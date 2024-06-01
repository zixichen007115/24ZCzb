import sys

import elastica as ea
from elastica import *
from elastica.timestepper import extend_stepper_interface
from elastica._calculus import _isnan_check

from coomm.actuations.muscles import force_length_weight_poly
from coomm.actuations.muscles import (
    MuscleGroup,
    LongitudinalMuscle,
    ApplyMuscleGroups
)

from coomm.forces import DragForce
from coomm.callback_func import RodCallBack



class BaseSimulator(BaseSystemCollection, Constraints, Connections, Forcing, CallBacks):
    pass


class ArmEnvironment:
    def __init__(self, final_time, time_step=3.0e-4, recording_fps=10, num_seg=1):
        # Integrator type
        self.StatefulStepper = PositionVerlet()

        self.final_time = final_time
        self.time_step = time_step
        self.total_steps = int(self.final_time / self.time_step)
        self.recording_fps = recording_fps
        self.youngs_modulus = 1e4
        self.poisson_ratio = 0.5
        self.LM_ratio_muscle_position_parameter = 0.0075
        self.step_skip = int(1.0 / (self.recording_fps * self.time_step))
        self.num_seg = num_seg

    def get_systems(self, ):
        return self.simulator

    def get_data(self, ):
        return [self.rod_parameters_dict]

    def set_arm(self):
        base_length, radius = self.set_rod()
        self.set_muscles(radius[0], self.shearable_rod)
        self.set_drag_force(
            base_length, radius[0], radius[-1],
            self.shearable_rod, self.rod_parameters_dict
        )

    def setup(self):
        self.set_arm()

    def set_rod(self):
        """ Set up a rod """
        # n_elements = 25 * self.num_seg  # number of discretized elements of the arm
        # base_length = 0.05 * self.num_seg  # total length of the arm

        n_elements = 100  # number of discretized elements of the arm
        base_length = 0.2  # total length of the arm

        radius_base = 0.01  # radius of the arm at the base
        radius_tip = 0.01  # radius of the arm at the tip

        radius = np.linspace(radius_base, radius_tip, n_elements + 1)
        radius_mean = (radius[:-1] + radius[1:]) / 2
        damp_coefficient = 0.05

        self.shearable_rod = CosseratRod.straight_rod(
            n_elements=n_elements,
            start=np.zeros((3,)),
            direction=np.array([0.0, 0.0, 1.0]),
            normal=np.array([1.0, 0.0, 0.0]),
            base_length=base_length,
            base_radius=radius_mean.copy(),
            density=1050,
            nu=damp_coefficient * ((radius_mean / radius_base) ** 2),
            youngs_modulus=self.youngs_modulus,
            shear_modulus=self.youngs_modulus / (2 * (1 + self.poisson_ratio)),
            nu_for_torques=damp_coefficient * ((radius_mean / radius_base) ** 4),
        )
        self.simulator.append(self.shearable_rod)

        self.rod_parameters_dict = defaultdict(list)
        self.simulator.collect_diagnostics(self.shearable_rod).using(
            RodCallBack,
            step_skip=self.step_skip,
            callback_params=self.rod_parameters_dict
        )

        """ Set up boundary conditions """
        self.simulator.constrain(self.shearable_rod).using(
            OneEndFixedRod,
            constrained_position_idx=(0,),
            constrained_director_idx=(0,)
        )

        return base_length, radius

    def set_muscles(self, base_radius, arm):
        """ Add muscle actuation """

        def add_muscle_actuation(radius_base, arm):

            muscle_groups = []
            LM_ratio_muscle_position = self.LM_ratio_muscle_position_parameter / radius_base
            LM_ratio_radius = 0.003 / radius_base
            shearable_rod_area = np.pi * arm.radius ** 2
            LM_rest_muscle_area = shearable_rod_area * (
                    LM_ratio_radius ** 2
            )

            LM_max_muscle_stress = 10_000.0

            muscle_dict = dict(
                force_length_weight=force_length_weight_poly,
            )

            # Add 4 longitudinal muscles
            for k in range(4):
                muscle_groups.append(
                    MuscleGroup(
                        muscles=[
                            LongitudinalMuscle(
                                muscle_init_angle=np.pi * 0.5 * k,
                                ratio_muscle_position=LM_ratio_muscle_position,
                                rest_muscle_area=LM_rest_muscle_area,
                                max_muscle_stress=LM_max_muscle_stress,
                                **muscle_dict
                            )
                        ],
                        type_name='LM',
                    )
                )

            for muscle_group in muscle_groups:
                muscle_group.set_current_length_as_rest_length(arm)

            return muscle_groups

        self.muscle_groups = add_muscle_actuation(
            base_radius, arm
        )
        self.muscle_callback_params_list = [
            defaultdict(list) for _ in self.muscle_groups
        ]
        self.simulator.add_forcing_to(self.shearable_rod).using(
            ApplyMuscleGroups,
            muscle_groups=self.muscle_groups,
            step_skip=self.step_skip,
            callback_params_list=self.muscle_callback_params_list,
        )

        gravitational_acc = 0.5

        self.simulator.add_forcing_to(self.shearable_rod).using(
            ea.GravityForces, acc_gravity=np.array([0.0, 0.0, gravitational_acc])
        )

    def set_drag_force(self,
                       base_length, base_radius, tip_radius,
                       arm, arm_parameters_dict
                       ):
        """ Add drag force """
        dl = base_length / arm.n_elems
        fluid_factor = 1
        r_bar = (base_radius + tip_radius) / 2
        sea_water_dentsity = 1022
        c_per = 0.41 / sea_water_dentsity / r_bar / dl * fluid_factor
        c_tan = 0.033 / sea_water_dentsity / np.pi / r_bar / dl * fluid_factor

        self.simulator.add_forcing_to(arm).using(
            DragForce,
            rho_environment=sea_water_dentsity,
            c_per=c_per,
            c_tan=c_tan,
            system=arm,
            step_skip=self.step_skip,
            callback_params=arm_parameters_dict  # self.rod_parameters_dict
        )

    def reset(self):
        self.simulator = BaseSimulator()

        self.setup()

        """ Finalize the simulator and create time stepper """
        self.simulator.finalize()
        self.do_step, self.stages_and_updates = extend_stepper_interface(
            self.StatefulStepper, self.simulator
        )

        """ Return 
            (1) total time steps for the simulation step iterations
            (2) systems for controller design
        """
        return self.total_steps, self.get_systems()

    def step(self, time, muscle_activations):

        """ Set muscle activations """
        for muscle_group, activation in zip(self.muscle_groups, muscle_activations):
            muscle_group.apply_activation(activation)

        """ Run the simulation for one step """
        time = self.do_step(
            self.StatefulStepper,
            self.stages_and_updates,
            self.simulator,
            time,
            self.time_step,
        )

        """ Done is a boolean to reset the environment before episode is completed """
        done = False
        # Position of the rod cannot be NaN, it is not valid, stop the simulation
        invalid_values_condition = _isnan_check(self.shearable_rod.position_collection)

        if invalid_values_condition == True:
            done = True
            sys.exit("NaN detected in the simulation !!!!!!!!")

        """ Return
            (1) current simulation time
            (2) current systems
            (3) a flag denotes whether the simulation runs correlectly
        """
        return time, self.get_systems(), done

    def save_data(self, filename="simulation", **kwargs):

        import pickle

        print("Saving data to pickle files ...", end='\r')

        with open(filename + "_data.pickle", "wb") as data_file:
            data = dict(
                recording_fps=self.recording_fps,
                systems=self.get_data(),
                muscle_groups=self.muscle_callback_params_list,
                **kwargs
            )
            pickle.dump(data, data_file)

        with open(filename + "_systems.pickle", "wb") as system_file:
            data = dict(
                systems=self.get_systems(),
                muscle_groups=self.muscle_groups,
            )
            pickle.dump(data, system_file)

        print("Saving data to pickle files ... Done!")
