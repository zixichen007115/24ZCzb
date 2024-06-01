import numpy as np
from tqdm import tqdm
from set_arm_environment import ArmEnvironment


class Environment(ArmEnvironment):
    def get_data(self):
        return [self.rod_parameters_dict]

    def setup(self):
        self.set_arm()


def main(ctrl_step=1, act_list=None):
    """ Create simulation environment """
    # time_step = 2.5e-4
    time_step = 2.5e-4

    controller_Hz = 1
    final_time = int(ctrl_step / controller_Hz)

    env = Environment(final_time, time_step=time_step)
    total_steps, systems = env.reset()

    controller_step_skip = int(1.0 / (controller_Hz * env.time_step))

    """ Read arm params """
    activations = []
    for m in range(len(env.muscle_groups)):
        activations.append(
            np.zeros(env.muscle_groups[m].activation.shape)
        )

    """ Start the simulation """
    print("Running simulation ...")
    time = np.float64(0.0)

    pos_list = np.zeros((3, len(act_list)))
    vel_list = np.zeros((3, len(act_list)))

    dir_list = np.zeros((3, 3, len(act_list)))
    ctrl_num = 0
    # act_list_ori = np.copy(act_list)
    # act_list = act_list + np.random.rand(ctrl_step, 8) * 0.1 - 0.05

    for k_sim in tqdm(range(total_steps)):

        if (k_sim % controller_step_skip) == 0:
            activations[0] = np.ones(100) * np.max([0, act_list[ctrl_num, 0]])
            activations[1] = np.ones(100) * np.max([0, act_list[ctrl_num, 1]])
            activations[2] = np.ones(100) * np.max([0, -act_list[ctrl_num, 0]])
            activations[3] = np.ones(100) * np.max([0, -act_list[ctrl_num, 1]])

            noise_weight = 0.05

            noise = np.random.rand(4, 100) * 2 * noise_weight - noise_weight
            activations = np.clip(activations + noise, -1, 1)


            pos_list[:, ctrl_num] = env.shearable_rod.position_collection[:, -1]
            vel_list[:, ctrl_num] = env.shearable_rod.velocity_collection[:, -1]
            dir_list[:, :, ctrl_num] = env.shearable_rod.director_collection[:, :, -1]

            ctrl_num = ctrl_num + 1

        time, systems, done = env.step(time, activations)
    print(ctrl_num)
    return pos_list, vel_list, dir_list, act_list
