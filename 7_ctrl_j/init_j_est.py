import sys

import numpy as np
from tqdm import tqdm
from set_arm_environment import ArmEnvironment



class Environment(ArmEnvironment):
    def get_data(self):
        return [self.rod_parameters_dict]

    def setup(self):
        self.set_arm()


def j_init_move(act=None):
    """ Create simulation environment """
    # time_step = 2.5e-4
    ctrl_step = 1
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

    pos_list = np.zeros((3, 2))
    ctrl_num = 0

    for k_sim in tqdm(range(total_steps + 1)):
        if (k_sim % controller_step_skip) == 0:
            activations[0] = np.ones(100) * np.max([0, act[0]])
            activations[1] = np.ones(100) * np.max([0, act[1]])
            activations[2] = np.ones(100) * np.max([0, -act[0]])
            activations[3] = np.ones(100) * np.max([0, -act[1]])

            noise_weight = 0.05

            noise = np.random.rand(4, 100) * 2 * noise_weight - noise_weight
            activations = np.clip(activations + noise, -1, 1)

            pos_list[:, ctrl_num] = env.shearable_rod.position_collection[:, -1]

            ctrl_num = ctrl_num + 1

        time, systems, done = env.step(time, activations)

    return pos_list


j_init = np.zeros([2, 2])

for i in range(2):
    act = np.zeros(2)
    act[i] = 1
    j_cols = j_init_move(act)
    j_col = j_cols[:, 1] - j_cols[:, 0]
    j_init[:, i] = j_col[:2]

np.save('../0_files/j_init.npy', j_init)
