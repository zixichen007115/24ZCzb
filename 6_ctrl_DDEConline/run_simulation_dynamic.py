import sys
import numpy as np
from tqdm import tqdm
from set_arm_environment import ArmEnvironment
import torch


class Environment(ArmEnvironment):
    def get_data(self):
        return [self.rod_parameters_dict]

    def setup(self):
        self.set_arm()


def matrix_update(c_input, c_matrix):
    X = c_matrix.T  #6,2

    Y = c_input[4:6, 1:].T  #(t_step - 1), 2
    A = c_input[:, :-1].T  #(t_step - 1), 6

    # Y = c_input[4:6, 3:].T  # (t_step - 1), 2
    # A = c_input[:, :-3].T  # (t_step - 1), 6

    # torch
    X_tensor = torch.tensor(X, requires_grad=True)
    Y_tensor = torch.tensor(Y, requires_grad=False)
    A_tensor = torch.tensor(A, requires_grad=False)
    optim = torch.optim.Adam([X_tensor], lr=1e-4)

    for i in range(10):
        loss = Y_tensor - A_tensor @ X_tensor

        optim.zero_grad()
        loss.norm().backward()
        optim.step()

    X = X_tensor.cpu().detach().numpy()

    # # linear optimization
    # alpha = 1e-3
    # for i in range(10):
    #     X_new = X + alpha * A.T.dot(Y - A.dot(X))
    #     X = np.copy(X_new)

    c_matrix = X.T

    return c_matrix

def main(ctrl_step=1, tar_list=None):
    """ Create simulation environment """
    time_step = 2.5e-4
    controller_Hz = 1
    final_time = int(ctrl_step / controller_Hz)

    env = Environment(final_time, time_step=time_step)
    total_steps, systems = env.reset()

    controller_step_skip = int(1.0 / (controller_Hz * env.time_step))

    data = np.load("../0_files/data_train.npz")
    pos_list = data["pos_list"]

    xy_range = np.max((np.max(pos_list[:2]), -np.min(pos_list[:2])))

    """ Read arm params """
    activations = []
    for m in range(len(env.muscle_groups)):
        activations.append(
            np.zeros(env.muscle_groups[m].activation.shape)
        )

    """ Start the simulation """
    print("Running simulation ...")
    time = np.float64(0.0)

    ctrl_num = 0
    t_step = 5

    shape_list = np.zeros((21, 3, ctrl_step))
    real_list = np.zeros((2, ctrl_step))

    seg_input = np.zeros([2, 2])
    act_list = np.zeros((ctrl_step, 2))
    pre_act = np.zeros(2)

    data_AB = np.load('../0_files/linear_LSTM_ctrl.npz')

    A = data_AB["A"]
    B = data_AB["B"]
    A0 = A[-1]
    A1 = A[-2]
    B0 = B[-1]

    c_matrix = np.concatenate([A0, A1, B0], axis=1)
    c_input = np.zeros([6, t_step])

    # a1 = A0 * s2 + A1 * s1 + B0 * a0

    for k_sim in tqdm(range(total_steps)):
        if (k_sim % controller_step_skip) == 0:

            for i in range(21):
                shape_list[i, :, ctrl_num] = env.shearable_rod.position_collection[:, 5 * i]

            real_list[:, ctrl_num] = env.shearable_rod.position_collection[:2, -1] / xy_range

            for i in range(len(c_input[0]) - 1):
                c_input[:, i] = c_input[:, i + 1]

            c_input[0:2, -2] = real_list[:, ctrl_num]

            c_input[4:6, -1] = np.copy(pre_act)
            c_input[2:4, -1] = real_list[:, ctrl_num]
            c_input[0:2, -1] = tar_list[:, ctrl_num + 1]

            c_matrix = matrix_update(c_input, c_matrix)

            act_lstm = c_matrix.dot(c_input[:, -1])

            act_noise = np.random.rand(2) * 0.1 - 0.05
            act_lstm_real = act_lstm + act_noise

            act_max = 1
            for i in range(2):
                if np.abs(act_lstm_real[i]) > act_max:
                    act_lstm_real[i] = act_max * np.sign(act_lstm_real[i])

            pre_act = np.copy(act_lstm)
            act_list[ctrl_num] = act_lstm_real

            activations[0] = np.ones(100) * np.max([0, act_lstm_real[0]])
            activations[1] = np.ones(100) * np.max([0, act_lstm_real[1]])
            activations[2] = np.ones(100) * np.max([0, -act_lstm_real[0]])
            activations[3] = np.ones(100) * np.max([0, -act_lstm_real[1]])

            ctrl_num = ctrl_num + 1

        time, systems, done = env.step(time, activations)
    return act_list, real_list, shape_list
