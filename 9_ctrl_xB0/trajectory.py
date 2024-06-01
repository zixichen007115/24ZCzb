import matplotlib.pyplot as plt
import numpy as np


def trajectory_generation(ctrl_step=100, task=None):
    ctrl_step = ctrl_step + 1
    x_tar = np.zeros(ctrl_step)
    y_tar = np.zeros(ctrl_step)
    radi = 0.8

    if task == 'spiral':

        len_tar = np.linspace(0, 1, ctrl_step)
        sin_tar = np.sin(np.linspace(0, 6 * np.pi, ctrl_step))
        cos_tar = np.cos(np.linspace(0, 6 * np.pi, ctrl_step))

        for i in range(ctrl_step):
            x_tar[i] = len_tar[i] * sin_tar[i]
            y_tar[i] = len_tar[i] * cos_tar[i]

    elif task == 'triangle':

        ei_step = int(ctrl_step / 8)
        x_tar[:ei_step] = 0
        x_tar[ei_step * 1:ei_step * 3] = np.linspace(0, 0.8, ei_step * 2)
        x_tar[ei_step * 3:ei_step * 5] = np.linspace(0.8, -0.8, ei_step * 2)
        x_tar[ei_step * 5:ei_step * 7] = np.linspace(-0.8, 0, ei_step * 2)
        x_tar[ei_step * 7:] = 0

        y_tar[:ei_step] = np.linspace(0, 1, ei_step)
        y_tar[ei_step * 1:ei_step * 3] = np.linspace(1, -0.6, ei_step * 2)
        y_tar[ei_step * 3:ei_step * 5] = -0.6
        y_tar[ei_step * 5:ei_step * 7] = np.linspace(-0.6, 1, ei_step * 2)
        y_tar[ei_step * 7:] = np.linspace(1, 0, ei_step + 1)

    elif task == 'step':

        four_step = int(ctrl_step / 4)
        x_tar[:four_step] = 1
        x_tar[four_step:four_step * 2] = -1
        x_tar[four_step * 2:four_step * 3] = 1
        x_tar[four_step * 3:] = -1



    trajectory = np.vstack((x_tar, -y_tar)) * radi

    return trajectory
