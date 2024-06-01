import sys

import numpy as np
from run_simulation_dynamic import main
from trajectory import trajectory_generation
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='spiral', choices=['spiral', 'triangle', 'step'])
config = parser.parse_args()

ctrl_step = 80

current_file_path = os.getcwd()
ctrl_kind = current_file_path.split('_')[-1]

if not os.path.exists("../0_files/data_" + ctrl_kind):
    os.mkdir("../0_files/data_" + ctrl_kind)

trial = 3
act_lists = np.zeros((trial, 2, ctrl_step))
real_lists = np.zeros((trial, 2, ctrl_step))
shape_lists = np.zeros((trial, 21, 3, ctrl_step))

tar_list = trajectory_generation(ctrl_step, config.task)


for i in range(trial):
    act_list, real_list, shape_list = main(ctrl_step=ctrl_step, tar_list=tar_list)
    # print(act_list, real_list, shape_list)
    act_lists[i] = act_list.T
    real_lists[i] = real_list
    shape_lists[i] = shape_list

np.savez('../0_files/data_' + ctrl_kind + '/data_'+config.task,
         act_lists=act_lists, real_lists=real_lists, shape_lists=shape_lists)
