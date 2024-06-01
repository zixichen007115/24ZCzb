import sys

import numpy as np
from run_simulation_dynamic import main
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_kind', type=str, default='train', choices=['train', 'test'])
config = parser.parse_args()

if config.data_kind == 'train':
    ctrl_step = 10000
else:
    ctrl_step = 1000


act_list = np.random.rand(ctrl_step, 2) * 2 - 1
print("%.3f_%.3f" % (np.min(act_list), np.max(act_list)))

pos_list, vel_list, dir_list, act_list = main(ctrl_step=ctrl_step, act_list=act_list)

np.savez('../0_files/data_' + config.data_kind, pos_list=pos_list, vel_list=vel_list, dir_list=dir_list,
         act_list=act_list)

print(pos_list.shape)
print("%.3f_%.3f" % (np.min(pos_list[:, 0]), np.max(pos_list[:, 0])))
print("%.3f_%.3f" % (np.min(pos_list[:, 1]), np.max(pos_list[:, 1])))
print("%.3f_%.3f" % (np.min(pos_list[:, 2]), np.max(pos_list[:, 2])))
print("%.3f_%.3f" % (np.min(act_list), np.max(act_list)))
