import sys

import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_kind', type=str, default='train', choices=['train', 'test'])
config = parser.parse_args()

data = np.load("../0_files/data_" + config.data_kind + ".npz")

pos_list = data["pos_list"]
dir_list = data["dir_list"]
act_list = data["act_list"]
# pos_list: 3, steps
# dir_list: 3, 3, steps
# pos_list: 2, steps

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-0.1, 0.1])
ax.set_ylim([-0.1, 0.1])
ax.set_zlim([0.0, 0.2])
label_font = 15
ax.set_xlabel('x(m)', fontsize=label_font)
ax.set_ylabel('y(m)', fontsize=label_font)
ax.set_zlabel('z(m)', fontsize=label_font)
ax.xaxis.labelpad = 20
ax.yaxis.labelpad = 20
ax.zaxis.labelpad = 20

ax.scatter(pos_list[0], pos_list[1], pos_list[2])
plt.legend()
plt.savefig('../0_files/dataset_' + config.data_kind)
plt.show()
