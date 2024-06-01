import numpy as np
import matplotlib.pyplot as plt
from trajectory import trajectory_generation
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='spiral', choices=['spiral', 'triangle', 'step'])
config = parser.parse_args()


current_file_path = os.getcwd()
ctrl_kind = current_file_path.split('_')[-1]

data = np.load('../0_files/data_' + ctrl_kind + '/data_' + config.task + ".npz")

act_list = data["act_lists"]
real_list = data["real_lists"]
shape_list = data["shape_lists"]
# act_list:   trial, 2, steps
# real_list:  trial, 2, steps
# shape_list: trial, 21, 3, steps

print(np.shape(act_list))
print(np.shape(real_list))
print(np.shape(shape_list))

data = np.load("../0_files/data_train.npz")
pos_list = data["pos_list"]
xy_range = np.max((np.max(pos_list[:2]), -np.min(pos_list[:2])))

trial = np.shape(shape_list)[0]
ctrl_step = np.shape(shape_list)[3]

tar_list = trajectory_generation(ctrl_step=ctrl_step, task=config.task)


err = np.zeros((trial, ctrl_step))

for t in range(trial):
    for i in range(ctrl_step):
        err[t, i] = np.linalg.norm(tar_list[:, i] - real_list[t, :, i])


print("err+-var: %.2f+-%.2f" % (np.mean(err * 100 / 2), np.var(err * 100 / 2)))


plt.figure(figsize=(8, 8))
plt.scatter(pos_list[0] / xy_range, pos_list[1] / xy_range, c='green', s=5, alpha=0.5)
plt.plot(tar_list[0, :ctrl_step], tar_list[1, :ctrl_step], c='red', label='desired trajectory', linewidth=5)
plt.plot(np.mean(real_list[:, 0], axis=0), np.mean(real_list[:, 1], axis=0), c='blue', label='real trajectory',
         linewidth=2.5)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.legend(fontsize=20)
plt.savefig('../0_files/data_' + ctrl_kind + '/' + config.task + ".png")
plt.show()

plt.figure(figsize=(8, 8))
plt.plot(tar_list[0, :ctrl_step], c='red', label='desired', linewidth=5)
plt.plot(np.mean(real_list[:, 0], axis=0), c='blue', label='real', linewidth=2.5)
plt.fill_between(np.linspace(0, ctrl_step - 1, ctrl_step), np.min(real_list[:, 0], axis=0), np.max(real_list[:, 0], axis=0), color='blue', alpha=0.5)
plt.plot(act_list[0, 0], label='act')
plt.ylim(-1.1, 1.1)
plt.legend(fontsize=20)
plt.savefig('../0_files/data_' + ctrl_kind + '/x_' + config.task + ".png")
plt.show()

plt.figure(figsize=(8, 8))
plt.plot(tar_list[1, :ctrl_step], c='red', label='desired', linewidth=5)
plt.plot(np.mean(real_list[:, 1], axis=0), c='blue', label='real', linewidth=2.5)
plt.fill_between(np.linspace(0, ctrl_step - 1, ctrl_step), np.min(real_list[:, 1], axis=0), np.max(real_list[:, 1], axis=0), color='blue', alpha=0.5)
plt.plot(act_list[0, 1], label='act')
plt.ylim(-1.1, 1.1)
plt.legend(fontsize=20)
plt.savefig('../0_files/data_' + ctrl_kind + '/y_' + config.task + ".png")
plt.show()
