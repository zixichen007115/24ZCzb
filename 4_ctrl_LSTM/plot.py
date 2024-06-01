import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import argparse
import imageio
from os import path

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='square', choices=['square', 'circle'])
config = parser.parse_args()

data = np.load('../0_files/data_lstm/data_' + config.task + ".npz")

act_lists = data["act_lists"]
real_lists = data["real_lists"]
shape_lists = data["shape_lists"]
# act_list:   trial, 2, steps
# real_list:  trial, 2, steps
# shape_list: trial, 21, 3, steps
act_list = act_lists[0]
real_list = real_lists[0]
shape_list = shape_lists[0]

print(np.shape(act_list))
print(np.shape(real_list))
print(np.shape(shape_list))

ctrl_step = np.shape(shape_list)[2]
img_dir = "../0_files/data_lstm/img_" + config.task

if os.path.exists(img_dir):
    shutil.rmtree(img_dir)
os.mkdir(img_dir)

for step in range(0, ctrl_step, 3):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    ax.set_xlim([-0.1, 0.1])
    ax.set_ylim([-0.1, 0.1])
    ax.set_zlim([0.0, 0.2])
    colors = ['b', 'g', 'k']

    for i in range(21):
        ax.scatter(shape_list[i, 0, step], shape_list[i, 1, step], shape_list[i, 2, step], c='r')

    plt.savefig(img_dir + "/%03d.png" % (step + 1))
    plt.close()

imgs = []
file = sorted(os.listdir(img_dir))
for f in file:
    real_url = path.join(img_dir, f)
    imgs.append(real_url)

frames = []
for image_name in imgs:
    frames.append(imageio.imread(image_name))

imageio.mimsave("../0_files/data_lstm/" + config.task, frames, 'GIF', duration=0.1)
