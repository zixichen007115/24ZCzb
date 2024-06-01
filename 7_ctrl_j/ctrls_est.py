import numpy as np
import torch


def j_cal(poses, acts, j_np):
    # poses: (t_step - 1) + 1, 2; -1 is target
    # acts:  (t_step - 1), 2
    # j:     2, 2
    # t_step: 5

    j_tensor = torch.tensor(j_np, requires_grad=True)
    optim_m = torch.optim.Adam([j_tensor], 0.075, [0.9, 0.999])

    dif_poses = poses[1:-1] - poses[:-2]
    dif_poses_tensor = torch.tensor(dif_poses.T, requires_grad=False)

    dif_acts = acts[1:] - acts[:-1]
    dif_acts_tensor = torch.tensor(dif_acts.T, requires_grad=False)

    for it in range(20):
        loss = dif_poses_tensor - j_tensor @ dif_acts_tensor
        loss_sum = loss.norm()
        optim_m.zero_grad()
        loss_sum.backward()
        optim_m.step()

    jac_np = j_tensor.detach().numpy()
    j_new_tensor = torch.tensor(jac_np, requires_grad=False)

    dif_act_tensor = torch.tensor(np.zeros(2), requires_grad=True)

    optim_c = torch.optim.Adam([dif_act_tensor], 0.075, [0.9, 0.999])

    dif_tar_poses = poses[-1] - poses[-2]
    dif_tar_poses_tensor = torch.tensor(dif_tar_poses, requires_grad=False)

    act_tensor = torch.tensor(acts[-1], requires_grad=False)

    for it in range(20):
        loss = dif_tar_poses_tensor - j_new_tensor @ (act_tensor + dif_act_tensor)
        loss_sum = loss.norm()
        optim_c.zero_grad()
        loss_sum.backward()
        optim_c.step()

    act = (act_tensor + dif_act_tensor).detach().numpy()

    return act, jac_np


def act_est(poses, acts, j_np):
    # poses: (t_step - 1) + 1, 2; -1 is target
    # acts:  (t_step - 1), 2
    # j:     2, 2

    act_j, j_np = j_cal(poses, acts, j_np)
    act = act_j

    act_max = 1
    for i in range(2):
        if np.abs(act[i]) > act_max:
            act[i] = act_max * act[i] / np.abs(act[i])

    return act, j_np
