import numpy as np
import torch


def k_cal(poses, acts, k_np):
    # poses: (t_step - 1) + 1, 2; -1 is target
    # acts:  (t_step - 1), 2
    # j:     2, 2
    # t_step: 5

    k_tensor = torch.tensor(k_np, requires_grad=True)
    optim_m = torch.optim.Adam([k_tensor], 0.02, [0.9, 0.999])

    dif_poses = poses[1:-1]
    dif_poses_tensor = torch.tensor(dif_poses.T, requires_grad=False)

    dif_acts = acts[1:]
    dif_acts_tensor = torch.tensor(dif_acts.T, requires_grad=False)

    for it in range(5):
        loss = dif_poses_tensor - k_tensor @ dif_acts_tensor
        loss_sum = loss.norm()
        optim_m.zero_grad()
        loss_sum.backward()
        optim_m.step()

    kin_np = k_tensor.detach().numpy()
    k_new_tensor = torch.tensor(kin_np, requires_grad=False)

    dif_act_tensor = torch.tensor(np.zeros(2), requires_grad=True)

    optim_c = torch.optim.Adam([dif_act_tensor], 0.02, [0.9, 0.999])

    dif_tar_poses = poses[-1]
    dif_tar_poses_tensor = torch.tensor(dif_tar_poses, requires_grad=False)

    act_tensor = torch.tensor(acts[-1], requires_grad=False)

    for it in range(5):
        loss = dif_tar_poses_tensor - k_new_tensor @ (act_tensor + dif_act_tensor)
        loss_sum = loss.norm()
        optim_c.zero_grad()
        loss_sum.backward()
        optim_c.step()

    act = (act_tensor + dif_act_tensor).detach().numpy()

    return act, kin_np


def act_est(poses, acts, k_np):
    # poses: (t_step - 1) + 1, 2; -1 is target
    # acts:  (t_step - 1), 2
    # j:     2, 2

    act_k, k_np = k_cal(poses, acts, k_np)
    act = act_k

    act_max = 1
    for i in range(2):
        if np.abs(act[i]) > act_max:
            act[i] = act_max * act[i] / np.abs(act[i])

    return act, k_np
