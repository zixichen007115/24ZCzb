# 24ZCzb

## Dependencies
`pip install -r requriements.txt`

## 0_files
All the files (NN dataset, NN, trajectory, etc.) generated during training and control

## 1_sim
To generate the dataset for LSTM training

`python data_generation_dynamic.py`

## 2_LSTM_ctrl_training

train a LSTM network and calculate the gradient matrices. 

## 3_k_j_init

estimate the initial kinematics and Jacobian matrices.

## 4_ctrl_LSTM

control the robot to fulfill tasks with the LSTM controller.

## 5_ctrl_DDEC

control the robot to fulfill tasks with the linear DDEC controller.

## 6_ctrl_DDEConline

control the robot to fulfill tasks with the online updating DDEC controller.

## 7_ctrl_j, 8_ctrl_k

control the robot to fulfill tasks with the kinematics and Jacobian controller.

## 9_ctrl_xA0/xA1/xB0

Abliation study. control the robot to fulfill tasks with the linear DDEC controller while blocking one componment each time.

## Citation

We ask that any publications which use this repository cite as following:

```
@article{chen2024data,
  title={Data-driven Explainable Controller for Soft Robots based on Recurrent Neural Networks},
  author={Chen, Zixi and Ren, Xuyang and Ciuti, Gastone and Stefanini, Cesare},
  journal={arXiv e-prints},
  pages={arXiv--2406},
  year={2024}
}



```
