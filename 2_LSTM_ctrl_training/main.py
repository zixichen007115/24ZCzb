import os
import argparse
# from solver import Solver
from solver_gradient import Solver

from data_loader import get_loader
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)

        # Data loader.
    if config.mode == 'train':
        loader_train = get_loader(config.batch_size, 'train')
        loader_test = get_loader(config.batch_size, 'val')

    else:
        loader_train = get_loader(config.batch_size, 'train')
        loader_test = get_loader(config.batch_size, 'test')

    # Solver for training and testing StarGAN.
    solver = Solver(loader_train, loader_test, config)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_iters', type=int, default=1000000, help='number of total iterations for training D')
    parser.add_argument('--r_lr', type=float, default=0.001, help='learning rate for ResNet')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=1000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    # Directories.
    parser.add_argument('--model_save_dir', type=str, default='../0_files')

    # Step size.
    parser.add_argument('--log_step', type=int, default=500)
    parser.add_argument('--early_stop_step', type=int, default=500)
    parser.add_argument('--model_save_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)
