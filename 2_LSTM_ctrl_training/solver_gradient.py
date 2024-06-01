import sys

import matplotlib.pyplot as plt

from model import LSTM
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, loader_train, loader_test, config):
        """Initialize configurations."""

        # Data loader.
        self.loader_train = loader_train
        self.loader_test = loader_test

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.r_lr = config.r_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.model_save_dir = config.model_save_dir

        # Step size.
        self.log_step = config.log_step
        self.early_stop_step = config.early_stop_step
        self.model_save_step = config.model_save_step

        # Early stop 
        self.patience = 5
        self.verbose = False
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = -0.00

        # Build the model.
        self.build_model()

    def build_model(self, input_size=4, hidden_size=64, num_layers=4, num_classes=2):
        """Create a classifier."""
        self.LSTM = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes,
                         device=self.device)
        self.optimizer = torch.optim.Adam(self.LSTM.parameters(), self.r_lr, [self.beta1, self.beta2])
        self.print_network(self.LSTM, 'LSTM')
        self.LSTM.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained classifier."""
        ResNet_path = os.path.join(self.model_save_dir, 'LSTM_ctrl.ckpt')
        self.LSTM.load_state_dict(torch.load(ResNet_path, map_location=lambda storage, loc: storage))

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.optimizer.zero_grad()

    def training_loss(self, logit, target):
        """Compute mse loss."""
        return F.mse_loss(logit.float(), target.float(), reduction='sum')

    def save_checkpoint(self):
        LSTM_path = os.path.join(self.model_save_dir, 'LSTM_ctrl.ckpt')
        torch.save(self.LSTM.state_dict(), LSTM_path)

    def early_stop_check(self, err):
        score = -err
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('EarlyStopping counter: %d out of %d, best score:%.3f, current score:%.3f' % (
                self.counter, self.patience, self.best_score, score))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0
            if score > self.best_score:
                self.best_score = score
                self.save_checkpoint()

    def train(self):
        """Train ResNet within a single dataset."""
        train_data_loader = self.loader_train
        test_data_loader = self.loader_test

        # training dataset
        train_data_iter = iter(train_data_loader)
        seg_input, output = next(train_data_iter)

        seg_input = seg_input.to(self.device)
        output = output.to(self.device)

        # Learning rate cache for decaying.
        r_lr = self.r_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                seg_input, output = next(train_data_iter)
            except:
                train_data_iter = iter(train_data_loader)
                seg_input, output = next(train_data_iter)

            seg_input = seg_input.to(self.device)
            output = output.to(self.device)

            # # =================================================================================== #
            # #                             2. Train the classifier                                 #
            # # =================================================================================== #

            # Compute loss with real images.
            est_output, _ = self.LSTM(seg_input)
            loss_t = self.training_loss(est_output, output)

            # Backward and optimize.
            self.reset_grad()
            loss_t.backward()
            self.optimizer.step()

            # Logging.
            loss = {}
            loss['loss'] = loss_t.item()

            # =================================================================================== #
            #                                 3. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i + 1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

            # =================================================================================== #
            #                                 4. Early Stopping                                   #
            # =================================================================================== #

            # Print out training information.
            if (i + 1) % self.early_stop_step == 0:

                val_data_loader = self.loader_test
                val_data_iter = iter(val_data_loader)
                err = 0
                with torch.no_grad():
                    for _, (seg_input, output) in enumerate(val_data_iter):
                        seg_input = seg_input.to(self.device)
                        est_output, _ = self.LSTM(seg_input)
                        est_output = est_output.cpu().numpy()
                        output = output.cpu().numpy()
                        err = err + np.mean(np.abs(est_output - output))
                self.early_stop_check(err)

            if self.early_stop:
                break

            # Save model checkpoints.
            if (i + 1) % self.model_save_step == 0:
                self.save_checkpoint()

    def ddec(self):
        # Load the trained classifier.
        self.restore_model(self.test_iters)

        # Load the dataset.
        train_data_loader = self.loader_train
        test_data_loader = self.loader_test

        # calculate test acc
        with (torch.no_grad()):
            grad_matrix_list = np.zeros([len(train_data_loader.dataset), 5, 4, 5, 2])

            num = 0
            for i, (pos, act) in enumerate(train_data_loader):
                for t_step in range(5):
                    for i_size in range(4):
                        dif = np.zeros([len(pos), 5, 4])
                        dif[:, t_step, i_size] = 0.1
                        dif_tensor = torch.from_numpy(dif)
                        pos_add = pos + dif_tensor
                        pos_add = pos_add.to(self.device)
                        pos_min = pos - dif_tensor
                        pos_min = pos_min.to(self.device)
                        act_est_add, _ = self.LSTM(pos_add)
                        act_est_add = act_est_add.cpu().numpy()
                        act_est_min, _ = self.LSTM(pos_min)
                        act_est_min = act_est_min.cpu().numpy()

                        # for k in range(len(pos)):
                        grad_matrix_list[64 * num: 64 * num + len(pos), t_step, i_size] = (
                                                                                                  act_est_add - act_est_min) / 0.2
                num += 1

        A = np.zeros([5, 2, 2])
        B = np.zeros([5, 2, 2])

        for i in range(5):
            for j in range(2):
                for l in range(2):
                    A[i, j, l] = np.mean(grad_matrix_list[:, i, j, 4, l], axis=0)
                    B[i, j, l] = np.mean(grad_matrix_list[:, i, j + 2, 4, l], axis=0)

        for i in range(5):
            print("norm of %d-th A matrix: %f" % (i + 1, np.linalg.norm(A[i])))

        for i in range(5):
            print("norm of %d-th B matrix: %f" % (i + 1, np.linalg.norm(B[i])))

        A[:3] = 0
        B[:4] = 0
        print(A[-2])
        print(A[-1])
        print(B[-1])
        sys.exit()

        np.savez("../0_files/linear_LSTM_ctrl.npz", A=A, B=B)

    def test(self):

        # Load the trained classifier.
        self.restore_model(self.test_iters)

        # Load the dataset.
        test_data_loader = self.loader_test

        # calculate test acc
        with torch.no_grad():

            err = np.zeros([len(self.loader_test.dataset), 5, 2])
            num = 0
            for i, (seg_input, output) in enumerate(test_data_loader):
                seg_input = seg_input.to(self.device)
                est_output, _ = self.LSTM(seg_input)
                est_output = est_output.cpu().numpy()
                output = output.cpu().numpy()
                for k in range(len(est_output)):
                    err[num] = np.abs(est_output[k] - output[k])
                    num += 1

        print("lstm_err:%.2f+-%.2f" % (np.mean(err[:, -1] / 2 * 100), np.var(err[:, -1] / 2 * 100)))

        self.ddec()