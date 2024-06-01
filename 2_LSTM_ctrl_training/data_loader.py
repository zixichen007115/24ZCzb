from torch.utils import data
import random
import numpy as np


class Data_sim(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.mode = mode
        self.train_dataset = []
        self.val_dataset = []
        self.test_dataset = []
        self.preprocess()

    def preprocess(self):

        data = np.load("../0_files/data_train.npz")
        pos_list = data["pos_list"]
        vel_list = data["vel_list"]
        act_list = data["act_list"].T
        # pos_list: 3, steps
        # act_list: 2, steps
        xy_range = np.max((np.max(pos_list[:2]), -np.min(pos_list[:2])))
        vel_range = np.max((np.max(vel_list[0]), -np.min(vel_list[0])))

        pos_list[:2] = pos_list[:2] / xy_range
        vel_list[0] = vel_list[0] / vel_range

        print("position  list shape:{}".format(np.shape(pos_list)))
        print("velocity  list shape:{}".format(np.shape(vel_list)))

        print("action    list shape:{}".format(np.shape(act_list)))
        print("pos    range:%.3f_%.3f" % (np.min(pos_list[:2]), np.max(pos_list[:2])))
        # print("vel    range:%.3f_%.3f" % (np.min(vel_list[0]), np.max(vel_list[0])))
        print("action range:%.3f_%.3f" % (np.min(act_list), np.max(act_list)))

        random.seed(1)
        list_length = np.shape(pos_list)[1]

        val_test_sample = random.sample(range(list_length), int(list_length * 0.3))
        val_sample = val_test_sample[:int(list_length * 0.1)]
        test_sample = val_test_sample[int(list_length * 0.1):]

        t_step = 5

        for i in range(list_length - t_step - 1):
            seg_input = np.zeros([t_step, 4])
            # s2, a0
            output = np.zeros([t_step, 2])
            # a1

            for k in range(t_step):
                seg_input[k, 0:2] = pos_list[0:2, i + k + 2]
                seg_input[k, 2:4] = act_list[:, i + k]

                output[k, 0:2] = act_list[:, i + k + 1]

            if i in val_sample:
                self.val_dataset.append([seg_input.transpose(), output.transpose()])
            elif i in test_sample:
                self.test_dataset.append([seg_input.transpose(), output.transpose()])
            else:
                self.train_dataset.append([seg_input.transpose(), output.transpose()])

        print('Finished preprocessing the dataset...')
        print('train sample number: %d.' % len(self.train_dataset))
        print('validation sample number: %d.' % len(self.val_dataset))
        print('test sample number: %d.' % len(self.test_dataset))

    def __getitem__(self, index):
        if self.mode == 'train':
            dataset = self.train_dataset
        elif self.mode == 'test':
            dataset = self.test_dataset
        else:
            dataset = self.val_dataset
        seg_input, output = dataset[index]
        return seg_input.transpose(), output.transpose()

    def __len__(self):
        """Return the number of images."""
        if self.mode == 'train':
            return len(self.train_dataset)
        elif self.mode == 'test':
            return len(self.test_dataset)
        else:
            return len(self.val_dataset)


def get_loader(batch_size=32, mode='train', num_workers=1):
    """Build and return a data loader."""
    dataset = Data_sim(mode=mode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True if mode == 'train' else False,
                                  # shuffle=True,

                                  num_workers=num_workers)
    return data_loader
