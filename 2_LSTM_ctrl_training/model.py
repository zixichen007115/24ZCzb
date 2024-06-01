import sys

import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.fc_0 = nn.Linear(hidden_size, self.num_classes)

    def forward(self, x):
        x = x.to(torch.float32)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # out, _ = self.lstm(x, h0)

        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))

        # print(out[0, -1])
        # sys.exit()
        out_0 = self.fc_0(out)
        return out_0, out
