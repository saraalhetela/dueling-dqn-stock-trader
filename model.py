# model.py
import torch
import torch.nn as nn

class DuelingConv1D(nn.Module):
    def __init__(self, input_channels, output_size):
        super(DuelingConv1D, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 128, 5)
        self.conv2 = nn.Conv1d(128, 128, 5)
        self.flatten = nn.Flatten()
        self.state_value_linear1 = nn.Linear(5376, 512)
        self.state_value_linear2 = nn.Linear(512, 1)
        self.adv_linear1 = nn.Linear(5376, 512)
        self.adv_linear2 = nn.Linear(512, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        val = self.relu(self.state_value_linear1(x))
        val = self.state_value_linear2(val)
        adv = self.relu(self.adv_linear1(x))
        adv = self.adv_linear2(adv)
        return val + adv - adv.mean(dim=1, keepdim=True)
