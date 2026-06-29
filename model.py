import torch
import torch.nn as nn

class DuelingConv1D(nn.Module):
    def __init__(self, input_channels, output_size, obs_bars=50):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5)
        self.relu  = nn.ReLU()
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, obs_bars)
            flat_size = self._conv_out(dummy).shape[1]
        self.value_fc1 = nn.Linear(flat_size, 128)
        self.value_fc2 = nn.Linear(128, 1)
        self.adv_fc1   = nn.Linear(flat_size, 128)
        self.adv_fc2   = nn.Linear(128, output_size)

    def _conv_out(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x.flatten(start_dim=1)

    def forward(self, x):
        f   = self._conv_out(x)
        val = self.relu(self.value_fc1(f))
        val = self.value_fc2(val)
        adv = self.relu(self.adv_fc1(f))
        adv = self.adv_fc2(adv)
        return val + adv - adv.mean(dim=1, keepdim=True)
