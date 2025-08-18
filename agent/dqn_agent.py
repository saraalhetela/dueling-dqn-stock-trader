import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from utils.memory import ReplayMemory

class dueling_Conv1D_Q_Net(nn.Module):
    def __init__(self, input_depth_length, output_shape):
        super(dueling_Conv1D_Q_Net, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_depth_length, 128, 5)
        self.conv2 = torch.nn.Conv1d(128, 128, 5)
        self.state_value_linear1 = torch.nn.Linear(5376, 512)
        self.state_value_linear2 = torch.nn.Linear(512, 1)
        self.advantage_linear1 = torch.nn.Linear(5376, 512)
        self.advantage_linear2 = torch.nn.Linear(512, output_shape)
        self.activation = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.flatten(x)
        val = self.activation(self.state_value_linear1(x))
        val = self.state_value_linear2(val)
        adv = self.activation(self.advantage_linear1(x))
        adv = self.advantage_linear2(adv)
        return val + adv - adv.mean(dim=1, keepdim=True)

def batch_target_for_nsteps_dqn(nsteps_reward_batch, gamma, maxQ, nsteps_done_batch, device="cpu"):
    Y_list = []
    for i, rewards in enumerate(nsteps_reward_batch):
        Y = torch.tensor(np.dot([gamma**j for j in range(len(rewards))], rewards)).to(device)
        Y += gamma**len(rewards) * maxQ[i] * (1 - nsteps_done_batch[i])
        Y_list.append(Y)
    return torch.stack(Y_list).float().to(device)


def preprocess_state(state, input_shape, add_noise=False):
    if add_noise:
        return state.reshape(*input_shape) + np.random.rand(*input_shape) / 100.0
    return state.reshape(*input_shape)


def get_action(Q_val, num_actions, epsilon):
    return np.random.randint(0, num_actions) if random.random() < epsilon else np.argmax(Q_val, 1)[0]


def update(loss, optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# DQN Helpers
def get_batch_for_nsteps_dqn(replay, batch_size, nsteps=1, device="cpu"):
    if nsteps < 1:
        nsteps = 1
    minibatch_idx = random.sample(range(len(replay)), batch_size)
    minibatch = [replay[idx] for idx in minibatch_idx]
    nsteps_next_state_batch = []
    nsteps_reward_batch = []
    nsteps_done_batch = []
    state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch]).float().to(device)
    action1_batch = torch.tensor([a for (s1, a, r, s2, d) in minibatch]).long().to(device)
    for exp_idx in minibatch_idx:
        nsteps_reward = []
        for step in range(nsteps):
            exp = replay[min(exp_idx + step, len(replay) - 1)]
            _, _, r, s2, d = exp
            nsteps_reward.append(r)
            if d or step == nsteps - 1:
                nsteps_next_state_batch.append(s2)
                nsteps_done_batch.append(d)
                break
        nsteps_reward_batch.append(nsteps_reward)
    nsteps_next_state_batch = torch.cat(nsteps_next_state_batch).float().to(device)
    nsteps_done_batch = torch.tensor(nsteps_done_batch).long().to(device)
    return state1_batch, action1_batch, nsteps_next_state_batch, nsteps_reward_batch, nsteps_done_batch
