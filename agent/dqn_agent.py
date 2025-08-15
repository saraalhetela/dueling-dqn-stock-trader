import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from utils.memory import ReplayMemory

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.lr)
        self.memory = ReplayMemory(config.memory_size)
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.target_update = config.target_update
        self.n_step = config.n_step
        self.steps_done = 0

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.policy_net.fc3.out_features)
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            return self.policy_net(state).argmax().item()

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Implement optimization step here

