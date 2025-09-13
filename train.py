# train.py
import torch
import numpy as np
from collections import deque
import random
import os
from config import *
from environment import TradingEnv, ACTIONS
from data_preprocessing import load_and_preprocess
from model import DuelingConv1D

data = load_and_preprocess(DATA_PATH)
device = DEVICE

# Initialize networks
input_shape = (5, OBS_BARS)
agent = DuelingConv1D(input_shape[0], len(ACTIONS)).to(device)
target = DuelingConv1D(input_shape[0], len(ACTIONS)).to(device)
target.load_state_dict(agent.state_dict())
optimizer = torch.optim.RMSprop(agent.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.MSELoss()

# Replay buffer
replay = deque(maxlen=MEMORY_SIZE)

# Training loop
epsilon = EPSILON_START
step_count = 0

while step_count < MAX_STEPS:
    env = TradingEnv(data, obs_bars=OBS_BARS)
    state = env.reset()
    state = torch.from_numpy(state).unsqueeze(0).float().to(device)
    done = False
    while not done:
        step_count += 1
        # epsilon-greedy
        qvals = agent(state)
        if random.random() < epsilon:
            action = random.randint(0, len(ACTIONS)-1)
        else:
            action = torch.argmax(qvals, dim=1).item()
        action_name = ACTIONS[action]

        next_state, reward, done = env.step(action_name)
        next_state_tensor = torch.from_numpy(next_state).unsqueeze(0).float().to(device)

        replay.append((state, action, reward, next_state_tensor, done))
        state = next_state_tensor

        # Update
        if len(replay) >= BATCH_SIZE:
            batch = random.sample(replay, BATCH_SIZE)
            states_b = torch.cat([b[0] for b in batch]).float().to(device)
            actions_b = torch.tensor([b[1] for b in batch]).long().to(device)
            rewards_b = torch.tensor([b[2] for b in batch]).float().to(device)
            next_states_b = torch.cat([b[3] for b in batch]).float().to(device)
            dones_b = torch.tensor([b[4] for b in batch]).float().to(device)

            q_vals = agent(states_b).gather(1, actions_b.unsqueeze(1)).squeeze()
            with torch.no_grad():
                q_next = target(next_states_b).max(1)[0]
                y = rewards_b + GAMMA * q_next * (1 - dones_b)
            loss = loss_fn(q_vals, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if step_count % SYNC_FREQ == 0:
            target.load_state_dict(agent.state_dict())

    epsilon = max(EPSILON_END, EPSILON_START - EPSILON_START * step_count / MAX_STEPS)

    if step_count % 5000 == 0:
        # Save checkpoint
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save(agent.state_dict(), f"{CHECKPOINT_DIR}/checkpoint_step{step_count}.pt")
        print(f"Step {step_count}: checkpoint saved")
