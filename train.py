# train.py
import copy
import os
import random
from collections import deque

import numpy as np
import torch

from config import *
from environment import TradingEnv, ACTIONS


def train_agent(model, train_data, val_data, device="cpu"):
    agent  = model
    target = copy.deepcopy(agent).to(device)
    target.load_state_dict(agent.state_dict())

    optimizer = torch.optim.RMSprop(agent.parameters(), lr=LEARNING_RATE)
    loss_fn   = torch.nn.SmoothL1Loss()

    # Replay buffer stores numpy arrays, NOT GPU tensors.
    # Storing tensors on GPU blows VRAM at large buffer sizes.
    replay     = deque(maxlen=MEMORY_SIZE)
    n_buf      = deque(maxlen=N_STEP)
    epsilon    = EPSILON_START
    step_count = 0
    train_profits  = []
    val_rewards    = []

    def flush_n(buf):
        """Compute N-step return and push numpy transition to replay."""
        if not buf:
            return
        s0, a0      = buf[0][0], buf[0][1]   # numpy array, int
        ret         = 0.0
        last_ns     = buf[-1][3]              # numpy array
        last_done   = buf[-1][4]
        for i, (_, _, r, _, d) in enumerate(buf):
            ret += (GAMMA ** i) * r
            if d:
                last_done = True
                break
        # Store as numpy — cheap on RAM, moved to device only at sample time
        replay.append((s0, a0, ret, last_ns, last_done))

    while step_count < MAX_STEPS:
        env   = TradingEnv(train_data, obs_bars=OBS_BARS)
        state = env.reset()                  # numpy (5, 50)
        done  = False
        ep_profit = 0.0
        n_buf.clear()

        while not done:
            step_count += 1

            # Move to device only for inference — don't keep it there
            state_t = torch.from_numpy(state).unsqueeze(0).float().to(device)

            if random.random() < epsilon:
                action = random.randint(0, len(ACTIONS) - 1)
            else:
                with torch.no_grad():
                    action = agent(state_t).argmax(dim=1).item()

            next_state, reward, done = env.step(ACTIONS[action])
            # next_state is numpy — keep it that way for the buffer

            n_buf.append((state, action, reward, next_state, done))
            if len(n_buf) == N_STEP or done:
                flush_n(n_buf)
                if done:
                    while len(n_buf) > 1:
                        n_buf.popleft()
                        flush_n(n_buf)
                    n_buf.clear()

            state = next_state        # stay as numpy
            ep_profit += reward

            # ── Learning step ──────────────────────────────────────────────
            if len(replay) >= BATCH_SIZE:
                batch = random.sample(replay, BATCH_SIZE)

                # Stack numpy arrays → single tensor → move to device once
                sb  = torch.from_numpy(np.stack([b[0] for b in batch])).float().to(device)
                ab  = torch.tensor([b[1] for b in batch], dtype=torch.long).to(device)
                rb  = torch.tensor([b[2] for b in batch], dtype=torch.float32).to(device)
                nsb = torch.from_numpy(np.stack([b[3] for b in batch])).float().to(device)
                db  = torch.tensor([b[4] for b in batch], dtype=torch.float32).to(device)

                q_pred = agent(sb).gather(1, ab.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    best_a = agent(nsb).argmax(dim=1, keepdim=True)
                    q_next = target(nsb).gather(1, best_a).squeeze(1)
                    q_tgt  = rb + (GAMMA ** N_STEP) * q_next * (1 - db)

                loss = loss_fn(q_pred, q_tgt)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=10.0)
                optimizer.step()

            if step_count % SYNC_FREQ == 0:
                target.load_state_dict(agent.state_dict())

            epsilon = max(
                EPSILON_END,
                EPSILON_START - (EPSILON_START - EPSILON_END) * step_count / MAX_STEPS
            )

            if step_count >= MAX_STEPS:
                break

        train_profits.append(ep_profit)

        if step_count % 5000 == 0 or step_count >= MAX_STEPS:
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            torch.save(agent.state_dict(), f"{CHECKPOINT_DIR}/ckpt_{step_count}.pt")
            vr = _quick_val(agent, val_data, device)
            val_rewards.append((step_count, vr))
            print(f"Step {step_count:>7} | ε={epsilon:.3f} | "
                  f"Train ep profit={ep_profit:.2f} | Val reward={vr:.2f}")

    return agent, train_profits, val_rewards


def _quick_val(model, val_data, device, max_steps=500):
    model.eval()
    env   = TradingEnv(val_data, obs_bars=OBS_BARS, test=True)
    state = env.reset()
    total, done, step = 0.0, False, 0
    while not done and step < max_steps:
        state_t = torch.from_numpy(state).unsqueeze(0).float().to(device)
        with torch.no_grad():
            a = model(state_t).argmax(dim=1).item()
        state, reward, done = env.step(ACTIONS[a])
        total += reward
        step  += 1
    model.train()
    return total
