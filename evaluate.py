# evaluate.py
import torch
import numpy as np
from environment import TradingEnv, ACTIONS

def evaluate_agent(model, data, device="cuda", episodes=100, obs_bars=50, max_steps=500):
    model.eval()
    total_rewards = []
    action_counts = {name: 0 for name in ACTIONS.values()}
    win_count = 0

    for episode in range(episodes):
        env = TradingEnv(data, obs_bars=obs_bars, test=True)
        state = env.reset()
        state = torch.from_numpy(state).unsqueeze(0).float().to(device)
        episode_reward = 0
        done = False
        step = 0

        while not done and step < max_steps:
            with torch.no_grad():
                qvals = model(state)
                action_idx = torch.argmax(qvals, dim=1).item()
                action_name = ACTIONS[action_idx]

            action_counts[action_name] += 1
            next_state, reward, done = env.step(action_name)
            state = torch.from_numpy(next_state).unsqueeze(0).float().to(device)
            episode_reward += reward
            step += 1

        total_rewards.append(episode_reward)
        if episode_reward > 0:
            win_count += 1

    print(f"Tested over {episodes} episodes.")
    print(f"Average reward: {np.mean(total_rewards):.2f}")
    print(f"Win rate (>0 reward): {win_count / episodes * 100:.2f}%")
    print("Action distribution:")
    for action_name, count in action_counts.items():
        print(f"  {action_name}: {count} ({count / sum(action_counts.values()) * 100:.2f}%)")

    model.train()
    return total_rewards, action_counts
