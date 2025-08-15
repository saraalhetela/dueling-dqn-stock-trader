import os
import pandas as pd
import matplotlib.pyplot as plt
from environment.stock_env import StockEnv
from agent.dqn_agent import DQNAgent
from config import Config

# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)

# Load stock data
data = pd.read_csv(Config.STOCK_CSV)
env = StockEnv(data)
agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, Config)

all_rewards = []

for episode in range(Config.EPISODES):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        agent.learn()
        state = next_state
        total_reward += reward

    all_rewards.append(total_reward)
    print(f"Episode {episode + 1}/{Config.EPISODES}, Reward: {total_reward}")

# Save trained model
os.makedirs("models", exist_ok=True)
agent.save_model("models/dqn_model.pth")

# Plot rewards
plt.plot(all_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Rewards")
plt.savefig("plots/rewards.png")
plt.close()
