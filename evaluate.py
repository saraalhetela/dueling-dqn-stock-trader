import os
import pandas as pd
import matplotlib.pyplot as plt
from environment.stock_env import StockEnv
from agent.dqn_agent import DQNAgent
from config import Config

# Create plots folder if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Load data
data = pd.read_csv(Config.STOCK_CSV)

# Initialize environment and agent
env = StockEnv(data)
agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, Config)

# Reset environment
state = env.reset()
done = False

# Track metrics
balance_history = []
positions = []
rewards_eval = []

while not done:
    action = agent.select_action(state, epsilon=0)  # deterministic
    next_state, reward, done, info = env.step(action)

    balance_history.append(env.balance)
    positions.append((env.current_step, action))
    rewards_eval.append(reward)

    state = next_state

# --- PLOTS ---

# 1. Portfolio Balance
plt.figure()
plt.plot(balance_history)
plt.title("Portfolio Balance Over Time")
plt.xlabel("Time Step")
plt.ylabel("Balance ($)")
plt.grid(True)
plt.savefig("plots/balance.png")
plt.close()

# 2. Buy/Sell Positions
plt.figure()
plt.plot(data["Close"], label="Close Price")
buys = [step for step, act in positions if act == 1]
sells = [step for step, act in positions if act == 2]
plt.scatter(buys, data["Close"].iloc[buys], marker="^", color="g", label="Buy")
plt.scatter(sells, data["Close"].iloc[sells], marker="v", color="r", label="Sell")
plt.title("Buy/Sell Positions on Stock Price")
plt.xlabel("Time Step")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.savefig("plots/positions.png")
plt.close()

# 3. Rewards During Evaluation
plt.figure()
plt.plot(rewards_eval)
plt.title("Reward per Step (Evaluation)")
plt.xlabel("Time Step")
plt.ylabel("Reward")
plt.grid(True)
plt.savefig("plots/rewards_eval.png")
plt.close()

# Summary
print(f"Total evaluation reward: {sum(rewards_eval)}")
