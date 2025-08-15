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

# Load trained model
agent.load_model("models/dqn_model.pth")

state = env.reset()
done = False

portfolio_values = []
positions = []  # 1 = buy, -1 = sell, 0 = hold
rewards_eval = []

while not done:
    action = agent.select_action(state, epsilon=0)  # deterministic
    next_state, reward, done, info = env.step(action)
    state = next_state

    portfolio_values.append(info.get("portfolio_value", 0))
    positions.append(action - 1)  # assuming action 0=hold,1=buy,2=sell
    rewards_eval.append(reward)

# Save portfolio balance plot
plt.plot(portfolio_values)
plt.xlabel("Step")
plt.ylabel("Portfolio Value")
plt.title("Portfolio Balance Over Time")
plt.savefig("plots/balance.png")
plt.close()

# Save positions plot
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label="Stock Price")
buy_signals = [i if pos==1 else None for i, pos in enumerate(positions)]
sell_signals = [i if pos==-1 else None for i, pos in enumerate(positions)]
plt.scatter(buy_signals, data['Close'][buy_signals], marker="^", color="g", label="Buy")
plt.scatter(sell_signals, data['Close'][sell_signals], marker="v", color="r", label="Sell")
plt.xlabel("Step")
plt.ylabel("Price")
plt.title("Trading Actions")
plt.legend()
plt.savefig("plots/positions.png")
plt.close()

# Save evaluation rewards
plt.plot(rewards_eval)
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("Evaluation Rewards")
plt.savefig("plots/rewards_eval.png")
plt.close()

print(f"Total evaluation reward: {sum(rewards_eval)}")
