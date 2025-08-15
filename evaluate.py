import pandas as pd
from environment.stock_env import StockEnv
from agent.dqn_agent import DQNAgent
from config import Config

data = pd.read_csv("data/stock_prices.csv")
env = StockEnv(data)
agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, Config)

state = env.reset()
done = False
total_reward = 0
while not done:
    action = agent.select_action(state, epsilon=0)  # deterministic
    state, reward, done, _ = env.step(action)
    total_reward += reward

print(f"Total evaluation reward: {total_reward}")

