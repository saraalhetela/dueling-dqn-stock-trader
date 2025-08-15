import pandas as pd
from config import Config
from environment.stock_env import StockEnv
from agent.dqn_agent import DQNAgent

data = pd.read_csv("data/stock_prices.csv")
env = StockEnv(data)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim, Config)

for episode in range(Config.num_episodes):
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state, epsilon=0.1)
        next_state, reward, done, _ = env.step(action)
        agent.memory.push(state, action, reward, next_state, done)
        state = next_state
        agent.optimize()
    if episode % Config.target_update == 0:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())

