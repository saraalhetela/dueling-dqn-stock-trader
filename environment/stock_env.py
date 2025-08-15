import gym
from gym import spaces
import numpy as np

class StockEnv(gym.Env):
    def __init__(self, data, initial_balance=10000):
        super(StockEnv, self).__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.reset()
        self.action_space = spaces.Discrete(3)  # Buy, Hold, Sell
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(len(data.columns)+2,), dtype=np.float32)

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        self.total_profit = 0
        return self._get_observation()

    def _get_observation(self):
        obs = np.array([self.balance, self.position] + self.data.iloc[self.current_step].tolist())
        return obs

    def step(self, action):
        # Implement buy/sell logic
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        reward = 0  # Define your reward function
        return self._get_observation(), reward, done, {}

