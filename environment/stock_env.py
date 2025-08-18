import gym
from gym import spaces
import numpy as np

class StockEnv(gym.Env):
        def __init__(self, data, obs_bars=10, test=False, commission_perc=0.1):
        self.data = data
        self.obs_bars = obs_bars
        self.have_position = False
        self.open_price = 0
        self.test = test
        self.commission_perc = commission_perc
        if not test:
            self.curr_step = np.random.choice(self.data.high.shape[0] - self.obs_bars * 10) + self.obs_bars
        else:
            self.curr_step = self.obs_bars
        self.state = self.data[self.curr_step - self.obs_bars: self.curr_step]

    def step(self, action):
        reward = 0
        done = False
        relative_close = self.state["close"][self.curr_step - 1]
        open_price_now = self.state["open"][self.curr_step - 1]
        close = open_price_now * (1 + relative_close)

        trade_occurred = False

        if action == "buy" and not self.have_position:
            self.have_position = True
            self.open_price = close
            trade_occurred = True
            # Minor penalty to prevent overtrading
            reward -= self.commission_perc * 0.5

        elif action == "close" and self.have_position:
            trade_occurred = True
            trade_profit = close - self.open_price
            reward += 100.0 * trade_profit / self.open_price  # Profit-focused reward
            reward -= self.commission_perc  # Commission fee
            self.have_position = False
            self.open_price = 0.0
            if not self.test:
                done = True

        elif action == "do_nothing":
            if self.have_position:
                unrealized_profit = (close - self.open_price) / self.open_price
                reward += 0.1 * unrealized_profit  # Encourage holding profitable positions

        # Slight penalty for taking actions to avoid overtrading
        if trade_occurred:
            reward -= 0.01

        self.curr_step += 1
        self.state = self.data[self.curr_step - self.obs_bars: self.curr_step]

        if self.curr_step >= len(self.data) - 1:
            done = True

        state = np.zeros((5, self.obs_bars), dtype=np.float32)
        state[0] = self.state.high.to_list()
        state[1] = self.state.low.to_list()
        state[2] = self.state.close.to_list()
        state[3] = int(self.have_position)
        if self.have_position:
            state[4] = (close - self.open_price) / self.open_price

        return state, reward, done

