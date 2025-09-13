# environment.py
import numpy as np

ACTIONS = {0: "do_nothing", 1: "buy", 2: "close"}

class TradingEnv:
    def __init__(self, data, obs_bars=50, test=False, commission_perc=0.1):
        self.data = data
        self.obs_bars = obs_bars
        self.test = test
        self.commission_perc = commission_perc
        self.have_position = False
        self.open_price = 0
        self.reset()

    def reset(self):
        if not self.test:
            self.curr_step = np.random.choice(len(self.data) - self.obs_bars * 10) + self.obs_bars
        else:
            self.curr_step = self.obs_bars
        self.have_position = False
        self.open_price = 0
        self.state = self.data[self.curr_step - self.obs_bars: self.curr_step]
        return self.get_state()

    def step(self, action_name):
        reward = 0
        done = False

        relative_close = self.state["close"].iloc[-1]
        open_price_now = self.state["open"].iloc[-1]
        close_price = open_price_now * (1 + relative_close)

        trade_occurred = False

        if action_name == "buy" and not self.have_position:
            self.have_position = True
            self.open_price = close_price
            trade_occurred = True
            reward += 1.0  # encourage learning
        elif action_name == "close" and self.have_position:
            trade_occurred = True
            trade_profit = close_price - self.open_price
            reward += 100.0 * trade_profit / self.open_price
            reward -= self.commission_perc
            self.have_position = False
            self.open_price = 0
        elif action_name == "do_nothing" and self.have_position:
            unrealized_profit = (close_price - self.open_price) / self.open_price
            reward += 0.1 * unrealized_profit

        if trade_occurred:
            reward -= 0.01

        self.curr_step += 1
        if self.curr_step >= len(self.data) - 1:
            done = True
        self.state = self.data[self.curr_step - self.obs_bars: self.curr_step]

        return self.get_state(), reward, done

    def get_state(self):
        state = np.zeros((5, self.obs_bars), dtype=np.float32)
        state[0] = self.state.high.to_numpy()
        state[1] = self.state.low.to_numpy()
        state[2] = self.state.close.to_numpy()
        state[3] = int(self.have_position)
        if self.have_position:
            last_close = self.state["open"].iloc[-1] * (1 + self.state["close"].iloc[-1])
            state[4] = (last_close - self.open_price) / self.open_price
        return state
