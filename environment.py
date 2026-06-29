import numpy as np
ACTIONS = {0: "do_nothing", 1: "buy", 2: "close"}

class TradingEnv:
    def __init__(self, data, obs_bars=50, test=False, commission_perc=0.1):
        self.data = data
        self.obs_bars = obs_bars
        self.test = test
        self.commission_perc = commission_perc
        self._test_start = obs_bars
        self.reset()

    def reset(self):
        if self.test:
            self.curr_step = self._test_start
        else:
            max_start = len(self.data) - self.obs_bars - 1
            self.curr_step = np.random.randint(self.obs_bars, max(self.obs_bars + 1, max_start))
        self.have_position = False
        self.open_price = 0.0
        self._window = self.data.iloc[self.curr_step - self.obs_bars: self.curr_step]
        return self._get_state()

    def step(self, action_name):
        reward = 0.0
        done = False
        open_now  = self._window["open"].iloc[-1]
        rel_close = self._window["close"].iloc[-1]
        close_price = open_now * (1.0 + rel_close)

        if action_name == "buy" and not self.have_position:
            self.have_position = True
            self.open_price = close_price
            reward -= self.commission_perc / 100.0
        elif action_name == "close" and self.have_position:
            pnl_pct = (close_price - self.open_price) / self.open_price
            reward += pnl_pct * 100.0
            reward -= self.commission_perc
            self.have_position = False
            self.open_price = 0.0
        elif action_name == "do_nothing" and self.have_position:
            unrealised = (close_price - self.open_price) / self.open_price
            reward += 0.1 * unrealised

        self.curr_step += 1
        if self.curr_step >= len(self.data) - 1:
            done = True
            if self.test:
                self._test_start = self.obs_bars
        else:
            self._window = self.data.iloc[self.curr_step - self.obs_bars: self.curr_step]
        return self._get_state(), reward, done

    def _get_state(self):
        window = self.data.iloc[self.curr_step - self.obs_bars: self.curr_step]
        state = np.zeros((5, self.obs_bars), dtype=np.float32)
        state[0] = window["high"].to_numpy()
        state[1] = window["low"].to_numpy()
        state[2] = window["close"].to_numpy()
        state[3] = float(self.have_position)
        if self.have_position:
            last_close = window["open"].iloc[-1] * (1.0 + window["close"].iloc[-1])
            state[4] = (last_close - self.open_price) / self.open_price
        return state
