
# Dueling Conv1D DQN Trading Agent
 
[![Python Version](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1-red)](https://pytorch.org/)
 
A modular PyTorch implementation of a **Dueling Double DQN agent** for stock trading, with N-step returns, proper chronological train/val/test splits, and a buy-and-hold baseline for honest benchmarking.
 
---
 
## Architecture
 
- **Dueling Conv1D Q-Network** — shared 1D convolutional feature extractor over a 50-bar lookback window, with separate value and advantage heads
- **Double DQN** — action selection by online network, Q-value evaluation by target network, reducing overestimation bias
- **N-step returns** (N=2) — multi-step TD targets for faster credit assignment
- **Huber loss** — more stable than MSE against large TD errors early in training
- **Gradient clipping** — prevents exploding gradients destabilising learned weights
- **CPU replay buffer** — transitions stored as numpy arrays and moved to GPU only at sample time, preventing VRAM exhaustion over long training runs
---
 
## Results
 
Trained on Apple (AAPL) daily OHLC data, Feb 2013 – Feb 2018 (1,259 bars).  
Data source: [Kaggle — S&P 500 5-Year Individual Stocks](https://www.kaggle.com/datasets/camnugent/sandp500) → `individual_stocks_5yr/AAPL_data.csv`  
Hardware: CUDA GPU, 150,000 training steps, ε decayed from 1.0 → 0.10.
 
### Data split (chronological — never shuffled)
 
| Split | Bars | Purpose |
|-------|------|---------|
| Train | 881  | Agent learning |
| Val   | 189  | Generalisation check during training |
| **Test** | **189** | **Final evaluation — never seen during training** |
 
### Test set performance (held-out data)
 
| Metric | Value |
|--------|-------|
| Agent cumulative reward | **+18.15%** |
| Buy-and-hold return | +3.85% |
| **Alpha vs buy-and-hold** | **+14.30%** |
| Total trades | 20 |
| Win rate | 70.0% |
| Avg trade P&L | +1.01% |
 
The agent completed 20 trades on the held-out test period with a 70% win rate and positive average P&L per trade, outperforming a passive buy-and-hold strategy by 14.30 percentage points over the same window.
 
### Test set: agent cumulative reward vs buy-and-hold
 
![Test vs Buy-and-Hold](plots/test_vs_bh.png)
 
The agent stays above the buy-and-hold line for the majority of the test period. The staircase pattern reflects the long-only strategy — reward accumulates in discrete steps at each trade close. Green shading marks profitable trade windows, red marks losing ones. The single mid-period drawdown (steps ~25–60) is the agent's only extended spell below its prior peak before recovering strongly.
 
### Training episode rewards
 
![Training Profits](plots/train_profits.png)
 
Consistent upward trend across 350 episodes with no plateau or collapse. The 20-episode moving average rises from ~0 to ~250, confirming the agent is learning and not just benefiting from reward shaping on held positions.
 
### Validation reward during training
 
![Validation Curve](plots/val_curve.png)
 
---
 
## Installation
 
```bash
git clone https://github.com/your-username/dueling-dqn-stock-trader.git
cd dueling-dqn-stock-trader
python -m venv venv
source venv/bin/activate      # Linux / Mac
# venv\Scripts\activate       # Windows
pip install -r requirements.txt
```
 
## Data
 
Place your stock CSV at:
 
```
data/AAPL_data.csv
```
 
Expected columns: `date, open, high, low, close, volume, Name`
 
Download from: [Kaggle S&P 500 dataset](https://www.kaggle.com/datasets/camnugent/sandp500) → `individual_stocks_5yr/AAPL_data.csv`
 
## Configuration
 
All hyperparameters are in `config.py`:
 
```python
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
OBS_BARS      = 50        # lookback window in bars
BATCH_SIZE    = 64
GAMMA         = 0.99
LEARNING_RATE = 1e-4
MEMORY_SIZE   = 100_000
SYNC_FREQ     = 1_000     # target network sync frequency (steps)
MAX_STEPS     = 150_000   # total training environment steps
EPSILON_START = 1.0
EPSILON_END   = 0.05
N_STEP        = 2
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
```
 
## Usage
 
```bash
python main.py
```
 
This will:
1. Load and split data chronologically (70 / 15 / 15)
2. Train the agent, printing a checkpoint log and saving weights every 5,000 steps
3. Evaluate on the held-out test set only
4. Save plots to `plots/` and metrics to `results.json`
---
 
## Project structure
 
```
dueling-dqn-stock-trader/
├── config.py               — hyperparameters and device config
├── data_preprocessing.py   — load, clean, and chronologically split data
├── environment.py          — Gym-style trading environment (long-only, 0.1% commission)
├── model.py                — Dueling Conv1D network with dynamic flat-size computation
├── train.py                — training loop (Double DQN, N-step, Huber loss, grad clipping)
├── evaluate.py             — test evaluation with buy-and-hold baseline and trade logging
├── utils.py                — plotting utilities
├── main.py                 — entry point
├── requirements.txt
├── plots/
│   ├── test_vs_bh.png
│   ├── train_profits.png
│   └── val_curve.png
└── data/
    └── AAPL_data.csv
```
 
---
 
## Known limitations and future work
 
- **Replay buffer:** uniform random sampling. Prioritised Experience Replay (PER) would improve sample efficiency by replaying high-error transitions more frequently.
- **State representation:** position flag and unrealised P&L are broadcast scalars across all 50 timesteps. A cleaner design would pass portfolio features through a dedicated FC branch separate from the convolutional feature extractor.
- **Single asset, long-only:** no short positions, no portfolio-level risk controls, no position sizing beyond all-in / all-out.
- **Commission model:** flat 0.1% per trade — a simplification that ignores slippage and market impact.
- **Single stock, single run:** results are on one ticker over one historical period. Testing across multiple assets and time periods would strengthen the generalisation claim.
---
 
## License
 
MIT License. See LICENSE for details.
