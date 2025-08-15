# Deep Q-Network Stock Trader

[![Python Version](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1-red)](https://pytorch.org/)
[![GitHub Issues](https://img.shields.io/github/issues/your-username/deep-dqn-stock-trader)](https://github.com/your-username/deep-dqn-stock-trader/issues)

A modular Python implementation of a **Deep Q-Network (DQN) agent** for stock trading. This framework allows you to train and evaluate reinforcement learning agents on historical stock data using PyTorch and Gym-style environments.

## Features

- Custom stock trading environment (`StockEnv`) compatible with Gym API  
- Deep Q-Network agent with experience replay and optional N-step returns  
- Modular project structure for easy experimentation and extension  
- Visualization of trading performance and training rewards  

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/deep-dqn-stock-trader.git
cd deep-dqn-stock-trader

Create a virtual environment and install dependencies:

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```


## Usage
### Training the Agent
```bash
python train.py
```

The agent will train on the stock data defined in config.py and save the model and reward plots to plots/.

### Evaluating the Agent
```bash
python evaluate.py
```

Generates trading performance plots and prints evaluation metrics.
Project Structure

```
deep-dqn-stock-trader/
├── README.md
├── requirements.txt
├── config.py
├── environment/
│   ├── __init__.py
│   └── stock_env.py
├── agent/
│   ├── __init__.py
│   └── dqn_agent.py
├── utils/
│   ├── __init__.py
│   └── memory.py
├── train.py
├── evaluate.py
├── plots/
└── data/
```

+ **config.py** — Hyperparameters and environment settings
+ **environment/** — Custom stock trading environment
+ **agent/** — DQN agent and neural network
+ **utils/** — Helper modules (e.g., replay memory)
+ **train.py** — Script to train the agent
+ **evaluate.p** — Script to evaluate a trained agent
+ **plots/** — Stores reward and performance plots
+ **data/** — Folder for stock price CSV files

### Data Format

Place your stock price CSV in the data/ folder. Example:
```bash
data/stock_prices.csv
```

The CSV should include:
| Date       | Open  | High  | Low  | Close | Volume |
| ---------- | ----- | ----- | ---- | ----- | ------ |
| 2025-01-01 | 100.0 | 102.0 | 99.5 | 101.0 | 10000  |
