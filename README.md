# Deep Q-Network Stock Trader

[![Python Version](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1-red)](https://pytorch.org/)
[![GitHub Issues](https://img.shields.io/github/issues/saraalhetela/deep-dqn-stock-trader)](https://github.com/your-username/deep-dqn-stock-trader/issues)

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
```
Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```
Optional: If you have CUDA-enabled GPU, make sure your PyTorch version supports GPU for faster training.

## Configuration
All hyperparameters and settings are in config.py:
```bash
# Example: config.py
STOCK_CSV = "data/AAPL_data.csv"
INITIAL_BALANCE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 0.001
EPISODES = 500
```
Update STOCK_CSV to point to your stock data CSV file.

## Data Format

Place your stock price CSV in the `data/` folder:
```bash
data/stock_prices.csv
```
The CSV should include:
| Date       | Open  | High  | Low  | Close | Volume |
| ---------- | ----- | ----- | ---- | ----- | ------ |
| 2025-01-01 | 100.0 | 102.0 | 99.5 | 101.0 | 10000  |

## Usage
### Training the Agent
```bash
python train.py
```
+ Trains the DQN agent on historical stock data
+ Saves the trained model to `models/`
+ Generates reward plots in `plots/`
  
Example reward plot generated:

<img width="583" height="455" alt="rewards" src="https://github.com/user-attachments/assets/97a7e491-f8c7-4010-b453-fabd01b1b7b4" />


### Evaluating the Agent
```bash
python evaluate.py
```
+ Loads a trained model
+ Simulates trading on historical data
+ Generates trading performance plots:
  1. `plots/balance.png` — Shows portfolio balance over time
  2. `plots/positions.png` — Shows buy/sell actions on stock price
  3. `plots/rewards_eval.png` — Reward per time step during evaluation

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
├── plots/        # stores reward/performance plots
└── data/         # stock CSV files
```

+ **config.py** — Hyperparameters and environment settings
+ **environment/** — Custom stock trading environment
+ **agent/** — DQN agent and neural network
+ **utils/** — Helper modules (e.g., replay memory)
+ **train.py** — Script to train the agent
+ **evaluate.p** — Script to evaluate a trained agent
+ **plots/** — Stores reward and performance plots
+ **data/** — Folder for stock price CSV files

## Data Source

The stock price data used in this project is sourced from [Kaggle: S&P 500 5-Year Individual Stocks](https://www.kaggle.com/datasets/szrlee/stock-time-series-5yr)  
Specifically, the AAPL stock data CSV: `individual_stocks_5yr/AAPL_data.csv`.

Please follow the original license/terms of use from Kaggle when using this data.


## License

MIT License. See LICENSE for details.

## Contributing

1. Fork the repository
2. Create your feature branch (git checkout -b feature-name)
3. Commit your changes (git commit -m "Add feature")
4. Push to the branch (git push origin feature-name)
5. Open a Pull Request


