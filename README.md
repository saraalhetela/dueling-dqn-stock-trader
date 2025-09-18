# Dueling Conv1D DQN Trading Agent

[![Python Version](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1-red)](https://pytorch.org/)
[![GitHub Issues](https://img.shields.io/github/issues/saraalhetela/deep-dqn-stock-trader)](https://github.com/your-username/deep-dqn-stock-trader/issues)

A modular Python implementation of a **Dueling Conv1D DQN agent** for stock trading. This framework allows you to train and evaluate reinforcement learning agents on historical stock data using PyTorch and Gym-style environments.

## Features

- Dueling Conv1D Q-Network
- N-step returns
- Replay buffer (deque)
- Reward shaping for profitable trades
- Evaluation scripts with action correctness metrics
- Checkpoints and cumulative profit plotting
- GPU support via PyTorch (CUDA)

---

## Installation
Clone the repository:
```bash
git clone https://github.com/saraalhetela/dueling-dqn-stock-trader.git
cd deep-dqn-stock-trader
```

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```
> [!NOTE]
> This project has been tested with Python 3.10+ and PyTorch 2.1.


## Configuration
### 1- Data: 
Place your stock CSV in the data/ folder

```bash
RL_Trading_Agent/data/AAPL_data.csv
```

### 2- Hyperparameters:
All hyperparameters and settings are in config.py:

```bash
DEVICE = "cuda"  # or "cpu"
DATA_PATH = "./data/AAPL_data.csv"
CHECKPOINT_DIR = "./checkpoints"
OBS_BARS = 50
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 0.0001
MEMORY_SIZE = 100000
SYNC_FREQ = 1000
MAX_STEPS = 150000
EPSILON_START = 1.0
EPSILON_END = 0.1
N_STEP = 2
```

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
### Training and Evaluating the Agent
```bash
python main.py
```
This will:

+ Load and preprocess the data
+ Train the RL agent
+ Evaluate its performance
+ Plot cumulative profits
+ create plots/ folder and save plots to plots/ as PNG images

## Running in a Jupyter Notebook

You can also train and evaluate the agent inside a notebook.

### Clone the repository

```bash
!git clone https://github.com/saraalhetela/dueling-dqn-stock-trader.git
%cd dueling-dqn-stock-trader
```

### Add your stock CSV file into the data/ folder 

e.g., data/AAPL_data.

### Install requirements

```bash
!pip install -r requirements.txt
```

### Set up notebook plotting

```bash
%matplotlib inline
```

### mport project modules

```bash
from config import *
from environment import TradingEnv, ACTIONS
from model import DuelingConv1D
from train import train_agent
from evaluate import evaluate_agent
from utils import preprocess_state, plot_profits
```

### Run training and evaluation

```bash
!python main.py
```

Alternatively, you can call main() directly from inside your notebook

```bash
from main import main
main()
```
This way, all output (logs + plots) will display inside your notebook

## Project Structure

```
dueling-dqn-stock-trader/
├── README.md
├── config.py
├── data_preprocessing.py
├── environment.py
├── evaluate.py
├── main.py 
├── model.py
├── requirements.txt
├── train.py
└── utils.py

```

+ **config.py** — Hyperparameters and settings
+ **data_preprocessing.py** — Load and normalize stock data
+ **environment.py** — Trading environment
+ **evaluate.py** — Evaluation scripts
+ **main.py** — Entry point: runs training, evaluation, and logging using all modules
+ **model.py** — Dueling Conv1D network
+ **train.py** — Training loop
+ **utils.py** — helper functions: preprocess_state, N-step return computation, plotting.

## Results
After training for 150,000 steps on Apple (AAPL) stock data, the agent learned to trade with consistent profit.

### Cumulative Profit Over Time

Below is an example plot showing cumulative profit during evaluation:

<img width="571" height="455" alt="Untitled" src="https://github.com/user-attachments/assets/6e95a1dd-990d-4060-a89b-29c72757b483" />


## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## Data Source

The stock price data used in this project is sourced from [Kaggle: S&P 500 5-Year Individual Stocks](https://www.kaggle.com/datasets/szrlee/stock-time-series-5yr)  
Specifically, the AAPL stock data CSV: `individual_stocks_5yr/AAPL_data.csv`.

Please follow the original license/terms of use from Kaggle when using this data.


## License

MIT License. See LICENSE for details.


