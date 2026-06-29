import torch
DEVICE         = "cpu"
DATA_PATH      = ".data/AAPL_data.csv"
CHECKPOINT_DIR = "./checkpoints"
OBS_BARS       = 50
BATCH_SIZE     = 64
GAMMA          = 0.99
LEARNING_RATE  = 1e-4
MEMORY_SIZE    = 20_000
SYNC_FREQ      = 300
MAX_STEPS      = 5_000
EPSILON_START  = 1.0
EPSILON_END    = 0.05
N_STEP         = 2
TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.15
