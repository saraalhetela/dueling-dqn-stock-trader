class Config:
    STOCK_CSV = "data/AAPL_data.csv"
    
    # Environment
    initial_balance = 10000
    max_steps = 200

    # DQN
    gamma = 0.99
    lr = 1e-4
    batch_size = 64
    target_update = 10
    memory_size = 10000
    n_step = 3
    epsilon = 0.1

    # Training
    EPISODES = 500
