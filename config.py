class Config:
    STOCK_CSV = "data/AAPL_data.csv"
    batch_size = 64
    memory_size = 10000
    gamma = 0.99
    learning_rate = 0.0001
    epsilon = 1.

    
    initial_balance = 10000
    max_steps = 200
    target_update = 10
    n_step = 3
    EPISODES = 500



