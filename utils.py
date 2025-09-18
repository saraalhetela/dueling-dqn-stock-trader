# utils.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_profits(profits, title="Cumulative Profit Over Time", filename="profits.png"):
    """
    Plot cumulative profits over time.
    """
    # Create folder if it doesn't exist
    os.makedirs("plots", exist_ok=True)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(profits)
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Profit (%)")
    plt.grid(True)

    # Save to file
    save_path = os.path.join("plots", filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # Show the plot (works in notebook with %matplotlib inline)
    plt.show()

    print(f"📂 Plot saved to {save_path}")
    plt.show()


def preprocess_state(state, input_shape, add_noise=False):
    """
    Reshape state for neural network input.
    Optionally add small noise for exploration.
    """
    if add_noise:
        return state.reshape(*input_shape) + np.random.rand(*input_shape) / 100.0
    return state.reshape(*input_shape)

def n_step_return(rewards, gamma, next_q=0.0, done=False):
    """
    Compute N-step discounted return.
    """
    ret = 0.0
    for i, r in enumerate(rewards):
        ret += (gamma ** i) * r
    if not done:
        ret += (gamma ** len(rewards)) * next_q
    return ret


