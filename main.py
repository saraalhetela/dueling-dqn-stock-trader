# main.py

import torch
from config import *
from data_preprocessing import load_and_preprocess
from environment import TradingEnv, ACTIONS
from model import DuelingConv1D
from train import train_agent  # We'll slightly modify train.py to expose a train_agent() function
from evaluate import evaluate_agent
from utils import plot_profits

def main():
    print("📊 Loading and preprocessing data...")
    data = load_and_preprocess(DATA_PATH)

    print("🤖 Initializing model...")
    device = DEVICE
    input_channels = 5
    output_size = len(ACTIONS)
    model = DuelingConv1D(input_channels, output_size).to(device)

    print("⚡ Starting training...")
    # This will return the trained model and profits history
    trained_model, profits = train_agent(model, data, device=device)

    print("✅ Training completed. Evaluating agent...")
    total_rewards, action_counts = evaluate_agent(trained_model, data, device=device, episodes=100)

    print("📈 Plotting cumulative profits...")
    plot_profits(profits)

    print("🎯 Evaluation finished.")
    print(f"Total Rewards Avg: {sum(total_rewards)/len(total_rewards):.2f}")
    print("Action distribution:")
    for action_name, count in action_counts.items():
        print(f"  {action_name}: {count} ({count / sum(action_counts.values()) * 100:.2f}%)")

if __name__ == "__main__":
    main()
