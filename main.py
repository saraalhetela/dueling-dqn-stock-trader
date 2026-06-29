import torch
from config import *
from data_preprocessing import load_and_preprocess
from environment import ACTIONS
from model import DuelingConv1D
from train import train_agent
from evaluate import evaluate_agent, plot_results
import pandas as pd

def main():
    print(f"Device: {DEVICE}")

    # Raw data for B&H baseline (before normalisation)
    raw_df = pd.read_csv(DATA_PATH)
    raw_df = raw_df.drop(columns=["volume", "Name"], errors="ignore")

    print("\nLoading and splitting data...")
    train_data, val_data, test_data = load_and_preprocess(
        DATA_PATH, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO
    )

    # Raw test slice for buy-and-hold calculation
    n = len(raw_df)
    test_start = int(n * (TRAIN_RATIO + VAL_RATIO))
    raw_test_df = raw_df.iloc[test_start:].reset_index(drop=True)

    print("\nInitializing model...")
    model = DuelingConv1D(
        input_channels=5,
        output_size=len(ACTIONS),
        obs_bars=OBS_BARS
    ).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTraining...")
    trained_model, train_profits, val_rewards = train_agent(
        model, train_data, val_data, device=DEVICE
    )

    print("\nEvaluating on TEST set...")
    cum_profits, action_counts, total_reward, bh_return, trades = evaluate_agent(
        trained_model, test_data, raw_test_df, device=DEVICE, obs_bars=OBS_BARS
    )

    plot_results(cum_profits, bh_return, train_profits, trades)

    # Save results for README generation
    results = {
        "agent_reward": total_reward,
        "bh_return": bh_return,
        "alpha": total_reward - bh_return,
        "n_trades": len(trades),
        "win_rate": len([t for t in trades if t[2]>0]) / len(trades) * 100 if trades else 0,
        "avg_pnl": sum(t[2] for t in trades) / len(trades) if trades else 0,
        "train_bars": len(train_data),
        "val_bars": len(val_data),
        "test_bars": len(test_data),
        "action_counts": action_counts,
        "total_steps": sum(action_counts.values()),
    }
    import json
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results.json")

if __name__ == "__main__":
    main()
