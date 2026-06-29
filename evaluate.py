# evaluate.py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from environment import TradingEnv, ACTIONS


def buy_and_hold_return(raw_test_df):
    """
    Simple buy-and-hold benchmark on the raw (un-normalised) test data.
    Buys at the first open, sells at the last close.
    Returns total % return.
    """
    first_open  = raw_test_df["open"].iloc[0]
    last_close  = raw_test_df["close"].iloc[-1]
    return (last_close - first_open) / first_open * 100.0


def evaluate_agent(model, test_data, raw_test_df, device="cpu", obs_bars=50, max_steps=5000):
    """
    Evaluate trained agent on the held-out test split.

    Single sequential pass over all test bars.
    Compares against buy-and-hold baseline.

    Args:
        model:       trained DuelingConv1D
        test_data:   normalised test DataFrame (used by TradingEnv)
        raw_test_df: raw (un-normalised) test DataFrame (used for B&H baseline)
        device:      torch device string
        obs_bars:    must match training config
        max_steps:   safety cap (set high — we want to traverse the whole test set)

    Returns:
        cumulative_profits: list of running cumulative profit at each step
        action_counts:      dict {action_name: count}
        total_reward:       float — agent's total reward over test period
        bh_return:          float — buy-and-hold % return over same period
        trades:             list of (entry_step, exit_step, pnl_pct) tuples
    """
    model.eval()
    env   = TradingEnv(test_data, obs_bars=obs_bars, test=True)
    state = env.reset()
    state = torch.from_numpy(state).unsqueeze(0).float().to(device)

    cumulative         = 0.0
    cumulative_profits = []
    action_counts      = {name: 0 for name in ACTIONS.values()}
    trades             = []
    entry_step         = None
    entry_price        = None
    done = False
    step = 0

    while not done and step < max_steps:
        with torch.no_grad():
            action_idx  = model(state).argmax(dim=1).item()
            action_name = ACTIONS[action_idx]

        # Track trade records
        if action_name == "buy" and entry_step is None:
            entry_step  = step
            entry_price = env._window["open"].iloc[-1] * (1 + env._window["close"].iloc[-1])
        elif action_name == "close" and entry_step is not None:
            exit_price = env._window["open"].iloc[-1] * (1 + env._window["close"].iloc[-1])
            pnl = (exit_price - entry_price) / entry_price * 100.0
            trades.append((entry_step, step, pnl))
            entry_step = entry_price = None

        action_counts[action_name] += 1
        next_state, reward, done = env.step(action_name)
        state = torch.from_numpy(next_state).unsqueeze(0).float().to(device)
        cumulative += reward
        cumulative_profits.append(cumulative)
        step += 1

    bh_return     = buy_and_hold_return(raw_test_df)
    total_actions = sum(action_counts.values()) or 1
    win_trades    = [t for t in trades if t[2] > 0]
    win_rate      = len(win_trades) / len(trades) * 100 if trades else 0.0
    avg_pnl       = np.mean([t[2] for t in trades]) if trades else 0.0

    print("\n" + "="*50)
    print("  TEST SET EVALUATION (held-out data, never seen in training)")
    print("="*50)
    print(f"  Steps evaluated    : {step}")
    print(f"  Agent total reward : {cumulative:.2f}%")
    print(f"  Buy-and-hold       : {bh_return:.2f}%")
    print(f"  Alpha vs B&H       : {cumulative - bh_return:+.2f}%")
    print(f"  Total trades       : {len(trades)}")
    print(f"  Win rate           : {win_rate:.1f}%")
    print(f"  Avg trade P&L      : {avg_pnl:.2f}%")
    print("-"*50)
    print("  Action distribution:")
    for name, count in action_counts.items():
        print(f"    {name:<12}: {count} ({count/total_actions*100:.1f}%)")
    print("="*50)

    model.train()
    return cumulative_profits, action_counts, cumulative, bh_return, trades


def plot_results(cumulative_profits, bh_return, train_profits, trades):
    """Save all result plots to plots/ folder."""
    os.makedirs("plots", exist_ok=True)

    # ── 1. Test cumulative profit vs B&H ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    steps = list(range(len(cumulative_profits)))
    ax.plot(steps, cumulative_profits, color="#2a78d6", linewidth=1.8, label="Agent")
    ax.axhline(bh_return, color="#e34948", linewidth=1.4, linestyle="--",
               label=f"Buy & Hold ({bh_return:.1f}%)")
    ax.axhline(0, color="#888780", linewidth=0.7, linestyle=":")

    # Mark trades
    for (entry, exit_, pnl) in trades:
        color = "#1baf7a" if pnl > 0 else "#e34948"
        ax.axvspan(entry, exit_, alpha=0.08, color=color)

    ax.set_title("Test Set: Agent Cumulative Profit vs Buy-and-Hold", fontsize=13)
    ax.set_xlabel("Test step")
    ax.set_ylabel("Cumulative reward (%)")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/test_vs_bh.png", dpi=150)
    plt.close()

    # ── 2. Training episode profits ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4))
    window = 20
    smoothed = np.convolve(train_profits, np.ones(window)/window, mode="valid")
    ax.plot(train_profits, color="#b4b2a9", linewidth=0.6, alpha=0.5, label="Raw")
    ax.plot(range(window-1, len(train_profits)), smoothed,
            color="#2a78d6", linewidth=1.8, label=f"{window}-ep moving avg")
    ax.set_title("Training Episode Profits", fontsize=13)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/train_profits.png", dpi=150)
    plt.close()

    print("\nPlots saved to plots/")
