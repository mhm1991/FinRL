"""
FinRL XAU/USD Part 5: Rule-based baseline comparison

Compares PPO backtest results against simple long/flat baselines:
buy-and-hold, SMA crossover, RSI, MACD, breakout, mean reversion, random, and
always-flat.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from FinRL_XAUUSD_common import DATA_DIR
from FinRL_XAUUSD_common import RESULTS_DIR
from FinRL_XAUUSD_common import build_rule_based_baselines
from FinRL_XAUUSD_common import buy_and_hold_baseline
from FinRL_XAUUSD_common import ensure_xau_dirs
from FinRL_XAUUSD_common import load_split
from FinRL_XAUUSD_common import summarize_equity_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare XAU rule-based baselines.")
    parser.add_argument("--initial-amount", type=float, default=100000.0)
    parser.add_argument("--cost-pct", type=float, default=0.0002)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--trade-file", default=str(DATA_DIR / "xau_trade.csv"))
    parser.add_argument(
        "--ppo-result-file",
        default=str(RESULTS_DIR / "backtest_result.csv"),
        help="Optional PPO backtest result CSV produced by FinRL_XAUUSD_3_backtest.py.",
    )
    return parser.parse_args()


def load_existing_ppo(path: str) -> pd.DataFrame:
    try:
        result = pd.read_csv(path, index_col=0)
    except FileNotFoundError:
        return pd.DataFrame()
    keep_cols = [column for column in result.columns if column != "buy_hold"]
    return result[keep_cols] if keep_cols else pd.DataFrame()


def main() -> None:
    args = parse_args()
    ensure_xau_dirs()

    trade = load_split(args.trade_file)
    buy_hold = buy_and_hold_baseline(trade, args.initial_amount)
    baselines = build_rule_based_baselines(
        trade,
        initial_amount=args.initial_amount,
        cost_pct=args.cost_pct,
        seed=args.random_seed,
    )
    ppo = load_existing_ppo(args.ppo_result_file)

    result = pd.concat([buy_hold, baselines, ppo], axis=1, join="inner").dropna()
    metrics = summarize_equity_curves(result).sort_values(
        "sharpe",
        ascending=False,
    )

    result.to_csv(RESULTS_DIR / "baseline_comparison.csv")
    metrics.to_csv(RESULTS_DIR / "baseline_comparison_metrics.csv")

    plt.rcParams["figure.figsize"] = (15, 6)
    result.plot()
    plt.title("XAU Strategy Baseline Comparison")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.savefig(RESULTS_DIR / "baseline_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n=== Baseline Comparison Metrics ===")
    print(metrics)
    print(f"\nSaved baseline comparison outputs under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
