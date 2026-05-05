"""
FinRL XAU/USD Part 3: Backtest

Loads trained FinRL agents, compares them with a buy-and-hold XAU baseline,
and saves both equity curves and summary metrics.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

from FinRL_XAUUSD_common import DATA_DIR
from FinRL_XAUUSD_common import MODEL_DIR
from FinRL_XAUUSD_common import RESULTS_DIR
from FinRL_XAUUSD_common import build_env_kwargs
from FinRL_XAUUSD_common import buy_and_hold_baseline
from FinRL_XAUUSD_common import ensure_xau_dirs
from FinRL_XAUUSD_common import load_split
from FinRL_XAUUSD_common import parse_algorithms
from FinRL_XAUUSD_common import summarize_equity_curves
from FinRL_XAUUSD_common import summarize_long_only_trades

MODEL_CLASSES = {
    "a2c": A2C,
    "ddpg": DDPG,
    "ppo": PPO,
    "td3": TD3,
    "sac": SAC,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest FinRL XAU agents.")
    parser.add_argument("--algo", default="ppo", help="ppo, sac, td3, ddpg, a2c, or all")
    parser.add_argument("--initial-amount", type=float, default=100000.0)
    parser.add_argument("--hmax", type=int, default=100)
    parser.add_argument("--cost-pct", type=float, default=0.0002)
    parser.add_argument("--reward-scaling", type=float, default=1e-4)
    parser.add_argument("--trade-file", default=str(DATA_DIR / "xau_trade.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_xau_dirs()

    trade = load_split(args.trade_file)
    env_kwargs = build_env_kwargs(
        trade,
        initial_amount=args.initial_amount,
        hmax=args.hmax,
        cost_pct=args.cost_pct,
        reward_scaling=args.reward_scaling,
    )

    result_series = {"buy_hold": buy_and_hold_baseline(trade, args.initial_amount)}
    trade_metric_frames = []

    for algo in parse_algorithms(args.algo):
        model_path = MODEL_DIR / f"agent_{algo}"
        zip_path = model_path.with_suffix(".zip")
        if not zip_path.exists():
            print(f"Skipping {algo}: missing {zip_path}")
            continue

        model = MODEL_CLASSES[algo].load(str(model_path))
        trade_env = StockTradingEnv(df=trade, **env_kwargs)
        account_value, actions = DRLAgent.DRL_prediction(model=model, environment=trade_env)
        account_series = account_value.set_index("date")["account_value"].rename(algo)
        result_series[algo] = account_series
        actions.to_csv(RESULTS_DIR / f"actions_{algo}.csv")
        trade_metrics = summarize_long_only_trades(
            trade,
            actions,
            cost_pct=args.cost_pct,
        )
        trade_metrics.insert(0, "strategy", algo)
        trade_metric_frames.append(trade_metrics)

    result = pd.concat(result_series.values(), axis=1, join="inner").dropna()
    metrics = summarize_equity_curves(result)
    trade_metrics = (
        pd.concat(trade_metric_frames, ignore_index=True).set_index("strategy")
        if trade_metric_frames
        else pd.DataFrame()
    )

    result.to_csv(RESULTS_DIR / "backtest_result.csv")
    metrics.to_csv(RESULTS_DIR / "backtest_metrics.csv")
    if not trade_metrics.empty:
        trade_metrics.to_csv(RESULTS_DIR / "trade_metrics.csv")

    plt.rcParams["figure.figsize"] = (15, 5)
    result.plot()
    plt.title("FinRL XAU Backtest")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.savefig(RESULTS_DIR / "backtest_result.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n=== Backtest Metrics ===")
    print(metrics)
    if not trade_metrics.empty:
        print("\n=== Trade Metrics ===")
        print(trade_metrics)
    print(f"\nSaved outputs under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
