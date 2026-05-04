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
from FinRL_XAUUSD_common import ensure_xau_dirs
from FinRL_XAUUSD_common import load_split
from FinRL_XAUUSD_common import parse_algorithms

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


def buy_and_hold_baseline(trade: pd.DataFrame, initial_amount: float) -> pd.Series:
    prices = trade[["date", "close"]].drop_duplicates("date").set_index("date")["close"]
    return prices.div(prices.iloc[0]).mul(initial_amount).rename("buy_hold")


def max_drawdown(values: pd.Series) -> float:
    drawdown = values.div(values.cummax()).sub(1.0)
    return float(drawdown.min())


def summarize_results(result: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in result.columns:
        series = result[name].dropna()
        returns = series.pct_change().dropna()
        sharpe = 0.0
        if returns.std() != 0:
            sharpe = float((252**0.5) * returns.mean() / returns.std())
        rows.append(
            {
                "strategy": name,
                "start_value": float(series.iloc[0]),
                "end_value": float(series.iloc[-1]),
                "total_return": float(series.iloc[-1] / series.iloc[0] - 1.0),
                "max_drawdown": max_drawdown(series),
                "sharpe": sharpe,
            }
        )
    return pd.DataFrame(rows).set_index("strategy")


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

    result = pd.concat(result_series.values(), axis=1, join="inner").dropna()
    metrics = summarize_results(result)

    result.to_csv(RESULTS_DIR / "backtest_result.csv")
    metrics.to_csv(RESULTS_DIR / "backtest_metrics.csv")

    plt.rcParams["figure.figsize"] = (15, 5)
    result.plot()
    plt.title("FinRL XAU Backtest")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.savefig(RESULTS_DIR / "backtest_result.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n=== Backtest Metrics ===")
    print(metrics)
    print(f"\nSaved outputs under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
