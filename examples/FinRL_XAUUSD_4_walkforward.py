"""
FinRL XAU/USD Part 4: Walk-forward validation

Retrains agents on rolling windows and evaluates each model on the following
out-of-sample period.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from stable_baselines3.common.logger import configure

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split

from FinRL_XAUUSD_2_train import MODEL_PARAMS
from FinRL_XAUUSD_3_backtest import MODEL_CLASSES
from FinRL_XAUUSD_common import DATA_DIR
from FinRL_XAUUSD_common import MODEL_DIR
from FinRL_XAUUSD_common import RESULTS_DIR
from FinRL_XAUUSD_common import build_env_kwargs
from FinRL_XAUUSD_common import buy_and_hold_baseline
from FinRL_XAUUSD_common import ensure_xau_dirs
from FinRL_XAUUSD_common import parse_algorithms
from FinRL_XAUUSD_common import summarize_equity_curves
from FinRL_XAUUSD_common import summarize_long_only_trades


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run walk-forward XAU validation.")
    parser.add_argument("--algo", default="ppo", help="ppo, sac, td3, ddpg, a2c, or all")
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--train-years", type=int, default=5)
    parser.add_argument("--test-months", type=int, default=12)
    parser.add_argument("--step-months", type=int, default=12)
    parser.add_argument("--initial-amount", type=float, default=100000.0)
    parser.add_argument("--hmax", type=int, default=100)
    parser.add_argument("--cost-pct", type=float, default=0.0002)
    parser.add_argument("--reward-scaling", type=float, default=1e-4)
    parser.add_argument("--processed-file", default=str(DATA_DIR / "xau_processed.csv"))
    return parser.parse_args()


def load_processed(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df["date_dt"] = pd.to_datetime(df["date"])
    return df.sort_values(["date_dt", "tic"]).reset_index(drop=True)


def date_string(value: pd.Timestamp) -> str:
    return value.strftime("%Y-%m-%d")


def main() -> None:
    args = parse_args()
    ensure_xau_dirs()

    processed = load_processed(args.processed_file)
    min_date = processed["date_dt"].min().normalize()
    max_date = processed["date_dt"].max().normalize()
    fold_start = min_date
    fold_id = 1

    all_equity_metrics = []
    all_trade_metrics = []
    all_equity_curves = []
    walkforward_result_dir = RESULTS_DIR / "walkforward"
    walkforward_model_dir = MODEL_DIR / "walkforward"
    walkforward_result_dir.mkdir(parents=True, exist_ok=True)
    walkforward_model_dir.mkdir(parents=True, exist_ok=True)

    while True:
        train_start = fold_start
        train_end = train_start + pd.DateOffset(years=args.train_years)
        trade_start = train_end
        trade_end = trade_start + pd.DateOffset(months=args.test_months)
        if trade_start >= max_date:
            break

        train = data_split(processed, date_string(train_start), date_string(train_end))
        trade = data_split(processed, date_string(trade_start), date_string(trade_end))
        if train.empty or trade.empty:
            fold_start = fold_start + pd.DateOffset(months=args.step_months)
            continue

        fold_name = f"fold_{fold_id:02d}"
        print(
            f"\n=== {fold_name}: train {date_string(train_start)} to "
            f"{date_string(train_end)}, test {date_string(trade_start)} to "
            f"{date_string(trade_end)} ==="
        )
        result_series = {
            "buy_hold": buy_and_hold_baseline(trade, args.initial_amount)
        }

        train_kwargs = build_env_kwargs(
            train,
            initial_amount=args.initial_amount,
            hmax=args.hmax,
            cost_pct=args.cost_pct,
            reward_scaling=args.reward_scaling,
        )
        trade_kwargs = build_env_kwargs(
            trade,
            initial_amount=args.initial_amount,
            hmax=args.hmax,
            cost_pct=args.cost_pct,
            reward_scaling=args.reward_scaling,
        )
        train_env = StockTradingEnv(df=train, **train_kwargs)
        env_train, _ = train_env.get_sb_env()

        for algo in parse_algorithms(args.algo):
            agent = DRLAgent(env=env_train)
            model = agent.get_model(algo, model_kwargs=MODEL_PARAMS[algo])
            model.set_logger(
                configure(
                    str(walkforward_result_dir / fold_name / algo),
                    ["stdout", "csv", "tensorboard"],
                )
            )
            trained = agent.train_model(
                model=model,
                tb_log_name=f"xau_{fold_name}_{algo}",
                total_timesteps=args.timesteps,
            )
            model_path = walkforward_model_dir / f"{fold_name}_agent_{algo}"
            trained.save(str(model_path))

            loaded = MODEL_CLASSES[algo].load(str(model_path))
            trade_env = StockTradingEnv(df=trade, **trade_kwargs)
            account_value, actions = DRLAgent.DRL_prediction(
                model=loaded,
                environment=trade_env,
            )
            strategy_name = f"{fold_name}_{algo}"
            result_series[strategy_name] = account_value.set_index("date")[
                "account_value"
            ].rename(strategy_name)
            actions.to_csv(walkforward_result_dir / f"{strategy_name}_actions.csv")

            trade_metrics = summarize_long_only_trades(
                trade,
                actions,
                cost_pct=args.cost_pct,
            )
            trade_metrics.insert(0, "strategy", strategy_name)
            trade_metrics.insert(0, "fold", fold_name)
            all_trade_metrics.append(trade_metrics)

        fold_result = pd.concat(result_series.values(), axis=1, join="inner").dropna()
        fold_metrics = summarize_equity_curves(fold_result)
        fold_metrics.insert(0, "fold", fold_name)
        fold_metrics.insert(1, "train_start", date_string(train_start))
        fold_metrics.insert(2, "train_end", date_string(train_end))
        fold_metrics.insert(3, "test_start", date_string(trade_start))
        fold_metrics.insert(4, "test_end", date_string(trade_end))
        all_equity_metrics.append(fold_metrics.reset_index())

        fold_result.to_csv(walkforward_result_dir / f"{fold_name}_equity.csv")
        all_equity_curves.append(fold_result.add_prefix(f"{fold_name}_"))
        fold_start = fold_start + pd.DateOffset(months=args.step_months)
        fold_id += 1

    if not all_equity_metrics:
        raise RuntimeError("No walk-forward folds were generated.")

    equity_metrics = pd.concat(all_equity_metrics, ignore_index=True)
    equity_metrics.to_csv(walkforward_result_dir / "walkforward_metrics.csv", index=False)
    if all_trade_metrics:
        trade_metrics = pd.concat(all_trade_metrics, ignore_index=True)
        trade_metrics.to_csv(
            walkforward_result_dir / "walkforward_trade_metrics.csv",
            index=False,
        )
    pd.concat(all_equity_curves, axis=1).to_csv(
        walkforward_result_dir / "walkforward_equity_curves.csv"
    )

    print("\n=== Walk-forward Metrics ===")
    print(equity_metrics)
    print(f"\nSaved walk-forward outputs under {walkforward_result_dir}")


if __name__ == "__main__":
    main()
