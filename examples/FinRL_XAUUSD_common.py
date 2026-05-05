from __future__ import annotations

from pathlib import Path
import ast

import numpy as np
import pandas as pd

from finrl.config import INDICATORS

DATA_DIR = Path("datasets/xauusd")
MODEL_DIR = Path("trained_models/xauusd")
RESULTS_DIR = Path("results/xauusd")
BASE_TECH_INDICATORS = INDICATORS
XAU_EXTRA_FEATURES = [
    "return_1",
    "return_5",
    "return_10",
    "log_return_1",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "range_pct",
    "body_pct",
    "sma_ratio_5",
    "sma_ratio_10",
    "sma_ratio_20",
    "atr_14",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
]
TECH_INDICATORS = BASE_TECH_INDICATORS + XAU_EXTRA_FEATURES

DEFAULT_TICKER = "XAUUSD=X"
FALLBACK_TICKERS = ("GC=F",)


def ensure_xau_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_split(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.set_index(df.columns[0])
    df.index.names = [""]
    return df


def build_env_kwargs(
    df: pd.DataFrame,
    *,
    initial_amount: float,
    hmax: int,
    cost_pct: float,
    reward_scaling: float,
) -> dict:
    stock_dimension = len(df.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(TECH_INDICATORS) * stock_dimension

    return {
        "hmax": hmax,
        "initial_amount": initial_amount,
        "num_stock_shares": [0] * stock_dimension,
        "buy_cost_pct": [cost_pct] * stock_dimension,
        "sell_cost_pct": [cost_pct] * stock_dimension,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": TECH_INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": reward_scaling,
    }


def add_xau_market_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["date_dt"] = pd.to_datetime(enriched["date"])

    frames = []
    for _, group in enriched.sort_values(["tic", "date_dt"]).groupby("tic"):
        group = group.copy()
        close = group["close"].replace(0, np.nan)
        high = group["high"]
        low = group["low"]
        previous_close = close.shift(1)

        group["return_1"] = close.pct_change(1)
        group["return_5"] = close.pct_change(5)
        group["return_10"] = close.pct_change(10)
        group["log_return_1"] = np.log(close / previous_close)
        group["volatility_5"] = group["return_1"].rolling(5).std()
        group["volatility_10"] = group["return_1"].rolling(10).std()
        group["volatility_20"] = group["return_1"].rolling(20).std()
        group["range_pct"] = (high - low) / close
        group["body_pct"] = (group["close"] - group["open"]) / close

        for window in (5, 10, 20):
            group[f"sma_ratio_{window}"] = close / close.rolling(window).mean() - 1.0

        true_range = pd.concat(
            [
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        group["atr_14"] = true_range.rolling(14).mean() / close

        hour = group["date_dt"].dt.hour + group["date_dt"].dt.minute / 60.0
        day = group["date_dt"].dt.dayofweek
        group["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        group["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        group["day_sin"] = np.sin(2 * np.pi * day / 7)
        group["day_cos"] = np.cos(2 * np.pi * day / 7)
        frames.append(group)

    enriched = pd.concat(frames, ignore_index=True)
    enriched = enriched.drop(columns=["date_dt"])
    enriched[XAU_EXTRA_FEATURES] = enriched[XAU_EXTRA_FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )
    enriched[XAU_EXTRA_FEATURES] = enriched.groupby("tic", group_keys=False)[
        XAU_EXTRA_FEATURES
    ].apply(lambda values: values.ffill().bfill().fillna(0))
    return enriched.sort_values(["date", "tic"]).reset_index(drop=True)


def parse_algorithms(value: str) -> list[str]:
    supported = ("a2c", "ddpg", "ppo", "td3", "sac")
    if value == "all":
        return list(supported)

    algos = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(algos) - set(supported))
    if unknown:
        raise ValueError(f"Unsupported algorithm(s): {', '.join(unknown)}")
    return algos


def buy_and_hold_baseline(trade: pd.DataFrame, initial_amount: float) -> pd.Series:
    prices = trade[["date", "close"]].drop_duplicates("date").set_index("date")["close"]
    return prices.div(prices.iloc[0]).mul(initial_amount).rename("buy_hold")


def max_drawdown(values: pd.Series) -> float:
    drawdown = values.div(values.cummax()).sub(1.0)
    return float(drawdown.min())


def summarize_equity_curves(result: pd.DataFrame) -> pd.DataFrame:
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
                "daily_win_rate": float((returns > 0).mean()) if len(returns) else 0.0,
                "sharpe": sharpe,
            }
        )
    return pd.DataFrame(rows).set_index("strategy")


def parse_action_value(value) -> int:
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple, np.ndarray)):
            return int(parsed[0])
        return int(parsed)
    if isinstance(value, (list, tuple, np.ndarray)):
        return int(value[0])
    return int(value)


def summarize_long_only_trades(
    trade: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    cost_pct: float,
) -> pd.DataFrame:
    prices = trade[["date", "close"]].drop_duplicates("date").set_index("date")["close"]
    action_df = actions.copy()
    if "date" not in action_df.columns:
        action_df = action_df.reset_index()
    if "actions" in action_df.columns:
        action_series = action_df.set_index("date")["actions"].map(parse_action_value)
    else:
        action_series = action_df.set_index("date").iloc[:, 0].map(parse_action_value)

    lots: list[dict] = []
    closed_trades: list[dict] = []
    for date, action in action_series.items():
        if date not in prices.index or action == 0:
            continue
        price = float(prices.loc[date])
        if action > 0:
            lots.append({"entry_date": date, "entry_price": price, "shares": action})
            continue

        shares_to_sell = abs(action)
        while shares_to_sell > 0 and lots:
            lot = lots[0]
            shares = min(shares_to_sell, lot["shares"])
            entry_cost = lot["entry_price"] * shares * (1 + cost_pct)
            exit_value = price * shares * (1 - cost_pct)
            pnl = exit_value - entry_cost
            closed_trades.append(
                {
                    "entry_date": lot["entry_date"],
                    "exit_date": date,
                    "entry_price": lot["entry_price"],
                    "exit_price": price,
                    "shares": shares,
                    "pnl": pnl,
                    "return": pnl / entry_cost if entry_cost else 0.0,
                }
            )
            lot["shares"] -= shares
            shares_to_sell -= shares
            if lot["shares"] == 0:
                lots.pop(0)

    if not closed_trades:
        return pd.DataFrame(
            [
                {
                    "closed_trades": 0,
                    "trade_win_rate": 0.0,
                    "profit_factor": 0.0,
                    "average_trade_return": 0.0,
                    "average_win": 0.0,
                    "average_loss": 0.0,
                    "total_trade_pnl": 0.0,
                    "open_lots": len(lots),
                }
            ]
        )

    closed = pd.DataFrame(closed_trades)
    wins = closed[closed["pnl"] > 0]
    losses = closed[closed["pnl"] < 0]
    gross_profit = wins["pnl"].sum()
    gross_loss = losses["pnl"].abs().sum()
    profit_factor = float(gross_profit / gross_loss) if gross_loss else float("inf")
    return pd.DataFrame(
        [
            {
                "closed_trades": int(len(closed)),
                "trade_win_rate": float((closed["pnl"] > 0).mean()),
                "profit_factor": profit_factor,
                "average_trade_return": float(closed["return"].mean()),
                "average_win": float(wins["pnl"].mean()) if len(wins) else 0.0,
                "average_loss": float(losses["pnl"].mean()) if len(losses) else 0.0,
                "total_trade_pnl": float(closed["pnl"].sum()),
                "open_lots": len(lots),
            }
        ]
    )
