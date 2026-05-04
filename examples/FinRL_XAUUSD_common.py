from __future__ import annotations

from pathlib import Path

import pandas as pd

from finrl.config import INDICATORS

DATA_DIR = Path("datasets/xauusd")
MODEL_DIR = Path("trained_models/xauusd")
RESULTS_DIR = Path("results/xauusd")
TECH_INDICATORS = INDICATORS

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


def parse_algorithms(value: str) -> list[str]:
    supported = ("a2c", "ddpg", "ppo", "td3", "sac")
    if value == "all":
        return list(supported)

    algos = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(algos) - set(supported))
    if unknown:
        raise ValueError(f"Unsupported algorithm(s): {', '.join(unknown)}")
    return algos
