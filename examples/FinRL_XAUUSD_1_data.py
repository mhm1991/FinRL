"""
FinRL XAU/USD Part 1: Data

This script prepares a single-asset XAU baseline dataset for FinRL's standard
StockTradingEnv. It defaults to Yahoo's XAU/USD symbol and falls back to gold
futures if the Forex symbol is unavailable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

from FinRL_XAUUSD_common import DATA_DIR
from FinRL_XAUUSD_common import BASE_TECH_INDICATORS
from FinRL_XAUUSD_common import DEFAULT_TICKER
from FinRL_XAUUSD_common import FALLBACK_TICKERS
from FinRL_XAUUSD_common import add_xau_market_features
from FinRL_XAUUSD_common import ensure_xau_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare XAU data for FinRL.")
    parser.add_argument(
        "--csv",
        help=(
            "Optional local OHLCV CSV. When provided, the script skips Yahoo "
            "download and imports this file instead."
        ),
    )
    parser.add_argument("--csv-ticker", default="XAUUSD")
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--train-start", default="2014-01-01")
    parser.add_argument("--train-end", default="2024-01-01")
    parser.add_argument("--trade-start", default="2024-01-01")
    parser.add_argument("--trade-end", default="2025-01-01")
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Only try --ticker and do not fall back to gold futures.",
    )
    return parser.parse_args()


def find_column(columns: list[str], aliases: tuple[str, ...], required_name: str) -> str:
    normalized = {column.lower().strip(): column for column in columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    raise ValueError(
        f"Could not find required {required_name} column. "
        f"Tried aliases: {', '.join(aliases)}"
    )


def normalize_ohlcv_csv(csv_path: str | Path, ticker: str) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    columns = list(raw.columns)

    date_col = find_column(
        columns,
        ("date", "datetime", "time", "timestamp", "gmt time", "local time"),
        "date/time",
    )
    open_col = find_column(columns, ("open", "o", "bidopen", "askopen"), "open")
    high_col = find_column(columns, ("high", "h", "bidhigh", "askhigh"), "high")
    low_col = find_column(columns, ("low", "l", "bidlow", "asklow"), "low")
    close_col = find_column(columns, ("close", "c", "bidclose", "askclose"), "close")

    volume_col = None
    for alias in ("volume", "vol", "tick_volume", "tickvolume", "ticks"):
        try:
            volume_col = find_column(columns, (alias,), "volume")
            break
        except ValueError:
            continue

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col], errors="coerce"),
            "open": pd.to_numeric(raw[open_col], errors="coerce"),
            "high": pd.to_numeric(raw[high_col], errors="coerce"),
            "low": pd.to_numeric(raw[low_col], errors="coerce"),
            "close": pd.to_numeric(raw[close_col], errors="coerce"),
            "volume": (
                pd.to_numeric(raw[volume_col], errors="coerce").fillna(0)
                if volume_col is not None
                else 0
            ),
            "tic": ticker,
        }
    )
    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    df = df.sort_values("date").drop_duplicates(["date", "tic"]).reset_index(drop=True)
    df["day"] = df["date"].dt.dayofweek
    has_intraday = (df["date"].dt.normalize() != df["date"]).any()
    date_format = "%Y-%m-%d %H:%M:%S" if has_intraday else "%Y-%m-%d"
    df["date"] = df["date"].dt.strftime(date_format)
    return df


def fetch_first_available(tickers: list[str], start_date: str, end_date: str):
    last_error = None
    for ticker in tickers:
        try:
            print(f"Downloading {ticker} from {start_date} to {end_date}")
            df = YahooDownloader(
                start_date=start_date,
                end_date=end_date,
                ticker_list=[ticker],
            ).fetch_data(auto_adjust=True)
            if not df.empty:
                return ticker, df
        except Exception as error:  # noqa: BLE001
            last_error = error
            print(f"Download failed for {ticker}: {error}")

    raise RuntimeError(f"No XAU data could be downloaded. Last error: {last_error}")


def main() -> None:
    args = parse_args()
    ensure_xau_dirs()

    if args.csv:
        raw = normalize_ohlcv_csv(args.csv, args.csv_ticker)
        print(f"Using local CSV: {args.csv}")
        print(f"Using ticker: {args.csv_ticker}")
    else:
        candidates = [args.ticker]
        if not args.no_fallback:
            candidates.extend(t for t in FALLBACK_TICKERS if t != args.ticker)

        ticker, raw = fetch_first_available(candidates, args.train_start, args.trade_end)
        print(f"Using ticker: {ticker}")

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=BASE_TECH_INDICATORS,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False,
    )
    processed = fe.preprocess_data(raw)
    processed = add_xau_market_features(processed)
    processed = processed.sort_values(["date", "tic"]).ffill().bfill()

    train = data_split(processed, args.train_start, args.train_end)
    trade = data_split(processed, args.trade_start, args.trade_end)

    if train.empty or trade.empty:
        raise RuntimeError(
            "The selected dates produced an empty train or trade split. "
            "Adjust --train-start, --train-end, --trade-start, or --trade-end."
        )

    raw.to_csv(DATA_DIR / "xau_raw.csv")
    processed.to_csv(DATA_DIR / "xau_processed.csv")
    train.to_csv(DATA_DIR / "xau_train.csv")
    trade.to_csv(DATA_DIR / "xau_trade.csv")

    print(f"Train rows: {len(train)}")
    print(f"Trade rows: {len(trade)}")
    print(f"Saved data under {DATA_DIR}")


if __name__ == "__main__":
    main()
