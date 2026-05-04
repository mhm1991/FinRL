"""
FinRL XAU/USD Part 1: Data

This script prepares a single-asset XAU baseline dataset for FinRL's standard
StockTradingEnv. It defaults to Yahoo's XAU/USD symbol and falls back to gold
futures if the Forex symbol is unavailable.
"""

from __future__ import annotations

import argparse

from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

from FinRL_XAUUSD_common import DATA_DIR
from FinRL_XAUUSD_common import DEFAULT_TICKER
from FinRL_XAUUSD_common import FALLBACK_TICKERS
from FinRL_XAUUSD_common import TECH_INDICATORS
from FinRL_XAUUSD_common import ensure_xau_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare XAU data for FinRL.")
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

    candidates = [args.ticker]
    if not args.no_fallback:
        candidates.extend(t for t in FALLBACK_TICKERS if t != args.ticker)

    ticker, raw = fetch_first_available(candidates, args.train_start, args.trade_end)
    print(f"Using ticker: {ticker}")

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=TECH_INDICATORS,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False,
    )
    processed = fe.preprocess_data(raw)
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
