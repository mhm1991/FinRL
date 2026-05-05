# FinRL-First XAU Trading MVP

This first milestone keeps the project inside the existing FinRL stack. The
goal is to produce a clear baseline for XAU/USD research before adding custom
Forex execution logic or Transformer-based models.

## Scope

- Instrument: XAU/USD from Yahoo Finance by default, with gold futures as a
  fallback data proxy.
- Environment: FinRL `StockTradingEnv`.
- Agents: PPO by default, with optional A2C, DDPG, TD3, and SAC.
- Baseline: buy-and-hold XAU exposure.
- Outputs: trained models, backtest equity curves, action logs, and summary
  metrics.
- Features: FinRL technical indicators plus XAU-specific lagged returns,
  rolling volatility, ATR, candle range/body ratios, moving-average ratios, and
  time/session signals.

## Important Limitation

FinRL's standard single-asset stock environment supports long/flat exposure.
It does not model full Forex behavior such as native short positions, leverage,
margin calls, rollover, broker spread dynamics, or pip-level execution. That is
acceptable for the first FinRL baseline, but a production XAU Forex tool should
later use a custom environment.

Transformer investigation is intentionally postponed until after this baseline
is measurable.

## Commands

Prepare data:

```bash
python examples/FinRL_XAUUSD_1_data.py
```

Prepare data from a local broker/exported CSV:

```bash
python examples/FinRL_XAUUSD_1_data.py --csv data/xauusd_m15.csv
```

The CSV importer accepts common date/time aliases such as `date`, `datetime`,
`time`, or `timestamp`; OHLC columns such as `open`, `high`, `low`, `close`;
and optional volume columns such as `volume`, `tick_volume`, or `ticks`.

Train PPO:

```bash
python examples/FinRL_XAUUSD_2_train.py --algo ppo --timesteps 20000
```

Backtest PPO:

```bash
python examples/FinRL_XAUUSD_3_backtest.py --algo ppo
```

Backtests save equity metrics, daily win rate, action logs, and trade-level
metrics under `results/xauusd/`.

Run walk-forward validation:

```bash
python examples/FinRL_XAUUSD_4_walkforward.py --algo ppo --timesteps 20000
```

Train and compare all supported agents:

```bash
python examples/FinRL_XAUUSD_2_train.py --algo all --timesteps 20000
python examples/FinRL_XAUUSD_3_backtest.py --algo all
```

## Next Milestone

After the FinRL baseline is stable:

- replace the stock environment with a custom XAU Forex environment,
- add long/short/flat actions,
- model spread, slippage, leverage, and margin,
- add walk-forward validation,
- compare against simple technical strategies,
- only then evaluate Transformer-based signal models.
