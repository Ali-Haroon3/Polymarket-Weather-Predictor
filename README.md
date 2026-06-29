# Polymarket Weather Predictor (Rust)

A Rust port of the full Polymarket Weather Prediction System, including:
- Multi-source weather data pipeline
- Bayesian probability modeling
- Monte Carlo trading simulation and market making
- End-to-end backtesting engine
- Polymarket API client and live trader scaffolding

## Project Layout

- `src/data_pipeline/`: weather fetchers, aggregation, processing
- `src/models/`: Bayesian model and calibration metrics
- `src/trading/`: Monte Carlo simulator and market maker
- `src/backtesting/`: market simulator, backtest engine, performance analytics
- `src/api/`: Polymarket API client + live trading bot
- `src/database/`: data model structs and connection/session abstractions
- `src/bin/`: runnable binaries (examples, backtest, init_db)
- `tests/`: integration tests covering models/trading/backtesting/data/api

## Build

```bash
cargo build
```

## Test

```bash
cargo test
```

## Run

Initialize database abstraction:

```bash
cargo run --bin init_db
```

Run the full backtest example:

```bash
cargo run --bin run_backtest
```

Download real Polymarket weather market history (CSV for backtesting):

```bash
cargo run --bin download_polymarket_history -- \
  --output data/polymarket_history.csv \
  --start 2025-01-01 \
  --end 2025-03-31 \
  --limit 500
```

Run backtest with real Polymarket market history (CSV/JSON):

```bash
cargo run --bin run_backtest_real -- \
  --markets data/polymarket_history.csv \
  --start 2025-01-01 \
  --end 2025-03-31
```

Typical flow:
1. `download_polymarket_history` to generate `data/polymarket_history.csv`
2. `run_backtest_real` to compute strategy performance on those markets

Required columns in CSV/JSON rows:
- `date` (`YYYY-MM-DD` or RFC3339 timestamp)
- `market_id`
- `market_title`
- `market_type` — one of:
  - `temperature` (legacy: P(high ≥ `threshold`), `threshold` in °F)
  - `temp_at_least` (P(high ≥ `threshold`)), `temp_at_most` (P(high ≤ `threshold`))
  - `temp_bucket` (P(`threshold` ≤ high ≤ `threshold_upper`); for an exact "be N" bucket set `threshold_upper` = `threshold`)
  - `precipitation`
- `threshold` (lower/primary bound, in `unit`)
- `threshold_upper` (optional; upper bound for `temp_bucket`)
- `unit` (optional; `C` or `F`; defaults to °F for legacy rows)
- `market_price` (0-1)
- `actual_outcome` (0 or 1)
- `city` (e.g., `NYC`, `LA`, `London`)

Bucket markets are priced round-half-up: the integer bucket "be N" is the interval `[N-0.5, N+0.5)` (applied in `unit`, then converted to °C). The model forecasts the daily **high**, so `lowest temperature` markets are skipped by the downloader.

Run end-to-end workflow example:

```bash
cargo run --bin example_workflow
```

Run live trading example in paper mode:

```bash
cargo run --bin live_trading_example
```

## Environment Variables

Optional variables (defaults are provided in `src/config.rs`):

- `DATABASE_URL`
- `NOAA_API_KEY`
- `NOAA_BASE_URL`
- `ACCUWEATHER_API_KEY`
- `AWC_BASE_URL`
- `OPENWEATHERMAP_API_KEY`
- `VISUAL_CROSSING_API_KEY`
- `WEATHERAPI_KEY`
- `TOMORROW_IO_API_KEY`
- `INITIAL_CAPITAL`
- `MIN_BID_ASK_SPREAD`

## Notes

- API-keyed weather sources are automatically skipped when keys are missing.
- Polymarket client supports paper trading mode by default.
- The database layer is a lightweight Rust abstraction in this port.
