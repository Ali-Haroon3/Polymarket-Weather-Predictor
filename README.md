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
