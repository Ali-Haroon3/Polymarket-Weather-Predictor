# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
cargo build                       # build lib + all bins
cargo test                        # all integration tests (tests/*.rs)
cargo test --test models          # one test file
cargo test test_calibration_metrics   # one test by name (substring match)
cargo clippy --all-targets        # lint
cargo fmt                         # format (no rustfmt.toml; uses defaults)
```

Runnable binaries (`src/bin/`): `init_db`, `run_backtest`, `run_backtest_real`, `download_polymarket_history`, `example_workflow`, `live_trading_example`. See README for their flags. The history downloader and backtest-real form the real-data pipeline: `download_polymarket_history` → CSV → `run_backtest_real --markets <csv>`.

## Architecture

Rust port of a Python weather-market trading system. The Python original is archived verbatim under `archive/python-legacy/` with a mirrored module layout — consult it when a Rust port behaves unexpectedly or a feature looks missing.

**Pipeline (the core data flow, spanning many files):**

1. **Fetchers** (`src/data_pipeline/*_fetcher.rs`) — one struct per weather source, all exposing `fetch_location(key, start, end) -> Vec<WeatherRecord>`. Free sources (open-meteo, nws, noaa, awc) always run; API-keyed sources (owm, visual_crossing, weatherapi, tomorrow_io, accuweather) implement `is_available()` and are **silently skipped when their key is unset**. A failed/erroring fetch returns an empty `Vec`, never panics — degradation is by design.
2. **`MultiSourceAggregator`** (`multi_source_aggregator.rs`) — calls every fetcher, groups records by date, and averages each field across sources. A day is marked `is_validated` only when ≥2 sources agree. This is the single funnel all weather data flows through.
3. **`BayesianWeatherModel`** (`models/bayesian_model.rs`) — despite param names like `mcmc_draws`/`tune`, there is **no MCMC**. It uses closed-form conjugate updates: Normal–Normal for temperature, Beta–Bernoulli for precipitation occurrence. Predictions sample from the posterior with **fixed per-method seeds** (42/43/44/45) so results are deterministic.
4. **`MarketSimulator`** (`backtesting/market_simulator.rs`) — synthesizes markets from weather: computes rolling **climatological** probabilities and prices each market at that prob plus seeded Gaussian noise. Used only in simulated-backtest mode.
5. **`BacktestEngine`** (`backtesting/backtest_engine.rs`) — the orchestrator. Walk-forward: for each market date it trains a *fresh* model on the trailing `model_lookback_days` window (needs ≥14 records), computes `edge = estimate − price`, filters by `edge_threshold`, sizes with **fractional Kelly** (`kelly_fraction`, capped at `max_position_pct`), and books PnL. Two entry points: `run()` (simulated markets) and `run_with_real_markets()` (markets loaded from CSV/JSON via `RealMarketLoader`).
6. **`PerformanceAnalyzer`** (`backtesting/performance_metrics.rs`) — turns the trade log + portfolio curve into Sharpe, drawdown, return, etc.

**Live trading** (`src/api/`): `PolymarketClient` (defaults to paper mode unless real auth is passed) and `LiveTrader`, which wires the same `BayesianWeatherModel` + `MarketMaker` against live orderbooks. Separate from the backtest path.

## Conventions & gotchas

- **Temperature units cross boundaries.** `WeatherRecord` stores °C. Config thresholds and `SimulatedMarket` use °F. Conversions happen at the edges — `backtest_engine::get_model_estimate` converts the °F threshold to °C before calling the model; `market_simulator` converts stored °C to °F. NOAA raw values are in tenths and divided by 10 in `normalize_noaa_data`. Get a unit wrong here and backtests silently produce garbage.
- **Config is functions, not a loaded struct.** `src/config.rs` exposes `bayesian_model_params()`, `backtest_params()`, `initial_capital()`, etc. Env-backed accessors each call `dotenvy::dotenv()` and fall back to a hardcoded default — there is no global config object.
- **Determinism is load-bearing.** Every RNG is a seeded `StdRng`. Tests and backtests rely on this; don't introduce `thread_rng()` or unseeded randomness in the model/sim/backtest paths.
- **Blocking vs async split.** Weather fetchers use `reqwest::blocking`. Only the Polymarket history downloader (`api/polymarket_history.rs`) is async (`reqwest` + `tokio`); its bin is `#[tokio::main]`. Don't call blocking fetchers from an async context.
- **The database layer is an in-memory abstraction** (`src/database/`), not a real DB connection, despite `DATABASE_URL` existing in config.
- **CLI args are parsed by hand** (no `clap`): a `--key value` loop into a `HashMap`. Match that pattern when adding flags to a bin.
- Errors that cross module boundaries use `thiserror` enums (`RealMarketLoadError`, `PolymarketHistoryError`); internal model/training errors are plain `Result<_, String>`.
