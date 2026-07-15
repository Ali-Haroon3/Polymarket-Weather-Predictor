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
- `src/bin/`: runnable binaries (examples, backtest, capture, dashboard)
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

## Dashboard and forward capture

Generate a self-contained HTML dashboard of the model's calibration against real resolved
Polymarket markets (reliability diagram, Brier/ECE/skill score, per-city and per-market tables):

```bash
cargo run --release --bin weather_dashboard -- \
  --markets data/polymarket_history.csv \
  --output dashboard.html
```

Weather is fetched once per city and cached under `data/weather_cache/` (pass `--refresh` to
re-fetch). Open `dashboard.html` in a browser.

Polymarket purges price history shortly after a market resolves, so real entry prices for a backtest
only exist while markets are live. The capture daemon snapshots every active weather market (its
current price + the model's probability) and finalizes outcomes as markets resolve, accruing a real
(price, estimate, outcome) dataset over time:

```bash
cargo run --release --bin capture_prices    # run daily (cron / schedule)
```

It appends to `data/captures.jsonl`, which the dashboard reads to populate the forward-PnL panel and
the live model-vs-market disagreement signals.

## Deployment (GitHub Pages)

The dashboard is published to a stable URL by two chained workflows:

- `.github/workflows/daily-capture.yml` — runs 15:00 UTC daily (and on manual dispatch): builds
  `capture_prices` + `weather_dashboard`, captures prices, re-renders `dashboard.html`, and commits
  it to `main`.
- `.github/workflows/deploy-dashboard.yml` — runs after a successful `daily-capture` (via
  `workflow_run`) or on manual dispatch: publishes `dashboard.html` as `index.html` to GitHub Pages.

Live URL: <https://ali-haroon3.github.io/Polymarket-Weather-Predictor/>

### One-time setup (required before the first deploy can succeed)

The deploy uses the **"GitHub Actions"** Pages source, which needs the Pages site to exist. A
workflow's `GITHUB_TOKEN` is **not allowed to create the site the first time** — that is the
`Create Pages site failed: Resource not accessible by integration` error you get if this step is
skipped. Enable Pages once, as the repo owner:

- **In the UI:** Settings → Pages → Build and deployment → **Source: "GitHub Actions"**.
- **Or via the API** (with a token that has Pages admin / `repo` scope — a workflow token will not
  work here):

  ```bash
  gh api -X POST repos/Ali-Haroon3/Polymarket-Weather-Predictor/pages -f build_type=workflow
  ```

Once Pages is enabled, deploys succeed automatically after each daily capture. To publish immediately
without waiting for 15:00 UTC, trigger a deploy manually:

```bash
gh workflow run deploy-dashboard.yml       # Actions → deploy-dashboard → Run workflow
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
