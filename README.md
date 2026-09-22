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

## Kalshi pilot and real money

`kalshi_pilot` defaults to the experimental market-shape strategy (Kalshi only, BUY YES or BUY NO,
lead ≥ 1, edge after fees plus a buffer, flat `--stake`, exposure caps and circuit breakers).
It is a DRY RUN unless `--live` is passed. Every live run, including demo, must pass the
fee-inclusive go-live gate before placing new orders. `python3` is required for live admission;
the scorer is embedded in the Rust binary at build time. Missing/invalid evidence or an
unavailable scorer blocks new orders. Earlier live orders are reconciled before admission and
breaker checks so standing down does not suppress expiry management. Total exposure includes
held positions and resting commitments, including orders outside the pilot ledger; unknown
resting quantities block new orders. The trade host remains
demo until `KALSHI_BASE_URL` points at production:

```bash
cargo run --release --bin kalshi_pilot              # dry run: log intended orders + skips
cargo run --release --bin kalshi_pilot -- --live    # real limit orders on the configured host
python3 scripts/go_live_gate.py                     # read the decision rule and current verdict
python3 scripts/go_live_gate.py --enforce            # exit nonzero unless admission passes
python3 scripts/pilot_alpha_audit.py                 # prospective side attribution and uncertainty
python3 scripts/pilot_alpha_audit.py --replay        # slower capture-time comparison of both vs NO
```

Every decision lands in `data/pilot_trades.jsonl`. Live orders are placed with a TTL
(`--order-ttl-mins`, default 240) and reconciled on the next credentialed run into `fill` /
`unfilled` rows recording what actually executed at what price; a live run also cancels any of its
own orders still resting on their target day. The daily-capture GitHub Action is the canonical
driver: dry by default, and live when the repository variable `PILOT_LIVE` is `1`, the
`KALSHI_API_KEY_ID` / `KALSHI_PRIVATE_KEY_PEM` secrets are set, and the `KALSHI_BASE_URL` variable
names the production host. `PILOT_DISABLE=1` is the kill switch in either driver.

As of captures through 2026-09-22 the default has 65 settled **paper** orders, −$42.75 after
modeled fees (−4.45%), and has not passed admission. Five newly resolved orders lost $41.44.
The frozen NO shadow still has zero prospective selections: today's only new order was YES.
See [the latest forward update](reports/2026-09-22-forward-update.md),
[the original loss audit](reports/2026-09-21-alpha-audit.md) and
[independent validation](reports/2026-09-21-alpha-validation.md). No trading rule is promoted
from these results. Python accounting tests run with
`python3 -m unittest discover -s tests -p 'test_*.py'`.

The [preregistered archive test](reports/2026-09-22-archive-preregistration.md) uses separate
May–June data, exact prior-day 15:00 UTC quote candles, and actual settlement timestamps for
training availability. It evaluates four fixed policies without changing the pilot:

```bash
python3 scripts/kalshi_archive_capture.py --download --workers 12 \
  --preregistration-commit e7c78f46121e4bfa9915c91c2165fba9377de118
python3 scripts/archive_alpha_validation.py \
  --captures data/raw/kalshi_archive_may_june_2026/captures.jsonl
```

Public responses are cached and hashed under ignored `data/raw/`; canonical captures remain
separate. Missing exact candles reject the whole event. Archive candles are sparse, so missing
minutes do not establish missing orderbooks. Coverage must be read alongside any performance
result, and historical quote replays are not execution evidence.
The [completed archive test](reports/2026-09-22-archive-validation.md) recovered only 61 events;
no entry had the 60 earlier settled ladders needed for training. All four policies remain
unevaluated on that sample. Compressed inputs are included for offline reproduction.

A [bounded training extension](reports/2026-09-22-archive-training-extension-preregistration.md)
adds March–April history while keeping the original May–June input and all four policies fixed.
It is a reanalysis of the already processed June sample, not a new untouched holdout. The
extension stops at April 30 regardless of results and cannot authorize live trading:

```bash
python3 scripts/kalshi_archive_warmup.py --download \
  --preregistration-commit f534598c21615f5cb97b61933f8927844a9217a0
python3 scripts/archive_alpha_validation.py \
  --captures data/raw/kalshi_archive_may_june_2026/captures.jsonl \
  --warmup data/raw/kalshi_archive_warmup_mar_apr_2026/captures.jsonl
```

The [completed extension](reports/2026-09-22-archive-training-extension.md) recovered 154
training events and supplied 182–208 causal ladders at every June entry. All four policies
lost after fees: joint −14.13%, parent-selected NO −1.19%, bias only −9.82%, scale only −7.93%.
All failed the frozen criteria. Only 30 June events had usable quotes, and intervals still
span loss and gain; this rejects promotion without proving negative expected returns.

The [current contract-source audit](reports/2026-09-22-current-weather-audit.md) found that
all 90 observed September 22 daily contracts name The Weather Company. Older NWS-based
calibration descriptions are not proof of the current contractual source or identical
measurement rules. New captures retain optional raw rules, an observed-source tag and a
rules hash; legacy metadata stays missing. Venue-reported outcome labels remain unchanged.
The same audit found no profitable full-ladder taker basket in one current 15-city snapshot.
A separate [next-day quote screen](reports/2026-09-22-next-day-basket-audit.md) covers all
15 September 23 ladders: 23 quoted basket sides and 46 budget comparisons, with no positive
result after fees. The ordinary basket payoff is conditional on binary settlement; contract
rules retain a last-fair-price exception when settlement data is unavailable.

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
- `KALSHI_BASE_URL` (trade host; demo by default), `KALSHI_API_KEY_ID`,
  `KALSHI_PRIVATE_KEY_PEM` or `KALSHI_PRIVATE_KEY_PATH` (trade endpoints only; market data is
  anonymous), `PILOT_DISABLE`

## Notes

- API-keyed weather sources are automatically skipped when keys are missing.
- Polymarket client supports paper trading mode by default.
- The database layer is a lightweight Rust abstraction in this port.
