# Live Trading Guide (Rust)

This guide explains how to run the Rust live trading components in paper mode and real mode.

## Quick Start (Paper Trading)

Paper mode requires no credentials.

```bash
cargo run --bin live_trading_example
```

The example initializes `LiveTrader`, trains the Bayesian model with synthetic weather data, runs trading iterations, and prints a summary.

## Real Trading Setup

### 1. Create and secure your Polymarket account
- Create account on [Polymarket](https://polymarket.com)
- Enable 2FA
- Fund with USDC

### 2. Set credentials

```bash
export POLYMARKET_API_KEY="your_api_key"
export POLYMARKET_API_SECRET="your_api_secret"
export POLYMARKET_PRIVATE_KEY="your_private_key"
```

### 3. Construct trader in real mode

```rust
use polymarket_weather_predictor::api::LiveTrader;

let mut trader = LiveTrader::new(
    std::env::var("POLYMARKET_API_KEY").ok(),
    std::env::var("POLYMARKET_API_SECRET").ok(),
    std::env::var("POLYMARKET_PRIVATE_KEY").ok(),
    Some(1_000.0),
    false,
    0.05,
);
```

## Core API Surface

### `PolymarketClient`
- `get_markets(filter, limit)`
- `get_orderbook(market_id)`
- `get_mid_price(market_id)`
- `place_order(...)`
- `market_order(...)`
- `close_position(...)`
- `get_account_balance()`
- `get_positions()`

### `LiveTrader`
- `initialize(historical_weather)`
- `scan_markets()`
- `analyze_market(market_id, market)`
- `calculate_order(analysis, available_capital)`
- `execute_order(order)`
- `run_iteration(markets)`
- `run_backtest(prices, forecasts, max_iterations)`
- `get_performance_summary()`
- `shutdown()`

## Risk Controls

Recommended starting settings:
- `initial_capital`: small (for example `1000`)
- `max_position_pct`: `0.05` or lower
- Keep `paper_trading=true` until behavior is validated

## Operational Notes

- The Rust client defaults to paper trading if real auth is not configured.
- Market/network APIs may fail intermittently; the implementation handles failures gracefully and continues.
- For production use, add external monitoring and persistent trade storage.
