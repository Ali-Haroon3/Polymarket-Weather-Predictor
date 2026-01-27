# Live Trading Guide - Polymarket Weather Predictor

This guide explains how to set up and run the live trading bot on Polymarket's paper trading and real trading accounts.

## Quick Start (Paper Trading)

No credentials needed! Paper trading lets you test the bot without real money.

```python
from polymarket.api import LiveTrader
from polymarket.data_pipeline import DataProcessor
import pandas as pd
import numpy as np

# Generate sample data
dates = pd.date_range("2023-01-01", periods=365)
weather_data = pd.DataFrame({
    "date": dates,
    "temperature_mean": np.random.normal(15, 8, 365),
    "precipitation_total": np.random.exponential(2, 365),
    "data_quality_score": np.ones(365),
    "is_validated": np.ones(365, dtype=bool),
})

# Initialize trader
trader = LiveTrader(
    paper_trading=True,
    initial_capital=100000
)

# Initialize with data
trader.initialize(weather_data)

# Run one trading iteration
stats = trader.run_iteration()
print(stats)
```

Run the example:
```bash
python scripts/live_trading_example.py
```

## Getting Real Trading Credentials

### 1. Create Polymarket Account
- Visit https://polymarket.com
- Sign up with email
- Verify email
- Enable 2FA for security

### 2. Generate API Keys
- Log in to Polymarket
- Go to Account → Settings → API
- Generate API Key and API Secret
- **Important**: Store these securely, never commit to git

### 3. Fund Your Account
- Add USDC to your Polymarket wallet
- Deposits are instant
- Start with small amounts for testing

## Configuration

### Environment Variables
Set credentials as environment variables:

```bash
export POLYMARKET_API_KEY="your_api_key_here"
export POLYMARKET_API_SECRET="your_api_secret_here"
export POLYMARKET_PRIVATE_KEY="your_private_key_here"
```

### .env File
Or create a `.env` file in the project root (add to .gitignore):

```
POLYMARKET_API_KEY=your_api_key_here
POLYMARKET_API_SECRET=your_api_secret_here
POLYMARKET_PRIVATE_KEY=your_private_key_here
```

Load in Python:
```python
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("POLYMARKET_API_KEY")
api_secret = os.getenv("POLYMARKET_API_SECRET")
private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
```

## Live Trading Setup

### Start Small
```python
from polymarket.api import LiveTrader
import os
from dotenv import load_dotenv

load_dotenv()

trader = LiveTrader(
    api_key=os.getenv("POLYMARKET_API_KEY"),
    api_secret=os.getenv("POLYMARKET_API_SECRET"),
    private_key=os.getenv("POLYMARKET_PRIVATE_KEY"),
    initial_capital=1000,  # Start with $1000
    paper_trading=False,   # Enable real trading
    max_position_pct=0.05  # Max 5% per position
)

# Initialize with historical weather data
historical_data = load_your_historical_data()
trader.initialize(historical_data)

# Run one iteration
stats = trader.run_iteration()
print(stats)
```

### Continuous Trading
```python
import time

# Run trading bot continuously
while True:
    try:
        stats = trader.run_iteration()
        print(f"[{datetime.now()}] Orders: {stats.get('orders_placed', 0)}")

        # Sleep before next iteration
        time.sleep(60)  # Check every minute

    except KeyboardInterrupt:
        print("Shutting down...")
        trader.shutdown()
        break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(10)
```

## API Methods

### PolymarketClient

```python
from polymarket.api import PolymarketClient

client = PolymarketClient(
    api_key="your_key",
    api_secret="your_secret",
    paper_trading=False
)

# Get available markets
markets = client.get_markets(filter_name="weather")

# Get market orderbook
orderbook = client.get_orderbook(market_id)

# Get mid price
mid_price = client.get_mid_price(market_id)

# Place order
order = client.place_order(
    market_id=market_id,
    outcome="YES",
    side="BUY",
    price=0.55,
    size=10
)

# Get account balance
balance = client.get_account_balance()

# Get positions
positions = client.get_positions()

# Market order (instant execution)
market_order = client.market_order(
    market_id=market_id,
    outcome="YES",
    side="BUY",
    size=5
)

# Close position
client.close_position(market_id, current_position=10)
```

### LiveTrader

```python
from polymarket.api import LiveTrader

trader = LiveTrader(...)

# Initialize
trader.initialize(historical_weather_data)

# Scan for available markets
markets = trader.scan_markets()

# Analyze specific market
analysis = trader.analyze_market(market_id, market_data)

# Calculate optimal order
order = trader.calculate_order(analysis, available_capital)

# Execute order
trader.execute_order(order)

# Run one iteration (scan, analyze, trade)
stats = trader.run_iteration()

# Get performance summary
summary = trader.get_performance_summary()

# Backtest strategy
backtest_results = trader.run_backtest(prices_df, forecasts_df)

# Shutdown (closes all positions)
trader.shutdown()
```

## Risk Management

### Position Sizing
The bot uses Kelly Criterion for position sizing, automatically scaled by `max_position_pct`:

```python
trader = LiveTrader(
    initial_capital=10000,
    max_position_pct=0.05  # Max 5% per trade
)
```

With $10,000 capital and 5% max:
- Maximum position per market: $500
- Actual size also considers probability edge

### Stop Loss & Take Profit
The bot automatically:
- Takes profits at 10% gains
- Closes positions with adverse outcomes
- Limits total position exposure

### Capital Preservation
- Never trades more than available capital
- Checks account balance before orders
- Graceful shutdown closes all positions

## Troubleshooting

### "Not authenticated"
- Verify API credentials are correct
- Check environment variables are set
- Ensure account has active sessions

### "Insufficient capital"
- Check account balance
- Reduce position sizes
- Fund account with more USDC

### API errors
- Check Polymarket status page
- Verify market still exists
- Check network connectivity

### Orders not filling
- Verify sufficient liquidity in market
- Check order price is within spread
- Try market order instead of limit

## Monitoring

### Log Output
Enable debug logging:
```python
import logging

logging.basicConfig(level=logging.DEBUG)
```

### Performance Metrics
```python
summary = trader.get_performance_summary()
print(f"P&L: {summary['current_pnl_pct']:.2f}%")
print(f"Trades: {summary['total_trades']}")
print(f"Positions: {summary['current_positions']}")
```

### Trade History
```python
for trade in trader.trade_history:
    print(f"{trade['timestamp']} - {trade['side']} {trade['size']} @ {trade['price']}")
```

## Safety Checklist

- [ ] Test with paper trading first
- [ ] Start with small capital ($100-$1000)
- [ ] Set max_position_pct to 5-10%
- [ ] Monitor first 10 trades manually
- [ ] Never commit API keys to git
- [ ] Use environment variables for credentials
- [ ] Enable 2FA on Polymarket account
- [ ] Have kill switch ready (Ctrl+C)
- [ ] Review code changes before deploying
- [ ] Test on paper trading after code changes

## Support

For issues or questions:
1. Check logs for error messages
2. Verify credentials and account setup
3. Test individual API methods
4. Run paper trading to isolate issues
5. Check Polymarket documentation

Good luck with live trading!
