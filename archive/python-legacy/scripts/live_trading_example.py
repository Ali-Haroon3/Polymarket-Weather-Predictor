"""
Example: Running live trading bot on Polymarket
Demonstrates paper trading and real trading setup
"""

import logging
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

from polymarket.api import LiveTrader
from polymarket.data_pipeline import DataProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_sample_weather_data(days: int = 365) -> pd.DataFrame:
    """Generate synthetic weather data for testing"""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=days, freq="D")

    return pd.DataFrame({
        "date": dates,
        "station_id": "TEST_STATION",
        "temperature_mean": np.random.normal(15, 8, days),
        "temperature_max": np.random.normal(22, 8, days),
        "temperature_min": np.random.normal(8, 8, days),
        "precipitation_total": np.random.exponential(2, days),
        "data_quality_score": np.ones(days),
        "is_validated": np.ones(days, dtype=bool),
    })


def example_paper_trading():
    """Example: Paper trading mode (no credentials needed)"""
    logger.info("=" * 60)
    logger.info("EXAMPLE: Paper Trading on Polymarket")
    logger.info("=" * 60)

    # Initialize trader in paper trading mode
    trader = LiveTrader(
        paper_trading=True,
        initial_capital=100000,
        max_position_pct=0.1
    )

    # Generate sample data
    logger.info("Generating sample weather data...")
    weather_data = generate_sample_weather_data()

    # Initialize trader
    logger.info("Initializing trader...")
    if not trader.initialize(weather_data):
        logger.error("Failed to initialize trader")
        return

    logger.info("Trader initialized in paper trading mode")
    logger.info(f"Starting capital: ${trader.initial_capital:,.2f}")

    # Simulate market analysis
    logger.info("\nSimulating market analysis...")
    simulated_markets = [
        {
            "id": "market_temp_above_90f",
            "title": "Will temperature exceed 90F in NYC next week?",
        },
        {
            "id": "market_precipitation",
            "title": "Will there be precipitation in LA next week?",
        },
        {
            "id": "market_temp_above_95f",
            "title": "Will temperature exceed 95F in Dallas next week?",
        },
    ]

    # Run trading iterations
    logger.info("\nRunning trading iterations...")
    for iteration in range(3):
        logger.info(f"\n--- Iteration {iteration + 1} ---")
        stats = trader.run_iteration(markets=simulated_markets)

        logger.info(f"Markets scanned: {stats['markets_scanned']}")
        logger.info(f"Opportunities found: {stats['opportunities']}")
        logger.info(f"Orders placed: {stats['orders_placed']}")
        logger.info(f"Active positions: {stats['positions']}")

        time.sleep(1)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRADING SUMMARY")
    logger.info("=" * 60)

    summary = trader.get_performance_summary()
    if summary:
        logger.info(f"Total trades: {summary['total_trades']}")
        logger.info(f"Current positions: {summary['current_positions']}")
        logger.info(f"Current P&L: {summary['current_pnl_pct']:.2f}%")

    # Shutdown
    trader.shutdown()
    logger.info("Paper trading session complete!")


def example_real_trading_setup():
    """Example: Setup for real trading (with credentials)"""
    logger.info("=" * 60)
    logger.info("REAL TRADING SETUP INSTRUCTIONS")
    logger.info("=" * 60)

    setup_instructions = """
    To enable real trading on Polymarket:

    1. CREATE POLYMARKET ACCOUNT
       - Go to https://polymarket.com
       - Create account with email
       - Set up 2FA for security

    2. GENERATE API CREDENTIALS
       - Navigate to account settings
       - Go to API section
       - Generate API key and secret
       - Store securely (never commit to git)

    3. FUNDING YOUR ACCOUNT
       - Deposit USDC to your Polymarket wallet
       - Funds are available for trading immediately

    4. CONFIGURE BOT
       - Set environment variables:
         export POLYMARKET_API_KEY="your_api_key"
         export POLYMARKET_API_SECRET="your_api_secret"
         export POLYMARKET_PRIVATE_KEY="your_private_key"

       OR create a .env file:
         POLYMARKET_API_KEY=your_api_key
         POLYMARKET_API_SECRET=your_api_secret
         POLYMARKET_PRIVATE_KEY=your_private_key

    5. START LIVE TRADING
       - Set paper_trading=False in LiveTrader
       - Start with small position sizes
       - Monitor closely for first few trades

    6. RISK MANAGEMENT
       - Set max_position_pct to limit exposure
       - Use stop-loss levels
       - Monitor P&L regularly
       - Start with small capital (e.g., $1,000)

    EXAMPLE CODE:
    ```python
    import os
    from polymarket.api import LiveTrader

    trader = LiveTrader(
        api_key=os.getenv("POLYMARKET_API_KEY"),
        api_secret=os.getenv("POLYMARKET_API_SECRET"),
        private_key=os.getenv("POLYMARKET_PRIVATE_KEY"),
        initial_capital=1000,  # Start small
        paper_trading=False,
        max_position_pct=0.05  # 5% max position
    )

    # Initialize with historical data
    historical_data = load_weather_data()
    trader.initialize(historical_data)

    # Run continuous trading
    while True:
        stats = trader.run_iteration()
        print(stats)
        time.sleep(60)  # Check every minute
    ```
    """

    logger.info(setup_instructions)


def example_backtesting():
    """Example: Backtesting trading strategy"""
    logger.info("=" * 60)
    logger.info("EXAMPLE: Backtesting")
    logger.info("=" * 60)

    # Initialize trader
    trader = LiveTrader(paper_trading=True)

    # Generate data
    logger.info("Generating historical data...")
    weather_data = generate_sample_weather_data(days=365)
    prices = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=365),
        "close": np.random.uniform(0.4, 0.6, 365),
    })
    forecasts = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=365),
        "probability": np.random.uniform(0.4, 0.6, 365),
    })

    # Initialize
    trader.initialize(weather_data)

    # Run backtest
    logger.info("Running backtest...")
    results = trader.run_backtest(prices, forecasts, max_iterations=100)

    logger.info("\nBacktest Results:")
    logger.info(f"Total trades: {results['total_trades']}")
    logger.info(f"Winning trades: {results['winning_trades']}")
    logger.info(f"Win rate: {results['win_rate']:.1%}")
    logger.info(f"Final P&L: ${results['final_pnl']:.2f}")


def example_api_client_usage():
    """Example: Using PolymarketClient directly"""
    logger.info("=" * 60)
    logger.info("EXAMPLE: Polymarket API Client")
    logger.info("=" * 60)

    from polymarket.api import PolymarketClient

    # Initialize client in paper trading mode
    client = PolymarketClient(paper_trading=True)

    logger.info("\nChecking API health...")
    is_healthy = client.health_check()
    logger.info(f"API healthy: {is_healthy}")

    logger.info("\nScanning available markets...")
    markets_response = client.get_markets(filter_name="weather", limit=5)

    if isinstance(markets_response, dict):
        markets = markets_response.get("data", [])
    else:
        markets = markets_response

    logger.info(f"Found {len(markets)} markets")

    if markets:
        for market in markets[:3]:
            logger.info(f"  - {market.get('title', 'Unknown')}")

    logger.info("\nGetting account balance (paper trading)...")
    balance = client.get_account_balance()
    logger.info(f"Account balance: ${balance:,.2f}")

    logger.info("\nPlacing paper trade...")
    order = client.place_order(
        market_id="test_market_123",
        outcome="YES",
        side="BUY",
        price=0.55,
        size=10
    )

    if order:
        logger.info(f"Order placed: {order['orderId']}")
        logger.info(f"Status: {order['status']}")


def main():
    """Run all examples"""
    logger.info("Polymarket Weather Prediction - Live Trading Examples\n")

    # Example 1: Paper trading
    example_paper_trading()

    logger.info("\n" * 2)

    # Example 2: API client usage
    example_api_client_usage()

    logger.info("\n" * 2)

    # Example 3: Backtesting
    example_backtesting()

    logger.info("\n" * 2)

    # Example 4: Setup instructions
    example_real_trading_setup()

    logger.info("\n" + "=" * 60)
    logger.info("All examples completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
