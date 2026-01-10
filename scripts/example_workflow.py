"""
Example workflow demonstrating the complete system
"""

import logging
import pandas as pd
import numpy as np

from polymarket.data_pipeline import NOAAFetcher, DataProcessor
from polymarket.models import BayesianWeatherModel, CalibrationAnalyzer
from polymarket.trading import MonteCarloSimulator, MarketMaker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_data_pipeline():
    """Example: Fetch and process weather data"""
    logger.info("=" * 50)
    logger.info("EXAMPLE 1: Data Pipeline")
    logger.info("=" * 50)

    # Initialize fetcher
    fetcher = NOAAFetcher()

    # Fetch data for major US stations
    logger.info("Fetching NOAA weather data...")
    data = fetcher.fetch_major_us_stations(
        start_date="2023-01-01",
        end_date="2023-12-31"
    )

    if not data.empty:
        logger.info(f"Fetched {len(data)} observations")

        # Process the data
        processor = DataProcessor()
        processed = processor.process_raw_observations(data)
        validated, quality_scores = processor.validate_observations(processed)

        logger.info(f"Valid records: {validated['is_validated'].sum()}")
        logger.info(f"Avg quality score: {quality_scores.mean():.3f}")

        # Generate statistics
        stats = processor.generate_statistics(validated)
        logger.info(f"Statistics: {stats}")

        return validated
    else:
        logger.warning("No data fetched. Using synthetic data for demo.")
        return generate_synthetic_weather_data()


def generate_synthetic_weather_data():
    """Generate synthetic weather data for demonstration"""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=365)

    return pd.DataFrame({
        "date": dates,
        "station_id": "TEST_STATION",
        "temperature_mean": np.random.normal(15, 8, 365),
        "temperature_max": np.random.normal(22, 8, 365),
        "temperature_min": np.random.normal(8, 8, 365),
        "precipitation_total": np.random.exponential(2, 365),
        "wind_speed_mean": np.random.gamma(2, 2, 365),
        "data_quality_score": np.ones(365),
        "is_validated": np.ones(365, dtype=bool),
    })


def example_probabilistic_forecasting(weather_data):
    """Example: Generate probabilistic forecasts"""
    logger.info("=" * 50)
    logger.info("EXAMPLE 2: Probabilistic Forecasting")
    logger.info("=" * 50)

    # Train model
    model = BayesianWeatherModel()
    logger.info("Training Bayesian weather model...")
    model.train(weather_data)

    # Generate forecasts
    logger.info("Generating probability forecasts...")
    forecast = model.forecast_event_probabilities()

    for event, prob in forecast.items():
        logger.info(f"{event}: {prob:.3f}")

    # Calibration analysis
    logger.info("Analyzing calibration...")
    predictions = np.random.uniform(0, 1, 100)
    outcomes = np.random.binomial(1, 0.5, 100)

    metrics = CalibrationAnalyzer.calibration_metrics(predictions, outcomes)
    logger.info(f"Brier Score: {metrics['brier_score']:.3f}")
    logger.info(f"ECE: {metrics['expected_calibration_error']:.3f}")

    return model


def example_monte_carlo_trading(model):
    """Example: Monte Carlo simulation for trading"""
    logger.info("=" * 50)
    logger.info("EXAMPLE 3: Monte Carlo Trading Simulation")
    logger.info("=" * 50)

    simulator = MonteCarloSimulator()

    # Get forecast
    forecast = model.forecast_event_probabilities()
    temp_prob = forecast.get("temp_above_90f", 0.3)

    # Simulate price paths
    logger.info("Simulating price paths...")
    paths = simulator.simulate_price_paths(
        initial_price=0.35,
        probability_estimate=temp_prob,
        volatility=0.20,
        days_to_expiry=30,
        n_simulations=1000
    )

    # Calculate P&L distribution
    pnl_stats = simulator.calculate_pnl_distribution(
        entry_price=0.35,
        position_size=0.05,
        price_paths=paths
    )

    logger.info(f"Expected P&L: ${pnl_stats['mean_pnl']:.2f}")
    logger.info(f"Std Dev: ${pnl_stats['std_pnl']:.2f}")
    logger.info(f"Sharpe Ratio: {pnl_stats['sharpe_ratio']:.3f}")
    logger.info(f"Prob Profit: {pnl_stats['prob_profit']:.1%}")

    # Optimize position size
    logger.info("Optimizing position size...")
    opt_result = simulator.optimize_position_size(
        entry_price=0.35,
        probability_estimate=temp_prob,
        volatility=0.20,
        days_to_expiry=30,
        max_loss_tolerance=5000
    )

    logger.info(f"Optimal Position Size: {opt_result['optimal_position_size']:.3f}")
    logger.info(f"Expected Sharpe: {opt_result['expected_sharpe_ratio']:.3f}")


def example_market_making():
    """Example: Market-making strategies"""
    logger.info("=" * 50)
    logger.info("EXAMPLE 4: Market-Making")
    logger.info("=" * 50)

    mm = MarketMaker(capital=100000)

    # Calculate spreads
    logger.info("Calculating optimal bid-ask spreads...")
    spread = mm.calculate_fair_value_spread(
        market_price=0.50,
        our_estimate=0.58,
        volatility=0.20,
        inventory=0.02
    )

    logger.info(f"Bid: {spread['bid']:.4f}")
    logger.info(f"Ask: {spread['ask']:.4f}")
    logger.info(f"Spread: {spread['spread']:.4f}")

    # Position sizing with Kelly Criterion
    logger.info("Calculating position sizes...")
    sizes = mm.calculate_position_sizes(
        probability_estimates={"MARKET_A": 0.60, "MARKET_B": 0.40},
        market_prices={"MARKET_A": 0.50, "MARKET_B": 0.45},
        volatility_estimates={"MARKET_A": 0.20, "MARKET_B": 0.18},
        max_position_pct=0.10
    )

    for market, size in sizes.items():
        logger.info(f"{market}: {size:.4f}")

    # Execute trades
    logger.info("Executing trades...")
    for market_id, size in sizes.items():
        trade = mm.execute_market_orders(
            market_id=market_id,
            side="BUY",
            size=size,
            price=0.50
        )
        logger.info(f"Bought {size:.4f} at {0.50}")

    # Portfolio metrics
    metrics = mm.get_portfolio_metrics()
    logger.info(f"Total Position: {metrics['total_position']:.4f}")
    logger.info(f"Net Exposure: {metrics['net_exposure']:.4f}")


def main():
    """Run all examples"""
    logger.info("Starting Polymarket Weather Prediction System Examples")
    logger.info("=" * 50)

    # Run examples
    weather_data = example_data_pipeline()
    model = example_probabilistic_forecasting(weather_data)
    example_monte_carlo_trading(model)
    example_market_making()

    logger.info("=" * 50)
    logger.info("Examples completed successfully!")


if __name__ == "__main__":
    main()
