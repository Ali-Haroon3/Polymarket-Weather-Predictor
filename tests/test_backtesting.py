"""
Tests for backtesting engine, market simulator, and performance metrics
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from polymarket.backtesting.market_simulator import MarketSimulator
from polymarket.backtesting.backtest_engine import BacktestEngine
from polymarket.backtesting.performance_metrics import PerformanceAnalyzer


# --- Helpers ---


def make_weather_data(n_days=90, seed=42):
    """Generate synthetic weather data for testing"""
    np.random.seed(seed)
    dates = pd.date_range("2025-08-01", periods=n_days)

    # Simulate realistic temperature pattern (summer -> fall)
    base_temp = 25 - np.arange(n_days) * 0.1  # Cooling trend
    noise = np.random.normal(0, 3, n_days)

    temp_mean = base_temp + noise
    temp_max = temp_mean + np.random.uniform(3, 8, n_days)
    temp_min = temp_mean - np.random.uniform(3, 8, n_days)
    precip = np.random.exponential(1.5, n_days)
    precip[np.random.random(n_days) > 0.3] = 0  # 70% dry days

    return pd.DataFrame({
        "date": dates,
        "station_id": "TEST",
        "temperature_max": temp_max,
        "temperature_min": temp_min,
        "temperature_mean": temp_mean,
        "precipitation_total": precip,
        "wind_speed_mean": np.random.gamma(2, 2, n_days),
        "source_count": 2,
        "sources": "open_meteo,nws",
        "data_quality_score": 1.0,
        "is_validated": True,
        "location": "NYC",
    })


# --- MarketSimulator Tests ---


class TestMarketSimulator:
    def test_generate_markets(self):
        weather = make_weather_data(30)
        sim = MarketSimulator(seed=42)

        markets = sim.generate_markets(weather, "NYC")

        assert not markets.empty
        assert "market_id" in markets.columns
        assert "market_price" in markets.columns
        assert "actual_outcome" in markets.columns
        assert "market_type" in markets.columns
        assert "city" in markets.columns

    def test_market_prices_in_valid_range(self):
        weather = make_weather_data(30)
        sim = MarketSimulator(seed=42)

        markets = sim.generate_markets(weather, "NYC")

        assert (markets["market_price"] >= 0.02).all()
        assert (markets["market_price"] <= 0.98).all()

    def test_actual_outcomes_binary(self):
        weather = make_weather_data(30)
        sim = MarketSimulator(seed=42)

        markets = sim.generate_markets(weather, "NYC")

        assert set(markets["actual_outcome"].unique()).issubset({0.0, 1.0})

    def test_temperature_and_precipitation_markets(self):
        weather = make_weather_data(30)
        sim = MarketSimulator(seed=42)

        markets = sim.generate_markets(weather, "NYC")

        market_types = markets["market_type"].unique()
        assert "temperature" in market_types
        assert "precipitation" in market_types

    def test_multiple_thresholds(self):
        weather = make_weather_data(30)
        sim = MarketSimulator(temp_thresholds_f=[32, 70, 90], seed=42)

        markets = sim.generate_markets(weather, "NYC")

        temp_markets = markets[markets["market_type"] == "temperature"]
        thresholds = temp_markets["threshold"].unique()
        assert 32 in thresholds
        assert 70 in thresholds
        assert 90 in thresholds

    def test_empty_weather_data(self):
        sim = MarketSimulator(seed=42)
        markets = sim.generate_markets(pd.DataFrame(), "NYC")
        assert markets.empty

    def test_reproducible_with_seed(self):
        weather = make_weather_data(30)
        sim1 = MarketSimulator(seed=123)
        sim2 = MarketSimulator(seed=123)

        markets1 = sim1.generate_markets(weather, "NYC")
        markets2 = sim2.generate_markets(weather, "NYC")

        pd.testing.assert_frame_equal(markets1, markets2)


# --- PerformanceAnalyzer Tests ---


class TestPerformanceAnalyzer:
    def test_portfolio_metrics_basic(self):
        values = [100000, 100500, 101000, 100800, 101500]
        metrics = PerformanceAnalyzer._portfolio_metrics(values, 100000)

        assert metrics["initial_capital"] == 100000
        assert metrics["final_value"] == 101500
        assert metrics["total_return_pct"] > 0
        assert "sharpe_ratio" in metrics
        assert "max_drawdown_pct" in metrics

    def test_portfolio_metrics_with_drawdown(self):
        values = [100000, 105000, 95000, 98000, 102000]
        metrics = PerformanceAnalyzer._portfolio_metrics(values, 100000)

        assert metrics["max_drawdown_pct"] < 0  # Should be negative

    def test_trade_metrics(self):
        trades = pd.DataFrame({
            "pnl": [100, -50, 200, -30, 150, -80, 300, -20],
        })

        metrics = PerformanceAnalyzer._trade_metrics(trades)

        assert metrics["total_trades"] == 8
        assert metrics["winning_trades"] == 4
        assert metrics["losing_trades"] == 4
        assert metrics["win_rate_pct"] == 50.0
        assert metrics["avg_win"] > 0
        assert metrics["avg_loss"] < 0
        assert metrics["profit_factor"] > 1

    def test_trade_metrics_empty(self):
        trades = pd.DataFrame()
        metrics = PerformanceAnalyzer._trade_metrics(trades)
        assert metrics["total_trades"] == 0

    def test_forecast_metrics(self):
        np.random.seed(42)
        trades = pd.DataFrame({
            "our_estimate": np.random.uniform(0.2, 0.8, 100),
            "actual_outcome": np.random.binomial(1, 0.5, 100).astype(float),
        })

        metrics = PerformanceAnalyzer._forecast_metrics(trades)

        assert "brier_score" in metrics
        assert "log_loss" in metrics
        assert "expected_calibration_error" in metrics
        assert 0 <= metrics["brier_score"] <= 1

    def test_calibration_curve(self):
        predictions = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        outcomes = np.array([0, 0, 1, 1, 1, 1])

        curve = PerformanceAnalyzer._calibration_curve(predictions, outcomes, n_bins=5)

        assert len(curve) > 0
        for point in curve:
            assert "bin_center" in point
            assert "predicted_mean" in point
            assert "observed_freq" in point

    def test_city_breakdown(self):
        trades = pd.DataFrame({
            "city": ["NYC", "NYC", "LA", "LA", "London"],
            "pnl": [100, -50, 200, -30, 150],
        })

        breakdown = PerformanceAnalyzer.compute_city_breakdown(trades)

        assert "NYC" in breakdown
        assert "LA" in breakdown
        assert "London" in breakdown
        assert breakdown["NYC"]["total_trades"] == 2
        assert breakdown["LA"]["total_trades"] == 2
        assert breakdown["London"]["total_trades"] == 1

    def test_market_type_breakdown(self):
        trades = pd.DataFrame({
            "market_type": ["temperature", "temperature", "precipitation"],
            "pnl": [100, -50, 200],
        })

        breakdown = PerformanceAnalyzer.compute_market_type_breakdown(trades)

        assert "temperature" in breakdown
        assert "precipitation" in breakdown

    def test_monthly_breakdown(self):
        trades = pd.DataFrame({
            "date": ["2025-08-15", "2025-08-20", "2025-09-10", "2025-09-15"],
            "pnl": [100, -50, 200, -30],
        })

        breakdown = PerformanceAnalyzer.compute_monthly_breakdown(trades)

        assert "2025-08" in breakdown
        assert "2025-09" in breakdown


# --- BacktestEngine Tests ---


class TestBacktestEngine:
    def test_kelly_position_size_buy(self):
        engine = BacktestEngine(
            cities=["NYC"],
            initial_capital=100000,
            kelly_fraction=0.25,
        )

        # Our estimate is higher than market -> should buy
        size = engine._kelly_position_size(0.7, 0.5, 100000)
        assert size > 0
        assert size <= 100000 * 0.10  # Max position cap

    def test_kelly_position_size_sell(self):
        engine = BacktestEngine(
            cities=["NYC"],
            initial_capital=100000,
            kelly_fraction=0.25,
        )

        # Our estimate is lower than market -> should sell
        size = engine._kelly_position_size(0.3, 0.5, 100000)
        assert size > 0
        assert size <= 100000 * 0.10

    def test_kelly_no_edge(self):
        engine = BacktestEngine(
            cities=["NYC"],
            initial_capital=100000,
            kelly_fraction=0.25,
        )

        # No edge -> small or zero position
        size = engine._kelly_position_size(0.5, 0.5, 100000)
        assert size < 100  # Essentially zero

    def test_run_with_synthetic_data(self):
        """Full integration test with synthetic data (no API calls)"""
        engine = BacktestEngine(
            cities=["NYC"],
            initial_capital=100000,
            edge_threshold=0.05,
            kelly_fraction=0.25,
            model_lookback_days=30,
            seed=42,
        )

        # Create synthetic weather data
        weather_data = {"NYC": make_weather_data(90)}

        results = engine.run(
            start_date="2025-09-01",
            end_date="2025-10-30",
            weather_data_override=weather_data,
        )

        assert "config" in results
        assert "metrics" in results
        assert "portfolio_values" in results
        assert "trades" in results

        portfolio = results["metrics"]["portfolio"]
        assert portfolio["initial_capital"] == 100000
        assert portfolio["final_value"] > 0  # Capital should still be positive

    def test_run_multiple_cities(self):
        """Test backtest with multiple cities"""
        engine = BacktestEngine(
            cities=["NYC", "LA"],
            initial_capital=100000,
            model_lookback_days=30,
            seed=42,
        )

        weather_data = {
            "NYC": make_weather_data(90, seed=42),
            "LA": make_weather_data(90, seed=43),
        }

        results = engine.run(
            start_date="2025-09-01",
            end_date="2025-10-30",
            weather_data_override=weather_data,
        )

        assert "config" in results
        assert len(results["config"]["cities"]) == 2

    def test_compute_full_metrics(self):
        """Test that full metrics computation works end to end"""
        trades = pd.DataFrame({
            "date": pd.date_range("2025-08-01", periods=10),
            "city": ["NYC"] * 5 + ["LA"] * 5,
            "market_type": ["temperature"] * 7 + ["precipitation"] * 3,
            "pnl": [100, -50, 200, -30, 150, -80, 300, -20, 50, -40],
            "our_estimate": np.random.uniform(0.3, 0.7, 10),
            "actual_outcome": np.random.binomial(1, 0.5, 10).astype(float),
        })

        portfolio_values = [100000 + i * 50 for i in range(11)]

        metrics = PerformanceAnalyzer.compute_metrics(
            trades, portfolio_values, 100000
        )

        assert "portfolio" in metrics
        assert "trades" in metrics
        assert "forecast" in metrics
