"""
Tests for trading and market-making strategies
"""

import pytest
import numpy as np
import pandas as pd

from polymarket.trading import MonteCarloSimulator, MarketMaker


class TestMonteCarloSimulator:
    """Tests for Monte Carlo simulation"""

    @pytest.fixture
    def simulator(self):
        """Initialize simulator"""
        return MonteCarloSimulator()

    def test_price_path_simulation(self, simulator):
        """Test price path generation"""
        paths = simulator.simulate_price_paths(
            initial_price=0.5,
            probability_estimate=0.6,
            volatility=0.2,
            days_to_expiry=30,
            n_simulations=100
        )

        assert paths.shape == (100, 30)
        assert np.all(paths >= 0) and np.all(paths <= 1)

    def test_pnl_distribution(self, simulator):
        """Test P&L calculation"""
        paths = simulator.simulate_price_paths(
            initial_price=0.5,
            probability_estimate=0.6,
            volatility=0.2,
            days_to_expiry=30,
            n_simulations=1000
        )

        pnl_stats = simulator.calculate_pnl_distribution(
            entry_price=0.5,
            position_size=0.1,
            price_paths=paths
        )

        assert "mean_pnl" in pnl_stats
        assert "std_pnl" in pnl_stats
        assert "sharpe_ratio" in pnl_stats

    def test_var_calculation(self, simulator):
        """Test Value at Risk calculation"""
        pnl = np.random.normal(100, 50, 10000)
        var = simulator.calculate_value_at_risk(pnl, confidence_level=0.95)
        assert var < 0  # VaR should be negative

    def test_position_sizing(self, simulator):
        """Test optimal position sizing"""
        result = simulator.optimize_position_size(
            entry_price=0.5,
            probability_estimate=0.6,
            volatility=0.2,
            days_to_expiry=30,
            max_loss_tolerance=5000,
            target_sharpe=1.0
        )

        assert "optimal_position_size" in result
        assert 0 <= result["optimal_position_size"] <= 0.5


class TestMarketMaker:
    """Tests for market-making strategies"""

    @pytest.fixture
    def market_maker(self):
        """Initialize market maker"""
        return MarketMaker(capital=100000)

    def test_spread_calculation(self, market_maker):
        """Test bid-ask spread calculation"""
        spread = market_maker.calculate_fair_value_spread(
            market_price=0.5,
            our_estimate=0.55,
            volatility=0.2,
            inventory=0
        )

        assert spread["bid"] < spread["ask"]
        assert spread["bid"] >= 0 and spread["ask"] <= 1

    def test_optimal_spreads(self, market_maker):
        """Test optimal spread calculation for basket"""
        market_prices = pd.DataFrame({
            "bid": [0.48, 0.45],
            "ask": [0.52, 0.55],
        }, index=["TEMP_MARKET", "RAIN_MARKET"])

        estimates = pd.Series([0.55, 0.6], index=["TEMP_MARKET", "RAIN_MARKET"])

        result = market_maker.optimal_bid_ask_spreads(
            market_prices, estimates, volatility=0.2
        )

        assert len(result) == 2
        assert "bid" in result.columns and "ask" in result.columns

    def test_position_sizing(self, market_maker):
        """Test position sizing calculation"""
        estimates = {"MARKET_A": 0.6, "MARKET_B": 0.4}
        prices = {"MARKET_A": 0.5, "MARKET_B": 0.5}
        volatility = {"MARKET_A": 0.2, "MARKET_B": 0.15}

        sizes = market_maker.calculate_position_sizes(
            estimates, prices, volatility, max_position_pct=0.1
        )

        assert len(sizes) == 2
        assert all(0 <= s <= 0.1 for s in sizes.values())

    def test_order_execution(self, market_maker):
        """Test order execution"""
        trade = market_maker.execute_market_orders(
            market_id="TEST_MARKET",
            side="BUY",
            size=0.05,
            price=0.5
        )

        assert trade["market_id"] == "TEST_MARKET"
        assert trade["side"] == "BUY"
        assert "TEST_MARKET" in market_maker.inventory

    def test_portfolio_metrics(self, market_maker):
        """Test portfolio metrics calculation"""
        market_maker.inventory = {"MARKET_A": 0.1, "MARKET_B": -0.05}
        metrics = market_maker.get_portfolio_metrics()

        assert metrics["total_position"] == 0.15
        assert metrics["net_exposure"] == 0.05
        assert metrics["number_of_markets"] == 2
