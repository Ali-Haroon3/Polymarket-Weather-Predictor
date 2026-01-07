"""
Monte Carlo simulation for weather prediction market pricing and risk management
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

from polymarket.config import MONTE_CARLO_PARAMS

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """
    Monte Carlo simulation engine for weather market scenarios
    Generates probability distributions for market prices and P&L
    """

    def __init__(self, params: Dict = None):
        self.params = params or MONTE_CARLO_PARAMS
        np.random.seed(self.params["random_seed"])

    def simulate_price_paths(
        self,
        initial_price: float,
        probability_estimate: float,
        volatility: float,
        days_to_expiry: int,
        n_simulations: int = None
    ) -> np.ndarray:
        """
        Simulate weather market price paths using geometric Brownian motion

        Args:
            initial_price: Current market price
            probability_estimate: Our probability estimate for the event
            volatility: Market volatility (annualized)
            days_to_expiry: Days until market expiry
            n_simulations: Number of simulation paths

        Returns:
            Array of shape (n_simulations, days_to_expiry) with price paths
        """
        n_sims = n_simulations or self.params["n_simulations"]
        dt = 1 / 365  # Daily steps

        # Initialize paths
        paths = np.zeros((n_sims, days_to_expiry))
        paths[:, 0] = initial_price

        # Expected return based on probability estimate vs market price
        # If our estimate > market price, expect price to rise
        fair_value = probability_estimate
        drift = (fair_value - initial_price) / days_to_expiry

        # Generate price paths
        for t in range(1, days_to_expiry):
            dW = np.random.normal(0, np.sqrt(dt), n_sims)
            paths[:, t] = paths[:, t - 1] * np.exp(
                (drift - 0.5 * volatility ** 2) * dt + volatility * dW
            )
            # Clamp prices to [0, 1]
            paths[:, t] = np.clip(paths[:, t], 0, 1)

        return paths

    def calculate_pnl_distribution(
        self,
        entry_price: float,
        position_size: float,
        price_paths: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate P&L distribution from simulated price paths

        Args:
            entry_price: Entry price
            position_size: Position size
            price_paths: Simulated price paths

        Returns:
            Dictionary with P&L statistics
        """
        # Final prices at expiry
        final_prices = price_paths[:, -1]

        # P&L calculation
        pnl = (final_prices - entry_price) * position_size * 100  # 100x leverage assumption

        return {
            "mean_pnl": np.mean(pnl),
            "std_pnl": np.std(pnl),
            "median_pnl": np.median(pnl),
            "min_pnl": np.min(pnl),
            "max_pnl": np.max(pnl),
            "percentile_5": np.percentile(pnl, 5),
            "percentile_95": np.percentile(pnl, 95),
            "prob_profit": np.mean(pnl > 0),
            "prob_loss": np.mean(pnl < 0),
            "sharpe_ratio": np.mean(pnl) / np.std(pnl) if np.std(pnl) > 0 else 0,
        }

    def calculate_value_at_risk(
        self,
        pnl_distribution: np.ndarray,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk (VaR) at given confidence level

        Args:
            pnl_distribution: Array of P&L values
            confidence_level: Confidence level (e.g., 0.95)

        Returns:
            VaR estimate (negative value for loss)
        """
        return np.percentile(pnl_distribution, (1 - confidence_level) * 100)

    def calculate_conditional_var(
        self,
        pnl_distribution: np.ndarray,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Conditional VaR (expected shortfall)

        Args:
            pnl_distribution: Array of P&L values
            confidence_level: Confidence level

        Returns:
            Conditional VaR estimate
        """
        var_threshold = np.percentile(pnl_distribution, (1 - confidence_level) * 100)
        return np.mean(pnl_distribution[pnl_distribution <= var_threshold])

    def optimize_position_size(
        self,
        entry_price: float,
        probability_estimate: float,
        volatility: float,
        days_to_expiry: int,
        max_loss_tolerance: float,
        target_sharpe: float = 1.0
    ) -> Dict[str, float]:
        """
        Optimize position size using Kelly Criterion and risk constraints

        Args:
            entry_price: Current market price
            probability_estimate: Probability estimate
            volatility: Market volatility
            days_to_expiry: Days to expiry
            max_loss_tolerance: Maximum tolerable loss
            target_sharpe: Target Sharpe ratio

        Returns:
            Optimal position sizing and risk metrics
        """
        # Simulate price paths
        paths = self.simulate_price_paths(
            entry_price,
            probability_estimate,
            volatility,
            days_to_expiry
        )

        # Test different position sizes
        position_sizes = np.linspace(0.01, 0.5, 50)
        best_sharpe = -np.inf
        best_position_size = 0

        for pos_size in position_sizes:
            # Calculate P&L
            pnl = (paths[:, -1] - entry_price) * pos_size * 100

            # Check loss constraint
            max_loss = np.percentile(pnl, 5)
            if max_loss < -max_loss_tolerance:
                continue

            # Calculate Sharpe ratio
            sharpe = np.mean(pnl) / np.std(pnl) if np.std(pnl) > 0 else -np.inf
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_position_size = pos_size

        return {
            "optimal_position_size": best_position_size,
            "expected_sharpe_ratio": best_sharpe,
            "expected_return": np.mean(
                (paths[:, -1] - entry_price) * best_position_size * 100
            ),
            "expected_volatility": np.std(
                (paths[:, -1] - entry_price) * best_position_size * 100
            ),
        }

    def backtest_strategy(
        self,
        historical_prices: pd.DataFrame,
        probability_forecasts: pd.DataFrame,
        initial_capital: float = 100000,
        max_position_size: float = 0.1
    ) -> Dict:
        """
        Backtest trading strategy on historical data

        Args:
            historical_prices: DataFrame with market prices
            probability_forecasts: DataFrame with probability forecasts
            initial_capital: Starting capital
            max_position_size: Maximum position size constraint

        Returns:
            Backtest results with returns, drawdown, etc.
        """
        capital = initial_capital
        portfolio_value = [capital]
        positions = []
        trades = []

        for idx in range(len(historical_prices) - 1):
            market_price = historical_prices.iloc[idx]["close"]
            forecast_price = probability_forecasts.iloc[idx]["probability"]

            # Simple strategy: buy if forecast > market price
            if forecast_price > market_price + 0.02:  # 2% edge threshold
                position_size = min(
                    max_position_size * capital / (market_price * 100),
                    capital / (market_price * 100)
                )

                # Execute trade
                positions.append({
                    "entry_price": market_price,
                    "position_size": position_size,
                    "entry_idx": idx
                })

                trades.append({
                    "type": "BUY",
                    "price": market_price,
                    "size": position_size,
                    "date": historical_prices.index[idx]
                })

            # Close positions if profitable or stop loss
            if positions:
                current_price = historical_prices.iloc[idx + 1]["close"]
                for pos in positions[:]:
                    pnl = (current_price - pos["entry_price"]) * pos["position_size"] * 100

                    # Exit conditions
                    if pnl > capital * 0.02 or pnl < -capital * 0.01:
                        capital += pnl
                        positions.remove(pos)
                        trades.append({
                            "type": "SELL",
                            "price": current_price,
                            "size": pos["position_size"],
                            "pnl": pnl,
                            "date": historical_prices.index[idx + 1]
                        })

            portfolio_value.append(capital)

        # Calculate metrics
        returns = np.diff(portfolio_value) / portfolio_value[:-1]
        cumulative_return = (portfolio_value[-1] - initial_capital) / initial_capital
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        max_dd = self._calculate_max_drawdown(portfolio_value)

        return {
            "final_value": portfolio_value[-1],
            "total_return": cumulative_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_dd,
            "total_trades": len([t for t in trades if t["type"] == "BUY"]),
            "portfolio_value": portfolio_value,
            "trades": trades,
        }

    @staticmethod
    def _calculate_max_drawdown(values: list) -> float:
        """Calculate maximum drawdown"""
        running_max = np.maximum.accumulate(values)
        drawdown = (values - running_max) / running_max
        return np.min(drawdown)
