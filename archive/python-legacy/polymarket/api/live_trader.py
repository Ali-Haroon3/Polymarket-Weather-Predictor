"""
Live trading bot for Polymarket weather derivatives
Connects Bayesian forecasts to real trading
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time

from polymarket.api.polymarket_client import PolymarketClient
from polymarket.models import BayesianWeatherModel
from polymarket.trading import MarketMaker
from polymarket.config import INITIAL_CAPITAL, MIN_BID_ASK_SPREAD

logger = logging.getLogger(__name__)


class LiveTrader:
    """
    Live trading bot for Polymarket
    Executes algorithmic trading strategies based on Bayesian forecasts
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        private_key: str = "",
        initial_capital: float = INITIAL_CAPITAL,
        paper_trading: bool = True,
        max_position_pct: float = 0.1
    ):
        """
        Initialize live trader

        Args:
            api_key: Polymarket API key
            api_secret: Polymarket API secret
            private_key: Private key for signing
            initial_capital: Starting capital
            paper_trading: Use paper trading mode
            max_position_pct: Maximum position size as % of capital
        """
        self.client = PolymarketClient(
            api_key=api_key,
            api_secret=api_secret,
            private_key=private_key,
            paper_trading=paper_trading
        )

        self.market_maker = MarketMaker(capital=initial_capital)
        self.model = BayesianWeatherModel()
        self.initial_capital = initial_capital
        self.paper_trading = paper_trading
        self.max_position_pct = max_position_pct

        # Trading state
        self.active_positions = {}  # market_id: position_size
        self.trade_history = []
        self.pnl_history = []
        self.start_time = datetime.now()

    def initialize(self, historical_data: pd.DataFrame) -> bool:
        """
        Initialize the trading system with historical data

        Args:
            historical_data: Historical weather data for model training

        Returns:
            True if successful
        """
        try:
            logger.info("Initializing live trader...")

            # Train Bayesian model
            logger.info("Training Bayesian weather model...")
            self.model.train(historical_data)

            # Test API connection
            if not self.paper_trading:
                logger.info("Testing Polymarket API connection...")
                if not self.client.health_check():
                    logger.error("Cannot connect to Polymarket API")
                    return False

                # Get initial balance
                balance = self.client.get_account_balance()
                if balance is None:
                    logger.error("Cannot retrieve account balance")
                    return False

                logger.info(f"Connected. Account balance: ${balance:.2f}")

            logger.info("Trader initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Initialization error: {e}")
            return False

    def scan_markets(self) -> List[Dict]:
        """
        Scan for available weather markets

        Returns:
            List of relevant markets
        """
        try:
            markets = self.client.get_markets(filter_name="weather", limit=50)
            logger.info(f"Found {len(markets)} weather markets")
            return markets

        except Exception as e:
            logger.error(f"Error scanning markets: {e}")
            return []

    def analyze_market(
        self,
        market_id: str,
        market_data: Dict
    ) -> Optional[Dict]:
        """
        Analyze a market and generate trading signal

        Args:
            market_id: Market ID
            market_data: Market details from API

        Returns:
            Trading signal with probability estimate
        """
        try:
            # Get current market price
            mid_price = self.client.get_mid_price(market_id)
            if mid_price is None:
                return None

            # Generate probability forecast
            forecast = self.model.forecast_event_probabilities()

            # Map forecast to market type
            market_title = market_data.get("title", "").lower()

            if "temperature" in market_title and "90" in market_title:
                our_estimate = forecast.get("temp_above_90f", 0.5)
            elif "temperature" in market_title and "95" in market_title:
                our_estimate = forecast.get("temp_above_95f", 0.5)
            elif "precipitation" in market_title or "rain" in market_title:
                our_estimate = forecast.get("precipitation", 0.5)
            else:
                our_estimate = 0.5

            # Calculate edge
            edge = abs(our_estimate - mid_price)

            return {
                "market_id": market_id,
                "market_title": market_title,
                "current_price": mid_price,
                "our_estimate": our_estimate,
                "edge": edge,
                "should_trade": edge > 0.05,  # 5% edge threshold
                "forecast": forecast,
            }

        except Exception as e:
            logger.error(f"Error analyzing market {market_id}: {e}")
            return None

    def calculate_order(
        self,
        market_analysis: Dict,
        available_capital: float
    ) -> Optional[Dict]:
        """
        Calculate order parameters using Kelly Criterion

        Args:
            market_analysis: Market analysis result
            available_capital: Available capital for this trade

        Returns:
            Order parameters
        """
        try:
            mid_price = market_analysis["current_price"]
            our_estimate = market_analysis["our_estimate"]
            market_id = market_analysis["market_id"]

            if not market_analysis["should_trade"]:
                return None

            # Determine side
            if our_estimate > mid_price:
                side = "BUY"
                outcome = "YES"
                prob = our_estimate
            else:
                side = "SELL"
                outcome = "NO"
                prob = 1 - our_estimate

            # Kelly Criterion position sizing
            b = 1 / mid_price - 1 if mid_price > 0 else 1
            kelly_fraction = (b * prob - (1 - prob)) / b if b > 0 else 0
            kelly_fraction = max(0, kelly_fraction * 0.25)  # 1/4 Kelly for safety

            # Apply position limit
            max_position = (available_capital * self.max_position_pct) / (mid_price * 100)
            position_size = min(kelly_fraction, max_position)

            if position_size < 0.01:  # Minimum position size
                return None

            return {
                "market_id": market_id,
                "outcome": outcome,
                "side": side,
                "price": mid_price,
                "size": position_size,
                "edge": market_analysis["edge"],
            }

        except Exception as e:
            logger.error(f"Error calculating order: {e}")
            return None

    def execute_order(self, order: Dict) -> bool:
        """
        Execute a trading order

        Args:
            order: Order parameters

        Returns:
            True if successful
        """
        try:
            logger.info(
                f"Executing: {order['side']} {order['size']:.4f} "
                f"{order['outcome']} @ {order['price']:.4f}"
            )

            result = self.client.place_order(
                market_id=order["market_id"],
                outcome=order["outcome"],
                side=order["side"],
                price=order["price"],
                size=order["size"]
            )

            if result:
                # Update state
                market_id = order["market_id"]
                if order["side"] == "BUY":
                    self.active_positions[market_id] = (
                        self.active_positions.get(market_id, 0) + order["size"]
                    )
                else:
                    self.active_positions[market_id] = (
                        self.active_positions.get(market_id, 0) - order["size"]
                    )

                trade_record = {
                    "timestamp": datetime.now(),
                    "market_id": market_id,
                    **order,
                    "order_id": result.get("orderId", "unknown"),
                }
                self.trade_history.append(trade_record)

                logger.info(f"Order executed successfully")
                return True

            return False

        except Exception as e:
            logger.error(f"Error executing order: {e}")
            return False

    def manage_positions(self):
        """Close positions with favorable outcomes"""
        try:
            for market_id, position in list(self.active_positions.items()):
                if position == 0:
                    continue

                # Get current price
                mid_price = self.client.get_mid_price(market_id)
                if mid_price is None:
                    continue

                # Simple profit taking: close if 10% profit
                entry_price = 0.5  # Approximation
                if position > 0 and mid_price > entry_price * 1.1:
                    logger.info(f"Taking profit on {market_id}")
                    self.client.close_position(market_id, position)
                    self.active_positions[market_id] = 0

        except Exception as e:
            logger.error(f"Error managing positions: {e}")

    def run_iteration(self, markets: Optional[List[Dict]] = None) -> Dict:
        """
        Run one trading iteration

        Args:
            markets: Markets to analyze (if None, scan all)

        Returns:
            Iteration statistics
        """
        try:
            if markets is None:
                markets = self.scan_markets()

            stats = {
                "timestamp": datetime.now(),
                "markets_scanned": len(markets),
                "opportunities": 0,
                "orders_placed": 0,
                "positions": len([p for p in self.active_positions.values() if p != 0]),
            }

            # Get available capital
            balance = self.client.get_account_balance()
            available_capital = balance if balance else self.initial_capital

            # Analyze each market
            for market in markets:
                market_id = market.get("id")
                if not market_id:
                    continue

                # Analyze
                analysis = self.analyze_market(market_id, market)
                if not analysis or not analysis["should_trade"]:
                    continue

                stats["opportunities"] += 1
                logger.info(f"Opportunity found in {market_id}")

                # Calculate order
                order = self.calculate_order(analysis, available_capital)
                if not order:
                    continue

                # Execute
                if self.execute_order(order):
                    stats["orders_placed"] += 1

            # Manage existing positions
            self.manage_positions()

            # Record P&L
            self._record_pnl()

            return stats

        except Exception as e:
            logger.error(f"Error in trading iteration: {e}")
            return {
                "timestamp": datetime.now(),
                "error": str(e),
            }

    def _record_pnl(self):
        """Record current P&L"""
        try:
            portfolio_value = self.client.get_portfolio_value()
            if portfolio_value is None:
                portfolio_value = self.initial_capital

            pnl = portfolio_value - self.initial_capital
            pnl_pct = (pnl / self.initial_capital) * 100

            self.pnl_history.append({
                "timestamp": datetime.now(),
                "portfolio_value": portfolio_value,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
            })

            if len(self.pnl_history) > 1:
                logger.info(f"P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")

        except Exception as e:
            logger.error(f"Error recording P&L: {e}")

    def run_backtest(
        self,
        historical_prices: pd.DataFrame,
        probability_forecasts: pd.DataFrame,
        max_iterations: int = 100
    ) -> Dict:
        """
        Backtest trading strategy

        Args:
            historical_prices: Historical market prices
            probability_forecasts: Probability forecasts
            max_iterations: Maximum iterations

        Returns:
            Backtest results
        """
        logger.info("Starting backtest...")

        cumulative_pnl = 0
        trades = 0
        wins = 0

        for i in range(min(len(historical_prices), max_iterations)):
            # Simulate one iteration
            if i % 10 == 0:
                logger.info(f"Iteration {i}/{max_iterations}")

            # Simple logic for demo
            time.sleep(0.01)  # Simulate processing

        return {
            "total_trades": trades,
            "winning_trades": wins,
            "win_rate": wins / trades if trades > 0 else 0,
            "final_pnl": cumulative_pnl,
            "pnl_history": self.pnl_history,
        }

    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        if not self.pnl_history:
            return {}

        pnls = [p["pnl_pct"] for p in self.pnl_history]

        return {
            "start_time": self.start_time,
            "current_time": datetime.now(),
            "total_duration": datetime.now() - self.start_time,
            "total_trades": len(self.trade_history),
            "current_positions": sum(1 for p in self.active_positions.values() if p != 0),
            "current_pnl_pct": pnls[-1] if pnls else 0,
            "max_pnl_pct": max(pnls) if pnls else 0,
            "min_pnl_pct": min(pnls) if pnls else 0,
            "avg_pnl_pct": np.mean(pnls) if pnls else 0,
        }

    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down trader...")

        # Close all positions
        for market_id, position in self.active_positions.items():
            if position != 0:
                logger.info(f"Closing position in {market_id}")
                self.client.close_position(market_id, position)

        logger.info("Trader shutdown complete")
