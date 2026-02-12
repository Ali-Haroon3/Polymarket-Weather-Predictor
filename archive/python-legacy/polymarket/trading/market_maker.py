"""
Market-making strategies for weather prediction markets
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from polymarket.config import MONTE_CARLO_PARAMS

logger = logging.getLogger(__name__)


class MarketMaker:
    """
    Market maker for Polymarket weather derivatives
    Manages bid-ask spreads and position sizing
    """

    def __init__(self, capital: float = 100000):
        self.capital = capital
        self.inventory = {}  # Track inventory by market
        self.pnl = 0.0

    def calculate_fair_value_spread(
        self,
        market_price: float,
        our_estimate: float,
        volatility: float,
        inventory: float = 0
    ) -> Dict[str, float]:
        """
        Calculate optimal bid-ask spread based on fair value and inventory

        Args:
            market_price: Current market mid price
            our_estimate: Our probability estimate (fair value)
            volatility: Market volatility
            inventory: Current inventory position

        Returns:
            Dictionary with bid and ask prices
        """
        # Base spread component (proportional to volatility)
        base_spread = volatility * 0.02  # 2% of volatility

        # Inventory component (penalize adverse positions)
        inventory_spread = max(0, abs(inventory) * 0.01)

        # Fair value component
        fair_value = our_estimate
        mid_price = market_price

        # Optimal spread
        half_spread = base_spread + inventory_spread

        # Asymmetric spread based on fair value
        if fair_value > mid_price:
            # We think price should go up, make bid tighter
            bid = mid_price - half_spread * 0.5
            ask = mid_price + half_spread * 1.5
        elif fair_value < mid_price:
            # We think price should go down, make ask tighter
            bid = mid_price - half_spread * 1.5
            ask = mid_price + half_spread * 0.5
        else:
            # Symmetric spread around market
            bid = mid_price - half_spread
            ask = mid_price + half_spread

        return {
            "bid": np.clip(bid, 0, 1),
            "ask": np.clip(ask, 0, 1),
            "spread": ask - bid,
            "mid": (bid + ask) / 2,
        }

    def optimal_bid_ask_spreads(
        self,
        market_prices: pd.DataFrame,
        probability_estimates: pd.Series,
        volatility: float = 0.15,
        inventory_limits: Dict[str, float] = None
    ) -> pd.DataFrame:
        """
        Calculate optimal spreads for a basket of markets

        Args:
            market_prices: DataFrame with market prices (columns: bid, ask, mid)
            probability_estimates: Series with our probability estimates
            volatility: Overall market volatility
            inventory_limits: Limits on inventory by market

        Returns:
            DataFrame with bid-ask recommendations
        """
        spreads = []

        for market_id in market_prices.index:
            market = market_prices.loc[market_id]
            estimate = probability_estimates.loc[market_id]
            inventory = self.inventory.get(market_id, 0)

            spread = self.calculate_fair_value_spread(
                market_price=market.get("mid", (market["bid"] + market["ask"]) / 2),
                our_estimate=estimate,
                volatility=volatility,
                inventory=inventory
            )

            spread["market_id"] = market_id
            spread["current_inventory"] = inventory
            spreads.append(spread)

        return pd.DataFrame(spreads).set_index("market_id")

    def calculate_position_sizes(
        self,
        probability_estimates: Dict[str, float],
        market_prices: Dict[str, float],
        volatility_estimates: Dict[str, float],
        max_position_pct: float = 0.1
    ) -> Dict[str, float]:
        """
        Calculate optimal position sizes using Kelly Criterion

        Args:
            probability_estimates: Probability estimates for each market
            market_prices: Current market prices
            volatility_estimates: Volatility estimates
            max_position_pct: Maximum position as % of capital

        Returns:
            Dictionary with recommended position sizes
        """
        position_sizes = {}

        for market_id, estimate in probability_estimates.items():
            market_price = market_prices.get(market_id, 0.5)
            volatility = volatility_estimates.get(market_id, 0.15)

            # Win probability
            p_win = estimate if estimate > market_price else 1 - estimate
            p_loss = 1 - p_win

            # Kelly Criterion: f* = (bp - q) / b
            # b = odds, p = probability, q = 1-p
            b = 1 / market_price - 1 if market_price > 0 else 1
            kelly_fraction = (b * p_win - p_loss) / b if b > 0 else 0

            # Apply fractional Kelly for safety (1/4 Kelly)
            kelly_fraction = max(0, kelly_fraction * 0.25)

            # Apply position size limit
            position_size = min(kelly_fraction, max_position_pct)

            position_sizes[market_id] = position_size

        return position_sizes

    def execute_market_orders(
        self,
        market_id: str,
        side: str,  # BUY or SELL
        size: float,
        price: float
    ) -> Dict:
        """
        Execute a market order

        Args:
            market_id: Market identifier
            side: BUY or SELL
            size: Position size
            price: Execution price

        Returns:
            Trade execution record
        """
        notional = size * price * 100  # 100x leverage

        if side == "BUY":
            if notional > self.capital:
                logger.warning(f"Insufficient capital for {notional} order")
                size = self.capital / (price * 100)
            self.inventory[market_id] = self.inventory.get(market_id, 0) + size
        elif side == "SELL":
            self.inventory[market_id] = self.inventory.get(market_id, 0) - size

        trade = {
            "market_id": market_id,
            "side": side,
            "size": size,
            "price": price,
            "notional": size * price * 100,
            "timestamp": pd.Timestamp.now(),
        }

        return trade

    def hedge_position(
        self,
        market_id: str,
        inventory: float,
        current_price: float,
        hedge_ratio: float = 0.5
    ) -> Dict:
        """
        Hedge an adverse inventory position

        Args:
            market_id: Market to hedge
            inventory: Current inventory
            current_price: Current market price
            hedge_ratio: Ratio to hedge (0-1)

        Returns:
            Hedge trade record
        """
        if inventory > 0:
            # Sell to reduce long position
            hedge_size = inventory * hedge_ratio
            return self.execute_market_orders(market_id, "SELL", hedge_size, current_price)
        else:
            # Buy to reduce short position
            hedge_size = abs(inventory) * hedge_ratio
            return self.execute_market_orders(market_id, "BUY", hedge_size, current_price)

    def calculate_pnl_by_market(
        self,
        current_prices: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate P&L for each market position

        Args:
            current_prices: Current market prices

        Returns:
            Dictionary with P&L by market
        """
        pnl_by_market = {}

        for market_id, inventory in self.inventory.items():
            if inventory != 0:
                current_price = current_prices.get(market_id, 0.5)
                # Simplified P&L calculation
                pnl = inventory * (current_price - 0.5) * 100

                pnl_by_market[market_id] = pnl

        return pnl_by_market

    def get_portfolio_metrics(self) -> Dict:
        """Calculate overall portfolio metrics"""
        total_position = sum(abs(v) for v in self.inventory.values())
        net_exposure = sum(self.inventory.values())

        return {
            "total_position": total_position,
            "net_exposure": net_exposure,
            "number_of_markets": len([m for m in self.inventory.values() if m != 0]),
            "total_pnl": self.pnl,
            "remaining_capital": self.capital - abs(self.pnl),
        }
