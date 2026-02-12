"""
Polymarket API client for authentication and trading
Handles all interactions with Polymarket CLOB API
"""

import logging
import requests
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time

from polymarket.config import INITIAL_CAPITAL

logger = logging.getLogger(__name__)


class PolymarketClient:
    """
    Client for interacting with Polymarket CLOB API
    Supports both authentication and paper trading operations
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        private_key: str = "",
        base_url: str = "https://clob.polymarket.com",
        paper_trading: bool = True
    ):
        """
        Initialize Polymarket client

        Args:
            api_key: Polymarket API key
            api_secret: Polymarket API secret
            private_key: Private key for signing transactions
            base_url: Base URL for Polymarket API
            paper_trading: Use paper trading account
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.private_key = private_key
        self.base_url = base_url.rstrip("/")
        self.paper_trading = paper_trading
        self.session = requests.Session()
        self.authenticated = False

        if api_key and api_secret:
            self._authenticate()

    def _authenticate(self) -> bool:
        """
        Authenticate with Polymarket API

        Returns:
            True if authentication successful
        """
        try:
            headers = {
                "POLY-API-KEY": self.api_key,
                "POLY-API-SECRET": self.api_secret,
            }

            response = self.session.get(
                f"{self.base_url}/user",
                headers=headers,
                timeout=10
            )

            if response.status_code == 200:
                self.authenticated = True
                logger.info("Successfully authenticated with Polymarket")
                return True
            else:
                logger.error(f"Authentication failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False

    def get_markets(
        self,
        filter_name: str = "weather",
        limit: int = 100
    ) -> List[Dict]:
        """
        Get available markets from Polymarket

        Args:
            filter_name: Filter markets by name
            limit: Maximum number of markets to return

        Returns:
            List of market objects
        """
        try:
            params = {
                "filter": filter_name,
                "limit": limit,
            }

            response = self.session.get(
                f"{self.base_url}/markets",
                params=params,
                timeout=10
            )
            response.raise_for_status()

            markets = response.json()
            logger.info(f"Retrieved {len(markets)} markets")
            return markets

        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
            return []

    def get_market_by_id(self, market_id: str) -> Optional[Dict]:
        """
        Get specific market details

        Args:
            market_id: Polymarket market ID

        Returns:
            Market details or None
        """
        try:
            response = self.session.get(
                f"{self.base_url}/markets/{market_id}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"Error fetching market {market_id}: {e}")
            return None

    def get_orderbook(self, market_id: str) -> Optional[Dict]:
        """
        Get current orderbook for a market

        Args:
            market_id: Polymarket market ID

        Returns:
            Orderbook data (bids, asks)
        """
        try:
            response = self.session.get(
                f"{self.base_url}/markets/{market_id}/orderbook",
                timeout=10
            )
            response.raise_for_status()

            orderbook = response.json()
            logger.debug(f"Retrieved orderbook for {market_id}")
            return orderbook

        except Exception as e:
            logger.error(f"Error fetching orderbook: {e}")
            return None

    def get_mid_price(self, market_id: str) -> Optional[float]:
        """
        Get mid price for a market

        Args:
            market_id: Polymarket market ID

        Returns:
            Mid price (average of best bid/ask)
        """
        orderbook = self.get_orderbook(market_id)
        if not orderbook:
            return None

        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        if bids and asks:
            best_bid = float(bids[0][0]) if isinstance(bids[0], (list, tuple)) else float(bids[0])
            best_ask = float(asks[0][0]) if isinstance(asks[0], (list, tuple)) else float(asks[0])
            return (best_bid + best_ask) / 2

        return None

    def place_order(
        self,
        market_id: str,
        outcome: str,
        side: str,
        price: float,
        size: float,
        order_type: str = "limit"
    ) -> Optional[Dict]:
        """
        Place an order on Polymarket

        Args:
            market_id: Market to trade
            outcome: YES or NO
            side: BUY or SELL
            price: Order price (0-1)
            size: Order size
            order_type: limit or market

        Returns:
            Order confirmation or None
        """
        if not self.authenticated and not self.paper_trading:
            logger.error("Not authenticated. Cannot place order.")
            return None

        try:
            order_data = {
                "marketId": market_id,
                "outcome": outcome,
                "side": side,
                "price": price,
                "size": size,
                "orderType": order_type,
                "timestamp": int(time.time() * 1000),
            }

            if self.paper_trading:
                # Paper trading: simulate order
                logger.info(f"PAPER TRADE: {side} {size} {outcome} @ {price} in {market_id}")
                return {
                    "orderId": f"paper_{int(time.time() * 1000)}",
                    "status": "accepted",
                    "paper_trade": True,
                    **order_data
                }

            # Real trading
            headers = {
                "POLY-API-KEY": self.api_key,
                "POLY-API-SECRET": self.api_secret,
            }

            response = self.session.post(
                f"{self.base_url}/orders",
                json=order_data,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()

            order = response.json()
            logger.info(f"Order placed: {order}")
            return order

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order

        Args:
            order_id: Order ID to cancel

        Returns:
            True if successful
        """
        if self.paper_trading:
            logger.info(f"PAPER TRADE: Cancel order {order_id}")
            return True

        try:
            headers = {
                "POLY-API-KEY": self.api_key,
                "POLY-API-SECRET": self.api_secret,
            }

            response = self.session.delete(
                f"{self.base_url}/orders/{order_id}",
                headers=headers,
                timeout=10
            )
            response.raise_for_status()

            logger.info(f"Order cancelled: {order_id}")
            return True

        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False

    def get_orders(self) -> List[Dict]:
        """
        Get all open orders

        Returns:
            List of open orders
        """
        if not self.authenticated and not self.paper_trading:
            logger.error("Not authenticated. Cannot retrieve orders.")
            return []

        try:
            if self.paper_trading:
                # Return empty for paper trading
                return []

            headers = {
                "POLY-API-KEY": self.api_key,
                "POLY-API-SECRET": self.api_secret,
            }

            response = self.session.get(
                f"{self.base_url}/orders",
                headers=headers,
                timeout=10
            )
            response.raise_for_status()

            orders = response.json()
            logger.debug(f"Retrieved {len(orders)} orders")
            return orders

        except Exception as e:
            logger.error(f"Error retrieving orders: {e}")
            return []

    def get_positions(self) -> Dict[str, float]:
        """
        Get current positions

        Returns:
            Dictionary of market_id: position_size
        """
        if not self.authenticated and not self.paper_trading:
            logger.error("Not authenticated. Cannot retrieve positions.")
            return {}

        try:
            if self.paper_trading:
                # Return empty for paper trading
                return {}

            headers = {
                "POLY-API-KEY": self.api_key,
                "POLY-API-SECRET": self.api_secret,
            }

            response = self.session.get(
                f"{self.base_url}/user/positions",
                headers=headers,
                timeout=10
            )
            response.raise_for_status()

            positions = response.json()
            logger.debug(f"Retrieved positions for {len(positions)} markets")
            return positions

        except Exception as e:
            logger.error(f"Error retrieving positions: {e}")
            return {}

    def get_account_balance(self) -> Optional[float]:
        """
        Get account balance in USDC

        Returns:
            Account balance or None
        """
        if not self.authenticated and not self.paper_trading:
            logger.error("Not authenticated.")
            return None

        try:
            if self.paper_trading:
                # Return paper trading balance
                return INITIAL_CAPITAL

            headers = {
                "POLY-API-KEY": self.api_key,
                "POLY-API-SECRET": self.api_secret,
            }

            response = self.session.get(
                f"{self.base_url}/user/balance",
                headers=headers,
                timeout=10
            )
            response.raise_for_status()

            balance = response.json().get("balance", 0)
            logger.info(f"Account balance: ${balance:.2f}")
            return balance

        except Exception as e:
            logger.error(f"Error retrieving balance: {e}")
            return None

    def get_portfolio_value(self) -> Optional[float]:
        """
        Get total portfolio value (cash + positions)

        Returns:
            Portfolio value or None
        """
        balance = self.get_account_balance()
        if balance is None:
            return None

        try:
            positions = self.get_positions()
            position_value = sum(abs(v) for v in positions.values()) * 0.5  # Rough estimate

            return balance + position_value

        except Exception as e:
            logger.error(f"Error calculating portfolio value: {e}")
            return balance

    def market_order(
        self,
        market_id: str,
        outcome: str,
        side: str,
        size: float
    ) -> Optional[Dict]:
        """
        Execute a market order (instant execution)

        Args:
            market_id: Market ID
            outcome: YES or NO
            side: BUY or SELL
            size: Order size

        Returns:
            Order confirmation
        """
        # Get current mid price
        mid_price = self.get_mid_price(market_id)
        if mid_price is None:
            logger.error(f"Cannot get price for {market_id}")
            return None

        # Apply slippage for market order
        if side == "BUY":
            price = mid_price + 0.005  # Slippage
        else:
            price = mid_price - 0.005

        return self.place_order(
            market_id=market_id,
            outcome=outcome,
            side=side,
            price=price,
            size=size,
            order_type="market"
        )

    def close_position(
        self,
        market_id: str,
        current_position: float
    ) -> Optional[Dict]:
        """
        Close an open position

        Args:
            market_id: Market ID
            current_position: Current position size

        Returns:
            Order confirmation
        """
        if current_position > 0:
            # Sell to close long position
            outcome = "YES"
            side = "SELL"
        else:
            # Buy to close short position
            outcome = "NO"
            side = "BUY"

        return self.market_order(
            market_id=market_id,
            outcome=outcome,
            side=side,
            size=abs(current_position)
        )

    def health_check(self) -> bool:
        """Check if API is accessible"""
        try:
            response = self.session.get(
                f"{self.base_url}/markets",
                params={"limit": 1},
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
