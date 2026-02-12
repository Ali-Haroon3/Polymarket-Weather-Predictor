"""
Backtesting engine for Polymarket weather prediction strategies
Orchestrates data fetching, market simulation, model prediction, and trade execution
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from polymarket.data_pipeline.multi_source_aggregator import MultiSourceAggregator
from polymarket.models.bayesian_model import BayesianWeatherModel
from polymarket.backtesting.market_simulator import MarketSimulator
from polymarket.backtesting.performance_metrics import PerformanceAnalyzer
from polymarket.config import BACKTEST_PARAMS, BAYESIAN_MODEL_PARAMS

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    End-to-end backtesting engine for the weather prediction trading strategy.

    Flow:
    1. Fetch historical weather data from multiple sources
    2. Generate simulated prediction markets from actual outcomes
    3. Run Bayesian model on trailing data to generate forecasts
    4. Trade based on model vs market disagreement (edge detection)
    5. Resolve trades against actual outcomes
    6. Compute performance metrics
    """

    def __init__(
        self,
        cities: List[str] = None,
        initial_capital: float = 100_000,
        max_position_pct: float = 0.10,
        edge_threshold: float = 0.05,
        kelly_fraction: float = 0.25,
        model_lookback_days: int = 60,
        market_noise_std: float = 0.10,
        seed: int = 42,
    ):
        """
        Args:
            cities: List of city keys to backtest
            initial_capital: Starting capital
            max_position_pct: Max position as fraction of capital
            edge_threshold: Minimum edge to trigger a trade
            kelly_fraction: Kelly criterion fraction (safety scaling)
            model_lookback_days: Days of trailing data to train model on
            market_noise_std: Noise in simulated market prices
            seed: Random seed
        """
        self.cities = cities or list(BACKTEST_PARAMS["cities"].keys())
        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.edge_threshold = edge_threshold
        self.kelly_fraction = kelly_fraction
        self.model_lookback_days = model_lookback_days
        self.seed = seed

        self.aggregator = MultiSourceAggregator()
        self.market_sim = MarketSimulator(
            market_noise_std=market_noise_std, seed=seed
        )

    def run(
        self,
        start_date: str = None,
        end_date: str = None,
        weather_data_override: Dict[str, pd.DataFrame] = None,
    ) -> Dict:
        """
        Run the full backtest

        Args:
            start_date: Backtest start date (YYYY-MM-DD). Defaults to 6 months ago.
            end_date: Backtest end date (YYYY-MM-DD). Defaults to today.
            weather_data_override: Optional pre-fetched weather data by city.

        Returns:
            Complete backtest results with metrics
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            months = BACKTEST_PARAMS["lookback_months"]
            start_dt = datetime.now() - timedelta(days=months * 30)
            start_date = start_dt.strftime("%Y-%m-%d")

        logger.info(f"Starting backtest: {start_date} to {end_date}")
        logger.info(f"Cities: {self.cities}")
        logger.info(f"Capital: ${self.initial_capital:,.0f}")

        # Step 1: Fetch weather data
        weather_data = weather_data_override or self._fetch_weather_data(
            start_date, end_date
        )

        # Step 2: Generate simulated markets
        all_markets = self._generate_markets(weather_data)

        # Step 3: Run trading simulation
        trades, portfolio_values = self._simulate_trading(
            weather_data, all_markets
        )

        # Step 4: Compute metrics
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        metrics = PerformanceAnalyzer.compute_metrics(
            trades_df, portfolio_values, self.initial_capital
        )

        # Add breakdowns
        if not trades_df.empty:
            metrics["by_city"] = PerformanceAnalyzer.compute_city_breakdown(trades_df)
            metrics["by_market_type"] = PerformanceAnalyzer.compute_market_type_breakdown(trades_df)
            metrics["by_month"] = PerformanceAnalyzer.compute_monthly_breakdown(trades_df)

        results = {
            "config": {
                "start_date": start_date,
                "end_date": end_date,
                "cities": self.cities,
                "initial_capital": self.initial_capital,
                "edge_threshold": self.edge_threshold,
                "kelly_fraction": self.kelly_fraction,
                "model_lookback_days": self.model_lookback_days,
            },
            "data_sources": {
                city: list(df["sources"].unique()) if "sources" in df.columns else ["unknown"]
                for city, df in weather_data.items()
            },
            "metrics": metrics,
            "portfolio_values": portfolio_values,
            "total_markets_generated": len(all_markets),
            "trades": trades,
        }

        logger.info("Backtest complete")
        return results

    def _fetch_weather_data(
        self, start_date: str, end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """Fetch weather data for all cities from all available sources"""
        # Fetch extra lookback for model training
        fetch_start = (
            datetime.strptime(start_date, "%Y-%m-%d")
            - timedelta(days=self.model_lookback_days)
        ).strftime("%Y-%m-%d")

        logger.info(f"Fetching weather data from {fetch_start} to {end_date}")
        return self.aggregator.aggregate_multiple(self.cities, fetch_start, end_date)

    def _generate_markets(
        self, weather_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Generate simulated markets for all cities"""
        all_markets = []
        for city, data in weather_data.items():
            markets = self.market_sim.generate_markets(data, city)
            all_markets.append(markets)

        if all_markets:
            combined = pd.concat(all_markets, ignore_index=True)
            logger.info(f"Generated {len(combined)} total simulated markets")
            return combined

        return pd.DataFrame()

    def _simulate_trading(
        self,
        weather_data: Dict[str, pd.DataFrame],
        markets: pd.DataFrame,
    ) -> Tuple[List[Dict], List[float]]:
        """
        Simulate trading across all cities and dates.
        For each day, trains model on trailing data and trades any markets with edge.
        """
        if markets.empty:
            return [], [self.initial_capital]

        capital = self.initial_capital
        portfolio_values = [capital]
        all_trades = []

        # Sort markets by date
        markets = markets.sort_values("date").reset_index(drop=True)
        dates = sorted(markets["date"].unique())

        for date in dates:
            day_markets = markets[markets["date"] == date]

            for _, market in day_markets.iterrows():
                city = market["city"]
                city_data = weather_data.get(city)
                if city_data is None:
                    continue

                # Get trailing data for model training
                city_data = city_data.copy()
                city_data["date"] = pd.to_datetime(city_data["date"])
                trailing = city_data[
                    (city_data["date"] < date)
                    & (city_data["date"] >= date - timedelta(days=self.model_lookback_days))
                ]

                if len(trailing) < 14:  # Need minimum data
                    continue

                # Train model on trailing data
                model = BayesianWeatherModel()
                try:
                    model.train(trailing)
                except Exception:
                    continue

                # Generate our probability estimate
                our_estimate = self._get_model_estimate(model, market)
                if our_estimate is None:
                    continue

                market_price = market["market_price"]
                edge = our_estimate - market_price

                # Trade if edge exceeds threshold
                if abs(edge) < self.edge_threshold:
                    continue

                # Position sizing via Kelly Criterion
                position_size = self._kelly_position_size(
                    our_estimate, market_price, capital
                )
                if position_size < 1.0:  # Minimum $1 trade
                    continue

                # Determine trade direction
                side = "BUY" if edge > 0 else "SELL"

                # Resolve trade against actual outcome
                actual = market["actual_outcome"]
                if side == "BUY":
                    pnl = position_size * (actual - market_price)
                else:
                    pnl = position_size * (market_price - actual)

                capital += pnl

                trade = {
                    "date": date,
                    "city": city,
                    "market_id": market["market_id"],
                    "market_title": market["market_title"],
                    "market_type": market["market_type"],
                    "threshold": market["threshold"],
                    "market_price": market_price,
                    "our_estimate": our_estimate,
                    "edge": edge,
                    "side": side,
                    "position_size": position_size,
                    "actual_outcome": actual,
                    "pnl": pnl,
                    "capital_after": capital,
                }
                all_trades.append(trade)

            portfolio_values.append(capital)

        logger.info(
            f"Simulated {len(all_trades)} trades over {len(dates)} days. "
            f"Final capital: ${capital:,.2f}"
        )
        return all_trades, portfolio_values

    def _get_model_estimate(
        self, model: BayesianWeatherModel, market: pd.Series
    ) -> Optional[float]:
        """Get the model's probability estimate for a market"""
        try:
            market_type = market["market_type"]

            if market_type == "temperature":
                threshold_f = market["threshold"]
                threshold_c = (threshold_f - 32) * 5 / 9
                prob = model.predict_temperature_exceeds(threshold_c)
                return float(np.clip(prob, 0.01, 0.99))

            elif market_type == "precipitation":
                prob, _ = model.predict_precipitation_probability()
                return float(np.clip(prob, 0.01, 0.99))

            return None

        except Exception as e:
            logger.debug(f"Model estimate error: {e}")
            return None

    def _kelly_position_size(
        self,
        probability: float,
        market_price: float,
        available_capital: float,
    ) -> float:
        """
        Calculate position size using fractional Kelly Criterion

        For a binary outcome market:
        - If buying YES at price p with true probability q:
          Kelly f* = (q - p) / (1 - p)  when q > p
        - If selling YES (buying NO) at price p with true probability q:
          Kelly f* = (p - q) / p  when q < p
        """
        if probability > market_price:
            # Buy YES
            edge = probability - market_price
            odds = 1 / market_price - 1
            if odds <= 0:
                return 0
            kelly = (odds * probability - (1 - probability)) / odds
        else:
            # Buy NO (sell YES)
            inv_prob = 1 - probability
            inv_price = 1 - market_price
            odds = 1 / inv_price - 1
            if odds <= 0:
                return 0
            kelly = (odds * inv_prob - probability) / odds

        kelly = max(0, kelly * self.kelly_fraction)

        # Cap at max position size
        max_pos = available_capital * self.max_position_pct
        position = min(kelly * available_capital, max_pos)

        return position
