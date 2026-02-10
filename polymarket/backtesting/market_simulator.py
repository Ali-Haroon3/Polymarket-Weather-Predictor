"""
Market simulator for backtesting
Generates realistic simulated Polymarket weather markets from actual weather data
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def fahrenheit_to_celsius(f: float) -> float:
    return (f - 32) * 5 / 9


class MarketSimulator:
    """
    Generates simulated Polymarket weather prediction markets
    based on actual historical weather data for backtesting.
    """

    # Temperature thresholds in Fahrenheit (as Polymarket uses F)
    DEFAULT_TEMP_THRESHOLDS_F = [32, 50, 60, 70, 80, 90]

    def __init__(
        self,
        temp_thresholds_f: List[int] = None,
        market_noise_std: float = 0.10,
        seed: int = 42,
    ):
        """
        Args:
            temp_thresholds_f: Temperature thresholds in Fahrenheit for market generation
            market_noise_std: Std dev of noise added to climatological probability
                              to simulate market inefficiency
            seed: Random seed for reproducibility
        """
        self.temp_thresholds_f = temp_thresholds_f or self.DEFAULT_TEMP_THRESHOLDS_F
        self.market_noise_std = market_noise_std
        self.rng = np.random.RandomState(seed)

    def generate_markets(
        self,
        weather_data: pd.DataFrame,
        city: str,
        climatology: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Generate simulated prediction markets from actual weather data.

        For each day, creates markets for:
        - "Will max temperature exceed X°F?" for each threshold
        - "Will it rain today?"

        Args:
            weather_data: DataFrame with columns: date, temperature_max,
                         temperature_min, temperature_mean, precipitation_total
            city: City name (for market naming)
            climatology: Optional climatological averages for computing market prices.
                        If None, uses a rolling 30-day window from the data itself.

        Returns:
            DataFrame with columns: date, market_id, market_title, market_type,
                                   threshold, market_price, actual_outcome
        """
        if weather_data.empty:
            return pd.DataFrame()

        weather_data = weather_data.copy()
        weather_data["date"] = pd.to_datetime(weather_data["date"])
        weather_data = weather_data.sort_values("date").reset_index(drop=True)

        markets = []

        # Compute rolling climatological probabilities for market price generation
        clim_probs = self._compute_climatological_probs(weather_data)

        for idx, row in weather_data.iterrows():
            date = row["date"]
            temp_max_c = row.get("temperature_max")
            temp_mean_c = row.get("temperature_mean")
            precip = row.get("precipitation_total", 0)

            if pd.isna(temp_max_c):
                continue

            # Convert to Fahrenheit for market thresholds
            temp_max_f = temp_max_c * 9 / 5 + 32

            # Temperature threshold markets
            for threshold_f in self.temp_thresholds_f:
                actual_outcome = 1.0 if temp_max_f >= threshold_f else 0.0

                # Market price = climatological probability + noise
                clim_key = f"temp_above_{threshold_f}f"
                clim_prob = clim_probs.get(clim_key, {}).get(idx, 0.5)
                market_price = self._generate_market_price(clim_prob)

                markets.append({
                    "date": date,
                    "market_id": f"{city}_temp_{threshold_f}f_{date.strftime('%Y%m%d')}",
                    "market_title": f"Will {city} max temperature exceed {threshold_f}°F?",
                    "market_type": "temperature",
                    "threshold": threshold_f,
                    "market_price": market_price,
                    "actual_outcome": actual_outcome,
                    "city": city,
                })

            # Precipitation market
            if not pd.isna(precip):
                actual_rain = 1.0 if precip > 0.1 else 0.0  # >0.1mm = rain
                clim_prob = clim_probs.get("precipitation", {}).get(idx, 0.3)
                market_price = self._generate_market_price(clim_prob)

                markets.append({
                    "date": date,
                    "market_id": f"{city}_rain_{date.strftime('%Y%m%d')}",
                    "market_title": f"Will it rain in {city} today?",
                    "market_type": "precipitation",
                    "threshold": 0.1,
                    "market_price": market_price,
                    "actual_outcome": actual_rain,
                    "city": city,
                })

        df = pd.DataFrame(markets)
        logger.info(
            f"Generated {len(df)} simulated markets for {city} "
            f"over {weather_data['date'].nunique()} days"
        )
        return df

    def _compute_climatological_probs(
        self, weather_data: pd.DataFrame, window: int = 30
    ) -> Dict[str, Dict[int, float]]:
        """
        Compute rolling climatological probabilities for each market type.
        Uses a trailing window to avoid look-ahead bias.
        """
        probs = {}

        # Temperature thresholds
        if "temperature_max" in weather_data.columns:
            temp_max_f = weather_data["temperature_max"] * 9 / 5 + 32
            for threshold_f in self.temp_thresholds_f:
                key = f"temp_above_{threshold_f}f"
                exceeded = (temp_max_f >= threshold_f).astype(float)
                rolling_prob = exceeded.rolling(window=window, min_periods=7).mean()
                # Fill NaN at start with overall mean
                rolling_prob = rolling_prob.fillna(exceeded.mean())
                probs[key] = rolling_prob.to_dict()

        # Precipitation
        if "precipitation_total" in weather_data.columns:
            rained = (weather_data["precipitation_total"] > 0.1).astype(float)
            rolling_prob = rained.rolling(window=window, min_periods=7).mean()
            rolling_prob = rolling_prob.fillna(rained.mean())
            probs["precipitation"] = rolling_prob.to_dict()

        return probs

    def _generate_market_price(self, climatological_prob: float) -> float:
        """
        Generate a realistic market price from climatological probability.
        Adds noise to simulate market inefficiency and disagreement.
        """
        noise = self.rng.normal(0, self.market_noise_std)
        price = climatological_prob + noise
        # Clamp to valid probability range with some spread from 0/1
        return float(np.clip(price, 0.02, 0.98))
