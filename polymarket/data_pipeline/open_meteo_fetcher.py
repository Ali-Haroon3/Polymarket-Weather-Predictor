"""
Open-Meteo weather data fetcher module
Retrieves historical weather data from Open-Meteo Archive API
Free, no API key required, global coverage
"""

import logging
import requests
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

LOCATIONS = {
    "NYC": {"lat": 40.71, "lon": -74.01, "name": "New York, NY"},
    "LA": {"lat": 34.05, "lon": -118.24, "name": "Los Angeles, CA"},
    "London": {"lat": 51.51, "lon": -0.13, "name": "London, UK"},
    "Chicago": {"lat": 41.88, "lon": -87.63, "name": "Chicago, IL"},
    "Dallas": {"lat": 32.78, "lon": -96.80, "name": "Dallas, TX"},
    "Denver": {"lat": 39.74, "lon": -104.99, "name": "Denver, CO"},
    "Miami": {"lat": 25.76, "lon": -80.19, "name": "Miami, FL"},
    "Boston": {"lat": 42.36, "lon": -71.06, "name": "Boston, MA"},
}

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"


class OpenMeteoFetcher:
    """Fetches historical weather data from Open-Meteo Archive API"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "PolymarketWeatherPredictor/1.0"})

    def fetch_daily_observations(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        location_id: str = "OPEN_METEO",
    ) -> pd.DataFrame:
        """
        Fetch daily weather observations for a location

        Args:
            latitude: Location latitude
            longitude: Location longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            location_id: Identifier for the location

        Returns:
            DataFrame with standardized weather observations
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join([
                "temperature_2m_max",
                "temperature_2m_min",
                "temperature_2m_mean",
                "precipitation_sum",
                "windspeed_10m_max",
                "winddirection_10m_dominant",
            ]),
            "timezone": "auto",
        }

        try:
            response = self.session.get(BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "daily" not in data:
                logger.warning(f"No daily data returned for {location_id}")
                return pd.DataFrame()

            daily = data["daily"]
            df = pd.DataFrame({
                "date": pd.to_datetime(daily["time"]),
                "station_id": location_id,
                "temperature_max": daily.get("temperature_2m_max"),
                "temperature_min": daily.get("temperature_2m_min"),
                "temperature_mean": daily.get("temperature_2m_mean"),
                "precipitation_total": daily.get("precipitation_sum"),
                "wind_speed_mean": daily.get("windspeed_10m_max"),
            })

            # Drop rows where all weather values are null
            weather_cols = ["temperature_max", "temperature_min", "temperature_mean",
                          "precipitation_total", "wind_speed_mean"]
            df = df.dropna(subset=weather_cols, how="all")

            logger.info(f"Open-Meteo: Fetched {len(df)} observations for {location_id}")
            return df

        except requests.RequestException as e:
            logger.error(f"Open-Meteo error for {location_id}: {e}")
            return pd.DataFrame()

    def fetch_location(
        self,
        location_key: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch data for a pre-configured location by key

        Args:
            location_key: Key from LOCATIONS dict (e.g., "NYC", "London")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with weather observations
        """
        location = LOCATIONS.get(location_key)
        if not location:
            logger.error(f"Unknown location: {location_key}")
            return pd.DataFrame()

        return self.fetch_daily_observations(
            latitude=location["lat"],
            longitude=location["lon"],
            start_date=start_date,
            end_date=end_date,
            location_id=f"OPEN_METEO_{location_key}",
        )

    def fetch_multiple_locations(
        self,
        location_keys: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch data for multiple locations and combine

        Args:
            location_keys: List of location keys
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Combined DataFrame with all observations
        """
        all_data = []
        for key in location_keys:
            df = self.fetch_location(key, start_date, end_date)
            if not df.empty:
                all_data.append(df)

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
