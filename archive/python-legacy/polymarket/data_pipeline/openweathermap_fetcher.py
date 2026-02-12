"""
OpenWeatherMap data fetcher module
Retrieves historical weather data from OpenWeatherMap API
Requires API key, global coverage
"""

import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from polymarket.config import OPENWEATHERMAP_API_KEY

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

BASE_URL = "https://api.openweathermap.org/data/3.0/onecall/day_summary"


class OpenWeatherMapFetcher:
    """Fetches historical weather data from OpenWeatherMap API"""

    def __init__(self, api_key: str = OPENWEATHERMAP_API_KEY):
        self.api_key = api_key
        self.session = requests.Session()

    def is_available(self) -> bool:
        """Check if API key is configured"""
        return bool(self.api_key)

    def fetch_daily_observations(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        location_id: str = "OWM",
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
        if not self.is_available():
            logger.info("OpenWeatherMap: No API key configured, skipping")
            return pd.DataFrame()

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        observations = []

        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            params = {
                "lat": latitude,
                "lon": longitude,
                "date": date_str,
                "appid": self.api_key,
                "units": "metric",
            }

            try:
                response = self.session.get(BASE_URL, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()

                temp = data.get("temperature", {})
                precip = data.get("precipitation", {})
                wind = data.get("wind", {})

                observations.append({
                    "date": pd.Timestamp(date_str),
                    "station_id": f"OWM_{location_id}",
                    "temperature_max": temp.get("max"),
                    "temperature_min": temp.get("min"),
                    "temperature_mean": (
                        (temp.get("max", 0) + temp.get("min", 0)) / 2
                        if temp.get("max") is not None and temp.get("min") is not None
                        else None
                    ),
                    "precipitation_total": precip.get("total", 0),
                    "wind_speed_mean": wind.get("max", {}).get("speed"),
                })

            except requests.RequestException as e:
                logger.debug(f"OpenWeatherMap error for {date_str}: {e}")

            current += timedelta(days=1)

        if not observations:
            return pd.DataFrame()

        df = pd.DataFrame(observations)
        logger.info(f"OpenWeatherMap: Fetched {len(df)} observations for {location_id}")
        return df

    def fetch_location(
        self,
        location_key: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch data for a pre-configured location"""
        location = LOCATIONS.get(location_key)
        if not location:
            logger.error(f"Unknown location: {location_key}")
            return pd.DataFrame()

        return self.fetch_daily_observations(
            latitude=location["lat"],
            longitude=location["lon"],
            start_date=start_date,
            end_date=end_date,
            location_id=location_key,
        )

    def fetch_multiple_locations(
        self,
        location_keys: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch data for multiple locations and combine"""
        all_data = []
        for key in location_keys:
            df = self.fetch_location(key, start_date, end_date)
            if not df.empty:
                all_data.append(df)

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
