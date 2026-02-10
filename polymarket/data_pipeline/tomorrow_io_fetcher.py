"""
Tomorrow.io weather data fetcher module
Retrieves historical weather data from Tomorrow.io Timeline API
Requires API key, global coverage
"""

import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional

from polymarket.config import TOMORROW_IO_API_KEY

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

BASE_URL = "https://api.tomorrow.io/v4/timelines"


class TomorrowIOFetcher:
    """Fetches historical weather data from Tomorrow.io API"""

    def __init__(self, api_key: str = TOMORROW_IO_API_KEY):
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
        location_id: str = "TIO",
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
            logger.info("Tomorrow.io: No API key configured, skipping")
            return pd.DataFrame()

        params = {
            "location": f"{latitude},{longitude}",
            "fields": ",".join([
                "temperatureMax",
                "temperatureMin",
                "temperatureAvg",
                "precipitationIntensityAvg",
                "windSpeedAvg",
            ]),
            "timesteps": "1d",
            "startTime": f"{start_date}T00:00:00Z",
            "endTime": f"{end_date}T23:59:59Z",
            "apikey": self.api_key,
            "units": "metric",
        }

        try:
            response = self.session.get(BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            timelines = data.get("data", {}).get("timelines", [])
            if not timelines:
                logger.warning(f"Tomorrow.io: No data for {location_id}")
                return pd.DataFrame()

            observations = []
            for timeline in timelines:
                for interval in timeline.get("intervals", []):
                    values = interval.get("values", {})
                    start_time = interval.get("startTime", "")

                    observations.append({
                        "date": pd.Timestamp(start_time[:10]),
                        "station_id": f"TIO_{location_id}",
                        "temperature_max": values.get("temperatureMax"),
                        "temperature_min": values.get("temperatureMin"),
                        "temperature_mean": values.get("temperatureAvg"),
                        "precipitation_total": (
                            values.get("precipitationIntensityAvg", 0) * 24
                        ),  # mm/hr to mm/day
                        "wind_speed_mean": values.get("windSpeedAvg"),
                    })

            if not observations:
                return pd.DataFrame()

            df = pd.DataFrame(observations)
            logger.info(f"Tomorrow.io: Fetched {len(df)} observations for {location_id}")
            return df

        except requests.RequestException as e:
            logger.error(f"Tomorrow.io error for {location_id}: {e}")
            return pd.DataFrame()

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
