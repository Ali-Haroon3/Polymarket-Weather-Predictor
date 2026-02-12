"""
WeatherAPI.com data fetcher module
Retrieves historical weather data from WeatherAPI.com
Requires API key, global coverage
"""

import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional

from polymarket.config import WEATHERAPI_KEY

logger = logging.getLogger(__name__)

LOCATIONS = {
    "NYC": {"query": "New York", "name": "New York, NY"},
    "LA": {"query": "Los Angeles", "name": "Los Angeles, CA"},
    "London": {"query": "London", "name": "London, UK"},
    "Chicago": {"query": "Chicago", "name": "Chicago, IL"},
    "Dallas": {"query": "Dallas", "name": "Dallas, TX"},
    "Denver": {"query": "Denver", "name": "Denver, CO"},
    "Miami": {"query": "Miami", "name": "Miami, FL"},
    "Boston": {"query": "Boston", "name": "Boston, MA"},
}

BASE_URL = "https://api.weatherapi.com/v1/history.json"


class WeatherAPIFetcher:
    """Fetches historical weather data from WeatherAPI.com"""

    def __init__(self, api_key: str = WEATHERAPI_KEY):
        self.api_key = api_key
        self.session = requests.Session()

    def is_available(self) -> bool:
        """Check if API key is configured"""
        return bool(self.api_key)

    def fetch_daily_observations(
        self,
        query: str,
        start_date: str,
        end_date: str,
        location_id: str = "WAPI",
    ) -> pd.DataFrame:
        """
        Fetch daily weather observations for a location

        Args:
            query: Location query string
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            location_id: Identifier for the location

        Returns:
            DataFrame with standardized weather observations
        """
        if not self.is_available():
            logger.info("WeatherAPI: No API key configured, skipping")
            return pd.DataFrame()

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        observations = []

        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            params = {
                "key": self.api_key,
                "q": query,
                "dt": date_str,
            }

            try:
                response = self.session.get(BASE_URL, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()

                forecast = data.get("forecast", {}).get("forecastday", [])
                if forecast:
                    day = forecast[0].get("day", {})
                    observations.append({
                        "date": pd.Timestamp(date_str),
                        "station_id": f"WAPI_{location_id}",
                        "temperature_max": day.get("maxtemp_c"),
                        "temperature_min": day.get("mintemp_c"),
                        "temperature_mean": day.get("avgtemp_c"),
                        "precipitation_total": day.get("totalprecip_mm", 0),
                        "wind_speed_mean": day.get("maxwind_kph", 0) / 3.6,  # km/h to m/s
                    })

            except requests.RequestException as e:
                logger.debug(f"WeatherAPI error for {date_str}: {e}")

            current += timedelta(days=1)

        if not observations:
            return pd.DataFrame()

        df = pd.DataFrame(observations)
        logger.info(f"WeatherAPI: Fetched {len(df)} observations for {location_id}")
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
            query=location["query"],
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
