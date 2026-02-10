"""
Visual Crossing weather data fetcher module
Retrieves historical weather data from Visual Crossing Weather API
Requires API key, global coverage
"""

import logging
import requests
import pandas as pd
from typing import List, Optional

from polymarket.config import VISUAL_CROSSING_API_KEY

logger = logging.getLogger(__name__)

LOCATIONS = {
    "NYC": {"query": "New York,NY,US", "name": "New York, NY"},
    "LA": {"query": "Los Angeles,CA,US", "name": "Los Angeles, CA"},
    "London": {"query": "London,UK", "name": "London, UK"},
    "Chicago": {"query": "Chicago,IL,US", "name": "Chicago, IL"},
    "Dallas": {"query": "Dallas,TX,US", "name": "Dallas, TX"},
    "Denver": {"query": "Denver,CO,US", "name": "Denver, CO"},
    "Miami": {"query": "Miami,FL,US", "name": "Miami, FL"},
    "Boston": {"query": "Boston,MA,US", "name": "Boston, MA"},
}

BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"


class VisualCrossingFetcher:
    """Fetches historical weather data from Visual Crossing API"""

    def __init__(self, api_key: str = VISUAL_CROSSING_API_KEY):
        self.api_key = api_key
        self.session = requests.Session()

    def is_available(self) -> bool:
        """Check if API key is configured"""
        return bool(self.api_key)

    def fetch_daily_observations(
        self,
        location_query: str,
        start_date: str,
        end_date: str,
        location_id: str = "VC",
    ) -> pd.DataFrame:
        """
        Fetch daily weather observations for a location

        Args:
            location_query: Location query string (e.g., "New York,NY,US")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            location_id: Identifier for the location

        Returns:
            DataFrame with standardized weather observations
        """
        if not self.is_available():
            logger.info("Visual Crossing: No API key configured, skipping")
            return pd.DataFrame()

        url = f"{BASE_URL}/{location_query}/{start_date}/{end_date}"
        params = {
            "unitGroup": "metric",
            "include": "days",
            "key": self.api_key,
            "contentType": "json",
        }

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            days = data.get("days", [])
            if not days:
                logger.warning(f"Visual Crossing: No data for {location_id}")
                return pd.DataFrame()

            observations = []
            for day in days:
                observations.append({
                    "date": pd.Timestamp(day["datetime"]),
                    "station_id": f"VC_{location_id}",
                    "temperature_max": day.get("tempmax"),
                    "temperature_min": day.get("tempmin"),
                    "temperature_mean": day.get("temp"),
                    "precipitation_total": day.get("precip", 0),
                    "wind_speed_mean": day.get("windspeed"),
                })

            df = pd.DataFrame(observations)
            logger.info(f"Visual Crossing: Fetched {len(df)} observations for {location_id}")
            return df

        except requests.RequestException as e:
            logger.error(f"Visual Crossing error for {location_id}: {e}")
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
            location_query=location["query"],
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
