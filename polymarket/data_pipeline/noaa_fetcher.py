"""
NOAA weather data fetcher module
Retrieves historical weather observations from NOAA API
"""

import logging
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd

from polymarket.config import NOAA_API_KEY, NOAA_BASE_URL

logger = logging.getLogger(__name__)

# Major weather stations for continental US
MAJOR_STATIONS = {
    "GHCND:USW00023023": {"name": "NEW YORK, NY", "lat": 40.77, "lon": -73.87},
    "GHCND:USW00094846": {"name": "CHICAGO, IL", "lat": 41.99, "lon": -87.93},
    "GHCND:USW00012918": {"name": "LOS ANGELES, CA", "lat": 34.05, "lon": -118.24},
    "GHCND:USW00013060": {"name": "DALLAS, TX", "lat": 32.85, "lon": -96.85},
    "GHCND:USW00014827": {"name": "DENVER, CO", "lat": 39.74, "lon": -104.99},
    "GHCND:USW00093017": {"name": "MIAMI, FL", "lat": 25.80, "lon": -80.27},
    "GHCND:USW00024234": {"name": "BOSTON, MA", "lat": 42.36, "lon": -71.01},
}


class NOAAFetcher:
    """Fetches historical weather data from NOAA API"""

    def __init__(self, api_key: str = NOAA_API_KEY):
        self.api_key = api_key
        self.base_url = NOAA_BASE_URL
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"token": api_key})

    def fetch_daily_observations(
        self,
        station_id: str,
        start_date: str,
        end_date: str,
        data_types: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Fetch daily weather observations for a specific station

        Args:
            station_id: NOAA station ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            data_types: List of data types to fetch (TMAX, TMIN, PRCP, etc.)

        Returns:
            DataFrame with weather observations
        """
        if data_types is None:
            data_types = ["TMAX", "TMIN", "PRCP", "AWND"]

        url = f"{self.base_url}data"
        params = {
            "datasetid": "GHCND",
            "stationid": station_id,
            "startDate": start_date,
            "endDate": end_date,
            "datatypeid": data_types,
            "limit": 1000,
            "offset": 1,
        }

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "results" not in data:
                logger.warning(f"No data returned for station {station_id}")
                return pd.DataFrame()

            # Parse results into DataFrame
            observations = []
            for result in data["results"]:
                observations.append({
                    "date": result["date"],
                    "datatype": result["datatype"],
                    "value": result["value"],
                    "attributes": result.get("attributes", ""),
                })

            return pd.DataFrame(observations)

        except requests.RequestException as e:
            logger.error(f"Error fetching data for {station_id}: {e}")
            return pd.DataFrame()

    def fetch_bulk_data(
        self,
        station_ids: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch data for multiple stations and combine

        Args:
            station_ids: List of NOAA station IDs
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Combined DataFrame with all observations
        """
        all_data = []

        for station_id in station_ids:
            logger.info(f"Fetching data for {station_id}")
            df = self.fetch_daily_observations(station_id, start_date, end_date)
            if not df.empty:
                df["station_id"] = station_id
                all_data.append(df)

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()

    def fetch_major_us_stations(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch data for major US weather stations

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with observations from all major stations
        """
        station_ids = list(MAJOR_STATIONS.keys())
        return self.fetch_bulk_data(station_ids, start_date, end_date)

    def get_station_metadata(self, station_id: str) -> Dict:
        """Get metadata for a specific station"""
        return MAJOR_STATIONS.get(station_id, {})
