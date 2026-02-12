"""
National Weather Service (NWS) data fetcher module
Retrieves weather observations from api.weather.gov
Free, no API key required, US only
"""

import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

NWS_STATIONS = {
    "NYC": {"station_id": "KNYC", "name": "New York City, NY"},
    "LA": {"station_id": "KLAX", "name": "Los Angeles, CA"},
    "Chicago": {"station_id": "KORD", "name": "Chicago O'Hare, IL"},
    "Dallas": {"station_id": "KDFW", "name": "Dallas/Fort Worth, TX"},
    "Denver": {"station_id": "KDEN", "name": "Denver, CO"},
    "Miami": {"station_id": "KMIA", "name": "Miami, FL"},
    "Boston": {"station_id": "KBOS", "name": "Boston, MA"},
}

BASE_URL = "https://api.weather.gov"


class NWSFetcher:
    """Fetches weather observations from National Weather Service API"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "PolymarketWeatherPredictor/1.0 (weather-prediction@example.com)",
            "Accept": "application/geo+json",
        })

    def fetch_daily_observations(
        self,
        station_id: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch daily weather observations for a station.
        NWS API returns hourly observations; we aggregate to daily.

        Args:
            station_id: NWS station ID (e.g., "KNYC")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with daily aggregated weather observations
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        all_observations = []
        # NWS API limits to 500 observations per request; fetch in weekly chunks
        current = start
        while current < end:
            chunk_end = min(current + timedelta(days=7), end)
            obs = self._fetch_observations_chunk(station_id, current, chunk_end)
            all_observations.extend(obs)
            current = chunk_end

        if not all_observations:
            logger.warning(f"NWS: No observations for {station_id}")
            return pd.DataFrame()

        df = pd.DataFrame(all_observations)
        daily = self._aggregate_to_daily(df, station_id)

        logger.info(f"NWS: Fetched {len(daily)} daily observations for {station_id}")
        return daily

    def _fetch_observations_chunk(
        self,
        station_id: str,
        start: datetime,
        end: datetime,
    ) -> List[Dict]:
        """Fetch a chunk of observations from NWS API"""
        url = f"{BASE_URL}/stations/{station_id}/observations"
        params = {
            "start": start.strftime("%Y-%m-%dT00:00:00Z"),
            "end": end.strftime("%Y-%m-%dT23:59:59Z"),
        }

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            observations = []
            for feature in data.get("features", []):
                props = feature.get("properties", {})
                temp = props.get("temperature", {}).get("value")
                wind = props.get("windSpeed", {}).get("value")
                precip = props.get("precipitationLastHour", {}).get("value")
                timestamp = props.get("timestamp")

                if timestamp:
                    observations.append({
                        "timestamp": timestamp,
                        "temperature": temp,
                        "wind_speed": wind / 3.6 if wind is not None else None,  # km/h to m/s
                        "precipitation": precip if precip is not None else 0,
                    })

            return observations

        except requests.RequestException as e:
            logger.error(f"NWS error for {station_id}: {e}")
            return []

    def _aggregate_to_daily(self, df: pd.DataFrame, station_id: str) -> pd.DataFrame:
        """Aggregate hourly observations to daily summaries"""
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date

        daily = df.groupby("date").agg(
            temperature_max=("temperature", "max"),
            temperature_min=("temperature", "min"),
            temperature_mean=("temperature", "mean"),
            precipitation_total=("precipitation", "sum"),
            wind_speed_mean=("wind_speed", "mean"),
        ).reset_index()

        daily["date"] = pd.to_datetime(daily["date"])
        daily["station_id"] = f"NWS_{station_id}"

        return daily

    def fetch_location(
        self,
        location_key: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch data for a pre-configured location"""
        station = NWS_STATIONS.get(location_key)
        if not station:
            logger.warning(f"NWS: No station for {location_key} (US only)")
            return pd.DataFrame()

        return self.fetch_daily_observations(
            station_id=station["station_id"],
            start_date=start_date,
            end_date=end_date,
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
