"""
Multi-source weather data aggregator
Fetches from all available sources and merges into a unified dataset
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

from polymarket.data_pipeline.open_meteo_fetcher import OpenMeteoFetcher
from polymarket.data_pipeline.nws_fetcher import NWSFetcher
from polymarket.data_pipeline.noaa_fetcher import NOAAFetcher, MAJOR_STATIONS
from polymarket.data_pipeline.openweathermap_fetcher import OpenWeatherMapFetcher
from polymarket.data_pipeline.visual_crossing_fetcher import VisualCrossingFetcher
from polymarket.data_pipeline.weatherapi_fetcher import WeatherAPIFetcher
from polymarket.data_pipeline.tomorrow_io_fetcher import TomorrowIOFetcher

logger = logging.getLogger(__name__)

CITY_NOAA_STATIONS = {
    "NYC": "GHCND:USW00023023",
    "LA": "GHCND:USW00012918",
    "Chicago": "GHCND:USW00094846",
    "Dallas": "GHCND:USW00013060",
    "Denver": "GHCND:USW00014827",
    "Miami": "GHCND:USW00093017",
    "Boston": "GHCND:USW00024234",
}


class MultiSourceAggregator:
    """
    Aggregates weather data from multiple sources into a single unified dataset.
    Handles graceful degradation when API keys are missing.
    """

    def __init__(self):
        self.open_meteo = OpenMeteoFetcher()
        self.nws = NWSFetcher()
        self.noaa = NOAAFetcher()
        self.owm = OpenWeatherMapFetcher()
        self.vc = VisualCrossingFetcher()
        self.wapi = WeatherAPIFetcher()
        self.tio = TomorrowIOFetcher()

    def fetch_all_sources(
        self,
        location_key: str,
        start_date: str,
        end_date: str,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data from all available sources for a location

        Args:
            location_key: Location key (e.g., "NYC", "LA", "London")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Dictionary mapping source name to DataFrame
        """
        results = {}

        # 1. Open-Meteo (always available, no API key)
        logger.info(f"Fetching Open-Meteo data for {location_key}...")
        df = self.open_meteo.fetch_location(location_key, start_date, end_date)
        if not df.empty:
            results["open_meteo"] = df

        # 2. NWS (US only, no API key)
        logger.info(f"Fetching NWS data for {location_key}...")
        df = self.nws.fetch_location(location_key, start_date, end_date)
        if not df.empty:
            results["nws"] = df

        # 3. NOAA (US only, requires API key)
        noaa_station = CITY_NOAA_STATIONS.get(location_key)
        if noaa_station:
            logger.info(f"Fetching NOAA data for {location_key}...")
            try:
                df = self.noaa.fetch_daily_observations(
                    noaa_station, start_date, end_date
                )
                if not df.empty:
                    df = self._normalize_noaa_data(df, noaa_station)
                    results["noaa"] = df
            except Exception as e:
                logger.warning(f"NOAA fetch failed for {location_key}: {e}")

        # 4. OpenWeatherMap (requires API key)
        if self.owm.is_available():
            logger.info(f"Fetching OpenWeatherMap data for {location_key}...")
            df = self.owm.fetch_location(location_key, start_date, end_date)
            if not df.empty:
                results["openweathermap"] = df

        # 5. Visual Crossing (requires API key)
        if self.vc.is_available():
            logger.info(f"Fetching Visual Crossing data for {location_key}...")
            df = self.vc.fetch_location(location_key, start_date, end_date)
            if not df.empty:
                results["visual_crossing"] = df

        # 6. WeatherAPI (requires API key)
        if self.wapi.is_available():
            logger.info(f"Fetching WeatherAPI data for {location_key}...")
            df = self.wapi.fetch_location(location_key, start_date, end_date)
            if not df.empty:
                results["weatherapi"] = df

        # 7. Tomorrow.io (requires API key)
        if self.tio.is_available():
            logger.info(f"Fetching Tomorrow.io data for {location_key}...")
            df = self.tio.fetch_location(location_key, start_date, end_date)
            if not df.empty:
                results["tomorrow_io"] = df

        logger.info(
            f"Fetched data from {len(results)} sources for {location_key}: "
            f"{list(results.keys())}"
        )
        return results

    def _normalize_noaa_data(self, df: pd.DataFrame, station_id: str) -> pd.DataFrame:
        """Normalize raw NOAA data to standard format"""
        if df.empty:
            return df

        # NOAA returns data in long format, pivot to wide
        pivot = df.pivot_table(
            index="date",
            columns="datatype",
            values="value",
            aggfunc="first",
        ).reset_index()

        result = pd.DataFrame({
            "date": pd.to_datetime(pivot["date"]),
            "station_id": f"NOAA_{station_id}",
        })

        # NOAA temperatures are in tenths of degrees C
        if "TMAX" in pivot.columns:
            result["temperature_max"] = pivot["TMAX"] / 10.0
        if "TMIN" in pivot.columns:
            result["temperature_min"] = pivot["TMIN"] / 10.0
        if "TMAX" in pivot.columns and "TMIN" in pivot.columns:
            result["temperature_mean"] = (pivot["TMAX"] / 10.0 + pivot["TMIN"] / 10.0) / 2
        # NOAA precipitation is in tenths of mm
        if "PRCP" in pivot.columns:
            result["precipitation_total"] = pivot["PRCP"] / 10.0
        # NOAA wind is in tenths of m/s
        if "AWND" in pivot.columns:
            result["wind_speed_mean"] = pivot["AWND"] / 10.0

        return result

    def aggregate(
        self,
        location_key: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch from all sources and merge into a single dataset.
        Uses mean of all available sources for each field on each date.

        Args:
            location_key: Location key
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Unified DataFrame with averaged values and source metadata
        """
        source_data = self.fetch_all_sources(location_key, start_date, end_date)

        if not source_data:
            logger.warning(f"No data from any source for {location_key}")
            return pd.DataFrame()

        weather_cols = [
            "temperature_max", "temperature_min", "temperature_mean",
            "precipitation_total", "wind_speed_mean",
        ]

        # Collect all data by date
        all_frames = []
        for source_name, df in source_data.items():
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            df["source"] = source_name
            all_frames.append(df[["date", "source"] + [c for c in weather_cols if c in df.columns]])

        combined = pd.concat(all_frames, ignore_index=True)

        # Average across sources for each date
        aggregated = combined.groupby("date").agg(
            temperature_max=("temperature_max", "mean"),
            temperature_min=("temperature_min", "mean"),
            temperature_mean=("temperature_mean", "mean"),
            precipitation_total=("precipitation_total", "mean"),
            wind_speed_mean=("wind_speed_mean", "mean"),
            source_count=("source", "nunique"),
            sources=("source", lambda x: ",".join(sorted(x.unique()))),
        ).reset_index()

        aggregated["station_id"] = f"MULTI_{location_key}"
        aggregated["location"] = location_key

        # Add quality score based on number of sources
        max_sources = aggregated["source_count"].max()
        aggregated["data_quality_score"] = aggregated["source_count"] / max(max_sources, 1)
        aggregated["is_validated"] = aggregated["source_count"] >= 2

        logger.info(
            f"Aggregated {len(aggregated)} days for {location_key} "
            f"from {aggregated['source_count'].max()} sources max"
        )

        return aggregated

    def aggregate_multiple(
        self,
        location_keys: List[str],
        start_date: str,
        end_date: str,
    ) -> Dict[str, pd.DataFrame]:
        """
        Aggregate data for multiple locations

        Args:
            location_keys: List of location keys
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary mapping location key to aggregated DataFrame
        """
        results = {}
        for key in location_keys:
            df = self.aggregate(key, start_date, end_date)
            if not df.empty:
                results[key] = df
        return results
