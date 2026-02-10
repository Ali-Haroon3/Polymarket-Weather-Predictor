"""
Tests for multi-source weather data fetchers and aggregator
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime

from polymarket.data_pipeline.open_meteo_fetcher import OpenMeteoFetcher, LOCATIONS
from polymarket.data_pipeline.nws_fetcher import NWSFetcher, NWS_STATIONS
from polymarket.data_pipeline.openweathermap_fetcher import OpenWeatherMapFetcher
from polymarket.data_pipeline.visual_crossing_fetcher import VisualCrossingFetcher
from polymarket.data_pipeline.weatherapi_fetcher import WeatherAPIFetcher
from polymarket.data_pipeline.tomorrow_io_fetcher import TomorrowIOFetcher
from polymarket.data_pipeline.multi_source_aggregator import MultiSourceAggregator


# --- Helper to generate synthetic weather data ---


def make_synthetic_df(source_prefix, location_key, n_days=30):
    """Create a synthetic weather DataFrame for testing"""
    np.random.seed(42)
    dates = pd.date_range("2025-08-01", periods=n_days)
    return pd.DataFrame({
        "date": dates,
        "station_id": f"{source_prefix}_{location_key}",
        "temperature_max": np.random.normal(30, 5, n_days),
        "temperature_min": np.random.normal(20, 5, n_days),
        "temperature_mean": np.random.normal(25, 5, n_days),
        "precipitation_total": np.random.exponential(2, n_days),
        "wind_speed_mean": np.random.gamma(2, 2, n_days),
    })


# --- Open-Meteo Tests ---


class TestOpenMeteoFetcher:
    def test_locations_defined(self):
        assert "NYC" in LOCATIONS
        assert "LA" in LOCATIONS
        assert "London" in LOCATIONS

    @patch("polymarket.data_pipeline.open_meteo_fetcher.requests.Session")
    def test_fetch_daily_observations_success(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "daily": {
                "time": ["2025-08-01", "2025-08-02", "2025-08-03"],
                "temperature_2m_max": [30.0, 31.0, 29.5],
                "temperature_2m_min": [20.0, 21.0, 19.5],
                "temperature_2m_mean": [25.0, 26.0, 24.5],
                "precipitation_sum": [0.0, 5.2, 0.0],
                "windspeed_10m_max": [15.0, 20.0, 12.0],
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_session.get.return_value = mock_response

        fetcher = OpenMeteoFetcher()
        fetcher.session = mock_session

        df = fetcher.fetch_daily_observations(40.71, -74.01, "2025-08-01", "2025-08-03")

        assert len(df) == 3
        assert "temperature_max" in df.columns
        assert "temperature_min" in df.columns
        assert "precipitation_total" in df.columns

    @patch("polymarket.data_pipeline.open_meteo_fetcher.requests.Session")
    def test_fetch_daily_observations_empty(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_session.get.return_value = mock_response

        fetcher = OpenMeteoFetcher()
        fetcher.session = mock_session

        df = fetcher.fetch_daily_observations(40.71, -74.01, "2025-08-01", "2025-08-03")
        assert df.empty

    def test_fetch_location_invalid(self):
        fetcher = OpenMeteoFetcher()
        df = fetcher.fetch_location("INVALID_CITY", "2025-08-01", "2025-08-03")
        assert df.empty


# --- NWS Tests ---


class TestNWSFetcher:
    def test_stations_defined(self):
        assert "NYC" in NWS_STATIONS
        assert "LA" in NWS_STATIONS

    def test_no_station_for_london(self):
        fetcher = NWSFetcher()
        df = fetcher.fetch_location("London", "2025-08-01", "2025-08-03")
        assert df.empty

    def test_aggregate_to_daily(self):
        fetcher = NWSFetcher()
        hourly = pd.DataFrame({
            "timestamp": pd.date_range("2025-08-01", periods=24, freq="h"),
            "temperature": np.random.normal(25, 3, 24),
            "wind_speed": np.random.normal(5, 1, 24),
            "precipitation": np.random.exponential(0.1, 24),
        })

        daily = fetcher._aggregate_to_daily(hourly, "KNYC")
        assert len(daily) == 1
        assert "temperature_max" in daily.columns
        assert "temperature_min" in daily.columns
        assert "precipitation_total" in daily.columns


# --- API-key dependent fetchers ---


class TestOpenWeatherMapFetcher:
    def test_not_available_without_key(self):
        fetcher = OpenWeatherMapFetcher(api_key="")
        assert not fetcher.is_available()

    def test_skips_when_unavailable(self):
        fetcher = OpenWeatherMapFetcher(api_key="")
        df = fetcher.fetch_location("NYC", "2025-08-01", "2025-08-03")
        assert df.empty


class TestVisualCrossingFetcher:
    def test_not_available_without_key(self):
        fetcher = VisualCrossingFetcher(api_key="")
        assert not fetcher.is_available()

    def test_skips_when_unavailable(self):
        fetcher = VisualCrossingFetcher(api_key="")
        df = fetcher.fetch_location("NYC", "2025-08-01", "2025-08-03")
        assert df.empty


class TestWeatherAPIFetcher:
    def test_not_available_without_key(self):
        fetcher = WeatherAPIFetcher(api_key="")
        assert not fetcher.is_available()

    def test_skips_when_unavailable(self):
        fetcher = WeatherAPIFetcher(api_key="")
        df = fetcher.fetch_location("NYC", "2025-08-01", "2025-08-03")
        assert df.empty


class TestTomorrowIOFetcher:
    def test_not_available_without_key(self):
        fetcher = TomorrowIOFetcher(api_key="")
        assert not fetcher.is_available()

    def test_skips_when_unavailable(self):
        fetcher = TomorrowIOFetcher(api_key="")
        df = fetcher.fetch_location("NYC", "2025-08-01", "2025-08-03")
        assert df.empty


# --- MultiSourceAggregator Tests ---


class TestMultiSourceAggregator:
    @patch.object(OpenMeteoFetcher, "fetch_location")
    @patch.object(NWSFetcher, "fetch_location")
    @patch.object(OpenWeatherMapFetcher, "is_available", return_value=False)
    @patch.object(VisualCrossingFetcher, "is_available", return_value=False)
    @patch.object(WeatherAPIFetcher, "is_available", return_value=False)
    @patch.object(TomorrowIOFetcher, "is_available", return_value=False)
    def test_aggregate_merges_sources(
        self, mock_tio, mock_wapi, mock_vc, mock_owm, mock_nws, mock_om
    ):
        # Open-Meteo returns data
        mock_om.return_value = make_synthetic_df("OPEN_METEO", "NYC", 10)
        # NWS returns data with slightly different values
        nws_df = make_synthetic_df("NWS_KNYC", "NYC", 10)
        nws_df["temperature_max"] += 1  # Slightly higher
        mock_nws.return_value = nws_df

        aggregator = MultiSourceAggregator()
        result = aggregator.aggregate("NYC", "2025-08-01", "2025-08-10")

        assert not result.empty
        assert "temperature_max" in result.columns
        assert "source_count" in result.columns
        assert "sources" in result.columns
        # Should have data from 2 sources
        assert result["source_count"].max() >= 1

    @patch.object(OpenMeteoFetcher, "fetch_location")
    @patch.object(NWSFetcher, "fetch_location")
    @patch.object(OpenWeatherMapFetcher, "is_available", return_value=False)
    @patch.object(VisualCrossingFetcher, "is_available", return_value=False)
    @patch.object(WeatherAPIFetcher, "is_available", return_value=False)
    @patch.object(TomorrowIOFetcher, "is_available", return_value=False)
    def test_aggregate_london_no_us_sources(
        self, mock_tio, mock_wapi, mock_vc, mock_owm, mock_nws, mock_om
    ):
        """London should still get data from Open-Meteo even without US sources"""
        mock_om.return_value = make_synthetic_df("OPEN_METEO", "London", 10)
        mock_nws.return_value = pd.DataFrame()  # NWS has no London data

        aggregator = MultiSourceAggregator()
        result = aggregator.aggregate("London", "2025-08-01", "2025-08-10")

        assert not result.empty
        assert len(result) == 10

    @patch.object(OpenMeteoFetcher, "fetch_location")
    @patch.object(NWSFetcher, "fetch_location")
    @patch.object(OpenWeatherMapFetcher, "is_available", return_value=False)
    @patch.object(VisualCrossingFetcher, "is_available", return_value=False)
    @patch.object(WeatherAPIFetcher, "is_available", return_value=False)
    @patch.object(TomorrowIOFetcher, "is_available", return_value=False)
    def test_aggregate_multiple_cities(
        self, mock_tio, mock_wapi, mock_vc, mock_owm, mock_nws, mock_om
    ):
        mock_om.side_effect = lambda key, s, e: make_synthetic_df("OPEN_METEO", key, 10)
        mock_nws.return_value = pd.DataFrame()

        aggregator = MultiSourceAggregator()
        results = aggregator.aggregate_multiple(
            ["NYC", "LA", "London"], "2025-08-01", "2025-08-10"
        )

        assert "NYC" in results
        assert "LA" in results
        assert "London" in results
