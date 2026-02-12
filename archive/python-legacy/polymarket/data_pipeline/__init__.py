"""
Data pipeline for multi-source weather data ingestion and processing
"""

from .noaa_fetcher import NOAAFetcher
from .data_processor import DataProcessor
from .open_meteo_fetcher import OpenMeteoFetcher
from .nws_fetcher import NWSFetcher
from .openweathermap_fetcher import OpenWeatherMapFetcher
from .visual_crossing_fetcher import VisualCrossingFetcher
from .weatherapi_fetcher import WeatherAPIFetcher
from .tomorrow_io_fetcher import TomorrowIOFetcher
from .multi_source_aggregator import MultiSourceAggregator

__all__ = [
    "NOAAFetcher",
    "DataProcessor",
    "OpenMeteoFetcher",
    "NWSFetcher",
    "OpenWeatherMapFetcher",
    "VisualCrossingFetcher",
    "WeatherAPIFetcher",
    "TomorrowIOFetcher",
    "MultiSourceAggregator",
]
