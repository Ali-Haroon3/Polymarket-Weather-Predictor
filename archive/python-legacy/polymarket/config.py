"""
Configuration settings for Polymarket Weather Prediction System
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Database Configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/polymarket_weather"
)

# NOAA API Configuration
NOAA_API_KEY = os.getenv("NOAA_API_KEY", "")
NOAA_BASE_URL = os.getenv(
    "NOAA_BASE_URL",
    "https://www.ncei.noaa.gov/api/access/"
)

# Model Configuration
MODEL_LOOKBACK_DAYS = int(os.getenv("MODEL_LOOKBACK_DAYS", "365"))
FORECAST_HORIZON_DAYS = int(os.getenv("FORECAST_HORIZON_DAYS", "30"))
PROBABILITY_THRESHOLD = float(os.getenv("PROBABILITY_THRESHOLD", "0.5"))

# Trading Configuration
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "100000"))
MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", "0.1"))
MIN_BID_ASK_SPREAD = float(os.getenv("MIN_BID_ASK_SPREAD", "0.02"))

# Additional Weather API Keys
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")
VISUAL_CROSSING_API_KEY = os.getenv("VISUAL_CROSSING_API_KEY", "")
WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY", "")
TOMORROW_IO_API_KEY = os.getenv("TOMORROW_IO_API_KEY", "")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")

# Model hyperparameters
BAYESIAN_MODEL_PARAMS = {
    "temperature_prior_mu": 60,
    "temperature_prior_sigma": 15,
    "precipitation_prior_alpha": 1.0,
    "precipitation_prior_beta": 1.0,
    "mcmc_draws": 2000,
    "tune": 1000,
}

# Monte Carlo parameters
MONTE_CARLO_PARAMS = {
    "n_simulations": 10000,
    "random_seed": 42,
    "volatility_window": 30,
    "correlation_lookback": 90,
}

# Data processing parameters
DATA_PIPELINE_PARAMS = {
    "batch_size": 1000,
    "validation_threshold": 0.95,
    "missing_data_threshold": 0.05,
}

# Backtest parameters
BACKTEST_PARAMS = {
    "lookback_months": 6,
    "cities": {
        "NYC": {
            "lat": 40.71, "lon": -74.01,
            "noaa_station": "GHCND:USW00023023",
            "nws_station": "KNYC",
        },
        "LA": {
            "lat": 34.05, "lon": -118.24,
            "noaa_station": "GHCND:USW00012918",
            "nws_station": "KLAX",
        },
        "London": {
            "lat": 51.51, "lon": -0.13,
            "noaa_station": None,
            "nws_station": None,
        },
    },
    "temperature_thresholds_f": [32, 50, 60, 70, 80, 90],
    "edge_threshold": 0.05,
    "kelly_fraction": 0.25,
}
