"""
SQLAlchemy ORM models for weather data and trading records
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class WeatherObservation(Base):
    """Historical weather observation data from NOAA"""

    __tablename__ = "weather_observations"

    id = Column(Integer, primary_key=True)
    station_id = Column(String(50), nullable=False, index=True)
    station_name = Column(String(255), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    observation_date = Column(DateTime, nullable=False, index=True)

    # Temperature measurements
    temperature_max = Column(Float)  # Celsius
    temperature_min = Column(Float)
    temperature_mean = Column(Float)

    # Precipitation
    precipitation_total = Column(Float)  # mm
    precipitation_probability = Column(Float)  # 0-1

    # Wind
    wind_speed_mean = Column(Float)  # m/s
    wind_speed_max = Column(Float)

    # Other metrics
    humidity = Column(Float)  # 0-1
    pressure = Column(Float)  # hPa
    cloud_cover = Column(Float)  # 0-1
    visibility = Column(Float)  # km

    # Data quality
    data_quality_score = Column(Float, default=1.0)
    is_validated = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    forecasts = relationship("ProbabilityForecast", back_populates="observation")

    def __repr__(self):
        return (
            f"<WeatherObservation(station={self.station_id}, "
            f"date={self.observation_date}, temp={self.temperature_mean}C)>"
        )


class ProbabilityForecast(Base):
    """Bayesian probability forecasts for weather events"""

    __tablename__ = "probability_forecasts"

    id = Column(Integer, primary_key=True)
    observation_id = Column(Integer, ForeignKey("weather_observations.id"))
    station_id = Column(String(50), nullable=False, index=True)
    forecast_date = Column(DateTime, nullable=False, index=True)
    horizon_days = Column(Integer, nullable=False)  # Days ahead

    # Temperature forecasts
    prob_temp_above_90f = Column(Float)  # P(T > 90°F)
    prob_temp_above_95f = Column(Float)  # P(T > 95°F)
    prob_temp_below_32f = Column(Float)  # P(T < 32°F)
    expected_temperature = Column(Float)  # Mean prediction
    temperature_std = Column(Float)  # Standard deviation

    # Precipitation forecasts
    prob_precipitation = Column(Float)  # P(precipitation > 0mm)
    prob_heavy_precipitation = Column(Float)  # P(precipitation > 25mm)
    expected_precipitation = Column(Float)  # Expected value
    precipitation_std = Column(Float)

    # Model metadata
    model_version = Column(String(50), nullable=False)
    calibration_score = Column(Float)  # Brier score
    credible_interval_lower = Column(Float)
    credible_interval_upper = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)
    observation = relationship("WeatherObservation", back_populates="forecasts")

    trades = relationship("TradeExecution", back_populates="forecast")

    def __repr__(self):
        return (
            f"<ProbabilityForecast(station={self.station_id}, "
            f"date={self.forecast_date}, horizon={self.horizon_days}d)>"
        )


class TradeExecution(Base):
    """Record of executed trades on Polymarket"""

    __tablename__ = "trade_executions"

    id = Column(Integer, primary_key=True)
    forecast_id = Column(Integer, ForeignKey("probability_forecasts.id"))
    market_id = Column(String(100), nullable=False, index=True)
    market_type = Column(String(50), nullable=False)  # e.g., "temperature", "precipitation"

    # Order details
    order_date = Column(DateTime, nullable=False)
    order_type = Column(String(20), nullable=False)  # BUY, SELL, HEDGE
    position_size = Column(Float, nullable=False)

    # Pricing
    entry_price = Column(Float, nullable=False)
    bid_price = Column(Float)
    ask_price = Column(Float)

    # Position management
    stop_loss = Column(Float)
    take_profit = Column(Float)
    position_status = Column(String(50), default="OPEN")  # OPEN, CLOSED, HEDGED

    # P&L
    current_price = Column(Float)
    realized_pnl = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)

    # Risk metrics
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    forecast = relationship("ProbabilityForecast", back_populates="trades")

    def __repr__(self):
        return (
            f"<TradeExecution(market={self.market_id}, "
            f"type={self.order_type}, size={self.position_size}, pnl={self.realized_pnl})>"
        )
