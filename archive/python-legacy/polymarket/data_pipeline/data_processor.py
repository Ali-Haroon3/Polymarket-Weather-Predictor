"""
Data processing and validation pipeline
Cleans, validates, and normalizes weather data
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple

from polymarket.config import DATA_PIPELINE_PARAMS
from polymarket.database import WeatherObservation, get_session

logger = logging.getLogger(__name__)


class DataProcessor:
    """Processes and validates weather observations"""

    def __init__(self):
        self.validation_threshold = DATA_PIPELINE_PARAMS["validation_threshold"]
        self.missing_data_threshold = DATA_PIPELINE_PARAMS["missing_data_threshold"]

    def process_raw_observations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw NOAA data into standardized format

        Args:
            df: Raw NOAA data DataFrame

        Returns:
            Processed DataFrame with standard schema
        """
        # Pivot data so each row is a date, columns are datatypes
        df_pivot = df.pivot_table(
            index=["date", "station_id"],
            columns="datatype",
            values="value",
            aggfunc="first"
        ).reset_index()

        # Rename columns to standard names
        df_pivot = df_pivot.rename(columns={
            "TMAX": "temperature_max",
            "TMIN": "temperature_min",
            "PRCP": "precipitation_total",
            "AWND": "wind_speed_mean",
        })

        # Convert to appropriate dtypes
        df_pivot["date"] = pd.to_datetime(df_pivot["date"])
        for col in ["temperature_max", "temperature_min", "precipitation_total"]:
            if col in df_pivot.columns:
                df_pivot[col] = pd.to_numeric(df_pivot[col], errors="coerce")

        # Calculate derived metrics
        df_pivot["temperature_mean"] = (
            df_pivot["temperature_max"] + df_pivot["temperature_min"]
        ) / 2

        return df_pivot

    def validate_observations(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Validate weather observations for quality

        Args:
            df: DataFrame with weather observations

        Returns:
            Tuple of (validated_df, quality_scores)
        """
        quality_scores = pd.Series(1.0, index=df.index)

        # Check for missing values
        missing_rate = df.isnull().sum() / len(df)
        for col, rate in missing_rate.items():
            if rate > self.missing_data_threshold:
                quality_scores *= (1 - rate)
                logger.warning(
                    f"Column {col} has {rate:.1%} missing data"
                )

        # Temperature range validation
        if "temperature_max" in df.columns and "temperature_min" in df.columns:
            invalid_temp = df["temperature_max"] < df["temperature_min"]
            quality_scores[invalid_temp] *= 0.5
            logger.info(f"Found {invalid_temp.sum()} invalid temperature pairs")

        # Physical plausibility checks
        if "temperature_mean" in df.columns:
            # Temperature should be between -50C and 50C for continental US
            implausible_temp = (df["temperature_mean"] < -50) | (df["temperature_mean"] > 50)
            quality_scores[implausible_temp] *= 0.1
            logger.info(f"Found {implausible_temp.sum()} implausible temperatures")

        if "precipitation_total" in df.columns:
            # Precipitation should be positive
            negative_precip = df["precipitation_total"] < 0
            quality_scores[negative_precip] = 0.0
            logger.info(f"Found {negative_precip.sum()} negative precipitation values")

        # Mark observations as validated if quality > threshold
        df["is_validated"] = quality_scores >= self.validation_threshold
        df["data_quality_score"] = quality_scores

        return df, quality_scores

    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize numerical features to [0, 1] range

        Args:
            df: DataFrame with weather observations

        Returns:
            DataFrame with normalized features
        """
        df_normalized = df.copy()

        # Temperature normalization (based on typical US range -20 to 45°C)
        if "temperature_mean" in df_normalized.columns:
            df_normalized["temperature_mean_norm"] = (
                (df_normalized["temperature_mean"] + 20) / 65
            ).clip(0, 1)

        # Precipitation normalization (mm, typical max ~100mm)
        if "precipitation_total" in df_normalized.columns:
            df_normalized["precipitation_norm"] = (
                df_normalized["precipitation_total"] / 100
            ).clip(0, 1)

        return df_normalized

    def save_to_database(self, df: pd.DataFrame, station_metadata: dict) -> int:
        """
        Save processed observations to database

        Args:
            df: Processed DataFrame
            station_metadata: Station information (name, lat, lon)

        Returns:
            Number of records inserted
        """
        session = get_session()
        count = 0

        try:
            for _, row in df.iterrows():
                obs = WeatherObservation(
                    station_id=row["station_id"],
                    station_name=station_metadata.get("name", "Unknown"),
                    latitude=station_metadata.get("lat", 0),
                    longitude=station_metadata.get("lon", 0),
                    observation_date=row["date"],
                    temperature_max=row.get("temperature_max"),
                    temperature_min=row.get("temperature_min"),
                    temperature_mean=row.get("temperature_mean"),
                    precipitation_total=row.get("precipitation_total"),
                    wind_speed_mean=row.get("wind_speed_mean"),
                    data_quality_score=row.get("data_quality_score", 1.0),
                    is_validated=row.get("is_validated", False),
                )
                session.add(obs)
                count += 1

            session.commit()
            logger.info(f"Inserted {count} records into database")

        except Exception as e:
            session.rollback()
            logger.error(f"Error saving to database: {e}")
            count = 0

        finally:
            session.close()

        return count

    def generate_statistics(self, df: pd.DataFrame) -> dict:
        """
        Generate summary statistics from observations

        Args:
            df: DataFrame with weather observations

        Returns:
            Dictionary with summary statistics
        """
        stats = {
            "total_records": len(df),
            "valid_records": (df["is_validated"] == True).sum(),
            "date_range": f"{df['date'].min()} to {df['date'].max()}",
            "temperature_stats": {
                "mean": df["temperature_mean"].mean(),
                "std": df["temperature_mean"].std(),
                "min": df["temperature_mean"].min(),
                "max": df["temperature_mean"].max(),
            },
            "precipitation_stats": {
                "mean": df["precipitation_total"].mean(),
                "std": df["precipitation_total"].std(),
                "max": df["precipitation_total"].max(),
                "days_with_rain": (df["precipitation_total"] > 0).sum(),
            },
            "data_quality": {
                "avg_quality_score": df["data_quality_score"].mean(),
                "min_quality_score": df["data_quality_score"].min(),
            },
        }
        return stats
