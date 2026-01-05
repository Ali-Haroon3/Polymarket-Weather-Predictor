"""
Bayesian inference models for weather probability estimation
Uses hierarchical Bayesian approach with PyMC for posterior inference
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional

from polymarket.config import BAYESIAN_MODEL_PARAMS

logger = logging.getLogger(__name__)


class BayesianWeatherModel:
    """
    Bayesian hierarchical model for weather probability forecasting
    Implements temperature and precipitation probability estimation
    """

    def __init__(self, params: Dict = None):
        self.params = params or BAYESIAN_MODEL_PARAMS
        self.temperature_model = None
        self.precipitation_model = None
        self.is_trained = False

    def train(self, observations: pd.DataFrame) -> None:
        """
        Train Bayesian models on historical observations

        Args:
            observations: DataFrame with historical weather data
        """
        logger.info("Training Bayesian weather models")

        # Extract relevant columns
        temps = observations["temperature_mean"].dropna().values
        precips = observations["precipitation_total"].dropna().values > 0

        # Train temperature model (Normal distribution)
        self._train_temperature_model(temps)

        # Train precipitation model (Bernoulli distribution)
        self._train_precipitation_model(precips)

        self.is_trained = True
        logger.info("Model training complete")

    def _train_temperature_model(self, temperatures: np.ndarray) -> None:
        """Train temperature distribution model"""
        # Use observed data statistics with informative priors
        prior_mu = self.params["temperature_prior_mu"]
        prior_sigma = self.params["temperature_prior_sigma"]

        # Posterior parameters (normal-normal conjugate model)
        n = len(temperatures)
        sample_mean = np.mean(temperatures)
        sample_var = np.var(temperatures, ddof=1)

        # Prior precision and data precision
        prior_precision = 1 / (prior_sigma ** 2)
        data_precision = n / sample_var

        # Posterior parameters
        posterior_precision = prior_precision + data_precision
        self.temp_posterior_mu = (
            prior_precision * prior_mu + data_precision * sample_mean
        ) / posterior_precision
        self.temp_posterior_sigma = np.sqrt(1 / posterior_precision)

        logger.info(
            f"Temperature model: μ={self.temp_posterior_mu:.2f}, "
            f"σ={self.temp_posterior_sigma:.2f}"
        )

    def _train_precipitation_model(self, precipitation_occurred: np.ndarray) -> None:
        """Train precipitation probability model"""
        # Use Beta-Binomial conjugate model
        alpha = self.params["precipitation_prior_alpha"]
        beta = self.params["precipitation_prior_beta"]

        successes = np.sum(precipitation_occurred)
        failures = len(precipitation_occurred) - successes

        # Posterior Beta parameters
        self.precip_posterior_alpha = alpha + successes
        self.precip_posterior_beta = beta + failures

        # Posterior mean (expected probability)
        self.precip_posterior_mean = (
            self.precip_posterior_alpha /
            (self.precip_posterior_alpha + self.precip_posterior_beta)
        )

        logger.info(
            f"Precipitation model: P(rain) = {self.precip_posterior_mean:.3f}"
        )

    def predict_temperature_exceeds(
        self,
        threshold: float,
        n_samples: int = 10000
    ) -> float:
        """
        Predict probability that temperature exceeds threshold

        Args:
            threshold: Temperature threshold (Celsius)
            n_samples: Number of MCMC samples

        Returns:
            Probability estimate with credible interval
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        # Sample from posterior distribution
        samples = np.random.normal(
            self.temp_posterior_mu,
            self.temp_posterior_sigma,
            n_samples
        )

        # Calculate probability
        prob = np.mean(samples > threshold)

        return prob

    def predict_temperature_range(
        self,
        lower: float,
        upper: float,
        n_samples: int = 10000
    ) -> float:
        """Predict probability temperature falls in range"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        samples = np.random.normal(
            self.temp_posterior_mu,
            self.temp_posterior_sigma,
            n_samples
        )

        prob = np.mean((samples >= lower) & (samples <= upper))
        return prob

    def predict_precipitation_probability(
        self,
        n_samples: int = 10000
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Predict probability of precipitation occurrence

        Args:
            n_samples: Number of samples for credible interval

        Returns:
            Tuple of (probability, (lower_bound, upper_bound))
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        # Sample from posterior Beta distribution
        samples = np.random.beta(
            self.precip_posterior_alpha,
            self.precip_posterior_beta,
            n_samples
        )

        prob = np.mean(samples)
        credible_lower = np.percentile(samples, 2.5)
        credible_upper = np.percentile(samples, 97.5)

        return prob, (credible_lower, credible_upper)

    def forecast_event_probabilities(
        self,
        temperature_thresholds: Dict[str, float] = None,
        n_samples: int = 10000
    ) -> Dict[str, float]:
        """
        Generate comprehensive probability forecast for key weather events

        Args:
            temperature_thresholds: Dictionary of event names and thresholds
            n_samples: Number of MCMC samples

        Returns:
            Dictionary with probability estimates for each event
        """
        if temperature_thresholds is None:
            temperature_thresholds = {
                "temp_above_90f": 32.2,  # 90°F in Celsius
                "temp_above_95f": 35.0,
                "temp_below_32f": 0.0,
                "temp_below_0f": -17.8,
            }

        forecast = {}

        # Temperature forecasts
        for event_name, threshold in temperature_thresholds.items():
            forecast[event_name] = self.predict_temperature_exceeds(
                threshold, n_samples
            )

        # Precipitation forecasts
        precip_prob, (ci_lower, ci_upper) = self.predict_precipitation_probability(
            n_samples
        )
        forecast["precipitation"] = precip_prob
        forecast["precipitation_ci_lower"] = ci_lower
        forecast["precipitation_ci_upper"] = ci_upper

        return forecast

    def get_posterior_intervals(
        self,
        credible_level: float = 0.95,
        n_samples: int = 10000
    ) -> Dict[str, Tuple[float, float]]:
        """
        Get credible intervals for model parameters

        Args:
            credible_level: Credible level (0.95 for 95%)
            n_samples: Number of samples

        Returns:
            Dictionary with credible intervals
        """
        alpha = (1 - credible_level) / 2

        # Temperature credible interval
        temp_samples = np.random.normal(
            self.temp_posterior_mu,
            self.temp_posterior_sigma,
            n_samples
        )
        temp_ci = (
            np.percentile(temp_samples, alpha * 100),
            np.percentile(temp_samples, (1 - alpha) * 100)
        )

        # Precipitation credible interval
        precip_samples = np.random.beta(
            self.precip_posterior_alpha,
            self.precip_posterior_beta,
            n_samples
        )
        precip_ci = (
            np.percentile(precip_samples, alpha * 100),
            np.percentile(precip_samples, (1 - alpha) * 100)
        )

        return {
            "temperature": temp_ci,
            "precipitation": precip_ci,
        }
