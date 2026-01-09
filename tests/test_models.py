"""
Tests for Bayesian weather models and calibration
"""

import pytest
import numpy as np
import pandas as pd

from polymarket.models import BayesianWeatherModel, CalibrationAnalyzer


class TestBayesianWeatherModel:
    """Tests for Bayesian weather forecasting model"""

    @pytest.fixture
    def sample_data(self):
        """Generate sample weather observations"""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=365)
        temps = np.random.normal(15, 8, 365)  # Mean 15°C, std 8°C
        precips = np.random.binomial(1, 0.3, 365)  # 30% chance of rain

        return pd.DataFrame({
            "temperature_mean": temps,
            "precipitation_total": precips,
            "date": dates,
        })

    @pytest.fixture
    def model(self):
        """Initialize model"""
        return BayesianWeatherModel()

    def test_model_training(self, model, sample_data):
        """Test model training"""
        model.train(sample_data)
        assert model.is_trained

    def test_temperature_prediction(self, model, sample_data):
        """Test temperature probability prediction"""
        model.train(sample_data)
        prob = model.predict_temperature_exceeds(20)
        assert 0 <= prob <= 1

    def test_precipitation_prediction(self, model, sample_data):
        """Test precipitation probability prediction"""
        model.train(sample_data)
        prob, ci = model.predict_precipitation_probability()
        assert 0 <= prob <= 1
        assert ci[0] <= prob <= ci[1]

    def test_forecast_event_probabilities(self, model, sample_data):
        """Test comprehensive forecast generation"""
        model.train(sample_data)
        forecast = model.forecast_event_probabilities()
        assert "precipitation" in forecast
        assert "temp_above_90f" in forecast

    def test_posterior_intervals(self, model, sample_data):
        """Test credible interval calculation"""
        model.train(sample_data)
        intervals = model.get_posterior_intervals()
        assert "temperature" in intervals
        assert "precipitation" in intervals


class TestCalibrationAnalyzer:
    """Tests for probability calibration"""

    def test_brier_score(self):
        """Test Brier score calculation"""
        predictions = np.array([0.9, 0.1, 0.8, 0.2])
        outcomes = np.array([1, 0, 1, 0])
        score = CalibrationAnalyzer.brier_score(predictions, outcomes)
        assert 0 <= score <= 1

    def test_calibration_metrics(self):
        """Test comprehensive calibration metrics"""
        predictions = np.random.uniform(0, 1, 1000)
        outcomes = np.random.binomial(1, 0.5, 1000)
        metrics = CalibrationAnalyzer.calibration_metrics(predictions, outcomes)

        assert "brier_score" in metrics
        assert "expected_calibration_error" in metrics
        assert "coverage" in metrics

    def test_reliability_diagram(self):
        """Test reliability diagram generation"""
        predictions = np.random.uniform(0, 1, 1000)
        outcomes = (predictions > 0.5).astype(int)

        bin_edges, bin_means, freq = CalibrationAnalyzer.reliability_diagram(
            predictions, outcomes, n_bins=10
        )

        assert len(bin_means) == 10
        assert len(freq) == 10
