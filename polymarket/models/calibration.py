"""
Probability calibration and model evaluation
Implements Brier score and calibration analysis
"""

import numpy as np
import pandas as pd
from typing import Tuple


class CalibrationAnalyzer:
    """Analyzes and improves probability calibration"""

    @staticmethod
    def brier_score(predicted_probs: np.ndarray, outcomes: np.ndarray) -> float:
        """
        Calculate Brier score - mean squared error between predictions and outcomes

        Args:
            predicted_probs: Array of probability predictions [0, 1]
            outcomes: Array of binary outcomes [0, 1]

        Returns:
            Brier score (lower is better, 0 is perfect)
        """
        return np.mean((predicted_probs - outcomes) ** 2)

    @staticmethod
    def reliability_diagram(
        predicted_probs: np.ndarray,
        outcomes: np.ndarray,
        n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create reliability diagram data for calibration visualization

        Args:
            predicted_probs: Array of probability predictions
            outcomes: Array of binary outcomes
            n_bins: Number of bins for binning predictions

        Returns:
            Tuple of (bin_edges, bin_means, observed_frequencies)
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_means = []
        observed_frequencies = []

        for i in range(n_bins):
            mask = (predicted_probs >= bin_edges[i]) & (predicted_probs < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_means.append(predicted_probs[mask].mean())
                observed_frequencies.append(outcomes[mask].mean())
            else:
                bin_means.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                observed_frequencies.append(np.nan)

        return bin_edges, np.array(bin_means), np.array(observed_frequencies)

    @staticmethod
    def calibration_metrics(
        predicted_probs: np.ndarray,
        outcomes: np.ndarray
    ) -> dict:
        """
        Calculate comprehensive calibration metrics

        Args:
            predicted_probs: Array of probability predictions
            outcomes: Array of binary outcomes

        Returns:
            Dictionary with calibration metrics
        """
        brier = CalibrationAnalyzer.brier_score(predicted_probs, outcomes)

        # Expected calibration error
        ece = CalibrationAnalyzer.expected_calibration_error(predicted_probs, outcomes)

        # Sharpness
        sharpness = np.std(predicted_probs)

        # Resolution
        base_rate = np.mean(outcomes)
        resolution = np.mean((predicted_probs - base_rate) ** 2)

        # Coverage
        in_credible = (predicted_probs >= 0) & (predicted_probs <= 1)
        coverage = np.mean(in_credible)

        return {
            "brier_score": brier,
            "expected_calibration_error": ece,
            "sharpness": sharpness,
            "resolution": resolution,
            "coverage": coverage,
        }

    @staticmethod
    def expected_calibration_error(
        predicted_probs: np.ndarray,
        outcomes: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calculate expected calibration error

        Args:
            predicted_probs: Array of probability predictions
            outcomes: Array of binary outcomes
            n_bins: Number of bins

        Returns:
            Expected calibration error
        """
        bin_edges, bin_means, observed_freq = CalibrationAnalyzer.reliability_diagram(
            predicted_probs, outcomes, n_bins
        )

        # Calculate ECE as weighted average of bin calibration errors
        bin_sizes = np.histogram(predicted_probs, bin_edges)[0]
        bin_sizes = bin_sizes[bin_sizes > 0] / len(predicted_probs)

        valid_bins = ~np.isnan(observed_freq[bin_sizes > 0])
        bin_means_valid = bin_means[bin_sizes > 0][valid_bins]
        observed_freq_valid = observed_freq[bin_sizes > 0][valid_bins]
        bin_sizes_valid = bin_sizes[valid_bins]

        ece = np.sum(bin_sizes_valid * np.abs(bin_means_valid - observed_freq_valid))

        return ece

    @staticmethod
    def temperature_calibration(
        predicted_temps: np.ndarray,
        observed_temps: np.ndarray
    ) -> dict:
        """
        Calibration analysis for continuous temperature predictions

        Args:
            predicted_temps: Array of predicted temperatures
            observed_temps: Array of observed temperatures

        Returns:
            Dictionary with calibration metrics
        """
        mae = np.mean(np.abs(predicted_temps - observed_temps))
        rmse = np.sqrt(np.mean((predicted_temps - observed_temps) ** 2))
        r_squared = 1 - (
            np.sum((observed_temps - predicted_temps) ** 2) /
            np.sum((observed_temps - np.mean(observed_temps)) ** 2)
        )

        # Prediction interval coverage
        std_error = np.std(predicted_temps - observed_temps)
        coverage_68 = np.mean(
            np.abs(predicted_temps - observed_temps) <= std_error
        )
        coverage_95 = np.mean(
            np.abs(predicted_temps - observed_temps) <= 1.96 * std_error
        )

        return {
            "mae": mae,
            "rmse": rmse,
            "r_squared": r_squared,
            "coverage_68pct": coverage_68,
            "coverage_95pct": coverage_95,
        }
