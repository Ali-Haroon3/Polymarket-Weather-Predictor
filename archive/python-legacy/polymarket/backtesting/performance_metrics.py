"""
Performance metrics and analysis for backtesting results
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """Computes comprehensive performance metrics for backtest results"""

    @staticmethod
    def compute_metrics(trades: pd.DataFrame, portfolio_values: List[float],
                        initial_capital: float) -> Dict:
        """
        Compute full suite of performance metrics

        Args:
            trades: DataFrame with trade records
            portfolio_values: List of daily portfolio values
            initial_capital: Starting capital

        Returns:
            Dictionary with all performance metrics
        """
        metrics = {}

        # Portfolio-level metrics
        metrics["portfolio"] = PerformanceAnalyzer._portfolio_metrics(
            portfolio_values, initial_capital
        )

        # Trade-level metrics
        if not trades.empty:
            metrics["trades"] = PerformanceAnalyzer._trade_metrics(trades)
        else:
            metrics["trades"] = {"total_trades": 0}

        # Forecast accuracy (if outcome data available)
        if "our_estimate" in trades.columns and "actual_outcome" in trades.columns:
            metrics["forecast"] = PerformanceAnalyzer._forecast_metrics(trades)

        return metrics

    @staticmethod
    def _portfolio_metrics(portfolio_values: List[float],
                           initial_capital: float) -> Dict:
        """Compute portfolio-level performance metrics"""
        values = np.array(portfolio_values)
        if len(values) < 2:
            return {
                "initial_capital": initial_capital,
                "final_value": values[-1] if len(values) > 0 else initial_capital,
                "total_return_pct": 0,
            }

        final_value = values[-1]
        total_return = (final_value - initial_capital) / initial_capital

        # Daily returns
        daily_returns = np.diff(values) / values[:-1]
        daily_returns = daily_returns[np.isfinite(daily_returns)]

        # Sharpe ratio (annualized, assuming 365 trading days for crypto/prediction markets)
        sharpe = 0
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365)

        # Max drawdown
        running_max = np.maximum.accumulate(values)
        drawdowns = (values - running_max) / running_max
        max_drawdown = float(np.min(drawdowns))

        # Sortino ratio
        downside_returns = daily_returns[daily_returns < 0]
        sortino = 0
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            sortino = np.mean(daily_returns) / np.std(downside_returns) * np.sqrt(365)

        return {
            "initial_capital": initial_capital,
            "final_value": float(final_value),
            "total_return_pct": float(total_return * 100),
            "total_pnl": float(final_value - initial_capital),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown_pct": float(max_drawdown * 100),
            "volatility_annual_pct": float(np.std(daily_returns) * np.sqrt(365) * 100) if len(daily_returns) > 0 else 0,
            "best_day_pct": float(np.max(daily_returns) * 100) if len(daily_returns) > 0 else 0,
            "worst_day_pct": float(np.min(daily_returns) * 100) if len(daily_returns) > 0 else 0,
            "positive_days": int(np.sum(daily_returns > 0)),
            "negative_days": int(np.sum(daily_returns < 0)),
            "total_days": len(daily_returns),
        }

    @staticmethod
    def _trade_metrics(trades: pd.DataFrame) -> Dict:
        """Compute trade-level performance metrics"""
        total = len(trades)
        if total == 0:
            return {"total_trades": 0}

        pnls = trades["pnl"].values if "pnl" in trades.columns else np.array([])

        wins = pnls[pnls > 0] if len(pnls) > 0 else np.array([])
        losses = pnls[pnls < 0] if len(pnls) > 0 else np.array([])

        win_rate = len(wins) / total if total > 0 else 0
        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0
        profit_factor = abs(np.sum(wins) / np.sum(losses)) if len(losses) > 0 and np.sum(losses) != 0 else float("inf")

        return {
            "total_trades": total,
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate_pct": float(win_rate * 100),
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total_pnl": float(np.sum(pnls)) if len(pnls) > 0 else 0,
            "profit_factor": float(profit_factor),
            "largest_win": float(np.max(wins)) if len(wins) > 0 else 0,
            "largest_loss": float(np.min(losses)) if len(losses) > 0 else 0,
            "avg_trade_pnl": float(np.mean(pnls)) if len(pnls) > 0 else 0,
        }

    @staticmethod
    def _forecast_metrics(trades: pd.DataFrame) -> Dict:
        """Compute forecast accuracy metrics"""
        predictions = trades["our_estimate"].values
        outcomes = trades["actual_outcome"].values

        # Brier score
        brier_score = float(np.mean((predictions - outcomes) ** 2))

        # Log loss
        eps = 1e-15
        clipped = np.clip(predictions, eps, 1 - eps)
        log_loss = -float(np.mean(
            outcomes * np.log(clipped) + (1 - outcomes) * np.log(1 - clipped)
        ))

        # Calibration (binned)
        calibration = PerformanceAnalyzer._calibration_curve(predictions, outcomes)

        # Expected Calibration Error
        ece = PerformanceAnalyzer._expected_calibration_error(predictions, outcomes)

        return {
            "brier_score": brier_score,
            "log_loss": log_loss,
            "expected_calibration_error": ece,
            "calibration_bins": calibration,
        }

    @staticmethod
    def _calibration_curve(
        predictions: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
    ) -> List[Dict]:
        """Compute calibration curve data"""
        bins = np.linspace(0, 1, n_bins + 1)
        curve = []

        for i in range(n_bins):
            mask = (predictions >= bins[i]) & (predictions < bins[i + 1])
            if np.sum(mask) > 0:
                curve.append({
                    "bin_center": float((bins[i] + bins[i + 1]) / 2),
                    "predicted_mean": float(np.mean(predictions[mask])),
                    "observed_freq": float(np.mean(outcomes[mask])),
                    "count": int(np.sum(mask)),
                })

        return curve

    @staticmethod
    def _expected_calibration_error(
        predictions: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
    ) -> float:
        """Compute Expected Calibration Error"""
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total = len(predictions)

        for i in range(n_bins):
            mask = (predictions >= bins[i]) & (predictions < bins[i + 1])
            count = np.sum(mask)
            if count > 0:
                avg_pred = np.mean(predictions[mask])
                avg_outcome = np.mean(outcomes[mask])
                ece += (count / total) * abs(avg_pred - avg_outcome)

        return float(ece)

    @staticmethod
    def compute_city_breakdown(
        all_trades: pd.DataFrame,
    ) -> Dict[str, Dict]:
        """Compute per-city performance breakdown"""
        if "city" not in all_trades.columns:
            return {}

        breakdown = {}
        for city in all_trades["city"].unique():
            city_trades = all_trades[all_trades["city"] == city]
            pnls = city_trades["pnl"].values if "pnl" in city_trades.columns else np.array([])
            wins = pnls[pnls > 0] if len(pnls) > 0 else np.array([])

            breakdown[city] = {
                "total_trades": len(city_trades),
                "total_pnl": float(np.sum(pnls)) if len(pnls) > 0 else 0,
                "win_rate_pct": float(len(wins) / len(city_trades) * 100) if len(city_trades) > 0 else 0,
                "avg_trade_pnl": float(np.mean(pnls)) if len(pnls) > 0 else 0,
            }

        return breakdown

    @staticmethod
    def compute_market_type_breakdown(
        all_trades: pd.DataFrame,
    ) -> Dict[str, Dict]:
        """Compute per-market-type performance breakdown"""
        if "market_type" not in all_trades.columns:
            return {}

        breakdown = {}
        for mtype in all_trades["market_type"].unique():
            type_trades = all_trades[all_trades["market_type"] == mtype]
            pnls = type_trades["pnl"].values if "pnl" in type_trades.columns else np.array([])
            wins = pnls[pnls > 0] if len(pnls) > 0 else np.array([])

            breakdown[mtype] = {
                "total_trades": len(type_trades),
                "total_pnl": float(np.sum(pnls)) if len(pnls) > 0 else 0,
                "win_rate_pct": float(len(wins) / len(type_trades) * 100) if len(type_trades) > 0 else 0,
                "avg_trade_pnl": float(np.mean(pnls)) if len(pnls) > 0 else 0,
            }

        return breakdown

    @staticmethod
    def compute_monthly_breakdown(
        all_trades: pd.DataFrame,
    ) -> Dict[str, Dict]:
        """Compute monthly performance breakdown"""
        if "date" not in all_trades.columns or all_trades.empty:
            return {}

        all_trades = all_trades.copy()
        all_trades["month"] = pd.to_datetime(all_trades["date"]).dt.to_period("M").astype(str)

        breakdown = {}
        for month in sorted(all_trades["month"].unique()):
            month_trades = all_trades[all_trades["month"] == month]
            pnls = month_trades["pnl"].values if "pnl" in month_trades.columns else np.array([])
            wins = pnls[pnls > 0] if len(pnls) > 0 else np.array([])

            breakdown[month] = {
                "total_trades": len(month_trades),
                "total_pnl": float(np.sum(pnls)) if len(pnls) > 0 else 0,
                "win_rate_pct": float(len(wins) / len(month_trades) * 100) if len(month_trades) > 0 else 0,
            }

        return breakdown
