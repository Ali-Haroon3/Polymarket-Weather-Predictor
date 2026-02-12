use std::collections::HashMap;

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

use crate::types::TradeRecord;
use crate::utils::{max_drawdown, mean, std_dev};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioMetrics {
    pub initial_capital: f64,
    pub final_value: f64,
    pub total_return_pct: f64,
    pub total_pnl: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown_pct: f64,
    pub volatility_annual_pct: f64,
    pub best_day_pct: f64,
    pub worst_day_pct: f64,
    pub positive_days: usize,
    pub negative_days: usize,
    pub total_days: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeMetrics {
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate_pct: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub total_pnl: f64,
    pub profit_factor: f64,
    pub largest_win: f64,
    pub largest_loss: f64,
    pub avg_trade_pnl: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastMetrics {
    pub brier_score: f64,
    pub log_loss: f64,
    pub expected_calibration_error: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub portfolio: PortfolioMetrics,
    pub trades: TradeMetrics,
    pub forecast: Option<ForecastMetrics>,
    pub by_city: HashMap<String, HashMap<String, f64>>,
    pub by_market_type: HashMap<String, HashMap<String, f64>>,
    pub by_month: HashMap<String, HashMap<String, f64>>,
}

pub struct PerformanceAnalyzer;

impl PerformanceAnalyzer {
    pub fn compute_metrics(
        trades: &[TradeRecord],
        portfolio_values: &[f64],
        initial_capital: f64,
    ) -> PerformanceMetrics {
        let portfolio = Self::portfolio_metrics(portfolio_values, initial_capital);
        let trade_metrics = Self::trade_metrics(trades);

        let forecast = if trades.is_empty() {
            None
        } else {
            Some(Self::forecast_metrics(trades))
        };

        PerformanceMetrics {
            portfolio,
            trades: trade_metrics,
            forecast,
            by_city: Self::compute_city_breakdown(trades),
            by_market_type: Self::compute_market_type_breakdown(trades),
            by_month: Self::compute_monthly_breakdown(trades),
        }
    }

    pub fn portfolio_metrics(portfolio_values: &[f64], initial_capital: f64) -> PortfolioMetrics {
        if portfolio_values.len() < 2 {
            return PortfolioMetrics {
                initial_capital,
                final_value: *portfolio_values.last().unwrap_or(&initial_capital),
                total_return_pct: 0.0,
                total_pnl: 0.0,
                sharpe_ratio: 0.0,
                sortino_ratio: 0.0,
                max_drawdown_pct: 0.0,
                volatility_annual_pct: 0.0,
                best_day_pct: 0.0,
                worst_day_pct: 0.0,
                positive_days: 0,
                negative_days: 0,
                total_days: 0,
            };
        }

        let final_value = *portfolio_values.last().unwrap_or(&initial_capital);
        let total_return = (final_value - initial_capital) / initial_capital;
        let daily_returns: Vec<f64> = portfolio_values
            .windows(2)
            .filter_map(|w| if w[0] != 0.0 { Some((w[1] - w[0]) / w[0]) } else { None })
            .collect();

        let sharpe = {
            let sd = std_dev(&daily_returns, 0);
            if daily_returns.len() > 1 && sd > 0.0 {
                mean(&daily_returns) / sd * 365.0_f64.sqrt()
            } else {
                0.0
            }
        };

        let downside: Vec<f64> = daily_returns.iter().copied().filter(|r| *r < 0.0).collect();
        let sortino = {
            let sd = std_dev(&downside, 0);
            if !downside.is_empty() && sd > 0.0 {
                mean(&daily_returns) / sd * 365.0_f64.sqrt()
            } else {
                0.0
            }
        };

        PortfolioMetrics {
            initial_capital,
            final_value,
            total_return_pct: total_return * 100.0,
            total_pnl: final_value - initial_capital,
            sharpe_ratio: sharpe,
            sortino_ratio: sortino,
            max_drawdown_pct: max_drawdown(portfolio_values) * 100.0,
            volatility_annual_pct: std_dev(&daily_returns, 0) * 365.0_f64.sqrt() * 100.0,
            best_day_pct: daily_returns
                .iter()
                .copied()
                .reduce(f64::max)
                .unwrap_or(0.0)
                * 100.0,
            worst_day_pct: daily_returns
                .iter()
                .copied()
                .reduce(f64::min)
                .unwrap_or(0.0)
                * 100.0,
            positive_days: daily_returns.iter().filter(|r| **r > 0.0).count(),
            negative_days: daily_returns.iter().filter(|r| **r < 0.0).count(),
            total_days: daily_returns.len(),
        }
    }

    pub fn trade_metrics(trades: &[TradeRecord]) -> TradeMetrics {
        if trades.is_empty() {
            return TradeMetrics {
                total_trades: 0,
                winning_trades: 0,
                losing_trades: 0,
                win_rate_pct: 0.0,
                avg_win: 0.0,
                avg_loss: 0.0,
                total_pnl: 0.0,
                profit_factor: f64::INFINITY,
                largest_win: 0.0,
                largest_loss: 0.0,
                avg_trade_pnl: 0.0,
            };
        }

        let pnls: Vec<f64> = trades.iter().map(|t| t.pnl).collect();
        let wins: Vec<f64> = pnls.iter().copied().filter(|p| *p > 0.0).collect();
        let losses: Vec<f64> = pnls.iter().copied().filter(|p| *p < 0.0).collect();
        let win_rate = wins.len() as f64 / trades.len() as f64;

        let profit_factor = if losses.is_empty() || losses.iter().sum::<f64>() == 0.0 {
            f64::INFINITY
        } else {
            (wins.iter().sum::<f64>() / losses.iter().sum::<f64>()).abs()
        };

        TradeMetrics {
            total_trades: trades.len(),
            winning_trades: wins.len(),
            losing_trades: losses.len(),
            win_rate_pct: win_rate * 100.0,
            avg_win: mean(&wins),
            avg_loss: mean(&losses),
            total_pnl: pnls.iter().sum(),
            profit_factor,
            largest_win: wins.iter().copied().reduce(f64::max).unwrap_or(0.0),
            largest_loss: losses.iter().copied().reduce(f64::min).unwrap_or(0.0),
            avg_trade_pnl: mean(&pnls),
        }
    }

    pub fn forecast_metrics(trades: &[TradeRecord]) -> ForecastMetrics {
        let preds: Vec<f64> = trades.iter().map(|t| t.our_estimate).collect();
        let outs: Vec<f64> = trades.iter().map(|t| t.actual_outcome).collect();

        let brier = preds
            .iter()
            .zip(outs.iter())
            .map(|(p, y)| {
                let d = p - y;
                d * d
            })
            .sum::<f64>()
            / preds.len().max(1) as f64;

        let eps = 1e-15;
        let log_loss = -preds
            .iter()
            .zip(outs.iter())
            .map(|(p, y)| {
                let clipped = p.max(eps).min(1.0 - eps);
                y * clipped.ln() + (1.0 - y) * (1.0 - clipped).ln()
            })
            .sum::<f64>()
            / preds.len().max(1) as f64;

        let ece = expected_calibration_error(&preds, &outs, 10);

        ForecastMetrics {
            brier_score: brier,
            log_loss,
            expected_calibration_error: ece,
        }
    }

    pub fn compute_city_breakdown(trades: &[TradeRecord]) -> HashMap<String, HashMap<String, f64>> {
        let mut buckets: HashMap<String, Vec<&TradeRecord>> = HashMap::new();
        for trade in trades {
            buckets.entry(trade.city.clone()).or_default().push(trade);
        }
        summarize_buckets(buckets)
    }

    pub fn compute_market_type_breakdown(
        trades: &[TradeRecord],
    ) -> HashMap<String, HashMap<String, f64>> {
        let mut buckets: HashMap<String, Vec<&TradeRecord>> = HashMap::new();
        for trade in trades {
            buckets
                .entry(trade.market_type.clone())
                .or_default()
                .push(trade);
        }
        summarize_buckets(buckets)
    }

    pub fn compute_monthly_breakdown(trades: &[TradeRecord]) -> HashMap<String, HashMap<String, f64>> {
        let mut buckets: HashMap<String, Vec<&TradeRecord>> = HashMap::new();
        for trade in trades {
            let month = trade.date.format("%Y-%m").to_string();
            buckets.entry(month).or_default().push(trade);
        }
        summarize_buckets(buckets)
    }
}

fn summarize_buckets(
    buckets: HashMap<String, Vec<&TradeRecord>>,
) -> HashMap<String, HashMap<String, f64>> {
    let mut out = HashMap::new();

    for (k, trades) in buckets {
        let pnls: Vec<f64> = trades.iter().map(|t| t.pnl).collect();
        let wins = pnls.iter().filter(|p| **p > 0.0).count() as f64;
        let total = trades.len().max(1) as f64;

        let mut row = HashMap::new();
        row.insert("total_trades".to_string(), trades.len() as f64);
        row.insert("total_pnl".to_string(), pnls.iter().sum());
        row.insert("win_rate_pct".to_string(), wins / total * 100.0);
        row.insert("avg_trade_pnl".to_string(), mean(&pnls));
        out.insert(k, row);
    }

    out
}

fn expected_calibration_error(predictions: &[f64], outcomes: &[f64], n_bins: usize) -> f64 {
    if predictions.is_empty() || predictions.len() != outcomes.len() {
        return 0.0;
    }

    let mut ece = 0.0;
    for i in 0..n_bins {
        let lo = i as f64 / n_bins as f64;
        let hi = (i + 1) as f64 / n_bins as f64;

        let mut p_bin = Vec::new();
        let mut y_bin = Vec::new();
        for (p, y) in predictions.iter().zip(outcomes.iter()) {
            if *p >= lo && *p < hi {
                p_bin.push(*p);
                y_bin.push(*y);
            }
        }

        if !p_bin.is_empty() {
            let w = p_bin.len() as f64 / predictions.len() as f64;
            ece += w * (mean(&p_bin) - mean(&y_bin)).abs();
        }
    }

    ece
}

pub fn make_trade_record(
    date: NaiveDate,
    city: &str,
    market_type: &str,
    pnl: f64,
    our_estimate: f64,
    actual_outcome: f64,
) -> TradeRecord {
    TradeRecord {
        date,
        city: city.to_string(),
        market_id: format!("{city}_{market_type}_{}", date.format("%Y%m%d")),
        market_title: format!("Synthetic {market_type} market"),
        market_type: market_type.to_string(),
        threshold: 0.0,
        market_price: 0.5,
        our_estimate,
        edge: our_estimate - 0.5,
        side: if our_estimate >= 0.5 {
            "BUY".to_string()
        } else {
            "SELL".to_string()
        },
        position_size: 100.0,
        actual_outcome,
        pnl,
        capital_after: 100_000.0 + pnl,
    }
}
