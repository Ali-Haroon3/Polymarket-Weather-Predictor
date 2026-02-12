use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use crate::config::{monte_carlo_params, MonteCarloParams};
use crate::types::{PricePoint, ProbabilityPoint};
use crate::utils::{clip, max_drawdown, mean, median, percentile, std_dev};

#[derive(Debug, Clone)]
pub struct PnlStats {
    pub mean_pnl: f64,
    pub std_pnl: f64,
    pub median_pnl: f64,
    pub min_pnl: f64,
    pub max_pnl: f64,
    pub percentile_5: f64,
    pub percentile_95: f64,
    pub prob_profit: f64,
    pub prob_loss: f64,
    pub sharpe_ratio: f64,
}

#[derive(Debug, Clone)]
pub struct PositionOptimization {
    pub optimal_position_size: f64,
    pub expected_sharpe_ratio: f64,
    pub expected_return: f64,
    pub expected_volatility: f64,
}

#[derive(Debug, Clone)]
pub struct StrategyBacktestResult {
    pub final_value: f64,
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub total_trades: usize,
    pub portfolio_value: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MonteCarloSimulator {
    pub params: MonteCarloParams,
}

impl Default for MonteCarloSimulator {
    fn default() -> Self {
        Self::new(None)
    }
}

impl MonteCarloSimulator {
    pub fn new(params: Option<MonteCarloParams>) -> Self {
        Self {
            params: params.unwrap_or_else(monte_carlo_params),
        }
    }

    pub fn simulate_price_paths(
        &self,
        initial_price: f64,
        probability_estimate: f64,
        volatility: f64,
        days_to_expiry: usize,
        n_simulations: Option<usize>,
    ) -> Vec<Vec<f64>> {
        let n_sims = n_simulations.unwrap_or(self.params.n_simulations);
        if days_to_expiry == 0 {
            return vec![vec![]; n_sims];
        }

        let dt: f64 = 1.0 / 365.0;
        let drift = (probability_estimate - initial_price) / days_to_expiry as f64;

        let mut paths = vec![vec![0.0; days_to_expiry]; n_sims];
        for path in &mut paths {
            path[0] = initial_price;
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(self.params.random_seed);
        let norm = Normal::new(0.0, dt.sqrt()).unwrap();

        for t in 1..days_to_expiry {
            for path in &mut paths {
                let dw = norm.sample(&mut rng);
                let prev = path[t - 1];
                let next = prev * ((drift - 0.5 * volatility.powi(2)) * dt + volatility * dw).exp();
                path[t] = clip(next, 0.0, 1.0);
            }
        }

        paths
    }

    pub fn calculate_pnl_distribution(
        &self,
        entry_price: f64,
        position_size: f64,
        price_paths: &[Vec<f64>],
    ) -> PnlStats {
        let final_prices: Vec<f64> = price_paths
            .iter()
            .filter_map(|path| path.last().copied())
            .collect();

        let pnl: Vec<f64> = final_prices
            .iter()
            .map(|fp| (fp - entry_price) * position_size * 100.0)
            .collect();

        let std = std_dev(&pnl, 0);
        PnlStats {
            mean_pnl: mean(&pnl),
            std_pnl: std,
            median_pnl: median(&pnl),
            min_pnl: pnl.iter().fold(f64::INFINITY, |a, b| a.min(*b)),
            max_pnl: pnl.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b)),
            percentile_5: percentile(&pnl, 5.0),
            percentile_95: percentile(&pnl, 95.0),
            prob_profit: pnl.iter().filter(|v| **v > 0.0).count() as f64 / pnl.len().max(1) as f64,
            prob_loss: pnl.iter().filter(|v| **v < 0.0).count() as f64 / pnl.len().max(1) as f64,
            sharpe_ratio: if std > 0.0 { mean(&pnl) / std } else { 0.0 },
        }
    }

    pub fn calculate_value_at_risk(&self, pnl_distribution: &[f64], confidence_level: f64) -> f64 {
        percentile(pnl_distribution, (1.0 - confidence_level) * 100.0)
    }

    pub fn calculate_conditional_var(&self, pnl_distribution: &[f64], confidence_level: f64) -> f64 {
        let threshold = self.calculate_value_at_risk(pnl_distribution, confidence_level);
        let tail: Vec<f64> = pnl_distribution
            .iter()
            .copied()
            .filter(|p| *p <= threshold)
            .collect();
        mean(&tail)
    }

    pub fn optimize_position_size(
        &self,
        entry_price: f64,
        probability_estimate: f64,
        volatility: f64,
        days_to_expiry: usize,
        max_loss_tolerance: f64,
    ) -> PositionOptimization {
        let paths = self.simulate_price_paths(
            entry_price,
            probability_estimate,
            volatility,
            days_to_expiry,
            None,
        );

        let mut best_sharpe = f64::NEG_INFINITY;
        let mut best_position_size = 0.0;

        for i in 1..=50 {
            let pos_size = 0.01 + (i as f64 - 1.0) * (0.5 - 0.01) / 49.0;
            let pnl: Vec<f64> = paths
                .iter()
                .filter_map(|p| p.last().copied())
                .map(|fp| (fp - entry_price) * pos_size * 100.0)
                .collect();

            let max_loss = percentile(&pnl, 5.0);
            if max_loss < -max_loss_tolerance {
                continue;
            }

            let std = std_dev(&pnl, 0);
            let sharpe = if std > 0.0 { mean(&pnl) / std } else { f64::NEG_INFINITY };
            if sharpe > best_sharpe {
                best_sharpe = sharpe;
                best_position_size = pos_size;
            }
        }

        let pnl: Vec<f64> = paths
            .iter()
            .filter_map(|p| p.last().copied())
            .map(|fp| (fp - entry_price) * best_position_size * 100.0)
            .collect();

        PositionOptimization {
            optimal_position_size: best_position_size,
            expected_sharpe_ratio: best_sharpe,
            expected_return: mean(&pnl),
            expected_volatility: std_dev(&pnl, 0),
        }
    }

    pub fn backtest_strategy(
        &self,
        historical_prices: &[PricePoint],
        probability_forecasts: &[ProbabilityPoint],
        initial_capital: f64,
        max_position_size: f64,
    ) -> StrategyBacktestResult {
        if historical_prices.len() < 2 || probability_forecasts.is_empty() {
            return StrategyBacktestResult {
                final_value: initial_capital,
                total_return: 0.0,
                sharpe_ratio: 0.0,
                max_drawdown: 0.0,
                total_trades: 0,
                portfolio_value: vec![initial_capital],
            };
        }

        let mut capital = initial_capital;
        let mut portfolio_value = vec![capital];
        let mut positions: Vec<(f64, f64)> = Vec::new();
        let mut trades = 0;

        let n = historical_prices.len().min(probability_forecasts.len());
        for idx in 0..(n - 1) {
            let market_price = historical_prices[idx].close;
            let forecast_price = probability_forecasts[idx].probability;

            if forecast_price > market_price + 0.02 {
                let position_size = (max_position_size * capital / (market_price * 100.0))
                    .min(capital / (market_price * 100.0));
                positions.push((market_price, position_size));
                trades += 1;
            }

            if !positions.is_empty() {
                let current_price = historical_prices[idx + 1].close;
                positions.retain(|(entry, size)| {
                    let pnl = (current_price - *entry) * *size * 100.0;
                    let close = pnl > capital * 0.02 || pnl < -capital * 0.01;
                    if close {
                        capital += pnl;
                    }
                    !close
                });
            }

            portfolio_value.push(capital);
        }

        let returns: Vec<f64> = portfolio_value
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        let sharpe = {
            let sd = std_dev(&returns, 0);
            if sd > 0.0 {
                mean(&returns) / sd * 252.0_f64.sqrt()
            } else {
                0.0
            }
        };

        StrategyBacktestResult {
            final_value: *portfolio_value.last().unwrap_or(&initial_capital),
            total_return: (*portfolio_value.last().unwrap_or(&initial_capital) - initial_capital)
                / initial_capital,
            sharpe_ratio: sharpe,
            max_drawdown: max_drawdown(&portfolio_value),
            total_trades: trades,
            portfolio_value,
        }
    }
}
