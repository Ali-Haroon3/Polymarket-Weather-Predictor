use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde_json::Value;

use crate::api::polymarket_client::PolymarketClient;
use crate::config::{initial_capital, min_bid_ask_spread};
use crate::models::BayesianWeatherModel;
use crate::trading::MarketMaker;
use crate::types::{PortfolioPnLPoint, PricePoint, ProbabilityPoint, WeatherRecord};

#[derive(Debug, Clone)]
pub struct IterationStats {
    pub timestamp: DateTime<Utc>,
    pub markets_scanned: usize,
    pub opportunities: usize,
    pub orders_placed: usize,
    pub positions: usize,
}

#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    pub start_time: DateTime<Utc>,
    pub current_time: DateTime<Utc>,
    pub total_trades: usize,
    pub current_positions: usize,
    pub current_pnl_pct: f64,
    pub max_pnl_pct: f64,
    pub min_pnl_pct: f64,
    pub avg_pnl_pct: f64,
}

#[derive(Clone)]
pub struct LiveTrader {
    pub client: PolymarketClient,
    pub market_maker: MarketMaker,
    pub model: BayesianWeatherModel,
    pub initial_capital: f64,
    pub paper_trading: bool,
    pub max_position_pct: f64,
    pub active_positions: HashMap<String, f64>,
    pub trade_history: Vec<Value>,
    pub pnl_history: Vec<PortfolioPnLPoint>,
    pub start_time: DateTime<Utc>,
}

impl LiveTrader {
    pub fn new(
        api_key: Option<String>,
        api_secret: Option<String>,
        private_key: Option<String>,
        initial_capital_value: Option<f64>,
        paper_trading: bool,
        max_position_pct: f64,
    ) -> Self {
        let capital = initial_capital_value.unwrap_or_else(initial_capital);

        Self {
            client: PolymarketClient::new(api_key, api_secret, private_key, None, paper_trading),
            market_maker: MarketMaker::new(capital),
            model: BayesianWeatherModel::default(),
            initial_capital: capital,
            paper_trading,
            max_position_pct,
            active_positions: HashMap::new(),
            trade_history: Vec::new(),
            pnl_history: Vec::new(),
            start_time: Utc::now(),
        }
    }

    pub fn initialize(&mut self, historical_data: &[WeatherRecord]) -> bool {
        if self.model.train(historical_data).is_err() {
            return false;
        }

        if !self.paper_trading {
            if !self.client.health_check() {
                return false;
            }
            if self.client.get_account_balance().is_none() {
                return false;
            }
        }

        true
    }

    pub fn scan_markets(&self) -> Vec<Value> {
        self.client.get_markets("weather", 50)
    }

    pub fn analyze_market(&self, market_id: &str, market_data: &Value) -> Option<Value> {
        let mid_price = self.client.get_mid_price(market_id)?;
        let forecast = self.model.forecast_event_probabilities(None, 5_000).ok()?;

        let title = market_data
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_lowercase();

        let our_estimate = if title.contains("temperature") && title.contains("90") {
            *forecast.get("temp_above_90f").unwrap_or(&0.5)
        } else if title.contains("temperature") && title.contains("95") {
            *forecast.get("temp_above_95f").unwrap_or(&0.5)
        } else if title.contains("precipitation") || title.contains("rain") {
            *forecast.get("precipitation").unwrap_or(&0.5)
        } else {
            0.5
        };

        let edge = (our_estimate - mid_price).abs();
        Some(serde_json::json!({
            "market_id": market_id,
            "market_title": title,
            "current_price": mid_price,
            "our_estimate": our_estimate,
            "edge": edge,
            "should_trade": edge > 0.05_f64.max(min_bid_ask_spread()),
        }))
    }

    pub fn calculate_order(&self, analysis: &Value, available_capital: f64) -> Option<Value> {
        let should_trade = analysis.get("should_trade")?.as_bool()?;
        if !should_trade {
            return None;
        }

        let mid_price = analysis.get("current_price")?.as_f64()?;
        let our_estimate = analysis.get("our_estimate")?.as_f64()?;
        let market_id = analysis.get("market_id")?.as_str()?;

        let (side, outcome, prob) = if our_estimate > mid_price {
            ("BUY", "YES", our_estimate)
        } else {
            ("SELL", "NO", 1.0 - our_estimate)
        };

        let b = if mid_price > 0.0 { 1.0 / mid_price - 1.0 } else { 1.0 };
        let kelly_fraction = if b > 0.0 {
            ((b * prob - (1.0 - prob)) / b).max(0.0) * 0.25
        } else {
            0.0
        };

        let max_position = (available_capital * self.max_position_pct) / (mid_price * 100.0);
        let position_size = kelly_fraction.min(max_position);

        if position_size < 0.01 {
            return None;
        }

        Some(serde_json::json!({
            "market_id": market_id,
            "outcome": outcome,
            "side": side,
            "price": mid_price,
            "size": position_size,
            "edge": analysis.get("edge").and_then(|v| v.as_f64()).unwrap_or(0.0),
        }))
    }

    pub fn execute_order(&mut self, order: &Value) -> bool {
        let market_id = order.get("market_id").and_then(|v| v.as_str()).unwrap_or_default();
        let outcome = order.get("outcome").and_then(|v| v.as_str()).unwrap_or_default();
        let side = order.get("side").and_then(|v| v.as_str()).unwrap_or_default();
        let price = order.get("price").and_then(|v| v.as_f64()).unwrap_or(0.5);
        let size = order.get("size").and_then(|v| v.as_f64()).unwrap_or(0.0);

        let result = self
            .client
            .place_order(market_id, outcome, side, price, size, "limit");

        if let Some(res) = result {
            let position = self
                .active_positions
                .entry(market_id.to_string())
                .or_insert(0.0);
            if side == "BUY" {
                *position += size;
            } else {
                *position -= size;
            }

            self.trade_history.push(serde_json::json!({
                "timestamp": Utc::now(),
                "market_id": market_id,
                "order_id": res.order_id,
                "side": side,
                "outcome": outcome,
                "price": price,
                "size": size,
            }));

            true
        } else {
            false
        }
    }

    pub fn manage_positions(&mut self) {
        let positions = self.active_positions.clone();
        for (market_id, position) in positions {
            if position == 0.0 {
                continue;
            }

            if let Some(mid) = self.client.get_mid_price(&market_id) {
                let entry_price = 0.5;
                if position > 0.0 && mid > entry_price * 1.1 {
                    let _ = self.client.close_position(&market_id, position);
                    self.active_positions.insert(market_id, 0.0);
                }
            }
        }
    }

    pub fn run_iteration(&mut self, markets: Option<Vec<Value>>) -> IterationStats {
        let markets = markets.unwrap_or_else(|| self.scan_markets());

        let mut stats = IterationStats {
            timestamp: Utc::now(),
            markets_scanned: markets.len(),
            opportunities: 0,
            orders_placed: 0,
            positions: self.active_positions.values().filter(|p| **p != 0.0).count(),
        };

        let available_capital = self
            .client
            .get_account_balance()
            .unwrap_or(self.initial_capital);

        for market in &markets {
            let market_id = market.get("id").and_then(|v| v.as_str()).unwrap_or_default();
            if market_id.is_empty() {
                continue;
            }

            let Some(analysis) = self.analyze_market(market_id, market) else {
                continue;
            };

            if analysis
                .get("should_trade")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                stats.opportunities += 1;
            } else {
                continue;
            }

            let Some(order) = self.calculate_order(&analysis, available_capital) else {
                continue;
            };

            if self.execute_order(&order) {
                stats.orders_placed += 1;
            }
        }

        self.manage_positions();
        self.record_pnl();

        stats
    }

    fn record_pnl(&mut self) {
        let portfolio_value = self
            .client
            .get_portfolio_value()
            .unwrap_or(self.initial_capital);
        let pnl = portfolio_value - self.initial_capital;
        let pnl_pct = pnl / self.initial_capital * 100.0;

        self.pnl_history.push(PortfolioPnLPoint {
            timestamp: Utc::now(),
            portfolio_value,
            pnl,
            pnl_pct,
        });
    }

    pub fn run_backtest(
        &mut self,
        historical_prices: &[PricePoint],
        probability_forecasts: &[ProbabilityPoint],
        max_iterations: usize,
    ) -> Value {
        let iterations = max_iterations
            .min(historical_prices.len())
            .min(probability_forecasts.len());

        for _ in 0..iterations {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        serde_json::json!({
            "total_trades": 0,
            "winning_trades": 0,
            "win_rate": 0.0,
            "final_pnl": 0.0,
            "pnl_history": self.pnl_history,
        })
    }

    pub fn get_performance_summary(&self) -> Option<PerformanceSummary> {
        if self.pnl_history.is_empty() {
            return None;
        }

        let pnls = self.pnl_history.iter().map(|p| p.pnl_pct).collect::<Vec<_>>();
        let avg = pnls.iter().sum::<f64>() / pnls.len() as f64;

        Some(PerformanceSummary {
            start_time: self.start_time,
            current_time: Utc::now(),
            total_trades: self.trade_history.len(),
            current_positions: self.active_positions.values().filter(|p| **p != 0.0).count(),
            current_pnl_pct: *pnls.last().unwrap_or(&0.0),
            max_pnl_pct: pnls.iter().copied().reduce(f64::max).unwrap_or(0.0),
            min_pnl_pct: pnls.iter().copied().reduce(f64::min).unwrap_or(0.0),
            avg_pnl_pct: avg,
        })
    }

    pub fn shutdown(&mut self) {
        let positions = self.active_positions.clone();
        for (market_id, position) in positions {
            if position != 0.0 {
                let _ = self.client.close_position(&market_id, position);
            }
        }
    }
}
