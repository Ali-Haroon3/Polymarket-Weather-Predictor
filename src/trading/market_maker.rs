use std::collections::HashMap;

use chrono::Utc;

use crate::types::{MarketQuote, SpreadQuote};
use crate::utils::clip;

#[derive(Debug, Clone)]
pub struct TradeExecution {
    pub market_id: String,
    pub side: String,
    pub size: f64,
    pub price: f64,
    pub notional: f64,
    pub timestamp: chrono::DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct PortfolioMetrics {
    pub total_position: f64,
    pub net_exposure: f64,
    pub number_of_markets: usize,
    pub total_pnl: f64,
    pub remaining_capital: f64,
}

#[derive(Debug, Clone)]
pub struct MarketMaker {
    pub capital: f64,
    pub inventory: HashMap<String, f64>,
    pub pnl: f64,
}

impl MarketMaker {
    pub fn new(capital: f64) -> Self {
        Self {
            capital,
            inventory: HashMap::new(),
            pnl: 0.0,
        }
    }

    pub fn calculate_fair_value_spread(
        &self,
        market_price: f64,
        our_estimate: f64,
        volatility: f64,
        inventory: f64,
    ) -> SpreadQuote {
        let base_spread = volatility * 0.02;
        let inventory_spread = inventory.abs() * 0.01;
        let half_spread = base_spread + inventory_spread;

        let (bid, ask) = if our_estimate > market_price {
            (market_price - half_spread * 0.5, market_price + half_spread * 1.5)
        } else if our_estimate < market_price {
            (market_price - half_spread * 1.5, market_price + half_spread * 0.5)
        } else {
            (market_price - half_spread, market_price + half_spread)
        };

        let bid = clip(bid, 0.0, 1.0);
        let ask = clip(ask, 0.0, 1.0);

        SpreadQuote {
            bid,
            ask,
            spread: ask - bid,
            mid: (bid + ask) / 2.0,
            market_id: String::new(),
            current_inventory: inventory,
        }
    }

    pub fn optimal_bid_ask_spreads(
        &self,
        market_prices: &HashMap<String, MarketQuote>,
        probability_estimates: &HashMap<String, f64>,
        volatility: f64,
    ) -> HashMap<String, SpreadQuote> {
        let mut out = HashMap::new();

        for (market_id, quote) in market_prices {
            let estimate = *probability_estimates.get(market_id).unwrap_or(&0.5);
            let inventory = *self.inventory.get(market_id).unwrap_or(&0.0);
            let mid = quote.mid.unwrap_or((quote.bid + quote.ask) / 2.0);

            let mut spread = self.calculate_fair_value_spread(mid, estimate, volatility, inventory);
            spread.market_id = market_id.clone();
            spread.current_inventory = inventory;
            out.insert(market_id.clone(), spread);
        }

        out
    }

    pub fn calculate_position_sizes(
        &self,
        probability_estimates: &HashMap<String, f64>,
        market_prices: &HashMap<String, f64>,
        max_position_pct: f64,
    ) -> HashMap<String, f64> {
        let mut sizes = HashMap::new();

        for (market_id, estimate) in probability_estimates {
            let market_price = *market_prices.get(market_id).unwrap_or(&0.5);
            let p_win = if *estimate > market_price {
                *estimate
            } else {
                1.0 - *estimate
            };
            let p_loss = 1.0 - p_win;

            let b = if market_price > 0.0 {
                1.0 / market_price - 1.0
            } else {
                1.0
            };
            let kelly = if b > 0.0 {
                ((b * p_win - p_loss) / b).max(0.0) * 0.25
            } else {
                0.0
            };

            sizes.insert(market_id.clone(), kelly.min(max_position_pct));
        }

        sizes
    }

    pub fn execute_market_orders(
        &mut self,
        market_id: &str,
        side: &str,
        size: f64,
        price: f64,
    ) -> TradeExecution {
        let mut adjusted_size = size;
        let notional = size * price * 100.0;

        if side == "BUY" {
            if notional > self.capital && price > 0.0 {
                adjusted_size = self.capital / (price * 100.0);
            }
            *self.inventory.entry(market_id.to_string()).or_insert(0.0) += adjusted_size;
        } else if side == "SELL" {
            *self.inventory.entry(market_id.to_string()).or_insert(0.0) -= adjusted_size;
        }

        TradeExecution {
            market_id: market_id.to_string(),
            side: side.to_string(),
            size: adjusted_size,
            price,
            notional: adjusted_size * price * 100.0,
            timestamp: Utc::now(),
        }
    }

    pub fn hedge_position(
        &mut self,
        market_id: &str,
        inventory: f64,
        current_price: f64,
        hedge_ratio: f64,
    ) -> TradeExecution {
        if inventory > 0.0 {
            self.execute_market_orders(market_id, "SELL", inventory * hedge_ratio, current_price)
        } else {
            self.execute_market_orders(
                market_id,
                "BUY",
                inventory.abs() * hedge_ratio,
                current_price,
            )
        }
    }

    pub fn calculate_pnl_by_market(&self, current_prices: &HashMap<String, f64>) -> HashMap<String, f64> {
        let mut pnl = HashMap::new();

        for (market_id, inventory) in &self.inventory {
            if *inventory != 0.0 {
                let current = *current_prices.get(market_id).unwrap_or(&0.5);
                pnl.insert(market_id.clone(), inventory * (current - 0.5) * 100.0);
            }
        }

        pnl
    }

    pub fn get_portfolio_metrics(&self) -> PortfolioMetrics {
        let total_position = self.inventory.values().map(|v| v.abs()).sum::<f64>();
        let net_exposure = self.inventory.values().sum::<f64>();
        let number_of_markets = self.inventory.values().filter(|v| **v != 0.0).count();

        PortfolioMetrics {
            total_position,
            net_exposure,
            number_of_markets,
            total_pnl: self.pnl,
            remaining_capital: self.capital - self.pnl.abs(),
        }
    }
}
