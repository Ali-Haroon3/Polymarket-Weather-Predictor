use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::initial_capital;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderConfirmation {
    pub order_id: String,
    pub status: String,
    pub paper_trade: bool,
    pub market_id: String,
    pub outcome: String,
    pub side: String,
    pub price: f64,
    pub size: f64,
    pub order_type: String,
}

#[derive(Clone)]
pub struct PolymarketClient {
    pub api_key: String,
    pub api_secret: String,
    pub private_key: String,
    pub base_url: String,
    pub paper_trading: bool,
    pub authenticated: bool,
    client: reqwest::blocking::Client,
}

impl PolymarketClient {
    pub fn new(
        api_key: Option<String>,
        api_secret: Option<String>,
        private_key: Option<String>,
        base_url: Option<String>,
        paper_trading: bool,
    ) -> Self {
        let mut c = Self {
            api_key: api_key.unwrap_or_default(),
            api_secret: api_secret.unwrap_or_default(),
            private_key: private_key.unwrap_or_default(),
            base_url: base_url
                .unwrap_or_else(|| "https://clob.polymarket.com".to_string())
                .trim_end_matches('/')
                .to_string(),
            paper_trading,
            authenticated: false,
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .unwrap_or_else(|_| reqwest::blocking::Client::new()),
        };

        if !c.api_key.is_empty() && !c.api_secret.is_empty() {
            c.authenticate();
        }

        c
    }

    pub fn authenticate(&mut self) -> bool {
        let headers = [
            ("POLY-API-KEY", self.api_key.clone()),
            ("POLY-API-SECRET", self.api_secret.clone()),
        ];

        let resp = self
            .client
            .get(format!("{}/user", self.base_url))
            .headers(headers_to_map(&headers))
            .send();

        self.authenticated = resp.map(|r| r.status().is_success()).unwrap_or(false);
        self.authenticated
    }

    pub fn get_markets(&self, filter_name: &str, limit: usize) -> Vec<Value> {
        let query = [("filter", filter_name.to_string()), ("limit", limit.to_string())];
        let Ok(resp) = self
            .client
            .get(format!("{}/markets", self.base_url))
            .query(&query)
            .send()
        else {
            return Vec::new();
        };

        if let Ok(val) = resp.json::<Value>() {
            if let Some(arr) = val.as_array() {
                return arr.clone();
            }
            if let Some(arr) = val.get("data").and_then(|d| d.as_array()) {
                return arr.clone();
            }
        }

        Vec::new()
    }

    pub fn get_market_by_id(&self, market_id: &str) -> Option<Value> {
        self.client
            .get(format!("{}/markets/{}", self.base_url, market_id))
            .send()
            .ok()?
            .json::<Value>()
            .ok()
    }

    pub fn get_orderbook(&self, market_id: &str) -> Option<Value> {
        self.client
            .get(format!("{}/markets/{}/orderbook", self.base_url, market_id))
            .send()
            .ok()?
            .json::<Value>()
            .ok()
    }

    pub fn get_mid_price(&self, market_id: &str) -> Option<f64> {
        let ob = self.get_orderbook(market_id)?;
        let bids = ob.get("bids")?.as_array()?.clone();
        let asks = ob.get("asks")?.as_array()?.clone();

        let bid = first_price(&bids)?;
        let ask = first_price(&asks)?;

        Some((bid + ask) / 2.0)
    }

    pub fn place_order(
        &self,
        market_id: &str,
        outcome: &str,
        side: &str,
        price: f64,
        size: f64,
        order_type: &str,
    ) -> Option<OrderConfirmation> {
        if !self.authenticated && !self.paper_trading {
            return None;
        }

        if self.paper_trading {
            return Some(OrderConfirmation {
                order_id: format!("paper_{}", chrono::Utc::now().timestamp_millis()),
                status: "accepted".to_string(),
                paper_trade: true,
                market_id: market_id.to_string(),
                outcome: outcome.to_string(),
                side: side.to_string(),
                price,
                size,
                order_type: order_type.to_string(),
            });
        }

        let payload = serde_json::json!({
            "marketId": market_id,
            "outcome": outcome,
            "side": side,
            "price": price,
            "size": size,
            "orderType": order_type,
            "timestamp": chrono::Utc::now().timestamp_millis(),
        });

        let headers = [
            ("POLY-API-KEY", self.api_key.clone()),
            ("POLY-API-SECRET", self.api_secret.clone()),
        ];

        let resp = self
            .client
            .post(format!("{}/orders", self.base_url))
            .headers(headers_to_map(&headers))
            .json(&payload)
            .send()
            .ok()?;

        let val = resp.json::<Value>().ok()?;
        Some(OrderConfirmation {
            order_id: val
                .get("orderId")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
            status: val
                .get("status")
                .and_then(|v| v.as_str())
                .unwrap_or("accepted")
                .to_string(),
            paper_trade: false,
            market_id: market_id.to_string(),
            outcome: outcome.to_string(),
            side: side.to_string(),
            price,
            size,
            order_type: order_type.to_string(),
        })
    }

    pub fn cancel_order(&self, order_id: &str) -> bool {
        if self.paper_trading {
            return true;
        }

        let headers = [
            ("POLY-API-KEY", self.api_key.clone()),
            ("POLY-API-SECRET", self.api_secret.clone()),
        ];

        self.client
            .delete(format!("{}/orders/{}", self.base_url, order_id))
            .headers(headers_to_map(&headers))
            .send()
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    pub fn get_positions(&self) -> HashMap<String, f64> {
        if !self.authenticated && !self.paper_trading {
            return HashMap::new();
        }
        if self.paper_trading {
            return HashMap::new();
        }

        let headers = [
            ("POLY-API-KEY", self.api_key.clone()),
            ("POLY-API-SECRET", self.api_secret.clone()),
        ];

        let Ok(resp) = self
            .client
            .get(format!("{}/user/positions", self.base_url))
            .headers(headers_to_map(&headers))
            .send()
        else {
            return HashMap::new();
        };

        resp.json::<HashMap<String, f64>>().unwrap_or_default()
    }

    pub fn get_account_balance(&self) -> Option<f64> {
        if !self.authenticated && !self.paper_trading {
            return None;
        }

        if self.paper_trading {
            return Some(initial_capital());
        }

        let headers = [
            ("POLY-API-KEY", self.api_key.clone()),
            ("POLY-API-SECRET", self.api_secret.clone()),
        ];

        let resp = self
            .client
            .get(format!("{}/user/balance", self.base_url))
            .headers(headers_to_map(&headers))
            .send()
            .ok()?;
        let json = resp.json::<Value>().ok()?;
        json.get("balance").and_then(|b| b.as_f64())
    }

    pub fn get_portfolio_value(&self) -> Option<f64> {
        let balance = self.get_account_balance()?;
        let positions = self.get_positions();
        let position_value = positions.values().map(|v| v.abs()).sum::<f64>() * 0.5;
        Some(balance + position_value)
    }

    pub fn market_order(&self, market_id: &str, outcome: &str, side: &str, size: f64) -> Option<OrderConfirmation> {
        let mid = self.get_mid_price(market_id)?;
        let price = if side == "BUY" { mid + 0.005 } else { mid - 0.005 };
        self.place_order(market_id, outcome, side, price, size, "market")
    }

    pub fn close_position(&self, market_id: &str, current_position: f64) -> Option<OrderConfirmation> {
        let (outcome, side) = if current_position > 0.0 {
            ("YES", "SELL")
        } else {
            ("NO", "BUY")
        };
        self.market_order(market_id, outcome, side, current_position.abs())
    }

    pub fn health_check(&self) -> bool {
        self.client
            .get(format!("{}/markets", self.base_url))
            .query(&[("limit", "1")])
            .send()
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }
}

fn headers_to_map(headers: &[(&str, String)]) -> reqwest::header::HeaderMap {
    let mut map = reqwest::header::HeaderMap::new();
    for (k, v) in headers {
        if let (Ok(name), Ok(val)) = (
            reqwest::header::HeaderName::from_bytes(k.as_bytes()),
            reqwest::header::HeaderValue::from_str(v),
        ) {
            map.insert(name, val);
        }
    }
    map
}

fn first_price(levels: &[Value]) -> Option<f64> {
    let first = levels.first()?;
    if let Some(arr) = first.as_array() {
        return arr.first()?.as_f64();
    }
    first.as_f64()
}
