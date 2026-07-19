//! Credentialed Kalshi TRADE client — balance, positions, resting orders, limit orders.
//!
//! Talks to `config::kalshi_base_url()`, which defaults to the DEMO/paper host: nothing touches
//! real money until `KALSHI_BASE_URL` is explicitly pointed at production. Requests are signed
//! with the same RSA-PSS machinery the market-data downloader carries (`KalshiAuth`); the client
//! refuses to construct without credentials, so callers can't half-configure their way into
//! unsigned trade calls.
//!
//! Kalshi mechanics the pilot leans on: "selling YES" is buying NO contracts — no shorting, max
//! loss is the price paid. Prices are integer cents (1..=99). Trading fees are
//! `ceil(0.07 × count × P × (1−P))` cents with P the contract price in dollars — see `fee_cents`.

use serde_json::{json, Value};

use super::kalshi_history::KalshiAuth;
use super::polymarket_history::value_as_f64;
use crate::config;

const API_PREFIX: &str = "/trade-api/v2";

#[derive(Debug, thiserror::Error)]
pub enum KalshiTradeError {
    #[error("request failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error("no Kalshi trade credentials (set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH/_PEM)")]
    NoAuth,
    #[error("unexpected response from {0}: {1}")]
    BadResponse(String, String),
}

/// One resting or placed order, as much of it as the pilot needs.
#[derive(Debug, Clone)]
pub struct KalshiOrder {
    pub order_id: String,
    pub ticker: String,
    pub status: String,
}

/// A held position in one market (contracts signed: positive = YES, negative = NO).
#[derive(Debug, Clone)]
pub struct KalshiPosition {
    pub ticker: String,
    pub position: i64,
}

pub struct KalshiTradeClient {
    base_url: String,
    client: reqwest::Client,
    auth: KalshiAuth,
}

impl KalshiTradeClient {
    /// Errors when signing credentials are absent — a trade client without auth is useless.
    pub fn new() -> Result<Self, KalshiTradeError> {
        let auth = KalshiAuth::from_env().ok_or(KalshiTradeError::NoAuth)?;
        Ok(Self {
            base_url: config::kalshi_base_url(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(20))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            auth,
        })
    }

    /// Host this client trades against (print it loudly — demo vs prod is the money boundary).
    pub fn host(&self) -> &str {
        &self.base_url
    }

    /// Whether the configured host is Kalshi's production (real-money) API.
    pub fn is_production(&self) -> bool {
        !self.base_url.contains("demo")
    }

    /// Available balance in dollars.
    pub async fn balance(&self) -> Result<f64, KalshiTradeError> {
        let v = self.get("/portfolio/balance").await?;
        v.get("balance")
            .and_then(value_as_f64)
            .map(|cents| cents / 100.0)
            .ok_or_else(|| KalshiTradeError::BadResponse("balance".into(), v.to_string()))
    }

    /// Nonzero per-market positions.
    pub async fn positions(&self) -> Result<Vec<KalshiPosition>, KalshiTradeError> {
        let v = self.get("/portfolio/positions").await?;
        let rows = v
            .get("market_positions")
            .and_then(|x| x.as_array())
            .cloned()
            .unwrap_or_default();
        Ok(rows
            .iter()
            .filter_map(|p| {
                let ticker = p.get("ticker")?.as_str()?.to_string();
                let position = p.get("position").and_then(value_as_f64)? as i64;
                (position != 0).then_some(KalshiPosition { ticker, position })
            })
            .collect())
    }

    /// Resting (open) orders.
    pub async fn resting_orders(&self) -> Result<Vec<KalshiOrder>, KalshiTradeError> {
        let v = self.get("/portfolio/orders?status=resting").await?;
        let rows = v
            .get("orders")
            .and_then(|x| x.as_array())
            .cloned()
            .unwrap_or_default();
        Ok(rows.iter().filter_map(parse_order).collect())
    }

    /// Place a limit BUY of `count` NO contracts at `no_price_cents` (1..=99). This IS the pilot's
    /// "SELL YES": identical payoff, no shorting involved. `client_order_id` should be stable per
    /// intent (e.g. derived from ticker + date) so a crashed-and-rerun pilot can't double-order.
    pub async fn buy_no_limit(
        &self,
        ticker: &str,
        count: i64,
        no_price_cents: i64,
        client_order_id: &str,
    ) -> Result<KalshiOrder, KalshiTradeError> {
        let body = json!({
            "action": "buy",
            "side": "no",
            "type": "limit",
            "ticker": ticker,
            "count": count,
            "no_price": no_price_cents,
            "client_order_id": client_order_id,
        });
        let path = format!("{API_PREFIX}/portfolio/orders");
        let mut req = self
            .client
            .post(format!("{}{}", self.base_url, path))
            .json(&body);
        for (k, val) in self.auth.headers("POST", &path) {
            req = req.header(k, val);
        }
        let v = req.send().await?.json::<Value>().await?;
        v.get("order")
            .and_then(parse_order)
            .ok_or_else(|| KalshiTradeError::BadResponse("create order".into(), v.to_string()))
    }

    /// Signed GET; `path_and_query`'s query part is excluded from the signed message per Kalshi's
    /// scheme (only `timestamp+method+path` is signed).
    async fn get(&self, path_and_query: &str) -> Result<Value, KalshiTradeError> {
        let path_only = path_and_query.split('?').next().unwrap_or(path_and_query);
        let mut req = self
            .client
            .get(format!("{}{API_PREFIX}{}", self.base_url, path_and_query));
        for (k, v) in self.auth.headers("GET", &format!("{API_PREFIX}{path_only}")) {
            req = req.header(k, v);
        }
        Ok(req.send().await?.json::<Value>().await?)
    }
}

fn parse_order(o: &Value) -> Option<KalshiOrder> {
    Some(KalshiOrder {
        order_id: o.get("order_id")?.as_str()?.to_string(),
        ticker: o.get("ticker")?.as_str()?.to_string(),
        status: o
            .get("status")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string(),
    })
}

/// Kalshi trading fee in CENTS for `count` contracts at `price_cents`:
/// ceil(0.07 × count × P × (1−P) × 100) with P in dollars. Symmetric in YES/NO price.
pub fn fee_cents(count: i64, price_cents: i64) -> i64 {
    let p = price_cents as f64 / 100.0;
    // Epsilon guard: exact-cent products (e.g. 100 × 50¢ ⇒ 175.000…03) must not ceil up a cent.
    (0.07 * count as f64 * p * (1.0 - p) * 100.0 - 1e-9).ceil() as i64
}

/// Same fee as a per-contract fraction of $1 notional — what an edge threshold must clear.
pub fn fee_frac(price: f64) -> f64 {
    0.07 * price * (1.0 - price)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fee_matches_kalshi_schedule() {
        // 100 contracts at 50¢: 0.07 × 100 × 0.25 = $1.75 = 175¢ exactly.
        assert_eq!(fee_cents(100, 50), 175);
        // 1 contract at 50¢: 1.75¢ rounds UP to 2¢ (ceil, per the fee schedule).
        assert_eq!(fee_cents(1, 50), 2);
        // Symmetric and smaller near the tails.
        assert_eq!(fee_cents(100, 10), fee_cents(100, 90));
        assert!(fee_cents(100, 10) < fee_cents(100, 50));
        // Fraction form: 1.75¢ per contract at p = 0.5.
        assert!((fee_frac(0.5) - 0.0175).abs() < 1e-12);
    }

    #[test]
    fn order_parsing_tolerates_missing_status() {
        let o = serde_json::json!({"order_id": "abc", "ticker": "KXHIGHNY-26JUL20-B90.5"});
        let parsed = parse_order(&o).unwrap();
        assert_eq!(parsed.order_id, "abc");
        assert_eq!(parsed.status, "");
        assert!(parse_order(&serde_json::json!({"ticker": "X"})).is_none());
    }
}
