//! Credentialed Kalshi TRADE client — balance, positions, resting orders, limit orders, and the
//! order lifecycle the pilot reconciles against (get order, fills, cancel).
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
//! A limit order rests until it fills, is cancelled, or its `expiration_ts` (unix seconds)
//! passes; its status is `pending`/`resting` while live and `executed`/`canceled` once terminal
//! (expiry surfaces as `canceled`), and `count − remaining_count` is what filled. Field names
//! follow Kalshi's official SDK (kalshi-python 2.1.4: CreateOrderRequest, Order, Fill) — the
//! parsers below also accept the older `yes_price`/`no_price`-per-fill and `*_dollars` shapes,
//! because a fill row the pilot cannot read is a real position it cannot account for.

use reqwest::Method;
use serde_json::{json, Value};

use super::kalshi_history::KalshiAuth;
use super::polymarket_history::value_as_f64;
use crate::config;

const API_PREFIX: &str = "/trade-api/v2";

#[derive(Debug, thiserror::Error)]
pub enum KalshiTradeError {
    #[error("request failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error(
        "no Kalshi trade credentials (set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH/_PEM)"
    )]
    NoAuth,
    #[error("unexpected response from {0}: {1}")]
    BadResponse(String, String),
}

/// One resting, placed, fetched or cancelled order, as much of it as the pilot needs. The count
/// fields are `None` when a response omits them; `filled()` then has no opinion.
#[derive(Debug, Clone, PartialEq)]
pub struct KalshiOrder {
    pub order_id: String,
    pub ticker: String,
    /// Kalshi's enum: `pending` / `resting` while live, `executed` / `canceled` once terminal.
    pub status: String,
    /// Contracts originally placed.
    pub count: Option<i64>,
    /// Contracts still unfilled and not cancelled.
    pub remaining_count: Option<i64>,
    /// The order's NO limit price in cents.
    pub no_price_cents: Option<i64>,
}

impl KalshiOrder {
    /// Whether Kalshi will never touch this order again: fully filled, cancelled, or expired
    /// (expiry surfaces as `canceled`).
    pub fn is_terminal(&self) -> bool {
        matches!(self.status.as_str(), "executed" | "canceled" | "cancelled")
    }

    /// Contracts filled so far — placed minus remaining — when the response carries both.
    pub fn filled(&self) -> Option<i64> {
        Some((self.count? - self.remaining_count?).max(0))
    }
}

/// One fill (a partial or whole execution) of an order, normalised to the NO side.
#[derive(Debug, Clone, PartialEq)]
pub struct KalshiFill {
    pub fill_id: String,
    pub order_id: String,
    pub ticker: String,
    pub count: i64,
    /// Price paid per NO contract, in cents.
    pub no_price_cents: i64,
    pub is_taker: bool,
    pub created_time: String,
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
        let v = self.get("balance", "/portfolio/balance").await?;
        v.get("balance")
            .and_then(value_as_f64)
            .map(|cents| cents / 100.0)
            .ok_or_else(|| KalshiTradeError::BadResponse("balance".into(), v.to_string()))
    }

    /// Nonzero per-market positions.
    pub async fn positions(&self) -> Result<Vec<KalshiPosition>, KalshiTradeError> {
        let v = self.get("positions", "/portfolio/positions").await?;
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
        let v = self
            .get("resting orders", "/portfolio/orders?status=resting")
            .await?;
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
    /// `expiration_ts` (unix seconds) bounds how long an unfilled remainder may rest; `None` is
    /// good-till-cancelled, which for a lead-1 strategy means "may fill on the day it must not".
    pub async fn buy_no_limit(
        &self,
        ticker: &str,
        count: i64,
        no_price_cents: i64,
        client_order_id: &str,
        expiration_ts: Option<i64>,
    ) -> Result<KalshiOrder, KalshiTradeError> {
        let mut body = json!({
            "action": "buy",
            "side": "no",
            "type": "limit",
            "ticker": ticker,
            "count": count,
            "no_price": no_price_cents,
            "client_order_id": client_order_id,
        });
        if let Some(ts) = expiration_ts {
            body["expiration_ts"] = json!(ts);
        }
        let v = self
            .send(
                "create order",
                self.signed(Method::POST, "/portfolio/orders").json(&body),
            )
            .await?;
        v.get("order")
            .and_then(parse_order)
            .ok_or_else(|| KalshiTradeError::BadResponse("create order".into(), v.to_string()))
    }

    /// One order by id — status and counts, for reconciling a placed order against what filled.
    pub async fn order(&self, order_id: &str) -> Result<KalshiOrder, KalshiTradeError> {
        let v = self
            .get("get order", &format!("/portfolio/orders/{order_id}"))
            .await?;
        v.get("order")
            .and_then(parse_order)
            .ok_or_else(|| KalshiTradeError::BadResponse("get order".into(), v.to_string()))
    }

    /// Cancel whatever of an order is still resting. Returns the order as Kalshi now sees it.
    pub async fn cancel_order(&self, order_id: &str) -> Result<KalshiOrder, KalshiTradeError> {
        let v = self
            .send(
                "cancel order",
                self.signed(Method::DELETE, &format!("/portfolio/orders/{order_id}")),
            )
            .await?;
        v.get("order")
            .and_then(parse_order)
            .ok_or_else(|| KalshiTradeError::BadResponse("cancel order".into(), v.to_string()))
    }

    /// Every fill of one order (paginated on `cursor`; an order has a handful at most). Fill
    /// PRICE is what the pilot exists to measure — a limit that crosses the book fills at the
    /// resting side's price, so it can be better than the limit, never worse.
    pub async fn fills(&self, order_id: &str) -> Result<Vec<KalshiFill>, KalshiTradeError> {
        let mut out = Vec::new();
        let mut cursor: Option<String> = None;
        loop {
            let mut q = format!("/portfolio/fills?order_id={order_id}&limit=100");
            if let Some(c) = &cursor {
                q.push_str("&cursor=");
                q.push_str(c);
            }
            let v = self.get("get fills", &q).await?;
            let rows = v
                .get("fills")
                .and_then(|x| x.as_array())
                .cloned()
                .unwrap_or_default();
            out.extend(rows.iter().filter_map(parse_fill));
            match v
                .get("cursor")
                .and_then(|c| c.as_str())
                .filter(|c| !c.is_empty())
            {
                Some(c) if !rows.is_empty() => cursor = Some(c.to_string()),
                _ => break,
            }
        }
        Ok(out)
    }

    /// Signed GET; `path_and_query`'s query part is excluded from the signed message per Kalshi's
    /// scheme (only `timestamp+method+path` is signed).
    async fn get(&self, what: &str, path_and_query: &str) -> Result<Value, KalshiTradeError> {
        self.send(what, self.signed(Method::GET, path_and_query))
            .await
    }

    /// A request with Kalshi's three auth headers. The signed path carries the API prefix and
    /// drops the query string; the URL carries both.
    fn signed(&self, method: Method, path_and_query: &str) -> reqwest::RequestBuilder {
        let path_only = path_and_query.split('?').next().unwrap_or(path_and_query);
        let signed_path = format!("{API_PREFIX}{path_only}");
        let mut req = self.client.request(
            method.clone(),
            format!("{}{API_PREFIX}{}", self.base_url, path_and_query),
        );
        for (k, v) in self.auth.headers(method.as_str(), &signed_path) {
            req = req.header(k, v);
        }
        req
    }

    /// Send and decode, folding a non-2xx status into `BadResponse` WITH the body, so a rejected
    /// order says why (insufficient balance, market closed, bad price) rather than "no order key".
    async fn send(
        &self,
        what: &str,
        req: reqwest::RequestBuilder,
    ) -> Result<Value, KalshiTradeError> {
        let resp = req.send().await?;
        let status = resp.status();
        let body = resp.text().await?;
        if !status.is_success() {
            return Err(KalshiTradeError::BadResponse(
                format!("{what} (HTTP {status})"),
                body,
            ));
        }
        Ok(serde_json::from_str(&body).unwrap_or(Value::Null))
    }
}

/// Cents from a price the API may state in cents (`34`) or dollars (`0.34`, `"0.3400"`): a
/// tradeable price is 1..=99 cents, so anything under 1 can only be dollars.
fn to_cents(x: f64) -> i64 {
    if x < 1.0 {
        (x * 100.0).round() as i64
    } else {
        x.round() as i64
    }
}

fn parse_order(o: &Value) -> Option<KalshiOrder> {
    let int = |keys: &[&str]| {
        keys.iter()
            .find_map(|k| o.get(*k).and_then(value_as_f64))
            .map(|x| x as i64)
    };
    Some(KalshiOrder {
        order_id: o.get("order_id")?.as_str()?.to_string(),
        ticker: o.get("ticker")?.as_str()?.to_string(),
        status: o
            .get("status")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string(),
        count: int(&["count", "initial_count", "place_count"]),
        remaining_count: int(&["remaining_count"]),
        no_price_cents: ["no_price", "no_price_dollars"]
            .iter()
            .find_map(|k| o.get(*k).and_then(value_as_f64))
            .map(to_cents),
    })
}

fn parse_fill(f: &Value) -> Option<KalshiFill> {
    let side = f.get("side").and_then(|s| s.as_str()).unwrap_or("no");
    let str_of = |keys: &[&str]| {
        keys.iter()
            .find_map(|k| f.get(*k).and_then(|s| s.as_str()))
            .unwrap_or("")
            .to_string()
    };
    Some(KalshiFill {
        fill_id: str_of(&["fill_id", "trade_id"]),
        order_id: f.get("order_id")?.as_str()?.to_string(),
        ticker: str_of(&["ticker"]),
        count: f.get("count").and_then(value_as_f64)? as i64,
        no_price_cents: fill_no_price_cents(f, side)?,
        is_taker: f.get("is_taker").and_then(|b| b.as_bool()).unwrap_or(false),
        created_time: str_of(&["created_time"]),
    })
}

/// The NO price of a fill in cents, whichever shape the API used: an explicit `no_price` /
/// `no_price_dollars`, else the single `price` of the fill's own side (the official SDK's `Fill`
/// model — a YES-side price is `100 − price` on the NO side), else `yes_price` likewise.
fn fill_no_price_cents(f: &Value, side: &str) -> Option<i64> {
    if let Some(c) = ["no_price", "no_price_dollars"]
        .iter()
        .find_map(|k| f.get(*k).and_then(value_as_f64))
    {
        return Some(to_cents(c));
    }
    if let Some(p) = f.get("price").and_then(value_as_f64) {
        let c = to_cents(p);
        return Some(if side == "yes" { 100 - c } else { c });
    }
    ["yes_price", "yes_price_dollars"]
        .iter()
        .find_map(|k| f.get(*k).and_then(value_as_f64))
        .map(|y| 100 - to_cents(y))
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
    fn order_parsing_tolerates_missing_status_and_counts() {
        let o = serde_json::json!({"order_id": "abc", "ticker": "KXHIGHNY-26JUL20-B90.5"});
        let parsed = parse_order(&o).unwrap();
        assert_eq!(parsed.order_id, "abc");
        assert_eq!(parsed.status, "");
        assert_eq!(parsed.filled(), None, "no counts ⇒ no opinion on fills");
        assert!(!parsed.is_terminal());
        assert!(parse_order(&serde_json::json!({"ticker": "X"})).is_none());
    }

    #[test]
    fn order_parsing_reads_counts_and_terminal_status() {
        // The official SDK's Order shape: count + remaining_count, cents prices.
        let o = serde_json::json!({
            "order_id": "o1", "ticker": "KXHIGHNY-26SEP08-B89.5", "status": "executed",
            "count": 25, "remaining_count": 0, "no_price": 60, "yes_price": 40
        });
        let p = parse_order(&o).unwrap();
        assert_eq!(p.filled(), Some(25));
        assert_eq!(p.no_price_cents, Some(60));
        assert!(p.is_terminal());
        // Partially filled then expired: Kalshi reports the remainder as canceled.
        let o = serde_json::json!({
            "order_id": "o2", "ticker": "T", "status": "canceled",
            "initial_count": 25, "remaining_count": 15, "no_price_dollars": "0.6000"
        });
        let p = parse_order(&o).unwrap();
        assert_eq!(p.filled(), Some(10));
        assert_eq!(p.no_price_cents, Some(60), "fixed-point dollars → cents");
        assert!(p.is_terminal());
        // Still resting.
        let o = serde_json::json!({
            "order_id": "o3", "ticker": "T", "status": "resting", "count": 25, "remaining_count": 25
        });
        let p = parse_order(&o).unwrap();
        assert_eq!(p.filled(), Some(0));
        assert!(!p.is_terminal());
    }

    #[test]
    fn fill_parsing_normalises_every_price_shape_to_the_no_side() {
        // Official SDK shape: one `price` on the fill's own side.
        let f = serde_json::json!({
            "fill_id": "f1", "order_id": "o1", "ticker": "T", "side": "no", "action": "buy",
            "count": 10, "price": 58, "is_taker": true, "created_time": "2026-09-07T15:01:00Z"
        });
        let p = parse_fill(&f).unwrap();
        assert_eq!((p.count, p.no_price_cents, p.is_taker), (10, 58, true));
        assert_eq!(p.fill_id, "f1");
        // The same fill seen from the YES side is the complement.
        let f = serde_json::json!({"order_id": "o1", "side": "yes", "count": 10, "price": 42});
        assert_eq!(parse_fill(&f).unwrap().no_price_cents, 58);
        // Older shape: explicit per-side cents; `no_price` wins over `price`.
        let f = serde_json::json!({
            "trade_id": "t9", "order_id": "o1", "side": "no", "count": 3,
            "yes_price": 41, "no_price": 59, "price": 1
        });
        let p = parse_fill(&f).unwrap();
        assert_eq!((p.no_price_cents, p.fill_id.as_str()), (59, "t9"));
        // Dollar-denominated fixed point.
        let f = serde_json::json!({"order_id": "o1", "side": "no", "count": 3, "no_price_dollars": "0.5900"});
        assert_eq!(parse_fill(&f).unwrap().no_price_cents, 59);
        let f = serde_json::json!({"order_id": "o1", "side": "yes", "count": 3, "yes_price_dollars": 0.41});
        assert_eq!(parse_fill(&f).unwrap().no_price_cents, 59);
        // No price at all, or no order id ⇒ unusable, never a phantom fill.
        assert!(parse_fill(&serde_json::json!({"order_id": "o1", "count": 3})).is_none());
        assert!(parse_fill(&serde_json::json!({"count": 3, "price": 50})).is_none());
    }

    #[test]
    fn cents_heuristic_only_treats_sub_unit_values_as_dollars() {
        assert_eq!(to_cents(34.0), 34);
        assert_eq!(to_cents(0.34), 34);
        assert_eq!(to_cents(1.0), 1, "1 is a legal cent price, not a dollar");
        assert_eq!(to_cents(0.995), 100, "rounds, never truncates");
    }
}
