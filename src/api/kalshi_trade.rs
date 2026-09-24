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
//! (expiry surfaces as `canceled`). Explicit `fill_count_fp` is authoritative; cancellation
//! clears the remainder and must not be mistaken for execution. Modern fixed-point fields and
//! legacy fields are accepted only when this integer-contract, whole-cent client can represent
//! them exactly. Unsupported or malformed account data fails closed, never silently drops risk.

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
    /// Explicitly executed contracts, independent of cancellation of the remainder.
    pub fill_count: Option<i64>,
    /// The order's NO limit price in cents.
    pub no_price_cents: Option<i64>,
}

impl KalshiOrder {
    /// Whether Kalshi will never touch this order again: fully filled, cancelled, or expired
    /// (expiry surfaces as `canceled`).
    pub fn is_terminal(&self) -> bool {
        matches!(self.status.as_str(), "executed" | "canceled" | "cancelled")
    }

    /// Explicit executed count first. Legacy subtraction is safe only before cancellation.
    pub fn filled(&self) -> Option<i64> {
        if let Some(filled) = self.fill_count {
            return (filled >= 0 && self.count.is_none_or(|count| filled <= count))
                .then_some(filled);
        }
        if !matches!(self.status.as_str(), "pending" | "resting" | "executed") {
            return None;
        }
        let (count, remaining) = (self.count?, self.remaining_count?);
        (count >= 0 && remaining >= 0 && remaining <= count).then_some(count - remaining)
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

/// Which contract a limit BUY takes. The pilot's SELL signal is a NO purchase (max loss = NO
/// price paid); its BUY signal is a YES purchase (max loss = YES price paid). Never a sell order
/// on either side — nothing is ever shorted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderSide {
    Yes,
    No,
}

impl OrderSide {
    pub fn as_str(self) -> &'static str {
        match self {
            OrderSide::Yes => "yes",
            OrderSide::No => "no",
        }
    }
}

/// The `CreateOrderRequest` body for a limit BUY on one side: the price field is the side's own
/// (`yes_price` / `no_price`, cents), per kalshi-python 2.1.4's model.
pub fn limit_order_body(
    side: OrderSide,
    ticker: &str,
    count: i64,
    price_cents: i64,
    client_order_id: &str,
    expiration_ts: Option<i64>,
) -> Value {
    let mut body = json!({
        "action": "buy",
        "side": side.as_str(),
        "type": "limit",
        "ticker": ticker,
        "count": count,
        "client_order_id": client_order_id,
    });
    body[match side {
        OrderSide::Yes => "yes_price",
        OrderSide::No => "no_price",
    }] = json!(price_cents);
    if let Some(ts) = expiration_ts {
        body["expiration_ts"] = json!(ts);
    }
    body
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
        let rows = self
            .get_all(
                "positions",
                "/portfolio/positions",
                "market_positions",
                parse_position,
            )
            .await?;
        Ok(rows.into_iter().filter(|p| p.position != 0).collect())
    }

    /// Resting (open) orders.
    pub async fn resting_orders(&self) -> Result<Vec<KalshiOrder>, KalshiTradeError> {
        self.get_all(
            "resting orders",
            "/portfolio/orders?status=resting",
            "orders",
            parse_order,
        )
        .await
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
        self.buy_limit(
            OrderSide::No,
            ticker,
            count,
            no_price_cents,
            client_order_id,
            expiration_ts,
        )
        .await
    }

    /// Place a limit BUY of `count` YES contracts at `yes_price_cents` (1..=99) — the other half
    /// of the market-shape strategy (2026-09-07), which buys the under-priced favorite cell as
    /// readily as it sells the over-priced tail. Same idempotency and TTL contract as
    /// `buy_no_limit`; max loss is the YES price paid.
    pub async fn buy_yes_limit(
        &self,
        ticker: &str,
        count: i64,
        yes_price_cents: i64,
        client_order_id: &str,
        expiration_ts: Option<i64>,
    ) -> Result<KalshiOrder, KalshiTradeError> {
        self.buy_limit(
            OrderSide::Yes,
            ticker,
            count,
            yes_price_cents,
            client_order_id,
            expiration_ts,
        )
        .await
    }

    async fn buy_limit(
        &self,
        side: OrderSide,
        ticker: &str,
        count: i64,
        price_cents: i64,
        client_order_id: &str,
        expiration_ts: Option<i64>,
    ) -> Result<KalshiOrder, KalshiTradeError> {
        let body = limit_order_body(
            side,
            ticker,
            count,
            price_cents,
            client_order_id,
            expiration_ts,
        );
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
        let fills = self
            .get_all(
                "get fills",
                &format!("/portfolio/fills?order_id={order_id}&limit=100"),
                "fills",
                parse_fill,
            )
            .await?;
        if fills.iter().any(|f| f.order_id != order_id) {
            return Err(KalshiTradeError::BadResponse(
                "get fills".into(),
                "response contains another order's fills".into(),
            ));
        }
        Ok(fills)
    }

    /// Account lists must be complete and parseable before they can be used as risk evidence.
    async fn get_all<T>(
        &self,
        what: &str,
        path: &str,
        field: &str,
        parser: fn(&Value) -> Option<T>,
    ) -> Result<Vec<T>, KalshiTradeError> {
        let mut out = Vec::new();
        let mut cursor: Option<String> = None;
        let mut seen_cursors = std::collections::HashSet::new();
        loop {
            let mut req = self.signed(Method::GET, path);
            if let Some(c) = &cursor {
                req = req.query(&[("cursor", c)]);
            }
            let v = self.send(what, req).await?;
            out.extend(parse_response_rows(&v, field, what, parser)?);
            cursor = response_cursor(&v, what)?;
            let Some(c) = &cursor else { break };
            if !seen_cursors.insert(c.clone()) {
                return Err(KalshiTradeError::BadResponse(
                    what.into(),
                    "pagination repeated a cursor; account data is incomplete".into(),
                ));
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

/// Never drop one malformed row from an otherwise successful account response.
fn parse_response_rows<T>(
    response: &Value,
    field: &str,
    what: &str,
    parser: fn(&Value) -> Option<T>,
) -> Result<Vec<T>, KalshiTradeError> {
    let rows = response
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| {
            KalshiTradeError::BadResponse(what.into(), format!("missing or invalid {field} array"))
        })?;
    rows.iter()
        .enumerate()
        .map(|(i, row)| {
            parser(row).ok_or_else(|| {
                KalshiTradeError::BadResponse(
                    what.into(),
                    format!("unsupported or malformed {field} row {i}"),
                )
            })
        })
        .collect()
}

fn response_cursor(response: &Value, what: &str) -> Result<Option<String>, KalshiTradeError> {
    match response.get("cursor") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(c)) if c.is_empty() => Ok(None),
        Some(Value::String(c)) => Ok(Some(c.clone())),
        _ => Err(KalshiTradeError::BadResponse(
            what.into(),
            "invalid pagination cursor".into(),
        )),
    }
}

/// This client trades whole contracts. Fractional, non-finite or imprecise quantities cannot
/// be truncated: even a fractional position represents money at risk.
fn exact_integer(value: &Value, signed: bool) -> Option<i64> {
    if let Some(text) = value.as_str() {
        let (whole, fraction) = text.split_once('.').unwrap_or((text, ""));
        if !fraction.bytes().all(|c| c == b'0') {
            return None;
        }
        let count = whole.parse::<i64>().ok()?;
        return (signed || count >= 0).then_some(count);
    }
    let x = value_as_f64(value)?;
    if !x.is_finite()
        || x.fract() != 0.0
        || x.abs() > 9_007_199_254_740_991.0
        || (!signed && x < 0.0)
    {
        return None;
    }
    Some(x as i64)
}

/// Outer None means invalid; Some(None) means absent. Reject conflicting modern/legacy data.
fn integer_field(o: &Value, keys: &[&str], signed: bool) -> Option<Option<i64>> {
    let mut result = None;
    for key in keys {
        if let Some(v) = o.get(*key).filter(|v| !v.is_null()) {
            let parsed = exact_integer(v, signed)?;
            if result.is_some_and(|old| old != parsed) {
                return None;
            }
            result = Some(parsed);
        }
    }
    Some(result)
}

/// Whole cents only; subcent values remain unsupported rather than rounded into false cost.
fn to_cents(x: f64) -> Option<i64> {
    let cents = if x < 1.0 { x * 100.0 } else { x };
    exact_cents(cents)
}

fn exact_cents(cents: f64) -> Option<i64> {
    (cents.is_finite() && (1.0..=99.0).contains(&cents) && (cents - cents.round()).abs() < 1e-8)
        .then_some(cents.round() as i64)
}

fn price_field(o: &Value, keys: &[(&str, bool)]) -> Option<Option<i64>> {
    let mut result = None;
    for (key, dollars) in keys {
        if let Some(v) = o.get(*key).filter(|v| !v.is_null()) {
            let x = value_as_f64(v)?;
            // Fixed-point strings retain decimal precision that an f64 might erase. A value
            // such as "0.6000000000000001" must not become an apparently exact 60-cent fill.
            if let Some(text) = v.as_str() {
                let (_, fraction) = text.split_once('.').unwrap_or((text, ""));
                let decimal_places = if *dollars || x < 1.0 { 2 } else { 0 };
                if !fraction.bytes().all(|c| c.is_ascii_digit())
                    || fraction.bytes().skip(decimal_places).any(|c| c != b'0')
                {
                    return None;
                }
            }
            let cents = if *dollars {
                exact_cents(x * 100.0)?
            } else {
                to_cents(x)?
            };
            if result.is_some_and(|old| old != cents) {
                return None;
            }
            result = Some(cents);
        }
    }
    Some(result)
}

fn no_price(o: &Value) -> Option<Option<i64>> {
    let no = price_field(o, &[("no_price_dollars", true), ("no_price", false)])?;
    let yes = price_field(o, &[("yes_price_dollars", true), ("yes_price", false)])?;
    if no.zip(yes).is_some_and(|(n, y)| n + y != 100) {
        return None;
    }
    Some(no.or_else(|| yes.map(|y| 100 - y)))
}

fn nonempty_string<'a>(o: &'a Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .find_map(|key| o.get(*key)?.as_str().filter(|s| !s.is_empty()))
}

fn parse_position(p: &Value) -> Option<KalshiPosition> {
    Some(KalshiPosition {
        ticker: nonempty_string(p, &["ticker"])?.to_string(),
        position: integer_field(p, &["position_fp", "position"], true)??,
    })
}

fn parse_order(o: &Value) -> Option<KalshiOrder> {
    let count = integer_field(
        o,
        &["initial_count_fp", "initial_count", "count", "place_count"],
        false,
    )?;
    let remaining_count = integer_field(o, &["remaining_count_fp", "remaining_count"], false)?;
    let fill_count = integer_field(o, &["fill_count_fp", "fill_count"], false)?;
    if count.zip(remaining_count).is_some_and(|(c, r)| r > c)
        || count.zip(fill_count).is_some_and(|(c, f)| f > c)
        || count
            .zip(remaining_count.zip(fill_count))
            .is_some_and(|(c, (r, f))| r.checked_add(f).is_none_or(|total| total > c))
    {
        return None;
    }
    Some(KalshiOrder {
        order_id: nonempty_string(o, &["order_id"])?.to_string(),
        ticker: nonempty_string(o, &["ticker"])?.to_string(),
        status: o
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        count,
        remaining_count,
        fill_count,
        no_price_cents: no_price(o)?,
    })
}

fn parse_fill(f: &Value) -> Option<KalshiFill> {
    let mut side = None;
    for key in ["outcome_side", "side"] {
        if let Some(value) = f.get(key).filter(|value| !value.is_null()) {
            let parsed = value.as_str()?;
            if !matches!(parsed, "yes" | "no") || side.is_some_and(|old| old != parsed) {
                return None;
            }
            side = Some(parsed);
        }
    }
    let count = integer_field(f, &["count_fp", "count"], false)??;
    if count <= 0 {
        return None;
    }
    Some(KalshiFill {
        fill_id: nonempty_string(f, &["fill_id", "trade_id"])?.to_string(),
        order_id: nonempty_string(f, &["order_id"])?.to_string(),
        ticker: nonempty_string(f, &["ticker", "market_ticker"])?.to_string(),
        count,
        no_price_cents: fill_no_price_cents(f, side)?,
        is_taker: f.get("is_taker").and_then(Value::as_bool).unwrap_or(false),
        created_time: f
            .get("created_time")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
    })
}

/// Explicit side prices take precedence over the legacy own-side `price` field.
fn fill_no_price_cents(f: &Value, side: Option<&str>) -> Option<i64> {
    if let Some(c) = no_price(f)? {
        return Some(c);
    }
    // A legacy `price` is paid on the fill's own side. Without that side its NO equivalent
    // is unknowable; defaulting to NO could turn an 80-cent YES fill into a 20-cent one.
    let side = side?;
    let c = price_field(f, &[("price", false)])??;
    Some(if side == "yes" { 100 - c } else { c })
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
            "initial_count": 25, "remaining_count": 0, "fill_count": 10,
            "no_price_dollars": "0.6000"
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
    fn limit_order_body_names_the_sides_own_price_field() {
        let no = limit_order_body(
            OrderSide::No,
            "KX-T",
            10,
            66,
            "pilot-KX-T-2026-09-08",
            Some(1),
        );
        assert_eq!(no["side"], "no");
        assert_eq!(no["no_price"], 66);
        assert!(no.get("yes_price").is_none());
        assert_eq!(no["expiration_ts"], 1);
        let yes = limit_order_body(OrderSide::Yes, "KX-T", 10, 41, "id", None);
        assert_eq!(yes["side"], "yes");
        assert_eq!(yes["yes_price"], 41);
        assert!(yes.get("no_price").is_none());
        assert!(yes.get("expiration_ts").is_none());
        assert_eq!(yes["action"], "buy");
        assert_eq!(yes["type"], "limit");
        // A YES order echoed back with only its yes price parses to the NO complement.
        let o = serde_json::json!({"order_id":"o","ticker":"KX-T","status":"resting","count":10,"remaining_count":10,"yes_price":41});
        assert_eq!(parse_order(&o).unwrap().no_price_cents, Some(59));
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
        let f = serde_json::json!({"fill_id":"f2", "ticker":"T", "order_id": "o1", "side": "yes", "count": 10, "price": 42});
        assert_eq!(parse_fill(&f).unwrap().no_price_cents, 58);
        // Older shape: explicit per-side cents; `no_price` wins over `price`.
        let f = serde_json::json!({
            "trade_id": "t9", "ticker":"T", "order_id": "o1", "side": "no", "count": 3,
            "yes_price": 41, "no_price": 59, "price": 1
        });
        let p = parse_fill(&f).unwrap();
        assert_eq!((p.no_price_cents, p.fill_id.as_str()), (59, "t9"));
        // Dollar-denominated fixed point.
        let f = serde_json::json!({"fill_id":"f3", "ticker":"T", "order_id": "o1", "side": "no", "count": 3, "no_price_dollars": "0.5900"});
        assert_eq!(parse_fill(&f).unwrap().no_price_cents, 59);
        let f = serde_json::json!({"fill_id":"f4", "ticker":"T", "order_id": "o1", "side": "yes", "count": 3, "yes_price_dollars": 0.41});
        assert_eq!(parse_fill(&f).unwrap().no_price_cents, 59);
        // No price at all, or no order id ⇒ unusable, never a phantom fill.
        assert!(parse_fill(&serde_json::json!({"order_id": "o1", "count": 3})).is_none());
        assert!(parse_fill(&serde_json::json!({"count": 3, "price": 50})).is_none());
    }

    #[test]
    fn whole_cent_prices_are_exact_and_subcents_fail_closed() {
        assert_eq!(to_cents(34.0), Some(34));
        assert_eq!(to_cents(0.34), Some(34));
        assert_eq!(to_cents(1.0), Some(1), "legacy 1 is one cent");
        for invalid in [
            0.995,
            0.605,
            60.5,
            0.0,
            -1.0,
            100.0,
            f64::NAN,
            f64::INFINITY,
        ] {
            assert_eq!(to_cents(invalid), None);
        }
        assert!(
            parse_order(&json!({"order_id":"o", "ticker":"T", "no_price_dollars":"1.0000"}))
                .is_none()
        );
    }

    #[test]
    fn ambiguous_legacy_fill_prices_require_a_consistent_explicit_side() {
        let mut fill = json!({"fill_id":"f", "order_id":"o", "ticker":"T", "count":10, "price":80});
        assert!(
            parse_fill(&fill).is_none(),
            "own-side price without side is ambiguous"
        );
        fill["side"] = json!("yes");
        assert_eq!(parse_fill(&fill).unwrap().no_price_cents, 20);
        fill["outcome_side"] = json!("no");
        assert!(
            parse_fill(&fill).is_none(),
            "conflicting side aliases cannot be normalized"
        );
        fill["no_price_dollars"] = json!("0.2000");
        assert!(
            parse_fill(&fill).is_none(),
            "named prices cannot excuse conflicting sides"
        );
        fill.as_object_mut().unwrap().remove("side");
        fill.as_object_mut().unwrap().remove("outcome_side");
        assert_eq!(
            parse_fill(&fill).unwrap().no_price_cents,
            20,
            "named NO price is unambiguous without side"
        );
        fill["side"] = json!("");
        assert!(parse_fill(&fill).is_none());
    }

    #[test]
    fn current_fixed_point_responses_preserve_execution_and_exposure() {
        let o = json!({
            "order_id":"o", "ticker":"T", "status":"canceled",
            "initial_count_fp":"25.00", "remaining_count_fp":"0.00", "fill_count_fp":"10.00",
            "no_price_dollars":"0.6000", "yes_price_dollars":"0.4000"
        });
        let order = parse_order(&o).unwrap();
        assert_eq!(order.filled(), Some(10), "canceled remainder is not a fill");
        assert_eq!(order.count, Some(25));
        let f = json!({
            "fill_id":"f", "order_id":"o", "ticker":"T", "outcome_side":"no",
            "count_fp":"10.00", "no_price_dollars":"0.6000", "is_taker":true
        });
        let fill = parse_fill(&f).unwrap();
        assert_eq!((fill.count, fill.no_price_cents), (10, 60));
        let p = parse_position(&json!({"ticker":"T", "position_fp":"-10.00"})).unwrap();
        assert_eq!(p.position, -10);
        let mut missing = o;
        missing.as_object_mut().unwrap().remove("fill_count_fp");
        assert_eq!(parse_order(&missing).unwrap().filled(), None);
    }

    #[test]
    fn unsupported_and_conflicting_counts_are_not_truncated_or_ignored() {
        for value in [
            json!("0.50"),
            json!("10.0000000000000001"),
            json!(-1),
            json!("NaN"),
            json!("Infinity"),
            json!("9223372036854775808"),
        ] {
            assert!(
                parse_order(&json!({"order_id":"o", "ticker":"T", "fill_count_fp":value}))
                    .is_none()
            );
        }
        assert!(parse_position(&json!({"ticker":"T", "position_fp":"-0.50"})).is_none());
        assert!(parse_position(&json!({"ticker":"T"})).is_none());
        assert!(parse_order(
            &json!({"order_id":"o", "ticker":"T", "initial_count_fp":"10.00", "count":11})
        )
        .is_none());
        assert!(parse_order(&json!({"order_id":"o", "ticker":"T", "initial_count_fp":"10.00", "fill_count_fp":"11.00"})).is_none());
        assert!(parse_order(&json!({"order_id":"o", "ticker":"T", "initial_count_fp":"9223372036854775807", "remaining_count_fp":"9223372036854775807", "fill_count_fp":"9223372036854775807"})).is_none());
    }

    #[test]
    fn account_pages_fail_as_a_whole_when_a_row_is_unreadable() {
        let good = json!({"fill_id":"f", "order_id":"o", "ticker":"T", "count_fp":"10.00", "no_price_dollars":"0.6000"});
        assert_eq!(
            parse_response_rows(
                &json!({"fills":[good.clone()]}),
                "fills",
                "test",
                parse_fill
            )
            .unwrap()
            .len(),
            1
        );
        for bad in [
            json!({"fill_id":"bad", "order_id":"o", "ticker":"T", "count_fp":"0.50", "no_price_dollars":"0.6000"}),
            json!({"fill_id":"bad", "order_id":"o", "ticker":"T", "count_fp":"10.00", "no_price_dollars":"0.6050"}),
            json!({"fill_id":"bad", "order_id":"o", "ticker":"T", "count_fp":"10.00", "no_price_dollars":"0.6000000000000001"}),
            json!({"order_id":"o", "ticker":"T", "count_fp":"10.00", "no_price_dollars":"0.6000"}),
            json!({"fill_id":"bad", "order_id":"o", "count_fp":"10.00", "no_price_dollars":"0.6000"}),
        ] {
            assert!(parse_response_rows(
                &json!({"fills":[good.clone(), bad]}),
                "fills",
                "test",
                parse_fill
            )
            .is_err());
        }
        assert!(parse_response_rows(&json!({}), "fills", "test", parse_fill).is_err());
        assert!(parse_response_rows(
            &json!({"market_positions":[{"ticker":"T","position_fp":"0.50"}]}),
            "market_positions",
            "test",
            parse_position
        )
        .is_err());
        assert!(parse_response_rows(
            &json!({"orders":[{"order_id":"o", "ticker":"T", "remaining_count_fp":"0.50"}]}),
            "orders",
            "test",
            parse_order
        )
        .is_err());
        assert!(response_cursor(&json!({"cursor":42}), "test").is_err());
    }
}
