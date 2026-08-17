//! Kalshi weather-market downloader — the venue-B analog of [`super::polymarket_history`].
//!
//! Fetches Kalshi temperature/weather markets and maps them into the same [`WeatherMarketRow`] the
//! capture daemon already consumes, so both venues flow through one pipeline.
//!
//! MARKET DATA (all this module fetches) is anonymous: Kalshi's production market-data endpoints
//! (`config::kalshi_public_base_url`, api.elections.kalshi.com) serve GET /markets without auth, and
//! real production prices are what forward calibration needs — signed DEMO-env fetches would fill
//! captures with paper prices. The RSA-PSS(SHA-256) signing support (`timestamp + method + path`)
//! is kept for the credentialed trade endpoints (`config::kalshi_base_url`, orders/portfolio) that
//! live trading will need; set `KALSHI_API_KEY_ID` + `KALSHI_PRIVATE_KEY_PATH`/`_PEM` for those.

use std::collections::HashSet;

use base64::Engine as _;
use chrono::{NaiveDate, Utc};
use rsa::pkcs1::DecodeRsaPrivateKey;
use rsa::pkcs8::DecodePrivateKey;
use rsa::pss::SigningKey;
use rsa::signature::{RandomizedSigner, SignatureEncoding};
use rsa::RsaPrivateKey;
use serde_json::Value;
use sha2::Sha256;

use super::polymarket_history::{
    infer_market_type_and_threshold, is_weather_like_market, json_str, value_as_f64,
};
use super::WeatherMarketRow;
use crate::config;
use crate::utils::parse_date;

/// Path prefix that is part of the signed message (host lives in `base_url`).
const API_PREFIX: &str = "/trade-api/v2";

/// Daily high-temperature series to scan. An untargeted `/markets?status=open` scan never reaches
/// them (thousands of non-weather markets come first), so fetching is per-series. These are the
/// series whose settlement stations are verified in `stations::KALSHI_STATIONS`.
const WEATHER_SERIES: &[&str] = &[
    "KXHIGHNY",
    "KXHIGHCHI",
    "KXHIGHAUS",
    "KXHIGHDEN",
    "KXHIGHLAX",
    "KXHIGHMIA",
    "KXHIGHPHIL",
    "KXHIGHTDAL",
    "KXHIGHTSEA",
    "KXHIGHTATL",
    "KXHIGHTBOS",
    "KXHIGHTPHX",
    "KXHIGHTLV",
    "KXHIGHTDC",
    "KXHIGHTHOU",
];

#[derive(Debug, thiserror::Error)]
pub enum KalshiHistoryError {
    #[error("request failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error("invalid Kalshi private key (expected an RSA PKCS#8 or PKCS#1 PEM)")]
    BadKey,
}

/// Signs Kalshi requests. Present only when both a key id and a parseable private key are
/// configured. Shared with the credentialed trade client (`kalshi_trade`).
pub(crate) struct KalshiAuth {
    key_id: String,
    signing_key: SigningKey<Sha256>,
}

impl KalshiAuth {
    pub(crate) fn from_env() -> Option<Self> {
        let key_id = config::kalshi_api_key_id();
        let pem = config::kalshi_private_key_pem();
        if key_id.is_empty() || pem.trim().is_empty() {
            return None;
        }
        Self::from_pem(key_id, &pem).ok()
    }

    fn from_pem(key_id: String, pem: &str) -> Result<Self, KalshiHistoryError> {
        let key = RsaPrivateKey::from_pkcs8_pem(pem)
            .or_else(|_| RsaPrivateKey::from_pkcs1_pem(pem))
            .map_err(|_| KalshiHistoryError::BadKey)?;
        Ok(Self {
            key_id,
            signing_key: SigningKey::<Sha256>::new(key),
        })
    }

    /// The three auth headers for `method path` at the current time. Signs `timestamp+method+path`
    /// with RSA-PSS(SHA-256) — the query string is NOT part of the signed message.
    pub(crate) fn headers(&self, method: &str, path: &str) -> Vec<(&'static str, String)> {
        let ts = Utc::now().timestamp_millis().to_string();
        let msg = format!("{ts}{method}{path}");
        // PSS salt is randomized by design; this is network auth, not a deterministic model path.
        let sig = self
            .signing_key
            .sign_with_rng(&mut rand::thread_rng(), msg.as_bytes());
        let b64 = base64::engine::general_purpose::STANDARD.encode(sig.to_bytes());
        vec![
            ("KALSHI-ACCESS-KEY", self.key_id.clone()),
            ("KALSHI-ACCESS-SIGNATURE", b64),
            ("KALSHI-ACCESS-TIMESTAMP", ts),
        ]
    }
}

pub struct KalshiHistoryDownloader {
    base_url: String,
    client: reqwest::Client,
    auth: Option<KalshiAuth>,
}

impl Default for KalshiHistoryDownloader {
    fn default() -> Self {
        Self::new()
    }
}

impl KalshiHistoryDownloader {
    pub fn new() -> Self {
        Self {
            base_url: config::kalshi_public_base_url(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(20))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            auth: KalshiAuth::from_env(),
        }
    }

    /// Market data is anonymous on Kalshi's public production API, so this downloader is always
    /// available. (Credentials — `has_trade_auth` — are only needed for the trade endpoints.)
    pub fn is_available(&self) -> bool {
        true
    }

    /// Whether signing credentials for the credentialed trade endpoints are configured.
    pub fn has_trade_auth(&self) -> bool {
        self.auth.is_some()
    }

    /// Weather markets for the requested side: `active` ⇒ open/unresolved, else settled/resolved.
    ///
    /// The anonymous market list NULLS the price fields (`last_price`/`yes_bid`/`yes_ask`), but the
    /// per-market ORDERBOOK is public — so for active markets the book is fetched per ticker, the
    /// top of book fills `best_bid`/`best_ask`, and the mid replaces the 0.50 no-trade fallback.
    pub async fn download_weather_markets(
        &self,
        active: bool,
        limit: usize,
    ) -> Result<Vec<WeatherMarketRow>, KalshiHistoryError> {
        let status = if active { "open" } else { "settled" };
        let mut out: Vec<WeatherMarketRow> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for series in WEATHER_SERIES {
            if out.len() >= limit {
                break;
            }
            let mut raw = self
                .fetch_markets(series, Some(status), limit - out.len())
                .await?;
            if active && raw.is_empty() {
                // 2026-08-14: the `status=open` series scan started returning empty for EVERY
                // weather series while direct ticker lookups kept working — the same silent
                // server-side breakage mode as `status=settled` in mid-July. Re-scan without the
                // status filter (bounded by min_close_ts=now inside fetch_markets) and let
                // market_looks_open() do the filtering here. Harmless once the filter recovers:
                // a genuinely empty series is one cheap extra request.
                raw = self.fetch_markets(series, None, limit - out.len()).await?;
                raw.retain(market_looks_open);
                if !raw.is_empty() {
                    eprintln!(
                        "  {series}: status=open scan empty; unfiltered fallback found {} open markets",
                        raw.len()
                    );
                }
            }
            self.ingest_markets(&raw, active, &mut seen, &mut out).await;
        }
        if active && out.is_empty() {
            // 2026-08-17: the unfiltered per-series fallback ALSO came back empty on every series
            // while the settled scan (same series_ticker param) kept returning rows — so the
            // series-level market list appears to hide open markets entirely. Last resort: query by
            // EVENT ticker ({series}-{YYMMMDD}, the same segment settlement tickers embed and
            // direct ticker lookups answer) for today and the next two target days.
            eprintln!(
                "  Kalshi series scans (status=open AND unfiltered) empty for all {} series; probing by event ticker...",
                WEATHER_SERIES.len()
            );
            let today = Utc::now().date_naive();
            for series in WEATHER_SERIES {
                if out.len() >= limit {
                    break;
                }
                for offset in 0..=2i64 {
                    let event = event_ticker(series, today + chrono::Duration::days(offset));
                    let mut raw = self.fetch_markets_by_event(&event).await;
                    raw.retain(market_looks_open);
                    if !raw.is_empty() {
                        eprintln!(
                            "  {event}: event-ticker probe found {} open markets",
                            raw.len()
                        );
                    }
                    self.ingest_markets(&raw, true, &mut seen, &mut out).await;
                }
            }
        }
        Ok(out)
    }

    /// Parse raw market objects and push the rows that match the requested side into `out`,
    /// filling top-of-book prices for active markets. Shared by every scan tier.
    async fn ingest_markets(
        &self,
        raw: &[Value],
        active: bool,
        seen: &mut HashSet<String>,
        out: &mut Vec<WeatherMarketRow>,
    ) {
        for m in raw {
            if let Some(mut row) = parse_kalshi_market(m) {
                if active == row.outcome.is_none() && seen.insert(row.market_id.clone()) {
                    if active {
                        let (bid, ask) = self.fetch_top_of_book(&row.market_id).await;
                        row.best_bid = bid;
                        row.best_ask = ask;
                        let no_last = m
                            .get("last_price")
                            .and_then(value_as_f64)
                            .filter(|&c| c > 0.0)
                            .is_none();
                        if no_last {
                            if let (Some(b), Some(a)) = (bid, ask) {
                                row.price = (b + a) / 2.0;
                            }
                        }
                        // gentle on the public API: ~10 req/s
                        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    }
                    out.push(row);
                }
            }
        }
    }

    /// One page of `GET /markets?event_ticker=<event>` with no status filter. A missing event or
    /// any fetch/parse failure is just an empty list — the caller treats this as a probe.
    async fn fetch_markets_by_event(&self, event: &str) -> Vec<Value> {
        let path = format!("{API_PREFIX}/markets");
        let url = format!("{}{path}", self.base_url);
        let mut req = self
            .client
            .get(&url)
            .query(&[("limit", "1000"), ("event_ticker", event)]);
        if let Some(auth) = &self.auth {
            for (k, v) in auth.headers("GET", &path) {
                req = req.header(k, v);
            }
        }
        let Ok(resp) = req.send().await else {
            return Vec::new();
        };
        let Ok(val) = resp.json::<Value>().await else {
            return Vec::new();
        };
        val.get("markets")
            .and_then(|m| m.as_array())
            .cloned()
            .unwrap_or_default()
    }

    /// Best (YES bid, YES ask) in [0,1] from the public per-market orderbook. Kalshi books rest
    /// YES buys and NO buys: best YES ask = 1 − best NO bid. Either side may be empty; any fetch
    /// or parse failure is just (None, None).
    async fn fetch_top_of_book(&self, ticker: &str) -> (Option<f64>, Option<f64>) {
        let url = format!("{}{API_PREFIX}/markets/{ticker}/orderbook", self.base_url);
        let Ok(resp) = self.client.get(&url).send().await else {
            return (None, None);
        };
        let Ok(val) = resp.json::<Value>().await else {
            return (None, None);
        };
        // "orderbook_fp": {"yes_dollars": [["0.0100","751.00"],...], "no_dollars": [...]} — dollar
        // strings; the legacy "orderbook" variant uses integer-cent pairs. Handle both.
        let best = |levels: Option<&Value>, cents: bool| -> Option<f64> {
            levels?
                .as_array()?
                .iter()
                .filter_map(|lvl| {
                    let p = lvl.get(0)?;
                    let x = p
                        .as_f64()
                        .or_else(|| p.as_str().and_then(|s| s.parse::<f64>().ok()))?;
                    Some(if cents { x / 100.0 } else { x })
                })
                .fold(None, |acc: Option<f64>, v| {
                    Some(acc.map_or(v, |a| a.max(v)))
                })
        };
        let (yes_best, no_best) = if let Some(fp) = val.get("orderbook_fp") {
            (
                best(fp.get("yes_dollars"), false),
                best(fp.get("no_dollars"), false),
            )
        } else if let Some(ob) = val.get("orderbook") {
            (best(ob.get("yes"), true), best(ob.get("no"), true))
        } else {
            (None, None)
        };
        let in_range = |x: f64| (x > 0.0 && x < 1.0).then_some(x);
        (
            yes_best.and_then(in_range),
            no_best.map(|n| 1.0 - n).and_then(in_range),
        )
    }

    /// Resolve outcomes for specific tickers via direct `GET /markets/<ticker>` lookups — the
    /// Kalshi analog of `PolymarketHistoryDownloader::fetch_outcomes_for_ids`. The bulk
    /// `status=settled` series scan silently stopped returning some series in mid-July 2026
    /// (AUS/LAX/MIA accrued unresolved snapshots for a week while NY/CHI/DEN/PHIL kept resolving),
    /// but a settled market always answers a direct ticker lookup, which is exactly what the
    /// capture daemon stored. Unresolved, voided, or missing tickers contribute nothing.
    pub async fn fetch_outcomes_for_tickers(
        &self,
        tickers: &[String],
    ) -> std::collections::HashMap<String, f64> {
        let mut out = std::collections::HashMap::new();
        for ticker in tickers {
            let path = format!("{API_PREFIX}/markets/{ticker}");
            let url = format!("{}{path}", self.base_url);
            let mut req = self.client.get(&url);
            if let Some(auth) = &self.auth {
                for (k, v) in auth.headers("GET", &path) {
                    req = req.header(k, v);
                }
            }
            match req.send().await {
                Ok(resp) => {
                    if let Ok(val) = resp.json::<Value>().await {
                        if let Some(o) = settled_outcome_from_market_response(&val) {
                            out.insert(ticker.clone(), o);
                        }
                    }
                }
                Err(e) => eprintln!("  direct Kalshi outcome lookup for {ticker} failed: {e}"),
            }
            // gentle on the public API: ~10 req/s
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        out
    }

    /// Cursor-paginate `GET /markets?series_ticker=<series>[&status=<status>]` up to `limit` rows.
    /// `status: None` is the fallback scan for when a server-side status filter silently breaks:
    /// it bounds the walk with `min_close_ts = now` instead, so pagination never wades into the
    /// hundreds of already-settled strikes each daily series accrues per season.
    async fn fetch_markets(
        &self,
        series: &str,
        status: Option<&str>,
        limit: usize,
    ) -> Result<Vec<Value>, KalshiHistoryError> {
        let path = format!("{API_PREFIX}/markets");
        let url = format!("{}{path}", self.base_url);

        let mut out: Vec<Value> = Vec::new();
        let mut cursor: Option<String> = None;
        loop {
            if out.len() >= limit {
                break;
            }
            let mut query: Vec<(&str, String)> = vec![
                ("limit", "1000".to_string()),
                ("series_ticker", series.to_string()),
            ];
            match status {
                Some("settled") => {
                    query.push(("status", "settled".to_string()));
                    // Only recent resolutions matter (finalizing our own snapshots); each series
                    // accrues hundreds of settled strikes per season.
                    let since = Utc::now().timestamp() - 14 * 24 * 3600;
                    query.push(("min_close_ts", since.to_string()));
                }
                Some(s) => query.push(("status", s.to_string())),
                None => {
                    // Unfiltered fallback: anything still trading closes in the future.
                    query.push(("min_close_ts", Utc::now().timestamp().to_string()));
                }
            }
            if let Some(c) = &cursor {
                query.push(("cursor", c.clone()));
            }

            // Anonymous GET — the public host serves market data unauthenticated. Signed headers
            // are still attached when creds exist (harmless there, required on the trade host).
            let mut req = self.client.get(&url).query(&query);
            if let Some(auth) = &self.auth {
                for (k, v) in auth.headers("GET", &path) {
                    req = req.header(k, v);
                }
            }
            let val = req.send().await?.json::<Value>().await?;

            let markets = val.get("markets").and_then(|m| m.as_array());
            match markets {
                Some(ms) if !ms.is_empty() => out.extend(ms.iter().cloned()),
                _ => break,
            }
            cursor = val
                .get("cursor")
                .and_then(|c| c.as_str())
                .filter(|s| !s.is_empty())
                .map(String::from);
            if cursor.is_none() {
                break;
            }
        }
        Ok(out)
    }
}

/// Daily-series event ticker for a target day: `KXHIGHNY` + 2026-08-18 ⇒ `KXHIGHNY-26AUG18`.
/// The exact inverse of the `YYMMMDD` segment `ticker_date` parses out of settlement tickers.
fn event_ticker(series: &str, date: NaiveDate) -> String {
    format!(
        "{series}-{}",
        date.format("%y%b%d").to_string().to_ascii_uppercase()
    )
}

/// Client-side "is this market open for trading?" for the unfiltered fallback scan. Trusts the
/// market's own `status` string when it's a value we recognize; otherwise falls back to "unresolved
/// and closes in the future" (which also excludes closed-but-unsettled markets). Deliberately
/// permissive about status VALUES we've never seen — if Kalshi renamed the states (the likely cause
/// of the `status=open` query filter breaking), hard-coding today's names would re-break silently.
fn market_looks_open(m: &Value) -> bool {
    match json_str(m, &["status"]).as_deref() {
        Some("active" | "open") => true,
        Some("closed" | "settled" | "finalized" | "determined" | "initialized" | "inactive") => {
            false
        }
        _ => {
            kalshi_outcome(m).is_none()
                && ["close_time", "expiration_time"]
                    .iter()
                    .find_map(|k| m.get(*k).and_then(|v| v.as_str()))
                    .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                    .is_some_and(|t| t.timestamp() > Utc::now().timestamp())
        }
    }
}

/// Parse one raw Kalshi market into a `WeatherMarketRow` (None if it isn't a priceable weather market).
/// City/weather-gate reuse the shared title parsers; the bucket shape comes from Kalshi's structured
/// strikes (with the title parser as a fallback for shapes without strikes, e.g. rain yes/no).
fn parse_kalshi_market(m: &Value) -> Option<WeatherMarketRow> {
    let title = json_str(m, &["title"])?;
    let subtitle = json_str(m, &["yes_sub_title", "subtitle"]).unwrap_or_default();
    let combined = format!("{title} {subtitle}");
    let lower = combined.to_ascii_lowercase();
    if !is_weather_like_market(&lower) {
        return None;
    }
    let city = crate::cities::infer_city(&combined)?;
    let (market_type, threshold, threshold_upper, unit) =
        strike_shape(m).or_else(|| infer_market_type_and_threshold(&combined))?;

    Some(WeatherMarketRow {
        target_date: kalshi_target_date(m)?,
        market_id: json_str(m, &["ticker", "id"])?,
        market_title: if subtitle.is_empty() {
            title
        } else {
            format!("{title} — {subtitle}")
        },
        market_type,
        threshold,
        threshold_upper,
        unit,
        city: city.to_string(),
        price: kalshi_price(m),
        outcome: kalshi_outcome(m),
        source: "kalshi".to_string(),
        best_bid: kalshi_cents(m, "yes_bid"),
        best_ask: kalshi_cents(m, "yes_ask"),
        volume: m.get("volume").and_then(value_as_f64),
        volume_24h: m.get("volume_24h").and_then(value_as_f64),
        open_interest: m.get("open_interest").and_then(value_as_f64),
        liquidity: m.get("liquidity").and_then(value_as_f64),
    })
}

/// One side of the YES book, cents → [0,1]. 0 (or 100 for the ask) means the side is empty.
fn kalshi_cents(m: &Value, key: &str) -> Option<f64> {
    m.get(key)
        .and_then(value_as_f64)
        .filter(|&c| c > 0.0 && c < 100.0)
        .map(|c| c / 100.0)
}

/// Bucket shape from Kalshi's structured strikes (US temperature markets are in °F). None when the
/// market has no usable strikes, so the caller can fall back to the shared title parser.
///
/// Kalshi's `greater`/`less` are STRICT: floor_strike 105 titled "106° or above" resolves YES iff
/// the integer CLI high ≥ 106, and cap_strike 98 titled "97° or below" iff ≤ 97 — so strict shapes
/// shift by one whole degree to the inclusive threshold our `temp_at_least`/`temp_at_most` mean.
/// `between` floor/cap are already the inclusive integer bucket ends ("98° to 99°" ⇒ [98, 99]).
/// Verified against 6,030 settled markets (100% consistent with NWS CLI highs).
fn strike_shape(m: &Value) -> Option<(String, f64, Option<f64>, Option<String>)> {
    let floor = m.get("floor_strike").and_then(value_as_f64);
    let cap = m.get("cap_strike").and_then(value_as_f64);
    let unit = Some("F".to_string());
    match json_str(m, &["strike_type"]).as_deref() {
        Some("greater") => Some(("temp_at_least".to_string(), floor? + 1.0, None, unit)),
        Some("greater_or_equal") => Some(("temp_at_least".to_string(), floor?, None, unit)),
        Some("less") => Some(("temp_at_most".to_string(), cap.or(floor)? - 1.0, None, unit)),
        Some("less_or_equal") => Some(("temp_at_most".to_string(), cap.or(floor)?, None, unit)),
        Some("between") => Some(("temp_bucket".to_string(), floor?, Some(cap?), unit)),
        _ => None,
    }
}

/// YES-side price in [0,1]. Kalshi quotes cents (0..100): prefer last trade, else the bid/ask mid,
/// else 0.50.
fn kalshi_price(m: &Value) -> f64 {
    let cents = m
        .get("last_price")
        .and_then(value_as_f64)
        .filter(|&c| c > 0.0)
        .or_else(|| {
            let bid = m.get("yes_bid").and_then(value_as_f64)?;
            let ask = m.get("yes_ask").and_then(value_as_f64)?;
            (ask > 0.0).then_some((bid + ask) / 2.0)
        })
        .unwrap_or(50.0);
    (cents / 100.0).clamp(0.0, 1.0)
}

/// Outcome from a single-market response, which nests the market under `"market"` (a bare market
/// object is accepted too, matching the bulk-list element shape).
fn settled_outcome_from_market_response(val: &Value) -> Option<f64> {
    kalshi_outcome(val.get("market").unwrap_or(val))
}

/// Realized outcome from Kalshi's `result` (yes/no), or None while unresolved / voided.
fn kalshi_outcome(m: &Value) -> Option<f64> {
    match json_str(m, &["result"]).as_deref() {
        Some("yes") => Some(1.0),
        Some("no") => Some(0.0),
        _ => None,
    }
}

/// The day the market is ABOUT. The ticker embeds it (`KXHIGHNY-26JUL04-B98.5` ⇒ 2026-07-04);
/// close/expiration times are the morning AFTER the target day, so parsing those (the old
/// behavior, kept only as a fallback) put every Kalshi market one day late.
fn kalshi_target_date(m: &Value) -> Option<NaiveDate> {
    json_str(m, &["ticker"])
        .as_deref()
        .and_then(ticker_date)
        .or_else(|| {
            ["close_time", "expiration_time", "expected_expiration_time"]
                .iter()
                .find_map(|k| m.get(*k).and_then(|v| v.as_str()))
                .and_then(parse_date)
        })
}

/// Parse the `YYMMMDD` segment of a Kalshi event/market ticker (`KXHIGHNY-26JUL04-…`).
fn ticker_date(ticker: &str) -> Option<NaiveDate> {
    let seg = ticker.split('-').nth(1)?;
    if seg.len() != 7 || !seg.is_char_boundary(2) {
        return None;
    }
    let yy: i32 = seg[0..2].parse().ok()?;
    let month = match &seg[2..5] {
        "JAN" => 1,
        "FEB" => 2,
        "MAR" => 3,
        "APR" => 4,
        "MAY" => 5,
        "JUN" => 6,
        "JUL" => 7,
        "AUG" => 8,
        "SEP" => 9,
        "OCT" => 10,
        "NOV" => 11,
        "DEC" => 12,
        _ => return None,
    };
    let dd: u32 = seg[5..7].parse().ok()?;
    NaiveDate::from_ymd_opt(2000 + yy, month, dd)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_ticker_roundtrips_through_ticker_date() {
        let d = NaiveDate::from_ymd_opt(2026, 8, 18).unwrap();
        let event = event_ticker("KXHIGHNY", d);
        assert_eq!(event, "KXHIGHNY-26AUG18");
        // A market ticker under this event parses back to the same target day.
        assert_eq!(ticker_date(&format!("{event}-B85.5")), Some(d));
        assert_eq!(ticker_date(&event), Some(d));
    }

    #[test]
    fn market_looks_open_trusts_known_statuses() {
        let open: Value = serde_json::from_str(r#"{"status":"active","result":""}"#).unwrap();
        assert!(market_looks_open(&open));
        let closed: Value =
            serde_json::from_str(r#"{"status":"closed","close_time":"2099-01-01T05:00:00Z"}"#)
                .unwrap();
        assert!(!market_looks_open(&closed));
        let settled: Value = serde_json::from_str(r#"{"status":"settled","result":"no"}"#).unwrap();
        assert!(!market_looks_open(&settled));
    }

    #[test]
    fn market_looks_open_unknown_status_falls_back_to_close_time() {
        // Unknown/missing status: open iff unresolved AND close_time is in the future.
        let future: Value = serde_json::from_str(
            r#"{"status":"something_new","result":"","close_time":"2099-01-01T05:00:00Z"}"#,
        )
        .unwrap();
        assert!(market_looks_open(&future));
        let past: Value =
            serde_json::from_str(r#"{"result":"","close_time":"2020-01-01T05:00:00Z"}"#).unwrap();
        assert!(!market_looks_open(&past));
        let resolved: Value =
            serde_json::from_str(r#"{"result":"yes","close_time":"2099-01-01T05:00:00Z"}"#)
                .unwrap();
        assert!(!market_looks_open(&resolved));
        let no_times: Value = serde_json::from_str(r#"{"result":""}"#).unwrap();
        assert!(!market_looks_open(&no_times));
    }

    // Throwaway 2048-bit RSA key (PKCS#8). Test-only — signs nothing real.
    const TEST_PEM: &str = "-----BEGIN PRIVATE KEY-----\n\
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCczuHezwXT3mD0\n\
23XpYm0v/H+v2XOA9xrJCo1AwOgnS5qbeiOvMdbSa13VvaRqPGLPVRelGmWh6sOh\n\
MgiDPvIp5OubI0BckB5PEErrhBlXwSuRBTc6tddAAvb1Zd57afghlZvF8NXAIwHj\n\
vELrhIe2MnHu1OpTwVXGNeCU0sOnAQtbb/SELwEW5sTTxkZsthk1KdoHbIlpzSGF\n\
ws8GO+yCBqg32U7y/vpPOF8Y6JGMKFtVtZ19K2S9z0KsNNu6OGVjX+a+5H/BKwrx\n\
+UnM393d6utnO9dHVRdj3lYPW9IG1usdKQgHscVdbFL8rA1/8KTWIc9EM0zJkDRp\n\
IldsWlopAgMBAAECggEAARskE4CP9yKR3rmImS/FhjYbl/IoCNnRyQ3dlhSz9hxj\n\
5cwEz6dgJjThUFVdHZSd/zLa7VUBea8+gpJd5jXyJd0eYcVFV6z4npUQlUOq4hfs\n\
M7qD8X8uBijUhvBsl06izR+uxGtRJmTd5cBmKA5Z4sOktE4i8f/IJk4q6o+GdN8O\n\
dtAvDdbAxR3+8tLfqtzgW6zhRhFoObICx+qO40c1FsenlxsiRsV9oESpfgCL9Wop\n\
CTOrxHQ7ZTDTGzRPam4nZrFB52boAkd42aQL838cdvWSFYziC3ubAFM3ppZEIhCH\n\
f/RsRQrI4F2CYcWlMNFHIS+7D2sTrkyIInUMzNyTeQKBgQDZn7/ac6wzGpWs07p3\n\
fb1OqJbs8o1q10a07eEvynV2dqg57LZVUdVsCIf16iTnw9Q+JVDWhhXcuTSapO1b\n\
43eg73XclG3n4qwbwqY1TbA9SNANFIIm/Y+HADpcmLMkm7kcNY/rXO+DNOxH1Fel\n\
HxqF/ggQ/nW2MdS6TSrlBH3H8wKBgQC4dbUdmnRZh3zIJkc5exv/kUAjVu8UNEg0\n\
SS9Fv3ViXEmn4+cwe4teUzShlaVE+cV5Z3QLwvTF6DY1awYxgkR5Q5IDrrMwGUsl\n\
xga459I9YqmYU68iso5WSWBM2qLzmGPzzaNlT1myAYUdeIUnmVQ+AS+iyiOZ39sT\n\
kqtcvepYcwKBgQDQYEYzxc58oFfwxEsnXx6E0qiw+Q6v3rG1TJDyUclnRPaPgDiW\n\
OWVBrGmC7k+oG7p+RvzAbGNClZPo/0LHWWaSkrcoHneeMUzax+ad42V9SNrtq1V0\n\
QBXODknTn+LoirMUb7T+iF5OI76aiJfjY4TiB/txSjUr30rxmDIaV9KYQQKBgHR3\n\
BWLqmZnJnPS2jnPxkgz8BdYKVquYExrINovARfpTsMHLeLY42xc0S9/WH2J8nb14\n\
n2Gpt2iZpFh3+ZIa4Ob7zd57WHH/Fl6EOMxYunq9p70g4Ux0FsDOVXpQ9V/+kOVn\n\
qkoWHtjwcr7X0KLfAbygfRY4sN+/4o/qJD5LPwKXAoGBAMP2IA82wRcHSAaF9y96\n\
6rgqOeJVm+QsYwm0B7pN9ShYPw4LCWQ4Hctlfu4gKsaBfhlOUs+yGQUvGQNCxTAm\n\
GMvVNHMr5JUZbcTF4UocAy94s9c9QGz5d+dHVLGDrRgtnsmclKBpHnoOun3mbP4Y\n\
b1qMwu767YVXiVRAobFRB/Gy\n\
-----END PRIVATE KEY-----\n";

    fn market(json: &str) -> Value {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn parses_between_bucket_active() {
        let row = parse_kalshi_market(&market(
            r#"{"ticker":"KXHIGHCHI-26JUN30-B71.5","title":"Highest temperature in Chicago",
                "yes_sub_title":"71° to 72°","strike_type":"between","floor_strike":71,"cap_strike":72,
                "last_price":34,"yes_bid":30,"yes_ask":37,"result":"","close_time":"2026-07-01T04:59:00Z",
                "volume":5120,"volume_24h":301,"open_interest":2214}"#,
        ))
        .expect("should parse");
        assert_eq!(row.source, "kalshi");
        assert_eq!(row.city, "Chicago");
        assert_eq!(row.market_type, "temp_bucket");
        assert_eq!(row.threshold, 71.0);
        assert_eq!(row.threshold_upper, Some(72.0));
        assert_eq!(row.unit.as_deref(), Some("F"));
        assert!((row.price - 0.34).abs() < 1e-9, "cents → 0..1");
        assert_eq!(row.best_bid, Some(0.30));
        assert_eq!(row.best_ask, Some(0.37));
        assert_eq!(row.outcome, None);
        assert_eq!(row.volume, Some(5120.0));
        assert_eq!(row.volume_24h, Some(301.0));
        assert_eq!(row.open_interest, Some(2214.0));
        assert_eq!(row.liquidity, None, "absent field stays None");
        // Target day comes from the TICKER; close_time is the morning after (would be off by one).
        assert_eq!(
            row.target_date,
            NaiveDate::from_ymd_opt(2026, 6, 30).unwrap()
        );
    }

    #[test]
    fn parses_greater_settled_yes() {
        // Kalshi 'greater' is strict: floor 90 is titled "91° or above" ⇒ inclusive threshold 91.
        let row = parse_kalshi_market(&market(
            r#"{"ticker":"KXHIGHMIA-26JUN29-T90","title":"High temperature in Miami",
                "yes_sub_title":"91° or above","strike_type":"greater","floor_strike":90,
                "yes_bid":0,"yes_ask":0,"result":"yes","close_time":"2026-06-30T04:59:00Z"}"#,
        ))
        .expect("should parse");
        assert_eq!(row.market_type, "temp_at_least");
        assert_eq!(row.threshold, 91.0);
        assert_eq!(row.outcome, Some(1.0));
        assert!(
            (row.price - 0.50).abs() < 1e-9,
            "no trade / no book → 0.50 fallback"
        );
        assert_eq!(row.best_bid, None, "0-cent bid = empty side");
        assert_eq!(row.best_ask, None);
        assert_eq!(
            row.target_date,
            NaiveDate::from_ymd_opt(2026, 6, 29).unwrap()
        );
    }

    #[test]
    fn parses_less_as_strict() {
        // 'less' cap 98 is titled "97° or below" ⇒ inclusive temp_at_most threshold 97.
        let row = parse_kalshi_market(&market(
            r#"{"ticker":"KXHIGHNY-26JUL04-T98","title":"High temp in NYC",
                "yes_sub_title":"97° or below","strike_type":"less","cap_strike":98,
                "last_price":12,"result":"","close_time":"2026-07-05T04:59:00Z"}"#,
        ))
        .expect("should parse");
        assert_eq!(row.market_type, "temp_at_most");
        assert_eq!(row.threshold, 97.0);
        assert_eq!(
            row.target_date,
            NaiveDate::from_ymd_opt(2026, 7, 4).unwrap()
        );
    }

    #[test]
    fn skips_non_weather() {
        assert!(parse_kalshi_market(&market(
            r#"{"ticker":"X","title":"Will the Fed cut rates","close_time":"2026-06-30T23:00:00Z"}"#
        ))
        .is_none());
    }

    #[test]
    fn settled_outcome_handles_wrapped_and_bare_market() {
        let wrapped: Value =
            serde_json::from_str(r#"{"market":{"ticker":"KXHIGHLAX-26JUL14-T81","result":"no"}}"#)
                .unwrap();
        assert_eq!(settled_outcome_from_market_response(&wrapped), Some(0.0));

        let bare: Value =
            serde_json::from_str(r#"{"ticker":"KXHIGHLAX-26JUL14-T81","result":"yes"}"#).unwrap();
        assert_eq!(settled_outcome_from_market_response(&bare), Some(1.0));

        let unresolved: Value =
            serde_json::from_str(r#"{"market":{"ticker":"KXHIGHLAX-26JUL22-T81","result":""}}"#)
                .unwrap();
        assert_eq!(settled_outcome_from_market_response(&unresolved), None);
    }

    #[test]
    fn signing_roundtrips() {
        use rsa::pss::VerifyingKey;
        use rsa::signature::{Keypair, Verifier};

        let auth = KalshiAuth::from_pem("key-123".to_string(), TEST_PEM).expect("valid PEM");
        let headers = auth.headers("GET", "/trade-api/v2/markets");
        // three expected headers present
        assert!(headers.iter().any(|(k, _)| *k == "KALSHI-ACCESS-KEY"));
        let ts = &headers
            .iter()
            .find(|(k, _)| *k == "KALSHI-ACCESS-TIMESTAMP")
            .unwrap()
            .1;
        let sig_b64 = &headers
            .iter()
            .find(|(k, _)| *k == "KALSHI-ACCESS-SIGNATURE")
            .unwrap()
            .1;

        // the signature verifies against the reconstructed message with the matching public key
        let msg = format!("{ts}GET/trade-api/v2/markets");
        let sig_bytes = base64::engine::general_purpose::STANDARD
            .decode(sig_b64)
            .unwrap();
        let sig = rsa::pss::Signature::try_from(sig_bytes.as_slice()).unwrap();
        let vk: VerifyingKey<Sha256> = auth.signing_key.verifying_key();
        assert!(
            vk.verify(msg.as_bytes(), &sig).is_ok(),
            "signature must verify"
        );
    }
}
