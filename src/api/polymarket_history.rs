use chrono::{DateTime, NaiveDate, Utc};
use serde_json::Value;

use crate::types::SimulatedMarket;
use crate::utils::{contains_word, parse_date};

const DEFAULT_GAMMA_BASE_URL: &str = "https://gamma-api.polymarket.com";
const DEFAULT_CLOB_BASE_URL: &str = "https://clob.polymarket.com";

#[derive(Debug, thiserror::Error)]
pub enum PolymarketHistoryError {
    #[error("request failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// A weather market parsed into the model's pricing inputs, with its current price and (if resolved)
/// its outcome. Used by the capture daemon to snapshot active markets and finalize resolved ones.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WeatherMarketRow {
    pub target_date: NaiveDate,
    pub market_id: String,
    pub market_title: String,
    pub market_type: String,
    pub threshold: f64,
    pub threshold_upper: Option<f64>,
    pub unit: Option<String>,
    pub city: String,
    /// Current/last traded price of the YES side (0..1).
    pub price: f64,
    /// Realized outcome (1.0 YES / 0.0 NO), or None while the market is unresolved.
    pub outcome: Option<f64>,
}

#[derive(Clone)]
pub struct PolymarketHistoryDownloader {
    gamma_base_url: String,
    clob_base_url: String,
    client: reqwest::Client,
}

impl PolymarketHistoryDownloader {
    pub fn new(gamma_base_url: Option<String>, clob_base_url: Option<String>) -> Self {
        Self {
            gamma_base_url: gamma_base_url
                .unwrap_or_else(|| DEFAULT_GAMMA_BASE_URL.to_string())
                .trim_end_matches('/')
                .to_string(),
            clob_base_url: clob_base_url
                .unwrap_or_else(|| DEFAULT_CLOB_BASE_URL.to_string())
                .trim_end_matches('/')
                .to_string(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(20))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        }
    }

    pub async fn download_weather_history(
        &self,
        start_date: Option<NaiveDate>,
        end_date: Option<NaiveDate>,
        limit: usize,
        delay_ms: u64,
    ) -> Result<Vec<SimulatedMarket>, PolymarketHistoryError> {
        let markets = self.fetch_gamma_markets(limit).await?;

        let mut out = Vec::new();
        for market in markets {
            let Some(title) = json_str(&market, &["question", "title", "name"]) else {
                continue;
            };

            let normalized_title = title.to_ascii_lowercase();
            if !is_weather_like_market(&normalized_title) {
                continue;
            }

            let Some(city) = infer_city(&title) else {
                continue;
            };

            let Some((market_type, threshold, threshold_upper, unit)) =
                infer_market_type_and_threshold(&title)
            else {
                continue;
            };

            let Some(actual_outcome) = infer_actual_outcome(&market) else {
                continue;
            };

            let market_id = json_str(&market, &["id", "conditionId", "slug", "questionID"])
                .unwrap_or_else(|| format!("{}_{}", city, threshold));

            let token_id = infer_yes_token_id(&market);
            let history = if let Some(id) = token_id.as_deref() {
                match self.fetch_price_history(id, start_date, end_date).await {
                    Ok(h) => h,
                    Err(e) => {
                        eprintln!("warning: failed to fetch price history for market '{}': {}", title, e);
                        Vec::new()
                    }
                }
            } else {
                Vec::new()
            };

            // One row per market: the bucket is about a single day's high. Entry price = the opening
            // (earliest) traded price; date = the market's resolution day.
            let (entry_date, entry_price) = match history.first().copied() {
                Some(point) => point,
                None => match fallback_market_point(&market) {
                    Some(fb) => {
                        eprintln!(
                            "warning: using fabricated fallback price {:.2} for market '{}' (no traded history)",
                            fb.1, title
                        );
                        fb
                    }
                    None => continue,
                },
            };

            let date = market_target_date(&market).unwrap_or(entry_date);
            if start_date.map(|s| date < s).unwrap_or(false)
                || end_date.map(|e| date > e).unwrap_or(false)
            {
                continue;
            }

            out.push(SimulatedMarket {
                date,
                market_id,
                market_title: title.clone(),
                market_type,
                threshold,
                threshold_upper,
                unit,
                market_price: entry_price.clamp(0.0, 1.0),
                actual_outcome,
                city,
            });

            if delay_ms > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            }
        }

        out.sort_by(|a, b| {
            a.date
                .cmp(&b.date)
                .then_with(|| a.market_id.cmp(&b.market_id))
        });

        Ok(out)
    }

    async fn fetch_gamma_markets(&self, limit: usize) -> Result<Vec<Value>, reqwest::Error> {
        let url = format!("{}/markets", self.gamma_base_url);
        let mut all: Vec<Value> = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

        // 1. Targeted discovery: closed "temperature"-tagged EVENTS, whose child markets are the
        // bucket ladders we want. This is by far the highest-yield source for weather markets.
        self.collect_temperature_event_markets(limit, &mut all, &mut seen)
            .await?;

        // 2. Top up from the weather-dense closed-market pool, then newest-by-id, to catch anything
        // the tag misses (precipitation/wind markets aren't under the temperature tag).
        let strategies: [&[(&str, &str)]; 2] = [
            &[("closed", "true"), ("order", "volume"), ("ascending", "false")],
            &[("order", "id"), ("ascending", "false")],
        ];

        for extra in strategies {
            if all.len() >= limit {
                break;
            }
            let mut offset = 0usize;
            loop {
                if all.len() >= limit {
                    break;
                }
                // The server caps a page at ~100 rows, so request 100 and advance by the count.
                let mut query: Vec<(&str, String)> =
                    vec![("limit", "100".to_string()), ("offset", offset.to_string())];
                for (k, v) in extra {
                    query.push((k, (*v).to_string()));
                }

                let resp = self.client.get(&url).query(&query).send().await?;
                let val = resp.json::<Value>().await?;
                let rows = extract_rows(&val);
                if rows.is_empty() {
                    break;
                }
                offset += rows.len();
                for row in rows {
                    let id = json_str(&row, &["id", "conditionId", "slug"]).unwrap_or_default();
                    if id.is_empty() || seen.insert(id) {
                        all.push(row);
                    }
                }
            }
        }

        all.truncate(limit);
        Ok(all)
    }

    /// Pull closed "temperature"-tagged events and flatten their child markets (the bucket ladders)
    /// into `all`, deduped by id. Each child market carries the same question/outcomes/clobTokenIds
    /// fields the rest of the pipeline expects.
    async fn collect_temperature_event_markets(
        &self,
        limit: usize,
        all: &mut Vec<Value>,
        seen: &mut std::collections::HashSet<String>,
    ) -> Result<(), reqwest::Error> {
        let url = format!("{}/events", self.gamma_base_url);
        let mut offset = 0usize;
        loop {
            if all.len() >= limit {
                break;
            }
            let query = [
                ("limit", "100".to_string()),
                ("offset", offset.to_string()),
                ("closed", "true".to_string()),
                ("tag_slug", "temperature".to_string()),
                ("order", "startDate".to_string()),
                ("ascending", "false".to_string()),
            ];
            let resp = self.client.get(&url).query(&query).send().await?;
            let val = resp.json::<Value>().await?;
            let events = extract_rows(&val);
            if events.is_empty() {
                break;
            }
            offset += events.len();
            for ev in &events {
                let Some(markets) = ev.get("markets").and_then(|m| m.as_array()) else {
                    continue;
                };
                for m in markets {
                    if all.len() >= limit {
                        break;
                    }
                    let id = json_str(m, &["id", "conditionId", "slug"]).unwrap_or_default();
                    if id.is_empty() || seen.insert(id) {
                        all.push(m.clone());
                    }
                }
            }
        }
        Ok(())
    }

    /// Fetch temperature-tagged weather markets parsed into pricing inputs. `active=true` returns
    /// open markets (current price, no outcome); `active=false` returns resolved markets (with
    /// outcome). Uses the events tag for discovery and `lastTradePrice` for the price (no per-market
    /// price-history call), so it is fast enough to run as a daily daemon.
    pub async fn download_weather_markets(
        &self,
        active: bool,
        limit: usize,
    ) -> Result<Vec<WeatherMarketRow>, PolymarketHistoryError> {
        let closed = if active { "false" } else { "true" };

        // Collect raw markets from BOTH sources (the temperature tag is reliable for the closed
        // batch but empty for current active markets, which only the /markets scan surfaces).
        let mut raw: Vec<Value> = Vec::new();
        let mut seen_ids: std::collections::HashSet<String> = std::collections::HashSet::new();

        // a) temperature-tagged events -> child markets
        let events_url = format!("{}/events", self.gamma_base_url);
        let mut offset = 0usize;
        loop {
            if raw.len() >= limit {
                break;
            }
            let query = [
                ("limit", "100".to_string()),
                ("offset", offset.to_string()),
                ("closed", closed.to_string()),
                ("tag_slug", "temperature".to_string()),
                ("order", "startDate".to_string()),
                ("ascending", "false".to_string()),
            ];
            let resp = self.client.get(&events_url).query(&query).send().await?;
            let val = resp.json::<Value>().await?;
            let events = extract_rows(&val);
            if events.is_empty() {
                break;
            }
            offset += events.len();
            for ev in &events {
                if let Some(markets) = ev.get("markets").and_then(|m| m.as_array()) {
                    for m in markets {
                        push_unique(&mut raw, &mut seen_ids, m, limit);
                    }
                }
            }
        }

        // b) /markets?closed=<>&order=volume scan (catches untagged active/closed markets)
        let markets_url = format!("{}/markets", self.gamma_base_url);
        let mut offset = 0usize;
        loop {
            if raw.len() >= limit {
                break;
            }
            let query = [
                ("limit", "100".to_string()),
                ("offset", offset.to_string()),
                ("closed", closed.to_string()),
                ("order", "volume".to_string()),
                ("ascending", "false".to_string()),
            ];
            let resp = self.client.get(&markets_url).query(&query).send().await?;
            let val = resp.json::<Value>().await?;
            let rows = extract_rows(&val);
            if rows.is_empty() {
                break;
            }
            offset += rows.len();
            for m in &rows {
                push_unique(&mut raw, &mut seen_ids, m, limit);
            }
        }

        // Parse + keep the rows matching the requested side (active => unresolved, closed => resolved).
        let mut out: Vec<WeatherMarketRow> = Vec::new();
        let mut seen_mid: std::collections::HashSet<String> = std::collections::HashSet::new();
        for m in &raw {
            if let Some(row) = parse_weather_market_row(m) {
                if active == row.outcome.is_none() && seen_mid.insert(row.market_id.clone()) {
                    out.push(row);
                }
            }
        }
        Ok(out)
    }

    /// Fetch the market's daily price series (one point per day). The CLOB rejects wide
    /// startTs/endTs spans ("interval too long"), so we ask for `interval=max` (the full traded
    /// history) and keep the OPENING (first) price of each day — the near-settlement price carries
    /// the resolution, so the opening is the least-leaky entry price for a backtest. Returns days
    /// in ascending order (so `.first()` is the overall opening price).
    async fn fetch_price_history(
        &self,
        token_id: &str,
        _start_date: Option<NaiveDate>,
        _end_date: Option<NaiveDate>,
    ) -> Result<Vec<(NaiveDate, f64)>, reqwest::Error> {
        let candidates: [Vec<(&str, String)>; 2] = [
            vec![("market", token_id.to_string()), ("interval", "max".to_string())],
            vec![("token_id", token_id.to_string()), ("interval", "max".to_string())],
        ];

        let url = format!("{}/prices-history", self.clob_base_url);
        for query in candidates {
            let Ok(resp) = self.client.get(&url).query(&query).send().await else {
                continue;
            };
            let Ok(val) = resp.json::<Value>().await else {
                continue;
            };

            let points = parse_price_history_points(&val); // time-ascending
            if !points.is_empty() {
                let mut seen: std::collections::HashSet<NaiveDate> = std::collections::HashSet::new();
                let mut out = Vec::new();
                for (d, p) in points {
                    if seen.insert(d) {
                        out.push((d, p)); // first (opening) price of day d
                    }
                }
                return Ok(out);
            }
        }

        Ok(Vec::new())
    }
}

impl Default for PolymarketHistoryDownloader {
    fn default() -> Self {
        Self::new(None, None)
    }
}

fn extract_rows(v: &Value) -> Vec<Value> {
    if let Some(arr) = v.as_array() {
        return arr.clone();
    }

    v.get("data")
        .and_then(|d| d.as_array())
        .cloned()
        .unwrap_or_default()
}

fn parse_price_history_points(v: &Value) -> Vec<(NaiveDate, f64)> {
    let history = v
        .get("history")
        .and_then(|h| h.as_array())
        .cloned()
        .or_else(|| v.as_array().cloned())
        .unwrap_or_default();

    history
        .into_iter()
        .filter_map(|row| {
            let ts = row
                .get("t")
                .or_else(|| row.get("timestamp"))
                .or_else(|| row.get("time"))
                .or_else(|| row.get("date"));

            let price = row
                .get("p")
                .or_else(|| row.get("price"))
                .or_else(|| row.get("close"))
                .and_then(value_as_f64)?;

            let date = ts.and_then(value_to_date)?;
            Some((date, price))
        })
        .collect()
}

fn json_str(v: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|k| v.get(*k).and_then(|x| x.as_str()).map(|s| s.to_string()))
}

fn value_as_f64(v: &Value) -> Option<f64> {
    v.as_f64()
        .or_else(|| v.as_i64().map(|x| x as f64))
        .or_else(|| v.as_u64().map(|x| x as f64))
        .or_else(|| v.as_str().and_then(|s| s.parse::<f64>().ok()))
}

fn value_to_date(v: &Value) -> Option<NaiveDate> {
    if let Some(ts) = value_as_f64(v) {
        // Accept both seconds and milliseconds.
        let secs = if ts > 10_000_000_000.0 {
            (ts / 1000.0) as i64
        } else {
            ts as i64
        };

        return DateTime::<Utc>::from_timestamp(secs, 0).map(|d| d.date_naive());
    }

    v.as_str().and_then(parse_date)
}

fn infer_yes_token_id(market: &Value) -> Option<String> {
    // Prefer explicit token arrays with outcome labels.
    if let Some(tokens) = market.get("tokens").and_then(|v| v.as_array()) {
        if let Some(token) = tokens.iter().find(|t| {
            t.get("outcome")
                .and_then(|v| v.as_str())
                .map(|s| s.eq_ignore_ascii_case("yes"))
                .unwrap_or(false)
        }) {
            if let Some(id) = token
                .get("token_id")
                .or_else(|| token.get("tokenId"))
                .or_else(|| token.get("id"))
                .and_then(|v| v.as_str())
            {
                return Some(id.to_string());
            }
        }
    }

    // Fallback to clobTokenIds as JSON string or raw array.
    if let Some(raw) = market.get("clobTokenIds") {
        if let Some(s) = raw.as_str() {
            if let Ok(ids) = serde_json::from_str::<Vec<String>>(s) {
                return ids.first().cloned();
            }
        }

        if let Some(arr) = raw.as_array() {
            return arr
                .first()
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
        }
    }

    None
}

fn infer_actual_outcome(market: &Value) -> Option<f64> {
    let direct = ["resolvedOutcome", "resolution", "winner", "outcome", "result"]
        .iter()
        .find_map(|k| market.get(*k));

    if let Some(v) = direct {
        if let Some(outcome) = parse_binary_outcome(v) {
            return Some(outcome);
        }
    }

    if let Some(tokens) = market.get("tokens").and_then(|v| v.as_array()) {
        for token in tokens {
            let is_winner = token
                .get("winner")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
                || token
                    .get("isWinner")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

            if !is_winner {
                continue;
            }

            let outcome = token
                .get("outcome")
                .and_then(|v| v.as_str())
                .unwrap_or_default();

            if outcome.eq_ignore_ascii_case("yes") {
                return Some(1.0);
            }
            if outcome.eq_ignore_ascii_case("no") {
                return Some(0.0);
            }
        }
    }

    // Gamma schema: a resolved market exposes parallel `outcomes` /
    // `outcomePrices` arrays where the winning outcome's price is exactly "1".
    // (Unresolved markets carry fractional prices, so they yield None here.)
    if let (Some(outcomes), Some(prices)) = (
        market.get("outcomes").and_then(parse_str_array),
        market.get("outcomePrices").and_then(parse_outcome_prices),
    ) {
        if let Some(yes_idx) = outcomes.iter().position(|o| o.eq_ignore_ascii_case("yes")) {
            if let Some(price) = prices.get(yes_idx) {
                if (price - 1.0).abs() < 1e-9 {
                    return Some(1.0);
                }
                if price.abs() < 1e-9 {
                    return Some(0.0);
                }
            }
        }
    }

    None
}

fn parse_str_array(v: &Value) -> Option<Vec<String>> {
    if let Some(arr) = v.as_array() {
        return Some(arr.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect());
    }
    if let Some(s) = v.as_str() {
        if let Ok(arr) = serde_json::from_str::<Vec<String>>(s) {
            return Some(arr);
        }
    }
    None
}

fn parse_binary_outcome(v: &Value) -> Option<f64> {
    if let Some(b) = v.as_bool() {
        return Some(if b { 1.0 } else { 0.0 });
    }

    if let Some(x) = value_as_f64(v) {
        if (x - 1.0).abs() < f64::EPSILON {
            return Some(1.0);
        }
        if x.abs() < f64::EPSILON {
            return Some(0.0);
        }
    }

    let s = v.as_str()?.trim().to_ascii_lowercase();
    if ["yes", "y", "true", "1"].contains(&s.as_str()) {
        return Some(1.0);
    }
    if ["no", "n", "false", "0"].contains(&s.as_str()) {
        return Some(0.0);
    }

    None
}

/// Push a market into `raw` if its id is new and the cap isn't reached.
fn push_unique(
    raw: &mut Vec<Value>,
    seen: &mut std::collections::HashSet<String>,
    m: &Value,
    limit: usize,
) {
    if raw.len() >= limit {
        return;
    }
    let id = json_str(m, &["id", "conditionId", "slug"]).unwrap_or_default();
    if id.is_empty() || seen.insert(id) {
        raw.push(m.clone());
    }
}

/// Parse a raw Gamma market into a `WeatherMarketRow` (None if it isn't a priceable weather market).
fn parse_weather_market_row(market: &Value) -> Option<WeatherMarketRow> {
    let title = json_str(market, &["question", "title", "name"])?;
    if !is_weather_like_market(&title.to_ascii_lowercase()) {
        return None;
    }
    let city = infer_city(&title)?;
    let (market_type, threshold, threshold_upper, unit) = infer_market_type_and_threshold(&title)?;
    let target_date = market_target_date(market)?;
    let market_id = json_str(market, &["id", "conditionId", "slug", "questionID"])
        .unwrap_or_else(|| format!("{city}_{threshold}"));
    let price = market
        .get("lastTradePrice")
        .and_then(value_as_f64)
        .unwrap_or(0.5)
        .clamp(0.0, 1.0);
    Some(WeatherMarketRow {
        target_date,
        market_id,
        market_title: title,
        market_type,
        threshold,
        threshold_upper,
        unit,
        city,
        price,
        outcome: infer_actual_outcome(market),
    })
}

/// The day a temperature bucket is measured/resolved (the day whose high the market is about).
fn market_target_date(market: &Value) -> Option<NaiveDate> {
    ["endDate", "endDateIso", "closedTime", "resolveTime"]
        .iter()
        .find_map(|k| market.get(*k).and_then(|v| v.as_str()))
        .and_then(parse_date)
}

fn fallback_market_point(market: &Value) -> Option<(NaiveDate, f64)> {
    let date = ["endDate", "endDateIso", "closedTime", "resolveTime", "createdAt"]
        .iter()
        .find_map(|k| market.get(*k).and_then(|v| v.as_str()))
        .and_then(parse_date)
        .unwrap_or_else(|| Utc::now().date_naive());

    let price = market
        .get("yesPrice")
        .and_then(value_as_f64)
        .or_else(|| {
            market
                .get("outcomePrices")
                .and_then(parse_outcome_prices)
                .and_then(|prices| prices.first().copied())
        })
        .unwrap_or(0.5);

    Some((date, price))
}

fn parse_outcome_prices(v: &Value) -> Option<Vec<f64>> {
    if let Some(arr) = v.as_array() {
        return Some(arr.iter().filter_map(value_as_f64).collect());
    }

    if let Some(s) = v.as_str() {
        if let Ok(arr) = serde_json::from_str::<Vec<Value>>(s) {
            return Some(arr.iter().filter_map(value_as_f64).collect());
        }
    }

    None
}

fn is_weather_like_market(title: &str) -> bool {
    let keys = [
        "weather",
        "temperature",
        "rain",
        "precip",
        "snow",
        "wind",
        "hurricane",
        "storm",
        "tornado",
        "cyclone",
        "gust",
    ];

    keys.iter().any(|k| title.contains(k))
}

/// Identify the city in a market title (delegates to the shared city registry).
fn infer_city(title: &str) -> Option<String> {
    crate::cities::infer_city(title).map(String::from)
}

/// Parse a market title into (market_type, threshold, threshold_upper, unit).
/// Temperature markets resolve into the bucket shapes the model can price (over the daily HIGH):
///   "N or higher" -> temp_at_least, "N or below" -> temp_at_most,
///   "between A-B" -> temp_bucket(A,B), "be N" -> temp_bucket(N,N).
/// "lowest temperature" markets are skipped (None) — the model forecasts the high only.
fn infer_market_type_and_threshold(
    title: &str,
) -> Option<(String, f64, Option<f64>, Option<String>)> {
    let t = title.to_ascii_lowercase();

    if t.contains("rain") || t.contains("precip") || t.contains("snow") {
        return Some(("precipitation".to_string(), 0.1, None, None));
    }

    if t.contains("temp") || t.contains("temperature") || t.contains('°') {
        // The model forecasts the daily high; it cannot price daily-low markets.
        if t.contains("lowest") || t.contains("coldest") || t.contains("minimum temp") || t.contains("low temp") {
            return None;
        }
        let unit = detect_temp_unit(&t);

        // "between A-B" range bucket. Gated on "between" so a date like "2024-01-15" can't be
        // misread as a range (real Polymarket range markets are always phrased "between A-B").
        if t.contains("between") {
            if let Some((a, b)) = extract_range(&t) {
                return Some(("temp_bucket".to_string(), a, Some(b), Some(unit)));
            }
        }

        let n = extract_first_number(&t)?;

        // Phrase forms are matched as substrings; bare single words use whole-word matching so a
        // city/word like "Dover" ("over") or "thunder" ("under") can't flip the bucket direction.
        let at_least = t.contains("or higher")
            || t.contains("or above")
            || t.contains("or more")
            || t.contains("at least")
            || t.contains("exceed")
            || contains_word(&t, "above")
            || contains_word(&t, "over")
            || contains_word(&t, "greater");
        let at_most = t.contains("or below")
            || t.contains("or lower")
            || t.contains("or less")
            || t.contains("at most")
            || t.contains("less than")
            || contains_word(&t, "below")
            || contains_word(&t, "under");
        if at_least {
            return Some(("temp_at_least".to_string(), n, None, Some(unit)));
        }
        if at_most {
            return Some(("temp_at_most".to_string(), n, None, Some(unit)));
        }
        // "be N" exact integer bucket.
        return Some(("temp_bucket".to_string(), n, Some(n), Some(unit)));
    }

    if t.contains("wind") || t.contains("gust") {
        let n = extract_first_number(&t).unwrap_or(20.0);
        return Some(("wind".to_string(), n, None, None));
    }

    if t.contains("hurricane") || t.contains("storm") || t.contains("tornado") || t.contains("cyclone") {
        return Some(("storm".to_string(), 0.5, None, None));
    }

    None
}

/// "F" or "C" from a temperature title. Reliable on "°F"/"°C"/"fahrenheit"/"celsius"; also
/// catches a bare suffix like "90f". Defaults to "F" (legacy) when no marker is present.
fn detect_temp_unit(t: &str) -> String {
    if t.contains("°f") || t.contains("fahrenheit") {
        return "F".to_string();
    }
    if t.contains("°c") || t.contains("celsius") {
        return "C".to_string();
    }
    // Bare suffix like "90f": the unit letter must come IMMEDIATELY after the number (an optional
    // "°" aside). Do NOT skip spaces/words, or a later word like "cooler" would latch as "C".
    let b = t.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_digit() {
            let mut j = i;
            while j < b.len() && (b[j].is_ascii_digit() || b[j] == b'.') {
                j += 1;
            }
            if j + 1 < b.len() && b[j] == 0xC2 && b[j + 1] == 0xB0 {
                j += 2; // skip a "°"
            }
            if j < b.len() {
                match b[j] | 0x20 {
                    b'f' => return "F".to_string(),
                    b'c' => return "C".to_string(),
                    _ => {}
                }
            }
            i = j.max(i + 1);
        } else {
            i += 1;
        }
    }
    "F".to_string()
}

/// Extract a "A-B" numeric range (e.g. "between 66-67°F" -> (66, 67)). Only a hyphen directly
/// flanked by digits counts, so date words like "on May 20" are not mistaken for a range.
fn extract_range(t: &str) -> Option<(f64, f64)> {
    let b = t.as_bytes();
    for i in 1..b.len() {
        if b[i] == b'-' && b[i - 1].is_ascii_digit() && i + 1 < b.len() && b[i + 1].is_ascii_digit() {
            let mut lo = i;
            while lo > 0 && (b[lo - 1].is_ascii_digit() || b[lo - 1] == b'.') {
                lo -= 1;
            }
            // Absorb a leading '-' so a negative lower bound keeps its sign ("between -5-3").
            if lo > 0 && b[lo - 1] == b'-' && (lo < 2 || !b[lo - 2].is_ascii_digit()) {
                lo -= 1;
            }
            let mut hi = i + 1;
            while hi < b.len() && (b[hi].is_ascii_digit() || b[hi] == b'.') {
                hi += 1;
            }
            let a = t[lo..i].parse::<f64>().ok()?;
            let c = t[i + 1..hi].parse::<f64>().ok()?;
            // Order the bounds so a reversed parse can't become an empty (lo > hi) bucket priced ~0.
            return Some(if a <= c { (a, c) } else { (c, a) });
        }
    }
    None
}

fn extract_first_number(text: &str) -> Option<f64> {
    let chars: Vec<char> = text.chars().collect();
    let mut buf = String::new();
    let mut started = false;

    for (i, &ch) in chars.iter().enumerate() {
        if !started && ch == '-' {
            // Treat as negative sign only if followed immediately by a digit.
            if chars.get(i + 1).map_or(false, |c| c.is_ascii_digit()) {
                buf.push(ch);
            }
        } else if ch.is_ascii_digit() {
            buf.push(ch);
            started = true;
        } else if started && ch == '.' && !buf.contains('.') {
            buf.push(ch);
        } else if started {
            break;
        } else if ch != '-' {
            // Non-digit non-dash: clear any pending '-' sign.
            buf.clear();
        }
    }

    if started {
        buf.parse::<f64>().ok()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use chrono::Datelike;

    use super::*;

    // ── existing tests (preserved) ──────────────────────────────────────────

    #[test]
    fn test_infer_market_signature() {
        assert_eq!(
            infer_market_type_and_threshold("Will NYC temperature exceed 90F?"),
            Some(("temp_at_least".to_string(), 90.0, None, Some("F".to_string())))
        );
        assert_eq!(
            infer_market_type_and_threshold("Will it rain in LA tomorrow?"),
            Some(("precipitation".to_string(), 0.1, None, None))
        );
    }

    #[test]
    fn test_infer_bucket_shapes() {
        // "or higher" -> temp_at_least, degC
        assert_eq!(
            infer_market_type_and_threshold("Will the highest temperature in Manila be 36°C or higher on July 1?"),
            Some(("temp_at_least".to_string(), 36.0, None, Some("C".to_string())))
        );
        // "or below" -> temp_at_most, degC
        assert_eq!(
            infer_market_type_and_threshold("Will the highest temperature in Manila be 26°C or below on July 1?"),
            Some(("temp_at_most".to_string(), 26.0, None, Some("C".to_string())))
        );
        // exact "be N" -> temp_bucket with upper == lower, degC
        assert_eq!(
            infer_market_type_and_threshold("Will the highest temperature in London be 18°C on February 25?"),
            Some(("temp_bucket".to_string(), 18.0, Some(18.0), Some("C".to_string())))
        );
        // "between A-B" -> temp_bucket, degF
        assert_eq!(
            infer_market_type_and_threshold("Will the highest temperature in Seattle be between 50-51°F on May 20?"),
            Some(("temp_bucket".to_string(), 50.0, Some(51.0), Some("F".to_string())))
        );
    }

    #[test]
    fn test_lowest_temperature_skipped() {
        // The model forecasts the daily high; low-temperature markets cannot be priced.
        assert_eq!(
            infer_market_type_and_threshold("Will the lowest temperature in London be 24°C or higher on June 1?"),
            None
        );
        // "coldest" is the same daily-low event.
        assert_eq!(
            infer_market_type_and_threshold("Will the coldest temperature in Denver be 2°C or higher?"),
            None
        );
    }

    #[test]
    fn test_negative_range_not_inverted() {
        // "between -5-3" must keep the negative lower bound and order the bounds (was (5,3) -> 0).
        assert_eq!(
            infer_market_type_and_threshold("Will the highest temperature in London be between -5-3°C on Jan 5?"),
            Some(("temp_bucket".to_string(), -5.0, Some(3.0), Some("C".to_string())))
        );
        assert_eq!(extract_range("between -5-3"), Some((-5.0, 3.0)));
        assert_eq!(extract_range("between 66-67"), Some((66.0, 67.0)));
    }

    #[test]
    fn test_unit_does_not_latch_to_later_word() {
        // Bare suffix only: a later word starting with c/f must not set the unit.
        assert_eq!(detect_temp_unit("be 18 then cooler later"), "F"); // not "C" from "cooler"
        assert_eq!(detect_temp_unit("be 40 for the day"), "F");
        // Real markers and adjacent bare suffixes still work.
        assert_eq!(detect_temp_unit("be 90f"), "F");
        assert_eq!(detect_temp_unit("be 25c"), "C");
        assert_eq!(detect_temp_unit("be 18°c"), "C");
        assert_eq!(detect_temp_unit("between 66-67°f"), "F");
    }

    #[test]
    fn test_directional_word_collision() {
        // "Dover" contains "over"; must not route to temp_at_least — it's an exact bucket.
        assert_eq!(
            infer_market_type_and_threshold("Will the highest temperature in Dover be 70°C on May 1?"),
            Some(("temp_bucket".to_string(), 70.0, Some(70.0), Some("C".to_string())))
        );
    }

    #[test]
    fn test_parse_weather_market_row() {
        let m = serde_json::json!({
            "question": "Will the highest temperature in London be between 66-67°F on May 20?",
            "lastTradePrice": 0.23,
            "endDate": "2026-05-20T12:00:00Z",
            "outcomes": "[\"Yes\", \"No\"]",
            "outcomePrices": "[\"0\", \"1\"]"
        });
        let row = parse_weather_market_row(&m).expect("should parse");
        assert_eq!(row.city, "London");
        assert_eq!(row.market_type, "temp_bucket");
        assert_eq!(row.threshold, 66.0);
        assert_eq!(row.threshold_upper, Some(67.0));
        assert_eq!(row.unit.as_deref(), Some("F"));
        assert!((row.price - 0.23).abs() < 1e-9);
        assert_eq!(row.outcome, Some(0.0)); // outcomePrices ["0","1"] -> NO won
        assert_eq!(row.target_date, NaiveDate::from_ymd_opt(2026, 5, 20).unwrap());

        // Active market (unresolved) -> outcome None.
        let active = serde_json::json!({
            "question": "Will the highest temperature in Tokyo be 30°C on July 1?",
            "lastTradePrice": 0.4, "endDate": "2026-07-01T12:00:00Z"
        });
        assert_eq!(parse_weather_market_row(&active).unwrap().outcome, None);

        // Non-weather -> None.
        assert!(parse_weather_market_row(&serde_json::json!({"question": "Will BTC hit 100k?"})).is_none());
    }

    #[test]
    fn test_city_inference() {
        assert_eq!(infer_city("New York rain market"), Some("NYC".to_string()));
        assert_eq!(infer_city("London temperature"), Some("London".to_string()));
        assert_eq!(infer_city("Unknown City"), None);
    }

    #[test]
    fn test_history_parsing() {
        let val = serde_json::json!({
            "history": [
                {"t": 1735689600, "p": 0.4},
                {"t": 1735776000, "p": 0.42}
            ]
        });

        let points = parse_price_history_points(&val);
        assert_eq!(points.len(), 2);
        assert_eq!(points[0].1, 0.4);
    }

    #[test]
    fn test_outcome_parse() {
        assert_eq!(parse_binary_outcome(&serde_json::json!("YES")), Some(1.0));
        assert_eq!(parse_binary_outcome(&serde_json::json!("no")), Some(0.0));
        assert_eq!(parse_binary_outcome(&serde_json::json!(true)), Some(1.0));
    }

    // ── new tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_is_weather_like_market() {
        assert!(is_weather_like_market("will it rain in chicago"));
        assert!(is_weather_like_market("nyc temperature above 90f"));
        assert!(is_weather_like_market("hurricane season prediction"));
        assert!(is_weather_like_market("will there be a tornado"));
        assert!(!is_weather_like_market("will bitcoin exceed 100k"));
        assert!(!is_weather_like_market("2024 us election winner"));
    }

    #[test]
    fn test_city_inference_la_start_of_title() {
        // Regression: "LA" at start was previously skipped by " la " pattern.
        assert_eq!(infer_city("LA temperature above 90F"), Some("LA".to_string()));
        assert_eq!(infer_city("LA rain this week"), Some("LA".to_string()));
    }

    #[test]
    fn test_city_inference_la_middle_and_end() {
        assert_eq!(infer_city("Will it rain in LA?"), Some("LA".to_string()));
        assert_eq!(infer_city("temperature in LA"), Some("LA".to_string()));
    }

    #[test]
    fn test_city_inference_la_no_false_positives() {
        // "la" embedded in another city name must not match as LA.
        assert_ne!(infer_city("Dallas temperature above 90F"), Some("LA".to_string()));
        assert_ne!(infer_city("Atlanta rain forecast"), Some("LA".to_string()));
    }

    #[test]
    fn test_city_inference_expanded() {
        assert_eq!(infer_city("Seattle rain this week"), Some("Seattle".to_string()));
        assert_eq!(infer_city("Atlanta temperature forecast"), Some("Atlanta".to_string()));
        assert_eq!(infer_city("Houston hurricane risk"), Some("Houston".to_string()));
        assert_eq!(infer_city("Phoenix heat advisory"), Some("Phoenix".to_string()));
        assert_eq!(infer_city("Portland snow expected"), Some("Portland".to_string()));
        assert_eq!(infer_city("San Francisco fog prediction"), Some("SF".to_string()));
        assert_eq!(infer_city("Toronto temperature below freezing"), Some("Toronto".to_string()));
        assert_eq!(infer_city("Tokyo typhoon season"), Some("Tokyo".to_string()));
        assert_eq!(infer_city("Sydney rainfall record"), Some("Sydney".to_string()));
    }

    #[test]
    fn test_infer_market_type_wind() {
        assert_eq!(
            infer_market_type_and_threshold("Will wind exceed 30mph in Chicago?"),
            Some(("wind".to_string(), 30.0, None, None))
        );
        assert_eq!(
            infer_market_type_and_threshold("Will gusts reach 50mph?"),
            Some(("wind".to_string(), 50.0, None, None))
        );
        // No number → default 20mph
        assert_eq!(
            infer_market_type_and_threshold("High wind warning issued"),
            Some(("wind".to_string(), 20.0, None, None))
        );
    }

    #[test]
    fn test_infer_market_type_storm() {
        assert_eq!(
            infer_market_type_and_threshold("Will hurricane Ian make landfall?"),
            Some(("storm".to_string(), 0.5, None, None))
        );
        assert_eq!(
            infer_market_type_and_threshold("Will a tornado strike Kansas City?"),
            Some(("storm".to_string(), 0.5, None, None))
        );
        assert_eq!(
            infer_market_type_and_threshold("Major storm expected this weekend"),
            Some(("storm".to_string(), 0.5, None, None))
        );
    }

    #[test]
    fn test_infer_market_type_negative_temp() {
        // "below N" routes to temp_at_most; negative thresholds are extracted (not the date hyphen).
        assert_eq!(
            infer_market_type_and_threshold("Will temperature drop below -10°C?"),
            Some(("temp_at_most".to_string(), -10.0, None, Some("C".to_string())))
        );
        assert_eq!(
            infer_market_type_and_threshold("Temperature below -32F tonight?"),
            Some(("temp_at_most".to_string(), -32.0, None, Some("F".to_string())))
        );
    }

    #[test]
    fn test_extract_first_number() {
        assert_eq!(extract_first_number("exceed 90F"), Some(90.0));
        assert_eq!(extract_first_number("below -10 degrees"), Some(-10.0));
        assert_eq!(extract_first_number("1.5 inches of rain"), Some(1.5));
        assert_eq!(extract_first_number("-3.5 below zero"), Some(-3.5));
        assert_eq!(extract_first_number("no numbers here"), None);
        // '-' not followed by digit → not a negative sign
        assert_eq!(extract_first_number("wind-driven rain"), None);
        // First number only
        assert_eq!(extract_first_number("10 to 20 mph"), Some(10.0));
    }

    #[test]
    fn test_infer_actual_outcome_direct_fields() {
        assert_eq!(
            infer_actual_outcome(&serde_json::json!({"resolvedOutcome": "Yes"})),
            Some(1.0)
        );
        assert_eq!(
            infer_actual_outcome(&serde_json::json!({"resolution": "No"})),
            Some(0.0)
        );
        assert_eq!(
            infer_actual_outcome(&serde_json::json!({"winner": true})),
            Some(1.0)
        );
        assert_eq!(
            infer_actual_outcome(&serde_json::json!({"outcome": 1.0})),
            Some(1.0)
        );
        assert_eq!(
            infer_actual_outcome(&serde_json::json!({"result": "0"})),
            Some(0.0)
        );
    }

    #[test]
    fn test_infer_actual_outcome_token_winner() {
        let m = serde_json::json!({
            "tokens": [
                {"outcome": "Yes", "winner": true},
                {"outcome": "No", "winner": false}
            ]
        });
        assert_eq!(infer_actual_outcome(&m), Some(1.0));

        let m = serde_json::json!({
            "tokens": [
                {"outcome": "Yes", "isWinner": false},
                {"outcome": "No", "isWinner": true}
            ]
        });
        assert_eq!(infer_actual_outcome(&m), Some(0.0));
    }

    #[test]
    fn test_infer_actual_outcome_from_gamma_outcome_prices() {
        // Real Gamma schema: parallel string-encoded arrays, winner price "1".
        let yes_won = serde_json::json!({
            "outcomes": "[\"Yes\", \"No\"]",
            "outcomePrices": "[\"1\", \"0\"]"
        });
        assert_eq!(infer_actual_outcome(&yes_won), Some(1.0));

        let no_won = serde_json::json!({
            "outcomes": "[\"Yes\", \"No\"]",
            "outcomePrices": "[\"0\", \"1\"]"
        });
        assert_eq!(infer_actual_outcome(&no_won), Some(0.0));

        // Unresolved (fractional prices) must NOT be read as resolved.
        let open = serde_json::json!({
            "outcomes": "[\"Yes\", \"No\"]",
            "outcomePrices": "[\"0.55\", \"0.45\"]"
        });
        assert_eq!(infer_actual_outcome(&open), None);
    }

    #[test]
    fn test_infer_actual_outcome_none_for_unresolved() {
        let m = serde_json::json!({"question": "Will it rain in NYC tomorrow?"});
        assert_eq!(infer_actual_outcome(&m), None);

        // Tokens present but none marked as winner
        let m = serde_json::json!({
            "tokens": [
                {"outcome": "Yes", "winner": false},
                {"outcome": "No", "winner": false}
            ]
        });
        assert_eq!(infer_actual_outcome(&m), None);
    }

    #[test]
    fn test_infer_yes_token_id_from_tokens_array() {
        let m = serde_json::json!({
            "tokens": [
                {"outcome": "No", "token_id": "no_token"},
                {"outcome": "Yes", "token_id": "yes_token"}
            ]
        });
        assert_eq!(infer_yes_token_id(&m), Some("yes_token".to_string()));
    }

    #[test]
    fn test_infer_yes_token_id_from_clob_json_string() {
        let m = serde_json::json!({
            "clobTokenIds": "[\"tok_yes\", \"tok_no\"]"
        });
        assert_eq!(infer_yes_token_id(&m), Some("tok_yes".to_string()));
    }

    #[test]
    fn test_infer_yes_token_id_from_clob_array() {
        let m = serde_json::json!({
            "clobTokenIds": ["tok_yes", "tok_no"]
        });
        assert_eq!(infer_yes_token_id(&m), Some("tok_yes".to_string()));
    }

    #[test]
    fn test_infer_yes_token_id_none() {
        let m = serde_json::json!({"question": "market with no tokens"});
        assert_eq!(infer_yes_token_id(&m), None);
    }

    #[test]
    fn test_value_to_date_millisecond_timestamp() {
        // 1735689600000 ms → 2025-01-01
        let v = serde_json::json!(1_735_689_600_000u64);
        let result = value_to_date(&v).expect("should parse ms timestamp");
        assert_eq!(result.year(), 2025);
        assert_eq!(result.month(), 1);
    }

    #[test]
    fn test_value_to_date_second_timestamp() {
        // 1735689600 s → 2025-01-01
        let v = serde_json::json!(1_735_689_600u64);
        let result = value_to_date(&v).expect("should parse second timestamp");
        assert_eq!(result.year(), 2025);
        assert_eq!(result.month(), 1);
    }

    #[test]
    fn test_value_to_date_string_formats() {
        let v = serde_json::json!("2025-03-15");
        assert_eq!(value_to_date(&v), NaiveDate::from_ymd_opt(2025, 3, 15));

        let v = serde_json::json!("2025-03-15T12:00:00Z");
        assert_eq!(value_to_date(&v), NaiveDate::from_ymd_opt(2025, 3, 15));
    }

    #[test]
    fn test_extract_rows_direct_array() {
        let v = serde_json::json!([{"id": "1"}, {"id": "2"}]);
        let rows = extract_rows(&v);
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_extract_rows_data_wrapper() {
        let v = serde_json::json!({"data": [{"id": "1"}, {"id": "2"}, {"id": "3"}]});
        let rows = extract_rows(&v);
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn test_extract_rows_empty_on_unknown_shape() {
        let v = serde_json::json!({"markets": [{"id": "1"}]});
        let rows = extract_rows(&v);
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn test_parse_price_history_flat_array() {
        let val = serde_json::json!([
            {"t": 1735689600, "p": 0.55},
            {"t": 1735776000, "p": 0.60}
        ]);
        let points = parse_price_history_points(&val);
        assert_eq!(points.len(), 2);
        assert_eq!(points[1].1, 0.60);
    }

    #[test]
    fn test_parse_price_history_alt_field_names() {
        let val = serde_json::json!({
            "history": [
                {"timestamp": 1735689600, "price": 0.3},
                {"time": 1735776000, "close": 0.35}
            ]
        });
        let points = parse_price_history_points(&val);
        assert_eq!(points.len(), 2);
    }

    #[test]
    fn test_parse_outcome_prices_array() {
        let v = serde_json::json!([0.65, 0.35]);
        let prices = parse_outcome_prices(&v).unwrap();
        assert_eq!(prices, vec![0.65, 0.35]);
    }

    #[test]
    fn test_parse_outcome_prices_json_string() {
        let v = serde_json::json!("[0.7, 0.3]");
        let prices = parse_outcome_prices(&v).unwrap();
        assert_eq!(prices, vec![0.7, 0.3]);
    }

    #[test]
    fn test_fallback_market_point_uses_end_date() {
        let m = serde_json::json!({"endDate": "2025-06-01", "yesPrice": 0.75});
        let (date, price) = fallback_market_point(&m).unwrap();
        assert_eq!(date, NaiveDate::from_ymd_opt(2025, 6, 1).unwrap());
        assert!((price - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fallback_market_point_default_price() {
        // No yesPrice → fallback to 0.5
        let m = serde_json::json!({"endDate": "2025-06-01"});
        let (_, price) = fallback_market_point(&m).unwrap();
        assert!((price - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parse_binary_outcome_numeric() {
        assert_eq!(parse_binary_outcome(&serde_json::json!(1.0)), Some(1.0));
        assert_eq!(parse_binary_outcome(&serde_json::json!(0.0)), Some(0.0));
        assert_eq!(parse_binary_outcome(&serde_json::json!("1")), Some(1.0));
        assert_eq!(parse_binary_outcome(&serde_json::json!("0")), Some(0.0));
        // Ambiguous value → None
        assert_eq!(parse_binary_outcome(&serde_json::json!(0.5)), None);
    }

    #[test]
    fn test_contains_word() {
        assert!(contains_word("la rain tomorrow", "la"));
        assert!(contains_word("rain in la", "la"));
        assert!(contains_word("la", "la"));
        // False positives that must NOT match
        assert!(!contains_word("dallas temperature", "la"));
        assert!(!contains_word("atlanta rain", "la"));
        assert!(!contains_word("philadelphia storm", "la"));
        // Regression: must not panic on multi-byte chars (the '°' degree sign).
        assert!(!contains_word("highest temperature in panama city be 39°c or higher", "la"));
        assert!(contains_word("will la reach 39°c", "la"));
    }
}
