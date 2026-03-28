use std::collections::HashMap;

use chrono::{DateTime, NaiveDate, Utc};
use serde_json::Value;

use crate::types::SimulatedMarket;
use crate::utils::parse_date;

const DEFAULT_GAMMA_BASE_URL: &str = "https://gamma-api.polymarket.com";
const DEFAULT_CLOB_BASE_URL: &str = "https://clob.polymarket.com";

#[derive(Debug, thiserror::Error)]
pub enum PolymarketHistoryError {
    #[error("request failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
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

            let Some((market_type, threshold)) = infer_market_type_and_threshold(&title) else {
                continue;
            };

            let Some(actual_outcome) = infer_actual_outcome(&market) else {
                continue;
            };

            let market_id = json_str(&market, &["id", "conditionId", "slug", "questionID"])
                .unwrap_or_else(|| format!("{}_{}", city, threshold));

            let token_id = infer_yes_token_id(&market);
            let mut history = if let Some(id) = token_id.as_deref() {
                match self.fetch_price_history(id, start_date, end_date).await {
                    Ok(h) => h,
                    Err(e) => {
                        eprintln!(
                            "warning: failed to fetch price history for market '{}': {}",
                            title, e
                        );
                        Vec::new()
                    }
                }
            } else {
                Vec::new()
            };

            if history.is_empty() {
                if let Some((d, p)) = fallback_market_point(&market) {
                    eprintln!(
                        "warning: using fabricated fallback price {:.2} for market '{}' (no price history available)",
                        p, title
                    );
                    history.push((d, p));
                }
            }

            for (date, price) in history {
                if let Some(start) = start_date {
                    if date < start {
                        continue;
                    }
                }
                if let Some(end) = end_date {
                    if date > end {
                        continue;
                    }
                }

                out.push(SimulatedMarket {
                    date,
                    market_id: market_id.clone(),
                    market_title: title.clone(),
                    market_type: market_type.clone(),
                    threshold,
                    market_price: price.clamp(0.0, 1.0),
                    actual_outcome,
                    city: city.clone(),
                });
            }

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

        let mut all = Vec::new();
        let mut offset = 0usize;
        let page_size = limit.min(200).max(1);

        while all.len() < limit {
            let query = [
                ("limit", page_size.to_string()),
                ("offset", offset.to_string()),
                ("order", "id".to_string()),
                ("ascending", "false".to_string()),
            ];

            let resp = self.client.get(&url).query(&query).send().await?;
            let val = resp.json::<Value>().await?;
            let rows = extract_rows(&val);
            let count = rows.len();
            if rows.is_empty() {
                break;
            }

            offset += count;
            all.extend(rows);

            if count < page_size {
                break;
            }
        }

        all.truncate(limit);
        Ok(all)
    }

    async fn fetch_price_history(
        &self,
        token_id: &str,
        start_date: Option<NaiveDate>,
        end_date: Option<NaiveDate>,
    ) -> Result<Vec<(NaiveDate, f64)>, reqwest::Error> {
        let start_ts = start_date
            .and_then(|d| d.and_hms_opt(0, 0, 0))
            .map(|dt| dt.and_utc().timestamp())
            .unwrap_or(0);
        let end_ts = end_date
            .and_then(|d| d.and_hms_opt(23, 59, 59))
            .map(|dt| dt.and_utc().timestamp())
            .unwrap_or_else(|| Utc::now().timestamp());

        let candidates: [(&str, Vec<(&str, String)>); 3] = [
            (
                "prices-history",
                vec![
                    ("market", token_id.to_string()),
                    ("interval", "1d".to_string()),
                    ("startTs", start_ts.to_string()),
                    ("endTs", end_ts.to_string()),
                ],
            ),
            (
                "prices-history",
                vec![
                    ("token_id", token_id.to_string()),
                    ("interval", "1d".to_string()),
                    ("startTs", start_ts.to_string()),
                    ("endTs", end_ts.to_string()),
                ],
            ),
            (
                "price-history",
                vec![
                    ("market", token_id.to_string()),
                    ("interval", "1d".to_string()),
                    ("startTs", start_ts.to_string()),
                    ("endTs", end_ts.to_string()),
                ],
            ),
        ];

        for (path, query) in candidates {
            let url = format!("{}/{}", self.clob_base_url, path);
            let Ok(resp) = self.client.get(&url).query(&query).send().await else {
                continue;
            };
            let Ok(val) = resp.json::<Value>().await else {
                continue;
            };

            let points = parse_price_history_points(&val);
            if !points.is_empty() {
                let mut by_day: HashMap<NaiveDate, f64> = HashMap::new();
                for (d, p) in points {
                    by_day.insert(d, p);
                }

                let mut out = by_day.into_iter().collect::<Vec<_>>();
                out.sort_by_key(|(d, _)| *d);
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

fn infer_city(title: &str) -> Option<String> {
    let t = title.to_ascii_lowercase();

    let mapping: &[(&str, &str)] = &[
        ("new york", "NYC"),
        ("nyc", "NYC"),
        ("los angeles", "LA"),
        ("l.a.", "LA"),
        ("london", "London"),
        ("chicago", "Chicago"),
        ("dallas", "Dallas"),
        ("denver", "Denver"),
        ("miami", "Miami"),
        ("boston", "Boston"),
        ("seattle", "Seattle"),
        ("atlanta", "Atlanta"),
        ("houston", "Houston"),
        ("phoenix", "Phoenix"),
        ("portland", "Portland"),
        ("san francisco", "SF"),
        ("toronto", "Toronto"),
        ("tokyo", "Tokyo"),
        ("sydney", "Sydney"),
    ];

    for (k, city) in mapping {
        if t.contains(k) {
            return Some((*city).to_string());
        }
    }

    // "la" is ambiguous — match only as a whole word to avoid false positives
    // in city names like "dallas" or "atlanta".
    if contains_word(&t, "la") {
        return Some("LA".to_string());
    }

    None
}

/// Returns true if `word` appears in `text` as a standalone word
/// (not surrounded by ASCII alphabetic characters).
fn contains_word(text: &str, word: &str) -> bool {
    let bytes = text.as_bytes();
    let wlen = word.len();
    let tlen = bytes.len();
    let mut pos = 0;
    while pos + wlen <= tlen {
        if &text[pos..pos + wlen] == word {
            let before_ok = pos == 0 || !bytes[pos - 1].is_ascii_alphabetic();
            let after_ok = pos + wlen == tlen || !bytes[pos + wlen].is_ascii_alphabetic();
            if before_ok && after_ok {
                return true;
            }
        }
        pos += 1;
    }
    false
}

fn infer_market_type_and_threshold(title: &str) -> Option<(String, f64)> {
    let t = title.to_ascii_lowercase();

    if t.contains("rain") || t.contains("precip") || t.contains("snow") {
        return Some(("precipitation".to_string(), 0.1));
    }

    if t.contains("temp") || t.contains("temperature") {
        if let Some(number) = extract_first_number(&t) {
            return Some(("temperature".to_string(), number));
        }
        return Some(("temperature".to_string(), 70.0));
    }

    if t.contains("wind") || t.contains("gust") {
        if let Some(number) = extract_first_number(&t) {
            return Some(("wind".to_string(), number));
        }
        return Some(("wind".to_string(), 20.0));
    }

    if t.contains("hurricane") || t.contains("storm") || t.contains("tornado") || t.contains("cyclone") {
        return Some(("storm".to_string(), 0.5));
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
            Some(("temperature".to_string(), 90.0))
        );
        assert_eq!(
            infer_market_type_and_threshold("Will it rain in LA tomorrow?"),
            Some(("precipitation".to_string(), 0.1))
        );
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
            Some(("wind".to_string(), 30.0))
        );
        assert_eq!(
            infer_market_type_and_threshold("Will gusts reach 50mph?"),
            Some(("wind".to_string(), 50.0))
        );
        // No number → default 20mph
        assert_eq!(
            infer_market_type_and_threshold("High wind warning issued"),
            Some(("wind".to_string(), 20.0))
        );
    }

    #[test]
    fn test_infer_market_type_storm() {
        assert_eq!(
            infer_market_type_and_threshold("Will hurricane Ian make landfall?"),
            Some(("storm".to_string(), 0.5))
        );
        assert_eq!(
            infer_market_type_and_threshold("Will a tornado strike Kansas City?"),
            Some(("storm".to_string(), 0.5))
        );
        assert_eq!(
            infer_market_type_and_threshold("Major storm expected this weekend"),
            Some(("storm".to_string(), 0.5))
        );
    }

    #[test]
    fn test_infer_market_type_negative_temp() {
        // Regression: negative temperatures were previously unextracted → 70.0 fallback.
        assert_eq!(
            infer_market_type_and_threshold("Will temperature drop below -10°C?"),
            Some(("temperature".to_string(), -10.0))
        );
        assert_eq!(
            infer_market_type_and_threshold("Temperature below -32F tonight?"),
            Some(("temperature".to_string(), -32.0))
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
    }
}
