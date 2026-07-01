//! Kalshi weather-market downloader — the venue-B analog of [`super::polymarket_history`].
//!
//! Fetches Kalshi temperature/weather markets and maps them into the same [`WeatherMarketRow`] the
//! capture daemon already consumes, so both venues flow through one pipeline. Defaults to Kalshi's
//! DEMO/paper environment (`config::kalshi_base_url`). Kalshi's trade API has no anonymous access, so
//! every request is signed with an RSA-PSS(SHA-256) signature over `timestamp + method + path`;
//! without credentials the downloader reports `is_available() == false` and is silently skipped, the
//! same graceful-degradation contract the API-keyed weather fetchers use.

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

#[derive(Debug, thiserror::Error)]
pub enum KalshiHistoryError {
    #[error("request failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error("invalid Kalshi private key (expected an RSA PKCS#8 or PKCS#1 PEM)")]
    BadKey,
}

/// Signs Kalshi requests. Present only when both a key id and a parseable private key are configured.
struct KalshiAuth {
    key_id: String,
    signing_key: SigningKey<Sha256>,
}

impl KalshiAuth {
    fn from_env() -> Option<Self> {
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
    fn headers(&self, method: &str, path: &str) -> Vec<(&'static str, String)> {
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
            base_url: config::kalshi_base_url(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(20))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            auth: KalshiAuth::from_env(),
        }
    }

    /// Whether Kalshi credentials are configured. The capture daemon checks this before fetching so
    /// an unconfigured Kalshi never errors the run — it's simply absent, like an unset weather API key.
    pub fn is_available(&self) -> bool {
        self.auth.is_some()
    }

    /// Weather markets for the requested side: `active` ⇒ open/unresolved, else settled/resolved.
    /// Returns an empty vec (never an error) when credentials aren't configured.
    pub async fn download_weather_markets(
        &self,
        active: bool,
        limit: usize,
    ) -> Result<Vec<WeatherMarketRow>, KalshiHistoryError> {
        let status = if active { "open" } else { "settled" };
        let raw = self.fetch_markets(status, limit).await?;

        let mut out: Vec<WeatherMarketRow> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for m in &raw {
            if let Some(row) = parse_kalshi_market(m) {
                if active == row.outcome.is_none() && seen.insert(row.market_id.clone()) {
                    out.push(row);
                }
            }
        }
        Ok(out)
    }

    /// Cursor-paginate `GET /markets?status=<status>` up to `limit` raw markets.
    async fn fetch_markets(
        &self,
        status: &str,
        limit: usize,
    ) -> Result<Vec<Value>, KalshiHistoryError> {
        let Some(auth) = &self.auth else {
            return Ok(Vec::new()); // no creds → silently absent
        };
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
                ("status", status.to_string()),
            ];
            if let Some(c) = &cursor {
                query.push(("cursor", c.clone()));
            }

            let mut req = self.client.get(&url).query(&query);
            for (k, v) in auth.headers("GET", &path) {
                req = req.header(k, v);
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
    })
}

/// Bucket shape from Kalshi's structured strikes (US temperature markets are in °F). None when the
/// market has no usable strikes, so the caller can fall back to the shared title parser.
fn strike_shape(m: &Value) -> Option<(String, f64, Option<f64>, Option<String>)> {
    let floor = m.get("floor_strike").and_then(value_as_f64);
    let cap = m.get("cap_strike").and_then(value_as_f64);
    let unit = Some("F".to_string());
    match json_str(m, &["strike_type"]).as_deref() {
        Some("greater") | Some("greater_or_equal") => {
            Some(("temp_at_least".to_string(), floor?, None, unit))
        }
        Some("less") | Some("less_or_equal") => {
            Some(("temp_at_most".to_string(), cap.or(floor)?, None, unit))
        }
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

/// Realized outcome from Kalshi's `result` (yes/no), or None while unresolved / voided.
fn kalshi_outcome(m: &Value) -> Option<f64> {
    match json_str(m, &["result"]).as_deref() {
        Some("yes") => Some(1.0),
        Some("no") => Some(0.0),
        _ => None,
    }
}

/// The day the market resolves (≈ the target day whose high it's about).
fn kalshi_target_date(m: &Value) -> Option<NaiveDate> {
    ["close_time", "expiration_time", "expected_expiration_time"]
        .iter()
        .find_map(|k| m.get(*k).and_then(|v| v.as_str()))
        .and_then(parse_date)
}

#[cfg(test)]
mod tests {
    use super::*;

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
            r#"{"ticker":"KXHIGHCHI-25JUN30-B71","title":"Highest temperature in Chicago",
                "yes_sub_title":"71° to 72°","strike_type":"between","floor_strike":71,"cap_strike":72,
                "last_price":34,"result":"","close_time":"2026-06-30T23:00:00Z"}"#,
        ))
        .expect("should parse");
        assert_eq!(row.source, "kalshi");
        assert_eq!(row.city, "Chicago");
        assert_eq!(row.market_type, "temp_bucket");
        assert_eq!(row.threshold, 71.0);
        assert_eq!(row.threshold_upper, Some(72.0));
        assert_eq!(row.unit.as_deref(), Some("F"));
        assert!((row.price - 0.34).abs() < 1e-9, "cents → 0..1");
        assert_eq!(row.outcome, None);
        assert_eq!(
            row.target_date,
            NaiveDate::from_ymd_opt(2026, 6, 30).unwrap()
        );
    }

    #[test]
    fn parses_greater_settled_yes() {
        let row = parse_kalshi_market(&market(
            r#"{"ticker":"KXHIGHMIA-25JUN29-T90","title":"High temperature in Miami",
                "yes_sub_title":"90° or above","strike_type":"greater","floor_strike":90,
                "yes_bid":0,"yes_ask":0,"result":"yes","close_time":"2026-06-29T23:00:00Z"}"#,
        ))
        .expect("should parse");
        assert_eq!(row.market_type, "temp_at_least");
        assert_eq!(row.threshold, 90.0);
        assert_eq!(row.outcome, Some(1.0));
        assert!(
            (row.price - 0.50).abs() < 1e-9,
            "no trade / no book → 0.50 fallback"
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
