use std::collections::HashMap;

use chrono::NaiveDate;
use serde_json::Value;

use crate::types::WeatherRecord;

pub const BASE_URL: &str = "https://archive-api.open-meteo.com/v1/archive";
/// Archived *forecasts* (what was predicted at the time), for backtesting forecast skill — distinct
/// from the reanalysis archive (observed truth) above.
pub const FORECAST_BASE_URL: &str = "https://historical-forecast-api.open-meteo.com/v1/forecast";
/// Live forecast (future dates), for pricing active markets at real trading lead.
pub const FORECAST_LIVE_BASE_URL: &str = "https://api.open-meteo.com/v1/forecast";
/// The equal-weight blend the station error model (`stations.rs`) is fitted against. Measured
/// 10–40% lower daily-high error than any single member (ECMWF-heavy weights did WORSE than equal
/// weights — ifs025 is the weakest member at station scale). Changing this list invalidates the
/// fitted sigma/bias tables.
pub const BLEND_MODELS: &[&str] = &[
    "ecmwf_ifs025",
    "gfs_seamless",
    "ukmo_seamless",
    "icon_seamless",
];
/// AI weather models logged ALONGSIDE the blend, per capture, so accrued captures can show whether
/// any beats the blend at station scale BEFORE it earns a slot in `BLEND_MODELS` (see the warning
/// above: changing the blend invalidates the fitted sigma/bias tables). Queried one model per
/// request so a renamed/unavailable model degrades to no data without failing the others.
///
/// Each logical model is a CANDIDATE LIST of (endpoint, model-string) pairs tried in order until
/// one returns data — the July 19 capture proved the sandbox-unverifiable single guesses wrong
/// (`ecmwf_aifs025` is the stale pre-operational AIFS name; the current registry says
/// `ecmwf_aifs025_single`), and family endpoints (`/v1/ecmwf`, `/v1/gfs`) are covered in case the
/// generic `/v1/forecast` doesn't route a model. Order is stable: index 0 = ECMWF AIFS,
/// 1 = NOAA AIGFS.
///
/// Slot 1 was GraphCast (`gfs_graphcast025`) until 2026-07-22: it logged null on every capture
/// because NCEP DISCONTINUED the experimental GraphCastGFS/EAGLE runs on 2025-12-17/18 — before
/// this logging even existed — and Open-Meteo's `gfs_graphcast025` alias still parses but serves
/// no data. NOAA's operational replacement is AIGFS (live on Open-Meteo as `ncep_aigfs025` since
/// 2026-01-07, per open-meteo/open-meteo#1633 / PR #1655), so slot 1 logs that instead.
pub const AI_LOG_MODELS: &[&str] = &["ecmwf_aifs025_single", "ncep_aigfs025"];
pub const AI_LOG_CANDIDATES: &[&[(&str, &str)]] = &[
    &[
        ("forecast", "ecmwf_aifs025_single"),
        ("ecmwf", "ecmwf_aifs025_single"),
        ("forecast", "ecmwf_aifs025"),
    ],
    &[("forecast", "ncep_aigfs025"), ("gfs", "ncep_aigfs025")],
];
/// Ensemble forecast endpoint (separate host, like air quality). Serves per-MEMBER series
/// (`temperature_2m`, `temperature_2m_member01`, …) instead of one deterministic value.
pub const ENSEMBLE_BASE_URL: &str = "https://ensemble-api.open-meteo.com/v1/ensemble";
/// Ensemble systems whose member SPREAD is logged per capture: the day-specific uncertainty
/// candidate. The pricing σ is a per-(city, lead) constant fitted seasonally, so it prices a
/// locked-in ridge day and a frontal-timing coin-flip day with the same width; member spread is
/// the standard day-of-forecast signal for that difference (spread–skill relationship). Log-only
/// until a spread→σ mapping is fitted walk-forward and beats the constant-σ tables on accrued
/// captures — same gating as the AI models above, and same reason: σ changes invalidate nothing
/// here because nothing in the pricing path reads these fields yet.
///
/// Candidate lists per logical system (0 = ECMWF ENS ~51 members, 1 = NOAA GEFS ~31), tried in
/// order, because the exact registry names could NOT be verified from the sandbox (its proxy
/// blocks the ensemble host) — the July 19 lesson says never ship a single unverified guess.
/// The daemon prints which candidate served each system, or that all failed, once per run.
pub const ENSEMBLE_LOG_MODELS: &[&str] = &["ecmwf_ifs025", "gfs025"];
pub const ENSEMBLE_LOG_CANDIDATES: &[&[&str]] = &[
    &["ecmwf_ifs025", "ecmwf_ifs04"],
    &["gfs025", "gfs05", "gfs_seamless"],
];
/// A "spread" from fewer members than this is noise (or a deterministic model that slipped
/// through a candidate list) — treat the date as missing rather than log a fake-tight σ.
pub const ENSEMBLE_MIN_MEMBERS: usize = 8;
/// Air-quality forecast endpoint (separate host from the weather API). Used to log forecast smoke
/// (PM2.5 / US AQI) for market target days: heavy wildfire smoke measurably suppresses daily highs
/// and the blend models only partly price the effect.
pub const AIR_QUALITY_BASE_URL: &str = "https://air-quality-api.open-meteo.com/v1/air-quality";
/// Per-UTC-date (max PM2.5 µg/m³, max US AQI) — the shape `fetch_air_quality_day_max` returns.
pub type AirQualityByDay = HashMap<NaiveDate, (Option<f64>, Option<f64>)>;
/// Per-date (member-mean daily high °C, std-dev °C) across ensemble members' daily maxes, plus
/// the model that served it (`None` = every candidate failed) — the shape
/// `fetch_ensemble_stats_tagged` returns. The SPREAD is the σ input (`backtesting::spread_sigma`);
/// the MEAN is log-only until it beats the blend on accrued captures (the μ gate).
pub type TaggedEnsembleStats = (HashMap<NaiveDate, (f64, f64)>, Option<String>);
/// Daily-max series plus the `(endpoint, model)` candidate that served it (`None` = all failed) —
/// the shape `fetch_forecast_max_live_candidates_tagged` returns.
pub type TaggedDailyMax = (Vec<(NaiveDate, f64)>, Option<(String, String)>);

#[derive(Clone)]
pub struct OpenMeteoFetcher {
    client: reqwest::blocking::Client,
    locations: HashMap<String, (f64, f64, String)>,
}

impl OpenMeteoFetcher {
    pub fn new() -> Self {
        Self {
            client: crate::data_pipeline::build_client(30),
            locations: default_locations(),
        }
    }

    /// Archived daily-high FORECASTS (degC) — what was predicted for past dates. For backtesting
    /// forecast skill against resolved markets. Empty vec on any failure (degradation by design).
    pub fn fetch_forecast_max(
        &self,
        latitude: f64,
        longitude: f64,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Vec<(NaiveDate, f64)> {
        self.fetch_forecast_max_at(FORECAST_BASE_URL, latitude, longitude, start_date, end_date)
    }

    /// LIVE daily-high forecasts (degC) for near-future dates — used to price active markets at the
    /// real trading lead (no leakage possible; the outcome hasn't happened yet).
    pub fn fetch_forecast_max_live(
        &self,
        latitude: f64,
        longitude: f64,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Vec<(NaiveDate, f64)> {
        self.fetch_forecast_max_at(
            FORECAST_LIVE_BASE_URL,
            latitude,
            longitude,
            start_date,
            end_date,
        )
    }

    /// LIVE hourly temperature forecast (°C) with UTC timestamps, one series PER MODEL of the
    /// equal-weight blend the station error tables were fitted against (`BLEND_MODELS`). The
    /// station nowcast maxes each series over a local-day window and averages the maxes. Models
    /// with no data are dropped; empty vec on total failure.
    pub fn fetch_forecast_hourly_models_utc(
        &self,
        latitude: f64,
        longitude: f64,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Vec<Vec<(chrono::NaiveDateTime, f64)>> {
        let query = [
            ("latitude", latitude.to_string()),
            ("longitude", longitude.to_string()),
            ("start_date", start_date.to_string()),
            ("end_date", end_date.to_string()),
            ("hourly", "temperature_2m".to_string()),
            ("models", BLEND_MODELS.join(",")),
            ("timezone", "UTC".to_string()),
        ];
        let Ok(resp) = self.client.get(FORECAST_LIVE_BASE_URL).query(&query).send() else {
            return Vec::new();
        };
        let Ok(json) = resp.json::<Value>() else {
            return Vec::new();
        };
        let hourly = json.get("hourly").cloned().unwrap_or(Value::Null);
        let times = hourly
            .get("time")
            .and_then(|x| x.as_array())
            .cloned()
            .unwrap_or_default();
        BLEND_MODELS
            .iter()
            .filter_map(|m| {
                let temps = as_opt_f64_vec(hourly.get(format!("temperature_2m_{m}").as_str()));
                let series: Vec<(chrono::NaiveDateTime, f64)> = times
                    .iter()
                    .enumerate()
                    .filter_map(|(i, tv)| {
                        let ts = tv.as_str()?;
                        let t = chrono::NaiveDateTime::parse_from_str(ts, "%Y-%m-%dT%H:%M").ok()?;
                        Some((t, temps.get(i).copied().flatten()?))
                    })
                    .collect();
                (!series.is_empty()).then_some(series)
            })
            .collect()
    }

    /// LIVE daily-high forecast (degC) from a single named model at the given API endpoint
    /// ("forecast", "ecmwf", "gfs", …), for logging alongside the blend. Empty vec on any failure
    /// — including an unknown model name, which the API rejects for the whole request (why
    /// callers pass one model at a time).
    pub fn fetch_forecast_max_live_model(
        &self,
        latitude: f64,
        longitude: f64,
        start_date: NaiveDate,
        end_date: NaiveDate,
        endpoint: &str,
        model: &str,
    ) -> Vec<(NaiveDate, f64)> {
        let url = format!("https://api.open-meteo.com/v1/{endpoint}");
        let query = [
            ("latitude", latitude.to_string()),
            ("longitude", longitude.to_string()),
            ("start_date", start_date.to_string()),
            ("end_date", end_date.to_string()),
            ("daily", "temperature_2m_max".to_string()),
            ("models", model.to_string()),
            ("timezone", "auto".to_string()),
        ];
        let Ok(resp) = self.client.get(&url).query(&query).send() else {
            return Vec::new();
        };
        let Ok(json) = resp.json::<Value>() else {
            return Vec::new();
        };
        parse_daily_max(&json, Some(model))
    }

    /// First non-empty result across a candidate list of (endpoint, model) pairs, with a final
    /// hourly-series fallback (collapsed to per-local-date maxima client-side) in case a model
    /// serves hourly but not daily aggregation. Empty when every candidate fails.
    pub fn fetch_forecast_max_live_candidates(
        &self,
        latitude: f64,
        longitude: f64,
        start_date: NaiveDate,
        end_date: NaiveDate,
        candidates: &[(&str, &str)],
    ) -> Vec<(NaiveDate, f64)> {
        self.fetch_forecast_max_live_candidates_tagged(
            latitude, longitude, start_date, end_date, candidates,
        )
        .0
    }

    /// Like [`Self::fetch_forecast_max_live_candidates`], but also reports WHICH
    /// `(endpoint, model)` candidate served the data — `None` when every candidate failed — so
    /// callers can surface upstream model renames/delistings instead of logging null forever
    /// (the July 19 silent-failure lesson).
    pub fn fetch_forecast_max_live_candidates_tagged(
        &self,
        latitude: f64,
        longitude: f64,
        start_date: NaiveDate,
        end_date: NaiveDate,
        candidates: &[(&str, &str)],
    ) -> TaggedDailyMax {
        for (endpoint, model) in candidates {
            let got = self.fetch_forecast_max_live_model(
                latitude, longitude, start_date, end_date, endpoint, model,
            );
            if !got.is_empty() {
                return (got, Some((endpoint.to_string(), model.to_string())));
            }
        }
        for (endpoint, model) in candidates {
            let got = self
                .fetch_hourly_day_max(latitude, longitude, start_date, end_date, endpoint, model);
            if !got.is_empty() {
                return (got, Some((endpoint.to_string(), model.to_string())));
            }
        }
        (Vec::new(), None)
    }

    /// Hourly temperature for one model collapsed to per-date maxima (the model's own timezone via
    /// `timezone=auto`, matching how daily aggregation would bucket hours).
    fn fetch_hourly_day_max(
        &self,
        latitude: f64,
        longitude: f64,
        start_date: NaiveDate,
        end_date: NaiveDate,
        endpoint: &str,
        model: &str,
    ) -> Vec<(NaiveDate, f64)> {
        let url = format!("https://api.open-meteo.com/v1/{endpoint}");
        let query = [
            ("latitude", latitude.to_string()),
            ("longitude", longitude.to_string()),
            ("start_date", start_date.to_string()),
            ("end_date", end_date.to_string()),
            ("hourly", "temperature_2m".to_string()),
            ("models", model.to_string()),
            ("timezone", "auto".to_string()),
        ];
        let Ok(resp) = self.client.get(&url).query(&query).send() else {
            return Vec::new();
        };
        let Ok(json) = resp.json::<Value>() else {
            return Vec::new();
        };
        parse_hourly_day_max(&json, model)
    }

    /// Per-date member stats (mean °C, std-dev °C of each member's daily high) from the first
    /// ensemble model candidate that serves data, tagged with which one it was (`None` = all
    /// failed).
    /// Tries daily aggregation first, then hourly series collapsed to per-date member maxima —
    /// the ensemble API's supported aggregations couldn't be verified from the sandbox, so both
    /// response shapes are handled (the AI-model playbook). Dates with fewer than
    /// `ENSEMBLE_MIN_MEMBERS` members are dropped, not logged as fake-tight spreads.
    pub fn fetch_ensemble_stats_tagged(
        &self,
        latitude: f64,
        longitude: f64,
        start_date: NaiveDate,
        end_date: NaiveDate,
        candidates: &[&str],
    ) -> TaggedEnsembleStats {
        for model in candidates {
            for (agg_query, agg_parse) in [
                ("daily", true),   // daily=temperature_2m_max, spread over member columns
                ("hourly", false), // hourly=temperature_2m, member series collapsed client-side
            ] {
                let query = [
                    ("latitude", latitude.to_string()),
                    ("longitude", longitude.to_string()),
                    ("start_date", start_date.to_string()),
                    ("end_date", end_date.to_string()),
                    (
                        agg_query,
                        if agg_parse {
                            "temperature_2m_max".to_string()
                        } else {
                            "temperature_2m".to_string()
                        },
                    ),
                    ("models", model.to_string()),
                    ("timezone", "auto".to_string()),
                ];
                let Ok(resp) = self.client.get(ENSEMBLE_BASE_URL).query(&query).send() else {
                    continue;
                };
                let Ok(json) = resp.json::<Value>() else {
                    continue;
                };
                let got = if agg_parse {
                    parse_ensemble_daily_stats(&json, ENSEMBLE_MIN_MEMBERS)
                } else {
                    parse_ensemble_hourly_stats(&json, ENSEMBLE_MIN_MEMBERS)
                };
                if !got.is_empty() {
                    return (got, Some(model.to_string()));
                }
            }
        }
        (HashMap::new(), None)
    }

    /// Forecast PM2.5 (µg/m³) and US AQI at these coords, collapsed to per-UTC-date hourly maxima
    /// over the API's ~week horizon. Empty map on any failure (degradation by design).
    pub fn fetch_air_quality_day_max(&self, latitude: f64, longitude: f64) -> AirQualityByDay {
        let query = [
            ("latitude", latitude.to_string()),
            ("longitude", longitude.to_string()),
            ("hourly", "pm2_5,us_aqi".to_string()),
            ("timezone", "UTC".to_string()),
            ("forecast_days", "7".to_string()),
        ];
        let Ok(resp) = self.client.get(AIR_QUALITY_BASE_URL).query(&query).send() else {
            return HashMap::new();
        };
        let Ok(json) = resp.json::<Value>() else {
            return HashMap::new();
        };
        parse_air_quality_day_max(&json)
    }

    fn fetch_forecast_max_at(
        &self,
        base_url: &str,
        latitude: f64,
        longitude: f64,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Vec<(NaiveDate, f64)> {
        let query = [
            ("latitude", latitude.to_string()),
            ("longitude", longitude.to_string()),
            ("start_date", start_date.to_string()),
            ("end_date", end_date.to_string()),
            ("daily", "temperature_2m_max".to_string()),
            ("timezone", "auto".to_string()),
        ];

        let Ok(resp) = self.client.get(base_url).query(&query).send() else {
            return Vec::new();
        };
        let Ok(json) = resp.json::<Value>() else {
            return Vec::new();
        };
        parse_daily_max(&json, None)
    }

    pub fn fetch_daily_observations(
        &self,
        latitude: f64,
        longitude: f64,
        start_date: NaiveDate,
        end_date: NaiveDate,
        location_id: &str,
    ) -> Vec<WeatherRecord> {
        let query = [
            ("latitude", latitude.to_string()),
            ("longitude", longitude.to_string()),
            ("start_date", start_date.to_string()),
            ("end_date", end_date.to_string()),
            (
                "daily",
                "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,windspeed_10m_max".to_string(),
            ),
            ("timezone", "auto".to_string()),
        ];

        let Ok(resp) = self.client.get(BASE_URL).query(&query).send() else {
            return Vec::new();
        };
        let Ok(json) = resp.json::<Value>() else {
            return Vec::new();
        };

        let daily = json.get("daily").cloned().unwrap_or(Value::Null);
        let times = daily
            .get("time")
            .and_then(|x| x.as_array())
            .cloned()
            .unwrap_or_default();

        let temp_max = as_opt_f64_vec(daily.get("temperature_2m_max"));
        let temp_min = as_opt_f64_vec(daily.get("temperature_2m_min"));
        let temp_mean = as_opt_f64_vec(daily.get("temperature_2m_mean"));
        let precip = as_opt_f64_vec(daily.get("precipitation_sum"));
        let wind = as_opt_f64_vec(daily.get("windspeed_10m_max"));

        let mut out = Vec::new();
        for (i, date_v) in times.iter().enumerate() {
            let Some(ds) = date_v.as_str() else {
                continue;
            };
            let Ok(date) = NaiveDate::parse_from_str(ds, "%Y-%m-%d") else {
                continue;
            };

            let mut row = WeatherRecord::new(date, location_id.to_string());
            row.temperature_max = temp_max.get(i).copied().flatten();
            row.temperature_min = temp_min.get(i).copied().flatten();
            row.temperature_mean = temp_mean.get(i).copied().flatten();
            row.precipitation_total = precip.get(i).copied().flatten();
            row.wind_speed_mean = wind.get(i).copied().flatten();
            out.push(row);
        }

        out
    }

    pub fn fetch_location(
        &self,
        location_key: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Vec<WeatherRecord> {
        let Some((lat, lon, _name)) = self.locations.get(location_key) else {
            return Vec::new();
        };

        self.fetch_daily_observations(
            *lat,
            *lon,
            start_date,
            end_date,
            &format!("OPEN_METEO_{location_key}"),
        )
    }
}

impl Default for OpenMeteoFetcher {
    fn default() -> Self {
        Self::new()
    }
}

fn as_opt_f64_vec(v: Option<&Value>) -> Vec<Option<f64>> {
    v.and_then(|x| x.as_array().cloned())
        .unwrap_or_default()
        .iter()
        .map(|n| n.as_f64())
        .collect()
}

/// (date, daily max) pairs from a forecast response. With a single `models=` value the API keeps
/// the plain `temperature_2m_max` key, but suffixed `temperature_2m_max_<model>` variants have
/// been observed — accept either when a model is given.
fn parse_daily_max(json: &Value, model: Option<&str>) -> Vec<(NaiveDate, f64)> {
    let daily = json.get("daily").cloned().unwrap_or(Value::Null);
    let times = daily
        .get("time")
        .and_then(|x| x.as_array())
        .cloned()
        .unwrap_or_default();
    let field = daily
        .get("temperature_2m_max")
        .or_else(|| model.and_then(|m| daily.get(format!("temperature_2m_max_{m}").as_str())));
    let highs = as_opt_f64_vec(field);

    let mut out = Vec::new();
    for (i, date_v) in times.iter().enumerate() {
        let Some(ds) = date_v.as_str() else { continue };
        let Ok(date) = NaiveDate::parse_from_str(ds, "%Y-%m-%d") else {
            continue;
        };
        if let Some(h) = highs.get(i).copied().flatten() {
            out.push((date, h));
        }
    }
    out
}

/// Hourly temperature_2m for one model collapsed to per-date maxima. Accepts the plain key or the
/// model-suffixed variant, like `parse_daily_max`.
fn parse_hourly_day_max(json: &Value, model: &str) -> Vec<(NaiveDate, f64)> {
    let hourly = json.get("hourly").cloned().unwrap_or(Value::Null);
    let times = hourly
        .get("time")
        .and_then(|x| x.as_array())
        .cloned()
        .unwrap_or_default();
    let field = hourly
        .get("temperature_2m")
        .or_else(|| hourly.get(format!("temperature_2m_{model}").as_str()));
    let temps = as_opt_f64_vec(field);

    let mut by_date: HashMap<NaiveDate, f64> = HashMap::new();
    for (i, tv) in times.iter().enumerate() {
        let Some(ts) = tv.as_str() else { continue };
        let Ok(t) = chrono::NaiveDateTime::parse_from_str(ts, "%Y-%m-%dT%H:%M") else {
            continue;
        };
        if let Some(v) = temps.get(i).copied().flatten() {
            let e = by_date.entry(t.date()).or_insert(f64::NEG_INFINITY);
            *e = e.max(v);
        }
    }
    let mut out: Vec<(NaiveDate, f64)> = by_date.into_iter().collect();
    out.sort_by_key(|(d, _)| *d);
    out
}

/// Every member series under `obj` for the base variable: the plain key plus any
/// `{base}_member01`-style (or model-suffixed) variant. The ensemble API's exact key shape
/// couldn't be sandbox-verified, so anything prefixed with the base counts as a member.
fn ensemble_member_series(obj: &Value, base: &str) -> Vec<Vec<Option<f64>>> {
    let Some(map) = obj.as_object() else {
        return Vec::new();
    };
    let prefix = format!("{base}_");
    let mut keys: Vec<&String> = map
        .keys()
        .filter(|k| *k == base || k.starts_with(&prefix))
        .collect();
    keys.sort(); // deterministic member order regardless of JSON map order
    keys.iter().map(|k| as_opt_f64_vec(map.get(*k))).collect()
}

/// (mean, sample std-dev); None below `min_n` values — a "spread" over a handful of members (or
/// a deterministic model that slipped through a candidate list) is noise, not uncertainty, and a
/// "mean" of one run is just that run.
fn member_stats(values: &[f64], min_n: usize) -> Option<(f64, f64)> {
    if values.len() < min_n.max(2) {
        return None;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    Some((mean, var.sqrt()))
}

/// Per-date member spread from a daily-aggregated ensemble response.
fn parse_ensemble_daily_stats(json: &Value, min_members: usize) -> HashMap<NaiveDate, (f64, f64)> {
    let daily = json.get("daily").cloned().unwrap_or(Value::Null);
    let times = daily
        .get("time")
        .and_then(|x| x.as_array())
        .cloned()
        .unwrap_or_default();
    let members = ensemble_member_series(&daily, "temperature_2m_max");

    let mut out = HashMap::new();
    for (i, date_v) in times.iter().enumerate() {
        let Some(ds) = date_v.as_str() else { continue };
        let Ok(date) = NaiveDate::parse_from_str(ds, "%Y-%m-%d") else {
            continue;
        };
        let vals: Vec<f64> = members
            .iter()
            .filter_map(|m| m.get(i).copied().flatten())
            .collect();
        if let Some(stats) = member_stats(&vals, min_members) {
            out.insert(date, stats);
        }
    }
    out
}

/// Per-date member spread from an hourly ensemble response: each member's hours collapse to its
/// per-date max first (same local-date bucketing as `parse_hourly_day_max`), THEN the spread is
/// taken across members — the spread of daily highs, not of any single hour.
fn parse_ensemble_hourly_stats(json: &Value, min_members: usize) -> HashMap<NaiveDate, (f64, f64)> {
    let hourly = json.get("hourly").cloned().unwrap_or(Value::Null);
    let times = hourly
        .get("time")
        .and_then(|x| x.as_array())
        .cloned()
        .unwrap_or_default();
    let members = ensemble_member_series(&hourly, "temperature_2m");

    let mut by_date: HashMap<NaiveDate, Vec<f64>> = HashMap::new();
    for member in &members {
        let mut day_max: HashMap<NaiveDate, f64> = HashMap::new();
        for (i, tv) in times.iter().enumerate() {
            let Some(ts) = tv.as_str() else { continue };
            let Ok(t) = chrono::NaiveDateTime::parse_from_str(ts, "%Y-%m-%dT%H:%M") else {
                continue;
            };
            if let Some(v) = member.get(i).copied().flatten() {
                let e = day_max.entry(t.date()).or_insert(f64::NEG_INFINITY);
                *e = e.max(v);
            }
        }
        for (date, mx) in day_max {
            by_date.entry(date).or_default().push(mx);
        }
    }

    by_date
        .into_iter()
        .filter_map(|(date, vals)| member_stats(&vals, min_members).map(|s| (date, s)))
        .collect()
}

/// Hourly pm2_5/us_aqi series collapsed to per-UTC-date maxima. A date appears whenever it has any
/// hour of either series; a series absent for that date stays None.
fn parse_air_quality_day_max(json: &Value) -> AirQualityByDay {
    let hourly = json.get("hourly").cloned().unwrap_or(Value::Null);
    let times = hourly
        .get("time")
        .and_then(|x| x.as_array())
        .cloned()
        .unwrap_or_default();
    let pm25 = as_opt_f64_vec(hourly.get("pm2_5"));
    let aqi = as_opt_f64_vec(hourly.get("us_aqi"));

    let mut out: AirQualityByDay = HashMap::new();
    for (i, tv) in times.iter().enumerate() {
        let Some(ts) = tv.as_str() else { continue };
        let Ok(t) = chrono::NaiveDateTime::parse_from_str(ts, "%Y-%m-%dT%H:%M") else {
            continue;
        };
        let entry = out.entry(t.date()).or_insert((None, None));
        if let Some(p) = pm25.get(i).copied().flatten() {
            entry.0 = Some(entry.0.map_or(p, |x: f64| x.max(p)));
        }
        if let Some(a) = aqi.get(i).copied().flatten() {
            entry.1 = Some(entry.1.map_or(a, |x: f64| x.max(a)));
        }
    }
    out
}

fn default_locations() -> HashMap<String, (f64, f64, String)> {
    // Built from the shared city registry so every recognizable city has coordinates (Open-Meteo's
    // archive works for any lat/lon globally, so coverage is just the registry).
    crate::cities::CITIES
        .iter()
        .map(|c| (c.key.to_string(), (c.lat, c.lon, c.key.to_string())))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_daily_max_plain_and_model_suffixed_keys() {
        let plain = serde_json::json!({"daily": {
            "time": ["2026-07-20", "2026-07-21"],
            "temperature_2m_max": [31.4, null]
        }});
        let got = parse_daily_max(&plain, Some("ecmwf_aifs025"));
        assert_eq!(
            got,
            vec![(NaiveDate::from_ymd_opt(2026, 7, 20).unwrap(), 31.4)],
            "null entries are dropped, plain key wins"
        );

        let suffixed = serde_json::json!({"daily": {
            "time": ["2026-07-20"],
            "temperature_2m_max_ecmwf_aifs025": [30.0]
        }});
        assert_eq!(
            parse_daily_max(&suffixed, Some("ecmwf_aifs025")),
            vec![(NaiveDate::from_ymd_opt(2026, 7, 20).unwrap(), 30.0)]
        );
        assert!(
            parse_daily_max(&suffixed, None).is_empty(),
            "no model given ⇒ suffixed key is not searched"
        );
    }

    #[test]
    fn parse_hourly_day_max_collapses_and_accepts_suffixed_key() {
        let json = serde_json::json!({"hourly": {
            "time": ["2026-07-20T00:00", "2026-07-20T14:00", "2026-07-21T13:00"],
            "temperature_2m_gfs_graphcast025": [24.1, 33.7, null]
        }});
        let got = parse_hourly_day_max(&json, "gfs_graphcast025");
        assert_eq!(
            got,
            vec![(NaiveDate::from_ymd_opt(2026, 7, 20).unwrap(), 33.7)],
            "hours collapse to the day max; all-null days are dropped"
        );
        assert!(parse_hourly_day_max(&serde_json::json!({}), "x").is_empty());
    }

    #[test]
    fn ai_log_candidates_align_with_model_names() {
        // Snapshot fields are positional (0 = AIFS, 1 = GraphCast) — keep the two constants in
        // lockstep so a reorder can't silently swap which model lands in which field.
        assert_eq!(AI_LOG_CANDIDATES.len(), AI_LOG_MODELS.len());
        for (i, name) in AI_LOG_MODELS.iter().enumerate() {
            assert!(
                AI_LOG_CANDIDATES[i].iter().any(|(_, m)| m == name),
                "AI_LOG_MODELS[{i}] = {name} missing from its candidate list"
            );
        }
    }

    #[test]
    fn ensemble_log_candidates_align_with_model_names() {
        // Snapshot fields are positional (0 = ECMWF ENS, 1 = GEFS) — keep the constants in
        // lockstep so a reorder can't silently swap which system lands in which field.
        assert_eq!(ENSEMBLE_LOG_CANDIDATES.len(), ENSEMBLE_LOG_MODELS.len());
        for (i, name) in ENSEMBLE_LOG_MODELS.iter().enumerate() {
            assert!(
                ENSEMBLE_LOG_CANDIDATES[i].contains(name),
                "ENSEMBLE_LOG_MODELS[{i}] = {name} missing from its candidate list"
            );
        }
    }

    #[test]
    fn ensemble_daily_spread_needs_min_members_and_skips_null_holes() {
        let json = serde_json::json!({"daily": {
            "time": ["2026-07-24", "2026-07-25"],
            "temperature_2m_max": [30.0, 31.0],
            "temperature_2m_max_member01": [31.0, null],
            "temperature_2m_max_member02": [32.0, 30.0]
        }});
        let got = parse_ensemble_daily_stats(&json, 3);
        let d24 = NaiveDate::from_ymd_opt(2026, 7, 24).unwrap();
        let d25 = NaiveDate::from_ymd_opt(2026, 7, 25).unwrap();
        assert_eq!(
            got.get(&d24),
            Some(&(31.0, 1.0)),
            "mean and sample std of [30, 31, 32]"
        );
        assert!(
            !got.contains_key(&d25),
            "a null member drops the date below the member floor"
        );
        assert!(
            parse_ensemble_daily_stats(&json, 4).is_empty(),
            "min_members above the member count yields nothing"
        );
    }

    #[test]
    fn ensemble_hourly_spread_collapses_each_member_before_spreading() {
        // Member A peaks at 33 (hour 14), member B at 31 (hour 13): the spread must be over the
        // per-member DAY MAXES {33, 31, 35}, not over any single hour's values.
        let json = serde_json::json!({"hourly": {
            "time": ["2026-07-24T00:00", "2026-07-24T13:00", "2026-07-24T14:00"],
            "temperature_2m": [20.0, 30.0, 33.0],
            "temperature_2m_member01": [21.0, 31.0, 29.0],
            "temperature_2m_member02": [22.0, 32.0, 35.0]
        }});
        let got = parse_ensemble_hourly_stats(&json, 3);
        let d24 = NaiveDate::from_ymd_opt(2026, 7, 24).unwrap();
        assert_eq!(
            got.get(&d24),
            Some(&(33.0, 2.0)),
            "mean and sample std of maxes [33, 31, 35]"
        );
        assert!(parse_ensemble_hourly_stats(&serde_json::json!({}), 2).is_empty());
    }

    #[test]
    fn member_stats_rejects_deterministic_singletons() {
        assert_eq!(member_stats(&[30.0], 1), None, "min_n is floored at 2");
        assert_eq!(
            member_stats(&[30.0, 32.0], 2),
            Some((31.0, std::f64::consts::SQRT_2))
        );
    }

    #[test]
    fn parse_air_quality_collapses_hours_to_day_max() {
        let json = serde_json::json!({"hourly": {
            "time": ["2026-07-20T00:00", "2026-07-20T13:00", "2026-07-21T00:00"],
            "pm2_5": [12.0, 87.5, null],
            "us_aqi": [40.0, 158.0, 61.0]
        }});
        let got = parse_air_quality_day_max(&json);
        let d20 = NaiveDate::from_ymd_opt(2026, 7, 20).unwrap();
        let d21 = NaiveDate::from_ymd_opt(2026, 7, 21).unwrap();
        assert_eq!(got.get(&d20), Some(&(Some(87.5), Some(158.0))));
        assert_eq!(
            got.get(&d21),
            Some(&(None, Some(61.0))),
            "date with AQI but no pm2.5 keeps pm2.5 = None"
        );
        assert!(parse_air_quality_day_max(&serde_json::json!({})).is_empty());
    }
}
