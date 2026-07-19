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
pub const AI_LOG_MODELS: &[&str] = &["ecmwf_aifs025", "gfs_graphcast025"];
/// Air-quality forecast endpoint (separate host from the weather API). Used to log forecast smoke
/// (PM2.5 / US AQI) for market target days: heavy wildfire smoke measurably suppresses daily highs
/// and the blend models only partly price the effect.
pub const AIR_QUALITY_BASE_URL: &str = "https://air-quality-api.open-meteo.com/v1/air-quality";
/// Per-UTC-date (max PM2.5 µg/m³, max US AQI) — the shape `fetch_air_quality_day_max` returns.
pub type AirQualityByDay = HashMap<NaiveDate, (Option<f64>, Option<f64>)>;

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

    /// LIVE daily-high forecast (degC) from a single named model (e.g. one of `AI_LOG_MODELS`),
    /// for logging alongside the blend. Empty vec on any failure — including an unknown model
    /// name, which the API rejects for the whole request (why callers pass one model at a time).
    pub fn fetch_forecast_max_live_model(
        &self,
        latitude: f64,
        longitude: f64,
        start_date: NaiveDate,
        end_date: NaiveDate,
        model: &str,
    ) -> Vec<(NaiveDate, f64)> {
        let query = [
            ("latitude", latitude.to_string()),
            ("longitude", longitude.to_string()),
            ("start_date", start_date.to_string()),
            ("end_date", end_date.to_string()),
            ("daily", "temperature_2m_max".to_string()),
            ("models", model.to_string()),
            ("timezone", "auto".to_string()),
        ];
        let Ok(resp) = self.client.get(FORECAST_LIVE_BASE_URL).query(&query).send() else {
            return Vec::new();
        };
        let Ok(json) = resp.json::<Value>() else {
            return Vec::new();
        };
        parse_daily_max(&json, Some(model))
    }

    /// Forecast PM2.5 (µg/m³) and US AQI at these coords, collapsed to per-UTC-date hourly maxima
    /// over the API's ~week horizon. Empty map on any failure (degradation by design).
    pub fn fetch_air_quality_day_max(
        &self,
        latitude: f64,
        longitude: f64,
    ) -> AirQualityByDay {
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
    let field = daily.get("temperature_2m_max").or_else(|| {
        model.and_then(|m| daily.get(format!("temperature_2m_max_{m}").as_str()))
    });
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
