use std::collections::HashMap;

use chrono::{Duration, NaiveDate};
use serde_json::Value;

use crate::config::awc_base_url;
use crate::types::WeatherRecord;

#[derive(Clone)]
pub struct AWCFetcher {
    base_url: String,
    client: reqwest::blocking::Client,
    stations: HashMap<String, String>,
}

impl AWCFetcher {
    pub fn new() -> Self {
        Self {
            base_url: awc_base_url(),
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .user_agent("polymarket-weather-predictor/1.0")
                .build()
                .unwrap_or_else(|_| reqwest::blocking::Client::new()),
            stations: default_stations(),
        }
    }

    pub fn fetch_metar(
        &self,
        icao: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Vec<WeatherRecord> {
        let mut out = Vec::new();
        let mut current = start_date;

        while current <= end_date {
            let url = format!("{}/metar", self.base_url);
            let query = [
                ("ids", icao.to_string()),
                ("format", "json".to_string()),
                ("date", current.format("%Y%m%d").to_string()),
                ("hours", "24".to_string()),
            ];

            if let Ok(resp) = self.client.get(&url).query(&query).send() {
                if let Ok(json) = resp.json::<Value>() {
                    if let Some(observations) = json.as_array() {
                        let mut temps: Vec<f64> = Vec::new();
                        let mut winds: Vec<f64> = Vec::new();
                        let mut precips: Vec<f64> = Vec::new();

                        for obs in observations {
                            if let Some(t) = obs.get("temp").and_then(|v| v.as_f64()) {
                                temps.push(t);
                            }
                            // METAR wind speed is in knots; convert to m/s
                            if let Some(w) = obs.get("wspd").and_then(|v| v.as_f64()) {
                                winds.push(w * 0.514_444);
                            }
                            if let Some(p) = obs.get("precip").and_then(|v| v.as_f64()) {
                                precips.push(p);
                            }
                        }

                        if !temps.is_empty() {
                            let mut row =
                                WeatherRecord::new(current, format!("AWC_METAR_{icao}"));
                            row.temperature_max = temps.iter().copied().reduce(f64::max);
                            row.temperature_min = temps.iter().copied().reduce(f64::min);
                            row.temperature_mean =
                                Some(temps.iter().sum::<f64>() / temps.len() as f64);
                            row.wind_speed_mean = if winds.is_empty() {
                                None
                            } else {
                                Some(winds.iter().sum::<f64>() / winds.len() as f64)
                            };
                            row.precipitation_total = Some(precips.iter().sum());
                            out.push(row);
                        }
                    }
                }
            }

            current += Duration::days(1);
        }

        out
    }

    pub fn fetch_taf(
        &self,
        icao: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Vec<WeatherRecord> {
        let mut out = Vec::new();
        let mut current = start_date;

        while current <= end_date {
            let url = format!("{}/taf", self.base_url);
            let query = [
                ("ids", icao.to_string()),
                ("format", "json".to_string()),
                ("date", current.format("%Y%m%d").to_string()),
            ];

            if let Ok(resp) = self.client.get(&url).query(&query).send() {
                if let Ok(json) = resp.json::<Value>() {
                    if let Some(forecasts) = json.as_array() {
                        for fc in forecasts {
                            let mut row =
                                WeatherRecord::new(current, format!("AWC_TAF_{icao}"));
                            // TAF wind speed is in knots; convert to m/s
                            if let Some(w) = fc.get("wspd").and_then(|v| v.as_f64()) {
                                row.wind_speed_mean = Some(w * 0.514_444);
                            }
                            if row.wind_speed_mean.is_some() {
                                out.push(row);
                                break;
                            }
                        }
                    }
                }
            }

            current += Duration::days(1);
        }

        out
    }

    pub fn fetch_metar_location(
        &self,
        location_key: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Vec<WeatherRecord> {
        let Some(icao) = self.stations.get(location_key) else {
            return Vec::new();
        };
        self.fetch_metar(icao, start_date, end_date)
    }

    pub fn fetch_taf_location(
        &self,
        location_key: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Vec<WeatherRecord> {
        let Some(icao) = self.stations.get(location_key) else {
            return Vec::new();
        };
        self.fetch_taf(icao, start_date, end_date)
    }
}

impl Default for AWCFetcher {
    fn default() -> Self {
        Self::new()
    }
}

fn default_stations() -> HashMap<String, String> {
    [
        ("NYC", "KNYC"),
        ("LA", "KLAX"),
        ("London", "EGLL"),
        ("Chicago", "KORD"),
        ("Dallas", "KDFW"),
        ("Denver", "KDEN"),
        ("Miami", "KMIA"),
        ("Boston", "KBOS"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}
