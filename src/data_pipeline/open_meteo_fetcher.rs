use std::collections::HashMap;

use chrono::NaiveDate;
use serde_json::Value;

use crate::types::WeatherRecord;

pub const BASE_URL: &str = "https://archive-api.open-meteo.com/v1/archive";

#[derive(Clone)]
pub struct OpenMeteoFetcher {
    client: reqwest::blocking::Client,
    locations: HashMap<String, (f64, f64, String)>,
}

impl OpenMeteoFetcher {
    pub fn new() -> Self {
        Self {
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap_or_else(|_| reqwest::blocking::Client::new()),
            locations: default_locations(),
        }
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

    pub fn fetch_location(&self, location_key: &str, start_date: NaiveDate, end_date: NaiveDate) -> Vec<WeatherRecord> {
        let Some((lat, lon, _name)) = self.locations.get(location_key) else {
            return Vec::new();
        };

        self.fetch_daily_observations(*lat, *lon, start_date, end_date, &format!("OPEN_METEO_{location_key}"))
    }

    pub fn fetch_multiple_locations(
        &self,
        location_keys: &[String],
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Vec<WeatherRecord> {
        let mut out = Vec::new();
        for key in location_keys {
            out.extend(self.fetch_location(key, start_date, end_date));
        }
        out
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

fn default_locations() -> HashMap<String, (f64, f64, String)> {
    // Built from the shared city registry so every recognizable city has coordinates (Open-Meteo's
    // archive works for any lat/lon globally, so coverage is just the registry).
    crate::cities::CITIES
        .iter()
        .map(|c| (c.key.to_string(), (c.lat, c.lon, c.key.to_string())))
        .collect()
}
