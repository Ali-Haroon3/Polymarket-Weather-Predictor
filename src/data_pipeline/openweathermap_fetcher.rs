use std::collections::HashMap;

use chrono::{Duration, NaiveDate};
use serde_json::Value;

use crate::config::openweathermap_api_key;
use crate::types::WeatherRecord;

const BASE_URL: &str = "https://api.openweathermap.org/data/3.0/onecall/day_summary";

#[derive(Clone)]
pub struct OpenWeatherMapFetcher {
    api_key: String,
    client: reqwest::blocking::Client,
    locations: HashMap<String, (f64, f64)>,
}

impl OpenWeatherMapFetcher {
    pub fn new(api_key: Option<String>) -> Self {
        Self {
            api_key: api_key.unwrap_or_else(openweathermap_api_key),
            client: crate::data_pipeline::build_client(15),
            locations: default_locations(),
        }
    }

    pub fn is_available(&self) -> bool {
        !self.api_key.is_empty()
    }

    pub fn fetch_daily_observations(
        &self,
        latitude: f64,
        longitude: f64,
        start_date: NaiveDate,
        end_date: NaiveDate,
        location_id: &str,
    ) -> Vec<WeatherRecord> {
        if !self.is_available() {
            return Vec::new();
        }

        let mut out = Vec::new();
        let mut current = start_date;

        while current <= end_date {
            let query = [
                ("lat", latitude.to_string()),
                ("lon", longitude.to_string()),
                ("date", current.to_string()),
                ("appid", self.api_key.clone()),
                ("units", "metric".to_string()),
            ];

            if let Ok(resp) = self.client.get(BASE_URL).query(&query).send() {
                if let Ok(json) = resp.json::<Value>() {
                    let mut row = WeatherRecord::new(current, format!("OWM_{location_id}"));
                    row.temperature_max = json
                        .get("temperature")
                        .and_then(|v| v.get("max"))
                        .and_then(|v| v.as_f64());
                    row.temperature_min = json
                        .get("temperature")
                        .and_then(|v| v.get("min"))
                        .and_then(|v| v.as_f64());

                    row.temperature_mean = match (row.temperature_max, row.temperature_min) {
                        (Some(mx), Some(mn)) => Some((mx + mn) / 2.0),
                        _ => None,
                    };

                    row.precipitation_total = json
                        .get("precipitation")
                        .and_then(|v| v.get("total"))
                        .and_then(|v| v.as_f64())
                        .or(Some(0.0));
                    row.wind_speed_mean = json
                        .get("wind")
                        .and_then(|v| v.get("max"))
                        .and_then(|v| v.get("speed"))
                        .and_then(|v| v.as_f64());
                    out.push(row);
                }
            }

            current += Duration::days(1);
        }

        out
    }

    pub fn fetch_location(&self, location_key: &str, start_date: NaiveDate, end_date: NaiveDate) -> Vec<WeatherRecord> {
        let Some((lat, lon)) = self.locations.get(location_key) else {
            return Vec::new();
        };
        self.fetch_daily_observations(*lat, *lon, start_date, end_date, location_key)
    }
}

impl Default for OpenWeatherMapFetcher {
    fn default() -> Self {
        Self::new(None)
    }
}

fn default_locations() -> HashMap<String, (f64, f64)> {
    [
        ("NYC", (40.71, -74.01)),
        ("LA", (34.05, -118.24)),
        ("London", (51.51, -0.13)),
        ("Chicago", (41.88, -87.63)),
        ("Dallas", (32.78, -96.80)),
        ("Denver", (39.74, -104.99)),
        ("Miami", (25.76, -80.19)),
        ("Boston", (42.36, -71.06)),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect()
}
