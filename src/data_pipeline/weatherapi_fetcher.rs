use std::collections::HashMap;

use chrono::{Duration, NaiveDate};
use serde_json::Value;

use crate::config::weatherapi_key;
use crate::types::WeatherRecord;

const BASE_URL: &str = "https://api.weatherapi.com/v1/history.json";

#[derive(Clone)]
pub struct WeatherAPIFetcher {
    api_key: String,
    client: reqwest::blocking::Client,
    locations: HashMap<String, String>,
}

impl WeatherAPIFetcher {
    pub fn new(api_key: Option<String>) -> Self {
        Self {
            api_key: api_key.unwrap_or_else(weatherapi_key),
            client: crate::data_pipeline::build_client(15),
            locations: default_locations(),
        }
    }

    pub fn is_available(&self) -> bool {
        !self.api_key.is_empty()
    }

    pub fn fetch_daily_observations(
        &self,
        query: &str,
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
            let params = [
                ("key", self.api_key.clone()),
                ("q", query.to_string()),
                ("dt", current.to_string()),
            ];

            if let Ok(resp) = self.client.get(BASE_URL).query(&params).send() {
                if let Ok(json) = resp.json::<Value>() {
                    if let Some(day) = json
                        .get("forecast")
                        .and_then(|v| v.get("forecastday"))
                        .and_then(|v| v.as_array())
                        .and_then(|arr| arr.first())
                        .and_then(|v| v.get("day"))
                    {
                        let mut row = WeatherRecord::new(current, format!("WAPI_{location_id}"));
                        row.temperature_max = day.get("maxtemp_c").and_then(|v| v.as_f64());
                        row.temperature_min = day.get("mintemp_c").and_then(|v| v.as_f64());
                        row.temperature_mean = day.get("avgtemp_c").and_then(|v| v.as_f64());
                        row.precipitation_total = day
                            .get("totalprecip_mm")
                            .and_then(|v| v.as_f64())
                            .or(Some(0.0));
                        row.wind_speed_mean = day
                            .get("maxwind_kph")
                            .and_then(|v| v.as_f64())
                            .map(|x| x / 3.6);
                        out.push(row);
                    }
                }
            }

            current += Duration::days(1);
        }

        out
    }

    pub fn fetch_location(&self, location_key: &str, start_date: NaiveDate, end_date: NaiveDate) -> Vec<WeatherRecord> {
        let Some(query) = self.locations.get(location_key) else {
            return Vec::new();
        };
        self.fetch_daily_observations(query, start_date, end_date, location_key)
    }
}

impl Default for WeatherAPIFetcher {
    fn default() -> Self {
        Self::new(None)
    }
}

fn default_locations() -> HashMap<String, String> {
    [
        ("NYC", "New York"),
        ("LA", "Los Angeles"),
        ("London", "London"),
        ("Chicago", "Chicago"),
        ("Dallas", "Dallas"),
        ("Denver", "Denver"),
        ("Miami", "Miami"),
        ("Boston", "Boston"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}
