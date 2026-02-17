use std::collections::HashMap;

use chrono::{Duration, NaiveDate};
use serde_json::Value;

use crate::config::accuweather_api_key;
use crate::types::WeatherRecord;

const BASE_URL: &str = "https://dataservice.accuweather.com";

#[derive(Clone)]
pub struct AccuWeatherFetcher {
    api_key: String,
    client: reqwest::blocking::Client,
    location_keys: HashMap<String, String>,
}

impl AccuWeatherFetcher {
    pub fn new(api_key: Option<String>) -> Self {
        Self {
            api_key: api_key.unwrap_or_else(accuweather_api_key),
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(15))
                .build()
                .unwrap_or_else(|_| reqwest::blocking::Client::new()),
            location_keys: default_location_keys(),
        }
    }

    pub fn is_available(&self) -> bool {
        !self.api_key.is_empty()
    }

    pub fn fetch_daily_conditions(
        &self,
        accu_location_key: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
        city_id: &str,
    ) -> Vec<WeatherRecord> {
        if !self.is_available() {
            return Vec::new();
        }

        let mut out = Vec::new();
        let mut current = start_date;

        while current <= end_date {
            let url = format!("{BASE_URL}/forecasts/v1/daily/1day/{accu_location_key}");
            let query = [
                ("apikey", self.api_key.clone()),
                ("metric", "true".to_string()),
                ("details", "true".to_string()),
            ];

            if let Ok(resp) = self.client.get(&url).query(&query).send() {
                if let Ok(json) = resp.json::<Value>() {
                    let day = json
                        .get("DailyForecasts")
                        .and_then(|v| v.as_array())
                        .and_then(|arr| arr.first())
                        .cloned();

                    if let Some(day) = day {
                        let mut row =
                            WeatherRecord::new(current, format!("ACCUWEATHER_{city_id}"));
                        row.temperature_max = day
                            .get("Temperature")
                            .and_then(|v| v.get("Maximum"))
                            .and_then(|v| v.get("Value"))
                            .and_then(|v| v.as_f64());
                        row.temperature_min = day
                            .get("Temperature")
                            .and_then(|v| v.get("Minimum"))
                            .and_then(|v| v.get("Value"))
                            .and_then(|v| v.as_f64());
                        row.temperature_mean = match (row.temperature_max, row.temperature_min) {
                            (Some(mx), Some(mn)) => Some((mx + mn) / 2.0),
                            _ => None,
                        };
                        row.precipitation_total = day
                            .get("Day")
                            .and_then(|v| v.get("TotalLiquid"))
                            .and_then(|v| v.get("Value"))
                            .and_then(|v| v.as_f64())
                            .or(Some(0.0));
                        row.wind_speed_mean = day
                            .get("Day")
                            .and_then(|v| v.get("Wind"))
                            .and_then(|v| v.get("Speed"))
                            .and_then(|v| v.get("Value"))
                            .and_then(|v| v.as_f64());
                        out.push(row);
                    }
                }
            }

            current += Duration::days(1);
        }

        out
    }

    pub fn fetch_location(
        &self,
        location_key: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Vec<WeatherRecord> {
        let Some(accu_key) = self.location_keys.get(location_key) else {
            return Vec::new();
        };
        self.fetch_daily_conditions(accu_key, start_date, end_date, location_key)
    }
}

impl Default for AccuWeatherFetcher {
    fn default() -> Self {
        Self::new(None)
    }
}

fn default_location_keys() -> HashMap<String, String> {
    [
        ("NYC", "349727"),
        ("LA", "347625"),
        ("London", "328328"),
        ("Chicago", "348308"),
        ("Dallas", "351194"),
        ("Denver", "347810"),
        ("Miami", "347936"),
        ("Boston", "348735"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}
