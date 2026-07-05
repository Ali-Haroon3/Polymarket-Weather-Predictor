use std::collections::HashMap;

use chrono::NaiveDate;
use serde_json::Value;

use crate::config::tomorrow_io_api_key;
use crate::types::WeatherRecord;

const BASE_URL: &str = "https://api.tomorrow.io/v4/timelines";

#[derive(Clone)]
pub struct TomorrowIOFetcher {
    api_key: String,
    client: reqwest::blocking::Client,
    locations: HashMap<String, (f64, f64)>,
}

impl TomorrowIOFetcher {
    pub fn new(api_key: Option<String>) -> Self {
        Self {
            api_key: api_key.unwrap_or_else(tomorrow_io_api_key),
            client: crate::data_pipeline::build_client(30),
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

        let params = [
            ("location", format!("{latitude},{longitude}")),
            (
                "fields",
                "temperatureMax,temperatureMin,temperatureAvg,precipitationIntensityAvg,windSpeedAvg".to_string(),
            ),
            ("timesteps", "1d".to_string()),
            (
                "startTime",
                format!("{}T00:00:00Z", start_date.format("%Y-%m-%d")),
            ),
            (
                "endTime",
                format!("{}T23:59:59Z", end_date.format("%Y-%m-%d")),
            ),
            ("apikey", self.api_key.clone()),
            ("units", "metric".to_string()),
        ];

        let Ok(resp) = self.client.get(BASE_URL).query(&params).send() else {
            return Vec::new();
        };
        let Ok(json) = resp.json::<Value>() else {
            return Vec::new();
        };

        let intervals = json
            .get("data")
            .and_then(|d| d.get("timelines"))
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|t| t.get("intervals"))
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let mut out = Vec::new();
        for interval in intervals {
            let start_time = interval.get("startTime").and_then(|v| v.as_str());
            let Some(start_time) = start_time else {
                continue;
            };

            let Ok(date) = NaiveDate::parse_from_str(&start_time[..10], "%Y-%m-%d") else {
                continue;
            };
            let values = interval.get("values").cloned().unwrap_or(Value::Null);

            let mut row = WeatherRecord::new(date, format!("TIO_{location_id}"));
            row.temperature_max = values.get("temperatureMax").and_then(|v| v.as_f64());
            row.temperature_min = values.get("temperatureMin").and_then(|v| v.as_f64());
            row.temperature_mean = values.get("temperatureAvg").and_then(|v| v.as_f64());
            row.precipitation_total = values
                .get("precipitationIntensityAvg")
                .and_then(|v| v.as_f64())
                .map(|x| x * 24.0);
            row.wind_speed_mean = values.get("windSpeedAvg").and_then(|v| v.as_f64());
            out.push(row);
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

impl Default for TomorrowIOFetcher {
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
