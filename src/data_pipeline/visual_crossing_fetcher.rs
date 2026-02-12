use std::collections::HashMap;

use chrono::NaiveDate;
use serde_json::Value;

use crate::config::visual_crossing_api_key;
use crate::types::WeatherRecord;

const BASE_URL: &str = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline";

#[derive(Clone)]
pub struct VisualCrossingFetcher {
    api_key: String,
    client: reqwest::blocking::Client,
    locations: HashMap<String, String>,
}

impl VisualCrossingFetcher {
    pub fn new(api_key: Option<String>) -> Self {
        Self {
            api_key: api_key.unwrap_or_else(visual_crossing_api_key),
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap_or_else(|_| reqwest::blocking::Client::new()),
            locations: default_locations(),
        }
    }

    pub fn is_available(&self) -> bool {
        !self.api_key.is_empty()
    }

    pub fn fetch_daily_observations(
        &self,
        location_query: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
        location_id: &str,
    ) -> Vec<WeatherRecord> {
        if !self.is_available() {
            return Vec::new();
        }

        let url = format!("{BASE_URL}/{location_query}/{start_date}/{end_date}");
        let query = [
            ("unitGroup", "metric".to_string()),
            ("include", "days".to_string()),
            ("key", self.api_key.clone()),
            ("contentType", "json".to_string()),
        ];

        let Ok(resp) = self.client.get(url).query(&query).send() else {
            return Vec::new();
        };
        let Ok(json) = resp.json::<Value>() else {
            return Vec::new();
        };

        json.get("days")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default()
            .iter()
            .filter_map(|day| {
                let date = day.get("datetime")?.as_str()?;
                let d = NaiveDate::parse_from_str(date, "%Y-%m-%d").ok()?;

                let mut row = WeatherRecord::new(d, format!("VC_{location_id}"));
                row.temperature_max = day.get("tempmax").and_then(|v| v.as_f64());
                row.temperature_min = day.get("tempmin").and_then(|v| v.as_f64());
                row.temperature_mean = day.get("temp").and_then(|v| v.as_f64());
                row.precipitation_total = day.get("precip").and_then(|v| v.as_f64()).or(Some(0.0));
                row.wind_speed_mean = day.get("windspeed").and_then(|v| v.as_f64());
                Some(row)
            })
            .collect()
    }

    pub fn fetch_location(&self, location_key: &str, start_date: NaiveDate, end_date: NaiveDate) -> Vec<WeatherRecord> {
        let Some(query) = self.locations.get(location_key) else {
            return Vec::new();
        };
        self.fetch_daily_observations(query, start_date, end_date, location_key)
    }
}

impl Default for VisualCrossingFetcher {
    fn default() -> Self {
        Self::new(None)
    }
}

fn default_locations() -> HashMap<String, String> {
    [
        ("NYC", "New York,NY,US"),
        ("LA", "Los Angeles,CA,US"),
        ("London", "London,UK"),
        ("Chicago", "Chicago,IL,US"),
        ("Dallas", "Dallas,TX,US"),
        ("Denver", "Denver,CO,US"),
        ("Miami", "Miami,FL,US"),
        ("Boston", "Boston,MA,US"),
    ]
    .into_iter()
    .map(|(k, q)| (k.to_string(), q.to_string()))
    .collect()
}
