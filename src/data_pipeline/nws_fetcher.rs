use std::collections::HashMap;

use chrono::{DateTime, Duration, NaiveDate, Utc};
use serde_json::Value;

use crate::types::WeatherRecord;

const BASE_URL: &str = "https://api.weather.gov";

// Type aliases for complex observation types
type Observation = (DateTime<Utc>, Option<f64>, Option<f64>, Option<f64>);
type DailyData = (Option<f64>, Option<f64>, Option<f64>);

#[derive(Clone)]
pub struct NWSFetcher {
    client: reqwest::blocking::Client,
    stations: HashMap<String, String>,
}

impl NWSFetcher {
    pub fn new() -> Self {
        Self {
            client: crate::data_pipeline::build_client(30),
            stations: default_stations(),
        }
    }

    pub fn fetch_daily_observations(
        &self,
        station_id: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Vec<WeatherRecord> {
        let mut all_observations: Vec<Observation> = Vec::new();

        let mut current = start_date;
        while current < end_date {
            let chunk_end = (current + Duration::days(7)).min(end_date);
            let mut chunk = self.fetch_observations_chunk(station_id, current, chunk_end);
            all_observations.append(&mut chunk);
            current = chunk_end;
        }

        self.aggregate_to_daily(all_observations, station_id)
    }

    fn fetch_observations_chunk(
        &self,
        station_id: &str,
        start: NaiveDate,
        end: NaiveDate,
    ) -> Vec<Observation> {
        let url = format!("{BASE_URL}/stations/{station_id}/observations");
        let query = [
            (
                "start",
                format!("{}T00:00:00Z", start.format("%Y-%m-%d")),
            ),
            (
                "end",
                format!("{}T23:59:59Z", end.format("%Y-%m-%d")),
            ),
        ];

        let Ok(resp) = self.client.get(url).query(&query).send() else {
            return Vec::new();
        };
        let Ok(json) = resp.json::<Value>() else {
            return Vec::new();
        };

        let mut out = Vec::new();
        for feature in json
            .get("features")
            .and_then(|f| f.as_array())
            .cloned()
            .unwrap_or_default()
        {
            let props = feature.get("properties").cloned().unwrap_or(Value::Null);
            let ts = props.get("timestamp").and_then(|v| v.as_str());
            let Some(ts) = ts else {
                continue;
            };
            let Ok(dt) = DateTime::parse_from_rfc3339(ts) else {
                continue;
            };
            let dt = dt.with_timezone(&Utc);
            let temp = props
                .get("temperature")
                .and_then(|v| v.get("value"))
                .and_then(|v| v.as_f64());
            let wind = props
                .get("windSpeed")
                .and_then(|v| v.get("value"))
                .and_then(|v| v.as_f64())
                .map(|x| x / 3.6);
            let precip = props
                .get("precipitationLastHour")
                .and_then(|v| v.get("value"))
                .and_then(|v| v.as_f64())
                .or(Some(0.0));

            out.push((dt, temp, wind, precip));
        }

        out
    }

    fn aggregate_to_daily(
        &self,
        observations: Vec<Observation>,
        station_id: &str,
    ) -> Vec<WeatherRecord> {
        let mut by_day: HashMap<NaiveDate, Vec<DailyData>> = HashMap::new();
        for (dt, temp, wind, precip) in observations {
            by_day.entry(dt.date_naive()).or_default().push((temp, wind, precip));
        }

        let mut days: Vec<NaiveDate> = by_day.keys().copied().collect();
        days.sort();

        let mut out = Vec::new();
        for day in days {
            let rows = by_day.remove(&day).unwrap_or_default();
            let temps: Vec<f64> = rows.iter().filter_map(|(t, _, _)| *t).collect();
            let winds: Vec<f64> = rows.iter().filter_map(|(_, w, _)| *w).collect();
            let precips: Vec<f64> = rows.iter().filter_map(|(_, _, p)| *p).collect();

            let mut rec = WeatherRecord::new(day, format!("NWS_{station_id}"));
            rec.temperature_max = temps.iter().copied().reduce(f64::max);
            rec.temperature_min = temps.iter().copied().reduce(f64::min);
            rec.temperature_mean = if temps.is_empty() {
                None
            } else {
                Some(temps.iter().sum::<f64>() / temps.len() as f64)
            };
            rec.precipitation_total = Some(precips.iter().sum());
            rec.wind_speed_mean = if winds.is_empty() {
                None
            } else {
                Some(winds.iter().sum::<f64>() / winds.len() as f64)
            };
            out.push(rec);
        }

        out
    }

    pub fn fetch_location(&self, location_key: &str, start_date: NaiveDate, end_date: NaiveDate) -> Vec<WeatherRecord> {
        let Some(station_id) = self.stations.get(location_key) else {
            return Vec::new();
        };
        self.fetch_daily_observations(station_id, start_date, end_date)
    }
}

impl Default for NWSFetcher {
    fn default() -> Self {
        Self::new()
    }
}

fn default_stations() -> HashMap<String, String> {
    [
        ("NYC", "KNYC"),
        ("LA", "KLAX"),
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
