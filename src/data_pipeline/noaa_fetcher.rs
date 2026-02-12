use std::collections::HashMap;

use chrono::NaiveDate;
use serde_json::Value;

use crate::config::{noaa_api_key, noaa_base_url};
use crate::types::RawObservation;

#[derive(Clone)]
pub struct NOAAFetcher {
    api_key: String,
    base_url: String,
    client: reqwest::blocking::Client,
}

impl NOAAFetcher {
    pub fn new() -> Self {
        Self {
            api_key: noaa_api_key(),
            base_url: noaa_base_url(),
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap_or_else(|_| reqwest::blocking::Client::new()),
        }
    }

    pub fn fetch_daily_observations(
        &self,
        station_id: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
        data_types: Option<Vec<&str>>,
    ) -> Vec<RawObservation> {
        let types = data_types.unwrap_or_else(|| vec!["TMAX", "TMIN", "PRCP", "AWND"]);
        let url = format!("{}data", self.base_url);

        let query = [
            ("datasetid", "GHCND".to_string()),
            ("stationid", station_id.to_string()),
            ("startDate", start_date.to_string()),
            ("endDate", end_date.to_string()),
            ("datatypeid", types.join(",")),
            ("limit", "1000".to_string()),
            ("offset", "1".to_string()),
        ];

        let mut request = self.client.get(url).query(&query);
        if !self.api_key.is_empty() {
            request = request.header("token", self.api_key.clone());
        }

        let Ok(resp) = request.send() else {
            return Vec::new();
        };
        let Ok(json) = resp.json::<Value>() else {
            return Vec::new();
        };

        json.get("results")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default()
            .iter()
            .filter_map(|r| {
                let date = r.get("date").and_then(|x| x.as_str())?;
                let datatype = r.get("datatype").and_then(|x| x.as_str())?;
                let value = r.get("value").and_then(|x| x.as_f64())?;
                let day = NaiveDate::parse_from_str(&date[..10], "%Y-%m-%d").ok()?;

                Some(RawObservation {
                    date: day,
                    station_id: station_id.to_string(),
                    datatype: datatype.to_string(),
                    value,
                })
            })
            .collect()
    }

    pub fn fetch_bulk_data(
        &self,
        station_ids: &[String],
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Vec<RawObservation> {
        let mut out = Vec::new();
        for station_id in station_ids {
            out.extend(self.fetch_daily_observations(station_id, start_date, end_date, None));
        }
        out
    }

    pub fn fetch_major_us_stations(&self, start_date: NaiveDate, end_date: NaiveDate) -> Vec<RawObservation> {
        let ids = major_stations().keys().cloned().collect::<Vec<_>>();
        self.fetch_bulk_data(&ids, start_date, end_date)
    }

    pub fn get_station_metadata(&self, station_id: &str) -> Option<StationMeta> {
        major_stations().get(station_id).cloned()
    }
}

impl Default for NOAAFetcher {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct StationMeta {
    pub name: &'static str,
    pub lat: f64,
    pub lon: f64,
}

pub fn major_stations() -> HashMap<String, StationMeta> {
    [
        (
            "GHCND:USW00023023",
            StationMeta {
                name: "NEW YORK, NY",
                lat: 40.77,
                lon: -73.87,
            },
        ),
        (
            "GHCND:USW00094846",
            StationMeta {
                name: "CHICAGO, IL",
                lat: 41.99,
                lon: -87.93,
            },
        ),
        (
            "GHCND:USW00012918",
            StationMeta {
                name: "LOS ANGELES, CA",
                lat: 34.05,
                lon: -118.24,
            },
        ),
        (
            "GHCND:USW00013060",
            StationMeta {
                name: "DALLAS, TX",
                lat: 32.85,
                lon: -96.85,
            },
        ),
        (
            "GHCND:USW00014827",
            StationMeta {
                name: "DENVER, CO",
                lat: 39.74,
                lon: -104.99,
            },
        ),
        (
            "GHCND:USW00093017",
            StationMeta {
                name: "MIAMI, FL",
                lat: 25.80,
                lon: -80.27,
            },
        ),
        (
            "GHCND:USW00024234",
            StationMeta {
                name: "BOSTON, MA",
                lat: 42.36,
                lon: -71.01,
            },
        ),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect()
}
