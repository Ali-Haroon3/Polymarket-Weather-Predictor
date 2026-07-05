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
            client: crate::data_pipeline::build_client(30),
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

}

impl Default for NOAAFetcher {
    fn default() -> Self {
        Self::new()
    }
}

