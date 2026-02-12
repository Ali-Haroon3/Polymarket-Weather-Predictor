use std::collections::HashMap;

use chrono::NaiveDate;

use crate::data_pipeline::noaa_fetcher::NOAAFetcher;
use crate::data_pipeline::nws_fetcher::NWSFetcher;
use crate::data_pipeline::open_meteo_fetcher::OpenMeteoFetcher;
use crate::data_pipeline::openweathermap_fetcher::OpenWeatherMapFetcher;
use crate::data_pipeline::tomorrow_io_fetcher::TomorrowIOFetcher;
use crate::data_pipeline::visual_crossing_fetcher::VisualCrossingFetcher;
use crate::data_pipeline::weatherapi_fetcher::WeatherAPIFetcher;
use crate::types::{RawObservation, WeatherRecord};

#[derive(Clone)]
pub struct MultiSourceAggregator {
    pub open_meteo: OpenMeteoFetcher,
    pub nws: NWSFetcher,
    pub noaa: NOAAFetcher,
    pub owm: OpenWeatherMapFetcher,
    pub vc: VisualCrossingFetcher,
    pub wapi: WeatherAPIFetcher,
    pub tio: TomorrowIOFetcher,
}

impl MultiSourceAggregator {
    pub fn new() -> Self {
        Self {
            open_meteo: OpenMeteoFetcher::new(),
            nws: NWSFetcher::new(),
            noaa: NOAAFetcher::new(),
            owm: OpenWeatherMapFetcher::new(None),
            vc: VisualCrossingFetcher::new(None),
            wapi: WeatherAPIFetcher::new(None),
            tio: TomorrowIOFetcher::new(None),
        }
    }

    pub fn fetch_all_sources(
        &self,
        location_key: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> HashMap<String, Vec<WeatherRecord>> {
        let mut results = HashMap::new();

        let om = self.open_meteo.fetch_location(location_key, start_date, end_date);
        if !om.is_empty() {
            results.insert("open_meteo".to_string(), om);
        }

        let nws = self.nws.fetch_location(location_key, start_date, end_date);
        if !nws.is_empty() {
            results.insert("nws".to_string(), nws);
        }

        if let Some(station) = city_noaa_stations().get(location_key) {
            let raw = self
                .noaa
                .fetch_daily_observations(station, start_date, end_date, None);
            let normalized = normalize_noaa_data(&raw, station);
            if !normalized.is_empty() {
                results.insert("noaa".to_string(), normalized);
            }
        }

        if self.owm.is_available() {
            let rows = self.owm.fetch_location(location_key, start_date, end_date);
            if !rows.is_empty() {
                results.insert("openweathermap".to_string(), rows);
            }
        }

        if self.vc.is_available() {
            let rows = self.vc.fetch_location(location_key, start_date, end_date);
            if !rows.is_empty() {
                results.insert("visual_crossing".to_string(), rows);
            }
        }

        if self.wapi.is_available() {
            let rows = self.wapi.fetch_location(location_key, start_date, end_date);
            if !rows.is_empty() {
                results.insert("weatherapi".to_string(), rows);
            }
        }

        if self.tio.is_available() {
            let rows = self.tio.fetch_location(location_key, start_date, end_date);
            if !rows.is_empty() {
                results.insert("tomorrow_io".to_string(), rows);
            }
        }

        results
    }

    pub fn aggregate(
        &self,
        location_key: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Vec<WeatherRecord> {
        let source_data = self.fetch_all_sources(location_key, start_date, end_date);
        if source_data.is_empty() {
            return Vec::new();
        }

        let mut date_map: HashMap<NaiveDate, Vec<(String, WeatherRecord)>> = HashMap::new();
        for (source, rows) in source_data {
            for row in rows {
                date_map
                    .entry(row.date)
                    .or_default()
                    .push((source.clone(), row));
            }
        }

        let mut dates: Vec<NaiveDate> = date_map.keys().copied().collect();
        dates.sort();

        let mut out = Vec::new();
        for date in dates {
            let rows = date_map.remove(&date).unwrap_or_default();
            let mut source_names = Vec::new();
            let mut temp_max = Vec::new();
            let mut temp_min = Vec::new();
            let mut temp_mean = Vec::new();
            let mut precip = Vec::new();
            let mut wind = Vec::new();

            for (source, row) in rows {
                source_names.push(source);
                if let Some(v) = row.temperature_max {
                    temp_max.push(v);
                }
                if let Some(v) = row.temperature_min {
                    temp_min.push(v);
                }
                if let Some(v) = row.temperature_mean {
                    temp_mean.push(v);
                }
                if let Some(v) = row.precipitation_total {
                    precip.push(v);
                }
                if let Some(v) = row.wind_speed_mean {
                    wind.push(v);
                }
            }

            source_names.sort();
            source_names.dedup();

            let mut agg = WeatherRecord::new(date, format!("MULTI_{location_key}"));
            agg.location = Some(location_key.to_string());
            agg.temperature_max = mean_opt(&temp_max);
            agg.temperature_min = mean_opt(&temp_min);
            agg.temperature_mean = mean_opt(&temp_mean);
            agg.precipitation_total = mean_opt(&precip);
            agg.wind_speed_mean = mean_opt(&wind);
            agg.source_count = Some(source_names.len() as u32);
            agg.sources = Some(source_names.join(","));
            agg.data_quality_score = Some(1.0);
            agg.is_validated = source_names.len() >= 2;
            out.push(agg);
        }

        out
    }

    pub fn aggregate_multiple(
        &self,
        location_keys: &[String],
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> HashMap<String, Vec<WeatherRecord>> {
        let mut out = HashMap::new();

        for key in location_keys {
            let rows = self.aggregate(key, start_date, end_date);
            if !rows.is_empty() {
                out.insert(key.clone(), rows);
            }
        }

        out
    }
}

impl Default for MultiSourceAggregator {
    fn default() -> Self {
        Self::new()
    }
}

fn mean_opt(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

fn normalize_noaa_data(raw: &[RawObservation], station_id: &str) -> Vec<WeatherRecord> {
    let mut map: HashMap<NaiveDate, WeatherRecord> = HashMap::new();

    for obs in raw {
        let entry = map
            .entry(obs.date)
            .or_insert_with(|| WeatherRecord::new(obs.date, format!("NOAA_{station_id}")));

        match obs.datatype.as_str() {
            "TMAX" => entry.temperature_max = Some(obs.value / 10.0),
            "TMIN" => entry.temperature_min = Some(obs.value / 10.0),
            "PRCP" => entry.precipitation_total = Some(obs.value / 10.0),
            "AWND" => entry.wind_speed_mean = Some(obs.value / 10.0),
            _ => {}
        }
    }

    let mut out: Vec<WeatherRecord> = map.into_values().collect();
    out.sort_by_key(|r| r.date);
    for row in &mut out {
        row.temperature_mean = match (row.temperature_max, row.temperature_min) {
            (Some(mx), Some(mn)) => Some((mx + mn) / 2.0),
            _ => None,
        };
    }
    out
}

fn city_noaa_stations() -> HashMap<String, String> {
    [
        ("NYC", "GHCND:USW00023023"),
        ("LA", "GHCND:USW00012918"),
        ("Chicago", "GHCND:USW00094846"),
        ("Dallas", "GHCND:USW00013060"),
        ("Denver", "GHCND:USW00014827"),
        ("Miami", "GHCND:USW00093017"),
        ("Boston", "GHCND:USW00024234"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}
