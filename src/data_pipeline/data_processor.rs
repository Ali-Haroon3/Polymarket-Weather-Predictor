use std::collections::HashMap;

use chrono::NaiveDate;

use crate::config::data_pipeline_params;
use crate::database::{connection::DatabaseSession, models::WeatherObservation};
use crate::types::{RawObservation, WeatherRecord};

#[derive(Debug, Clone)]
pub struct DataProcessor {
    pub validation_threshold: f64,
    pub missing_data_threshold: f64,
}

impl Default for DataProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl DataProcessor {
    pub fn new() -> Self {
        let p = data_pipeline_params();
        Self {
            validation_threshold: p.validation_threshold,
            missing_data_threshold: p.missing_data_threshold,
        }
    }

    pub fn process_raw_observations(&self, raw: &[RawObservation]) -> Vec<WeatherRecord> {
        let mut map: HashMap<(NaiveDate, String), WeatherRecord> = HashMap::new();

        for obs in raw {
            let key = (obs.date, obs.station_id.clone());
            let entry = map
                .entry(key.clone())
                .or_insert_with(|| WeatherRecord::new(obs.date, obs.station_id.clone()));

            match obs.datatype.as_str() {
                "TMAX" => entry.temperature_max = Some(obs.value),
                "TMIN" => entry.temperature_min = Some(obs.value),
                "PRCP" => entry.precipitation_total = Some(obs.value),
                "AWND" => entry.wind_speed_mean = Some(obs.value),
                _ => {}
            }
        }

        let mut out: Vec<WeatherRecord> = map.into_values().collect();
        for row in &mut out {
            if row.temperature_mean.is_none() {
                row.temperature_mean = match (row.temperature_max, row.temperature_min) {
                    (Some(mx), Some(mn)) => Some((mx + mn) / 2.0),
                    _ => None,
                };
            }
        }
        out.sort_by_key(|r| r.date);
        out
    }

    pub fn validate_observations(&self, rows: &[WeatherRecord]) -> (Vec<WeatherRecord>, Vec<f64>) {
        if rows.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let n = rows.len() as f64;
        let missing_temp = rows.iter().filter(|r| r.temperature_mean.is_none()).count() as f64 / n;
        let missing_precip = rows
            .iter()
            .filter(|r| r.precipitation_total.is_none())
            .count() as f64
            / n;

        let mut out = Vec::new();
        let mut quality = Vec::new();

        for row in rows {
            let mut score = 1.0;
            if missing_temp > self.missing_data_threshold {
                score *= 1.0 - missing_temp;
            }
            if missing_precip > self.missing_data_threshold {
                score *= 1.0 - missing_precip;
            }

            if let (Some(mx), Some(mn)) = (row.temperature_max, row.temperature_min) {
                if mx < mn {
                    score *= 0.5;
                }
            }

            if let Some(temp) = row.temperature_mean {
                if !(-50.0..=50.0).contains(&temp) {
                    score *= 0.1;
                }
            }

            if let Some(p) = row.precipitation_total {
                if p < 0.0 {
                    score = 0.0;
                }
            }

            let mut r = row.clone();
            r.data_quality_score = Some(score);
            r.is_validated = score >= self.validation_threshold;
            out.push(r);
            quality.push(score);
        }

        (out, quality)
    }

    pub fn normalize_features(&self, rows: &[WeatherRecord]) -> Vec<HashMap<String, f64>> {
        rows.iter()
            .map(|r| {
                let mut m = HashMap::new();
                if let Some(t) = r.temperature_mean {
                    m.insert("temperature_mean_norm".to_string(), ((t + 20.0) / 65.0).clamp(0.0, 1.0));
                }
                if let Some(p) = r.precipitation_total {
                    m.insert("precipitation_norm".to_string(), (p / 100.0).clamp(0.0, 1.0));
                }
                m
            })
            .collect()
    }

    pub fn save_to_database(
        &self,
        rows: &[WeatherRecord],
        station_name: &str,
        latitude: f64,
        longitude: f64,
        session: &mut DatabaseSession,
    ) -> usize {
        let mut count = 0;
        for row in rows {
            let model = WeatherObservation {
                id: count as i64 + 1,
                station_id: row.station_id.clone(),
                station_name: station_name.to_string(),
                latitude,
                longitude,
                observation_date: row.date,
                temperature_max: row.temperature_max,
                temperature_min: row.temperature_min,
                temperature_mean: row.temperature_mean,
                precipitation_total: row.precipitation_total,
                wind_speed_mean: row.wind_speed_mean,
                data_quality_score: row.data_quality_score.unwrap_or(1.0),
                is_validated: row.is_validated,
            };
            session.weather_observations.push(model);
            count += 1;
        }
        count
    }

    pub fn generate_statistics(&self, rows: &[WeatherRecord]) -> HashMap<String, f64> {
        let total = rows.len() as f64;
        if total == 0.0 {
            return HashMap::new();
        }

        let valid = rows.iter().filter(|r| r.is_validated).count() as f64;
        let temps: Vec<f64> = rows.iter().filter_map(|r| r.temperature_mean).collect();
        let precips: Vec<f64> = rows.iter().filter_map(|r| r.precipitation_total).collect();

        let mut out = HashMap::new();
        out.insert("total_records".to_string(), total);
        out.insert("valid_records".to_string(), valid);
        out.insert("temperature_mean".to_string(), avg(&temps));
        out.insert("temperature_min".to_string(), temps.iter().copied().reduce(f64::min).unwrap_or(0.0));
        out.insert("temperature_max".to_string(), temps.iter().copied().reduce(f64::max).unwrap_or(0.0));
        out.insert("precipitation_mean".to_string(), avg(&precips));
        out.insert("precipitation_max".to_string(), precips.iter().copied().reduce(f64::max).unwrap_or(0.0));
        out.insert(
            "avg_quality_score".to_string(),
            rows.iter()
                .filter_map(|r| r.data_quality_score)
                .sum::<f64>()
                / rows.len().max(1) as f64,
        );
        out
    }
}

fn avg(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}
