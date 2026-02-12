use std::collections::HashMap;

use rand::{Rng, SeedableRng};

use crate::types::{SimulatedMarket, WeatherRecord};
use crate::utils::clip;

pub fn fahrenheit_to_celsius(f: f64) -> f64 {
    (f - 32.0) * 5.0 / 9.0
}

#[derive(Debug, Clone)]
pub struct MarketSimulator {
    pub temp_thresholds_f: Vec<i32>,
    pub market_noise_std: f64,
    rng: rand::rngs::StdRng,
}

impl MarketSimulator {
    pub fn new(temp_thresholds_f: Option<Vec<i32>>, market_noise_std: f64, seed: u64) -> Self {
        Self {
            temp_thresholds_f: temp_thresholds_f.unwrap_or_else(|| vec![32, 50, 60, 70, 80, 90]),
            market_noise_std,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }

    pub fn generate_markets(&mut self, weather_data: &[WeatherRecord], city: &str) -> Vec<SimulatedMarket> {
        if weather_data.is_empty() {
            return Vec::new();
        }

        let mut sorted = weather_data.to_vec();
        sorted.sort_by_key(|r| r.date);
        let clim_probs = self.compute_climatological_probs(&sorted, 30);

        let mut markets = Vec::new();

        for (idx, row) in sorted.iter().enumerate() {
            let Some(temp_max_c) = row.temperature_max else {
                continue;
            };
            let temp_max_f = temp_max_c * 9.0 / 5.0 + 32.0;

            let thresholds = self.temp_thresholds_f.clone();
            for threshold in thresholds {
                let actual_outcome = if temp_max_f >= threshold as f64 { 1.0 } else { 0.0 };
                let key = format!("temp_above_{threshold}f");
                let clim_prob = clim_probs
                    .get(&key)
                    .and_then(|m| m.get(&idx))
                    .copied()
                    .unwrap_or(0.5);

                markets.push(SimulatedMarket {
                    date: row.date,
                    market_id: format!("{city}_temp_{threshold}f_{}", row.date.format("%Y%m%d")),
                    market_title: format!("Will {city} max temperature exceed {threshold}°F?"),
                    market_type: "temperature".to_string(),
                    threshold: threshold as f64,
                    market_price: self.generate_market_price(clim_prob),
                    actual_outcome,
                    city: city.to_string(),
                });
            }

            if let Some(precip) = row.precipitation_total {
                let actual_rain = if precip > 0.1 { 1.0 } else { 0.0 };
                let clim_prob = clim_probs
                    .get("precipitation")
                    .and_then(|m| m.get(&idx))
                    .copied()
                    .unwrap_or(0.3);

                markets.push(SimulatedMarket {
                    date: row.date,
                    market_id: format!("{city}_rain_{}", row.date.format("%Y%m%d")),
                    market_title: format!("Will it rain in {city} today?"),
                    market_type: "precipitation".to_string(),
                    threshold: 0.1,
                    market_price: self.generate_market_price(clim_prob),
                    actual_outcome: actual_rain,
                    city: city.to_string(),
                });
            }
        }

        markets
    }

    fn compute_climatological_probs(
        &self,
        weather_data: &[WeatherRecord],
        window: usize,
    ) -> HashMap<String, HashMap<usize, f64>> {
        let mut probs = HashMap::new();

        for threshold in &self.temp_thresholds_f {
            let key = format!("temp_above_{threshold}f");
            let exceeded: Vec<f64> = weather_data
                .iter()
                .map(|w| {
                    if w.temperature_max.map(|t| t * 9.0 / 5.0 + 32.0 >= *threshold as f64).unwrap_or(false)
                    {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            probs.insert(key, rolling_probabilities(&exceeded, window, 7));
        }

        let rained: Vec<f64> = weather_data
            .iter()
            .map(|w| if w.precipitation_total.unwrap_or(0.0) > 0.1 { 1.0 } else { 0.0 })
            .collect();
        probs.insert(
            "precipitation".to_string(),
            rolling_probabilities(&rained, window, 7),
        );

        probs
    }

    fn generate_market_price(&mut self, climatological_prob: f64) -> f64 {
        let noise = rand_distr::Normal::new(0.0, self.market_noise_std)
            .ok()
            .map(|dist| self.rng.sample(dist))
            .unwrap_or(0.0);
        clip(climatological_prob + noise, 0.02, 0.98)
    }
}

fn rolling_probabilities(values: &[f64], window: usize, min_periods: usize) -> HashMap<usize, f64> {
    let overall = if values.is_empty() {
        0.5
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    };

    let mut out = HashMap::new();
    for i in 0..values.len() {
        let start = i.saturating_sub(window.saturating_sub(1));
        let slice = &values[start..=i];
        let val = if slice.len() >= min_periods {
            slice.iter().sum::<f64>() / slice.len() as f64
        } else {
            overall
        };
        out.insert(i, val);
    }
    out
}
