use rand::SeedableRng;
use rand_distr::{Beta, Distribution};

use crate::config::{bayesian_model_params, BayesianModelParams};
use crate::types::WeatherRecord;
use crate::utils::{mean, percentile, sample_normal};

#[derive(Debug, Clone)]
pub struct BayesianWeatherModel {
    pub params: BayesianModelParams,
    pub is_trained: bool,
    pub temp_posterior_mu: f64,
    pub temp_posterior_sigma: f64,
    pub precip_posterior_alpha: f64,
    pub precip_posterior_beta: f64,
    pub precip_posterior_mean: f64,
}

impl Default for BayesianWeatherModel {
    fn default() -> Self {
        Self::new(None)
    }
}

impl BayesianWeatherModel {
    pub fn new(params: Option<BayesianModelParams>) -> Self {
        Self {
            params: params.unwrap_or_else(bayesian_model_params),
            is_trained: false,
            temp_posterior_mu: 0.0,
            temp_posterior_sigma: 1.0,
            precip_posterior_alpha: 1.0,
            precip_posterior_beta: 1.0,
            precip_posterior_mean: 0.5,
        }
    }

    pub fn train(&mut self, observations: &[WeatherRecord]) -> Result<(), String> {
        let temps: Vec<f64> = observations
            .iter()
            .filter_map(|r| r.temperature_mean)
            .collect();

        let precips: Vec<bool> = observations
            .iter()
            .filter_map(|r| r.precipitation_total)
            .map(|p| p > 0.0)
            .collect();

        self.train_from_arrays(&temps, &precips)
    }

    pub fn train_from_arrays(
        &mut self,
        temperatures: &[f64],
        precipitation_occurred: &[bool],
    ) -> Result<(), String> {
        if temperatures.len() < 2 {
            return Err("need at least two temperature observations".to_string());
        }
        if precipitation_occurred.is_empty() {
            return Err("need precipitation observations".to_string());
        }

        self.train_temperature_model(temperatures)?;
        self.train_precipitation_model(precipitation_occurred);
        self.is_trained = true;
        Ok(())
    }

    fn train_temperature_model(&mut self, temperatures: &[f64]) -> Result<(), String> {
        let n = temperatures.len() as f64;
        let sample_mean = mean(temperatures);
        let sample_var = {
            let mu = sample_mean;
            let sse = temperatures
                .iter()
                .map(|t| {
                    let d = t - mu;
                    d * d
                })
                .sum::<f64>();
            sse / (temperatures.len() - 1) as f64
        };

        if sample_var <= 0.0 {
            return Err("temperature variance must be positive".to_string());
        }

        let prior_mu = self.params.temperature_prior_mu;
        let prior_sigma = self.params.temperature_prior_sigma;

        let prior_precision = 1.0 / prior_sigma.powi(2);
        let data_precision = n / sample_var;
        let posterior_precision = prior_precision + data_precision;

        self.temp_posterior_mu =
            (prior_precision * prior_mu + data_precision * sample_mean) / posterior_precision;
        self.temp_posterior_sigma = (1.0 / posterior_precision).sqrt();

        Ok(())
    }

    fn train_precipitation_model(&mut self, precipitation_occurred: &[bool]) {
        let alpha = self.params.precipitation_prior_alpha;
        let beta = self.params.precipitation_prior_beta;

        let successes = precipitation_occurred.iter().filter(|x| **x).count() as f64;
        let failures = precipitation_occurred.len() as f64 - successes;

        self.precip_posterior_alpha = alpha + successes;
        self.precip_posterior_beta = beta + failures;
        self.precip_posterior_mean =
            self.precip_posterior_alpha / (self.precip_posterior_alpha + self.precip_posterior_beta);
    }

    pub fn predict_temperature_exceeds(
        &self,
        threshold: f64,
        n_samples: usize,
    ) -> Result<f64, String> {
        if !self.is_trained {
            return Err("model must be trained before prediction".to_string());
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let samples = sample_normal(
            self.temp_posterior_mu,
            self.temp_posterior_sigma,
            n_samples,
            &mut rng,
        );
        let count = samples.iter().filter(|s| **s > threshold).count();
        Ok(count as f64 / n_samples as f64)
    }

    pub fn predict_temperature_range(
        &self,
        lower: f64,
        upper: f64,
        n_samples: usize,
    ) -> Result<f64, String> {
        if !self.is_trained {
            return Err("model must be trained before prediction".to_string());
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(43);
        let samples = sample_normal(
            self.temp_posterior_mu,
            self.temp_posterior_sigma,
            n_samples,
            &mut rng,
        );

        let count = samples
            .iter()
            .filter(|s| **s >= lower && **s <= upper)
            .count();
        Ok(count as f64 / n_samples as f64)
    }

    pub fn predict_precipitation_probability(
        &self,
        n_samples: usize,
    ) -> Result<(f64, (f64, f64)), String> {
        if !self.is_trained {
            return Err("model must be trained before prediction".to_string());
        }

        let beta_dist = Beta::new(self.precip_posterior_alpha, self.precip_posterior_beta)
            .map_err(|e| format!("invalid beta params: {e}"))?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(44);
        let samples: Vec<f64> = (0..n_samples).map(|_| beta_dist.sample(&mut rng)).collect();

        let prob = mean(&samples);
        let low = percentile(&samples, 2.5);
        let high = percentile(&samples, 97.5);

        Ok((prob, (low, high)))
    }

    pub fn forecast_event_probabilities(
        &self,
        temperature_thresholds: Option<Vec<(String, f64)>>,
        n_samples: usize,
    ) -> Result<std::collections::HashMap<String, f64>, String> {
        if !self.is_trained {
            return Err("model must be trained before prediction".to_string());
        }

        let defaults = vec![
            ("temp_above_90f".to_string(), 32.2),
            ("temp_above_95f".to_string(), 35.0),
            ("temp_below_32f".to_string(), 0.0),
            ("temp_below_0f".to_string(), -17.8),
        ];

        let thresholds = temperature_thresholds.unwrap_or(defaults);
        let mut forecast = std::collections::HashMap::new();

        for (name, threshold) in thresholds {
            forecast.insert(
                name,
                self.predict_temperature_exceeds(threshold, n_samples)?,
            );
        }

        let (precip_prob, (low, high)) = self.predict_precipitation_probability(n_samples)?;
        forecast.insert("precipitation".to_string(), precip_prob);
        forecast.insert("precipitation_ci_lower".to_string(), low);
        forecast.insert("precipitation_ci_upper".to_string(), high);

        Ok(forecast)
    }

    pub fn get_posterior_intervals(
        &self,
        credible_level: f64,
        n_samples: usize,
    ) -> Result<std::collections::HashMap<String, (f64, f64)>, String> {
        if !self.is_trained {
            return Err("model must be trained before prediction".to_string());
        }

        let alpha = (1.0 - credible_level) / 2.0;

        let mut rng = rand::rngs::StdRng::seed_from_u64(45);
        let temp_samples = sample_normal(
            self.temp_posterior_mu,
            self.temp_posterior_sigma,
            n_samples,
            &mut rng,
        );

        let beta_dist = Beta::new(self.precip_posterior_alpha, self.precip_posterior_beta)
            .map_err(|e| format!("invalid beta params: {e}"))?;
        let precip_samples: Vec<f64> = (0..n_samples).map(|_| beta_dist.sample(&mut rng)).collect();

        let mut out = std::collections::HashMap::new();
        out.insert(
            "temperature".to_string(),
            (
                percentile(&temp_samples, alpha * 100.0),
                percentile(&temp_samples, (1.0 - alpha) * 100.0),
            ),
        );
        out.insert(
            "precipitation".to_string(),
            (
                percentile(&precip_samples, alpha * 100.0),
                percentile(&precip_samples, (1.0 - alpha) * 100.0),
            ),
        );

        Ok(out)
    }
}
