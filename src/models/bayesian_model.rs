use chrono::NaiveDate;
use rand::SeedableRng;
use rand_distr::{Beta, Distribution};

use crate::config::{bayesian_model_params, BayesianModelParams};
use crate::types::WeatherRecord;
use crate::utils::{mean, normal_cdf, percentile, sample_normal};

#[derive(Debug, Clone)]
pub struct BayesianWeatherModel {
    pub params: BayesianModelParams,
    pub is_trained: bool,
    pub temp_posterior_mu: f64,
    /// Standard error of the posterior *mean* (sqrt(tau_n^2)). Used by the mean credible band.
    pub temp_posterior_sigma: f64,
    /// Spread of a new daily-high observation: sqrt(tau_n^2 + sample_var). This is the
    /// posterior *predictive* sigma — what every "will the high exceed/fall in X" market needs.
    pub temp_predictive_sigma: f64,
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
            temp_predictive_sigma: 1.0,
            precip_posterior_alpha: 1.0,
            precip_posterior_beta: 1.0,
            precip_posterior_mean: 0.5,
        }
    }

    pub fn train(&mut self, observations: &[WeatherRecord]) -> Result<(), String> {
        // Markets resolve on the daily HIGH ("highest temperature ..."), so train on
        // temperature_max, falling back to temperature_mean only on rows that lack a max so
        // mean-only feeds still train. ponytail: a feed mixing max-rows and mean-only-rows will
        // blend the two series here; acceptable since real feeds populate max consistently.
        let temps: Vec<f64> = observations
            .iter()
            .filter_map(|r| r.temperature_max.or(r.temperature_mean))
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

        // NaN/inf compare false against <= 0.0, so test finiteness explicitly to keep a bad
        // value from leaking through the (unclipped) predictive probabilities downstream.
        if !sample_var.is_finite() || sample_var <= 0.0 {
            return Err("temperature variance must be positive and finite".to_string());
        }

        let prior_mu = self.params.temperature_prior_mu;
        let prior_sigma = self.params.temperature_prior_sigma;

        let prior_precision = 1.0 / prior_sigma.powi(2);
        let data_precision = n / sample_var;
        let posterior_precision = prior_precision + data_precision;

        self.temp_posterior_mu =
            (prior_precision * prior_mu + data_precision * sample_mean) / posterior_precision;

        // Posterior of the MEAN has variance tau_n^2 = 1/posterior_precision.
        let posterior_mean_var = 1.0 / posterior_precision;
        self.temp_posterior_sigma = posterior_mean_var.sqrt();

        // THE FIX: a *new* daily high is mean + noise, so the predictive variance adds back the
        // day-to-day spread: predictive_var = tau_n^2 + sample_var. The old code used only tau_n^2
        // (the standard error of the mean, ~1 degC), which made tail probabilities wildly
        // overconfident. Predictive sigma is ~the sample SD (~8 degC).
        self.temp_predictive_sigma = (posterior_mean_var + sample_var).sqrt();

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

    /// Dated, target-aware training. Fits a local linear trend of the daily high on day-offset
    /// from `target`, then evaluates the predictive Normal AT the target day — removing the
    /// seasonal lag a stationary trailing-window mean suffers. Writes the same three temp_* fields
    /// as `train`, so all `prob_*` pricing is unchanged. (Precipitation is trained stationary.)
    pub fn train_for_target(
        &mut self,
        observations: &[WeatherRecord],
        target: NaiveDate,
    ) -> Result<(), String> {
        // (day-offset-from-target, daily high) pairs; same max-or-mean fallback as train().
        let dated: Vec<(f64, f64)> = observations
            .iter()
            .filter_map(|r| {
                r.temperature_max
                    .or(r.temperature_mean)
                    .map(|y| ((r.date - target).num_days() as f64, y))
            })
            .collect();

        let precips: Vec<bool> = observations
            .iter()
            .filter_map(|r| r.precipitation_total)
            .map(|p| p > 0.0)
            .collect();

        if precips.is_empty() {
            return Err("need precipitation observations".to_string());
        }

        self.fit_trend_temperature(&dated)?;
        self.train_precipitation_model(&precips);
        self.is_trained = true;
        Ok(())
    }

    /// Local linear detrend in day-offset space: fit `high = a + b·offset` by OLS, predict at the
    /// target day (offset 0) so `mu = a`. Predictive sigma keeps the step-1 philosophy — projection
    /// uncertainty + residual day-to-day variance, never the SE of the mean. Falls back to the
    /// stationary conjugate fit when a slope can't be identified (too few points or zero x-spread).
    fn fit_trend_temperature(&mut self, dated: &[(f64, f64)]) -> Result<(), String> {
        // Match the stationary path's contract: too few temperature points is a clean Err, never a
        // panic. (A window can have ≥14 records but <2 temperatures — e.g. precip-only NOAA days.)
        if dated.len() < 2 {
            return Err("need at least two temperature observations".to_string());
        }
        let ys: Vec<f64> = dated.iter().map(|&(_, y)| y).collect();
        let n = dated.len();
        // Below the walk-forward's own 14-record floor a slope is just noise; use the stationary fit.
        if n < 14 {
            return self.train_temperature_model(&ys);
        }

        let nf = n as f64;
        let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
        for &(x, y) in dated {
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
        let d = nf * sxx - sx * sx; // = n·Σ(x − x̄)² ≥ 0
        if d <= 1e-9 {
            // Singular design (all records share one date) — no slope to fit.
            return self.train_temperature_model(&ys);
        }

        let b = (nf * sxy - sx * sy) / d;
        let a = (sy - b * sx) / nf; // intercept = prediction at target (offset 0)

        let sse: f64 = dated
            .iter()
            .map(|&(x, y)| {
                let r = y - a - b * x;
                r * r
            })
            .sum();
        let s2 = sse / (nf - 2.0); // residual day-to-day variance, df = n−2
        if !s2.is_finite() || s2 <= 0.0 {
            return Err("temperature variance must be positive and finite".to_string());
        }

        // Slope-significance gate. SE(b) = sqrt(s2·n/D). Only detrend when the slope is statistically
        // real (|t| ≥ 2 ≈ 95%): a real spring/fall ramp scores t ~ 10–20 and is kept; a spurious slope
        // on flat (e.g. mid-summer) data scores |t| < 2 and collapses to the stationary fit — which
        // avoids the ~2°C per-window mean wander that a predictive band widens for but can't re-center,
        // and incidentally avoids over-extrapolating curvature near seasonal turning points.
        let se_b = (s2 * nf / d).sqrt();
        if !se_b.is_finite() || se_b <= 0.0 || b.abs() / se_b < 2.0 {
            return self.train_temperature_model(&ys);
        }

        let x_bar = sx / nf;
        let proj_var = s2 * (1.0 / nf + x_bar * x_bar * nf / d); // Var(fitted line) at offset 0

        // ponytail: weak prior (sigma^2=225) vs data precision at n>=14 is a no-op; skip the shrink.
        self.temp_posterior_mu = a;
        self.temp_posterior_sigma = proj_var.sqrt(); // trend-aware SE of the target-day mean
        self.temp_predictive_sigma = (s2 + proj_var).sqrt(); // projection + residual; step-1 philosophy
        Ok(())
    }

    /// The posterior predictive distribution of a new daily high: N(mu, predictive_sigma), in degC.
    fn temperature_predictive(&self) -> (f64, f64) {
        (self.temp_posterior_mu, self.temp_predictive_sigma)
    }

    /// P(daily high >= t). Analytic over the predictive Normal (degC). Infallible; prices the
    /// "T degC or higher" bucket form. For a continuous Normal, P(>=t) == P(>t).
    pub fn prob_at_least(&self, t: f64) -> f64 {
        let (mu, sigma) = self.temperature_predictive();
        if sigma <= 0.0 {
            return if t <= mu { 1.0 } else { 0.0 };
        }
        1.0 - normal_cdf((t - mu) / sigma)
    }

    /// P(daily high <= t). Prices the "T degC or below" bucket form.
    pub fn prob_at_most(&self, t: f64) -> f64 {
        let (mu, sigma) = self.temperature_predictive();
        if sigma <= 0.0 {
            return if t >= mu { 1.0 } else { 0.0 };
        }
        normal_cdf((t - mu) / sigma)
    }

    /// P(lo <= daily high <= hi). Prices an exact bucket ("be T degC" -> [T-0.5, T+0.5)) or a
    /// range ("between A-B"). A swapped interval (lo > hi) is a caller bug and yields 0.
    pub fn prob_between(&self, lo: f64, hi: f64) -> f64 {
        if lo > hi {
            return 0.0;
        }
        let (mu, sigma) = self.temperature_predictive();
        if sigma <= 0.0 {
            return if lo <= mu && mu <= hi { 1.0 } else { 0.0 };
        }
        normal_cdf((hi - mu) / sigma) - normal_cdf((lo - mu) / sigma)
    }

    /// P(high > threshold). Analytic and deterministic; `_n_samples` is retained for API
    /// compatibility (the old Monte Carlo sampler is gone).
    pub fn predict_temperature_exceeds(
        &self,
        threshold: f64,
        _n_samples: usize,
    ) -> Result<f64, String> {
        if !self.is_trained {
            return Err("model must be trained before prediction".to_string());
        }
        Ok(self.prob_at_least(threshold))
    }

    pub fn predict_temperature_range(
        &self,
        lower: f64,
        upper: f64,
        _n_samples: usize,
    ) -> Result<f64, String> {
        if !self.is_trained {
            return Err("model must be trained before prediction".to_string());
        }
        Ok(self.prob_between(lower, upper))
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
            // A "below" event is P(high <= t); everything else is P(high >= t). Routing by name
            // keeps temp_below_* honest (they previously returned the complement, P(high >= t)).
            let prob = if name.contains("below") {
                self.prob_at_most(threshold)
            } else {
                self.prob_at_least(threshold)
            };
            forecast.insert(name, prob);
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

        // Predictive band for a day's high (predictive sigma), not a credible band for the mean.
        let mut rng = rand::rngs::StdRng::seed_from_u64(45);
        let temp_samples = sample_normal(
            self.temp_posterior_mu,
            self.temp_predictive_sigma,
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
