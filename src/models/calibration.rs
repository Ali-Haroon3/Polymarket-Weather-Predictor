use crate::utils::{mean, std_dev};

pub struct CalibrationAnalyzer;

impl CalibrationAnalyzer {
    pub fn brier_score(predicted_probs: &[f64], outcomes: &[f64]) -> f64 {
        if predicted_probs.is_empty() || predicted_probs.len() != outcomes.len() {
            return 0.0;
        }

        predicted_probs
            .iter()
            .zip(outcomes.iter())
            .map(|(p, y)| {
                let d = p - y;
                d * d
            })
            .sum::<f64>()
            / predicted_probs.len() as f64
    }

    pub fn reliability_diagram(
        predicted_probs: &[f64],
        outcomes: &[f64],
        n_bins: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let edges: Vec<f64> = (0..=n_bins).map(|i| i as f64 / n_bins as f64).collect();
        let mut bin_means = Vec::with_capacity(n_bins);
        let mut observed = Vec::with_capacity(n_bins);

        for i in 0..n_bins {
            let lo = edges[i];
            let hi = edges[i + 1];
            let mut probs = Vec::new();
            let mut outs = Vec::new();

            for (p, y) in predicted_probs.iter().zip(outcomes.iter()) {
                if *p >= lo && *p < hi {
                    probs.push(*p);
                    outs.push(*y);
                }
            }

            if probs.is_empty() {
                bin_means.push((lo + hi) / 2.0);
                observed.push(f64::NAN);
            } else {
                bin_means.push(mean(&probs));
                observed.push(mean(&outs));
            }
        }

        (edges, bin_means, observed)
    }

    pub fn expected_calibration_error(
        predicted_probs: &[f64],
        outcomes: &[f64],
        n_bins: usize,
    ) -> f64 {
        if predicted_probs.is_empty() || predicted_probs.len() != outcomes.len() {
            return 0.0;
        }

        let (edges, bin_means, obs) = Self::reliability_diagram(predicted_probs, outcomes, n_bins);
        let total = predicted_probs.len() as f64;
        let mut ece = 0.0;

        for i in 0..n_bins {
            let lo = edges[i];
            let hi = edges[i + 1];
            let count = predicted_probs
                .iter()
                .filter(|p| **p >= lo && **p < hi)
                .count() as f64;

            if count > 0.0 && obs[i].is_finite() {
                ece += (count / total) * (bin_means[i] - obs[i]).abs();
            }
        }

        ece
    }

    pub fn calibration_metrics(predicted_probs: &[f64], outcomes: &[f64]) -> std::collections::HashMap<String, f64> {
        let mut out = std::collections::HashMap::new();
        let brier = Self::brier_score(predicted_probs, outcomes);
        let ece = Self::expected_calibration_error(predicted_probs, outcomes, 10);

        let sharpness = std_dev(predicted_probs, 0);
        let base_rate = mean(outcomes);
        let resolution = predicted_probs
            .iter()
            .map(|p| {
                let d = p - base_rate;
                d * d
            })
            .sum::<f64>()
            / predicted_probs.len().max(1) as f64;

        let coverage = if predicted_probs.is_empty() {
            0.0
        } else {
            predicted_probs
                .iter()
                .filter(|p| **p >= 0.0 && **p <= 1.0)
                .count() as f64
                / predicted_probs.len() as f64
        };

        out.insert("brier_score".to_string(), brier);
        out.insert("expected_calibration_error".to_string(), ece);
        out.insert("sharpness".to_string(), sharpness);
        out.insert("resolution".to_string(), resolution);
        out.insert("coverage".to_string(), coverage);
        out
    }

    pub fn temperature_calibration(
        predicted_temps: &[f64],
        observed_temps: &[f64],
    ) -> std::collections::HashMap<String, f64> {
        let n = predicted_temps.len().min(observed_temps.len());
        if n == 0 {
            return std::collections::HashMap::new();
        }

        let mut abs_errors = Vec::with_capacity(n);
        let mut sq_errors = Vec::with_capacity(n);
        let mut residuals = Vec::with_capacity(n);

        for i in 0..n {
            let err = predicted_temps[i] - observed_temps[i];
            abs_errors.push(err.abs());
            sq_errors.push(err * err);
            residuals.push(err);
        }

        let mae = mean(&abs_errors);
        let rmse = mean(&sq_errors).sqrt();

        let observed_mean = mean(&observed_temps[..n]);
        let ss_res = sq_errors.iter().sum::<f64>();
        let ss_tot = observed_temps[..n]
            .iter()
            .map(|y| {
                let d = y - observed_mean;
                d * d
            })
            .sum::<f64>();
        let r_squared = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };

        let std_error = std_dev(&residuals, 0);
        let coverage_68 = residuals
            .iter()
            .filter(|e| e.abs() <= std_error)
            .count() as f64
            / n as f64;
        let coverage_95 = residuals
            .iter()
            .filter(|e| e.abs() <= 1.96 * std_error)
            .count() as f64
            / n as f64;

        let mut out = std::collections::HashMap::new();
        out.insert("mae".to_string(), mae);
        out.insert("rmse".to_string(), rmse);
        out.insert("r_squared".to_string(), r_squared);
        out.insert("coverage_68pct".to_string(), coverage_68);
        out.insert("coverage_95pct".to_string(), coverage_95);
        out
    }
}
