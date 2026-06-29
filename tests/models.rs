use chrono::{Duration, NaiveDate};
use rand::SeedableRng;
use rand_distr::{Distribution, Exp, Normal};

use polymarket_weather_predictor::models::{BayesianWeatherModel, CalibrationAnalyzer};
use polymarket_weather_predictor::types::WeatherRecord;

fn sample_data() -> Vec<WeatherRecord> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let start = NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();

    let temp = Normal::new(15.0, 8.0).unwrap();
    let tmax = Normal::new(22.0, 8.0).unwrap();
    let precip = Exp::new(1.0 / 0.3).unwrap();

    (0..365)
        .map(|i| {
            let mut row = WeatherRecord::new(start + Duration::days(i), "TEST");
            row.temperature_mean = Some(temp.sample(&mut rng));
            row.temperature_max = Some(tmax.sample(&mut rng));
            row.precipitation_total = Some(if rand::random::<f64>() > 0.7 {
                precip.sample(&mut rng)
            } else {
                0.0
            });
            row
        })
        .collect()
}

#[test]
fn test_model_training_and_predictions() {
    let mut model = BayesianWeatherModel::default();
    let data = sample_data();

    model.train(&data).unwrap();
    assert!(model.is_trained);

    let temp_prob = model.predict_temperature_exceeds(20.0, 5000).unwrap();
    assert!((0.0..=1.0).contains(&temp_prob));

    let (precip_prob, ci) = model.predict_precipitation_probability(5000).unwrap();
    assert!((0.0..=1.0).contains(&precip_prob));
    assert!(ci.0 <= precip_prob && precip_prob <= ci.1);

    let forecast = model.forecast_event_probabilities(None, 5000).unwrap();
    assert!(forecast.contains_key("precipitation"));
    assert!(forecast.contains_key("temp_above_90f"));

    let intervals = model.get_posterior_intervals(0.95, 5000).unwrap();
    assert!(intervals.contains_key("temperature"));
    assert!(intervals.contains_key("precipitation"));
}

#[test]
fn test_calibration_metrics() {
    let predictions = vec![0.9, 0.1, 0.8, 0.2];
    let outcomes = vec![1.0, 0.0, 1.0, 0.0];

    let brier = CalibrationAnalyzer::brier_score(&predictions, &outcomes);
    assert!((0.0..=1.0).contains(&brier));

    let metrics = CalibrationAnalyzer::calibration_metrics(&predictions, &outcomes);
    assert!(metrics.contains_key("brier_score"));
    assert!(metrics.contains_key("expected_calibration_error"));
    assert!(metrics.contains_key("coverage"));

    let (_edges, means, freq) = CalibrationAnalyzer::reliability_diagram(&predictions, &outcomes, 10);
    assert_eq!(means.len(), 10);
    assert_eq!(freq.len(), 10);
}

// ── predictive-distribution calibration (the predictive-variance fix) ────────

use polymarket_weather_predictor::utils::{erf, normal_cdf};

/// Daily-high observations drawn from a known Normal(mu, s), set on temperature_max.
fn sample_max_data(mu: f64, s: f64, seed: u64, n: usize) -> Vec<WeatherRecord> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let dist = Normal::new(mu, s).unwrap();
    let start = NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();
    (0..n as i64)
        .map(|i| {
            let mut row = WeatherRecord::new(start + Duration::days(i), "TEST");
            row.temperature_max = Some(dist.sample(&mut rng));
            row.precipitation_total = Some(0.0);
            row
        })
        .collect()
}

/// Ground-truth upper-tail probability of a Normal(mu, s), via the crate's own erf.
fn true_tail(t: f64, mu: f64, s: f64) -> f64 {
    1.0 - normal_cdf((t - mu) / s)
}

#[test]
fn test_erf_phi_known_values() {
    let close = |a: f64, b: f64, tol: f64| assert!((a - b).abs() < tol, "{a} vs {b}");
    close(erf(1.0), 0.8427007929, 1e-6);
    close(erf(-1.0), -0.8427007929, 1e-6);
    close(erf(0.0), 0.0, 1e-7); // A&S 7.1.26 residual at 0 is ~1e-9, within its 1.5e-7 bound
    close(normal_cdf(0.0), 0.5, 1e-9);
    close(normal_cdf(1.96), 0.9750, 1e-3);
    close(normal_cdf(-1.96), 0.0250, 1e-3);
    close(normal_cdf(2.5) + normal_cdf(-2.5), 1.0, 1e-7);
}

#[test]
fn test_predictive_matches_truth() {
    // Train on 400 daily highs ~ N(25, 8). The predictive should track the TRUE daily
    // distribution (spread ~8 degC), not the standard error of the mean (~0.4 degC).
    let mut model = BayesianWeatherModel::default();
    model.train(&sample_max_data(25.0, 8.0, 7, 400)).unwrap();

    for &t in &[5.0, 13.0, 21.0, 25.0, 29.0, 37.0, 45.0] {
        let p = model.prob_at_least(t);
        let truth = true_tail(t, 25.0, 8.0);
        assert!((p - truth).abs() < 0.02, "t={t}: model={p} truth={truth}");
    }

    // Contrast: the SE-of-the-mean sigma (the old bug) collapses tails to ~0 where the
    // real probability at +1.5 sigma is ~6.7%.
    assert!(model.temp_posterior_sigma < 1.0, "SE of mean should be sub-degree at n=400");
    let buggy = true_tail(37.0, model.temp_posterior_mu, model.temp_posterior_sigma);
    assert!(buggy < 1e-3, "buggy (SE-of-mean) tail at +1.5 sigma ~ {buggy}");
    assert!(true_tail(37.0, 25.0, 8.0) > 0.05, "true tail at +1.5 sigma is material");
}

#[test]
fn test_bucket_partition() {
    let mut model = BayesianWeatherModel::default();
    model.train(&sample_max_data(25.0, 8.0, 7, 400)).unwrap();

    // A full integer-degree ladder must tile the real line: it sums to exactly 1.
    let (lo, hi) = (-15i32, 65i32);
    let mut total = model.prob_at_most(lo as f64);
    for k in lo..hi {
        total += model.prob_between(k as f64, (k + 1) as f64);
    }
    total += model.prob_at_least(hi as f64);
    assert!((total - 1.0).abs() < 1e-9, "ladder sums to {total}");

    // Internal consistency of the three forms.
    assert!((model.prob_at_least(25.0) + model.prob_at_most(25.0) - 1.0).abs() < 1e-12);
    assert!(
        (model.prob_between(20.0, 30.0)
            - (model.prob_at_most(30.0) - model.prob_at_most(20.0)))
        .abs()
            < 1e-12
    );
    assert_eq!(model.prob_between(30.0, 20.0), 0.0); // swapped interval -> 0
}

#[test]
fn test_brier_beats_buggy() {
    // Train on the first 400 highs, evaluate calibration on the next 400 held-out highs.
    let all = sample_max_data(25.0, 8.0, 11, 800);
    let mut model = BayesianWeatherModel::default();
    model.train(&all[..400]).unwrap();
    let held: Vec<f64> = all[400..].iter().filter_map(|r| r.temperature_max).collect();

    let (mu, se_mean) = (model.temp_posterior_mu, model.temp_posterior_sigma);
    let mut outcomes = Vec::new();
    let mut preds_fixed = Vec::new();
    let mut preds_buggy = Vec::new();
    for &t in &[13.0, 17.0, 21.0, 25.0, 29.0, 33.0, 37.0] {
        for &high in &held {
            outcomes.push(if high >= t { 1.0 } else { 0.0 });
            preds_fixed.push(model.prob_at_least(t));
            // The old model: tail under the SE-of-mean sigma, clipped like get_model_estimate.
            preds_buggy.push(true_tail(t, mu, se_mean).clamp(0.01, 0.99));
        }
    }

    // Brier is diluted by the irreducible noise of binary outcomes (the fixed model sits at the
    // Bayes-optimal ~0.15), so the strict-better check is modest. ECE — the direct calibration
    // metric — is where the overconfident model fails dramatically.
    let brier_fixed = CalibrationAnalyzer::brier_score(&preds_fixed, &outcomes);
    let brier_buggy = CalibrationAnalyzer::brier_score(&preds_buggy, &outcomes);
    assert!(
        brier_fixed < brier_buggy,
        "fixed Brier {brier_fixed} should beat buggy {brier_buggy}"
    );

    let ece_fixed = CalibrationAnalyzer::expected_calibration_error(&preds_fixed, &outcomes, 10);
    let ece_buggy = CalibrationAnalyzer::expected_calibration_error(&preds_buggy, &outcomes, 10);
    assert!(ece_fixed < 0.05, "fixed model should be well-calibrated, ECE {ece_fixed}");
    assert!(
        ece_fixed < ece_buggy - 0.05,
        "fixed ECE {ece_fixed} should crush buggy ECE {ece_buggy}"
    );
}
