use chrono::{Datelike, Duration, NaiveDate};
use rand::SeedableRng;
use rand_distr::{Distribution, Exp, Normal};

use polymarket_weather_predictor::backtesting::market_estimate;
use polymarket_weather_predictor::models::{BayesianWeatherModel, CalibrationAnalyzer};
use polymarket_weather_predictor::types::{SimulatedMarket, WeatherRecord};

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

// ── day-of-year seasonality (local linear detrend) ──────────────────────────

/// Daily highs on a linear ramp on real dates [target-n, target-1]: high(i) = base + slope*i + noise.
/// Truth at the target day (i = n) is base + slope*n. Stationary training averages to the window
/// midpoint (~base + slope*n/2); seasonal training extrapolates the trend to the target.
fn ramp_data(target: NaiveDate, n: i64, base: f64, slope: f64, noise: f64, seed: u64) -> Vec<WeatherRecord> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let dist = Normal::new(0.0, noise).unwrap();
    (0..n)
        .map(|i| {
            let date = target - Duration::days(n - i);
            let mut row = WeatherRecord::new(date, "TEST");
            row.temperature_max = Some(base + slope * i as f64 + dist.sample(&mut rng));
            row.precipitation_total = Some(0.0);
            row
        })
        .collect()
}

#[test]
fn test_seasonal_tracks_ramp_not_midpoint() {
    let target = NaiveDate::from_ymd_opt(2021, 3, 1).unwrap();
    let data = ramp_data(target, 60, 14.0, 0.2, 2.0, 7);

    let mut seas = BayesianWeatherModel::default();
    seas.train_for_target(&data, target).unwrap();
    let mut stat = BayesianWeatherModel::default();
    stat.train(&data).unwrap();

    let true_target = 14.0 + 0.2 * 60.0; // 26.0
    assert!(
        (seas.temp_posterior_mu - true_target).abs() < 1.0,
        "seasonal mu {} should track true target {true_target}",
        seas.temp_posterior_mu
    );
    assert!(
        stat.temp_posterior_mu < seas.temp_posterior_mu - 3.0,
        "stationary {} must lag seasonal {} by >3C on a rising ramp",
        stat.temp_posterior_mu,
        seas.temp_posterior_mu
    );
}

#[test]
fn test_seasonal_residual_variance() {
    let target = NaiveDate::from_ymd_opt(2021, 3, 1).unwrap();
    let data = ramp_data(target, 60, 14.0, 0.2, 2.0, 7);

    let mut seas = BayesianWeatherModel::default();
    seas.train_for_target(&data, target).unwrap();
    let mut stat = BayesianWeatherModel::default();
    stat.train(&data).unwrap();

    // Seasonal sigma ~ true noise SD (2.0); stationary sigma is inflated by the trend.
    assert!(
        (seas.temp_predictive_sigma - 2.0).abs() < 0.7,
        "seasonal sigma {} should be near noise SD 2.0",
        seas.temp_predictive_sigma
    );
    assert!(
        stat.temp_predictive_sigma > seas.temp_predictive_sigma + 1.0,
        "trend should inflate stationary sigma {} vs seasonal {}",
        stat.temp_predictive_sigma,
        seas.temp_predictive_sigma
    );
}

#[test]
fn test_seasonal_fallback_short_window() {
    // Below the 14-record floor the seasonal fit must be byte-identical to the stationary fit.
    let target = NaiveDate::from_ymd_opt(2021, 3, 1).unwrap();
    let data = ramp_data(target, 10, 14.0, 0.2, 2.0, 1);

    let mut seas = BayesianWeatherModel::default();
    seas.train_for_target(&data, target).unwrap();
    let mut stat = BayesianWeatherModel::default();
    stat.train(&data).unwrap();

    assert!((seas.temp_posterior_mu - stat.temp_posterior_mu).abs() < 1e-9);
    assert!((seas.temp_predictive_sigma - stat.temp_predictive_sigma).abs() < 1e-9);
    assert!(seas.temp_predictive_sigma.is_finite() && seas.temp_predictive_sigma > 0.0);
}

#[test]
fn test_seasonal_deterministic() {
    let target = NaiveDate::from_ymd_opt(2021, 3, 1).unwrap();
    let data = ramp_data(target, 60, 14.0, 0.2, 2.0, 99);
    let mut a = BayesianWeatherModel::default();
    a.train_for_target(&data, target).unwrap();
    let mut b = BayesianWeatherModel::default();
    b.train_for_target(&data, target).unwrap();
    for &t in &[15.0, 22.0, 28.0] {
        assert_eq!(a.prob_at_least(t).to_bits(), b.prob_at_least(t).to_bits());
    }
}

#[test]
fn test_seasonal_no_regression_flat() {
    // On data with NO trend, the slope the seasonal fit finds is spurious — but its predictive
    // sigma widens (via proj_var) to cover the added mean uncertainty, so CALIBRATION is preserved:
    // Brier on held-out flat days is no worse than the stationary model's. (We test the property
    // that matters — calibration — not point-estimate stability, which detrending legitimately
    // trades for the ability to track real trends.)
    let target = NaiveDate::from_ymd_opt(2020, 3, 1).unwrap(); // day after the 60-record window
    let train = sample_max_data(25.0, 8.0, 11, 60); // flat, dates end 2020-02-29
    let held: Vec<f64> = sample_max_data(25.0, 8.0, 23, 200)
        .iter()
        .filter_map(|r| r.temperature_max)
        .collect();

    let mut seas = BayesianWeatherModel::default();
    seas.train_for_target(&train, target).unwrap();
    let mut stat = BayesianWeatherModel::default();
    stat.train(&train).unwrap();

    let (mut outcomes, mut p_seas, mut p_stat) = (Vec::new(), Vec::new(), Vec::new());
    for &t in &[17.0, 21.0, 25.0, 29.0, 33.0] {
        for &h in &held {
            outcomes.push(if h >= t { 1.0 } else { 0.0 });
            p_seas.push(seas.prob_at_least(t));
            p_stat.push(stat.prob_at_least(t));
        }
    }
    let brier_seas = CalibrationAnalyzer::brier_score(&p_seas, &outcomes);
    let brier_stat = CalibrationAnalyzer::brier_score(&p_stat, &outcomes);
    assert!(
        brier_seas < brier_stat + 0.03,
        "flat data: seasonal Brier {brier_seas} should be no worse than stationary {brier_stat}"
    );
    assert!(
        seas.temp_predictive_sigma.is_finite() && seas.temp_predictive_sigma > 4.0,
        "seasonal sigma {} stays sane on flat data",
        seas.temp_predictive_sigma
    );
}

#[test]
fn test_seasonal_no_regression_flat_many_seeds() {
    // Averaged over many flat windows the slope-significance gate keeps seasonal Brier from
    // regressing vs stationary (a single seed could pass by luck; the mean gap is the real test).
    let target = NaiveDate::from_ymd_opt(2020, 3, 1).unwrap();
    let held: Vec<f64> = sample_max_data(25.0, 8.0, 23, 200)
        .iter()
        .filter_map(|r| r.temperature_max)
        .collect();
    let mut sum_gap = 0.0;
    let seeds = 30u64;
    for seed in 0..seeds {
        let train = sample_max_data(25.0, 8.0, 100 + seed, 60);
        let mut seas = BayesianWeatherModel::default();
        seas.train_for_target(&train, target).unwrap();
        let mut stat = BayesianWeatherModel::default();
        stat.train(&train).unwrap();
        let (mut o, mut ps, mut pt) = (Vec::new(), Vec::new(), Vec::new());
        for &t in &[17.0, 21.0, 25.0, 29.0, 33.0] {
            for &h in &held {
                o.push(if h >= t { 1.0 } else { 0.0 });
                ps.push(seas.prob_at_least(t));
                pt.push(stat.prob_at_least(t));
            }
        }
        sum_gap +=
            CalibrationAnalyzer::brier_score(&ps, &o) - CalibrationAnalyzer::brier_score(&pt, &o);
    }
    let mean_gap = sum_gap / seeds as f64;
    assert!(
        mean_gap < 0.005,
        "flat data: mean seasonal-minus-stationary Brier gap {mean_gap} should be ~0"
    );
}

#[test]
fn test_seasonal_tracks_downward_ramp() {
    // Fall cooling: stationary lags ABOVE the target; seasonal tracks it. (Catches an offset/sign bug.)
    let target = NaiveDate::from_ymd_opt(2021, 10, 1).unwrap();
    let data = ramp_data(target, 60, 30.0, -0.2, 2.0, 5);
    let mut seas = BayesianWeatherModel::default();
    seas.train_for_target(&data, target).unwrap();
    let mut stat = BayesianWeatherModel::default();
    stat.train(&data).unwrap();

    let true_target = 30.0 - 0.2 * 60.0; // 18.0
    assert!(
        (seas.temp_posterior_mu - true_target).abs() < 1.0,
        "seasonal mu {} should track true target {true_target}",
        seas.temp_posterior_mu
    );
    assert!(
        stat.temp_posterior_mu > seas.temp_posterior_mu + 3.0,
        "stationary {} must lag ABOVE seasonal {} on a falling ramp",
        stat.temp_posterior_mu,
        seas.temp_posterior_mu
    );
}

#[test]
fn test_seasonal_beats_stationary_on_seasonal_curve() {
    // Real-ish sinusoidal annual cycle, target on the steep ascending part: the local linear
    // projection tracks the target's true climatology far better than the lagging stationary mean.
    let target = NaiveDate::from_ymd_opt(2021, 2, 14).unwrap(); // doy ~45, rising steeply
    let seasonal_high = |doy: f64| 15.0 + 12.0 * (2.0 * std::f64::consts::PI * doy / 365.0).sin();
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let noise = Normal::new(0.0, 2.0).unwrap();
    let data: Vec<WeatherRecord> = (1..=60i64)
        .map(|k| {
            let date = target - Duration::days(k);
            let mut r = WeatherRecord::new(date, "T");
            r.temperature_max = Some(seasonal_high(date.ordinal() as f64) + noise.sample(&mut rng));
            r.precipitation_total = Some(0.0);
            r
        })
        .collect();
    let truth = seasonal_high(target.ordinal() as f64);

    let mut seas = BayesianWeatherModel::default();
    seas.train_for_target(&data, target).unwrap();
    let mut stat = BayesianWeatherModel::default();
    stat.train(&data).unwrap();

    let err_seas = (seas.temp_posterior_mu - truth).abs();
    let err_stat = (stat.temp_posterior_mu - truth).abs();
    assert!(
        err_seas < err_stat,
        "on a seasonal curve, seasonal error {err_seas} should beat stationary error {err_stat} (truth {truth})"
    );
}

#[test]
fn test_seasonal_temp_empty_window_errors() {
    // Precip-only records (all temperature fields None, e.g. NOAA PRCP-only days) must Err cleanly,
    // never panic — even with ≥14 records (which clear the walk-forward's record-count gate).
    let target = NaiveDate::from_ymd_opt(2021, 3, 1).unwrap();
    let recs: Vec<WeatherRecord> = (1..=20i64)
        .map(|k| {
            let mut r = WeatherRecord::new(target - Duration::days(k), "T");
            r.precipitation_total = Some(0.0); // temperature_max and temperature_mean stay None
            r
        })
        .collect();
    let mut m = BayesianWeatherModel::default();
    assert!(m.train_for_target(&recs, target).is_err());
    assert!(!m.is_trained);
}

/// The 2026-09-01 defect: `evaluate_markets_inner` builds a point-forecast model for any city with
/// a temperature forecast, then asks `market_estimate` every market shape — rain included.
/// `set_point_forecast` never touches the Beta, so the answer was its Default (1, 1) prior (0.496
/// after sampling) reported as a real opinion. In production it showed up as the SAME 0.496 across
/// 11 climatically unrelated cities (Dubai, Seattle, LA, Hong Kong, ...), scoring Brier 0.233
/// against the market's 0.066 on 33 resolved captures.
#[test]
fn point_forecast_model_refuses_to_answer_rain_from_the_untouched_prior() {
    let mut temp_only = BayesianWeatherModel::default();
    temp_only.set_point_forecast(30.0, 1.5);
    assert!(
        temp_only.prob_at_least(28.0) > 0.5,
        "the temperature side must still answer"
    );
    assert!(
        temp_only.predict_precipitation_probability(5000).is_err(),
        "a temperature-only model must not answer a rain question from the Beta prior"
    );

    // The trading path turns that refusal into "no estimate", which is "no trade".
    let rain = SimulatedMarket {
        date: NaiveDate::from_ymd_opt(2026, 9, 1).unwrap(),
        market_id: "m".into(),
        market_title: "Will it rain in LA?".into(),
        market_type: "precipitation".into(),
        threshold: 0.0,
        threshold_upper: None,
        unit: None,
        market_price: 0.03,
        actual_outcome: 0.0,
        city: "LA".into(),
    };
    assert_eq!(market_estimate(&temp_only, &rain), None);

    // A model that DID see rain observations still answers, and reflects what it saw.
    let mut trained = BayesianWeatherModel::default();
    trained
        .train_from_arrays(&[20.0, 21.0, 22.0, 23.0], &[false, false, false, true])
        .unwrap();
    let (p, _) = trained.predict_precipitation_probability(5000).unwrap();
    assert!(
        p > 0.0 && p < 0.5,
        "one wet day in four should price well under the 0.5 prior, got {p}"
    );
}
