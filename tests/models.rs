use chrono::{Duration, NaiveDate};
use rand::SeedableRng;
use rand_distr::{Distribution, Exp, Normal};

use polymarket_weather_predictor::models::{BayesianWeatherModel, CalibrationAnalyzer};
use polymarket_weather_predictor::types::WeatherRecord;

fn sample_data() -> Vec<WeatherRecord> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let start = NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();

    let temp = Normal::new(15.0, 8.0).unwrap();
    let precip = Exp::new(1.0 / 0.3).unwrap();

    (0..365)
        .map(|i| {
            let mut row = WeatherRecord::new(start + Duration::days(i), "TEST");
            row.temperature_mean = Some(temp.sample(&mut rng));
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
