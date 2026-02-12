use chrono::{Duration, NaiveDate};
use rand::SeedableRng;
use rand_distr::{Distribution, Exp, Gamma, Normal};

use polymarket_weather_predictor::models::{BayesianWeatherModel, CalibrationAnalyzer};
use polymarket_weather_predictor::trading::{MarketMaker, MonteCarloSimulator};
use polymarket_weather_predictor::types::WeatherRecord;

fn synthetic_weather(days: usize) -> Vec<WeatherRecord> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let start = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();

    let n_temp = Normal::new(15.0, 8.0).unwrap();
    let n_max = Normal::new(22.0, 8.0).unwrap();
    let n_min = Normal::new(8.0, 8.0).unwrap();
    let exp = Exp::new(0.5).unwrap();
    let gamma = Gamma::new(2.0, 2.0).unwrap();

    (0..days)
        .map(|i| {
            let mut row = WeatherRecord::new(start + Duration::days(i as i64), "TEST_STATION");
            row.temperature_mean = Some(n_temp.sample(&mut rng));
            row.temperature_max = Some(n_max.sample(&mut rng));
            row.temperature_min = Some(n_min.sample(&mut rng));
            row.precipitation_total = Some(exp.sample(&mut rng));
            row.wind_speed_mean = Some(gamma.sample(&mut rng));
            row
        })
        .collect()
}

fn main() {
    let weather = synthetic_weather(365);

    let mut model = BayesianWeatherModel::default();
    model.train(&weather).expect("model train");

    let forecast = model.forecast_event_probabilities(None, 5000).expect("forecast");
    println!("Forecast: {forecast:?}");

    let preds = vec![0.2, 0.4, 0.6, 0.8];
    let outs = vec![0.0, 0.0, 1.0, 1.0];
    let calib = CalibrationAnalyzer::calibration_metrics(&preds, &outs);
    println!("Calibration: {calib:?}");

    let sim = MonteCarloSimulator::default();
    let paths = sim.simulate_price_paths(0.35, 0.55, 0.2, 30, Some(1000));
    let pnl = sim.calculate_pnl_distribution(0.35, 0.05, &paths);
    println!("P&L stats: {pnl:?}");

    let mut mm = MarketMaker::new(100_000.0);
    let spread = mm.calculate_fair_value_spread(0.5, 0.58, 0.2, 0.02);
    println!("Spread: {spread:?}");

    let trade = mm.execute_market_orders("MARKET_A", "BUY", 0.05, 0.5);
    println!("Trade: {trade:?}");
}
