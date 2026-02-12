use chrono::{Duration, NaiveDate};
use rand::SeedableRng;
use rand_distr::{Distribution, Exp, Gamma, Normal};

use polymarket_weather_predictor::backtesting::{BacktestConfig, BacktestEngine};
use polymarket_weather_predictor::types::WeatherRecord;

fn make_weather(city: &str, start: NaiveDate, days: usize, seed: u64) -> Vec<WeatherRecord> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let n_base = Normal::new(25.0, 3.0).unwrap();
    let n_offset = Normal::new(0.0, 2.0).unwrap();
    let gamma = Gamma::new(2.0, 2.0).unwrap();
    let exp = Exp::new(0.8).unwrap();

    (0..days)
        .map(|i| {
            let mut row = WeatherRecord::new(start + Duration::days(i as i64), format!("MULTI_{city}"));
            let mean = n_base.sample(&mut rng) - i as f64 * 0.1 + n_offset.sample(&mut rng);
            row.temperature_mean = Some(mean);
            row.temperature_max = Some(mean + 5.0 + n_offset.sample(&mut rng));
            row.temperature_min = Some(mean - 5.0 + n_offset.sample(&mut rng));
            let p = exp.sample(&mut rng);
            row.precipitation_total = Some(if rand::random::<f64>() > 0.3 { 0.0 } else { p });
            row.wind_speed_mean = Some(gamma.sample(&mut rng));
            row.sources = Some("open_meteo,nws".to_string());
            row.source_count = Some(2);
            row.location = Some(city.to_string());
            row
        })
        .collect()
}

fn main() {
    let start = NaiveDate::from_ymd_opt(2025, 9, 1).unwrap();
    let end = NaiveDate::from_ymd_opt(2025, 10, 30).unwrap();

    let config = BacktestConfig {
        cities: Some(vec!["NYC".to_string(), "LA".to_string(), "London".to_string()]),
        initial_capital: 100_000.0,
        max_position_pct: 0.10,
        edge_threshold: 0.05,
        kelly_fraction: 0.25,
        model_lookback_days: 60,
        market_noise_std: 0.10,
        seed: 42,
    };
    let mut engine = BacktestEngine::new(config);

    let mut weather = std::collections::HashMap::new();
    weather.insert("NYC".to_string(), make_weather("NYC", start - Duration::days(60), 150, 42));
    weather.insert("LA".to_string(), make_weather("LA", start - Duration::days(60), 150, 43));
    weather.insert(
        "London".to_string(),
        make_weather("London", start - Duration::days(60), 150, 44),
    );

    let results = engine.run(Some(start), Some(end), Some(weather));

    println!("Backtest complete");
    println!("Config: {:?}", results.config);
    println!("Portfolio: {:?}", results.metrics.portfolio);
    println!("Trade metrics: {:?}", results.metrics.trades);
    println!("Trades executed: {}", results.trades.len());
    println!("Total markets generated: {}", results.total_markets_generated);
}
