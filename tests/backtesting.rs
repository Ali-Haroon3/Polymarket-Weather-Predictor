use std::collections::HashMap;

use chrono::{Duration, NaiveDate};
use rand::SeedableRng;
use rand_distr::{Distribution, Exp, Gamma, Normal};

use polymarket_weather_predictor::backtesting::{
    BacktestConfig, BacktestEngine, MarketSimulator, PerformanceAnalyzer, RealMarketLoader,
};
use polymarket_weather_predictor::types::{SimulatedMarket, WeatherRecord};

fn make_weather_data(n_days: usize, seed: u64) -> Vec<WeatherRecord> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let start = NaiveDate::from_ymd_opt(2025, 8, 1).unwrap();

    let temp_noise = Normal::new(0.0, 3.0).unwrap();
    let temp_high = Normal::new(5.0, 1.0).unwrap();
    let temp_low = Normal::new(5.0, 1.0).unwrap();
    let precip = Exp::new(1.0 / 1.5).unwrap();
    let wind = Gamma::new(2.0, 2.0).unwrap();

    (0..n_days)
        .map(|i| {
            let base_temp = 25.0 - i as f64 * 0.1;
            let mean = base_temp + temp_noise.sample(&mut rng);

            let mut row = WeatherRecord::new(start + Duration::days(i as i64), "TEST");
            row.temperature_mean = Some(mean);
            row.temperature_max = Some(mean + temp_high.sample(&mut rng));
            row.temperature_min = Some(mean - temp_low.sample(&mut rng));

            let r = if rand::random::<f64>() > 0.3 {
                0.0
            } else {
                precip.sample(&mut rng)
            };
            row.precipitation_total = Some(r);
            row.wind_speed_mean = Some(wind.sample(&mut rng));
            row.source_count = Some(2);
            row.sources = Some("open_meteo,nws".to_string());
            row.data_quality_score = Some(1.0);
            row.is_validated = true;
            row.location = Some("NYC".to_string());
            row
        })
        .collect()
}

#[test]
fn test_market_simulator_basics() {
    let weather = make_weather_data(30, 42);
    let mut sim = MarketSimulator::new(None, 0.10, 42);

    let markets = sim.generate_markets(&weather, "NYC");
    assert!(!markets.is_empty());
    assert!(markets.iter().all(|m| m.market_price >= 0.02 && m.market_price <= 0.98));
    assert!(markets
        .iter()
        .all(|m| m.actual_outcome == 0.0 || m.actual_outcome == 1.0));

    let has_temp = markets.iter().any(|m| m.market_type == "temperature");
    let has_precip = markets.iter().any(|m| m.market_type == "precipitation");
    assert!(has_temp && has_precip);
}

#[test]
fn test_backtest_engine_with_override() {
    let config = BacktestConfig {
        cities: Some(vec!["NYC".to_string(), "LA".to_string()]),
        initial_capital: 100_000.0,
        max_position_pct: 0.10,
        edge_threshold: 0.05,
        kelly_fraction: 0.25,
        model_lookback_days: 30,
        market_noise_std: 0.10,
        seed: 42,
    };
    let mut engine = BacktestEngine::new(config);

    let mut override_data: HashMap<String, Vec<WeatherRecord>> = HashMap::new();
    override_data.insert("NYC".to_string(), make_weather_data(90, 42));
    override_data.insert("LA".to_string(), make_weather_data(90, 43));

    let results = engine.run(
        Some(NaiveDate::from_ymd_opt(2025, 9, 1).unwrap()),
        Some(NaiveDate::from_ymd_opt(2025, 10, 30).unwrap()),
        Some(override_data),
    );

    assert!(results.config.contains_key("start_date"));
    assert!(results.config.contains_key("end_date"));
    assert!(results.metrics.portfolio.final_value > 0.0);
}

#[test]
fn test_performance_metrics_full() {
    let trades = vec![
        polymarket_weather_predictor::backtesting::performance_metrics::make_trade_record(
            NaiveDate::from_ymd_opt(2025, 8, 1).unwrap(),
            "NYC",
            "temperature",
            100.0,
            0.6,
            1.0,
        ),
        polymarket_weather_predictor::backtesting::performance_metrics::make_trade_record(
            NaiveDate::from_ymd_opt(2025, 8, 2).unwrap(),
            "NYC",
            "temperature",
            -50.0,
            0.4,
            0.0,
        ),
        polymarket_weather_predictor::backtesting::performance_metrics::make_trade_record(
            NaiveDate::from_ymd_opt(2025, 9, 1).unwrap(),
            "LA",
            "precipitation",
            200.0,
            0.7,
            1.0,
        ),
    ];

    let portfolio_values = vec![100_000.0, 100_050.0, 100_100.0, 100_250.0];
    let metrics = PerformanceAnalyzer::compute_metrics(&trades, &portfolio_values, 100_000.0);

    assert!(metrics.portfolio.final_value > 0.0);
    assert_eq!(metrics.trades.total_trades, 3);
    assert!(metrics.forecast.is_some());
    assert!(metrics.by_city.contains_key("NYC"));
    assert!(metrics.by_market_type.contains_key("temperature"));
    assert!(metrics.by_month.contains_key("2025-08"));
}

#[test]
fn test_real_market_loader_csv() {
    let tmp = std::env::temp_dir().join(format!(
        "polymarket_markets_{}.csv",
        chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
    ));

    let csv_data = "date,market_id,market_title,market_type,threshold,market_price,actual_outcome,city\n\
2025-09-10,m1,Temp > 90,temperature,90,0.45,1,NYC\n\
2025-09-11,m2,Rain market,precipitation,0.1,0.35,0,NYC\n";
    std::fs::write(&tmp, csv_data).expect("write temp csv");

    let rows = RealMarketLoader::load_from_path(&tmp).expect("load csv");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].market_id, "m1");
    assert_eq!(rows[1].market_type, "precipitation");

    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn test_backtest_engine_with_real_markets() {
    let config = BacktestConfig {
        cities: Some(vec!["NYC".to_string()]),
        initial_capital: 100_000.0,
        max_position_pct: 0.10,
        edge_threshold: 0.01,
        kelly_fraction: 0.25,
        model_lookback_days: 30,
        market_noise_std: 0.10,
        seed: 42,
    };
    let mut engine = BacktestEngine::new(config);

    let markets = vec![
        SimulatedMarket {
            date: NaiveDate::from_ymd_opt(2025, 9, 15).unwrap(),
            market_id: "real_1".to_string(),
            market_title: "Will NYC max temperature exceed 90°F?".to_string(),
            market_type: "temperature".to_string(),
            threshold: 90.0,
            market_price: 0.40,
            actual_outcome: 1.0,
            city: "NYC".to_string(),
        },
        SimulatedMarket {
            date: NaiveDate::from_ymd_opt(2025, 9, 16).unwrap(),
            market_id: "real_2".to_string(),
            market_title: "Will it rain in NYC today?".to_string(),
            market_type: "precipitation".to_string(),
            threshold: 0.1,
            market_price: 0.30,
            actual_outcome: 0.0,
            city: "NYC".to_string(),
        },
    ];

    let mut weather_override: HashMap<String, Vec<WeatherRecord>> = HashMap::new();
    weather_override.insert("NYC".to_string(), make_weather_data(120, 11));

    let results = engine.run_with_real_markets(
        markets,
        Some(NaiveDate::from_ymd_opt(2025, 9, 1).unwrap()),
        Some(NaiveDate::from_ymd_opt(2025, 9, 30).unwrap()),
        Some(weather_override),
    );

    assert_eq!(
        results.config.get("market_mode").map(|s| s.as_str()),
        Some("real_markets")
    );
    assert_eq!(results.total_markets_generated, 2);
    assert!(results.metrics.portfolio.final_value > 0.0);
}
