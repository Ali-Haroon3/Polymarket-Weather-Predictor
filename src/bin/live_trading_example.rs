use chrono::{Duration, NaiveDate};
use rand::SeedableRng;
use rand_distr::{Distribution, Exp, Normal};

use polymarket_weather_predictor::api::LiveTrader;
use polymarket_weather_predictor::types::{PricePoint, ProbabilityPoint, WeatherRecord};

fn generate_sample_weather_data(days: usize) -> Vec<WeatherRecord> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let start = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();

    let temp = Normal::new(15.0, 8.0).unwrap();
    let tmax = Normal::new(22.0, 8.0).unwrap();
    let tmin = Normal::new(8.0, 8.0).unwrap();
    let precip = Exp::new(0.5).unwrap();

    (0..days)
        .map(|i| {
            let mut row = WeatherRecord::new(start + Duration::days(i as i64), "TEST_STATION");
            row.temperature_mean = Some(temp.sample(&mut rng));
            row.temperature_max = Some(tmax.sample(&mut rng));
            row.temperature_min = Some(tmin.sample(&mut rng));
            row.precipitation_total = Some(precip.sample(&mut rng));
            row
        })
        .collect()
}

fn main() {
    let mut trader = LiveTrader::new(None, None, None, Some(100_000.0), true, 0.1);
    let weather = generate_sample_weather_data(365);

    if !trader.initialize(&weather) {
        eprintln!("failed to initialize trader");
        std::process::exit(1);
    }

    let markets = vec![
        serde_json::json!({"id": "market_1", "title": "Temperature above 90F"}),
        serde_json::json!({"id": "market_2", "title": "Precipitation event"}),
    ];

    for _ in 0..3 {
        let stats = trader.run_iteration(Some(markets.clone()));
        println!(
            "scanned={} opportunities={} orders={} positions={}",
            stats.markets_scanned, stats.opportunities, stats.orders_placed, stats.positions
        );
    }

    let prices: Vec<PricePoint> = (0..100)
        .map(|i| PricePoint {
            date: NaiveDate::from_ymd_opt(2023, 1, 1).unwrap() + Duration::days(i),
            close: 0.4 + (i as f64 % 20.0) / 100.0,
        })
        .collect();

    let forecasts: Vec<ProbabilityPoint> = (0..100)
        .map(|i| ProbabilityPoint {
            date: NaiveDate::from_ymd_opt(2023, 1, 1).unwrap() + Duration::days(i),
            probability: 0.45 + (i as f64 % 10.0) / 100.0,
        })
        .collect();

    let backtest = trader.run_backtest(&prices, &forecasts, 10);
    println!("Backtest: {backtest}");

    if let Some(summary) = trader.get_performance_summary() {
        println!(
            "Performance: trades={} pnl={:.2}%",
            summary.total_trades, summary.current_pnl_pct
        );
    }

    trader.shutdown();
}
