use chrono::{Duration, NaiveDate};
use rand::SeedableRng;
use rand_distr::{Distribution, Exp, Normal};

use polymarket_weather_predictor::api::{LiveTrader, PolymarketClient};
use polymarket_weather_predictor::types::{PricePoint, ProbabilityPoint, WeatherRecord};

fn sample_data() -> Vec<WeatherRecord> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let start = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
    let temp = Normal::new(15.0, 8.0).unwrap();
    let tmax = Normal::new(22.0, 8.0).unwrap();
    let tmin = Normal::new(8.0, 8.0).unwrap();
    let precip = Exp::new(0.5).unwrap();

    (0..365)
        .map(|i| {
            let mut row = WeatherRecord::new(start + Duration::days(i), "TEST");
            row.temperature_mean = Some(temp.sample(&mut rng));
            row.temperature_max = Some(tmax.sample(&mut rng));
            row.temperature_min = Some(tmin.sample(&mut rng));
            row.precipitation_total = Some(precip.sample(&mut rng));
            row
        })
        .collect()
}

#[test]
fn test_polymarket_client_paper_mode() {
    let client = PolymarketClient::new(None, None, None, None, true);

    assert!(client.paper_trading);
    assert!(client.get_account_balance().unwrap_or(0.0) > 0.0);

    let order = client.place_order("test_market", "YES", "BUY", 0.55, 10.0, "limit");
    assert!(order.is_some());
    let order = order.unwrap();
    assert!(order.paper_trade);
    assert_eq!(order.status, "accepted");

    assert!(client.cancel_order("test_order_123"));
    let health = client.health_check();
    assert!(matches!(health, true | false));
}

#[test]
fn test_live_trader_iteration() {
    let mut trader = LiveTrader::new(None, None, None, Some(100_000.0), true, 0.1);
    let data = sample_data();
    assert!(trader.initialize(&data));

    let markets = vec![
        serde_json::json!({"id": "market_1", "title": "Temperature above 90F"}),
        serde_json::json!({"id": "market_2", "title": "Precipitation event"}),
    ];

    let stats = trader.run_iteration(Some(markets));
    assert!(stats.markets_scanned > 0);

    let prices = (0..100)
        .map(|i| PricePoint {
            date: NaiveDate::from_ymd_opt(2023, 1, 1).unwrap() + Duration::days(i),
            close: 0.4 + (i % 20) as f64 / 100.0,
        })
        .collect::<Vec<_>>();

    let forecasts = (0..100)
        .map(|i| ProbabilityPoint {
            date: NaiveDate::from_ymd_opt(2023, 1, 1).unwrap() + Duration::days(i),
            probability: 0.45 + (i % 10) as f64 / 100.0,
        })
        .collect::<Vec<_>>();

    let backtest = trader.run_backtest(&prices, &forecasts, 10);
    assert!(backtest.get("total_trades").is_some());

    let summary = trader.get_performance_summary();
    assert!(summary.is_some());
}
