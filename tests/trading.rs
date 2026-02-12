use std::collections::HashMap;

use approx::assert_relative_eq;

use polymarket_weather_predictor::trading::{MarketMaker, MonteCarloSimulator};
use polymarket_weather_predictor::types::{MarketQuote, PricePoint, ProbabilityPoint};

#[test]
fn test_monte_carlo_core() {
    let sim = MonteCarloSimulator::default();

    let paths = sim.simulate_price_paths(0.5, 0.6, 0.2, 30, Some(100));
    assert_eq!(paths.len(), 100);
    assert_eq!(paths[0].len(), 30);
    assert!(paths.iter().flatten().all(|p| *p >= 0.0 && *p <= 1.0));

    let stats = sim.calculate_pnl_distribution(0.5, 0.1, &paths);
    assert!(stats.std_pnl >= 0.0);

    let pnl = (0..10_000)
        .map(|i| -100.0 + (i as f64 % 100.0) - 50.0)
        .collect::<Vec<_>>();
    let var = sim.calculate_value_at_risk(&pnl, 0.95);
    assert!(var < 0.0);

    let opt = sim.optimize_position_size(0.5, 0.6, 0.2, 30, 5000.0);
    assert!((0.0..=0.5).contains(&opt.optimal_position_size));
}

#[test]
fn test_market_maker_core() {
    let mut mm = MarketMaker::new(100_000.0);

    let spread = mm.calculate_fair_value_spread(0.5, 0.55, 0.2, 0.0);
    assert!(spread.bid < spread.ask);
    assert!(spread.bid >= 0.0 && spread.ask <= 1.0);

    let market_prices = HashMap::from([
        (
            "TEMP_MARKET".to_string(),
            MarketQuote {
                bid: 0.48,
                ask: 0.52,
                mid: None,
            },
        ),
        (
            "RAIN_MARKET".to_string(),
            MarketQuote {
                bid: 0.45,
                ask: 0.55,
                mid: None,
            },
        ),
    ]);

    let estimates = HashMap::from([
        ("TEMP_MARKET".to_string(), 0.55),
        ("RAIN_MARKET".to_string(), 0.60),
    ]);

    let spreads = mm.optimal_bid_ask_spreads(&market_prices, &estimates, 0.2);
    assert_eq!(spreads.len(), 2);

    let prices = HashMap::from([
        ("MARKET_A".to_string(), 0.5),
        ("MARKET_B".to_string(), 0.5),
    ]);
    let sizes = mm.calculate_position_sizes(&HashMap::from([
        ("MARKET_A".to_string(), 0.6),
        ("MARKET_B".to_string(), 0.4),
    ]), &prices, 0.1);

    assert_eq!(sizes.len(), 2);
    assert!(sizes.values().all(|s| *s >= 0.0 && *s <= 0.1));

    let trade = mm.execute_market_orders("TEST_MARKET", "BUY", 0.05, 0.5);
    assert_eq!(trade.market_id, "TEST_MARKET");
    assert_eq!(trade.side, "BUY");
    assert!(mm.inventory.contains_key("TEST_MARKET"));

    mm.inventory.insert("MARKET_A".to_string(), 0.1);
    mm.inventory.insert("MARKET_B".to_string(), -0.05);
    let metrics = mm.get_portfolio_metrics();
    assert_relative_eq!(metrics.total_position, 0.20, epsilon = 1e-8);
    assert_relative_eq!(metrics.net_exposure, 0.10, epsilon = 1e-8);
    assert_eq!(metrics.number_of_markets, 3);
}

#[test]
fn test_backtest_strategy_minimum() {
    let sim = MonteCarloSimulator::default();
    let prices = vec![
        PricePoint {
            date: chrono::NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            close: 0.5,
        },
        PricePoint {
            date: chrono::NaiveDate::from_ymd_opt(2025, 1, 2).unwrap(),
            close: 0.55,
        },
    ];

    let forecasts = vec![
        ProbabilityPoint {
            date: chrono::NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            probability: 0.6,
        },
        ProbabilityPoint {
            date: chrono::NaiveDate::from_ymd_opt(2025, 1, 2).unwrap(),
            probability: 0.6,
        },
    ];

    let result = sim.backtest_strategy(&prices, &forecasts, 100000.0, 0.1);
    assert!(result.final_value > 0.0);
}
