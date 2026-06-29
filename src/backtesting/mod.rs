pub mod backtest_engine;
pub mod market_simulator;
pub mod performance_metrics;
pub mod real_market_loader;

pub use backtest_engine::{
    evaluate_markets, market_estimate, BacktestConfig, BacktestEngine, BacktestResults,
    MarketEvaluation,
};
pub use market_simulator::{fahrenheit_to_celsius, MarketSimulator};
pub use performance_metrics::{PerformanceAnalyzer, PerformanceMetrics};
pub use real_market_loader::{RealMarketLoadError, RealMarketLoader};
