pub mod backtest_engine;
pub mod market_simulator;
pub mod performance_metrics;

pub use backtest_engine::{BacktestConfig, BacktestEngine, BacktestResults};
pub use market_simulator::{fahrenheit_to_celsius, MarketSimulator};
pub use performance_metrics::{PerformanceAnalyzer, PerformanceMetrics};
