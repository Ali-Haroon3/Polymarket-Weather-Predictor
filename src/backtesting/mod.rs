pub mod backtest_engine;
pub mod market_simulator;
pub mod performance_metrics;
pub mod real_market_loader;

pub use backtest_engine::{market_estimate, BacktestConfig, BacktestEngine, BacktestResults};
pub use market_simulator::{fahrenheit_to_celsius, MarketSimulator};
pub use performance_metrics::{PerformanceAnalyzer, PerformanceMetrics};
pub use real_market_loader::{RealMarketLoadError, RealMarketLoader};
