pub mod backtest_engine;
pub mod market_shape;
pub mod market_simulator;
pub mod performance_metrics;
pub mod real_market_loader;
pub mod shrinkage;
pub mod spread_sigma;

pub use backtest_engine::{
    evaluate_markets, evaluate_markets_with_forecast, kelly_fraction_of_capital, market_estimate,
    BacktestConfig, BacktestEngine, BacktestResults, MarketEvaluation,
};
pub use market_shape::{
    build_ladders, decide_cell, fit_shape, fit_shape_as_of, replay, replay_roi, shape_history,
    walk_forward_estimates, Ladder, LadderInput, ReplayTrade, ShapeDecision, ShapeParams,
    ShapeSide,
};
pub use market_simulator::{fahrenheit_to_celsius, MarketSimulator};
pub use performance_metrics::{PerformanceAnalyzer, PerformanceMetrics};
pub use real_market_loader::{RealMarketLoadError, RealMarketLoader};
pub use shrinkage::{
    fill_prices, lambda_segment, reference_price, segment_veto, shape_segment, SegmentVeto,
    ShrinkageFit, TRAIL_MIN_N, TRAIL_WINDOW_DAYS,
};
