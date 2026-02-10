"""
Backtest runner script
Runs a 6-month backtest of the weather prediction strategy
for NYC, LA, and London using multi-source weather data
"""

import logging
import json
import os
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polymarket.backtesting import BacktestEngine, PerformanceAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def print_separator(char="=", width=70):
    print(char * width)


def print_header(title):
    print_separator()
    print(f"  {title}")
    print_separator()


def format_pct(value):
    return f"{value:+.2f}%"


def format_usd(value):
    return f"${value:,.2f}"


def print_results(results):
    """Print a comprehensive results report"""
    config = results["config"]
    metrics = results["metrics"]
    portfolio = metrics.get("portfolio", {})
    trade_metrics = metrics.get("trades", {})
    forecast = metrics.get("forecast", {})

    print()
    print_header("POLYMARKET WEATHER PREDICTION BACKTEST RESULTS")
    print()

    # Config
    print("Configuration:")
    print(f"  Period:           {config['start_date']} to {config['end_date']}")
    print(f"  Cities:           {', '.join(config['cities'])}")
    print(f"  Initial Capital:  {format_usd(config['initial_capital'])}")
    print(f"  Edge Threshold:   {config['edge_threshold']:.0%}")
    print(f"  Kelly Fraction:   {config['kelly_fraction']}")
    print(f"  Model Lookback:   {config['model_lookback_days']} days")
    print()

    # Data sources
    print("Data Sources Used:")
    for city, sources in results.get("data_sources", {}).items():
        src_list = []
        for s in sources:
            if isinstance(s, str):
                src_list.extend(s.split(","))
        src_list = sorted(set(src_list))
        print(f"  {city}: {', '.join(src_list)}")
    print()

    # Portfolio performance
    print_header("PORTFOLIO PERFORMANCE")
    print()
    print(f"  Initial Capital:      {format_usd(portfolio.get('initial_capital', 0))}")
    print(f"  Final Value:          {format_usd(portfolio.get('final_value', 0))}")
    print(f"  Total P&L:            {format_usd(portfolio.get('total_pnl', 0))}")
    print(f"  Total Return:         {format_pct(portfolio.get('total_return_pct', 0))}")
    print(f"  Sharpe Ratio:         {portfolio.get('sharpe_ratio', 0):.3f}")
    print(f"  Sortino Ratio:        {portfolio.get('sortino_ratio', 0):.3f}")
    print(f"  Max Drawdown:         {format_pct(portfolio.get('max_drawdown_pct', 0))}")
    print(f"  Annual Volatility:    {format_pct(portfolio.get('volatility_annual_pct', 0))}")
    print(f"  Best Day:             {format_pct(portfolio.get('best_day_pct', 0))}")
    print(f"  Worst Day:            {format_pct(portfolio.get('worst_day_pct', 0))}")
    print(f"  Positive Days:        {portfolio.get('positive_days', 0)}")
    print(f"  Negative Days:        {portfolio.get('negative_days', 0)}")
    print()

    # Trade statistics
    print_header("TRADE STATISTICS")
    print()
    print(f"  Total Trades:         {trade_metrics.get('total_trades', 0)}")
    print(f"  Winning Trades:       {trade_metrics.get('winning_trades', 0)}")
    print(f"  Losing Trades:        {trade_metrics.get('losing_trades', 0)}")
    print(f"  Win Rate:             {format_pct(trade_metrics.get('win_rate_pct', 0))}")
    print(f"  Avg Win:              {format_usd(trade_metrics.get('avg_win', 0))}")
    print(f"  Avg Loss:             {format_usd(trade_metrics.get('avg_loss', 0))}")
    print(f"  Largest Win:          {format_usd(trade_metrics.get('largest_win', 0))}")
    print(f"  Largest Loss:         {format_usd(trade_metrics.get('largest_loss', 0))}")
    print(f"  Profit Factor:        {trade_metrics.get('profit_factor', 0):.3f}")
    print(f"  Avg Trade P&L:        {format_usd(trade_metrics.get('avg_trade_pnl', 0))}")
    print()

    # Forecast accuracy
    if forecast:
        print_header("FORECAST ACCURACY")
        print()
        print(f"  Brier Score:          {forecast.get('brier_score', 0):.4f}")
        print(f"  Log Loss:             {forecast.get('log_loss', 0):.4f}")
        print(f"  ECE:                  {forecast.get('expected_calibration_error', 0):.4f}")
        print()

    # City breakdown
    city_breakdown = metrics.get("by_city", {})
    if city_breakdown:
        print_header("PERFORMANCE BY CITY")
        print()
        print(f"  {'City':<12} {'Trades':>8} {'P&L':>12} {'Win Rate':>10} {'Avg P&L':>10}")
        print(f"  {'-'*12} {'-'*8} {'-'*12} {'-'*10} {'-'*10}")
        for city, data in city_breakdown.items():
            print(
                f"  {city:<12} "
                f"{data['total_trades']:>8} "
                f"{format_usd(data['total_pnl']):>12} "
                f"{format_pct(data['win_rate_pct']):>10} "
                f"{format_usd(data['avg_trade_pnl']):>10}"
            )
        print()

    # Market type breakdown
    type_breakdown = metrics.get("by_market_type", {})
    if type_breakdown:
        print_header("PERFORMANCE BY MARKET TYPE")
        print()
        print(f"  {'Type':<16} {'Trades':>8} {'P&L':>12} {'Win Rate':>10} {'Avg P&L':>10}")
        print(f"  {'-'*16} {'-'*8} {'-'*12} {'-'*10} {'-'*10}")
        for mtype, data in type_breakdown.items():
            print(
                f"  {mtype:<16} "
                f"{data['total_trades']:>8} "
                f"{format_usd(data['total_pnl']):>12} "
                f"{format_pct(data['win_rate_pct']):>10} "
                f"{format_usd(data['avg_trade_pnl']):>10}"
            )
        print()

    # Monthly breakdown
    monthly = metrics.get("by_month", {})
    if monthly:
        print_header("MONTHLY PERFORMANCE")
        print()
        print(f"  {'Month':<12} {'Trades':>8} {'P&L':>12} {'Win Rate':>10}")
        print(f"  {'-'*12} {'-'*8} {'-'*12} {'-'*10}")
        for month, data in monthly.items():
            print(
                f"  {month:<12} "
                f"{data['total_trades']:>8} "
                f"{format_usd(data['total_pnl']):>12} "
                f"{format_pct(data['win_rate_pct']):>10}"
            )
        print()

    print(f"  Total markets generated: {results.get('total_markets_generated', 0)}")
    print_separator()


def save_results(results, output_dir="backtest_results"):
    """Save results to files"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save metrics as JSON (exclude non-serializable data)
    metrics_file = os.path.join(output_dir, f"backtest_metrics_{timestamp}.json")
    serializable = {
        "config": results["config"],
        "data_sources": results["data_sources"],
        "metrics": results["metrics"],
        "total_markets_generated": results["total_markets_generated"],
    }
    with open(metrics_file, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    logger.info(f"Results saved to {metrics_file}")

    # Save trades as CSV
    if results.get("trades"):
        import pandas as pd
        trades_file = os.path.join(output_dir, f"backtest_trades_{timestamp}.csv")
        pd.DataFrame(results["trades"]).to_csv(trades_file, index=False)
        logger.info(f"Trades saved to {trades_file}")


def main():
    """Run the 6-month backtest for NYC, LA, and London"""
    print_header("POLYMARKET WEATHER PREDICTION BACKTESTER")
    print()
    print("Initializing backtest engine...")
    print("Cities: NYC, LA, London")
    print(f"Period: Last 6 months ({(datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')})")
    print()

    engine = BacktestEngine(
        cities=["NYC", "LA", "London"],
        initial_capital=100_000,
        max_position_pct=0.10,
        edge_threshold=0.05,
        kelly_fraction=0.25,
        model_lookback_days=60,
        market_noise_std=0.10,
    )

    print("Running backtest (this may take a few minutes)...")
    print()

    results = engine.run()

    print_results(results)
    save_results(results)

    print()
    print("Backtest complete!")


if __name__ == "__main__":
    main()
