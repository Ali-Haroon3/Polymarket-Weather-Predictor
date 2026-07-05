use std::collections::HashMap;
use std::path::PathBuf;

use chrono::NaiveDate;

use polymarket_weather_predictor::backtesting::{
    BacktestConfig, BacktestEngine, RealMarketLoader,
};

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = CliArgs::parse(std::env::args().skip(1).collect())?;

    let markets = RealMarketLoader::load_from_path(&args.markets)
        .map_err(|e| format!("failed to load market file: {e}"))?;

    if markets.is_empty() {
        return Err("no rows found in market data file".to_string());
    }

    let mut cities = markets.iter().map(|m| m.city.clone()).collect::<Vec<_>>();
    cities.sort();
    cities.dedup();

    let config = BacktestConfig {
        cities: Some(cities),
        initial_capital: args.initial_capital,
        max_position_pct: args.max_position_pct,
        edge_threshold: args.edge_threshold,
        kelly_fraction: args.kelly_fraction,
        model_lookback_days: args.model_lookback_days,
        market_noise_std: 0.10,
        seed: args.seed,
    };

    let mut engine = BacktestEngine::new(config);
    let results = engine.run_with_real_markets(markets, args.start_date, args.end_date, None);

    println!("Real-market backtest complete");
    println!("Mode: {}", results.config.get("market_mode").unwrap_or(&"unknown".to_string()));
    println!("Period: {} -> {}", results.config["start_date"], results.config["end_date"]);
    println!("Markets: {}", results.total_markets_generated);
    println!("Trades: {}", results.metrics.trades.total_trades);
    println!("Final value: ${:.2}", results.metrics.portfolio.final_value);
    println!("Total return: {:.2}%", results.metrics.portfolio.total_return_pct);
    println!("Sharpe: {:.3}", results.metrics.portfolio.sharpe_ratio);
    println!("Max drawdown: {:.2}%", results.metrics.portfolio.max_drawdown_pct);

    let output_dir = args.output_dir.unwrap_or_else(|| PathBuf::from("backtest_results"));
    std::fs::create_dir_all(&output_dir)
        .map_err(|e| format!("failed to create output dir {}: {e}", output_dir.display()))?;

    let ts = chrono::Utc::now().format("%Y%m%d_%H%M%S").to_string();
    let metrics_path = output_dir.join(format!("real_backtest_metrics_{ts}.json"));
    let trades_path = output_dir.join(format!("real_backtest_trades_{ts}.csv"));

    let metrics_json = serde_json::to_vec_pretty(&results)
        .map_err(|e| format!("failed to serialize results: {e}"))?;
    std::fs::write(&metrics_path, metrics_json)
        .map_err(|e| format!("failed writing {}: {e}", metrics_path.display()))?;

    write_trades_csv(&trades_path, &results.trades)
        .map_err(|e| format!("failed writing {}: {e}", trades_path.display()))?;

    println!("Saved metrics: {}", metrics_path.display());
    println!("Saved trades: {}", trades_path.display());

    Ok(())
}

fn write_trades_csv(path: &PathBuf, trades: &[polymarket_weather_predictor::types::TradeRecord]) -> Result<(), csv::Error> {
    let mut writer = csv::Writer::from_path(path)?;
    for trade in trades {
        writer.serialize(trade)?;
    }
    writer.flush()?;
    Ok(())
}

#[derive(Debug)]
struct CliArgs {
    markets: PathBuf,
    start_date: Option<NaiveDate>,
    end_date: Option<NaiveDate>,
    initial_capital: f64,
    max_position_pct: f64,
    edge_threshold: f64,
    kelly_fraction: f64,
    model_lookback_days: i64,
    seed: u64,
    output_dir: Option<PathBuf>,
}

impl CliArgs {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        if args.is_empty() {
            return Err(usage());
        }

        let mut map: HashMap<String, String> = HashMap::new();
        let mut i = 0;
        while i < args.len() {
            let key = args[i].clone();
            if !key.starts_with("--") {
                return Err(format!("unexpected argument '{key}'\n{}", usage()));
            }

            if i + 1 >= args.len() {
                return Err(format!("missing value for '{key}'\n{}", usage()));
            }

            map.insert(key, args[i + 1].clone());
            i += 2;
        }

        let markets = map
            .get("--markets")
            .ok_or_else(|| format!("--markets is required\n{}", usage()))
            .map(PathBuf::from)?;

        let start_date = map
            .get("--start")
            .map(|d| parse_date(d).map_err(|e| format!("--start {e}")))
            .transpose()?;

        let end_date = map
            .get("--end")
            .map(|d| parse_date(d).map_err(|e| format!("--end {e}")))
            .transpose()?;

        let initial_capital = parse_opt(&map, "--initial-capital", 100_000.0_f64)?;
        let max_position_pct = parse_opt(&map, "--max-position-pct", 0.10_f64)?;
        let edge_threshold = parse_opt(&map, "--edge-threshold", 0.05_f64)?;
        let kelly_fraction = parse_opt(&map, "--kelly-fraction", 0.25_f64)?;
        let model_lookback_days = parse_opt(&map, "--model-lookback-days", 60_i64)?;
        let seed = parse_opt(&map, "--seed", 42_u64)?;
        let output_dir = map.get("--output-dir").map(PathBuf::from);

        Ok(Self {
            markets,
            start_date,
            end_date,
            initial_capital,
            max_position_pct,
            edge_threshold,
            kelly_fraction,
            model_lookback_days,
            seed,
            output_dir,
        })
    }
}

fn usage() -> String {
    "usage: cargo run --bin run_backtest_real -- \\
  --markets <path.csv|path.json> \\
  [--start YYYY-MM-DD] [--end YYYY-MM-DD] \\
  [--initial-capital 100000] [--max-position-pct 0.10] \\
  [--edge-threshold 0.05] [--kelly-fraction 0.25] \\
  [--model-lookback-days 60] [--seed 42] \\
  [--output-dir backtest_results]"
        .to_string()
}

fn parse_date(value: &str) -> Result<NaiveDate, String> {
    polymarket_weather_predictor::utils::parse_date(value)
        .ok_or_else(|| format!("invalid date '{value}', expected YYYY-MM-DD"))
}

fn parse_opt<T: std::str::FromStr>(
    map: &HashMap<String, String>,
    key: &str,
    default: T,
) -> Result<T, String> {
    map.get(key)
        .map(|v| v.parse::<T>().map_err(|_| format!("invalid value for {key}: {v}")))
        .transpose()
        .map(|o| o.unwrap_or(default))
}
