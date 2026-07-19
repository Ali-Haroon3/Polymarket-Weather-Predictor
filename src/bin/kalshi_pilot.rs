//! Kalshi real-money PILOT — the evidence-backed strategy at pocket-change size.
//!
//! Encodes the configuration every slice of forward evidence agrees on (July 2026, ~600 settled
//! paper trades): KALSHI only, SELL only (executed as buying NO — max loss is the price paid),
//! lead ≥ 1 only, thresholded on the SHRUNK edge λ·(bid − model) with λ fitted per-venue on
//! resolved lead ≥ 1 captures, and the edge must ALSO clear Kalshi's trading fee plus a buffer —
//! the cost paper trading never modeled, and the number this pilot exists to measure.
//!
//! Sizing is FLAT stakes, not Kelly: at a couple-hundred-dollar bankroll the job is a clean,
//! uniform measurement sample (intended vs filled vs fee vs outcome), not compounding.
//!
//! Safety: DRY RUN unless `--live` is passed; the trade host defaults to Kalshi's DEMO
//! environment until `KALSHI_BASE_URL` points at production; `PILOT_DISABLE=1` is a kill switch;
//! total exposure and orders-per-run are hard-capped; and the ledger (`data/pilot_trades.jsonl`)
//! plus resting orders plus held positions all dedupe re-runs, so restarting the pilot can't
//! double-order a market. Every decision (including skips) is appended to the ledger.
//!
//!   cargo run --release --bin kalshi_pilot            # dry run: print + log intended orders
//!   cargo run --release --bin kalshi_pilot -- --live  # place real limit orders

use std::collections::HashSet;
use std::path::PathBuf;

use chrono::{NaiveDate, Utc};
use serde::{Deserialize, Serialize};

use polymarket_weather_predictor::api::kalshi_trade::{fee_frac, KalshiTradeClient};
use polymarket_weather_predictor::api::{KalshiHistoryDownloader, WeatherMarketRow};
use polymarket_weather_predictor::backtesting::{market_estimate, ShrinkageFit};
use polymarket_weather_predictor::config;
use polymarket_weather_predictor::data_pipeline::StationPricer;
use polymarket_weather_predictor::models::BayesianWeatherModel;
use polymarket_weather_predictor::types::SimulatedMarket;

/// Hard ceiling on capital committed across ALL pilot positions (dollars), unless overridden.
const DEFAULT_MAX_EXPOSURE: f64 = 200.0;
/// Flat stake per trade (dollars of NO-contract cost).
const DEFAULT_STAKE: f64 = 15.0;
/// Max new orders per run — a runaway-model backstop on top of the exposure cap.
const DEFAULT_MAX_ORDERS: usize = 5;
/// Shrunk edge must exceed threshold + fee + this buffer before an order is placed.
const DEFAULT_FEE_BUFFER: f64 = 0.01;

/// One ledger line: every decision the pilot makes, tradable or not, dry or live.
#[derive(Debug, Serialize, Deserialize)]
struct LedgerRow {
    run_at: chrono::DateTime<Utc>,
    ticker: String,
    city: String,
    target_date: NaiveDate,
    /// "order" (placed / would place), or a skip reason.
    decision: String,
    dry_run: bool,
    yes_bid: Option<f64>,
    no_price: Option<f64>,
    model_estimate: Option<f64>,
    lambda: f64,
    claimed_edge: Option<f64>,
    shrunk_edge: Option<f64>,
    fee_frac: Option<f64>,
    contracts: i64,
    cost: f64,
    order_id: Option<String>,
    order_status: Option<String>,
    error: Option<String>,
}

/// The subset of a capture row the λ fit needs. Extra fields in captures.jsonl are ignored.
#[derive(Debug, Deserialize)]
struct CaptureRow {
    captured_at: NaiveDate,
    target_date: NaiveDate,
    entry_price: f64,
    model_estimate: Option<f64>,
    outcome: Option<f64>,
    #[serde(default = "default_source")]
    source: String,
    #[serde(default)]
    best_bid: Option<f64>,
    #[serde(default)]
    best_ask: Option<f64>,
}

fn default_source() -> String {
    "polymarket".to_string()
}

struct PilotConfig {
    live: bool,
    stake: f64,
    max_exposure: f64,
    max_orders: usize,
    edge_threshold: f64,
    fee_buffer: f64,
    captures_path: PathBuf,
    ledger_path: PathBuf,
}

#[tokio::main]
async fn main() {
    if std::env::var("PILOT_DISABLE").map(|v| v == "1").unwrap_or(false) {
        eprintln!("PILOT_DISABLE=1 — kill switch engaged, exiting without doing anything.");
        return;
    }
    if let Err(e) = run().await {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn parse_args() -> PilotConfig {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| args.iter().any(|a| a == name);
    let val = |name: &str| {
        args.windows(2)
            .find(|w| w[0] == name)
            .map(|w| w[1].clone())
    };
    let fval = |name: &str, d: f64| val(name).and_then(|v| v.parse().ok()).unwrap_or(d);
    PilotConfig {
        live: flag("--live"),
        stake: fval("--stake", DEFAULT_STAKE),
        max_exposure: fval("--max-exposure", DEFAULT_MAX_EXPOSURE),
        max_orders: val("--max-orders")
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_MAX_ORDERS),
        edge_threshold: fval("--edge-threshold", config::backtest_params().edge_threshold),
        fee_buffer: fval("--fee-buffer", DEFAULT_FEE_BUFFER),
        captures_path: PathBuf::from(
            val("--captures").unwrap_or_else(|| "data/captures.jsonl".into()),
        ),
        ledger_path: PathBuf::from(
            val("--ledger").unwrap_or_else(|| "data/pilot_trades.jsonl".into()),
        ),
    }
}

async fn run() -> Result<(), String> {
    let cfg = parse_args();
    let today = Utc::now().date_naive();

    // λ from the same captures the paper evidence came from: resolved, lead ≥ 1 only.
    let lambda = fit_lambda_from_captures(&cfg.captures_path);
    println!(
        "λ(kalshi) = {:.3} (fitted from {})",
        lambda,
        cfg.captures_path.display()
    );

    // Trade client is required even for a dry run: a pilot that can't see its own positions
    // can't dedupe, and finding out auth is broken on arming day defeats the rehearsal.
    let trader = KalshiTradeClient::new().map_err(|e| e.to_string())?;
    println!(
        "Trade host: {} ({}) — {}",
        trader.host(),
        if trader.is_production() {
            "PRODUCTION, real money"
        } else {
            "demo/paper"
        },
        if cfg.live { "LIVE" } else { "DRY RUN" }
    );
    let balance = trader.balance().await.map_err(|e| e.to_string())?;
    let positions = trader.positions().await.map_err(|e| e.to_string())?;
    let resting = trader.resting_orders().await.map_err(|e| e.to_string())?;
    println!(
        "Balance ${balance:.2} · {} open positions · {} resting orders",
        positions.len(),
        resting.len()
    );

    // Dedupe set: anything held, resting, or ever decided "order" in the ledger.
    let mut committed: HashSet<String> = positions.iter().map(|p| p.ticker.clone()).collect();
    committed.extend(resting.iter().map(|o| o.ticker.clone()));
    committed.extend(load_ordered_tickers(&cfg.ledger_path));

    // Live open markets, priced exactly like the capture daemon prices them.
    let kalshi = KalshiHistoryDownloader::new();
    let markets = kalshi
        .download_weather_markets(true, 4000)
        .await
        .map_err(|e| format!("market fetch failed: {e}"))?;
    println!("{} open Kalshi weather markets", markets.len());

    let mut pricer = StationPricer::new(today);
    let mut ledger: Vec<LedgerRow> = Vec::new();
    let mut placed = 0usize;
    let mut exposure = position_cost_estimate(&positions);

    // Deterministic scan order (venue fetch order varies): by target date then ticker.
    let mut sorted: Vec<&WeatherMarketRow> = markets.iter().collect();
    sorted.sort_by(|a, b| (a.target_date, &a.market_id).cmp(&(b.target_date, &b.market_id)));

    for r in sorted {
        if committed.contains(&r.market_id) {
            continue; // silently: already handled in a previous run
        }
        let est = pricer.estimate(r).and_then(|(mu, sigma)| {
            let mut model = BayesianWeatherModel::default();
            model.set_point_forecast(mu, sigma);
            market_estimate(&model, &to_sim(r))
        });
        let d = decide_sell(
            today,
            r.target_date,
            r.best_bid,
            est,
            lambda,
            cfg.edge_threshold,
            cfg.fee_buffer,
        );
        let mut row = ledger_row(r, &d, est, lambda, cfg.live);
        if let Decision::Order {
            no_price,
            claimed: _,
            shrunk: _,
        } = d
        {
            if placed >= cfg.max_orders {
                row.decision = "skip_max_orders".into();
            } else {
                let contracts = size_contracts(cfg.stake, no_price);
                let cost = contracts as f64 * no_price;
                if contracts == 0 {
                    row.decision = "skip_stake_below_one_contract".into();
                } else if exposure + cost > cfg.max_exposure {
                    row.decision = "skip_exposure_cap".into();
                } else {
                    row.contracts = contracts;
                    row.cost = cost;
                    let cents = (no_price * 100.0).round() as i64;
                    // Stable per (ticker, day): a crashed-and-rerun pilot reuses the same id and
                    // Kalshi rejects the duplicate instead of double-filling.
                    let coid = format!("pilot-{}-{}", r.market_id, today);
                    if cfg.live {
                        match trader
                            .buy_no_limit(&r.market_id, contracts, cents, &coid)
                            .await
                        {
                            Ok(o) => {
                                row.order_id = Some(o.order_id);
                                row.order_status = Some(o.status);
                            }
                            Err(e) => {
                                row.decision = "order_error".into();
                                row.error = Some(e.to_string());
                            }
                        }
                    }
                    if row.error.is_none() {
                        placed += 1;
                        exposure += cost;
                        committed.insert(r.market_id.clone());
                    }
                }
            }
        }
        println!(
            "{:<28} {} lead={} bid={} est={} shrunk={} -> {}{}",
            r.market_id,
            r.target_date,
            (r.target_date - today).num_days(),
            fmt(r.best_bid),
            fmt(est),
            fmt(row.shrunk_edge),
            row.decision,
            if row.contracts > 0 {
                format!(" ({} @ ~${:.2})", row.contracts, row.cost)
            } else {
                String::new()
            }
        );
        ledger.push(row);
    }

    append_ledger(&cfg.ledger_path, &ledger)?;
    println!(
        "\n{} orders {} · ${:.2} committed exposure · full decision log appended to {}",
        placed,
        if cfg.live { "PLACED" } else { "would be placed (dry run)" },
        exposure,
        cfg.ledger_path.display()
    );
    Ok(())
}

/// The tradable decision for one market, or why not.
#[derive(Debug, PartialEq)]
enum Decision {
    /// SELL YES via BUY NO at `no_price` (= 1 − yes bid, the executable taker price).
    Order { no_price: f64, claimed: f64, shrunk: f64 },
    Skip(&'static str),
    /// Skip carrying the diagnostics that were computed before the threshold failed.
    SkipWithEdge {
        reason: &'static str,
        claimed: f64,
        shrunk: f64,
    },
}

/// The pilot's entire strategy in one pure function. SELL only (est below the executable bid),
/// lead ≥ 1 only, and the SHRUNK edge must clear threshold + fee + buffer. Fee is charged on the
/// NO price actually traded.
fn decide_sell(
    today: NaiveDate,
    target: NaiveDate,
    yes_bid: Option<f64>,
    est: Option<f64>,
    lambda: f64,
    threshold: f64,
    fee_buffer: f64,
) -> Decision {
    if (target - today).num_days() < 1 {
        return Decision::Skip("skip_lead0"); // day-of: the market's intraday info wins
    }
    let Some(est) = est else {
        return Decision::Skip("skip_no_estimate");
    };
    let Some(bid) = yes_bid.filter(|b| *b > 0.0 && *b < 1.0) else {
        return Decision::Skip("skip_no_bid");
    };
    let claimed = bid - est;
    let shrunk = lambda * claimed;
    if claimed <= 0.0 {
        return Decision::SkipWithEdge {
            reason: "skip_not_sell",
            claimed,
            shrunk,
        };
    }
    let no_price = 1.0 - bid;
    let required = threshold + fee_frac(no_price) + fee_buffer;
    if shrunk < required {
        return Decision::SkipWithEdge {
            reason: "skip_edge_below_costs",
            claimed,
            shrunk,
        };
    }
    Decision::Order {
        no_price,
        claimed,
        shrunk,
    }
}

/// Whole contracts a flat dollar stake buys at `no_price`. Never rounds up past the stake.
fn size_contracts(stake: f64, no_price: f64) -> i64 {
    if no_price <= 0.0 {
        return 0;
    }
    (stake / no_price).floor() as i64
}

/// λ for Kalshi from resolved lead ≥ 1 captures — the same fit and hygiene as the dashboard's
/// full-sample fit (book mid as reference price, lead ≤ 0 excluded).
fn fit_lambda_from_captures(path: &PathBuf) -> f64 {
    let mut fit = ShrinkageFit::default();
    let Ok(text) = std::fs::read_to_string(path) else {
        eprintln!(
            "warning: no captures at {} — λ falls back to 1.0 (NO shrink); pass --captures",
            path.display()
        );
        return 1.0;
    };
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let Ok(c) = serde_json::from_str::<CaptureRow>(line) else {
            continue;
        };
        let (Some(est), Some(outcome)) = (c.model_estimate, c.outcome) else {
            continue;
        };
        if (c.target_date - c.captured_at).num_days() < 1 {
            continue; // lead ≤ 0 prices already embed the outcome
        }
        let px = match (c.best_bid, c.best_ask) {
            (Some(b), Some(a)) if b > 0.0 && a < 1.0 && b <= a => (a + b) / 2.0,
            _ => c.entry_price,
        };
        if px > 0.0 && px < 1.0 {
            fit.observe(&c.source, est - px, outcome - px);
        }
    }
    fit.lambda("kalshi")
}

/// Dollars already tied up in held positions, conservatively estimated at worst-case $1/contract.
/// Kalshi doesn't return cost basis on this endpoint; over-counting exposure only makes the pilot
/// MORE cautious, never less.
fn position_cost_estimate(positions: &[polymarket_weather_predictor::api::KalshiPosition]) -> f64 {
    positions.iter().map(|p| p.position.unsigned_abs() as f64).sum()
}

fn load_ordered_tickers(path: &PathBuf) -> HashSet<String> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return HashSet::new();
    };
    text.lines()
        .filter_map(|l| serde_json::from_str::<LedgerRow>(l).ok())
        .filter(|r| r.decision == "order" && r.error.is_none())
        .map(|r| r.ticker)
        .collect()
}

fn append_ledger(path: &PathBuf, rows: &[LedgerRow]) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
    }
    let mut body = String::new();
    for r in rows {
        body.push_str(&serde_json::to_string(r).map_err(|e| e.to_string())?);
        body.push('\n');
    }
    use std::io::Write as _;
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut f| f.write_all(body.as_bytes()))
        .map_err(|e| format!("append {}: {e}", path.display()))
}

fn ledger_row(
    r: &WeatherMarketRow,
    d: &Decision,
    est: Option<f64>,
    lambda: f64,
    live: bool,
) -> LedgerRow {
    let (decision, no_price, claimed, shrunk) = match d {
        Decision::Order {
            no_price,
            claimed,
            shrunk,
        } => ("order".to_string(), Some(*no_price), Some(*claimed), Some(*shrunk)),
        Decision::Skip(reason) => (reason.to_string(), None, None, None),
        Decision::SkipWithEdge {
            reason,
            claimed,
            shrunk,
        } => (reason.to_string(), None, Some(*claimed), Some(*shrunk)),
    };
    LedgerRow {
        run_at: Utc::now(),
        ticker: r.market_id.clone(),
        city: r.city.clone(),
        target_date: r.target_date,
        decision,
        dry_run: !live,
        yes_bid: r.best_bid,
        no_price,
        model_estimate: est,
        lambda,
        claimed_edge: claimed,
        shrunk_edge: shrunk,
        fee_frac: no_price.map(fee_frac),
        contracts: 0,
        cost: 0.0,
        order_id: None,
        order_status: None,
        error: None,
    }
}

fn to_sim(r: &WeatherMarketRow) -> SimulatedMarket {
    SimulatedMarket {
        date: r.target_date,
        market_id: r.market_id.clone(),
        market_title: r.market_title.clone(),
        market_type: r.market_type.clone(),
        threshold: r.threshold,
        threshold_upper: r.threshold_upper,
        unit: r.unit.clone(),
        market_price: r.price,
        actual_outcome: 0.0, // unknown for an open market
        city: r.city.clone(),
    }
}

fn fmt(x: Option<f64>) -> String {
    x.map_or("—".into(), |v| format!("{v:.3}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(s: &str) -> NaiveDate {
        NaiveDate::parse_from_str(s, "%Y-%m-%d").unwrap()
    }

    #[test]
    fn sell_decision_encodes_the_evidence_backed_filters() {
        let today = d("2026-07-19");
        let tomorrow = d("2026-07-20");
        // Clean SELL: bid 0.40, est 0.10, λ 0.5 → shrunk 0.15; NO at 0.60 costs
        // fee_frac(0.6) ≈ 0.0168; required = 0.05 + 0.0168 + 0.01 ≈ 0.077 → order.
        match decide_sell(today, tomorrow, Some(0.40), Some(0.10), 0.5, 0.05, 0.01) {
            Decision::Order {
                no_price,
                claimed,
                shrunk,
            } => {
                assert!((no_price - 0.60).abs() < 1e-9);
                assert!((claimed - 0.30).abs() < 1e-9);
                assert!((shrunk - 0.15).abs() < 1e-9);
            }
            other => panic!("expected order, got {other:?}"),
        }

        // Same numbers at lead 0 → skipped (day-of markets are the market's edge, not ours).
        assert_eq!(
            decide_sell(today, today, Some(0.40), Some(0.10), 0.5, 0.05, 0.01),
            Decision::Skip("skip_lead0")
        );

        // BUY signal (est above bid) is never traded.
        assert!(matches!(
            decide_sell(today, tomorrow, Some(0.40), Some(0.70), 0.5, 0.05, 0.01),
            Decision::SkipWithEdge {
                reason: "skip_not_sell",
                ..
            }
        ));

        // A raw edge that clears the threshold but whose SHRUNK edge can't pay the fee → skip.
        // bid 0.40, est 0.28 → claimed 0.12, λ 0.5 → shrunk 0.06 < 0.077 required.
        assert!(matches!(
            decide_sell(today, tomorrow, Some(0.40), Some(0.28), 0.5, 0.05, 0.01),
            Decision::SkipWithEdge {
                reason: "skip_edge_below_costs",
                ..
            }
        ));

        // λ = 0 (anti-signal venue) turns everything off.
        assert!(matches!(
            decide_sell(today, tomorrow, Some(0.40), Some(0.05), 0.0, 0.05, 0.01),
            Decision::SkipWithEdge {
                reason: "skip_edge_below_costs",
                ..
            }
        ));

        // No book / degenerate bid → skip.
        assert_eq!(
            decide_sell(today, tomorrow, None, Some(0.10), 0.5, 0.05, 0.01),
            Decision::Skip("skip_no_bid")
        );
        assert_eq!(
            decide_sell(today, tomorrow, Some(1.0), Some(0.10), 0.5, 0.05, 0.01),
            Decision::Skip("skip_no_bid")
        );
    }

    #[test]
    fn flat_sizing_floors_and_never_overspends() {
        assert_eq!(size_contracts(15.0, 0.60), 25);
        assert_eq!(size_contracts(15.0, 0.95), 15);
        // Stake below one contract → 0 (the caller skips, never rounds up).
        assert_eq!(size_contracts(0.50, 0.60), 0);
        assert_eq!(size_contracts(15.0, 0.0), 0);
        // Cost check: floor guarantees cost ≤ stake.
        let c = size_contracts(15.0, 0.61);
        assert!(c as f64 * 0.61 <= 15.0);
    }

    #[test]
    fn lambda_fit_reads_capture_lines_and_skips_lead0() {
        let dir = std::env::temp_dir().join("pilot_test_captures");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("captures.jsonl");
        // ≥ MIN_N resolved kalshi lead-1 rows with predicted edge 0.2, realized 0.1 → λ = 0.5;
        // plus a lead-0 row with realized = predicted that would bias λ up if not excluded.
        let mut lines: Vec<String> = (0..ShrinkageFit::MIN_N)
            .map(|i| {
                format!(
                    r#"{{"captured_at":"2026-07-01","target_date":"2026-07-02","market_id":"m{i}","market_title":"t","market_type":"temp_bucket","threshold":1.0,"threshold_upper":null,"unit":"F","city":"NYC","entry_price":0.5,"model_estimate":0.7,"outcome":0.6,"source":"kalshi"}}"#
                )
            })
            .collect();
        lines.push(
            r#"{"captured_at":"2026-07-02","target_date":"2026-07-02","market_id":"day0","market_title":"t","market_type":"temp_bucket","threshold":1.0,"threshold_upper":null,"unit":"F","city":"NYC","entry_price":0.5,"model_estimate":0.9,"outcome":0.9,"source":"kalshi"}"#.to_string(),
        );
        std::fs::write(&path, lines.join("\n")).unwrap();
        let lambda = fit_lambda_from_captures(&path);
        assert!(
            (lambda - 0.5).abs() < 1e-9,
            "outcome−price = 0.1 over est−price = 0.2 ⇒ λ = 0.5, lead-0 row excluded; got {lambda}"
        );
        // Missing file → 1.0 (no shrink) with a warning, never a crash.
        assert_eq!(fit_lambda_from_captures(&dir.join("nope.jsonl")), 1.0);
    }
}
