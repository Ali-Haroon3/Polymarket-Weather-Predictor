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
//! Automatic circuit breakers (added after the first negative out-of-sample week, Jul 13–19):
//! the pilot STANDS DOWN — no orders, before the trade API is even touched — when the fitted λ
//! falls below `--lambda-floor` (realized edge too thin to be worth trading) or when its own
//! settled orders lost more than `--max-weekly-loss` dollars over the trailing 7 days. Both are
//! mode-scoped rehearsals: dry runs are gated by dry-run ledger rows, live runs by live ones.
//! Correlated exposure is capped per (city, target day) via `--max-city-exposure` — N bucket
//! markets on one city-day are one weather bet, not N independent bets.
//!
//!   cargo run --release --bin kalshi_pilot            # dry run: print + log intended orders
//!   cargo run --release --bin kalshi_pilot -- --live  # place real limit orders

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use chrono::{NaiveDate, Utc};
use serde::{Deserialize, Serialize};

use polymarket_weather_predictor::api::kalshi_trade::{fee_frac, KalshiTradeClient};
use polymarket_weather_predictor::api::{KalshiHistoryDownloader, WeatherMarketRow};
use polymarket_weather_predictor::backtesting::{lambda_segment, market_estimate, ShrinkageFit};
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
/// Stand down when the fitted λ drops below this: at λ < 0.2 the model realizes under a fifth of
/// the edge it claims, and the correct stake is zero. (λ fell 0.48 → 0.38 over Jul 13–19; this
/// floor turns "keep an eye on it" into an automatic stop.)
const DEFAULT_LAMBDA_FLOOR: f64 = 0.2;
/// Stand down when the pilot's own settled orders lost more than this (dollars) over the
/// trailing 7 days. Resuming after a trip is a human decision (raise the flag or wait it out).
const DEFAULT_MAX_WEEKLY_LOSS: f64 = 50.0;
/// Per-(city, target-day) exposure cap, in multiples of the stake: bucket markets on the same
/// city-day settle on the SAME daily high, so stacking them is pyramiding one bet.
const DEFAULT_CITY_EXPOSURE_STAKES: f64 = 2.0;

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

/// The subset of a capture row the λ fit and the loss breaker need. Extra fields in
/// captures.jsonl are ignored.
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
    #[serde(default)]
    market_id: Option<String>,
    /// For the λ-fit hygiene filter (temperature markets only, matching the dashboard and
    /// `scripts/lambda_diagnostics.py`). Defaults empty for old rows, which then don't pass the
    /// `starts_with("temp")` gate — every real capture carries the field.
    #[serde(default)]
    market_type: String,
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
    lambda_floor: f64,
    max_weekly_loss: f64,
    max_city_exposure: f64,
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
    let stake = fval("--stake", DEFAULT_STAKE);
    PilotConfig {
        live: flag("--live"),
        stake,
        max_exposure: fval("--max-exposure", DEFAULT_MAX_EXPOSURE),
        max_orders: val("--max-orders")
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_MAX_ORDERS),
        edge_threshold: fval("--edge-threshold", config::backtest_params().edge_threshold),
        fee_buffer: fval("--fee-buffer", DEFAULT_FEE_BUFFER),
        lambda_floor: fval("--lambda-floor", DEFAULT_LAMBDA_FLOOR),
        max_weekly_loss: fval("--max-weekly-loss", DEFAULT_MAX_WEEKLY_LOSS),
        max_city_exposure: fval("--max-city-exposure", DEFAULT_CITY_EXPOSURE_STAKES * stake),
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

    // Circuit breakers — evaluated BEFORE the trade API is touched, so a tripped breaker can
    // never be defeated by an auth failure path or a partial run.
    let (week_pnl, week_settled) =
        realized_week_pnl(&cfg.ledger_path, &cfg.captures_path, today, cfg.live);
    if week_settled > 0 {
        println!(
            "Trailing-7-day realized PnL: ${week_pnl:+.2} over {week_settled} settled {} orders",
            if cfg.live { "live" } else { "dry-run" },
        );
    }
    if let Some(reason) = stand_down_reason(
        lambda,
        cfg.lambda_floor,
        week_pnl,
        week_settled,
        cfg.max_weekly_loss,
    ) {
        eprintln!(
            "STAND DOWN: {reason}. No orders this run. Resuming is a human decision — \
             re-run with --lambda-floor / --max-weekly-loss overridden once you've looked."
        );
        return Ok(());
    }

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
    // Live orders can never commit more than the funds actually there, whatever --max-exposure
    // says. Dry runs keep the configured cap so the ledger shows what a funded account would do.
    let exposure_cap = if cfg.live {
        cfg.max_exposure.min(balance)
    } else {
        cfg.max_exposure
    };

    // Dedupe set: anything held, resting, or ever decided "order" in the ledger.
    let mut committed: HashSet<String> = positions.iter().map(|p| p.ticker.clone()).collect();
    committed.extend(resting.iter().map(|o| o.ticker.clone()));
    committed.extend(load_ordered_tickers(&cfg.ledger_path));

    // Correlated-exposure ledger: dollars already committed per (city, target day) by earlier
    // runs whose markets are still open. Same-mode rows only, like the loss breaker.
    let mut city_exposure = open_city_exposure(&cfg.ledger_path, today, cfg.live);

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
                let city_key = (r.city.clone(), r.target_date);
                let city_spent = city_exposure.get(&city_key).copied().unwrap_or(0.0);
                if contracts == 0 {
                    row.decision = "skip_stake_below_one_contract".into();
                } else if exposure + cost > exposure_cap {
                    row.decision = "skip_exposure_cap".into();
                } else if city_spent + cost > cfg.max_city_exposure {
                    // Same city-day = same daily high = one bet. Don't pyramid it.
                    row.decision = "skip_city_exposure_cap".into();
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
                        *city_exposure.entry(city_key).or_insert(0.0) += cost;
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
        exposure.max(0.0), // .max(0.0) irons out "-0.00" (negative-zero display artifact)
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
        if !c.market_type.starts_with("temp") {
            continue; // λ hygiene: temperature markets only, same as the dashboard's fit
        }
        let px = match (c.best_bid, c.best_ask) {
            (Some(b), Some(a)) if b > 0.0 && a < 1.0 && b <= a => (a + b) / 2.0,
            _ => c.entry_price,
        };
        if px > 0.0 && px < 1.0 {
            // Tagged by price band even though the pilot still trades the venue fold: venue
            // lookups are unchanged by tagging (proven by the fold-identity test in shrinkage.rs),
            // and an untagged fit would make a future `lambda_seg` call a silent no-op.
            fit.observe_seg(&c.source, lambda_segment(px), est - px, outcome - px);
        }
    }
    fit.lambda("kalshi")
}

/// Why the pilot must not trade this run, if any breaker tripped. λ floor first: a too-thin
/// realized edge makes the loss question moot. The loss breaker only arms once at least one
/// order has actually settled — an empty ledger (or a fresh week) is not a loss.
fn stand_down_reason(
    lambda: f64,
    lambda_floor: f64,
    week_pnl: f64,
    week_settled: usize,
    max_weekly_loss: f64,
) -> Option<String> {
    if lambda < lambda_floor {
        return Some(format!(
            "λ {lambda:.3} is below the {lambda_floor:.2} floor — the model realizes too little \
             of its claimed edge to be worth trading"
        ));
    }
    if week_settled > 0 && week_pnl < -max_weekly_loss {
        return Some(format!(
            "trailing-7-day realized PnL ${week_pnl:+.2} breaches the −${max_weekly_loss:.2} \
             weekly loss breaker"
        ));
    }
    None
}

/// Realized PnL (dollars) of the pilot's own orders whose markets settled in the trailing 7
/// days, joined against capture outcomes; returns (pnl, settled-order count). Mode-scoped: live
/// runs are judged by live orders and dry runs by dry-run orders, so the breaker logic rehearses
/// during the dry-run phase but paper losses can never trip a funded run (arming live after a
/// bad paper week is a human call, not this function's).
///
/// Fills are assumed: a resting limit that never filled is counted as if it did. That can only
/// overstate a loss (the order was placed because the model liked it), so the breaker errs
/// toward standing down — acceptable for a safety rail, not for PnL reporting.
fn realized_week_pnl(
    ledger_path: &PathBuf,
    captures_path: &PathBuf,
    today: NaiveDate,
    live: bool,
) -> (f64, usize) {
    let Ok(text) = std::fs::read_to_string(ledger_path) else {
        return (0.0, 0);
    };
    let outcomes = load_outcomes(captures_path);
    let week_ago = today - chrono::Duration::days(7);
    let (mut pnl, mut settled) = (0.0, 0usize);
    for row in text
        .lines()
        .filter_map(|l| serde_json::from_str::<LedgerRow>(l).ok())
    {
        if row.decision != "order" || row.error.is_some() || row.dry_run == live {
            continue;
        }
        if row.target_date < week_ago || row.target_date >= today {
            continue; // outside the trailing week, or not yet settled
        }
        let (Some(no_price), Some(outcome)) = (row.no_price, outcomes.get(&row.ticker).copied())
        else {
            continue;
        };
        // BUY NO pays $1/contract when the market resolves NO (outcome 0); fee paid either way.
        let per_contract = (1.0 - outcome) - no_price - fee_frac(no_price);
        pnl += row.contracts as f64 * per_contract;
        settled += 1;
    }
    (pnl, settled)
}

/// Resolved outcome per ticker from captures.jsonl (the capture daemon fills outcomes in as
/// markets settle, so this is the pilot's resolution source too — no extra API surface).
fn load_outcomes(captures_path: &PathBuf) -> HashMap<String, f64> {
    let Ok(text) = std::fs::read_to_string(captures_path) else {
        return HashMap::new();
    };
    let mut out = HashMap::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let Ok(c) = serde_json::from_str::<CaptureRow>(line) else {
            continue;
        };
        if let (Some(id), Some(outcome)) = (c.market_id, c.outcome) {
            out.insert(id, outcome);
        }
    }
    out
}

/// Dollars committed per (city, target day) by prior runs' orders on still-open markets —
/// same-mode rows only, mirroring `realized_week_pnl` — so the per-city cap holds across
/// restarts, not just within one run.
fn open_city_exposure(
    ledger_path: &PathBuf,
    today: NaiveDate,
    live: bool,
) -> HashMap<(String, NaiveDate), f64> {
    let Ok(text) = std::fs::read_to_string(ledger_path) else {
        return HashMap::new();
    };
    let mut out: HashMap<(String, NaiveDate), f64> = HashMap::new();
    for row in text
        .lines()
        .filter_map(|l| serde_json::from_str::<LedgerRow>(l).ok())
    {
        if row.decision == "order"
            && row.error.is_none()
            && row.dry_run != live
            && row.target_date >= today
        {
            *out.entry((row.city, row.target_date)).or_insert(0.0) += row.cost;
        }
    }
    out
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

    /// A minimal ledger "order" row for breaker tests.
    fn order_row(ticker: &str, city: &str, target: &str, no_price: f64, contracts: i64, dry: bool) -> LedgerRow {
        LedgerRow {
            run_at: Utc::now(),
            ticker: ticker.into(),
            city: city.into(),
            target_date: d(target),
            decision: "order".into(),
            dry_run: dry,
            yes_bid: Some(1.0 - no_price),
            no_price: Some(no_price),
            model_estimate: Some(0.1),
            lambda: 0.4,
            claimed_edge: Some(0.3),
            shrunk_edge: Some(0.12),
            fee_frac: Some(fee_frac(no_price)),
            contracts,
            cost: contracts as f64 * no_price,
            order_id: None,
            order_status: None,
            error: None,
        }
    }

    fn write_jsonl<T: serde::Serialize>(path: &PathBuf, rows: &[T]) {
        let body: String = rows
            .iter()
            .map(|r| serde_json::to_string(r).unwrap() + "\n")
            .collect();
        std::fs::write(path, body).unwrap();
    }

    #[test]
    fn stand_down_trips_on_lambda_floor_and_weekly_loss() {
        // Healthy: λ above floor, small profit.
        assert_eq!(stand_down_reason(0.35, 0.2, 4.0, 3, 50.0), None);
        // λ below floor trips regardless of PnL.
        assert!(stand_down_reason(0.19, 0.2, 100.0, 3, 50.0).is_some());
        // Loss beyond the line trips…
        assert!(stand_down_reason(0.35, 0.2, -50.01, 3, 50.0).is_some());
        // …but only once at least one order has settled: a fresh ledger can't trip it.
        assert_eq!(stand_down_reason(0.35, 0.2, -50.01, 0, 50.0), None);
        // Loss exactly at the line does not trip (breach is strict).
        assert_eq!(stand_down_reason(0.35, 0.2, -50.0, 3, 50.0), None);
    }

    #[test]
    fn weekly_pnl_joins_ledger_orders_to_capture_outcomes() {
        let dir = std::env::temp_dir().join("pilot_test_weekly_pnl");
        let _ = std::fs::create_dir_all(&dir);
        let ledger = dir.join("ledger.jsonl");
        let captures = dir.join("captures.jsonl");
        let today = d("2026-07-20");

        write_jsonl(
            &ledger,
            &[
                order_row("WIN", "NYC", "2026-07-18", 0.60, 10, false), // resolved NO → win
                order_row("LOSE", "NYC", "2026-07-19", 0.60, 10, false), // resolved YES → lose
                order_row("OLD", "NYC", "2026-07-10", 0.60, 10, false), // outside the window
                order_row("FUT", "NYC", "2026-07-22", 0.60, 10, false), // not settled yet
                order_row("DRY", "NYC", "2026-07-18", 0.60, 10, true), // wrong mode for live
            ],
        );
        // Outcomes come from capture rows (only market_id + outcome matter for the join).
        let cap_line = |id: &str, outcome: f64| {
            format!(
                r#"{{"captured_at":"2026-07-17","target_date":"2026-07-18","market_id":"{id}","entry_price":0.5,"model_estimate":0.4,"outcome":{outcome},"source":"kalshi"}}"#
            )
        };
        std::fs::write(
            &captures,
            [cap_line("WIN", 0.0), cap_line("LOSE", 1.0), cap_line("OLD", 0.0)].join("\n"),
        )
        .unwrap();

        let (pnl, settled) = realized_week_pnl(&ledger, &captures, today, true);
        assert_eq!(settled, 2, "only WIN and LOSE are live, in-window, and resolved");
        // WIN: 10×(0.40 − fee), LOSE: 10×(−0.60 − fee), fee = 0.07·0.6·0.4 = 0.0168.
        let fee = fee_frac(0.60);
        let expect = 10.0 * (0.40 - fee) + 10.0 * (-0.60 - fee);
        assert!((pnl - expect).abs() < 1e-9, "got {pnl}, want {expect}");

        // Dry mode sees only the dry row (which resolved NO via WIN? no — DRY has no outcome
        // under its own ticker), so nothing settles.
        let (_, dry_settled) = realized_week_pnl(&ledger, &captures, today, false);
        assert_eq!(dry_settled, 0, "DRY ticker has no capture outcome");
    }

    #[test]
    fn city_exposure_accumulates_same_mode_open_orders() {
        let dir = std::env::temp_dir().join("pilot_test_city_exposure");
        let _ = std::fs::create_dir_all(&dir);
        let ledger = dir.join("ledger.jsonl");
        let today = d("2026-07-20");
        write_jsonl(
            &ledger,
            &[
                order_row("A", "NYC", "2026-07-21", 0.50, 20, false), // $10 open
                order_row("B", "NYC", "2026-07-21", 0.25, 20, false), // $5 more, same city-day
                order_row("C", "NYC", "2026-07-19", 0.50, 20, false), // already settled → out
                order_row("D", "Denver", "2026-07-21", 0.50, 20, true), // wrong mode
            ],
        );
        let exp = open_city_exposure(&ledger, today, true);
        assert_eq!(exp.len(), 1);
        let nyc = exp[&("NYC".to_string(), d("2026-07-21"))];
        assert!((nyc - 15.0).abs() < 1e-9, "20×0.50 + 20×0.25 = $15, got {nyc}");
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
