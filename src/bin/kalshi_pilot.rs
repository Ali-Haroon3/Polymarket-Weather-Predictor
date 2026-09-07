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
//! of the TRADED segment falls below `--lambda-floor` (realized edge too thin to be worth
//! trading) or when its own settled orders lost more than `--max-weekly-loss` dollars over the
//! trailing 7 days. Both are mode-scoped rehearsals: dry runs are gated by dry-run ledger rows,
//! live runs by live ones. Correlated exposure is capped per (city, target day) via
//! `--max-city-exposure` — N bucket markets on one city-day are one weather bet, not N
//! independent bets.
//!
//! λ enters in two places, and BOTH read the candidate's own segment since 2026-08-09. The FLOOR
//! BREAKER (2026-07-30) watches the λ of the segment the pilot's orders actually live in —
//! px ≥ 0.10 via the shared `lambda_segment` boundary — because the venue fold averages in the
//! sub-10¢ tail, a segment whose candidates essentially never clear threshold + fee and whose
//! negative λ was dragging the fold toward the floor (0.33 and trailing ~0.21 vs +0.51 for the
//! traded band at the 2026-07-26 look). The EDGE GATE originally kept the venue fold, waiting on
//! the dashboard's Filter A/B rows — but the fold has the same dilution problem there, and it
//! starved the paper sample outright: the first four CI dry runs (2026-08-05..08) produced ZERO
//! orders, with the best candidate's fold-shrunk edge (0.31 × 0.207 = 0.064) under the required
//! 0.077 while its own ≥ 10¢ band fitted λ = 0.44. A gate that prices every candidate with a tail
//! it cannot trade measures nothing, so since 2026-08-09 each candidate's edge is shrunk by
//! `lambda_seg` at its OWN bid band (segment → venue fold → pooled fallback on thin samples, so
//! young data behaves exactly like the old gate). Sub-10¢ bands with a negative fitted slope
//! clamp to λ = 0 and can never clear the gate — an adaptive price floor — and the per-order
//! guard still applies `--lambda-floor` at the candidate's band as defense in depth (binding when
//! a large claimed edge would clear the gate at a band λ under the floor, e.g. λ 0.15 × 0.60).
//!
//! Since 2026-08-25 the same two ideas — no fit, no trade; a fitted anti-signal slope clamps to
//! zero — are ALSO applied on the CITY axis (`city_gate`), because the tradeable universe turned
//! out not to be static. PR #34 (2026-08-19) fixed title-less Kalshi parsing and the eight
//! `KXHIGHT*` cities, which had never produced a single capture row, entered the universe at once
//! carrying per-(city, lead) bias/σ constants fitted Jan–Apr 2026 that neither seasonal refit
//! (07-20, 08-05) could touch — a city with no captures contributes nothing to a refit. Vegas
//! priced four straight days 2–4σ below the realized high (mean z +2.52; its 24 rows fit λ −0.84)
//! and took four of the pilot's next six paper orders, both settled ones losing. So a (kalshi,
//! city) is now withheld until it has `ShrinkageFit::MIN_N` resolved rows of its own, and withheld
//! after that whenever its own fitted λ sits under `--lambda-floor`. Both gates reuse constants
//! that were already argued for elsewhere; neither adds a city list to keep up to date. On the
//! captures as of 2026-08-25 the rule withholds all eight new cities on coverage and Denver
//! (λ −0.20) and Chicago (λ −0.08) on slope — reproducing, from data alone, exactly the pair the
//! dashboard's hand-picked `skip_anti_lambda_cities` row froze on 08-17 and has been unable to
//! extend since.
//!
//! The REAL-MONEY path (2026-09-07) closes the gaps a paper run never exercised. Every live
//! limit order carries an `expiration_ts` (`--order-ttl-mins`, default 4 h): the order is priced
//! off the ~15:00 UTC capture geometry at lead ≥ 1, and a good-till-cancelled remainder that
//! filled the next morning would be exactly the lead-0 trade the evidence says loses. Each run
//! then RECONCILES its earlier live orders — asks Kalshi for the order and its fills and appends
//! a `fill` (or `unfilled`) row naming what actually executed, at what average NO price, with
//! which terminal status — so the ledger holds intended-vs-filled per order, the number the
//! pilot exists to measure. A remainder still resting once its target day has arrived is
//! cancelled by a live run before anything else happens (a dry run only says it would). The
//! loss breaker, the city-exposure cap and the go-live gate read the fill row when one exists
//! and fall back to the intended fill (conservative: assumes the whole order filled) until it
//! does. Ledger dedupe is same-mode only, like every other ledger read: a dry row is a paper
//! decision and must not block the live order it rehearsed. The daily-capture Action is the
//! canonical driver in BOTH modes — it runs `--live` when the repository variable `PILOT_LIVE`
//! is `1` and the `KALSHI_API_KEY_ID` / `KALSHI_PRIVATE_KEY_PEM` secrets are set, against
//! `vars.KALSHI_BASE_URL` (unset ⇒ the demo host), so arming real money is two settings in the
//! GitHub UI and never a code change.
//!
//!   cargo run --release --bin kalshi_pilot            # dry run: print + log intended orders
//!   cargo run --release --bin kalshi_pilot -- --live  # place real limit orders

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};

use polymarket_weather_predictor::api::kalshi_trade::{
    fee_frac, KalshiFill, KalshiOrder, KalshiTradeClient,
};
use polymarket_weather_predictor::api::{KalshiHistoryDownloader, WeatherMarketRow};
use polymarket_weather_predictor::backtesting::spread_sigma::{
    fit_spread_sigma_scale, spread_obs, SpreadObs,
};
use polymarket_weather_predictor::backtesting::{
    lambda_segment, market_estimate, segment_veto, SegmentVeto, ShrinkageFit,
};
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
/// Stand down when the fitted λ of the TRADED segment (px ≥ 0.10) drops below this: at λ < 0.2
/// the model realizes under a fifth of the edge it claims, and the correct stake is zero. (λ fell
/// 0.48 → 0.38 over Jul 13–19; this floor turns "keep an eye on it" into an automatic stop.)
const DEFAULT_LAMBDA_FLOOR: f64 = 0.2;
/// Stand down when the pilot's own settled orders lost more than this (dollars) over the
/// trailing 7 days. Resuming after a trip is a human decision (raise the flag or wait it out).
const DEFAULT_MAX_WEEKLY_LOSS: f64 = 50.0;
/// Per-(city, target-day) exposure cap, in multiples of the stake: bucket markets on the same
/// city-day settle on the SAME daily high, so stacking them is pyramiding one bet.
const DEFAULT_CITY_EXPOSURE_STAKES: f64 = 2.0;
/// How long an unfilled live limit order may rest (minutes) before Kalshi expires it. A limit at
/// the executable bid fills as a taker at once when the book has size; whatever rests is the
/// market having moved away, and chasing it into the target day is the lead-0 trade the
/// strategy forbids. 4 h from a ~15:00 UTC run ends before any US city's local day rolls over.
/// 0 ⇒ good-till-cancelled (not recommended).
const DEFAULT_ORDER_TTL_MINS: i64 = 240;
/// Stop asking Kalshi about a live order this many days after its target day: by then its
/// market has long settled and the answer cannot change what the pilot does next.
const RECONCILE_GIVE_UP_DAYS: i64 = 14;

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
    /// The rest exist only for the spread→σ scale fit (`fit_spread_scale_from_captures`), which
    /// re-prices resolved rows and so needs the full market shape plus the stored forecast and
    /// logged ensemble spread. All defaulted so a row missing any of them still feeds the λ fit.
    #[serde(default)]
    city: String,
    #[serde(default)]
    threshold: Option<f64>,
    #[serde(default)]
    threshold_upper: Option<f64>,
    #[serde(default)]
    unit: Option<String>,
    #[serde(default)]
    forecast_high: Option<f64>,
    #[serde(default)]
    forecast_sigma: Option<f64>,
    #[serde(default)]
    ensemble_spread_ecmwf: Option<f64>,
    #[serde(default)]
    ensemble_spread_gfs: Option<f64>,
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
    order_ttl_mins: i64,
    captures_path: PathBuf,
    ledger_path: PathBuf,
}

#[tokio::main]
async fn main() {
    if std::env::var("PILOT_DISABLE")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
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
    let val = |name: &str| args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone());
    let fval = |name: &str, d: f64| val(name).and_then(|v| v.parse().ok()).unwrap_or(d);
    let stake = fval("--stake", DEFAULT_STAKE);
    PilotConfig {
        live: flag("--live"),
        stake,
        max_exposure: fval("--max-exposure", DEFAULT_MAX_EXPOSURE),
        max_orders: val("--max-orders")
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_MAX_ORDERS),
        // 0.10 since 2026-08-17, matching the dashboard's promoted default: with shrunk edges a
        // 10% floor led its forward A/B window at both checkpoints (+32.7% vs +15.0%, n=77, at
        // promotion). The old source, config::backtest_params().edge_threshold (0.05), still
        // governs the raw-edge backtest engine, where 10% was never the validated cut.
        edge_threshold: fval("--edge-threshold", 0.10),
        fee_buffer: fval("--fee-buffer", DEFAULT_FEE_BUFFER),
        lambda_floor: fval("--lambda-floor", DEFAULT_LAMBDA_FLOOR),
        max_weekly_loss: fval("--max-weekly-loss", DEFAULT_MAX_WEEKLY_LOSS),
        max_city_exposure: fval("--max-city-exposure", DEFAULT_CITY_EXPOSURE_STAKES * stake),
        order_ttl_mins: val("--order-ttl-mins")
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_ORDER_TTL_MINS),
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

    // λ from the same captures the paper evidence came from: resolved, lead ≥ 1 only. Each
    // candidate's edge gate reads its OWN bid band via `gate_lambda` (see module docs); the
    // breaker watches the traded (px ≥ 0.10) band; the venue fold remains only as the thin-
    // segment fallback inside `lambda_seg` and the logged value for no-bid rows.
    let fits = fit_shrinkage_from_captures(&cfg.captures_path);
    let fit = &fits.band;
    let lambda = fit.lambda("kalshi");
    let traded_seg = lambda_segment(0.10); // boundary value ⇒ the ≥ 10¢ band's canonical name
    let lambda_traded = fit.lambda_seg("kalshi", traded_seg);
    println!(
        "λ(kalshi fold) = {lambda:.3} (fallback) · λ({traded_seg}) = {lambda_traded:.3} for the \
         floor breaker; each candidate's gate is shrunk at its own bid band (fitted from {})",
        cfg.captures_path.display()
    );

    // Spread→σ scale from the same captures, so the pilot prices with the SAME σ regime the
    // capture daemon stores evidence under (both feed `StationPricer::new` below).
    let spread_scale = fit_spread_sigma_scale(&spread_obs_from_captures(&cfg.captures_path));
    match spread_scale {
        Some(a) => println!(
            "spread→σ: a = {a:.1} — lead ≥ 1 rows with an ensemble spread price at \
             σ = max(a·spread, 0.2 °C), same as the capture daemon"
        ),
        None => println!("spread→σ: too few resolved rows carrying spread — constant σ tables"),
    }

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
        lambda_traded,
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

    // Trade client: REQUIRED live, attempted for a dry run. With credentials a dry run rehearses
    // auth and position/resting dedupe exactly like arming day; without them (the cred-free CI
    // dry run) it degrades — loudly — to ledger-only dedupe rather than not rehearsing at all,
    // because the daily decision sample is the evidence the go-live gate needs.
    let trader = match KalshiTradeClient::new() {
        Ok(t) => Some(t),
        Err(e) if !cfg.live => {
            eprintln!(
                "warning: {e} — dry run continues with LEDGER-ONLY dedupe (no balance, no \
                 position/resting-order rehearsal)"
            );
            None
        }
        Err(e) => return Err(e.to_string()),
    };
    let (balance, positions, resting) = match &trader {
        Some(t) => {
            println!(
                "Trade host: {} ({}) — {}",
                t.host(),
                if t.is_production() {
                    "PRODUCTION, real money"
                } else {
                    "demo/paper"
                },
                if cfg.live { "LIVE" } else { "DRY RUN" }
            );
            if cfg.live && t.is_production() {
                println!(
                    "*** REAL MONEY *** stake ${:.2} · max {} orders/run · exposure cap ${:.2} · \
                     city cap ${:.2} · order TTL {} min · weekly-loss breaker ${:.2}",
                    cfg.stake,
                    cfg.max_orders,
                    cfg.max_exposure,
                    cfg.max_city_exposure,
                    cfg.order_ttl_mins,
                    cfg.max_weekly_loss
                );
            }
            let balance = t.balance().await.map_err(|e| e.to_string())?;
            let positions = t.positions().await.map_err(|e| e.to_string())?;
            let resting = t.resting_orders().await.map_err(|e| e.to_string())?;
            println!(
                "Balance ${balance:.2} · {} open positions · {} resting orders",
                positions.len(),
                resting.len()
            );
            (balance, positions, resting)
        }
        None => (0.0, Vec::new(), Vec::new()),
    };

    // What became of the live orders placed by earlier runs: fills, expiries, and — in a live
    // run — the cancel of anything still resting on its target day. Written down before today's
    // decisions so the breaker's next read and the go-live gate see real fills, not intentions.
    if let Some(t) = &trader {
        let reconciled = reconcile_orders(t, &cfg.ledger_path, today, cfg.live).await;
        if !reconciled.is_empty() {
            append_ledger(&cfg.ledger_path, &reconciled)?;
        }
    }
    // Live orders can never commit more than the funds actually there, whatever --max-exposure
    // says. Dry runs keep the configured cap so the ledger shows what a funded account would do.
    let exposure_cap = if cfg.live {
        cfg.max_exposure.min(balance)
    } else {
        cfg.max_exposure
    };

    // Dedupe set: anything held, resting, or decided "order" by an earlier run IN THIS MODE —
    // a dry row is a paper decision and must not block the live order it rehearsed.
    let mut committed: HashSet<String> = positions.iter().map(|p| p.ticker.clone()).collect();
    committed.extend(resting.iter().map(|o| o.ticker.clone()));
    committed.extend(load_ordered_tickers(&cfg.ledger_path, cfg.live));

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

    let mut pricer = StationPricer::new(today, spread_scale);
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
        let candidate_lambda = gate_lambda(fit, r.best_bid, lambda);
        let d = decide_sell(
            today,
            r.target_date,
            r.best_bid,
            est,
            candidate_lambda,
            cfg.edge_threshold,
            cfg.fee_buffer,
        );
        let mut row = ledger_row(r, &d, est, candidate_lambda, cfg.live);
        if let Decision::Order {
            no_price,
            claimed: _,
            shrunk: _,
        } = d
        {
            // Defense in depth for `--lambda-floor`: the gate above already shrank by this same
            // band λ, but a large claimed edge can clear the gate at a band λ under the floor —
            // the floor is an absolute stop, not a scale, so it's applied per order too.
            let seg_lambda = fit.lambda_seg("kalshi", lambda_segment(1.0 - no_price));
            // The city gate runs first among the order-path guards: it disqualifies the CITY, not
            // this order, so a row logged under it says plainly which city was withheld and why.
            if let Some(reason) = city_gate(&fits.city, &r.city, cfg.lambda_floor, today) {
                row.decision = reason.into();
            } else if seg_lambda < cfg.lambda_floor {
                row.decision = "skip_segment_lambda_floor".into();
            } else if placed >= cfg.max_orders {
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
                    // Kalshi prices are whole cents in 1..=99; a bid of 0.995 must not round
                    // into an illegal 100.
                    let cents = ((no_price * 100.0).round() as i64).clamp(1, 99);
                    // Stable per (ticker, day): a crashed-and-rerun pilot reuses the same id and
                    // Kalshi rejects the duplicate instead of double-filling.
                    let coid = format!("pilot-{}-{}", r.market_id, today);
                    if cfg.live {
                        // Live always has a client: the no-auth path above errors out for --live.
                        let t = trader.as_ref().expect("live run without trade client");
                        let expires = expiration_ts(Utc::now(), cfg.order_ttl_mins);
                        match t
                            .buy_no_limit(&r.market_id, contracts, cents, &coid, expires)
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
    if cfg.live && placed > 0 {
        // A limit at the executable bid is a taker order when the book has size, so most fills
        // are immediate: give Kalshi a moment, then write down what already executed. Anything
        // still resting is asked about again next run (or expires under its TTL first).
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
        if let Some(t) = &trader {
            let reconciled = reconcile_orders(t, &cfg.ledger_path, today, cfg.live).await;
            if !reconciled.is_empty() {
                append_ledger(&cfg.ledger_path, &reconciled)?;
            }
        }
    }
    println!(
        "\n{} orders {} · ${:.2} committed exposure · full decision log appended to {}",
        placed,
        if cfg.live {
            "PLACED"
        } else {
            "would be placed (dry run)"
        },
        exposure.max(0.0), // .max(0.0) irons out "-0.00" (negative-zero display artifact)
        cfg.ledger_path.display()
    );
    Ok(())
}

/// The tradable decision for one market, or why not.
#[derive(Debug, PartialEq)]
enum Decision {
    /// SELL YES via BUY NO at `no_price` (= 1 − yes bid, the executable taker price).
    Order {
        no_price: f64,
        claimed: f64,
        shrunk: f64,
    },
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

/// The same observations keyed two ways. One pass, two `ShrinkageFit`s: identical venue folds,
/// different segment axes. Keeping them separate (rather than mixing band and city tags into one
/// map) leaves `rows_seg()` diagnostics and every venue fold counting each row exactly once.
#[derive(Default)]
struct PilotFits {
    /// Tagged by price band: sizing, the edge gate (`gate_lambda`) and the λ-floor breaker.
    band: ShrinkageFit,
    /// Tagged by city: the per-city trading gate (`city_gate`).
    city: ShrinkageFit,
}

/// Shrinkage fits for Kalshi from resolved lead ≥ 1 captures — the same fit and hygiene as the
/// dashboard's full-sample fit (book mid as reference price, lead ≤ 0 excluded). Returns the
/// whole fit so the caller can read both the venue fold (sizing) and the traded-segment λ (the
/// floor breaker); a missing file yields an empty fit, whose every lookup is 1.0 (NO shrink).
fn fit_shrinkage_from_captures(path: &PathBuf) -> PilotFits {
    let mut fits = PilotFits::default();
    let Ok(text) = std::fs::read_to_string(path) else {
        eprintln!(
            "warning: no captures at {} — λ falls back to 1.0 (NO shrink); pass --captures",
            path.display()
        );
        return fits;
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
            // Tagged by price band: the edge gate and the floor breaker read `lambda_seg`; venue-
            // level lookups fold across segments, which tagging never changes (proven by the
            // fold-identity test in shrinkage.rs).
            fits.band
                .observe_seg(&c.source, lambda_segment(px), est - px, outcome - px);
            // Same row, keyed by city and DATED, for `city_gate`: the trailing check needs to
            // know when each observation resolved, not just that it did.
            fits.city
                .observe_seg_dated(&c.source, &c.city, c.target_date, est - px, outcome - px);
        }
    }
    fits
}

/// λ the edge gate shrinks a candidate by: its own bid band's segment fit (a SELL fills at the
/// bid, so the bid picks the band — the dashboard's per-side convention), falling back
/// segment → venue fold → pooled inside `lambda_seg` on thin samples. An unusable bid returns
/// the venue fold: those rows die at `skip_no_bid` and the value is only logged.
fn gate_lambda(fit: &ShrinkageFit, best_bid: Option<f64>, venue_fold: f64) -> f64 {
    match best_bid {
        Some(b) if b > 0.0 && b < 1.0 => fit.lambda_seg("kalshi", lambda_segment(b)),
        _ => venue_fold,
    }
}

/// Whether the pilot may trade a (kalshi, city) AT ALL, independent of any one market's edge.
/// Two ways a city fails, both read off its own forward fit so there is no list to maintain:
///
///   * fewer than `MIN_N` resolved rows — the city's pricing constants (per-(city, lead) bias and
///     σ, `stations.rs`) were fitted Jan–Apr 2026 and neither seasonal refit could touch a city
///     with no captures, so nothing has yet checked them forward. `lambda_seg` cannot express
///     this: under `MIN_N` it answers from the venue fold, which reports a healthy λ for a city
///     that has never been measured. Hence `n_seg` first.
///   * a fitted λ under the floor — its disagreements are anti-signal. This is the band gate's
///     adaptive price floor applied on the city axis: a clamped-to-zero city can never clear the
///     edge threshold, so no hand-picked city list can go stale behind it.
///
/// Both matter because the Kalshi universe is not static: PR #34 (2026-08-19) fixed title-less
/// parsing and eight `KXHIGHT*` cities that had never produced a capture row entered the
/// tradeable universe at once, carrying exactly those unvalidated constants. Vegas priced four
/// straight days 2–4σ under the realized high (λ −0.84 on its first 24 rows) and took four of the
/// pilot's next six paper orders.
fn city_gate(fit: &ShrinkageFit, city: &str, floor: f64, today: NaiveDate) -> Option<&'static str> {
    // The rule itself lives in `backtesting::shrinkage` so the dashboard's computed A/B row and
    // this gate cannot drift; only the ledger's decision strings are the pilot's own.
    segment_veto(fit, "kalshi", city, floor, today).map(|v| match v {
        SegmentVeto::Unvalidated => "skip_city_unvalidated",
        SegmentVeto::BelowFloor => "skip_city_lambda_floor",
        SegmentVeto::TrailingBelowFloor => "skip_city_trailing_lambda_floor",
    })
}

/// Spread→σ fit observations from the captures file — the population hygiene lives in the shared
/// `backtesting::spread_sigma::spread_obs`, so this fit and the capture daemon's cannot drift.
/// Missing file ⇒ empty ⇒ `fit_spread_sigma_scale` returns None ⇒ constant σ tables.
fn spread_obs_from_captures(path: &PathBuf) -> Vec<SpreadObs> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str::<CaptureRow>(l).ok())
        .filter_map(|c| {
            spread_obs(
                SimulatedMarket {
                    date: c.target_date,
                    market_id: c.market_id.clone().unwrap_or_default(),
                    market_title: String::new(),
                    market_type: c.market_type.clone(),
                    threshold: c.threshold?,
                    threshold_upper: c.threshold_upper,
                    unit: c.unit.clone(),
                    market_price: c.entry_price,
                    actual_outcome: 0.0, // unused by the fit; the resolved outcome is passed below
                    city: c.city.clone(),
                },
                c.captured_at,
                c.best_bid,
                c.best_ask,
                c.model_estimate,
                c.outcome,
                c.forecast_high,
                c.forecast_sigma,
                c.ensemble_spread_ecmwf,
                c.ensemble_spread_gfs,
            )
        })
        .collect()
}

/// Why the pilot must not trade this run, if any breaker tripped. λ floor first: a too-thin
/// realized edge makes the loss question moot. `lambda` is the TRADED segment's λ (px ≥ 0.10,
/// falling back to the venue fold on thin samples) — see the module docs for why the fold alone
/// would eventually stand the pilot down on tail rows it never trades. The loss breaker only
/// arms once at least one order has actually settled — an empty ledger (or a fresh week) is not
/// a loss.
fn stand_down_reason(
    lambda: f64,
    lambda_floor: f64,
    week_pnl: f64,
    week_settled: usize,
    max_weekly_loss: f64,
) -> Option<String> {
    if lambda < lambda_floor {
        return Some(format!(
            "traded-segment λ {lambda:.3} is below the {lambda_floor:.2} floor — the model \
             realizes too little of its claimed edge to be worth trading"
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

/// Unix-seconds expiry for a live order placed at `now`: `ttl_mins` ≤ 0 means good-till-
/// cancelled (no expiry sent).
fn expiration_ts(now: DateTime<Utc>, ttl_mins: i64) -> Option<i64> {
    (ttl_mins > 0).then(|| (now + chrono::Duration::minutes(ttl_mins)).timestamp())
}

/// Every parseable ledger row, in file order. A missing file is an empty ledger.
fn load_ledger(path: &PathBuf) -> Vec<LedgerRow> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    text.lines()
        .filter_map(|l| serde_json::from_str::<LedgerRow>(l).ok())
        .collect()
}

/// Whether a row is a reconciliation verdict (`fill` / `unfilled`) rather than a decision.
fn is_reconciliation(r: &LedgerRow) -> bool {
    r.decision == "fill" || r.decision == "unfilled"
}

/// The reconciliation row per live order id, where one has been written.
fn reconciliation_by_order(rows: &[LedgerRow]) -> HashMap<String, &LedgerRow> {
    rows.iter()
        .filter(|r| is_reconciliation(r))
        .filter_map(|r| r.order_id.as_ref().map(|id| (id.clone(), r)))
        .collect()
}

/// What an order row is worth for exposure and PnL: the reconciled fill when Kalshi has told us
/// (0 contracts for an order that never filled), else the intended fill — conservative in the
/// direction the breaker wants, since an order was placed because the model liked it.
fn effective_fill(row: &LedgerRow, fills: &HashMap<String, &LedgerRow>) -> (i64, Option<f64>, f64) {
    match row.order_id.as_ref().and_then(|id| fills.get(id)) {
        Some(f) => (f.contracts, f.no_price, f.cost),
        None => (row.contracts, row.no_price, row.cost),
    }
}

/// Live `order` rows Kalshi has not yet been asked about: they carry an `order_id` and no
/// `fill`/`unfilled` row names it yet. Dry rows never have an order id and never qualify.
fn pending_reconciliation(rows: &[LedgerRow]) -> Vec<&LedgerRow> {
    let done: HashSet<&String> = rows
        .iter()
        .filter(|r| is_reconciliation(r))
        .filter_map(|r| r.order_id.as_ref())
        .collect();
    rows.iter()
        .filter(|r| r.decision == "order" && !r.dry_run && r.error.is_none())
        .filter(|r| r.order_id.as_ref().is_some_and(|id| !done.contains(id)))
        .collect()
}

/// Ask Kalshi what became of each unreconciled live order and write the answer down. An order
/// is final once Kalshi says so (`executed`, or `canceled` — which is also how expiry surfaces)
/// or once its target day has arrived: a remainder still resting then is exactly the lead-0
/// fill the strategy must not take, so a LIVE run cancels it first (a dry run only says it
/// would). Anything still resting inside its TTL is left alone and asked about next run. A
/// fetch or cancel failure is logged and retried next run rather than guessed at; after
/// `RECONCILE_GIVE_UP_DAYS` the pilot stops asking and the PnL readers keep their conservative
/// intended-fill fallback for that order.
async fn reconcile_orders(
    t: &KalshiTradeClient,
    ledger_path: &PathBuf,
    today: NaiveDate,
    live: bool,
) -> Vec<LedgerRow> {
    let rows = load_ledger(ledger_path);
    let mut out = Vec::new();
    for row in pending_reconciliation(&rows) {
        let Some(order_id) = row.order_id.clone() else {
            continue;
        };
        if (today - row.target_date).num_days() > RECONCILE_GIVE_UP_DAYS {
            continue;
        }
        let mut order = match t.order(&order_id).await {
            Ok(o) => o,
            Err(e) => {
                eprintln!(
                    "warning: could not fetch order {order_id} ({}): {e}",
                    row.ticker
                );
                continue;
            }
        };
        if !order.is_terminal() && row.target_date <= today {
            if !live {
                eprintln!(
                    "warning: {order_id} ({}) is still {} on its target day — a --live run would \
                     cancel it; leaving it alone in a dry run",
                    row.ticker, order.status
                );
                continue;
            }
            match t.cancel_order(&order_id).await {
                Ok(o) => {
                    println!(
                        "cancelled {order_id} ({}): still {} and its target day {} has arrived",
                        row.ticker, order.status, row.target_date
                    );
                    // Re-read rather than trust the cancel echo: the fill counts are what matter.
                    order = t.order(&order_id).await.unwrap_or(o);
                }
                Err(e) => {
                    eprintln!(
                        "warning: could not cancel {order_id} ({}): {e} — will retry next run",
                        row.ticker
                    );
                    continue;
                }
            }
        }
        if !order.is_terminal() {
            println!(
                "{order_id} ({}) still {} inside its TTL — asking again next run",
                row.ticker, order.status
            );
            continue;
        }
        let fills: Vec<KalshiFill> = match t.fills(&order_id).await {
            Ok(f) => f.into_iter().filter(|f| f.order_id == order_id).collect(),
            Err(e) => {
                eprintln!(
                    "warning: fills for {order_id} ({}): {e} — using the order's own counts",
                    row.ticker
                );
                Vec::new()
            }
        };
        let rec = reconciliation_row(row, &order, &fills, Utc::now());
        println!(
            "{:<28} {} — {} of {} contracts filled @ avg {} (intended {}), status {}",
            row.ticker,
            rec.decision,
            rec.contracts,
            row.contracts,
            fmt(rec.no_price),
            fmt(row.no_price),
            order.status
        );
        out.push(rec);
    }
    out
}

/// The `fill` / `unfilled` row for a terminal live order: contracts and cost from its fills
/// (each at the price it actually traded), or from the order's own placed − remaining counts at
/// its limit price when no fill detail came back (a fill never trades worse than the limit, so
/// that can only overstate cost).
fn reconciliation_row(
    order_row: &LedgerRow,
    order: &KalshiOrder,
    fills: &[KalshiFill],
    now: DateTime<Utc>,
) -> LedgerRow {
    let (filled, cost) = if fills.is_empty() {
        let n = order.filled().unwrap_or(0).max(0);
        let px = order
            .no_price_cents
            .map(|c| c as f64 / 100.0)
            .or(order_row.no_price)
            .unwrap_or(0.0);
        (n, n as f64 * px)
    } else {
        fills.iter().fold((0i64, 0.0f64), |(n, c), f| {
            (
                n + f.count,
                c + f.count as f64 * f.no_price_cents as f64 / 100.0,
            )
        })
    };
    let avg = (filled > 0).then(|| cost / filled as f64);
    LedgerRow {
        run_at: now,
        ticker: order_row.ticker.clone(),
        city: order_row.city.clone(),
        target_date: order_row.target_date,
        decision: if filled > 0 { "fill" } else { "unfilled" }.into(),
        dry_run: false,
        yes_bid: None,
        no_price: avg,
        model_estimate: None,
        lambda: 0.0,
        claimed_edge: None,
        shrunk_edge: None,
        fee_frac: avg.map(fee_frac),
        contracts: filled,
        cost,
        order_id: order_row.order_id.clone(),
        order_status: Some(order.status.clone()),
        error: None,
    }
}

/// Realized PnL (dollars) of the pilot's own orders whose markets settled in the trailing 7
/// days, joined against capture outcomes; returns (pnl, settled-order count). Mode-scoped: live
/// runs are judged by live orders and dry runs by dry-run orders, so the breaker logic rehearses
/// during the dry-run phase but paper losses can never trip a funded run (arming live after a
/// bad paper week is a human call, not this function's).
///
/// Live orders count what their `fill` row says filled (an `unfilled` order is no bet and does
/// not settle); until that row exists, and for every dry row, the intended fill is assumed. That
/// can only overstate a loss (the order was placed because the model liked it), so the breaker
/// errs toward standing down — acceptable for a safety rail, not for PnL reporting.
fn realized_week_pnl(
    ledger_path: &PathBuf,
    captures_path: &PathBuf,
    today: NaiveDate,
    live: bool,
) -> (f64, usize) {
    let rows = load_ledger(ledger_path);
    if rows.is_empty() {
        return (0.0, 0);
    }
    let outcomes = load_outcomes(captures_path);
    let fills = reconciliation_by_order(&rows);
    let week_ago = today - chrono::Duration::days(7);
    let (mut pnl, mut settled) = (0.0, 0usize);
    for row in &rows {
        if row.decision != "order" || row.error.is_some() || row.dry_run == live {
            continue;
        }
        if row.target_date < week_ago || row.target_date >= today {
            continue; // outside the trailing week, or not yet settled
        }
        let (contracts, no_price, _) = effective_fill(row, &fills);
        let (Some(no_price), Some(outcome)) = (no_price, outcomes.get(&row.ticker).copied()) else {
            continue;
        };
        if contracts <= 0 {
            continue; // never filled: nothing was at risk
        }
        // BUY NO pays $1/contract when the market resolves NO (outcome 0); fee paid either way.
        let per_contract = (1.0 - outcome) - no_price - fee_frac(no_price);
        pnl += contracts as f64 * per_contract;
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
/// same-mode rows only, mirroring `realized_week_pnl`, and at the reconciled cost where a fill
/// row exists — so the per-city cap holds across restarts, not just within one run.
fn open_city_exposure(
    ledger_path: &PathBuf,
    today: NaiveDate,
    live: bool,
) -> HashMap<(String, NaiveDate), f64> {
    let rows = load_ledger(ledger_path);
    let fills = reconciliation_by_order(&rows);
    let mut out: HashMap<(String, NaiveDate), f64> = HashMap::new();
    for row in &rows {
        if row.decision == "order"
            && row.error.is_none()
            && row.dry_run != live
            && row.target_date >= today
        {
            let (_, _, cost) = effective_fill(row, &fills);
            *out.entry((row.city.clone(), row.target_date))
                .or_insert(0.0) += cost;
        }
    }
    out
}

/// Dollars already tied up in held positions, conservatively estimated at worst-case $1/contract.
/// Kalshi doesn't return cost basis on this endpoint; over-counting exposure only makes the pilot
/// MORE cautious, never less.
fn position_cost_estimate(positions: &[polymarket_weather_predictor::api::KalshiPosition]) -> f64 {
    positions
        .iter()
        .map(|p| p.position.unsigned_abs() as f64)
        .sum()
}

/// Tickers an earlier run in the SAME mode decided to order. Dry rows are paper decisions: they
/// rehearse the live order, they don't stand in for it.
fn load_ordered_tickers(path: &PathBuf, live: bool) -> HashSet<String> {
    load_ledger(path)
        .into_iter()
        .filter(|r| r.decision == "order" && r.error.is_none() && r.dry_run != live)
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
        } => (
            "order".to_string(),
            Some(*no_price),
            Some(*claimed),
            Some(*shrunk),
        ),
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
    fn order_row(
        ticker: &str,
        city: &str,
        target: &str,
        no_price: f64,
        contracts: i64,
        dry: bool,
    ) -> LedgerRow {
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
    fn gate_lambda_reads_the_candidates_own_bid_band() {
        let mut fit = ShrinkageFit::default();
        // Thick, distinct fits in both bands: ≥ 10¢ realizes half its claims, sub-10¢ none.
        for _ in 0..ShrinkageFit::MIN_N {
            fit.observe_seg("kalshi", lambda_segment(0.40), 0.10, 0.05);
            fit.observe_seg("kalshi", lambda_segment(0.05), 0.10, -0.02);
        }
        let fold = fit.lambda("kalshi");
        // A 40¢ bid gates at the ≥ 10¢ band's λ, a 5¢ bid at the (clamped-to-zero) tail band's.
        assert!((gate_lambda(&fit, Some(0.40), fold) - 0.5).abs() < 1e-9);
        assert_eq!(gate_lambda(&fit, Some(0.05), fold), 0.0);
        // No usable bid → the venue fold, logged on a row that dies at skip_no_bid anyway.
        assert_eq!(gate_lambda(&fit, None, fold), fold);
        assert_eq!(gate_lambda(&fit, Some(0.0), fold), fold);
        // Thin band (no observations at all for a fresh fit) → falls back to the venue fold.
        let empty = ShrinkageFit::default();
        assert_eq!(gate_lambda(&empty, Some(0.40), 1.0), 1.0);
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
                order_row("DRY", "NYC", "2026-07-18", 0.60, 10, true),  // wrong mode for live
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
            [
                cap_line("WIN", 0.0),
                cap_line("LOSE", 1.0),
                cap_line("OLD", 0.0),
            ]
            .join("\n"),
        )
        .unwrap();

        let (pnl, settled) = realized_week_pnl(&ledger, &captures, today, true);
        assert_eq!(
            settled, 2,
            "only WIN and LOSE are live, in-window, and resolved"
        );
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
        assert!(
            (nyc - 15.0).abs() < 1e-9,
            "20×0.50 + 20×0.25 = $15, got {nyc}"
        );
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
        let fits = fit_shrinkage_from_captures(&path);
        let lambda = fits.band.lambda("kalshi");
        assert!(
            (lambda - 0.5).abs() < 1e-9,
            "outcome−price = 0.1 over est−price = 0.2 ⇒ λ = 0.5, lead-0 row excluded; got {lambda}"
        );
        // The rows sit at mid 0.5, so the traded (≥ 10¢) segment carries the same fit — the
        // breaker's lookup must see it, proving observations were tagged, not bare-venue.
        let traded = fits.band.lambda_seg("kalshi", lambda_segment(0.10));
        assert!(
            (traded - 0.5).abs() < 1e-9,
            "traded-segment λ, got {traded}"
        );
        // The SAME rows are keyed by city in the second fit — same count, same venue fold, so the
        // two views can never disagree about how much evidence a run is standing on.
        assert_eq!(fits.city.n_seg("kalshi", "NYC"), ShrinkageFit::MIN_N);
        assert!((fits.city.lambda("kalshi") - lambda).abs() < 1e-12);
        // Missing file → an empty fit: every lookup is 1.0 (no shrink), never a crash.
        let empty = fit_shrinkage_from_captures(&dir.join("nope.jsonl"));
        assert_eq!(empty.band.lambda("kalshi"), 1.0);
        assert_eq!(empty.band.lambda_seg("kalshi", lambda_segment(0.10)), 1.0);
        assert_eq!(empty.city.n_seg("kalshi", "NYC"), 0);
    }

    #[test]
    fn city_gate_withholds_unvalidated_and_anti_signal_cities() {
        let mut fit = ShrinkageFit::default();
        // An established, healthy city; an established anti-signal one; a young one.
        for _ in 0..ShrinkageFit::MIN_N {
            fit.observe_seg("kalshi", "Miami", 0.10, 0.075); // λ 0.75
            fit.observe_seg("kalshi", "Denver", 0.10, -0.02); // λ −0.2 ⇒ clamps to 0
        }
        for _ in 0..ShrinkageFit::MIN_N - 1 {
            fit.observe_seg("kalshi", "Vegas", 0.10, 0.09); // λ 0.9, but one row short
        }
        let floor = 0.2;
        assert_eq!(
            city_gate(&fit, "Miami", floor, d("2026-09-06")),
            None,
            "healthy city trades"
        );
        assert_eq!(
            city_gate(&fit, "Denver", floor, d("2026-09-06")),
            Some("skip_city_lambda_floor"),
            "a fitted anti-signal city is withheld on its own λ — no hand-picked list"
        );
        // One row short of a fit, and a λ that would sail through the floor if it were trusted:
        // coverage is checked FIRST precisely so a flattering thin slope can't open the gate.
        assert_eq!(fit.n_seg("kalshi", "Vegas"), ShrinkageFit::MIN_N - 1);
        assert!(fit.lambda_seg("kalshi", "Vegas") > floor);
        assert_eq!(
            city_gate(&fit, "Vegas", floor, d("2026-09-06")),
            Some("skip_city_unvalidated")
        );
        // A city with no rows at all — a series that just started listing — is withheld too.
        assert_eq!(
            city_gate(&fit, "Phoenix", floor, d("2026-09-06")),
            Some("skip_city_unvalidated")
        );
        // An empty fit withholds everything rather than trading on the 1.0 no-shrink fallback.
        let empty = ShrinkageFit::default();
        assert_eq!(
            city_gate(&empty, "Miami", floor, d("2026-09-06")),
            Some("skip_city_unvalidated")
        );
    }

    #[test]
    fn floor_breaker_watches_the_traded_segment_not_the_venue_fold() {
        // The 2026-07 failure mode in miniature: a big anti-signal sub-10¢ tail drags the venue
        // fold under the floor while the segment the pilot actually trades stays healthy.
        let mut fit = ShrinkageFit::default();
        for _ in 0..ShrinkageFit::MIN_N * 3 {
            fit.observe_seg("kalshi", lambda_segment(0.05), 0.1, -0.01); // tail slope −0.1
        }
        for _ in 0..ShrinkageFit::MIN_N {
            fit.observe_seg("kalshi", lambda_segment(0.50), 0.1, 0.05); // traded slope +0.5
        }
        let fold = fit.lambda("kalshi");
        let traded = fit.lambda_seg("kalshi", lambda_segment(0.10));
        assert!(
            fold < 0.2,
            "venue fold {fold} sits below the floor (old breaker trips)"
        );
        assert!(
            (traded - 0.5).abs() < 1e-9,
            "traded segment healthy, got {traded}"
        );
        // New breaker: healthy traded segment ⇒ no stand-down; the fold alone would have stood
        // the pilot down for rows it is structurally unable to trade.
        assert_eq!(stand_down_reason(traded, 0.2, 0.0, 0, 50.0), None);
        assert!(stand_down_reason(fold, 0.2, 0.0, 0, 50.0).is_some());
        // And the per-order guard blocks a candidate whose own band is the toxic tail: its
        // segment λ (clamped to 0) sits below the floor.
        assert!(fit.lambda_seg("kalshi", lambda_segment(0.05)) < 0.2);
    }
    #[test]
    fn ordered_tickers_dedupe_is_mode_scoped() {
        let dir = std::env::temp_dir().join("pilot_test_dedupe_mode");
        let _ = std::fs::create_dir_all(&dir);
        let ledger = dir.join("ledger.jsonl");
        write_jsonl(
            &ledger,
            &[
                order_row("PAPER", "NYC", "2026-09-08", 0.60, 25, true),
                order_row("REAL", "NYC", "2026-09-08", 0.60, 25, false),
            ],
        );
        // A live run is blocked only by live rows: the paper decision rehearsed the order, it
        // does not stand in for it. A dry run likewise ignores live rows.
        let live = load_ordered_tickers(&ledger, true);
        assert!(live.contains("REAL") && !live.contains("PAPER"));
        let dry = load_ordered_tickers(&ledger, false);
        assert!(dry.contains("PAPER") && !dry.contains("REAL"));
    }

    #[test]
    fn expiration_is_bounded_by_ttl() {
        let now = chrono::DateTime::parse_from_rfc3339("2026-09-07T15:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        assert_eq!(
            expiration_ts(now, 240),
            Some(now.timestamp() + 4 * 3600),
            "4 h after a 15:00 UTC run is 19:00 UTC — before any US city's local day rolls over"
        );
        assert_eq!(expiration_ts(now, 0), None, "0 ⇒ good-till-cancelled");
        assert_eq!(expiration_ts(now, -5), None);
    }

    fn kalshi_order(order_id: &str, status: &str, count: i64, remaining: i64) -> KalshiOrder {
        KalshiOrder {
            order_id: order_id.into(),
            ticker: "KXHIGHNY-26SEP08-B89.5".into(),
            status: status.into(),
            count: Some(count),
            remaining_count: Some(remaining),
            no_price_cents: Some(60),
        }
    }

    fn kalshi_fill(order_id: &str, count: i64, no_price_cents: i64) -> KalshiFill {
        KalshiFill {
            fill_id: format!("f-{count}-{no_price_cents}"),
            order_id: order_id.into(),
            ticker: "KXHIGHNY-26SEP08-B89.5".into(),
            count,
            no_price_cents,
            is_taker: true,
            created_time: String::new(),
        }
    }

    #[test]
    fn reconciliation_row_records_what_actually_filled() {
        let mut intended = order_row(
            "KXHIGHNY-26SEP08-B89.5",
            "NYC",
            "2026-09-08",
            0.60,
            25,
            false,
        );
        intended.order_id = Some("o1".into());
        let now = Utc::now();

        // Two fills, one better than the limit: 10 @ 58¢ + 5 @ 60¢ = $8.80 for 15 contracts.
        let fills = [kalshi_fill("o1", 10, 58), kalshi_fill("o1", 5, 60)];
        let rec = reconciliation_row(
            &intended,
            &kalshi_order("o1", "canceled", 25, 10),
            &fills,
            now,
        );
        assert_eq!(rec.decision, "fill");
        assert_eq!(rec.contracts, 15);
        assert!((rec.cost - 8.80).abs() < 1e-9, "cost {}", rec.cost);
        assert!((rec.no_price.unwrap() - 8.80 / 15.0).abs() < 1e-9);
        assert_eq!(rec.order_id.as_deref(), Some("o1"));
        assert_eq!(rec.order_status.as_deref(), Some("canceled"));
        assert!(!rec.dry_run);
        assert_eq!(
            (rec.city.as_str(), rec.target_date),
            ("NYC", d("2026-09-08"))
        );
        // Fee is re-derived from the average price actually paid.
        assert!((rec.fee_frac.unwrap() - fee_frac(8.80 / 15.0)).abs() < 1e-12);

        // No fill detail: the order's own placed − remaining at its limit price.
        let rec = reconciliation_row(&intended, &kalshi_order("o1", "executed", 25, 0), &[], now);
        assert_eq!((rec.decision.as_str(), rec.contracts), ("fill", 25));
        assert!((rec.cost - 15.0).abs() < 1e-9);
        assert!((rec.no_price.unwrap() - 0.60).abs() < 1e-9);

        // Expired untouched: an `unfilled` row with nothing at risk.
        let rec = reconciliation_row(&intended, &kalshi_order("o1", "canceled", 25, 25), &[], now);
        assert_eq!((rec.decision.as_str(), rec.contracts), ("unfilled", 0));
        assert_eq!(rec.cost, 0.0);
        assert_eq!(rec.no_price, None);
        assert_eq!(rec.fee_frac, None);
    }

    #[test]
    fn pending_reconciliation_skips_reconciled_and_dry_rows() {
        let mut done = order_row("A", "NYC", "2026-09-08", 0.60, 25, false);
        done.order_id = Some("o-done".into());
        let mut open = order_row("B", "NYC", "2026-09-08", 0.60, 25, false);
        open.order_id = Some("o-open".into());
        let mut failed = order_row("C", "NYC", "2026-09-08", 0.60, 25, false);
        failed.order_id = Some("o-err".into());
        failed.error = Some("rejected".into());
        let dry = order_row("D", "NYC", "2026-09-08", 0.60, 25, true);
        let verdict = reconciliation_row(
            &done,
            &kalshi_order("o-done", "executed", 25, 0),
            &[],
            Utc::now(),
        );
        let rows = vec![done, open, failed, dry, verdict];
        let pending: Vec<&str> = pending_reconciliation(&rows)
            .iter()
            .map(|r| r.ticker.as_str())
            .collect();
        assert_eq!(
            pending,
            vec!["B"],
            "only the live order with an id and no verdict yet is asked about"
        );
        let by_order = reconciliation_by_order(&rows);
        assert_eq!(by_order["o-done"].decision, "fill");
        assert!(!by_order.contains_key("o-open"));
    }

    #[test]
    fn weekly_pnl_and_city_exposure_prefer_fill_rows_for_live_orders() {
        let dir = std::env::temp_dir().join("pilot_test_fill_rows");
        let _ = std::fs::create_dir_all(&dir);
        let ledger = dir.join("ledger.jsonl");
        let captures = dir.join("captures.jsonl");
        let today = d("2026-09-09");
        let now = Utc::now();

        // Intended 25 @ 0.60; actually filled 10 @ 0.58 (won). Resolved NO.
        let mut partial = order_row("PART", "NYC", "2026-09-08", 0.60, 25, false);
        partial.order_id = Some("o-part".into());
        let partial_fill = reconciliation_row(
            &partial,
            &kalshi_order("o-part", "canceled", 25, 15),
            &[kalshi_fill("o-part", 10, 58)],
            now,
        );
        // Intended 25 @ 0.60; expired untouched. Resolved YES — would have LOST had it filled.
        let mut never = order_row("NEVER", "NYC", "2026-09-08", 0.60, 25, false);
        never.order_id = Some("o-never".into());
        let never_fill = reconciliation_row(
            &never,
            &kalshi_order("o-never", "canceled", 25, 25),
            &[],
            now,
        );
        // Not yet reconciled: the intended fill is assumed. Resolved YES (loss).
        let mut assumed = order_row("ASSUMED", "NYC", "2026-09-08", 0.60, 25, false);
        assumed.order_id = Some("o-assumed".into());
        // Still open tomorrow: exposure only. Filled 20 of 25.
        let mut open = order_row("OPEN", "NYC", "2026-09-10", 0.50, 30, false);
        open.order_id = Some("o-open".into());
        let open_fill = reconciliation_row(
            &open,
            &kalshi_order("o-open", "canceled", 30, 10),
            &[kalshi_fill("o-open", 20, 50)],
            now,
        );
        write_jsonl(
            &ledger,
            &[
                partial,
                partial_fill,
                never,
                never_fill,
                assumed,
                open,
                open_fill,
            ],
        );
        let cap_line = |id: &str, outcome: f64| {
            format!(
                r#"{{"captured_at":"2026-09-07","target_date":"2026-09-08","market_id":"{id}","entry_price":0.5,"model_estimate":0.4,"outcome":{outcome},"source":"kalshi"}}"#
            )
        };
        std::fs::write(
            &captures,
            [
                cap_line("PART", 0.0),
                cap_line("NEVER", 1.0),
                cap_line("ASSUMED", 1.0),
            ]
            .join("\n"),
        )
        .unwrap();

        let (pnl, settled) = realized_week_pnl(&ledger, &captures, today, true);
        assert_eq!(
            settled, 2,
            "PART (filled) and ASSUMED (assumed); NEVER was no bet"
        );
        let expect = 10.0 * (0.42 - fee_frac(0.58)) + 25.0 * (-0.60 - fee_frac(0.60));
        assert!((pnl - expect).abs() < 1e-9, "got {pnl}, want {expect}");

        // Open exposure counts the reconciled $10, not the intended $15.
        let exp = open_city_exposure(&ledger, today, true);
        let nyc = exp[&("NYC".to_string(), d("2026-09-10"))];
        assert!((nyc - 10.0).abs() < 1e-9, "got {nyc}");
    }
}
