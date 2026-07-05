//! Generate a self-contained HTML dashboard from real Polymarket weather markets: model
//! calibration vs realized outcomes, plus PnL on any markets that carry a real (traded) price.
//!
//! Weather is fetched once per city and cached to disk, so re-generating the dashboard is fast.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use chrono::{Duration, NaiveDate};

use polymarket_weather_predictor::backtesting::{
    evaluate_markets, evaluate_markets_with_forecast, kelly_fraction_of_capital, MarketEvaluation,
    RealMarketLoader,
};
use polymarket_weather_predictor::cities;
use polymarket_weather_predictor::config;
use polymarket_weather_predictor::data_pipeline::{MultiSourceAggregator, OpenMeteoFetcher};
use polymarket_weather_predictor::models::CalibrationAnalyzer;
use polymarket_weather_predictor::types::{SimulatedMarket, WeatherRecord};

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse(std::env::args().skip(1).collect())?;

    // The resolved-markets CSV drives the calibration/reliability sections; it's optional so the
    // forward trading-strategy readout can render from captures.jsonl alone (Polymarket purges price
    // history after resolution, so a resolved-markets CSV isn't always on hand).
    let evals = match &args.markets {
        Some(path) => {
            let markets = RealMarketLoader::load_from_path(path)
                .map_err(|e| format!("failed to load markets: {e}"))?;
            if markets.is_empty() {
                return Err("no markets in input".to_string());
            }
            println!("Loaded {} markets", markets.len());

            let weather =
                load_weather(&markets, args.lookback_days, &args.cache_dir, args.refresh)?;
            let covered: Vec<&String> = weather.keys().collect();
            println!(
                "Weather available for {} cities: {:?}",
                covered.len(),
                covered
            );

            if args.forecast {
                let forecasts = load_forecasts(&markets, &args.forecast_cache_dir, args.refresh)?;
                let (bias, computed_sigma) = calibrate_forecast(&forecasts, &weather);
                let sigma = args.forecast_sigma.unwrap_or(computed_sigma);
                println!(
                    "Forecast model: debias {bias:+.2}C, error sigma {sigma:.2}C{} (vs archive highs)",
                    if args.forecast_sigma.is_some() {
                        format!(" [override; computed {computed_sigma:.2}]")
                    } else {
                        String::new()
                    }
                );
                // Debias the forecast (subtract its mean error) and price each market from N(forecast, sigma).
                let debiased: HashMap<String, HashMap<NaiveDate, f64>> = forecasts
                    .into_iter()
                    .map(|(c, m)| (c, m.into_iter().map(|(d, h)| (d, h - bias)).collect()))
                    .collect();
                evaluate_markets_with_forecast(
                    &markets,
                    &weather,
                    args.lookback_days,
                    &debiased,
                    sigma,
                )
            } else {
                evaluate_markets(&markets, &weather, args.lookback_days)
            }
        }
        None => {
            println!("No --markets given; rendering forward-strategy readout from captures only.");
            Vec::new()
        }
    };
    let captures = load_captures(&args.captures);
    if !captures.is_empty() {
        println!(
            "Loaded {} forward captures from {}",
            captures.len(),
            args.captures.display()
        );
    }
    let sp = StrategyParams {
        edge_threshold: args.edge_threshold,
        kelly_fraction: args.kelly_fraction,
        max_position_pct: args.max_position_pct,
        bankroll: args.bankroll,
        max_edge: args.max_edge,
        no_sell_buckets: args.no_sell_buckets,
        trade_day_of: args.trade_day_of,
    };
    let html = render_dashboard(&evals, &captures, &sp);

    std::fs::write(&args.output, html)
        .map_err(|e| format!("failed writing {}: {e}", args.output.display()))?;
    println!("Wrote dashboard: {}", args.output.display());
    Ok(())
}

/// Fetch (and cache) aggregated weather per city covering every market's trailing window.
fn load_weather(
    markets: &[SimulatedMarket],
    lookback_days: i64,
    cache_dir: &PathBuf,
    refresh: bool,
) -> Result<HashMap<String, Vec<WeatherRecord>>, String> {
    std::fs::create_dir_all(cache_dir)
        .map_err(|e| format!("cannot create cache dir {}: {e}", cache_dir.display()))?;

    // date range per city
    let mut ranges: HashMap<&str, (NaiveDate, NaiveDate)> = HashMap::new();
    for m in markets {
        let e = ranges.entry(m.city.as_str()).or_insert((m.date, m.date));
        if m.date < e.0 {
            e.0 = m.date;
        }
        if m.date > e.1 {
            e.1 = m.date;
        }
    }

    let aggregator = MultiSourceAggregator::new();
    let mut out: HashMap<String, Vec<WeatherRecord>> = HashMap::new();

    for (city, (min_d, max_d)) in ranges {
        let cache_file = cache_dir.join(format!("{}.json", city.replace(['/', ' '], "_")));
        if !refresh && cache_file.exists() {
            if let Ok(bytes) = std::fs::read(&cache_file) {
                if let Ok(rows) = serde_json::from_slice::<Vec<WeatherRecord>>(&bytes) {
                    if !rows.is_empty() {
                        out.insert(city.to_string(), rows);
                        println!("  {city}: {} cached records", out[city].len());
                        continue;
                    }
                }
            }
        }

        let start = min_d - Duration::days(lookback_days + 5);
        print!("  {city}: fetching {start}..{max_d} ... ");
        let rows = aggregator.aggregate(city, start, max_d);
        println!("{} records", rows.len());
        if !rows.is_empty() {
            let _ = std::fs::write(&cache_file, serde_json::to_vec(&rows).unwrap_or_default());
            out.insert(city.to_string(), rows);
        }
    }

    Ok(out)
}

/// Fetch (and cache) archived daily-high FORECASTS per city, keyed by date (degC). Uses Open-Meteo's
/// historical-forecast API (what was predicted), distinct from the reanalysis archive (observed).
fn load_forecasts(
    markets: &[SimulatedMarket],
    cache_dir: &PathBuf,
    refresh: bool,
) -> Result<HashMap<String, HashMap<NaiveDate, f64>>, String> {
    std::fs::create_dir_all(cache_dir).map_err(|e| {
        format!(
            "cannot create forecast cache dir {}: {e}",
            cache_dir.display()
        )
    })?;

    let mut ranges: HashMap<&str, (NaiveDate, NaiveDate)> = HashMap::new();
    for m in markets {
        let e = ranges.entry(m.city.as_str()).or_insert((m.date, m.date));
        e.0 = e.0.min(m.date);
        e.1 = e.1.max(m.date);
    }

    let fetcher = OpenMeteoFetcher::new();
    let mut out: HashMap<String, HashMap<NaiveDate, f64>> = HashMap::new();

    for (city, (min_d, max_d)) in ranges {
        let cache_file = cache_dir.join(format!("{}.json", city.replace(['/', ' '], "_")));
        if !refresh && cache_file.exists() {
            if let Ok(bytes) = std::fs::read(&cache_file) {
                if let Ok(pairs) = serde_json::from_slice::<Vec<(NaiveDate, f64)>>(&bytes) {
                    if !pairs.is_empty() {
                        let map: HashMap<NaiveDate, f64> = pairs.into_iter().collect();
                        println!("  {city}: {} cached forecasts", map.len());
                        out.insert(city.to_string(), map);
                        continue;
                    }
                }
            }
        }

        let Some((lat, lon)) = cities::coords(city) else {
            println!("  {city}: no coords, skipping forecast");
            continue;
        };
        print!("  {city}: fetching forecasts {min_d}..{max_d} ... ");
        let pairs = fetcher.fetch_forecast_max(lat, lon, min_d, max_d);
        println!("{} forecasts", pairs.len());
        if !pairs.is_empty() {
            let _ = std::fs::write(&cache_file, serde_json::to_vec(&pairs).unwrap_or_default());
            out.insert(city.to_string(), pairs.into_iter().collect());
        }
    }

    Ok(out)
}

/// Forecast error vs the cached archive highs: (mean bias, residual sigma) in degC. The sigma becomes
/// the predictive spread; the bias is removed before pricing. ponytail: in-sample calibration over the
/// whole window — fine for measurement; make it rolling/causal before trusting live PnL.
fn calibrate_forecast(
    forecasts: &HashMap<String, HashMap<NaiveDate, f64>>,
    weather: &HashMap<String, Vec<WeatherRecord>>,
) -> (f64, f64) {
    let mut residuals = Vec::new();
    for (city, fmap) in forecasts {
        let Some(rows) = weather.get(city) else {
            continue;
        };
        let archive: HashMap<NaiveDate, f64> = rows
            .iter()
            .filter_map(|r| r.temperature_max.map(|h| (r.date, h)))
            .collect();
        for (date, f) in fmap {
            if let Some(a) = archive.get(date) {
                residuals.push(f - a);
            }
        }
    }
    if residuals.len() < 2 {
        return (0.0, 2.5); // no overlap to calibrate; a sane default short-lead sigma
    }
    let bias = residuals.iter().sum::<f64>() / residuals.len() as f64;
    let var =
        residuals.iter().map(|r| (r - bias).powi(2)).sum::<f64>() / (residuals.len() - 1) as f64;
    (bias, var.sqrt().max(0.5))
}

// ── metrics ────────────────────────────────────────────────────────────────

struct Calib {
    n: usize,
    brier: f64,
    ece: f64,
    accuracy: f64,
    base_rate: f64,
    avg_pred: f64,
}

fn calibration(evals: &[&MarketEvaluation]) -> Option<Calib> {
    let preds: Vec<f64> = evals.iter().filter_map(|e| e.model_estimate).collect();
    let outs: Vec<f64> = evals
        .iter()
        .filter(|e| e.model_estimate.is_some())
        .map(|e| e.actual_outcome)
        .collect();
    if preds.is_empty() {
        return None;
    }
    let n = preds.len();
    let correct = preds
        .iter()
        .zip(&outs)
        .filter(|(p, o)| (**p > 0.5) == (**o > 0.5))
        .count();
    Some(Calib {
        n,
        brier: CalibrationAnalyzer::brier_score(&preds, &outs),
        ece: CalibrationAnalyzer::expected_calibration_error(&preds, &outs, 10),
        accuracy: correct as f64 / n as f64,
        base_rate: outs.iter().sum::<f64>() / n as f64,
        avg_pred: preds.iter().sum::<f64>() / n as f64,
    })
}

// ── forward strategy: edge threshold → fractional Kelly → compounding PnL ─────

/// Strategy knobs. Defaults mirror `backtest_params()` / `config` so the forward readout applies the
/// same rules the backtest was tuned on.
#[derive(Clone, Copy)]
struct StrategyParams {
    edge_threshold: f64,
    kelly_fraction: f64,
    max_position_pct: f64,
    bankroll: f64,
    /// Skip trades whose |model − price| exceeds this. Large edges turned out to be the model's own
    /// miscalibration (the price forecasts buckets ~3× better), not alpha. None = no cap.
    max_edge: Option<f64>,
    /// Suppress SELL trades on `temp_bucket` markets — the model systematically under-estimates narrow
    /// bucket probabilities, so its bucket-SELL signals are structurally its own errors.
    no_sell_buckets: bool,
    /// Take trades captured ON the target day (lead 0). Off by default: at day-of the market prices
    /// intraday information the daemon doesn't have — measured out-of-sample, the market's bucket
    /// Brier beat even a leak-optimistic reconstruction of our day-of estimate (0.104 vs 0.127), and
    /// day-of trades were the entire −8.1% loss. The model's edge is at lead ≥ 1 (model 0.034 vs
    /// market 0.099) and post-day scraps. `--trade-day-of` re-enables for forward A/B.
    trade_day_of: bool,
}

/// One booked trade: a settled capture that cleared the edge filter.
struct Trade {
    date: NaiveDate,
    city: String,
    market_type: String,
    source: String,
    lead: i64,
    label: String,
    side: &'static str, // BUY YES / SELL (buy NO)
    price: f64,
    est: f64,
    stake: f64,
    pnl: f64,
    equity: f64, // bankroll after booking this trade
}

/// Decide a trade from one capture's (model estimate, entry price): the side and the fractional-Kelly
/// stake as a fraction of capital. `None` when the price isn't tradable, the edge is too thin, or Kelly
/// says don't bet. Shared by settled PnL and the open-book preview so both size positions identically.
fn decide(
    est: f64,
    price: f64,
    market_type: &str,
    lead: i64,
    p: &StrategyParams,
) -> Option<(&'static str, f64)> {
    if price <= 0.0 || price >= 1.0 {
        return None; // fabricated / resolved price, not a real entry
    }
    if lead == 0 && !p.trade_day_of {
        return None; // day-of: the market's intraday information set beats ours (see StrategyParams)
    }
    let edge = est - price;
    if edge.abs() < p.edge_threshold {
        return None;
    }
    if let Some(cap) = p.max_edge {
        if edge.abs() > cap {
            return None; // oversized edge = model miscalibration, not opportunity
        }
    }
    let side = if edge > 0.0 { "BUY" } else { "SELL" };
    if p.no_sell_buckets && side == "SELL" && market_type == "temp_bucket" {
        return None;
    }
    let frac = kelly_fraction_of_capital(est, price, p.kelly_fraction);
    if frac <= 0.0 {
        return None;
    }
    Some((side, frac))
}

/// Cap the Kelly fraction to `max_position_pct` of capital and return the dollar stake.
fn stake_dollars(frac: f64, capital: f64, p: &StrategyParams) -> f64 {
    (frac * capital).min(capital * p.max_position_pct)
}

/// Walk settled captures in resolution order, edge-filter, fractional-Kelly size on the compounding
/// bankroll, and book PnL using the same per-share convention as the backtest engine
/// (stake × (outcome − price) for a BUY, negated for a SELL).
fn run_strategy(captures: &[Capture], p: &StrategyParams) -> Vec<Trade> {
    let mut settled: Vec<&Capture> = captures
        .iter()
        .filter(|c| c.outcome.is_some() && c.model_estimate.is_some())
        .collect();
    settled.sort_by(|a, b| a.target_date.cmp(&b.target_date).then(a.city.cmp(&b.city)));

    let mut equity = p.bankroll;
    let mut trades = Vec::new();
    for c in settled {
        let est = c.model_estimate.unwrap();
        let outcome = c.outcome.unwrap();
        let Some((side, frac)) = decide(est, c.entry_price, &c.market_type, c.lead(), p) else {
            continue;
        };
        let stake = stake_dollars(frac, equity, p);
        if stake < 1.0 {
            continue;
        }
        let pnl = if side == "BUY" {
            stake * (outcome - c.entry_price)
        } else {
            stake * (c.entry_price - outcome)
        };
        equity += pnl;
        trades.push(Trade {
            date: c.target_date,
            city: c.city.clone(),
            market_type: c.market_type.clone(),
            source: c.source.clone(),
            lead: c.lead(),
            label: label(
                &c.market_type,
                c.threshold,
                c.threshold_upper,
                c.unit.as_deref(),
            ),
            side,
            price: c.entry_price,
            est,
            stake,
            pnl,
            equity,
        });
    }
    trades
}

/// Rolled-up performance over a slice of trades.
struct Agg {
    n: usize,
    pnl: f64,
    staked: f64,
    wins: usize,
}

fn agg(trades: &[&Trade]) -> Agg {
    let mut a = Agg {
        n: 0,
        pnl: 0.0,
        staked: 0.0,
        wins: 0,
    };
    for t in trades {
        a.n += 1;
        a.pnl += t.pnl;
        a.staked += t.stake;
        if t.pnl > 0.0 {
            a.wins += 1;
        }
    }
    a
}

impl Agg {
    fn roi(&self) -> f64 {
        if self.staked > 0.0 {
            self.pnl / self.staked
        } else {
            0.0
        }
    }
    fn hit_rate(&self) -> f64 {
        if self.n > 0 {
            self.wins as f64 / self.n as f64
        } else {
            0.0
        }
    }
}

// ── forward captures (from the capture_prices daemon) ───────────────────────

#[derive(serde::Deserialize)]
struct Capture {
    captured_at: NaiveDate,
    target_date: NaiveDate,
    market_type: String,
    threshold: f64,
    threshold_upper: Option<f64>,
    unit: Option<String>,
    city: String,
    entry_price: f64,
    model_estimate: Option<f64>,
    outcome: Option<f64>,
    #[serde(default = "default_source")]
    source: String,
}

fn default_source() -> String {
    "polymarket".to_string()
}

impl Capture {
    /// Trading lead in days at capture: negative = captured after the target day (post), 0 =
    /// day-of, ≥1 = genuine forecast lead.
    fn lead(&self) -> i64 {
        (self.target_date - self.captured_at).num_days()
    }
}

fn load_captures(path: &PathBuf) -> Vec<Capture> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str::<Capture>(l).ok())
        .collect()
}

// ── HTML ─────────────────────────────────────────────────────────────────────

fn render_dashboard(
    evals: &[MarketEvaluation],
    captures: &[Capture],
    sp: &StrategyParams,
) -> String {
    let all: Vec<&MarketEvaluation> = evals.iter().collect();
    let total = all.len();
    let evaluated = all.iter().filter(|e| e.model_estimate.is_some()).count();
    let overall = calibration(&all);

    // per-city
    let mut by_city: BTreeMap<&str, Vec<&MarketEvaluation>> = BTreeMap::new();
    for e in &all {
        by_city.entry(e.city.as_str()).or_default().push(e);
    }

    let mut s = String::new();
    s.push_str(HEAD);

    // summary cards
    s.push_str("<div class=\"cards\">");
    s.push_str(&card(
        "Markets",
        &total.to_string(),
        "real resolved Polymarket buckets",
    ));
    s.push_str(&card(
        "Evaluated",
        &format!("{evaluated}"),
        &format!("{}% have model estimates", pct(evaluated, total)),
    ));
    s.push_str(&card("Cities", &by_city.len().to_string(), "with markets"));
    if let Some(c) = &overall {
        // Baseline = always predict the base rate. Brier skill score > 0 means the model beats it.
        let baseline = c.base_rate * (1.0 - c.base_rate);
        let bss = if baseline > 0.0 {
            1.0 - c.brier / baseline
        } else {
            0.0
        };
        s.push_str(&card(
            "Brier",
            &format!("{:.3}", c.brier),
            &format!("baseline {baseline:.3} (predict base rate)"),
        ));
        s.push_str(&card(
            "ECE",
            &format!("{:.3}", c.ece),
            "calibration error, lower better",
        ));
        s.push_str(&card(
            "Brier skill",
            &format!("{:+.2}", bss),
            if bss > 0.0 {
                "beats the base-rate baseline"
            } else {
                "below baseline — buckets need seasonality"
            },
        ));
        s.push_str(&card(
            "Accuracy",
            &format!("{:.0}%", c.accuracy * 100.0),
            &format!(
                "base rate {:.0}% (so accuracy flatters)",
                c.base_rate * 100.0
            ),
        ));
    }
    s.push_str("</div>");

    s.push_str(&render_strategy(captures, sp));

    // reliability diagram
    if let Some(c) = &overall {
        let preds: Vec<f64> = all.iter().filter_map(|e| e.model_estimate).collect();
        let outs: Vec<f64> = all
            .iter()
            .filter(|e| e.model_estimate.is_some())
            .map(|e| e.actual_outcome)
            .collect();
        s.push_str(
            "<div class=\"panel\"><h2>Reliability (model probability vs realized frequency)</h2>",
        );
        s.push_str("<div class=\"reliab\">");
        s.push_str(&reliability_svg(&preds, &outs));
        s.push_str(&format!(
            "<div class=\"legend\"><p>Points on the diagonal = perfectly calibrated. Above = model under-confident, below = over-confident.</p><p class=\"muted\">n={} · avg model P {:.2} · realized {:.2}</p></div>",
            c.n, c.avg_pred, c.base_rate
        ));
        s.push_str("</div></div>");
    }

    // per-city table
    s.push_str("<div class=\"panel\"><h2>By city</h2><table><thead><tr><th>City</th><th>Markets</th><th>Evaluated</th><th>Brier</th><th>Accuracy</th></tr></thead><tbody>");
    for (city, es) in &by_city {
        let c = calibration(es);
        let evald = es.iter().filter(|e| e.model_estimate.is_some()).count();
        s.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
            esc(city),
            es.len(),
            evald,
            c.as_ref()
                .map(|c| format!("{:.3}", c.brier))
                .unwrap_or_else(|| "–".into()),
            c.as_ref()
                .map(|c| format!("{:.0}%", c.accuracy * 100.0))
                .unwrap_or_else(|| "–".into()),
        ));
    }
    s.push_str("</tbody></table></div>");

    // markets table
    s.push_str("<div class=\"panel\"><h2>Markets</h2><table><thead><tr><th>Date</th><th>City</th><th>Bucket</th><th>Model P</th><th>Price</th><th>Outcome</th></tr></thead><tbody>");
    let mut sorted = all.clone();
    sorted.sort_by(|a, b| b.date.cmp(&a.date).then(a.city.cmp(&b.city)));
    for e in sorted.iter().take(600) {
        let est = e
            .model_estimate
            .map(|p| format!("{:.3}", p))
            .unwrap_or_else(|| "<span class=\"muted\">n/a</span>".into());
        let price = if e.market_price <= 0.0 || e.market_price >= 1.0 {
            "<span class=\"muted\">–</span>".to_string()
        } else {
            format!("{:.3}", e.market_price)
        };
        let (oc, ocls) = if e.actual_outcome > 0.5 {
            ("YES", "yes")
        } else {
            ("NO", "no")
        };
        // mark correctness of a >0.5 call
        let hit = e
            .model_estimate
            .map(|p| (p > 0.5) == (e.actual_outcome > 0.5))
            .unwrap_or(false);
        let estcls = if e.model_estimate.is_some() && hit {
            "hit"
        } else if e.model_estimate.is_some() {
            "miss"
        } else {
            ""
        };
        s.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td class=\"{}\">{}</td><td>{}</td><td class=\"{}\">{}</td></tr>",
            e.date,
            esc(&e.city),
            esc(&label(&e.market_type, e.threshold, e.threshold_upper, e.unit.as_deref())),
            estcls,
            est,
            price,
            ocls,
            oc,
        ));
    }
    s.push_str("</tbody></table>");
    if total > 600 {
        s.push_str(&format!(
            "<p class=\"muted\">Showing 600 of {total} markets.</p>"
        ));
    }
    s.push_str("</div>");

    s.push_str(FOOT);
    s
}

/// The forward trading-strategy panel: apply edge threshold → fractional Kelly → compounding PnL over
/// resolved captures, then break the result down by city and market type. When nothing has resolved
/// yet, it still shows the open book — the positions (and Kelly stakes) the strategy would take today.
fn render_strategy(captures: &[Capture], sp: &StrategyParams) -> String {
    let mut s = String::new();
    s.push_str("<div class=\"panel\"><h2>Forward trading strategy (real entry prices)</h2>");
    let filters = {
        let mut f = Vec::new();
        if let Some(cap) = sp.max_edge {
            f.push(format!("skip |edge| > {:.0}%", cap * 100.0));
        }
        if sp.no_sell_buckets {
            f.push("no SELL on temp buckets".to_string());
        }
        if !sp.trade_day_of {
            f.push("skip day-of (lead 0) trades".to_string());
        }
        if f.is_empty() {
            "no extra filters".to_string()
        } else {
            f.join(" · ")
        }
    };
    s.push_str(&format!(
        "<p class=\"muted\">Rule: trade when |model − price| ≥ {:.0}%, size at {:.2}× Kelly (cap {:.0}% of bankroll), compound a ${:.0} bankroll. Filters: {}. Override with <code>--edge-threshold</code> / <code>--kelly-fraction</code> / <code>--bankroll</code> / <code>--max-edge</code> / <code>--no-sell-buckets</code>.</p>",
        sp.edge_threshold * 100.0,
        sp.kelly_fraction,
        sp.max_position_pct * 100.0,
        sp.bankroll,
        esc(&filters),
    ));
    s.push_str("<p class=\"muted\" style=\"font-size:11px\">Day-of (lead 0) trades are skipped by default: out-of-sample the market's intraday information beats the daemon's morning snapshot there, while at lead ≥ 1 the model's bucket Brier beats the market ~3×. <code>--trade-day-of</code> re-enables.</p>");

    if captures.is_empty() {
        s.push_str("<p class=\"muted\">No captures yet. Polymarket purges price history after resolution, so real entry prices are only available live. Run <code>capture_prices</code> daily (cron / schedule) to snapshot active markets; PnL settles as they resolve.</p></div>");
        return s;
    }

    let trades = run_strategy(captures, sp);
    let trade_refs: Vec<&Trade> = trades.iter().collect();
    let a = agg(&trade_refs);
    let captured = captures.len();
    let resolved = captures.iter().filter(|c| c.outcome.is_some()).count();
    let final_bankroll = trades.last().map(|t| t.equity).unwrap_or(sp.bankroll);
    let growth = 100.0 * (final_bankroll - sp.bankroll) / sp.bankroll;

    // ── headline cards ───────────────────────────────────────────────────────
    s.push_str(&format!(
        "<div class=\"cards\">{}{}{}{}{}</div>",
        card(
            "Captured",
            &captured.to_string(),
            &format!("{resolved} resolved · {} pending", captured - resolved),
        ),
        card(
            "Settled trades",
            &a.n.to_string(),
            "cleared the edge filter"
        ),
        card(
            "Cumulative PnL",
            &usd_signed(a.pnl),
            &format!("{:+.1}% on capital risked", a.roi() * 100.0),
        ),
        card(
            "Win rate",
            &format!("{:.0}%", a.hit_rate() * 100.0),
            &format!("{}/{} trades green", a.wins, a.n),
        ),
        card(
            "Bankroll",
            &format!("${:.0}", final_bankroll),
            &format!("{growth:+.1}% from ${:.0}", sp.bankroll),
        ),
    ));

    // ── forward A/B tracker: what each candidate filter would have done on the SAME settled set.
    // Always computed (independent of the active flags) so the daily baseline dashboard reveals the
    // divergence as pending markets resolve — the out-of-sample test the in-sample sweep can't give.
    if resolved > 0 {
        let variants = [
            (
                "Baseline (no filters, incl. day-of)",
                StrategyParams {
                    max_edge: None,
                    no_sell_buckets: false,
                    trade_day_of: true,
                    ..*sp
                },
            ),
            (
                "Skip day-of (default)",
                StrategyParams {
                    max_edge: None,
                    no_sell_buckets: false,
                    trade_day_of: false,
                    ..*sp
                },
            ),
            (
                "Edge cap 30%",
                StrategyParams {
                    max_edge: Some(0.30),
                    no_sell_buckets: false,
                    trade_day_of: true,
                    ..*sp
                },
            ),
            (
                "No SELL on buckets",
                StrategyParams {
                    max_edge: None,
                    no_sell_buckets: true,
                    trade_day_of: true,
                    ..*sp
                },
            ),
        ];
        s.push_str("<h3 class=\"sub-h\">Filter A/B — forward tracker</h3>");
        s.push_str("<table><thead><tr><th>Config</th><th>Trades</th><th>PnL</th><th>ROI</th><th>Win</th></tr></thead><tbody>");
        for (name, v) in &variants {
            let refs: Vec<Trade> = run_strategy(captures, v);
            let rr: Vec<&Trade> = refs.iter().collect();
            let av = agg(&rr);
            let cls = if av.pnl >= 0.0 { "yes" } else { "miss" };
            s.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td class=\"{}\">{}</td><td>{:+.1}%</td><td>{:.0}%</td></tr>",
                esc(name), av.n, cls, usd_signed(av.pnl), av.roi() * 100.0, av.hit_rate() * 100.0
            ));
        }
        s.push_str("</tbody></table><p class=\"muted\" style=\"font-size:11px\">In-sample so far — watch these diverge as pending markets resolve to validate the filter out-of-sample.</p>");
    }

    if a.n == 0 {
        s.push_str("<p class=\"muted\">No captures have resolved into settled trades yet — the PnL answer arrives as outcomes fill in. Below is the open book: what the strategy would trade right now.</p>");
    } else {
        // equity curve
        s.push_str("<h3 class=\"sub-h\">Equity curve</h3>");
        s.push_str(&equity_svg(sp.bankroll, &trades));

        // per-city / per-market-type / per-venue breakdowns
        s.push_str("<div class=\"reliab\" style=\"align-items:flex-start\">");
        s.push_str(&breakdown_table(
            "By venue",
            "Venue",
            group_trades(&trade_refs, |t| t.source.clone()),
        ));
        s.push_str(&breakdown_table(
            "By city",
            "City",
            group_trades(&trade_refs, |t| t.city.clone()),
        ));
        s.push_str(&breakdown_table(
            "By market type",
            "Type",
            group_trades(&trade_refs, |t| t.market_type.clone()),
        ));
        s.push_str(&breakdown_table(
            "By trading lead",
            "Lead",
            group_trades(&trade_refs, |t| match t.lead {
                l if l < 0 => "post (after day)".to_string(),
                0 => "day-of".to_string(),
                1 => "lead 1".to_string(),
                _ => "lead 2+".to_string(),
            }),
        ));
        s.push_str("</div>");

        // trade log (most recent first) — the auditable per-trade PnL
        s.push_str("<h3 class=\"sub-h\">Settled trades</h3>");
        s.push_str("<table><thead><tr><th>Resolved</th><th>Venue</th><th>City</th><th>Bucket</th><th>Side</th><th>Price</th><th>Model</th><th>Stake</th><th>PnL</th><th>Bankroll</th></tr></thead><tbody>");
        for t in trades.iter().rev().take(150) {
            let scls = if t.side == "BUY" { "yes" } else { "miss" };
            let pcls = if t.pnl >= 0.0 { "yes" } else { "miss" };
            s.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td class=\"{}\">{}</td><td>{:.3}</td><td>{:.3}</td><td>${:.0}</td><td class=\"{}\">{}</td><td>${:.0}</td></tr>",
                t.date, esc(&t.source), esc(&t.city), esc(&t.label), scls, t.side, t.price, t.est, t.stake, pcls, usd_signed(t.pnl), t.equity,
            ));
        }
        s.push_str("</tbody></table>");
    }

    // ── open book: positions the strategy would take today ───────────────────
    let mut open: Vec<(&Capture, &'static str, f64)> = captures
        .iter()
        .filter(|c| c.outcome.is_none())
        .filter_map(|c| {
            let est = c.model_estimate?;
            let (side, frac) = decide(est, c.entry_price, &c.market_type, c.lead(), sp)?;
            Some((c, side, stake_dollars(frac, sp.bankroll, sp)))
        })
        .collect();
    open.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // Worst-case downside: a BUY of `stake` shares at price p loses stake·p if it resolves NO; a SELL
    // loses stake·(1−p) if it resolves YES. Each stake is sized against the full bankroll independently,
    // so if this sum exceeds the bankroll the naive per-bet Kelly is over-committing across simultaneous
    // signals — a real read on the params, not double-counting. ponytail: no portfolio Kelly here.
    let downside: f64 = open
        .iter()
        .map(|(c, side, st)| {
            if *side == "BUY" {
                st * c.entry_price
            } else {
                st * (1.0 - c.entry_price)
            }
        })
        .sum();
    s.push_str(&format!(
        "<h3 class=\"sub-h\">Open book — {} positions the strategy would take now (${:.0} max downside if all lose)</h3>",
        open.len(),
        downside,
    ));
    if open.is_empty() {
        s.push_str("<p class=\"muted\">No open captures clear the edge threshold right now.</p>");
    } else {
        s.push_str("<table><thead><tr><th>Resolves</th><th>Venue</th><th>City</th><th>Bucket</th><th>Price</th><th>Model</th><th>Edge</th><th>Side</th><th>Stake</th></tr></thead><tbody>");
        for (c, side, stake) in open.iter().take(25) {
            let est = c.model_estimate.unwrap();
            let edge = est - c.entry_price;
            let scls = if *side == "BUY" { "yes" } else { "miss" };
            s.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{:.3}</td><td>{:.3}</td><td>{:+.3}</td><td class=\"{}\">{}</td><td>${:.0}</td></tr>",
                c.target_date, esc(&c.source), esc(&c.city), esc(&label(&c.market_type, c.threshold, c.threshold_upper, c.unit.as_deref())), c.entry_price, est, edge, scls, side, stake
            ));
        }
        s.push_str("</tbody></table>");
        if open.len() > 25 {
            s.push_str(&format!(
                "<p class=\"muted\">Showing 25 of {} open positions.</p>",
                open.len()
            ));
        }
    }

    s.push_str("</div>");
    s
}

/// Group trades by a key and roll up each group, returning rows sorted by PnL descending.
fn group_trades(trades: &[&Trade], key: impl Fn(&Trade) -> String) -> Vec<(String, Agg)> {
    let mut groups: BTreeMap<String, Vec<&Trade>> = BTreeMap::new();
    for t in trades {
        groups.entry(key(t)).or_default().push(t);
    }
    let mut rows: Vec<(String, Agg)> = groups.into_iter().map(|(k, v)| (k, agg(&v))).collect();
    rows.sort_by(|a, b| {
        b.1.pnl
            .partial_cmp(&a.1.pnl)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    rows
}

fn breakdown_table(title: &str, col: &str, rows: Vec<(String, Agg)>) -> String {
    let mut s = format!(
        "<div style=\"flex:1;min-width:280px\"><h4 style=\"font-size:12px;color:#8a93a6;margin:0 0 8px\">{}</h4><table><thead><tr><th>{}</th><th>Trades</th><th>PnL</th><th>ROI</th><th>Hit</th></tr></thead><tbody>",
        esc(title),
        esc(col),
    );
    for (k, a) in &rows {
        let pcls = if a.pnl >= 0.0 { "yes" } else { "miss" };
        s.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td class=\"{}\">{}</td><td>{:+.1}%</td><td>{:.0}%</td></tr>",
            esc(k),
            a.n,
            pcls,
            usd_signed(a.pnl),
            a.roi() * 100.0,
            a.hit_rate() * 100.0,
        ));
    }
    s.push_str("</tbody></table></div>");
    s
}

/// Compounding-bankroll equity curve as an inline SVG (starting bankroll → after each settled trade).
fn equity_svg(bankroll: f64, trades: &[Trade]) -> String {
    let (w, h, pad) = (640.0, 160.0, 28.0);
    let mut ys = vec![bankroll];
    ys.extend(trades.iter().map(|t| t.equity));
    let lo = ys.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let span = (hi - lo).max(1.0);
    let n = ys.len();
    let x = |i: usize| pad + (i as f64) / ((n - 1).max(1) as f64) * (w - 2.0 * pad);
    let y = |v: f64| h - pad - (v - lo) / span * (h - 2.0 * pad);

    let mut p = format!(
        "<svg viewBox=\"0 0 {w} {h}\" width=\"100%\" height=\"{h}\" style=\"max-width:{w}px\">"
    );
    // starting-bankroll baseline
    p.push_str(&format!(
        "<line x1=\"{}\" y1=\"{:.1}\" x2=\"{}\" y2=\"{:.1}\" stroke=\"#3a4253\" stroke-dasharray=\"4 4\"/>",
        pad, y(bankroll), w - pad, y(bankroll)
    ));
    let pts: String = ys
        .iter()
        .enumerate()
        .map(|(i, v)| format!("{:.1},{:.1}", x(i), y(*v)))
        .collect::<Vec<_>>()
        .join(" ");
    let stroke = if ys[n - 1] >= bankroll {
        "#3fb950"
    } else {
        "#f85149"
    };
    p.push_str(&format!(
        "<polyline points=\"{pts}\" fill=\"none\" stroke=\"{stroke}\" stroke-width=\"2\"/>"
    ));
    p.push_str(&format!(
        "<text x=\"{}\" y=\"{:.1}\" fill=\"#6b7488\" font-size=\"10\">${:.0}</text>",
        pad + 2.0,
        y(bankroll) - 4.0,
        bankroll
    ));
    p.push_str("</svg>");
    p
}

fn usd_signed(v: f64) -> String {
    let sign = if v < 0.0 { "−" } else { "+" };
    format!("{sign}${:.2}", v.abs())
}

/// Human-readable bucket label from a market's shape fields (shared by MarketEvaluation and Capture).
fn label(
    market_type: &str,
    threshold: f64,
    threshold_upper: Option<f64>,
    unit: Option<&str>,
) -> String {
    let u = unit.unwrap_or("F");
    match market_type {
        "temp_at_least" => format!("≥ {:.0}°{u}", threshold),
        "temp_at_most" => format!("≤ {:.0}°{u}", threshold),
        "temp_bucket" => match threshold_upper {
            Some(hi) if (hi - threshold).abs() > 1e-9 => format!("{:.0}–{:.0}°{u}", threshold, hi),
            _ => format!("= {:.0}°{u}", threshold),
        },
        "temperature" => format!("≥ {:.0}°F", threshold),
        "precipitation" => "rain".to_string(),
        other => other.to_string(),
    }
}

fn reliability_svg(preds: &[f64], outs: &[f64]) -> String {
    let (_edges, means, obs) = CalibrationAnalyzer::reliability_diagram(preds, outs, 10);
    let (w, h, pad) = (320.0, 320.0, 36.0);
    let x = |v: f64| pad + v * (w - 2.0 * pad);
    let y = |v: f64| h - pad - v * (h - 2.0 * pad);
    let mut p = format!("<svg viewBox=\"0 0 {w} {h}\" width=\"{w}\" height=\"{h}\">");
    // frame + diagonal
    p.push_str(&format!(
        "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"none\" stroke=\"#2a3142\"/>",
        pad,
        pad,
        w - 2.0 * pad,
        h - 2.0 * pad
    ));
    p.push_str(&format!(
        "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#3a4253\" stroke-dasharray=\"4 4\"/>",
        x(0.0), y(0.0), x(1.0), y(1.0)
    ));
    // counts per bin for sizing
    let mut counts = [0usize; 10];
    for pr in preds {
        let b = ((*pr * 10.0) as usize).min(9);
        counts[b] += 1;
    }
    for i in 0..10 {
        if !obs[i].is_finite() || counts[i] == 0 {
            continue;
        }
        let r = 3.0 + (counts[i] as f64).sqrt().min(7.0);
        p.push_str(&format!(
            "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"{:.1}\" fill=\"#5b8def\" fill-opacity=\"0.8\"/>",
            x(means[i]),
            y(obs[i]),
            r
        ));
    }
    p.push_str(&format!(
        "<text x=\"{}\" y=\"{}\" fill=\"#8a93a6\" font-size=\"11\" text-anchor=\"middle\">model probability →</text>",
        w / 2.0, h - 8.0
    ));
    p.push_str(&format!(
        "<text x=\"12\" y=\"{}\" fill=\"#8a93a6\" font-size=\"11\" transform=\"rotate(-90 12 {})\" text-anchor=\"middle\">realized frequency →</text>",
        h / 2.0, h / 2.0
    ));
    p.push_str("</svg>");
    p
}

fn card(label: &str, value: &str, sub: &str) -> String {
    format!(
        "<div class=\"card\"><div class=\"label\">{}</div><div class=\"value\">{}</div><div class=\"sub\">{}</div></div>",
        esc(label), value, esc(sub)
    )
}

fn pct(a: usize, b: usize) -> usize {
    if b == 0 {
        0
    } else {
        100 * a / b
    }
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

const HEAD: &str = r#"<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Weather Market Dashboard</title><style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:#0e1117;color:#e6e9ef;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:22px;margin:0 0 4px}
.tag{color:#8a93a6;margin:0 0 24px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:0 0 8px}
.card{background:#161b24;border:1px solid #232a37;border-radius:10px;padding:14px 16px}
.card .label{color:#8a93a6;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.card .value{font-size:26px;font-weight:600;margin:4px 0 2px}
.card .sub{color:#6b7488;font-size:12px}
.panel{background:#11151d;border:1px solid #1e242f;border-radius:12px;padding:18px 20px;margin:20px 0}
.panel h2{font-size:15px;margin:0 0 14px;color:#cfd5e1}
.sub-h{font-size:13px;color:#cfd5e1;margin:22px 0 8px;font-weight:600}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid #1c222d}
th{color:#8a93a6;font-weight:600;position:sticky;top:0;background:#11151d}
.panel:has(table){max-height:560px;overflow:auto}
.muted{color:#6b7488}
.yes{color:#3fb950;font-weight:600}.no{color:#8a93a6}
.hit{color:#3fb950}.miss{color:#f85149}
.reliab{display:flex;gap:24px;align-items:center;flex-wrap:wrap}
.legend{max-width:340px}
code{background:#1c222d;padding:1px 6px;border-radius:4px;font-size:12px}
a{color:#5b8def}
</style></head><body><div class="wrap">
<h1>Weather Market Dashboard</h1>
<p class="tag">Calibrated daily-high model vs real resolved Polymarket weather markets</p>"#;

const FOOT: &str = r#"<p class="muted" style="margin-top:32px">Generated by <code>weather_dashboard</code> · model = posterior-predictive daily-high (degC), bucket-priced round-half-up.</p>
</div></body></html>"#;

#[derive(Debug)]
struct Args {
    markets: Option<PathBuf>,
    output: PathBuf,
    cache_dir: PathBuf,
    captures: PathBuf,
    lookback_days: i64,
    refresh: bool,
    forecast: bool,
    forecast_cache_dir: PathBuf,
    forecast_sigma: Option<f64>,
    edge_threshold: f64,
    kelly_fraction: f64,
    max_position_pct: f64,
    bankroll: f64,
    max_edge: Option<f64>,
    no_sell_buckets: bool,
    trade_day_of: bool,
}

impl Args {
    fn parse(argv: Vec<String>) -> Result<Self, String> {
        let mut map: HashMap<String, String> = HashMap::new();
        let mut flags: Vec<String> = Vec::new();
        let mut i = 0;
        while i < argv.len() {
            let k = argv[i].clone();
            if k == "--refresh"
                || k == "--forecast"
                || k == "--no-sell-buckets"
                || k == "--trade-day-of"
            {
                flags.push(k);
                i += 1;
                continue;
            }
            if !k.starts_with("--") || i + 1 >= argv.len() {
                return Err(format!("bad argument '{k}'\n{}", usage()));
            }
            map.insert(k, argv[i + 1].clone());
            i += 2;
        }
        Ok(Self {
            markets: map.get("--markets").map(PathBuf::from),
            output: map
                .get("--output")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("dashboard.html")),
            cache_dir: map
                .get("--cache-dir")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("data/weather_cache")),
            captures: map
                .get("--captures")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("data/captures.jsonl")),
            lookback_days: map
                .get("--lookback-days")
                .and_then(|v| v.parse().ok())
                .unwrap_or(45),
            refresh: flags.iter().any(|f| f == "--refresh"),
            forecast: flags.iter().any(|f| f == "--forecast"),
            forecast_cache_dir: map
                .get("--forecast-cache-dir")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("data/forecast_cache")),
            forecast_sigma: map.get("--forecast-sigma").and_then(|v| v.parse().ok()),
            edge_threshold: map
                .get("--edge-threshold")
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| config::backtest_params().edge_threshold),
            kelly_fraction: map
                .get("--kelly-fraction")
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| config::backtest_params().kelly_fraction),
            max_position_pct: map
                .get("--max-position-pct")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.10),
            bankroll: map
                .get("--bankroll")
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(config::initial_capital),
            max_edge: map.get("--max-edge").and_then(|v| v.parse().ok()),
            no_sell_buckets: flags.iter().any(|f| f == "--no-sell-buckets"),
            trade_day_of: flags.iter().any(|f| f == "--trade-day-of"),
        })
    }
}

fn usage() -> String {
    "usage: cargo run --bin weather_dashboard -- [--markets <path.csv>] [--output dashboard.html] \
     [--cache-dir data/weather_cache] [--lookback-days 45] [--refresh] \
     [--forecast] [--forecast-cache-dir data/forecast_cache] \
     [--edge-threshold 0.05] [--kelly-fraction 0.25] [--max-position-pct 0.10] [--bankroll 100000] \
     [--max-edge 0.30] [--no-sell-buckets] [--trade-day-of]"
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cap(date: &str, city: &str, price: f64, est: Option<f64>, outcome: Option<f64>) -> Capture {
        let target: NaiveDate = date.parse().unwrap();
        Capture {
            captured_at: target - chrono::Duration::days(1), // lead 1: the tradable regime
            target_date: target,
            market_type: "temp_bucket".into(),
            threshold: 90.0,
            threshold_upper: Some(91.0),
            unit: Some("F".into()),
            city: city.into(),
            entry_price: price,
            model_estimate: est,
            outcome,
            source: "polymarket".into(),
        }
    }

    fn params() -> StrategyParams {
        StrategyParams {
            edge_threshold: 0.05,
            kelly_fraction: 0.25,
            max_position_pct: 0.10,
            bankroll: 100_000.0,
            max_edge: None,
            no_sell_buckets: false,
            trade_day_of: false,
        }
    }

    #[test]
    fn strategy_books_edges_and_compounds() {
        let sp = params();
        let caps = vec![
            cap("2026-06-10", "Denver", 0.40, Some(0.60), Some(1.0)), // BUY, wins
            cap("2026-06-11", "Denver", 0.50, Some(0.20), Some(0.0)), // SELL, wins
            cap("2026-06-12", "Denver", 0.50, Some(0.52), Some(1.0)), // thin edge → skipped
            cap("2026-06-13", "Denver", 0.40, Some(0.60), None),      // unresolved → not settled
        ];
        let trades = run_strategy(&caps, &sp);
        assert_eq!(
            trades.len(),
            2,
            "only the two resolved, edge-clearing captures trade"
        );
        assert_eq!(trades[0].side, "BUY");
        assert!(trades[0].pnl > 0.0);
        assert_eq!(trades[1].side, "SELL");
        assert!(trades[1].pnl > 0.0);
        // compounding: second trade's equity reflects the first trade's PnL
        assert!(trades[1].equity > trades[0].equity);

        let refs: Vec<&Trade> = trades.iter().collect();
        let a = agg(&refs);
        assert_eq!(a.n, 2);
        assert_eq!(a.wins, 2);
        assert!((a.hit_rate() - 1.0).abs() < 1e-9);
        assert!(a.pnl > 0.0 && a.roi() > 0.0);
    }

    #[test]
    fn decide_filters_thin_edges_and_untradable_prices() {
        let sp = params();
        let mt = "temp_bucket";
        assert!(decide(0.60, 0.40, mt, 1, &sp).is_some_and(|(side, _)| side == "BUY"));
        assert!(decide(0.20, 0.50, mt, 1, &sp).is_some_and(|(side, _)| side == "SELL"));
        assert!(
            decide(0.52, 0.50, mt, 1, &sp).is_none(),
            "edge below threshold"
        );
        assert!(
            decide(0.60, 0.00, mt, 1, &sp).is_none(),
            "0/1 price is not a real entry"
        );
        assert!(decide(0.60, 1.00, mt, 1, &sp).is_none());
    }

    #[test]
    fn decide_gates_day_of_trades_by_default() {
        let sp = params();
        assert!(
            decide(0.60, 0.40, "temp_bucket", 0, &sp).is_none(),
            "day-of gated by default"
        );
        assert!(
            decide(0.60, 0.40, "temp_bucket", -1, &sp).is_some(),
            "post-day allowed"
        );
        let open = StrategyParams {
            trade_day_of: true,
            ..params()
        };
        assert!(
            decide(0.60, 0.40, "temp_bucket", 0, &open).is_some(),
            "--trade-day-of re-enables"
        );
    }

    #[test]
    fn decide_respects_max_edge_and_no_sell_buckets() {
        // max_edge caps oversized disagreements (the model-miscalibration signal)
        let capped = StrategyParams {
            max_edge: Some(0.30),
            ..params()
        };
        assert!(
            decide(0.20, 0.50, "temp_bucket", 1, &capped).is_some(),
            "0.30 edge is within cap"
        );
        assert!(
            decide(0.05, 0.50, "temp_bucket", 1, &capped).is_none(),
            "0.45 edge exceeds cap"
        );

        // no_sell_buckets suppresses only SELL on temp_bucket, not BUYs or other market types
        let nsb = StrategyParams {
            no_sell_buckets: true,
            ..params()
        };
        assert!(
            decide(0.20, 0.50, "temp_bucket", 1, &nsb).is_none(),
            "bucket SELL suppressed"
        );
        assert!(
            decide(0.20, 0.50, "temp_at_most", 1, &nsb).is_some(),
            "non-bucket SELL allowed"
        );
        assert!(
            decide(0.60, 0.40, "temp_bucket", 1, &nsb).is_some_and(|(s, _)| s == "BUY"),
            "bucket BUY allowed"
        );
    }
}
