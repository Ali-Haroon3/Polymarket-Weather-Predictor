//! Generate a self-contained HTML dashboard from real Polymarket weather markets: model
//! calibration vs realized outcomes, plus PnL on any markets that carry a real (traded) price.
//!
//! Weather is fetched once per city and cached to disk, so re-generating the dashboard is fast.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use chrono::{Duration, NaiveDate};

use polymarket_weather_predictor::backtesting::{
    evaluate_markets, evaluate_markets_with_forecast, MarketEvaluation, RealMarketLoader,
};
use polymarket_weather_predictor::cities;
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

    let markets = RealMarketLoader::load_from_path(&args.markets)
        .map_err(|e| format!("failed to load markets: {e}"))?;
    if markets.is_empty() {
        return Err("no markets in input".to_string());
    }
    println!("Loaded {} markets", markets.len());

    let weather = load_weather(&markets, args.lookback_days, &args.cache_dir, args.refresh)?;
    let covered: Vec<&String> = weather.keys().collect();
    println!("Weather available for {} cities: {:?}", covered.len(), covered);

    let evals = if args.forecast {
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
        evaluate_markets_with_forecast(&markets, &weather, args.lookback_days, &debiased, sigma)
    } else {
        evaluate_markets(&markets, &weather, args.lookback_days)
    };
    let captures = load_captures(&args.captures);
    if !captures.is_empty() {
        println!("Loaded {} forward captures from {}", captures.len(), args.captures.display());
    }
    let html = render_dashboard(&evals, &captures);

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
        let e = ranges
            .entry(m.city.as_str())
            .or_insert((m.date, m.date));
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
            let _ = std::fs::write(
                &cache_file,
                serde_json::to_vec(&rows).unwrap_or_default(),
            );
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
    std::fs::create_dir_all(cache_dir)
        .map_err(|e| format!("cannot create forecast cache dir {}: {e}", cache_dir.display()))?;

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
        let Some(rows) = weather.get(city) else { continue };
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
    let var = residuals.iter().map(|r| (r - bias).powi(2)).sum::<f64>() / (residuals.len() - 1) as f64;
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

/// Simple PnL on markets with a real (non 0/1) traded price: bet the model's edge, settle on outcome.
struct Pnl {
    trades: usize,
    total: f64,
    wins: usize,
}

fn compute_pnl(evals: &[&MarketEvaluation], edge_threshold: f64) -> Pnl {
    let mut p = Pnl { trades: 0, total: 0.0, wins: 0 };
    for e in evals {
        let Some(est) = e.model_estimate else { continue };
        let price = e.market_price;
        if price <= 0.0 || price >= 1.0 {
            continue; // fabricated / resolved price, not a tradable entry
        }
        let edge = est - price;
        if edge.abs() < edge_threshold {
            continue;
        }
        p.trades += 1;
        // $1 stake; BUY YES if model richer than market, else SELL.
        let pnl = if edge > 0.0 {
            e.actual_outcome - price
        } else {
            price - e.actual_outcome
        };
        p.total += pnl;
        if pnl > 0.0 {
            p.wins += 1;
        }
    }
    p
}

// ── forward captures (from the capture_prices daemon) ───────────────────────

#[derive(serde::Deserialize)]
struct Capture {
    target_date: NaiveDate,
    market_type: String,
    threshold: f64,
    threshold_upper: Option<f64>,
    unit: Option<String>,
    city: String,
    entry_price: f64,
    model_estimate: Option<f64>,
    outcome: Option<f64>,
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

fn render_dashboard(evals: &[MarketEvaluation], captures: &[Capture]) -> String {
    let all: Vec<&MarketEvaluation> = evals.iter().collect();
    let total = all.len();
    let evaluated = all.iter().filter(|e| e.model_estimate.is_some()).count();
    let overall = calibration(&all);

    // Forward captures: settle the ones that have resolved into MarketEvaluations for PnL.
    let settled: Vec<MarketEvaluation> = captures
        .iter()
        .filter_map(|c| {
            let o = c.outcome?;
            Some(MarketEvaluation {
                date: c.target_date,
                city: c.city.clone(),
                market_id: String::new(),
                market_title: String::new(),
                market_type: c.market_type.clone(),
                threshold: c.threshold,
                threshold_upper: c.threshold_upper,
                unit: c.unit.clone(),
                market_price: c.entry_price,
                actual_outcome: o,
                model_estimate: c.model_estimate,
            })
        })
        .collect();
    let settled_refs: Vec<&MarketEvaluation> = settled.iter().collect();
    let pnl = compute_pnl(&settled_refs, 0.05);
    let captured = captures.len();
    let pending = captured - settled.len();

    // per-city
    let mut by_city: BTreeMap<&str, Vec<&MarketEvaluation>> = BTreeMap::new();
    for e in &all {
        by_city.entry(e.city.as_str()).or_default().push(e);
    }

    let mut s = String::new();
    s.push_str(HEAD);

    // summary cards
    s.push_str("<div class=\"cards\">");
    s.push_str(&card("Markets", &total.to_string(), "real resolved Polymarket buckets"));
    s.push_str(&card(
        "Evaluated",
        &format!("{evaluated}"),
        &format!("{}% have model estimates", pct(evaluated, total)),
    ));
    s.push_str(&card("Cities", &by_city.len().to_string(), "with markets"));
    if let Some(c) = &overall {
        // Baseline = always predict the base rate. Brier skill score > 0 means the model beats it.
        let baseline = c.base_rate * (1.0 - c.base_rate);
        let bss = if baseline > 0.0 { 1.0 - c.brier / baseline } else { 0.0 };
        s.push_str(&card("Brier", &format!("{:.3}", c.brier), &format!("baseline {baseline:.3} (predict base rate)")));
        s.push_str(&card("ECE", &format!("{:.3}", c.ece), "calibration error, lower better"));
        s.push_str(&card(
            "Brier skill",
            &format!("{:+.2}", bss),
            if bss > 0.0 { "beats the base-rate baseline" } else { "below baseline — buckets need seasonality" },
        ));
        s.push_str(&card(
            "Accuracy",
            &format!("{:.0}%", c.accuracy * 100.0),
            &format!("base rate {:.0}% (so accuracy flatters)", c.base_rate * 100.0),
        ));
    }
    s.push_str("</div>");

    // Forward-capture PnL panel
    s.push_str("<div class=\"panel\"><h2>Forward capture (real entry prices)</h2>");
    if captured == 0 {
        s.push_str("<p class=\"muted\">No captures yet. Polymarket purges price history after resolution, so real entry prices are only available live. Run <code>capture_prices</code> daily (cron / schedule) to snapshot active markets; PnL settles as they resolve.</p>");
    } else {
        let avg = if pnl.trades > 0 { pnl.total / pnl.trades as f64 } else { 0.0 };
        let wr = if pnl.trades > 0 { 100.0 * pnl.wins as f64 / pnl.trades as f64 } else { 0.0 };
        s.push_str(&format!(
            "<div class=\"cards\">{}{}{}{}</div>",
            card("Captured", &captured.to_string(), &format!("{pending} still pending resolution")),
            card("Settled trades", &pnl.trades.to_string(), "edge > 5%, resolved"),
            card("Total PnL", &format!("{:+.3}", pnl.total), "per $1 staked"),
            card("Win rate", &format!("{:.0}%", wr), &format!("avg {:+.3}/trade", avg)),
        ));
        // live signals: biggest model-vs-market disagreements among pending captures
        let mut live: Vec<&Capture> = captures
            .iter()
            .filter(|c| c.outcome.is_none() && c.model_estimate.is_some() && c.entry_price > 0.0 && c.entry_price < 1.0)
            .collect();
        live.sort_by(|a, b| {
            let ea = (a.model_estimate.unwrap() - a.entry_price).abs();
            let eb = (b.model_estimate.unwrap() - b.entry_price).abs();
            eb.partial_cmp(&ea).unwrap_or(std::cmp::Ordering::Equal)
        });
        if !live.is_empty() {
            s.push_str("<h3 style=\"font-size:13px;color:#8a93a6;margin:16px 0 8px\">Largest open model-vs-market disagreements</h3>");
            s.push_str("<table><thead><tr><th>Resolves</th><th>City</th><th>Bucket</th><th>Market</th><th>Model</th><th>Edge</th><th>Side</th></tr></thead><tbody>");
            for c in live.iter().take(15) {
                let est = c.model_estimate.unwrap();
                let edge = est - c.entry_price;
                let (side, scls) = if edge > 0.0 { ("BUY", "yes") } else { ("SELL", "miss") };
                s.push_str(&format!(
                    "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.3}</td><td>{:.3}</td><td>{:+.3}</td><td class=\"{}\">{}</td></tr>",
                    c.target_date, esc(&c.city), esc(&cap_bucket_label(c)), c.entry_price, est, edge, scls, side
                ));
            }
            s.push_str("</tbody></table>");
        }
    }
    s.push_str("</div>");

    // reliability diagram
    if let Some(c) = &overall {
        let preds: Vec<f64> = all.iter().filter_map(|e| e.model_estimate).collect();
        let outs: Vec<f64> = all
            .iter()
            .filter(|e| e.model_estimate.is_some())
            .map(|e| e.actual_outcome)
            .collect();
        s.push_str("<div class=\"panel\"><h2>Reliability (model probability vs realized frequency)</h2>");
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
            c.as_ref().map(|c| format!("{:.3}", c.brier)).unwrap_or_else(|| "–".into()),
            c.as_ref().map(|c| format!("{:.0}%", c.accuracy * 100.0)).unwrap_or_else(|| "–".into()),
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
        let (oc, ocls) = if e.actual_outcome > 0.5 { ("YES", "yes") } else { ("NO", "no") };
        // mark correctness of a >0.5 call
        let hit = e
            .model_estimate
            .map(|p| (p > 0.5) == (e.actual_outcome > 0.5))
            .unwrap_or(false);
        let estcls = if e.model_estimate.is_some() && hit { "hit" } else if e.model_estimate.is_some() { "miss" } else { "" };
        s.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td class=\"{}\">{}</td><td>{}</td><td class=\"{}\">{}</td></tr>",
            e.date,
            esc(&e.city),
            esc(&bucket_label(e)),
            estcls,
            est,
            price,
            ocls,
            oc,
        ));
    }
    s.push_str("</tbody></table>");
    if total > 600 {
        s.push_str(&format!("<p class=\"muted\">Showing 600 of {total} markets.</p>"));
    }
    s.push_str("</div>");

    s.push_str(FOOT);
    s
}

fn bucket_label(e: &MarketEvaluation) -> String {
    let u = e.unit.as_deref().unwrap_or("F");
    match e.market_type.as_str() {
        "temp_at_least" => format!("≥ {:.0}°{u}", e.threshold),
        "temp_at_most" => format!("≤ {:.0}°{u}", e.threshold),
        "temp_bucket" => match e.threshold_upper {
            Some(hi) if (hi - e.threshold).abs() > 1e-9 => format!("{:.0}–{:.0}°{u}", e.threshold, hi),
            _ => format!("= {:.0}°{u}", e.threshold),
        },
        "temperature" => format!("≥ {:.0}°F", e.threshold),
        "precipitation" => "rain".to_string(),
        other => other.to_string(),
    }
}

fn cap_bucket_label(c: &Capture) -> String {
    let u = c.unit.as_deref().unwrap_or("F");
    match c.market_type.as_str() {
        "temp_at_least" => format!("≥ {:.0}°{u}", c.threshold),
        "temp_at_most" => format!("≤ {:.0}°{u}", c.threshold),
        "temp_bucket" => match c.threshold_upper {
            Some(hi) if (hi - c.threshold).abs() > 1e-9 => format!("{:.0}–{:.0}°{u}", c.threshold, hi),
            _ => format!("= {:.0}°{u}", c.threshold),
        },
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
        pad, pad, w - 2.0 * pad, h - 2.0 * pad
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
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
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
    markets: PathBuf,
    output: PathBuf,
    cache_dir: PathBuf,
    captures: PathBuf,
    lookback_days: i64,
    refresh: bool,
    forecast: bool,
    forecast_cache_dir: PathBuf,
    forecast_sigma: Option<f64>,
}

impl Args {
    fn parse(argv: Vec<String>) -> Result<Self, String> {
        let mut map: HashMap<String, String> = HashMap::new();
        let mut flags: Vec<String> = Vec::new();
        let mut i = 0;
        while i < argv.len() {
            let k = argv[i].clone();
            if k == "--refresh" || k == "--forecast" {
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
        let markets = map
            .get("--markets")
            .map(PathBuf::from)
            .ok_or_else(|| format!("--markets is required\n{}", usage()))?;
        Ok(Self {
            markets,
            output: map.get("--output").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("dashboard.html")),
            cache_dir: map.get("--cache-dir").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("data/weather_cache")),
            captures: map.get("--captures").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("data/captures.jsonl")),
            lookback_days: map.get("--lookback-days").and_then(|v| v.parse().ok()).unwrap_or(45),
            refresh: flags.iter().any(|f| f == "--refresh"),
            forecast: flags.iter().any(|f| f == "--forecast"),
            forecast_cache_dir: map
                .get("--forecast-cache-dir")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("data/forecast_cache")),
            forecast_sigma: map.get("--forecast-sigma").and_then(|v| v.parse().ok()),
        })
    }
}

fn usage() -> String {
    "usage: cargo run --bin weather_dashboard -- --markets <path.csv> [--output dashboard.html] \
     [--cache-dir data/weather_cache] [--lookback-days 45] [--refresh] \
     [--forecast] [--forecast-cache-dir data/forecast_cache]"
        .to_string()
}
