//! Daily capture daemon for forward, real-price evaluation.
//!
//! Each run: snapshot every ACTIVE Polymarket weather market (its current price + the model's
//! probability computed from the LIVE forecast of its target day, falling back to climatology beyond
//! the forecast horizon), and finalize any previously-captured market that has since resolved (fill
//! in its outcome). Over time this accrues a real dataset of (entry price, model estimate, realized
//! outcome) — at genuine trading lead with real prices — that the dashboard turns into calibration
//! and PnL.
//!
//! Run daily, e.g. via cron or the schedule skill:  cargo run --release --bin capture_prices

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use chrono::{Duration, NaiveDate, Utc};
use serde::{Deserialize, Serialize};

use chrono::NaiveDateTime;

use polymarket_weather_predictor::api::{
    KalshiHistoryDownloader, PolymarketHistoryDownloader, WeatherMarketRow,
};
use polymarket_weather_predictor::backtesting::{evaluate_markets_with_forecast, market_estimate};
use polymarket_weather_predictor::cities;
use polymarket_weather_predictor::config;
use polymarket_weather_predictor::data_pipeline::station_obs::{
    forecast_day_max_c, nowcast_mu_sigma, phase_for, wu_running_max_c, IemObsFetcher, Phase,
};
use polymarket_weather_predictor::data_pipeline::{MultiSourceAggregator, OpenMeteoFetcher};
use polymarket_weather_predictor::models::BayesianWeatherModel;
use polymarket_weather_predictor::stations::{station_for, Station};
use polymarket_weather_predictor::types::{SimulatedMarket, WeatherRecord};

/// LEGACY forecast-error spread (degC): only for markets the station-aware path can't price —
/// cities without a verified resolution station, Kalshi rows (different resolution source: NWS CLI
/// day-high, not the WU ob-max; mapping unverified until Kalshi captures settle), and any row whose
/// obs/forecast fetch came up empty.
///
/// Mapped Polymarket cities are priced by `station_nowcast` instead: markets resolve on the max of
/// whole-degree METAR obs at a specific airport station (verified 43/43 against settled outcomes),
/// so post/day-of markets read the same METAR feed and lead-k markets use the station-coordinate
/// forecast with per-(city, lead) sigma/bias fitted on Jan–Apr 2026 (see `src/stations.rs`). That
/// removed the structural bucket miscalibration this constant could never fix: the old single-sigma
/// grid model priced finished days as if still uncertain (fake edges) and the wrong microclimate.
const FORECAST_SIGMA: f64 = 2.0;

/// Only direct-lookup markets whose target day is within this many days behind today. A market that
/// hasn't resolved this long after its date is voided/stuck; stop re-querying it every run.
const RESOLVE_WINDOW_DAYS: i64 = 45;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Snapshot {
    captured_at: NaiveDate,
    target_date: NaiveDate,
    market_id: String,
    market_title: String,
    market_type: String,
    threshold: f64,
    threshold_upper: Option<f64>,
    unit: Option<String>,
    city: String,
    entry_price: f64,
    model_estimate: Option<f64>,
    outcome: Option<f64>,
    /// Venue ("polymarket" / "kalshi"). Defaulted for snapshots captured before multi-venue support.
    #[serde(default = "default_source")]
    source: String,
    /// Live daily-high forecast (°C) used to price this market and the σ (°C) applied. Stored because
    /// the live forecast is ephemeral — it can't be recovered after the fact — and it's the raw input
    /// needed to recalibrate bucket pricing against realized highs as settled captures accrue.
    /// None ⇒ priced from climatology (no forecast reached this city/date).
    #[serde(default)]
    forecast_high: Option<f64>,
    #[serde(default)]
    forecast_sigma: Option<f64>,
}

fn default_source() -> String {
    "polymarket".to_string()
}

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), String> {
    let out_path = std::env::args()
        .skip(1)
        .collect::<Vec<_>>()
        .windows(2)
        .find(|w| w[0] == "--out")
        .map(|w| PathBuf::from(&w[1]))
        .unwrap_or_else(|| PathBuf::from("data/captures.jsonl"));
    let cache_dir = PathBuf::from("data/weather_cache");
    let lookback_days = 45i64;

    let downloader = PolymarketHistoryDownloader::default();
    let today = Utc::now().date_naive();

    // ── async: pull active markets (entry prices) ──
    println!("Fetching active Polymarket weather markets...");
    let mut active = downloader
        .download_weather_markets(true, 4000)
        .await
        .map_err(|e| format!("active fetch failed: {e}"))?;
    println!("  {} active weather markets", active.len());

    // Finalize by DIRECT id lookup of our own unresolved, past-dated snapshots. Recently-resolved
    // low-volume weather markets never surface in Polymarket's bulk closed-market feeds — the
    // `temperature` tag lags by weeks (newest closed events sit ~10 weeks back) and the volume scan
    // is capped at high-volume markets — so the old bulk-resolved discovery finalized nothing. Each
    // market resolves cleanly when queried by id, which is exactly what we stored at capture time.
    let mut outcomes: HashMap<String, f64> = HashMap::new();
    let need_ids: Vec<String> = load_snapshots(&out_path)
        .into_iter()
        .filter(|s| {
            s.outcome.is_none()
                && s.source == "polymarket"
                && s.target_date <= today
                && s.target_date >= today - Duration::days(RESOLVE_WINDOW_DAYS)
        })
        .map(|s| s.market_id)
        .collect();
    if !need_ids.is_empty() {
        println!(
            "Finalizing {} past-dated unresolved Polymarket snapshots by direct id lookup...",
            need_ids.len()
        );
        let found = downloader.fetch_outcomes_for_ids(&need_ids).await;
        println!("  {} newly resolved via direct lookup", found.len());
        outcomes.extend(found);
    }

    // Kalshi (demo by default) — additive and best-effort: no creds ⇒ skipped, a fetch error is
    // logged but never aborts the Polymarket run (same degrade-by-design contract as the fetchers).
    let kalshi = KalshiHistoryDownloader::new();
    if kalshi.is_available() {
        println!(
            "Fetching Kalshi weather markets ({})...",
            config::kalshi_base_url()
        );
        match kalshi.download_weather_markets(true, 4000).await {
            Ok(k) => {
                println!("  {} active Kalshi weather markets", k.len());
                active.extend(k);
            }
            Err(e) => eprintln!("  Kalshi active fetch failed, continuing without it: {e}"),
        }
        match kalshi.download_weather_markets(false, 4000).await {
            Ok(k) => {
                let before = outcomes.len();
                for r in k {
                    if let Some(o) = r.outcome {
                        outcomes.insert(r.market_id, o);
                    }
                }
                println!(
                    "  {} Kalshi resolved outcomes added",
                    outcomes.len() - before
                );
            }
            Err(e) => eprintln!("  Kalshi resolved fetch failed, continuing without it: {e}"),
        }
    } else {
        println!(
            "Kalshi not configured — skipping (set KALSHI_API_KEY_ID + KALSHI_PRIVATE_KEY_PATH to enable)."
        );
    }

    // ── blocking: weather + model + file IO, off the async executor ──
    tokio::task::spawn_blocking(move || {
        process(
            active,
            outcomes,
            today,
            lookback_days,
            &cache_dir,
            &out_path,
        )
    })
    .await
    .map_err(|e| format!("worker join failed: {e}"))?
}

fn process(
    active: Vec<WeatherMarketRow>,
    outcomes: HashMap<String, f64>,
    today: NaiveDate,
    lookback_days: i64,
    cache_dir: &PathBuf,
    out_path: &PathBuf,
) -> Result<(), String> {
    let mut snaps = load_snapshots(out_path);
    let existing: HashSet<String> = snaps.iter().map(|s| s.market_id.clone()).collect();

    // 1. Finalize: fill outcomes for snapshots that have since resolved.
    let mut finalized = 0;
    for s in snaps.iter_mut() {
        if s.outcome.is_none() {
            if let Some(o) = outcomes.get(&s.market_id) {
                s.outcome = Some(*o);
                finalized += 1;
            }
        }
    }
    println!("Finalized {finalized} previously-captured markets");

    // 2. Snapshot new active markets: compute the model estimate from current climatology.
    let fresh: Vec<&WeatherMarketRow> = active
        .iter()
        .filter(|r| !existing.contains(&r.market_id))
        .collect();
    println!("{} new active markets to snapshot", fresh.len());

    if !fresh.is_empty() {
        // Station-aware pricing first (verified Polymarket resolution stations); everything it
        // can't price falls through to the legacy forecast/climatology path.
        let mut est: HashMap<String, f64> = HashMap::new();
        let mut used: HashMap<String, (f64, f64)> = HashMap::new(); // market_id -> (mu, sigma) °C
        let mut legacy: Vec<&WeatherMarketRow> = Vec::new();
        let mut pricer = StationPricer::new(today);
        for r in &fresh {
            match pricer.estimate(r) {
                Some((mu, sigma)) => {
                    let mut model = BayesianWeatherModel::default();
                    model.set_point_forecast(mu, sigma);
                    match market_estimate(&model, &to_sim(r)) {
                        Some(p) => {
                            est.insert(r.market_id.clone(), p);
                            used.insert(r.market_id.clone(), (mu, sigma));
                        }
                        None => legacy.push(r),
                    }
                }
                None => legacy.push(r),
            }
        }
        println!(
            "{} markets priced from resolution-station nowcast, {} on legacy path",
            est.len(),
            legacy.len()
        );

        let sims: Vec<SimulatedMarket> = legacy.iter().map(|r| to_sim(r)).collect();
        // Price legacy markets from the LIVE forecast of their target day (real trading lead).
        let forecasts = load_forecasts(&sims);
        // Climatology is only the fallback for markets the forecast can't reach (beyond horizon).
        // Active markets are future-dated, so the archive returns nothing for the rest anyway — skip
        // the slow multi-source aggregator entirely when every market already has a forecast.
        let need_weather: Vec<SimulatedMarket> = sims
            .iter()
            .filter(|m| {
                !forecasts
                    .get(&m.city)
                    .map(|f| f.contains_key(&m.date))
                    .unwrap_or(false)
            })
            .cloned()
            .collect();
        let weather = if need_weather.is_empty() {
            HashMap::new()
        } else {
            println!(
                "{} markets beyond forecast horizon; loading climatology",
                need_weather.len()
            );
            load_weather(&need_weather, lookback_days, cache_dir)
        };
        let evals = evaluate_markets_with_forecast(
            &sims,
            &weather,
            lookback_days,
            &forecasts,
            FORECAST_SIGMA,
        );
        for e in &evals {
            if let Some(p) = e.model_estimate {
                est.insert(e.market_id.clone(), p);
            }
        }
        for r in &sims {
            if let Some(fh) = forecasts.get(&r.city).and_then(|m| m.get(&r.date)) {
                used.insert(r.market_id.clone(), (*fh, FORECAST_SIGMA));
            }
        }

        for r in fresh {
            let fs = used.get(r.market_id.as_str()).copied();
            snaps.push(Snapshot {
                captured_at: today,
                target_date: r.target_date,
                market_id: r.market_id.clone(),
                market_title: r.market_title.clone(),
                market_type: r.market_type.clone(),
                threshold: r.threshold,
                threshold_upper: r.threshold_upper,
                unit: r.unit.clone(),
                city: r.city.clone(),
                entry_price: r.price,
                model_estimate: est.get(r.market_id.as_str()).copied(),
                outcome: outcomes.get(&r.market_id).copied(),
                source: r.source.clone(),
                forecast_high: fs.map(|(mu, _)| mu),
                forecast_sigma: fs.map(|(_, sigma)| sigma),
            });
        }
    }

    write_snapshots(out_path, &snaps)?;
    let priced = snaps.iter().filter(|s| s.model_estimate.is_some()).count();
    let settled = snaps.iter().filter(|s| s.outcome.is_some()).count();
    println!(
        "captures.jsonl: {} total · {} with model estimate · {} settled outcomes",
        snaps.len(),
        priced,
        settled
    );
    println!("Wrote {}", out_path.display());
    Ok(())
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
        actual_outcome: 0.0, // unknown at capture time
        city: r.city.clone(),
    }
}

/// Prices a market against its Polymarket resolution station: METAR obs (the very feed the market
/// resolves on) for elapsed hours, the station-coordinate hourly forecast for the rest, and the
/// per-(city, lead) fitted sigma/bias from `stations.rs`. Returns the (mu, sigma) in °C for
/// `set_point_forecast`, or None to fall back to the legacy path. One obs fetch and one forecast
/// fetch per city per run, cached.
struct StationPricer {
    now_utc: NaiveDateTime,
    today: chrono::NaiveDate,
    obs_fetcher: IemObsFetcher,
    open_meteo: OpenMeteoFetcher,
    obs_cache: HashMap<String, Vec<(NaiveDateTime, f64)>>,
    forecast_cache: HashMap<String, Vec<(NaiveDateTime, f64)>>,
}

impl StationPricer {
    fn new(today: chrono::NaiveDate) -> Self {
        Self {
            now_utc: Utc::now().naive_utc(),
            today,
            obs_fetcher: IemObsFetcher::new(),
            open_meteo: OpenMeteoFetcher::new(),
            obs_cache: HashMap::new(),
            forecast_cache: HashMap::new(),
        }
    }

    fn estimate(&mut self, r: &WeatherMarketRow) -> Option<(f64, f64)> {
        if r.source != "polymarket" || !r.market_type.starts_with("temp") {
            return None; // Kalshi resolves on NWS CLI (unverified mapping); precip has no station model
        }
        let st = station_for(&r.city)?;
        let phase = phase_for(self.now_utc, r.target_date, st);
        let (runmax, rest) = match phase {
            Phase::Post => (
                wu_running_max_c(self.obs(st, r.target_date)?, r.target_date, st, None),
                None,
            ),
            Phase::DayOf => {
                let cutoff = self.now_utc;
                let run = wu_running_max_c(
                    self.obs(st, r.target_date)?,
                    r.target_date,
                    st,
                    Some(cutoff),
                );
                let rest = forecast_day_max_c(self.forecast(st)?, r.target_date, st, Some(cutoff));
                (run, rest)
            }
            Phase::Lead(_) => (
                None,
                forecast_day_max_c(self.forecast(st)?, r.target_date, st, None),
            ),
        };
        nowcast_mu_sigma(st, phase, runmax, rest)
    }

    fn obs(&mut self, st: &Station, target: chrono::NaiveDate) -> Option<&[(NaiveDateTime, f64)]> {
        if !self.obs_cache.contains_key(st.city) {
            let start = (target - Duration::days(1)).min(self.today - Duration::days(1));
            let got = self.obs_fetcher.fetch_tmpf_utc(st, start, self.today);
            self.obs_cache.insert(st.city.to_string(), got);
        }
        let v = self.obs_cache.get(st.city).unwrap();
        (!v.is_empty()).then_some(v.as_slice())
    }

    fn forecast(&mut self, st: &Station) -> Option<&[(NaiveDateTime, f64)]> {
        if !self.forecast_cache.contains_key(st.city) {
            // 16-day horizon; a target beyond it simply yields no hours -> legacy fallback.
            let got = self.open_meteo.fetch_forecast_hourly_utc(
                st.lat,
                st.lon,
                self.today - Duration::days(1),
                self.today + Duration::days(15),
            );
            self.forecast_cache.insert(st.city.to_string(), got);
        }
        let v = self.forecast_cache.get(st.city).unwrap();
        (!v.is_empty()).then_some(v.as_slice())
    }
}

fn load_snapshots(path: &PathBuf) -> Vec<Snapshot> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str::<Snapshot>(l).ok())
        .collect()
}

fn write_snapshots(path: &PathBuf, snaps: &[Snapshot]) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
    }
    let mut body = String::new();
    for s in snaps {
        body.push_str(&serde_json::to_string(s).map_err(|e| e.to_string())?);
        body.push('\n');
    }
    std::fs::write(path, body).map_err(|e| format!("write {}: {e}", path.display()))
}

/// Cached aggregated weather per city covering the markets' trailing windows.
fn load_weather(
    markets: &[SimulatedMarket],
    lookback_days: i64,
    cache_dir: &PathBuf,
) -> HashMap<String, Vec<WeatherRecord>> {
    let _ = std::fs::create_dir_all(cache_dir);
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
        let start = min_d - Duration::days(lookback_days + 5);
        // Daily-refresh cache: re-fetch if the cached range doesn't reach max_d.
        print!("  weather {city}: ");
        let rows = aggregator.aggregate(city, start, max_d);
        println!("{} records", rows.len());
        if !rows.is_empty() {
            out.insert(city.to_string(), rows);
        }
    }
    out
}

/// Live daily-high forecasts per city for the markets' target dates (city -> date -> high, degC).
/// Fetched fresh each run since forecasts change daily. Cities without coords, or targets beyond the
/// ~16-day forecast horizon, simply won't appear — the eval falls back to climatology for those.
fn load_forecasts(markets: &[SimulatedMarket]) -> HashMap<String, HashMap<NaiveDate, f64>> {
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

    let fetcher = OpenMeteoFetcher::new();
    let mut out: HashMap<String, HashMap<NaiveDate, f64>> = HashMap::new();
    for (city, (min_d, max_d)) in ranges {
        let Some((lat, lon)) = cities::coords(city) else {
            continue;
        };
        let pairs = fetcher.fetch_forecast_max_live(lat, lon, min_d, max_d);
        if !pairs.is_empty() {
            println!("  forecast {city}: {} days", pairs.len());
            out.insert(city.to_string(), pairs.into_iter().collect());
        }
    }
    out
}
