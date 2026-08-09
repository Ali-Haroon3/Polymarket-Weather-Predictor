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

use polymarket_weather_predictor::api::{
    KalshiHistoryDownloader, PolymarketHistoryDownloader, WeatherMarketRow,
};
use polymarket_weather_predictor::backtesting::spread_sigma::{
    fit_spread_sigma_scale, spread_obs, SpreadObs, SPREAD_SIGMA_MIN_N,
};
use polymarket_weather_predictor::backtesting::{evaluate_markets_with_forecast, market_estimate};
use polymarket_weather_predictor::cities;
use polymarket_weather_predictor::config;
use polymarket_weather_predictor::data_pipeline::open_meteo_fetcher::{
    AirQualityByDay, AI_LOG_CANDIDATES, AI_LOG_MODELS,
};
use polymarket_weather_predictor::data_pipeline::{
    MultiSourceAggregator, OpenMeteoFetcher, StationPricer,
};
use polymarket_weather_predictor::models::BayesianWeatherModel;
use polymarket_weather_predictor::stations::station_for;
use polymarket_weather_predictor::types::{SimulatedMarket, WeatherRecord};

/// LEGACY forecast-error spread (degC): only for markets the station-aware path can't price —
/// cities without a verified resolution station, Kalshi rows (different resolution source: NWS CLI
/// day-high, not the WU ob-max; mapping unverified until Kalshi captures settle), and any row whose
/// obs/forecast fetch came up empty. The 2026-08-09 spread→σ mapping does NOT reach this path —
/// it lives in `StationPricer`, so legacy rows keep this constant (their spreads are still logged,
/// so extending the mapping to them can be validated later the same way).
///
/// Mapped Polymarket cities are priced by `station_nowcast` instead: markets resolve on the max of
/// whole-degree METAR obs at a specific airport station (verified 43/43 against settled outcomes),
/// so post/day-of markets read the same METAR feed and lead-k markets use the station-coordinate
/// forecast with per-(city, lead) sigma/bias fitted on Jan–Apr 2026 (see `src/stations.rs`). That
/// removed the structural bucket miscalibration this constant could never fix: the old single-sigma
/// grid model priced finished days as if still uncertain (fake edges) and the wrong microclimate.
const FORECAST_SIGMA: f64 = 2.0;

/// Summer bias (°C, ADDED to the forecast) for legacy-path cities, measured Jul 2026 against
/// realized highs recovered from winning bucket markets and shrunk toward 0 by n/(n+6). These
/// cities have no station mapping, so the seasonal drift the station tables absorbed in their
/// 2026-07-20 refit has to live here. Istanbul's −1.5 °C is the whole reason its markets kept
/// producing fake SELL edges. Hong Kong had too little settled data to fit (n=2) — left at 0.
const LEGACY_SUMMER_BIAS_C: &[(&str, f64)] =
    &[("Moscow", -0.48), ("Istanbul", -1.55), ("Tel Aviv", 0.62)];

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
    /// Top of the YES book at capture (entry_price is the last trade). What a BUY/SELL would
    /// actually fill at — the dashboard evaluates executable edge from these when present.
    #[serde(default)]
    best_bid: Option<f64>,
    #[serde(default)]
    best_ask: Option<f64>,
    /// Attention proxies as the venue reported them at capture (sentiment research: does hype
    /// money widen mispricing?). Log-only — the trading path never reads them. None ⇒ captured
    /// before these were logged, or the venue omits the field.
    #[serde(default)]
    volume: Option<f64>,
    #[serde(default)]
    volume_24h: Option<f64>,
    #[serde(default)]
    open_interest: Option<f64>,
    #[serde(default)]
    liquidity: Option<f64>,
    /// Forecast smoke for the TARGET day at the city (UTC-day max of hourly PM2.5 µg/m³ / US AQI),
    /// logged to test a smoke-day bias correction (hypothesis: smoke suppresses highs and the
    /// blend only partly prices it). First look 2026-08-09: DEAD on 208 deduped (city, day)
    /// residuals — heavy-smoke days (>55 µg/m³, n=20) realized +0.53 °C ABOVE forecast (the same
    /// sign as the general summer under-forecast, not suppression) and r(pm25, residual) ≈ +0.03.
    /// No correction warranted; the field stays logged for a bigger-sample recheck. None ⇒ beyond
    /// the ~week air-quality horizon or fetch failed. Log-only.
    #[serde(default)]
    pm25_max: Option<f64>,
    #[serde(default)]
    us_aqi_max: Option<f64>,
    /// Daily-high forecasts (°C) for the target day from the AI models in `AI_LOG_MODELS`, at the
    /// same coords the row was priced at (resolution station when mapped, else city). Log-only:
    /// they must beat the blend on accrued captures BEFORE entering it — changing `BLEND_MODELS`
    /// invalidates the fitted sigma/bias tables.
    #[serde(default)]
    forecast_high_aifs: Option<f64>,
    /// Replaced `forecast_high_graphcast` on 2026-07-22: NCEP discontinued GraphCastGFS in
    /// Dec 2025 (the old field never held a value), and AIGFS is its operational successor.
    #[serde(default)]
    forecast_high_aigfs: Option<f64>,
    /// Day-specific uncertainty (°C): std-dev across ensemble members' daily highs for the
    /// TARGET day at the coords the row was priced at (0 = ECMWF ENS ~51 members, 1 = NOAA
    /// GEFS ~31; `ENSEMBLE_LOG_MODELS`). Log-only from 2026-07-23 until the walk-forward gate was
    /// met at the 2026-08-09 check (Brier 0.0947 vs the constant tables' 0.0971 on 1038 rows,
    /// `scripts/lambda_diagnostics.py`) — since then lead ≥ 1 STATION rows price at
    /// σ = max(a·spread, 0.2 °C) (`backtesting::spread_sigma`), and the value priced with is the
    /// value stored here (both come from the same `StationPricer` fetch cache). None ⇒
    /// pre-logging row, fetch failed, or under the member floor.
    #[serde(default)]
    ensemble_spread_ecmwf: Option<f64>,
    #[serde(default)]
    ensemble_spread_gfs: Option<f64>,
    /// Member-MEAN daily highs (°C) from the same ensemble fetch as the spreads (since
    /// 2026-08-09). μ candidates: ensemble means routinely beat deterministic runs at lead 1–2,
    /// and these ride a fetch we already make — but they are LOG-ONLY until they beat the blend's
    /// RMSE on accrued captures (`scripts/lambda_diagnostics.py` ai_means), the same gate the AI
    /// means are still failing. None ⇒ pre-logging row or no data for the day.
    #[serde(default)]
    ensemble_mean_ecmwf: Option<f64>,
    #[serde(default)]
    ensemble_mean_gfs: Option<f64>,
    /// Per-model daily highs (°C, keyed by `BLEND_MODELS` name) that the blended μ was averaged
    /// from, at the row's resolution station, full target day (since 2026-08-09; station rows
    /// only). LOG-ONLY: stored so per-city blend WEIGHTS can be fitted walk-forward from accrued
    /// captures — nothing existed to fit while only the average survived the run. BTreeMap so the
    /// daily full-file rewrite serializes in stable order (a HashMap would churn every committed
    /// row every day). None ⇒ pre-logging row, unmapped city, or failed forecast fetch.
    #[serde(default)]
    blend_model_highs: Option<std::collections::BTreeMap<String, f64>>,
    /// "ensemble" when this row's σ came from the walk-forward spread→σ mapping instead of the
    /// per-(city, lead) constant tables; None ⇒ constant σ (legacy path, day-of/post phases, no
    /// spread available, or a pre-2026-08-09 row). Lets diagnostics and station-table refits
    /// split the two σ regimes instead of pooling them unknowingly.
    #[serde(default)]
    sigma_source: Option<String>,
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

    // Kalshi — market data is anonymous on the public production host, so this always runs; a
    // fetch error is logged but never aborts the Polymarket run (degrade-by-design).
    let kalshi = KalshiHistoryDownloader::new();
    println!(
        "Fetching Kalshi weather markets ({})...",
        config::kalshi_public_base_url()
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

    // Kalshi direct-ticker finalize, mirroring the Polymarket direct-id lookup above: the bulk
    // `status=settled` series scan silently dropped some series in mid-July 2026 (AUS/LAX/MIA
    // accrued a week of unresolved snapshots while other series kept resolving), but every settled
    // market still answers a direct ticker lookup. Only past-dated snapshots the bulk scan didn't
    // just resolve are queried.
    let kalshi_need: Vec<String> = load_snapshots(&out_path)
        .into_iter()
        .filter(|s| {
            s.outcome.is_none()
                && s.source == "kalshi"
                && s.target_date < today
                && s.target_date >= today - Duration::days(RESOLVE_WINDOW_DAYS)
                && !outcomes.contains_key(&s.market_id)
        })
        .map(|s| s.market_id)
        .collect();
    if !kalshi_need.is_empty() {
        println!(
            "Finalizing {} past-dated unresolved Kalshi snapshots by direct ticker lookup...",
            kalshi_need.len()
        );
        let found = kalshi.fetch_outcomes_for_tickers(&kalshi_need).await;
        println!("  {} newly resolved via direct lookup", found.len());
        outcomes.extend(found);
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
        // Walk-forward spread→σ: fit the scale on RESOLVED history before pricing today's
        // captures (everything resolved is strictly earlier than any market being priced, so
        // applying the fit forward is out-of-sample by construction). Under MIN_N rows the
        // pricer keeps the constant per-(city, lead) tables.
        let spread_fit_rows: Vec<SpreadObs> =
            snaps.iter().filter_map(snapshot_spread_obs).collect();
        let spread_scale = fit_spread_sigma_scale(&spread_fit_rows);
        match spread_scale {
            Some(a) => println!(
                "spread→σ: a = {a:.1} fitted on {} resolved rows — lead ≥ 1 station rows with an \
                 ensemble spread price at σ = max(a·spread, 0.2 °C)",
                spread_fit_rows.len()
            ),
            None => println!(
                "spread→σ: {} resolved rows carrying spread < {SPREAD_SIGMA_MIN_N} — constant σ \
                 tables this run",
                spread_fit_rows.len()
            ),
        }

        // Station-aware pricing first (verified Polymarket resolution stations); everything it
        // can't price falls through to the legacy forecast/climatology path.
        let mut est: HashMap<String, f64> = HashMap::new();
        let mut used: HashMap<String, (f64, f64)> = HashMap::new(); // market_id -> (mu, sigma) °C
        let mut dynamic_sigma_ids: HashSet<String> = HashSet::new();
        let mut legacy: Vec<&WeatherMarketRow> = Vec::new();
        let mut pricer = StationPricer::new(today, spread_scale);
        for r in &fresh {
            match pricer.estimate_tagged(r) {
                Some((mu, sigma, dynamic)) => {
                    let mut model = BayesianWeatherModel::default();
                    model.set_point_forecast(mu, sigma);
                    match market_estimate(&model, &to_sim(r)) {
                        Some(p) => {
                            est.insert(r.market_id.clone(), p);
                            used.insert(r.market_id.clone(), (mu, sigma));
                            if dynamic {
                                dynamic_sigma_ids.insert(r.market_id.clone());
                            }
                        }
                        None => legacy.push(r),
                    }
                }
                None => legacy.push(r),
            }
        }
        println!(
            "{} markets priced from resolution-station nowcast ({} with spread-σ), {} on legacy \
             path",
            est.len(),
            dynamic_sigma_ids.len(),
            legacy.len()
        );

        let sims: Vec<SimulatedMarket> = legacy.iter().map(|r| to_sim(r)).collect();
        // Price legacy markets from the LIVE forecast of their target day (real trading lead),
        // debiased by the fitted per-city summer offset (see LEGACY_SUMMER_BIAS_C).
        let mut forecasts = load_forecasts(&sims);
        for (city, delta) in LEGACY_SUMMER_BIAS_C {
            if let Some(days) = forecasts.get_mut(*city) {
                for v in days.values_mut() {
                    *v += delta;
                }
            }
        }
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

        let mut signals = SignalLogger::new(today);
        for r in fresh {
            let fs = used.get(r.market_id.as_str()).copied();
            let (pm25_max, us_aqi_max) = signals.air_quality(r);
            let (forecast_high_aifs, forecast_high_aigfs) = signals.ai_forecasts(r);
            // From the PRICER's caches, not a second fetch: the spread stored on the row is the
            // spread the row's σ was (or would have been) priced with, and the blend members are
            // the very series its μ was averaged from.
            let ens = pricer.ensemble_stats(&r.city, &r.source, r.target_date);
            let blend_model_highs = pricer.blend_model_highs(&r.city, &r.source, r.target_date);
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
                best_bid: r.best_bid,
                best_ask: r.best_ask,
                volume: r.volume,
                volume_24h: r.volume_24h,
                open_interest: r.open_interest,
                liquidity: r.liquidity,
                pm25_max,
                us_aqi_max,
                forecast_high_aifs,
                forecast_high_aigfs,
                ensemble_spread_ecmwf: ens.spread_ecmwf,
                ensemble_spread_gfs: ens.spread_gfs,
                ensemble_mean_ecmwf: ens.mean_ecmwf,
                ensemble_mean_gfs: ens.mean_gfs,
                blend_model_highs,
                sigma_source: dynamic_sigma_ids
                    .contains(&r.market_id)
                    .then(|| "ensemble".to_string()),
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

/// Log-only extra signals for fresh snapshots (attention / smoke / AI-model fields). One fetch per
/// city (air quality) or per priced coordinate per model (AI forecasts) per run, cached; any
/// failure is just None fields — never blocks the capture. Ensemble spreads used to live here too;
/// they moved into `StationPricer` when the spread→σ mapping entered pricing (2026-08-09) so the
/// logged value and the priced-with value come from one fetch.
struct SignalLogger {
    today: NaiveDate,
    open_meteo: OpenMeteoFetcher,
    aq_cache: HashMap<String, AirQualityByDay>,
    ai_cache: HashMap<String, Vec<HashMap<NaiveDate, f64>>>,
    ai_diag_done: bool,
}

impl SignalLogger {
    fn new(today: NaiveDate) -> Self {
        Self {
            today,
            open_meteo: OpenMeteoFetcher::new(),
            aq_cache: HashMap::new(),
            ai_cache: HashMap::new(),
            ai_diag_done: false,
        }
    }

    /// (pm2.5 max, US AQI max) forecast for the row's target day at the CITY coords — smoke is
    /// regional, so downtown vs airport doesn't matter the way it does for temperature.
    fn air_quality(&mut self, r: &WeatherMarketRow) -> (Option<f64>, Option<f64>) {
        let Some((lat, lon)) = cities::coords(&r.city) else {
            return (None, None);
        };
        if !self.aq_cache.contains_key(&r.city) {
            let got = self.open_meteo.fetch_air_quality_day_max(lat, lon);
            self.aq_cache.insert(r.city.clone(), got);
        }
        self.aq_cache[&r.city]
            .get(&r.target_date)
            .copied()
            .unwrap_or((None, None))
    }

    /// Target-day daily high (°C) per logical model in `AI_LOG_CANDIDATES` (0 = ECMWF AIFS,
    /// 1 = NOAA AIGFS), at the coords the row was priced at: the venue's resolution station when
    /// mapped, else the city registry.
    fn ai_forecasts(&mut self, r: &WeatherMarketRow) -> (Option<f64>, Option<f64>) {
        let coords = station_for(&r.city, &r.source)
            .map(|st| (st.lat, st.lon))
            .or_else(|| cities::coords(&r.city));
        let Some((lat, lon)) = coords else {
            return (None, None);
        };
        let key = format!("{lat:.3},{lon:.3}");
        if !self.ai_cache.contains_key(&key) {
            let maps: Vec<HashMap<NaiveDate, f64>> = AI_LOG_CANDIDATES
                .iter()
                .enumerate()
                .map(|(i, cands)| {
                    let (got, served_by) =
                        self.open_meteo.fetch_forecast_max_live_candidates_tagged(
                            lat,
                            lon,
                            self.today,
                            self.today + Duration::days(15),
                            cands,
                        );
                    // Once per run, say which candidate actually served each logical model — a
                    // rename/delisting upstream otherwise just logs null forever (July 19 lesson).
                    if !self.ai_diag_done {
                        let name = AI_LOG_MODELS.get(i).copied().unwrap_or("?");
                        match &served_by {
                            Some((endpoint, model)) => {
                                println!(
                                    "  AI model {name}: served by /v1/{endpoint} models={model}"
                                )
                            }
                            None => eprintln!(
                                "  AI model {name}: ALL candidates failed — likely renamed or \
                                 delisted upstream; field will be null this run"
                            ),
                        }
                    }
                    got.into_iter().collect()
                })
                .collect();
            self.ai_diag_done = true;
            self.ai_cache.insert(key.clone(), maps);
        }
        let maps = &self.ai_cache[&key];
        (
            maps.first().and_then(|m| m.get(&r.target_date)).copied(),
            maps.get(1).and_then(|m| m.get(&r.target_date)).copied(),
        )
    }
}

/// One spread→σ fit observation from a resolved snapshot, or None where the row is outside the
/// fit population (hygiene lives in `backtesting::spread_sigma::spread_obs`, shared with the
/// pilot's identical fit).
fn snapshot_spread_obs(s: &Snapshot) -> Option<SpreadObs> {
    spread_obs(
        SimulatedMarket {
            date: s.target_date,
            market_id: s.market_id.clone(),
            market_title: s.market_title.clone(),
            market_type: s.market_type.clone(),
            threshold: s.threshold,
            threshold_upper: s.threshold_upper,
            unit: s.unit.clone(),
            market_price: s.entry_price,
            actual_outcome: 0.0, // unused by the fit; the resolved outcome is passed below
            city: s.city.clone(),
        },
        s.captured_at,
        s.best_bid,
        s.best_ask,
        s.model_estimate,
        s.outcome,
        s.forecast_high,
        s.forecast_sigma,
        s.ensemble_spread_ecmwf,
        s.ensemble_spread_gfs,
    )
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_capture_rows_deserialize_with_new_fields_none() {
        // A pre-signal-logging line, verbatim shape from data/captures.jsonl.
        let line = r#"{"captured_at":"2026-07-01","target_date":"2026-07-02","market_id":"m1",
            "market_title":"Highest temperature in NYC on July 2?","market_type":"temp_bucket",
            "threshold":88.0,"threshold_upper":89.0,"unit":"F","city":"NYC","entry_price":0.3,
            "model_estimate":0.25,"outcome":null,"source":"polymarket",
            "forecast_high":31.2,"forecast_sigma":1.1,"best_bid":0.28,"best_ask":0.33,
            "forecast_high_graphcast":null}"#;
        let s: Snapshot = serde_json::from_str(line).expect("old rows must keep parsing");
        assert_eq!(s.volume, None);
        assert_eq!(s.open_interest, None);
        assert_eq!(s.pm25_max, None);
        assert_eq!(s.forecast_high_aifs, None);
        // The retired GraphCast field (never held a value; NCEP killed the model Dec 2025) is
        // simply ignored on read and dropped on the next rewrite.
        assert_eq!(s.forecast_high_aigfs, None);
        assert_eq!(s.ensemble_spread_ecmwf, None);
        assert_eq!(s.ensemble_spread_gfs, None);
        assert_eq!(s.ensemble_mean_ecmwf, None);
        assert_eq!(s.blend_model_highs, None);
        assert_eq!(s.sigma_source, None);

        // And the new fields survive a write→read cycle (outcome-filling rewrites every row).
        let mut s2 = s;
        s2.volume = Some(1234.5);
        s2.us_aqi_max = Some(158.0);
        s2.forecast_high_aigfs = Some(33.1);
        s2.ensemble_spread_ecmwf = Some(1.7);
        s2.ensemble_spread_gfs = Some(2.3);
        s2.ensemble_mean_ecmwf = Some(32.4);
        s2.ensemble_mean_gfs = Some(31.9);
        s2.blend_model_highs = Some(
            [
                ("ecmwf_ifs025".to_string(), 32.1),
                ("gfs_seamless".to_string(), 33.0),
            ]
            .into_iter()
            .collect(),
        );
        s2.sigma_source = Some("ensemble".to_string());
        let round: Snapshot = serde_json::from_str(&serde_json::to_string(&s2).unwrap()).unwrap();
        assert_eq!(round.volume, Some(1234.5));
        assert_eq!(round.us_aqi_max, Some(158.0));
        assert_eq!(round.forecast_high_aigfs, Some(33.1));
        assert_eq!(round.ensemble_spread_ecmwf, Some(1.7));
        assert_eq!(round.ensemble_spread_gfs, Some(2.3));
        assert_eq!(round.ensemble_mean_ecmwf, Some(32.4));
        assert_eq!(round.ensemble_mean_gfs, Some(31.9));
        assert_eq!(round.blend_model_highs, s2.blend_model_highs);
        assert_eq!(round.sigma_source, Some("ensemble".to_string()));

        // BTreeMap keys serialize in sorted order — the daily full-file rewrite must be
        // byte-stable or every committed row churns every day.
        let json = serde_json::to_string(&s2).unwrap();
        assert!(json.find("ecmwf_ifs025").unwrap() < json.find("gfs_seamless").unwrap());
    }

    #[test]
    fn snapshot_spread_obs_mirrors_the_fit_population() {
        let base = Snapshot {
            captured_at: NaiveDate::from_ymd_opt(2026, 8, 1).unwrap(),
            target_date: NaiveDate::from_ymd_opt(2026, 8, 2).unwrap(),
            market_id: "m1".into(),
            market_title: "t".into(),
            market_type: "temp_bucket".into(),
            threshold: 88.0,
            threshold_upper: Some(89.0),
            unit: Some("F".into()),
            city: "NYC".into(),
            entry_price: 0.3,
            model_estimate: Some(0.25),
            outcome: Some(1.0),
            source: "kalshi".into(),
            forecast_high: Some(31.2),
            forecast_sigma: Some(1.1),
            best_bid: Some(0.28),
            best_ask: Some(0.33),
            volume: None,
            volume_24h: None,
            open_interest: None,
            liquidity: None,
            pm25_max: None,
            us_aqi_max: None,
            forecast_high_aifs: None,
            forecast_high_aigfs: None,
            ensemble_spread_ecmwf: Some(1.4),
            ensemble_spread_gfs: None,
            ensemble_mean_ecmwf: None,
            ensemble_mean_gfs: None,
            blend_model_highs: None,
            sigma_source: None,
        };
        let o = snapshot_spread_obs(&base).expect("resolved lead-1 temp row with spread qualifies");
        assert!((o.spread - 1.4).abs() < 1e-12); // GEFS falls back to ECMWF
        assert!((o.outcome - 1.0).abs() < 1e-12);

        let mut lead0 = base.clone();
        lead0.captured_at = lead0.target_date;
        assert!(snapshot_spread_obs(&lead0).is_none());

        let mut unresolved = base.clone();
        unresolved.outcome = None;
        assert!(snapshot_spread_obs(&unresolved).is_none());

        let mut no_spread = base;
        no_spread.ensemble_spread_ecmwf = None;
        assert!(snapshot_spread_obs(&no_spread).is_none());
    }
}
