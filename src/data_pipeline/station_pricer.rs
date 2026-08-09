//! Station-aware (mu, sigma) pricing for active temperature markets — extracted from the capture
//! daemon so the live pilot prices markets EXACTLY the way the calibration evidence was accrued.

use std::collections::HashMap;

use chrono::{Duration, NaiveDate, NaiveDateTime, Utc};

use crate::api::WeatherMarketRow;
use crate::backtesting::spread_sigma::{dynamic_sigma, mean_spread};
use crate::cities;
use crate::data_pipeline::open_meteo_fetcher::{ENSEMBLE_LOG_CANDIDATES, ENSEMBLE_LOG_MODELS};
use crate::data_pipeline::station_obs::{
    blend_forecast_day_max_c, nowcast_mu_sigma, phase_for, wu_running_max_c, IemObsFetcher, Phase,
};
use crate::data_pipeline::OpenMeteoFetcher;
use crate::stations::{station_for, Station};

/// Prices a market against its venue's resolution station: METAR obs for elapsed hours (the very
/// feed Polymarket resolves on; Kalshi's CLI gap is a fitted post/bias term), the
/// station-coordinate hourly forecast for the rest, and the per-(city, lead) fitted sigma/bias
/// from `stations.rs` — Polymarket and Kalshi have separate verified tables (different stations
/// AND truth variables). Returns the (mu, sigma) in °C for `set_point_forecast`, or None to fall
/// back to the legacy path. One obs fetch and one forecast fetch per STATION per run, cached
/// (the same city can map to two stations across venues).
///
/// Since 2026-08-09 the constructor takes the walk-forward spread→σ scale
/// (`backtesting::spread_sigma`): when set, lead ≥ 1 rows whose target day has an ensemble
/// member spread price at σ = max(a·spread, 0.2 °C) instead of the per-(city, lead) constant.
/// The parameter is REQUIRED (not a builder default) so the daemon and the pilot cannot silently
/// diverge — every construction site states what σ regime it prices under.
pub struct StationPricer {
    now_utc: NaiveDateTime,
    today: NaiveDate,
    obs_fetcher: IemObsFetcher,
    open_meteo: OpenMeteoFetcher,
    obs_cache: HashMap<String, Vec<(NaiveDateTime, f64)>>,
    forecast_cache: HashMap<String, Vec<Vec<(NaiveDateTime, f64)>>>,
    /// Fitted spread→σ scale `a`; None ⇒ constant tables only (pre-gate behavior).
    spread_sigma_scale: Option<f64>,
    ens_cache: HashMap<String, Vec<HashMap<NaiveDate, f64>>>,
    ens_diag_done: bool,
    /// Per-slot circuit breaker: a slot whose FIRST coordinate came back all-candidates-failed is
    /// skipped for every later coordinate this run. Without it, an unreachable-but-hanging
    /// ensemble host (a separate host nothing else uses) costs its full candidate matrix × the
    /// 30 s client timeout at EVERY station.
    ens_slot_dead: Vec<bool>,
}

impl StationPricer {
    pub fn new(today: NaiveDate, spread_sigma_scale: Option<f64>) -> Self {
        Self {
            now_utc: Utc::now().naive_utc(),
            today,
            obs_fetcher: IemObsFetcher::new(),
            open_meteo: OpenMeteoFetcher::new(),
            obs_cache: HashMap::new(),
            forecast_cache: HashMap::new(),
            spread_sigma_scale,
            ens_cache: HashMap::new(),
            ens_diag_done: false,
            ens_slot_dead: vec![false; ENSEMBLE_LOG_CANDIDATES.len()],
        }
    }

    pub fn estimate(&mut self, r: &WeatherMarketRow) -> Option<(f64, f64)> {
        self.estimate_tagged(r).map(|(mu, sigma, _)| (mu, sigma))
    }

    /// Like `estimate`, plus whether σ came from the fitted spread→σ mapping rather than the
    /// per-(city, lead) constant table — the capture daemon stores the flag (`sigma_source`) so
    /// diagnostics and refits can split the two σ regimes instead of pooling them unknowingly.
    pub fn estimate_tagged(&mut self, r: &WeatherMarketRow) -> Option<(f64, f64, bool)> {
        if !r.market_type.starts_with("temp") {
            return None; // precip has no station model
        }
        let st = station_for(&r.city, &r.source)?;
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
                let rest =
                    blend_forecast_day_max_c(self.forecast(st)?, r.target_date, st, Some(cutoff));
                (run, rest)
            }
            Phase::Lead(_) => (
                None,
                blend_forecast_day_max_c(self.forecast(st)?, r.target_date, st, None),
            ),
        };
        let (mu, sigma) = nowcast_mu_sigma(st, phase, runmax, rest)?;
        // Dynamic σ on pure-forecast rows only: day-of/post σ describe obs-vs-truth noise, not
        // forecast uncertainty, and lead ≥ 1 rows are what the walk-forward validation scored.
        // μ (incl. the fitted bias) is untouched — the mapping was fitted against stored μ.
        if let (Phase::Lead(_), Some(a)) = (phase, self.spread_sigma_scale) {
            let (e, g) = self.ensemble_spreads(&r.city, &r.source, r.target_date);
            if let Some(sp) = mean_spread(e, g) {
                return Some((mu, dynamic_sigma(a, sp), true));
            }
        }
        Some((mu, sigma, false))
    }

    /// Target-day ensemble member spread (°C) per system in `ENSEMBLE_LOG_MODELS` (0 = ECMWF ENS,
    /// 1 = GEFS), at the coords the row is priced at — the venue's resolution station when
    /// mapped, else the city registry. One fetch per coordinate per run, cached; the window is a
    /// week out (leads beyond 2 are neither traded nor fitted, and ~50-member hourly payloads are
    /// big enough without tail days nobody reads). This is ALSO the capture daemon's log source
    /// for `ensemble_spread_*`, so the spread a row is priced with is the spread stored on it.
    pub fn ensemble_spreads(
        &mut self,
        city: &str,
        source: &str,
        target: NaiveDate,
    ) -> (Option<f64>, Option<f64>) {
        let coords = station_for(city, source)
            .map(|st| (st.lat, st.lon))
            .or_else(|| cities::coords(city));
        let Some((lat, lon)) = coords else {
            return (None, None);
        };
        let key = format!("{lat:.3},{lon:.3}");
        if !self.ens_cache.contains_key(&key) {
            let mut maps: Vec<HashMap<NaiveDate, f64>> = Vec::new();
            for (i, cands) in ENSEMBLE_LOG_CANDIDATES.iter().enumerate() {
                if self.ens_slot_dead[i] {
                    maps.push(HashMap::new());
                    continue;
                }
                let (got, served_by) = self.open_meteo.fetch_ensemble_spread_tagged(
                    lat,
                    lon,
                    self.today,
                    self.today + Duration::days(7),
                    cands,
                );
                if served_by.is_none() {
                    self.ens_slot_dead[i] = true;
                }
                // Same per-run visibility as the AI models: a renamed/delisted ensemble
                // otherwise logs null forever (July 19 lesson).
                if !self.ens_diag_done {
                    let name = ENSEMBLE_LOG_MODELS.get(i).copied().unwrap_or("?");
                    match &served_by {
                        Some(model) => println!(
                            "  ensemble {name}: served by models={model} ({} dates)",
                            got.len()
                        ),
                        None => eprintln!(
                            "  ensemble {name}: ALL candidates failed — host down, or \
                             renamed/delisted upstream; skipping this slot for the rest of \
                             the run (field will be null)"
                        ),
                    }
                }
                maps.push(got);
            }
            self.ens_diag_done = true;
            self.ens_cache.insert(key.clone(), maps);
        }
        let maps = &self.ens_cache[&key];
        (
            maps.first().and_then(|m| m.get(&target)).copied(),
            maps.get(1).and_then(|m| m.get(&target)).copied(),
        )
    }

    fn obs(&mut self, st: &Station, target: NaiveDate) -> Option<&[(NaiveDateTime, f64)]> {
        if !self.obs_cache.contains_key(st.iem_id) {
            let start = (target - Duration::days(1)).min(self.today - Duration::days(1));
            let got = self.obs_fetcher.fetch_tmpf_utc(st, start, self.today);
            self.obs_cache.insert(st.iem_id.to_string(), got);
        }
        let v = self.obs_cache.get(st.iem_id).unwrap();
        (!v.is_empty()).then_some(v.as_slice())
    }

    fn forecast(&mut self, st: &Station) -> Option<&[Vec<(NaiveDateTime, f64)>]> {
        if !self.forecast_cache.contains_key(st.iem_id) {
            // 16-day horizon; a target beyond it simply yields no hours -> legacy fallback.
            let got = self.open_meteo.fetch_forecast_hourly_models_utc(
                st.lat,
                st.lon,
                self.today - Duration::days(1),
                self.today + Duration::days(15),
            );
            self.forecast_cache.insert(st.iem_id.to_string(), got);
        }
        let v = self.forecast_cache.get(st.iem_id).unwrap();
        (!v.is_empty()).then_some(v.as_slice())
    }
}
