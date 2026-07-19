//! Station-aware (mu, sigma) pricing for active temperature markets — extracted from the capture
//! daemon so the live pilot prices markets EXACTLY the way the calibration evidence was accrued.

use std::collections::HashMap;

use chrono::{Duration, NaiveDate, NaiveDateTime, Utc};

use crate::api::WeatherMarketRow;
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
pub struct StationPricer {
    now_utc: NaiveDateTime,
    today: NaiveDate,
    obs_fetcher: IemObsFetcher,
    open_meteo: OpenMeteoFetcher,
    obs_cache: HashMap<String, Vec<(NaiveDateTime, f64)>>,
    forecast_cache: HashMap<String, Vec<Vec<(NaiveDateTime, f64)>>>,
}

impl StationPricer {
    pub fn new(today: NaiveDate) -> Self {
        Self {
            now_utc: Utc::now().naive_utc(),
            today,
            obs_fetcher: IemObsFetcher::new(),
            open_meteo: OpenMeteoFetcher::new(),
            obs_cache: HashMap::new(),
            forecast_cache: HashMap::new(),
        }
    }

    pub fn estimate(&mut self, r: &WeatherMarketRow) -> Option<(f64, f64)> {
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
        nowcast_mu_sigma(st, phase, runmax, rest)
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
