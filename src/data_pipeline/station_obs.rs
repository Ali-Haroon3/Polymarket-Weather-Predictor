//! METAR observations for resolution stations (IEM ASOS archive) + the pure nowcast math that
//! turns obs and forecasts into a (mu, sigma) for `set_point_forecast`.
//!
//! The resolution variable is Weather Underground's daily max: every METAR ob rounded to the
//! station's display unit, maxed over the LOCAL calendar day. `wu_running_max_c` reproduces that
//! from raw tmpf obs (verified 43/43 against settled Polymarket outcomes). All timestamps here are
//! UTC (obs are requested with tz=Etc/UTC); local-day windows come from the station's standard
//! UTC offset — DST shifts the window edges by 1h, which can only move midnight-adjacent obs in or
//! out, never the day's max.

use chrono::{Duration, NaiveDate, NaiveDateTime};

use crate::stations::{Station, FAR_SIGMA};

/// Where the capture instant falls relative to the market's target LOCAL day.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Local day fully elapsed — the WU max is known from obs.
    Post,
    /// Capture is inside the target local day — fuse running obs max with the remaining forecast.
    DayOf,
    /// Pure forecast, this many local days ahead (≥1).
    Lead(u32),
}

pub fn phase_for(now_utc: NaiveDateTime, target: NaiveDate, station: &Station) -> Phase {
    let local_date = (now_utc + Duration::hours(station.utc_offset_hours as i64)).date();
    if local_date > target {
        Phase::Post
    } else if local_date == target {
        Phase::DayOf
    } else {
        Phase::Lead((target - local_date).num_days() as u32)
    }
}

/// UTC window [start, end) of the station's local calendar day.
pub fn local_day_utc(target: NaiveDate, station: &Station) -> (NaiveDateTime, NaiveDateTime) {
    let start =
        target.and_hms_opt(0, 0, 0).unwrap() - Duration::hours(station.utc_offset_hours as i64);
    (start, start + Duration::hours(24))
}

/// One METAR ob rounded to the station's display unit, returned in °C (the model's unit).
fn wu_ob_c(tmpf: f64, station: &Station) -> f64 {
    if station.celsius {
        ((tmpf - 32.0) / 1.8).round()
    } else {
        (tmpf.round() - 32.0) / 1.8
    }
}

/// WU-style max (°C) over the target local day, from obs strictly before `cutoff` (None = all).
pub fn wu_running_max_c(
    obs_utc: &[(NaiveDateTime, f64)],
    target: NaiveDate,
    station: &Station,
    cutoff_utc: Option<NaiveDateTime>,
) -> Option<f64> {
    let (start, end) = local_day_utc(target, station);
    obs_utc
        .iter()
        .filter(|(t, _)| *t >= start && *t < end && cutoff_utc.map(|c| *t < c).unwrap_or(true))
        .map(|(_, f)| wu_ob_c(*f, station))
        .fold(None, |acc: Option<f64>, v| {
            Some(acc.map_or(v, |a| a.max(v)))
        })
}

/// Max (°C) of hourly forecast points falling in the target local day, at or after `from` (None =
/// whole day). `hourly_utc` is (UTC timestamp, °C).
pub fn forecast_day_max_c(
    hourly_utc: &[(NaiveDateTime, f64)],
    target: NaiveDate,
    station: &Station,
    from_utc: Option<NaiveDateTime>,
) -> Option<f64> {
    let (start, end) = local_day_utc(target, station);
    let lo = from_utc.map_or(start, |f| f.max(start));
    hourly_utc
        .iter()
        .filter(|(t, _)| *t >= lo && *t < end)
        .map(|(_, v)| *v)
        .fold(None, |acc: Option<f64>, v| {
            Some(acc.map_or(v, |a| a.max(v)))
        })
}

/// Assemble the phase-aware (mu, sigma) in °C. `runmax_c` = obs so far (Post/DayOf), `rest_c` =
/// forecast max of the not-yet-elapsed part (DayOf) or of the whole day (Lead). None when the
/// needed inputs are missing — callers fall back to the legacy pricing path.
pub fn nowcast_mu_sigma(
    station: &Station,
    phase: Phase,
    runmax_c: Option<f64>,
    rest_c: Option<f64>,
) -> Option<(f64, f64)> {
    match phase {
        // Post-phase truth differs by venue: WU (Polymarket) IS the ob-max (bias 0, tiny sigma);
        // Kalshi's CLI reads the continuous sensor, so its fitted ob-max gap applies here.
        Phase::Post => runmax_c.map(|m| (m + station.post_bias, station.post_sigma)),
        Phase::DayOf => {
            let mu = match (runmax_c, rest_c) {
                (Some(a), Some(b)) => a.max(b),
                (a, b) => a.or(b)?,
            };
            Some((mu + station.bias[0], station.sigma[0]))
        }
        Phase::Lead(k) => {
            let mu = rest_c?;
            let i = (k as usize).min(3);
            if i >= 3 {
                Some((mu, FAR_SIGMA))
            } else {
                Some((mu + station.bias[i], station.sigma[i]))
            }
        }
    }
}

/// Raw METAR temperatures (UTC timestamp, tmpf) from the IEM ASOS archive, `start`..=`end` UTC
/// dates. Empty on any failure (degradation by design, like every fetcher).
pub struct IemObsFetcher {
    client: reqwest::blocking::Client,
}

impl IemObsFetcher {
    pub fn new() -> Self {
        Self {
            client: crate::data_pipeline::build_client(30),
        }
    }

    /// One `obhistory.json` call per UTC date — unlike the bulk asos.py archive (which lags a day
    /// or more at some stations), this endpoint is near-real-time, which the day-of running max
    /// depends on. A failed date contributes nothing.
    pub fn fetch_tmpf_utc(
        &self,
        station: &Station,
        start: NaiveDate,
        end: NaiveDate,
    ) -> Vec<(NaiveDateTime, f64)> {
        let mut out = Vec::new();
        let mut d = start;
        while d <= end {
            let query = [
                ("station", station.iem_id.to_string()),
                ("network", station.iem_network.to_string()),
                ("date", d.to_string()),
            ];
            if let Ok(resp) = self
                .client
                .get("https://mesonet.agron.iastate.edu/api/1/obhistory.json")
                .query(&query)
                .send()
            {
                if let Ok(json) = resp.json::<serde_json::Value>() {
                    for row in json
                        .get("data")
                        .and_then(|v| v.as_array())
                        .unwrap_or(&Vec::new())
                    {
                        let Some(ts) = row.get("utc_valid").and_then(|v| v.as_str()) else {
                            continue;
                        };
                        let Ok(t) = NaiveDateTime::parse_from_str(ts, "%Y-%m-%dT%H:%MZ")
                            .or_else(|_| NaiveDateTime::parse_from_str(ts, "%Y-%m-%dT%H:%M:%SZ"))
                        else {
                            continue;
                        };
                        if let Some(f) = row.get("tmpf").and_then(|v| v.as_f64()) {
                            out.push((t, f));
                        }
                    }
                }
            }
            d += Duration::days(1);
        }
        out
    }
}

impl Default for IemObsFetcher {
    fn default() -> Self {
        Self::new()
    }
}
