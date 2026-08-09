//! Walk-forward ensemble-spread → σ mapping — the first DAY-SPECIFIC uncertainty in pricing.
//!
//! The pricing σ has always been a per-(city, lead) CONSTANT, so it prices a locked-in ridge day
//! and a frontal coin-flip day identically. Ensemble member spread is the standard signal for that
//! difference, logged per capture since 2026-07-23 (`ensemble_spread_ecmwf`/`_gfs`) precisely so
//! this mapping could be fitted and judged before touching pricing. The promotion gate — beat the
//! constant-σ tables walk-forward on accrued captures — was met at the 2026-08-09 check
//! (`scripts/lambda_diagnostics.py`, 1108 rows carrying spread): σ = a·spread scored bucket Brier
//! 0.0947 vs the constant tables' 0.0971 on the same 1038 walk-forward rows (λ +0.347 vs the
//! 0.328 pooled fit). The spread terciles show why: the constant σ is fine on calm days but the
//! high-spread tercile is where it loses (Brier 0.1041), exactly the days spread flags.
//!
//! The mapping mirrors the validated script EXACTLY: spread is the ECMWF-ENS/GEFS mean (GEFS
//! falls back to ECMWF; no ECMWF ⇒ no mapping), σ = max(a · spread, 0.2 °C), and `a` is a coarse
//! grid search (0.4..=3.0 step 0.1) minimizing the Brier of re-priced shapes over RESOLVED
//! lead ≥ 1 temperature captures — the λ fit's population hygiene plus stored forecast μ/σ and
//! spread. Refitting each run on resolved history and applying only to markets still being priced
//! is walk-forward by construction: everything resolved is strictly earlier than any open market.
//!
//! Deliberately NOT a per-(city, lead) fit: one pooled scalar is what the validation validated,
//! and slicing it thinner re-opens the overconfidence door the σ-floor and MIN_N gate close.

use chrono::NaiveDate;

use crate::models::BayesianWeatherModel;
use crate::types::SimulatedMarket;

use super::market_estimate;

/// σ floor (°C): a near-zero member spread means "locked in", but the resolution variable still
/// carries whole-degree rounding and ob-vs-CLI noise — never price sharper than this. Matches the
/// validated script's `max(a * sp, 0.2)`.
pub const SPREAD_SIGMA_FLOOR_C: f64 = 0.2;

/// Minimum resolved observations before the mapping is used at all, matching the script's
/// walk-forward gate (and `ShrinkageFit::MIN_N`). Under it, pricing keeps the constant tables.
pub const SPREAD_SIGMA_MIN_N: usize = 40;

/// One resolved capture the scale fit can learn from.
pub struct SpreadObs {
    /// The market shape re-priced during the grid search (its `actual_outcome` field is unused —
    /// the resolved outcome lives on this struct so the two can't drift).
    pub market: SimulatedMarket,
    /// Stored μ (°C) the row was priced at — post-bias, exactly what a dynamic σ pairs with.
    pub forecast_high: f64,
    /// Mean ensemble member spread (°C) logged on the row.
    pub spread: f64,
    /// Resolved outcome, 0.0 or 1.0.
    pub outcome: f64,
}

/// The script's `sp`: mean of the ECMWF-ENS and GEFS member spreads, GEFS falling back to ECMWF
/// when absent. No ECMWF ⇒ None — GEFS-only rows were never part of the validated sample.
pub fn mean_spread(ecmwf: Option<f64>, gfs: Option<f64>) -> Option<f64> {
    let e = ecmwf?;
    Some((e + gfs.unwrap_or(e)) / 2.0)
}

/// σ (°C) under the fitted mapping.
pub fn dynamic_sigma(scale: f64, spread: f64) -> f64 {
    (scale * spread).max(SPREAD_SIGMA_FLOOR_C)
}

/// Build one fit observation from a capture row's fields, applying the SAME row hygiene as the
/// validated script (its `usable` filter plus the ensemble section's extras): resolved, priced
/// (model estimate present), temperature shape, lead ≥ 1, usable reference price (book mid when
/// sane, else last trade), priced from a stored forecast (μ present, σ > 0), ECMWF spread
/// present. None for any row outside that population. Callers hand in whatever their capture-row
/// struct holds — the hygiene lives here once so the daemon and the pilot cannot drift.
#[allow(clippy::too_many_arguments)] // one arg per capture field; a builder would just rename them
pub fn spread_obs(
    market: SimulatedMarket,
    captured_at: NaiveDate,
    best_bid: Option<f64>,
    best_ask: Option<f64>,
    model_estimate: Option<f64>,
    outcome: Option<f64>,
    forecast_high: Option<f64>,
    forecast_sigma: Option<f64>,
    spread_ecmwf: Option<f64>,
    spread_gfs: Option<f64>,
) -> Option<SpreadObs> {
    let outcome = outcome?;
    model_estimate?;
    if !market.market_type.starts_with("temp") {
        return None;
    }
    if (market.date - captured_at).num_days() < 1 {
        return None; // lead ≤ 0 rows were priced with intraday information; not this regime
    }
    let px = match (best_bid, best_ask) {
        (Some(b), Some(a)) if b > 0.0 && a < 1.0 && b <= a => (a + b) / 2.0,
        _ => market.market_price,
    };
    if !(px > 0.0 && px < 1.0) {
        return None;
    }
    let forecast_high = forecast_high?;
    forecast_sigma.filter(|s| *s > 0.0)?;
    let spread = mean_spread(spread_ecmwf, spread_gfs)?;
    Some(SpreadObs {
        market,
        forecast_high,
        spread,
        outcome,
    })
}

/// Grid-fit the scale `a` minimizing Σ(reprice − outcome)² — the script's coarse search,
/// reproduced with the PRODUCTION pricing path (`set_point_forecast` + `market_estimate`) rather
/// than a re-implementation, so fit-time and apply-time pricing are the same code. Ascending grid
/// with strict-improvement keeps ties at the smallest `a`, like the script. None under
/// `SPREAD_SIGMA_MIN_N` rows: keep the constant tables.
pub fn fit_spread_sigma_scale(obs: &[SpreadObs]) -> Option<f64> {
    if obs.len() < SPREAD_SIGMA_MIN_N {
        return None;
    }
    let mut best: Option<(f64, f64)> = None;
    for a10 in 4..=30u32 {
        let a = f64::from(a10) / 10.0;
        let mut v = 0.0;
        for o in obs {
            if let Some(p) = reprice(&o.market, o.forecast_high, dynamic_sigma(a, o.spread)) {
                v += (p - o.outcome).powi(2);
            }
        }
        if best.is_none_or(|(_, bv)| v < bv) {
            best = Some((a, v));
        }
    }
    best.map(|(a, _)| a)
}

fn reprice(market: &SimulatedMarket, mu_c: f64, sigma_c: f64) -> Option<f64> {
    let mut model = BayesianWeatherModel::default();
    model.set_point_forecast(mu_c, sigma_c);
    market_estimate(&model, market)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn d(s: &str) -> NaiveDate {
        NaiveDate::parse_from_str(s, "%Y-%m-%d").unwrap()
    }

    fn market(mt: &str, threshold_c: f64) -> SimulatedMarket {
        SimulatedMarket {
            date: d("2026-08-02"),
            market_id: "m".into(),
            market_title: String::new(),
            market_type: mt.into(),
            threshold: threshold_c,
            threshold_upper: None,
            unit: Some("C".into()),
            market_price: 0.5,
            actual_outcome: 0.0,
            city: "NYC".into(),
        }
    }

    fn obs(mt: &str, threshold_c: f64, mu: f64, spread: f64, outcome: f64) -> SpreadObs {
        SpreadObs {
            market: market(mt, threshold_c),
            forecast_high: mu,
            spread,
            outcome,
        }
    }

    #[test]
    fn mean_spread_requires_ecmwf_and_averages_with_gfs_fallback() {
        assert_eq!(mean_spread(Some(1.0), Some(2.0)), Some(1.5));
        assert_eq!(mean_spread(Some(1.0), None), Some(1.0)); // GEFS falls back to ECMWF
        assert_eq!(mean_spread(None, Some(2.0)), None); // GEFS alone was never validated
        assert_eq!(mean_spread(None, None), None);
    }

    #[test]
    fn dynamic_sigma_is_floored() {
        assert!((dynamic_sigma(1.5, 2.0) - 3.0).abs() < 1e-12);
        assert!((dynamic_sigma(1.5, 0.01) - SPREAD_SIGMA_FLOOR_C).abs() < 1e-12);
    }

    #[test]
    fn spread_obs_applies_the_fit_population_hygiene() {
        let ok = || {
            (
                market("temp_at_least", 30.0),
                d("2026-08-01"), // lead 1
                Some(0.4),
                Some(0.5),
                Some(0.45),
                Some(1.0),
                Some(31.0),
                Some(1.5),
                Some(1.0),
                Some(1.2),
            )
        };
        let build = |m, cap, b, a, est, out, fh, fs, se, sg| {
            spread_obs(m, cap, b, a, est, out, fh, fs, se, sg)
        };
        let (m, cap, b, a, est, out, fh, fs, se, sg) = ok();
        assert!(build(m, cap, b, a, est, out, fh, fs, se, sg).is_some());

        // Unresolved, lead-0, precip, σ=0, and no-ECMWF-spread rows are all outside the population.
        let (m, cap, b, a, est, _, fh, fs, se, sg) = ok();
        assert!(build(m, cap, b, a, est, None, fh, fs, se, sg).is_none());
        let (m, _, b, a, est, out, fh, fs, se, sg) = ok();
        assert!(build(m, d("2026-08-02"), b, a, est, out, fh, fs, se, sg).is_none());
        let (_, cap, b, a, est, out, fh, fs, se, sg) = ok();
        assert!(build(
            market("precipitation", 0.0),
            cap,
            b,
            a,
            est,
            out,
            fh,
            fs,
            se,
            sg
        )
        .is_none());
        let (m, cap, b, a, est, out, fh, _, se, sg) = ok();
        assert!(build(m, cap, b, a, est, out, fh, Some(0.0), se, sg).is_none());
        let (m, cap, b, a, est, out, fh, fs, _, sg) = ok();
        assert!(build(m, cap, b, a, est, out, fh, fs, None, sg).is_none());

        // A junk book (bid > ask) falls back to the last trade, which here is usable.
        let (m, cap, _, _, est, out, fh, fs, se, sg) = ok();
        assert!(build(m, cap, Some(0.9), Some(0.1), est, out, fh, fs, se, sg).is_some());
    }

    #[test]
    fn fit_needs_min_n_rows() {
        let rows: Vec<SpreadObs> = (0..SPREAD_SIGMA_MIN_N - 1)
            .map(|_| obs("temp_at_least", 35.0, 30.0, 1.0, 0.0))
            .collect();
        assert_eq!(fit_spread_sigma_scale(&rows), None);
    }

    #[test]
    fn fit_recovers_the_brier_optimal_grid_end_in_both_directions() {
        // Outcomes agree with the point forecast (threshold far above μ, never hit): every
        // sharpening lowers Brier, so the grid's smallest scale wins.
        let sharp: Vec<SpreadObs> = (0..SPREAD_SIGMA_MIN_N)
            .map(|_| obs("temp_at_least", 40.0, 30.0, 2.0, 0.0))
            .collect();
        assert_eq!(fit_spread_sigma_scale(&sharp), Some(0.4));

        // Outcomes contradict the point forecast (same far threshold, but it HIT): the widest σ
        // concedes the most probability to the tail, so the grid's largest scale wins.
        let wide: Vec<SpreadObs> = (0..SPREAD_SIGMA_MIN_N)
            .map(|_| obs("temp_at_least", 40.0, 30.0, 2.0, 1.0))
            .collect();
        assert_eq!(fit_spread_sigma_scale(&wide), Some(3.0));
    }
}
