//! Per-venue edge-shrinkage fit, shared by the dashboard's strategy replay and the live pilot.
//!
//! OLS through the origin of realized `(outcome − price)` on predicted `(model − price)`. λ = 1
//! means the model's disagreements with the market are fully real; λ = ⅓ (roughly what July 2026
//! captures show) means only a third of each claimed edge survives contact with the outcome, so
//! λ·edge is the calibrated bet size for thresholding and Kelly.
//!
//! Data hygiene: callers must only `observe()` captures with lead ≥ 1. Day-of and post-day rows
//! (lead ≤ 0) have prices that already embed the outcome — the market Brier at lead −1 is ~0.0005 —
//! and including them would bias λ toward 1.

use std::collections::HashMap;

#[derive(Default)]
pub struct ShrinkageFit {
    /// venue → (Σx², Σxy, n) running sums for the through-origin slope Σxy/Σx².
    by_venue: HashMap<String, (f64, f64, usize)>,
}

impl ShrinkageFit {
    /// Minimum resolved rows before a fitted λ is trusted over the fallback chain
    /// (venue → pooled → 1.0). Below this the slope is mostly noise.
    pub const MIN_N: usize = 40;

    pub fn observe(&mut self, venue: &str, predicted: f64, realized: f64) {
        let e = self.by_venue.entry(venue.to_string()).or_default();
        e.0 += predicted * predicted;
        e.1 += predicted * realized;
        e.2 += 1;
    }

    fn slope(sums: &(f64, f64, usize)) -> Option<f64> {
        (sums.2 >= Self::MIN_N && sums.0 > 0.0).then(|| (sums.1 / sums.0).clamp(0.0, 1.0))
    }

    fn pooled(&self) -> Option<f64> {
        let sums = self
            .by_venue
            .values()
            .fold((0.0, 0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
        Self::slope(&sums)
    }

    /// λ to apply to a venue's edges: per-venue when it has enough resolved rows, else pooled
    /// across venues, else 1.0 (no shrink) while the sample is too thin to fit. Clamped to [0, 1]:
    /// a negative slope means the model's disagreement is anti-signal (shrink to zero, which stops
    /// trading), and slopes above 1 are never amplified.
    pub fn lambda(&self, venue: &str) -> f64 {
        self.by_venue
            .get(venue)
            .and_then(Self::slope)
            .or_else(|| self.pooled())
            .unwrap_or(1.0)
    }

    /// (venue, raw unclamped slope, n) per venue, for diagnostics tables.
    pub fn rows(&self) -> Vec<(String, f64, usize)> {
        let mut out: Vec<(String, f64, usize)> = self
            .by_venue
            .iter()
            .map(|(v, &(xx, xy, n))| (v.clone(), if xx > 0.0 { xy / xx } else { 0.0 }, n))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }
}
