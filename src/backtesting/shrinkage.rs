//! Per-venue edge-shrinkage fit, shared by the dashboard's strategy replay and the live pilot.
//!
//! OLS through the origin of realized `(outcome − price)` on predicted `(model − price)`. λ = 1
//! means the model's disagreements with the market are fully real; λ = ⅓ (roughly what July 2026
//! captures show) means only a third of each claimed edge survives contact with the outcome, so
//! λ·edge is the calibrated bet size for thresholding and Kelly.
//!
//! Observations may optionally carry a SEGMENT tag (`observe_seg`) so λ can be conditioned on
//! (venue, segment) instead of venue alone. Motivation (`scripts/lambda_diagnostics.py`,
//! 2026-07-26): within-venue λ spread dwarfs the between-venue spread — sub-10¢ markets carry
//! negative λ on BOTH sides (BUY −0.086, SELL −0.105) while everything ≥ 10¢ is healthy (BUY
//! +0.363, SELL +0.540), yet the per-venue fit averages those segments together. A per-segment λ
//! zeroes out anti-signal segments by construction (negative slopes clamp to 0, which stops
//! trading them) instead of needing a hand-picked filter. Venue-level lookups fold across all of
//! the venue's segments, so `lambda()` behaves identically whether or not observations were
//! tagged.
//!
//! Data hygiene: callers must only `observe()` captures with lead ≥ 1. Day-of and post-day rows
//! (lead ≤ 0) have prices that already embed the outcome — the market Brier at lead −1 is ~0.0005 —
//! and including them would bias λ toward 1.

use std::collections::BTreeMap;

/// The λ segment a price falls in. One boundary, at 10¢: `scripts/lambda_diagnostics.py`
/// (2026-07-26) found λ negative on BOTH sides below it and healthy on both sides above it, and
/// the walk-forward prototype showed the single-boundary scheme is where nearly all the gain is
/// (OOS calibration slope 0.674 → 0.701; finer bands add thin-segment fallback churn for slope
/// noise). Lives here — not in a bin — so the dashboard and the pilot can never drift on the
/// boundary. Fit observations are tagged by the ref (mid) price; trade decisions look up by the
/// side's fill price — rows straddling the boundary land differently in rare cases, accepted.
pub fn lambda_segment(px: f64) -> &'static str {
    if px < 0.10 {
        "px<0.10"
    } else {
        "px≥0.10"
    }
}

#[derive(Default)]
pub struct ShrinkageFit {
    /// (venue, segment) → (Σx², Σxy, n) running sums for the through-origin slope Σxy/Σx².
    /// Untagged observations land under segment `""`. BTreeMap, not HashMap: folds sum floats in
    /// key order, so every fold is deterministic run to run (determinism is load-bearing here).
    by_key: BTreeMap<(String, String), (f64, f64, usize)>,
}

impl ShrinkageFit {
    /// Minimum resolved rows before a fitted λ is trusted over the fallback chain
    /// (segment → venue → pooled → 1.0). Below this the slope is mostly noise.
    pub const MIN_N: usize = 40;

    pub fn observe(&mut self, venue: &str, predicted: f64, realized: f64) {
        self.observe_seg(venue, "", predicted, realized);
    }

    /// Observe with a segment tag (e.g. a price band). The segment key is caller-defined; venue
    /// -level lookups fold across segments, so tagging never changes `lambda()`.
    pub fn observe_seg(&mut self, venue: &str, segment: &str, predicted: f64, realized: f64) {
        let e = self
            .by_key
            .entry((venue.to_string(), segment.to_string()))
            .or_default();
        e.0 += predicted * predicted;
        e.1 += predicted * realized;
        e.2 += 1;
    }

    fn slope(sums: &(f64, f64, usize)) -> Option<f64> {
        (sums.2 >= Self::MIN_N && sums.0 > 0.0).then(|| (sums.1 / sums.0).clamp(0.0, 1.0))
    }

    /// Fold the running sums over every key matching `pred`.
    fn fold(&self, pred: impl Fn(&(String, String)) -> bool) -> (f64, f64, usize) {
        self.by_key
            .iter()
            .filter(|(k, _)| pred(k))
            .fold((0.0, 0.0, 0), |a, (_, b)| (a.0 + b.0, a.1 + b.1, a.2 + b.2))
    }

    fn pooled(&self) -> Option<f64> {
        Self::slope(&self.fold(|_| true))
    }

    /// λ to apply to a venue's edges: per-venue when it has enough resolved rows, else pooled
    /// across venues, else 1.0 (no shrink) while the sample is too thin to fit. Clamped to [0, 1]:
    /// a negative slope means the model's disagreement is anti-signal (shrink to zero, which stops
    /// trading), and slopes above 1 are never amplified.
    pub fn lambda(&self, venue: &str) -> f64 {
        Self::slope(&self.fold(|k| k.0 == venue))
            .or_else(|| self.pooled())
            .unwrap_or(1.0)
    }

    /// λ for a (venue, segment): the segment's own fit when it has ≥ `MIN_N` rows, else the
    /// venue fold, else pooled, else 1.0. Same clamping as `lambda`. An empty segment means "no
    /// segment" and resolves to the venue fold — otherwise a fit built via untagged `observe()`
    /// would answer `lambda_seg(venue, "")` from the `""` bucket alone, silently excluding any
    /// tagged rows.
    pub fn lambda_seg(&self, venue: &str, segment: &str) -> f64 {
        if segment.is_empty() {
            return self.lambda(venue);
        }
        self.by_key
            .get(&(venue.to_string(), segment.to_string()))
            .and_then(Self::slope)
            .unwrap_or_else(|| self.lambda(venue))
    }

    /// Resolved rows behind one (venue, segment) key — the COVERAGE question, separate from the
    /// slope question `lambda_seg` answers. `lambda_seg` deliberately falls back to the venue fold
    /// under `MIN_N`, so a key with no fit of its own still returns a plausible λ; a caller that
    /// must distinguish "fitted and healthy" from "never fitted" has to ask this first.
    pub fn n_seg(&self, venue: &str, segment: &str) -> usize {
        self.by_key
            .get(&(venue.to_string(), segment.to_string()))
            .map_or(0, |s| s.2)
    }

    /// (venue, raw unclamped slope, n) per venue, for diagnostics tables. Folds across segments.
    pub fn rows(&self) -> Vec<(String, f64, usize)> {
        let mut venues: Vec<String> = self.by_key.keys().map(|k| k.0.clone()).collect();
        venues.sort();
        venues.dedup();
        venues
            .into_iter()
            .map(|v| {
                let (xx, xy, n) = self.fold(|k| k.0 == v);
                (v, if xx > 0.0 { xy / xx } else { 0.0 }, n)
            })
            .collect()
    }

    /// (venue, segment, raw unclamped slope, n) per tagged segment, for diagnostics tables.
    /// Untagged (`""`) entries are skipped — they're already visible via `rows()`.
    pub fn rows_seg(&self) -> Vec<(String, String, f64, usize)> {
        let mut out: Vec<(String, String, f64, usize)> = self
            .by_key
            .iter()
            .filter(|(k, _)| !k.1.is_empty())
            .map(|(k, &(xx, xy, n))| {
                (
                    k.0.clone(),
                    k.1.clone(),
                    if xx > 0.0 { xy / xx } else { 0.0 },
                    n,
                )
            })
            .collect();
        out.sort_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)));
        out
    }
}

/// Why a (venue, segment) has not earned the right to be traded on its own forward evidence, or
/// `None` when it has. The rule the pilot's city gate and the dashboard's computed A/B row BOTH
/// apply — it lives here, next to `lambda_segment`, for the same reason that boundary does: two
/// bins asking the same question must not drift apart on the answer.
///
/// Order matters and is the whole subtlety. `lambda_seg` deliberately falls back to the venue fold
/// under `MIN_N`, so a segment with no fit of its own still reports a plausible λ — on 2026-08-25
/// every one of the eight then-new Kalshi cities answered with the venue's healthy 0.355 while
/// three of them were running negative slopes. Coverage is therefore checked FIRST, via `n_seg`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentVeto {
    /// Fewer than `ShrinkageFit::MIN_N` resolved rows: nothing has measured this segment forward.
    Unvalidated,
    /// Fitted λ under the caller's floor: measured, and the disagreements are anti-signal.
    BelowFloor,
}

pub fn segment_veto(
    fit: &ShrinkageFit,
    venue: &str,
    segment: &str,
    floor: f64,
) -> Option<SegmentVeto> {
    if fit.n_seg(venue, segment) < ShrinkageFit::MIN_N {
        return Some(SegmentVeto::Unvalidated);
    }
    (fit.lambda_seg(venue, segment) < floor).then_some(SegmentVeto::BelowFloor)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feed n observations with exact slope `s` (x = 0.1, y = 0.1·s) under one key.
    fn feed(fit: &mut ShrinkageFit, venue: &str, seg: &str, s: f64, n: usize) {
        for _ in 0..n {
            fit.observe_seg(venue, seg, 0.1, 0.1 * s);
        }
    }

    #[test]
    fn venue_lookup_folds_across_segments_identically_to_untagged() {
        // Tagged fit: two segments with different slopes.
        let mut tagged = ShrinkageFit::default();
        feed(&mut tagged, "kalshi", "lo", -0.2, 30);
        feed(&mut tagged, "kalshi", "hi", 0.6, 30);
        // Untagged fit: the same observations under the bare venue.
        let mut plain = ShrinkageFit::default();
        feed(&mut plain, "kalshi", "", -0.2, 30);
        feed(&mut plain, "kalshi", "", 0.6, 30);
        assert!((tagged.lambda("kalshi") - plain.lambda("kalshi")).abs() < 1e-12);
    }

    #[test]
    fn segment_lambda_falls_back_segment_to_venue_to_pooled_to_one() {
        let mut fit = ShrinkageFit::default();
        // Thin segment (< MIN_N) inside a thick venue: falls back to the venue fold.
        feed(&mut fit, "kalshi", "lo", -0.5, 10);
        feed(&mut fit, "kalshi", "hi", 0.5, 50);
        let venue_lambda = fit.lambda("kalshi");
        assert_eq!(fit.lambda_seg("kalshi", "lo"), venue_lambda);
        // Thick segment: its own fit, clamped ≥ 0 would not apply here (0.5 > 0).
        assert!((fit.lambda_seg("kalshi", "hi") - 0.5).abs() < 1e-12);
        // Unknown venue: pooled.
        let pooled = fit.lambda_seg("polymarket", "lo");
        assert!((pooled - fit.lambda("polymarket")).abs() < 1e-12);
        // Empty fit: 1.0.
        let empty = ShrinkageFit::default();
        assert_eq!(empty.lambda_seg("kalshi", "lo"), 1.0);
    }

    #[test]
    fn anti_signal_segment_clamps_to_zero_and_stops_trading() {
        let mut fit = ShrinkageFit::default();
        feed(&mut fit, "polymarket", "lo", -0.1, 50); // the sub-10¢ tail
        feed(&mut fit, "polymarket", "hi", 0.4, 50);
        assert_eq!(
            fit.lambda_seg("polymarket", "lo"),
            0.0,
            "negative segment slope clamps to zero — shrunk edge can never clear a threshold"
        );
        assert!((fit.lambda_seg("polymarket", "hi") - 0.4).abs() < 1e-12);
        // The venue fold sits between them, exactly as the per-venue fit always did.
        let v = fit.lambda("polymarket");
        assert!(v > 0.0 && v < 0.4);
    }

    #[test]
    fn n_seg_counts_only_its_own_key_and_never_falls_back() {
        let mut fit = ShrinkageFit::default();
        feed(&mut fit, "kalshi", "Vegas", -0.9, 24); // a young city, under MIN_N
        feed(&mut fit, "kalshi", "NYC", 0.4, 258);
        assert_eq!(fit.n_seg("kalshi", "Vegas"), 24);
        assert_eq!(fit.n_seg("kalshi", "NYC"), 258);
        assert_eq!(fit.n_seg("kalshi", "Nowhere"), 0);
        assert_eq!(fit.n_seg("polymarket", "NYC"), 0, "keys are per venue");
        // The point of the accessor: λ hides the thin sample behind the venue fold, so a caller
        // reading λ alone cannot tell a never-fitted city from a healthy one.
        assert_eq!(fit.lambda_seg("kalshi", "Vegas"), fit.lambda("kalshi"));
        assert!(fit.lambda_seg("kalshi", "Vegas") > 0.0);
    }

    #[test]
    fn segment_veto_checks_coverage_before_slope() {
        let mut fit = ShrinkageFit::default();
        feed(&mut fit, "kalshi", "Miami", 0.75, ShrinkageFit::MIN_N);
        feed(&mut fit, "kalshi", "Denver", -0.2, ShrinkageFit::MIN_N);
        feed(&mut fit, "kalshi", "Vegas", 0.9, ShrinkageFit::MIN_N - 1);
        let floor = 0.2;
        assert_eq!(segment_veto(&fit, "kalshi", "Miami", floor), None);
        assert_eq!(
            segment_veto(&fit, "kalshi", "Denver", floor),
            Some(SegmentVeto::BelowFloor)
        );
        // One row short, with a slope that would sail through the floor if it were trusted —
        // and `lambda_seg` reports the venue fold for it, which is exactly the trap.
        assert!(fit.lambda_seg("kalshi", "Vegas") > floor);
        assert_eq!(
            segment_veto(&fit, "kalshi", "Vegas", floor),
            Some(SegmentVeto::Unvalidated)
        );
        // Never seen, and an empty fit: withheld, not waved through on the 1.0 no-shrink default.
        assert_eq!(
            segment_veto(&fit, "kalshi", "Nowhere", floor),
            Some(SegmentVeto::Unvalidated)
        );
        assert_eq!(
            segment_veto(&ShrinkageFit::default(), "kalshi", "Miami", floor),
            Some(SegmentVeto::Unvalidated)
        );
    }

    #[test]
    fn rows_and_rows_seg_report_the_expected_shapes() {
        let mut fit = ShrinkageFit::default();
        feed(&mut fit, "kalshi", "lo", 0.2, 5);
        feed(&mut fit, "kalshi", "hi", 0.8, 5);
        feed(&mut fit, "polymarket", "", 0.5, 5);
        let rows = fit.rows();
        assert_eq!(rows.len(), 2, "one folded row per venue");
        let k = rows.iter().find(|r| r.0 == "kalshi").unwrap();
        assert_eq!(k.2, 10);
        assert!((k.1 - 0.5).abs() < 1e-12, "fold averages the two segments");
        let segs = fit.rows_seg();
        assert_eq!(segs.len(), 2, "untagged entries are skipped");
        assert!(segs.iter().all(|r| r.0 == "kalshi"));
    }
}
