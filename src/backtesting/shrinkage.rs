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

use chrono::{Duration, NaiveDate};

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

/// Trailing window (in days of resolved target dates) for the drift view of λ. The full-sample
/// slope grows sluggish as history accrues — a two-week anti-signal stretch barely moves it — so
/// the trailing slope is the early-warning view. Shared by the dashboard's λ diagnostics and the
/// `segment_veto` trailing check, so the number a human watches on the dashboard is the number
/// the pilot acts on.
pub const TRAIL_WINDOW_DAYS: i64 = 14;
/// Minimum rows inside the trailing window before its slope is trusted at all; under this it is
/// mostly noise and the trailing check simply does not apply.
pub const TRAIL_MIN_N: usize = 20;

#[derive(Default)]
pub struct ShrinkageFit {
    /// (venue, segment) → (Σx², Σxy, n) running sums for the through-origin slope Σxy/Σx².
    /// Untagged observations land under segment `""`. BTreeMap, not HashMap: folds sum floats in
    /// key order, so every fold is deterministic run to run (determinism is load-bearing here).
    by_key: BTreeMap<(String, String), (f64, f64, usize)>,
    /// (venue, segment) → dated (x, y) observations, populated only by `observe_seg_dated`, for
    /// `trailing_slope`. The running sums above answer "what has this segment done, ever"; a
    /// trailing window answers "what has it done lately", which is a different question once the
    /// segment has enough history that its full-sample slope can no longer move on a bad fortnight.
    dated: BTreeMap<(String, String), Vec<DatedObs>>,
}

/// One dated λ observation: (target date, claimed edge, realized edge).
type DatedObs = (NaiveDate, f64, f64);

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

    /// `observe_seg`, additionally remembering the observation's target date so `trailing_slope`
    /// can answer for a window. The running sums are updated identically, so every existing lookup
    /// (`lambda`, `lambda_seg`, `n_seg`, `rows*`) is unchanged by which observe variant fed it.
    pub fn observe_seg_dated(
        &mut self,
        venue: &str,
        segment: &str,
        date: NaiveDate,
        predicted: f64,
        realized: f64,
    ) {
        self.observe_seg(venue, segment, predicted, realized);
        self.dated
            .entry((venue.to_string(), segment.to_string()))
            .or_default()
            .push((date, predicted, realized));
    }

    /// Raw (unclamped, no fallback) through-origin slope over the segment's dated observations
    /// with target date in the `TRAIL_WINDOW_DAYS`-day window ending at `as_of`, and the row count
    /// behind it. A diagnostic value, not a trading value: it can be negative, and it is `None`
    /// under `TRAIL_MIN_N` rows rather than falling back to anything — a thin window says nothing.
    /// Only observations recorded through `observe_seg_dated` are visible here.
    pub fn trailing_slope(&self, venue: &str, segment: &str, as_of: NaiveDate) -> Option<f64> {
        let from = as_of - Duration::days(TRAIL_WINDOW_DAYS - 1);
        let (xx, xy, n) = self
            .dated
            .get(&(venue.to_string(), segment.to_string()))
            .map(|v| {
                v.iter()
                    .filter(|(d, _, _)| *d >= from && *d <= as_of)
                    .fold((0.0, 0.0, 0usize), |a, (_, x, y)| {
                        (a.0 + x * x, a.1 + x * y, a.2 + 1)
                    })
            })
            .unwrap_or((0.0, 0.0, 0));
        (n >= TRAIL_MIN_N && xx > 0.0).then(|| xy / xx)
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
///
/// The third check exists because the first two have a blind spot that cost the pilot three
/// straight trades. A full-sample λ over hundreds of rows cannot move on a bad fortnight: Kalshi LA
/// flipped from a −0.96 °C to a +1.35 °C residual regime on 2026-08-31 (mean z +1.72 over four
/// days), the pilot sold three LA buckets that all hit, and LA's full-sample λ stayed at +0.35 on
/// n=318 — comfortably above the floor. Vegas was caught only because it was NEW; a mature city
/// going wrong is invisible to the first two checks for weeks. So the segment's trailing
/// `TRAIL_WINDOW_DAYS` slope, as of the caller's date, is held to the same floor. The window and
/// its minimum are the dashboard's own drift-diagnostic constants, chosen there long before this
/// gate existed — not fitted to the LA sample that motivated it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentVeto {
    /// Fewer than `ShrinkageFit::MIN_N` resolved rows: nothing has measured this segment forward.
    Unvalidated,
    /// Fitted full-sample λ under the caller's floor: measured, and the disagreements are anti-signal.
    BelowFloor,
    /// Full-sample λ is fine but the trailing `TRAIL_WINDOW_DAYS` slope (≥ `TRAIL_MIN_N` rows)
    /// is under the floor: the segment has gone wrong RECENTLY and history is masking it.
    TrailingBelowFloor,
}

pub fn segment_veto(
    fit: &ShrinkageFit,
    venue: &str,
    segment: &str,
    floor: f64,
    as_of: NaiveDate,
) -> Option<SegmentVeto> {
    if fit.n_seg(venue, segment) < ShrinkageFit::MIN_N {
        return Some(SegmentVeto::Unvalidated);
    }
    if fit.lambda_seg(venue, segment) < floor {
        return Some(SegmentVeto::BelowFloor);
    }
    fit.trailing_slope(venue, segment, as_of)
        .filter(|s| *s < floor)
        .map(|_| SegmentVeto::TrailingBelowFloor)
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

    fn d(s: &str) -> NaiveDate {
        NaiveDate::parse_from_str(s, "%Y-%m-%d").unwrap()
    }

    #[test]
    fn segment_veto_checks_coverage_before_slope() {
        let mut fit = ShrinkageFit::default();
        feed(&mut fit, "kalshi", "Miami", 0.75, ShrinkageFit::MIN_N);
        feed(&mut fit, "kalshi", "Denver", -0.2, ShrinkageFit::MIN_N);
        feed(&mut fit, "kalshi", "Vegas", 0.9, ShrinkageFit::MIN_N - 1);
        let (floor, today) = (0.2, d("2026-09-06"));
        assert_eq!(segment_veto(&fit, "kalshi", "Miami", floor, today), None);
        assert_eq!(
            segment_veto(&fit, "kalshi", "Denver", floor, today),
            Some(SegmentVeto::BelowFloor)
        );
        // One row short, with a slope that would sail through the floor if it were trusted —
        // and `lambda_seg` reports the venue fold for it, which is exactly the trap.
        assert!(fit.lambda_seg("kalshi", "Vegas") > floor);
        assert_eq!(
            segment_veto(&fit, "kalshi", "Vegas", floor, today),
            Some(SegmentVeto::Unvalidated)
        );
        // Never seen, and an empty fit: withheld, not waved through on the 1.0 no-shrink default.
        assert_eq!(
            segment_veto(&fit, "kalshi", "Nowhere", floor, today),
            Some(SegmentVeto::Unvalidated)
        );
        assert_eq!(
            segment_veto(&ShrinkageFit::default(), "kalshi", "Miami", floor, today),
            Some(SegmentVeto::Unvalidated)
        );
    }

    /// The LA shape in miniature: a long healthy history that pins the full-sample λ well above
    /// the floor, then a fortnight of anti-signal that the full-sample fit cannot see.
    #[test]
    fn trailing_veto_catches_a_mature_segment_that_has_gone_wrong_recently() {
        let mut fit = ShrinkageFit::default();
        let floor = 0.2;
        // 60 days × 5 rows of healthy λ 0.5, target dates 2026-06-01 .. 2026-07-30.
        for i in 0..60 {
            let day = d("2026-06-01") + Duration::days(i);
            for _ in 0..5 {
                fit.observe_seg_dated("kalshi", "LA", day, 0.1, 0.05);
            }
        }
        // Then 5 days × 5 rows of λ −0.5 — 25 rows, over TRAIL_MIN_N — ending 2026-08-04.
        for i in 0..5 {
            let day = d("2026-07-31") + Duration::days(i);
            for _ in 0..5 {
                fit.observe_seg_dated("kalshi", "LA", day, 0.1, -0.05);
            }
        }
        let today = d("2026-08-04");
        // Full-sample λ barely notices: 300 healthy rows against 25 bad ones.
        let full = fit.lambda_seg("kalshi", "LA");
        assert!(full > floor, "full-sample λ should still look healthy, got {full}");
        // The trailing 14-day window ending today is 9 healthy days (45 rows) + 5 bad (25 rows):
        // slope (45·0.005 − 25·0.005)/(70·0.01) = 0.143 < 0.2. Vetoed on the trailing check.
        let trail = fit.trailing_slope("kalshi", "LA", today).unwrap();
        assert!(trail < floor, "trailing slope should be under the floor, got {trail}");
        assert_eq!(
            segment_veto(&fit, "kalshi", "LA", floor, today),
            Some(SegmentVeto::TrailingBelowFloor)
        );
        // Asked as of a date before the bad stretch, the same fit passes: the veto is causal.
        assert_eq!(
            segment_veto(&fit, "kalshi", "LA", floor, d("2026-07-30")),
            None
        );
        // A window with too few rows says nothing rather than something: undated observations
        // never populate it, so a fit fed via plain `observe_seg` has no trailing opinion.
        let mut undated = ShrinkageFit::default();
        feed(&mut undated, "kalshi", "LA", 0.5, ShrinkageFit::MIN_N);
        assert_eq!(undated.trailing_slope("kalshi", "LA", today), None);
        assert_eq!(segment_veto(&undated, "kalshi", "LA", floor, today), None);
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
