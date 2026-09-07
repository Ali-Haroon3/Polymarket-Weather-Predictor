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

/// The λ segment for a market SHAPE and price — the second segment axis (2026-09-07). The 10¢
/// band above was drawn when the tail was the visible heterogeneity; by September the dominant
/// axis on the Kalshi SELL side (lead ≥ 1, px ≥ 0.10) was the market's shape. Bucket markets
/// realized λ +0.06 / +0.03 on the full sample (n=547 / 369 in the two bands below) and
/// −0.24 / −0.06 over the trailing 30 days, while threshold markets realized +0.22 / +0.98
/// (n=90 / 143) and +0.74 / +1.00 — and the pilot's own ledger agreed: 20 bucket orders at −19%
/// ROI, 3 threshold orders at +98%. A 2 °F bucket's probability is hypersensitive to the
/// half-degree bias errors the seasonal refits chase, where a tail threshold integrates over
/// them, and prediction-market tails carry the classic longshot overprice. Within each shape the
/// 35¢ line separates the mid-book, where disagreements realize almost fully, from the 10–35¢
/// band. Six keys; a thin key answers from the venue fold inside `lambda_seg`, so young data
/// gates exactly like the venue λ. Non-bucket shapes (`temp_at_least`, `temp_at_most`, legacy
/// `temperature`) are all thresholds.
///
/// Calibration is not profit, and the pilot's gate does NOT read this key (it keeps
/// `lambda_segment`): a flat-stake, fee-inclusive replay of the pilot's exact rule over all
/// captures under this key LOST — −26% on 10 trades since 2026-08-09 against +8.8% on 11 for
/// the band key — because a λ near 1 on mid-book thresholds admits the 13–20% claimed-edge
/// candidates, and that claimed-edge band realizes nothing (the winner's curse the go-live gate
/// documents). A coarser first replay had read +12%; its profit was seven small cheap-NO wins at
/// 18–31¢ bids that this six-key scheme refuses. The dashboard's `--shape-lambda` A/B rows
/// (frozen 2026-09-07) exist to settle that disagreement forward. Lives here so no bin can drift
/// on the boundaries if one ever adopts it.
pub fn shape_segment(market_type: &str, px: f64) -> &'static str {
    let bucket = market_type == "temp_bucket";
    match (bucket, px) {
        (true, p) if p < 0.10 => "bucket·px<0.10",
        (true, p) if p < 0.35 => "bucket·px0.10–0.35",
        (true, _) => "bucket·px≥0.35",
        (false, p) if p < 0.10 => "threshold·px<0.10",
        (false, p) if p < 0.35 => "threshold·px0.10–0.35",
        (false, _) => "threshold·px≥0.35",
    }
}

/// The price a capture's model-vs-market disagreement is measured against, shared by every λ fit
/// (the dashboard's `ref_price`, the pilot's capture fit, and mirrored by the analysis scripts):
/// the mid of a sane two-sided book; the one side a one-sided book quotes; the last trade
/// (`entry_price`) ONLY for rows on which the venue reported no book at all (legacy captures
/// predating book capture). `None` when nothing usable — a 0/1 price is an empty side, not a
/// price, and a crossed book is not trusted.
///
/// Why the one-sided case is its own rule (2026-09-07): Kalshi's anonymous market list nulls
/// `last_price`, so `kalshi_price` writes the 0.50 "no trade / no book" PLACEHOLDER as
/// `entry_price` whenever the book is not two-sided — and every consumer that fell back to the
/// last trade read that placeholder as a real 50¢ price on markets with a 1¢ ask and no bid.
/// 133 resolved lead ≥ 1 Kalshi rows carried it (all resolved NO, as a 1¢ market does), entering
/// the fit as x ≈ −0.48, y = −0.50: realized ≈ claimed, at ~25× the x² weight of a typical row.
/// Those rows alone made the Kalshi px ≥ 0.10 λ read +0.474 (n=1801); without them it is −0.006
/// (n=1668), and the venue fold goes +0.380 → −0.019. The dashboard's `decide()` was worse off
/// still: it "sold" those markets at 0.50 (see `fill_prices`).
pub fn reference_price(
    entry_price: f64,
    best_bid: Option<f64>,
    best_ask: Option<f64>,
) -> Option<f64> {
    let usable = |x: f64| (x > 0.0 && x < 1.0).then_some(x);
    let reported_book = best_bid.is_some() || best_ask.is_some();
    let px = match (best_bid.and_then(usable), best_ask.and_then(usable)) {
        (Some(b), Some(a)) if b <= a => (a + b) / 2.0,
        (Some(_), Some(_)) => return None, // crossed book: nothing to trust
        (Some(b), None) => b,
        (None, Some(a)) => a,
        (None, None) if reported_book => return None, // a reported book with nothing on it
        (None, None) => entry_price,
    };
    usable(px)
}

/// The EXECUTABLE prices a strategy replay may fill at: `(buy_px, sell_px)` — a BUY fills at the
/// ask, a SELL at the bid. The last trade (`entry_price`) is the fallback for BOTH sides only on
/// rows with no book at all (legacy captures); once the venue reported a book, a missing side is
/// simply untradable — there is nobody to sell to without a bid and nothing to buy without an
/// ask. Before 2026-09-07 the fallback applied PER SIDE, so a Kalshi row with a 1¢ ask and no bid
/// became a SELL at its 0.50 placeholder (`reference_price`): 133 phantom wins at the position
/// cap, compounding a $100k bankroll to $517M on the dashboard, and 7 of the open book's 10
/// positions on the day it was found.
pub fn fill_prices(
    entry_price: f64,
    best_bid: Option<f64>,
    best_ask: Option<f64>,
) -> (Option<f64>, Option<f64>) {
    let usable = |x: f64| (x > 0.0 && x < 1.0).then_some(x);
    let fallback = if best_bid.is_some() || best_ask.is_some() {
        None
    } else {
        usable(entry_price)
    };
    (
        best_ask.and_then(usable).or(fallback),
        best_bid.and_then(usable).or(fallback),
    )
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

    #[test]
    fn reference_price_never_reads_the_placeholder_behind_a_reported_book() {
        // Two-sided sane book: the mid, whatever the last trade says.
        assert!((reference_price(0.5, Some(0.40), Some(0.44)).unwrap() - 0.42).abs() < 1e-12);
        // The 2026-09-07 defect: a Kalshi row with a 1¢ ask, no bid, and the 0.50 placeholder
        // stored as its "last trade" is a 1¢ market, not a 50¢ one.
        assert!((reference_price(0.5, None, Some(0.01)).unwrap() - 0.01).abs() < 1e-12);
        // Bid-only book (a 99¢ market with no ask): the bid.
        assert!((reference_price(0.5, Some(0.99), None).unwrap() - 0.99).abs() < 1e-12);
        // No book at all (legacy rows): the last trade is all there is.
        assert!((reference_price(0.31, None, None).unwrap() - 0.31).abs() < 1e-12);
        assert_eq!(reference_price(0.0, None, None), None, "0/1 is not a price");
        // A reported book with nothing usable on it never falls through to the last trade —
        // a 0-cent bid and a 100-cent ask are empty sides.
        assert_eq!(reference_price(0.5, Some(0.0), Some(1.0)), None);
        // Crossed book: not trusted.
        assert_eq!(reference_price(0.5, Some(0.60), Some(0.40)), None);
    }

    #[test]
    fn fill_prices_need_the_side_they_hit_once_a_book_was_reported() {
        // Two-sided: BUY at the ask, SELL at the bid.
        assert_eq!(
            fill_prices(0.5, Some(0.40), Some(0.44)),
            (Some(0.44), Some(0.40))
        );
        // The phantom: ask 1¢, no bid, placeholder 0.50 — a BUY can fill at 1¢, a SELL cannot
        // fill at all (and certainly not at 0.50).
        assert_eq!(fill_prices(0.5, None, Some(0.01)), (Some(0.01), None));
        // Bid-only: SELL at the bid, nothing to buy.
        assert_eq!(fill_prices(0.5, Some(0.99), None), (None, Some(0.99)));
        // No book at all: the last trade fills both sides (legacy rows).
        assert_eq!(fill_prices(0.31, None, None), (Some(0.31), Some(0.31)));
        assert_eq!(fill_prices(0.0, None, None), (None, None));
        // Empty sides reported as 0 / 1 are not fills either.
        assert_eq!(fill_prices(0.5, Some(0.0), Some(1.0)), (None, None));
    }

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
        assert!(
            full > floor,
            "full-sample λ should still look healthy, got {full}"
        );
        // The trailing 14-day window ending today is 9 healthy days (45 rows) + 5 bad (25 rows):
        // slope (45·0.005 − 25·0.005)/(70·0.01) = 0.143 < 0.2. Vetoed on the trailing check.
        let trail = fit.trailing_slope("kalshi", "LA", today).unwrap();
        assert!(
            trail < floor,
            "trailing slope should be under the floor, got {trail}"
        );
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
    #[test]
    fn shape_segment_splits_on_shape_then_the_two_price_lines() {
        assert_eq!(shape_segment("temp_bucket", 0.05), "bucket·px<0.10");
        assert_eq!(shape_segment("temp_bucket", 0.10), "bucket·px0.10–0.35");
        assert_eq!(shape_segment("temp_bucket", 0.349), "bucket·px0.10–0.35");
        assert_eq!(shape_segment("temp_bucket", 0.35), "bucket·px≥0.35");
        assert_eq!(shape_segment("temp_at_least", 0.05), "threshold·px<0.10");
        assert_eq!(shape_segment("temp_at_most", 0.20), "threshold·px0.10–0.35");
        assert_eq!(shape_segment("temperature", 0.60), "threshold·px≥0.35");
        // The tail line is the band fn's line, so the two axes never disagree about the tail.
        for px in [0.05, 0.0999, 0.10, 0.5] {
            assert_eq!(
                lambda_segment(px) == "px<0.10",
                shape_segment("temp_bucket", px).ends_with("px<0.10")
            );
        }
        // Six distinct keys, all non-empty (an empty segment would mean "no segment").
        let keys: std::collections::BTreeSet<&str> = [
            ("temp_bucket", 0.05),
            ("temp_bucket", 0.2),
            ("temp_bucket", 0.5),
            ("temp_at_least", 0.05),
            ("temp_at_least", 0.2),
            ("temp_at_least", 0.5),
        ]
        .iter()
        .map(|(m, p)| shape_segment(m, *p))
        .collect();
        assert_eq!(keys.len(), 6);
        assert!(keys.iter().all(|k| !k.is_empty()));
    }

    #[test]
    fn shape_keyed_fit_clamps_buckets_and_trusts_thresholds_independently() {
        let mut fit = ShrinkageFit::default();
        for _ in 0..ShrinkageFit::MIN_N {
            fit.observe_seg("kalshi", shape_segment("temp_bucket", 0.5), 0.10, -0.01);
            fit.observe_seg("kalshi", shape_segment("temp_at_least", 0.5), 0.10, 0.095);
        }
        assert_eq!(
            fit.lambda_seg("kalshi", shape_segment("temp_bucket", 0.5)),
            0.0
        );
        assert!(
            (fit.lambda_seg("kalshi", shape_segment("temp_at_least", 0.5)) - 0.95).abs() < 1e-9
        );
        // A thin key (no threshold rows in the 10–35¢ band) answers from the venue fold, which
        // averages the two: the fallback is the old per-venue behaviour, never 1.0.
        let fold = fit.lambda("kalshi");
        assert!(fold > 0.0 && fold < 0.95);
        assert!(
            (fit.lambda_seg("kalshi", shape_segment("temp_at_least", 0.2)) - fold).abs() < 1e-12
        );
    }
}
