//! Market-implied distribution ("market shape") strategy — the model-free alpha found 2026-09-07.
//!
//! The finding. On Kalshi a city-day's temperature markets form a LADDER: four 2 °F buckets plus
//! an open-ended threshold at each end, mutually exclusive and jointly exhaustive, so their mids
//! sum to ~1 and imply a full distribution of the settlement high. Fitting a Normal(μ, σ) to that
//! ladder by least squares gives the MARKET's forecast — and against realized highs it is better
//! than the weather model's (RMSE 0.92 °C vs 1.22 on 458 uncensored ladders; a regression of the
//! realized high on both gives the model 2% weight). So the model has nothing to add on the mean.
//! What the market gets wrong is its own SHAPE, in two persistent ways:
//!
//!   * it runs COLD: realized − market μ = +0.27 °C, positive in 8 of 9 target weeks and 14 of 15
//!     cities. The plausible mechanism is structural: Kalshi settles on the NWS CLI daily max, a
//!     continuous-sensor reading that sits 0–2 °F ABOVE the whole-degree METAR observations
//!     traders watch (the same gap `stations.rs` fits a post-day bias for), so a ladder priced
//!     off the obs runs a half-degree cold against the number that actually settles it;
//!   * it is TOO WIDE: the market's own z-scores have sd 0.86 with 75% inside ±1σ (Normal: 68%),
//!     the classic longshot overpricing — tails at 10–20¢ realize 9%, favorites at 45–55¢
//!     realize 62%.
//!
//! The strategy re-shapes the market's own ladder — Normal(μ + b, k·σ) — and trades every cell
//! whose re-shaped probability differs from the EXECUTABLE price by more than a threshold after
//! Kalshi's fee: BUY YES at the ask (the under-priced favorite and its neighbours), BUY NO at the
//! bid (the over-priced tail). `b` and `k` are fitted WALK-FORWARD, by minimizing the cell Brier
//! score over the ladders that had resolved before the trading day (strictly earlier target
//! dates), so no trade sees its own or a same-day outcome. Over captures 07-05..09-06 (strict
//! walk-forward, executable prices, fees): 899 trades at +10.7% ROI on risk, 64% win, t = 4.6
//! raw and 4.1 clustered by date; every month positive (+17% / +10% / +11%); both sides
//! independently significant (BUY +23% ROI, SELL ≥ 10¢ +10%); robust to the fit window and
//! cadence (+9.7..11.4% across six variants); and the claimed edges realize almost one-for-one
//! (claimed 0.043 → realized 0.035; 0.078 → 0.090), which the weather model's never did. The
//! bias is the larger contributor (b alone: +11.9% at θ = 3%; k alone: +1.3%), the sharpening
//! adds breadth. Sub-10¢ NO purchases realize nothing (−0.4% on 164) and are floored out.
//!
//! Hygiene. Only COMPLETE ladders are trusted (mids summing to 0.8–1.2 over ≥ 4 cells — a
//! partial ladder's Normal fit is unconstrained); Polymarket captures hold one to three markets
//! per city-day, so this is Kalshi-only until that changes. Reference prices come from the shared
//! `reference_price` rule (a one-sided book is priced at its quoted side, never the 0.50
//! placeholder — the defect that produced this repo's earlier phantom edge). The dashboard's A/B
//! rows and the pilot both call THIS module for the ladder fit, the walk-forward (b, k) and the
//! cell decision, so the two bins cannot drift on the rule.

use std::collections::BTreeMap;

use chrono::NaiveDate;

use crate::utils::normal_cdf;

/// Resolved, complete, lead ≥ 1 ladders needed before (b, k) is fitted at all; below it the
/// strategy stands down rather than trade on a guess.
pub const MIN_LADDERS: usize = 60;
/// Most recent resolved ladders the fit reads. The sensitivity sweep (150 / 300 / all) moved ROI
/// by ±1 point; 300 is ~6 weeks of the current Kalshi universe, long enough to be stable and
/// short enough to follow a seasonal drift in the market's bias.
pub const HIST_CAP: usize = 300;
/// Edge after fees a cell must show at the executable price. θ = 3% is the sweep's sweet spot
/// (θ = 0: +4.5% on 2069, θ = 3%: +10.7% on 899, θ = 6%: +17.6% on 411 — ROI rises with θ, so
/// the threshold trades breadth for quality, and 3% keeps ~2 trades per ladder).
pub const DEFAULT_EDGE_THRESHOLD: f64 = 0.03;
/// Executable-price floor on BOTH sides: a NO purchase against a sub-10¢ YES bid realized −0.4%
/// on 164 trades (the bid is one tick wide there), and a sub-10¢ YES purchase is a lottery
/// ticket the sizing can't hold.
pub const MIN_PRICE: f64 = 0.10;
/// Mids of a complete ladder sum to about 1 (the ask-side overround pushes a little over).
pub const COMPLETE_SUM: (f64, f64) = (0.8, 1.2);
pub const COMPLETE_CELLS: usize = 4;

/// One market of a ladder, its settlement interval in °C, half-open `[lo, hi)`; an open-ended
/// threshold carries ±∞ on the open side.
#[derive(Debug, Clone)]
pub struct LadderCell {
    pub market_id: String,
    pub market_type: String,
    pub lo_c: f64,
    pub hi_c: f64,
    /// Reference price (the shared `reference_price` rule).
    pub price: f64,
    pub bid: Option<f64>,
    pub ask: Option<f64>,
    pub outcome: Option<f64>,
}

/// One city-day's ladder as captured on one day, with the market-implied Normal fitted to it.
#[derive(Debug, Clone)]
pub struct Ladder {
    pub venue: String,
    pub city: String,
    pub target: NaiveDate,
    pub captured: NaiveDate,
    pub cells: Vec<LadderCell>,
    /// Market-implied Normal in °C: least-squares fit of cell probabilities to mids.
    pub mu: f64,
    pub sigma: f64,
    pub fit_sse: f64,
    pub price_sum: f64,
}

/// One market row, as either a capture or a live venue row presents it; `price` is already the
/// shared reference price (`None` ⇒ the row carries no usable price and is dropped).
#[derive(Debug, Clone)]
pub struct LadderInput {
    pub venue: String,
    pub city: String,
    pub target: NaiveDate,
    pub captured: NaiveDate,
    pub market_id: String,
    pub market_type: String,
    pub threshold: f64,
    pub threshold_upper: Option<f64>,
    pub unit: Option<String>,
    pub price: Option<f64>,
    pub bid: Option<f64>,
    pub ask: Option<f64>,
    pub outcome: Option<f64>,
}

/// The two re-shaping parameters: shift the market's mean by `bias_c` (°C) and scale its σ by
/// `sigma_scale`. (0, 1) is the market itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeParams {
    pub bias_c: f64,
    pub sigma_scale: f64,
}

/// Which contract a signal buys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeSide {
    /// The cell is under-priced: buy YES at the ask.
    BuyYes,
    /// The cell is over-priced: buy NO at (1 − bid) — the pilot's "SELL".
    BuyNo,
}

/// A tradable cell: the contract, its price per contract on that side, the YES price it was
/// judged at, the re-shaped probability and the edge net of fee that cleared the threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct ShapeDecision {
    pub side: ShapeSide,
    /// Price paid per contract on the bought side (ask for YES, 1 − bid for NO).
    pub price: f64,
    /// The executable YES price the edge was measured against (ask or bid).
    pub yes_price: f64,
    pub prob: f64,
    pub edge: f64,
}

/// A market's settlement interval in °C, half-open, from its shape fields. Buckets are
/// inclusive whole-degree ranges ("92–93" ⇒ [91.5, 93.5) °F; a Polymarket 1 °C bucket has
/// `threshold_upper == threshold`); `temp_at_least T` is [T − 0.5, ∞) and `temp_at_most T` is
/// (−∞, T + 0.5) in the market's unit, because settlement highs are whole degrees.
pub fn cell_bounds_c(
    market_type: &str,
    threshold: f64,
    threshold_upper: Option<f64>,
    unit: Option<&str>,
) -> Option<(f64, f64)> {
    let to_c = |v: f64| match unit.unwrap_or("F") {
        "C" | "c" => v,
        _ => (v - 32.0) * 5.0 / 9.0,
    };
    match market_type {
        "temp_bucket" => Some((
            to_c(threshold - 0.5),
            to_c(threshold_upper.unwrap_or(threshold) + 0.5),
        )),
        "temp_at_least" => Some((to_c(threshold - 0.5), f64::INFINITY)),
        "temp_at_most" => Some((f64::NEG_INFINITY, to_c(threshold + 0.5))),
        _ => None,
    }
}

/// P(lo ≤ X < hi) under Normal(mu, sigma), open ends allowed.
pub fn normal_cell_prob(lo: f64, hi: f64, mu: f64, sigma: f64) -> f64 {
    let cdf = |x: f64| {
        if x == f64::INFINITY {
            1.0
        } else if x == f64::NEG_INFINITY {
            0.0
        } else {
            normal_cdf((x - mu) / sigma)
        }
    };
    (cdf(hi) - cdf(lo)).clamp(0.0, 1.0)
}

/// Least-squares Normal(μ, σ) through a ladder's `(lo, hi, price)` cells: coarse grid (μ ± 8 °C
/// by 0.5, σ 0.5..5.0 by 0.5) then a fine pass (± 0.5 by 0.1). Deterministic; ties keep the
/// first (lowest μ, then lowest σ). Returns (μ, σ, SSE).
pub fn fit_market_normal(cells: &[(f64, f64, f64)], mu0: f64) -> (f64, f64, f64) {
    let sse = |mu: f64, sigma: f64| -> f64 {
        cells
            .iter()
            .map(|&(lo, hi, p)| {
                let d = normal_cell_prob(lo, hi, mu, sigma) - p;
                d * d
            })
            .sum()
    };
    let mut best = (f64::INFINITY, mu0, 1.0);
    for i in -16..=16 {
        let mu = mu0 + 0.5 * i as f64;
        for j in 1..=10 {
            let sigma = 0.5 * j as f64;
            let v = sse(mu, sigma);
            if v < best.0 {
                best = (v, mu, sigma);
            }
        }
    }
    let (_, cmu, csig) = best;
    for i in -5..=5 {
        let mu = cmu + 0.1 * i as f64;
        for j in -5..=5 {
            let sigma = (csig + 0.1 * j as f64).max(0.2);
            let v = sse(mu, sigma);
            if v < best.0 {
                best = (v, mu, sigma);
            }
        }
    }
    (best.1, best.2, best.0)
}

/// Group rows into ladders by (venue, city, target, captured) and fit each one. Rows without a
/// usable price or a recognised shape are dropped; a group needs at least three priced cells to
/// be a ladder at all (completeness is judged separately by `Ladder::is_complete`).
pub fn build_ladders(rows: Vec<LadderInput>) -> Vec<Ladder> {
    let mut groups: BTreeMap<(String, String, NaiveDate, NaiveDate), Vec<LadderCell>> =
        BTreeMap::new();
    for r in rows {
        let Some(price) = r.price else { continue };
        let Some((lo, hi)) = cell_bounds_c(
            &r.market_type,
            r.threshold,
            r.threshold_upper,
            r.unit.as_deref(),
        ) else {
            continue;
        };
        groups
            .entry((r.venue, r.city, r.target, r.captured))
            .or_default()
            .push(LadderCell {
                market_id: r.market_id,
                market_type: r.market_type,
                lo_c: lo,
                hi_c: hi,
                price,
                bid: r.bid,
                ask: r.ask,
                outcome: r.outcome,
            });
    }
    let mut out = Vec::new();
    for ((venue, city, target, captured), mut cells) in groups {
        if cells.len() < 3 {
            continue;
        }
        cells.sort_by(|a, b| {
            a.lo_c
                .partial_cmp(&b.lo_c)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.market_id.cmp(&b.market_id))
        });
        let price_sum: f64 = cells.iter().map(|c| c.price).sum();
        // Grid centre: the price-weighted centre of the finite cells (an open end contributes
        // its finite edge nudged 1 °C outward). ±8 °C of coarse grid around it covers any
        // plausible ladder, so the centre only needs to be roughly right.
        let mu0 = {
            let mut w = 0.0;
            let mut s = 0.0;
            for c in &cells {
                let lo = if c.lo_c.is_finite() {
                    c.lo_c
                } else {
                    c.hi_c - 1.0
                };
                let hi = if c.hi_c.is_finite() {
                    c.hi_c
                } else {
                    c.lo_c + 1.0
                };
                w += c.price;
                s += c.price * (lo + hi) / 2.0;
            }
            if w > 0.0 {
                s / w
            } else {
                cells
                    .iter()
                    .filter(|c| c.lo_c.is_finite())
                    .map(|c| c.lo_c)
                    .sum::<f64>()
                    / cells.iter().filter(|c| c.lo_c.is_finite()).count().max(1) as f64
            }
        };
        let flat: Vec<(f64, f64, f64)> = cells.iter().map(|c| (c.lo_c, c.hi_c, c.price)).collect();
        let (mu, sigma, fit_sse) = fit_market_normal(&flat, mu0);
        out.push(Ladder {
            venue,
            city,
            target,
            captured,
            cells,
            mu,
            sigma,
            fit_sse,
            price_sum,
        });
    }
    out
}

impl Ladder {
    /// Mids sum to ~1 over at least `COMPLETE_CELLS` cells: the whole distribution was captured,
    /// so its Normal fit means something.
    pub fn is_complete(&self) -> bool {
        self.cells.len() >= COMPLETE_CELLS
            && self.price_sum >= COMPLETE_SUM.0
            && self.price_sum <= COMPLETE_SUM.1
    }
    pub fn is_resolved(&self) -> bool {
        self.cells.iter().all(|c| c.outcome.is_some())
    }
    pub fn lead(&self) -> i64 {
        (self.target - self.captured).num_days()
    }
    /// Each cell's probability under the re-shaped Normal(μ + b, k·σ).
    pub fn shaped_probs(&self, p: &ShapeParams) -> Vec<f64> {
        let mu = self.mu + p.bias_c;
        let sigma = (self.sigma * p.sigma_scale).max(0.2);
        self.cells
            .iter()
            .map(|c| normal_cell_prob(c.lo_c, c.hi_c, mu, sigma))
            .collect()
    }
    /// Σ (shaped prob − outcome)² over the cells; the walk-forward fit's objective.
    pub fn brier(&self, p: &ShapeParams) -> f64 {
        self.shaped_probs(p)
            .iter()
            .zip(&self.cells)
            .map(|(q, c)| {
                let d = q - c.outcome.unwrap_or(0.0);
                d * d
            })
            .sum()
    }
}

/// The resolved evidence a trading day may read: complete, lead ≥ 1 ladders of `venue` whose
/// TARGET date is strictly before `as_of` (they settled the evening before at the latest), newest
/// `HIST_CAP` by target date. Strictly earlier because same-day ladders in other cities have not
/// resolved either when the day's decisions are made.
pub fn shape_history<'a>(ladders: &'a [Ladder], venue: &str, as_of: NaiveDate) -> Vec<&'a Ladder> {
    let mut hist: Vec<&Ladder> = ladders
        .iter()
        .filter(|l| l.venue == venue && l.is_complete() && l.is_resolved() && l.lead() >= 1)
        .filter(|l| l.target < as_of)
        .collect();
    hist.sort_by(|a, b| (a.target, a.captured, &a.city).cmp(&(b.target, b.captured, &b.city)));
    if hist.len() > HIST_CAP {
        hist.drain(..hist.len() - HIST_CAP);
    }
    hist
}

/// Grid-fit (b, k) minimizing the summed cell Brier over `hist`: b ∈ [−0.6, 0.6] °C by 0.1,
/// k ∈ [0.6, 1.2] by 0.05. `None` under `MIN_LADDERS`. Deterministic: ties keep the first
/// (lowest b, then lowest k), so (0, 1) — the market itself — wins any exact tie near it.
pub fn fit_shape(hist: &[&Ladder]) -> Option<ShapeParams> {
    if hist.len() < MIN_LADDERS {
        return None;
    }
    let mut best: Option<(f64, ShapeParams)> = None;
    for bi in -6..=6 {
        for ki in 0..=12 {
            let p = ShapeParams {
                bias_c: 0.1 * bi as f64,
                sigma_scale: 0.6 + 0.05 * ki as f64,
            };
            let tot: f64 = hist.iter().map(|l| l.brier(&p)).sum();
            if best.is_none_or(|(b, _)| tot < b) {
                best = Some((tot, p));
            }
        }
    }
    best.map(|(_, p)| p)
}

/// The parameters a trading day at `as_of` may use — `shape_history` then `fit_shape`.
pub fn fit_shape_as_of(ladders: &[Ladder], venue: &str, as_of: NaiveDate) -> Option<ShapeParams> {
    fit_shape(&shape_history(ladders, venue, as_of))
}

/// Trade one cell, or not. BUY YES when the re-shaped probability beats the ask by more than
/// `theta` after `fee(ask)`; BUY NO when the bid beats it by more than `theta` after `fee(bid)`;
/// the executable YES price must be at least `min_price` on either side. Fees are charged on
/// the YES price (Kalshi's `0.07·P·(1−P)` is symmetric in P).
pub fn decide_cell(
    prob: f64,
    bid: Option<f64>,
    ask: Option<f64>,
    fee: impl Fn(f64) -> f64,
    theta: f64,
    min_price: f64,
) -> Option<ShapeDecision> {
    let usable = |x: f64| (x > 0.0 && x < 1.0 && x >= min_price).then_some(x);
    if let Some(a) = ask.and_then(usable) {
        let edge = prob - a - fee(a);
        if edge > theta {
            return Some(ShapeDecision {
                side: ShapeSide::BuyYes,
                price: a,
                yes_price: a,
                prob,
                edge,
            });
        }
    }
    if let Some(b) = bid.and_then(usable) {
        let edge = b - prob - fee(b);
        if edge > theta {
            return Some(ShapeDecision {
                side: ShapeSide::BuyNo,
                price: 1.0 - b,
                yes_price: b,
                prob,
                edge,
            });
        }
    }
    None
}

/// Every complete ladder's re-shaped cell probabilities under the parameters its capture day
/// could have known — keyed by (market_id, captured) — the walk-forward estimate the
/// dashboard's replay reads in place of the weather model. One (b, k) fit per capture date.
pub fn walk_forward_estimates(
    ladders: &[Ladder],
    venue: &str,
) -> BTreeMap<(String, NaiveDate), f64> {
    let mut by_date: BTreeMap<NaiveDate, Option<ShapeParams>> = BTreeMap::new();
    let mut out = BTreeMap::new();
    for l in ladders
        .iter()
        .filter(|l| l.venue == venue && l.is_complete())
    {
        let params = *by_date
            .entry(l.captured)
            .or_insert_with(|| fit_shape_as_of(ladders, venue, l.captured));
        let Some(p) = params else { continue };
        for (q, c) in l.shaped_probs(&p).iter().zip(&l.cells) {
            out.insert((c.market_id.clone(), l.captured), *q);
        }
    }
    out
}

/// One replayed trade of the strategy over resolved ladders — the pilot's health check.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayTrade {
    pub target: NaiveDate,
    pub city: String,
    pub market_id: String,
    pub side: ShapeSide,
    /// Price paid per contract on the bought side.
    pub price: f64,
    /// PnL per contract net of fee: +(1 − price) − fee on a win, −price − fee on a loss.
    pub pnl: f64,
}

/// Replay the strategy walk-forward over every resolved complete ladder of `venue` whose target
/// is in `[from, to)`: each ladder priced under the (b, k) its own capture day could have
/// fitted, each cell decided by `decide_cell` at `theta` / `min_price` with `fee`, settled at
/// its outcome. The pilot's trailing-window health breaker reads this (the analogue of the
/// model strategy's λ floor: "has the edge stopped realizing lately?"), and the dashboard's
/// market-shape rows are the same replay with Kelly sizing on top.
pub fn replay(
    ladders: &[Ladder],
    venue: &str,
    from: NaiveDate,
    to: NaiveDate,
    fee: impl Fn(f64) -> f64 + Copy,
    theta: f64,
    min_price: f64,
) -> Vec<ReplayTrade> {
    let est = walk_forward_estimates(ladders, venue);
    let mut out = Vec::new();
    for l in ladders
        .iter()
        .filter(|l| l.venue == venue && l.is_complete() && l.is_resolved() && l.lead() >= 1)
        .filter(|l| l.target >= from && l.target < to)
    {
        for c in &l.cells {
            let Some(&q) = est.get(&(c.market_id.clone(), l.captured)) else {
                continue;
            };
            let Some(d) = decide_cell(q, c.bid, c.ask, fee, theta, min_price) else {
                continue;
            };
            let outcome = c.outcome.unwrap_or(0.0);
            let won = match d.side {
                ShapeSide::BuyYes => outcome >= 0.5,
                ShapeSide::BuyNo => outcome < 0.5,
            };
            let pnl = if won { 1.0 - d.price } else { -d.price } - fee(d.yes_price);
            out.push(ReplayTrade {
                target: l.target,
                city: l.city.clone(),
                market_id: c.market_id.clone(),
                side: d.side,
                price: d.price,
                pnl,
            });
        }
    }
    out
}

/// ROI on capital risked (Σ pnl / Σ price paid) and trade count of a replay slice.
pub fn replay_roi(trades: &[ReplayTrade]) -> (f64, usize) {
    let risk: f64 = trades.iter().map(|t| t.price).sum();
    let pnl: f64 = trades.iter().map(|t| t.pnl).sum();
    (if risk > 0.0 { pnl / risk } else { 0.0 }, trades.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(s: &str) -> NaiveDate {
        NaiveDate::parse_from_str(s, "%Y-%m-%d").unwrap()
    }

    #[test]
    fn cell_bounds_follow_whole_degree_settlement() {
        // Kalshi 2 °F bucket 92–93 ⇒ [91.5, 93.5) °F.
        let (lo, hi) = cell_bounds_c("temp_bucket", 92.0, Some(93.0), Some("F")).unwrap();
        assert!((lo - (91.5 - 32.0) * 5.0 / 9.0).abs() < 1e-9);
        assert!((hi - (93.5 - 32.0) * 5.0 / 9.0).abs() < 1e-9);
        // Polymarket 1 °C bucket "30" ⇒ [29.5, 30.5).
        assert_eq!(
            cell_bounds_c("temp_bucket", 30.0, Some(30.0), Some("C")),
            Some((29.5, 30.5))
        );
        let (lo, hi) = cell_bounds_c("temp_at_least", 94.0, None, Some("F")).unwrap();
        assert!((lo - (93.5 - 32.0) * 5.0 / 9.0).abs() < 1e-9 && hi == f64::INFINITY);
        let (lo, hi) = cell_bounds_c("temp_at_most", 85.0, None, None).unwrap();
        assert!(lo == f64::NEG_INFINITY && (hi - (85.5 - 32.0) * 5.0 / 9.0).abs() < 1e-9);
        assert_eq!(cell_bounds_c("precipitation", 0.0, None, None), None);
    }

    /// A synthetic Kalshi-style ladder priced exactly from Normal(30.0, 1.1) fits back to it.
    fn synthetic(
        mu: f64,
        sigma: f64,
        target: &str,
        captured: &str,
        outcome_high: Option<f64>,
    ) -> LadderRows {
        let edges_f = [
            (f64::NEG_INFINITY, 85.5),
            (85.5, 87.5),
            (87.5, 89.5),
            (89.5, 91.5),
            (91.5, 93.5),
            (93.5, f64::INFINITY),
        ];
        let mut rows = Vec::new();
        for (i, (lo_f, hi_f)) in edges_f.iter().enumerate() {
            let lo = if lo_f.is_finite() {
                (lo_f - 32.0) * 5.0 / 9.0
            } else {
                *lo_f
            };
            let hi = if hi_f.is_finite() {
                (hi_f - 32.0) * 5.0 / 9.0
            } else {
                *hi_f
            };
            let p = normal_cell_prob(lo, hi, mu, sigma);
            let (mt, t, tu) = match i {
                0 => ("temp_at_most", 85.0, None),
                5 => ("temp_at_least", 94.0, None),
                _ => ("temp_bucket", lo_f + 0.5, Some(hi_f - 0.5)),
            };
            let outcome = outcome_high.map(|h| if h >= lo && h < hi { 1.0 } else { 0.0 });
            rows.push(LadderInput {
                venue: "kalshi".into(),
                city: "NYC".into(),
                target: d(target),
                captured: d(captured),
                market_id: format!("m{i}-{target}"),
                market_type: mt.into(),
                threshold: t,
                threshold_upper: tu,
                unit: Some("F".into()),
                price: Some(p.clamp(0.01, 0.99)),
                bid: Some((p - 0.01).clamp(0.01, 0.98)),
                ask: Some((p + 0.01).clamp(0.02, 0.99)),
                outcome,
            });
        }
        rows
    }
    type LadderRows = Vec<LadderInput>;

    #[test]
    fn normal_fit_recovers_the_ladders_distribution() {
        let l = build_ladders(synthetic(30.0, 1.1, "2026-09-08", "2026-09-07", None));
        assert_eq!(l.len(), 1);
        let l = &l[0];
        assert!(l.is_complete(), "sum {}", l.price_sum);
        assert!((l.mu - 30.0).abs() <= 0.11, "mu {}", l.mu);
        assert!((l.sigma - 1.1).abs() <= 0.11, "sigma {}", l.sigma);
        assert!(l.fit_sse < 1e-3);
        assert_eq!(l.lead(), 1);
        assert!(!l.is_resolved());
        // An incomplete ladder (three cells) is built but not complete.
        let mut rows = synthetic(30.0, 1.1, "2026-09-08", "2026-09-07", None);
        rows.truncate(3);
        let l = build_ladders(rows);
        assert_eq!(l.len(), 1);
        assert!(!l[0].is_complete());
    }

    #[test]
    fn shape_fit_recovers_a_planted_bias_and_sharpening_walk_forward() {
        // 70 resolved ladders priced from Normal(30, 1.2) whose highs actually come from
        // Normal(30.3, 0.9): the market runs 0.3 °C cold and is too wide. Deterministic highs
        // from a fixed sequence of z-scores so the test is exact.
        let zs = [-1.4, -0.8, -0.3, 0.0, 0.3, 0.8, 1.4];
        let mut rows = Vec::new();
        for i in 0..70 {
            let day = d("2026-07-01") + chrono::Duration::days(i as i64);
            let high = 30.3 + 0.9 * zs[i % zs.len()];
            rows.extend(synthetic(
                30.0,
                1.2,
                &day.to_string(),
                &(day - chrono::Duration::days(1)).to_string(),
                Some(high),
            ));
        }
        let ladders = build_ladders(rows);
        assert_eq!(ladders.len(), 70);
        let today = d("2026-09-10");
        let hist = shape_history(&ladders, "kalshi", today);
        assert_eq!(hist.len(), 70);
        let p = fit_shape(&hist).unwrap();
        assert!(p.bias_c > 0.15 && p.bias_c < 0.5, "bias {}", p.bias_c);
        assert!(p.sigma_scale < 0.9, "scale {}", p.sigma_scale);
        // Causality: as of the 30th day only 29 resolved targets are visible, under MIN_LADDERS.
        assert_eq!(shape_history(&ladders, "kalshi", d("2026-07-30")).len(), 29);
        assert_eq!(fit_shape_as_of(&ladders, "kalshi", d("2026-07-30")), None);
        // A same-day target is not evidence yet.
        assert_eq!(shape_history(&ladders, "kalshi", d("2026-07-31")).len(), 30);
        // Another venue sees nothing.
        assert!(shape_history(&ladders, "polymarket", today).is_empty());
        // The walk-forward estimate map covers every cell of every complete ladder captured on a
        // day with enough history, and none before.
        let est = walk_forward_estimates(&ladders, "kalshi");
        let first_day_with_fit = d("2026-07-01") + chrono::Duration::days(MIN_LADDERS as i64 + 1);
        assert!(est
            .keys()
            .all(|(_, cap)| *cap >= first_day_with_fit - chrono::Duration::days(1)));
        assert!(!est.is_empty());
        // The replay trades only the ladders after the fit exists, settles each at its outcome,
        // and — since the planted highs really do come from the re-shaped distribution — makes
        // money net of a Kalshi-style fee.
        let fee = |p: f64| 0.07 * p * (1.0 - p);
        let trades = replay(
            &ladders,
            "kalshi",
            d("2026-01-01"),
            d("2027-01-01"),
            fee,
            0.03,
            MIN_PRICE,
        );
        assert!(!trades.is_empty());
        assert!(trades.iter().all(|t| t.target >= first_day_with_fit));
        let (roi, n) = replay_roi(&trades);
        assert_eq!(n, trades.len());
        assert!(roi > 0.0, "roi {roi} on {n}");
        // A window with no resolved ladders is an empty replay, not a crash.
        assert!(replay(
            &ladders,
            "kalshi",
            d("2027-01-01"),
            d("2027-02-01"),
            fee,
            0.03,
            MIN_PRICE
        )
        .is_empty());
        assert_eq!(replay_roi(&[]), (0.0, 0));
    }

    #[test]
    fn decide_cell_trades_both_sides_net_of_fee_above_the_floor() {
        let fee = |p: f64| 0.07 * p * (1.0 - p);
        // Under-priced favourite: prob 0.60 vs ask 0.50 → BUY YES, edge 0.10 − fee.
        let dcs = decide_cell(0.60, Some(0.48), Some(0.50), fee, 0.03, MIN_PRICE).unwrap();
        assert_eq!(dcs.side, ShapeSide::BuyYes);
        assert!((dcs.price - 0.50).abs() < 1e-12);
        assert!((dcs.edge - (0.10 - fee(0.50))).abs() < 1e-12);
        // Over-priced tail: prob 0.05 vs bid 0.15 → BUY NO at 0.85.
        let dcs = decide_cell(0.05, Some(0.15), Some(0.17), fee, 0.03, MIN_PRICE).unwrap();
        assert_eq!(dcs.side, ShapeSide::BuyNo);
        assert!((dcs.price - 0.85).abs() < 1e-12 && (dcs.yes_price - 0.15).abs() < 1e-12);
        // Same tail with a 9¢ bid: below the floor, no trade.
        assert_eq!(
            decide_cell(0.02, Some(0.09), Some(0.11), fee, 0.03, MIN_PRICE),
            None
        );
        // Edge inside the threshold: no trade. No book: no trade.
        assert_eq!(
            decide_cell(0.52, Some(0.49), Some(0.50), fee, 0.03, MIN_PRICE),
            None
        );
        assert_eq!(decide_cell(0.60, None, None, fee, 0.03, MIN_PRICE), None);
        // A 0 / 1 quote is an empty side.
        assert_eq!(
            decide_cell(0.60, Some(0.0), Some(1.0), fee, 0.03, MIN_PRICE),
            None
        );
    }
}
