#!/usr/bin/env python3
"""Why is edge-shrinkage lambda ~0.35, and what would raise it?

Lambda is the through-origin slope of realized (outcome - price) on claimed (model - price) over
resolved lead >= 1 captures -- the fraction of each claimed edge that survives the outcome. It has
decayed from ~0.6 (early July 2026) to ~0.35, and the trailing-14d slope sits near the pilot's 0.2
circuit breaker, so the question is which part of the pricing chain is leaking.

Lambda falls for exactly three reasons, and this script separates them:

  1. OVERCONFIDENCE -- the estimate is too extreme for its own accuracy. A model that says 0.02 when
     the truth is 0.15 claims a large edge that cannot realize. Diagnosed by the calibration slope
     (regress outcome on estimate) and directly by the SIGMA SWEEP: re-price every capture under
     N(forecast_high, c * forecast_sigma) and find the c that maximizes lambda / minimizes Brier.
     c > 1 means the pricing sigma is too tight and the fix is a wider sigma, not a better mean.
  2. BIAS -- the point forecast is off, so estimates skew one way. Shows up as a lambda that differs
     sharply between BUY-side and SELL-side claims, and in the residual mean when realized highs are
     recoverable.
  3. HETEROGENEITY -- lambda is genuinely high in some segments and negative in others, so the
     pooled fit understates the tradeable part. Diagnosed by the per-segment breakdown; the fix is a
     filter, not a model change.

Also tests the two logged-but-unused candidates as sigma replacements: the ECMWF/GFS ENSEMBLE SPREAD
(day-specific uncertainty vs the per-(city, lead) constant) and the AI model highs (AIFS/AIGFS) as
alternate means. Both are evaluated walk-forward -- fitted on strictly earlier target days -- because
an in-sample sigma always improves Brier and would prove nothing.

First look (2026-07-26, 1864 resolved lead >= 1 rows): lambda is NOT a tuning parameter -- it is a
scoreboard of how much sharper the model is than the market, and it tracks that skill gap almost
one-for-one by week:

    week of      n   model Brier  market Brier   edge    lambda
    2026-06-29  122    0.0654       0.0921     +0.0267   +0.851
    2026-07-06  651    0.1072       0.1044     -0.0028   +0.451
    2026-07-13  608    0.1106       0.0888     -0.0218   +0.087
    2026-07-20  483    0.0942       0.0917     -0.0025   +0.415

Both "fix the pricing" hypotheses are DEAD on this sample:
  * Overconfidence: the sigma sweep is Brier-optimal at c = 1.0 exactly (0.1049 vs 0.1052 at 0.8,
    0.1058 at 1.2), and widening sigma LOWERS lambda (0.305 -> 0.192 at c = 2.0). The estimates are
    also well calibrated in aggregate -- model 0.0-0.1 realizes 0.043, 0.2-0.3 realizes 0.257,
    through-origin slope 0.854, mean estimate 0.132 vs mean outcome 0.126.
  * Bias: the 2026-07-20 seasonal refit worked. Kalshi's weekly residual went -0.80/-1.18 -> -0.04
    degC and Polymarket's -0.55/-0.57 -> +0.44. A walk-forward bias correction on top now moves
    Brier only 0.1026 -> 0.1020 and LOWERS lambda (0.198 -> 0.165).

What is left is HETEROGENEITY, and the dominant axis is not the one the strategy currently filters
on. Sub-10c markets carry negative lambda on BOTH sides while everything above 10c is healthy:

    BUY  px < 0.10   lambda -0.086  n=778      BUY  px >= 0.10   lambda +0.363  n=219
    SELL px < 0.10   lambda -0.105  n=203      SELL px >= 0.10   lambda +0.540  n=664

That reframes the standing "BUYs are worthless" result: the dead segment is the TAIL (981 of 1864
rows), not the buy side, and the model's own Normal tail is where it has no information. Shape is
the second axis -- temp_at_most +0.671 / temp_at_least +0.366 / temp_bucket +0.239 -- and city
spread is large (kalshi/Miami +0.768, kalshi/Austin +0.750 vs kalshi/Chicago -0.250,
polymarket/Istanbul -0.618). All of this is WITHIN venue, yet lambda is currently fitted per venue
only (kalshi 0.341 vs polymarket 0.391), so the fit averages healthy and anti-signal segments
together. Conditioning ShrinkageFit on (venue, segment) would zero out the bad segments by
construction instead of needing a hand-picked filter.

Neither logged sharpening candidate is ready: AIFS RMSE 2.31 degC vs the blend's 1.46 on the same
rows (AIGFS n=28, too few), and ensemble spread has only 67 usable rows -- both stay log-only.

Stdlib only, no deps.

Usage: python3 scripts/lambda_diagnostics.py [--captures data/captures.jsonl] [--since YYYY-MM-DD]
"""
import argparse
import collections
import datetime
import json
import math

MIN_N = 40  # matches backtesting::ShrinkageFit::MIN_N


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def to_c(v, unit):
    return (v - 32.0) / 1.8 if (unit or "F") == "F" else v


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def lead(r):
    return (
        datetime.date.fromisoformat(r["target_date"])
        - datetime.date.fromisoformat(r["captured_at"])
    ).days


def ref_price(r):
    """Mirrors weather_dashboard::ref_price -- book mid when sane, else last trade."""
    b, a = r.get("best_bid"), r.get("best_ask")
    px = (a + b) / 2.0 if (b is not None and a is not None and b > 0 and a < 1 and b <= a) else r.get("entry_price")
    return px if (px is not None and 0.0 < px < 1.0) else None


def slope(obs):
    """Through-origin OLS slope of realized on claimed, plus n."""
    sxx = sum(x * x for x, _ in obs)
    sxy = sum(x * y for x, y in obs)
    return (sxy / sxx if sxx > 0 else float("nan")), len(obs)


def price_shape(r, mu_c, sigma_c):
    """Re-price a market shape under N(mu_c, sigma_c), matching the round-half-up bucket convention.

    The reported high is a whole degree in the market's unit, so a bucket [lo, hi] wins when the
    true high falls in [lo - 0.5, hi + 0.5) before unit conversion.
    """
    if sigma_c <= 0:
        return None
    mt, unit = r["market_type"], r.get("unit")
    lo = r.get("threshold")
    if lo is None:
        return None
    z = lambda edge_c: norm_cdf((edge_c - mu_c) / sigma_c)
    if mt == "temp_bucket":
        hi = r.get("threshold_upper")
        hi = lo if hi is None else hi
        return z(to_c(hi + 0.5, unit)) - z(to_c(lo - 0.5, unit))
    if mt == "temp_at_least":
        return 1.0 - z(to_c(lo - 0.5, unit))
    if mt == "temp_at_most":
        return z(to_c(lo + 0.5, unit))
    return None


def usable(rows, since=None):
    """Resolved lead >= 1 temperature captures with a usable price -- the lambda fit population."""
    out = []
    for r in rows:
        if r.get("outcome") is None or r.get("model_estimate") is None:
            continue
        if not r.get("market_type", "").startswith("temp"):
            continue
        if lead(r) < 1:
            continue
        if since and r["target_date"] < since:
            continue
        px = ref_price(r)
        if px is None:
            continue
        r = dict(r, _px=px, _lead=lead(r))
        out.append(r)
    return out


def obs_of(rows, est_key="model_estimate"):
    return [(r[est_key] - r["_px"], r["outcome"] - r["_px"]) for r in rows]


def band(x):
    a = abs(x)
    if a < 0.05:
        return "  <5%"
    if a < 0.10:
        return " 5-10%"
    if a < 0.20:
        return "10-20%"
    if a < 0.35:
        return "20-35%"
    return " >=35%"


def report_segments(rows):
    print("\n== 3. HETEROGENEITY: lambda by segment (n >= %d shown) ==" % MIN_N)
    segs = {
        "venue": lambda r: r.get("source", "polymarket"),
        "side (claim direction)": lambda r: "model ABOVE price (BUY)" if r["model_estimate"] > r["_px"] else "model BELOW price (SELL)",
        "lead": lambda r: f"lead {r['_lead']}",
        "claimed edge band": lambda r: band(r["model_estimate"] - r["_px"]),
        "market type": lambda r: r["market_type"],
        "price band": lambda r: "px <0.10" if r["_px"] < 0.10 else ("px 0.10-0.35" if r["_px"] < 0.35 else ("px 0.35-0.65" if r["_px"] < 0.65 else "px >=0.65")),
    }
    for name, key in segs.items():
        groups = collections.defaultdict(list)
        for r in rows:
            groups[key(r)].append(r)
        print(f"\n  -- by {name} --")
        rowsout = []
        for k, v in groups.items():
            lam, n = slope(obs_of(v))
            if n >= MIN_N:
                rowsout.append((k, lam, n))
        for k, lam, n in sorted(rowsout, key=lambda t: -t[1]):
            print(f"     {str(k):26s} lambda={lam:+.3f}  n={n}")


def report_cities(rows):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r.get("source", "polymarket"), r["city"])].append(r)
    out = []
    for k, v in groups.items():
        lam, n = slope(obs_of(v))
        if n >= MIN_N:
            out.append((k, lam, n))
    out.sort(key=lambda t: -t[1])
    print("\n  -- by (venue, city), best and worst --")
    for k, lam, n in out[:6]:
        print(f"     {k[0]+'/'+k[1]:26s} lambda={lam:+.3f}  n={n}")
    print("     ...")
    for k, lam, n in out[-6:]:
        print(f"     {k[0]+'/'+k[1]:26s} lambda={lam:+.3f}  n={n}")


def report_calibration(rows):
    print("\n== 1a. OVERCONFIDENCE: reliability of the model estimate ==")
    print("  model says p, outcomes realize at:")
    buckets = collections.defaultdict(list)
    for r in rows:
        buckets[min(int(r["model_estimate"] * 10), 9)].append(r["outcome"])
    print(f"     {'model p':>12s} {'realized':>9s} {'n':>6s}")
    for b in sorted(buckets):
        v = buckets[b]
        print(f"     {b/10:.1f} - {b/10+0.1:.1f}  {sum(v)/len(v):9.3f} {len(v):6d}")
    # Through-origin slope of outcome on estimate: <1 means estimates are too extreme.
    sxx = sum(r["model_estimate"] ** 2 for r in rows)
    sxy = sum(r["model_estimate"] * r["outcome"] for r in rows)
    print(f"\n  through-origin slope of outcome on estimate: {sxy/sxx:.3f}  (1.0 = calibrated)")
    print(f"  mean estimate {sum(r['model_estimate'] for r in rows)/len(rows):.3f}"
          f"  vs mean outcome {sum(r['outcome'] for r in rows)/len(rows):.3f}")


def sigma_sweep(rows):
    """Re-price under N(mu, c*sigma) and report lambda / Brier vs the scale c."""
    have = [r for r in rows if r.get("forecast_high") is not None and r.get("forecast_sigma")]
    print(f"\n== 1b. OVERCONFIDENCE: sigma sweep ({len(have)} of {len(rows)} rows carry mu/sigma) ==")
    if not have:
        return
    # Reproduction check: does price_shape at c=1 recover the stored estimate?
    err = [abs(price_shape(r, r["forecast_high"], r["forecast_sigma"]) - r["model_estimate"])
           for r in have if price_shape(r, r["forecast_high"], r["forecast_sigma"]) is not None]
    print(f"  reproduction check at c=1.0: median |recomputed - stored| = {sorted(err)[len(err)//2]:.4f}"
          f"  (max {max(err):.4f}, n={len(err)})")
    if sorted(err)[len(err) // 2] > 0.02:
        print("  WARNING: poor reproduction -- the sweep below is unreliable")
    print(f"\n  {'c':>5s} {'lambda':>8s} {'Brier':>8s} {'mean|edge|':>11s}")
    base_brier = sum((r["model_estimate"] - r["outcome"]) ** 2 for r in have) / len(have)
    print(f"  {'stored':>5s} {slope(obs_of(have))[0]:8.3f} {base_brier:8.4f}"
          f" {sum(abs(r['model_estimate']-r['_px']) for r in have)/len(have):11.4f}")
    best = None
    for c in [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.25, 2.5, 3.0]:
        obs, sq, edges = [], [], []
        for r in have:
            p = price_shape(r, r["forecast_high"], r["forecast_sigma"] * c)
            if p is None:
                continue
            obs.append((p - r["_px"], r["outcome"] - r["_px"]))
            sq.append((p - r["outcome"]) ** 2)
            edges.append(abs(p - r["_px"]))
        lam, n = slope(obs)
        brier = sum(sq) / len(sq)
        print(f"  {c:5.2f} {lam:8.3f} {brier:8.4f} {sum(edges)/len(edges):11.4f}")
        if best is None or brier < best[2]:
            best = (c, lam, brier)
    print(f"\n  best Brier at c={best[0]} (lambda {best[1]:.3f}, Brier {best[2]:.4f} vs stored {base_brier:.4f})")
    print("  NOTE: a wider sigma shrinks every claimed edge, so lambda rising here means the model's")
    print("  disagreements were inflated by overconfidence rather than being genuinely wrong.")


def market_brier(rows):
    mb = sum((r["_px"] - r["outcome"]) ** 2 for r in rows) / len(rows)
    modb = sum((r["model_estimate"] - r["outcome"]) ** 2 for r in rows) / len(rows)
    print(f"\n  reference: market Brier {mb:.4f} vs model Brier {modb:.4f} on the same {len(rows)} rows")


def ensemble_sigma(rows):
    """Does the logged ensemble spread beat the constant sigma, walk-forward?"""
    have = [r for r in rows
            if r.get("forecast_high") is not None and r.get("forecast_sigma")
            and r.get("ensemble_spread_ecmwf") is not None]
    print(f"\n== SIGMA CANDIDATE: ensemble spread ({len(have)} rows carry it) ==")
    if len(have) < 2 * MIN_N:
        print("  too few rows carrying spread to fit -- accrue more captures")
        return
    sp = lambda r: (r["ensemble_spread_ecmwf"] + (r.get("ensemble_spread_gfs") or r["ensemble_spread_ecmwf"])) / 2.0
    lo = sorted(sp(r) for r in have)
    print(f"  spread distribution (degC): p10={lo[len(lo)//10]:.2f} median={lo[len(lo)//2]:.2f} p90={lo[9*len(lo)//10]:.2f}")
    # Is spread informative about |error| at all? Compare Brier in spread terciles under constant sigma.
    t1, t2 = lo[len(lo) // 3], lo[2 * len(lo) // 3]
    for name, sel in [("low spread", lambda s: s <= t1), ("mid", lambda s: t1 < s <= t2), ("high spread", lambda s: s > t2)]:
        sub = [r for r in have if sel(sp(r))]
        if not sub:
            continue
        b = sum((r["model_estimate"] - r["outcome"]) ** 2 for r in sub) / len(sub)
        lam, n = slope(obs_of(sub))
        print(f"  {name:12s} n={n:4d}  model Brier={b:.4f}  lambda={lam:+.3f}")
    # Walk-forward: sigma = a * spread, a fitted on strictly earlier target days.
    have.sort(key=lambda r: r["target_date"])
    dates = sorted({r["target_date"] for r in have})
    hist = []
    new_sq, old_sq, new_obs = [], [], []
    for d in dates:
        day = [r for r in have if r["target_date"] == d]
        if len(hist) >= MIN_N:
            # Fit the scale a minimizing Brier over history, by coarse search.
            best_a, best_v = None, None
            for a in [x / 10.0 for x in range(4, 31)]:
                v = 0.0
                for h in hist:
                    p = price_shape(h, h["forecast_high"], max(a * sp(h), 0.2))
                    if p is not None:
                        v += (p - h["outcome"]) ** 2
                if best_v is None or v < best_v:
                    best_a, best_v = a, v
            for r in day:
                p = price_shape(r, r["forecast_high"], max(best_a * sp(r), 0.2))
                if p is None:
                    continue
                new_sq.append((p - r["outcome"]) ** 2)
                old_sq.append((r["model_estimate"] - r["outcome"]) ** 2)
                new_obs.append((p - r["_px"], r["outcome"] - r["_px"]))
        hist.extend(day)
    if new_sq:
        lam, n = slope(new_obs)
        print(f"\n  walk-forward spread-sigma: Brier {sum(new_sq)/len(new_sq):.4f} vs constant-sigma"
              f" {sum(old_sq)/len(old_sq):.4f} on the same {len(new_sq)} rows; lambda {lam:+.3f}")
        print("  (ship a dynamic sigma only if this Brier beats the constant one out-of-sample)")
    else:
        print("  not enough history to walk forward yet")


def ai_means(rows):
    """Do the logged AI model highs beat the blend's forecast_high as the mean?"""
    print("\n== MEAN CANDIDATE: AI model highs vs the blend ==")
    truth = realized_highs(rows)
    recs = []
    for r in rows:
        k = (r.get("source", "polymarket"), r["city"], r["target_date"])
        if k not in truth or r.get("forecast_high") is None:
            continue
        recs.append((r, truth[k]))
    if not recs:
        print("  no recoverable realized highs")
        return
    # One observation per (venue, city, target-day, lead) so a wide ladder can't dominate.
    seen, ded = set(), []
    for r, t in recs:
        k = (r.get("source", "polymarket"), r["city"], r["target_date"], r["_lead"])
        if k in seen:
            continue
        seen.add(k)
        ded.append((r, t))
    print(f"  {len(ded)} deduped (venue, city, day, lead) observations with a recovered high")
    for name, key in [("blend forecast_high", "forecast_high"),
                      ("AIFS", "forecast_high_aifs"),
                      ("AIGFS", "forecast_high_aigfs"),
                      # Member means (logged since 2026-08-09): the standard result is that an
                      # ensemble mean beats the deterministic run at these leads -- if one beats
                      # the blend here, it is a real mu candidate (same gate as the AI means).
                      ("ENS mean (ECMWF)", "ensemble_mean_ecmwf"),
                      ("ENS mean (GEFS)", "ensemble_mean_gfs")]:
        errs = [t - to_c(r[key], "C") for r, t in ded if r.get(key) is not None]
        if len(errs) < MIN_N:
            print(f"  {name:22s} n={len(errs)} -- too few")
            continue
        m = sum(errs) / len(errs)
        rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
        print(f"  {name:22s} n={len(errs):4d}  bias={m:+.2f}degC  RMSE={rmse:.2f}degC")
    # Blend members (logged since 2026-08-09): per-model error on the same recovered highs -- the
    # raw material for a fitted re-weighting of BLEND_MODELS. Report-only; a weight fit needs
    # enough rows to be walk-forward-validated first.
    per = {}
    for r, t in ded:
        for name, v in (r.get("blend_model_highs") or {}).items():
            per.setdefault(name, []).append(t - v)
    for name in sorted(per):
        errs = per[name]
        if len(errs) < MIN_N:
            print(f"  member {name:18s} n={len(errs)} -- too few")
            continue
        m = sum(errs) / len(errs)
        rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
        print(f"  member {name:18s} n={len(errs):4d}  bias={m:+.2f}degC  RMSE={rmse:.2f}degC")

    # Simple average of blend + both AI models, where all three exist.
    trio = [(r, t) for r, t in ded
            if r.get("forecast_high_aifs") is not None and r.get("forecast_high_aigfs") is not None]
    if len(trio) >= MIN_N:
        errs = [t - (r["forecast_high"] + r["forecast_high_aifs"] + r["forecast_high_aigfs"]) / 3.0 for r, t in trio]
        m = sum(errs) / len(errs)
        rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
        base = [t - r["forecast_high"] for r, t in trio]
        brmse = math.sqrt(sum(e * e for e in base) / len(base))
        print(f"  {'equal-weight trio':22s} n={len(trio):4d}  bias={m:+.2f}degC  RMSE={rmse:.2f}degC"
              f"   (blend alone on same rows: {brmse:.2f})")
    print("  NOTE: CLAUDE.md gate -- AI models must beat BLEND_MODELS on accrued captures BEFORE")
    print("  entering the blend, since changing the blend invalidates the fitted sigma/bias tables.")


def weekly_skill(rows):
    """Model Brier vs market Brier, and lambda, per target week.

    The single most important series here: lambda is the slope of realized on claimed edge, so it
    can only be high if the model is SHARPER than the market. If the model's Brier edge over the
    market has eroded, lambda decay is a skill problem, not a shrinkage-parameter problem.
    """
    print("\n== 4. DECAY: model vs market skill by target week ==")
    wk = lambda r: (datetime.date.fromisoformat(r["target_date"])
                    - datetime.timedelta(days=datetime.date.fromisoformat(r["target_date"]).weekday())).isoformat()
    groups = collections.defaultdict(list)
    for r in rows:
        groups[wk(r)].append(r)
    print(f"  {'week of':>12s} {'n':>5s} {'model':>8s} {'market':>8s} {'edge':>8s} {'lambda':>8s}")
    for w in sorted(groups):
        v = groups[w]
        if len(v) < 30:
            continue
        mod = sum((r["model_estimate"] - r["outcome"]) ** 2 for r in v) / len(v)
        mkt = sum((r["_px"] - r["outcome"]) ** 2 for r in v) / len(v)
        lam, n = slope(obs_of(v))
        print(f"  {w:>12s} {n:5d} {mod:8.4f} {mkt:8.4f} {mkt-mod:+8.4f} {lam:+8.3f}")
    print("  edge > 0 means the model beat the market that week.")


def bias_drift(rows):
    """Is the point forecast drifting warm/cold since the last refit -- and does fixing it raise lambda?

    A stale bias is the one defect that manufactures edge in a SYSTEMATIC direction: if the forecast
    runs warm, every high bucket looks over-priced (fake SELL edge) and every low bucket looks
    under-priced. Unlike noise, it survives averaging, so it lands straight in lambda.

    The walk-forward test re-prices each target day with a per-venue bias fitted ONLY on strictly
    earlier resolved days. If that raises lambda and lowers Brier, the fix is refitting the station
    tables (or tracking bias online) rather than anything about the shrinkage itself.
    """
    print("\n== 2. BIAS: residual drift of the point forecast ==")
    truth = realized_highs(rows)
    seen, recs = set(), []
    for r in rows:
        k = (r.get("source", "polymarket"), r["city"], r["target_date"])
        if k not in truth or r.get("forecast_high") is None or not r.get("forecast_sigma"):
            continue
        dk = k + (r["_lead"],)
        if dk in seen:
            continue
        seen.add(dk)
        recs.append((r, truth[k]))
    if len(recs) < MIN_N:
        print(f"  only {len(recs)} recoverable observations -- too few")
        return
    print(f"  {len(recs)} deduped (venue, city, day, lead) residuals; residual = realized - forecast")
    for v in sorted({r.get("source", "polymarket") for r, _ in recs}):
        sub = [(r, t) for r, t in recs if r.get("source", "polymarket") == v]
        print(f"\n  -- {v} --")
        byw = collections.defaultdict(list)
        for r, t in sub:
            w = datetime.date.fromisoformat(r["target_date"])
            byw[(w - datetime.timedelta(days=w.weekday())).isoformat()].append(t - r["forecast_high"])
        for w in sorted(byw):
            e = byw[w]
            if len(e) < 8:
                continue
            print(f"     week {w}  n={len(e):3d}  mean residual {sum(e)/len(e):+.2f} degC")

    # Walk-forward per-venue bias correction.
    recs.sort(key=lambda rt: rt[0]["target_date"])
    dates = sorted({r["target_date"] for r, _ in recs})
    hist = collections.defaultdict(list)
    new_sq, old_sq, new_obs, old_obs = [], [], [], []
    all_rows_by_date = collections.defaultdict(list)
    for r in rows:
        if r.get("forecast_high") is not None and r.get("forecast_sigma"):
            all_rows_by_date[r["target_date"]].append(r)
    for d in dates:
        for r in all_rows_by_date.get(d, []):
            v = r.get("source", "polymarket")
            if len(hist[v]) < MIN_N:
                continue
            b = sum(hist[v]) / len(hist[v])
            p = price_shape(r, r["forecast_high"] + b, r["forecast_sigma"])
            if p is None:
                continue
            new_sq.append((p - r["outcome"]) ** 2)
            old_sq.append((r["model_estimate"] - r["outcome"]) ** 2)
            new_obs.append((p - r["_px"], r["outcome"] - r["_px"]))
            old_obs.append((r["model_estimate"] - r["_px"], r["outcome"] - r["_px"]))
        for r, t in recs:
            if r["target_date"] == d:
                hist[r.get("source", "polymarket")].append(t - r["forecast_high"])
    if new_sq:
        ln, _ = slope(new_obs)
        lo, _ = slope(old_obs)
        print(f"\n  walk-forward bias correction on {len(new_sq)} rows:")
        print(f"     Brier  {sum(old_sq)/len(old_sq):.4f} -> {sum(new_sq)/len(new_sq):.4f}")
        print(f"     lambda {lo:+.3f} -> {ln:+.3f}")
        print("  If lambda rises materially here, the decay is stale station tables -- refit them")
        print("  (scripts/refit_station_tables.py) rather than touching the shrinkage.")
    else:
        print("\n  not enough history to walk forward yet")


def realized_highs(rows):
    """Recover each (venue, city, day)'s realized high from the unique winning bucket."""
    groups = collections.defaultdict(dict)
    for r in rows:
        if r.get("market_type") == "temp_bucket" and r.get("outcome") is not None:
            groups[(r.get("source", "polymarket"), r["city"], r["target_date"])][r["market_id"]] = r
    truth = {}
    for key, best in groups.items():
        winners = [r for r in best.values() if r["outcome"] == 1.0]
        if len(winners) == 1:
            w = winners[0]
            lo = w["threshold"]
            hi = w.get("threshold_upper") or lo
            truth[key] = to_c((lo + hi) / 2.0, w.get("unit"))
    return truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", default="data/captures.jsonl")
    ap.add_argument("--since", default=None, help="only target dates >= this (YYYY-MM-DD)")
    a = ap.parse_args()

    raw = load(a.captures)
    rows = usable(raw, a.since)
    print(f"{len(raw)} captures -> {len(rows)} resolved lead>=1 temperature rows with a usable price")
    lam, n = slope(obs_of(rows))
    print(f"pooled lambda = {lam:.3f} over n={n}")
    for v in sorted({r.get("source", "polymarket") for r in rows}):
        sub = [r for r in rows if r.get("source", "polymarket") == v]
        l2, n2 = slope(obs_of(sub))
        print(f"  {v:12s} lambda = {l2:.3f}  n={n2}")
    market_brier(rows)

    report_calibration(rows)
    sigma_sweep(rows)
    bias_drift(rows)
    report_segments(rows)
    report_cities(rows)
    weekly_skill(rows)
    ensemble_sigma(rows)
    ai_means(rows)


if __name__ == "__main__":
    main()
