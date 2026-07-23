#!/usr/bin/env python3
"""Are forecast residuals fat-tailed enough that the Normal bucket CDF misprices tails?

The pricer turns (mu, sigma) into bucket probabilities with a NORMAL CDF
(backtest_engine::market_estimate / the station nowcast). If the true residual
distribution has fat tails, far buckets are systematically underpriced — one candidate
explanation for a measured house result: BUY-longshot trades break even while SELLs
carry the PnL. The strongest public competitor prices with Student-t distributions.
This script measures, on accrued resolved captures, whether t tails beat Normal tails
BEFORE anything in the pricing path changes (same gating as every other candidate).

Method (stdlib only, no deps):
  1. Truth: realized high per (venue, city, target-day) from the unique WINNING
     temp_bucket market (bucket midpoint -> degC) — refit_station_tables.py hygiene.
     CRUCIALLY, days whose high landed OFF the captured bucket ladder are not dropped:
     they resolve YES on an open-ended temp_at_least/temp_at_most market instead, and
     dropping them would censor exactly the tail the script is fitting (the censoring
     is outcome-dependent — a day goes missing precisely when the high lands far from
     forecast). Those days enter as one-sided/interval bounds (winning at_least T =>
     realized >= T - 0.5 in the market's unit; at_most T => realized <= T + 0.5;
     conservative half-degree widening, tightest bound across winners).
  2. z = (realized - forecast_high) / forecast_sigma per resolved lead 1-2 temp row,
     deduped to one obs per (venue, city, target, lead). forecast_high/sigma are the
     stored pricing inputs, so z measures exactly the standardization the CDF assumes.
     Bounded days contribute standardized censoring intervals instead of a point z.
  3. Censoring-aware MLE grid fit of z ~ s * t_nu (nu = inf recovers a rescaled
     Normal): exact obs contribute log pdf, censored obs log(F(hi) - F(lo)); profile
     the scale s per nu, compare log-likelihoods, and count |z| > 2 / > 3 (exact plus
     PROVABLE censored minima) against each fit's expectation.
  4. Decision test: re-price every resolved lead 1-2 temp market from its stored
     (forecast_high, forecast_sigma, threshold(s), unit) under
        normal        the shipping convention (sanity: matches stored model_estimate)
        t-matched     t_nu, variance-matched to sigma (tail shape changes, width doesn't)
        t-fitted      s * t_nu as fitted (tails AND recalibrated width)
        normal-s      s_norm * Normal (width recalibrated, Gaussian tails — the fair
                      baseline for t-fitted: is it the tails or just the s?)
     and compare Brier overall + on the tail slice (normal p < 0.05 or > 0.95), with a
     walk-forward split (fit on the older half of target days, score on the newer).
     Bucket widening mirrors market_estimate: +-0.5 in the market's unit, clip 0.01/0.99.

First look (2026-07-23, censoring-aware: 240 exact + 29 censored obs / 1,503 market
rows; the first cut of this script DROPPED off-ladder days and adversarial review
caught it — outcome-dependent truncation of exactly the tail being fitted): the
LIKELIHOOD strongly prefers t(nu~2.5-3, s~0.85) over Normal (s=1.40), dLL +35.8 — but
the shape is a peaky center with rare blowups, NOT uniformly fat shoulders (|z|>2:
>=17 provable incl. censored bounds vs 41.2 the Normal fit expects), and the PRICING
effect is negligible: walk-forward Brier 0.1051 (normal) vs 0.1046 (t-fitted), with
the tail-slice gain coming from the scale recalibration, not the tail shape (normal-s
matched it). The 0.01/0.99 clip already absorbs what the tails would change. VERDICT:
leave the Normal CDF alone; the longshot break-even result is not a CDF-shape problem.
Caveats: obs are correlated across venue/lead pairs sharing a physical day (collapsing
to independent city-days reproduces the same best nu, dLL ~ -7%), and nu rides the
grid edge (extending below 2.5 changes nothing to 4 decimals of Brier). Re-run after
major regime changes or if the clip bounds ever move.

Usage: python3 scripts/fit_tail_distribution.py [--captures data/captures.jsonl]
                                                [--station-only]  (drop legacy sigma=2.0 rows)
"""

import argparse
import collections
import datetime as dt
import json
import math

BUCKET_HALF_WIDTH = 0.5
CLIP_LO, CLIP_HI = 0.01, 0.99
NU_GRID = [2.5, 3, 3.5, 4, 5, 6, 8, 10, 12, 15, 20, 30, 50, 100, float("inf")]
S_GRID = [0.50 + 0.01 * i for i in range(151)]  # 0.50 .. 2.00


def to_c(v, unit):
    return (v - 32.0) / 1.8 if (unit or "F") == "F" else v


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def lead(r):
    return (
        dt.date.fromisoformat(r["target_date"]) - dt.date.fromisoformat(r["captured_at"])
    ).days


def realized_highs(rows):
    """{(venue, city, target): ("exact", high_c) | ("interval", lo_c|None, hi_c|None)}.

    Exact from the unique winning bucket; otherwise censoring bounds from winning
    open-ended markets (tightest across winners, +-0.5 widened in the market's unit).
    """
    groups = collections.defaultdict(dict)
    for r in rows:
        if r["market_type"].startswith("temp") and r.get("outcome") is not None:
            key = (r.get("source", "polymarket"), r["city"], r["target_date"])
            groups[key][r["market_id"]] = r
    truth = {}
    for key, best in groups.items():
        winners = [r for r in best.values() if r["outcome"] == 1.0]
        buckets = [r for r in winners if r["market_type"] == "temp_bucket"]
        if len(buckets) == 1:
            w = buckets[0]
            lo = w["threshold"]
            hi = w["threshold_upper"] if w.get("threshold_upper") is not None else lo
            truth[key] = ("exact", to_c((lo + hi) / 2.0, w.get("unit")))
            continue
        lo_bound, hi_bound = None, None
        for w in winners:
            if w["market_type"] == "temp_at_least":
                b = to_c(w["threshold"] - BUCKET_HALF_WIDTH, w.get("unit"))
                lo_bound = b if lo_bound is None else max(lo_bound, b)
            elif w["market_type"] == "temp_at_most":
                b = to_c(w["threshold"] + BUCKET_HALF_WIDTH, w.get("unit"))
                hi_bound = b if hi_bound is None else min(hi_bound, b)
        if lo_bound is not None or hi_bound is not None:
            truth[key] = ("interval", lo_bound, hi_bound)
    return truth


# ---------- Student-t machinery (Numerical-Recipes-style incomplete beta) ----------


def betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3e-12, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    bt = math.exp(ln_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * betacf(a, b, x) / a
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b


def t_cdf(t, nu):
    if math.isinf(nu):
        return 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))
    p = 0.5 * betai(nu / 2.0, 0.5, nu / (nu + t * t))
    return 1.0 - p if t > 0 else p


def t_logpdf(z, nu, s):
    if math.isinf(nu):
        return -0.5 * math.log(2.0 * math.pi) - math.log(s) - 0.5 * (z / s) ** 2
    return (
        math.lgamma((nu + 1.0) / 2.0)
        - math.lgamma(nu / 2.0)
        - 0.5 * math.log(nu * math.pi)
        - math.log(s)
        - (nu + 1.0) / 2.0 * math.log1p((z / s) ** 2 / nu)
    )


def fit_scaled_t(zs, cens=()):
    """{nu: (best s, loglik at best s)} over the grid.

    zs are exact standardized obs (log pdf terms); cens are (zlo|None, zhi|None)
    standardized censoring intervals contributing log(F(zhi) - F(zlo)) — dropping
    them would truncate exactly the tail being fitted.
    """

    def cens_ll(zlo, zhi, nu, s):
        p_hi = t_cdf(zhi / s, nu) if zhi is not None else 1.0
        p_lo = t_cdf(zlo / s, nu) if zlo is not None else 0.0
        return math.log(max(p_hi - p_lo, 1e-300))

    out = {}
    for nu in NU_GRID:
        best = max(
            (
                sum(t_logpdf(z, nu, s) for z in zs)
                + sum(cens_ll(zlo, zhi, nu, s) for zlo, zhi in cens),
                s,
            )
            for s in S_GRID
        )
        out[nu] = (best[1], best[0])
    return out


# ---------- repricing ----------


def reprice(r, cdf):
    """Market probability from stored forecast inputs under an arbitrary residual CDF."""
    mu, sigma = r["forecast_high"], r["forecast_sigma"]
    unit = r.get("unit")
    F = lambda x_unit: cdf((to_c(x_unit, unit) - mu) / sigma)
    mt = r["market_type"]
    if mt == "temp_at_least":
        p = 1.0 - F(r["threshold"] - BUCKET_HALF_WIDTH)
    elif mt == "temp_at_most":
        p = F(r["threshold"] + BUCKET_HALF_WIDTH)
    elif mt == "temp_bucket":
        upper = r["threshold_upper"] if r.get("threshold_upper") is not None else r["threshold"]
        p = F(upper + BUCKET_HALF_WIDTH) - F(r["threshold"] - BUCKET_HALF_WIDTH)
    else:
        return None
    return min(max(p, CLIP_LO), CLIP_HI)


def brier(pairs):
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs) if pairs else float("nan")


def cdf_normal(s):
    return lambda z: 0.5 * (1.0 + math.erf(z / (s * math.sqrt(2.0))))


def cdf_t_matched(nu):
    tau = math.sqrt((nu - 2.0) / nu) if not math.isinf(nu) else 1.0
    return lambda z: t_cdf(z / tau, nu)


def cdf_t_fitted(nu, s):
    return lambda z: t_cdf(z / s, nu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", default="data/captures.jsonl")
    ap.add_argument("--station-only", action="store_true")
    args = ap.parse_args()

    rows = load(args.captures)
    truth = realized_highs(rows)
    n_exact = sum(1 for t in truth.values() if t[0] == "exact")
    print(
        f"city-days with recovered truth: {len(truth)} "
        f"({n_exact} exact, {len(truth) - n_exact} censored bounds)"
    )

    def usable(r):
        if r.get("outcome") is None or not r["market_type"].startswith("temp"):
            return False
        if r.get("forecast_high") is None or not r.get("forecast_sigma"):
            return False
        if args.station_only and r["forecast_sigma"] == 2.0:
            return False  # legacy flat-sigma path, not the fitted tables
        return lead(r) in (1, 2)

    fit_rows = [r for r in rows if usable(r)]

    # --- step 2: deduped z sample (exact points + censoring intervals) ---
    seen, zs, z_dates, cens, cens_dates = set(), [], [], [], []
    for r in fit_rows:
        key = (r.get("source", "polymarket"), r["city"], r["target_date"], lead(r))
        t = truth.get(key[:3])
        if t is None or key in seen:
            continue
        seen.add(key)
        mu, sigma = r["forecast_high"], r["forecast_sigma"]
        if t[0] == "exact":
            zs.append((t[1] - mu) / sigma)
            z_dates.append(r["target_date"])
        else:
            _, lo, hi = t
            zlo = (lo - mu) / sigma if lo is not None else None
            zhi = (hi - mu) / sigma if hi is not None else None
            cens.append((zlo, zhi))
            cens_dates.append(r["target_date"])
    print(f"deduped obs (lead 1-2): {len(zs)} exact + {len(cens)} censored")
    if len(zs) < 40:
        print("too few observations to fit tails; accrue more captures first")
        return

    # --- step 3: censoring-aware MLE fit ---
    fits = fit_scaled_t(zs, cens)
    ll_norm = fits[float("inf")]
    best_nu = max(fits, key=lambda nu: fits[nu][1])
    print(f"\nNormal (nu=inf): s={ll_norm[0]:.2f}  loglik={ll_norm[1]:.2f}")
    print("nu grid (s, delta-loglik vs Normal):")
    for nu in NU_GRID:
        s, ll = fits[nu]
        tag = "  <-- best" if nu == best_nu else ""
        print(f"  nu={nu:<6} s={s:.2f}  dLL={ll - ll_norm[1]:+.2f}{tag}")
    # Tail counts: exact obs plus PROVABLE censored minima (bound already beyond the cut).
    c2 = sum(1 for zlo, zhi in cens if (zlo is not None and zlo > 2) or (zhi is not None and zhi < -2))
    c3 = sum(1 for zlo, zhi in cens if (zlo is not None and zlo > 3) or (zhi is not None and zhi < -3))
    n2 = sum(1 for z in zs if abs(z) > 2)
    n3 = sum(1 for z in zs if abs(z) > 3)
    n_all = len(zs) + len(cens)
    for label, nu, s in [
        ("Normal", float("inf"), ll_norm[0]),
        (f"t(nu={best_nu})", best_nu, fits[best_nu][0]),
    ]:
        e2 = n_all * 2 * (1 - t_cdf(2 / s, nu))
        e3 = n_all * 2 * (1 - t_cdf(3 / s, nu))
        print(
            f"  |z|>2: >={n2 + c2} obs ({n2} exact + {c2} provable censored) vs {e2:.1f} "
            f"expected; |z|>3: >={n3 + c3} vs {e3:.1f}  [{label}]"
        )

    # --- step 4: repricing Brier, walk-forward split by target day ---
    all_dates = sorted(z_dates + cens_dates)
    split = all_dates[len(all_dates) // 2]
    train = [z for z, d in zip(zs, z_dates) if d < split]
    train_cens = [c for c, d in zip(cens, cens_dates) if d < split]
    if len(train) >= 40:
        f_tr = fit_scaled_t(train, train_cens)
        nu_tr = max(f_tr, key=lambda nu: f_tr[nu][1])
        s_norm_tr, s_t_tr = f_tr[float("inf")][0], f_tr[nu_tr][0]
    else:
        nu_tr, s_norm_tr, s_t_tr = best_nu, ll_norm[0], fits[best_nu][0]
        print(f"(train half too small; walk-forward columns reuse the full fit)")
    print(f"\nsplit at target {split}: train fit nu={nu_tr} s={s_t_tr:.2f} s_norm={s_norm_tr:.2f}")

    variants = [
        ("normal", cdf_normal(1.0)),
        ("normal-s", cdf_normal(s_norm_tr)),
        ("t-matched", cdf_t_matched(nu_tr)),
        ("t-fitted", cdf_t_fitted(nu_tr, s_t_tr)),
    ]
    for scope, sel in [
        ("in-sample (all)", lambda r: True),
        (f"walk-forward (target >= {split})", lambda r: r["target_date"] >= split),
    ]:
        srows = [r for r in fit_rows if sel(r)]
        print(f"\n  {scope}: {len(srows)} market rows")
        p_norm = {id(r): reprice(r, variants[0][1]) for r in srows}
        stored = brier([(r["model_estimate"], r["outcome"]) for r in srows if r.get("model_estimate") is not None])
        mkt = brier([(r["entry_price"], r["outcome"]) for r in srows])
        print(f"    stored estimate Brier={stored:.4f}   market Brier={mkt:.4f}")
        for name, cdf in variants:
            pairs = [(reprice(r, cdf), r["outcome"]) for r in srows]
            tails = [
                (p, y)
                for (p, y), r in zip(pairs, srows)
                if p_norm[id(r)] is not None and (p_norm[id(r)] < 0.05 or p_norm[id(r)] > 0.95)
            ]
            print(
                f"    {name:<10} Brier={brier(pairs):.4f}   tail slice (n={len(tails)}): {brier(tails):.4f}"
            )


if __name__ == "__main__":
    main()
