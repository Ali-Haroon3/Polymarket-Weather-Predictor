#!/usr/bin/env python3
"""The market-shape alpha search over accrued captures (stdlib only) — first look 2026-09-07.

Written the day the phantom-fill defect (backtesting::reference_price) took the weather model's
realized edge to zero on both venues, to answer the only question left: is there ANY edge in the
captures that survives a strict walk-forward, executable prices and Kalshi's fee? There is, and
it does not involve the weather model. This script is the evidence behind
`backtesting::market_shape`, kept runnable so the numbers can be re-read as captures accrue.

What it does, in order:

  A. Rebuilds every Kalshi city-day LADDER (the 2 °F buckets plus both open-ended thresholds a
     capture day recorded for one city and target) and fits a Normal(μ, σ) to its mids by least
     squares — the MARKET's own forecast of the settlement high. Polymarket captures hold one to
     three markets per city-day, never a ladder, so everything below is Kalshi.
  B. Scores that market forecast against the realized high (recovered from the unique winning
     bucket; threshold winners are censored and skipped), beside the weather model's
     `forecast_high` and every alternative source the captures log (ensemble means, AIFS,
     AIGFS), and regresses the realized high on model and market together.
  C. Measures the market's own shape errors by week and by city: the bias of realized − μ and
     the dispersion of (realized − μ)/σ.
  D. Replays the strategy STRICTLY walk-forward: for each capture day, (b, k) minimizing the
     cell Brier over the resolved complete ladders whose target was BEFORE that day (newest
     HIST_CAP), then every cell of that day's complete ladders decided against its executable
     quote net of fee — BUY YES at the ask above θ, BUY NO at the bid above θ, both floored at
     MIN_PRICE — and settled at its outcome. Reports ROI on capital risked, t-statistics raw and
     clustered by ladder and by date, and breakdowns by month, side × band, shape and city, plus
     the same replay with only b or only k to attribute the edge.

First look (captures 06-29..09-07, 5536 resolved lead ≥ 1 rows, 458 uncensored Kalshi ladders):

  * The market's μ beats the model's: RMSE 0.92 °C vs 1.22; OLS realized = 0.02·model +
    0.98·market. The model adds nothing on the mean. Ensemble means and the AI models are
    worse still (RMSE 1.7–2.4).
  * The market runs COLD (realized − μ = +0.27 °C, positive in 8 of 9 weeks and 14 of 15
    cities — consistent with settlement on the NWS CLI max, which sits 0–2 °F above the METAR
    obs traders watch) and TOO WIDE (sd of its z-scores 0.86, 75% inside ±1σ).
  * Strict walk-forward, fees, executable prices, θ = 3%: 899 trades, +10.7% ROI on risk, 64%
    win, t = 4.6 raw / 4.3 by ladder / 4.1 by date; Jul +17%, Aug +10%, Sep +11%; BUY side
    +23% ROI on 464, NO side ≥ 10¢ +10% on 271; positive in 12 of 15 cities. b alone gives
    +11.9%, k alone +1.3%: the bias carries it, the sharpening adds breadth. Window / cadence
    variants land at +9.7..11.4%. Claimed edges realize about one-for-one (0.043 → 0.035,
    0.078 → 0.090), which the weather model's never did.

Usage: python3 scripts/market_shape_alpha.py [--captures data/captures.jsonl] [--theta 0.03]
                                              [--min-price 0.10] [--section A|B|C|D|all]
"""
import argparse
import collections
import datetime as dt
import json
import math
import statistics as st

MIN_LADDERS = 60
HIST_CAP = 300
COMPLETE_SUM = (0.8, 1.2)
COMPLETE_CELLS = 4
B_GRID = [round(-0.6 + 0.1 * i, 1) for i in range(13)]
K_GRID = [round(0.6 + 0.05 * i, 2) for i in range(13)]


# ── shared rules (mirrors of the Rust) ───────────────────────────────────────


def usable(x):
    return x if (x is not None and 0.0 < x < 1.0) else None


def reference_price(r):
    """backtesting::reference_price: mid of a sane book, the quoted side of a one-sided book,
    the last trade only with no book at all and never the 0.50 never-traded default."""
    b, a = r.get("best_bid"), r.get("best_ask")
    reported_book = b is not None or a is not None
    b, a = usable(b), usable(a)
    if b is not None and a is not None:
        return (a + b) / 2.0 if b <= a else None
    if b is not None or a is not None:
        return b if b is not None else a
    px = None if reported_book else r.get("entry_price")
    if px is not None and abs(px - 0.5) < 1e-9:
        px = None
    return usable(px)


def fee(p):
    return 0.07 * p * (1.0 - p)


def to_c(v, unit):
    return v if (unit or "F").upper() == "C" else (v - 32.0) * 5.0 / 9.0


def cell_bounds_c(r):
    mt, t, tu, unit = r["market_type"], r["threshold"], r.get("threshold_upper"), r.get("unit")
    if mt == "temp_bucket":
        return to_c(t - 0.5, unit), to_c((tu if tu is not None else t) + 0.5, unit)
    if mt == "temp_at_least":
        return to_c(t - 0.5, unit), math.inf
    if mt == "temp_at_most":
        return -math.inf, to_c(t + 0.5, unit)
    return None


def phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def cell_prob(lo, hi, mu, sig):
    a = 0.0 if lo == -math.inf else phi((lo - mu) / sig)
    b = 1.0 if hi == math.inf else phi((hi - mu) / sig)
    return max(0.0, min(1.0, b - a))


def fit_market_normal(cells, mu0):
    def sse(mu, sig):
        return sum((cell_prob(lo, hi, mu, sig) - p) ** 2 for lo, hi, p in cells)

    best = None
    for i in range(-16, 17):
        mu = mu0 + 0.5 * i
        for j in range(1, 11):
            v = sse(mu, 0.5 * j)
            if best is None or v < best[0]:
                best = (v, mu, 0.5 * j)
    _, cmu, csig = best
    for i in range(-5, 6):
        for j in range(-5, 6):
            mu, sig = cmu + 0.1 * i, max(0.2, csig + 0.1 * j)
            v = sse(mu, sig)
            if v < best[0]:
                best = (v, mu, sig)
    return best[1], best[2], best[0]


def lead(r):
    return (dt.date.fromisoformat(r["target_date"]) - dt.date.fromisoformat(r["captured_at"][:10])).days


def build_ladders(rows):
    groups = collections.defaultdict(list)
    for r in rows:
        if not r.get("market_type", "").startswith("temp"):
            continue
        px = reference_price(r)
        e = cell_bounds_c(r)
        if px is None or e is None or r.get("market_id") is None:
            continue
        key = (r.get("source", "polymarket"), r["city"], r["target_date"], r["captured_at"][:10])
        groups[key].append(
            dict(
                lo=e[0],
                hi=e[1],
                px=px,
                bid=usable(r.get("best_bid")),
                ask=usable(r.get("best_ask")),
                outcome=r.get("outcome"),
                mt=r["market_type"],
                mid=r["market_id"],
                unit=r.get("unit"),
                t=r["threshold"],
                tu=r.get("threshold_upper"),
                forecast_high=r.get("forecast_high"),
                forecast_sigma=r.get("forecast_sigma"),
                extra=r,
            )
        )
    out = []
    for (venue, city, target, cap), cells in groups.items():
        if len(cells) < 3:
            continue
        cells.sort(key=lambda c: (c["lo"], c["mid"]))
        psum = sum(c["px"] for c in cells)
        w = sum(c["px"] for c in cells)
        mu0 = (
            sum(
                c["px"] * (((c["lo"] if c["lo"] != -math.inf else c["hi"] - 1.0) + (c["hi"] if c["hi"] != math.inf else c["lo"] + 1.0)) / 2.0)
                for c in cells
            )
            / w
            if w > 0
            else 0.0
        )
        mu, sig, err = fit_market_normal([(c["lo"], c["hi"], c["px"]) for c in cells], mu0)
        winners = [c for c in cells if c["mt"] == "temp_bucket" and c["outcome"] == 1.0]
        realized = None
        if len(winners) == 1:
            wv = winners[0]
            realized = to_c((wv["t"] + (wv["tu"] if wv["tu"] is not None else wv["t"])) / 2.0, wv["unit"])
        first = cells[0]["extra"]
        out.append(
            dict(
                venue=venue,
                city=city,
                target=target,
                cap=cap,
                lead=(dt.date.fromisoformat(target) - dt.date.fromisoformat(cap)).days,
                cells=cells,
                psum=psum,
                mu=mu,
                sig=sig,
                err=err,
                realized=realized,
                resolved=all(c["outcome"] is not None for c in cells),
                model_mu=next((c["forecast_high"] for c in cells if c["forecast_high"] is not None), None),
                model_sig=next((c["forecast_sigma"] for c in cells if c["forecast_sigma"] is not None), None),
                ens_mean_ecmwf=first.get("ensemble_mean_ecmwf"),
                ens_mean_gfs=first.get("ensemble_mean_gfs"),
                aifs=first.get("forecast_high_aifs"),
                aigfs=first.get("forecast_high_aigfs"),
            )
        )
    return out


def complete(l):
    return len(l["cells"]) >= COMPLETE_CELLS and COMPLETE_SUM[0] <= l["psum"] <= COMPLETE_SUM[1]


def shaped_probs(l, b, k):
    mu, sig = l["mu"] + b, max(0.2, l["sig"] * k)
    return [cell_prob(c["lo"], c["hi"], mu, sig) for c in l["cells"]]


def brier(l, b, k):
    return sum((p - (c["outcome"] or 0.0)) ** 2 for p, c in zip(shaped_probs(l, b, k), l["cells"]))


def fit_shape(hist, fit_b=True, fit_k=True):
    if len(hist) < MIN_LADDERS:
        return None
    best = None
    for b in B_GRID if fit_b else [0.0]:
        for k in K_GRID if fit_k else [1.0]:
            tot = sum(brier(l, b, k) for l in hist)
            if best is None or tot < best[0]:
                best = (tot, b, k)
    return best[1], best[2]


def shape_history(ladders, venue, as_of):
    hist = [l for l in ladders if l["venue"] == venue and complete(l) and l["resolved"] and l["lead"] >= 1 and l["target"] < as_of]
    hist.sort(key=lambda l: (l["target"], l["cap"], l["city"]))
    return hist[-HIST_CAP:]


def decide_cell(q, bid, ask, theta, min_price):
    ok = lambda x: x if (x is not None and 0.0 < x < 1.0 and x >= min_price) else None
    a, b = ok(ask), ok(bid)
    if a is not None and q - a - fee(a) > theta:
        return "BUY", a, a, q - a - fee(a)
    if b is not None and b - q - fee(b) > theta:
        return "SELL", 1.0 - b, b, b - q - fee(b)
    return None


# ── helpers ──────────────────────────────────────────────────────────────────


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def se(v):
    n = len(v)
    if n < 2:
        return float("nan")
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1) / n)


def rmse(v):
    return math.sqrt(mean([x * x for x in v]))


def ols2(ys, x1s, x2s):
    n = len(ys)
    X = [[1.0, a, b] for a, b in zip(x1s, x2s)]
    A = [[sum(X[i][p] * X[i][q] for i in range(n)) for q in range(3)] for p in range(3)]
    y = [sum(X[i][p] * ys[i] for i in range(n)) for p in range(3)]
    for c in range(3):
        piv = A[c][c]
        for r in range(3):
            if r != c:
                f = A[r][c] / piv
                A[r] = [A[r][k] - f * A[c][k] for k in range(3)]
                y[r] -= f * y[c]
    return [y[i] / A[i][i] for i in range(3)]


# ── sections ─────────────────────────────────────────────────────────────────


def section_b(ladders):
    print("\n== B. MEAN ACCURACY vs realized high (°C): the market's implied μ against every forecast ==")
    ok = [l for l in ladders if l["venue"] == "kalshi" and l["resolved"] and l["realized"] is not None and complete(l) and l["lead"] >= 1 and l["model_mu"] is not None]
    print(f"   {len(ok)} usable Kalshi ladders (resolved, complete, uncensored realized, lead >= 1)")
    for name, key in [("model μ", "model_mu"), ("market μ", "mu"), ("ens mean ECMWF", "ens_mean_ecmwf"), ("ens mean GFS", "ens_mean_gfs"), ("AIFS", "aifs"), ("AIGFS", "aigfs")]:
        sub = [l for l in ok if l.get(key) is not None]
        if len(sub) < 20:
            continue
        res = [l["realized"] - l[key] for l in sub]
        print(f"   {name:16s} n={len(sub):4d} bias={mean(res):+.2f} MAE={mean([abs(x) for x in res]):.2f} RMSE={rmse(res):.2f}")
    d = [abs(l["realized"] - l["model_mu"]) - abs(l["realized"] - l["mu"]) for l in ok]
    print(f"   paired |model err| − |market err|: {mean(d):+.3f} ± {se(d):.3f}")
    a, b1, b2 = ols2([l["realized"] for l in ok], [l["model_mu"] for l in ok], [l["mu"] for l in ok])
    print(f"   OLS realized = {a:+.2f} + {b1:.3f}·model + {b2:.3f}·market  (weight on model {b1 / (b1 + b2):.2f})")


def section_c(ladders):
    print("\n== C. THE MARKET'S OWN SHAPE ERRORS (kalshi, lead 1, complete, uncensored) ==")
    ok = [l for l in ladders if l["venue"] == "kalshi" and l["resolved"] and l["realized"] is not None and complete(l) and l["lead"] == 1]
    by = collections.defaultdict(list)
    for l in ok:
        d = dt.date.fromisoformat(l["target"])
        by[(d - dt.timedelta(days=d.weekday())).isoformat()].append(l)
    for wk in sorted(by):
        rs = by[wk]
        res = [l["realized"] - l["mu"] for l in rs]
        z = [(l["realized"] - l["mu"]) / l["sig"] for l in rs]
        print(f"   week {wk} n={len(rs):3d} bias={mean(res):+.2f}±{se(res):.2f} °C  sd(z)={st.pstdev(z):.2f}  within ±1σ {mean([abs(x) < 1 for x in z]):.2f}")
    by = collections.defaultdict(list)
    for l in ok:
        by[l["city"]].append(l)
    for c in sorted(by):
        rs = by[c]
        res = [l["realized"] - l["mu"] for l in rs]
        z = [(l["realized"] - l["mu"]) / l["sig"] for l in rs]
        print(f"   {c:14s} n={len(rs):3d} bias={mean(res):+.2f}±{se(res):.2f}  sd(z)={st.pstdev(z):.2f}")


def replay(ladders, theta, min_price, fit_b=True, fit_k=True, venue="kalshi"):
    lad = [l for l in ladders if l["venue"] == venue and l["resolved"] and complete(l) and l["lead"] >= 1]
    lad.sort(key=lambda l: (l["cap"], l["target"]))
    params_by_day = {}
    trades = []
    for l in lad:
        if l["cap"] not in params_by_day:
            params_by_day[l["cap"]] = fit_shape(shape_history(ladders, venue, l["cap"]), fit_b, fit_k)
        p = params_by_day[l["cap"]]
        if p is None:
            continue
        for q, c in zip(shaped_probs(l, *p), l["cells"]):
            d = decide_cell(q, c["bid"], c["ask"], theta, min_price)
            if d is None:
                continue
            side, price, yes_px, edge = d
            won = (c["outcome"] >= 0.5) if side == "BUY" else (c["outcome"] < 0.5)
            pnl = (1.0 - price if won else -price) - fee(yes_px)
            trades.append(dict(side=side, price=price, yes_px=yes_px, edge=edge, pnl=pnl, key=(l["city"], l["target"]), date=l["target"], city=l["city"], mt=c["mt"], month=l["target"][:7]))
    path = sorted((d, p) for d, p in params_by_day.items() if p is not None)
    return trades, path


def clustered_t(tr, keyf):
    g = collections.defaultdict(float)
    for t in tr:
        g[keyf(t)] += t["pnl"]
    v = list(g.values())
    return (mean(v) / se(v) if len(v) > 2 and se(v) > 0 else float("nan")), len(v)


def report(name, tr, by=None):
    if not tr:
        print(f"   {name:36s} no trades")
        return
    pnl = [t["pnl"] for t in tr]
    risk = sum(t["price"] for t in tr)
    tl, nl = clustered_t(tr, lambda t: t["key"])
    td, nd = clustered_t(tr, lambda t: t["date"])
    print(f"   {name:36s} n={len(tr):4d} ROI={sum(pnl) / risk:+.1%} win={mean([p > 0 for p in pnl]):.0%} pnl/ct={mean(pnl):+.4f} t(raw)={mean(pnl) / se(pnl):+.1f} t(ladder,{nl})={tl:+.1f} t(date,{nd})={td:+.1f} BUY={sum(t['side'] == 'BUY' for t in tr)} NO={sum(t['side'] == 'SELL' for t in tr)}")
    if by:
        g = collections.defaultdict(list)
        for t in tr:
            g[by(t)].append(t)
        for k in sorted(g, key=str):
            p = [t["pnl"] for t in g[k]]
            rk = sum(t["price"] for t in g[k])
            print(f"        {str(k):26s} n={len(p):4d} ROI={sum(p) / rk:+.1%} win={mean([x > 0 for x in p]):.0%} pnl/ct={mean(p):+.4f}±{se(p) if len(p) > 1 else 0:.4f}")


def section_d(ladders, theta, min_price):
    print(f"\n== D. STRICT WALK-FORWARD REPLAY (kalshi, executable, fees, θ={theta}, floor {min_price}) ==")
    tr, path = replay(ladders, theta, min_price)
    print("   (b, k) path:", " ".join(f"{d[5:]}:{b:+.1f}/{k:.2f}" for d, (b, k) in path[:: max(1, len(path) // 8)]))
    report("b + k (the strategy)", tr, by=lambda t: t["month"])
    report("   by side × band", tr, by=lambda t: (t["side"], "<0.10" if t["yes_px"] < 0.10 else "0.10-0.35" if t["yes_px"] < 0.35 else "0.35-0.65" if t["yes_px"] < 0.65 else ">=0.65"))
    report("   by shape", tr, by=lambda t: (t["side"], t["mt"]))
    report("   by city", tr, by=lambda t: t["city"])
    g = collections.defaultdict(list)
    for t in tr:
        e = t["edge"]
        g["<0.03" if e < 0.03 else "0.03-0.06" if e < 0.06 else "0.06-0.10" if e < 0.10 else ">=0.10"].append(t)
    print("   claimed (net) edge vs realized per contract:")
    for k in ["<0.03", "0.03-0.06", "0.06-0.10", ">=0.10"]:
        v = g.get(k, [])
        if v:
            print(f"        {k:10s} n={len(v):4d} claimed={mean([t['edge'] for t in v]):.3f} realized={mean([t['pnl'] for t in v]):+.3f}±{se([t['pnl'] for t in v]):.3f}")
    for label, fb, fk in [("b only (k = 1)", True, False), ("k only (b = 0)", False, True)]:
        t2, _ = replay(ladders, theta, min_price, fb, fk)
        report(label, t2)
    for th in (0.0, 0.06):
        t2, _ = replay(ladders, th, min_price)
        report(f"θ={th}", t2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", default="data/captures.jsonl")
    ap.add_argument("--theta", type=float, default=0.03)
    ap.add_argument("--min-price", type=float, default=0.10)
    ap.add_argument("--section", default="all", choices=["A", "B", "C", "D", "all"])
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.captures) if l.strip()]
    ladders = build_ladders(rows)
    kal = [l for l in ladders if l["venue"] == "kalshi"]
    print(f"== A. LADDERS == {len(rows)} captures → {len(ladders)} ladders; kalshi {len(kal)}, complete {sum(complete(l) for l in kal)}, resolved complete lead>=1 {sum(complete(l) and l['resolved'] and l['lead'] >= 1 for l in kal)}, with uncensored realized {sum(l['realized'] is not None for l in kal)}")
    pm = [l for l in ladders if l["venue"] == "polymarket"]
    print(f"   polymarket {len(pm)} ladders, cells per ladder {dict(collections.Counter(len(l['cells']) for l in pm))} — never a full ladder, hence Kalshi-only")
    if a.section in ("B", "all"):
        section_b(ladders)
    if a.section in ("C", "all"):
        section_c(ladders)
    if a.section in ("D", "all"):
        section_d(ladders, a.theta, a.min_price)


if __name__ == "__main__":
    main()
