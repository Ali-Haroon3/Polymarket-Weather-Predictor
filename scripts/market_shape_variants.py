#!/usr/bin/env python3
"""Variants of the market-shape strategy, replayed strictly walk-forward (stdlib only) —
second look 2026-09-09, two days after `backtesting::market_shape` shipped.

Builds on scripts/market_shape_alpha.py (imported; same ladders, same (b, k) fit, same
decide_cell, same fee) and asks the obvious follow-up questions the first look left open.
Every variant fixes its parameters as of each capture day from ladders whose TARGET preceded
that day, so each trade is out-of-sample for the parameters even though the variant LIST is
chosen here. ROI is on capital risked with flat contracts; "flat-stake" is the mean of
pnl/price, i.e. what $-per-order sizing (the pilot's) realizes.

Second look (captures 06-29..09-08, 536 resolved complete lead ≥ 1 Kalshi ladders; the shipped
rule reads 731 trades, +13.7% ROI, 60% win, t = 3.6 clustered by date):

  * PER-CITY b is DEAD, both ways: a per-city Brier fit (≥ 30 own ladders, else venue) drops
    the replay to +11.0% / t 2.8, and per-city trailing residuals shrunk to the venue b (14/21/
    30-day windows) land at +11.5..12.9%. Pooling across cities is what makes the bias
    estimable; the CLAUDE.md "per-city b is the obvious next test" is answered, negatively,
    at this sample size. Philadelphia/Boston stay negative under every variant.
  * SKEW (two-piece Normal, right/left scale ratio fitted on top of (b, k)) adds nothing:
    +13.2% on 662; the fitted ratio hovers 0.9–1.0.
  * A causal CITY VETO (skip a city while its own replay is negative on ≥ 15 or ≥ 30 trades)
    is a no-op: +13.6% / +13.9%.
  * REGIME-FOLLOWING b (trailing 3–14-day national residual, shrunk to the fitted b) is the
    only variant that edges the rule — best 5-day / n0 = 20 at +14.5% on 799, t 4.1 — and it
    is within noise of the base; a candidate for an A/B row once the paper sample exists, not
    a rule change.
  * LEAD 0 (day-of ladders) is unreadable: 49 trades on 3 capture dates (the daemon rarely
    records a complete two-sided day-of ladder), +16.5% but t ≈ 1.
  * θ × FLOOR: raising the floor to 15¢ reads +15.6% (from +13.7%) by dropping cheap BUYs
    that win 24% of the time, with no gain in t — in-sample selection, rule unchanged.
  * WHERE THE EDGE SITS: sub-1.3 °C-σ ladders and 1.3–1.8 both ≈ +14%; ≥ 1.8 loses on n = 8.
    Target weekday is the one striking slice — Thu/Fri/Sat targets +22..25% vs +3..7% for the
    rest, roughly 2.8σ apart on pooled cells — but it is one of seven slices inspected, has no
    mechanism yet, and is recorded as a watch item, not a filter.
  * PILOT-FAITHFUL (top-5 net edges per run day, ≤ 2 per city-day, flat $15): 206 orders,
    +21.3% ROI, 60% win, t 3.8 by date; +$620 over 46 run days; worst week −$14 and the $50
    weekly breaker never trips. Ranking by claimed net edge CONCENTRATES the edge (top-5 +21%
    > top-10 +18% > top-15 +15% > all +13.7%), the opposite of the weather model's winner's
    curse — the claimed edge is informative here. That is the shape the go-live gate's paper
    sample will have; at ~4.5 candidates/day the 100-order criterion is ~3 weeks out.

Usage: python3 scripts/market_shape_variants.py [--captures data/captures.jsonl]
"""
import os
import collections
import datetime as dt
import json
import math
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import market_shape_alpha as M  # noqa: E402

CAPTURES = sys.argv[sys.argv.index("--captures") + 1] if "--captures" in sys.argv else "data/captures.jsonl"
rows = [json.loads(l) for l in open(CAPTURES) if l.strip()]
ladders = M.build_ladders(rows)
KAL = [l for l in ladders if l["venue"] == "kalshi"]
print(f"kalshi ladders {len(KAL)}; complete+resolved lead>=1 {sum(M.complete(l) and l['resolved'] and l['lead']>=1 for l in KAL)}; lead0 complete+resolved {sum(M.complete(l) and l['resolved'] and l['lead']==0 for l in KAL)}")

# ── base (b, k) per capture day, cached ───────────────────────────────────────
DAYS = sorted({l["cap"] for l in KAL if l["resolved"] and M.complete(l)})
BASE = {}
for d in DAYS:
    BASE[d] = M.fit_shape(M.shape_history(ladders, "kalshi", d))
print("base (b,k) path:", " ".join(f"{d[5:]}:{p[0]:+.1f}/{p[1]:.2f}" for d, p in sorted(BASE.items()) if p)[:400])


def resid_hist(as_of, city=None, days=None):
    out = []
    for l in KAL:
        if l["realized"] is None or not M.complete(l) or l["lead"] < 1 or l["target"] >= as_of:
            continue
        if city is not None and l["city"] != city:
            continue
        if days is not None and (dt.date.fromisoformat(as_of) - dt.date.fromisoformat(l["target"])).days > days:
            continue
        out.append(l["realized"] - l["mu"])
    return out


def twopiece_probs(l, b, k, r):
    """Two-piece normal: left scale s1, right scale s2 = r*s1, mean scale k*sig."""
    mu = l["mu"] + b
    s = max(0.2, l["sig"] * k)
    s1 = 2 * s / (1 + r)
    s2 = r * s1

    def F(x):
        if x == -math.inf:
            return 0.0
        if x == math.inf:
            return 1.0
        if x < mu:
            return (2 * s1 / (s1 + s2)) * M.phi((x - mu) / s1)
        return (s1 - s2) / (s1 + s2) + (2 * s2 / (s1 + s2)) * M.phi((x - mu) / s2)

    return [max(0.0, min(1.0, F(c["hi"]) - F(c["lo"]))) for c in l["cells"]]


def brier_probs(l, probs):
    return sum((p - (c["outcome"] or 0.0)) ** 2 for p, c in zip(probs, l["cells"]))


R_GRID = [round(0.6 + 0.1 * i, 1) for i in range(10)]
SKEW = {}
for d in DAYS:
    p = BASE[d]
    if p is None:
        SKEW[d] = None
        continue
    hist = M.shape_history(ladders, "kalshi", d)
    best = None
    for r in R_GRID:
        tot = sum(brier_probs(l, twopiece_probs(l, p[0], p[1], r)) for l in hist)
        if best is None or tot < best[0]:
            best = (tot, r)
    SKEW[d] = best[1]
print("skew r path:", " ".join(f"{d[5:]}:{r}" for d, r in sorted(SKEW.items()) if r)[:300])

# per-city Brier b (k from venue), min 30 ladders
CITY_B = {}
for d in DAYS:
    p = BASE[d]
    if p is None:
        continue
    hist = M.shape_history(ladders, "kalshi", d)
    byc = collections.defaultdict(list)
    for l in hist:
        byc[l["city"]].append(l)
    for c, ls in byc.items():
        if len(ls) < 30:
            continue
        best = None
        for b in M.B_GRID:
            tot = sum(M.brier(l, b, p[1]) for l in ls)
            if best is None or tot < best[0]:
                best = (tot, b)
        CITY_B[(d, c)] = best[1]


def generic_replay(theta=0.03, min_price=0.10, lead_ok=lambda L: L >= 1, params=None, probs=None, city_ok=None):
    lad = [l for l in KAL if l["resolved"] and M.complete(l) and lead_ok(l["lead"])]
    lad.sort(key=lambda l: (l["cap"], l["target"]))
    trades = []
    for l in lad:
        p = (params or (lambda l: BASE.get(l["cap"])))(l)
        if p is None:
            continue
        if city_ok is not None and not city_ok(l):
            continue
        qs = (probs or (lambda l, p: M.shaped_probs(l, *p)))(l, p)
        for q, c in zip(qs, l["cells"]):
            d = M.decide_cell(q, c["bid"], c["ask"], theta, min_price)
            if d is None:
                continue
            side, price, yes_px, edge = d
            won = (c["outcome"] >= 0.5) if side == "BUY" else (c["outcome"] < 0.5)
            pnl = (1.0 - price if won else -price) - M.fee(yes_px)
            trades.append(dict(side=side, price=price, yes_px=yes_px, edge=edge, pnl=pnl, key=(l["city"], l["target"]), date=l["target"], cap=l["cap"], city=l["city"], mt=c["mt"], month=l["target"][:7], lead=l["lead"], sig=l["sig"], q=q))
    return trades


def rep(name, tr, by=None):
    M.report(name, tr, by)
    if tr:
        flat = M.mean([t["pnl"] / t["price"] for t in tr])
        print(f"        flat-stake ROI (pnl/price mean) {flat:+.1%}")


print("\n== 0. BASELINE (module rule, lead >= 1) ==")
base_tr = generic_replay()
rep("base", base_tr, by=lambda t: t["lead"])
rep("   by month", base_tr, by=lambda t: t["month"])

print("\n== 1. LEAD 0 (day-of ladders, params from lead>=1 history) ==")
l0 = generic_replay(lead_ok=lambda L: L == 0)
rep("lead0, base params", l0, by=lambda t: t["side"])
rep("   by month", l0, by=lambda t: t["month"])
# lead0 with b = 0 (shape only) and b only
l0b = generic_replay(lead_ok=lambda L: L == 0, params=lambda l: (BASE[l["cap"]][0], 1.0) if BASE.get(l["cap"]) else None)
rep("lead0, b only", l0b)
l0k = generic_replay(lead_ok=lambda L: L == 0, params=lambda l: (0.0, BASE[l["cap"]][1]) if BASE.get(l["cap"]) else None)
rep("lead0, k only", l0k)
# lead 0 residual stats
r0 = [(l["realized"] - l["mu"], (l["realized"] - l["mu"]) / l["sig"]) for l in KAL if l["lead"] == 0 and l["realized"] is not None and M.complete(l)]
if r0:
    print(f"   lead0 ladders w/ realized n={len(r0)} bias={M.mean([a for a,_ in r0]):+.2f}±{M.se([a for a,_ in r0]):.2f} sd(z)={M.se([b for _,b in r0])*math.sqrt(len(r0)):.2f}")

print("\n== 2. REGIME-FOLLOWING b (trailing national residual, shrunk to fitted b) ==")
for W, n0 in [(3, 10), (5, 20), (7, 30), (14, 30), (7, 0)]:
    def params(l, W=W, n0=n0):
        p = BASE.get(l["cap"])
        if p is None:
            return None
        h = resid_hist(l["cap"], days=W)
        if not h:
            return p
        b = (sum(h) + n0 * p[0]) / (len(h) + n0)
        return (b, p[1])
    rep(f"trail{W}d n0={n0}", generic_replay(params=params))

print("\n== 3. PER-CITY b ==")
def city_brier(l):
    p = BASE.get(l["cap"])
    if p is None:
        return None
    return (CITY_B.get((l["cap"], l["city"]), p[0]), p[1])
rep("per-city Brier b (min30, else venue)", generic_replay(params=city_brier), by=lambda t: t["city"])
for W, n0 in [(14, 10), (21, 15), (30, 20)]:
    def params(l, W=W, n0=n0):
        p = BASE.get(l["cap"])
        if p is None:
            return None
        h = resid_hist(l["cap"], city=l["city"], days=W)
        b = (sum(h) + n0 * p[0]) / (len(h) + n0) if h else p[0]
        return (b, p[1])
    rep(f"per-city trailing{W}d n0={n0}", generic_replay(params=params))

print("\n== 4. SKEW (two-piece normal, r fitted on top of (b,k)) ==")
sk = generic_replay(params=lambda l: (BASE[l["cap"]], SKEW[l["cap"]]) if BASE.get(l["cap"]) else None, probs=lambda l, p: twopiece_probs(l, p[0][0], p[0][1], p[1]))
rep("b + k + skew", sk, by=lambda t: t["side"])

print("\n== 5. CITY VETO (skip city when its own causal replay is negative on >= N trades) ==")
hist_by_city = collections.defaultdict(list)
for t in sorted(base_tr, key=lambda t: t["date"]):
    hist_by_city[t["city"]].append(t)
for N in (15, 30):
    keep = []
    for t in base_tr:
        prior = [u for u in hist_by_city[t["city"]] if u["date"] < t["cap"]]
        if len(prior) >= N and sum(u["pnl"] for u in prior) < 0:
            continue
        keep.append(t)
    rep(f"veto city if cum pnl<0 on >={N}", keep)

print("\n== 6. WHERE IS THE EDGE (base trades) ==")
rep("   by side x mt", base_tr, by=lambda t: (t["side"], t["mt"]))
rep("   by market sigma tercile", base_tr, by=lambda t: "sig<1.3" if t["sig"] < 1.3 else "1.3-1.8" if t["sig"] < 1.8 else ">=1.8")
rep("   by weekday(target)", base_tr, by=lambda t: dt.date.fromisoformat(t["date"]).strftime("%a"))
rep("   by q (our prob) band", base_tr, by=lambda t: (t["side"], "q<0.3" if t["q"] < 0.3 else "0.3-0.6" if t["q"] < 0.6 else "q>=0.6"))
cnt = collections.Counter(t["cap"] for t in base_tr)
rep("   by candidates/day", base_tr, by=lambda t: "<=10" if cnt[t["cap"]] <= 10 else "11-20" if cnt[t["cap"]] <= 20 else ">20")

print("\n== 7. PILOT-FAITHFUL: top-5 net edges per run day, max 2 per city-day, flat $15 ==")
for cap_n, per_city in [(5, 2), (5, 99), (10, 2), (15, 2), (999, 99)]:
    byday = collections.defaultdict(list)
    for t in base_tr:
        byday[t["cap"]].append(t)
    sel = []
    for d, ts in byday.items():
        ts = sorted(ts, key=lambda t: -t["edge"])
        cc = collections.Counter()
        n = 0
        for t in ts:
            if n >= cap_n:
                break
            if cc[t["key"]] >= per_city:
                continue
            cc[t["key"]] += 1
            n += 1
            sel.append(t)
    rep(f"top{cap_n}/day, <= {per_city}/city-day", sel, by=lambda t: t["month"])
    if cap_n == 5 and per_city == 2:
        # $15 stakes: dollars
        dollars = [15.0 * t["pnl"] / t["price"] for t in sel]
        wk = collections.defaultdict(float)
        for t, d in zip(sel, dollars):
            wk[dt.date.fromisoformat(t["date"]).isocalendar()[1]] += d
        print(f"        $15 stakes: total ${sum(dollars):+.2f} over {len(sel)} orders; worst week ${min(wk.values()):+.2f}, weeks<-50: {sum(v < -50 for v in wk.values())}/{len(wk)}")
        print(f"        per-run-day count: {M.mean([len(v) for v in byday.values()]):.1f} candidates/day over {len(byday)} days")

print("\n== 8. THETA x FLOOR grid (base) ==")
for th in (0.02, 0.03, 0.04, 0.05):
    for mp in (0.05, 0.10, 0.15, 0.20):
        tr = generic_replay(theta=th, min_price=mp)
        if tr:
            pnl = [t["pnl"] for t in tr]
            risk = sum(t["price"] for t in tr)
            td, nd = M.clustered_t(tr, lambda t: t["date"])
            print(f"   θ={th:.2f} floor={mp:.2f} n={len(tr):4d} ROI={sum(pnl)/risk:+.1%} t(date)={td:+.1f}")
