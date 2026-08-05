#!/usr/bin/env python3
"""Seasonal refit of the station sigma/bias tables from accrued forward captures.

Measures the drift of the CURRENT tables (stdlib only, no deps):

  1. Recover each (venue, city, target-day)'s realized high from the unique WINNING
     temp_bucket market (bucket midpoint, converted to degC). Ladders without exactly one
     winner are skipped. This uses the venue's own resolution truth by construction
     (Polymarket buckets resolve on WU ob-max, Kalshi on NWS CLI).
  2. Residual = realized - forecast_high per resolved lead 1-2 capture row, deduped to one
     observation per (venue, city, target-day, lead). forecast_high already includes the
     fitted bias, so residuals measure the REMAINING seasonal drift; z = residual/sigma_used
     would be ~N(0,1) if the tables were still calibrated.
  3. Per-(venue, city) bias delta, empirical-Bayes shrunk toward the venue mean with weight
     n/(n+K_SHRINK); cities without data get the venue mean. Then a per-venue sigma scale =
     RMS(z) after bias removal, clamped to [--sigma-floor, 2.0] (never more than doubled;
     the floor bounds how far one refit may SHARPEN sigma, the direction that causes
     overconfidence).
  4. Walk-forward validation: each capture day is re-priced with a fit from strictly earlier
     target days only. Ship the constants only if Brier improves here.

The sigma floor is a per-refit step limit, NOT a belief about the true scale: on 2026-08-05
the unconstrained fit wanted x0.61 Polymarket / x0.68 Kalshi (the 07-20 refit's x1.32 rescale
having over-widened), and walk-forward Brier improved monotonically as the floor was relaxed
(+3.9% at 0.9, +5.5% at 0.8, +6.6% at 0.7, +7.3% unconstrained). The default takes ~half the
move per refit so each step is re-validated on a fresh sample rather than trusting one
fortnight; lower it deliberately once the sample supports the rest.

Prints the suggested per-station bias[1..2] / sigma[1..2] updates; applying them to
src/stations.rs is a deliberate manual step (repo convention: the tables are the constants).
Post/k=0 slots are never touched — day-of is obs-anchored and post is (near-)deterministic.

Usage: python3 scripts/refit_station_tables.py [--captures data/captures.jsonl]
                                              [--since 2026-07-20] [--sigma-floor 0.8]
"""
import argparse
import collections
import datetime
import json
import math

K_SHRINK = 6.0
NORM = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))


def to_c(v, unit):
    return (v - 32.0) / 1.8 if (unit or "F") == "F" else v


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def realized_highs(rows):
    groups = collections.defaultdict(dict)
    for r in rows:
        if r.get("market_type") == "temp_bucket" and r.get("outcome") is not None:
            key = (r.get("source", "polymarket"), r["city"], r["target_date"])
            groups[key][r["market_id"]] = r
    truth = {}
    for key, best in groups.items():
        winners = [r for r in best.values() if r["outcome"] == 1.0]
        if len(winners) == 1:
            w = winners[0]
            lo, hi = w["threshold"], w.get("threshold_upper") or w["threshold"]
            truth[key] = to_c((lo + hi) / 2.0, w.get("unit"))
    return truth


def residuals(rows, truth, since):
    recs = {}
    for r in rows:
        if r.get("outcome") is None or r.get("forecast_high") is None:
            continue
        if r.get("forecast_sigma") in (None, 2.0):  # 2.0 = legacy path, fitted separately
            continue
        if r["captured_at"][:10] < since:
            continue
        key = (r.get("source", "polymarket"), r["city"], r["target_date"])
        if key not in truth:
            continue
        lead = (
            datetime.date.fromisoformat(r["target_date"])
            - datetime.date.fromisoformat(r["captured_at"][:10])
        ).days
        if lead not in (1, 2):
            continue
        recs[key + (lead,)] = (
            r["target_date"],
            r["source"],
            r["city"],
            lead,
            truth[key] - r["forecast_high"],
            r["forecast_sigma"],
        )
    return list(recs.values())


def fit(train, sigma_floor):
    by_v = collections.defaultdict(list)
    by_vc = collections.defaultdict(list)
    for (_, s, c, _, res, _) in train:
        by_v[s].append(res)
        by_vc[(s, c)].append(res)
    venue_mean = {v: sum(x) / len(x) for v, x in by_v.items()}
    delta = {}
    for (s, c), x in by_vc.items():
        n, m, vm = len(x), sum(x) / len(x), venue_mean[s]
        delta[(s, c)] = vm + (m - vm) * n / (n + K_SHRINK)
    scale = {}
    z2 = collections.defaultdict(list)
    for (_, s, c, _, res, sig) in train:
        z2[s].append(((res - delta[(s, c)]) / sig) ** 2)
    for s, x in z2.items():
        scale[s] = (
            min(2.0, max(sigma_floor, math.sqrt(sum(x) / len(x)))) if len(x) >= 20 else 1.0
        )
    return venue_mean, delta, scale


def price(mu_c, sig_c, r):
    u = r.get("unit") or "F"

    def cdf(x_disp, side):
        x = x_disp + (0.5 if side == "hi" else -0.5)
        xc = (x - 32.0) / 1.8 if u == "F" else x
        return NORM((xc - mu_c) / sig_c)

    lo, hi = r["threshold"], r.get("threshold_upper")
    mt = r["market_type"]
    if mt == "temp_bucket":
        return cdf(hi if hi is not None else lo, "hi") - cdf(lo, "lo")
    if mt == "temp_at_least":
        return 1 - cdf(lo, "lo")
    if mt == "temp_at_most":
        return cdf(lo, "hi")
    return None


def walk_forward(rows, recs, sigma_floor):
    old, new = [], []
    days = sorted(set(r["captured_at"][:10] for r in rows))
    for d in days:
        train = [t for t in recs if t[0] < d]
        if len(train) < 40:
            continue
        vm, delta, scale = fit(train, sigma_floor)
        for r in rows:
            if r["captured_at"][:10] != d or r.get("outcome") is None:
                continue
            if r.get("forecast_high") is None or r.get("forecast_sigma") in (None, 2.0):
                continue
            lead = (
                datetime.date.fromisoformat(r["target_date"])
                - datetime.date.fromisoformat(r["captured_at"][:10])
            ).days
            if lead not in (1, 2):
                continue
            p_old = price(r["forecast_high"], r["forecast_sigma"], r)
            if p_old is None:
                continue
            s, c = r["source"], r["city"]
            mu = r["forecast_high"] + delta.get((s, c), vm.get(s, 0.0))
            p_new = price(mu, r["forecast_sigma"] * scale.get(s, 1.0), r)
            old.append((p_old - r["outcome"]) ** 2)
            new.append((p_new - r["outcome"]) ** 2)
    return old, new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", default="data/captures.jsonl")
    ap.add_argument(
        "--since",
        default="2026-07-20",
        help="ignore captures before this date (last blend/table change). Residuals measure drift "
        "of the CURRENT tables, so rows priced under an EARLIER set of constants must be excluded "
        "-- pooling the 07-20 refit's before and after mixes two regimes and the walk-forward check "
        "correctly rejects the result (-2.8% vs +5.5% on post-refit rows alone).",
    )
    ap.add_argument(
        "--sigma-floor",
        type=float,
        default=0.8,
        help="lower clamp on the per-venue sigma scale: how far ONE refit may sharpen sigma "
        "(see module docstring)",
    )
    args = ap.parse_args()

    rows = load(args.captures)
    truth = realized_highs(rows)
    recs = residuals(rows, truth, args.since)
    print(f"{len(truth)} realized city-days, {len(recs)} residual observations (lead 1-2)")

    old, new = walk_forward(rows, recs, args.sigma_floor)
    if old:
        bo, bn = sum(old) / len(old), sum(new) / len(new)
        print(
            f"walk-forward Brier: old {bo:.4f} -> new {bn:.4f} "
            f"({(1 - bn / bo) * 100:+.1f}%) on n={len(old)}"
        )
        if bn >= bo:
            print("NO IMPROVEMENT — do not ship this fit.")

    vm, delta, scale = fit(recs, args.sigma_floor)
    print(f"\nvenue mean bias delta (degC, ADD to bias[1], bias[2]): "
          f"{ {k: round(v, 3) for k, v in vm.items()} }")
    print(f"venue sigma scale (MULTIPLY sigma[1], sigma[2]): "
          f"{ {k: round(v, 3) for k, v in scale.items()} }")
    counts = collections.Counter((s, c) for (_, s, c, _, _, _) in recs)
    print("\nper-city shrunk deltas (n>=4):")
    for (s, c), dv in sorted(delta.items(), key=lambda kv: kv[1]):
        if counts[(s, c)] >= 4:
            print(f"  {s:11s} {c:14s} {dv:+.2f}C (n={counts[(s, c)]})")


if __name__ == "__main__":
    main()
