#!/usr/bin/env python3
"""Does market microstructure predict how much of the model's claimed edge is REAL?

The capture daemon logs attention proxies (volume / volume_24h / liquidity, populated
since 2026-07-19 and Polymarket-only in practice — Kalshi's anonymous feed nulls them;
open_interest has NEVER held a value on either venue, Polymarket structurally has
none, and it is kept in FEATURES only for the day authenticated Kalshi fetches fill
it) and the top of book (best_bid / best_ask) on every fresh snapshot, log-only. This script is the first consumer: it conditions the
edge-shrinkage slope λ (through-origin regression of realized on claimed edge, the
same statistic `backtesting::ShrinkageFit` fits) on those fields, per venue.

Motivation: the strongest public competitor (a Kalshi XGBoost bot) feeds market
microstructure into its model outright. Before copying that, measure whether OUR λ
even varies with microstructure — if λ collapses in thin/wide books, a cheap filter
(the house Filter-A/B route) captures most of the benefit with none of the ML.

Fit hygiene matches ShrinkageFit / diagnose_lambda_window.py: resolved lead >= 1 rows
only, priced at book mid falling back to last trade. Two extra views:

  spread     needs both book sides; quartiles of (ask - bid). Also reports λ_exec —
             claimed/realized edge re-priced at the EXECUTABLE side (BUY fills at the
             ask, SELL at the bid, direction by estimate vs mid) — the number a trade
             would actually have earned.
  missing    rows where the field is null (old captures or venue omits it), reported
             as its own bucket so absence can't silently skew the quartiles.

Stdlib only, no deps.

First look (2026-07-23, 1,628 resolved lead>=1 rows, tick-rounded spreads): Polymarket
wide books are where claimed edge goes to die — spread Q4 (> 2c) has lambda_exec =
-0.29 (n=232), i.e. the model's disagreements with wide-book prices are
ANTI-predictive at the fill price, while tight books (Q1, < 1c) are efficiently priced
with little edge either way (lambda_exec 0.16). A "skip Polymarket books wider than
~2c" Filter-A/B candidate is the concrete follow-up. Kalshi has NO sub-1c books (min
tick; Q1 is empty by construction) and is NOT monotone above it: 1c books ~0 (n=319),
2-3c negative (-0.15, n=160), >3c +0.44 with the best pnl/trade (n=205) — likely
confounded with bucket moneyness; needs more data before any filter. Attention proxies
(volume/liquidity) are only ~27 resolved rows per quartile so far — noise; revisit in
August. The "missing" buckets are old pre-book-logging rows: a TIME cohort (lambda
0.83 era), not a liquidity readout — never compare quartiles against them.

Usage: python3 scripts/microstructure_lambda.py [--captures data/captures.jsonl]
                                                [--since YYYY-MM-DD] [--buckets 4]
"""

import argparse
import collections
import datetime as dt
import json


FEATURES = ["volume", "volume_24h", "open_interest", "liquidity", "spread"]


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def lead(r):
    return (
        dt.date.fromisoformat(r["target_date"]) - dt.date.fromisoformat(r["captured_at"])
    ).days


def mid_price(r):
    # Same in-range gate as the dashboard's ref_price: a 0/1 side is an empty book side.
    b, a = r.get("best_bid"), r.get("best_ask")
    if b is not None and a is not None and 0.0 < b <= a < 1.0:
        return (b + a) / 2.0
    return r["entry_price"]


def lam(pairs):
    """Through-origin slope of realized on claimed edge; None on zero claimed mass."""
    sxy = sum(x * y for x, y in pairs)
    sxx = sum(x * x for x, y in pairs)
    return (sxy / sxx) if sxx > 0 else None


def rows_for_fit(rows, since):
    out = []
    for r in rows:
        if r.get("outcome") is None or r.get("model_estimate") is None:
            continue
        if lead(r) < 1:
            continue
        if since and r["captured_at"] < since:
            continue
        out.append(r)
    return out


def quartile_edges(values, buckets):
    vs = sorted(values)
    return [vs[int(len(vs) * i / buckets)] for i in range(1, buckets)]


def bucket_of(v, edges):
    for i, e in enumerate(edges):
        if v < e:
            return i
    return len(edges)


def feature_value(r, feat):
    if feat == "spread":
        b, a = r.get("best_bid"), r.get("best_ask")
        if b is None or a is None:
            return None
        # Round away float noise: nominal 1-cent spreads otherwise carry a dozen distinct
        # float representations that straddle quartile edges arbitrarily.
        return round(a - b, 4)
    return r.get(feat)


def executable_pair(r):
    """(claimed, realized) at the fill price a trade would get, or None without a book.

    Direction by estimate vs mid, like the dashboard's decide(): BUY fills at the ask
    (claimed = est - ask, realized = outcome - ask); SELL fills at the bid
    (claimed = bid - est, realized = bid - outcome).
    """
    b, a = r.get("best_bid"), r.get("best_ask")
    if b is None or a is None:
        return None
    est, out = r["model_estimate"], r["outcome"]
    if est >= (b + a) / 2.0:
        return (est - a, out - a)
    return (b - est, b - out)


def print_table(title, buckets_dict, order):
    print(f"\n  {title}")
    print(f"    {'bucket':<18}{'n':>6}{'lambda':>9}{'mean|claim|':>13}{'pnl/trade':>11}")
    for name in order:
        pairs = buckets_dict.get(name, [])
        if not pairs:
            continue
        l = lam(pairs)
        mean_claim = sum(abs(x) for x, _ in pairs) / len(pairs)
        # PnL of always trading the claimed direction for 1 unit: sign(claim) * realized.
        pnl = sum((y if x >= 0 else -y) for x, y in pairs) / len(pairs)
        lstr = f"{l:.3f}" if l is not None else "-"
        print(f"    {name:<18}{len(pairs):>6}{lstr:>9}{mean_claim:>13.4f}{pnl:>11.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", default="data/captures.jsonl")
    ap.add_argument("--since", default=None, help="only captures on/after this date")
    ap.add_argument("--buckets", type=int, default=4)
    args = ap.parse_args()

    rows = rows_for_fit(load(args.captures), args.since)
    print(f"resolved lead>=1 rows in fit: {len(rows)}")

    for venue in ("polymarket", "kalshi"):
        vrows = [r for r in rows if r.get("source", "polymarket") == venue]
        if not vrows:
            continue
        print(f"\n== {venue} (n={len(vrows)}) ==")
        present = {
            f: sum(1 for r in vrows if feature_value(r, f) is not None) for f in FEATURES
        }
        print("  field coverage:", ", ".join(f"{f}={n}" for f, n in present.items()))

        for feat in FEATURES:
            valued = [(r, feature_value(r, feat)) for r in vrows]
            have = [(r, v) for r, v in valued if v is not None]
            if len(have) < 4 * args.buckets:
                continue
            edges = quartile_edges([v for _, v in have], args.buckets)
            groups = collections.defaultdict(list)
            for r, v in valued:
                x = r["model_estimate"] - mid_price(r)
                y = r["outcome"] - mid_price(r)
                name = "missing" if v is None else f"Q{bucket_of(v, edges) + 1}"
                groups[name].append((x, y))
            order = [f"Q{i + 1}" for i in range(args.buckets)] + ["missing"]
            label = f"{feat} (quartile edges: " + ", ".join(f"{e:.3g}" for e in edges) + ")"
            print_table(label, groups, order)

            if feat == "spread":
                exec_groups = collections.defaultdict(list)
                for r, v in have:
                    pair = executable_pair(r)
                    if pair is not None:
                        exec_groups[f"Q{bucket_of(v, edges) + 1}"].append(pair)
                print_table("spread -> EXECUTABLE-price edge (λ_exec)", exec_groups, order)


if __name__ == "__main__":
    main()
