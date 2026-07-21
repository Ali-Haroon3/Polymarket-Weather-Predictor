#!/usr/bin/env python3
"""Attribute a bad λ window to cities and forecast errors.

When the dashboard's trailing-window λ slope dips (see "Edge shrinkage λ over time"), this
answers "who did it, and was the model too hot or too cold?" from data/captures.jsonl alone:

  python3 scripts/diagnose_lambda_window.py 2026-07-13 2026-07-16

Per (city, venue) it reports the contribution to the window's through-origin slope
(Σxy decomposes exactly across rows, so `share` sums the negative mass) and the mean
forecast − realized high error (°C), with the realized high recovered from the winning
resolved bucket market of the same (city, venue, day). Fit hygiene matches ShrinkageFit:
resolved lead ≥ 1 rows only, priced at book mid falling back to last trade.

First used on the 2026-07-13..16 window (venue-pooled slope −0.23): Kalshi
Chicago/NYC/Philadelphia/Denver carried ~47% of the negative mass, with the model
over-forecasting highs by +1.9..+3.1 °C everywhere the damage was — the summer
over-forecast signature the 2026-07-20 seasonal refit corrects.
"""

import collections
import datetime as dt
import json
import signal
import sys
from pathlib import Path

signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # exit quietly when piped into `head`

CAPTURES = Path(__file__).resolve().parent.parent / "data" / "captures.jsonl"


def lead(r):
    return (dt.date.fromisoformat(r["target_date"]) - dt.date.fromisoformat(r["captured_at"])).days


def ref_price(r):
    b, a = r.get("best_bid"), r.get("best_ask")
    if b is not None and a is not None and b > 0.0 and a < 1.0 and b <= a:
        px = (a + b) / 2.0
    else:
        px = r["entry_price"]
    return px if 0.0 < px < 1.0 else None


def f2c(f):
    return (f - 32.0) * 5.0 / 9.0


def realized_highs(rows):
    """(city, venue, target_date) -> realized high °C, from the winning resolved bucket market
    (bucket midpoint; half-open buckets use the threshold itself)."""
    out = {}
    for r in rows:
        if r.get("outcome") != 1.0 or r["market_type"] != "temp_bucket":
            continue
        t, tu = r["threshold"], r.get("threshold_upper")
        mid = (t + tu) / 2.0 if tu is not None else t
        c = f2c(mid) if (r.get("unit") or "F") == "F" else mid
        out[(r["city"], r.get("source", "polymarket"), r["target_date"])] = c
    return out


def main():
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <window-start> <window-end>  (target dates, YYYY-MM-DD)")
    w1, w2 = (dt.date.fromisoformat(a) for a in sys.argv[1:3])
    rows = [json.loads(l) for l in CAPTURES.read_text().splitlines() if l.strip()]
    highs = realized_highs(rows)

    contrib = collections.defaultdict(lambda: [0.0, 0.0, 0])  # (city, venue) -> [Sxx, Sxy, n]
    ferr = collections.defaultdict(dict)  # (city, venue) -> {(date, lead): forecast - realized}
    for r in rows:
        est, out = r.get("model_estimate"), r.get("outcome")
        if est is None or out is None or lead(r) < 1:
            continue
        td = dt.date.fromisoformat(r["target_date"])
        if not (w1 <= td <= w2):
            continue
        px = ref_price(r)
        if px is None:
            continue
        key = (r["city"], r.get("source", "polymarket"))
        x = est - px
        c = contrib[key]
        c[0] += x * x
        c[1] += x * (out - px)
        c[2] += 1
        fh = r.get("forecast_high")
        rh = highs.get((key[0], key[1], r["target_date"]))
        if fh is not None and rh is not None:
            ferr[key][(r["target_date"], lead(r))] = fh - rh

    sxx = sum(c[0] for c in contrib.values())
    sxy = sum(c[1] for c in contrib.values())
    neg = sum(c[1] for c in contrib.values() if c[1] < 0)
    print(f"window {w1}..{w2}: venue-pooled slope {sxy / sxx:+.3f} (Sxx={sxx:.3f}, Sxy={sxy:.3f})")
    print(f"{'city':<14}{'venue':<12}{'n':>4}{'Sxy':>9}{'slope':>8}{'share':>7}  mean fcst−real °C")
    for (city, venue), (xx, xy, n) in sorted(contrib.items(), key=lambda kv: kv[1][1]):
        share = f"{xy / neg * 100.0:5.1f}%" if xy < 0 and neg < 0 else "     –"
        errs = list(ferr[(city, venue)].values())
        e = f"{sum(errs) / len(errs):+5.2f} ({len(errs)} d/l)" if errs else "    – (no bucket win)"
        print(f"{city:<14}{venue:<12}{n:>4}{xy:>9.3f}{xy / xx if xx > 0 else 0.0:>8.2f}{share:>7}  {e}")


if __name__ == "__main__":
    main()
