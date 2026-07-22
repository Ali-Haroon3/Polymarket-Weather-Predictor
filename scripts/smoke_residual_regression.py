#!/usr/bin/env python3
"""Does target-day smoke/haze (PM2.5 / US AQI) suppress realized daily highs
relative to the blend forecast? Tests the capture daemon's log-only smoke fields
before anything in the pricing path reads them (stdlib only, no deps).

Design:
  y = realized_high - forecast_high  per resolved lead-1/2 capture row, deduped to
      one obs per (venue, city, target, lead); realized recovered from the unique
      WINNING bucket market (same hygiene as refit_station_tables.py), forecast_high
      already embeds the fitted station bias.
  x = target-day AQI/PM2.5 at city coords. Two sources:
      --stored   the daemon's logged us_aqi_max/pm25_max (a FORECAST of the target
                 day made at capture time — the tradeable version of the signal;
                 only populated since 2026-07-19)
      default    ACTUAL AQI backfilled from the Open-Meteo air-quality API
                 (per-UTC-date hourly max, the open_meteo_fetcher.rs collapse),
                 covering every resolved city-day since captures began 2026-06-29.

  The primary fit is WITHIN-CITY (city fixed effects via demeaning): chronically
  hazy cities (Beijing, Mexico City) must not proxy for city identity — the
  priceable claim is "an unusually smoky day for THAT city comes in cooler than
  its forecast", which would justify a mu shift / sigma widen keyed on the AQI
  forecast. Pooled OLS is printed for reference only.

First look (2026-07-22, --stored): only 19 resolved city-days carried the fields
(logging began 07-19) and the window contained NO smoke episode — max AQI 131 was
Mexico City's chronic urban haze; high-AQI residuals sat at ~0. Inconclusive, not
refuted. The backfill path needs open-meteo network access (the cloud session's
policy blocks it) — run it from the capture machine.

Usage: python3 scripts/smoke_residual_regression.py [--stored]
           [--captures data/captures.jsonl] [--min-city-days 3]
"""
import argparse
import collections
import datetime as dt
import json
import math
import re
import sys
import time
import urllib.parse
import urllib.request

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
US_CITIES = {"NYC", "LA", "Chicago", "Dallas", "Denver", "Miami", "Boston",
             "Seattle", "Atlanta", "Houston", "Phoenix", "Portland", "SF",
             "Austin", "Washington", "Philadelphia", "Vegas"}


def to_c(v, unit):
    return (v - 32.0) / 1.8 if (unit or "F") == "F" else v


def realized_highs(rows):
    """(venue, city, target) -> realized high degC from the UNIQUE winning bucket."""
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


def residual_obs(rows, truth):
    """(venue, city, target, lead) -> (residual degC, stored (pm25, aqi))."""
    recs = {}
    for r in rows:
        if r.get("outcome") is None or r.get("forecast_high") is None:
            continue
        key = (r.get("source", "polymarket"), r["city"], r["target_date"])
        if key not in truth:
            continue
        lead = (dt.date.fromisoformat(r["target_date"])
                - dt.date.fromisoformat(r["captured_at"])).days
        if lead not in (1, 2):
            continue
        recs[key + (lead,)] = (truth[key] - r["forecast_high"],
                               (r.get("pm25_max"), r.get("us_aqi_max")))
    return recs


def city_coords(path):
    """key -> (lat, lon) parsed from the City rows of src/cities.rs."""
    txt = open(path).read()
    pat = re.compile(r'key:\s*"([^"]+)".*?lat:\s*(-?[\d.]+),\s*lon:\s*(-?[\d.]+)', re.S)
    out = {}
    for m in re.finditer(r'City\s*\{(.*?)\}', txt, re.S):
        mm = pat.search(m.group(1))
        if mm:
            out[mm.group(1)] = (float(mm.group(2)), float(mm.group(3)))
    return out


def fetch_aqi(lat, lon, d0, d1):
    """date -> (pm25_max, aqi_max) per UTC date — the fetcher's hourly-max collapse."""
    q = urllib.parse.urlencode(dict(latitude=lat, longitude=lon,
                                    hourly="pm2_5,us_aqi", timezone="UTC",
                                    start_date=d0, end_date=d1))
    with urllib.request.urlopen(f"{AQ_URL}?{q}", timeout=30) as resp:
        h = json.load(resp).get("hourly", {})
    times, pm, aqi = h.get("time", []), h.get("pm2_5", []), h.get("us_aqi", [])
    out = collections.defaultdict(lambda: [None, None])
    for i, t in enumerate(times):
        for j, series in ((0, pm), (1, aqi)):
            v = series[i] if i < len(series) else None
            if v is not None:
                cur = out[t[:10]][j]
                out[t[:10]][j] = v if cur is None else max(cur, v)
    return {d: tuple(v) for d, v in out.items()}


def ols(pairs):
    """OLS y = a + b*x -> (n, b, se_b, t, r2)."""
    n = len(pairs)
    if n < 3:
        return n, float("nan"), float("nan"), float("nan"), float("nan")
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    if sxx <= 0:
        return n, float("nan"), float("nan"), float("nan"), float("nan")
    b = sxy / sxx
    a = my - b * mx
    sse = sum((y - a - b * x) ** 2 for x, y in pairs)
    syy = sum((y - my) ** 2 for y in ys)
    s2 = sse / (n - 2)
    se = math.sqrt(s2 / sxx) if s2 > 0 else 0.0
    return n, b, se, b / se if se > 0 else float("nan"), 1 - sse / syy if syy > 0 else 0.0


def within_city(day_rows, xi, min_days):
    """City-demeaned (x, y) pairs from (city, date, res, pm, aqi) rows; xi picks pm|aqi."""
    by_city = collections.defaultdict(list)
    for r in day_rows:
        if r[xi] is not None:
            by_city[r[0]].append(r)
    pairs = []
    for lst in by_city.values():
        if len(lst) < min_days:
            continue
        mx = sum(r[xi] for r in lst) / len(lst)
        my = sum(r[2] for r in lst) / len(lst)
        pairs.extend((r[xi] - mx, r[2] - my) for r in lst)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", default="data/captures.jsonl")
    ap.add_argument("--cities-rs", default="src/cities.rs")
    ap.add_argument("--stored", action="store_true",
                    help="use the daemon's logged AQI forecasts instead of backfilling actuals")
    ap.add_argument("--min-city-days", type=int, default=3,
                    help="min days per city before it enters the within-city fit")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.captures) if l.strip()]
    truth = realized_highs(rows)
    recs = residual_obs(rows, truth)
    print(f"{len(truth)} realized city-days, {len(recs)} residual obs (lead 1-2)")

    # one obs per (city, target): venue/lead residuals share the smoke day, so
    # averaging avoids counting the same day as several observations.
    per_day = collections.defaultdict(list)
    for (venue, city, target, lead), (res, stored) in recs.items():
        per_day[(city, target)].append((res, stored))

    aqi = {}  # (city, date) -> (pm25, aqi)
    if args.stored:
        for (city, target), lst in per_day.items():
            pms = [s[0] for _, s in lst if s[0] is not None]
            aqs = [s[1] for _, s in lst if s[1] is not None]
            if pms or aqs:
                aqi[(city, target)] = (max(pms) if pms else None, max(aqs) if aqs else None)
    else:
        need = collections.defaultdict(set)
        for (city, target) in per_day:
            need[city].add(target)
        coords = city_coords(args.cities_rs)
        for city, dates in sorted(need.items()):
            if city not in coords:
                print(f"  no coords, skipped: {city}", file=sys.stderr)
                continue
            try:
                got = fetch_aqi(*coords[city], min(dates), max(dates))
            except Exception as e:
                print(f"  fetch FAILED {city}: {e}", file=sys.stderr)
                continue
            for d in dates:
                if d in got and got[d][1] is not None:
                    aqi[(city, d)] = got[d]
            time.sleep(0.3)

    day_rows = [(city, target, sum(r for r, _ in lst) / len(lst)) + aqi[(city, target)]
                for (city, target), lst in per_day.items() if (city, target) in aqi]
    print(f"{len(day_rows)} city-days with AQI ({'stored forecasts' if args.stored else 'backfilled actuals'})\n")
    if not day_rows:
        sys.exit("nothing to fit")

    for tag, sel in [("ALL cities", day_rows),
                     ("US only", [r for r in day_rows if r[0] in US_CITIES])]:
        print(f"== {tag} ==")
        n, b, se, t, r2 = ols([(r[4], r[2]) for r in sel if r[4] is not None])
        print(f"  pooled  res~AQI       n={n:4d}  slope={b * 100:+.3f} C/100AQI  t={t:+.2f}  r2={r2:.3f}")
        n, b, se, t, r2 = ols(within_city(sel, 4, args.min_city_days))
        print(f"  within  res~AQI anom  n={n:4d}  slope={b * 100:+.3f} C/100AQI  t={t:+.2f}  r2={r2:.3f}")
        n, b, se, t, r2 = ols(within_city(sel, 3, args.min_city_days))
        print(f"  within  res~PM25 anom n={n:4d}  slope={b * 10:+.3f} C/10ug    t={t:+.2f}  r2={r2:.3f}")

    anom = []
    by_city = collections.defaultdict(list)
    for r in day_rows:
        if r[4] is not None:
            by_city[r[0]].append(r)
    for lst in by_city.values():
        if len(lst) < args.min_city_days:
            continue
        ma = sum(r[4] for r in lst) / len(lst)
        anom.extend((r[4] - ma, r[2], r[0], r[1]) for r in lst)

    print("\n== residual by within-city AQI-anomaly bin ==")
    for lo, hi, lab in [(-1e9, -20, "calm   (<-20)"), (-20, 20, "normal"),
                        (20, 50, "hazy  (+20..50)"), (50, 1e9, "smoke  (>+50)")]:
        grp = [a[1] for a in anom if lo <= a[0] < hi]
        if grp:
            m = sum(grp) / len(grp)
            sd = math.sqrt(sum((g - m) ** 2 for g in grp) / max(1, len(grp) - 1))
            print(f"  {lab:16s} n={len(grp):4d}  mean_res={m:+.2f}C  se={sd / math.sqrt(len(grp)):.2f}")

    print("\n== 12 largest AQI-anomaly city-days ==")
    for a, res, city, d in sorted(anom, reverse=True)[:12]:
        print(f"  {city:14s} {d}  anom={a:+6.1f}  residual={res:+.2f}C")


if __name__ == "__main__":
    main()
