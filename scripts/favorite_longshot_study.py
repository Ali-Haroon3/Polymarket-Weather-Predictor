#!/usr/bin/env python3
"""Favorite-longshot study on Kalshi daily-high ladders (research only; no orders, no canonical writes).

Three questions, all on lead-1 Kalshi capture rows (captured_at = target - 1):

1. bands  - taker calibration: realized return of buying YES at the captured ask, or NO at
            1 - captured bid, by price band, after Kalshi's rounded taker fee.
2. rule   - the frozen favorite rule (reports/2026-10-12-favorite-longshot-study.md): buy YES at
            the captured ask whenever 0.50 <= ask < 0.99, $15 including the rounded fee. No fitted
            parameters. Reported for the in-sample period and for the forward window
            (capture dates FORWARD_START..FORWARD_END), which is the only part that counts.
3. ledger - the market-shape pilot's actual paper orders split by side and YES price >= 0.50.

With --tape (from --fetch-tape), also simulates RESTING (maker) orders against Kalshi's public
trade tape: an order posted at cap-day 15:30 UTC fills only when a later taker trade crosses it.

python3 scripts/favorite_longshot_study.py
python3 scripts/favorite_longshot_study.py --fetch-tape research-output/kalshi-tape.jsonl
python3 scripts/favorite_longshot_study.py --tape research-output/kalshi-tape.jsonl
"""

import argparse
import collections
import datetime as dt
import json
import math
from pathlib import Path
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import go_live_gate as gate
import weather_challenger as challenger

THRESHOLD, CEILING = 0.50, 0.99
FREEZE = "2026-10-12"                       # last capture date inspected before the rule was frozen
FORWARD_START, FORWARD_END = "2026-10-13", "2026-12-11"
BANDS = ((0.02, 0.10), (0.10, 0.20), (0.20, 0.35), (0.35, 0.50), (0.50, 0.65), (0.65, 0.80),
         (0.80, 0.90), (0.90, 0.98))
TAPE_BASE = "https://api.elections.kalshi.com/trade-api/v2/"


def lead1_rows(path):
    for line in open(path):
        r = json.loads(line)
        if r.get("source") != "kalshi":
            continue
        if (dt.date.fromisoformat(r["target_date"]) - dt.date.fromisoformat(r["captured_at"][:10])).days == 1:
            yield r


def price(value):
    return challenger.usable_price(value)


def taker_bands(rows):
    out = {}
    for side in ("yes", "no"):
        for lo, hi in BANDS:
            n = wins = cost = net = 0
            for r in rows:
                if r["outcome"] is None:
                    continue
                p = price(r["best_ask"]) if side == "yes" else (
                    1 - r["best_bid"] if price(r["best_bid"]) is not None else None)
                if p is None or not lo <= p < hi:
                    continue
                order = challenger.size_order(p)
                if order is None:
                    continue
                won = r["outcome"] == (1 if side == "yes" else 0)
                n, wins = n + 1, wins + won
                cost, net = cost + order["cost"], net + order["contracts"] * won - order["cost"]
            if n:
                out[f"buy_{side}_{lo:.2f}_{hi:.2f}"] = dict(orders=n, win_rate=wins / n, roi=net / cost)
    return out


def favorite_trades(rows, first, last):
    """The frozen rule. Outcomes are read only after the order list is fixed."""
    trades, unresolved = [], 0
    for r in rows:
        ask = price(r["best_ask"])
        if ask is None or not THRESHOLD <= ask < CEILING or not first <= r["captured_at"][:10] <= last:
            continue
        order = challenger.size_order(ask)
        if order is None:
            continue
        if r["outcome"] is None:
            unresolved += 1
            continue
        won = r["outcome"] == 1
        stress = challenger.size_order(ask + 0.01)
        trades.append(dict(order, ticker=r["market_id"], city=r["city"], target_date=r["target_date"],
                           run_at=r["captured_at"][:10], side="yes", price=ask, won=won,
                           net=order["contracts"] * won - order["cost"],
                           stress_1c=None if stress is None else dict(
                               stress, net=stress["contracts"] * won - stress["cost"])))
    return trades, unresolved


def ledger_split(captures, ledger):
    outcomes = {}
    for line in open(captures):
        r = json.loads(line)
        if r["outcome"] is not None:
            outcomes[r["market_id"]] = r["outcome"]
    groups = collections.defaultdict(lambda: dict(orders=0, wins=0, cost=0.0, net=0.0))
    for line in open(ledger):
        r = json.loads(line)
        if r.get("strategy") != "market-shape" or r.get("decision") != "order" or not r.get("contracts"):
            continue
        outcome = outcomes.get(r["ticker"])
        if outcome is None:
            continue
        c, p = r["contracts"], r["price"]
        won = outcome == (1 if r["side"] == "yes" else 0)
        cost = c * p + gate.kalshi_fee(c, p)
        key = "no" if r["side"] == "no" else ("yes_at_or_above_0.50" if p >= THRESHOLD else "yes_below_0.50")
        g = groups[key]
        g["orders"] += 1; g["wins"] += won; g["cost"] += cost; g["net"] += c * won - cost
    return {k: dict(v, roi=v["net"] / v["cost"]) for k, v in sorted(groups.items())}


_lock, _next = threading.Lock(), [0.0]


def _get(path, params):
    url = TAPE_BASE + path + "?" + urllib.parse.urlencode(params)
    for attempt in range(10):
        with _lock:  # ponytail: one global ~12 req/s spacing; Kalshi's basic tier 429s near 20/s
            now = time.monotonic(); wait = max(0.0, _next[0] - now); _next[0] = max(now, _next[0]) + 1 / 12
        time.sleep(wait)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research"}), timeout=30) as f:
                return json.loads(f.read())
        except Exception:
            time.sleep(min(30, 2 * (attempt + 1)))
    raise RuntimeError(url)


def _tape(row):
    min_ts = int(dt.datetime.fromisoformat(row["captured_at"][:10] + "T14:00:00+00:00").timestamp())
    for path in ("markets/trades", "historical/trades"):  # recent markets, then archived ones
        trades, cursor = [], None
        while True:
            params = dict(ticker=row["market_id"], limit=1000, min_ts=min_ts, **({"cursor": cursor} if cursor else {}))
            page = _get(path, params)
            trades += page.get("trades", [])
            cursor = page.get("cursor")
            if not cursor:
                break
        if trades:
            break
    return dict(ticker=row["market_id"], endpoint=path if trades else None, trades=[
        dict(ts=t["created_time"], side=t["taker_side"], yes=float(t["yes_price_dollars"]), n=float(t["count_fp"]))
        for t in trades])


def fetch_tape(rows, out):
    done = set()
    if out.exists():
        done = {json.loads(line)["ticker"] for line in open(out)}
    todo = [r for r in rows if r["outcome"] is not None and r["market_id"] not in done]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f, ThreadPoolExecutor(6) as pool:
        for rec in pool.map(_tape, todo):
            f.write(json.dumps(rec) + "\n"); f.flush()
    return len(todo)


HOLDOUT_BEFORE = "2026-07-05"   # first capture date; report section 5 protocol
SERIES = ("KXHIGHNY", "KXHIGHCHI", "KXHIGHAUS", "KXHIGHDEN", "KXHIGHLAX", "KXHIGHMIA", "KXHIGHPHIL", "KXHIGHTDAL",
          "KXHIGHTSEA", "KXHIGHTATL", "KXHIGHTBOS", "KXHIGHTPHX", "KXHIGHTLV", "KXHIGHTDC", "KXHIGHTHOU")
MONTHS = dict(JAN=1, FEB=2, MAR=3, APR=4, MAY=5, JUN=6, JUL=7, AUG=8, SEP=9, OCT=10, NOV=11, DEC=12)


def decision_ts(target):
    return int((dt.datetime.fromisoformat(target + "T15:00:00+00:00") - dt.timedelta(days=1)).timestamp())


REPLICATION_LAST = "2026-10-11"  # report section 6 protocol
REPLICATION_EXCLUDE = set(SERIES) | {"HIGHAUS", "HIGHCHI", "HIGHMIA", "HIGHNY", "KXHIGHHOU", "KXHIGHOU", "KXHOUHIGH",
                                     "KXDENHIGH", "KXHIGHTEMPDEN", "KXPHILHIGH", "HIGHUS", "KXHIGHUS", "KXHIGHNYD", "KXSJCLOWT"}


def _window(market):
    t = decision_ts(market["target"])
    paths = ("historical/trades", "markets/trades") if market["target"] < "2026-08-15" else ("markets/trades", "historical/trades")
    for path in paths:  # archived markets live under /historical; recent ones under /markets
        trades, cursor = [], None
        while True:
            page = _get(path, dict(ticker=market["ticker"], limit=1000, min_ts=t - 6 * 3600,
                                   max_ts=t + 2 * 3600, **({"cursor": cursor} if cursor else {})))
            trades += page.get("trades", [])
            cursor = page.get("cursor")
            if not cursor:
                break
        if trades:
            break
    return dict(market, trades=[dict(ts=x["created_time"], side=x["taker_side"], yes=float(x["yes_price_dollars"]),
                                     n=float(x["count_fp"])) for x in trades])


def list_markets(series, keep):
    """Finalized yes/no markets of the given series whose ticker target date satisfies keep(target)."""
    markets, seen = [], set()
    for name in series:
        for path, extra in (("historical/markets", {}), ("markets", {"status": "settled"})):
            cursor = None
            while True:
                page = _get(path, dict(series_ticker=name, limit=1000, **extra, **({"cursor": cursor} if cursor else {})))
                for m in page.get("markets", []):
                    parts = m["ticker"].split("-")
                    code = parts[1] if len(parts) > 2 else ""
                    if len(code) != 7 or code[2:5] not in MONTHS or m.get("result") not in ("yes", "no") or m["ticker"] in seen:
                        continue
                    target = f"20{code[:2]}-{MONTHS[code[2:5]]:02d}-{code[5:]}"
                    if keep(target):
                        seen.add(m["ticker"])
                        markets.append(dict(ticker=m["ticker"], series=name, target=target, result=m["result"]))
                cursor = page.get("cursor")
                if not cursor or not page.get("markets"):
                    break
    return markets


def replication_series():
    """Section 6: daily Climate-and-Weather series titled as a high/low temperature, minus the exclusions."""
    out = []
    for s in _get("series", {"category": "Climate and Weather", "limit": 1000})["series"]:
        title = (s.get("title") or "").lower()
        if s["ticker"] in REPLICATION_EXCLUDE or s.get("frequency") != "daily" or "temp" not in title:
            continue
        if any(w in title.split() for w in ("highest", "high", "max", "maximum", "lowest", "low", "min", "minimum")):
            out.append(s["ticker"])
    return sorted(out)


def fetch_windows(markets, out):
    done = {json.loads(line)["ticker"] for line in open(out)} if out.exists() else set()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f, ThreadPoolExecutor(5) as pool:
        for rec in pool.map(_window, [m for m in markets if m["ticker"] not in done]):
            f.write(json.dumps(rec) + "\n"); f.flush()
    return len(markets)


def fetch_holdout(out):
    """Section 5: every finalized pre-capture market in SERIES, with taker prints in [T-6h, T+2h)."""
    return fetch_windows(list_markets(SERIES, lambda target: target < HOLDOUT_BEFORE), out)


def fetch_replication(out):
    """Section 6: every finalized market of the untouched series with target <= REPLICATION_LAST."""
    return fetch_windows(list_markets(replication_series(), lambda target: target <= REPLICATION_LAST), out)


def holdout_trades(tape_path):
    """Section 5 protocol: signal = last taker-YES print before T; fill = first taker-YES print in [T, T+2h)."""
    trades = []
    for line in open(tape_path):
        m = json.loads(line)
        t = decision_ts(m["target"])
        yes = sorted((dt.datetime.fromisoformat(x["ts"].replace("Z", "+00:00")).timestamp(), x["yes"])
                     for x in m["trades"] if x["side"] == "yes")
        before = [p for ts, p in yes if t - 6 * 3600 <= ts < t]
        after = [p for ts, p in yes if t <= ts < t + 2 * 3600]
        if not before or not THRESHOLD <= before[-1] < CEILING or not after:
            continue
        fill = after[0]
        order = challenger.size_order(fill)
        if order is None:
            continue
        won = m["result"] == "yes"
        stress = challenger.size_order(fill + 0.01)
        trades.append(dict(order, ticker=m["ticker"], city=m["ticker"].split("-")[0].removeprefix("KX").removeprefix("HIGHT").removeprefix("HIGH"),
                           target_date=m["target"],
                           run_at=m["target"], side="yes", price=fill, won=won,
                           net=order["contracts"] * won - order["cost"],
                           stress_1c=None if stress is None else dict(stress, net=stress["contracts"] * won - stress["cost"])))
    return trades


def maker_sim(rows, tape_path, post_hour=15.5, stake=15.0):
    """Resting order 1c inside the spread (alone at the top), filled by later taker trades at or through it."""
    snap = {r["market_id"]: r for r in rows if r["outcome"] is not None}
    results = collections.defaultdict(lambda: dict(posted=0, filled=0, wins=0, cost=0.0, net=0.0))
    for line in open(tape_path):
        tape = json.loads(line)
        r = snap.get(tape["ticker"])
        if r is None or price(r["best_bid"]) is None or price(r["best_ask"]) is None:
            continue
        b, a = r["best_bid"], r["best_ask"]
        t0 = dt.datetime.fromisoformat(r["captured_at"][:10] + "T00:00:00+00:00") + dt.timedelta(hours=post_hour)
        trades = sorted(((dt.datetime.fromisoformat(t["ts"].replace("Z", "+00:00")), t) for t in tape["trades"]),
                        key=lambda pair: pair[0])
        for side in ("sell_yes", "buy_yes"):
            p = round(a - 0.01, 2) if side == "sell_yes" else round(b + 0.01, 2)
            if not b < p < a:
                continue
            cost_per = 1 - p if side == "sell_yes" else p
            won = r["outcome"] == (0 if side == "sell_yes" else 1)
            want, filled = int(stake // cost_per), 0.0
            for ts, t in trades:
                if ts < t0:
                    continue
                crosses = (t["side"] == "yes" and t["yes"] >= p - 1e-9) if side == "sell_yes" else (
                    t["side"] == "no" and t["yes"] <= p + 1e-9)
                if crosses:
                    filled += t["n"]
                    if filled >= want:
                        break
            filled = min(filled, want)
            band = next((f"{lo:.2f}_{hi:.2f}" for lo, hi in BANDS if lo <= p < hi), "other")
            g = results[f"{side}_{band}"]
            g["posted"] += 1
            if filled:
                g["filled"] += 1; g["wins"] += won; g["cost"] += filled * cost_per
                g["net"] += filled * ((1 - cost_per) if won else -cost_per)
    return {k: dict(v, roi=v["net"] / v["cost"] if v["cost"] else None) for k, v in sorted(results.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--captures", type=Path, default=Path("data/captures.jsonl"))
    parser.add_argument("--ledger", type=Path, default=Path("data/pilot_trades.jsonl"))
    parser.add_argument("--tape", type=Path, help="Trade-tape JSONL from --fetch-tape; enables the maker simulation.")
    parser.add_argument("--fetch-tape", type=Path, help="Download the public trade tape for resolved lead-1 rows, then exit.")
    parser.add_argument("--holdout", type=Path, help="Window tape from --fetch-holdout or --fetch-replication; scores it.")
    parser.add_argument("--fetch-holdout", type=Path, help="Download pre-capture markets' decision windows, then exit (~1 h).")
    parser.add_argument("--fetch-replication", type=Path, help="Download section 6 untouched-series windows, then exit (~1.5 h).")
    args = parser.parse_args()
    rows = list(lead1_rows(args.captures))
    if args.fetch_replication:
        print(f"{fetch_replication(args.fetch_replication)} replication markets in {args.fetch_replication}")
        return
    if args.fetch_holdout:
        print(f"{fetch_holdout(args.fetch_holdout)} holdout markets in {args.fetch_holdout}")
        return
    if args.fetch_tape:
        print(f"fetched {fetch_tape(rows, args.fetch_tape)} markets into {args.fetch_tape}")
        return
    in_sample, _ = favorite_trades(rows, "0000-00-00", FREEZE)
    forward, unresolved = favorite_trades(rows, FORWARD_START, FORWARD_END)
    report = dict(rule=dict(threshold=THRESHOLD, ceiling=CEILING, stake_including_fee=challenger.STAKE,
                            freeze=FREEZE, forward_window=[FORWARD_START, FORWARD_END]),
                  taker_bands=taker_bands(rows),
                  favorite_in_sample=challenger.summarize(in_sample),
                  favorite_forward=dict(challenger.summarize(forward), unresolved=unresolved),
                  pilot_ledger=ledger_split(args.captures, args.ledger))
    if args.holdout:
        trades = holdout_trades(args.holdout)
        report["holdout"] = dict(challenger.summarize(trades), by_year={
            year: challenger.summarize([t for t in trades if t["target_date"].startswith(year)])
            for year in sorted({t["target_date"][:4] for t in trades})})
    if args.tape:
        report["maker_improve_by_band"] = maker_sim(rows, args.tape)
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
