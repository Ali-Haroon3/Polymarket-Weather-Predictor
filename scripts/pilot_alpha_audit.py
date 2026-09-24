#!/usr/bin/env python3
"""Fee-inclusive prospective pilot audit; stdlib only, no orders or network calls.

Compare the two sides already frozen in the 2026-09-07 dashboard experiment. The
ledger subset is attribution, NOT a replay of a NO-only strategy with replacement
orders. Optional --replay evaluates capture-time candidates with integer $15
stakes, 5 orders/run, $30/city-target and ticker deduplication. Select BEFORE
looking at outcomes; open bets still consume capacity. Its snapshots differ from
the pilot's later API scan, so it is a research comparison, not fill evidence.
"""
import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import random
from pathlib import Path

import go_live_gate as gate
import market_shape_alpha as shape

FREEZE = "2026-09-07"
SHADOW_FREEZE = "2026-09-21"


def summarize(trades):
    cost = sum(t["cost"] for t in trades)
    net = sum(t["net"] for t in trades)
    # Resample whole target days: same-day weather and ladder bets are correlated.
    groups = collections.defaultdict(lambda: [0.0, 0.0])
    for t in trades:
        v = groups[t["target_date"]]
        v[0] += t["net"]
        v[1] += t["cost"]
    days = [groups[k] for k in sorted(groups)]
    ci = None
    if len(days) >= 2:
        rng = random.Random(42)
        draws = []
        for _ in range(10000):
            sample = rng.choices(days, k=len(days))
            risk = sum(x[1] for x in sample)
            if risk:
                draws.append(sum(x[0] for x in sample) / risk)
        draws.sort()
        ci = [draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]]
    return dict(
        settled=len(trades), target_days=len(days), wins=sum(t["won"] for t in trades),
        cost=cost, net=net, fees=sum(t["fee"] for t in trades),
        roi=net / cost if cost else None, bootstrap_95=ci,
        # Hold selected contracts fixed; a 1-cent worse entry on every contract.
        net_with_1c_adverse_fill=net - sum(t["contracts"] for t in trades) * 0.01,
    )


def select_candidates(candidates, side=None):
    """Fixed research policy; no outcomes consulted until the full selection is final."""
    selected, committed = [], set()
    city_exposure = collections.defaultdict(float)
    counts = collections.Counter()
    exposure = collections.defaultdict(float)
    for t in sorted(candidates, key=lambda x: (x["run_at"], -x["edge"], x["ticker"])):
        day = t["run_at"]
        city_key = (t["city"], t["target_date"])
        if side and t["side"] != side:
            continue
        if t["ticker"] in committed or counts[day] >= 5:
            continue
        n = math.floor(15.0 / t["price"] + 1e-10)
        cost = n * t["price"]
        if not n or city_exposure[city_key] + cost > 30.0 + 1e-9:
            continue
        if exposure[day] + cost > 150.0:
            continue
        counts[day] += 1
        exposure[day] += cost
        city_exposure[city_key] += cost
        committed.add(t["ticker"])
        selected.append(dict(t, contracts=n, cost=cost))
    settled, opened = [], 0
    for t in selected:
        if t["outcome"] is None:
            opened += 1
            continue
        won = t["outcome"] == (1 if t["side"] == "yes" else 0)
        fee = gate.kalshi_fee(t["contracts"], t["price"])
        settled.append(dict(t, won=won, fee=fee,
                            net=(t["contracts"] if won else 0) - t["cost"] - fee))
    return settled, opened


def replay_candidates(captures, since):
    ladders = shape.build_ladders(captures)
    fits, candidates = {}, []
    for l in sorted(ladders, key=lambda x: (x["cap"], x["city"], x["target"])):
        if l["venue"] != "kalshi" or l["cap"] <= since or l["lead"] < 1 or not shape.complete(l):
            continue
        day = l["cap"]
        if day not in fits:
            fits[day] = shape.fit_shape(shape.shape_history(ladders, "kalshi", day))
        params = fits[day]
        if params is None:
            continue
        for q, cell in zip(shape.shaped_probs(l, *params), l["cells"]):
            # Shipped pilot: 3% edge + 1% buffer AFTER fee, not the dashboard's 3%.
            decision = shape.decide_cell(q, cell["bid"], cell["ask"], 0.04, 0.10)
            if decision is None:
                continue
            side, price, _, edge = decision
            candidates.append(dict(
                run_at=day, target_date=l["target"], city=l["city"], ticker=cell["mid"],
                side="yes" if side == "BUY" else "no", price=price, edge=edge,
                outcome=cell["outcome"],
            ))
    return candidates


def audit(ledger_path, captures_path, since, replay=False):
    ledger, captures = gate.load_jsonl(ledger_path), gate.load_jsonl(captures_path)
    settled, opened, unfilled = gate.settle(ledger, captures)
    prospective = [t for t in settled if t["run_at"] > since]
    r = dict(
        freeze=since, latest_capture=max(r["captured_at"][:10] for r in captures),
        inputs={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (ledger_path, captures_path)},
        all=summarize(prospective),
        by_side={s: summarize([t for t in prospective if t["side"] == s]) for s in ("yes", "no")},
        by_mode={m: summarize([t for t in prospective if t["mode"] == m]) for m in ("paper", "live")},
        open=len(opened), unfilled=len(unfilled),
        go_live=gate.report(ledger, captures)["go_live"],
        shadow_rank_then_no=dict(
            freeze=SHADOW_FREEZE,
            rule="Keep NO orders from the existing pilot's selected top five; do not replace YES orders.",
            promotion="Research only. No live admission from this subset; accrue a new prospective sample.",
            forward=summarize([t for t in settled
                               if t["run_at"] > SHADOW_FREEZE and t["side"] == "no"]),
        ),
    )
    if replay:
        candidates = replay_candidates(captures, since)
        r["capture_replay"] = {}
        for side in (None, "no"):
            trades, count_open = select_candidates(candidates, side)
            r["capture_replay"][side or "both"] = dict(summarize(trades), open=count_open)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", default="data/pilot_trades.jsonl")
    ap.add_argument("--captures", default="data/captures.jsonl")
    ap.add_argument("--since", default=FREEZE, type=dt.date.fromisoformat)
    ap.add_argument("--replay", action="store_true")
    a = ap.parse_args()
    print(json.dumps(audit(a.ledger, a.captures, str(a.since), a.replay), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
