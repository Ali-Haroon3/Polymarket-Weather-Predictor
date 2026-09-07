#!/usr/bin/env python3
"""The go-live gate for the Kalshi pilot, evaluated from the paper ledger (stdlib only).

Written down 2026-09-06 because until then the repo referred to "the go-live gate's paper sample"
in several places without ever stating what the gate was — evidence was accruing toward a decision
rule that did not exist. This script IS the rule. Its thresholds were chosen before the sample
that will be scored against them, and they are not to be moved to fit that sample.

Three criteria, all required:

  1. SAMPLE + SIGN. At least MIN_SETTLED settled paper orders, and their fee-inclusive ROI is
     positive. Fees are Kalshi's ceil(0.07 * C * P * (1 - P)) per order, charged on the NO price
     actually traded — the cost paper trading never modeled and the number the pilot exists to
     measure. ~100 trades is roughly three months at the pilot's ~1 order/day; that is the price of
     distinguishing a thin edge from noise, and widening the pilot to hurry it is how Vegas got in.

  2. NO SINGLE-CITY BLEED. No one city accounts for more than MAX_CITY_LOSS_SHARE of the total
     realized losses. This is the regime-shift check: both of the pilot's loss mechanisms to date
     (Vegas 08-21..24, LA 09-01..03) were one city's pricing constants going wrong while everything
     else was fine. A positive aggregate that hides one bleeding city is not a strategy that works;
     it is a strategy that has not yet been found out.

  3. STRUCTURAL. The city gate carries the trailing-window check (segment_veto::TrailingBelowFloor,
     landed 2026-09-06), so a mature city going wrong is caught in days rather than weeks. A code
     property, not a data one: this script reports it satisfied from that date and does not
     re-verify it — read backtesting::shrinkage if in doubt.

Passing all three makes the POCKET-CHANGE pilot defensible ($15 stakes, $50/week breaker): its
stated purpose is to measure fills and slippage that paper cannot. It says nothing about size.

Usage: python3 scripts/go_live_gate.py [--ledger data/pilot_trades.jsonl]
                                       [--captures data/captures.jsonl] [--json]
"""
import argparse
import collections
import json
import math

MIN_SETTLED = 100
MAX_CITY_LOSS_SHARE = 1.0 / 3.0
STRUCTURAL_SINCE = "2026-09-06"


def kalshi_fee(contracts, price):
    # api::kalshi_trade::fee_frac's formula, in dollars, for one order.
    return math.ceil(0.07 * contracts * price * (1.0 - price) * 100.0) / 100.0


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def settle(ledger, captures):
    """Join every 'order' ledger row to its resolved capture. Returns (settled, open)."""
    outcome = {}
    for r in captures:
        if r.get("source") == "kalshi" and r.get("outcome") is not None:
            outcome[(r.get("market_id"), r.get("target_date"))] = r["outcome"]
    settled, still_open = [], []
    for o in ledger:
        if o.get("decision") != "order" or o.get("error"):
            continue
        key = (o["ticker"], o["target_date"])
        if key not in outcome:
            still_open.append(o)
            continue
        n, cost = o["contracts"], o["cost"]
        fee = kalshi_fee(n, o["no_price"])
        # The pilot buys NO: a YES outcome of 0 pays $1/contract.
        gross = (n - cost) if outcome[key] == 0 else -cost
        settled.append(
            {
                "run_at": o["run_at"][:10],
                "ticker": o["ticker"],
                "city": o["city"],
                "cost": cost,
                "fee": fee,
                "gross": gross,
                "net": gross - fee,
                "won": outcome[key] == 0,
            }
        )
    return settled, still_open


def evaluate(settled):
    n = len(settled)
    staked = sum(t["cost"] for t in settled)
    net = sum(t["net"] for t in settled)
    fees = sum(t["fee"] for t in settled)
    roi = net / staked if staked else 0.0
    wins = sum(1 for t in settled if t["won"])

    losses_by_city = collections.Counter()
    for t in settled:
        if t["net"] < 0:
            losses_by_city[t["city"]] += -t["net"]
    total_loss = sum(losses_by_city.values())
    worst_city, worst_loss = (losses_by_city.most_common(1) or [(None, 0.0)])[0]
    worst_share = worst_loss / total_loss if total_loss else 0.0

    return {
        "settled": n,
        "wins": wins,
        "staked": staked,
        "fees": fees,
        "net_pnl": net,
        "roi_after_fees": roi,
        "worst_city": worst_city,
        "worst_city_loss_share": worst_share,
        "criteria": {
            "1_sample_and_sign": {
                "pass": n >= MIN_SETTLED and roi > 0.0,
                "detail": f"{n}/{MIN_SETTLED} settled, ROI after fees {roi:+.1%}",
            },
            "2_no_single_city_bleed": {
                "pass": worst_share <= MAX_CITY_LOSS_SHARE,
                "detail": (
                    f"{worst_city} carries {worst_share:.0%} of losses "
                    f"(limit {MAX_CITY_LOSS_SHARE:.0%})"
                    if worst_city
                    else "no losses yet"
                ),
            },
            "3_trailing_gate_structural": {
                "pass": True,
                "detail": f"segment_veto::TrailingBelowFloor in place since {STRUCTURAL_SINCE}",
            },
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="data/pilot_trades.jsonl")
    ap.add_argument("--captures", default="data/captures.jsonl")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    a = ap.parse_args()

    settled, still_open = settle(load_jsonl(a.ledger), load_jsonl(a.captures))
    r = evaluate(settled)
    r["open"] = len(still_open)
    r["go_live"] = all(c["pass"] for c in r["criteria"].values())

    if a.json:
        print(json.dumps(r, indent=2))
        return

    print(f"GO-LIVE GATE — {r['settled']} settled paper orders, {r['open']} open")
    print(
        f"  {r['wins']}/{r['settled']} green · ${r['net_pnl']:+.2f} net on ${r['staked']:.2f} "
        f"staked after ${r['fees']:.2f} fees · ROI {r['roi_after_fees']:+.1%}"
    )
    for name, c in r["criteria"].items():
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {name}: {c['detail']}")
    print(f"  => {'GO' if r['go_live'] else 'NO-GO'}")


if __name__ == "__main__":
    main()
