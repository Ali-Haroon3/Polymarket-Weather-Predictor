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

Since 2026-09-07 the ledger can also hold LIVE orders and their reconciliation rows (`fill` /
`unfilled`, keyed by order_id — what Kalshi actually executed, at what average NO price). A live
order is scored on its fill row when one exists (an `unfilled` order was no bet and is dropped),
as unverified until then; paper rows are scored as intended, as before. The criteria pool
paper and live — it is one rule producing one sample — and the readout splits them so the
paper-vs-real comparison (fill ratio, price improvement vs the intended limit) is visible.

STRATEGIES (2026-09-07). The same day the phantom-fill defect took the weather model's edge to
zero, the pilot switched to the MARKET-SHAPE strategy (backtesting::market_shape): the Kalshi
ladder's own implied distribution, bias-shifted and sharpened by walk-forward (b, k), traded on
BOTH sides — BUY NO against an over-priced tail, BUY YES on an under-priced favourite. Each
ledger row carries its `strategy` (rows without the field are the legacy `model-shrunk` ones)
and its `side` / `price` (the contract bought and the price paid per contract; legacy rows are
NO-side at `no_price`). A gate is a rule about ONE strategy's sample, so this script scores one
strategy at a time (`--strategy`, default market-shape) and never pools the 23 model-strategy
orders into the new sample. Criterion 3 for market-shape is the module's own structure: (b, k)
fitted only on ladders resolved before the trading day, at least MIN_LADDERS of them, plus the
trailing-30-day replay breaker the pilot runs before placing new orders.

Since the 2026-09-21 audit, --enforce makes NO-GO a nonzero exit. The Rust pilot embeds this
scorer and invokes it on every --live run after reconciliation and before new orders. The
original three research thresholds are unchanged. An additional execution-integrity check
requires every prior live order (across strategies) to have a fill/expiry verdict. Unverified
intentions never count as settled profits; zero fills count in the fill-rate denominator.

Usage: python3 scripts/go_live_gate.py [--ledger data/pilot_trades.jsonl]
                                       [--captures data/captures.jsonl] [--json] [--enforce]
                                       [--strategy market-shape|model-shrunk]
"""
import argparse
import collections
import json
import math
import sys

MIN_SETTLED = 100
MAX_CITY_LOSS_SHARE = 1.0 / 3.0
STRUCTURAL = {
    "model-shrunk": "segment_veto::TrailingBelowFloor in place since 2026-09-06",
    "market-shape": (
        "market_shape: walk-forward (b, k) over >= MIN_LADDERS resolved ladders + "
        "trailing-30-day replay breaker, in place since 2026-09-07"
    ),
}
DEFAULT_STRATEGY = "market-shape"


def strategy_of(row):
    return row.get("strategy") or "model-shrunk"


def side_of(row):
    return row.get("side") or "no"


def paid_price(row):
    """Price paid per contract on the row's own side (legacy rows: the NO price)."""
    return row["price"] if row.get("price") is not None else row.get("no_price")


def kalshi_fee(contracts, price):
    # api::kalshi_trade::fee_cents' formula, in dollars, for one order — including its epsilon
    # guard: an exact-cent product (25 × 60¢ ⇒ 42.000…01) must not ceil up a phantom cent.
    return math.ceil(0.07 * contracts * price * (1.0 - price) * 100.0 - 1e-9) / 100.0


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def reconciliations(ledger):
    """The `fill` / `unfilled` verdict per live order id (the pilot writes at most one)."""
    return {
        r["order_id"]: r
        for r in ledger
        if r.get("decision") in ("fill", "unfilled") and r.get("order_id")
    }


def settle(ledger, captures, strategy=DEFAULT_STRATEGY):
    """Join every 'order' ledger row of `strategy` to its resolved capture. Returns
    (settled, open, unfilled).

    Live orders count only after reconciliation; unresolved fills stay open even if the
    market has settled. Intended fills are not evidence of profit. An
    order that never filled is returned separately (no bet, so neither settled nor open). A
    NO-side order wins when the market resolves NO, a YES-side order when it resolves YES; the
    fee is charged on the price paid either way (Kalshi's formula is symmetric in P).
    """
    outcome = {}
    for r in captures:
        if r.get("source") == "kalshi" and r.get("outcome") is not None:
            key = (r.get("market_id"), r.get("target_date"))
            value = r["outcome"]
            if value not in (0, 1) or (key in outcome and outcome[key] != value):
                raise ValueError(f"invalid or conflicting settlement for {key}")
            outcome[key] = value
    fills = reconciliations(ledger)
    settled, still_open, unfilled = [], [], []
    seen = set()
    for o in ledger:
        if o.get("decision") != "order" or o.get("error") or strategy_of(o) != strategy:
            continue
        live = not o.get("dry_run", True)
        identity = (("live", o.get("order_id")) if live and o.get("order_id") else
                    ("paper" if not live else "unverified", o["ticker"], o["target_date"]))
        if identity in seen:
            raise ValueError(f"duplicate order evidence: {identity}")
        seen.add(identity)
        side = side_of(o)
        n, cost, price = o["contracts"], o["cost"], paid_price(o)
        fill = fills.get(o.get("order_id")) if live else None
        if live and fill is None:
            still_open.append(o)
            continue
        if fill is not None:
            if fill["contracts"] <= 0:
                unfilled.append(o)
                continue
            n, cost, price = fill["contracts"], fill["cost"], paid_price(fill)
        if (side not in ("yes", "no") or n <= 0 or price is None or
                not math.isfinite(price) or not 0 < price < 1 or
                not math.isfinite(cost) or abs(cost - n * price) > 0.011):
            raise ValueError(f"invalid order economics: {identity}")
        key = (o["ticker"], o["target_date"])
        if key not in outcome:
            still_open.append(o)
            continue
        fee = kalshi_fee(n, price)
        won = outcome[key] == (1 if side == "yes" else 0)
        # A winning contract pays $1; the cost is what was paid for it.
        gross = (n - cost) if won else -cost
        settled.append(
            {
                "run_at": o["run_at"][:10],
                "target_date": o["target_date"],
                "ticker": o["ticker"],
                "city": o["city"],
                "side": side,
                "mode": "live" if live else "paper",
                "reconciled": fill is not None,
                "intended_contracts": o["contracts"],
                "intended_price": paid_price(o),
                "contracts": n,
                "price": price,
                "cost": cost,
                "fee": fee,
                "gross": gross,
                "net": gross - fee,
                "won": won,
            }
        )
    return settled, still_open, unfilled


def live_readout(settled, unfilled):
    """Paper-vs-real: how much of what the pilot intended actually traded, and at what price.

    Only reconciled live orders carry real fill data; the ratios below are over those. Price
    improvement is intended limit minus average fill, in cents — a limit that crosses the book
    fills at the resting side's price, so this is ≥ 0 by construction and measures how far
    inside the bid the pilot's evidence-side price actually sat.
    """
    live = [t for t in settled if t["mode"] == "live"]
    real = [t for t in live if t["reconciled"]]
    intended = sum(t["intended_contracts"] for t in real) + sum(
        t["contracts"] for t in unfilled
    )
    filled = sum(t["contracts"] for t in real)
    improvement = [
        (t["intended_price"] - t["price"]) * 100.0 for t in real if t["contracts"] > 0
    ]
    staked = sum(t["cost"] for t in live)
    net = sum(t["net"] for t in live)
    return {
        "live_settled": len(live),
        "live_reconciled": len(real),
        "live_unfilled_orders": len(unfilled),
        "live_net_pnl": net,
        "live_roi_after_fees": net / staked if staked else 0.0,
        "fill_ratio": filled / intended if intended else None,
        "mean_price_improvement_cents": (
            sum(improvement) / len(improvement) if improvement else None
        ),
    }


def evaluate(settled, strategy=DEFAULT_STRATEGY):
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
            "3_structural": {
                "pass": strategy in STRUCTURAL,
                "detail": STRUCTURAL.get(strategy, f"no structural check known for {strategy}"),
            },
        },
    }


def report(ledger, captures, strategy=DEFAULT_STRATEGY):
    settled, still_open, unfilled = settle(ledger, captures, strategy)
    r = evaluate(settled, strategy)
    r["strategy"] = strategy
    r["open"] = len(still_open)
    fills = reconciliations(ledger)
    # Admission may not assume outstanding live orders filled as intended. Keep managing
    # them, but require verified executions before adding new exposure (across strategies).
    unverified = sum(
        o.get("decision") == "order" and not o.get("error") and
        not o.get("dry_run", True) and o.get("order_id") not in fills
        for o in ledger
    )
    r["unverified_live_orders"] = unverified
    r["criteria"]["4_live_fill_evidence"] = {
        "pass": unverified == 0,
        "detail": f"{unverified} live orders await verified fill/expiry information",
    }
    r["go_live"] = all(c["pass"] for c in r["criteria"].values())
    r["live"] = live_readout(settled, unfilled)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="data/pilot_trades.jsonl")
    ap.add_argument("--captures", default="data/captures.jsonl")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument(
        "--enforce", action="store_true",
        help="exit 1 on NO-GO; the live pilot requires this check before new orders",
    )
    ap.add_argument(
        "--strategy",
        default=DEFAULT_STRATEGY,
        choices=sorted(STRUCTURAL),
        help="which strategy's orders to score (one gate, one sample)",
    )
    a = ap.parse_args()

    r = report(
        load_jsonl(a.ledger), load_jsonl(a.captures), a.strategy
    )

    if a.json:
        print(json.dumps(r, indent=2))
        return 1 if a.enforce and not r["go_live"] else 0

    paper = r["settled"] - r["live"]["live_settled"]
    print(
        f"GO-LIVE GATE [{a.strategy}] — {r['settled']} settled orders ({paper} paper, "
        f"{r['live']['live_settled']} live), {r['open']} open"
    )
    print(
        f"  {r['wins']}/{r['settled']} green · ${r['net_pnl']:+.2f} net on ${r['staked']:.2f} "
        f"staked after ${r['fees']:.2f} fees · ROI {r['roi_after_fees']:+.1%}"
    )
    for name, c in r["criteria"].items():
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {name}: {c['detail']}")
    print(f"  => {'GO' if r['go_live'] else 'NO-GO'}")
    lv = r["live"]
    if lv["live_settled"] or lv["live_unfilled_orders"]:
        ratio = "n/a" if lv["fill_ratio"] is None else f"{lv['fill_ratio']:.0%}"
        impr = (
            "n/a"
            if lv["mean_price_improvement_cents"] is None
            else f"{lv['mean_price_improvement_cents']:+.2f}¢"
        )
        print(
            f"  LIVE: {lv['live_settled']} settled ({lv['live_reconciled']} on real fills), "
            f"{lv['live_unfilled_orders']} never filled · ${lv['live_net_pnl']:+.2f} net · "
            f"ROI {lv['live_roi_after_fees']:+.1%} · fill ratio {ratio} · "
            f"mean price improvement {impr} vs the intended limit"
        )
    return 1 if a.enforce and not r["go_live"] else 0


if __name__ == "__main__":
    sys.exit(main())
