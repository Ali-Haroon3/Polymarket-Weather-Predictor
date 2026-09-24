#!/usr/bin/env python3
"""Audit full Kalshi temperature baskets without using outcomes (stdlib only).

A complete, mutually exclusive ladder pays $1 per full YES basket and N-1 dollars
per full NO basket. Price every leg at its captured ask or 1-bid; use equal integer
quantities and go_live_gate.kalshi_fee rounded per leg. The budget includes fees.
Search every affordable quantity because rounded fees make net profit nonmonotonic.

These are captured-quote research candidates, not executable guaranteed profits:
leg snapshots were not simultaneous, available depth is unknown, and partial fills
can leave directional exposure. The fee model assumes the general taker schedule
with multiplier 1 and one fill per leg. No trades are placed.

Usage: python3 scripts/ladder_arbitrage_audit.py [--budgets 15 150] [--json]
"""
import argparse
import collections
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

from go_live_gate import kalshi_fee, load_jsonl
from market_shape_alpha import cell_bounds_c

EPSILON = 1e-9
LIMITATIONS = (
    "Captured leg quotes were not simultaneous; depth and fill availability are unknown. "
    "Candidates are not executable guaranteed profits. Partial fills can leave directional "
    "exposure. Fees assume multiplier 1 and one fill per leg."
)


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def full_partition(rows):
    """Return ordered rows only for unique, valid intervals covering the whole line."""
    if len(rows) < 2:
        return None
    intervals, ids = [], set()
    for row in rows:
        market_id = row.get("market_id")
        if not isinstance(market_id, str) or not market_id or market_id in ids:
            return None
        ids.add(market_id)
        if (row.get("market_type") not in ("temp_bucket", "temp_at_least", "temp_at_most")
                or not finite_number(row.get("threshold"))
                or row.get("unit") not in (None, "F", "f", "C", "c")):
            return None
        upper = row.get("threshold_upper")
        if row["market_type"] == "temp_bucket" and not finite_number(upper):
            return None
        lo, hi = cell_bounds_c(row)
        if math.isnan(lo) or math.isnan(hi) or lo >= hi:
            return None
        intervals.append((lo, hi, row))
    intervals.sort(key=lambda value: (value[0], value[1]))
    if intervals[0][0] != -math.inf or intervals[-1][1] != math.inf:
        return None
    if any(not math.isfinite(left[1]) or not math.isfinite(right[0])
           or abs(left[1] - right[0]) > EPSILON
           for left, right in zip(intervals, intervals[1:])):
        return None
    return [value[2] for value in intervals]


def executable_prices(rows, side):
    """No last-price fallback: every leg needs the actual side being purchased."""
    if side not in ("yes", "no"):
        raise ValueError("side must be yes or no")
    prices = []
    for row in rows:
        bid, ask = row.get("best_bid"), row.get("best_ask")
        if (finite_number(bid) and finite_number(ask) and bid > ask + EPSILON):
            return None
        quote = ask if side == "yes" else bid
        if not finite_number(quote) or not 0 < quote < 1:
            return None
        prices.append(quote if side == "yes" else 1.0 - quote)
    return prices


def basket_cost(prices, quantity):
    principal = quantity * sum(prices)
    fees = sum(kalshi_fee(quantity, price) for price in prices)
    return principal, fees, principal + fees


def best_basket(prices, payout_per_basket, budget):
    """Best net among all equal integer quantities affordable INCLUDING per-leg fees."""
    if not finite_number(budget) or budget <= 0:
        raise ValueError("budget must be positive and finite")
    if not prices or any(not finite_number(p) or not 0 < p < 1 for p in prices):
        raise ValueError("prices must all be finite and inside (0, 1)")
    if not finite_number(payout_per_basket) or payout_per_basket <= 0:
        raise ValueError("basket payout must be positive and finite")
    best, max_affordable = None, 0
    # Fees are nonnegative, so principal alone supplies a safe upper bound.
    for quantity in range(1, math.floor((budget + EPSILON) / sum(prices)) + 1):
        principal, fees, total = basket_cost(prices, quantity)
        if total > budget + EPSILON:
            continue
        max_affordable = quantity
        payout = quantity * payout_per_basket
        candidate = dict(quantity=quantity, principal=principal, fees=fees,
                         total_cost=total, payout=payout, net=payout - total)
        if best is None or candidate["net"] > best["net"] + EPSILON:
            best = candidate
    if best is None:
        return None
    best["max_affordable_quantity"] = max_affordable
    stress = []
    for index, price in enumerate(prices):
        adverse = price + 0.01
        if adverse > 1.0 + EPSILON:
            # Do not silently turn the stated 1-cent stress into a smaller movement.
            best["single_leg_adverse_cent"] = None
            return best
        stressed_prices = list(prices)
        stressed_prices[index] = min(1.0, adverse)
        _, fees, cost = basket_cost(stressed_prices, best["quantity"])
        stress.append(dict(leg_index=index, price=stressed_prices[index], fees=fees,
                           total_cost=cost, net=best["payout"] - cost,
                           within_budget=cost <= budget + EPSILON))
    best["single_leg_adverse_cent"] = min(stress, key=lambda value: value["net"])
    return best


def audit(rows, budgets=(15.0,)):
    groups = collections.defaultdict(list)
    for row in rows:
        if row.get("source") != "kalshi":
            continue
        captured = dt.date.fromisoformat(row["captured_at"][:10])
        target = dt.date.fromisoformat(row["target_date"])
        if (target - captured).days < 1:
            continue
        groups[(row["city"], target.isoformat(), captured.isoformat())].append(row)
    for budget in budgets:
        if not finite_number(budget) or budget <= 0:
            raise ValueError("budgets must be positive and finite")
    summary = dict(eligible_groups=len(groups), full_partitions=0, rejected_partitions=0,
                   quoted_baskets=dict(yes=0, no=0), gross_positive_baskets=0,
                   unrounded_fee_positive_baskets=0, before_rounding_candidates=[],
                   budgets=[dict(budget=b, affordable_baskets=0, positive_candidates=[])
                            for b in budgets], limitations=LIMITATIONS)
    for (city, target, captured), group in sorted(groups.items()):
        partition = full_partition(group)
        if partition is None:
            summary["rejected_partitions"] += 1
            continue
        summary["full_partitions"] += 1
        for side in ("yes", "no"):
            prices = executable_prices(partition, side)
            if prices is None:
                continue
            summary["quoted_baskets"][side] += 1
            payout = 1.0 if side == "yes" else len(partition) - 1.0
            gross = payout - sum(prices)
            unrounded_net = gross - sum(0.07 * p * (1.0 - p) for p in prices)
            summary["gross_positive_baskets"] += gross > EPSILON
            summary["unrounded_fee_positive_baskets"] += unrounded_net > EPSILON
            identity = dict(city=city, target_date=target, captured_at=captured, side=side,
                            markets=[row["market_id"] for row in partition], prices=prices,
                            payout_per_basket=payout, gross_per_basket=gross,
                            unrounded_net_per_basket=unrounded_net)
            if unrounded_net > EPSILON:
                summary["before_rounding_candidates"].append(identity)
            for budget_summary in summary["budgets"]:
                best = best_basket(prices, payout, budget_summary["budget"])
                if best is None:
                    continue
                budget_summary["affordable_baskets"] += 1
                if best["net"] > EPSILON:
                    budget_summary["positive_candidates"].append(dict(identity, **best))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, default=Path("data/captures.jsonl"))
    parser.add_argument("--budgets", type=float, nargs="+", default=[15.0])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = audit(load_jsonl(args.captures), args.budgets)
    result["captures_sha256"] = hashlib.sha256(args.captures.read_bytes()).hexdigest()
    if args.json:
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    print("Full-ladder basket audit; no outcomes used; no orders placed")
    print(f"Capture SHA-256: {result['captures_sha256']}")
    print(f"Lead>=1 Kalshi: {result['eligible_groups']} groups; "
          f"{result['full_partitions']} full partitions; "
          f"{result['rejected_partitions']} rejected")
    print(f"All required quotes present: {result['quoted_baskets']}; "
          f"gross-positive baskets: {result['gross_positive_baskets']}; "
          f"positive before fee rounding: {result['unrounded_fee_positive_baskets']}")
    for b in result["budgets"]:
        positive = b["positive_candidates"]
        print(f"Budget ${b['budget']:.2f} including fees: {len(positive)} positive candidates "
              f"among {b['affordable_baskets']} affordable baskets")
        for candidate in positive:
            stress = candidate["single_leg_adverse_cent"]
            stressed = (f"${stress['net']:+.2f}; within budget={stress['within_budget']}"
                        if stress is not None else "unavailable: price would exceed $1")
            print(f"  {candidate['city']} captured {candidate['captured_at']} / "
                  f"target {candidate['target_date']} BUY {candidate['side'].upper()}: "
                  f"{candidate['quantity']} per leg, cost ${candidate['total_cost']:.2f}, "
                  f"fees ${candidate['fees']:.2f}, net ${candidate['net']:+.2f}; "
                  f"worst single-leg +1c stress {stressed}")
    print(LIMITATIONS)


if __name__ == "__main__":
    main()
