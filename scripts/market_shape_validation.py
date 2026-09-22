#!/usr/bin/env python3
"""Validate the three market-shape fit families already recorded on 2026-09-07.

The original script compared all qualifying cells at a 3% fee-adjusted hurdle,
with flat contracts and unrounded per-contract fees. Report that specification
separately from applying the pilot's 4% hurdle, integer $15 stakes, top-five
selection, city caps, and ticker deduplication. Combining an ablation with the
pilot's selection is a research comparison, not a separately frozen strategy.

Every family is refitted independently from its prior-target-date history. The
bias-only family fits (b, 1); scale-only fits (0, k). Neither borrows parameters
from the joint fit. Selection occurs before inspecting settlement outcomes.
No orders, network calls, parameter sweeps, or promotion decisions occur here.
"""

import argparse
import collections
import datetime as dt
import hashlib
import json
import random
import statistics
from pathlib import Path

import go_live_gate as gate
import market_shape_alpha as shape
import pilot_alpha_audit as pilot

FAMILIES = (("joint", True, True), ("bias_only", True, False),
            ("scale_only", False, True))


def summarize(trades):
    """Fee-inclusive statistics, clustered by target day, with city attribution."""
    result = pilot.summarize(trades)
    cities = collections.defaultdict(list)
    days = collections.defaultdict(float)
    for trade in trades:
        cities[trade["city"]].append(trade)
        days[trade["target_date"]] += trade["net"]
    by_city = {}
    for city, rows in sorted(cities.items()):
        net = sum(row["net"] for row in rows)
        cost = sum(row["cost"] for row in rows)
        by_city[city] = dict(settled=len(rows), net=net, roi=net / cost)
    by_side = {}
    for side in ("yes", "no"):
        rows = [row for row in trades if row["side"] == side]
        by_side[side] = dict(settled=len(rows), net=sum(row["net"] for row in rows),
                             cost=sum(row["cost"] for row in rows))
    result.update(
        by_city=by_city,
        positive_cities=sum(row["net"] > 0 for row in by_city.values()),
        negative_cities=sum(row["net"] < 0 for row in by_city.values()),
        positive_days=sum(net > 0 for net in days.values()),
        negative_days=sum(net < 0 for net in days.values()),
        worst_leave_one_city_out_net=(
            result["net"] - max(row["net"] for row in by_city.values())
            if len(by_city) > 1 else None
        ),
        worst_leave_one_day_out_net=(
            result["net"] - max(days.values()) if len(days) > 1 else None
        ),
        by_side=by_side,
    )
    return result


def period_summary(trades, freeze):
    return {
        "through_freeze": summarize([t for t in trades if t["run_at"] <= freeze]),
        "after_freeze": summarize([t for t in trades if t["run_at"] > freeze]),
    }


def candidates_for(ladders, fits, hurdle):
    candidates = []
    for ladder in ladders:
        params = fits[ladder["cap"]]
        if params is None:
            continue
        for probability, cell in zip(shape.shaped_probs(ladder, *params), ladder["cells"]):
            decision = shape.decide_cell(probability, cell["bid"], cell["ask"], hurdle, 0.10)
            if decision is None:
                continue
            side, price, _, edge = decision
            candidates.append(dict(
                run_at=ladder["cap"], target_date=ladder["target"],
                city=ladder["city"], ticker=cell["mid"],
                side="yes" if side == "BUY" else "no",
                price=price, edge=edge, outcome=cell["outcome"],
            ))
    return candidates


def settle_original(candidates):
    """Mirror market_shape_alpha.replay's flat-contract, fractional-fee economics."""
    trades = []
    for candidate in candidates:
        if candidate["outcome"] is None:
            continue
        won = candidate["outcome"] == (1 if candidate["side"] == "yes" else 0)
        price = candidate["price"]
        fee = shape.fee(price)
        trades.append(dict(candidate, won=won, contracts=1, cost=price, fee=fee,
                           net=(1 if won else 0) - price - fee))
    return trades


def day_bootstrap_mean(groups):
    """Bootstrap a sample mean while keeping each target day's observations together."""
    groups = [(sum(groups[day]), len(groups[day])) for day in sorted(groups)]
    if len(groups) < 2:
        return None
    rng = random.Random(42)
    draws = []
    for _ in range(10000):
        sample = rng.choices(groups, k=len(groups))
        draws.append(sum(row[0] for row in sample) / sum(row[1] for row in sample))
    draws.sort()
    return [draws[250], draws[9750]]


def paired_brier_summary(ladders, fits, freeze):
    """Score complete post-freeze ladders, paired against the uncorrected market normal."""
    differences = collections.defaultdict(list)
    shaped_scores, market_scores = [], []
    for ladder in ladders:
        params = fits[ladder["cap"]]
        if ladder["cap"] <= freeze or not ladder["resolved"] or params is None:
            continue
        shaped = shape.brier(ladder, *params)
        market = shape.brier(ladder, 0.0, 1.0)
        shaped_scores.append(shaped)
        market_scores.append(market)
        differences[ladder["target"]].append(shaped - market)
    deltas = [value for rows in differences.values() for value in rows]
    return dict(
        resolved_ladders=len(deltas), target_days=len(differences),
        mean_shaped_brier=statistics.mean(shaped_scores) if shaped_scores else None,
        mean_market_normal_brier=statistics.mean(market_scores) if market_scores else None,
        mean_paired_delta=statistics.mean(deltas) if deltas else None,
        day_bootstrap_95=day_bootstrap_mean(differences),
        interpretation="Sum of squared cell errors per ladder; positive delta means "
            "the correction scores worse than the uncorrected fitted market normal.",
    )


def bias_summary(ladders):
    """Describe uncensored market residuals; bootstrap whole target days."""
    days = collections.defaultdict(list)
    residuals, standardized = [], []
    for ladder in ladders:
        if not ladder["resolved"] or ladder["realized"] is None:
            continue
        residual = ladder["realized"] - ladder["mu"]
        residuals.append(residual)
        standardized.append(residual / ladder["sig"])
        days[ladder["target"]].append(residual)
    return dict(
        uncensored_ladders=len(residuals), target_days=len(days),
        realized_minus_market_c=statistics.mean(residuals) if residuals else None,
        day_bootstrap_95_c=day_bootstrap_mean(days),
        standardized_residual_sd=statistics.pstdev(standardized) if standardized else None,
    )


def audit(captures_path, freeze):
    captures = gate.load_jsonl(captures_path)
    ladders = shape.build_ladders(captures)
    eligible = sorted(
        [ladder for ladder in ladders if ladder["venue"] == "kalshi"
         and ladder["lead"] >= 1 and shape.complete(ladder)],
        key=lambda ladder: (ladder["cap"], ladder["city"], ladder["target"]),
    )
    days = sorted({ladder["cap"] for ladder in eligible})
    histories = {day: shape.shape_history(ladders, "kalshi", day) for day in days}
    result = dict(
        freeze=freeze,
        original_specification_commit="14a1898",
        capture_sha256=hashlib.sha256(Path(captures_path).read_bytes()).hexdigest(),
        capture_rows=len(captures),
        latest_capture=max((row["captured_at"][:10] for row in captures), default=None),
        interpretation={
            "original_03_all_cells": "Previously specified fit families; flat contracts, "
                "unrounded per-contract fees, no ranking or exposure caps.",
            "pilot_04_top5": "Same families under existing pilot selection and rounded "
                "order fees. This combination was not a separately frozen variant.",
            "limitations": "Capture quotes are not fills; historical settlement publication "
                "times are unavailable; replay omits dynamic breakers and account inventory. "
                "Day bootstrap does not correct variant selection or serial dependence. "
                "One-cent stress holds contracts and modeled fees fixed. No rule promoted.",
        },
        market_bias={
            "through_freeze": bias_summary([l for l in eligible if l["cap"] <= freeze]),
            "after_freeze": bias_summary([l for l in eligible if l["cap"] > freeze]),
        },
        families={},
    )
    for name, fit_b, fit_k in FAMILIES:
        fits = {day: shape.fit_shape(histories[day], fit_b=fit_b, fit_k=fit_k) for day in days}
        original = settle_original(candidates_for(
            [ladder for ladder in eligible if ladder["resolved"]], fits, 0.03))
        # The shared pilot selector commits open trades before filtering settlement outcomes.
        selected, opened = pilot.select_candidates(candidates_for(eligible, fits, 0.04))
        result["families"][name] = dict(
            original_03_all_cells=period_summary(original, freeze),
            pilot_04_top5=period_summary(selected, freeze),
            pilot_04_open=opened,
            post_freeze_fit_path={day: fits[day] for day in days if day > freeze},
            post_brier_delta=paired_brier_summary(eligible, fits, freeze),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", default="data/captures.jsonl")
    parser.add_argument("--freeze", type=dt.date.fromisoformat, default=dt.date(2026, 9, 7))
    args = parser.parse_args()
    print(json.dumps(audit(args.captures, str(args.freeze)), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
