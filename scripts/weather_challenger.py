#!/usr/bin/env python3
"""Bounded, offline comparison of five fixed weather-market probability families.

Uses existing Kalshi capture books and forecast_high (Celsius). No API calls,
orders, canonical ledger writes, parameter search, or automatic promotion. The
new weather family regresses prior uncensored event residuals on forecast minus
market mean with strong fixed ridge shrinkage; its weather weight is capped at
25%. Joint, bias-only, and scale-only retain the repository's original grids,
but each is independently refitted under this study's stricter history rules.

Study selections use the earliest qualifying complete lead >= 1 snapshot per
city/target, one position/event, at most five orders/day, integer contracts, and
$15 INCLUDING rounded order fees. Every selected contract needs >= 4 cents net
expected edge. Outcomes are consulted only after selection. Historical quote
snapshots are not execution proof and these previously inspected dates are not
an untouched holdout.

Known outcome_observed_at dates must precede the entry day. Legacy rows have no
publication clock: target + 2 days is an explicit approximation, not proof that
settlement was available then. Uncensored winning-bucket midpoints approximate
realized temperature. Shadow mode masks current/future outcomes and emits only
the latest capture day's eligible selections, using the same earliest-event rule.

python3 scripts/weather_challenger.py --output /tmp/weather-study.json
python3 scripts/weather_challenger.py --mode shadow --output /tmp/weather-shadow.json
"""

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import random

import go_live_gate as gate
import market_shape_alpha as shape

FAMILIES = ("market_normal", "joint_shape", "bias_only", "scale_only", "weather_blend")
POLICY_VERSION = "weather-challenger-v1"
MIN_HISTORY = 60
HISTORY_CAP = 300
STAKE = 15.0
MIN_PRICE = 0.10
MIN_EDGE = 0.04
MAX_ORDERS_PER_DAY = 5
GRID = tuple((b, k) for b in shape.B_GRID for k in shape.K_GRID)


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def usable_price(value):
    return value if number(value) and 0 < value < 1 else None


def exact_partition(cells):
    """Require exhaustive, non-overlapping coverage, including both infinite tails."""
    if len(cells) < 4 or len({c["mid"] for c in cells}) != len(cells):
        return False
    ordered = sorted(cells, key=lambda c: c["lo"])
    if ordered[0]["lo"] != -math.inf or ordered[-1]["hi"] != math.inf:
        return False
    if any(c["lo"] >= c["hi"] for c in ordered):
        return False
    return all(math.isclose(a["hi"], b["lo"], abs_tol=1e-8, rel_tol=0)
               for a, b in zip(ordered, ordered[1:]))


def availability(row):
    """First usable decision date; intraday entry time is unavailable in old captures."""
    stamp = row.get("outcome_observed_at")
    if stamp is not None:
        try:
            observed = dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            if observed.tzinfo is None:
                raise ValueError("outcome_observed_at must have a timezone")
            observed_day = observed.astimezone(dt.timezone.utc).date()
            return str(observed_day + dt.timedelta(days=1)), "observed"
        except (TypeError, ValueError):
            # Malformed explicit evidence must not become an optimistic legacy fallback.
            return "9999-12-31", "invalid_observed_timestamp"
    return str(dt.date.fromisoformat(row["target_date"]) + dt.timedelta(days=2)), "legacy_target_plus_2"


def build_events(rows):
    """Choose earliest valid snapshot without looking at settlement or forecast values."""
    groups = collections.defaultdict(list)
    rejected = collections.Counter()
    for row in rows:
        if row.get("source") != "kalshi" or row.get("market_type") not in {
            "temp_bucket", "temp_at_most", "temp_at_least"
        }:
            continue
        target, cap = row["target_date"], row["captured_at"]
        if (dt.date.fromisoformat(target) - dt.date.fromisoformat(cap[:10])).days < 1:
            continue
        groups[(cap, row["city"], target)].append(row)
    events, seen = [], set()
    for (stamp, city, target), group in sorted(groups.items()):
        event_key = (city, target)
        if event_key in seen:
            rejected["later_snapshot_same_event"] += 1
            continue
        cells, invalid = [], False
        for row in group:
            b, a = usable_price(row.get("best_bid")), usable_price(row.get("best_ask"))
            raw_quotes = [row.get("best_bid"), row.get("best_ask")]
            if any(v is not None and (not number(v) or not 0 <= v <= 1) for v in raw_quotes):
                invalid = True
                break
            if (b is None and a is None) or (b is not None and a is not None and b > a):
                invalid = True
                break
            if not row.get("market_id") or not number(row.get("threshold")):
                invalid = True
                break
            if row.get("threshold_upper") is not None and not number(row["threshold_upper"]):
                invalid = True
                break
            if (row.get("unit") or "F").upper() not in {"F", "C"}:
                invalid = True
                break
            lo, hi = shape.cell_bounds_c(row)
            available_on, source = availability(row)
            cells.append(dict(lo=lo, hi=hi, mid=row["market_id"], bid=b, ask=a,
                              px=(a + b) / 2 if a is not None and b is not None else (a if a is not None else b),
                              outcome=row.get("outcome"), available_on=available_on,
                              availability_source=source, mt=row["market_type"],
                              forecast_high=row.get("forecast_high")))
        if invalid:
            rejected["invalid_book_or_contract"] += 1
            continue
        if not exact_partition(cells):
            rejected["incomplete_or_overlapping_partition"] += 1
            continue
        psum = sum(c["px"] for c in cells)
        if not 0.8 <= psum <= 1.2:
            rejected["incoherent_mid_sum"] += 1
            continue
        cells.sort(key=lambda c: c["lo"])
        seen.add(event_key)
        mu0 = sum(c["px"] * (
            (c["lo"] if math.isfinite(c["lo"]) else c["hi"] - 1)
            + (c["hi"] if math.isfinite(c["hi"]) else c["lo"] + 1)
        ) / 2 for c in cells) / psum
        mu, sig, err = shape.fit_market_normal([(c["lo"], c["hi"], c["px"]) for c in cells], mu0)
        outcomes = [c["outcome"] for c in cells]
        resolved = all(number(v) and v in (0, 1) for v in outcomes) and sum(outcomes) == 1
        winner = next((c for c in cells if resolved and c["outcome"] == 1), None)
        realized = ((winner["lo"] + winner["hi"]) / 2
                    if winner is not None and winner["mt"] == "temp_bucket" else None)
        forecasts = [c["forecast_high"] for c in cells if number(c["forecast_high"])]
        # A snapshot contains one weather forecast. Conflicting per-cell values fail to market.
        forecast = forecasts[0] if forecasts and max(forecasts) - min(forecasts) < 1e-6 else None
        events.append(dict(city=city, target=target, cap=stamp[:10], captured_at=stamp,
                           cells=cells, mu=mu, sig=sig, err=err, psum=psum,
                           forecast_high=forecast, realized=realized, resolved=resolved,
                           available_on=max(c["available_on"] for c in cells),
                           availability_sources=sorted({c["availability_source"] for c in cells})))
    return events, dict(rejected)


def eligible_history(events, decision_day):
    """No repeated city-day events; only outcomes available before the decision."""
    eligible = [e for e in events if e["resolved"] and e["target"] < decision_day
                and e["cap"] < decision_day and e["available_on"] <= decision_day]
    eligible.sort(key=lambda e: (e["target"], e["city"], e["cap"]))
    return eligible[-HISTORY_CAP:]


def score_cache(events):
    """Finite fixed-grid scores cached once/event; fitting indexes only eligible history."""
    return {(e["city"], e["target"]): tuple(shape.brier(e, b, k) for b, k in GRID)
            for e in events if e["resolved"]}


def fit_weather(history, minimum=MIN_HISTORY):
    usable = [e for e in history if e["realized"] is not None and number(e["forecast_high"])]
    base = dict(bias_c=0.0, weather_weight=0.0, scale=1.0,
                weather_training_events=len(usable), weather_fit_available=False)
    if len(usable) < minimum:
        return base
    xs = [e["forecast_high"] - e["mu"] for e in usable]
    ys = [e["realized"] - e["mu"] for e in usable]
    # Fixed ridge prior: 60 zero-residual pseudo-events; slope penalty 300 C^2.
    nn, xx, sx, sy = len(xs) + 60.0, sum(x * x for x in xs) + 300.0, sum(xs), sum(ys)
    xy = sum(x * y for x, y in zip(xs, ys))
    beta = max(0.0, min(0.25, (nn * xy - sx * sy) / (nn * xx - sx * sx)))
    bias = max(-0.6, min(0.6, (sy - beta * sx) / nn))
    # Calibrate on every categorical outcome, including tail winners. Estimating
    # dispersion only from interior winning-bucket residuals would truncate tails.
    calibration = [e for e in history if number(e["forecast_high"])]
    scale = min(shape.K_GRID, key=lambda k: sum(
        shape.brier(e, bias + beta * (e["forecast_high"] - e["mu"]), k)
        for e in calibration))
    return dict(bias_c=bias, weather_weight=beta, scale=scale,
                weather_training_events=len(usable), weather_fit_available=True)


def fit_families(history, cache, minimum=MIN_HISTORY):
    if len(history) < minimum:
        return None
    scores = [sum(cache[(e["city"], e["target"])][i] for e in history) for i in range(len(GRID))]

    def best(fit_bias, fit_scale):
        indices = [i for i, (b, k) in enumerate(GRID) if (fit_bias or b == 0) and (fit_scale or k == 1)]
        b, k = GRID[min(indices, key=lambda i: scores[i])]
        return dict(bias_c=b, scale=k)

    return dict(market_normal=dict(bias_c=0.0, scale=1.0), joint_shape=best(True, True),
                bias_only=best(True, False), scale_only=best(False, True),
                weather_blend=fit_weather(history, minimum))


def probabilities(event, family, fit):
    bias, scale = fit["bias_c"], fit["scale"]
    if family == "weather_blend":
        if not number(event["forecast_high"]) or not fit["weather_fit_available"]:
            bias, scale = 0.0, 1.0
        else:
            bias += fit["weather_weight"] * (event["forecast_high"] - event["mu"])
    return shape.shaped_probs(event, bias, scale)


def size_order(price, budget=STAKE):
    """Largest integer order whose price plus rounded taker fee fits the budget."""
    if usable_price(price) is None:
        return None
    contracts = math.floor(budget / price + 1e-10)
    while contracts > 0:
        fee = gate.kalshi_fee(contracts, price)
        principal = contracts * price
        if principal + fee <= budget + 1e-9:
            return dict(contracts=contracts, principal=principal, fee=fee, cost=principal + fee)
        contracts -= 1
    return None


def candidates_for(event, family, fit):
    """Candidate generation never reads outcomes; fee hurdle uses actual integer size."""
    candidates = []
    for p_yes, cell in zip(probabilities(event, family, fit), event["cells"]):
        for side, price, probability in (("yes", cell["ask"], p_yes),
                                         ("no", 1 - cell["bid"] if cell["bid"] is not None else None, 1 - p_yes)):
            if usable_price(price) is None or price < MIN_PRICE:
                continue
            size = size_order(price)
            if size is None:
                continue
            edge = probability - price - size["fee"] / size["contracts"]
            if edge + 1e-12 < MIN_EDGE:
                continue
            candidates.append(dict(run_at=event["cap"], target_date=event["target"], city=event["city"],
                                   ticker=cell["mid"], side=side, price=price,
                                   probability=probability, edge=edge, **size))
    return candidates


def select_candidates(candidates):
    """Commit before settlement; unresolved orders consume the same capacity."""
    selected, seen_events, seen_tickers = [], set(), set()
    counts = collections.Counter()
    for candidate in sorted(candidates, key=lambda c: (c["run_at"], -c["edge"], c["ticker"], c["side"])):
        event = (candidate["city"], candidate["target_date"])
        if event in seen_events or candidate["ticker"] in seen_tickers or counts[candidate["run_at"]] >= MAX_ORDERS_PER_DAY:
            continue
        selected.append(dict(candidate))
        seen_events.add(event)
        seen_tickers.add(candidate["ticker"])
        counts[candidate["run_at"]] += 1
    return selected


def settle(selected, events):
    outcomes = {(e["city"], e["target"], c["mid"]): c["outcome"]
                for e in events if e["resolved"] for c in e["cells"]}
    trades, opened = [], 0
    for candidate in selected:
        outcome = outcomes.get((candidate["city"], candidate["target_date"], candidate["ticker"]))
        if outcome is None:
            opened += 1
            continue
        won = outcome == (1 if candidate["side"] == "yes" else 0)
        stress_price = candidate["price"] + 0.01
        stressed = size_order(stress_price)
        trades.append(dict(candidate, won=won, outcome=outcome,
                           net=(candidate["contracts"] if won else 0) - candidate["cost"],
                           stress_1c=dict(price=stress_price, **stressed,
                                          net=(stressed["contracts"] if won else 0) - stressed["cost"])
                           if stressed is not None else None))
    return trades, opened


def bootstrap(groups, draws=4000):
    if len(groups) < 2:
        return None
    rng = random.Random(20260907)
    ratios = []
    for _ in range(draws):
        sample = rng.choices(groups, k=len(groups))
        ratios.append(sum(g[0] for g in sample) / sum(g[1] for g in sample))
    ratios.sort()
    return [ratios[int(draws * 0.025)], ratios[int(draws * 0.975)]]


def summarize(trades):
    cost, net = sum(t["cost"] for t in trades), sum(t["net"] for t in trades)
    day_groups, by_city = collections.defaultdict(lambda: [0.0, 0.0]), {}
    for trade in trades:
        day_groups[trade["target_date"]][0] += trade["net"]
        day_groups[trade["target_date"]][1] += trade["cost"]
    for city in sorted({t["city"] for t in trades}):
        rows = [t for t in trades if t["city"] == city]
        city_cost, city_net = sum(t["cost"] for t in rows), sum(t["net"] for t in rows)
        by_city[city] = dict(orders=len(rows), net=city_net, cost=city_cost,
                             roi=city_net / city_cost, capital_share=city_cost / cost)
    stress = [t["stress_1c"] for t in trades if t["stress_1c"] is not None]
    stress_cost, stress_net = sum(t["cost"] for t in stress), sum(t["net"] for t in stress)
    cumulative = high_water = max_drawdown = 0.0
    for target in sorted(day_groups):
        cumulative += day_groups[target][0]
        high_water = max(high_water, cumulative)
        max_drawdown = max(max_drawdown, high_water - cumulative)
    return dict(orders=len(trades), target_days=len(day_groups), wins=sum(t["won"] for t in trades),
                cost=cost, fees=sum(t["fee"] for t in trades), net=net, roi=net / cost if cost else None,
                target_day_bootstrap_95_roi=bootstrap(list(day_groups.values())),
                max_target_day_drawdown=max_drawdown, by_city=by_city,
                largest_city_capital_share=max((c["capital_share"] for c in by_city.values()), default=None),
                net_without_best_city=net - max(c["net"] for c in by_city.values()) if len(by_city) > 1 else None,
                stress_1c=dict(orders=len(stress), unpriceable=len(trades) - len(stress),
                               cost=stress_cost, net=stress_net, fees=sum(t["fee"] for t in stress),
                               roi=stress_net / stress_cost if stress_cost else None))


def run(rows, mode="study"):
    latest = max((r["captured_at"][:10] for r in rows), default=None)
    if mode == "shadow" and latest:
        # Mask before event construction, not merely when formatting output.
        rows = [dict(r, outcome=None) if r["target_date"] >= latest else r for r in rows]
    events, exclusions = build_events(rows)
    cache = score_cache(events)
    days = sorted({e["cap"] for e in events}) if mode == "study" else ([latest] if latest else [])
    fits, history_counts = {}, {}
    for day in days:
        history = eligible_history(events, day)
        history_counts[day] = len(history)
        fits[day] = fit_families(history, cache)
    result = dict(
        mode=mode, latest_capture=latest, capture_rows=len(rows), qualifying_events=len(events),
        policy_version=POLICY_VERSION,
        forecast_events=sum(number(e["forecast_high"]) for e in events), exclusions=exclusions,
        outcome_availability_sources=dict(collections.Counter(s for e in events for s in e["availability_sources"])),
        specification=dict(families=list(FAMILIES), history_min=MIN_HISTORY, history_cap=HISTORY_CAP,
                           fee_inclusive_stake=STAKE, min_price=MIN_PRICE, min_net_edge=MIN_EDGE,
                           max_orders_per_day=MAX_ORDERS_PER_DAY, max_positions_per_event=1,
                           weather_ridge_intercept=60, weather_ridge_slope=300, max_weather_weight=0.25,
                           weather_scale_fit="All-event categorical Brier, including tail winners, on the original fixed K grid.",
                           selection="Earliest qualifying complete lead>=1 book per city/target; rank by rounded-fee net edge."),
        limitations=["Capture quotes are not execution proof; book depth and order fills are unverified.",
                     "Legacy target+2 outcome availability is an approximation, not observed publication evidence.",
                     "These historical dates were inspected before this study; no untouched holdout is claimed.",
                     "Winning interior bucket midpoints approximate realized temperature; tail winners are excluded from weather mean regression (possible truncation bias), but included in scale calibration.",
                     "Target-day bootstrap does not correct research selection or serial dependence across days.",
                     "Adverse 1c stress keeps ticker/side selection, then resizes and recomputes rounded fees inside $15.",
                     "All fitted families share 60 prior eligible events; no account inventory, dynamic breaker, or live promotion.",
                     "Challenger families have no live-admission authority and cannot inherit another strategy's go-live gate."],
        history_counts=history_counts, families={})
    for family in FAMILIES:
        candidates = []
        for event in events:
            day_fits = fits.get(event["cap"])
            if day_fits is not None:
                candidates.extend(candidates_for(event, family, day_fits[family]))
        selected = select_candidates(candidates)
        family_result = dict(fit_path={day: fit[family] for day, fit in fits.items() if fit is not None},
                             selected=selected)
        if mode == "study":
            trades, opened = settle(selected, events)
            family_result.update(open_orders=opened, trades=trades, periods={
                "all": summarize(trades),
                "through_2026_09_07": summarize([t for t in trades if t["run_at"] <= "2026-09-07"]),
                "after_2026_09_07": summarize([t for t in trades if t["run_at"] > "2026-09-07"]),
                "after_2026_09_21": summarize([t for t in trades if t["run_at"] > "2026-09-21"]),
            })
        result["families"][family] = family_result
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, default=Path("data/captures.jsonl"))
    parser.add_argument("--mode", choices=("study", "shadow"), default="study")
    parser.add_argument("--shadow", action="store_true", help="Alias for --mode shadow.")
    parser.add_argument("--require-capture-date", type=dt.date.fromisoformat,
                        help="Fail before fitting/output unless the newest capture has this UTC date.")
    parser.add_argument("--output", type=Path, help="Optional separate .json artifact; otherwise print JSON.")
    args = parser.parse_args()
    if args.output and (args.output.suffix != ".json" or args.output.resolve() == args.captures.resolve()):
        parser.error("--output must be a separate .json artifact, never a capture or ledger JSONL")
    data = args.captures.read_bytes()
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    latest = max((r["captured_at"][:10] for r in rows), default=None)
    if args.require_capture_date and latest != str(args.require_capture_date):
        parser.error(f"latest capture {latest!r} does not match required date {args.require_capture_date}")
    mode = "shadow" if args.shadow else args.mode
    result = run(rows, mode)
    result["required_capture_date"] = str(args.require_capture_date) if args.require_capture_date else None
    result["capture_sha256"] = hashlib.sha256(data).hexdigest()
    code_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in (Path(__file__), Path(shape.__file__), Path(gate.__file__))}
    result["policy_code_sha256"] = code_hashes
    result["policy_sha256"] = hashlib.sha256(json.dumps(
        dict(version=POLICY_VERSION, specification=result["specification"], code_hashes=code_hashes),
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
        print(json.dumps(dict(output=str(args.output.resolve()), mode=mode,
                              latest_capture=result["latest_capture"], events=result["qualifying_events"])))
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
