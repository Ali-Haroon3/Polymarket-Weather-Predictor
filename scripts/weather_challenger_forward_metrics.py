#!/usr/bin/env python3
"""Pure accounting for the registered challenger; never select, refit, fetch or trade.

The caller must validate complete prospective artifact/input coverage separately. A favorable
verdict here is conditional on that coverage and has no live-admission authority. Inputs and
all dated outcome versions are retained; post-cutoff observations cannot change primary results.
"""
import collections
import datetime as dt
from decimal import Decimal
import math
import random

import weather_challenger as challenger

UTC = dt.timezone.utc
CUTOFF = dt.datetime(2026, 12, 22, tzinfo=UTC)
START = dt.date(2026, 10, 8)
END = dt.date(2026, 12, 6)
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 20261007
RESTART_PROBABILITY = 1 / 7
POLICY_SHA256 = "d7509940235a42ab998357cf067b868bd3689e27e7686a069299ac24d2a95df1"


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _preserve(value):
    """Keep invalid evidence reviewable without emitting non-standard JSON numbers."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {"invalid_numeric_value": repr(value)}
    if isinstance(value, dict):
        return {str(key): _preserve(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_preserve(item) for item in value]
    return {"invalid_value_type": type(value).__name__, "representation": repr(value)}


def _decimal(value):
    return Decimal(str(value))


def _timestamp(value):
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO string")
    result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("timestamp must have a timezone")
    return result.astimezone(UTC)


def _day(value):
    if not isinstance(value, str):
        raise ValueError("date must be YYYY-MM-DD")
    result = dt.date.fromisoformat(value)
    if result.isoformat() != value:
        raise ValueError("date must be YYYY-MM-DD")
    return result


def _protocol_reasons(protocol):
    """Refuse parameter changes rather than turn a frozen endpoint into a tuning interface."""
    expected = {
        "schema_version": 1,
        "protocol_id": "weather-challenger-forward-20261007-v1",
        "primary_family": "scale_only",
        "decision_policy.policy_version": "weather-challenger-v1",
        "decision_policy.policy_sha256": POLICY_SHA256,
        "decision_policy.specification.fee_inclusive_stake": 15.0,
        "decision_policy.specification.min_price": 0.1,
        "decision_policy.specification.min_net_edge": 0.04,
        "decision_policy.specification.max_orders_per_day": 5,
        "decision_policy.specification.max_positions_per_event": 1,
        "calendar.capture_date_from": "2026-10-08",
        "calendar.capture_date_through": "2026-12-06",
        "calendar.capture_days": 60,
        "calendar.outcome_receipt_cutoff_exclusive": "2026-12-22T00:00:00Z",
        "calendar.earliest_final_assessment": "2026-12-22T00:00:00Z",
        "calendar.early_success_allowed": False,
        "calendar.adaptive_extension_allowed": False,
        "settlement.source": "kalshi",
        "settlement.require_outcome_observed_at": True,
        "settlement.require_receipt_after_artifact_publication_bound": True,
        "settlement.legacy_clock_fallback_allowed": False,
        "accounting.order_budget_including_fee": 15.0,
        "accounting.integer_contracts": True,
        "accounting.baseline_or_stress_reselection_allowed": False,
        "inference.restart_probability": RESTART_PROBABILITY,
        "inference.mean_block_length_days": 7,
        "inference.draws": BOOTSTRAP_DRAWS,
        "inference.seed": BOOTSTRAP_SEED,
        "inference.zero_cost_draw_roi": 0.0,
        "final_criteria.minimum_settled_orders": 100,
        "final_criteria.minimum_distinct_target_dates_with_orders": 30,
        "final_criteria.minimum_strictly_negative_target_day_groups": 2,
        "final_criteria.net_strictly_positive": True,
        "final_criteria.roi_bootstrap_lower_2_5_percent_strictly_positive": True,
        "final_criteria.stress_net_strictly_positive": True,
        "final_criteria.net_without_best_city_strictly_positive": True,
        "final_criteria.require_multiple_cities": True,
        "final_criteria.maximum_observed_target_day_drawdown": 50.0,
        "final_criteria.all_criteria_required": True,
        "verdicts.live_authorization": False,
    }
    reasons = []
    for path, wanted in expected.items():
        value = protocol
        for key in path.split("."):
            value = value.get(key) if isinstance(value, dict) else None
        # Reject bools masquerading as integer/numeric constants, including True == 1.
        valid = (type(value) is type(wanted) if type(wanted) in (bool, int)
                 else _number(value) if type(wanted) is float else isinstance(value, str))
        if not valid or value != wanted:
            reasons.append(f"protocol mismatch: {path}")
    return reasons


def stationary_bootstrap(groups):
    """Fixed circular stationary bootstrap of daily (net, fee-inclusive cost) pairs.

    No alternate seed, block length, draw count or percentile is accepted. Report all zero-cost
    replicates as ROI zero. This helper takes aggregates only, never market data or selections.
    """
    if not groups:
        return None
    pairs = [(float(net), float(cost)) for net, cost in groups]
    if any(not math.isfinite(net) or not math.isfinite(cost) or cost < 0
           for net, cost in pairs):
        raise ValueError("bootstrap pairs must have finite net and nonnegative cost")
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(pairs)
    ratios, zero_cost = [], 0
    for _ in range(BOOTSTRAP_DRAWS):
        index = rng.randrange(n)
        net, cost = pairs[index]
        for _ in range(1, n):
            if rng.random() < RESTART_PROBABILITY:
                index = rng.randrange(n)
            else:
                index = (index + 1) % n
            value, stake = pairs[index]
            net += value
            cost += stake
        if cost == 0:
            zero_cost += 1
            ratios.append(0.0)
        else:
            ratios.append(net / cost)
    ratios.sort()
    return dict(method="circular_stationary", draws=BOOTSTRAP_DRAWS,
                seed=BOOTSTRAP_SEED, restart_probability=RESTART_PROBABILITY,
                calendar_days=n, quantile_indices=[500, 19500],
                roi_95=[ratios[500], ratios[19500]], zero_cost_draws=zero_cost)


def _validate_order(order):
    errors = []
    if not isinstance(order, dict):
        return ["order must be an object"]
    for key in ("ticker", "city"):
        if not isinstance(order.get(key), str) or not order[key].strip():
            errors.append(f"invalid {key}")
    if order.get("side") not in ("yes", "no"):
        errors.append("invalid side")
    try:
        capture_day, target_day = _day(order.get("run_at")), _day(order.get("target_date"))
        bound = _timestamp(order.get("publication_bound"))
        if not START <= capture_day <= END:
            errors.append("capture date outside registration")
        if target_day <= capture_day:
            errors.append("target must follow capture date")
        if bound.date() != capture_day or bound >= dt.datetime.combine(target_day, dt.time(), UTC):
            errors.append("publication bound must be on capture day and before target midnight")
    except (ValueError, TypeError, OverflowError):
        errors.append("invalid capture/target date or publication bound")
    for key in ("price", "principal", "fee", "cost", "probability", "edge"):
        if not _number(order.get(key)):
            errors.append(f"invalid finite numeric {key}")
    if type(order.get("contracts")) is not int or order["contracts"] <= 0:
        errors.append("contracts must be a positive integer")
    if errors:
        return errors
    if not challenger.MIN_PRICE <= order["price"] < 1:
        errors.append("price outside policy bounds")
        return errors
    expected = challenger.size_order(order["price"])
    if expected is None or order["contracts"] != expected["contracts"]:
        errors.append("saved contracts do not match fee-inclusive integer sizing")
        return errors
    if expected is not None:
        for key in ("principal", "fee", "cost"):
            if not math.isclose(order[key], expected[key], rel_tol=0, abs_tol=1e-9):
                errors.append(f"saved {key} does not match sizing/rounded fees")
    if not 0 <= order["probability"] <= 1:
        errors.append("probability outside [0,1]")
    edge = order["probability"] - order["price"] - order["fee"] / order["contracts"]
    if not math.isclose(order["edge"], edge, rel_tol=0, abs_tol=1e-9):
        errors.append("saved edge inconsistent with probability, price and rounded fee")
    if edge + 1e-12 < challenger.MIN_EDGE:
        errors.append("saved order below fixed net-edge threshold")
    return errors


def _resolve(order, matching, now):
    """Only valid, compatible pre-cutoff receipts establish a primary outcome."""
    errors, values, admitted, late, future, pending = [], set(), [], [], [], []
    bound = _timestamp(order["publication_bound"])
    for row in matching:
        if row.get("source") != "kalshi":
            continue
        value = row.get("outcome")
        stamp = row.get("outcome_observed_at")
        if value is None and stamp is None:
            pending.append(_preserve(row))
            continue  # An earlier unresolved snapshot is not an outcome or a contradiction.
        try:
            observed = _timestamp(stamp)
        except (ValueError, TypeError, OverflowError):
            errors.append("outcome has missing or invalid observation timestamp")
            continue
        if observed >= CUTOFF:
            late.append(_preserve(row))
            continue
        if observed > now:
            future.append(_preserve(row))
            continue
        if row.get("target_date") != order["target_date"] or row.get("city") != order["city"]:
            errors.append("outcome target/city identity conflicts with saved order")
            continue
        if observed <= bound:
            errors.append("outcome receipt must strictly follow decision publication")
            continue
        if not _number(value) or value not in (0, 1):
            errors.append("outcome must be numeric binary 0 or 1")
            continue
        values.add(int(value))
        admitted.append(_preserve(row))
    if len(values) > 1:
        errors.append("conflicting pre-cutoff binary outcomes")
    resolved = next(iter(values)) if len(values) == 1 and not errors else None
    return dict(outcome=resolved, reasons=sorted(set(errors)),
                admitted_evidence=admitted, post_cutoff_evidence=late,
                future_evidence=future, unresolved_observations=pending,
                matching_evidence=_preserve(matching))


def assess(protocol, orders, outcomes, now):
    """Evaluate saved primary selections; complete artifact coverage is a caller obligation.

    Invalid caller clock raises ValueError. Invalid protocol/orders/outcome evidence yields
    pending before the registered cutoff or inconclusive afterward, never a passing result.
    Neither inputs nor other modules are modified. Amounts are accumulated with Decimal from
    saved values; JSON numeric outputs have corresponding exact saved-value decimal totals.
    """
    if not isinstance(now, dt.datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be a timezone-aware datetime")
    now = now.astimezone(UTC)
    reasons = _protocol_reasons(protocol)
    result = dict(schema_version=1, protocol_id="weather-challenger-forward-20261007-v1",
                  assessment_at_utc=now.isoformat(), cutoff_exclusive=CUTOFF.isoformat(),
                  paper_only=True, live_authorization=False,
                  coverage_verified_by_this_function=False,
                  coverage_required="Caller must override favorable verdict for incomplete/invalid artifact coverage.",
                  verdict="pending" if now < CUTOFF else "inconclusive", reasons=reasons,
                  selected_orders=len(orders) if isinstance(orders, list) else None,
                  order_results=[], post_cutoff_evidence=[], future_evidence=[],
                  metrics=None, criteria=None)
    if not isinstance(orders, list) or not isinstance(outcomes, list):
        reasons.append("orders and outcomes must be lists")
    if reasons:
        return result
    valid, tickers, events, counts = [], set(), set(), collections.Counter()
    for index, order in enumerate(orders):
        errors = _validate_order(order)
        entry = dict(index=index, saved_order=_preserve(order), state="invalid_order", reasons=errors)
        result["order_results"].append(entry)
        if errors:
            reasons.extend(f"order {index}: {message}" for message in errors)
            continue
        event = (order["city"], order["target_date"])
        if order["ticker"] in tickers or event in events:
            errors.append("duplicate ticker or city/target selection")
        tickers.add(order["ticker"])
        events.add(event)
        counts[order["run_at"]] += 1
        if counts[order["run_at"]] > challenger.MAX_ORDERS_PER_DAY:
            errors.append("more than five saved orders on one capture date")
        if _timestamp(order["publication_bound"]) > now:
            errors.append("decision publication bound is after assessment time")
        if errors:
            reasons.extend(f"order {index}: {message}" for message in errors)
        valid.append((order, entry))
    by_ticker = collections.defaultdict(list)
    for row in outcomes:
        if not isinstance(row, dict):
            reasons.append("outcome row must be an object")
            continue
        ticker = row.get("market_id")
        if isinstance(ticker, str) and ticker in tickers:
            by_ticker[ticker].append(row)
    groups, saved_targets = {}, []
    for order in orders:
        try:
            saved_targets.append(_day(order.get("target_date")))
        except (AttributeError, ValueError, TypeError, OverflowError):
            pass  # Invalid dates already prevent a complete financial assessment.
    if saved_targets:
        first, last = min(saved_targets), max(saved_targets)
        for offset in range((last - first).days + 1):
            day = str(first + dt.timedelta(days=offset))
            groups[day] = dict(orders=0, invalid_orders=0, unresolved=0, cost=Decimal(0),
                               fees=Decimal(0), settled_cost=Decimal(0),
                               settled_fees=Decimal(0), settled_net=Decimal(0),
                               unresolved_cost=Decimal(0))
    city_groups = collections.defaultdict(lambda: [Decimal(0), Decimal(0)])
    stress_cost = stress_fee = stress_net = Decimal(0)
    settled = wins = unpriceable = 0
    valid_indices = {entry["index"] for _, entry in valid}
    invalid_entries = [entry for entry in result["order_results"] if entry["index"] not in valid_indices]
    for entry in invalid_entries:
        raw = orders[entry["index"]]
        target = raw.get("target_date") if isinstance(raw, dict) else None
        if isinstance(target, str) and target in groups:
            groups[target]["orders"] += 1
            groups[target]["invalid_orders"] += 1
    for order, entry in valid:
        resolution = _resolve(order, by_ticker[order["ticker"]], now)
        prior_errors = entry["reasons"]
        entry.update(resolution)
        entry["reasons"] = prior_errors + resolution["reasons"]
        reasons.extend(f"order {entry['index']}: {message}" for message in resolution["reasons"])
        result["post_cutoff_evidence"].extend(resolution["post_cutoff_evidence"])
        result["future_evidence"].extend(resolution["future_evidence"])
        group = groups[order["target_date"]]
        cost, fee = _decimal(order["cost"]), _decimal(order["fee"])
        group["orders"] += 1
        group["cost"] += cost
        group["fees"] += fee
        if resolution["outcome"] is None:
            entry["state"] = "invalid_evidence" if resolution["reasons"] else "unresolved"
            entry["unresolved_worst_case_net"] = -float(cost)
            group["unresolved"] += 1
            group["unresolved_cost"] += cost
            continue
        won = resolution["outcome"] == int(order["side"] == "yes")
        net = (Decimal(order["contracts"]) if won else Decimal(0)) - cost
        entry.update(state="settled", won=won, net=float(net))
        settled += 1
        wins += won
        group["settled_cost"] += cost
        group["settled_fees"] += fee
        group["settled_net"] += net
        city_groups[order["city"]][0] += net
        city_groups[order["city"]][1] += cost
        stressed = challenger.size_order(order["price"] + 0.01)
        if stressed is None:
            unpriceable += 1
            entry["stress_1c"] = None
        else:
            stressed_net = (Decimal(stressed["contracts"]) if won else Decimal(0)) - _decimal(stressed["cost"])
            stress_net += stressed_net
            stress_cost += _decimal(stressed["cost"])
            stress_fee += _decimal(stressed["fee"])
            entry["stress_1c"] = dict(price=order["price"] + 0.01, **stressed, net=float(stressed_net))
    sums = {key: sum((g[key] for g in groups.values()), Decimal(0))
            for key in ("cost", "fees", "settled_cost", "settled_fees", "settled_net", "unresolved_cost")}
    unresolved = len(valid) - settled
    if unresolved:
        reasons.append(f"{unresolved} selected orders lack unambiguous eligible outcomes")
    complete = not reasons and len(valid) == len(orders) and unresolved == 0
    calendar_groups = []
    for target, group in groups.items():
        item = {key: float(value) if isinstance(value, Decimal) else value
                for key, value in group.items()}
        item.update(target_date=target, net=None if group["unresolved"] or group["invalid_orders"]
                    else float(group["settled_net"]))
        calendar_groups.append(item)
    traded_days = sum(g["orders"] > 0 for g in groups.values())
    net, cost = sums["settled_net"], sums["cost"]
    negative_days = sum(g["settled_net"] < 0 for g in groups.values()) if complete else None
    bootstrap = stationary_bootstrap([(g["settled_net"], g["cost"]) for g in groups.values()]) if complete else None
    cumulative = peak = max_drawdown = Decimal(0)
    if complete:
        for group in groups.values():
            cumulative += group["settled_net"]
            peak = max(peak, cumulative)
            max_drawdown = max(max_drawdown, peak - cumulative)
    best_city = (sorted(city_groups, key=lambda city: (-city_groups[city][0], city))[0]
                 if city_groups else None)
    without_best = net - city_groups[best_city][0] if complete and len(city_groups) >= 2 else None
    metrics = dict(accounting_complete=complete, invalid_orders=len(invalid_entries),
                   calendar_grid_complete=len(saved_targets) == len(orders),
                   exposure_complete=not invalid_entries,
                   accounting_scope="economically validated orders; invalid saved orders have unknown exposure",
                   validated_orders=len(valid), settled_orders=settled, wins=wins,
                   unresolved_orders=unresolved, target_dates_with_orders=traded_days,
                   strictly_negative_target_days=negative_days,
                   **{key: float(value) for key, value in sums.items()},
                   decimal_totals={key: str(value) for key, value in sums.items()},
                   net=float(net) if complete else None,
                   roi=float(net / cost) if complete and cost else None,
                   unresolved_worst_case_net=-float(sums["unresolved_cost"]) if not invalid_entries else None,
                   net_if_all_unresolved_lose=float(net - sums["unresolved_cost"]) if not invalid_entries else None,
                   calendar_groups=calendar_groups, bootstrap=bootstrap,
                   by_city_scope="settled orders only",
                   by_city={city: dict(net=float(values[0]), cost=float(values[1]))
                            for city, values in sorted(city_groups.items())},
                   best_city=best_city, net_without_best_city=float(without_best) if without_best is not None else None,
                   maximum_observed_target_day_drawdown=float(max_drawdown) if complete else None,
                   stress_1c=dict(net=float(stress_net), cost=float(stress_cost), fees=float(stress_fee),
                                  priceable_orders=settled - unpriceable, unpriceable_orders=unpriceable))
    result["metrics"] = metrics
    info = dict(minimum_100_settled_orders=complete and settled >= 100,
                minimum_30_traded_target_dates=complete and traded_days >= 30,
                minimum_2_negative_target_days=complete and negative_days >= 2)
    performance = dict(net_strictly_positive=complete and net > 0,
                       bootstrap_lower_strictly_positive=complete and bootstrap is not None and bootstrap["roi_95"][0] > 0,
                       stress_net_strictly_positive=complete and unpriceable == 0 and stress_net > 0,
                       net_without_best_city_strictly_positive=without_best is not None and without_best > 0,
                       maximum_drawdown_at_most_50=complete and max_drawdown <= Decimal(50))
    result["criteria"] = dict(information=info, performance=performance)
    if now < CUTOFF:
        reasons.append("final assessment is not allowed before the fixed cutoff")
    elif not complete:
        result["verdict"] = "inconclusive"
    elif not all(info.values()):
        result["verdict"] = "insufficient_sample_or_tail_evidence"
        reasons.extend(f"information criterion failed: {name}" for name, passed in info.items() if not passed)
    elif not all(performance.values()):
        result["verdict"] = "does_not_meet_preregistered_paper_research_criteria"
        reasons.extend(f"performance criterion failed: {name}" for name, passed in performance.items() if not passed)
    else:
        result["verdict"] = "meets_preregistered_paper_research_criteria"
    return result
