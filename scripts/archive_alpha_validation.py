#!/usr/bin/env python3
"""Evaluate the four policies frozen in the 2026-09-22 archive preregistration.

Offline historical validation, not prospective results or evidence of fills. The
default audit accepts only the independently downloaded May 1--June 28 archive.
An optional, separately supplied March--April warmup extends training only, with
the original June input held byte-for-byte fixed. Never initialize this test from
canonical June--September captures. Run after preregistration and implementation
have been committed, and after retrieval is authorized.
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
import pilot_alpha_audit as pilot

UTC = dt.timezone.utc
FIRST_TARGET = dt.date(2026, 5, 1)
WARMUP_FIRST_TARGET = dt.date(2026, 3, 1)
WARMUP_LAST_TARGET = dt.date(2026, 4, 30)
FROZEN_BASE_CAPTURE_SHA256 = "094dafd21aee0c78f6e0912a6caca5dcc8f5b86020855a1ac56d35b23224fe26"
LAST_TARGET = dt.date(2026, 6, 28)
FIRST_ENTRY = dt.date(2026, 6, 1)
LAST_ENTRY = dt.date(2026, 6, 27)
EXCLUDED = {("NYC", "2026-06-15")}
CITY_SERIES = {
    "NYC": "KXHIGHNY", "Chicago": "KXHIGHCHI", "Austin": "KXHIGHAUS",
    "Denver": "KXHIGHDEN", "LA": "KXHIGHLAX", "Miami": "KXHIGHMIA",
    "Philadelphia": "KXHIGHPHIL", "Dallas": "KXHIGHTDAL",
    "Seattle": "KXHIGHTSEA", "Atlanta": "KXHIGHTATL", "Boston": "KXHIGHTBOS",
    "Phoenix": "KXHIGHTPHX", "Vegas": "KXHIGHTLV", "Washington": "KXHIGHTDC",
    "Houston": "KXHIGHTHOU",
}
FAMILIES = (("joint", True, True), ("bias_only", True, False),
            ("scale_only", False, True))
POLICIES = ("joint", "joint_parent_no", "bias_only", "scale_only")
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 42
BLOCK_DAYS = 7
FROZEN_SHARED_SHA256 = {
    "market_shape_alpha.py": "2c309132a39db89c55dda8daeab995a5dde12f0cfdde3f6a20056129aecf12e7",
    "pilot_alpha_audit.py": "615df82678438722b1c6ffc8ff08a4fd479485aa0b47d186c6eb7c5c5c5343fc",
    "go_live_gate.py": "061692e86b19748c878458e3ed303cc563f854fb77b871ec269958e01d396da4",
}


def dates(first, last):
    return [first + dt.timedelta(days=i) for i in range((last - first).days + 1)]


def entry_time(target):
    return int(dt.datetime.combine(target - dt.timedelta(days=1),
                                   dt.time(15), tzinfo=UTC).timestamp())


def unix_time(value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("timestamps must be positive integer Unix seconds")
    return value


def iso_time(value):
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("archive ISO timestamps must include a timezone")
    return parsed.timestamp()


def prepare_ladders(rows, first_target=FIRST_TARGET):
    """Validate normalized archive data, excluding the disclosed probe before all use."""
    if first_target not in (FIRST_TARGET, WARMUP_FIRST_TARGET):
        raise ValueError("training may start only at the original or frozen warmup boundary")
    grouped = collections.defaultdict(list)
    seen = set()
    excluded_rows = 0
    for row in rows:
        city, target_string = row["city"], row["target_date"]
        if (city, target_string) in EXCLUDED:
            excluded_rows += 1
            continue
        target = dt.date.fromisoformat(target_string)
        if city not in CITY_SERIES or not first_target <= target <= LAST_TARGET:
            raise ValueError("row is outside the fixed archive universe or dates")
        expected_entry = entry_time(target)
        if (unix_time(row["entry_ts"]) != expected_entry or
                unix_time(row["quote_close_ts"]) != expected_entry):
            raise ValueError("quote must close exactly at prior-day 15:00 UTC")
        if row["captured_at"][:10] != str(target - dt.timedelta(days=1)):
            raise ValueError("capture date disagrees with fixed entry timestamp")
        if row.get("captured_at_utc") and iso_time(row["captured_at_utc"]) != expected_entry:
            raise ValueError("capture timestamp disagrees with entry timestamp")
        if row.get("source") != "kalshi" or row.get("archive_method") != "historical_1m_bid_ask_close":
            raise ValueError("archive must contain historical one-minute Kalshi bid/ask closes")
        if iso_time(row["open_time"]) > expected_entry:
            raise ValueError("market was not open at fixed entry")
        event = CITY_SERIES[city] + "-" + target.strftime("%y%b%d").upper()
        ticker = row["market_id"]
        if not isinstance(ticker, str) or not ticker.startswith(event + "-"):
            raise ValueError("ticker disagrees with fixed city and target date")
        if row.get("event_ticker", event) != event or ticker in seen:
            raise ValueError("duplicate ticker or inconsistent event identifier")
        seen.add(ticker)
        for key in ("best_bid", "best_ask"):
            quote = row.get(key)
            if quote is not None and (isinstance(quote, bool) or
                    not isinstance(quote, (int, float)) or not math.isfinite(quote) or
                    not 0 <= quote <= 1):
                raise ValueError("invalid archive quote")
        bid, ask = row.get("best_bid"), row.get("best_ask")
        if bid is not None and ask is not None and bid > ask:
            raise ValueError("crossed archive quote")
        if not any(value is not None and 0 < value < 1 for value in (bid, ask)):
            raise ValueError("every event leg needs at least one usable quote side")
        if row.get("unit") != "F" or row.get("market_type") not in (
                "temp_bucket", "temp_at_least", "temp_at_most"):
            raise ValueError("unexpected archive unit or market type")
        for key in ("threshold", "threshold_upper"):
            value = row.get(key)
            if (key == "threshold" or value is not None) and (
                    isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value)):
                raise ValueError("invalid temperature threshold")
        outcome = row.get("outcome")
        if isinstance(outcome, bool) or outcome not in (None, 0, 1):
            raise ValueError("archive outcomes must be binary or unknown")
        settled = row.get("settlement_ts")
        if settled is not None and unix_time(settled) <= expected_entry:
            raise ValueError("market was already settled at entry")
        if settled is not None and row.get("settlement_time_utc"):
            if math.ceil(iso_time(row["settlement_time_utc"])) != settled:
                raise ValueError("settlement timestamps disagree")
        grouped[(city, target_string)].append(row)
    for key, event_rows in grouped.items():
        bounds = sorted(shape.cell_bounds_c(row) for row in event_rows)
        if (len(bounds) < shape.COMPLETE_CELLS or bounds[0][0] != -math.inf or
                bounds[-1][1] != math.inf or any(lo >= hi for lo, hi in bounds) or
                any(abs(left[1] - right[0]) > 1e-8
                    for left, right in zip(bounds, bounds[1:]))):
            raise ValueError(f"event does not form a complete contiguous partition: {key}")
        results = [row.get("outcome") for row in event_rows]
        if all(value is not None for value in results) and sum(results) != 1:
            raise ValueError(f"complete event must have exactly one winner: {key}")
    normalized = [row for event_rows in grouped.values() for row in event_rows]
    ladders = shape.build_ladders(normalized)
    for ladder in ladders:
        event_rows = grouped[(ladder["city"], ladder["target"])]
        ladder["entry_ts"] = event_rows[0]["entry_ts"]
        stamps = [row.get("settlement_ts") for row in event_rows]
        ladder["settlement_ts"] = (max(stamps) if ladder["resolved"] and
                                   all(stamp is not None for stamp in stamps) else None)
    return ladders, excluded_rows


def causal_history(ladders, as_of):
    """A prior target is insufficient: every leg must actually have settled by entry."""
    entry_date = dt.datetime.fromtimestamp(as_of, UTC).date().isoformat()
    history = [ladder for ladder in ladders
               if (ladder["city"], ladder["target"]) not in EXCLUDED
               and shape.complete(ladder) and ladder["resolved"]
               and ladder["target"] < entry_date
               and ladder.get("settlement_ts") is not None
               and ladder["settlement_ts"] <= as_of]
    history.sort(key=lambda ladder: (ladder["target"], ladder["cap"], ladder["city"]))
    return history[-shape.HIST_CAP:]


def candidates_for(ladders, params):
    """Build decisions without reading the current ladder's outcomes."""
    candidates = []
    if params is None:
        return candidates
    for ladder in ladders:
        for probability, cell in zip(shape.shaped_probs(ladder, *params), ladder["cells"]):
            decision = shape.decide_cell(probability, cell["bid"], cell["ask"], 0.04, 0.10)
            if decision is None:
                continue
            side, price, _, edge = decision
            candidates.append(dict(run_at=ladder["cap"], target_date=ladder["target"],
                                   city=ladder["city"], ticker=cell["mid"],
                                   side="yes" if side == "BUY" else "no",
                                   price=price, edge=edge))
    return candidates


def select_outcome_blind(candidates):
    """Use the shared selector with a constant placeholder, retaining all chosen orders.

    Its public API returns only settled rows, so a constant dummy outcome exposes
    chosen quantities without dropping unknown actual outcomes. Dummy P&L and fees
    are discarded. Real settlement is joined only after every selection is fixed.
    """
    chosen, _ = pilot.select_candidates([dict(row, outcome=0) for row in candidates])
    return [{key: value for key, value in row.items()
             if key not in ("outcome", "won", "fee", "net")} for row in chosen]


def select_policies(candidates):
    selected = {name: select_outcome_blind(candidates[name]) for name, _, _ in FAMILIES}
    selected["joint_parent_no"] = [row for row in selected["joint"] if row["side"] == "no"]
    return selected


def settle_selected(selected, outcomes):
    settled, opened = [], []
    for row in selected:
        outcome = outcomes.get((row["ticker"], row["target_date"]))
        if outcome is None:
            opened.append(row)
            continue
        won = outcome == (1 if row["side"] == "yes" else 0)
        fee = gate.kalshi_fee(row["contracts"], row["price"])
        settled.append(dict(row, won=won, fee=fee,
                            net=(row["contracts"] if won else 0) - row["cost"] - fee))
    return settled, opened


def quantile_interval(values, tail):
    ordered = sorted(values)
    return [ordered[min(int(tail * len(ordered)), len(ordered) - 1)],
            ordered[min(int((1 - tail) * len(ordered)), len(ordered) - 1)]]


def bootstrap_roi(trades, block_days):
    """Fixed-calendar day or overlapping seven-day moving-block bootstrap."""
    calendar = [str(day) for day in dates(FIRST_ENTRY + dt.timedelta(days=1), LAST_TARGET)]
    totals = {day: [0.0, 0.0] for day in calendar}
    for trade in trades:
        totals[trade["target_date"]][0] += trade["net"]
        totals[trade["target_date"]][1] += trade["cost"]
    values = [totals[day] for day in calendar]
    rng = random.Random(BOOTSTRAP_SEED)
    draws, empty = [], 0
    for _ in range(BOOTSTRAP_DRAWS):
        if block_days == 1:
            sample = rng.choices(values, k=len(values))
        else:
            sample = []
            while len(sample) < len(values):
                start = rng.randrange(len(values) - block_days + 1)
                sample.extend(values[start:start + block_days])
            sample = sample[:len(values)]
        cost = sum(row[1] for row in sample)
        empty += cost == 0
        # Retain empty resamples conservatively; never condition on having placed a bet.
        draws.append(sum(row[0] for row in sample) / cost if cost else 0.0)
    return dict(calendar_days=len(calendar), block_days=block_days,
                draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED, zero_risk_draws=empty,
                nominal_95=quantile_interval(draws, 0.025),
                simultaneous_98_75=quantile_interval(draws, 0.00625))


def adverse_fill_stress(trades):
    """Keep quantities fixed, charge one cent more, and recompute rounded order fees."""
    invalid, net, cost, fees, extra, over_stake = [], 0.0, 0.0, 0.0, 0.0, 0
    city_cost = collections.defaultdict(float)
    for trade in trades:
        contracts, price = trade["contracts"], trade["price"] + 0.01
        stressed_cost = contracts * price
        extra += contracts * 0.01
        cost += stressed_cost
        over_stake += stressed_cost > 15.0 + 1e-9
        city_cost[(trade["city"], trade["target_date"])] += stressed_cost
        if not 0 < price < 1:
            invalid.append(dict(ticker=trade["ticker"], stressed_price=price,
                                contracts=contracts))
            continue
        fee = gate.kalshi_fee(contracts, price)
        fees += fee
        net += (contracts if trade["won"] else 0) - stressed_cost - fee
    return dict(contracts_unchanged=True, extra_capital_required=extra,
                requested_stressed_cost=cost, fees=fees if not invalid else None,
                net=net if not invalid else None,
                impossible_prices=invalid, orders_over_15_stake=over_stake,
                city_targets_over_30=sum(value > 30.0 + 1e-9 for value in city_cost.values()),
                note="Settlement-scored orders only; quantities are not resized. Extra capital "
                     "can exceed original stake/city caps. Impossible prices invalidate stress.")


def policy_report(selected, outcomes):
    trades, opened = settle_selected(selected, outcomes)
    result = pilot.summarize(trades)
    # Replace the older fee-fixed sensitivity with the explicitly recomputed archive stress.
    result.pop("net_with_1c_adverse_fill", None)
    result.pop("bootstrap_95", None)
    result["selected"] = len(selected)
    result["open"] = len(opened)
    result["unsettled_capital"] = sum(row["cost"] for row in opened)
    result["selection_sha256"] = hashlib.sha256(
        json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    result["by_side"] = {}
    result["by_city"] = {}
    for field, groups in (("side", ("yes", "no")), ("city", CITY_SERIES)):
        for group in groups:
            rows = [trade for trade in trades if trade[field] == group]
            result["by_" + field][group] = dict(
                settled=len(rows), net=sum(row["net"] for row in rows),
                cost=sum(row["cost"] for row in rows))
    best = sorted(result["by_city"].items(), key=lambda item: (-item[1]["net"], item[0]))[:2]
    remaining_net = result["net"] - sum(value["net"] for _, value in best)
    result["remove_two_best_cities"] = dict(cities=[city for city, _ in best], net=remaining_net)
    result["day_bootstrap"] = bootstrap_roi(trades, 1)
    result["moving_block_bootstrap"] = bootstrap_roi(trades, BLOCK_DAYS)
    result["one_cent_stress"] = adverse_fill_stress(trades)
    stress_net = result["one_cent_stress"]["net"]
    result["criteria"] = dict(
        all_selected_outcomes_known=not opened,
        positive_simultaneous_day_lower=result["day_bootstrap"]["simultaneous_98_75"][0] > 0,
        positive_simultaneous_block_lower=result["moving_block_bootstrap"]["simultaneous_98_75"][0] > 0,
        positive_repriced_stress_net=stress_net is not None and stress_net > 0,
        positive_after_removing_best_two_cities=remaining_net > 0,
    )
    result["credible_candidate_for_prospective_testing"] = all(result["criteria"].values())
    result["live_promotion"] = False
    return result


def audit_rows(rows, first_target=FIRST_TARGET):
    """Audit rows after the file-level warmup/date/hash checks in ``audit``."""
    ladders, excluded_rows = prepare_ladders(rows, first_target=first_target)
    complete = [ladder for ladder in ladders if shape.complete(ladder)]
    candidates = {name: [] for name, _, _ in FAMILIES}
    fits = {}
    for day in dates(FIRST_ENTRY, LAST_ENTRY):
        stamp = int(dt.datetime.combine(day, dt.time(15), tzinfo=UTC).timestamp())
        history = causal_history(complete, stamp)
        todays = [ladder for ladder in complete if ladder["entry_ts"] == stamp]
        fits[str(day)] = dict(history_ladders=len(history), eligible_ladders=len(todays), families={})
        for name, fit_b, fit_k in FAMILIES:
            params = shape.fit_shape(history, fit_b=fit_b, fit_k=fit_k)
            fits[str(day)]["families"][name] = params
            candidates[name].extend(candidates_for(todays, params))
    # Finish every policy's selection before joining any current contract's actual outcome.
    selected = select_policies(candidates)
    outcomes = {(row["market_id"], row["target_date"]): row.get("outcome")
                for row in rows if (row["city"], row["target_date"]) not in EXCLUDED}
    expected = {(city, str(day)) for city in CITY_SERIES for day in dates(first_target, LAST_TARGET)}
    holdout = {(city, str(day)) for city in CITY_SERIES
               for day in dates(FIRST_ENTRY + dt.timedelta(days=1), LAST_TARGET)} - EXCLUDED
    available = {(ladder["city"], ladder["target"]) for ladder in ladders}
    usable = {(ladder["city"], ladder["target"]) for ladder in complete}
    return dict(
        specification="reports/2026-09-22-archive-preregistration.md",
        interpretation="Out-of-period historical validation, not prospective results or fills. "
            "Four fixed policies; no live promotion. Bootstrap intervals are approximate. "
            "Zero-risk resamples have ROI 0 and remain in both intervals.",
        rule_coverage=dict(
            requested_event_ladders=len(expected), disclosed_probe_exclusions=sorted(EXCLUDED),
            excluded_probe_rows=excluded_rows, normalized_event_ladders=len(available),
            missing_event_ladders=len((expected - EXCLUDED) - available),
            probability_sum_incomplete_ladders=len(available - usable),
            expected_holdout_ladders=len(holdout), available_holdout_ladders=len(holdout & available),
            usable_holdout_ladders=len(holdout & usable),
            fixed_holdout_calendar_days=27,
            entry_days_with_minimum_history=sum(
                day["history_ladders"] >= shape.MIN_LADDERS for day in fits.values()),
            ladders_missing_settlement_availability=sum(
                ladder["settlement_ts"] is None for ladder in complete),
            per_city={city: dict(
                expected_holdout=sum(key[0] == city for key in holdout),
                usable_holdout=sum(key[0] == city for key in holdout & usable),
            ) for city in CITY_SERIES},
        ),
        fit_paths=fits,
        policies={name: policy_report(selected[name], outcomes) for name in POLICIES},
    )


def audit(path, warmup_path=None):
    scripts = Path(__file__).resolve().parent
    hashes = {name: hashlib.sha256((scripts / name).read_bytes()).hexdigest()
              for name in FROZEN_SHARED_SHA256}
    if hashes != FROZEN_SHARED_SHA256:
        raise ValueError("shared research code differs from the preregistered hashes")
    capture_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    rows = gate.load_jsonl(path)
    warmup = None
    if warmup_path is not None:
        if capture_hash != FROZEN_BASE_CAPTURE_SHA256:
            raise ValueError("training extension requires the unchanged original May--June archive hash")
        warmup = gate.load_jsonl(warmup_path)
        if any(not WARMUP_FIRST_TARGET <= dt.date.fromisoformat(row["target_date"])
               <= WARMUP_LAST_TARGET for row in warmup):
            raise ValueError("warmup targets must be within March 1--April 30, 2026 only")
        # All original validation and duplicate checks apply to the combined rows.
        result = audit_rows(warmup + rows, first_target=WARMUP_FIRST_TARGET)
    else:
        result = audit_rows(rows)
    result["capture_sha256"] = capture_hash
    result["shared_code_sha256"] = hashes
    if warmup is not None:
        result["evaluation_mode"] = "frozen_training_extension"
        result["base_specification"] = result["specification"]
        result["specification"] = "reports/2026-09-22-archive-training-extension-preregistration.md"
        result["interpretation"] = (
            "Frozen training extension using the same already-inspected June holdout; "
            "not a new untouched holdout or prospective experiment. Only March--April "
            "training data are added. Selection, causal availability, minimum history, "
            "fit grids, four policies, fees, stress, bootstrap and criteria are unchanged. "
            + result["interpretation"])
        result["warmup_capture_sha256"] = hashlib.sha256(Path(warmup_path).read_bytes()).hexdigest()
        result["training_extension"] = dict(
            first_target=str(WARMUP_FIRST_TARGET), last_target=str(WARMUP_LAST_TARGET),
            rows=len(warmup), expected_event_ladders=len(CITY_SERIES) * len(
                dates(WARMUP_FIRST_TARGET, WARMUP_LAST_TARGET)),
            normalized_event_ladders=len({(row["city"], row["target_date"]) for row in warmup}),
            unchanged_holdout_first_target="2026-06-02", unchanged_holdout_last_target="2026-06-28",
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", required=True, help="normalized independent archive JSONL")
    parser.add_argument("--warmup", help="separate preregistered March--April training JSONL")
    args = parser.parse_args()
    print(json.dumps(audit(args.captures, args.warmup), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
