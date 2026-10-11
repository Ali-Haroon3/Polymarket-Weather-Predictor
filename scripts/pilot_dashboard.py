#!/usr/bin/env python3
"""Read-only dashboard accounting from the original pilot ledger and captures.

Paper intentions and reconciled live orders remain separate. Fees are the frozen
modeled fees, not account statements. This command does not fetch or write data.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys

import go_live_gate as gate

UTC = dt.timezone.utc
STRATEGY = "market-shape"


def _day(value):
    if not isinstance(value, str) or len(value) != 10:
        raise ValueError("date must use YYYY-MM-DD")
    result = dt.date.fromisoformat(value)
    if result.isoformat() != value:
        raise ValueError("date must use YYYY-MM-DD")
    return result


def _clock(value):
    if not isinstance(value, str):
        raise ValueError("order/capture clock must be a string")
    if len(value) == 10:
        return dt.datetime.combine(_day(value), dt.time(), UTC)
    result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("order/capture timestamp requires a timezone")
    return result.astimezone(UTC)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _nonfinite(_):
    raise ValueError("nonfinite JSON number")


def _float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite JSON number")
    return result


def _read(path):
    path = Path(path)
    raw = path.read_bytes()
    rows = []
    for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, object_pairs_hook=_object, parse_constant=_nonfinite,
                             parse_float=_float)
            if not isinstance(row, dict):
                raise ValueError("JSONL row must be an object")
        except ValueError as exc:
            raise ValueError(f"invalid JSONL at {path}:{number}: {exc}") from exc
        rows.append(row)
    return rows, dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _validate(ledger, captures):
    if not captures:
        raise ValueError("capture input is empty")
    for row in captures:
        _clock(row.get("captured_at"))
        _day(row.get("target_date"))
        value = row.get("outcome")
        if value is not None and (not _number(value) or value not in (0, 1)):
            raise ValueError("capture outcome must be numeric binary or null")
        if row.get("source") == "kalshi" and not (
                isinstance(row.get("market_id"), str) and row["market_id"]):
            raise ValueError("Kalshi capture requires a market ID")
    for row in ledger:
        if "dry_run" in row and type(row["dry_run"]) is not bool:
            raise ValueError("ledger dry_run must be boolean when supplied")
        if row.get("decision") != "order" or row.get("error"):
            continue
        _clock(row.get("run_at"))
        _day(row.get("target_date"))
        if not all(isinstance(row.get(k), str) and row[k] for k in ("ticker", "city")):
            raise ValueError("order requires ticker and city")
        price, n, cost = gate.paid_price(row), row.get("contracts"), row.get("cost")
        if (type(n) is not int or n <= 0 or not _number(price) or not 0 < price < 1
                or not _number(cost) or abs(cost - n * price) > 0.011
                or gate.side_of(row) not in ("yes", "no")):
            raise ValueError("invalid order economics")


def _summary(settled, opened, unfilled, mode):
    rows = [row for row in settled if row["mode"] == mode]
    principal = math.fsum(row["cost"] for row in rows)
    fees = math.fsum(row["fee"] for row in rows)
    net = math.fsum(row["net"] for row in rows) if rows else None
    is_paper = mode == "paper"
    return dict(settled=len(rows), wins=sum(row["won"] for row in rows),
                principal=principal, fees=fees, net=net,
                roi=net / principal if net is not None and principal else None,
                open=sum(row.get("dry_run", True) == is_paper for row in opened),
                unfilled=sum(row.get("dry_run", True) == is_paper for row in unfilled))


def build_readout(ledger_path, captures_path):
    """Hash the exact bytes parsed, then reuse the unchanged pilot accounting."""
    ledger, ledger_input = _read(ledger_path)
    captures, capture_input = _read(captures_path)
    _validate(ledger, captures)
    settled, opened, unfilled = gate.settle(ledger, captures, strategy=STRATEGY)
    admission = gate.report(ledger, captures, strategy=STRATEGY)
    orders = [row for row in ledger if row.get("decision") == "order"
              and not row.get("error") and gate.strategy_of(row) == STRATEGY]
    latest = max(orders, key=lambda row: _clock(row["run_at"])) if orders else None
    return dict(schema_version=1, strategy=STRATEGY,
                latest_capture=max(_clock(row["captured_at"]).date().isoformat() for row in captures),
                latest_order_run=latest["run_at"] if latest else None,
                inputs=dict(ledger=ledger_input, captures=capture_input),
                gate_passed=admission["go_live"],
                unverified_live_orders=admission["unverified_live_orders"],
                paper=_summary(settled, opened, unfilled, "paper"),
                live=_summary(settled, opened, unfilled, "live"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", default="data/pilot_trades.jsonl")
    parser.add_argument("--captures", default="data/captures.jsonl")
    args = parser.parse_args(argv)
    try:
        result = build_readout(args.ledger, args.captures)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        print(f"pilot readout unavailable: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
