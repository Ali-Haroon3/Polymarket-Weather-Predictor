#!/usr/bin/env python3
"""Offline assessment of saved challenger decisions under the fixed registration.

This command never fetches data, refits a model, chooses orders or authorizes live
trading. Exit zero means a report was produced, including a pending/inconclusive
report; it is not a trading-admission signal. Evidence construction and schema
are described in weather_challenger_forward_evidence.py.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

import weather_challenger_forward_evidence as evidence
import weather_challenger_forward_metrics as metrics

UTC = dt.timezone.utc
ROOT = Path(__file__).resolve().parents[1]
REGISTRATION_PATH = ROOT / "reports/2026-10-07-challenger-preregistration.json"
REGISTRATION_SHA256 = "624762fa748ff93577cd68238a40382f785dca479d6d97f90455c55b6d525194"
REGISTRATION_COMMIT = "3af3c8416e8592d56a2698f2d15acb74424e6eaf"


def clock(value):
    """Require a timezone-bearing clock, normalized to UTC."""
    result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("assessment clock requires a timezone")
    return result.astimezone(UTC)


def load_registration():
    """Reject altered criteria or a changed decision implementation."""
    raw = REGISTRATION_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != REGISTRATION_SHA256:
        raise ValueError("registration bytes do not match the preregistered hash")
    protocol = json.loads(raw)
    policy = protocol["decision_policy"]
    for name, expected in policy["policy_code_sha256"].items():
        if hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"decision implementation changed: {name}")
    encoded = json.dumps(dict(version=policy["policy_version"],
                              specification=policy["specification"],
                              code_hashes=policy["policy_code_sha256"]),
                         sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != policy["policy_sha256"]:
        raise ValueError("decision policy digest does not match its specification")
    return protocol


def evaluate(root, protocol, now):
    """Combine evidence and economics; incomplete coverage can never pass."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("assessment clock requires a timezone")
    now = now.astimezone(UTC)
    observed = evidence.load_evidence(Path(root), protocol, now)
    assessment = metrics.assess(protocol, observed.get("orders", []),
                                observed.get("outcomes", []), now)
    first = dt.date.fromisoformat(protocol["calendar"]["capture_date_from"])
    last = dt.date.fromisoformat(protocol["calendar"]["capture_date_through"])
    expected = {str(first + dt.timedelta(days=i)) for i in range((last - first).days + 1)}
    days = observed.get("days", [])
    complete = (observed.get("state") == "valid"
                and len(days) == len(expected)
                and {day.get("date") for day in days} == expected
                and all(day.get("state") == "valid" for day in days)
                and observed.get("provenance", {}).get("coverage_complete") is True
                and observed.get("provenance", {}).get("inventory_window_closed") is True)
    final_time = clock(protocol["calendar"]["earliest_final_assessment"])
    if now < final_time:
        verdict = "pending"
    elif not complete:
        verdict = protocol["verdicts"]["invalid_or_incomplete_evidence"]
    else:
        verdict = assessment["verdict"]
    evidence_summary = {key: value for key, value in observed.items()
                        if key not in ("orders", "outcomes")}
    evidence_summary["saved_primary_orders"] = len(observed.get("orders", []))
    evidence_summary["outcome_input_rows"] = len(observed.get("outcomes", []))
    return dict(schema_version=1, protocol_id=protocol["protocol_id"],
                registration_sha256=REGISTRATION_SHA256,
                registration_commit=REGISTRATION_COMMIT,
                assessed_at_utc=now.isoformat(), primary_family=protocol["primary_family"],
                verdict=verdict, complete_evidence=complete,
                live_authorization=False, evidence=evidence_summary, assessment=assessment,
                limitations=[
                    "Passing means only meeting preregistered paper-research criteria.",
                    "Quote selections and modeled fees do not establish fills or account return.",
                    "The bootstrap relies on weak dependence and stationarity; unseen losses remain possible.",
                    "No result authorizes orders or inherits the original pilot's admission sample."])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path,
                        help="Preserved raw inventory, archives, inputs and outcomes directory.")
    parser.add_argument("--as-of", type=clock,
                        help="Archived assessment clock; future dates are rejected.")
    parser.add_argument("--output", type=Path,
                        help="New JSON report outside the evidence directory; existing files are never overwritten.")
    args = parser.parse_args(argv)
    actual_now = dt.datetime.now(UTC)
    now = args.as_of or actual_now
    if now > actual_now:
        parser.error("--as-of cannot claim a future assessment")
    try:
        protocol = load_registration()
        result = evaluate(args.evidence, protocol, now)
        encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            output = args.output.resolve()
            if output.suffix != ".json" or output.is_relative_to(args.evidence.resolve()):
                raise ValueError("output must be a separate .json file outside evidence")
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
            print(json.dumps(dict(output=str(output), verdict=result["verdict"], live_authorization=False)))
        else:
            print(encoded, end="")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"cannot assess saved challenger evidence: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
