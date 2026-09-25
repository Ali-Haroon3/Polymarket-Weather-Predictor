#!/usr/bin/env python3
"""Offline station-level availability evaluation, never a trading return estimate.

The study entry point must reconcile inventory and select a primary invocation
before calling analyze_capture. Pure snapshot results do not certify coverage.
"""
import base64
import argparse
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import sys
import zlib

from weather_daily_observation import parse_daily
from weather_market_observation import (
    STATIONS, displayed_offer, fee_terms, market_lifecycle,
    report_consistent_side, validate_event,
)
from weather_source_evidence import (
    CaptureEvidence, EvidenceError, RELEASE_TIME, RESERVED_END, RESERVED_START,
    pair_source_book, strict_json, timestamp,
)
from weather_source_inventory import planned_slots, select_primary, validate_inventory

UTC = dt.timezone.utc
CLIENT_URL = "https://weather.com/vc-ap-7f3f87/_next/static/chunks/16tzq6chnik5a.js?dpl=dpl_6EuN2eXkZ12qptNwuwESajgPCKR5"
CLIENT_SHA256 = "cf82c9a3d87111091a12054bb904be96f4d6c587606c54629f0b4f5c5cff7a5d"


def unit_basis(path, now):
    """Verify preserved, explicitly supported public-client unit evidence."""
    unknown = {"state": "unknown", "reason": "fahrenheit_evidence_unavailable"}
    if path is None:
        return unknown
    try:
        payload = strict_json(gzip.decompress(Path(path).read_bytes()))
        if (type(payload.get("schema_version")) is not int or payload["schema_version"] != 1
                or payload.get("purpose") != "public_source_authority"):
            raise EvidenceError("unsupported_unit_evidence_schema")
        records = [row for row in payload["requests"] if row.get("role") == "fahrenheit_client"]
        if len(records) != 1:
            raise EvidenceError("missing_or_duplicate_unit_evidence")
        record = records[0]
        raw = base64.b64decode(record["body_base64"], validate=True)
        if (record.get("original_url") != CLIENT_URL or record.get("final_url") != CLIENT_URL
                or record.get("status") != 200 or record.get("body_complete") is not True
                or record.get("error") is not None
                or type(record.get("bytes")) is not int or len(raw) != record["bytes"]
                or record.get("sha256") != CLIENT_SHA256
                or hashlib.sha256(raw).hexdigest() != CLIENT_SHA256):
            raise EvidenceError("unit_evidence_identity_or_bytes_mismatch")
        requested, received, finished = map(timestamp, (record["request_at_utc"], record["receipt_at_utc"],
                                                        record["attempt_finished_at_utc"]))
        if now.tzinfo is None or not requested <= received <= finished <= now.astimezone(UTC):
            raise EvidenceError("unit_evidence_chronology_invalid")
        return {"state": "verified", "url": CLIENT_URL, "sha256": CLIENT_SHA256,
                "receipt_at_utc": record["receipt_at_utc"],
                "limitation": "This receipt does not reconstruct the earlier missing client receipt."}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError, zlib.error):
        return unknown


def unknown_station(series, reason):
    return {"series": series, "city": STATIONS[series][0], "state": "unknown",
            "reason": reason, "source": None, "markets": [], "fee": None,
            "complete_components": False, "payout_comparison_available": False}


def _read(capture, role, release, now, **identity):
    try:
        return capture.response(role, validation_released=release, now=now, **identity), None
    except EvidenceError as exc:
        return None, str(exc)


def _station(capture, series, climate, climate_error, basis, release, now):
    result = unknown_station(series, "incomplete_evidence")
    markets, market_error = _read(capture, "markets", release, now, series=series)
    if markets is None:
        result["reason"] = market_error
        return result
    event = validate_event(markets.payload, series, capture.target.isoformat())
    if event["state"] != "valid":
        result.update(reason="unsupported_market_event", event_reasons=event["reasons"])
        return result
    result["market_provenance"] = markets.provenance
    if climate is None:
        result["reason"] = climate_error
        return result
    source = parse_daily(climate.payload, capture.target.isoformat(), event["station"]["cli_id"],
                         fahrenheit_basis_verified=basis["state"] == "verified")
    result.update(source=source, source_provenance=climate.provenance)
    if source["state"] == "unknown":
        result["reason"] = source["reason"]
        return result
    saved_series, fee_error = _read(capture, "series", release, now, series=series)
    fee = (fee_terms(saved_series.payload, series) if saved_series is not None else
           {"state": "unknown", "reasons": [fee_error], "payout_comparison_available": False})
    result["fee"] = fee
    if saved_series is not None:
        result["fee_provenance"] = saved_series.provenance
    components_known = fee["state"] == "supported_metadata"
    for normalized in event["markets"]:
        ticker = normalized["ticker"]
        record = {"ticker": ticker, "state": "unknown", "reason": "incomplete_book"}
        result["markets"].append(record)
        book, error = _read(capture, "orderbook", release, now, series=series, ticker=ticker)
        if book is None:
            record["reason"] = error
            components_known = False
            continue
        record["book_provenance"] = book.provenance
        raw_market = normalized["raw_market"]
        lifecycle = market_lifecycle(raw_market, markets.receipt_at.isoformat(),
                                     book.request_at.isoformat(), book.receipt_at.isoformat(),
                                     target_date=capture.target.isoformat(),
                                     timezone=event["station"]["timezone"])
        record["lifecycle"] = lifecycle
        pairing = pair_source_book(climate, markets, book, raw_market.get("close_time"))
        record["pairing"] = pairing
        # Verify both sides even when source explicitly has no report. Empty
        # valid books are distinct from a missing or corrupt book response.
        offers = {side: displayed_offer(book.payload, side) for side in ("yes", "no")}
        if any(item["state"] == "unknown" for item in offers.values()):
            record["reason"] = "unsupported_orderbook"
            components_known = False
            continue
        if lifecycle["state"] == "unknown":
            record["reason"] = "uncertain_market_lifecycle"
            components_known = False
            continue
        if lifecycle["state"] == "closed":
            record.update(state="absence", reason="market_observed_closed")
            continue
        if pairing["state"] != "eligible":
            record["reason"] = "source_book_pair_ineligible"
            components_known = False
            continue
        if source["state"] == "absent":
            record.update(state="absence", reason="official_report_not_observed")
            continue
        consistent = report_consistent_side(normalized, source["max_temp_f"])
        if consistent["state"] != "valid":
            record["reason"] = "report_consistent_side_unknown"
            components_known = False
            continue
        side = consistent["side"]
        offer = offers[side]
        record.update(side=side, offer=offer,
                      state="overlap" if offer["state"] == "available" else "absence",
                      reason="official_report_and_displayed_offer" if offer["state"] == "available"
                      else "insufficient_displayed_depth")
    result["complete_components"] = components_known
    if any(row["state"] == "overlap" for row in result["markets"]):
        result.update(state="overlap", reason="at_least_one_verified_report_consistent_offer")
    elif components_known and all(row["state"] == "absence" for row in result["markets"]):
        result.update(state="absence", reason="no_overlap_observed_with_complete_components")
    return result


def analyze_capture(directory, *, expected_target, authority_path=None,
                    validation_released=False, now=None):
    """Evaluate only a separately selected primary snapshot; never refresh APIs."""
    now = now or dt.datetime.now(UTC)
    if not isinstance(now, dt.datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise EvidenceError("analysis_clock_timezone_missing")
    now = now.astimezone(UTC)
    try:
        target = dt.date.fromisoformat(expected_target)
    except (TypeError, ValueError) as exc:
        raise EvidenceError("invalid_expected_target") from exc
    # Refuse even an aggregate of unknown source results during the reserved embargo.
    if RESERVED_START <= target <= RESERVED_END:
        if now < RELEASE_TIME or validation_released is not True:
            raise EvidenceError("reserved_analysis_locked")
    result = {"target_date": expected_target, "unit_basis": unit_basis(authority_path, now),
              "snapshot_only": True, "payout_comparison_available": False, "stations": [],
              "limitations": ["No primary-invocation or complete-inventory claim from this snapshot.",
                              "Displayed availability is not a fill, expected return, or validated alpha."]}
    try:
        capture = CaptureEvidence(directory)
        if capture.target != target:
            raise EvidenceError("capture_target_mismatch")
    except (EvidenceError, OSError) as exc:
        result["stations"] = [unknown_station(series, str(exc)) for series in STATIONS]
        return result
    result["manifest_sha256"] = capture.manifest_sha256
    climate, error = _read(capture, "climate", validation_released, now)
    result["stations"] = [_station(capture, series, climate, error, result["unit_basis"],
                                   validation_released, now) for series in STATIONS]
    return result


def _bound_capture(primary, mappings, target_date):
    """Bind a local capture to its selected coordinator evidence before body reads."""
    matches = [item for item in mappings if isinstance(item, dict)
               and item.get("run_id") == primary["id"]
               and item.get("run_attempt") == primary["run_attempt"]]
    if len(matches) != 1 or not isinstance(matches[0].get("directory"), str):
        raise EvidenceError("missing_or_duplicate_local_artifact")
    directory = Path(matches[0]["directory"]).resolve(strict=True)
    raw_status = (directory / "run-status.json").read_bytes()
    provenance = primary.get("coordinator_provenance", {})
    if hashlib.sha256(raw_status).hexdigest() != provenance.get("source_sha256"):
        raise EvidenceError("coordinator_file_hash_mismatch")
    # Status metadata contains no source values; retain its exact original bytes.
    status = strict_json(raw_status)
    canonical = json.dumps(status, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest() != provenance.get("canonical_status_sha256"):
        raise EvidenceError("coordinator_inventory_mismatch")
    manifest_path = directory / "capture" / "manifest.json"
    raw_manifest = manifest_path.read_bytes()
    recorded = status.get("evidence", {}).get("manifest")
    if (not isinstance(recorded, dict) or type(recorded.get("bytes")) is not int
            or recorded["bytes"] != len(raw_manifest)
            or recorded.get("sha256") != hashlib.sha256(raw_manifest).hexdigest()):
        raise EvidenceError("coordinator_manifest_bytes_mismatch")
    manifest = strict_json(raw_manifest)
    coordinator = primary["coordinator"]
    if (manifest.get("target_date") != target_date
            or timestamp(manifest.get("started_at_utc")) != timestamp(coordinator.get("collector_manifest_started_at_utc"))
            or timestamp(manifest.get("finished_at_utc")) > timestamp(coordinator.get("finished_at_utc"))):
        raise EvidenceError("capture_does_not_match_selected_invocation")
    return directory / "capture"


def load_inventory(path):
    """Verify exporter-preserved bytes before validating decoded GitHub metadata."""
    path = Path(path).resolve(strict=True)
    payload = strict_json(path.read_bytes())
    if not isinstance(payload, dict):
        raise EvidenceError("invalid_inventory_object")
    for category in ("pages", "attempts", "coordinator_statuses"):
        records = payload.get(category)
        if not isinstance(records, list):
            raise EvidenceError("missing_inventory_records")
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("raw_file"), str):
                raise EvidenceError("missing_inventory_raw_evidence")
            source = (path.parent / record["raw_file"]).resolve(strict=True)
            if not source.is_relative_to(path.parent) or not source.is_file():
                raise EvidenceError("inventory_raw_path_outside_snapshot")
            if source.stat().st_size > 8 * 1024 * 1024:
                raise EvidenceError("inventory_raw_body_too_large")
            raw = source.read_bytes()
            if category == "coordinator_statuses":
                digest, decoded = record.get("source_sha256"), record.get("status")
            else:
                if (record.get("body_complete") is not True or record.get("error") is not None
                        or type(record.get("bytes")) is not int or record["bytes"] != len(raw)):
                    raise EvidenceError("incomplete_inventory_response")
                digest, decoded = record.get("sha256"), record.get("body")
            if hashlib.sha256(raw).hexdigest() != digest or strict_json(raw) != decoded:
                raise EvidenceError("inventory_raw_bytes_or_decoded_body_mismatch")
    return payload


def analyze_study(payload, phase, *, authority_path=None, now=None):
    """Retain the full frozen denominator, including missing and failed observations."""
    now = now or dt.datetime.now(UTC)
    if not isinstance(now, dt.datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise EvidenceError("analysis_clock_timezone_missing")
    now = now.astimezone(UTC)
    slots = planned_slots(phase)
    inventory = validate_inventory(payload, now)
    result = {"schema_version": 1, "phase": phase, "analysis_at_utc": now.isoformat(),
              "planned_slots": len(slots), "planned_station_checkpoints": len(slots) * len(STATIONS),
              "inventory": {key: inventory[key] for key in ("state", "reasons", "provenance", "reserved_release_ready")},
              "payout_comparison_available": False, "trading_candidate_defined": False,
              "limitations": ["Repeated checkpoints and cities on one target date are not independent outcomes.",
                              "Observed availability does not establish fills, expected return or validated alpha.",
                              "Missing observations are unknown, never replaced by a successful duplicate."]}
    # During the reserved embargo, emit operational inventory only, never source
    # comparisons, eligibility totals, values, quotes, or inferred outcomes.
    if phase == "reserved_validation" and not inventory["reserved_release_ready"]:
        result.update(state="locked", reason="reserved_inventory_or_time_gate")
        return result
    selected = select_primary(inventory["records"], phase, now) if inventory["state"] == "valid" else None
    mappings = payload.get("artifact_paths", []) if isinstance(payload, dict) else []
    if not isinstance(mappings, list):
        mappings = []
    records, evaluated = [], 0
    for slot in (selected["slots"] if selected is not None else slots):
        row = {key: slot[key] for key in ("target_date", "slot_utc", "phase")}
        row.update(selection_state=slot.get("state", "unknown"),
                   reasons=slot.get("reasons", ["inventory_incomplete_or_invalid"]),
                   candidates=slot.get("candidates", []), duplicates=slot.get("duplicates", []),
                   primary=None)
        snapshot = None
        if slot.get("state") == "selected":
            primary = slot["primary"]
            row["primary"] = {"run_id": primary["id"], "run_attempt": primary["run_attempt"]}
            try:
                capture_path = _bound_capture(primary, mappings, slot["target_date"])
                snapshot = analyze_capture(capture_path, expected_target=slot["target_date"],
                                           authority_path=authority_path,
                                           validation_released=inventory["reserved_release_ready"], now=now)
                evaluated += 1
            except (EvidenceError, OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                row["reasons"] = [str(exc) if isinstance(exc, EvidenceError) else "local_artifact_unavailable_or_invalid"]
        row["snapshot"] = {key: value for key, value in snapshot.items() if key != "stations"} if snapshot is not None else None
        reason = row["reasons"][0] if row["reasons"] else "snapshot_unavailable"
        row["stations"] = snapshot["stations"] if snapshot is not None else [unknown_station(s, reason) for s in STATIONS]
        records.append(row)
    by_target = {}
    by_city = {series: {"city": STATIONS[series][0], "overlap": 0, "absence": 0, "unknown": 0}
               for series in STATIONS}
    counts = {"overlap": 0, "absence": 0, "unknown": 0}
    for row in records:
        target = by_target.setdefault(row["target_date"], {"overlap": 0, "absence": 0, "unknown": 0,
                                                          "slots": 0, "pending_station_checkpoints": 0,
                                                          "by_city": {series: {"overlap": 0, "absence": 0, "unknown": 0}
                                                                      for series in STATIONS}})
        target["slots"] += 1
        if row["selection_state"] == "pending":
            target["pending_station_checkpoints"] += len(STATIONS)
        for station in row["stations"]:
            target[station["state"]] += 1
            counts[station["state"]] += 1
            by_city[station["series"]][station["state"]] += 1
            target["by_city"][station["series"]][station["state"]] += 1
    result.update(state="evaluated_snapshots" if evaluated else "not_evaluated", evaluated_snapshots=evaluated,
                  counts=counts, by_target=by_target, by_city=by_city, slots=records,
                  skipped=selected["skipped"] if selected is not None else [],
                  pending_station_checkpoints=sum(v["pending_station_checkpoints"] for v in by_target.values()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--phase", choices=("development", "reserved_validation"), required=True)
    parser.add_argument("--authority-evidence", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        # Inventory contains operational GitHub/status metadata, never quote bodies.
        payload = load_inventory(args.inventory)
        result = analyze_study(payload, args.phase, authority_path=args.authority_evidence)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({"state": "failed", "error_type": type(exc).__name__}), file=sys.stderr)
        return 1
    print(json.dumps({key: result[key] for key in ("state", "phase", "planned_station_checkpoints")}))
    return 0 if result["inventory"]["state"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
