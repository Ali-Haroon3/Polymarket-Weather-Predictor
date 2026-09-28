#!/usr/bin/env python3
"""Validate saved GitHub metadata and select frozen source-study invocations offline.

No network, files, weather bodies, prices, or outcomes are read. Completeness is
relative to the preserved GitHub response evidence, not a cryptographic claim
about GitHub. Raw responses must be retained by the caller for reproduction.
"""
import datetime as dt
import hashlib
import json
import math
import re
from urllib.parse import parse_qs, quote, urlsplit

from weather_source_schedule import (FIRST_TARGET, LAST_TARGET, LAST_DEVELOPMENT_TARGET,
                                     TOLERANCE, parse_time, slot_at, stamp, utc)

REPOSITORY = "Ali-Haroon3/Polymarket-Weather-Predictor"
WORKFLOW_PATH = ".github/workflows/weather-source-research.yml"
WORKFLOW_NAME = WORKFLOW_PATH.rsplit("/", 1)[-1]
QUERY_FROM = "2026-09-26T13:00:00Z"
QUERY_THROUGH = "2026-10-24T09:30:00Z"
RELEASE_TIME = parse_time(QUERY_THROUGH)
PHASES = ("development", "reserved_validation")
TERMINAL_CONCLUSIONS = frozenset(("success", "failure", "neutral", "cancelled", "skipped",
                                  "timed_out", "action_required", "stale", "startup_failure"))


class InventoryError(ValueError):
    """Preserved metadata cannot establish an unambiguous inventory."""


def planned_slots(phase="development"):
    if phase not in PHASES:
        raise ValueError("unknown_phase")
    result = []
    for offset in range((LAST_TARGET - FIRST_TARGET).days + 1):
        target = FIRST_TARGET + dt.timedelta(days=offset)
        actual_phase = "development" if target <= LAST_DEVELOPMENT_TARGET else "reserved_validation"
        if actual_phase != phase:
            continue
        for day, hour in ((0, 13), (0, 17), (0, 21), (1, 1), (1, 5), (1, 9)):
            instant = dt.datetime.combine(target + dt.timedelta(days=day), dt.time(hour, 15), dt.timezone.utc)
            result.append({"target_date": target.isoformat(), "slot_utc": stamp(instant), "phase": phase})
    return result


def _require(condition, reason):
    if not condition:
        raise InventoryError(reason)


def _integer(value):
    return type(value) is int and value > 0


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _response(entry, now, expected_path, expected_query=None):
    _require(isinstance(entry, dict) and type(entry.get("status")) is int and entry["status"] == 200,
             "metadata_response_unsuccessful")
    url = urlsplit(entry.get("request_url", ""))
    _require(url.scheme == "https" and url.netloc == "api.github.com" and not url.fragment
             and url.path == expected_path, "metadata_request_identity_mismatch")
    _require(parse_qs(url.query, keep_blank_values=True) == (expected_query or {}),
             "metadata_request_query_mismatch")
    requested, received = parse_time(entry.get("requested_at_utc")), parse_time(entry.get("received_at_utc"))
    _require(requested <= received <= now, "metadata_retrieval_timestamp_order_invalid")
    body = entry.get("body")
    _require(isinstance(body, dict), "invalid_metadata_body")
    return body, {"request_url": entry["request_url"], "requested_at_utc": stamp(requested),
                  "received_at_utc": stamp(received), "canonical_body_sha256": _digest(body)}


def _run(body, received):
    _require(_integer(body.get("id")) and _integer(body.get("run_attempt")), "invalid_run_identity")
    _require(body.get("event") == "schedule" and body.get("path") == WORKFLOW_PATH,
             "run_workflow_identity_mismatch")
    repository = body.get("repository")
    _require(isinstance(repository, dict) and repository.get("full_name") == REPOSITORY,
             "run_repository_identity_mismatch")
    _require(body.get("html_url") == f"https://github.com/{REPOSITORY}/actions/runs/{body['id']}",
             "run_url_identity_mismatch")
    _require(isinstance(body.get("head_sha"), str) and re.fullmatch(r"[0-9a-f]{40}", body["head_sha"]),
             "invalid_head_sha")
    created, updated = parse_time(body.get("created_at")), parse_time(body.get("updated_at"))
    _require(parse_time(QUERY_FROM) <= created <= RELEASE_TIME and created <= updated <= received,
             "run_timestamp_order_or_query_bounds_invalid")
    started = body.get("run_started_at")
    if started is not None:
        _require(created <= parse_time(started) <= updated, "run_timestamp_order_invalid")
    status, conclusion = body.get("status"), body.get("conclusion")
    _require(status in ("queued", "in_progress", "completed", "waiting", "requested", "pending"),
             "unknown_run_status")
    _require((status == "completed" and conclusion in TERMINAL_CONCLUSIONS)
             or (status != "completed" and conclusion is None), "run_terminal_metadata_inconsistent")
    return {key: body.get(key) for key in ("id", "run_attempt", "event", "created_at", "updated_at",
                                          "run_started_at", "status", "conclusion", "head_sha", "html_url")}


def validate_inventory(payload, now):
    """Check explicit schema-v1 pagination/attempt evidence; assertions are ignored.

    The query is the entire fixed study creation window with event=schedule and
    per_page=100. Every contiguous page carries its request URL/times, HTTP status,
    and original decoded body {total_count, workflow_runs}. Every listed run has
    separate attempt responses for 1..run_attempt, including reruns. Optional
    coordinator_statuses contain run_id, run_attempt, received_at_utc and status
    (the saved run-status.json), plus optional artifact_id/source_sha256.
    """
    result = {"state": "unknown", "reasons": [], "records": [], "provenance": {},
              "reserved_release_ready": False}
    try:
        now = utc(now)
        _require(isinstance(payload, dict) and type(payload.get("schema_version")) is int
                 and payload["schema_version"] == 1, "unsupported_inventory_schema")
        _require(payload.get("repository") == REPOSITORY and payload.get("workflow_path") == WORKFLOW_PATH,
                 "inventory_identity_mismatch")
        _require(payload.get("query") == {"event": "schedule", "created_from": QUERY_FROM,
                                          "created_through": QUERY_THROUGH, "per_page": 100},
                 "inventory_query_bounds_mismatch")
        pages, attempts = payload.get("pages"), payload.get("attempts")
        _require(isinstance(pages, list) and 1 <= len(pages) <= 10 and isinstance(attempts, list) and len(attempts) <= 10000,
                 "missing_metadata_pages_or_attempts")
        page_provenance, listed, counts = [], {}, []
        for index, page in enumerate(pages, 1):
            _require(type(page.get("page")) is int and page["page"] == index, "metadata_pages_not_contiguous")
            query = {"event": ["schedule"], "created": [f"{QUERY_FROM}..{QUERY_THROUGH}"],
                     "per_page": ["100"], "page": [str(index)]}
            body, provenance = _response(page, now,
                f"/repos/{REPOSITORY}/actions/workflows/{quote(WORKFLOW_NAME)}/runs", query)
            total = body.get("total_count")
            _require(type(total) is int and 0 <= total < 1000, "metadata_total_unknown_or_api_cap")
            rows = body.get("workflow_runs")
            _require(isinstance(rows, list), "invalid_metadata_run_list")
            counts.append(total)
            _require(len(rows) == min(100, max(0, total - (index - 1) * 100)), "metadata_page_size_mismatch")
            for row in rows:
                _require(isinstance(row, dict), "invalid_metadata_run")
                run = _run(row, parse_time(provenance["received_at_utc"]))
                _require(run["id"] not in listed, "duplicate_paginated_run")
                listed[run["id"]] = run
            page_provenance.append(provenance)
        _require(len(set(counts)) == 1 and len(listed) == counts[0]
                 and len(pages) == max(1, math.ceil(counts[0] / 100)), "metadata_pagination_incomplete")
        indexed, attempt_provenance = {}, []
        for entry in attempts:
            _require(isinstance(entry, dict), "invalid_attempt_entry")
            identity = (entry.get("run_id"), entry.get("run_attempt"))
            _require(all(_integer(part) for part in identity) and identity not in indexed,
                     "duplicate_or_invalid_attempt_identity")
            run_id, attempt = identity
            _require(run_id in listed and attempt <= listed[run_id]["run_attempt"], "unlisted_attempt")
            body, provenance = _response(entry, now, f"/repos/{REPOSITORY}/actions/runs/{run_id}/attempts/{attempt}")
            run = _run(body, parse_time(provenance["received_at_utc"]))
            _require((run["id"], run["run_attempt"]) == identity, "attempt_response_identity_mismatch")
            _require(all(run[key] == listed[run_id][key] for key in ("created_at", "head_sha", "html_url")),
                     "attempt_and_listing_identity_mismatch")
            if attempt == listed[run_id]["run_attempt"]:
                _require(parse_time(run["updated_at"]) >= parse_time(listed[run_id]["updated_at"]),
                         "attempt_response_older_than_listing")
            indexed[identity] = run | {"metadata_provenance": provenance, "coordinator": None}
            attempt_provenance.append(provenance)
        _require(sum(run["run_attempt"] for run in listed.values()) <= 10000, "attempt_inventory_limit_exceeded")
        expected = {(run_id, n) for run_id, run in listed.items() for n in range(1, run["run_attempt"] + 1)}
        _require(set(indexed) == expected, "attempt_inventory_incomplete")
        statuses = payload.get("coordinator_statuses", [])
        _require(isinstance(statuses, list), "invalid_coordinator_inventory")
        seen_statuses = set()
        for entry in statuses:
            _require(isinstance(entry, dict), "invalid_coordinator_entry")
            identity = (entry.get("run_id"), entry.get("run_attempt"))
            _require(all(_integer(part) for part in identity) and identity in indexed and identity not in seen_statuses,
                     "duplicate_or_unlisted_coordinator")
            seen_statuses.add(identity)
            received = parse_time(entry.get("received_at_utc"))
            _require(received <= now, "coordinator_retrieval_in_future")
            body = entry.get("status")
            _require(isinstance(body, dict) and _integer(body.get("run_id")) and _integer(body.get("run_attempt"))
                     and (body.get("run_id"), body.get("run_attempt")) == identity,
                     "coordinator_identity_mismatch")
            # Preserve operational fields only; never carry a response body into output.
            fields = ("schema_version", "state", "reason", "run_id", "run_attempt", "run_metadata",
                      "coordinator_started_at_utc", "finished_at_utc", "actual_start_at_utc",
                      "pre_request_check_at_utc", "collector_manifest_started_at_utc", "slot", "creation_slot")
            indexed[identity]["coordinator"] = {key: body.get(key) for key in fields}
            if isinstance(body.get("run_metadata"), dict):
                indexed[identity]["coordinator"]["run_metadata"] = {key: body["run_metadata"].get(key)
                    for key in ("id", "run_attempt", "event", "created_at", "run_started_at", "head_sha", "html_url")}
            _require(entry.get("artifact_id") is None or _integer(entry["artifact_id"]), "invalid_artifact_id")
            _require(entry.get("source_sha256") is None or (isinstance(entry["source_sha256"], str)
                and re.fullmatch(r"[0-9a-f]{64}", entry["source_sha256"])), "invalid_status_file_sha256")
            indexed[identity]["coordinator_provenance"] = {
                "received_at_utc": stamp(received), "canonical_status_sha256": _digest(body),
                "artifact_id": entry.get("artifact_id"), "source_sha256": entry.get("source_sha256")}
        observations = page_provenance + attempt_provenance
        inventory_from = min(row["requested_at_utc"] for row in page_provenance)
        inventory_through = max(row["received_at_utc"] for row in page_provenance)
        for record in indexed.values():
            record["inventory_observed_from_utc"] = inventory_from
            record["inventory_observed_through_utc"] = inventory_through
        result.update(state="valid", records=list(indexed.values()), provenance={
            "repository": REPOSITORY, "workflow_path": WORKFLOW_PATH, "query": payload["query"],
            "run_count": len(listed), "attempt_count": len(indexed), "page_count": len(pages),
            "pages": page_provenance, "attempts": attempt_provenance,
            "scope": "completeness within preserved GitHub metadata; no server authenticity assertion"})
        result["reserved_release_ready"] = (now >= RELEASE_TIME
            and all(parse_time(row["requested_at_utc"]) >= RELEASE_TIME for row in observations)
            and all(row["status"] == "completed" for row in indexed.values()))
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
        result["reasons"] = [str(exc) if isinstance(exc, InventoryError) else "invalid_inventory_value"]
    return result


def _candidate(record, now):
    """Return validated invocation timing or a reason retaining its unknown slot."""
    coordinator = record.get("coordinator")
    if not isinstance(coordinator, dict):
        return None, "coordinator_timing_missing"
    try:
        _require(type(coordinator.get("schema_version")) is int and coordinator["schema_version"] == 1,
                 "coordinator_schema_unknown")
        _require((coordinator.get("run_id"), coordinator.get("run_attempt"))
                 == (record["id"], record["run_attempt"]), "coordinator_identity_mismatch")
        metadata = coordinator.get("run_metadata")
        _require(isinstance(metadata, dict) and _integer(metadata.get("id"))
                 and _integer(metadata.get("run_attempt")) and all(metadata.get(key) == record.get(key)
                 for key in ("id", "run_attempt", "event", "created_at", "run_started_at", "head_sha", "html_url")),
                 "coordinator_run_metadata_mismatch")
        created, github_started = parse_time(record["created_at"]), parse_time(record.get("run_started_at"))
        started = parse_time(coordinator.get("coordinator_started_at_utc"))
        _require(created <= github_started <= started <= now, "coordinator_timestamp_order_invalid")
        received = parse_time(record.get("coordinator_provenance", {}).get("received_at_utc"))
        _require(started <= received <= now, "coordinator_timestamp_order_invalid")
        for field in ("finished_at_utc", "actual_start_at_utc", "pre_request_check_at_utc",
                      "collector_manifest_started_at_utc"):
            if coordinator.get(field) is not None:
                _require(started <= parse_time(coordinator[field]) <= received,
                         "coordinator_timestamp_order_invalid")
        return started, None
    except (ValueError, TypeError, KeyError) as exc:
        return None, str(exc) if isinstance(exc, InventoryError) else "coordinator_timing_invalid"


def select_primary(records, phase, now):
    """Choose first started invocation without using weather, quote, or success data.

    Callers must separately propagate validate_inventory's unknown state. This
    selector never establishes inventory completeness or unlocks reserved data.
    """
    now = utc(now)
    slots = {slot["slot_utc"]: slot | {"state": "unknown", "reasons": [], "primary": None,
             "duplicates": [], "candidates": []} for slot in planned_slots(phase)}
    skipped, grouped, seen = [], {key: [] for key in slots}, set()
    for record in records:
        identity = {"run_id": record.get("id"), "run_attempt": record.get("run_attempt")}
        if record.get("event") != "schedule" or record.get("run_attempt") != 1:
            skipped.append(identity | {"reason": "not_original_scheduled_attempt"})
            continue
        try:
            created = parse_time(record.get("created_at"))
            slot = slot_at(created)
        except (ValueError, TypeError):
            slot = None
        if slot is None or slot["phase"] != phase:
            skipped.append(identity | {"reason": "creation_outside_phase_or_schedule"})
            continue
        key = slot["slot_utc"]
        started, reason = _candidate(record, now)
        pair = (identity["run_id"], identity["run_attempt"])
        if pair in seen:
            reason, started = "duplicate_run_attempt_record", None
        seen.add(pair)
        if started is not None and slot_at(started) != slot:
            skipped.append(identity | {"reason": "coordinator_start_off_schedule"})
            continue
        grouped[key].append((started, reason, record))
    for key, output in slots.items():
        candidates = grouped[key]
        output["candidates"] = [{"run_id": r["id"], "run_attempt": r["run_attempt"]} for _, _, r in candidates]
        if now < parse_time(key) + TOLERANCE:
            output.update(state="pending", reasons=["slot_window_not_closed"])
            continue
        if not candidates:
            output["reasons"] = ["missing_scheduled_invocation"]
            continue
        try:
            observed_from = [parse_time(record.get("inventory_observed_from_utc")) for _, _, record in candidates]
            _require(all(value >= parse_time(key) + TOLERANCE for value in observed_from),
                     "inventory_observed_before_slot_window_closed")
        except (ValueError, TypeError) as exc:
            output["reasons"] = [str(exc) if isinstance(exc, InventoryError) else "inventory_observation_time_missing"]
            continue
        errors = sorted({reason for _, reason, _ in candidates if reason})
        if errors:
            output["reasons"] = errors
            continue
        ordered = sorted(candidates, key=lambda row: row[0])
        if len(ordered) > 1 and ordered[0][0] == ordered[1][0]:
            output["reasons"] = ["earliest_start_tie"]
            continue
        started, _, primary = ordered[0]
        output["primary"] = primary
        output["duplicates"] = [{"run_id": r["id"], "run_attempt": r["run_attempt"]} for _, _, r in ordered[1:]]
        coordinator = primary["coordinator"]
        if coordinator.get("state") not in ("complete", "failed"):
            output["reasons"] = ["primary_coordinator_not_terminal"]
            continue
        if primary.get("status") != "completed":
            output["reasons"] = ["primary_run_nonterminal"]
            continue
        try:
            actual = parse_time(coordinator.get("actual_start_at_utc"))
            initialized = parse_time(coordinator.get("collector_manifest_started_at_utc"))
            gate = parse_time(coordinator.get("pre_request_check_at_utc"))
            finished = parse_time(coordinator.get("finished_at_utc"))
            received = parse_time(primary.get("coordinator_provenance", {}).get("received_at_utc"))
            _require(started <= actual <= initialized <= gate <= finished <= received <= now,
                     "primary_collection_timestamp_order_invalid")
            expected = {name: output[name] for name in ("target_date", "slot_utc", "phase")}
            _require(all(slot_at(value) == expected for value in (actual, initialized, gate))
                     and coordinator.get("slot") == expected and coordinator.get("creation_slot") == expected,
                     "primary_collection_off_schedule")
            output.update(state="selected", reasons=[])
        except (ValueError, TypeError, KeyError) as exc:
            output["reasons"] = [str(exc) if isinstance(exc, InventoryError) else "primary_collection_timing_missing"]
    return {"phase": phase, "planned_slots": 84, "planned_station_checkpoints": 1260,
            "slots": list(slots.values()), "skipped": skipped}
