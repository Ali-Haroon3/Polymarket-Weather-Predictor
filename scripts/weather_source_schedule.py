#!/usr/bin/env python3
"""Run one frozen UTC source-availability checkpoint, or preserve why it was skipped.

This coordinator never chooses a replacement invocation or inspects body values.
Every invocation is retained separately; primary-slot selection belongs to analysis.
"""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

UTC = dt.timezone.utc
ROOT = Path(__file__).resolve().parents[1]
FIRST_TARGET = dt.date(2026, 9, 26)
LAST_TARGET = dt.date(2026, 10, 23)
LAST_DEVELOPMENT_TARGET = dt.date(2026, 10, 9)
TOLERANCE = dt.timedelta(minutes=15)
PROTOCOL = Path("reports/2026-09-25-source-collection-protocol.md")
COLLECTOR = Path("scripts/weather_source_capture.py")
PROTOCOL_SHA256 = "57c61d4c6495251e8b09f00897975893c4727dd25f33d16d3162e837fd2cd98c"
COLLECTOR_SHA256 = "652fb66e90caf065a3324e7d0d83ddb23a917c5e0859bbadf1704eb1a46a3ab6"
RUN_FIELDS = ("id", "run_attempt", "event", "created_at", "run_started_at", "head_sha", "html_url")


def now_utc():
    return dt.datetime.now(UTC)


def utc(value):
    if not isinstance(value, dt.datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timezone_required")
    return value.astimezone(UTC)


def stamp(value):
    return utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


def parse_time(value):
    if not isinstance(value, str):
        raise ValueError("invalid_timestamp")
    return utc(dt.datetime.fromisoformat(value.replace("Z", "+00:00")))


def slot_at(value):
    value = utc(value)
    for day in range((LAST_TARGET - FIRST_TARGET).days + 1):
        target = FIRST_TARGET + dt.timedelta(days=day)
        for offset, hour in ((0, 13), (0, 17), (0, 21), (1, 1), (1, 5), (1, 9)):
            slot = dt.datetime.combine(target + dt.timedelta(days=offset), dt.time(hour, 15), UTC)
            if abs(value - slot) <= TOLERANCE:
                return {"target_date": target.isoformat(), "slot_utc": stamp(slot),
                        "phase": "development" if target <= LAST_DEVELOPMENT_TARGET else "reserved_validation"}
    return None


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def fingerprint(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            size += len(block)
            digest.update(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def checked_metadata(path, run_id, run_attempt):
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict) or any(key not in value for key in RUN_FIELDS):
        raise ValueError("missing_run_metadata")
    if (type(value["id"]) is not int or value["id"] != run_id or run_id < 1
            or type(value["run_attempt"]) is not int
            or value["run_attempt"] != run_attempt or run_attempt < 1):
        raise ValueError("run_identity_mismatch")
    if not isinstance(value["head_sha"], str) or not re.fullmatch(r"[0-9a-f]{40}", value["head_sha"]):
        raise ValueError("invalid_head_sha")
    if not isinstance(value["html_url"], str) or not value["html_url"].startswith("https://"):
        raise ValueError("invalid_run_url")
    parse_time(value["created_at"])
    parse_time(value["run_started_at"])
    return {key: value[key] for key in RUN_FIELDS}


def frozen_files(repo_root, output):
    checks = {}
    for name, relative, expected in (("protocol", PROTOCOL, PROTOCOL_SHA256),
                                     ("collector", COLLECTOR, COLLECTOR_SHA256)):
        path = repo_root / relative
        record = {"path": str(relative), "expected_sha256": expected, "sha256": None, "matches": False}
        if path.is_file():
            record.update(fingerprint(path))
            record["matches"] = record["sha256"] == expected
            if name == "protocol":
                (output / "protocol.md").write_bytes(path.read_bytes())
        checks[name] = record
    return checks


def load_collector(path):
    spec = importlib.util.spec_from_file_location("frozen_weather_source_capture", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Collector


def evidence_summary(output):
    files = [{"file": str(path.relative_to(output)), **fingerprint(path)}
             for path in sorted((output / "capture" / "responses").glob("*.body")) if path.is_file()]
    result = {"body_files": len(files), "body_bytes": sum(item["bytes"] for item in files),
              "bodies": files, "manifest": None}
    manifest_path = output / "capture" / "manifest.json"
    if manifest_path.is_file():
        result["manifest"] = {"file": "capture/manifest.json", **fingerprint(manifest_path)}
        try:
            manifest = json.loads(manifest_path.read_bytes())
            result["manifest"].update(complete=manifest.get("complete") is True,
                                      stop_reason=manifest.get("stop_reason"),
                                      requests=len(manifest.get("requests", [])))
        except (ValueError, TypeError, AttributeError):
            result["manifest"]["parse_error"] = True
    return result


def run_schedule(metadata_path, output_dir, run_id, run_attempt, *, repo_root=ROOT,
                 now=now_utc, collector_factory=None):
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = {"schema_version": 1, "state": "started", "reason": None,
              "run_id": run_id, "run_attempt": run_attempt,
              "coordinator_started_at_utc": stamp(now()), "finished_at_utc": None,
              "actual_start_at_utc": None, "pre_request_check_at_utc": None,
              "slot": None, "inspection_scope": "operational_only",
              "primary_invocation": "not_selected; retain every invocation for earliest-start analysis",
              "runner_sha256": fingerprint(Path(__file__))["sha256"],
              "run_metadata": None, "frozen_files": {}, "evidence": None}
    status_path = output / "run-status.json"
    write_json(status_path, status)

    def skip(reason):
        status.update(state="skipped", reason=reason)

    try:
        status["frozen_files"] = frozen_files(Path(repo_root), output)
        metadata = checked_metadata(Path(metadata_path), run_id, run_attempt)
        status["run_metadata"] = metadata
        write_json(output / "run-metadata.json", metadata)
        if metadata["event"] != "schedule":
            skip("not_schedule_event")
        elif metadata["run_attempt"] != 1:
            skip("rerun_attempt")
        else:
            created = parse_time(metadata["created_at"])
            creation_slot = slot_at(created)
            status["creation_slot"] = creation_slot
            if created.year != 2026:
                skip("outside_frozen_year")
            elif creation_slot is None:
                skip("creation_off_schedule")
            elif not (created <= parse_time(metadata["run_started_at"])
                      <= parse_time(status["coordinator_started_at_utc"])):
                status.update(state="failed", reason="run_timestamp_order_invalid")
            elif not all(item["matches"] for item in status["frozen_files"].values()):
                status.update(state="failed", reason="missing_or_changed_frozen_file")
            else:
                factory = collector_factory or load_collector(Path(repo_root) / COLLECTOR)
                # Recheck after metadata, disk work and import, immediately before initialization.
                actual = utc(now())
                status["actual_start_at_utc"] = stamp(actual)
                actual_slot = slot_at(actual)
                status["slot"] = creation_slot
                if actual.year != 2026:
                    skip("outside_frozen_year")
                elif actual_slot is None:
                    skip("collector_start_off_schedule")
                elif actual_slot != creation_slot:
                    skip("creation_and_start_slots_differ")
                elif parse_time(metadata["run_started_at"]) > actual:
                    status.update(state="failed", reason="run_timestamp_order_invalid")
                else:
                    collector = factory(creation_slot["target_date"], output / "capture", output / "protocol.md")
                    # Initialization writes a manifest but issues no requests. Never let its
                    # delay turn a timely invocation into an off-slot source retrieval.
                    initialized = parse_time(collector.manifest["started_at_utc"])
                    gate_time = utc(now())
                    status["collector_manifest_started_at_utc"] = stamp(initialized)
                    status["pre_request_check_at_utc"] = stamp(gate_time)
                    if slot_at(initialized) != creation_slot or slot_at(gate_time) != creation_slot:
                        skip("initialization_missed_slot")
                    else:
                        status.update(state="collecting", reason=None)
                        write_json(status_path, status)
                        manifest = collector.run()
                        status.update(state="complete" if manifest.get("complete") is True else "failed",
                                      reason="collected" if manifest.get("complete") is True else "collector_incomplete")
    except (KeyboardInterrupt, SystemExit) as exc:
        status.update(state="failed", reason="collector_interrupted", error_type=type(exc).__name__)
    except Exception as exc:
        # Body values and exception text are not printed, especially in the reserved window.
        status.update(state="failed", reason="coordinator_or_collector_error", error_type=type(exc).__name__)
    finally:
        status["finished_at_utc"] = stamp(now())
        status["evidence"] = evidence_summary(output)
        write_json(status_path, status)
    return status, 1 if status["state"] == "failed" else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--run-attempt", required=True, type=int)
    args = parser.parse_args()
    try:
        status, code = run_schedule(args.run_metadata, args.output_dir, args.run_id, args.run_attempt)
    except (OSError, ValueError) as exc:
        print(json.dumps({"state": "failed", "reason": "runner_initialization_failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps({key: status[key] for key in ("state", "reason", "run_id", "run_attempt", "slot")}
                     | {"body_files": status["evidence"]["body_files"]}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
