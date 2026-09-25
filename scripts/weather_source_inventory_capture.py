#!/usr/bin/env python3
"""Preserve read-only GitHub metadata for the fixed source-availability study.

This command never dispatches workflows or downloads source/quote artifacts. An
optional local artifact root supplies coordinator status files only. Completeness
comes from weather_source_inventory.validate_inventory, not a capture assertion.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import urlencode

import weather_source_inventory as inventory

API_VERSION = "2026-03-10"
MAX_REQUESTS = 512
MAX_PAGES = 10
REQUEST_TIMEOUT = 30
MAX_BODY_BYTES = 8 * 1024 * 1024
SAFE_HEADERS = frozenset(("date", "content-type", "content-length", "content-encoding",
                         "cache-control", "etag", "last-modified", "link", "retry-after",
                         "x-github-request-id", "x-github-api-version-selected",
                         "x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset"))


def now_utc():
    return dt.datetime.now(dt.timezone.utc)


def strict_json(body):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("nonfinite_json_number")

    return json.loads(body, object_pairs_hook=unique, parse_constant=invalid)


def split_response(raw):
    """Separate gh --include headers without changing one body byte.

    gh versions/platforms can mix LF and CRLF. Only a safe response-header
    allowlist is retained; authentication, OAuth scopes and cookies are omitted.
    """
    if not isinstance(raw, bytes):
        raise ValueError("nonbinary_gh_output")
    offset = 0
    while True:
        match = re.match(br"HTTP/[^\s]+[ \t]+([0-9]{3})(?:[^\r\n]*)\r?\n", raw[offset:])
        if match is None:
            raise ValueError("missing_http_status_line")
        status = int(match.group(1))
        cursor = offset + match.end()
        headers = {}
        while True:
            end = raw.find(b"\n", cursor)
            if end == -1:
                raise ValueError("incomplete_http_headers")
            line = raw[cursor:end].removesuffix(b"\r")
            cursor = end + 1
            if not line:
                break
            if b":" not in line or line[:1] in (b" ", b"\t"):
                raise ValueError("invalid_http_header")
            name, value = line.split(b":", 1)
            name = name.decode("ascii").lower()
            if name in SAFE_HEADERS:
                value = value.decode("latin-1").strip()
                headers[name] = headers[name] + ", " + value if name in headers else value
        if 100 <= status < 200:
            offset = cursor
            continue
        return status, headers, raw[cursor:]


def listing_url(page):
    query = urlencode({"event": "schedule", "created": f"{inventory.QUERY_FROM}..{inventory.QUERY_THROUGH}",
                       "per_page": 100, "page": page})
    return (f"https://api.github.com/repos/{inventory.REPOSITORY}/actions/workflows/"
            f"{inventory.WORKFLOW_NAME}/runs?{query}")


def attempt_url(run_id, attempt):
    return f"https://api.github.com/repos/{inventory.REPOSITORY}/actions/runs/{run_id}/attempts/{attempt}"


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


class InventoryCapture:
    def __init__(self, output_dir, *, artifacts_root=None, runner=subprocess.run, clock=now_utc):
        self.output = Path(output_dir).absolute()
        self.output.mkdir(mode=0o700, parents=True, exist_ok=False)
        (self.output / "responses").mkdir()
        self.artifacts_root = Path(artifacts_root).absolute() if artifacts_root is not None else None
        self.runner, self.clock = runner, clock
        self.request_count = 0
        self.errors = []
        self.started = self.timestamp()
        self.payload = {"schema_version": 1, "repository": inventory.REPOSITORY,
                        "workflow_path": inventory.WORKFLOW_PATH,
                        "query": {"event": "schedule", "created_from": inventory.QUERY_FROM,
                                  "created_through": inventory.QUERY_THROUGH, "per_page": 100},
                        "pages": [], "attempts": [], "coordinator_statuses": [], "artifact_paths": [],
                        "artifact_observations": [], "capture": {
                            "api_version": API_VERSION, "maximum_requests": MAX_REQUESTS,
                            "maximum_pages": MAX_PAGES, "request_timeout_seconds": REQUEST_TIMEOUT,
                            "automatic_retries": False,
                            "body_semantics": "Exact JSON body bytes emitted by gh api --include; gh performs HTTP transfer decoding.",
                            "timestamp_semantics": "Client request timestamp precedes gh execution; receipt follows process completion. Local status receipt follows its byte read."}}
        self.checkpoint("running")

    def timestamp(self):
        return inventory.stamp(inventory.utc(self.clock()))

    def checkpoint(self, state, validation=None):
        atomic_json(self.output / "inventory.json", self.payload)
        status = {"schema_version": 1, "state": state, "started_at_utc": self.started,
                  "observed_at_utc": self.timestamp(), "request_count": self.request_count,
                  "errors": self.errors}
        if validation is not None:
            status.update(validation_state=validation["state"], validation_reasons=validation["reasons"])
        atomic_json(self.output / "capture-status.json", status)

    def get(self, url, label):
        if self.request_count >= MAX_REQUESTS:
            raise ValueError("request_limit_exceeded")
        self.request_count += 1
        requested = self.timestamp()
        entry = {"request_url": url, "requested_at_utc": requested, "received_at_utc": None,
                 "status": None, "headers": {}, "body": None, "body_complete": False,
                 "raw_file": None, "bytes": None, "sha256": None, "error": None}
        raw, completed, succeeded = b"", False, False
        try:
            result = self.runner(["gh", "api", "--include", "--method", "GET", "--header",
                                  "Accept: application/vnd.github+json", "--header",
                                  f"X-GitHub-Api-Version: {API_VERSION}", url],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                 timeout=REQUEST_TIMEOUT, check=False)
            raw, completed, succeeded = result.stdout, True, result.returncode == 0
            if not succeeded:
                entry["error"] = "gh_nonzero_exit"
        except subprocess.TimeoutExpired as exc:
            raw = exc.stdout or b""
            entry["error"] = "gh_timeout"
        except (OSError, KeyboardInterrupt) as exc:
            entry["error"] = "gh_interrupted" if isinstance(exc, KeyboardInterrupt) else "gh_execution_failed"
        finished = self.timestamp()
        entry["attempt_finished_at_utc"] = finished
        try:
            status, headers, body = split_response(raw)
            entry.update(status=status, headers=headers)
            # Partial bodies from a timed-out process are retained with no receipt
            # or decoded eligibility evidence. Never save the unfiltered headers.
            path = self.output / "responses" / f"{self.request_count:03d}-{label}.json"
            path.write_bytes(body)
            entry.update(raw_file=str(path.relative_to(self.output)), bytes=len(body),
                         sha256=hashlib.sha256(body).hexdigest())
            if completed:
                entry["received_at_utc"] = finished
            if len(body) > MAX_BODY_BYTES:
                raise ValueError("metadata_body_too_large")
            decoded = strict_json(body)
            if not isinstance(decoded, dict):
                raise ValueError("metadata_body_not_object")
            entry["body_complete"] = completed
            if succeeded:
                entry["body"] = decoded
            if status != 200 and entry["error"] is None:
                entry["error"] = "http_unsuccessful"
        except (ValueError, UnicodeError, TypeError):
            if entry["error"] is None:
                entry["error"] = "invalid_metadata_response"
        return entry

    def collect_status(self, run_id, attempt):
        artifact = self.artifacts_root / f"weather-source-{run_id}-{attempt}"
        run_dir = artifact / "run"
        path = run_dir / "run-status.json"
        observation = {"run_id": run_id, "run_attempt": attempt}
        # Check only these exact known paths. Never recurse into an artifact or
        # follow a symlink into source bodies or unrelated local material.
        for component in (artifact, run_dir, path):
            if component.is_symlink():
                self.payload["artifact_observations"].append(observation | {"state": "symlink_rejected"})
                return
        if not path.is_file():
            self.payload["artifact_observations"].append(observation | {"state": "missing_status"})
            return
        entry = observation | {"received_at_utc": None, "source_sha256": None, "status": None}
        try:
            with path.open("rb") as stream:
                raw = stream.read(MAX_BODY_BYTES + 1)
            entry["received_at_utc"] = self.timestamp()
            if len(raw) > MAX_BODY_BYTES:
                raise ValueError("local_status_too_large")
            entry["source_sha256"] = hashlib.sha256(raw).hexdigest()
            destination = self.output / "responses" / f"status-{run_id}-{attempt}.json"
            destination.write_bytes(raw)
            entry["raw_file"] = str(destination.relative_to(self.output))
            entry["status"] = strict_json(raw)
            if not isinstance(entry["status"], dict):
                raise ValueError("local_status_not_object")
            self.payload["artifact_paths"].append(observation | {"directory": str(run_dir)})
            self.payload["artifact_observations"].append(observation | {"state": "status_read"})
        except (OSError, ValueError, UnicodeError):
            self.payload["artifact_observations"].append(observation | {
                "state": "unreadable_status", "received_at_utc": entry["received_at_utc"],
                "source_sha256": entry["source_sha256"], "raw_file": entry.get("raw_file")})
            return
        self.payload["coordinator_statuses"].append(entry)

    def run(self):
        listed, total = {}, None
        try:
            for page in range(1, MAX_PAGES + 1):
                entry = self.get(listing_url(page), f"page-{page}") | {"page": page}
                self.payload["pages"].append(entry)
                self.checkpoint("running")
                if entry["error"] is not None:
                    raise ValueError(entry["error"])
                body = entry["body"]
                count, rows = body.get("total_count"), body.get("workflow_runs")
                if type(count) is not int or not 0 <= count < 1000:
                    raise ValueError("metadata_total_unknown_or_api_cap")
                if total is None:
                    total = count
                if count != total or not isinstance(rows, list) or len(rows) != min(100, max(0, total - (page-1)*100)):
                    raise ValueError("metadata_pagination_changed_or_incomplete")
                for row in rows:
                    if not isinstance(row, dict) or type(row.get("id")) is not int or row["id"] <= 0:
                        raise ValueError("invalid_listed_run")
                    if type(row.get("run_attempt")) is not int or not 1 <= row["run_attempt"] <= MAX_REQUESTS:
                        raise ValueError("invalid_or_excessive_attempt_count")
                    if row["id"] in listed:
                        raise ValueError("duplicate_paginated_run")
                    listed[row["id"]] = row
                if len(listed) == total:
                    break
            if len(listed) != total:
                raise ValueError("metadata_pagination_incomplete")
            expected = [(run_id, n) for run_id in sorted(listed)
                        for n in range(1, listed[run_id]["run_attempt"] + 1)]
            if self.request_count + len(expected) > MAX_REQUESTS:
                raise ValueError("request_limit_exceeded")
            for run_id, attempt in expected:
                entry = self.get(attempt_url(run_id, attempt), f"run-{run_id}-attempt-{attempt}")
                self.payload["attempts"].append(entry | {"run_id": run_id, "run_attempt": attempt})
                self.checkpoint("running")
                if entry["error"] is not None:
                    raise ValueError(entry["error"])
            if self.artifacts_root is not None:
                if self.artifacts_root.is_symlink() or not self.artifacts_root.is_dir():
                    raise ValueError("invalid_artifacts_root")
                for run_id, attempt in expected:
                    self.collect_status(run_id, attempt)
                    self.checkpoint("running")
        except (ValueError, OSError, KeyboardInterrupt) as exc:
            # Never serialize arbitrary OS/gh diagnostics: they can contain
            # credential-bearing environment details or unrelated paths.
            reason = str(exc) if isinstance(exc, ValueError) else "local_capture_failure"
            self.errors.append(reason)
        checked = inventory.validate_inventory(self.payload, self.clock())
        self.checkpoint("failed" if self.errors or checked["state"] != "valid" else "finished", checked)
        return checked, 0 if not self.errors and checked["state"] == "valid" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing paths are refused")
    parser.add_argument("--artifacts-root", type=Path, help="Previously downloaded artifacts; reads only exact coordinator status paths")
    args = parser.parse_args(argv)
    try:
        capture = InventoryCapture(args.output_dir, artifacts_root=args.artifacts_root)
    except OSError:
        parser.exit(1, "Could not create a new inventory output directory.\n")
    checked, code = capture.run()
    print(json.dumps({"inventory": str(capture.output / "inventory.json"),
                      "capture_status": str(capture.output / "capture-status.json"),
                      "validation_state": checked["state"], "validation_reasons": checked["reasons"]}, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
