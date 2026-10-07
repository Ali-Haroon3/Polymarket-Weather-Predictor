"""Offline evidence checks for the frozen challenger trial; never refit or fetch.

The root contains inventory.json and relative, non-symlink evidence files. Inventory
schema 1 has pages, attempts, artifact_pages, optional jobs_pages/guard_logs,
artifacts, captures, and outcomes. Response records preserve exact raw_file bytes,
HTTP identity/status, request/receipt/finish times and a hashed response_headers
file. Saved ZIP records carry the same receipt fields for the artifact /zip URL.
Jobs use the original-attempt
jobs endpoint; guard_logs use /actions/jobs/JOB_ID/logs and preserve plain UTF-8.
Hashes establish consistency of saved evidence, not server authenticity.
"""

import datetime as dt
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from urllib.parse import parse_qs, quote, urlsplit
import zipfile

UTC = dt.timezone.utc
MAX_RAW = 8 * 1024 * 1024
MAX_CAPTURE = 256 * 1024 * 1024
MAX_ZIP = 32 * 1024 * 1024
TERMINAL = {"success", "failure", "neutral", "cancelled", "skipped", "timed_out",
            "action_required", "stale", "startup_failure"}


class EvidenceError(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise EvidenceError(reason)


def timestamp(value):
    require(isinstance(value, str), "invalid_timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidenceError("invalid_timestamp") from exc
    require(parsed.tzinfo is not None and parsed.utcoffset() is not None, "naive_timestamp")
    return parsed.astimezone(UTC)


def day(value):
    require(isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value), "invalid_date")
    return dt.date.fromisoformat(value)


def capture_date(value):
    """Match the frozen policy's day prefix, accepting an explicit UTC clock."""
    require(isinstance(value, str), "invalid_capture_date")
    parsed_day = day(value[:10])
    if len(value) != 10:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        require(parsed.tzinfo is not None and parsed.utcoffset() == dt.timedelta(0)
                and parsed.date() == parsed_day, "invalid_capture_utc_timestamp")
    return str(parsed_day)


def stamp(value):
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def integer(value, minimum=1):
    return type(value) is int and value >= minimum


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate_json_key")
            result[key] = value
        return result

    def finite(value):
        parsed = float(value)
        require(math.isfinite(parsed), "nonfinite_json_number")
        return parsed

    def invalid(_):
        raise EvidenceError("nonfinite_json_number")

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                          parse_float=finite, parse_constant=invalid)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError("invalid_json") from exc


class Reader:
    def __init__(self, root, now):
        root = Path(root)
        require(not root.is_symlink(), "symlink_root")
        self.root = root.resolve(strict=True)
        require(self.root.is_dir(), "invalid_evidence_root")
        self.now = now
        self.files = {}

    def raw(self, name, limit):
        require(isinstance(name, str) and name and "\\" not in name
                and not PurePosixPath(name).is_absolute()
                and all(part not in ("", ".", "..") for part in name.split("/")),
                "unsafe_evidence_path")
        path = self.root
        for part in name.split("/"):
            path = path / part
            require(not path.is_symlink(), "symlink_evidence_path")
        require(path.resolve(strict=True).is_relative_to(self.root), "evidence_path_escape")
        require(stat.S_ISREG(path.stat(follow_symlinks=False).st_mode), "evidence_file_size_or_type")
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as handle:
            info = os.fstat(handle.fileno())
            require(stat.S_ISREG(info.st_mode) and info.st_size <= limit, "evidence_file_size_or_type")
            raw = handle.read(limit + 1)
        require(len(raw) <= limit, "evidence_file_too_large")
        self.files[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        return raw

    def file(self, entry, limit=MAX_RAW, key="path"):
        require(isinstance(entry, dict), "invalid_file_record")
        require(integer(entry.get("bytes"), 0) and entry["bytes"] <= limit
                and isinstance(entry.get("sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), "invalid_file_digest_record")
        raw = self.raw(entry.get(key), limit)
        require(len(raw) == entry["bytes"] and hashlib.sha256(raw).hexdigest() == entry["sha256"],
                "file_hash_or_length_mismatch")
        return raw

    def receipt(self, entry, path, query=None):
        require(isinstance(entry, dict) and type(entry.get("status")) is int
                and entry["status"] == 200 and entry.get("body_complete") is True
                and entry.get("error") is None, "unsuccessful_or_incomplete_response")
        url = urlsplit(entry.get("request_url", ""))
        require(url.scheme == "https" and url.netloc == "api.github.com"
                and not url.fragment and url.path == path, "response_endpoint_mismatch")
        require(parse_qs(url.query, keep_blank_values=True) == (query or {}), "response_query_mismatch")
        requested = timestamp(entry.get("requested_at_utc"))
        received = timestamp(entry.get("received_at_utc"))
        finished = timestamp(entry.get("attempt_finished_at_utc"))
        require(requested <= received <= finished <= self.now, "response_receipt_order_invalid")
        require(isinstance(entry.get("response_headers"), dict), "missing_response_headers")
        headers = self.file(entry["response_headers"])
        require(bool(headers.strip()), "empty_response_headers")
        # Preserve redirect/proxy blocks. Only compare the final saved HTTP
        # status with the response record; this is consistency, not authenticity.
        statuses = re.findall(rb"(?m)^HTTP/\d(?:\.\d)?[ \t]+(\d{3})(?:[ \t][^\r\n]*)?\r?$", headers)
        require(bool(statuses) and int(statuses[-1]) == entry["status"], "response_headers_status_mismatch")
        return requested, received

    def response(self, entry, path, query=None, text=False):
        requested, received = self.receipt(entry, path, query)
        raw = self.file(entry, key="raw_file")
        value = raw.decode("utf-8") if text else strict_json(raw)
        if "body" in entry:
            require(canonical(entry["body"]) == canonical(value), "decoded_body_mismatch")
        if not text:
            require(isinstance(value, dict), "response_body_not_object")
        return value, requested, received


def paginated(reader, entries, path, item_key, extra_query=None):
    require(isinstance(entries, list) and 1 <= len(entries) <= 10, "missing_or_excessive_pages")
    rows, total, ids, receipts = [], None, set(), []
    for index, entry in enumerate(entries, 1):
        require(type(entry.get("page")) is int and entry["page"] == index, "pages_not_contiguous")
        query = dict(extra_query or {}, per_page=["100"], page=[str(index)])
        body, requested, received = reader.response(entry, path, query)
        count, batch = body.get("total_count"), body.get(item_key)
        require(integer(count, 0) and count < 1000, "total_count_unknown_or_api_cap")
        require(total is None or total == count, "pagination_total_changed")
        total = count
        require(isinstance(batch, list) and len(batch) == min(100, max(0, count - (index - 1) * 100)),
                "page_size_mismatch")
        for row in batch:
            require(isinstance(row, dict) and integer(row.get("id")) and row["id"] not in ids,
                    "duplicate_or_invalid_paginated_identity")
            ids.add(row["id"])
            rows.append((row, requested, received))
        receipts.append(requested)
    require(len(rows) == total and len(entries) == max(1, math.ceil(total / 100)), "pagination_incomplete")
    return rows, receipts


def grouped(entries, listed, attempt=False):
    require(isinstance(entries, list) and len(entries) <= 10000, "invalid_grouped_inventory")
    result = {}
    for entry in entries:
        require(isinstance(entry, dict) and integer(entry.get("run_id"))
                and entry["run_id"] in listed, "unlisted_grouped_run")
        if attempt:
            require(type(entry.get("run_attempt")) is int and entry["run_attempt"] == 1,
                    "nonoriginal_grouped_attempt")
        result.setdefault(entry["run_id"], []).append(entry)
    return result


def validate_run(run, evidence, received, first, last, original=True):
    repo = evidence["repository"]
    require(integer(run.get("id")) and integer(run.get("run_attempt")), "invalid_run_identity")
    require(run.get("event") == "schedule" and run.get("path") == evidence["workflow_path"]
            and isinstance(run.get("repository"), dict)
            and run["repository"].get("full_name") == repo, "run_repository_workflow_mismatch")
    require(run.get("html_url") == f"https://github.com/{repo}/actions/runs/{run['id']}"
            and isinstance(run.get("head_sha"), str)
            and re.fullmatch(r"[0-9a-f]{40}", run["head_sha"]), "run_url_or_head_mismatch")
    created, updated = timestamp(run.get("created_at")), timestamp(run.get("updated_at"))
    require(first <= created.date() <= last and created <= updated <= received, "run_time_or_window_invalid")
    # Provider fields need not agree to the second. A :24 creation/:23 start is
    # accepted when both independently satisfy the frozen same-day qualification.
    if run.get("run_started_at") is not None:
        started = timestamp(run["run_started_at"])
        require(started <= updated and (not original or started.date() == created.date()), "run_start_day_invalid")
    if original:
        require(run.get("status") == "completed" and run.get("conclusion") in TERMINAL,
                "original_run_not_terminal")
    else:
        require((run.get("status") == "completed" and run.get("conclusion") in TERMINAL)
                or (run.get("status") in ("queued", "in_progress", "waiting", "requested", "pending")
                    and run.get("conclusion") is None), "invalid_listing_run_status")
    return created.date()


def jsonl_rows(raw):
    for line in io.BytesIO(raw):
        if line.strip():
            require(len(line) <= MAX_RAW, "jsonl_row_too_large")
            value = strict_json(line)
            require(isinstance(value, dict), "jsonl_row_not_object")
            yield value


def shadow_from_zip(raw, member):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        require(1 <= len(entries) <= 16, "zip_member_count_invalid")
        names, files = set(), []
        for info in entries:
            name = info.filename
            require(name not in names and "\\" not in name
                    and not PurePosixPath(name).is_absolute()
                    and all(p not in ("", ".", "..") for p in name.rstrip("/").split("/")),
                    "unsafe_or_duplicate_zip_member")
            names.add(name)
            require(not stat.S_ISLNK(info.external_attr >> 16) and not (info.flag_bits & 1),
                    "zip_symlink_or_encryption")
            require(info.file_size <= MAX_RAW, "zip_member_too_large")
            if not info.is_dir():
                files.append(info)
        require(len(files) == 1 and files[0].filename == member, "unexpected_zip_members")
        with archive.open(files[0]) as stream:
            value_raw = stream.read(MAX_RAW + 1)
        require(len(value_raw) == files[0].file_size and len(value_raw) <= MAX_RAW, "zip_member_too_large")
        value = strict_json(value_raw)
        require(isinstance(value, dict), "shadow_not_object")
        return value, hashlib.sha256(value_raw).hexdigest()


def validate_entry_quote(order, rows):
    matches = [row for row in rows if row.get("source") == "kalshi"
               and row.get("market_id") == order["ticker"] and row.get("city") == order["city"]
               and row.get("target_date") == order["target_date"]]
    require(len(matches) == 1, "missing_or_duplicate_entry_book_identity")
    quote_value = matches[0].get("best_ask" if order["side"] == "yes" else "best_bid")
    require(type(quote_value) in (int, float) and math.isfinite(quote_value) and 0 < quote_value < 1,
            "missing_or_invalid_entry_quote")
    expected = quote_value if order["side"] == "yes" else 1 - quote_value
    paid = order.get("price")
    require(type(paid) in (int, float) and math.isfinite(paid) and 0 < paid < 1
            and abs(paid - expected) <= 1e-9, "saved_price_differs_from_entry_quote")


def candidate(reader, artifact, saved, run, listed, protocol, captures):
    evidence, policy = protocol["evidence"], protocol["decision_policy"]
    run_id = run["id"]
    require(artifact.get("name") == evidence["artifact_name"].format(run_id=run_id), "artifact_name_mismatch")
    # Hosted expiration does not invalidate exact bytes archived beforehand.
    require(type(artifact.get("expired")) is bool, "artifact_expiration_unknown")
    binding = artifact.get("workflow_run")
    require(isinstance(binding, dict) and integer(binding.get("id")) and binding["id"] == run_id
            and binding.get("head_sha") == run["head_sha"], "artifact_run_binding_mismatch")
    if "repository_id" in binding:
        require(integer(binding["repository_id"]) and binding["repository_id"] == run["repository"].get("id"),
                "artifact_repository_binding_mismatch")
    _, zip_received = reader.receipt(saved,
        f"/repos/{evidence['repository']}/actions/artifacts/{artifact['id']}/zip")
    require(timestamp(artifact.get("created_at")) <= zip_received, "artifact_download_before_creation")
    if artifact["expired"] and artifact.get("expires_at") is not None:
        require(zip_received < timestamp(artifact["expires_at"]), "expired_artifact_download_after_expiration")
    raw = reader.file(saved, MAX_ZIP)
    digest = artifact.get("digest")
    if digest is not None:
        require(digest == "sha256:" + hashlib.sha256(raw).hexdigest(), "github_artifact_digest_mismatch")
    value, member_hash = shadow_from_zip(raw, evidence["artifact_member"])
    capture_day = day(value.get("latest_capture"))
    require(value.get("mode") == "shadow" and value.get("required_capture_date") == str(capture_day),
            "shadow_mode_or_required_day_mismatch")
    for key in ("policy_version", "policy_sha256", "policy_code_sha256", "specification"):
        require(canonical(value.get(key)) == canonical(policy[key]), "shadow_policy_mismatch")
    capture_hash = value.get("capture_sha256")
    require(capture_hash in captures, "missing_exact_capture_input")
    require(captures[capture_hash]["latest_capture"] == str(capture_day), "capture_input_latest_day_mismatch")
    created, updated = timestamp(artifact.get("created_at")), timestamp(artifact.get("updated_at"))
    require(created <= updated <= reader.now, "artifact_timestamp_order_invalid")
    bound = max(created, updated, timestamp(run["updated_at"]))
    require(bound.date() == capture_day and timestamp(run["created_at"]).date() == capture_day
            and timestamp(listed["created_at"]).date() == capture_day, "publication_or_creation_day_mismatch")
    families = value.get("families")
    require(isinstance(families, dict) and set(families) == set(policy["specification"]["families"]),
            "shadow_families_mismatch")
    selections = {}
    for family, result in families.items():
        require(isinstance(result, dict) and isinstance(result.get("selected"), list)
                and len(result["selected"]) <= policy["specification"]["max_orders_per_day"],
                "invalid_selected_orders")
        seen = set()
        for order in result["selected"]:
            require(isinstance(order, dict) and order.get("run_at") == str(capture_day)
                    and isinstance(order.get("ticker"), str) and order["ticker"]
                    and isinstance(order.get("city"), str) and order["city"]
                    and order.get("side") in ("yes", "no"), "invalid_selected_identity")
            target = day(order.get("target_date"))
            require(bound < dt.datetime.combine(target, dt.time(), UTC), "publication_not_before_target")
            event = (order["city"], order["target_date"])
            require(event not in seen, "duplicate_selected_event")
            seen.add(event)
        selections[family] = result["selected"]
    for order in selections[protocol["primary_family"]]:
        validate_entry_quote(order, captures[capture_hash]["latest_rows"])
    comparison = {key: value[key] for key in ("latest_capture", "required_capture_date", "capture_sha256",
                                             "policy_version", "policy_sha256", "policy_code_sha256", "specification")}
    comparison["selections"] = selections
    return {"capture_date": str(capture_day), "run_id": run_id, "artifact_id": artifact["id"],
            "publication_bound": stamp(bound), "capture_sha256": capture_hash,
            "member_sha256": member_hash, "comparison": canonical(comparison),
            "orders": selections[protocol["primary_family"]]}


def jobs_and_logs(reader, run, jobs_entries, logs_entries, repository):
    """Verify every supplied optional record, including evidence not used to skip."""
    path = f"/repos/{repository}/actions/runs/{run['id']}/attempts/1/jobs"
    jobs, _ = paginated(reader, jobs_entries, path, "jobs")
    indexed = {}
    for job, requested, received in jobs:
        require(integer(job.get("run_id")) and job["run_id"] == run["id"]
                and type(job.get("run_attempt")) is int and job["run_attempt"] == 1,
                "job_attempt_binding_mismatch")
        require(timestamp(run["updated_at"]) <= requested, "jobs_inventory_before_terminal_attempt")
        indexed[job["id"]] = job
    seen, verified_logs = set(), []
    for entry in logs_entries:
        job_id = entry.get("job_id")
        require(integer(job_id) and job_id in indexed and job_id not in seen, "guard_log_job_binding_mismatch")
        seen.add(job_id)
        job = indexed[job_id]
        text, requested, received = reader.response(entry, f"/repos/{repository}/actions/jobs/{job_id}/logs", text=True)
        require(job.get("status") == "completed" and job.get("conclusion") in TERMINAL
                and timestamp(job.get("completed_at")) <= requested, "guard_job_not_successfully_terminal")
        verified_logs.append((job, text, received))
    return indexed, verified_logs


def guard_proven(run, verified_logs, capture_day):
    if run["conclusion"] != "success":
        return False
    proved = False
    for job, text, received in verified_logs:
        if job["conclusion"] != "success":
            continue
        steps = job.get("steps")
        require(isinstance(steps, list), "guard_steps_missing")
        guard = [s for s in steps if s.get("name") == "Stand down if today is already captured"]
        shadow = [s for s in steps if s.get("name") == "Recovery challenger paper signals"]
        if not (len(guard) == len(shadow) == 1 and guard[0].get("status") == "completed"
                and guard[0].get("conclusion") == "success" and shadow[0].get("status") == "completed"
                and shadow[0].get("conclusion") == "skipped"):
            continue
        pattern = re.compile(r"^(\S+) (?:##\[notice\]|::notice::)captures for "
                             + re.escape(capture_day) + r" already committed — standing down\s*$")
        for line in text.splitlines():
            match = pattern.fullmatch(line)
            if match:
                notice = timestamp(match.group(1))
                require(notice.date().isoformat() == capture_day and notice <= received,
                        "guard_notice_timestamp_invalid")
                proved = True
    return proved


def union_outcomes(reader, payload, orders, input_publications):
    """Preserve all relevant versions; a separate settlement file cannot hide one.

    Receipt/identity validity is decided by metrics under the fixed cutoff. In
    particular, malformed relevant clocks are retained, never transformed into
    a legacy fallback. Unrelated historical receipt clocks need not be valid.
    """
    tickers = {order["ticker"] for order in orders}
    versions, sources, total_bytes = {}, {}, 0
    records = payload["captures"] + [payload["outcomes"]]
    for record in records:
        input_bound = input_publications.get(record["sha256"])
        for row in jsonl_rows(reader.file(record, MAX_CAPTURE)):
            ticker = row.get("market_id")
            if row.get("source") != "kalshi" or not isinstance(ticker, str) or ticker not in tickers:
                continue
            if input_bound is not None and (row.get("outcome") is not None or row.get("outcome_observed_at") is not None):
                try:
                    observed = timestamp(row.get("outcome_observed_at"))
                except (ValueError, TypeError, OverflowError):
                    pass  # Retain malformed clocks for the settlement validator.
                else:
                    require(observed <= input_bound, "outcome_receipt_after_input_publication")
            key = canonical(row)
            if key not in versions:
                total_bytes += len(key.encode())
                require(total_bytes <= MAX_CAPTURE, "relevant_outcome_versions_too_large")
                versions[key] = row
            digest = hashlib.sha256(key.encode()).hexdigest()
            sources.setdefault(digest, set()).add(record["path"])
    return list(versions.values()), {digest: sorted(paths) for digest, paths in sources.items()}


def load_evidence(root, protocol, now):
    """Return verified saved primary intentions and outcome rows, or unknown.

    Unknown output is diagnostic only and can never authorize a final passing
    verdict. Missing future dates are individually pending; all 60 closed dates
    and a complete inventory observed after the capture window are needed valid.
    """
    result = {"state": "unknown", "errors": [], "days": [], "orders": [],
              "outcomes": [], "provenance": {}}
    try:
        require(isinstance(now, dt.datetime) and now.tzinfo is not None and now.utcoffset() is not None,
                "now_must_be_timezone_aware")
        now = now.astimezone(UTC)
        first = day(protocol["calendar"]["capture_date_from"])
        last = day(protocol["calendar"]["capture_date_through"])
        dates = [str(first + dt.timedelta(days=i)) for i in range((last - first).days + 1)]
        require(len(dates) == protocol["calendar"]["capture_days"], "protocol_calendar_mismatch")
        result["days"] = [{"date": d, "state": "pending" if day(d) >= now.date() else "unknown"}
                          for d in dates]
        reader = Reader(root, now)
        payload = strict_json(reader.raw("inventory.json", MAX_RAW))
        require(isinstance(payload, dict) and type(payload.get("schema_version")) is int
                and payload["schema_version"] == 1, "unsupported_inventory_schema")
        evidence = protocol["evidence"]
        repo, workflow = evidence["repository"], evidence["workflow_path"]
        require(payload.get("repository") == repo and payload.get("workflow_path") == workflow,
                "inventory_repository_workflow_mismatch")
        expected_query = {"created_from": str(first), "created_through": str(last), "event": "schedule", "per_page": 100}
        require(canonical(payload.get("query")) == canonical(expected_query), "inventory_query_mismatch")
        runs, page_requests = paginated(reader, payload.get("pages"),
            f"/repos/{repo}/actions/workflows/{quote(workflow.rsplit('/', 1)[-1])}/runs", "workflow_runs",
            {"event": ["schedule"], "created": [f"{first}..{last}"]})
        listed = {}
        for run, requested, received in runs:
            validate_run(run, evidence, received, first, last, original=False)
            listed[run["id"]] = run
        attempts = grouped(payload.get("attempts"), listed, attempt=True)
        require(set(attempts) == set(listed) and all(len(v) == 1 for v in attempts.values()),
                "original_attempt_inventory_incomplete_or_duplicate")
        originals = {}
        for run_id, entries in attempts.items():
            run, requested, received = reader.response(entries[0], f"/repos/{repo}/actions/runs/{run_id}/attempts/1")
            validate_run(run, evidence, received, first, last)
            require(run["id"] == run_id and run["run_attempt"] == 1, "original_attempt_identity_mismatch")
            require(run["head_sha"] == listed[run_id]["head_sha"]
                    and timestamp(run["created_at"]).date() == timestamp(listed[run_id]["created_at"]).date(),
                    "attempt_listing_identity_or_day_mismatch")
            originals[run_id] = run
        art_pages = grouped(payload.get("artifact_pages"), listed)
        require(set(art_pages) == set(listed), "artifact_listing_inventory_incomplete")
        all_artifacts, per_run = {}, {}
        for run_id, entries in art_pages.items():
            artifacts, requests = paginated(reader, entries, f"/repos/{repo}/actions/runs/{run_id}/artifacts", "artifacts")
            require(all(timestamp(originals[run_id]["updated_at"]) <= t for t in requests),
                    "artifact_inventory_before_terminal_attempt")
            per_run[run_id] = []
            for artifact, requested, received in artifacts:
                require(artifact["id"] not in all_artifacts, "duplicate_artifact_across_runs")
                require(timestamp(artifact.get("created_at")) <= received
                        and timestamp(artifact.get("updated_at")) <= received, "artifact_time_after_receipt")
                all_artifacts[artifact["id"]] = artifact
                per_run[run_id].append(artifact)
        saved_artifacts = {}
        require(isinstance(payload.get("artifacts"), list), "missing_saved_artifact_inventory")
        for saved in payload["artifacts"]:
            require(isinstance(saved, dict) and integer(saved.get("artifact_id"))
                    and saved["artifact_id"] in all_artifacts and saved["artifact_id"] not in saved_artifacts,
                    "duplicate_or_unlisted_saved_artifact")
            saved_artifacts[saved["artifact_id"]] = saved
        captures = {}
        require(isinstance(payload.get("captures"), list) and len(payload["captures"]) <= 1000,
                "missing_capture_inventory")
        for record in payload["captures"]:
            raw = reader.file(record, MAX_CAPTURE)
            latest, latest_rows = None, []
            for row in jsonl_rows(raw):
                # Only entry identity/quotes are compared; no fit, ranking, or outcomes.
                captured = capture_date(row.get("captured_at"))
                if latest is None or captured > latest:
                    latest, latest_rows = captured, [row]
                elif captured == latest:
                    latest_rows.append(row)
            # Exact duplicate decisions may archive the same input at several paths.
            # Every copy was hash-checked; retain their file provenance, index once.
            captures.setdefault(record["sha256"], {"path": record["path"], "latest_capture": latest,
                                                    "latest_rows": latest_rows})
        # Read the required outcome record before assembling candidate evidence,
        # but combine its versions with captured histories only after selection.
        reader.file(payload.get("outcomes"), MAX_CAPTURE)
        jobs = grouped(payload.get("jobs_pages", []), listed, attempt=True)
        logs = grouped(payload.get("guard_logs", []), listed, attempt=True)
        require(set(logs) <= set(jobs), "guard_logs_without_jobs")
        verified_logs, job_ids = {}, set()
        for run_id, entries in jobs.items():
            indexed, verified_logs[run_id] = jobs_and_logs(reader, originals[run_id], entries, logs.get(run_id, []), repo)
            require(not job_ids.intersection(indexed), "duplicate_job_across_runs")
            job_ids.update(indexed)
        candidates, missing = {}, {}
        for run_id, run in originals.items():
            capture_day = timestamp(run["created_at"]).date().isoformat()
            matches = [a for a in per_run[run_id]
                       if a.get("name") == evidence["artifact_name"].format(run_id=run_id)]
            require(len(matches) <= 1, "duplicate_named_original_artifact")
            if not matches:
                missing.setdefault(capture_day, []).append(run_id)
                continue
            artifact = matches[0]
            require(artifact["id"] in saved_artifacts, "named_artifact_bytes_missing")
            item = candidate(reader, artifact, saved_artifacts[artifact["id"]], run, listed[run_id], protocol, captures)
            candidates.setdefault(capture_day, []).append(item)
        seen_events, seen_tickers = set(), set()
        for entry in result["days"]:
            capture_day = entry["date"]
            values = candidates.get(capture_day, [])
            if not values:
                if entry["state"] != "pending":
                    result["errors"].append(f"missing_decision_day:{capture_day}")
                continue
            if len({v["comparison"] for v in values}) != 1:
                entry["state"] = "unknown"
                result["errors"].append(f"conflicting_same_day_decisions:{capture_day}")
                continue
            unproved = [rid for rid in missing.get(capture_day, [])
                        if not guard_proven(originals[rid], verified_logs.get(rid, []), capture_day)]
            if unproved:
                entry["state"] = "unknown"
                result["errors"].append(f"unavailable_decision_attempt:{capture_day}:{','.join(map(str, unproved))}")
                continue
            chosen = min(values, key=lambda v: (timestamp(v["publication_bound"]), v["run_id"]))
            entry.update({k: v for k, v in chosen.items() if k not in ("comparison", "orders")})
            entry.update(state="valid", candidate_run_ids=sorted(v["run_id"] for v in values),
                         guard_skipped_run_ids=sorted(missing.get(capture_day, [])), orders=len(chosen["orders"]))
            for order in chosen["orders"]:
                event = (order["city"], order["target_date"])
                require(event not in seen_events and order["ticker"] not in seen_tickers,
                        "duplicate_primary_event_across_days")
                seen_events.add(event)
                seen_tickers.add(order["ticker"])
                result["orders"].append(dict(order, publication_bound=chosen["publication_bound"]))
        input_publications = {}
        for values in candidates.values():
            for item in values:
                digest, bound = item["capture_sha256"], timestamp(item["publication_bound"])
                input_publications[digest] = min(bound, input_publications.get(digest, bound))
        result["outcomes"], outcome_sources = union_outcomes(reader, payload, result["orders"], input_publications)
        window_end = dt.datetime.combine(last + dt.timedelta(days=1), dt.time(), UTC)
        inventory_window_closed = all(t >= window_end for t in page_requests)
        if now >= window_end and not inventory_window_closed:
            result["errors"].append("inventory_observed_before_capture_window_closed")
        result["provenance"] = {"files": reader.files, "run_count": len(listed),
                                "artifact_count": len(all_artifacts), "analysis_at_utc": stamp(now),
                                "inventory_window_closed": inventory_window_closed,
                                "coverage_complete": all(d["state"] == "valid" for d in result["days"]),
                                "outcome_version_sources": outcome_sources,
                                "input_publication_bounds": {key: stamp(value) for key, value in input_publications.items()},
                                "scope": "Consistency of preserved evidence; no server authenticity assertion."}
        if not result["errors"]:
            result["state"] = "valid"
    except (EvidenceError, OSError, ValueError, TypeError, KeyError, AttributeError,
            zipfile.BadZipFile, RuntimeError, OverflowError) as exc:
        result["errors"].append(str(exc) if isinstance(exc, EvidenceError) else f"unreadable_or_invalid_evidence:{type(exc).__name__}")
    return result
