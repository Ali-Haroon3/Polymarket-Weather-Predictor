"""Synthetic, offline provenance and completeness tests; no actual trade inputs."""

import copy
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from urllib.parse import urlencode
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import weather_challenger_forward_evidence as evidence
import weather_challenger_forward_metrics as metrics

PROTOCOL = json.loads((Path(__file__).resolve().parents[1] / "reports" /
                      "2026-10-07-challenger-preregistration.json").read_text())
NOW = dt.datetime(2026, 12, 23, 1, tzinfo=dt.timezone.utc)
RECEIVED = "2026-12-23T00:00:01Z"
REPO = PROTOCOL["evidence"]["repository"]
PREFIX = f"/repos/{REPO}/actions"


class Fixture:
    def __init__(self, root, days=60):
        self.root, self.counter = root, 0
        self.runs, self.artifacts, self.attempts, self.art_pages = [], {}, {}, {}
        self.shadows, self.saved, self.capture_records = {}, {}, {}
        self.inventory = dict(schema_version=1, repository=REPO,
            workflow_path=PROTOCOL["evidence"]["workflow_path"],
            query=dict(created_from="2026-10-08", created_through="2026-12-06", event="schedule", per_page=100),
            pages=[], attempts=[], artifact_pages=[], artifacts=[], captures=[], jobs_pages=[], guard_logs=[])
        for n in range(days):
            self.add_run(str(dt.date(2026, 10, 8) + dt.timedelta(days=n)))
        self.inventory["outcomes"] = self.file("outcomes.jsonl", b"")
        self.listing()

    def file(self, path, raw):
        destination = self.root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        return dict(path=path, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

    def response(self, body, endpoint, query=None, **extra):
        self.counter += 1
        raw = json.dumps(body, separators=(",", ":")).encode()
        record = self.file(f"responses/{self.counter}.json", raw)
        record["raw_file"] = record.pop("path")
        headers = self.file(f"responses/{self.counter}.headers",
                            b"HTTP/2 200\r\ncontent-type: application/json\r\nx-github-request-id: synthetic\r\n\r\n")
        return dict(record, request_url="https://api.github.com" + endpoint +
                    ("?" + urlencode(query) if query else ""), status=200,
                    body_complete=True, error=None, requested_at_utc="2026-12-23T00:00:00Z",
                    received_at_utc=RECEIVED, attempt_finished_at_utc=RECEIVED,
                    response_headers=headers,
                    body=copy.deepcopy(body), **extra)

    def rewrite(self, record, body):
        saved = self.file(record["raw_file"], json.dumps(body, separators=(",", ":")).encode())
        record.update(bytes=saved["bytes"], sha256=saved["sha256"], body=copy.deepcopy(body))

    def listing(self):
        pages = []
        for offset in range(0, max(1, len(self.runs)), 100):
            page = offset // 100 + 1
            pages.append(self.response(dict(total_count=len(self.runs), workflow_runs=self.runs[offset:offset + 100]),
                PREFIX + "/workflows/daily-capture.yml/runs",
                dict(event="schedule", created="2026-10-08..2026-12-06", per_page=100, page=page), page=page))
        self.inventory["pages"] = pages

    def add_run(self, capture_day, orders=True, artifact=True):
        run_id = 10000 + len(self.runs)
        target = str(dt.date.fromisoformat(capture_day) + dt.timedelta(days=1))
        run = dict(id=run_id, run_attempt=1, event="schedule", path=PROTOCOL["evidence"]["workflow_path"],
            repository=dict(full_name=REPO, id=42), html_url=f"https://github.com/{REPO}/actions/runs/{run_id}",
            head_sha="a" * 40, created_at=capture_day + "T15:00:00Z", run_started_at=capture_day + "T15:00:00Z",
            updated_at=capture_day + "T15:03:00Z", status="completed", conclusion="success")
        self.runs.append(run)
        attempt = self.response(run, PREFIX + f"/runs/{run_id}/attempts/1", run_id=run_id, run_attempt=1)
        self.attempts[run_id] = attempt
        self.inventory["attempts"].append(attempt)
        metadata = []
        if artifact:
            if capture_day not in self.capture_records:
                captured = dict(source="kalshi", captured_at=capture_day, target_date=target, city="NYC",
                                market_id="T-" + target, outcome=None, best_ask=0.5, best_bid=0.49)
                record = self.file(f"captures/{capture_day}.jsonl", (json.dumps(captured) + "\n").encode())
                self.capture_records[capture_day] = record
                self.inventory["captures"].append(record)
            policy = PROTOCOL["decision_policy"]
            selected = [dict(run_at=capture_day, target_date=target, city="NYC", ticker="T-" + target,
                             side="yes", price=0.5, probability=0.7, edge=0.1825, contracts=28,
                             principal=14.0, fee=0.49, cost=14.49)] if orders else []
            shadow = dict(mode="shadow", latest_capture=capture_day, required_capture_date=capture_day,
                capture_sha256=self.capture_records[capture_day]["sha256"],
                families={family: dict(selected=copy.deepcopy(selected))
                          for family in policy["specification"]["families"]},
                **{k: copy.deepcopy(policy[k]) for k in ("policy_version", "policy_sha256", "policy_code_sha256", "specification")})
            self.shadows[run_id] = shadow
            artifact_id = run_id + 20000
            zip_bytes = self.zip(shadow)
            saved = self.file(f"artifacts/{artifact_id}.zip", zip_bytes)
            saved["artifact_id"] = artifact_id
            receipt = self.response({}, PREFIX + f"/artifacts/{artifact_id}/zip")
            saved.update({key: value for key, value in receipt.items()
                          if key not in ("raw_file", "bytes", "sha256", "body")})
            self.inventory["artifacts"].append(saved)
            self.saved[run_id] = saved
            meta = dict(id=artifact_id, name=f"weather-recovery-{run_id}-1", expired=False,
                        created_at=capture_day + "T15:01:00Z", updated_at=capture_day + "T15:02:00Z",
                        digest="sha256:" + saved["sha256"], workflow_run=dict(id=run_id, head_sha="a" * 40, repository_id=42))
            metadata.append(meta)
            self.artifacts[run_id] = meta
        page = self.response(dict(total_count=len(metadata), artifacts=metadata), PREFIX + f"/runs/{run_id}/artifacts",
                             dict(per_page=100, page=1), run_id=run_id, page=1)
        self.art_pages[run_id] = page
        self.inventory["artifact_pages"].append(page)
        return run_id

    @staticmethod
    def zip(shadow, extra=None):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("weather-shadow.json", json.dumps(shadow).encode())
            if extra:
                archive.writestr(*extra)
        return stream.getvalue()

    def rewrite_shadow(self, run_id, raw=None):
        raw = self.zip(self.shadows[run_id]) if raw is None else raw
        saved = self.saved[run_id]
        saved.update(self.file(saved["path"], raw))
        self.artifacts[run_id]["digest"] = "sha256:" + saved["sha256"]
        self.rewrite(self.art_pages[run_id], dict(total_count=1, artifacts=[self.artifacts[run_id]]))

    def rewrite_capture(self, capture_day, rows):
        record = self.capture_records[capture_day]
        record.update(self.file(record["path"], "\n".join(map(json.dumps, rows)).encode()))
        for rid, shadow in self.shadows.items():
            if shadow["latest_capture"] == capture_day:
                shadow["capture_sha256"] = record["sha256"]
                self.rewrite_shadow(rid)

    def guard(self, run_id, notice=None, conclusion="success"):
        capture_day = self.attempts[run_id]["body"]["created_at"][:10]
        job_id = run_id + 50000
        job = dict(id=job_id, run_id=run_id, run_attempt=1, status="completed", conclusion=conclusion,
                   started_at=capture_day + "T15:00:00Z", completed_at=capture_day + "T15:03:00Z",
                   steps=[dict(name="Stand down if today is already captured", status="completed", conclusion="success"),
                          dict(name="Recovery challenger paper signals", status="completed", conclusion="skipped")])
        self.inventory["jobs_pages"].append(self.response(dict(total_count=1, jobs=[job]),
            PREFIX + f"/runs/{run_id}/attempts/1/jobs", dict(per_page=100, page=1), run_id=run_id, run_attempt=1, page=1))
        notice = notice if notice is not None else f"{capture_day}T15:01:02.123Z ##[notice]captures for {capture_day} already committed — standing down\n"
        log = self.response({}, PREFIX + f"/jobs/{job_id}/logs", run_id=run_id, run_attempt=1, job_id=job_id)
        raw = self.file(log["raw_file"], notice.encode())
        log.update(bytes=raw["bytes"], sha256=raw["sha256"])
        log.pop("body")
        self.inventory["guard_logs"].append(log)

    def check(self, now=NOW):
        self.file("inventory.json", json.dumps(self.inventory).encode())
        return evidence.load_evidence(self.root, PROTOCOL, now)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixture = Fixture(self.root)

    def assert_unknown(self, reason):
        result = self.fixture.check()
        self.assertEqual(result["state"], "unknown", result)
        self.assertTrue(any(reason in error for error in result["errors"]), result["errors"])
        return result

    def test_complete_calendar_preserves_saved_orders_and_publication_bound(self):
        result = self.fixture.check()
        self.assertEqual(result["state"], "valid", result["errors"])
        self.assertEqual(len(result["days"]), 60)
        self.assertTrue(all(d["state"] == "valid" for d in result["days"]))
        self.assertEqual(result["days"][0]["date"], "2026-10-08")
        self.assertEqual(result["orders"][0]["publication_bound"], "2026-10-08T15:03:00Z")
        self.assertTrue(result["provenance"]["coverage_complete"])
        self.assertEqual(len(result["orders"]), 60)

    def test_valid_zero_order_artifact_is_not_a_missing_day(self):
        for rid, shadow in self.fixture.shadows.items():
            for family in shadow["families"].values():
                family["selected"] = []
            self.fixture.rewrite_shadow(rid)
        result = self.fixture.check()
        self.assertEqual(result["state"], "valid", result["errors"])
        self.assertEqual(result["orders"], [])

    def test_missing_named_zip_is_unknown_even_with_successful_run(self):
        self.fixture.inventory["artifacts"].pop(0)
        self.assert_unknown("named_artifact_bytes_missing")

    def test_missing_day_cannot_reduce_the_fixed_denominator(self):
        self.fixture.runs.pop()
        self.fixture.inventory["attempts"].pop()
        self.fixture.inventory["artifact_pages"].pop()
        self.fixture.inventory["artifacts"].pop()
        self.fixture.listing()
        result = self.assert_unknown("missing_decision_day:2026-12-06")
        self.assertEqual(result["days"][-1]["state"], "unknown")

    def test_empty_future_calendar_is_pending_with_valid_partial_metadata(self):
        other = Fixture(self.root / "future", days=0)
        for page in other.inventory["pages"]:
            for key in ("requested_at_utc", "received_at_utc", "attempt_finished_at_utc"):
                page[key] = "2026-10-07T12:00:00Z"
        result = other.check(dt.datetime(2026, 10, 7, 13, tzinfo=dt.timezone.utc))
        self.assertEqual(result["state"], "valid", result["errors"])
        self.assertTrue(all(d["state"] == "pending" for d in result["days"]))
        self.assertFalse(result["provenance"]["coverage_complete"])

    def test_raw_hash_tampering_and_decoded_mismatch_fail_closed(self):
        page = self.fixture.inventory["pages"][0]
        raw = self.root / page["raw_file"]
        raw.write_bytes(raw.read_bytes() + b" ")
        self.assert_unknown("file_hash_or_length_mismatch")
        self.fixture.listing()
        self.fixture.inventory["pages"][0]["body"]["total_count"] = 59
        self.assert_unknown("decoded_body_mismatch")

    def test_duplicate_json_keys_and_nonfinite_values_are_not_accepted(self):
        page = self.fixture.inventory["pages"][0]
        for malformed in (b'{"total_count":0,"total_count":60}', b'{"total_count":NaN}', b'{"total_count":1e999}'):
            saved = self.fixture.file(page["raw_file"], malformed)
            page.update(bytes=saved["bytes"], sha256=saved["sha256"])
            self.assertEqual(self.fixture.check()["state"], "unknown")

    def test_traversal_and_symlink_paths_are_rejected(self):
        page = self.fixture.inventory["pages"][0]
        name = page["raw_file"]
        page["raw_file"] = "../outside.json"
        self.assert_unknown("unsafe_evidence_path")
        page["raw_file"] = name
        raw = self.root / name
        original = raw.read_bytes()
        raw.unlink()
        other = self.root / "original.json"
        other.write_bytes(original)
        raw.symlink_to(other)
        self.assert_unknown("symlink_evidence_path")

    def test_nested_symlink_and_oversized_record_are_rejected(self):
        page = self.fixture.inventory["pages"][0]
        (self.root / "alias").symlink_to(self.root / "responses", target_is_directory=True)
        page["raw_file"] = "alias/1.json"
        self.assert_unknown("symlink_evidence_path")
        page["bytes"] = evidence.MAX_RAW + 1
        self.assert_unknown("invalid_file_digest_record")

    def test_exact_endpoint_query_receipt_and_complete_flags_are_required(self):
        page = self.fixture.inventory["pages"][0]
        original = copy.deepcopy(page)
        for key, value, reason in (
            ("request_url", original["request_url"].replace("api.github.com", "example.com"), "response_endpoint"),
            ("request_url", original["request_url"] + "&page=1", "response_query"),
            ("received_at_utc", "2026-12-23T00:00:00", "naive_timestamp"),
            ("received_at_utc", "2027-01-01T00:00:00Z", "response_receipt_order"),
            ("body_complete", False, "unsuccessful_or_incomplete"),
            ("error", "timeout", "unsuccessful_or_incomplete")):
            page.clear(); page.update(original); page[key] = value
            self.assert_unknown(reason)

    def test_api_cap_and_inconsistent_page_total_fail(self):
        page = self.fixture.inventory["pages"][0]
        body = copy.deepcopy(page["body"])
        body["total_count"] = 1000
        self.fixture.rewrite(page, body)
        self.assert_unknown("total_count_unknown_or_api_cap")

    def test_two_page_listing_requires_every_unique_run_and_original_attempt(self):
        for run in self.fixture.runs[:41]:
            self.fixture.add_run(run["created_at"][:10])
        self.fixture.listing()
        self.assertEqual(len(self.fixture.inventory["pages"]), 2)
        self.assertEqual(self.fixture.check()["state"], "valid")
        self.fixture.inventory["attempts"].pop()
        self.assert_unknown("original_attempt_inventory_incomplete")
        self.fixture.inventory["pages"].pop()
        self.assert_unknown("pagination_incomplete")

    def test_duplicate_run_across_pages_cannot_satisfy_count(self):
        for run in self.fixture.runs[:41]:
            self.fixture.add_run(run["created_at"][:10])
        self.fixture.listing()
        page = self.fixture.inventory["pages"][1]
        body = copy.deepcopy(page["body"])
        body["workflow_runs"] = [self.fixture.runs[0]]
        self.fixture.rewrite(page, body)
        self.assert_unknown("duplicate_or_invalid_paginated_identity")

    def test_complete_artifact_listing_required_for_every_run(self):
        self.fixture.inventory["artifact_pages"].pop()
        self.assert_unknown("artifact_listing_inventory_incomplete")

    def test_same_day_timestamp_disagreement_does_not_import_old_source_gate(self):
        attempt = self.fixture.inventory["attempts"][0]
        body = copy.deepcopy(attempt["body"])
        body["created_at"] = "2026-10-08T15:00:01Z"
        self.fixture.rewrite(attempt, body)
        self.assertEqual(self.fixture.check()["state"], "valid")

    def test_later_rerun_listing_does_not_replace_or_veto_original_decision(self):
        self.fixture.runs[0].update(run_attempt=2, status="in_progress", conclusion=None,
                                  run_started_at="2026-12-22T15:00:00Z", updated_at="2026-12-22T15:01:00Z")
        self.fixture.listing()
        result = self.fixture.check()
        self.assertEqual(result["state"], "valid", result["errors"])
        self.assertEqual(result["orders"][0]["publication_bound"], "2026-10-08T15:03:00Z")

    def test_nonoriginal_attempt_wrong_identity_and_nonterminal_fail(self):
        attempt = self.fixture.inventory["attempts"][0]
        original = copy.deepcopy(attempt["body"])
        for key, value, reason in (("run_attempt", 2, "original_attempt_identity"),
                                   ("event", "workflow_dispatch", "run_repository_workflow"),
                                   ("status", "in_progress", "original_run_not_terminal")):
            body = dict(original, **{key: value})
            self.fixture.rewrite(attempt, body)
            self.assert_unknown(reason)

    def test_late_artifact_or_past_target_cannot_be_prospective(self):
        rid = self.fixture.runs[0]["id"]
        self.fixture.artifacts[rid]["updated_at"] = "2026-10-09T00:00:00Z"
        self.fixture.rewrite_shadow(rid)
        self.assert_unknown("publication_or_creation_day_mismatch")
        self.fixture.artifacts[rid]["updated_at"] = "2026-10-08T15:02:00Z"
        self.fixture.shadows[rid]["families"]["scale_only"]["selected"][0]["target_date"] = "2026-10-08"
        self.fixture.rewrite_shadow(rid)
        self.assert_unknown("publication_not_before_target")

    def test_policy_input_linkage_and_github_digest_are_required(self):
        rid = self.fixture.runs[0]["id"]
        self.fixture.shadows[rid]["policy_sha256"] = "0" * 64
        self.fixture.rewrite_shadow(rid)
        self.assert_unknown("shadow_policy_mismatch")
        self.fixture.shadows[rid]["policy_sha256"] = PROTOCOL["decision_policy"]["policy_sha256"]
        self.fixture.rewrite_shadow(rid)
        self.fixture.inventory["captures"].pop(0)
        self.assert_unknown("missing_exact_capture_input")
        self.fixture.inventory["captures"].insert(0, self.fixture.capture_records["2026-10-08"])
        self.fixture.artifacts[rid]["digest"] = "sha256:" + "0" * 64
        self.fixture.rewrite(self.fixture.art_pages[rid], dict(total_count=1, artifacts=[self.fixture.artifacts[rid]]))
        self.assert_unknown("github_artifact_digest_mismatch")

    def test_zip_extra_members_and_traversal_are_rejected_without_extraction(self):
        rid = self.fixture.runs[0]["id"]
        for name, reason in (("extra.json", "unexpected_zip_members"), ("../escape", "unsafe_or_duplicate_zip_member")):
            self.fixture.rewrite_shadow(rid, self.fixture.zip(self.fixture.shadows[rid], (name, "bad")))
            self.assert_unknown(reason)
        self.assertFalse((self.root.parent / "escape").exists())

    def test_exact_duplicate_decisions_count_once_and_conflicts_are_unknown(self):
        rid = self.fixture.add_run("2026-10-08")
        self.fixture.listing()
        result = self.fixture.check()
        self.assertEqual(result["state"], "valid", result["errors"])
        self.assertEqual(len(result["orders"]), 60)
        self.assertEqual(result["days"][0]["run_id"], 10000)
        # Even a nonprimary-family conflict invalidates the day under the frozen rule.
        self.fixture.shadows[rid]["families"]["weather_blend"]["selected"] = []
        self.fixture.rewrite_shadow(rid)
        self.assert_unknown("conflicting_same_day_decisions")

    def test_missing_earlier_decision_is_not_replaced_by_later_artifact(self):
        rid = self.fixture.add_run("2026-10-08", artifact=False)
        self.fixture.listing()
        self.assert_unknown("unavailable_decision_attempt:2026-10-08")

    def test_exact_emitted_guard_notice_proves_harmless_backup_skip(self):
        rid = self.fixture.add_run("2026-10-08", artifact=False)
        self.fixture.guard(rid)
        self.fixture.listing()
        result = self.fixture.check()
        self.assertEqual(result["state"], "valid", result["errors"])
        self.assertEqual(result["days"][0]["guard_skipped_run_ids"], [rid])

    def test_preview_notice_wrong_day_or_cancelled_job_cannot_prove_guard(self):
        rid = self.fixture.add_run("2026-10-08", artifact=False)
        self.fixture.listing()
        self.fixture.guard(rid, notice='2026-10-08T15:01:02Z echo "::notice::captures for $today already committed — standing down"\n')
        self.assert_unknown("unavailable_decision_attempt")
        self.fixture.inventory["guard_logs"] = []
        self.fixture.inventory["jobs_pages"] = []
        self.fixture.guard(rid, conclusion="cancelled")
        self.assert_unknown("unavailable_decision_attempt")

    def test_expired_hosted_artifact_keeps_verifiable_archived_bytes(self):
        rid = self.fixture.runs[0]["id"]
        self.fixture.artifacts[rid]["expired"] = True
        self.fixture.rewrite_shadow(rid)
        self.assertEqual(self.fixture.check()["state"], "valid")

    def test_stale_input_cannot_be_relabelled_as_fresh_shadow(self):
        rid = self.fixture.runs[0]["id"]
        record = self.fixture.capture_records["2026-10-08"]
        stale = dict(source="kalshi", captured_at="2026-10-07")
        record.update(self.fixture.file(record["path"], json.dumps(stale).encode()))
        self.fixture.shadows[rid]["capture_sha256"] = record["sha256"]
        self.fixture.rewrite_shadow(rid)
        self.assert_unknown("capture_input_latest_day_mismatch")

    def test_preclose_inventory_cannot_establish_final_completeness(self):
        for page in self.fixture.inventory["pages"]:
            for key in ("requested_at_utc", "received_at_utc", "attempt_finished_at_utc"):
                page[key] = "2026-12-06T23:59:00Z"
        self.assert_unknown("inventory_observed_before_capture_window_closed")

    def test_optional_job_records_are_verified_even_when_not_needed(self):
        rid = self.fixture.runs[0]["id"]
        self.fixture.guard(rid)
        entry = self.fixture.inventory["guard_logs"][0]
        (self.root / entry["raw_file"]).write_text("tampered log")
        self.assert_unknown("file_hash_or_length_mismatch")

    def test_guard_notice_from_an_unbound_job_is_rejected(self):
        rid = self.fixture.add_run("2026-10-08", artifact=False)
        self.fixture.guard(rid)
        self.fixture.listing()
        self.fixture.inventory["guard_logs"][0]["job_id"] += 1
        self.assert_unknown("guard_log_job_binding_mismatch")

    def test_nonregular_file_does_not_block_reader(self):
        self.fixture.inventory["outcomes"]["path"] = "pipe"
        os.mkfifo(self.root / "pipe")
        self.assert_unknown("evidence_file_size_or_type")

    def test_primary_event_cannot_be_reentered_on_another_capture_day(self):
        for rid in (10000, 10001):
            selected = self.fixture.shadows[rid]["families"]["scale_only"]["selected"][0]
            selected.update(ticker="SHARED", target_date="2026-10-10")
            capture_day = self.fixture.shadows[rid]["latest_capture"]
            record = self.fixture.capture_records[capture_day]
            row = json.loads((self.root / record["path"]).read_bytes())
            row.update(market_id="SHARED", target_date="2026-10-10")
            self.fixture.rewrite_capture(capture_day, [row])
            self.fixture.rewrite_shadow(rid)
        self.assert_unknown("duplicate_primary_event_across_days")

    def test_paid_yes_price_must_match_archived_ask_not_midpoint(self):
        rid = 10000
        self.fixture.shadows[rid]["families"]["scale_only"]["selected"][0]["price"] = 0.495
        self.fixture.rewrite_shadow(rid)
        self.assert_unknown("saved_price_differs_from_entry_quote")

    def test_paid_no_price_uses_bid_complement(self):
        rid = 10000
        order = self.fixture.shadows[rid]["families"]["scale_only"]["selected"][0]
        order.update(side="no", price=0.51)
        self.fixture.rewrite_shadow(rid)
        self.assertEqual(self.fixture.check()["state"], "valid")
        order["price"] = 0.49
        self.fixture.rewrite_shadow(rid)
        self.assert_unknown("saved_price_differs_from_entry_quote")

    def test_missing_quote_wrong_identity_or_duplicate_book_rows_fail(self):
        record = self.fixture.capture_records["2026-10-08"]
        original = json.loads((self.root / record["path"]).read_bytes())
        for updates, reason in ((dict(best_ask=None), "missing_or_invalid_entry_quote"),
                                (dict(best_ask=True), "missing_or_invalid_entry_quote"),
                                (dict(city="Boston"), "missing_or_duplicate_entry_book_identity"),
                                (dict(source="polymarket"), "missing_or_duplicate_entry_book_identity")):
            self.fixture.rewrite_capture("2026-10-08", [dict(original, **updates)])
            self.assert_unknown(reason)
        self.fixture.rewrite_capture("2026-10-08", [original, original])
        self.assert_unknown("missing_or_duplicate_entry_book_identity")

    def test_capture_date_or_explicit_utc_timestamp_link_to_same_day(self):
        record = self.fixture.capture_records["2026-10-08"]
        original = json.loads((self.root / record["path"]).read_bytes())
        self.fixture.rewrite_capture("2026-10-08", [dict(original, captured_at="2026-10-08T15:00:00Z")])
        self.assertEqual(self.fixture.check()["state"], "valid")
        self.fixture.rewrite_capture("2026-10-08", [dict(original, captured_at="2026-10-08T15:00:00")])
        self.assert_unknown("invalid_capture_utc_timestamp")

    def test_identical_archived_input_copies_are_verified_and_retained(self):
        record = self.fixture.capture_records["2026-10-08"]
        duplicate = self.fixture.file("captures/duplicate.jsonl", (self.root / record["path"]).read_bytes())
        self.fixture.inventory["captures"].append(duplicate)
        result = self.fixture.check()
        self.assertEqual(result["state"], "valid", result["errors"])
        self.assertIn("captures/duplicate.jsonl", result["provenance"]["files"])

    def historical_version(self, **updates):
        row = dict(source="kalshi", captured_at="2026-10-08", target_date="2026-10-09",
                   city="NYC", market_id="T-2026-10-09", outcome=0,
                   outcome_observed_at="2026-10-09T16:00:00Z")
        row.update(updates)
        return row

    def add_history(self, capture_day, historical):
        record = self.fixture.capture_records[capture_day]
        rows = [json.loads(line) for line in (self.root / record["path"]).read_bytes().splitlines()]
        self.fixture.rewrite_capture(capture_day, rows + [historical])

    def test_archived_losing_outcome_cannot_be_hidden_by_winning_settlement_file(self):
        losing = self.historical_version()
        winning = dict(losing, outcome=1)
        self.add_history("2026-10-10", losing)
        self.fixture.inventory["outcomes"] = self.fixture.file("outcomes.jsonl", json.dumps(winning).encode())
        result = self.fixture.check()
        self.assertEqual(result["state"], "valid", result["errors"])
        self.assertIn(losing, result["outcomes"])
        self.assertIn(winning, result["outcomes"])
        assessment = metrics.assess(PROTOCOL, result["orders"], result["outcomes"], NOW)
        self.assertEqual(assessment["verdict"], "inconclusive")
        self.assertIn("conflicting pre-cutoff binary outcomes", assessment["order_results"][0]["reasons"])

    def test_relevant_malformed_receipt_and_identity_survive_union_and_block_metrics(self):
        winner = self.historical_version(outcome=1)
        self.fixture.inventory["outcomes"] = self.fixture.file("outcomes.jsonl", json.dumps(winner).encode())
        record = self.fixture.capture_records["2026-10-10"]
        current = json.loads((self.root / record["path"]).read_bytes())
        for changes, reason in (
            (dict(outcome_observed_at=None), "outcome has missing or invalid observation timestamp"),
            (dict(outcome_observed_at="invalid"), "outcome has missing or invalid observation timestamp"),
            (dict(city="Boston"), "outcome target/city identity conflicts with saved order"),
            (dict(target_date="2026-10-10"), "outcome target/city identity conflicts with saved order")):
            bad = self.historical_version(**changes)
            self.fixture.rewrite_capture("2026-10-10", [current, bad])
            result = self.fixture.check()
            self.assertEqual(result["state"], "valid", result["errors"])
            self.assertIn(bad, result["outcomes"])
            assessment = metrics.assess(PROTOCOL, result["orders"], result["outcomes"], NOW)
            self.assertEqual(assessment["verdict"], "inconclusive")
            self.assertIn(reason, assessment["order_results"][0]["reasons"])

    def test_exact_versions_deduplicate_but_all_source_paths_are_preserved(self):
        row = self.historical_version(outcome=1)
        self.add_history("2026-10-10", row)
        self.add_history("2026-10-11", row)
        self.fixture.inventory["outcomes"] = self.fixture.file("outcomes.jsonl", json.dumps(row).encode())
        result = self.fixture.check()
        self.assertEqual(result["state"], "valid", result["errors"])
        self.assertEqual(result["outcomes"].count(row), 1)
        digest = hashlib.sha256(evidence.canonical(row).encode()).hexdigest()
        self.assertEqual(result["provenance"]["outcome_version_sources"][digest],
                         ["captures/2026-10-10.jsonl", "captures/2026-10-11.jsonl", "outcomes.jsonl"])

    def test_unrelated_legacy_missing_clock_does_not_poison_relevant_union(self):
        unrelated = self.historical_version(market_id="OLD-UNSELECTED", outcome_observed_at="bad legacy clock")
        self.add_history("2026-10-10", unrelated)
        result = self.fixture.check()
        self.assertEqual(result["state"], "valid", result["errors"])
        self.assertNotIn(unrelated, result["outcomes"])
        self.assertTrue(any(row.get("outcome") is None for row in result["outcomes"]))

    def test_genuine_standalone_postcutoff_revision_does_not_replace_primary(self):
        winner = self.historical_version(outcome=1)
        late_loser = self.historical_version(outcome_observed_at="2026-12-22T00:00:00Z")
        self.add_history("2026-10-10", winner)
        self.fixture.inventory["outcomes"] = self.fixture.file("outcomes.jsonl", json.dumps(late_loser).encode())
        result = self.fixture.check()
        assessment = metrics.assess(PROTOCOL, result["orders"], result["outcomes"], NOW)
        self.assertEqual(assessment["order_results"][0]["outcome"], 1)
        self.assertEqual(assessment["order_results"][0]["reasons"], [])
        self.assertTrue(assessment["order_results"][0]["post_cutoff_evidence"])

    def test_input_loser_cannot_be_hidden_by_impossible_postcutoff_receipt(self):
        winner = self.historical_version(outcome=1)
        self.add_history("2026-10-10", self.historical_version(outcome_observed_at="2026-12-22T00:00:00Z"))
        self.fixture.inventory["outcomes"] = self.fixture.file("outcomes.jsonl", json.dumps(winner).encode())
        self.assert_unknown("outcome_receipt_after_input_publication")

    def test_raw_response_headers_are_mandatory_and_hash_verified(self):
        page = self.fixture.inventory["pages"][0]
        headers = page.pop("response_headers")
        self.assert_unknown("missing_response_headers")
        page["response_headers"] = headers
        (self.root / headers["path"]).write_bytes(b"HTTP/2 500\r\n\r\n")
        self.assert_unknown("file_hash_or_length_mismatch")
        page["response_headers"] = self.fixture.file("empty.headers", b" \r\n")
        self.assert_unknown("empty_response_headers")

    def test_zip_download_receipt_endpoint_and_clock_are_required(self):
        saved = self.fixture.saved[10000]
        url = saved.pop("request_url")
        self.assert_unknown("response_endpoint_mismatch")
        saved["request_url"] = url.replace("/30000/zip", "/30001/zip")
        self.assert_unknown("response_endpoint_mismatch")
        saved["request_url"] = url
        for key in ("requested_at_utc", "received_at_utc", "attempt_finished_at_utc"):
            saved[key] = "2026-10-08T14:00:00Z"
        self.assert_unknown("artifact_download_before_creation")

    def test_final_header_status_must_agree_and_redirect_chain_is_preserved(self):
        page = self.fixture.inventory["pages"][0]
        page["response_headers"] = self.fixture.file("wrong-status.headers", b"HTTP/2 404 Not Found\r\n\r\n")
        self.assert_unknown("response_headers_status_mismatch")
        page["response_headers"] = self.fixture.file("redirect.headers",
            b"HTTP/1.1 200 Connection established\r\n\r\nHTTP/2 302 Found\r\nlocation: https://example.test/\r\n\r\nHTTP/2 200 OK\r\n\r\n")
        self.assertEqual(self.fixture.check()["state"], "valid")

    def test_expired_zip_receipt_must_precede_known_expiration(self):
        rid = 10000
        self.fixture.artifacts[rid].update(expired=True, expires_at="2026-12-20T00:00:00Z")
        self.fixture.rewrite_shadow(rid)
        self.assert_unknown("expired_artifact_download_after_expiration")
        saved = self.fixture.saved[rid]
        for key in ("requested_at_utc", "received_at_utc", "attempt_finished_at_utc"):
            saved[key] = "2026-12-19T12:00:00Z"
        self.assertEqual(self.fixture.check()["state"], "valid")

    def test_outcome_bytes_are_verified_and_relevant_rows_pass_through(self):
        rows = [dict(source="kalshi", market_id="T-2026-10-09", outcome=1, outcome_observed_at="2026-10-09T16:00:00Z"),
                dict(source="kalshi", market_id="T-2026-10-09", outcome=0, outcome_observed_at="2026-10-10T16:00:00Z")]
        self.fixture.inventory["outcomes"] = self.fixture.file("outcomes.jsonl", "\n".join(map(json.dumps, rows)).encode())
        result = self.fixture.check()
        self.assertEqual(result["state"], "valid", result["errors"])
        for row in rows:
            self.assertIn(row, result["outcomes"])  # Metrics, not provenance, resolves conflicts.
        (self.root / "outcomes.jsonl").write_text("{}")
        self.assert_unknown("file_hash_or_length_mismatch")

    def test_naive_now_fails_closed(self):
        result = self.fixture.check(NOW.replace(tzinfo=None))
        self.assertEqual(result["state"], "unknown")
        self.assertIn("now_must_be_timezone_aware", result["errors"])


if __name__ == "__main__":
    unittest.main()
