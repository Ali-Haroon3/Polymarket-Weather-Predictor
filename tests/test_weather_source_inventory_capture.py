"""Offline-only exporter tests: fake gh responses, fixed clocks and temp files."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import weather_source_inventory_capture as capture
import weather_source_inventory as inventory

NOW = dt.datetime(2026, 10, 25, 12, tzinfo=dt.timezone.utc)


def row(run_id=11, attempt=1, status="completed", conclusion="failure"):
    return {"id": run_id, "run_attempt": attempt, "event": "schedule", "path": inventory.WORKFLOW_PATH,
            "repository": {"full_name": inventory.REPOSITORY}, "created_at": "2026-09-26T13:15:00Z",
            "run_started_at": "2026-09-26T13:15:01Z", "updated_at": "2026-10-24T10:00:00Z",
            "status": status, "conclusion": conclusion, "head_sha": "a" * 40,
            "html_url": f"https://github.com/{inventory.REPOSITORY}/actions/runs/{run_id}"}


def wire(body, status=200, extra=b""):
    raw = json.dumps(body, indent=1).encode() + b"\n"
    return b"HTTP/2.0 " + str(status).encode() + b" Status\r\nContent-Type: application/json\nDate: Sun, 25 Oct 2026 12:00:00 GMT\r\n" + extra + b"\n" + raw


class FakeGH:
    def __init__(self, rows=None, *, mutate=None):
        self.rows = [] if rows is None else rows
        self.calls = []
        self.mutate = mutate

    def __call__(self, command, **options):
        self.calls.append((command, options))
        url = command[-1]
        parts = urlsplit(url)
        if parts.path.endswith("/runs"):
            page = int(parse_qs(parts.query)["page"][0])
            body = {"total_count": len(self.rows), "workflow_runs": self.rows[(page-1)*100:page*100]}
        else:
            run_id, _, attempt = parts.path.rsplit("/", 3)[1:]
            body = dict(next(r for r in self.rows if r["id"] == int(run_id)), run_attempt=int(attempt))
        result = subprocess.CompletedProcess(command, 0, wire(body), None)
        return self.mutate(result, len(self.calls)) if self.mutate else result


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def run_capture(self, fake=None, *, artifacts=None):
        fake = fake or FakeGH()
        worker = capture.InventoryCapture(self.root / "result", artifacts_root=artifacts,
                                          runner=fake, clock=lambda: NOW)
        checked, code = worker.run()
        return worker, checked, code, fake

    def test_empty_query_is_exact_read_only_and_preserves_raw_body(self):
        worker, checked, code, fake = self.run_capture()
        self.assertEqual((code, checked["state"]), (0, "valid"))
        self.assertEqual(checked["provenance"]["run_count"], 0)
        command, options = fake.calls[0]
        self.assertEqual(command[:5], ["gh", "api", "--include", "--method", "GET"])
        self.assertIn("X-GitHub-Api-Version: 2026-03-10", command)
        self.assertEqual(options["timeout"], 30)
        self.assertEqual(options["stderr"], subprocess.DEVNULL)
        self.assertEqual(options["stdin"], subprocess.DEVNULL)
        self.assertFalse(options["check"])
        self.assertEqual(parse_qs(urlsplit(command[-1]).query), {
            "event": ["schedule"], "created": [f"{inventory.QUERY_FROM}..{inventory.QUERY_THROUGH}"],
            "per_page": ["100"], "page": ["1"]})
        page = worker.payload["pages"][0]
        raw = (worker.output / page["raw_file"]).read_bytes()
        self.assertEqual(raw, json.dumps({"total_count": 0, "workflow_runs": []}, indent=1).encode() + b"\n")
        self.assertEqual(hashlib.sha256(raw).hexdigest(), page["sha256"])
        self.assertEqual(len(raw), page["bytes"])
        self.assertEqual(page["requested_at_utc"], page["received_at_utc"])
        self.assertNotIn("complete", worker.payload)

    def test_mixed_header_separators_preserve_body_and_strip_secrets(self):
        body = b' {"total_count":0,"workflow_runs":[]}\r\n'
        raw = b"HTTP/2 200 OK\nContent-Type: application/json\r\nX-OAuth-Scopes: token-secret\nAuthorization: secret\r\nSet-Cookie: secret\nX-GitHub-Request-Id: request-1\r\n\n" + body
        status, headers, exact = capture.split_response(raw)
        self.assertEqual((status, exact), (200, body))
        self.assertEqual(headers, {"content-type": "application/json", "x-github-request-id": "request-1"})
        fake = FakeGH(mutate=lambda result, n: subprocess.CompletedProcess(result.args, 0, raw))
        worker, checked, code, _ = self.run_capture(fake)
        self.assertEqual(code, 0)
        self.assertNotIn("secret", (worker.output / "inventory.json").read_text())
        self.assertNotIn(b"secret", (worker.output / worker.payload["pages"][0]["raw_file"]).read_bytes())

    def test_informational_headers_do_not_become_body(self):
        raw = b"HTTP/1.1 100 Continue\r\n\r\n" + wire({"ok": True})
        status, headers, body = capture.split_response(raw)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"ok": True})

    def test_all_pages_and_original_rerun_attempts_include_failed_runs(self):
        rows = [row(i) for i in range(1, 102)]
        rows[0] = row(1, 2, "in_progress", None)
        worker, checked, code, fake = self.run_capture(FakeGH(rows))
        self.assertEqual((code, checked["state"]), (0, "valid"))
        self.assertFalse(checked["reserved_release_ready"])
        self.assertEqual((len(worker.payload["pages"]), len(worker.payload["attempts"])), (2, 102))
        self.assertEqual(len(fake.calls), 104)
        self.assertEqual([(e["run_id"], e["run_attempt"]) for e in worker.payload["attempts"][:2]], [(1, 1), (1, 2)])
        self.assertTrue(all("status=" not in command[-1] for command, _ in fake.calls))

    def test_cap_and_attempt_budget_fail_closed_before_excess_requests(self):
        for rows, expected in (([row(i) for i in range(1, 1001)], "metadata_total_unknown_or_api_cap"),
                               ([row(attempt=512)], "request_limit_exceeded")):
            with self.subTest(expected=expected):
                destination = self.root / expected
                fake = FakeGH(rows)
                worker = capture.InventoryCapture(destination, runner=fake, clock=lambda: NOW)
                checked, code = worker.run()
                self.assertEqual(code, 1)
                self.assertEqual(checked["state"], "unknown")
                self.assertEqual(worker.errors, [expected])
                self.assertEqual(len(fake.calls), 1)

    def test_changing_pagination_is_retained_and_unknown(self):
        def mutate(result, n):
            if n == 2:
                result.stdout = wire({"total_count": 100, "workflow_runs": []})
            return result
        worker, checked, code, fake = self.run_capture(FakeGH([row(i) for i in range(1, 102)], mutate=mutate))
        self.assertEqual((code, checked["state"], len(fake.calls)), (1, "unknown", 2))
        self.assertEqual(len(worker.payload["pages"]), 2)

    def test_http_error_checkpoint_retains_body_without_stderr(self):
        def mutate(result, n):
            return subprocess.CompletedProcess(result.args, 1, wire({"message": "rate limit"}, 403), b"sensitive-token")
        worker, checked, code, fake = self.run_capture(FakeGH(mutate=mutate))
        page = worker.payload["pages"][0]
        self.assertEqual((code, checked["state"], page["status"]), (1, "unknown", 403))
        self.assertEqual(json.loads((worker.output / page["raw_file"]).read_bytes()), {"message": "rate limit"})
        self.assertNotIn("sensitive", (worker.output / "inventory.json").read_text())
        self.assertEqual(json.loads((worker.output / "capture-status.json").read_text())["state"], "failed")
        self.assertEqual(len(fake.calls), 1)

    def test_timeout_keeps_partial_body_without_receipt_or_success(self):
        def timeout(command, **options):
            raise subprocess.TimeoutExpired(command, 30, output=b"HTTP/2.0 200 OK\n\n{\"total_count\":")
        worker, checked, code, _ = self.run_capture(timeout)
        page = worker.payload["pages"][0]
        self.assertEqual((code, checked["state"], page["error"]), (1, "unknown", "gh_timeout"))
        self.assertFalse(page["body_complete"])
        self.assertIsNone(page["received_at_utc"])
        self.assertIsNone(page["body"])
        self.assertEqual((worker.output / page["raw_file"]).read_bytes(), b'{"total_count":')

    def test_missing_gh_is_checkpointed_without_arbitrary_exception_text(self):
        def missing(command, **options):
            raise FileNotFoundError("secret value in arbitrary diagnostic")
        worker, checked, code, _ = self.run_capture(missing)
        self.assertEqual((code, checked["state"]), (1, "unknown"))
        self.assertEqual(worker.errors, ["gh_execution_failed"])
        self.assertNotIn("secret", (worker.output / "capture-status.json").read_text())

    def test_duplicate_json_keys_do_not_become_decoded_evidence(self):
        raw = b'HTTP/2 200 OK\n\n{"total_count":1,"total_count":0,"workflow_runs":[]}'
        fake = FakeGH(mutate=lambda result, n: subprocess.CompletedProcess(result.args, 0, raw))
        worker, checked, code, _ = self.run_capture(fake)
        self.assertEqual((code, checked["state"]), (1, "unknown"))
        self.assertIsNone(worker.payload["pages"][0]["body"])

    def test_existing_output_refused_before_any_request_or_write(self):
        destination = self.root / "result"
        destination.mkdir()
        sentinel = destination / "inventory.json"
        sentinel.write_text("preserve me")
        fake = FakeGH()
        with self.assertRaises(FileExistsError):
            capture.InventoryCapture(destination, runner=fake, clock=lambda: NOW)
        self.assertEqual(fake.calls, [])
        self.assertEqual(sentinel.read_text(), "preserve me")

    def test_local_status_reads_only_discovered_exact_path_and_preserves_hash(self):
        artifact_root = self.root / "artifacts"
        run_dir = artifact_root / "weather-source-11-1" / "run"
        run_dir.mkdir(parents=True)
        status = {"schema_version": 1, "run_id": 11, "run_attempt": 1, "state": "failed"}
        status_raw = json.dumps(status).encode() + b"\n"
        path = run_dir / "run-status.json"
        path.write_bytes(status_raw)
        (run_dir / "capture").mkdir()
        source = run_dir / "capture" / "manifest.json"
        source.write_text("must not read")
        foreign = artifact_root / "weather-source-99-1" / "run"
        foreign.mkdir(parents=True)
        (foreign / "run-status.json").write_text("must not read")
        reads = []
        original = Path.open
        def guarded(file, mode="r", *args, **kwargs):
            if "r" in mode and artifact_root in file.parents:
                reads.append(file)
                self.assertEqual(file, path)
            return original(file, mode, *args, **kwargs)
        with patch.object(Path, "open", guarded):
            worker, checked, code, _ = self.run_capture(FakeGH([row()]), artifacts=artifact_root)
        self.assertEqual((code, checked["state"]), (0, "valid"))
        self.assertEqual(reads, [path])
        saved = worker.payload["coordinator_statuses"][0]
        self.assertEqual(saved["status"], status)
        self.assertEqual(saved["source_sha256"], hashlib.sha256(status_raw).hexdigest())
        self.assertEqual(saved["received_at_utc"], inventory.stamp(NOW))
        self.assertEqual((worker.output / saved["raw_file"]).read_bytes(), status_raw)
        self.assertEqual(worker.payload["artifact_paths"], [{"run_id": 11, "run_attempt": 1, "directory": str(run_dir)}])

    def test_missing_or_symlinked_local_status_keeps_run_with_unknown_timing(self):
        artifact_root = self.root / "artifacts"
        artifact_root.mkdir()
        external = self.root / "unrelated"
        external.mkdir()
        (artifact_root / "weather-source-11-1").symlink_to(external, target_is_directory=True)
        worker, checked, code, _ = self.run_capture(FakeGH([row()]), artifacts=artifact_root)
        self.assertEqual((code, checked["state"]), (0, "valid"))
        self.assertEqual(worker.payload["artifact_observations"][0]["state"], "symlink_rejected")
        self.assertEqual(worker.payload["coordinator_statuses"], [])
        selected = inventory.select_primary(checked["records"], "development", NOW)
        self.assertEqual(selected["slots"][0]["reasons"], ["coordinator_timing_missing"])

    def test_malformed_local_status_preserves_metadata_but_marks_slot_unknown(self):
        artifacts = self.root / "artifacts"
        run_dir = artifacts / "weather-source-11-1" / "run"
        run_dir.mkdir(parents=True)
        (run_dir / "run-status.json").write_text('{"run_id":11,')
        worker, checked, code, _ = self.run_capture(FakeGH([row()]), artifacts=artifacts)
        self.assertEqual((code, checked["state"]), (0, "valid"))
        self.assertEqual(worker.payload["artifact_observations"][0]["state"], "unreadable_status")
        self.assertTrue((worker.output / "responses/status-11-1.json").exists())
        self.assertEqual(worker.payload["coordinator_statuses"], [])
        selected = inventory.select_primary(checked["records"], "development", NOW)
        self.assertEqual(selected["slots"][0]["reasons"], ["coordinator_timing_missing"])


if __name__ == "__main__":
    unittest.main()
