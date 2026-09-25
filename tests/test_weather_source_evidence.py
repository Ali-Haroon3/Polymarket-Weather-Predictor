"""Offline evidence and causal-order checks; no endpoint is queried."""
import base64
from dataclasses import replace
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import weather_source_evidence as evidence
from weather_daily_observation import parse_daily


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "capture"
        self.path.mkdir()
        (self.path / "responses").mkdir()
        protocol = (ROOT / "reports/2026-09-25-source-collection-protocol.md").read_bytes()
        (self.path / "protocol.md").write_bytes(protocol)
        self.target = dt.date(2026, 9, 26)
        self.manifest = {"schema_version": 1, "purpose": "public_source_and_quote_availability_only",
                         "script_sha256": evidence.COLLECTOR_SHA256,
                         "target_date": self.target.isoformat(),
                         "protocol": {"file": "protocol.md", "bytes": len(protocol),
                                      "sha256": evidence.PROTOCOL_SHA256},
                         "started_at_utc": "2026-09-26T13:15:00Z",
                         "finished_at_utc": "2026-09-26T13:16:00Z",
                         "complete": False, "requests": []}
        self.add_response("climate", b'{"date":"2026-09-26","results":[]}')

    def add_response(self, role, raw, series=None, ticker=None):
        identifier = len(self.manifest["requests"]) + 1
        relative = f"responses/{identifier:03d}-{role}.body"
        (self.path / relative).write_bytes(raw)
        url = evidence.expected_url(self.target, role, series, ticker)
        record = {"id": identifier, "role": role, "original_url": url, "final_url": url,
                  "request_at_utc": "2026-09-26T13:15:01Z", "receipt_at_utc": "2026-09-26T13:15:02Z",
                  "attempt_finished_at_utc": "2026-09-26T13:15:03Z", "status": 200,
                  "body_complete": True, "json_valid": True, "error": None,
                  "raw_file": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                  "headers": {"age": "4", "content-length": str(len(raw))}}
        if series is not None:
            record["series"] = series
        if ticker is not None:
            record["ticker"] = ticker
        self.manifest["requests"].append(record)
        return record

    def load(self):
        (self.path / "manifest.json").write_text(json.dumps(self.manifest))
        return evidence.CaptureEvidence(self.path)

    def test_partial_capture_can_retain_verified_response_with_provenance(self):
        item = self.load().response("climate")
        self.assertEqual(item.payload["date"], "2026-09-26")
        self.assertEqual(item.provenance["headers"]["age"], "4")
        self.assertEqual(len(item.provenance["manifest_sha256"]), 64)
        self.assertFalse(self.manifest["complete"])

    def test_missing_or_unreadable_evidence_remains_an_evidence_error(self):
        with self.assertRaisesRegex(evidence.EvidenceError, "evidence_file_unavailable"):
            evidence.CaptureEvidence(self.path / "missing")
        with self.assertRaisesRegex(evidence.EvidenceError, "evidence_file_unavailable"):
            evidence.CaptureEvidence(self.path)
        capture = self.load()
        (self.path / self.manifest["requests"][0]["raw_file"]).unlink()
        with self.assertRaisesRegex(evidence.EvidenceError, "evidence_file_unavailable"):
            capture.response("climate")
        with patch.object(Path, "read_bytes", side_effect=PermissionError("fixture")):
            with self.assertRaisesRegex(evidence.EvidenceError, "evidence_file_unavailable"):
                evidence.CaptureEvidence(self.path)

    def test_mutated_truncated_redirected_or_failed_response_is_unknown(self):
        original = dict(self.manifest["requests"][0])
        cases = [{"sha256": "0" * 64}, {"bytes": 1}, {"bytes": True}, {"body_complete": False},
                 {"status": 500}, {"json_valid": False}, {"error": "failure"},
                 {"final_url": "https://example.com/"}, {"original_url": "https://example.com/"},
                 {"raw_file": "../outside.body"}, {"headers": {"content-length": "1"}},
                 {"receipt_at_utc": None}, {"request_at_utc": "2026-09-26T13:15:04Z"},
                 {"attempt_finished_at_utc": "2026-09-26T13:17:00Z"}]
        for change in cases:
            with self.subTest(change=change):
                self.manifest["requests"][0] = original | change
                with self.assertRaises(evidence.EvidenceError):
                    self.load().response("climate")
        self.manifest["requests"][0] = original
        (self.path / original["raw_file"]).write_bytes(b"changed")
        with self.assertRaises(evidence.EvidenceError):
            self.load().response("climate")

    def test_symlink_cannot_read_outside_capture(self):
        record = self.manifest["requests"][0]
        body = self.path / record["raw_file"]
        outside = self.path.parent / "outside.body"
        shutil.move(body, outside)
        body.symlink_to(outside)
        with self.assertRaisesRegex(evidence.EvidenceError, "outside_capture"):
            self.load().response("climate")

    def test_duplicate_response_keys_and_request_identities_fail_closed(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}'):
            with self.subTest(raw=raw), self.assertRaises(evidence.EvidenceError):
                evidence.strict_json(raw)
        original = dict(self.manifest["requests"][0])
        self.manifest["requests"].append(original.copy())
        with self.assertRaisesRegex(evidence.EvidenceError, "ambiguous_request_inventory"):
            self.load()
        self.manifest["requests"][1]["id"] = 2
        with self.assertRaisesRegex(evidence.EvidenceError, "missing_or_duplicate_response"):
            self.load().response("climate")

    def test_raw_decimal_precision_cannot_turn_fractional_or_extreme_maximum_into_weather(self):
        for number, reason in (("64.0000000000000001", "unsupported_fractional_daily_maximum"),
                               ("1e999", "unsupported_numeric_domain")):
            raw = ('{"date":"2026-09-26","totalStations":1,"officialReports":1,'
                   '"revisedReports":0,"preliminaryReports":0,"noReports":0,'
                   '"results":[{"station":{"cliId":"BOS"},"status":"official",'
                   '"data":{"stationId":"BOS","reportDate":"2026-09-26",'
                   '"isOfficial":true,"issueTime":"","maxTemp":' + number + '}}]}').encode()
            self.manifest["requests"] = []
            self.add_response("climate", raw)
            source = self.load().response("climate")
            result = parse_daily(source.payload, "2026-09-26", "BOS", fahrenheit_basis_verified=True)
            self.assertEqual((result["state"], result["reason"]), ("unknown", reason))

    def test_frozen_identity_protocol_and_terminal_time_required(self):
        original = self.manifest.copy()
        for change in ({"script_sha256": "0" * 64}, {"schema_version": True},
                       {"target_date": "2026-02-30"}, {"finished_at_utc": None},
                       {"finished_at_utc": "2026-09-26T13:14:59Z"}):
            with self.subTest(change=change):
                self.manifest = original | change
                with self.assertRaises(evidence.EvidenceError):
                    self.load()
        self.manifest = original
        (self.path / "protocol.md").write_text("changed")
        with self.assertRaisesRegex(evidence.EvidenceError, "protocol_bytes_mismatch"):
            self.load()

    def test_reserved_body_never_read_before_time_and_terminal_inventory_release(self):
        self.manifest["target_date"] = "2026-10-10"
        capture = self.load()
        for now, release in (("2026-10-24T09:29:59Z", True), ("2026-10-24T09:30:00Z", False)):
            with self.subTest(now=now, release=release), patch.object(capture, "safe_file") as read:
                with self.assertRaisesRegex(evidence.EvidenceError, "reserved_bodies_locked"):
                    capture.response("climate", now=evidence.timestamp(now), validation_released=release)
                read.assert_not_called()
        # Release enables validation, not automatic acceptance of mismatched evidence.
        with self.assertRaisesRegex(evidence.EvidenceError, "response_identity_mismatch"):
            capture.response("climate", now=evidence.RELEASE_TIME, validation_released=True)

    def test_pairing_window_is_causal_inclusive_and_same_invocation(self):
        daily = self.load().response("climate")
        markets = replace(daily, role="markets", series="KXHIGHNY",
                          receipt_at=daily.receipt_at + dt.timedelta(seconds=1))
        book = replace(markets, role="orderbook", ticker="KXHIGHNY-26SEP26-B70.5",
                       request_at=daily.receipt_at + dt.timedelta(seconds=2),
                       receipt_at=daily.receipt_at + dt.timedelta(seconds=120))
        close = "2026-09-27T05:00:00Z"
        self.assertEqual(evidence.pair_source_book(daily, markets, book, close)["state"], "eligible")
        for changed in (
            replace(book, receipt_at=book.receipt_at + dt.timedelta(microseconds=1)),
            replace(book, request_at=daily.receipt_at),
            replace(book, target_date="2026-09-27"),
            replace(book, provenance=book.provenance | {"manifest_sha256": "other"}),
        ):
            with self.subTest(book=changed):
                self.assertEqual(evidence.pair_source_book(daily, markets, changed, close)["state"], "unknown")
        self.assertEqual(evidence.pair_source_book(daily, markets, book, book.receipt_at.isoformat())["state"], "unknown")

    def test_excluded_engineering_bundle_preserves_all_122_response_identities(self):
        bundle = json.loads(gzip.decompress((ROOT / "reports/2026-09-25-source-engineering-evidence.json.gz").read_bytes()))
        self.manifest = bundle["manifest"]
        (self.path / "protocol.md").write_bytes(base64.b64decode(bundle["protocol_base64"], validate=True))
        for relative, value in bundle["raw_response_bodies_base64"].items():
            (self.path / relative).write_bytes(base64.b64decode(value, validate=True))
        capture = self.load()
        count = 0
        for row in self.manifest["requests"]:
            item = capture.response(row["role"], series=row.get("series"), ticker=row.get("ticker"))
            self.assertEqual(item.provenance["sha256"], row["sha256"])
            count += 1
        self.assertEqual(count, 122)
        self.assertLess(capture.target, dt.date(2026, 9, 26))


if __name__ == "__main__":
    unittest.main()
