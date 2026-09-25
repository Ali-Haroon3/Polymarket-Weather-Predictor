"""Offline fixtures for public raw-evidence preservation and bounded collection."""
import ast
import copy
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import urllib.error
import urllib.parse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import weather_source_capture as capture


class Clock:
    def __init__(self):
        self.seconds = 0.0

    def monotonic(self):
        return self.seconds

    def sleep(self, seconds):
        self.seconds += seconds

    def now(self):
        return (dt.datetime(2026, 9, 24, 15, tzinfo=dt.timezone.utc)
                + dt.timedelta(seconds=self.seconds)).isoformat(timespec="microseconds").replace("+00:00", "Z")


def markets(event):
    return {"markets": [{"ticker": f"{event}-{suffix}", "event_ticker": event,
                         "market_type": "binary", "rules_primary": "Exact raw rule.\n",
                         "rules_secondary": "Corrections may occur."}
                        for suffix in ["T70", "B70.5", "B72.5", "B74.5", "B76.5", "T77"]],
            "cursor": ""}


class FixtureTransport:
    def __init__(self, clock, output):
        self.clock, self.output = clock, output
        self.calls, self.checkpoints = [], []
        self.event_mutation = None
        self.first_response = None
        self.delay = 0.125

    def __call__(self, url, timeout):
        self.calls.append((url, timeout, self.clock.seconds))
        self.checkpoints.append(json.loads((self.output / "manifest.json").read_bytes()))
        self.clock.sleep(min(self.delay, timeout))
        if self.delay > timeout:
            return capture.Response(error="request wall-clock timeout")
        if len(self.calls) == 1 and self.first_response is not None:
            return self.first_response
        parsed = urllib.parse.urlsplit(url)
        query = urllib.parse.parse_qs(parsed.query)
        if parsed.path.endswith("/climate/primary"):
            body = b' {"label":"caf\xc3\xa9", "data": []}\n'
        elif parsed.path.endswith("/metar"):
            body = b'{"data": []}\n'
        elif parsed.path.endswith("/markets"):
            event = query["event_ticker"][0]
            payload = markets(event)
            if event.startswith("KXHIGHNY-") and self.event_mutation:
                self.event_mutation(payload)
            body = json.dumps(payload).encode()
        elif "/series/" in parsed.path:
            body = json.dumps({"series": {"ticker": parsed.path.rsplit("/", 1)[-1],
                                          "fee_type": "quadratic", "fee_multiplier": 1}}).encode()
        else:
            body = b'{"orderbook_fp":{"yes_dollars":[["0.15","120"]],"no_dollars":[]}}\n'
        return capture.Response(200, url, {"Date": "Thu, 24 Sep 2026 15:00:00 GMT",
                                           "Content-Type": "application/json",
                                           "Cache-Control": "no-cache", "Set-Cookie": "omit-me",
                                           "Authorization": "omit-me"}, body, True)


class FakeHTTPResponse(io.BytesIO):
    def __init__(self, body, headers=None, status=200):
        super().__init__(body)
        self.headers, self.code = headers or {}, status

    def getcode(self):
        return self.code

    def geturl(self):
        return "https://weather.com/kalshi/api/climate/primary?date=2026-09-25"


class WeatherSourceCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.protocol = self.root / "frozen-protocol.md"
        self.protocol.write_bytes(b"\xef\xbb\xbf# Availability protocol\r\nExact preserved bytes.\n")

    def collector(self, name="capture"):
        output = self.root / name
        clock = Clock()
        transport = FixtureTransport(clock, output)
        collector = capture.Collector("2026-09-25", output, self.protocol,
                                      transport=transport, monotonic=clock.monotonic,
                                      sleep=clock.sleep, now=clock.now)
        return collector, transport, clock

    def test_fixed_universe_complete_capture_preserves_bytes_and_checkpoints(self):
        tree = ast.parse((Path(capture.__file__).with_name("kalshi_archive_capture.py")).read_text())
        existing = next(ast.literal_eval(node.value) for node in tree.body
                        if isinstance(node, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "SERIES" for t in node.targets))
        self.assertEqual(capture.SERIES, existing)
        collector, transport, _ = self.collector()
        manifest = collector.run()
        self.assertTrue(manifest["complete"])
        self.assertEqual(len(transport.calls), 122)
        self.assertEqual(len(manifest["events"]), 15)
        self.assertTrue(all(len(e["book_request_ids"]) == 6 for e in manifest["events"]))
        self.assertEqual(manifest["week_start"], "2026-09-21")
        self.assertEqual((collector.output / "protocol.md").read_bytes(), self.protocol.read_bytes())
        self.assertEqual(manifest["protocol"]["sha256"], hashlib.sha256(self.protocol.read_bytes()).hexdigest())
        self.assertEqual(manifest["script_sha256"], hashlib.sha256(Path(capture.__file__).read_bytes()).hexdigest())
        first = manifest["requests"][0]
        raw = (collector.output / first["raw_file"]).read_bytes()
        self.assertEqual(raw, b' {"label":"caf\xc3\xa9", "data": []}\n')
        self.assertEqual(first["bytes"], len(raw))
        self.assertEqual(first["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(first["request_at_utc"], "2026-09-24T15:00:00.000000Z")
        self.assertEqual(first["receipt_at_utc"], "2026-09-24T15:00:00.125000Z")
        self.assertEqual(first["final_url"], first["original_url"])
        self.assertEqual(set(first["headers"]), {"date", "content-type", "cache-control"})
        self.assertTrue(all(b[2] - a[2] >= 0.5 for a, b in zip(transport.calls, transport.calls[1:])))
        self.assertTrue(all(0 < timeout <= 15 for _, timeout, _ in transport.calls))
        first_book = next(r for r in manifest["requests"] if r["role"] == "orderbook")
        self.assertLess(manifest["requests"][1]["receipt_at_utc"], first_book["request_at_utc"])
        for index, checkpoint in enumerate(transport.checkpoints):
            self.assertFalse(checkpoint["complete"])
            self.assertIsNone(checkpoint["finished_at_utc"])
            self.assertEqual(len(checkpoint["requests"]), index + 1)
            if index:
                self.assertIsNotNone(checkpoint["requests"][-2]["attempt_finished_at_utc"])
                self.assertIsNotNone(checkpoint["requests"][-2]["sha256"])
        self.assertEqual(json.loads((collector.output / "manifest.json").read_bytes()), manifest)

    def test_existing_directory_and_missing_protocol_refuse_before_transport(self):
        output = self.root / "existing"
        output.mkdir()
        marker = output / "manifest.json"
        marker.write_bytes(b"do not replace")
        transport = Mock()
        with self.assertRaises(FileExistsError):
            capture.Collector("2026-09-25", output, self.protocol, transport=transport)
        self.assertEqual(marker.read_bytes(), b"do not replace")
        fresh = self.root / "fresh"
        with self.assertRaises(FileNotFoundError):
            capture.Collector("2026-09-25", fresh, self.root / "missing", transport=transport)
        self.assertFalse(fresh.exists())
        transport.assert_not_called()
        for value in ("20260925", "2026-9-25", "2026-02-30", "2126-09-25"):
            with self.assertRaises(ValueError):
                capture.target_date(value)

    def test_http_error_retains_exact_body_and_continues_without_success(self):
        collector, transport, _ = self.collector()
        body = b"\xffupstream failure\r\n"
        transport.first_response = capture.Response(503, "https://weather.com/kalshi/api/climate/primary?date=2026-09-25",
                                                    {"Date": "observed", "Set-Cookie": "omit"}, body, True)
        manifest = collector.run()
        self.assertFalse(manifest["complete"])
        self.assertEqual(len(transport.calls), 122)
        record = manifest["requests"][0]
        self.assertEqual(record["status"], 503)
        self.assertEqual((collector.output / record["raw_file"]).read_bytes(), body)
        self.assertEqual(record["sha256"], hashlib.sha256(body).hexdigest())
        self.assertIsNotNone(record["receipt_at_utc"])
        self.assertIsNotNone(record["error"])
        self.assertFalse(record["json_valid"])

    def test_invalid_or_paginated_market_identities_never_trigger_books(self):
        mutations = {
            "missing": lambda p: p["markets"].pop(),
            "duplicate": lambda p: p["markets"].__setitem__(1, copy.deepcopy(p["markets"][0])),
            "wrong_date": lambda p: p["markets"][0].update(ticker="KXHIGHNY-26SEP24-T70"),
            "wrong_event": lambda p: p["markets"][0].update(event_ticker="KXHIGHCHI-26SEP25"),
            "pagination": lambda p: p.update(cursor="another-page"),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                collector, transport, _ = self.collector(name)
                transport.event_mutation = mutation
                manifest = collector.run()
                self.assertFalse(manifest["complete"])
                self.assertEqual(len(transport.calls), 116)
                self.assertEqual(manifest["events"][0]["identity_status"], "invalid")
                self.assertEqual(manifest["events"][0]["book_request_ids"], [])
                self.assertFalse(any("/markets/KXHIGHNY-" in url for url, _, _ in transport.calls))
                self.assertTrue(any("/series/KXHIGHNY" in url for url, _, _ in transport.calls))
                self.assertEqual(len(manifest["events"][1]["book_request_ids"]), 6)

    def test_no_response_has_no_receipt_and_partial_json_is_not_complete(self):
        for name, response in [
            ("dns", capture.Response(error="DNS lookup failed")),
            ("partial", capture.Response(200, "https://weather.com/kalshi/api/climate/primary?date=2026-09-25",
                                         {}, b"{}", False, "Content-Length mismatch")),
        ]:
            with self.subTest(name=name):
                collector, transport, _ = self.collector(name)
                transport.first_response = response
                manifest = collector.run()
                record = manifest["requests"][0]
                self.assertFalse(manifest["complete"])
                self.assertIsNotNone(record["attempt_finished_at_utc"])
                self.assertEqual(record["receipt_at_utc"] is None, name == "dns")
                self.assertEqual(record["json_valid"], name == "partial")
                self.assertFalse(record["body_complete"])

    def test_deadline_stops_requests_and_caps_last_timeout(self):
        collector, transport, clock = self.collector()
        transport.delay = 14.7
        manifest = collector.run()
        self.assertFalse(manifest["complete"])
        self.assertEqual(manifest["stop_reason"], "deadline")
        self.assertEqual(clock.seconds, 300.0)
        self.assertLess(len(transport.calls), 122)
        self.assertAlmostEqual(transport.calls[-1][1], 6.0)
        self.assertTrue(all(start < 300 for _, _, start in transport.calls))
        self.assertIsNone(manifest["requests"][-1]["receipt_at_utc"])
        self.assertIsNotNone(manifest["finished_at_utc"])

    def test_worker_preserves_http_error_and_rejects_short_or_oversized_bodies(self):
        cases = [
            ("short", FakeHTTPResponse(b"{}", {"Content-Length": "10"}), False, b"{}", 200),
            ("large", FakeHTTPResponse(b"123456789"), False, b"12345678", 200),
            ("complete", FakeHTTPResponse(b"{}", {"Content-Length": "2"}), True, b"{}", 200),
            ("http_error", urllib.error.HTTPError("https://weather.com/kalshi/api/climate/primary?date=2026-09-25",
                                                 429, "Slow down", {"Date": "observed", "Set-Cookie": "omit"},
                                                 io.BytesIO(b"error\n")), True, b"error\n", 429),
        ]
        for name, response, complete, expected_body, status in cases:
            with self.subTest(name=name):
                opener = Mock()
                if name == "http_error":
                    opener.open.side_effect = response
                else:
                    opener.open.return_value = response
                metadata, body = self.root / f"{name}.json", self.root / f"{name}.body"
                with patch.object(capture.urllib.request, "build_opener", return_value=opener), \
                        patch.object(capture, "MAX_BODY_BYTES", 8):
                    capture.http_worker("https://weather.com/kalshi/api/climate/primary?date=2026-09-25",
                                        15, metadata, body)
                saved = json.loads(metadata.read_bytes())
                self.assertEqual(saved["status"], status)
                self.assertEqual(saved["body_complete"], complete)
                self.assertEqual(body.read_bytes(), expected_body)
                self.assertNotIn("set-cookie", saved["headers"])
                opener.open.assert_called_once()
        self.assertIsNone(capture.NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://other.test"))

    def test_transport_timeout_kills_worker_but_retains_partial_response(self):
        process = Mock(returncode=-9)
        process.wait.side_effect = [subprocess.TimeoutExpired("worker", 1), None]

        def launch(command, **kwargs):
            self.assertEqual(kwargs["env"], {})
            Path(command[6]).write_text(json.dumps({"status": 200, "final_url": command[4],
                                                   "headers": {"date": "observed"}, "body_complete": False}))
            Path(command[7]).write_bytes(b"partial bytes")
            return process

        with patch.object(capture.subprocess, "Popen", side_effect=launch):
            response = capture.PublicTransport()("https://weather.com/kalshi/api/climate/primary?date=2026-09-25", 1)
        process.kill.assert_called_once()
        self.assertEqual(response.body, b"partial bytes")
        self.assertEqual(response.status, 200)
        self.assertFalse(response.body_complete)
        self.assertEqual(response.error, "request wall-clock timeout")

    def test_interruption_reaps_worker_and_checkpoints_incomplete_attempt(self):
        process = Mock(returncode=-9)
        process.wait.side_effect = [KeyboardInterrupt(), None]
        collector, _, _ = self.collector()
        collector.transport = capture.PublicTransport()
        with patch.object(capture.subprocess, "Popen", return_value=process):
            with self.assertRaises(KeyboardInterrupt):
                collector.run()
        process.kill.assert_called_once()
        self.assertEqual(process.wait.call_count, 2)
        manifest = json.loads((collector.output / "manifest.json").read_bytes())
        self.assertFalse(manifest["complete"])
        self.assertEqual(manifest["stop_reason"], "interrupted")
        self.assertEqual(len(manifest["requests"]), 1)
        self.assertEqual(manifest["requests"][0]["error"], "interrupted")
        self.assertIsNone(manifest["requests"][0]["receipt_at_utc"])
        self.assertIsNotNone(manifest["requests"][0]["attempt_finished_at_utc"])


if __name__ == "__main__":
    unittest.main()
