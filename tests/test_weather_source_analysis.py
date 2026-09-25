"""Offline composition checks using excluded market schemas and synthetic observations.

No source values or books from the prospective development/reserved windows are
read. Market/series schemas come only from the excluded engineering artifact.
"""
import base64
import copy
from dataclasses import replace
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import weather_source_analysis as analysis
from weather_source_evidence import EvidenceError, ResponseEvidence

UTC = dt.timezone.utc
TARGET = dt.date(2026, 9, 24)
NOW = dt.datetime(2026, 9, 25, 23, tzinfo=UTC)
AUTHORITY = ROOT / "reports/2026-09-25-source-authority-evidence.json.gz"


class SyntheticCapture:
    """Stand in for independently verified transport while preserving pairing IDs."""
    target = TARGET
    manifest_sha256 = "a" * 64

    def __init__(self, events, series):
        self.events = copy.deepcopy(events)
        self.series = copy.deepcopy(series)
        self.calls = []
        self.errors = {}
        self.books = {}
        self.book_times = {}
        self.daily = {"date": TARGET.isoformat(), "totalStations": len(analysis.STATIONS),
                      "officialReports": 0, "revisedReports": 0, "preliminaryReports": 0,
                      "noReports": len(analysis.STATIONS), "results": [
                          {"station": {"cliId": station[1]}, "status": "no_report", "data": None}
                          for station in analysis.STATIONS.values()]}
        for event in self.events.values():
            for market in event["markets"]:
                market.update(status="active", open_time="2026-09-23T12:00:00Z",
                              result="", settlement_ts=None, settlement_time=None)

    def official(self, series, maximum=72):
        cli = analysis.STATIONS[series][1]
        for row in self.daily["results"]:
            if row["station"]["cliId"] == cli:
                if row["status"] != "official":
                    self.daily["officialReports"] += 1
                    self.daily["noReports"] -= 1
                row.update(status="official", data={"stationId": cli,
                           "reportDate": TARGET.isoformat(), "isOfficial": True,
                           "issueTime": "", "maxTemp": maximum})

    def response(self, role, *, series=None, ticker=None, **_):
        key = (role, series, ticker)
        self.calls.append(key)
        if key in self.errors:
            raise EvidenceError(self.errors[key])
        if role == "climate":
            payload, request, receipt = self.daily, "18:00:00", "18:00:01"
        elif role == "markets":
            payload, request, receipt = self.events[series], "18:00:02", "18:00:03"
        elif role == "series":
            payload, request, receipt = self.series[series], "18:00:03", "18:00:04"
        elif role == "orderbook":
            payload = self.books.get(ticker, {"orderbook_fp": {"yes_dollars": [], "no_dollars": []}})
            request, receipt = self.book_times.get(ticker, ("18:00:04", "18:00:05"))
        else:
            raise AssertionError("unexpected response role")
        def at(value):
            return dt.datetime.fromisoformat(f"2026-09-24T{value}+00:00")
        return ResponseEvidence(TARGET.isoformat(), role, series, ticker, at(request), at(receipt),
                                payload, {"manifest_sha256": self.manifest_sha256, "headers": {},
                                          "sha256": "b" * 64, "raw_file": "synthetic"})


class SourceAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = gzip.decompress((ROOT / "reports/2026-09-25-source-engineering-evidence.json.gz").read_bytes())
        assert hashlib.sha256(raw).hexdigest() == "cdc77bef1ce57629f192a1e4b1979d43b11058f54de165d2b379b0182c45e2cb"
        bundle = json.loads(raw)
        cls.events, cls.series = {}, {}
        for request in bundle["manifest"]["requests"]:
            if request["role"] not in ("markets", "series"):
                continue
            body = base64.b64decode(bundle["raw_response_bodies_base64"][request["raw_file"]], validate=True)
            assert len(body) == request["bytes"]
            assert hashlib.sha256(body).hexdigest() == request["sha256"]
            destination = cls.events if request["role"] == "markets" else cls.series
            destination[request["series"]] = json.loads(body)

    def setUp(self):
        self.capture = SyntheticCapture(self.events, self.series)
        self.ny = self.capture.events["KXHIGHNY"]["markets"]

    def run_analysis(self, *, unit_verified=True, capture=None, **kwargs):
        with patch.object(analysis, "CaptureEvidence", return_value=capture or self.capture), \
                patch.object(analysis, "unit_basis", return_value={
                    "state": "verified" if unit_verified else "unknown"}):
            return analysis.analyze_capture("synthetic", expected_target=TARGET.isoformat(), now=NOW, **kwargs)

    @staticmethod
    def station(result, series="KXHIGHNY"):
        return next(row for row in result["stations"] if row["series"] == series)

    @staticmethod
    def book(*, yes=(), no=()):
        return {"orderbook_fp": {"yes_dollars": list(yes), "no_dollars": list(no)}}

    def test_explicit_no_report_and_complete_books_prove_snapshot_absence_without_units(self):
        result = self.run_analysis(unit_verified=False)
        self.assertEqual(len(result["stations"]), 15)
        self.assertTrue(all(row["state"] == "absence" for row in result["stations"]))
        self.assertTrue(all(row["complete_components"] for row in result["stations"]))
        self.assertTrue(result["snapshot_only"])
        self.assertFalse(result["payout_comparison_available"])

    def test_missing_corrupt_and_schema_unknown_source_do_not_become_no_report(self):
        for reason in ("missing_or_duplicate_response", "response_bytes_mismatch"):
            with self.subTest(reason=reason):
                self.capture.errors[("climate", None, None)] = reason
                result = self.run_analysis()
                self.assertTrue(all(row["state"] == "unknown" for row in result["stations"]))
                self.assertEqual(self.station(result)["reason"], reason)
        self.capture.errors.clear()
        self.capture.daily["results"][0]["status"] = "unrecognized"
        result = self.run_analysis()
        self.assertTrue(all(row["state"] == "unknown" for row in result["stations"]))
        self.assertEqual(self.station(result)["source"]["reason"], "unsupported_daily_status")

    def test_no_report_with_one_missing_book_stays_unknown(self):
        ticker = self.ny[0]["ticker"]
        self.capture.errors[("orderbook", "KXHIGHNY", ticker)] = "missing_or_duplicate_response"
        station = self.station(self.run_analysis())
        self.assertEqual(station["state"], "unknown")
        self.assertFalse(station["complete_components"])
        self.assertEqual(sum(row["state"] == "absence" for row in station["markets"]), 5)
        self.assertEqual(next(row for row in station["markets"] if row["ticker"] == ticker)["reason"],
                         "missing_or_duplicate_response")

    def test_no_report_with_malformed_book_stays_unknown(self):
        self.capture.books[self.ny[0]["ticker"]] = {"orderbook_fp": {"yes_dollars": []}}
        station = self.station(self.run_analysis())
        self.assertEqual(station["state"], "unknown")
        self.assertFalse(station["complete_components"])
        self.assertIn("unsupported_orderbook", [row["reason"] for row in station["markets"]])

    def test_official_values_require_preserved_fahrenheit_basis(self):
        self.capture.official("KXHIGHNY")
        station = self.station(self.run_analysis(unit_verified=False))
        self.assertEqual(station["state"], "unknown")
        self.assertEqual(station["reason"], "fahrenheit_basis_unverified")
        self.assertFalse(any(role == "orderbook" and series == "KXHIGHNY"
                             for role, series, _ in self.capture.calls))

    def test_official_with_empty_valid_books_is_absence_not_overlap(self):
        self.capture.official("KXHIGHNY")
        station = self.station(self.run_analysis())
        self.assertEqual(station["state"], "absence")
        self.assertTrue(station["complete_components"])
        self.assertTrue(all(row["reason"] == "insufficient_displayed_depth" for row in station["markets"]))

    def test_one_verified_overlap_survives_another_failed_book_without_complete_claim(self):
        self.capture.official("KXHIGHNY", 72)
        winner = next(row for row in self.ny if row.get("floor_strike") == 72)
        self.capture.books[winner["ticker"]] = self.book(no=[["0.70", "1.00"]])
        failed = next(row for row in self.ny if row["ticker"] != winner["ticker"])
        self.capture.errors[("orderbook", "KXHIGHNY", failed["ticker"])] = "response_bytes_mismatch"
        station = self.station(self.run_analysis())
        self.assertEqual(station["state"], "overlap")
        self.assertFalse(station["complete_components"])
        observed = next(row for row in station["markets"] if row["ticker"] == winner["ticker"])
        self.assertEqual(observed["side"], "yes")
        self.assertEqual(observed["offer"]["price"], "0.30")
        self.assertFalse(station["payout_comparison_available"])

    def test_only_report_consistent_side_counts_as_overlap(self):
        self.capture.official("KXHIGHNY", 72)
        winner = next(row for row in self.ny if row.get("floor_strike") == 72)
        self.capture.books[winner["ticker"]] = self.book(yes=[["0.70", "1.00"]])
        self.assertEqual(self.station(self.run_analysis())["state"], "absence")
        loser = next(row for row in self.ny if row["strike_type"] == "less")
        self.capture.books[loser["ticker"]] = self.book(yes=[["0.80", "1.00"]])
        station = self.station(self.run_analysis())
        overlap = next(row for row in station["markets"] if row["state"] == "overlap")
        self.assertEqual(overlap["ticker"], loser["ticker"])
        self.assertEqual(overlap["side"], "no")
        self.assertEqual(overlap["offer"]["price"], "0.20")

    def test_every_station_uses_exact_cli_mapping_and_exhaustive_market_rules(self):
        for series, event in self.capture.events.items():
            self.capture.official(series)
            for market in event["markets"]:
                self.capture.books[market["ticker"]] = self.book(yes=[["0.40", "1"]], no=[["0.40", "1"]])
        result = self.run_analysis()
        self.assertTrue(all(row["state"] == "overlap" for row in result["stations"]))
        for station in result["stations"]:
            self.assertEqual(station["source"]["station_id"], analysis.STATIONS[station["series"]][1])
            self.assertEqual(sum(row["side"] == "yes" for row in station["markets"]), 1)
        self.assertEqual(self.station(result, "KXHIGHCHI")["source"]["station_id"], "MDW")
        self.assertEqual(self.station(result, "KXHIGHTHOU")["source"]["station_id"], "HOU")

    def test_rule_station_date_source_or_unit_mismatch_blocks_station(self):
        original = self.ny[0]["rules_primary"]
        for old, new in (("CLINYC", "CLIEWR"), ("Sep 24", "Sep 23"),
                         ("fahrenheit", "celsius"), ("The Weather Company", "Other Source")):
            with self.subTest(change=new):
                self.ny[0]["rules_primary"] = original.replace(old, new)
                station = self.station(self.run_analysis())
                self.assertEqual(station["state"], "unknown")
                self.assertEqual(station["reason"], "unsupported_market_event")
        self.ny[0]["rules_primary"] = original

    def test_unknown_fee_metadata_blocks_complete_absence_but_not_verified_overlap(self):
        self.capture.series["KXHIGHNY"]["series"]["fee_multiplier"] = None
        station = self.station(self.run_analysis())
        self.assertEqual(station["state"], "unknown")
        self.assertFalse(station["complete_components"])
        self.capture.official("KXHIGHNY", 72)
        winner = next(row for row in self.ny if row.get("floor_strike") == 72)
        self.capture.books[winner["ticker"]] = self.book(no=[["0.70", "1"]])
        station = self.station(self.run_analysis())
        self.assertEqual(station["state"], "overlap")
        self.assertFalse(station["complete_components"])
        self.assertFalse(station["payout_comparison_available"])
        self.assertFalse(station["fee"]["payout_comparison_available"])
        for prohibited in ("profit", "net_profit", "expected_return", "coefficient"):
            self.assertNotIn(prohibited, station)
            self.assertNotIn(prohibited, station["fee"])

    def test_stale_or_noncausal_source_cannot_turn_no_report_into_absence(self):
        ticker = self.ny[0]["ticker"]
        for times in (("18:02:01", "18:02:02"), ("18:00:01", "18:00:02")):
            with self.subTest(times=times):
                self.capture.book_times[ticker] = times
                station = self.station(self.run_analysis())
                self.assertEqual(station["state"], "unknown")
                self.assertFalse(station["complete_components"])

    def test_conflicting_close_bounds_are_unknown_even_when_report_absent(self):
        # The engineering NY event's text cuts off at 03:59Z but close_time is 05:00Z.
        # Move synthetic source/metadata/book receipts together into that disputed interval.
        original = self.capture.response
        def disputed(*args, **kwargs):
            response = original(*args, **kwargs)
            return replace(response, request_at=response.request_at + dt.timedelta(hours=10, minutes=30),
                           receipt_at=response.receipt_at + dt.timedelta(hours=10, minutes=30))
        self.capture.response = disputed
        station = self.station(self.run_analysis())
        self.assertEqual(station["state"], "unknown")
        self.assertFalse(station["complete_components"])
        self.assertTrue(all(row["reason"] == "uncertain_market_lifecycle" for row in station["markets"]))

    def test_constructor_failure_and_target_mismatch_preserve_full_unknown_denominator(self):
        for failure in (EvidenceError("manifest_missing"), OSError("unreadable")):
            with patch.object(analysis, "CaptureEvidence", side_effect=failure):
                result = analysis.analyze_capture("missing", expected_target=TARGET.isoformat(), now=NOW)
            self.assertEqual(len(result["stations"]), 15)
            self.assertTrue(all(row["state"] == "unknown" for row in result["stations"]))
        self.capture.target = dt.date(2026, 9, 23)
        result = self.run_analysis()
        self.assertEqual({row["reason"] for row in result["stations"]}, {"capture_target_mismatch"})
        self.assertEqual(self.capture.calls, [])

    def test_missing_timezone_fails_before_constructor_and_unit_reads(self):
        with patch.object(analysis, "CaptureEvidence") as constructor, \
                patch.object(analysis, "unit_basis") as basis, \
                self.assertRaisesRegex(EvidenceError, "analysis_clock_timezone_missing"):
            analysis.analyze_capture("reserved", expected_target="2026-10-10",
                                     now=dt.datetime(2026, 10, 25), validation_released=True)
        constructor.assert_not_called()
        basis.assert_not_called()

    def test_verified_authority_composes_with_official_source_and_supported_book(self):
        self.capture.official("KXHIGHNY", 72)
        winner = next(row for row in self.ny if row.get("floor_strike") == 72)
        self.capture.books[winner["ticker"]] = self.book(no=[["0.70", "1.00"]])
        with patch.object(analysis, "CaptureEvidence", return_value=self.capture):
            result = analysis.analyze_capture("synthetic", expected_target=TARGET.isoformat(),
                                              authority_path=AUTHORITY, now=NOW)
        self.assertEqual(result["unit_basis"]["state"], "verified")
        self.assertEqual(self.station(result)["state"], "overlap")
        self.assertFalse(result["payout_comparison_available"])

    def test_reserved_embargo_precedes_constructor_unit_file_and_body_reads(self):
        for now, released in ((analysis.RELEASE_TIME - dt.timedelta(seconds=1), True),
                              (analysis.RELEASE_TIME, False),
                              (analysis.RELEASE_TIME, 1)):
            with self.subTest(now=now, released=released), \
                    patch.object(analysis, "CaptureEvidence") as constructor, \
                    patch.object(analysis, "unit_basis") as basis, \
                    self.assertRaisesRegex(EvidenceError, "reserved_analysis_locked"):
                analysis.analyze_capture("reserved", expected_target="2026-10-10", now=now,
                                         validation_released=released)
            constructor.assert_not_called()
            basis.assert_not_called()


class UnitEvidenceTests(unittest.TestCase):
    def test_preserved_client_bytes_verify_supported_fahrenheit_basis(self):
        result = analysis.unit_basis(AUTHORITY, NOW)
        self.assertEqual(result["state"], "verified", result)
        self.assertEqual(result["sha256"], analysis.CLIENT_SHA256)
        self.assertEqual(result["url"], analysis.CLIENT_URL)
        self.assertIn("does not reconstruct", result["limitation"])

    def test_success_bytes_do_not_override_error_or_invalid_attempt_chronology(self):
        original = json.loads(gzip.decompress(AUTHORITY.read_bytes()))
        mutations = [
            ("request error", lambda row: row.update(error="truncated response")),
            ("missing finish", lambda row: row.pop("attempt_finished_at_utc")),
            ("receipt after finish", lambda row: row.update(
                attempt_finished_at_utc=row["request_at_utc"])),
            ("finish after evaluation", lambda row: row.update(
                attempt_finished_at_utc="2026-09-26T00:00:00Z")),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "authority.json.gz"
            for label, mutate in mutations:
                with self.subTest(condition=label):
                    payload = copy.deepcopy(original)
                    mutate(payload["requests"][0])
                    path.write_bytes(gzip.compress(json.dumps(payload).encode()))
                    self.assertEqual(analysis.unit_basis(path, NOW)["state"], "unknown")

    def test_missing_corrupt_and_wrong_identity_unit_evidence_is_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "authority.json.gz"
            self.assertEqual(analysis.unit_basis(None, NOW)["state"], "unknown")
            self.assertEqual(analysis.unit_basis(path, NOW)["state"], "unknown")
            for body in (b"not gzip", gzip.compress(b"not json"), gzip.compress(b"[]"),
                         AUTHORITY.read_bytes()[:-8]):
                path.write_bytes(body)
                self.assertEqual(analysis.unit_basis(path, NOW)["state"], "unknown")
            original = json.loads(gzip.decompress(AUTHORITY.read_bytes()))
            changes = [{"sha256": "0" * 64}, {"status": 500}, {"body_complete": False},
                       {"bytes": True}, {"final_url": "https://example.com/redirect"},
                       {"body_base64": "*invalid*"}, {"receipt_at_utc": "2026-09-26T00:00:00Z"},
                       {"request_at_utc": "2026-09-25T23:30:00Z"}]
            for change in changes:
                with self.subTest(change=change):
                    payload = copy.deepcopy(original)
                    payload["requests"][0].update(change)
                    path.write_bytes(gzip.compress(json.dumps(payload).encode()))
                    self.assertEqual(analysis.unit_basis(path, NOW)["state"], "unknown")
            for duplicate in ([], original["requests"] * 2):
                payload = copy.deepcopy(original)
                payload["requests"] = duplicate
                path.write_bytes(gzip.compress(json.dumps(payload).encode()))
                self.assertEqual(analysis.unit_basis(path, NOW)["state"], "unknown")


if __name__ == "__main__":
    unittest.main()
