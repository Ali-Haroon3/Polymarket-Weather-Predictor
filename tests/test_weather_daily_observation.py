"""Daily-source schema checks using synthetic data and no external requests."""

import copy
from decimal import Decimal
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from weather_daily_observation import parse_daily


def fixture():
    return {
        "date": "2026-09-23", "totalStations": 1,
        "officialReports": 1, "revisedReports": 0,
        "preliminaryReports": 0, "noReports": 0,
        "results": [{"station": {"cliId": "BOS", "city": "Boston"},
                     "status": "official", "avgTemp": 57.3,
                     "data": {"stationId": "BOS", "reportDate": "2026-09-23",
                              "maxTemp": 64, "minTemp": 51,
                              "isOfficial": True, "issueTime": ""}}],
    }


def parse(payload, **kwargs):
    return parse_daily(payload, "2026-09-23", "BOS", fahrenheit_basis_verified=True, **kwargs)


class DailyObservationTests(unittest.TestCase):
    def assert_unknown(self, payload, reason=None):
        result = parse(payload)
        self.assertEqual(result["state"], "unknown")
        self.assertIsNone(result["max_temp_f"])
        self.assertIs(result["publication_time_known"], False)
        if reason:
            self.assertEqual(result["reason"], reason)
        json.dumps(result, allow_nan=False)
        return result

    def test_exact_official_maximum_and_input_is_not_mutated(self):
        payload = fixture()
        original = copy.deepcopy(payload)
        result = parse(payload)
        self.assertEqual(result["state"], "official")
        self.assertEqual(result["max_temp_f"], "64")
        self.assertEqual(result["station_id"], "BOS")
        self.assertEqual(result["report_date"], "2026-09-23")
        self.assertEqual(result["issue_time"], "")
        self.assertIs(result["publication_time_known"], False)
        self.assertEqual(payload, original)
        json.dumps(result, allow_nan=False)

    def test_missing_unit_evidence_cannot_be_inferred_from_url_city_or_label(self):
        payload = fixture()
        payload["unit"] = "F"
        payload["source"] = "https://weather.com/kalshi/api/climate/primary"
        for flag in (False, None, 1, "true", "F"):
            with self.subTest(flag=flag):
                result = parse_daily(payload, "2026-09-23", "BOS", fahrenheit_basis_verified=flag)
                self.assertEqual(result["reason"], "fahrenheit_basis_unverified")
                self.assertIsNone(result["max_temp_f"])

    def test_no_report_explicit_absence_does_not_substitute_average(self):
        payload = fixture()
        payload.update(officialReports=0, noReports=1)
        payload["results"][0].update(status="no_report", data=None)
        result = parse_daily(payload, "2026-09-23", "BOS")
        self.assertEqual(result["state"], "absent")
        self.assertIsNone(result["max_temp_f"])
        self.assertFalse(result["fahrenheit_basis_verified"])

    def test_missing_or_duplicate_station_is_unknown_not_absent(self):
        payload = fixture()
        payload["results"][0]["station"]["cliId"] = "PVD"
        self.assert_unknown(payload, "station_missing")
        payload = fixture()
        payload["results"].append(copy.deepcopy(payload["results"][0]))
        payload.update(totalStations=2, officialReports=2)
        self.assert_unknown(payload, "duplicate_station")

    def test_city_name_cannot_replace_exact_cli_and_report_identity(self):
        for field, value, reason in (
            ("stationId", "KBOS", "report_station_mismatch"),
            ("stationId", "Boston", "report_station_mismatch"),
            ("reportDate", "2026-09-24", "report_date_mismatch"),
        ):
            with self.subTest(field=field, value=value):
                payload = fixture()
                payload["results"][0]["data"][field] = value
                self.assert_unknown(payload, reason)

    def test_envelope_date_and_requested_identity_are_exact(self):
        for value in ("2026-09-24", "2026-9-23", "2026-02-30", None):
            with self.subTest(value=value):
                payload = fixture()
                payload["date"] = value
                self.assert_unknown(payload)
        for target, station in (("2026-02-30", "BOS"), ("2026-09-23", "bos"),
                                ("2026-09-23", "CLIBOS"), (None, None)):
            self.assertEqual(parse_daily(fixture(), target, station)["reason"], "invalid_requested_identity")

    def test_count_and_status_conflicts_fail_closed(self):
        for field, value in (("totalStations", 2), ("totalStations", True),
                             ("officialReports", 0), ("noReports", 1),
                             ("revisedReports", True)):
            with self.subTest(field=field, value=value):
                payload = fixture()
                payload[field] = value
                self.assert_unknown(payload)
        payload = fixture()
        payload["results"][0]["status"] = "final"
        self.assert_unknown(payload, "unsupported_daily_status")

    def test_malformed_rows_and_missing_data_are_unknown(self):
        for value in (None, [], "BOS", {"station": None}, {"station": {"cliId": None}}):
            with self.subTest(value=value):
                payload = fixture()
                payload["results"] = [value]
                self.assert_unknown(payload)
        payload = fixture()
        del payload["results"][0]["data"]
        self.assert_unknown(payload, "missing_report_data")

    def test_official_flags_must_be_boolean_true(self):
        for value in (None, False, 1, "true"):
            with self.subTest(value=value):
                payload = fixture()
                payload["results"][0]["data"]["isOfficial"] = value
                self.assert_unknown(payload, "official_status_conflict")

    def test_preliminary_revised_and_conflicting_flags_are_unsupported(self):
        for status, counter in (("preliminary", "preliminaryReports"), ("revised", "revisedReports")):
            with self.subTest(status=status):
                payload = fixture()
                payload.update(officialReports=0)
                payload[counter] = 1
                payload["results"][0]["status"] = status
                self.assert_unknown(payload, "unsupported_nonofficial_report")
        for flag in ("isRevised", "isPreliminary"):
            payload = fixture()
            payload["results"][0]["data"][flag] = True
            self.assert_unknown(payload, "unsupported_revision_or_preliminary_flag")

    def test_no_report_with_embedded_value_is_conflict(self):
        payload = fixture()
        payload.update(officialReports=0, noReports=1)
        payload["results"][0]["status"] = "no_report"
        self.assert_unknown(payload, "no_report_data_conflict")

    def test_numeric_missing_and_nonfinite_values_cannot_be_weather(self):
        for value in (None, "64", "M", "T", "-9999", True, [], {},
                      float("nan"), float("inf"), float("-inf"),
                      Decimal("NaN"), Decimal("sNaN"), Decimal("Infinity")):
            with self.subTest(value=repr(value)):
                payload = fixture()
                payload["results"][0]["data"]["maxTemp"] = value
                self.assert_unknown(payload, "invalid_daily_maximum")

    def test_only_integral_supported_numeric_domain_is_accepted_without_rounding(self):
        for value in (-150, -0.0, 64.0, Decimal("64.000"), 150):
            with self.subTest(value=value):
                payload = fixture()
                payload["results"][0]["data"].update(maxTemp=value, minTemp=None)
                result = parse(payload)
                self.assertEqual(result["state"], "official")
                self.assertEqual(result["max_temp_f"], str(int(value)))
        for value in (-9999, -999, -151, 151, 999, 9999, Decimal("1e999")):
            payload = fixture()
            payload["results"][0]["data"]["maxTemp"] = value
            self.assert_unknown(payload, "unsupported_numeric_domain")
        for value in (64.1, Decimal("64.0000000000000001")):
            payload = fixture()
            payload["results"][0]["data"]["maxTemp"] = value
            self.assert_unknown(payload, "unsupported_fractional_daily_maximum")

    def test_explicit_unit_conflicts_override_external_basis(self):
        for container in ("root", "row", "station", "data"):
            payload = fixture()
            row = payload["results"][0]
            target = {"root": payload, "row": row, "station": row["station"], "data": row["data"]}[container]
            target["temperatureUnit"] = "C"
            self.assert_unknown(payload, "conflicting_or_unsupported_unit")

    def test_issue_time_never_becomes_known_publication_time(self):
        for value in ("", "2026-09-24T05:00:00Z", "unrecognized source text"):
            payload = fixture()
            payload["results"][0]["data"]["issueTime"] = value
            result = parse(payload)
            self.assertEqual(result["state"], "official")
            self.assertEqual(result["issue_time"], value)
            self.assertIs(result["publication_time_known"], False)
        payload = fixture()
        del payload["results"][0]["data"]["issueTime"]
        self.assertEqual(parse(payload)["state"], "official")
        payload["results"][0]["data"]["issueTime"] = []
        self.assert_unknown(payload, "invalid_issue_time")

    def test_max_below_min_is_unknown_and_non_max_precipitation_is_not_substituted(self):
        payload = fixture()
        payload["results"][0]["data"].update(maxTemp=50, minTemp=51)
        self.assert_unknown(payload, "maximum_below_minimum")
        payload = fixture()
        payload["results"][0]["data"].update(precipitation="T", snowfall=None)
        self.assertEqual(parse(payload)["max_temp_f"], "64")
        del payload["results"][0]["data"]["maxTemp"]
        self.assert_unknown(payload, "invalid_daily_maximum")


if __name__ == "__main__":
    unittest.main()
