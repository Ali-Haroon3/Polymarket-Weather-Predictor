"""Offline semantics tests; excluded engineering responses supply schema only."""
import base64
import copy
from decimal import Decimal, localcontext
import gzip
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import weather_market_observation as observation


class MarketObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = gzip.decompress((ROOT / "reports/2026-09-25-source-engineering-evidence.json.gz").read_bytes())
        assert hashlib.sha256(raw).hexdigest() == "cdc77bef1ce57629f192a1e4b1979d43b11058f54de165d2b379b0182c45e2cb"
        bundle = json.loads(raw)
        cls.events, cls.series = {}, {}
        for request in bundle["manifest"]["requests"]:
            # Only inspect schema-bearing market/series evidence. No source/quote comparison.
            if request["role"] not in ("markets", "series"):
                continue
            body = base64.b64decode(bundle["raw_response_bodies_base64"][request["raw_file"]], validate=True)
            assert len(body) == request["bytes"]
            assert hashlib.sha256(body).hexdigest() == request["sha256"]
            target = cls.events if request["role"] == "markets" else cls.series
            target[request["series"]] = json.loads(body)

    def event(self):
        return copy.deepcopy(self.events["KXHIGHNY"])

    def validate(self, payload):
        return observation.validate_event(payload, "KXHIGHNY", "2026-09-24")

    def market(self):
        return self.event()["markets"][0]

    def lifecycle(self, market=None, received="2026-09-25T01:00:00Z",
                  requested="2026-09-25T01:00:01Z", booked="2026-09-25T01:00:02Z"):
        return observation.market_lifecycle(
            self.market() if market is None else market, received, requested, booked,
            target_date="2026-09-24", timezone="America/New_York")

    @staticmethod
    def book(yes=None, no=None):
        return {"orderbook_fp": {"yes_dollars": [] if yes is None else yes,
                                 "no_dollars": [] if no is None else no}}

    def test_all_fifteen_saved_event_schemas_and_station_identities(self):
        for series, payload in self.events.items():
            with self.subTest(series=series):
                result = observation.validate_event(payload, series, "2026-09-24")
                self.assertEqual(result["state"], "valid", result)
                self.assertEqual(len(result["markets"]), 6)
                self.assertIsNone(result["markets"][0]["lower"])
                self.assertIsNone(result["markets"][-1]["upper"])
        self.assertEqual(observation.STATIONS["KXHIGHCHI"][1:3], ("MDW", "KMDW"))
        self.assertEqual(observation.STATIONS["KXHIGHTHOU"][1:3], ("HOU", "KHOU"))

    def test_event_identity_and_schema_mismatches_are_unknown(self):
        mutations = [
            lambda p: p.update(cursor="more"),
            lambda p: p.update(next_cursor="more"),
            lambda p: p["markets"].pop(),
            lambda p: p["markets"].__setitem__(1, p["markets"][0]),
            lambda p: p["markets"][0].update(event_ticker="KXHIGHNY-26SEP23"),
            lambda p: p["markets"][0].update(ticker="KXHIGHNY-26SEP24-B71.5"),
            lambda p: p["markets"][0].update(market_type="scalar"),
            lambda p: p["markets"][0].update(notional_value_dollars="0.5000"),
            lambda p: p["markets"][0].update(floor_strike=True),
            lambda p: p["markets"][0].update(floor_strike=72.1),
            lambda p: p["markets"][0].update(strike_type="greater_or_equal"),
            lambda p: p["markets"][0].update(rules_secondary=""),
        ]
        for mutate in mutations:
            payload = self.event()
            mutate(payload)
            self.assertEqual(self.validate(payload)["state"], "unknown")
        for payload in (None, [], {}, {"markets": [None] * 6}):
            self.assertEqual(self.validate(payload)["state"], "unknown")

    def test_rule_source_unit_date_station_and_condition_are_exact(self):
        for old, new in (("CLINYC", "CLIEWR"), ("Sep 24", "Sep 23"),
                         ("fahrenheit", "celsius"), ("The Weather Company", "Another Source"),
                         ("72-73", "72-74"), ("maximum", "minimum")):
            payload = self.event()
            payload["markets"][0]["rules_primary"] = payload["markets"][0]["rules_primary"].replace(old, new)
            self.assertEqual(self.validate(payload)["state"], "unknown")

    def test_changed_nonempty_secondary_rule_is_unknown(self):
        payload = self.event()
        payload["markets"][0]["rules_secondary"] += " Override: use an alternative settlement source."
        result = self.validate(payload)
        self.assertEqual(result["state"], "unknown")
        self.assertIn("unsupported correction or exceptional-settlement rules", result["reasons"][0])

    def test_raw_decimal_fraction_cannot_round_into_valid_integer_strike(self):
        payload = self.event()
        payload["markets"][0]["floor_strike"] = Decimal("72.00000000000000000000000000001")
        self.assertEqual(self.validate(payload)["state"], "unknown")
        payload = self.event()
        payload["markets"][0]["floor_strike"] = Decimal("72.0")
        self.assertEqual(self.validate(payload)["state"], "valid")
        series = copy.deepcopy(self.series["KXHIGHNY"])
        series["series"]["fee_multiplier"] = Decimal("1.00000000000000000000000000001")
        self.assertEqual(observation.fee_terms(series, "KXHIGHNY")["state"], "unknown")

    def test_huge_strike_exponents_fail_before_integer_allocation(self):
        for strike in (Decimal("1e999999999"), Decimal("-1e999999999")):
            payload = self.event()
            payload["markets"][0]["floor_strike"] = strike
            result = self.validate(payload)
            self.assertEqual(result["state"], "unknown")
            self.assertIn("parser support/size limit", result["reasons"][0])

    def test_identity_and_matching_rules_do_not_prove_partition(self):
        # Move one entire bucket and its text/ticker; every leg is internally consistent.
        payload = self.event()
        bucket = payload["markets"][0]
        bucket.update(floor_strike=80, cap_strike=81, ticker="KXHIGHNY-26SEP24-B80.5")
        bucket["rules_primary"] = bucket["rules_primary"].replace("72-73", "80-81")
        result = self.validate(payload)
        self.assertEqual(result["state"], "unknown")
        self.assertIn("partition", result["reasons"][0])

    def test_report_consistent_sides_use_integer_inclusive_bounds(self):
        rows = self.validate(self.event())["markets"]
        for value in (-100, 65, 66, 67, 68, 73, 74, 200):
            sides = [observation.report_consistent_side(row, value) for row in rows]
            self.assertTrue(all(side["state"] == "valid" for side in sides))
            self.assertEqual(sum(side["side"] == "yes" for side in sides), 1)
        for value in (True, None, "NaN", 66.5):
            self.assertEqual(observation.report_consistent_side(rows[0], value)["state"], "unknown")
        self.assertEqual(observation.report_consistent_side({}, 67)["state"], "unknown")

    def test_before_both_lifecycle_bounds_is_active_with_disclosure(self):
        result = self.lifecycle()
        self.assertEqual(result["state"], "active")
        self.assertTrue(result["close_bounds_disagree"])
        self.assertEqual(result["textual_cutoff"], "2026-09-25T03:59:00+00:00")

    def test_at_earlier_boundary_and_between_bounds_are_unknown(self):
        for booked in ("2026-09-25T03:59:00Z", "2026-09-25T04:30:00Z"):
            self.assertEqual(self.lifecycle(booked=booked)["state"], "unknown")
        self.assertEqual(self.lifecycle(booked="2026-09-25T05:00:00Z")["state"], "closed")

    def test_lifecycle_missing_conflicting_or_naive_evidence_is_unknown(self):
        for change in ({"status": "mystery"}, {"close_time": None}, {"early_close_condition": ""},
                       {"open_time": "2026-09-25T03:00:00Z"}, {"result": "yes"}):
            market = self.market()
            market.update(change)
            self.assertEqual(self.lifecycle(market)["state"], "unknown")
        self.assertEqual(self.lifecycle(received="2026-09-25T01:01:00Z")["state"], "unknown")
        self.assertEqual(self.lifecycle(booked="2026-09-25T01:00:00")["state"], "unknown")
        self.assertEqual(self.lifecycle(market=None)["state"], "active")

    def test_explicit_closed_status_is_observed_not_missing(self):
        market = self.market()
        market["status"] = "finalized"
        self.assertEqual(self.lifecycle(market)["state"], "closed")

    def test_bid_complement_direction_and_decimal_depth(self):
        result = observation.displayed_offer(self.book(yes=[["0.72", "1.20"]]), "no")
        self.assertEqual(result["state"], "available")
        self.assertEqual(result["price"], "0.28")
        self.assertEqual(result["depth"], "1.20")
        self.assertEqual(observation.displayed_offer(self.book(yes=[["0.72", "1.20"]]), "yes")["state"], "absent")

    def test_fractional_levels_accumulate_to_one_at_worst_limit(self):
        result = observation.displayed_offer(self.book(no=[["0.70", "0.60"], ["0.80", "0.60"]]), "yes")
        self.assertEqual(result["state"], "available")
        self.assertEqual(result["price"], "0.30")
        self.assertEqual(result["depth"], "1.20")
        self.assertEqual(result["levels"][0]["offer_price"], "0.20")
        self.assertTrue(result["price_is_limit_upper_bound"])

    def test_insufficient_depth_and_boundary_prices_are_observed_absence(self):
        for yes in ([], [["0.80", "0.99"]], [["0.80", "0.00"]],
                    [["0.00", "100.00"]], [["1.00", "100.00"]]):
            self.assertEqual(observation.displayed_offer(self.book(yes=yes), "no")["state"], "absent")

    def test_decimal_context_cannot_round_subcontract_depth_up(self):
        depth = "0." + "9" * 50
        with localcontext() as context:
            context.prec = 3
            result = observation.displayed_offer(self.book(yes=[["0.80", depth]]), "no")
        self.assertEqual(result["state"], "absent")
        self.assertEqual(result["depth"], depth)

    def test_corrupt_crossed_or_unsupported_books_are_unknown(self):
        bad = [None, {}, {"orderbook": {"yes": [], "no": []}},
               {"orderbook_fp": {"yes_dollars": []}},
               self.book(yes=[["0.7", "1"], ["0.7", "1"]]),
               self.book(yes=[["0.7", "1"]], no=[["0.4", "1"]]),
               self.book(yes=[[0.7, "1"]]), self.book(yes=[["0.7", -1]]),
               self.book(yes=[["0.7", "NaN"]]), self.book(yes=[["Infinity", "1"]]),
               self.book(yes=[["0.7", "1e3"]]), self.book(yes=[["-0.1", "1"]]),
               self.book(yes=[["0.7", "1" * 129]]),
               self.book(yes=[["0.7"]]), self.book(yes=[["1.1", "1"]]),
               self.book(yes=[["0.01", "1"]] * 101)]
        for payload in bad:
            self.assertEqual(observation.displayed_offer(payload, "no")["state"], "unknown", payload)

    def test_all_saved_fee_metadata_is_supported_but_not_a_fee_formula(self):
        for series, payload in self.series.items():
            result = observation.fee_terms(payload, series)
            self.assertEqual(result["state"], "supported_metadata")
            self.assertFalse(result["payout_comparison_available"])
            self.assertNotIn("coefficient", result)

    def test_missing_unsupported_and_conflicting_fee_metadata_stays_unknown(self):
        for change in ({"fee_type": None}, {"fee_multiplier": None}, {"fee_multiplier": True},
                       {"fee_multiplier": 2}, {"ticker": "KXHIGHCHI"},
                       {"settlement_sources": []}):
            payload = copy.deepcopy(self.series["KXHIGHNY"])
            payload["series"].update(change)
            result = observation.fee_terms(payload, "KXHIGHNY")
            self.assertEqual(result["state"], "unknown")
            self.assertFalse(result["payout_comparison_available"])

    def test_results_are_json_serializable(self):
        results = [self.validate(self.event()), self.lifecycle(),
                   observation.displayed_offer(self.book(yes=[["0.7", "1"]]), "no"),
                   observation.fee_terms(self.series["KXHIGHNY"], "KXHIGHNY")]
        json.dumps(results, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
