"""Causal-history, coverage, sizing, and shadow regressions for the fixed challenger."""

import copy
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import weather_challenger as challenger


def snapshot(cap="2026-09-01", target="2026-09-02", city="NYC", winner=2):
    cells = (("temp_at_most", 19, None, 0.10, 0.14),
             ("temp_bucket", 20, 20, 0.18, 0.22),
             ("temp_bucket", 21, 21, 0.28, 0.32),
             ("temp_at_least", 22, None, 0.36, 0.40))
    return [dict(source="kalshi", captured_at=cap, target_date=target, city=city,
                 market_id=f"{city}-{target}-{i}", market_type=mt, threshold=lo,
                 threshold_upper=hi, unit="C", best_bid=bid, best_ask=ask,
                 forecast_high=21.5, outcome=int(i == winner))
            for i, (mt, lo, hi, bid, ask) in enumerate(cells)]


def training_event(i, realized=21.0):
    target = dt.date(2026, 6, 1) + dt.timedelta(days=i)
    event, _ = challenger.build_events(snapshot(str(target - dt.timedelta(days=1)), str(target), f"C{i}"))
    return dict(event[0], realized=realized)


class PartitionTests(unittest.TestCase):
    def test_exact_ladder_rejects_gaps_overlap_missing_tails_and_duplicate_ticker(self):
        events, _ = challenger.build_events(snapshot())
        self.assertEqual(len(events), 1)
        cells = events[0]["cells"]
        self.assertTrue(challenger.exact_partition(cells))
        for changed in (
            cells[1:],
            [dict(c, lo=c["lo"] + 0.1) if i == 2 else c for i, c in enumerate(cells)],
            [dict(c, lo=c["lo"] - 0.1) if i == 2 else c for i, c in enumerate(cells)],
            [dict(c, mid=cells[0]["mid"]) if i == 1 else c for i, c in enumerate(cells)],
        ):
            with self.subTest(cells=changed):
                self.assertFalse(challenger.exact_partition(changed))

    def test_book_cross_or_invalid_quote_cannot_be_used(self):
        for bad in (dict(best_bid=0.9, best_ask=0.1), dict(best_ask=float("nan")),
                    dict(best_ask=1.2), dict(best_bid=None, best_ask=None)):
            rows = snapshot()
            rows[1].update(bad)
            self.assertEqual(challenger.build_events(rows)[0], [])

    def test_first_complete_snapshot_selected_without_outcome_filter(self):
        early = snapshot(cap="2026-08-31")
        later = snapshot(cap="2026-09-01", winner=1)
        for row in early:
            row["outcome"] = None
        events, rejected = challenger.build_events(later + early)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["cap"], "2026-08-31")
        self.assertFalse(events[0]["resolved"])
        self.assertEqual(rejected["later_snapshot_same_event"], 1)

    def test_exact_timestamps_are_not_combined_into_synthetic_day_books(self):
        rows = snapshot(cap="2026-09-01T08:00:00Z")[:2]
        rows += snapshot(cap="2026-09-01T09:00:00Z")[2:]
        self.assertEqual(challenger.build_events(rows)[0], [])


class CausalityTests(unittest.TestCase):
    def test_legacy_resolution_delay_and_observed_clock(self):
        event = challenger.build_events(snapshot())[0][0]
        self.assertEqual(challenger.eligible_history([event], "2026-09-03"), [])
        self.assertEqual(len(challenger.eligible_history([event], "2026-09-04")), 1)
        rows = snapshot()
        for row in rows:
            row["outcome_observed_at"] = "2026-09-06T00:00:00Z"
        known = challenger.build_events(rows)[0][0]
        self.assertEqual(challenger.eligible_history([known], "2026-09-06"), [])
        self.assertEqual(len(challenger.eligible_history([known], "2026-09-07")), 1)
        rows[0]["outcome_observed_at"] = "invalid"
        invalid = challenger.build_events(rows)[0][0]
        self.assertEqual(challenger.eligible_history([invalid], "2026-10-01"), [])

    def test_missing_or_future_outcome_cannot_change_earlier_fits_or_decisions(self):
        old = [training_event(i) for i in range(3)]
        future = challenger.build_events(snapshot(cap="2026-09-01", target="2026-09-02"))[0][0]
        events = old + [future]
        fit1 = challenger.fit_families(challenger.eligible_history(events, "2026-09-01"),
                                      challenger.score_cache(events), minimum=3)
        changed = copy.deepcopy(events)
        changed[-1]["realized"] = 500
        for cell in changed[-1]["cells"]:
            cell["outcome"] = 1 - cell["outcome"]
        fit2 = challenger.fit_families(challenger.eligible_history(changed, "2026-09-01"),
                                      challenger.score_cache(changed), minimum=3)
        self.assertEqual(fit1, fit2)
        for family in challenger.FAMILIES:
            self.assertEqual(challenger.candidates_for(future, family, fit1[family]),
                             challenger.candidates_for(changed[-1], family, fit2[family]))

    def test_scale_and_bias_are_independent_refits(self):
        hist = [training_event(0)]
        # Joint winner (0.3, 0.6), scale winner (0, 1.1), bias winner (-0.2, 1).
        scores = [10.0] * len(challenger.GRID)
        for pair, value in (((0.3, 0.6), 0.0), ((0.0, 1.1), 1.0), ((-0.2, 1.0), 2.0)):
            scores[challenger.GRID.index(pair)] = value
        fitted = challenger.fit_families(hist, {(hist[0]["city"], hist[0]["target"]): scores}, minimum=1)
        self.assertEqual(fitted["joint_shape"], dict(bias_c=0.3, scale=0.6))
        self.assertEqual(fitted["scale_only"], dict(bias_c=0.0, scale=1.1))
        self.assertEqual(fitted["bias_only"], dict(bias_c=-0.2, scale=1.0))

    def test_weather_fit_is_shrunk_and_missing_forecast_falls_to_market(self):
        hist = [training_event(i, realized=22) for i in range(3)]
        fitted = challenger.fit_weather(hist, minimum=3)
        self.assertTrue(fitted["weather_fit_available"])
        self.assertLessEqual(fitted["weather_weight"], 0.25)
        self.assertGreaterEqual(fitted["weather_weight"], 0)
        event = dict(hist[0], forecast_high=None)
        self.assertEqual(challenger.probabilities(event, "weather_blend", fitted),
                         challenger.probabilities(event, "market_normal", dict(bias_c=0, scale=1)))

    def test_weather_scale_calibration_includes_tail_winners(self):
        interior = [training_event(i) for i in range(3)]
        tails = [training_event(i) for i in range(3, 25)]
        for event in tails:
            event["realized"] = None
            for cell in event["cells"]:
                cell["outcome"] = int(cell["hi"] == float("inf"))
        interior_fit = challenger.fit_weather(interior, minimum=3)
        with_tails = challenger.fit_weather(interior + tails, minimum=3)
        self.assertEqual(interior_fit["bias_c"], with_tails["bias_c"])
        self.assertEqual(interior_fit["weather_weight"], with_tails["weather_weight"])
        self.assertGreater(with_tails["scale"], interior_fit["scale"])

    def test_shadow_masks_current_future_labels_before_building_events(self):
        rows = snapshot()
        actual_builder = challenger.build_events
        seen = []

        def capture(masked):
            seen.extend(masked)
            return actual_builder(masked)

        with patch.object(challenger, "build_events", side_effect=capture):
            result = challenger.run(rows, "shadow")
        self.assertTrue(all(r["outcome"] is None for r in seen))
        for family in result["families"].values():
            self.assertNotIn("trades", family)
            self.assertNotIn("periods", family)

    def test_cli_rejects_stale_capture_before_writing_shadow(self):
        with tempfile.TemporaryDirectory() as directory:
            captures = Path(directory) / "captures.jsonl"
            output = Path(directory) / "shadow.json"
            captures.write_text("".join(json.dumps(row) + "\n" for row in snapshot()))
            output.write_text("do not overwrite a prior artifact")
            process = subprocess.run([sys.executable, challenger.__file__, "--shadow",
                                      "--captures", str(captures), "--output", str(output),
                                      "--require-capture-date", "2026-09-02"],
                                     text=True, capture_output=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("does not match required date", process.stderr)
            self.assertEqual(output.read_text(), "do not overwrite a prior artifact")


class ExecutionTests(unittest.TestCase):
    def test_integer_contracts_and_rounded_fee_fit_inside_15_dollars(self):
        for price in (0.10, 0.11, 0.24, 0.60, 0.95):
            size = challenger.size_order(price)
            self.assertIsInstance(size["contracts"], int)
            self.assertLessEqual(size["cost"], 15 + 1e-9)
            self.assertAlmostEqual(size["fee"] * 100, round(size["fee"] * 100))
            n = size["contracts"] + 1
            self.assertGreater(n * price + challenger.gate.kalshi_fee(n, price), 15)
        self.assertEqual(challenger.size_order(0.60)["contracts"], 24)

    def test_rounded_fee_hurdle_and_executable_side(self):
        event = challenger.build_events(snapshot())[0][0]
        for c in event["cells"]:
            c.update(ask=None, bid=None)
        event["cells"][0]["ask"] = 0.60
        fit = dict(bias_c=0.0, scale=1.0)
        size = challenger.size_order(0.60)
        threshold = 0.60 + 0.04 + size["fee"] / size["contracts"]
        with patch.object(challenger, "probabilities", return_value=[threshold - 1e-5, 0, 0, 0]):
            self.assertEqual(challenger.candidates_for(event, "market_normal", fit), [])
        with patch.object(challenger, "probabilities", return_value=[threshold, 0, 0, 0]):
            candidates = challenger.candidates_for(event, "market_normal", fit)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["side"], "yes")

    def test_cap_one_event_max_five_and_ticker_dedupe(self):
        candidates = [dict(run_at="2026-09-01", target_date="2026-09-02", city=str(i),
                           ticker=str(i), side="yes", price=0.6, edge=0.2 - i / 100,
                           **challenger.size_order(0.6)) for i in range(8)]
        candidates += [dict(candidates[0], ticker="same-event", edge=0.25),
                       dict(candidates[1], run_at="2026-09-02", city="new-city")]
        selected = challenger.select_candidates(candidates)
        self.assertEqual(len(selected), 5)
        self.assertEqual(selected[0]["ticker"], "same-event")
        self.assertEqual(len({(t["city"], t["target_date"]) for t in selected}), 5)
        # Unknown outcomes cannot release slots: selection does not have access to them.
        trades, opened = challenger.settle(selected, [])
        self.assertEqual(trades, [])
        self.assertEqual(opened, 5)

    def test_adverse_quote_reprices_and_resizes_fees(self):
        event = challenger.build_events(snapshot())[0][0]
        candidate = dict(run_at=event["cap"], target_date=event["target"], city=event["city"],
                         ticker=event["cells"][2]["mid"], side="yes", price=0.60, edge=0.1,
                         **challenger.size_order(0.60))
        trades, opened = challenger.settle([candidate], [event])
        self.assertEqual(opened, 0)
        stress = trades[0]["stress_1c"]
        self.assertAlmostEqual(stress["price"], 0.61)
        self.assertEqual(stress["contracts"], 23)
        self.assertNotEqual(stress["fee"], candidate["fee"])
        self.assertLessEqual(stress["cost"], 15)
        self.assertAlmostEqual(stress["net"], stress["contracts"] - stress["cost"])


if __name__ == "__main__":
    unittest.main()
