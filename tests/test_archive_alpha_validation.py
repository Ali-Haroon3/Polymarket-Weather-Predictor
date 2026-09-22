"""Causal and accounting regressions for the preregistered archive evaluation."""

import copy
import datetime as dt
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import archive_alpha_validation as archive


def event_rows(city="NYC", target="2026-05-30"):
    date = dt.date.fromisoformat(target)
    stamp = archive.entry_time(date)
    event = archive.CITY_SERIES[city] + "-" + date.strftime("%y%b%d").upper()
    settlement = int(dt.datetime.combine(date + dt.timedelta(days=1),
                                         dt.time(12), tzinfo=archive.UTC).timestamp())
    cells = [("T69", "temp_at_most", 69, None, 0.20),
             ("B70.5", "temp_bucket", 70, 71, 0.30),
             ("B72.5", "temp_bucket", 72, 73, 0.30),
             ("T74", "temp_at_least", 74, None, 0.20)]
    return [dict(
        source="kalshi", archive_method="historical_1m_bid_ask_close", city=city,
        target_date=target, captured_at=str(date - dt.timedelta(days=1)),
        entry_ts=stamp, quote_close_ts=stamp,
        open_time=dt.datetime.fromtimestamp(stamp - 86400, archive.UTC).isoformat(),
        market_id=event + "-" + suffix, event_ticker=event, market_type=kind,
        threshold=lower, threshold_upper=upper, unit="F",
        best_bid=price - 0.02, best_ask=price + 0.02,
        outcome=int(index == 1), settlement_ts=settlement,
    ) for index, (suffix, kind, lower, upper, price) in enumerate(cells)]


def candidate(index, side="yes", outcome=None):
    return dict(run_at="2026-06-01", target_date="2026-06-02",
                city=list(archive.CITY_SERIES)[index], ticker=f"T{index}",
                side=side, price=0.3, edge=0.20 - index * 0.01, outcome=outcome)


class ArchiveCausalityTests(unittest.TestCase):
    def test_exact_entry_and_open_time_are_required(self):
        rows = event_rows()
        rows[0]["quote_close_ts"] += 60
        with self.assertRaisesRegex(ValueError, "exactly"):
            archive.prepare_ladders(rows)
        rows = event_rows()
        rows[0]["open_time"] = dt.datetime.fromtimestamp(
            rows[0]["entry_ts"] + 0.1, archive.UTC).isoformat()
        with self.assertRaisesRegex(ValueError, "not open"):
            archive.prepare_ladders(rows)

    def test_delayed_settlement_of_any_leg_blocks_training(self):
        rows = event_rows()
        entry = int(dt.datetime(2026, 6, 1, 15, tzinfo=archive.UTC).timestamp())
        rows[0]["settlement_ts"] = entry + 1
        ladders, _ = archive.prepare_ladders(rows)
        self.assertEqual(archive.causal_history(ladders, entry), [])
        self.assertEqual(len(archive.causal_history(ladders, entry + 1)), 1)
        rows[0]["settlement_ts"] = entry
        ladders, _ = archive.prepare_ladders(rows)
        self.assertEqual(len(archive.causal_history(ladders, entry)), 1)

    def test_fractional_settlement_is_rounded_up_for_availability(self):
        rows = event_rows()
        entry = int(dt.datetime(2026, 6, 1, 15, tzinfo=archive.UTC).timestamp())
        rows[0]["settlement_time_utc"] = "2026-06-01T15:00:00.100000Z"
        rows[0]["settlement_ts"] = entry + 1
        ladders, _ = archive.prepare_ladders(rows)
        self.assertFalse(archive.causal_history(ladders, entry))
        self.assertEqual(len(archive.causal_history(ladders, entry + 1)), 1)

    def test_target_date_and_publication_timestamp_are_both_required(self):
        ladders, _ = archive.prepare_ladders(event_rows(target="2026-06-01"))
        entry = int(dt.datetime(2026, 6, 1, 15, tzinfo=archive.UTC).timestamp())
        ladders[0]["settlement_ts"] = entry - 1
        self.assertFalse(archive.causal_history(ladders, entry))
        ladders[0]["target"] = "2026-05-31"
        ladders[0]["settlement_ts"] = None
        self.assertFalse(archive.causal_history(ladders, entry))

    def test_disclosed_probe_is_removed_from_every_stage(self):
        ordinary = event_rows(target="2026-06-14")
        probe = event_rows(target="2026-06-15")
        ladders, excluded = archive.prepare_ladders(ordinary + probe)
        self.assertEqual(excluded, 4)
        self.assertEqual([ladder["target"] for ladder in ladders], ["2026-06-14"])
        # The defensive history check also rejects a manually supplied probe ladder.
        forbidden = dict(ladders[0], target="2026-06-15")
        entry = int(dt.datetime(2026, 6, 20, 15, tzinfo=archive.UTC).timestamp())
        self.assertFalse(archive.causal_history([forbidden], entry))

    def test_newest_300_cap_applies_after_availability_filter(self):
        prototype = archive.prepare_ladders(event_rows())[0][0]
        ladders = []
        for offset in range(35):
            target = dt.date(2026, 5, 1) + dt.timedelta(days=offset)
            for city in list(archive.CITY_SERIES)[:10]:
                ladders.append(dict(prototype, city=city, target=str(target),
                                    cap=str(target - dt.timedelta(days=1)), settlement_ts=1))
        entry = int(dt.datetime(2026, 6, 27, 15, tzinfo=archive.UTC).timestamp())
        history = archive.causal_history(ladders, entry)
        self.assertEqual(len(history), 300)
        self.assertEqual(min(row["target"] for row in history), "2026-05-06")
        late = [dict(row, settlement_ts=entry + 1) for row in ladders[-100:]]
        self.assertEqual(len(archive.causal_history(ladders[:-100] + late, entry)), 250)
        self.assertIsNone(archive.shape.fit_shape(history[:59]))

    def test_current_outcomes_do_not_affect_candidates_and_hurdle_is_four_percent(self):
        ladder = archive.prepare_ladders(event_rows(target="2026-06-02"))[0][0]
        for cell in ladder["cells"]:
            cell["bid"], cell["ask"] = 0.48, 0.49
        with patch.object(archive.shape, "shaped_probs", return_value=[0.54] * 4):
            self.assertEqual(archive.candidates_for([ladder], (0, 1)), [])
        with patch.object(archive.shape, "shaped_probs", return_value=[0.55] * 4):
            before = archive.candidates_for([ladder], (0, 1))
            for cell in ladder["cells"]:
                cell["outcome"] = 1 - cell["outcome"]
            after = archive.candidates_for([ladder], (0, 1))
        self.assertEqual(before, after)
        self.assertEqual(len(before), 4)
        self.assertTrue(all("outcome" not in row for row in before))


class ArchiveSelectionAndAccountingTests(unittest.TestCase):
    def test_parent_no_never_backfills_and_unknowns_consume_capacity(self):
        candidates = [candidate(i, "no" if i in (1, 5) else "yes") for i in range(6)]
        families = {name: copy.deepcopy(candidates) for name, _, _ in archive.FAMILIES}
        selected = archive.select_policies(families)
        self.assertEqual(len(selected["joint"]), 5)
        self.assertEqual([row["ticker"] for row in selected["joint_parent_no"]], ["T1"])
        outcomes = {(f"T{i}", "2026-06-02"): None if i == 0 else 1 for i in range(6)}
        settled, opened = archive.settle_selected(selected["joint"], outcomes)
        self.assertEqual(len(settled), 4)
        self.assertEqual(len(opened), 1)
        self.assertNotIn("T5", [row["ticker"] for row in settled + opened])
        for row in candidates:
            row["outcome"] = 1
        self.assertEqual(archive.select_outcome_blind(candidates), selected["joint"])

    def test_integer_sizing_city_caps_and_dedup_are_inherited(self):
        rows = [dict(candidate(i), city="NYC", price=0.78) for i in range(6)]
        rows.append(copy.deepcopy(rows[0]))
        selected = archive.select_outcome_blind(rows)
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(row["contracts"] == 19 for row in selected))
        self.assertAlmostEqual(sum(row["cost"] for row in selected), 29.64)

    def test_stress_recomputes_fees_and_reports_extra_capital_without_resizing(self):
        trade = dict(candidate(0), contracts=50, cost=15.0, won=True)
        result = archive.adverse_fill_stress([trade])
        self.assertAlmostEqual(result["requested_stressed_cost"], 15.5)
        self.assertAlmostEqual(result["extra_capital_required"], 0.5)
        self.assertEqual(result["fees"], 0.75)
        self.assertEqual(result["net"], 33.75)
        self.assertEqual(result["orders_over_15_stake"], 1)
        self.assertTrue(result["contracts_unchanged"])

    def test_impossible_stressed_price_invalidates_total_without_dropping_order(self):
        trade = dict(candidate(0), contracts=15, price=0.99, cost=14.85, won=True)
        result = archive.adverse_fill_stress([trade])
        self.assertIsNone(result["net"])
        self.assertIsNone(result["fees"])
        self.assertEqual(len(result["impossible_prices"]), 1)
        self.assertEqual(result["impossible_prices"][0]["contracts"], 15)
        self.assertAlmostEqual(result["extra_capital_required"], 0.15)

    def test_family_quantiles_are_98_75_not_97_5_percent(self):
        self.assertEqual(archive.quantile_interval(list(range(10000)), 0.00625), [62, 9937])

    def test_fixed_calendar_and_empty_resamples_are_retained(self):
        for block in (1, 7):
            result = archive.bootstrap_roi([], block)
            self.assertEqual(result["calendar_days"], 27)
            self.assertEqual(result["block_days"], block)
            self.assertEqual(result["zero_risk_draws"], 10000)
            self.assertEqual(result["simultaneous_98_75"], [0.0, 0.0])


class ArchiveIntegrityTests(unittest.TestCase):
    def test_partial_partition_and_invalid_settlements_fail_closed(self):
        rows = event_rows()
        with self.assertRaisesRegex(ValueError, "partition"):
            archive.prepare_ladders(rows[:-1])
        rows[0]["outcome"] = 1
        with self.assertRaisesRegex(ValueError, "exactly one winner"):
            archive.prepare_ladders(rows)

    def test_duplicate_and_crossed_quotes_fail_closed(self):
        rows = event_rows()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            archive.prepare_ladders(rows + [rows[0]])
        rows[0]["best_bid"] = 0.9
        with self.assertRaisesRegex(ValueError, "crossed"):
            archive.prepare_ladders(rows)

    def test_leg_without_usable_quote_cannot_be_rescued_by_midpoint(self):
        rows = event_rows()
        rows[0].update(best_bid=0, best_ask=1, entry_price=0.5)
        with self.assertRaisesRegex(ValueError, "usable quote"):
            archive.prepare_ladders(rows)

    def test_ordinary_capture_data_cannot_initialize_archive_test(self):
        rows = event_rows(target="2026-09-09")
        with self.assertRaisesRegex(ValueError, "outside"):
            archive.prepare_ladders(rows)


if __name__ == "__main__":
    unittest.main()
