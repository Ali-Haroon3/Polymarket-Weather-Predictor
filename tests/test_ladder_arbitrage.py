"""Offline regression tests for basket topology, quotes, fees, and integer sizing."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ladder_arbitrage_audit as audit


def ladder():
    common = dict(source="kalshi", city="Test", captured_at="2026-09-21",
                  target_date="2026-09-22", unit="F", best_bid=0.30, best_ask=0.40)
    return [dict(common, market_id="LOW", market_type="temp_at_most", threshold=79),
            dict(common, market_id="MID", market_type="temp_bucket", threshold=80,
                 threshold_upper=81),
            dict(common, market_id="HIGH", market_type="temp_at_least", threshold=82)]


class LadderArbitrageTests(unittest.TestCase):
    def test_full_partition_needs_both_tails_no_gaps_or_overlaps(self):
        rows = ladder()
        self.assertEqual([r["market_id"] for r in audit.full_partition(rows[::-1])],
                         ["LOW", "MID", "HIGH"])
        self.assertIsNone(audit.full_partition(rows[:-1]))
        for threshold in (81, 83):
            invalid = copy.deepcopy(rows)
            invalid[-1]["threshold"] = threshold
            self.assertIsNone(audit.full_partition(invalid))
        invalid = copy.deepcopy(rows)
        invalid[1]["market_id"] = "LOW"
        self.assertIsNone(audit.full_partition(invalid))

    def test_missing_quotes_never_fall_back_to_entry_prices(self):
        rows = ladder()
        rows[0]["best_ask"] = None
        rows[0]["entry_price"] = 0.01
        self.assertIsNone(audit.executable_prices(rows, "yes"))
        self.assertEqual(audit.executable_prices(rows, "no"), [0.7, 0.7, 0.7])
        rows[1]["best_bid"] = None
        self.assertIsNone(audit.executable_prices(rows, "no"))

    def test_crossed_or_nonfinite_quotes_are_rejected(self):
        rows = ladder()
        rows[0]["best_bid"] = 0.5
        self.assertIsNone(audit.executable_prices(rows, "yes"))
        self.assertIsNone(audit.executable_prices(rows, "no"))
        rows[0]["best_bid"] = 0.3
        rows[0]["best_ask"] = float("nan")
        self.assertIsNone(audit.executable_prices(rows, "yes"))

    def test_fees_can_erase_positive_gross_arbitrage(self):
        basket = audit.best_basket([0.48, 0.48], 1.0, 1.0)
        self.assertEqual(basket["quantity"], 1)
        self.assertAlmostEqual(basket["fees"], 0.04)
        self.assertAlmostEqual(basket["net"], 0.0)
        self.assertIsNone(audit.best_basket([0.48, 0.48], 1.0, 0.99))

    def test_favorable_basket_includes_fees_in_budget(self):
        basket = audit.best_basket([0.4, 0.4], 1.0, 15.0)
        self.assertEqual(basket["quantity"], 17)
        self.assertAlmostEqual(basket["fees"], 0.58)
        self.assertAlmostEqual(basket["net"], 2.82)
        self.assertLessEqual(basket["total_cost"], 15.0)
        stress = basket["single_leg_adverse_cent"]
        self.assertAlmostEqual(stress["fees"], 0.58)
        self.assertAlmostEqual(stress["net"], 2.65)

    def test_search_checks_smaller_quantities_when_rounding_reduces_profit(self):
        # At n=3 fee rounding consumes a cent more than the small gross increment.
        basket = audit.best_basket([0.06, 0.93], 1.0, 3.10)
        values = [(n, n - audit.basket_cost([0.06, 0.93], n)[2]) for n in (1, 2, 3)]
        self.assertGreater(values[1][1], values[2][1])
        self.assertEqual(basket["quantity"], 2)
        self.assertEqual(basket["max_affordable_quantity"], 3)

    def test_adverse_single_leg_stress_recalculates_rounded_fee(self):
        basket = audit.best_basket([0.17, 0.80], 1.0, 1.0)
        self.assertAlmostEqual(basket["fees"], 0.03)
        stress = basket["single_leg_adverse_cent"]
        self.assertEqual(stress["leg_index"], 0)
        self.assertAlmostEqual(stress["fees"], 0.04)
        self.assertAlmostEqual(stress["net"], -0.02)
        self.assertFalse(stress["within_budget"])

    def test_no_basket_pays_n_minus_one_independent_of_outcomes(self):
        rows = ladder()
        for row in rows:
            row["best_bid"], row["best_ask"] = 0.4, 0.5
        result = audit.audit(rows, [15.0])
        candidates = result["budgets"][0]["positive_candidates"]
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["side"], "no")
        self.assertEqual(candidates[0]["payout_per_basket"], 2.0)
        changed = [dict(row, outcome=0.0) for row in rows]
        self.assertEqual(result, audit.audit(changed, [15.0]))

    def test_only_lead_one_or_more_kalshi_is_eligible(self):
        rows = ladder()
        rows += [dict(row, source="polymarket") for row in ladder()]
        rows += [dict(row, captured_at="2026-09-22") for row in ladder()]
        result = audit.audit(rows)
        self.assertEqual(result["eligible_groups"], 1)
        self.assertEqual(result["full_partitions"], 1)


if __name__ == "__main__":
    unittest.main()
