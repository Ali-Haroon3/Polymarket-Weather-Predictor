"""Synthetic accounting and fixed-cutoff regressions; no prospective market data."""

import copy
import datetime as dt
from decimal import Decimal
import json
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import weather_challenger as challenger
import weather_challenger_forward_metrics as metrics

PROTOCOL = json.loads((Path(__file__).resolve().parents[1] /
                       "reports/2026-10-07-challenger-preregistration.json").read_text())
FINAL = dt.datetime(2026, 12, 22, tzinfo=dt.timezone.utc)


def order(day=0, city="A", price=0.5, side="yes", target_offset=1):
    capture = dt.date(2026, 10, 8) + dt.timedelta(days=day)
    target = capture + dt.timedelta(days=target_offset)
    sized = challenger.size_order(price)
    probability = min(1.0, price + sized["fee"] / sized["contracts"] + 0.1)
    return dict(run_at=str(capture), target_date=str(target), city=city,
                ticker=f"SYNTHETIC-{city}-{target}", side=side, price=price,
                **sized, probability=probability,
                edge=probability - price - sized["fee"] / sized["contracts"],
                publication_bound=f"{capture}T15:00:00Z")


def receipt(saved, win=True, **changes):
    target = dt.date.fromisoformat(saved["target_date"])
    observed = target + dt.timedelta(days=1)
    row = dict(market_id=saved["ticker"], source="kalshi", city=saved["city"],
               target_date=saved["target_date"],
               outcome=int(win == (saved["side"] == "yes")),
               outcome_observed_at=f"{observed}T14:00:00Z")
    row.update(changes)
    return row


def study(loss_days=(8, 24), price_by_city=None):
    orders, outcomes = [], []
    for day in range(35):
        for city in ("A", "B", "C"):
            saved = order(day, city, price=(price_by_city or {}).get(city, 0.5))
            orders.append(saved)
            outcomes.append(receipt(saved, win=day not in loss_days))
    return orders, outcomes


class AccountingTests(unittest.TestCase):
    def test_exact_saved_cost_fees_payout_and_resized_stress(self):
        yes, no = order(city="A", price=0.37), order(city="B", price=0.62, side="no")
        before = copy.deepcopy([yes, no])
        result = metrics.assess(PROTOCOL, [yes, no], [receipt(yes), receipt(no, False)], FINAL)
        value = result["metrics"]
        cost = Decimal(str(yes["cost"])) + Decimal(str(no["cost"]))
        fee = Decimal(str(yes["fee"])) + Decimal(str(no["fee"]))
        net = Decimal(yes["contracts"]) - cost
        self.assertEqual(value["decimal_totals"]["cost"], str(cost))
        self.assertEqual(value["decimal_totals"]["fees"], str(fee))
        self.assertEqual(value["decimal_totals"]["settled_net"], str(net))
        self.assertEqual(value["roi"], float(net / cost))
        stressed_yes = challenger.size_order(yes["price"] + 0.01)
        stressed_no = challenger.size_order(no["price"] + 0.01)
        stress_cost = Decimal(str(stressed_yes["cost"])) + Decimal(str(stressed_no["cost"]))
        self.assertEqual(value["stress_1c"]["net"], float(Decimal(stressed_yes["contracts"]) - stress_cost))
        self.assertLess(stressed_yes["contracts"], yes["contracts"])
        self.assertEqual([yes, no], before)
        self.assertFalse(result["live_authorization"])
        self.assertTrue(result["paper_only"])
        self.assertFalse(result["coverage_verified_by_this_function"])

    def test_no_side_wins_only_when_binary_outcome_is_zero(self):
        saved = order(side="no")
        won = metrics.assess(PROTOCOL, [saved], [receipt(saved)], FINAL)
        lost = metrics.assess(PROTOCOL, [saved], [receipt(saved, False)], FINAL)
        self.assertEqual(won["metrics"]["net"], saved["contracts"] - saved["cost"])
        self.assertEqual(lost["metrics"]["net"], -saved["cost"])

    def test_all_saved_targets_define_calendar_including_unresolved_endpoints(self):
        first, middle, last = order(0), order(2), order(4)
        result = metrics.assess(PROTOCOL, [last, middle, first], [receipt(middle)], FINAL)
        value = result["metrics"]
        groups = value["calendar_groups"]
        self.assertEqual([g["target_date"] for g in groups],
                         [f"2026-10-{day:02d}" for day in range(9, 14)])
        self.assertEqual([g["orders"] for g in groups], [1, 0, 1, 0, 1])
        self.assertEqual([g["net"] for g in groups], [None, 0, middle["contracts"] - middle["cost"], 0, None])
        self.assertEqual(value["target_dates_with_orders"], 3)
        self.assertEqual(value["unresolved_cost"], first["cost"] + last["cost"])
        self.assertEqual(value["unresolved_worst_case_net"], -(first["cost"] + last["cost"]))
        self.assertEqual(value["net_if_all_unresolved_lose"],
                         float(Decimal(middle["contracts"]) - sum(Decimal(str(o["cost"])) for o in (first, middle, last))))
        for key in ("net", "roi", "bootstrap", "maximum_observed_target_day_drawdown"):
            self.assertIsNone(value[key])
        self.assertEqual(result["verdict"], "inconclusive")

    def test_group_cities_before_drawdown_and_count_traded_days_only(self):
        a, b, c = order(0, "A"), order(0, "B"), order(2, "A")
        result = metrics.assess(PROTOCOL, [a, b, c], [receipt(a), receipt(b, False), receipt(c)], FINAL)
        value = result["metrics"]
        # One same-day win/loss group has a small loss; it is not an intraday $14.99 drawdown.
        group_net = Decimal(a["contracts"]) - Decimal(str(a["cost"])) - Decimal(str(b["cost"]))
        self.assertEqual(value["calendar_groups"][0]["net"], float(group_net))
        self.assertEqual(value["maximum_observed_target_day_drawdown"], float(-group_net))
        self.assertEqual(value["strictly_negative_target_days"], 1)
        self.assertEqual(value["target_dates_with_orders"], 2)
        self.assertEqual(value["bootstrap"]["calendar_days"], 3)

    def test_invalid_economic_order_keeps_target_and_exposure_unknown(self):
        good, bad = order(0), order(2)
        bad["cost"] = float("nan")
        result = metrics.assess(PROTOCOL, [good, bad], [receipt(good)], FINAL)
        self.assertEqual(result["verdict"], "inconclusive")
        self.assertEqual(len(result["metrics"]["calendar_groups"]), 3)
        self.assertEqual(result["metrics"]["calendar_groups"][-1]["invalid_orders"], 1)
        self.assertIsNone(result["metrics"]["calendar_groups"][-1]["net"])
        self.assertFalse(result["metrics"]["exposure_complete"])
        self.assertIsNone(result["metrics"]["net_if_all_unresolved_lose"])
        json.dumps(result, allow_nan=False)


class OrderValidationTests(unittest.TestCase):
    def test_finite_scalar_economics_and_integer_contracts_are_required(self):
        changes = [dict(price=True), dict(price=float("nan")), dict(price=float("inf")),
                   dict(price=10 ** 400), dict(cost=False), dict(fee="0.50"),
                   dict(contracts=True), dict(contracts=28.0), dict(contracts=10 ** 400),
                   dict(contracts=0), dict(principal=0), dict(fee=0), dict(cost=15),
                   dict(probability=False), dict(probability=1.01), dict(edge=0.5),
                   dict(price=0.09), dict(side="invalid"), dict(city=""),
                   dict(target_date=[]), dict(run_at="2026-10-07"),
                   dict(run_at="2026-12-07"), dict(target_date="2026-10-08"),
                   dict(publication_bound="2026-10-08T15:00:00"),
                   dict(publication_bound="2026-10-09T00:00:00Z")]
        for changed in changes:
            with self.subTest(changed=changed):
                saved = dict(order(), **changed)
                result = metrics.assess(PROTOCOL, [saved], [], FINAL)
                self.assertEqual(result["verdict"], "inconclusive")
                self.assertEqual(result["order_results"][0]["state"], "invalid_order")
                json.dumps(result, allow_nan=False)

    def test_net_edge_threshold_and_native_sizing_are_not_reoptimized(self):
        saved = order()
        saved["probability"] = saved["price"] + saved["fee"] / saved["contracts"] + 0.039
        saved["edge"] = saved["probability"] - saved["price"] - saved["fee"] / saved["contracts"]
        result = metrics.assess(PROTOCOL, [saved], [receipt(saved)], FINAL)
        self.assertEqual(result["verdict"], "inconclusive")
        self.assertIn("below fixed net-edge", " ".join(result["reasons"]))

    def test_duplicate_ticker_or_event_across_capture_dates_is_invalid(self):
        first = order(0, target_offset=2)
        same_event = order(1)
        same_event["ticker"] = "DIFFERENT-TICKER"
        same_ticker = order(1, city="B")
        same_ticker["ticker"] = first["ticker"]
        for second in (same_event, same_ticker):
            with self.subTest(second=second):
                result = metrics.assess(PROTOCOL, [first, second], [receipt(first), receipt(second)], FINAL)
                self.assertEqual(result["verdict"], "inconclusive")
                self.assertIn("duplicate ticker or city/target selection", result["order_results"][1]["reasons"])

    def test_six_saved_orders_in_one_capture_day_is_invalid(self):
        saved = [order(city=str(i)) for i in range(6)]
        result = metrics.assess(PROTOCOL, saved, [receipt(o) for o in saved], FINAL)
        self.assertEqual(result["verdict"], "inconclusive")
        self.assertIn("more than five", " ".join(result["reasons"]))

    def test_protocol_changes_and_naive_assessment_clock_are_rejected(self):
        for path, replacement in [("draws", 1000), ("seed", 123), ("restart_probability", 1 / 2)]:
            changed = copy.deepcopy(PROTOCOL)
            changed["inference"][path] = replacement
            result = metrics.assess(changed, [], [], FINAL)
            self.assertEqual(result["verdict"], "inconclusive")
            self.assertIsNone(result["metrics"])
        changed = copy.deepcopy(PROTOCOL)
        changed["schema_version"] = True
        self.assertEqual(metrics.assess(changed, [], [], FINAL)["verdict"], "inconclusive")
        with self.assertRaises(ValueError):
            metrics.assess(PROTOCOL, [], [], dt.datetime(2026, 12, 22))


class OutcomeValidationTests(unittest.TestCase):
    def test_receipt_must_follow_publication_with_no_legacy_clock_fallback(self):
        saved = order()
        for stamp in (None, "bad", "2026-10-10T14:00:00", saved["publication_bound"], "2026-10-08T14:59:59Z"):
            with self.subTest(stamp=stamp):
                bad = receipt(saved, outcome_observed_at=stamp)
                bad["captured_at"] = "2026-10-11T00:00:00Z"
                result = metrics.assess(PROTOCOL, [saved], [bad, receipt(saved)], FINAL)
                self.assertEqual(result["verdict"], "inconclusive")
                self.assertEqual(result["metrics"]["settled_orders"], 0)
                self.assertEqual(len(result["order_results"][0]["matching_evidence"]), 2)

    def test_identity_or_binary_conflicts_prevent_settlement_and_retain_versions(self):
        saved = order()
        for changes in (dict(city="B"), dict(target_date="2026-10-10"), dict(outcome=True),
                        dict(outcome="1"), dict(outcome=float("nan")), dict(outcome=0.5), dict(outcome=None)):
            with self.subTest(changes=changes):
                rows = [receipt(saved), receipt(saved, **changes)]
                result = metrics.assess(PROTOCOL, [saved], rows, FINAL)
                self.assertEqual(result["verdict"], "inconclusive")
                self.assertEqual(result["metrics"]["settled_orders"], 0)
                self.assertEqual(len(result["order_results"][0]["matching_evidence"]), 2)
                json.dumps(result, allow_nan=False)
        rows = [receipt(saved), receipt(saved, False)]
        for versions in (rows, list(reversed(rows))):
            result = metrics.assess(PROTOCOL, [saved], versions, FINAL)
            self.assertIn("conflicting pre-cutoff binary outcomes", result["order_results"][0]["reasons"])
            self.assertEqual(result["metrics"]["unresolved_cost"], saved["cost"])

    def test_ordinary_unresolved_snapshot_does_not_poison_later_receipt(self):
        saved = order()
        rows = [receipt(saved, outcome=None, outcome_observed_at=None), receipt(saved), receipt(saved)]
        result = metrics.assess(PROTOCOL, [saved], rows, FINAL)
        self.assertEqual(result["metrics"]["settled_orders"], 1)
        self.assertEqual(len(result["order_results"][0]["unresolved_observations"]), 1)
        self.assertEqual(len(result["order_results"][0]["admitted_evidence"]), 2)

    def test_cutoff_is_exclusive_and_later_revision_cannot_rescue_or_replace(self):
        saved = order()
        last_valid = receipt(saved, outcome_observed_at="2026-12-21T23:59:59.999999Z")
        late = receipt(saved, False, outcome_observed_at="2026-12-22T00:00:00Z")
        for clock in (FINAL, FINAL + dt.timedelta(days=20)):
            result = metrics.assess(PROTOCOL, [saved], [last_valid, late], clock)
            self.assertEqual(result["metrics"]["wins"], 1)
            self.assertEqual(result["post_cutoff_evidence"], [late])
            result = metrics.assess(PROTOCOL, [saved], [late], clock)
            self.assertEqual(result["verdict"], "inconclusive")
            self.assertEqual(result["metrics"]["settled_orders"], 0)
        equivalent = receipt(saved, outcome_observed_at="2026-12-21T16:00:00-08:00")
        self.assertEqual(metrics.assess(PROTOCOL, [saved], [equivalent], FINAL)["metrics"]["settled_orders"], 0)

    def test_future_receipt_is_retained_but_not_used_early(self):
        saved = order()
        row = receipt(saved)
        now = dt.datetime(2026, 10, 9, tzinfo=dt.timezone.utc)
        result = metrics.assess(PROTOCOL, [saved], [row], now)
        self.assertEqual(result["verdict"], "pending")
        self.assertEqual(result["metrics"]["settled_orders"], 0)
        self.assertEqual(result["future_evidence"], [row])

    def test_other_sources_and_tickers_cannot_settle_saved_order(self):
        saved = order()
        rows = [receipt(saved, source="polymarket"), receipt(saved, market_id="OTHER")]
        result = metrics.assess(PROTOCOL, [saved], rows, FINAL)
        self.assertEqual(result["verdict"], "inconclusive")
        self.assertEqual(result["metrics"]["settled_orders"], 0)


class BootstrapTests(unittest.TestCase):
    def test_fixed_seed_joint_pairs_and_all_draws_match_independent_reference(self):
        pairs = [(10, 20), (0, 0), (-5, 10)]
        rng = random.Random(20261007)
        draws, zero_count = [], 0
        for _ in range(20000):
            indices = [rng.randrange(3)]
            for _ in range(2):
                restart = rng.random() < 1 / 7
                indices.append(rng.randrange(3) if restart else (indices[-1] + 1) % 3)
            net = sum(pairs[i][0] for i in indices)
            cost = sum(pairs[i][1] for i in indices)
            zero_count += cost == 0
            draws.append(net / cost if cost else 0)
        draws.sort()
        result = metrics.stationary_bootstrap(pairs)
        self.assertEqual(result["roi_95"], [draws[500], draws[19500]])
        self.assertEqual(result["zero_cost_draws"], zero_count)
        self.assertEqual(zero_count, 15)
        self.assertEqual(result, metrics.stationary_bootstrap(pairs))
        self.assertEqual(result["draws"], 20000)
        self.assertEqual(result["quantile_indices"], [500, 19500])

    def test_zero_cost_draws_are_zero_not_dropped_and_empty_has_no_ci(self):
        value = metrics.stationary_bootstrap([(0, 0)])
        self.assertEqual(value["zero_cost_draws"], 20000)
        self.assertEqual(value["roi_95"], [0, 0])
        self.assertIsNone(metrics.stationary_bootstrap([]))
        for bad in ([(float("nan"), 1)], [(0, -1)]):
            with self.assertRaises(ValueError):
                metrics.stationary_bootstrap(bad)


class VerdictTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orders, cls.outcomes = study()
        cls.complete = metrics.assess(PROTOCOL, cls.orders, cls.outcomes, FINAL)

    def test_sufficient_synthetic_sample_passes_only_paper_research_criteria(self):
        result = self.complete
        self.assertEqual(result["verdict"], "meets_preregistered_paper_research_criteria")
        self.assertEqual(result["metrics"]["settled_orders"], 105)
        self.assertEqual(result["metrics"]["target_dates_with_orders"], 35)
        self.assertEqual(result["metrics"]["strictly_negative_target_days"], 2)
        self.assertTrue(all(result["criteria"]["information"].values()))
        self.assertTrue(all(result["criteria"]["performance"].values()))
        self.assertFalse(result["live_authorization"])

    def test_even_all_passing_metrics_cannot_pass_before_cutoff(self):
        result = metrics.assess(PROTOCOL, self.orders, self.outcomes, FINAL - dt.timedelta(microseconds=1))
        self.assertTrue(all(result["criteria"]["performance"].values()))
        self.assertEqual(result["verdict"], "pending")

    def test_all_wins_remain_insufficient_tail_evidence(self):
        saved, rows = study(loss_days=())
        result = metrics.assess(PROTOCOL, saved, rows, FINAL)
        self.assertEqual(result["verdict"], "insufficient_sample_or_tail_evidence")
        self.assertGreater(result["metrics"]["bootstrap"]["roi_95"][0], 0)
        self.assertEqual(result["metrics"]["strictly_negative_target_days"], 0)

    def test_no_orders_are_insufficient_and_not_a_positive_zero_risk_result(self):
        result = metrics.assess(PROTOCOL, [], [], FINAL)
        self.assertEqual(result["verdict"], "insufficient_sample_or_tail_evidence")
        self.assertEqual(result["metrics"]["calendar_groups"], [])
        self.assertIsNone(result["metrics"]["bootstrap"])
        self.assertEqual(result["metrics"]["cost"], 0)

    def test_adjacent_negative_target_days_fail_drawdown_criterion(self):
        saved, rows = study(loss_days=(8, 9))
        result = metrics.assess(PROTOCOL, saved, rows, FINAL)
        self.assertEqual(result["verdict"], "does_not_meet_preregistered_paper_research_criteria")
        self.assertTrue(all(result["criteria"]["information"].values()))
        self.assertFalse(result["criteria"]["performance"]["maximum_drawdown_at_most_50"])
        self.assertGreater(result["metrics"]["maximum_observed_target_day_drawdown"], 50)
        self.assertGreater(result["metrics"]["net"], 0)

    def test_removing_best_city_must_leave_positive_net(self):
        saved, rows = study(price_by_city={"A": 0.1})
        rows = [receipt(o, win=(o["city"] == "A" and (dt.date.fromisoformat(o["run_at"]) - metrics.START).days not in (8, 24)))
                for o in saved]
        result = metrics.assess(PROTOCOL, saved, rows, FINAL)
        self.assertEqual(result["verdict"], "does_not_meet_preregistered_paper_research_criteria")
        self.assertEqual(result["metrics"]["best_city"], "A")
        self.assertLess(result["metrics"]["net_without_best_city"], 0)
        self.assertGreater(result["metrics"]["net"], 0)
        self.assertTrue(all(result["criteria"]["information"].values()))

    def test_unpriceable_stress_order_cannot_pass(self):
        native = challenger.size_order
        # Exercise the defensive branch without inventing an executable high-price order.
        with patch.object(challenger, "size_order", side_effect=lambda price: None if price == 0.51 else native(price)):
            result = metrics.assess(PROTOCOL, self.orders, self.outcomes, FINAL)
        self.assertEqual(result["metrics"]["stress_1c"]["unpriceable_orders"], 105)
        self.assertFalse(result["criteria"]["performance"]["stress_net_strictly_positive"])
        self.assertEqual(result["verdict"], "does_not_meet_preregistered_paper_research_criteria")


if __name__ == "__main__":
    unittest.main()
