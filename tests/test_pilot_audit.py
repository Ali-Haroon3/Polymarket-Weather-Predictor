"""Offline regressions for live admission and prospective alpha accounting."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import go_live_gate as gate
import pilot_alpha_audit as audit


def order(ticker="TEST", live=False):
    return dict(decision="order", strategy="market-shape", ticker=ticker, city="NYC",
                run_at="2026-09-08T15:00:00Z", target_date="2026-09-09", dry_run=not live,
                side="no", price=0.60, contracts=10, cost=6.0,
                order_id=ticker if live else None)


def outcome(ticker="TEST", value=0):
    return dict(source="kalshi", market_id=ticker, target_date="2026-09-09", outcome=value)


class GateTests(unittest.TestCase):
    def test_live_intentions_never_count_as_settled_profit(self):
        settled, opened, unfilled = gate.settle([order(live=True)], [outcome()])
        self.assertEqual(settled, [])
        self.assertEqual(len(opened), 1)
        self.assertFalse(unfilled)

    def test_reconciled_partial_fills_and_zero_fills(self):
        filled, empty = order("F", True), order("E", True)
        rec = dict(filled, decision="fill", contracts=4, cost=2.20, price=0.55)
        zero = dict(empty, decision="unfilled", contracts=0, cost=0.0, price=None)
        settled, _, unfilled = gate.settle([filled, empty, rec, zero], [outcome("F"), outcome("E")])
        self.assertEqual(len(settled), 1)
        self.assertAlmostEqual(settled[0]["gross"], 1.8)
        self.assertAlmostEqual(gate.live_readout(settled, unfilled)["fill_ratio"], 4 / 20)

    def test_yes_and_legacy_no_settlement(self):
        yes = dict(order("Y"), side="yes", price=0.3, cost=3.0)
        legacy = order("N")
        legacy.pop("strategy")
        legacy.pop("side")
        legacy["no_price"] = legacy.pop("price")
        settled, _, _ = gate.settle([yes, legacy], [outcome("Y", 1), outcome("N", 0)])
        self.assertEqual(len(settled), 1)
        self.assertAlmostEqual(settled[0]["net"], 6.85)
        old, _, _ = gate.settle([yes, legacy], [outcome("N", 0)], "model-shrunk")
        self.assertAlmostEqual(old[0]["net"], 3.83)

    def test_duplicate_or_conflicting_evidence_fails_closed(self):
        with self.assertRaises(ValueError):
            gate.settle([order(), order()], [outcome()])
        with self.assertRaises(ValueError):
            gate.settle([order()], [outcome(value=0), outcome(value=1)])
        with self.assertRaises(ValueError):
            gate.settle([dict(order(), cost=float("nan"))], [outcome()])

    def test_gate_boundaries_keep_precommitted_thresholds(self):
        won = dict(cost=10.0, net=1.0, fee=0.10, won=True, city="NYC")
        self.assertFalse(gate.evaluate([won] * 99)["criteria"]["1_sample_and_sign"]["pass"])
        self.assertTrue(gate.evaluate([won] * 100)["criteria"]["1_sample_and_sign"]["pass"])
        zero = dict(won, net=0.0)
        self.assertFalse(gate.evaluate([zero] * 100)["criteria"]["1_sample_and_sign"]["pass"])
        loss = dict(won, net=-1.0, won=False)
        self.assertFalse(gate.evaluate([won] * 100 + [loss])["criteria"]["2_no_single_city_bleed"]["pass"])

    def test_pending_live_orders_block_even_a_profitable_paper_sample(self):
        orders = [order(str(i)) for i in range(100)]
        outcomes = [outcome(str(i)) for i in range(100)]
        self.assertTrue(gate.report(orders, outcomes)["go_live"])
        orders.append(dict(order("OLD", live=True), strategy="model-shrunk"))
        r = gate.report(orders, outcomes)
        self.assertFalse(r["go_live"])
        self.assertEqual(r["unverified_live_orders"], 1)

    def test_enforce_exit_status_and_machine_readout(self):
        with tempfile.TemporaryDirectory() as d:
            ledger, caps = Path(d) / "ledger.jsonl", Path(d) / "captures.jsonl"
            ledger.write_text(json.dumps(order()) + "\n")
            caps.write_text(json.dumps(outcome()) + "\n")
            args = [sys.executable, str(Path(gate.__file__)), "--ledger", str(ledger),
                    "--captures", str(caps), "--json"]
            advisory = subprocess.run(args, capture_output=True, text=True)
            enforced = subprocess.run(args + ["--enforce"], capture_output=True, text=True)
            self.assertEqual(advisory.returncode, 0)
            self.assertEqual(enforced.returncode, 1)
            self.assertFalse(json.loads(enforced.stdout)["go_live"])
            caps.write_text("broken json\n")
            self.assertNotEqual(subprocess.run(args + ["--enforce"], capture_output=True).returncode, 0)


class AlphaAuditTests(unittest.TestCase):
    def candidates(self):
        return [dict(run_at="2026-09-08", target_date="2026-09-09", ticker=str(i),
                     city=str(i), side="yes" if i < 3 else "no", price=0.6,
                     edge=0.2 - i / 100, outcome=0) for i in range(8)]

    def test_unresolved_candidates_consume_capacity(self):
        rows = self.candidates()
        rows[0]["outcome"] = None
        trades, opened = audit.select_candidates(rows)
        self.assertEqual(opened, 1)
        self.assertEqual([t["ticker"] for t in trades], ["1", "2", "3", "4"])
        changed = copy.deepcopy(rows)
        for t in changed:
            t["outcome"] = 1
        trades2, _ = audit.select_candidates(changed)
        self.assertEqual([t["ticker"] for t in trades2], [str(i) for i in range(5)])

    def test_no_only_replacement_is_not_attribution(self):
        rows = self.candidates()
        both, _ = audit.select_candidates(rows)
        no, _ = audit.select_candidates(rows, "no")
        self.assertEqual(sum(t["side"] == "no" for t in both), 2)
        self.assertEqual(len(no), 5)

    def test_city_cap_and_ticker_dedupe_span_runs(self):
        rows = self.candidates()
        for t in rows:
            t["city"] = "NYC"
        rows.append(dict(rows[0], run_at="2026-09-09"))
        trades, _ = audit.select_candidates(rows)
        self.assertEqual(len(trades), 2)
        self.assertLessEqual(sum(t["cost"] for t in trades), 30.0)

    def test_uncertainty_clusters_correlated_trades_by_target_day(self):
        trades, _ = audit.select_candidates(self.candidates())
        result = audit.summarize(trades)
        self.assertEqual(result["target_days"], 1)
        self.assertIsNone(result["bootstrap_95"])
        self.assertEqual(result, audit.summarize(trades))


if __name__ == "__main__":
    unittest.main()
