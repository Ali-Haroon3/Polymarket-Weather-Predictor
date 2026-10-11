import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pilot_dashboard as dashboard


class PilotDashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ledger = self.root / "ledger.jsonl"
        self.captures = self.root / "captures.jsonl"

    def write(self, path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def order(self, ticker, **changes):
        row = dict(decision="order", strategy="market-shape", dry_run=True,
                   run_at="2026-09-29T14:55:00Z", target_date="2026-10-01",
                   ticker=ticker, city="Dallas", side="yes", contracts=10,
                   price=0.4, cost=4.0)
        return dict(row, **changes)

    def capture(self, ticker, outcome=1, **changes):
        return dict(dict(source="kalshi", captured_at="2026-10-02",
                         target_date="2026-10-01", market_id=ticker,
                         outcome=outcome), **changes)

    def fill(self, order, **changes):
        row = dict(order, decision="fill", reconciliation_verified=True,
                   order_status="executed", contracts=6, price=0.35, cost=2.1)
        return dict(row, **changes)

    def test_modes_and_verified_execution_remain_separate(self):
        live = self.order("live", dry_run=False, order_id="live-id")
        unverified = self.order("unknown", dry_run=False, order_id="unknown-id")
        nofill = self.order("nofill", dry_run=False, order_id="nofill-id")
        legacy = self.order("legacy", strategy="model-shrunk")
        rows = [self.order("paper"), live, self.fill(live), unverified,
                self.fill(unverified, reconciliation_verified=False), nofill,
                self.fill(nofill, decision="unfilled", order_status="canceled",
                          contracts=0, price=None, cost=0), legacy]
        self.write(self.ledger, rows)
        self.write(self.captures, [self.capture("paper", 0), self.capture("live"),
                                  self.capture("unknown"), self.capture("nofill"),
                                  self.capture("legacy")])
        result = dashboard.build_readout(self.ledger, self.captures)
        self.assertEqual(result["paper"]["settled"], 1)
        self.assertAlmostEqual(result["paper"]["net"], -4.17)
        self.assertEqual(result["paper"]["open"], 0)
        self.assertEqual(result["live"]["settled"], 1)
        self.assertEqual(result["live"]["wins"], 1)
        self.assertAlmostEqual(result["live"]["principal"], 2.1)
        self.assertAlmostEqual(result["live"]["fees"], 0.1)
        self.assertAlmostEqual(result["live"]["net"], 3.8)
        self.assertAlmostEqual(result["live"]["roi"], 3.8 / 2.1)
        self.assertEqual(result["live"]["open"], 1)
        self.assertEqual(result["live"]["unfilled"], 1)
        self.assertEqual(result["unverified_live_orders"], 1)
        self.assertFalse(result["gate_passed"])

    def test_other_strategy_unverified_live_blocks_gate_without_polluting_market_shape(self):
        paper = [self.order(f"paper-{i}") for i in range(100)]
        self.write(self.ledger, paper)
        self.write(self.captures, [self.capture(row["ticker"]) for row in paper]
                   + [self.capture("legacy-live")])
        self.assertTrue(dashboard.build_readout(self.ledger, self.captures)["gate_passed"])
        self.write(self.ledger, paper + [self.order("legacy-live", strategy="model-shrunk",
                                                   dry_run=False, order_id="legacy-live-id")])
        result = dashboard.build_readout(self.ledger, self.captures)
        self.assertFalse(result["gate_passed"])
        self.assertEqual(result["unverified_live_orders"], 1)
        self.assertEqual(result["paper"]["settled"], 100)
        self.assertEqual(result["live"]["open"], 0)
        self.assertEqual(result["live"]["settled"], 0)
        self.assertIsNone(result["live"]["net"])
        self.assertIsNone(result["live"]["roi"])

    def test_empty_sample_is_not_zero_profit(self):
        self.write(self.ledger, [])
        self.write(self.captures, [self.capture("irrelevant")])
        result = dashboard.build_readout(self.ledger, self.captures)
        for mode in ("paper", "live"):
            self.assertIsNone(result[mode]["net"])
            self.assertIsNone(result[mode]["roi"])
            self.assertEqual(result[mode]["settled"], 0)
        self.assertIsNone(result["latest_order_run"])

    def test_unresolved_paper_intention_not_settled(self):
        self.write(self.ledger, [self.order("open")])
        self.write(self.captures, [self.capture("open", None)])
        result = dashboard.build_readout(self.ledger, self.captures)
        self.assertEqual(result["paper"]["open"], 1)
        self.assertIsNone(result["paper"]["net"])
        self.assertEqual(result["latest_capture"], "2026-10-02")
        self.assertEqual(result["latest_order_run"], "2026-09-29T14:55:00Z")

    def test_hashes_cover_exact_single_read_bytes(self):
        self.write(self.ledger, [self.order("paper")])
        self.write(self.captures, [self.capture("paper")])
        originals = {p: p.read_bytes() for p in (self.ledger, self.captures)}
        reads = []
        def read_once(path):
            reads.append(path)
            if reads.count(path) > 1:
                raise AssertionError("input was reread")
            return originals[path]
        with mock.patch.object(Path, "read_bytes", read_once):
            result = dashboard.build_readout(self.ledger, self.captures)
        self.assertCountEqual(reads, originals)
        for key, path in (("ledger", self.ledger), ("captures", self.captures)):
            self.assertEqual(result["inputs"][key]["sha256"], hashlib.sha256(originals[path]).hexdigest())
            self.assertEqual(result["inputs"][key]["bytes"], len(originals[path]))

    def test_missing_empty_malformed_and_conflicting_captures_fail(self):
        self.write(self.ledger, [self.order("paper")])
        with self.assertRaises(OSError):
            dashboard.build_readout(self.ledger, self.captures)
        bad_inputs = ["", "{bad}\n", "[]\n", '{"x":1,"x":2}\n',
                      json.dumps(self.capture("paper", True)) + "\n",
                      json.dumps(self.capture("paper", captured_at="2026-99-01")) + "\n",
                      json.dumps(self.capture("paper", 0)) + "\n" + json.dumps(self.capture("paper", 1)) + "\n"]
        for raw in bad_inputs:
            with self.subTest(raw=raw):
                self.captures.write_text(raw)
                with self.assertRaises((ValueError, KeyError)):
                    dashboard.build_readout(self.ledger, self.captures)

    def test_malformed_ledger_evidence_fails(self):
        self.write(self.captures, [self.capture("paper")])
        for changes in ({"dry_run": "false"}, {"contracts": True}, {"price": float("nan")},
                        {"run_at": "2026-09-29T12:00:00"}, {"cost": 3.0}):
            with self.subTest(changes=changes):
                self.write(self.ledger, [self.order("paper", **changes)])
                with self.assertRaises(ValueError):
                    dashboard.build_readout(self.ledger, self.captures)
        self.write(self.ledger, [self.order("paper"), self.order("paper")])
        with self.assertRaises(ValueError):
            dashboard.build_readout(self.ledger, self.captures)

    def test_cli_failure_is_nonzero_without_json_or_zero_profit(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = dashboard.main(["--ledger", str(self.ledger), "--captures", str(self.captures)])
        self.assertEqual(code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("unavailable", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
