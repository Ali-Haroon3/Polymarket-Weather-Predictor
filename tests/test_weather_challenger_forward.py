"""Synthetic composition/CLI checks; never inspect prospective performance.

Boundary checks supply a hypothetical passing assessment to challenge coverage
and clock gates. End-to-end checks use only generated archives, quotes and
settlements to exercise the real evidence and accounting modules together.
"""
import contextlib
import copy
import datetime as dt
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import weather_challenger_forward as forward
from test_weather_challenger_forward_evidence import Fixture, NOW as SYNTHETIC_NOW


class ChallengerForwardEndToEndTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.protocol = forward.load_registration()
        self.fixture = Fixture(self.root)
        outcomes = []
        for index, (run_id, shadow) in enumerate(self.fixture.shadows.items()):
            capture_day = shadow["latest_capture"]
            first = shadow["families"]["scale_only"]["selected"][0]
            second = dict(first, city="CHI", ticker=first["ticker"] + "-CHI")
            for family in shadow["families"].values():
                family["selected"] = [copy.deepcopy(first), copy.deepcopy(second)]
            quotes = [dict(source="kalshi", captured_at=capture_day,
                           target_date=order["target_date"], city=order["city"],
                           market_id=order["ticker"], outcome=None,
                           best_ask=0.5, best_bid=0.49)
                      for order in (first, second)]
            self.fixture.rewrite_capture(capture_day, quotes)
            for order in (first, second):
                # Two separated losing target days satisfy the frozen tail
                # minimum without exceeding its observed drawdown threshold.
                outcomes.append(dict(source="kalshi", market_id=order["ticker"],
                                     city=order["city"], target_date=order["target_date"],
                                     outcome=0 if index in (10, 30) else 1,
                                     outcome_observed_at=order["target_date"] + "T16:00:00Z"))
        self.fixture.inventory["outcomes"] = self.fixture.file(
            "outcomes.jsonl", ("\n".join(map(json.dumps, outcomes)) + "\n").encode())

    def evaluate(self):
        self.fixture.file("inventory.json", json.dumps(self.fixture.inventory).encode())
        return forward.evaluate(self.root, self.protocol, SYNTHETIC_NOW)

    def test_complete_generated_trial_passes_only_paper_research_criteria(self):
        result = self.evaluate()
        self.assertEqual(result["verdict"], "meets_preregistered_paper_research_criteria")
        self.assertIs(result["complete_evidence"], True)
        self.assertEqual(len(result["evidence"]["days"]), 60)
        self.assertEqual(result["evidence"]["saved_primary_orders"], 120)
        self.assertEqual(result["assessment"]["metrics"]["settled_orders"], 120)
        self.assertIs(result["live_authorization"], False)
        self.assertIs(result["assessment"]["live_authorization"], False)
        self.assertIs(result["assessment"]["paper_only"], True)

    def test_separate_winning_outcome_cannot_hide_loss_in_exact_capture_input(self):
        capture_day = "2026-10-10"
        record = self.fixture.capture_records[capture_day]
        rows = [json.loads(line) for line in (self.root / record["path"]).read_text().splitlines()]
        rows.append(dict(source="kalshi", captured_at="2026-10-08",
                         target_date="2026-10-09", city="NYC", market_id="T-2026-10-09",
                         outcome=0, outcome_observed_at="2026-10-09T16:00:00Z",
                         best_ask=0.5, best_bid=0.49))
        self.fixture.rewrite_capture(capture_day, rows)
        result = self.evaluate()
        self.assertEqual(result["verdict"], "inconclusive")
        self.assertEqual(result["evidence"]["saved_primary_orders"], 120)
        self.assertIs(result["assessment"]["metrics"]["accounting_complete"], False)
        conflicted = next(row for row in result["assessment"]["order_results"]
                          if row["saved_order"]["ticker"] == "T-2026-10-09")
        self.assertIsNone(conflicted["outcome"])
        self.assertIn("conflicting pre-cutoff binary outcomes", conflicted["reasons"])
        self.assertEqual({row["outcome"] for row in conflicted["admitted_evidence"]}, {0, 1})
        self.assertIs(result["live_authorization"], False)


class ChallengerForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registration_bytes = forward.REGISTRATION_PATH.read_bytes()
        cls.protocol = json.loads(cls.registration_bytes)
        cls.final_at = forward.clock(cls.protocol["calendar"]["earliest_final_assessment"])
        cls.pass_verdict = cls.protocol["verdicts"]["complete_and_all_criteria_pass"]

    def complete_evidence(self):
        first = dt.date.fromisoformat(self.protocol["calendar"]["capture_date_from"])
        return {
            "state": "valid",
            "days": [{"date": str(first + dt.timedelta(days=i)), "state": "valid"}
                     for i in range(self.protocol["calendar"]["capture_days"])],
            "provenance": {"coverage_complete": True, "inventory_window_closed": True},
            "orders": [{"synthetic_fixture": "saved selection; never scored"}],
            "outcomes": [{"synthetic_fixture": "saved outcome; never scored"}],
        }

    def evaluate_fixture(self, observed, now=None, metric_verdict=None):
        assessment = {"verdict": metric_verdict or self.pass_verdict,
                      "synthetic_fixture": "hypothetical economics only"}
        with patch.object(forward.evidence, "load_evidence", return_value=observed), \
                patch.object(forward.metrics, "assess", return_value=assessment):
            result = forward.evaluate(Path("unused-synthetic-evidence"), self.protocol,
                                      now or self.final_at)
        self.assertIs(result["live_authorization"], False)
        return result

    def test_incomplete_day_coverage_cannot_promote_passing_metrics(self):
        complete = self.complete_evidence()
        cases = {}
        cases["missing day"] = copy.deepcopy(complete)
        cases["missing day"]["days"].pop()
        cases["missing all day evidence"] = copy.deepcopy(complete)
        del cases["missing all day evidence"]["days"]
        cases["unknown day"] = copy.deepcopy(complete)
        cases["unknown day"]["days"][7]["state"] = "unknown"
        cases["duplicate replaces missing date"] = copy.deepcopy(complete)
        cases["duplicate replaces missing date"]["days"][-1] = copy.deepcopy(complete["days"][0])
        cases["extra duplicate"] = copy.deepcopy(complete)
        cases["extra duplicate"]["days"].append(copy.deepcopy(complete["days"][0]))
        cases["unknown inventory"] = copy.deepcopy(complete)
        cases["unknown inventory"]["state"] = "unknown"
        for label, observed in cases.items():
            with self.subTest(label=label):
                result = self.evaluate_fixture(observed)
                self.assertEqual(result["assessment"]["verdict"], self.pass_verdict)
                self.assertIs(result["complete_evidence"], False)
                self.assertEqual(result["verdict"], "inconclusive")

    def test_valid_days_still_require_explicit_coverage_and_closed_inventory(self):
        for field in ("coverage_complete", "inventory_window_closed"):
            for value in (False, None, 1, "true"):
                with self.subTest(field=field, value=value):
                    observed = self.complete_evidence()
                    if value is None:
                        del observed["provenance"][field]
                    else:
                        observed["provenance"][field] = value
                    result = self.evaluate_fixture(observed)
                    self.assertIs(result["complete_evidence"], False)
                    self.assertEqual(result["verdict"], "inconclusive")

    def test_before_cutoff_remains_pending_even_with_passing_metrics(self):
        for complete in (True, False):
            with self.subTest(complete=complete):
                observed = self.complete_evidence()
                if not complete:
                    observed["days"].pop()
                result = self.evaluate_fixture(observed, self.final_at - dt.timedelta(microseconds=1))
                self.assertEqual(result["verdict"], "pending")

    def test_complete_sixty_days_at_cutoff_preserve_economic_verdict_without_live_authority(self):
        observed = self.complete_evidence()
        self.assertEqual(len(observed["days"]), 60)
        for verdict in (self.pass_verdict,
                        self.protocol["verdicts"]["insufficient_minimum_information"],
                        self.protocol["verdicts"]["complete_but_any_performance_or_risk_criterion_fails"]):
            with self.subTest(verdict=verdict):
                result = self.evaluate_fixture(observed, self.final_at, verdict)
                self.assertIs(result["complete_evidence"], True)
                self.assertEqual(result["verdict"], verdict)

    def isolated_registration(self, root):
        registration = root / "registration.json"
        registration.write_bytes(self.registration_bytes)
        (root / "scripts").mkdir()
        for name in self.protocol["decision_policy"]["policy_code_sha256"]:
            (root / "scripts" / name).write_bytes((ROOT / "scripts" / name).read_bytes())
        return registration

    def test_registration_and_all_three_implementation_hashes_are_enforced(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            registration = self.isolated_registration(root)
            with patch.object(forward, "ROOT", root), \
                    patch.object(forward, "REGISTRATION_PATH", registration):
                self.assertEqual(forward.load_registration(), self.protocol)
                names = self.protocol["decision_policy"]["policy_code_sha256"]
                self.assertEqual(len(names), 3)
                for name in names:
                    with self.subTest(implementation=name):
                        path = root / "scripts" / name
                        original = path.read_bytes()
                        path.write_bytes(original + b"\n# synthetic changed implementation\n")
                        try:
                            with self.assertRaisesRegex(ValueError, "decision implementation changed"):
                                forward.load_registration()
                        finally:
                            path.write_bytes(original)

    def test_changed_registration_bytes_fail_before_evaluation_or_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            registration = self.isolated_registration(root)
            # Even semantically identical JSON must preserve the committed bytes.
            registration.write_bytes(self.registration_bytes + b"\n")
            output = root / "report.json"
            errors = io.StringIO()
            with patch.object(forward, "ROOT", root), \
                    patch.object(forward, "REGISTRATION_PATH", registration), \
                    patch.object(forward, "evaluate") as evaluate, \
                    contextlib.redirect_stderr(errors):
                code = forward.main(["--evidence", str(root / "evidence"), "--output", str(output)])
            self.assertEqual(code, 2)
            evaluate.assert_not_called()
            self.assertIn("registration bytes", errors.getvalue())
            self.assertFalse(output.exists())

    def run_output_fixture(self, evidence, output):
        stdout, stderr = io.StringIO(), io.StringIO()
        report = {"verdict": "pending", "live_authorization": False,
                  "synthetic_fixture": "CLI persistence test; no scoring"}
        with patch.object(forward, "load_registration", return_value=self.protocol), \
                patch.object(forward, "evaluate", return_value=report), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = forward.main(["--evidence", str(evidence), "--output", str(output)])
        return code, stdout.getvalue(), stderr.getvalue(), report

    def test_cli_future_clock_rejected_before_registration_evidence_or_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "report.json"
            future = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=1)
            with patch.object(forward, "load_registration") as registration, \
                    patch.object(forward, "evaluate") as evaluate, \
                    contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as caught:
                    forward.main(["--evidence", str(root / "evidence"),
                                  "--as-of", future.isoformat(), "--output", str(output)])
            self.assertEqual(caught.exception.code, 2)
            self.assertIn("future assessment", stderr.getvalue())
            registration.assert_not_called()
            evaluate.assert_not_called()
            self.assertFalse(output.exists())

    def test_cli_existing_report_and_input_paths_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            evidence = root / "evidence"
            evidence.mkdir()
            original = b"preserved original bytes\n"
            existing_report = root / "existing-report.json"
            input_file = evidence / "inventory.json"
            existing_report.write_bytes(original)
            input_file.write_bytes(original)
            linked_input = root / "linked-input.json"
            linked_input.symlink_to(input_file)
            for output in (existing_report, input_file, linked_input, evidence / "new-report.json"):
                with self.subTest(output=output.name):
                    code, _, _, _ = self.run_output_fixture(evidence, output)
                    self.assertEqual(code, 2)
                    self.assertEqual(existing_report.read_bytes(), original)
                    self.assertEqual(input_file.read_bytes(), original)
                    self.assertFalse((evidence / "new-report.json").exists())

    def test_cli_new_separate_report_preserves_explicit_no_live_authority(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            evidence = root / "evidence"
            evidence.mkdir()
            output = root / "reports" / "result.json"
            code, stdout, stderr, expected = self.run_output_fixture(evidence, output)
            self.assertEqual(code, 0)
            self.assertEqual(stderr, "")
            self.assertEqual(json.loads(output.read_text()), expected)
            self.assertIs(json.loads(stdout)["live_authorization"], False)


if __name__ == "__main__":
    unittest.main()
