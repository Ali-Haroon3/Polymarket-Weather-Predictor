"""Offline checks for frozen scheduling and retention of failed observations."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import weather_source_schedule as schedule


def instant(value):
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


class FixtureCollector:
    """Writes representative evidence without importing or invoking the collector."""

    def __init__(self, target, output, protocol, started, *, complete=True, error=None):
        self.target, self.output = target, output
        self.complete, self.error, self.calls = complete, error, 0
        self.output.mkdir()
        (self.output / "responses").mkdir()
        shutil.copyfile(protocol, self.output / "protocol.md")
        self.manifest = {"started_at_utc": schedule.stamp(started), "complete": False,
                         "stop_reason": None, "requests": []}
        self.checkpoint()

    def checkpoint(self):
        schedule.write_json(self.output / "manifest.json", self.manifest)

    def run(self):
        self.calls += 1
        (self.output / "responses" / "0001.body").write_bytes(b' {"private_fixture_value": 71}\r\n')
        self.manifest.update(complete=self.complete and self.error is None,
                             stop_reason="fixture_failure" if self.error or not self.complete else None,
                             requests=[{"raw_file": "responses/0001.body"}])
        self.checkpoint()
        if self.error:
            raise self.error
        return self.manifest


class WeatherSourceScheduleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        for relative in (schedule.PROTOCOL, schedule.COLLECTOR):
            target = self.repo / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(schedule.ROOT / relative, target)
        self.metadata = self.root / "github-run.json"
        self.created = instant("2026-09-26T13:15:00Z")
        self.write_metadata()
        # Every test must use an explicit fake; an accidental real import fails closed.
        blocker = patch.object(schedule, "load_collector", side_effect=AssertionError("real collector forbidden"))
        blocker.start()
        self.addCleanup(blocker.stop)

    def write_metadata(self, **changes):
        value = {"id": 101, "run_attempt": 1, "event": "schedule",
                 "created_at": schedule.stamp(self.created),
                 "run_started_at": schedule.stamp(self.created),
                 "head_sha": "a" * 40, "html_url": "https://github.com/example/repo/actions/runs/101"}
        value.update(changes)
        self.metadata.write_text(json.dumps(value), encoding="utf-8")

    def run_fixture(self, *, actual=None, initialized=None, gate=None, complete=True,
                    error=None, attempt=1, name="result"):
        actual = actual or self.created
        initialized = initialized or actual
        gate = gate or actual
        clocks = iter((actual, actual, gate, gate))
        instances = []

        def create(target, output, protocol):
            item = FixtureCollector(target, output, protocol, initialized, complete=complete, error=error)
            instances.append(item)
            return item

        factory = Mock(side_effect=create)
        status, code = schedule.run_schedule(self.metadata, self.root / name, 101, attempt,
                                             repo_root=self.repo, now=lambda: next(clocks),
                                             collector_factory=factory)
        self.assertEqual(json.loads((self.root / name / "run-status.json").read_bytes()), status)
        return status, code, factory, instances

    def test_exact_date_year_midnight_and_inclusive_tolerance_boundaries(self):
        for stamp, target, phase in (
            ("2026-09-26T13:00:00Z", "2026-09-26", "development"),
            ("2026-09-26T13:30:00Z", "2026-09-26", "development"),
            ("2026-09-27T01:15:00Z", "2026-09-26", "development"),
            ("2026-10-01T01:15:00Z", "2026-09-30", "development"),
            ("2026-10-10T09:15:00Z", "2026-10-09", "development"),
            ("2026-10-10T13:15:00Z", "2026-10-10", "reserved_validation"),
            ("2026-10-24T09:30:00Z", "2026-10-23", "reserved_validation"),
            ("2026-09-26T19:15:00-06:00", "2026-09-26", "development"),
        ):
            with self.subTest(stamp=stamp):
                slot = schedule.slot_at(instant(stamp))
                self.assertEqual((slot["target_date"], slot["phase"]), (target, phase))
        for stamp in ("2026-09-26T12:59:59.999999Z", "2026-09-26T13:30:00.000001Z",
                      "2026-09-26T09:15:00Z", "2026-09-27T00:00:00Z",
                      "2026-10-24T09:30:00.000001Z", "2026-10-24T13:15:00Z",
                      "2025-09-26T13:15:00Z", "2027-09-26T13:15:00Z"):
            with self.subTest(stamp=stamp):
                self.assertIsNone(schedule.slot_at(instant(stamp)))
        with self.assertRaises(ValueError):
            schedule.slot_at(dt.datetime(2026, 9, 26, 13, 15))

    def test_workflow_calendar_covers_exactly_168_slots_and_is_read_only(self):
        text = (schedule.ROOT / ".github/workflows/weather-source-research.yml").read_text()
        expressions = re.findall(r'cron: "([^"]+)"', text)
        self.assertEqual(len(expressions), 5)
        actual = []
        for expression in expressions:
            minute, hours, days, month, weekday = expression.split()
            self.assertEqual(weekday, "*")
            first, last = (int(value) for value in days.split("-")) if "-" in days else (int(days), int(days))
            actual.extend(dt.datetime(2026, int(month), day, int(hour), int(minute), tzinfo=schedule.UTC)
                          for day in range(first, last + 1) for hour in hours.split(","))
        expected = [self.created + dt.timedelta(hours=4 * i) for i in range(168)]
        self.assertEqual(sorted(actual), expected)
        for value in expected:
            expected_target = value.date() - dt.timedelta(days=value.hour < 12)
            self.assertEqual(schedule.slot_at(value)["target_date"], expected_target.isoformat())
        self.assertNotIn("workflow_dispatch", text)
        self.assertNotIn("secrets.", text)
        self.assertIn("contents: read\n  actions: read", text)
        self.assertIn("persist-credentials: false", text)
        self.assertIn("cancel-in-progress: false", text)
        self.assertIn("retention-days: 90", text)
        self.assertIn("overwrite: false", text)
        self.assertEqual(text.count("GH_TOKEN:"), 1)
        self.assertEqual(text.count("if: always()"), 2)
        self.assertIn("actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1", text)
        self.assertIn("actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a", text)
        bootstrap = text.split("          python3 - <<'PY'\n", 1)[1].split("          PY", 1)[0]
        compile("\n".join(line[10:] for line in bootstrap.splitlines()), "workflow bootstrap", "exec")

    def test_reserved_capture_preserves_exact_evidence_without_body_values_in_status(self):
        self.created = instant("2026-10-10T13:15:00Z")
        self.write_metadata(unneeded_field="must not copy")
        actual = self.created + dt.timedelta(minutes=2)
        status, code, factory, instances = self.run_fixture(actual=actual)
        self.assertEqual((code, status["state"], status["reason"]), (0, "complete", "collected"))
        self.assertEqual(status["slot"]["phase"], "reserved_validation")
        self.assertEqual(status["actual_start_at_utc"], schedule.stamp(actual))
        self.assertEqual(status["pre_request_check_at_utc"], schedule.stamp(actual))
        self.assertEqual(instances[0].calls, 1)
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(instances[0].target, "2026-10-10")
        self.assertNotIn("unneeded_field", status["run_metadata"])
        self.assertTrue(all(item["matches"] for item in status["frozen_files"].values()))
        self.assertEqual((self.root / "result/protocol.md").read_bytes(),
                         (schedule.ROOT / schedule.PROTOCOL).read_bytes())
        raw = (self.root / "result/capture/responses/0001.body").read_bytes()
        self.assertEqual(status["evidence"]["bodies"][0]["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(status["evidence"]["body_bytes"], len(raw))
        self.assertNotIn("private_fixture_value", json.dumps(status))

    def test_queue_cannot_move_to_later_slot_or_run_after_tolerance(self):
        for name, actual, reason in (
            ("late", self.created + dt.timedelta(minutes=16), "collector_start_off_schedule"),
            ("next", self.created + dt.timedelta(hours=4), "creation_and_start_slots_differ"),
            ("year", self.created.replace(year=2027), "outside_frozen_year"),
        ):
            with self.subTest(name=name):
                status, code, factory, _ = self.run_fixture(actual=actual, name=name)
                self.assertEqual((code, status["state"], status["reason"]), (0, "skipped", reason))
                factory.assert_not_called()

    def test_creation_time_and_attempt_are_checked_before_initialization(self):
        for name, changes, attempt, reason in (
            ("off", {"created_at": "2026-09-26T12:00:00Z"}, 1, "creation_off_schedule"),
            ("year", {"created_at": "2027-09-26T13:15:00Z"}, 1, "outside_frozen_year"),
            ("rerun", {"run_attempt": 2}, 2, "rerun_attempt"),
            ("manual", {"event": "workflow_dispatch"}, 1, "not_schedule_event"),
        ):
            with self.subTest(name=name):
                self.write_metadata(**changes)
                status, code, factory, _ = self.run_fixture(attempt=attempt, name=name)
                self.assertEqual((code, status["reason"]), (0, reason))
                factory.assert_not_called()

    def test_initialization_delay_preserves_manifest_but_sends_no_requests(self):
        late = self.created + dt.timedelta(minutes=16)
        for name, initialized, gate in (("constructor", late, late), ("gate", self.created, late)):
            with self.subTest(name=name):
                status, code, _, instances = self.run_fixture(initialized=initialized, gate=gate, name=name)
                self.assertEqual((code, status["reason"]), (0, "initialization_missed_slot"))
                self.assertEqual(instances[0].calls, 0)
                self.assertEqual(status["evidence"]["body_files"], 0)
                self.assertFalse(status["evidence"]["manifest"]["complete"])

    def test_inconsistent_run_chronology_fails_before_collector_initialization(self):
        for name, created, started in (
            ("future_creation", "2026-09-26T13:16:00Z", "2026-09-26T13:16:00Z"),
            ("future_start", "2026-09-26T13:15:00Z", "2026-09-26T13:16:00Z"),
            ("start_before_creation", "2026-09-26T13:15:00Z", "2026-09-26T13:14:00Z"),
        ):
            with self.subTest(name=name):
                self.write_metadata(created_at=created, run_started_at=started)
                status, code, factory, _ = self.run_fixture(name=name)
                self.assertEqual((code, status["reason"]), (1, "run_timestamp_order_invalid"))
                self.assertEqual(status["run_metadata"]["created_at"], created)
                factory.assert_not_called()

    def test_missing_or_changed_frozen_input_never_initializes_collector(self):
        for name, relative, missing in (("protocol_drift", schedule.PROTOCOL, False),
                                        ("collector_drift", schedule.COLLECTOR, False),
                                        ("protocol_missing", schedule.PROTOCOL, True),
                                        ("collector_missing", schedule.COLLECTOR, True)):
            with self.subTest(name=name):
                path = self.repo / relative
                original = path.read_bytes()
                if missing:
                    path.unlink()
                else:
                    path.write_bytes(original + b"\n")
                status, code, factory, _ = self.run_fixture(name=name)
                self.assertEqual((code, status["reason"]), (1, "missing_or_changed_frozen_file"))
                factory.assert_not_called()
                path.write_bytes(original)

    def test_missing_invalid_or_mismatched_run_metadata_fails_closed(self):
        for name, changes in (("identity", {"id": 102}), ("boolean_id", {"id": True}),
                              ("attempt", {"run_attempt": 2}), ("time", {"created_at": "not-a-date"}),
                              ("sha", {"head_sha": "invalid"}), ("missing", None)):
            with self.subTest(name=name):
                self.write_metadata(**(changes or {}))
                if changes is None:
                    self.metadata.unlink()
                status, code, factory, _ = self.run_fixture(name=name)
                self.assertEqual((code, status["reason"]), (1, "coordinator_or_collector_error"))
                factory.assert_not_called()
                self.assertTrue((self.root / name / "protocol.md").is_file())

    def test_incomplete_exception_and_interrupt_retain_body_and_manifest_and_fail(self):
        for name, error, reason in (("incomplete", None, "collector_incomplete"),
                                     ("error", RuntimeError("private_fixture_value"), "coordinator_or_collector_error"),
                                     ("interrupt", KeyboardInterrupt(), "collector_interrupted")):
            with self.subTest(name=name):
                status, code, _, instances = self.run_fixture(complete=False, error=error, name=name)
                self.assertEqual((code, status["state"], status["reason"]), (1, "failed", reason))
                self.assertEqual(instances[0].calls, 1)
                self.assertEqual(status["evidence"]["body_files"], 1)
                self.assertFalse(status["evidence"]["manifest"]["complete"])
                self.assertNotIn("private_fixture_value", json.dumps(status))

    def test_duplicate_invocations_are_separate_and_existing_directory_is_untouched(self):
        first, _, _, _ = self.run_fixture(name="first", complete=False)
        second, _, _, _ = self.run_fixture(name="second")
        self.assertEqual(first["slot"], second["slot"])
        self.assertEqual(first["primary_invocation"], second["primary_invocation"])
        saved = (self.root / "first/run-status.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_fixture(name="first")
        self.assertEqual((self.root / "first/run-status.json").read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
