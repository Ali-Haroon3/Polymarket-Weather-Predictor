"""Offline integration of inventory gates, artifact binding, and fixed denominators."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import weather_source_analysis as analysis
import weather_source_inventory as inventory
from test_weather_source_inventory import NOW, OBSERVED, clock, coordinator, fixture, run


class WeatherSourceStudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def artifact(self, data, run_id=11, attempt=1):
        """Build matching synthetic status/manifest bytes; no response bodies exist."""
        entry = next(row for row in data["coordinator_statuses"]
                     if (row["run_id"], row["run_attempt"]) == (run_id, attempt))
        status = entry["status"]
        directory = self.root / f"run-{run_id}-{attempt}"
        (directory / "capture").mkdir(parents=True)
        manifest = {"schema_version": 1, "target_date": status["slot"]["target_date"],
                    "started_at_utc": status["collector_manifest_started_at_utc"],
                    "finished_at_utc": status["finished_at_utc"], "requests": [], "complete": False}
        manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
        (directory / "capture/manifest.json").write_bytes(manifest_bytes)
        status["evidence"] = {"manifest": {"file": "capture/manifest.json", "bytes": len(manifest_bytes),
                                           "sha256": hashlib.sha256(manifest_bytes).hexdigest()}}
        self.write_status(entry, directory)
        data.setdefault("artifact_paths", []).append({"run_id": run_id, "run_attempt": attempt,
                                                      "directory": str(directory)})
        return directory

    def write_status(self, entry, directory):
        raw = (json.dumps(entry["status"], indent=2) + "\n").encode()
        (directory / "run-status.json").write_bytes(raw)
        entry["source_sha256"] = hashlib.sha256(raw).hexdigest()

    def snapshot(self, target="2026-09-26"):
        stations = [analysis.unknown_station(series, "synthetic_unknown") for series in analysis.STATIONS]
        for i, station in enumerate(stations):
            station["state"] = "overlap" if i < 3 else "absence" if i < 5 else "unknown"
        return {"target_date": target, "snapshot_only": True, "manifest_sha256": "f" * 64,
                "payout_comparison_available": False, "stations": stations}

    def evaluate(self, data, phase="development", *, now=NOW):
        # Real binding always runs. Only semantic snapshot evaluation is stubbed.
        with patch.object(analysis, "analyze_capture", return_value=self.snapshot()) as evaluate:
            result = analysis.analyze_study(data, phase, now=now)
        return result, evaluate

    def assert_unknown_binding(self, data, reason):
        result, evaluate = self.evaluate(data)
        evaluate.assert_not_called()
        self.assertEqual(result["evaluated_snapshots"], 0)
        self.assertEqual(result["counts"], {"overlap": 0, "absence": 0, "unknown": 1260})
        self.assertEqual(result["slots"][0]["reasons"], [reason])
        self.assertEqual(result["slots"][0]["primary"], {"run_id": 11, "run_attempt": 1})
        return result

    def test_empty_complete_inventory_keeps_all_missing_observations(self):
        result, evaluate = self.evaluate(fixture([]))
        evaluate.assert_not_called()
        self.assertEqual(result["state"], "not_evaluated")
        self.assertEqual((result["planned_slots"], len(result["slots"])), (84, 84))
        self.assertEqual(result["planned_station_checkpoints"], 1260)
        self.assertEqual(result["counts"], {"overlap": 0, "absence": 0, "unknown": 1260})
        self.assertEqual(result["pending_station_checkpoints"], 0)
        self.assertEqual(len(result["by_target"]), 14)
        self.assertTrue(all(row["unknown"] == 90 and row["slots"] == 6 for row in result["by_target"].values()))

    def test_future_slots_are_pending_and_still_in_full_denominator(self):
        data = fixture([])
        for page in data["pages"]:
            page.update(requested_at_utc="2026-09-25T12:00:00Z", received_at_utc="2026-09-25T12:00:00Z")
        result, evaluate = self.evaluate(data, now=clock("2026-09-25T13:00:00Z"))
        evaluate.assert_not_called()
        self.assertEqual(result["inventory"]["state"], "valid")
        self.assertEqual(result["pending_station_checkpoints"], 1260)
        self.assertEqual(result["counts"]["unknown"], 1260)
        self.assertTrue(all(slot["selection_state"] == "pending" for slot in result["slots"]))

    def test_one_real_binding_leaves_other_83_slots_unknown(self):
        data = fixture()
        directory = self.artifact(data)
        result, evaluate = self.evaluate(data)
        evaluate.assert_called_once_with(directory / "capture", expected_target="2026-09-26",
                                         authority_path=None, validation_released=True, now=NOW)
        self.assertEqual(result["state"], "evaluated_snapshots")
        self.assertEqual(result["evaluated_snapshots"], 1)
        self.assertEqual(result["counts"], {"overlap": 3, "absence": 2, "unknown": 1255})
        self.assertEqual(sum(result["counts"].values()), 1260)
        self.assertEqual(result["by_target"]["2026-09-26"]["unknown"], 85)
        self.assertEqual(result["by_target"]["2026-09-27"]["unknown"], 90)
        self.assertEqual(result["slots"][0]["primary"], {"run_id": 11, "run_attempt": 1})
        self.assertFalse(result["trading_candidate_defined"])
        self.assertFalse(result["payout_comparison_available"])
        json.dumps(result, allow_nan=False)

    def test_failed_earliest_partial_evidence_not_replaced_by_later_success(self):
        data = fixture([run(11), run(12, created="2026-09-26T13:17:00Z")])
        data["coordinator_statuses"][0]["status"]["state"] = "failed"
        directory = self.artifact(data, 11)
        self.artifact(data, 12)
        result, evaluate = self.evaluate(data)
        self.assertEqual(evaluate.call_count, 1)
        self.assertEqual(evaluate.call_args.args[0], directory / "capture")
        self.assertEqual(result["slots"][0]["primary"], {"run_id": 11, "run_attempt": 1})
        self.assertEqual(result["slots"][0]["duplicates"], [{"run_id": 12, "run_attempt": 1}])

    def test_missing_first_artifact_never_uses_later_successful_duplicate(self):
        data = fixture([run(11), run(12, created="2026-09-26T13:17:00Z")])
        self.artifact(data, 12)
        self.assert_unknown_binding(data, "missing_or_duplicate_local_artifact")

    def test_invalid_inventory_calls_neither_binding_nor_snapshot_reader(self):
        data = fixture()
        data["pages"] = []
        with patch.object(analysis, "_bound_capture") as bind, patch.object(analysis, "analyze_capture") as evaluate:
            result = analysis.analyze_study(data, "development", now=NOW)
        bind.assert_not_called()
        evaluate.assert_not_called()
        self.assertEqual(result["inventory"]["state"], "unknown")
        self.assertEqual(result["counts"]["unknown"], 1260)
        self.assertTrue(all(slot["reasons"] == ["inventory_incomplete_or_invalid"] for slot in result["slots"]))

    def test_reserved_before_release_reads_no_artifact_and_emits_no_source_counts(self):
        data = fixture([run(created="2026-10-10T13:15:00Z")])
        # Coherent operational metadata available before the reserved window ends.
        for entry in data["pages"] + data["attempts"]:
            entry.update(requested_at_utc="2026-10-11T12:00:00Z", received_at_utc="2026-10-11T12:00:00Z")
        data["pages"][0]["body"]["workflow_runs"][0]["updated_at"] = "2026-10-10T13:20:00Z"
        data["coordinator_statuses"][0]["received_at_utc"] = "2026-10-11T12:00:00Z"
        with patch.object(analysis, "_bound_capture") as bind, patch.object(analysis, "analyze_capture") as evaluate:
            result = analysis.analyze_study(data, "reserved_validation", now=clock("2026-10-11T13:00:00Z"))
        self.assertEqual(result["inventory"]["state"], "valid", result)
        self.assertEqual(result["state"], "locked")
        bind.assert_not_called()
        evaluate.assert_not_called()
        for key in ("counts", "slots", "by_target", "by_city", "by_target_city", "evaluated_snapshots"):
            self.assertNotIn(key, result)

    def test_reserved_nonterminal_attempt_after_release_still_locked_without_counts(self):
        data = fixture([run(created="2026-10-10T13:15:00Z", status="in_progress", conclusion=None)])
        self.artifact(data)
        with patch.object(analysis, "_bound_capture") as bind, patch.object(analysis, "analyze_capture") as evaluate:
            result = analysis.analyze_study(data, "reserved_validation", now=NOW)
        self.assertEqual(result["inventory"]["state"], "valid")
        self.assertEqual(result["state"], "locked")
        self.assertNotIn("counts", result)
        self.assertNotIn("slots", result)
        bind.assert_not_called()
        evaluate.assert_not_called()

    def test_reserved_complete_terminal_inventory_releases_bound_original(self):
        data = fixture([run(created="2026-10-10T13:15:00Z")])
        directory = self.artifact(data)
        with patch.object(analysis, "analyze_capture", return_value=self.snapshot("2026-10-10")) as evaluate:
            result = analysis.analyze_study(data, "reserved_validation", now=NOW)
        self.assertTrue(result["inventory"]["reserved_release_ready"])
        self.assertEqual(result["state"], "evaluated_snapshots")
        evaluate.assert_called_once_with(directory / "capture", expected_target="2026-10-10",
                                         authority_path=None, validation_released=True, now=NOW)
        self.assertEqual(sum(result["counts"].values()), 1260)

    def test_reserved_bare_complete_assertion_never_unlocks(self):
        data = fixture([])
        data.update(inventory_complete=True, pages=[])
        result, evaluate = self.evaluate(data, "reserved_validation")
        self.assertEqual(result["state"], "locked")
        self.assertNotIn("counts", result)
        evaluate.assert_not_called()

    def test_raw_status_byte_tamper_rejected_before_snapshot_reader(self):
        data = fixture()
        directory = self.artifact(data)
        path = directory / "run-status.json"
        path.write_bytes(path.read_bytes() + b" ")
        self.assert_unknown_binding(data, "coordinator_file_hash_mismatch")

    def test_canonical_status_binding_prevents_updated_raw_hash_hiding_tamper(self):
        data = fixture()
        directory = self.artifact(data)
        path = directory / "run-status.json"
        value = json.loads(path.read_bytes())
        value["reason"] = "changed_only_on_disk"
        raw = json.dumps(value).encode()
        path.write_bytes(raw)
        data["coordinator_statuses"][0]["source_sha256"] = hashlib.sha256(raw).hexdigest()
        self.assert_unknown_binding(data, "coordinator_inventory_mismatch")

    def test_duplicate_status_json_keys_rejected_even_with_matching_hashes(self):
        data = fixture()
        directory = self.artifact(data)
        path = directory / "run-status.json"
        raw = path.read_bytes().replace(b'{', b'{"state":"ambiguous",', 1)
        path.write_bytes(raw)
        data["coordinator_statuses"][0]["source_sha256"] = hashlib.sha256(raw).hexdigest()
        result, evaluate = self.evaluate(data)
        evaluate.assert_not_called()
        self.assertEqual(result["evaluated_snapshots"], 0)
        self.assertEqual(result["slots"][0]["reasons"], ["invalid_json"])

    def test_manifest_bytes_hash_and_size_both_bound(self):
        for field in ("bytes", "sha256"):
            with self.subTest(field=field):
                data = fixture()
                directory = self.artifact(data)
                entry = data["coordinator_statuses"][0]
                entry["status"]["evidence"]["manifest"][field] = (1 if field == "bytes" else "0" * 64)
                self.write_status(entry, directory)
                self.assert_unknown_binding(data, "coordinator_manifest_bytes_mismatch")
                # Reuse the temporary root for the next independent artifact fixture.
                (directory / "capture/manifest.json").unlink()
                (directory / "capture").rmdir()
                (directory / "run-status.json").unlink()
                directory.rmdir()

    def test_manifest_target_and_collector_start_bind_to_selected_invocation(self):
        for field, value in (("target_date", "2026-09-27"),
                             ("started_at_utc", "2026-09-26T13:15:01Z"),
                             ("finished_at_utc", "2026-09-26T13:20:00Z")):
            with self.subTest(field=field):
                data = fixture()
                directory = self.artifact(data)
                manifest_path = directory / "capture/manifest.json"
                body = json.loads(manifest_path.read_bytes())
                body[field] = value
                raw = json.dumps(body).encode()
                manifest_path.write_bytes(raw)
                entry = data["coordinator_statuses"][0]
                entry["status"]["evidence"]["manifest"].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                self.write_status(entry, directory)
                self.assert_unknown_binding(data, "capture_does_not_match_selected_invocation")
                manifest_path.unlink()
                (directory / "capture").rmdir()
                (directory / "run-status.json").unlink()
                directory.rmdir()

    def test_missing_mismatched_or_duplicate_local_identity_stays_unknown(self):
        data = fixture()
        directory = self.artifact(data)
        for mappings in ([], [{"run_id": 12, "run_attempt": 1, "directory": str(directory)}],
                         [{"run_id": 11, "run_attempt": 2, "directory": str(directory)}],
                         data["artifact_paths"] * 2):
            changed = copy.deepcopy(data)
            changed["artifact_paths"] = mappings
            self.assert_unknown_binding(changed, "missing_or_duplicate_local_artifact")

    def test_missing_manifest_file_is_unknown_without_falling_back_to_any_other_capture(self):
        data = fixture()
        directory = self.artifact(data)
        (directory / "capture/manifest.json").unlink()
        self.assert_unknown_binding(data, "local_artifact_unavailable_or_invalid")

    def saved_inventory(self):
        data = fixture()
        for category in ("pages", "attempts", "coordinator_statuses"):
            for index, record in enumerate(data[category]):
                body = record["status"] if category == "coordinator_statuses" else record["body"]
                raw = (json.dumps(body, indent=2) + "\n").encode()
                relative = f"{category}-{index}.json"
                (self.root / relative).write_bytes(raw)
                record["raw_file"] = relative
                if category == "coordinator_statuses":
                    record["source_sha256"] = hashlib.sha256(raw).hexdigest()
                else:
                    record.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                                  body_complete=True, error=None)
        path = self.root / "inventory.json"
        path.write_text(json.dumps(data))
        return path, data

    def save_inventory_payload(self, path, data):
        path.write_text(json.dumps(data))

    def test_load_inventory_verifies_every_preserved_response_and_status(self):
        path, data = self.saved_inventory()
        self.assertEqual(analysis.load_inventory(path), data)
        result, evaluate = self.evaluate(analysis.load_inventory(path))
        evaluate.assert_not_called()
        self.assertEqual(result["inventory"]["state"], "valid")
        self.assertEqual(result["slots"][0]["reasons"], ["missing_or_duplicate_local_artifact"])

    def test_load_inventory_rejects_metadata_raw_tamper_even_with_updated_length(self):
        path, data = self.saved_inventory()
        record = data["pages"][0]
        target = self.root / record["raw_file"]
        target.write_bytes(target.read_bytes() + b" ")
        record["bytes"] += 1
        self.save_inventory_payload(path, data)
        with self.assertRaisesRegex(analysis.EvidenceError, "inventory_raw_bytes_or_decoded_body_mismatch"):
            analysis.load_inventory(path)

    def test_load_inventory_rejects_decoded_body_drift_and_incomplete_transport(self):
        path, original = self.saved_inventory()
        for change in ("body", "body_complete", "error", "bytes"):
            with self.subTest(change=change):
                data = copy.deepcopy(original)
                record = data["pages"][0]
                if change == "body":
                    record["body"] = {"total_count": 0, "workflow_runs": []}
                elif change == "body_complete":
                    record[change] = False
                elif change == "error":
                    record[change] = "truncated"
                else:
                    record[change] = 0
                self.save_inventory_payload(path, data)
                with self.assertRaises(analysis.EvidenceError):
                    analysis.load_inventory(path)

    def test_load_inventory_rejects_missing_file_and_parent_or_symlink_traversal(self):
        path, original = self.saved_inventory()
        data = copy.deepcopy(original)
        data["pages"][0]["raw_file"] = "missing.json"
        self.save_inventory_payload(path, data)
        with self.assertRaises(OSError):
            analysis.load_inventory(path)
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside).resolve() / "outside.json"
            target.write_text("{}")
            link = self.root / "escape.json"
            link.symlink_to(target)
            for relative in (str(target), "escape.json"):
                data = copy.deepcopy(original)
                data["pages"][0]["raw_file"] = relative
                self.save_inventory_payload(path, data)
                with self.assertRaisesRegex(analysis.EvidenceError, "inventory_raw_path_outside_snapshot"):
                    analysis.load_inventory(path)

    def test_load_inventory_rejects_duplicate_status_keys_despite_valid_raw_digest(self):
        path, data = self.saved_inventory()
        record = data["coordinator_statuses"][0]
        target = self.root / record["raw_file"]
        raw = target.read_bytes().replace(b"{", b'{"state":"ambiguous",', 1)
        target.write_bytes(raw)
        record["source_sha256"] = hashlib.sha256(raw).hexdigest()
        self.save_inventory_payload(path, data)
        with self.assertRaisesRegex(analysis.EvidenceError, "invalid_json"):
            analysis.load_inventory(path)

    def test_load_inventory_requires_all_three_metadata_categories(self):
        path, original = self.saved_inventory()
        for category in ("pages", "attempts", "coordinator_statuses"):
            data = copy.deepcopy(original)
            del data[category]
            self.save_inventory_payload(path, data)
            with self.assertRaisesRegex(analysis.EvidenceError, "missing_inventory_records"):
                analysis.load_inventory(path)

    def test_city_and_target_city_summaries_preserve_full_denominators_without_duplicate_stations(self):
        data = fixture()
        self.artifact(data)
        result, _ = self.evaluate(data)
        self.assertEqual(len(result["by_city"]), 15)
        self.assertTrue(all(sum(row[state] for state in ("overlap", "absence", "unknown")) == 84
                            for row in result["by_city"].values()))
        self.assertEqual(len(result["by_target"]), 14)
        for target in result["by_target"].values():
            cities = target["by_city"]
            self.assertEqual(len(cities), 15)
            self.assertTrue(all(sum(row[state] for state in ("overlap", "absence", "unknown")) == 6
                                for row in cities.values()))
        self.assertNotIn("stations", result["slots"][0]["snapshot"])
        self.assertEqual(len(result["slots"][0]["stations"]), 15)


if __name__ == "__main__":
    unittest.main()
