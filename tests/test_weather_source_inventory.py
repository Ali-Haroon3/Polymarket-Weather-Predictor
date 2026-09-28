"""Saved-metadata-only coverage, embargo, and primary-selection regressions."""
import copy
import datetime as dt
import json
from pathlib import Path
import sys
import unittest
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import weather_source_inventory as inventory


def clock(value):
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


OBSERVED = "2026-10-25T12:00:00Z"
NOW = clock("2026-10-25T13:00:00Z")
START = "2026-09-26T13:15:00Z"


def run(run_id=11, attempt=1, *, created=START, started=None, status="completed", conclusion="success"):
    return {"id": run_id, "run_attempt": attempt, "event": "schedule", "path": inventory.WORKFLOW_PATH,
            "repository": {"full_name": inventory.REPOSITORY}, "created_at": created,
            "run_started_at": started or created, "updated_at": "2026-10-24T10:00:00Z",
            "status": status, "conclusion": conclusion, "head_sha": "a" * 40,
            "html_url": f"https://github.com/{inventory.REPOSITORY}/actions/runs/{run_id}"}


def response(url, body):
    return {"request_url": url, "requested_at_utc": OBSERVED, "received_at_utc": OBSERVED,
            "status": 200, "body": body}


def coordinator(row, *, started=None, state="complete"):
    begun = started or row["run_started_at"]
    finished = inventory.stamp(clock(begun) + dt.timedelta(seconds=60))
    slot = inventory.slot_at(clock(row["created_at"]))
    keys = ("id", "run_attempt", "event", "created_at", "run_started_at", "head_sha", "html_url")
    return {"run_id": row["id"], "run_attempt": row["run_attempt"], "received_at_utc": OBSERVED,
            "artifact_id": row["id"] + 1000, "source_sha256": "b" * 64,
            "status": {"schema_version": 1, "run_id": row["id"], "run_attempt": row["run_attempt"],
                       "run_metadata": {key: row[key] for key in keys}, "state": state, "reason": None,
                       "coordinator_started_at_utc": begun, "actual_start_at_utc": begun,
                       "collector_manifest_started_at_utc": begun, "pre_request_check_at_utc": begun,
                       "finished_at_utc": finished, "slot": slot, "creation_slot": slot}}


def fixture(rows=None, original_attempts=None):
    rows = [run()] if rows is None else rows
    all_attempts = original_attempts if original_attempts is not None else rows
    query = {"event": "schedule", "created_from": inventory.QUERY_FROM,
             "created_through": inventory.QUERY_THROUGH, "per_page": 100}
    pages = []
    for page in range(1, max(1, (len(rows) + 99) // 100) + 1):
        params = {"event": "schedule", "created": f"{inventory.QUERY_FROM}..{inventory.QUERY_THROUGH}",
                  "per_page": 100, "page": page}
        url = (f"https://api.github.com/repos/{inventory.REPOSITORY}/actions/workflows/"
               f"{inventory.WORKFLOW_NAME}/runs?{urlencode(params)}")
        pages.append(response(url, {"total_count": len(rows), "workflow_runs": rows[(page-1)*100:page*100]})
                     | {"page": page})
    attempts = [response(f"https://api.github.com/repos/{inventory.REPOSITORY}/actions/runs/{r['id']}/attempts/{r['run_attempt']}", r)
                | {"run_id": r["id"], "run_attempt": r["run_attempt"]} for r in all_attempts]
    return {"schema_version": 1, "repository": inventory.REPOSITORY, "workflow_path": inventory.WORKFLOW_PATH,
            "query": query, "pages": pages, "attempts": attempts,
            "coordinator_statuses": [coordinator(row) for row in all_attempts]}


class InventoryValidationTests(unittest.TestCase):
    def validate(self, data=None, now=NOW):
        return inventory.validate_inventory(fixture() if data is None else data, now)

    def assert_unknown(self, data, reason=None):
        value = self.validate(data)
        self.assertEqual(value["state"], "unknown", value)
        self.assertFalse(value["reserved_release_ready"])
        if reason:
            self.assertIn(reason, value["reasons"])

    def test_preserved_complete_metadata_releases_only_after_final_window(self):
        checked = self.validate()
        self.assertEqual(checked["state"], "valid", checked)
        self.assertTrue(checked["reserved_release_ready"])
        self.assertEqual(checked["provenance"]["attempt_count"], 1)
        json.dumps(checked, allow_nan=False)
        early = fixture()
        for entry in early["pages"] + early["attempts"]:
            entry["requested_at_utc"] = "2026-10-24T09:29:59Z"
        self.assertFalse(self.validate(early)["reserved_release_ready"])

    def test_assertion_without_page_evidence_never_unlocks_reserved_bodies(self):
        data = fixture()
        data.update(inventory_complete=True, pages=[])
        self.assert_unknown(data)
        self.assert_unknown({"inventory_complete": True})

    def test_empty_complete_inventory_retains_denominator_and_can_be_released(self):
        checked = self.validate(fixture([]))
        self.assertEqual(checked["state"], "valid")
        self.assertTrue(checked["reserved_release_ready"])
        chosen = inventory.select_primary(checked["records"], "reserved_validation", NOW)
        self.assertEqual(chosen["planned_station_checkpoints"], 1260)
        self.assertEqual(len(chosen["slots"]), 84)
        self.assertTrue(all(row["reasons"] == ["missing_scheduled_invocation"] for row in chosen["slots"]))

    def test_pages_require_contiguous_count_complete_unique_records(self):
        rows = [run(i) for i in range(1, 102)]
        data = fixture(rows)
        self.assertEqual(self.validate(data)["state"], "valid")
        for mutate in (lambda x: x["pages"].pop(),
                       lambda x: x["pages"][1].update(page=3),
                       lambda x: x["pages"][1]["body"].update(total_count=100),
                       lambda x: x["pages"][0]["body"]["workflow_runs"].pop(),
                       lambda x: x["pages"][1]["body"].update(workflow_runs=[rows[0]])):
            changed = copy.deepcopy(data)
            mutate(changed)
            self.assert_unknown(changed)

    def test_api_thousand_record_cap_cannot_claim_complete(self):
        data = fixture()
        data["pages"][0]["body"]["total_count"] = 1000
        self.assert_unknown(data, "metadata_total_unknown_or_api_cap")

    def test_query_identity_bounds_and_filters_are_exact(self):
        for mutate in (lambda x: x.update(repository="other/repo"),
                       lambda x: x.update(workflow_path="other.yml"),
                       lambda x: x["query"].update(created_through="2026-10-23T09:30:00Z"),
                       lambda x: x["pages"][0].update(request_url=x["pages"][0]["request_url"] + "&status=success"),
                       lambda x: x["pages"][0].update(request_url=x["pages"][0]["request_url"].replace("api.github.com", "example.com")),
                       lambda x: x["pages"][0].update(request_url=x["pages"][0]["request_url"].replace("page=1", "page=2"))):
            data = fixture()
            mutate(data)
            self.assert_unknown(data)

    def test_metadata_transport_and_future_retrieval_rejected(self):
        for changes in ({"status": 500}, {"requested_at_utc": "2026-10-26T00:00:00Z"},
                        {"received_at_utc": "2026-10-25T14:00:00Z"}, {"received_at_utc": "2026-10-25T11:00:00Z"}):
            data = fixture()
            data["pages"][0].update(changes)
            self.assert_unknown(data)

    def test_engineering_future_year_manual_and_other_workflow_are_rejected(self):
        for changes in ({"created_at": "2026-09-25T13:15:00Z"}, {"created_at": "2027-09-26T13:15:00Z"},
                        {"event": "workflow_dispatch"}, {"path": "other.yml"},
                        {"repository": {"full_name": "other/repo"}}, {"head_sha": "wrong"}):
            row = run()
            row.update(changes)
            self.assert_unknown(fixture([row]))

    def test_reruns_must_include_every_attempt_and_all_terminal_to_release(self):
        original, latest = run(), run(attempt=2, started="2026-10-24T09:45:00Z", status="in_progress", conclusion=None)
        data = fixture([latest], [original, latest])
        checked = self.validate(data)
        self.assertEqual(checked["state"], "valid", checked)
        self.assertFalse(checked["reserved_release_ready"])
        data["attempts"].pop(0)
        self.assert_unknown(data, "attempt_inventory_incomplete")

    def test_duplicate_and_foreign_attempts_rejected(self):
        data = fixture()
        data["attempts"].append(copy.deepcopy(data["attempts"][0]))
        self.assert_unknown(data, "duplicate_or_invalid_attempt_identity")
        data = fixture()
        data["attempts"][0]["run_id"] = 999
        self.assert_unknown(data, "unlisted_attempt")

    def test_attempt_body_identity_and_listing_identity_must_match(self):
        for changes in ({"id": 12}, {"run_attempt": 2}, {"head_sha": "b" * 40}):
            data = copy.deepcopy(fixture())
            # Break object aliasing to leave the preserved listing unchanged.
            data["attempts"][0]["body"] = dict(data["attempts"][0]["body"], **changes)
            self.assert_unknown(data)

    def test_missing_coordinator_does_not_hide_run_or_assert_unknown_start_absent(self):
        data = fixture()
        data["coordinator_statuses"] = []
        checked = self.validate(data)
        self.assertEqual(checked["state"], "valid")
        chosen = inventory.select_primary(checked["records"], "development", NOW)["slots"][0]
        self.assertEqual(chosen["reasons"], ["coordinator_timing_missing"])
        self.assertIsNone(chosen["primary"])

    def test_status_inventory_cannot_reuse_duplicate_foreign_or_future_artifact(self):
        for mutate in (lambda x: x["coordinator_statuses"].append(x["coordinator_statuses"][0]),
                       lambda x: x["coordinator_statuses"][0].update(run_id=999),
                       lambda x: x["coordinator_statuses"][0]["status"].update(run_id=999),
                       lambda x: x["coordinator_statuses"][0].update(received_at_utc="2026-10-26T00:00:00Z")):
            data = fixture()
            mutate(data)
            self.assert_unknown(data)

    def test_invalid_timezone_boolean_ids_and_unbounded_attempts_fail_closed(self):
        for mutate in (lambda x: x["pages"][0].update(received_at_utc="2026-10-25T12:00:00"),
                       lambda x: x["pages"][0]["body"]["workflow_runs"][0].update(id=True),
                       lambda x: x["pages"][0]["body"]["workflow_runs"][0].update(run_attempt=10**10)):
            data = fixture()
            mutate(data)
            self.assert_unknown(data)


class PrimarySelectionTests(unittest.TestCase):
    def records(self, data):
        checked = inventory.validate_inventory(data, NOW)
        self.assertEqual(checked["state"], "valid", checked)
        return checked["records"]

    def chosen(self, data, phase="development", now=NOW):
        return inventory.select_primary(self.records(data), phase, now)

    def test_exact_84_slots_each_and_midnight_maps_to_previous_target(self):
        development, reserved = (inventory.planned_slots(phase) for phase in inventory.PHASES)
        self.assertEqual((len(development), len(reserved)), (84, 84))
        self.assertEqual(development[3]["target_date"], "2026-09-26")
        self.assertEqual(development[3]["slot_utc"], "2026-09-27T01:15:00.000000Z")
        self.assertEqual(development[-1]["target_date"], "2026-10-09")
        self.assertEqual(reserved[-1]["slot_utc"], "2026-10-24T09:15:00.000000Z")
        with self.assertRaises(ValueError):
            inventory.planned_slots("engineering")

    def test_earliest_failed_invocation_remains_primary_and_later_success_duplicate(self):
        first, later = run(11), run(12, created="2026-09-26T13:17:00Z")
        data = fixture([later, first])
        data["coordinator_statuses"][1]["status"]["state"] = "failed"
        selected = self.chosen(data)["slots"][0]
        self.assertEqual(selected["state"], "selected")
        self.assertEqual(selected["primary"]["id"], 11)
        self.assertEqual(selected["primary"]["coordinator"]["state"], "failed")
        self.assertEqual(selected["duplicates"], [{"run_id": 12, "run_attempt": 1}])

    def test_tied_earliest_start_unknown_without_success_tiebreak(self):
        data = fixture([run(11), run(12)])
        data["coordinator_statuses"][0]["status"]["state"] = "failed"
        selected = self.chosen(data)["slots"][0]
        self.assertEqual(selected["reasons"], ["earliest_start_tie"])
        self.assertIsNone(selected["primary"])

    def test_missing_start_blocks_later_replacement(self):
        data = fixture([run(11), run(12, created="2026-09-26T13:17:00Z")])
        data["coordinator_statuses"][0]["status"]["coordinator_started_at_utc"] = None
        selected = self.chosen(data)["slots"][0]
        self.assertEqual(selected["state"], "unknown")
        self.assertIsNone(selected["primary"])

    def test_off_schedule_or_wrong_phase_excluded_and_reruns_never_replace(self):
        off = run(12, created="2026-09-26T15:00:00Z")
        data = fixture([run(), off, run(13, created="2026-10-10T13:15:00Z")])
        result = self.chosen(data)
        self.assertEqual(len(result["skipped"]), 2)
        original, rerun = run(), run(attempt=2)
        data = fixture([rerun], [original, rerun])
        data["coordinator_statuses"] = [coordinator(rerun)]
        result = self.chosen(data)
        self.assertIsNone(result["slots"][0]["primary"])
        self.assertEqual(result["skipped"][0]["reason"], "not_original_scheduled_attempt")

    def test_nonterminal_first_run_retained_unknown(self):
        first = run(status="in_progress", conclusion=None)
        selected = self.chosen(fixture([first]))["slots"][0]
        self.assertEqual(selected["reasons"], ["primary_run_nonterminal"])
        self.assertEqual(selected["primary"]["id"], 11)

    def test_creation_and_coordinator_windows_both_required(self):
        data = fixture()
        data["coordinator_statuses"][0] = coordinator(run(), started="2026-09-26T13:30:01Z")
        result = self.chosen(data)
        self.assertEqual(result["skipped"][0]["reason"], "coordinator_start_off_schedule")
        self.assertIsNone(result["slots"][0]["primary"])

    def test_future_and_open_windows_pending_even_with_empty_inventory(self):
        result = inventory.select_primary([], "development", clock("2026-09-26T13:29:59Z"))
        self.assertTrue(all(row["state"] == "pending" for row in result["slots"]))
        result = inventory.select_primary([], "development", clock("2026-09-26T13:30:00Z"))
        self.assertEqual(result["slots"][0]["state"], "unknown")
        self.assertEqual(result["slots"][1]["state"], "pending")

    def test_cross_run_metadata_mismatch_contaminates_slot(self):
        for field, value in (("id", 12), ("head_sha", "b" * 40), ("run_attempt", 2),
                             ("event", "workflow_dispatch"), ("created_at", "2026-09-26T13:14:00Z"),
                             ("run_started_at", "2026-09-26T13:16:00Z")):
            data = fixture()
            data["coordinator_statuses"][0]["status"]["run_metadata"][field] = value
            selected = self.chosen(data)["slots"][0]
            self.assertEqual(selected["reasons"], ["coordinator_run_metadata_mismatch"])

    def test_status_times_after_artifact_receipt_or_before_github_start_unknown(self):
        for field, value in (("finished_at_utc", "2026-10-26T00:00:00Z"),
                             ("coordinator_started_at_utc", "2026-09-26T13:14:59Z")):
            data = fixture()
            data["coordinator_statuses"][0]["status"][field] = value
            selected = self.chosen(data)["slots"][0]
            self.assertEqual(selected["reasons"], ["coordinator_timestamp_order_invalid"])
            self.assertIsNone(selected["primary"])

    def test_late_actual_collection_retains_primary_but_cannot_qualify(self):
        data = fixture()
        item = data["coordinator_statuses"][0]["status"]
        item.update(actual_start_at_utc="2026-09-26T13:31:00Z",
                    collector_manifest_started_at_utc="2026-09-26T13:31:00Z",
                    pre_request_check_at_utc="2026-09-26T13:31:00Z", finished_at_utc="2026-09-26T13:32:00Z")
        selected = self.chosen(data)["slots"][0]
        self.assertEqual(selected["reasons"], ["primary_collection_off_schedule"])
        self.assertEqual(selected["primary"]["id"], 11)

    def test_stale_inventory_from_before_slot_closed_cannot_certify_primary(self):
        records = self.records(fixture())
        records[0]["inventory_observed_from_utc"] = "2026-09-26T13:29:59Z"
        result = inventory.select_primary(records, "development", NOW)
        self.assertEqual(result["slots"][0]["reasons"], ["inventory_observed_before_slot_window_closed"])
        self.assertIsNone(result["slots"][0]["primary"])

    def test_duplicate_same_identity_never_counts_as_independent_evidence(self):
        records = self.records(fixture())
        result = inventory.select_primary(records + records, "development", NOW)
        self.assertEqual(result["slots"][0]["reasons"], ["duplicate_run_attempt_record"])
        self.assertIsNone(result["slots"][0]["primary"])

    def test_output_has_operational_provenance_and_no_body_or_price_fields(self):
        data = fixture()
        data["coordinator_statuses"][0]["status"]["response_body"] = {"private_value": 999}
        selected = self.chosen(data)
        encoded = json.dumps(selected, allow_nan=False)
        self.assertNotIn("private_value", encoded)
        self.assertNotIn("response_body", encoded)
        self.assertIn("canonical_status_sha256", encoded)


if __name__ == "__main__":
    unittest.main()
