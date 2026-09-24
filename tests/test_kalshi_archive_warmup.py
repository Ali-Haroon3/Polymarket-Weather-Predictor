"""Training-extension fixtures preserve dates, transport bounds, and original data."""
import contextlib
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import kalshi_archive_warmup as warmup


COMMIT = "b" * 40


class ArchiveWarmupTests(unittest.TestCase):
    def test_fixed_extension_has_all_915_events_and_no_holdout_dates(self):
        events = warmup.extension_events()
        self.assertEqual(len(events), 915)
        self.assertEqual(len(set(events)), 915)
        self.assertEqual(min(day for _, day in events), dt.date(2026, 3, 1))
        self.assertEqual(max(day for _, day in events), dt.date(2026, 4, 30))
        for day in {day for _, day in events}:
            self.assertEqual([series for series, target in events if target == day],
                             list(warmup.archive.SERIES))
        entry = warmup.archive.entry_timestamp(events[0][1])
        self.assertEqual(warmup.archive.utc_string(entry), "2026-02-28T15:00:00Z")

    def test_extension_requires_its_own_preregistration_before_network(self):
        with patch.object(warmup.archive, "PublicArchiveClient") as client:
            for invalid in (None, "", "a" * 39, "g" * 40, warmup.ORIGINAL_PREREGISTRATION):
                with self.assertRaises(ValueError):
                    warmup.run(invalid)
            client.assert_not_called()

    def test_uses_unchanged_importer_and_records_correct_extension_manifest(self):
        rows = [dict(target_date="2026-03-01", market_id="FIXTURE", best_bid=0.1,
                     best_ask=0.2, settlement_ts=1772440000)]
        manifest = dict(sample_start="2026-05-01", sample_end="2026-06-28",
                        expected_events=915, accepted_events=1,
                        raw_responses=[{"sha256": "c" * 64}], event_coverage=[{"status": "accepted"}])
        with tempfile.TemporaryDirectory() as directory:
            output, cache = Path(directory) / "output", Path(directory) / "cache"
            with patch.object(warmup.archive, "PublicArchiveClient") as client_type, \
                    patch.object(warmup.archive, "download_sample", return_value=(rows, manifest)) as download, \
                    contextlib.redirect_stdout(io.StringIO()):
                result = warmup.run(COMMIT, output, cache)
            client_type.assert_called_once_with(cache, max_requests=10000, interval=0.2)
            download.assert_called_once_with(client_type.return_value, warmup.extension_events(), workers=12)
            self.assertEqual(result["sample_start"], "2026-03-01")
            self.assertEqual(result["sample_end"], "2026-04-30")
            self.assertEqual(result["mode"], "frozen_training_extension")
            self.assertTrue(result["training_only"])
            self.assertEqual(result["preregistration_commit"], COMMIT)
            self.assertEqual(result["raw_responses"], [{"sha256": "c" * 64}])
            self.assertEqual(result["captures_sha256"],
                             hashlib.sha256((output / "captures.jsonl").read_bytes()).hexdigest())
            self.assertEqual(result["wrapper_sha256"], warmup.code_hashes()["wrapper_sha256"])
            self.assertEqual(result["base_importer_sha256"], warmup.code_hashes()["base_importer_sha256"])
            self.assertEqual(json.loads((output / "manifest.json").read_text()), result)

    def test_original_archive_and_canonical_paths_cannot_be_overwritten(self):
        with patch.object(warmup.archive, "PublicArchiveClient") as client:
            for path in (warmup.archive.REPO_ROOT / "data",
                         warmup.archive.REPO_ROOT / "data/raw/kalshi_archive_may_june_2026"):
                with self.assertRaises(ValueError):
                    warmup.run(COMMIT, path)
            client.assert_not_called()

    def test_out_of_window_rows_or_changed_code_do_not_publish_captures(self):
        for changed_code in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "out"
                rows = [dict(target_date="2026-06-02" if not changed_code else "2026-04-30")]
                hashes = [dict(wrapper_sha256="w", base_importer_sha256="b")]
                hashes.append(dict(wrapper_sha256="changed" if changed_code else "w",
                                   base_importer_sha256="b"))
                with patch.object(warmup.archive, "PublicArchiveClient"), \
                        patch.object(warmup.archive, "download_sample", return_value=(rows, {})), \
                        patch.object(warmup, "code_hashes", side_effect=hashes):
                    with self.assertRaises((warmup.archive.InvalidArchive, RuntimeError)):
                        warmup.run(COMMIT, output, Path(directory) / "cache")
                self.assertFalse((output / "captures.jsonl").exists())

    def test_cli_has_no_date_timing_or_universe_overrides(self):
        with patch.object(warmup, "run") as run, contextlib.redirect_stderr(io.StringIO()):
            for argument in ("--start", "--end", "--time", "--series"):
                with self.assertRaises(SystemExit):
                    warmup.main(["--download", "--preregistration-commit", COMMIT, argument, "value"])
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
