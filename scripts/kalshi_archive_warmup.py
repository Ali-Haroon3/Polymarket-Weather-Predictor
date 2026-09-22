#!/usr/bin/env python3
"""Retrieve a fixed March–April 2026 training extension, separate from the holdout.

Requires its own preregistration commit before any network request. Targets are
March 1 through April 30 inclusive, all 15 original Kalshi series. The unchanged
base importer supplies exact prior-day 15:00 UTC candles, complete partitions,
lifecycle validation, raw-response hashes, retries, and the shared HTTP cache.
This wrapper changes neither the June holdout nor any strategy/execution rule.

Only public archive GETs run: 12 workers, a shared maximum of five requests per
second, and 10,000 attempts including retries. Output stays separate from the
original archive and canonical captures. No flexible date/time/universe options.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import time

import kalshi_archive_capture as archive

WARMUP_START = dt.date(2026, 3, 1)
WARMUP_END = dt.date(2026, 4, 30)
WORKERS = 12
REQUEST_INTERVAL = 0.2
MAX_REQUESTS = 10000
DEFAULT_OUTPUT = archive.REPO_ROOT / "data/raw/kalshi_archive_warmup_mar_apr_2026"
DEFAULT_CACHE = archive.REPO_ROOT / "data/raw/kalshi_archive_http"
ORIGINAL_PREREGISTRATION = "e7c78f46121e4bfa9915c91c2165fba9377de118"


def extension_events():
    """Exactly 915 city-target events, ordered by target day then original series."""
    return [(series, WARMUP_START + dt.timedelta(days=day))
            for day in range((WARMUP_END - WARMUP_START).days + 1)
            for series in archive.SERIES]


def validate_preregistration(commit):
    if (not isinstance(commit, str) or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)):
        raise ValueError("training extension requires its new full preregistration commit SHA")
    if commit == ORIGINAL_PREREGISTRATION:
        raise ValueError("original May–June preregistration does not authorize the training extension")


def code_hashes():
    return dict(wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                base_importer_sha256=hashlib.sha256(Path(archive.__file__).read_bytes()).hexdigest())


def run(preregistration_commit, out_dir=DEFAULT_OUTPUT, cache_dir=DEFAULT_CACHE):
    validate_preregistration(preregistration_commit)
    output = archive.guarded_directory(out_dir)
    original = (archive.REPO_ROOT / "data/raw/kalshi_archive_may_june_2026").resolve()
    if output == original:
        raise ValueError("training extension must not overwrite the original archive")
    hashes = code_hashes()
    client = archive.PublicArchiveClient(cache_dir, max_requests=MAX_REQUESTS,
                                         interval=REQUEST_INTERVAL)
    rows, manifest = archive.download_sample(client, extension_events(), workers=WORKERS)
    if hashes != code_hashes():
        raise RuntimeError("importer source changed during retrieval; cache preserved, output not published")
    for row in rows:
        target = dt.date.fromisoformat(row["target_date"])
        if not WARMUP_START <= target <= WARMUP_END:
            raise archive.InvalidArchive("base importer returned a row outside frozen training dates")
    content = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    manifest.update(
        sample_start=WARMUP_START.isoformat(), sample_end=WARMUP_END.isoformat(),
        mode="frozen_training_extension", training_only=True,
        preregistration_commit=preregistration_commit,
        original_archive_preregistration_commit=ORIGINAL_PREREGISTRATION,
        captures_sha256=hashlib.sha256(content.encode()).hexdigest(),
        importer_sha256=hashes["base_importer_sha256"], **hashes,
        request_interval_seconds=REQUEST_INTERVAL, max_requests=MAX_REQUESTS,
        generated_at=archive.utc_string(time.time()),
        purpose="Additional causal training only; June holdout, four candidates, and execution rules remain fixed.",
    )
    archive.atomic_text(output / "captures.jsonl", content)
    archive.atomic_text(output / "manifest.json",
                        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"Wrote {len(rows)} training rows; {manifest['accepted_events']}/{manifest['expected_events']} "
          f"events accepted; capture SHA-256 {manifest['captures_sha256']}; output {output}")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", required=True,
                        help="retrieve the fixed training extension after its preregistration commit")
    parser.add_argument("--preregistration-commit", required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args(argv)
    try:
        validate_preregistration(args.preregistration_commit)
    except ValueError as exc:
        parser.error(str(exc))
    run(args.preregistration_commit, args.out_dir, args.cache_dir)


if __name__ == "__main__":
    main()
