# Offline scorer for the registered challenger trial

`scripts/weather_challenger_forward.py` assesses saved scale-only selections under the
[October 7 registration](2026-10-07-challenger-preregistration.md). Registration commit
`3af3c8416e8592d56a2698f2d15acb74424e6eaf` preceded the first capture date. The command pins
the registration's exact bytes and all three decision implementation hashes. The registration
JSON's `offline_scorer_status_at_registration` remains an accurate historical statement.

This implementation was developed and tested with synthetic evidence before inspecting any
prospective performance. It does not fetch evidence, refit models, rerank selections, generate
orders, modify the ledger, or grant live authority. Evidence collection is a separate operation.

## Usage and interpretation

```bash
python3 scripts/weather_challenger_forward.py \
  --evidence /absolute/path/to/preserved-challenger-evidence \
  --output research-output/challenger-assessment.json

python3 -m unittest discover -s tests -p 'test_weather_challenger_forward*.py'
```

The output path must be a new `.json` file outside the evidence directory. Omitting `--output`
prints the report. Optional `--as-of` requires a timezone and cannot exceed the actual clock.
Exit status 0 means a report was produced, including pending or inconclusive reports; it is
never an admission signal. Fatal protocol, input or output errors use status 2. Every report
sets `live_authorization` to false.

Before December 22, 2026 at 00:00 UTC, the top-level verdict is `pending`. Thereafter, all
60 capture dates must be valid, the complete run listing must have been obtained after the
capture window closed, and every primary selection must have an eligible unambiguous outcome.
Missing evidence is `inconclusive`, never a zero-order day. Information minima and all economic
criteria use the frozen registration. Paper criteria do not establish fills, account return,
future profitability, or validated alpha.

Use the top-level verdict and `complete_evidence` together. The nested accounting result is
conditional on evidence coverage; its own favorable value cannot override missing coverage.
Partial reports expose settled totals and unresolved cost/worst-case exposure, while withholding
full-sample net, ROI, drawdown and bootstrap inference. Post-cutoff outcome versions are retained
separately and cannot rescue the primary result.

## Evidence directory, schema 1

Keep exact raw GitHub metadata responses, decision ZIPs, captured input bytes and dated outcome
versions under one directory. Obtain metadata through trusted GitHub access. Hashes verify
consistency of saved bytes; they do not authenticate the server or prove an independently
supplied inventory truthful. Preserve response headers with the archive as collection provenance;
the scorer verifies their bytes and successful final status, not a header signature.

`inventory.json` contains:

| Field | Required contents |
| --- | --- |
| `schema_version` | Integer `1` |
| `repository` | `Ali-Haroon3/Polymarket-Weather-Predictor` |
| `workflow_path` | `.github/workflows/daily-capture.yml` |
| `query` | `{"created_from":"2026-10-08","created_through":"2026-12-06","event":"schedule","per_page":100}` |
| `pages` | Ordered workflow-run response records, each with integer `page` starting at 1 |
| `attempts` | One original-attempt response record per listed run, each with `run_id` and `run_attempt: 1` |
| `artifact_pages` | All artifact-list response pages for every listed run; each with `run_id` and `page` |
| `jobs_pages` | Optional original-attempt jobs response pages with `run_id`, `run_attempt: 1`, and `page` |
| `guard_logs` | Optional plain UTF-8 job-log response records with `run_id`, `run_attempt: 1`, and `job_id` |
| `artifacts` | Saved decision ZIP records: `artifact_id`, `path`, `bytes`, `sha256`, plus the HTTP receipt fields below |
| `captures` | Exact JSONL input records: `path`, `bytes`, `sha256` |
| `outcomes` | One JSONL file record: `path`, `bytes`, `sha256`; retain all dated outcome versions |

Each HTTP response record includes `request_url`, integer `status: 200`, `body_complete: true`,
`error: null`, `requested_at_utc`, `received_at_utc`, `attempt_finished_at_utc`, `raw_file`, `bytes`
and lowercase `sha256`. Each also requires `response_headers`, a file record with `path`,
`bytes` and `sha256` for the exact response headers, including redirect headers when present.
Times require offsets and request ≤ receipt ≤ finish ≤ assessment.
An optional decoded `body` must agree with the preserved raw bytes. Paths must be relative,
contained regular files with no symlinks. JSON rejects duplicate keys and nonfinite numbers.

Under `https://api.github.com/repos/Ali-Haroon3/Polymarket-Weather-Predictor`, the exact endpoints are:

```text
/actions/workflows/daily-capture.yml/runs?event=schedule&created=2026-10-08..2026-12-06&per_page=100&page=N
/actions/runs/RUN_ID/attempts/1
/actions/runs/RUN_ID/artifacts?per_page=100&page=N
/actions/runs/RUN_ID/attempts/1/jobs?per_page=100&page=N
/actions/jobs/JOB_ID/logs
/actions/artifacts/ARTIFACT_ID/zip
```

Page totals and identities must agree without reaching GitHub's 1,000-result cap. Failed runs,
missing artifacts and expired artifacts stay in the inventory. Later reruns do not substitute
for the original attempt. Expired hosted artifacts can qualify only when their exact original
ZIP bytes were already preserved and all links and digests still validate. ZIP records include
the same URL/status/completeness/error/timestamp/header receipt fields as metadata records, using
`path` for their bytes instead of `raw_file`. Their request URL is the GitHub artifact ZIP endpoint
above; terminal status is 200 after redirects. Download receipt cannot precede artifact creation;
an expired artifact with a supplied expiration time must have been downloaded before that time.
Downloading later than the original capture day is allowed.

Decision archives must contain exactly `weather-shadow.json` as their only file. Its frozen
policy, required date, capture digest, family set and selected lists must agree. Every primary
order must match a unique same-day Kalshi input row by ticker, city and target date, and its
paid price must equal the saved executable YES ask or complement of the NO-side YES bid.
This is an input/quote check, not a rerun of the model or selector.

Same-day candidate artifacts must agree across all five families. They are counted once by
earliest publication bound, then lowest run ID. A missing candidate cannot be ignored merely
because another run succeeded. To prove a harmless backup skip, preserve original-attempt jobs
and actual timestamped log output showing the daily capture guard stood down, together with
the skipped recovery-shadow step and a valid same-day candidate. A shell preview of the guard
script alone is insufficient.

Outcome JSONL rows use `market_id`, `source: "kalshi"`, `target_date`, `city`, numeric binary
`outcome`, and `outcome_observed_at`. An earlier unresolved null/null snapshot is permissible.
Binary outcomes with missing clocks, conflicting identities or conflicting pre-cutoff labels
prevent complete accounting. Receipts must strictly follow the saved publication bound and
precede the exclusive December 22 cutoff. Receipt time is not exchange settlement time.
The scorer unions relevant outcome versions from every preserved input capture and the separate
outcomes file, deduplicating exact rows and retaining their source-file provenance. A separate
outcomes file cannot suppress a known conflicting result already saved in capture inputs.
A relevant outcome receipt inside an already-published input cannot claim a later observation
time than that input's earliest referencing artifact publication bound. Such inconsistent
clocks make evidence unknown rather than relabeling an existing loss as a post-cutoff revision.

## Implementation and boundaries

- `weather_challenger_forward_evidence.py` checks archived provenance, calendar coverage,
  duplicate/guard behavior, exact input linkage and executable quotes.
- `weather_challenger_forward_metrics.py` validates saved sizing and modeled rounded fees,
  resolves dated outcomes, and implements fixed stress, city omission, target-date drawdown
  and stationary bootstrap rules. It cannot independently establish artifact coverage.
- `weather_challenger_forward.py` pins the registration and implementation, combines both
  layers, enforces the final clock and prevents incomplete evidence from passing.

Synthetic tests cover both passing and failing economics, fixed resampling, cutoff/conflict
handling, missing evidence and CLI overwrite protection. No real prospective performance is
reported here. Hosted first-run verification and later evidence collection remain necessary.
The separate source-study embargo, freeze and release gates remain unchanged.
