# Operational addendum: scheduled prospective source collection

The [frozen protocol](2026-09-25-source-collection-protocol.md) defines a prospective study but its original commit supplied only an on-demand collector. This addendum prepares a separate GitHub Actions runner for those exact observation times. **It does not change the protocol, collector, sample windows, trading model or admission criteria.** No source values, quotes or returns from the reserved validation period are inspected by the runner.

## Scope and calendar

The proposed `.github/workflows/weather-source-research.yml` invokes `scripts/weather_source_schedule.py`. For target date D, its slots are 13:15, 17:15 and 21:15 UTC on D, then 01:15, 05:15 and 09:15 UTC on D+1. Targets are limited to **September 26–October 23, 2026**, with the original development/validation split unchanged. The first slot is September 26 at 13:15 UTC; the last is October 24 at 09:15 UTC.

Five calendar expressions cover the month boundary without generating extra study dates. An explicit year/date guard prevents source requests outside the 2026 study. Cron syntax has no year field, so leaving the workflow installed can produce guarded no-op runs in later years; removing it after the study avoids those runner invocations. No permanent source-monitoring service is introduced.

GitHub schedules run on the default branch and can be delayed or dropped. A draft-branch workflow does not establish an active scheduled collector, and a merged workflow does not establish successful observations. These limits are documented in [GitHub's scheduled-workflow reference](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule). At this addendum's preparation, the new workflow has not been merged or dispatched.

## Timing, duplicates and failures

The runner obtains the current workflow run's minimal metadata and requires a scheduled first attempt. It maps run creation to an eligible slot, then requires the actual collection start to fall within the same slot's 15-minute tolerance. A run queued into another slot cannot silently become a replacement observation. Timing is checked again immediately before collection. Wrong-year, off-window, late, non-scheduled and rerun invocations produce an operational status without source requests.

Run creation must precede or equal the recorded run start, which must precede or equal the coordinator's observation. Inconsistent chronology fails before collector initialization; the metadata and failure status remain available.

Every invocation gets a distinct artifact identity using its run ID and attempt. Separately delivered duplicate scheduled runs remain separate evidence; neither a successful duplicate nor a favorable quote replaces an earlier failed invocation. The protocol's earliest-started-invocation rule remains the analysis rule. A missing artifact or a job that fails before collection remains unknown coverage; workflow run history must be included when reconciling the planned denominator. No backup or manual catch-up collection is used.

Operational status and any partial raw evidence are retained even if collection fails. A collection error fails the job rather than producing a successful scientific result. Retrying artifact upload is distinct from repeating source requests: the source collector still has zero retries. No source values, quotes, inferred outcomes or profitability metrics are printed to job logs.

## Frozen inputs and artifact handling

Before any source request, the runner checks the original protocol and collector hashes:

- Protocol: `57c61d4c6495251e8b09f00897975893c4727dd25f33d16d3162e837fd2cd98c`
- Collector: `652fb66e90caf065a3324e7d0d83ddb23a917c5e0859bbadf1704eb1a46a3ab6`

A changed or missing file is an operational failure, not permission to substitute a new version silently. The existing collector retains its 122-request, two-starts-per-second, 15-second-per-request and five-minute request bounds. Each observation uses a new directory; earlier raw responses are never overwritten.

Artifacts contain operational status, run identity, copied protocol, manifest and exact response bytes. They are kept out of canonical captures and the pilot ledger. The workflow requests 90-day retention and unique artifact names, without overwrite. Those options follow the [official upload-artifact documentation](https://github.com/actions/upload-artifact#retention-period). Retention is finite: preserve required artifacts before they expire, rather than later reconstructing a missed vintage from current APIs.

The workflow has only `contents: read` and `actions: read` repository permissions. Its GitHub token is scoped to reading its own run metadata. Checkout does not persist credentials, and the anonymous source HTTP workers inherit no environment or trading credentials. There are no order endpoints, pilot steps, repository-data commits or trade-secret references. Checkout and artifact actions use verified immutable commit pins.

## Validation before deployment

All 12 focused offline scheduling tests pass. They cover the complete 168-slot calendar (84 development and 84 reserved), date/year and inclusive tolerance boundaries, queue delays, initialization delays, reruns, invalid metadata and chronology, frozen-file drift, separate duplicate evidence, and retention after partial failures or interruption. An executed bootstrap test verifies directory export outside checkout, paths containing spaces, and preservation on duplicate initialization. Tests use a fake collector and forbid importing the real collector; they issue no source requests.

The workflow parses successfully as YAML, and its embedded bootstrap Python compiles. Independent review verified the five calendar expressions against the original protocol. Both frozen file hashes remain unchanged. These checks do not establish GitHub schedule delivery, artifact upload success or collection coverage; those require actual default-branch run evidence. No workflow was dispatched during this validation.

The initial branch push produced a [GitHub validation failure](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/36172910879) before any jobs existed. The YAML-only check missed an invalid `runner.temp` expression in job-level environment configuration. GitHub's [context availability table](https://docs.github.com/en/actions/reference/workflows-and-actions/contexts#context-availability) disallows `runner` there. The corrected bootstrap derives its path from the runtime `RUNNER_TEMP` environment variable and exports it through `GITHUB_ENV`; artifact upload uses the permitted step-level context, independently of bootstrap success. No scientific observation was collected by the failed validation run.

Actionlint 1.7.12, downloaded from its official release and checked against the published archive checksum, reproduces the original context error and passes the corrected workflow with exit 0 and no diagnostics. ShellCheck was unavailable; this validation covers actionlint's workflow and expression checks, not ShellCheck analysis or actual hosted execution.

## Interpretation

Successful scheduling and complete HTTP responses establish data collection only. Missing official reports, absent books, uncertain fees, incompatible rules, revisions, closed markets and failed requests retain their distinct meanings under the original protocol. Source/quote comparisons from reserved targets remain uninspected until the protocol's end time and all available final-slot attempts are terminal. Missing slots remain missing.

Any later trading candidate still needs its own definition frozen before inspecting reserved data. The existing first prospective NO paper selection and this source-availability study remain separate experiments. Neither schedule deployment nor collector test success establishes alpha.
