# Prospective weather-source availability study

This protocol is frozen before retrieving a new source/book snapshot. Its purpose is to establish whether information from the contract's official weather source is observable while relevant market quotes remain available. **It does not define a profitable strategy, permit orders, or establish that an official or hourly value cannot be revised.** The existing pilot, its admission criteria, and its frozen experiments are unchanged.

## Motivation and excluded evidence

The [September 23 source reconciliation](2026-09-24-weather-source-reconciliation.md) found agreement between later official daily maxima and 90 venue outcomes. The earlier incomplete hourly maxima differed from the later daily maxima at 11 of 15 stations after rounding. That inspected date cannot validate a new temperature buffer, publication-delay strategy, or intraday model. Canonical daily captures also lack raw source vintages and contemporaneous book depth.

Any engineering capture of targets before September 26, 2026 is excluded from development and validation. Its only purposes are checking endpoint compatibility, evidence preservation and coverage. No return or parameter selection from an engineering capture is independent evidence.

## Fixed universe and requested evidence

Preserve all 15 existing US daily-high series: KXHIGHNY, KXHIGHCHI, KXHIGHAUS, KXHIGHDEN, KXHIGHLAX, KXHIGHMIA, KXHIGHPHIL, KXHIGHTDAL, KXHIGHTSEA, KXHIGHTATL, KXHIGHTBOS, KXHIGHTPHX, KXHIGHTLV, KXHIGHTDC and KXHIGHTHOU. Do not select cities or contracts based on prices, later outcomes or apparent opportunity.

For each explicit target date, preserve the public daily climate response and weekly hourly-observation response; each series' event markets with raw rules, station, strikes and lifecycle fields; series fee metadata; and each of the six identified contracts' order books with up to 100 levels. A missing or inconsistent event is recorded, not repaired with a city-name guess or a different date. Six matching market identities alone do not prove an exhaustive strike partition.

Every attempted request records client dispatch and response-observation UTC times, original and final URL, HTTP status, relevant cache headers, byte count, exact-body SHA-256 and any transport/parse failure. Attempts without an HTTP response have no receipt timestamp; their completion time is separate. These are client observation times, not precise network first-byte or source publication times. Exact response bytes are retained, including HTTP error bodies. The manifest records the collector and protocol hashes and checkpoints after each request. Existing output directories cannot be overwritten. A run that misses required request coverage is marked incomplete, with successfully preserved evidence still available for inspection. Request completeness and JSON validity do not establish valid weather observations, usable quotes or semantic API correctness; those remain separate analysis checks.

Requests are anonymous GETs only, with no trading credentials, cookies or order endpoints. The collector is serial, permits at most two starts per second and 122 requests per invocation, uses at most 15 seconds per request within a five-minute overall request deadline, and makes no retries. Redirects are recorded as failures rather than followed. Canonical captures and the pilot ledger are never written.

Each response body is capped at 8 MiB. A body interrupted by that cap, a timeout or a transport failure is explicitly incomplete; its stored bytes are only the received prefix. They must not be interpreted as a full response even if that prefix happens to parse as JSON.

## Prospective windows and collection clock

The proposed development targets are **September 26–October 9, 2026**, inclusive. The reserved validation targets are **October 10–23, 2026**, inclusive. Neither window may be extended or moved after inspecting its results. A short or empty sample is insufficient evidence, not permission to backfill.

Freeze the collection schema, parsing and analysis eligibility before October 10 at 00:00 UTC. During the reserved period, inspect operational health only; do not inspect source/quote comparisons or candidate results until **October 24 at 09:30 UTC**, after the last planned checkpoint. Any necessary operational repair must retain its version and failure history and cannot silently replace earlier evidence.

For a target date D, the fixed collection slots are **13:15, 17:15 and 21:15 UTC on D**, and **01:15, 05:15 and 09:15 UTC on D+1**. Record actual times; a run starting more than 15 minutes from its slot is off-schedule and excluded from the primary scheduled-sample analysis. Preserve off-schedule observations separately. Failed or missed slots remain in the coverage denominator. Never describe a later historical response as having been available at the missed decision time.

Each window therefore contains 1,260 planned station-checkpoint observations (14 dates × 6 slots × 15 stations). Report observed overlap, observed absence, and unknown/incomplete observations separately. Failed or missed retrievals are unknown, not proof that no opportunity existed. Repeated checkpoints are not independent weather outcomes.

If multiple invocations fall within one target/slot window, the earliest-started invocation is primary, including when it fails. Retain later invocations as nonprimary duplicates; never select a successful or favorable replacement retrospectively. An unresolved tie or missing timing evidence makes that slot unknown rather than allowing a result-dependent choice.

This commit provides an on-demand collector, **not a running scheduler**. No workflow, cron, trading setting or recurring automation is changed. The dated schedule above defines which future observations would qualify; it does not claim those observations have been collected or will arrive automatically. If the scheduled sample does not materialize, the study is not evaluated.

## Availability question and causal limits

The primary question is whether any scheduled snapshot contains both an official, date- and station-matched daily report and an active market with a displayed offer on the report-consistent side. Report coverage and such occurrences by city and target date, including zero-occurrence dates. Exact rule and unit matching is required; hourly maxima, daily averages and city labels cannot substitute for the contract's daily maximum.

The lifecycle response must identify the exact market as active and unclosed when received; a book received at or after its stated close does not qualify. Price availability requires an interior price strictly between $0 and $1. A count of zero means no overlap was observed at these checkpoints, not that overlap was impossible between checkpoints.

The daily response must have been fully received before the book request starts, with no more than 120 seconds between daily receipt and book receipt. Quote requests that fail this pairing window remain saved but are excluded from the primary availability count. The quote's own cached age, the source's issue/status fields and any inconsistencies must be disclosed. An unknown source publication time stays unknown. A receipt timestamp is evidence of observation, not first publication or a measured information advantage.

Book-side conversion must be explicit: a YES bid corresponds to a NO offer and a NO bid to a YES offer. A zero-size metadata ask is not liquidity. At least one whole contract of displayed depth is required. Use decimal arithmetic and the saved series fee rules before describing any price as below ordinary binary payout after fees. Missing or unsupported fee metadata makes that comparison unavailable. Sequential public requests are not atomic or evidence of an actual fill, and the finite depth limit can omit deeper quotes.

An official daily value can be revised, the venue may apply corrections or exceptional settlement provisions, and a market can stop accepting orders between requests. Preserve later source versions and authoritative venue outcomes under their later receipt times. Do not overwrite an earlier version with the final value. The collection study establishes observable data and displayed liquidity only; apparent payout headroom is not a risk-adjusted expected return or validated alpha.

## Separation from a future trading test

Development data may be used to specify a candidate after the collection process is checked. A trading candidate must then have a separately committed definition of selection, prices, fees, sizing, timing, exclusions, revision treatment, evaluation metric and admission threshold **before October 10 and before inspecting reserved source values, quotes, inferred outcomes, availability counts, payout headroom or strategy returns**. Reserved availability data are not automatically an untouched strategy holdout. Data inspected before a candidate's freeze become development evidence for that candidate and cannot later be relabeled independent validation. No candidate has been chosen here. If that freeze is missing, the reserved period is not an independent strategy test.

Any subsequent uncertainty calculation must preserve dependence across contracts and cities sharing a target day. Missing books, incomplete source coverage, unknown order availability and unsuccessful fetches cannot be silently omitted to manufacture a favorable rate. The existing NO shadow and archive results remain distinct experiments with their original freezes.

## On-demand use

Run the collector only into a new directory under ignored `data/raw/`, supplying this exact protocol file. Inspect `manifest.json` for completeness and failures before using the output. An engineering run is a connectivity and data-integrity check, not validation of the availability hypothesis. Later evaluation must use the saved raw bodies and recorded receipt times rather than refreshing endpoints to reconstruct old observations.

```bash
python3 scripts/weather_source_capture.py \
  --target-date YYYY-MM-DD \
  --output-dir data/raw/UNIQUE-RUN \
  --protocol-file reports/2026-09-25-source-collection-protocol.md
```

Replace the date and directory placeholders explicitly. The requested target date selects a weather/event date; it never backdates the response's actual receipt time.
