# October 7 source audit: preserved captures, current evaluation unavailable

**Thirty-four new scheduled jobs produced twenty collection artifacts and fourteen late skips, but the current study cannot be evaluated.** GitHub's listing and attempt endpoints disagree on one run's creation timestamp. The attempt response also places its start one second before creation. The unchanged inventory validator rejects that evidence; successful workflows and intact source files do not resolve the contradiction.

This extends the [October 1 audit](2026-10-01-hosted-source-study.md). Its corrected result of 0 overlap / 194 absence / 1,066 unknown remains a dated historical result, not the October 7 aggregate. No current availability result, candidate, fee-inclusive opportunity or validated alpha follows from this update.

## What was preserved

The complete listing page reports 58 scheduled runs, and separate original-attempt responses were retained for all 58. Every response describes a completed job with GitHub conclusion `success` and `run_attempt=1`. These are operational metadata observations; they do not mean the inventory passed validation.

| Coordinator artifacts | Previously preserved | New | Total |
|---|---:|---:|---:|
| Completed collection | 16 | 20 | 36 |
| Skipped for `creation_off_schedule` | 8 | 14 | 22 |
| Total artifacts | 24 | 34 | 58 |

The newest preserved run is [37658692085](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/37658692085), associated by its coordinator with the October 7 17:15 UTC checkpoint. No workflow was dispatched, retried or backfilled during this audit. The thirty-four new original ZIPs were downloaded through bounded GitHub GET requests; the earlier twenty-four archives retained their original bytes and receipt provenance.

All **58 archive digests** match the preserved GitHub artifact metadata. All **4,754 extracted files** match the ZIP contents. Across the 36 completed collections, all **4,392 responses** have complete HTTP-200 bodies, valid JSON, matching byte lengths and SHA-256 hashes. Those checks establish archive and transport integrity only. They do not establish official-report status, quote eligibility, primary selection, fills or returns. Every collected target date is before the reserved period; no reserved source body was inspected.

## Exact metadata contradiction

The affected run is [37470401551](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/37470401551), associated by its artifact with the October 6 13:15 UTC checkpoint. Its metadata head is `533c6bc0be54e549f0944fab8ba1bc856e0e2ee9`.

| Preserved source | `created_at` | `run_started_at` |
|---|---|---|
| GitHub workflow-run listing | October 6 13:23:23 UTC | October 6 13:23:23 UTC |
| GitHub `/attempts/1` response | October 6 13:23:24 UTC | October 6 13:23:23 UTC |
| Original coordinator artifact | October 6 13:23:23 UTC | October 6 13:23:23 UTC |

The coordinator records an actual collector start of 13:23:33.925213 UTC. That artifact does not authorize replacing the attempt endpoint's contradictory value. Both discovery and final metadata captures preserve the same discrepancy. The 14,011-byte attempt body has SHA-256:

`ba66742a2dda383eed29125e36b41eb2c9d8a17673ddbd5c24a419065b5e7e5e`

The discovery listing was received at **October 7 17:40:46.543947 UTC** and the affected attempt at **17:41:10.313028**. After all artifact downloads, the final listing was received at **17:44:28.326227**, the affected attempt at **17:45:11.341188**, and the last attempt response at **17:45:15.905687**. Each inventory capture made 59 bounded GETs with no automatic retries. Raw listing, attempt and coordinator bytes remain unmodified.

The validator returns `inventory.state="unknown"` with `run_timestamp_order_invalid`. The same run also has a cross-endpoint creation-identity disagreement; the first chronology error prevents the validator from reaching that later check. No timestamp was rounded, substituted or assigned a tolerance. The run was not omitted, and no later successful invocation was selected as a replacement.

## Current output and frozen boundaries

Development evaluation at **October 7 17:46:17.359217 UTC** returns `not_evaluated`, with **zero evaluated snapshots** and all **1,260 planned station checkpoints unknown**. Its machine counts of zero overlap and zero absence describe an unavailable evaluation, not observed zero opportunities. The output's zero pending count likewise does not mean the remaining calendar checkpoints occurred: invalid inventory prevents the selector from assigning any slot, including future ones.

Reserved evaluation at **17:46:17.551069 UTC** remains **locked**, with no source-comparison counts. The current source analysis does not bypass inventory to inspect new official/preliminary-report or quote comparisons. The separately preserved integrity checks are operational only.

The original study dates, 84 slots per phase, 15-station universe, ±15-minute observation windows, earliest-invocation rule and unknown-evidence treatment are unchanged. Parsing and analysis eligibility must be frozen before **October 10 at 00:00 UTC**. A trading candidate still needs its own committed selection, prices, fees, sizing, timing, exclusions, revision treatment, evaluation metric and admission threshold before that deadline and before reserved comparisons. No candidate is defined by this audit.

Reserved comparisons cannot be released before **October 24 at 09:30 UTC**, and still require fresh complete terminal-inventory evidence. The missing or contradictory metadata cannot be repaired by extending the windows or reusing later observations. The next scheduled dispatch after this audit is October 7 at **21:05 UTC**, for the unchanged 21:15 nominal checkpoint; this report makes no claim about its delivery.

## Evidence locations and reproduction

Analysis revision is `2ccd75ef003d74afb43baefcf7ae51c8f6c926d1`. The previously corrected market parser remains SHA-256 `7dfccfd9728c3cbff7128d88e8840d51dc17b2eb1cce56bf9e088700c646620d`. Collector and protocol hashes remain respectively `652fb66e90caf065a3324e7d0d83ddb23a917c5e0859bbadf1704eb1a46a3ab6` and `57c61d4c6495251e8b09f00897975893c4727dd25f33d16d3162e837fd2cd98c`.

The original evidence is retained under new ignored directories:

- `data/raw/source-study-artifacts-20261007T1742Z/`: original ZIPs, artifact metadata, exact download records and prior receipt provenance.
- `data/raw/source-study-inventory-20261007T1741Z-discovery/`: discovery listing and all attempt responses, including the original failed validation.
- `data/raw/source-study-inventory-20261007T1744Z/`: final listing, attempts and bound coordinator statuses; unchanged development and reserved outputs; `artifact-integrity-summary.json`; and `metadata-discrepancy-summary.json`.

The final inventory SHA-256 is `452668802ac3391b7d968dcd0661ab7bd91b683aba727577ad389ca122b36928`. Reproduction must preserve the contradictory values and report the failed inventory check. Do not edit provider timestamps, drop the affected run or reinterpret artifact counts as evaluated snapshots. No runtime, protocol, collector, schedule, trading rule, risk limit or live setting changed in this source audit.

## Portable evidence and offline reproduction

The [portable evidence bundle](2026-10-07-hosted-source-evidence.json.gz) preserves **424 exact files**: all final inventory and discovery files, all 58 original artifact ZIPs, artifact metadata, download histories, and integrity/discrepancy summaries. Extracted source files are represented by their original ZIPs rather than duplicated in the bundle. Every file has its relative path, byte length, SHA-256 and base64 bytes.

Bundle size is **7,189,923 compressed bytes / 25,067,085 uncompressed bytes**. Uncompressed SHA-256:

`a801726bc31b93d2528489b823c98fcf28ae983acf748082e4410c25029905bf`

The bundle uses sorted compact JSON followed by a newline and gzip with `mtime=0`; deterministic compression was verified. After decoding into a fresh temporary directory, all 424 file hashes and lengths verified, all 58 ZIP digests matched artifact metadata, and all 4,754 safely extracted files matched their archive bytes. Coordinator status bytes matched the final inventory's saved raw statuses. Discovery and final raw metadata also passed byte/hash/decoded-body verification; that verification preserves rather than resolves the provider's contradictory timestamps.

For reproduction, keep the decoded inventory intact and create a separate copy rebasing only `artifact_paths[*].directory` to the extracted `artifacts/weather-source-<run_id>-<run_attempt>/run` directories. Use analysis revision `2ccd75ef003d74afb43baefcf7ae51c8f6c926d1`, the existing September 25 Fahrenheit authority bundle, and each output's recorded evaluation time. Both development and reserved output objects reproduced exactly: development remains `not_evaluated` with all 1,260 checkpoints unknown; reserved remains `locked` without comparison counts. The separately verified discovery inventory retains the same rejection reason. No endpoint was refreshed, source value substituted, metadata contradiction hidden or reserved gate bypassed during reproduction.
