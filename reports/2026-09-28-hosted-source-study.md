# First hosted source observations: two eligible captures, no observed overlap

**Two scheduled development captures are now verified; neither observed an official daily report alongside an eligible quote.** Two other jobs arrived outside the frozen ±15-minute tolerance and correctly skipped collection. These four successful GitHub jobs therefore supply two evaluated snapshots, not four. The evidence does not establish alpha or a trading candidate.

This check follows the [04:33 UTC deployment and separate off-schedule observation](2026-09-28-source-activation.md). The earlier off-schedule capture remains excluded. The original development targets (September 26–October 9), reserved targets (October 10–23), six daily slots and eligibility rules are unchanged.

## Delivered runs

All times below are September 28, 2026 UTC. Each run is an original scheduled attempt (`run_attempt=1`), terminal with GitHub conclusion `success`. Coordinator evidence determines whether collection actually occurred.

| Nominal slot | Target date | GitHub creation | Collector start | Result |
|---|---|---|---|---|
| 05:15 | September 27 | 05:28:58 | 05:29:05.021034 | [Selected run 36382095119](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/36382095119), 122 responses |
| 09:15 | September 27 | 09:38:11 | None | [Run 36404762932](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/36404762932) skipped: `creation_off_schedule` |
| 13:15 | September 28 | 13:32:27 | None | [Run 36429335521](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/36429335521) skipped: `creation_off_schedule` |
| 17:15 | September 28 | 17:27:16 | 17:27:27.505414 | [Selected run 36458227853](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/36458227853), 122 responses |

The two late creations are outside every eligible slot; the table names the preceding scheduled checkpoint for explanation, not an eligible assignment. Their coordinator artifacts contain no source capture. No retry, shifted observation, wider tolerance or replacement was used.

The first three runs used deployed revision `dccd89dd4d0f1e4b3122c0dab53791fb6a6ec327`; the last used daily-capture revision `0edf4c348388062feabc04c65a12c6625214f424`. Frozen collector and protocol bytes match at both revisions. All 244 collected responses have valid JSON, HTTP 200, matching saved byte lengths and SHA-256 digests. Transport success does not imply an official report was available.

## Source and market evidence

Both daily responses explicitly contain 39 `no_report` records with null data, including all 15 study stations. Their exact receipt times are **05:29:05.187594 UTC** for the September 27 target and **17:27:27.721854 UTC** for September 28. No hourly value was substituted for the official daily maximum.

| Selected checkpoint | Observed overlap | Observed absence | Unknown |
|---|---:|---:|---:|
| September 28, 05:15 slot | 0 | 11 | 4 |
| September 28, 17:15 slot | 0 | 15 | 0 |
| Selected observations only | **0** | **26** | **4** |

At the 05:15 slot, Chicago, Austin, Dallas and Houston have unknown eligibility: their 24 books fall after the textual cutoff of 04:59 UTC but before the API close time of 06:00 UTC. Across the 90 market components, 36 are closed, 24 uncertain and 30 active. The other 11 station checkpoints have complete components establishing absence of the defined overlap; this does not mean all 11 cities were tradable. Maximum source-to-book receipt lag is 60.517540 seconds.

At the 17:15 slot, all 90 markets are active under both close bounds, all transport pairs are eligible, and all 15 official daily reports are explicitly absent. Maximum receipt lag is 60.967334 seconds. Receipt times bound these observations; they do not establish first publication time or absence between checkpoints.

## Coverage and reserved evidence

The complete read-only inventory page was requested at **19:56:23.146258 UTC** and received at **19:56:23.739110 UTC**. It reports four runs. All four attempt responses were received by 19:56:25.758695 UTC, and local coordinator statuses were bound to their exact saved bytes. The original artifacts were downloaded beforehand and their ZIP digests match the preserved GitHub artifact metadata.

The development analysis at **19:58:28.023738 UTC** retains all **84 slots / 1,260 station checkpoints**:

| Contribution | Station checkpoints |
|---|---:|
| Observed overlap | 0 |
| Observed absence | 26 |
| Unknown components within selected snapshots | 4 |
| Twelve closed slots without an eligible invocation | 180 |
| Seventy future slots, pending | 1,050 |
| Total | **1,260** |

The machine output therefore reports **0 overlap / 26 absence / 1,234 unknown**. Ten slots were missed before activation; two more have only late, skipped jobs. Pending and missing observations are not observed absences. Repeated stations and checkpoints are not independent return observations.

Reserved validation remains **locked** and emits no source-comparison counts. No reserved bodies were inspected. The original analysis freeze before October 10 and release no earlier than October 24 at 09:30 UTC remain in force, including the complete terminal-inventory requirement. A separately frozen trading candidate and fee-inclusive payout comparison remain undefined. The next nominal slot after this check is 21:15 UTC; this report makes no claim about its delivery.

## Portable evidence and reproduction

The [evidence bundle](2026-09-28-hosted-source-evidence.json.gz) contains 23 exact files: all four original artifact ZIPs, artifact metadata and download records, the complete inventory and its raw responses/statuses, and both analysis outputs. Each file records its relative path, byte length, SHA-256 and base64 bytes. The bundle is 525,532 compressed bytes and 2,562,854 uncompressed bytes; uncompressed SHA-256:

`34365eb94a1aedf5fc94a32a0e797e67c8044cd3840d654bb7010cbd2426fb7d`

| Run | Original ZIP bytes | SHA-256 |
|---|---:|---|
| 36382095119 | 262,793 | `3cdb2c545a9ae960fe1758c192b7176b58c87a5e4f336bdb6cbf7b8350823c7e` |
| 36404762932 | 6,345 | `55ca693a7b1391b1555d594c47f0794a6fcf1659594e2faefb1e91885863df5a` |
| 36429335521 | 6,346 | `e9cd9df1c69dca3adca54d637c7bf8cd7bf565f6c55966e98e8c2c93e56d61e7` |
| 36458227853 | 130,926 | `0426c99916adfd5f5ac7f5006aae67190bb13a64f99440d13d577021e42ef762` |

To reproduce offline, verify and decode `files[*]` into a temporary root. Extract each original ZIP under `artifacts/<ZIP stem>/`. Preserve the original inventory; make a separate copy and rebase only `artifact_paths[*].directory` to the extracted `artifacts/weather-source-<run_id>-<run_attempt>/run` directories. Raw GitHub metadata, status, manifest and response bytes must remain unchanged.

Load that copy with `weather_source_analysis.load_inventory`, then call `analyze_study` for each phase with the previously preserved `reports/2026-09-25-source-authority-evidence.json.gz` and the corresponding output's recorded `analysis_at_utc`. Historical evaluation times are used solely to reproduce the archived decisions, not to release reserved evidence. This procedure exactly reproduced both outputs after extraction into a fresh temporary directory. Independent review also verified all original ZIP/extracted-file bytes, selected-run bindings, response hashes and classifications.

The analysis implementation is recorded at `91eece5c3ed729d5590e0fca417c18296c5f1a36`. No source collector, protocol, evaluator, trading rule, risk limit or live setting changed in this update. The separate [paper audit](2026-09-28-forward-update.md) records the latest losses and admission state.
