# September 29 source study: six eligible captures, no observed overlap

**Four new eligible captures still show no official daily report alongside an eligible quote.** Three other new jobs arrived late and correctly skipped collection. Across all eleven original scheduled runs, six snapshots are eligible and five are skipped. This extends the [September 28 evidence](2026-09-28-hosted-source-study.md); it does not establish an alpha candidate.

## New runs and preserved observations

All seven new jobs finished with GitHub conclusion `success`. The coordinator statuses distinguish actual collection from guarded skips. Times below are UTC; nominal slots for late jobs identify the preceding planned checkpoint for explanation, not an eligible assignment.

| Run | Nominal slot | Target | Actual start or skipped creation | Classification |
|---|---|---|---|---|
| [36486059773](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/36486059773) | Sep 28 21:15 | Sep 28 | Start 21:26:15.195655 | 15 absence |
| [36508957148](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/36508957148) | Sep 29 01:15 | Sep 28 | Created 01:39:49; no capture | Unknown, late |
| [36526217028](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/36526217028) | Sep 29 05:15 | Sep 28 | Start 05:27:06.884469 | 11 absence, 4 unknown |
| [36549576712](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/36549576712) | Sep 29 09:15 | Sep 28 | Created 09:30:08; no capture | Unknown, late |
| [36575994095](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/36575994095) | Sep 29 13:15 | Sep 29 | Created 13:33:42; no capture | Unknown, late |
| [36605050427](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/36605050427) | Sep 29 17:15 | Sep 29 | Start 17:27:38.030991 | 15 absence |
| [36633407320](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/36633407320) | Sep 29 21:15 | Sep 29 | Start 21:27:46.716788 | 15 absence |

Each new selected snapshot contains 122 complete HTTP-200 JSON responses. All four daily bodies contain 39 explicit `no_report`/null rows, including the 15 study stations. Source receipt times are September 28 at **21:26:15.483567**, and September 29 at **05:27:07.050919**, **17:27:38.347046** and **21:27:46.933240** UTC. Maximum source-to-book receipt lags are respectively 60.394064, 60.515814, 61.198004 and 60.565808 seconds.

At the new 05:15 checkpoint, Chicago, Austin, Dallas and Houston remain unknown because their 24 books fall between the textual cutoff at 04:59 UTC and API close at 06:00 UTC. The other components comprise 36 closed and 30 active markets. An absence classification does not imply every city was tradable. The other three new selected snapshots have all 90 markets active under both close bounds, with eligible transport pairs and explicit official-report absence.

No hourly maximum, later source value or current response was substituted for an unavailable daily report. These observations do not determine publication time or exclude opportunities between checkpoints.

## Complete inventory and fixed denominator

The complete inventory page was received at **September 29, 21:30:59.656636 UTC**, after the last nominal slot's 21:30 cutoff. All eleven original-attempt metadata responses were received by **21:31:04.881247 UTC**. Every run is terminal; all eleven artifacts are present and their original ZIP digests match preserved GitHub metadata.

The development evaluation at **21:31:11.592376 UTC** reports:

| Contribution | Station checkpoints |
|---|---:|
| Observed overlap | 0 |
| Observed absence across six selected snapshots | 82 |
| Unknown components within selected snapshots | 8 |
| Fifteen closed slots without an eligible invocation | 225 |
| Sixty-three future slots, pending | 945 |
| Total | **1,260** |

The machine totals are therefore **0 overlap / 82 absence / 1,178 unknown**. The fifteen unobserved closed slots comprise ten missed before deployment and five late jobs. Neither missing nor pending evidence is observed absence. The six selected checkpoints span only three target dates, September 27–29; the station counts are not independent return observations.

Reserved evaluation at **21:31:12.291906 UTC** remains **locked**, with no source-comparison counts or reserved body inspection. The original development/validation dates, October 10 freeze, October 24 09:30 earliest release, and terminal-inventory requirement remain unchanged. A trading candidate and fee-inclusive payout comparison are still undefined.

The repeated dispatch delays motivated an [operational repair](2026-09-29-source-dispatch.md). It was undeployed at this 21:31 evaluation and subsequently merged at 21:41:35 UTC, as preserved in the [deployment record](2026-09-29-source-dispatch-deployment.json). It cannot recover any missed observation. All runs analyzed here used the original :15 dispatch schedule.

## Evidence and reproduction

The [portable bundle](2026-09-29-hosted-source-evidence.json.gz) preserves 69 files, including eleven exact artifact ZIPs, their metadata/download provenance, complete inventory and raw status/API responses, integrity summary, and both analysis outputs. The four previously downloaded archives were copied with their original receipt provenance; seven new archives were downloaded on September 29. All 799 extracted files match ZIP bytes. Across six collected snapshots, all **732 responses** match their recorded lengths and SHA-256 and parse as JSON.

Bundle size is **1,340,592 compressed bytes / 6,236,709 uncompressed bytes**. Uncompressed SHA-256:

`6165f07e4a3a3385a8575acefa4fe81eba2d9deb59a483b6c72818e3324ad796`

The bundle retains exact per-file hashes and the compact `inventory/artifact-integrity-summary.json`, including each ZIP/manifest/daily-response hash and actual timestamp. Analysis revision is `a8f95c8ee378ba8d0146e8912cbe76812b254326`; the runtime evaluator, coordinator, source collector and protocol are unchanged.

The [previous reproduction procedure](2026-09-28-hosted-source-study.md#portable-evidence-and-reproduction) applies: verify/decode each file, extract each original ZIP under its named artifact directory, and rebase only local artifact mappings in a separate inventory copy. Using the preserved Fahrenheit authority and each output's historical evaluation time reproduces both results exactly from a fresh temporary directory. It does not release reserved data or refresh source endpoints.

The separate [paper audit](2026-09-29-forward-update.md) records the latest trading losses. Source availability evidence does not establish fills, expected returns or permission to promote a strategy.
