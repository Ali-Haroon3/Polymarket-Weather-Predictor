# September 30 source study: four post-repair captures, still no qualifying overlap

**Four of the first five jobs after the dispatch repair collected eligible snapshots; one arrived late and skipped collection.** The new 09:15 checkpoint contains preliminary reports for all 15 study stations, but all 90 corresponding markets were already closed. No official-report/eligible-quote overlap has been observed, and no trading candidate is established.

This extends the [September 29 audit](2026-09-29-hosted-source-study.md). The [dispatch repair](2026-09-29-source-dispatch.md) moved cron launches ten minutes earlier while preserving the nominal slots and every eligibility window. Its first five observed jobs produced four captures, compared with six captures in eleven earlier jobs. This small, nonrandom comparison does not establish a causal improvement in delivery or any trading edge. Missing observations remain missing.

## First post-repair observations

All five new jobs finished with GitHub conclusion `success`; the coordinator status distinguishes actual collection from a guarded skip. The first four ran at metadata head `0af79968641744d005f434a8bcf93134b6b974db`, and the last at `fccf3165e375ceb5917f4a98a939b5594e17341f`. Both revisions contain the repaired minute-05 crons. Times below are UTC. The late job's nominal slot identifies the preceding planned checkpoint for explanation, not an eligible assignment.

| Run | Nominal September 30 slot | Target | Actual start or skipped creation | Primary classification |
|---|---|---|---|---|
| [36656162899](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/36656162899) | 01:15 | September 29 | Created 01:39:34; no capture | Unknown, late |
| [36673179893](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/36673179893) | 05:15 | September 29 | Start 05:23:35.802107 | 11 absence, 4 unknown |
| [36695969689](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/36695969689) | 09:15 | September 29 | Start 09:25:11.273192 | 15 unknown |
| [36721445750](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/36721445750) | 13:15 | September 30 | Start 13:24:55.403350 | 15 absence |
| [36750732025](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/36750732025) | 17:15 | September 30 | Start 17:21:10.101921 | 15 absence |

The new 05:15, 13:15 and 17:15 daily responses contain 39 explicit `no_report`/null rows each, including all study stations. Their source receipt times are respectively **05:23:36.018775**, **13:24:55.569614** and **17:21:10.320752**. Maximum source-to-book receipt lags are 60.617115, 60.515806 and 60.744502 seconds.

At 05:15, Chicago, Austin, Dallas and Houston remain unknown because their 24 books fall between the textual cutoff at 04:59 and API close at 06:00. The other market components comprise 36 closed and 30 active markets. An absence classification does not imply that the city was tradable. At 13:15 and 17:15, all 90 markets are active under both close bounds, with eligible transport pairs and explicit official-report absence.

### Preliminary reports observed after market closure

The 09:15 checkpoint's daily response, received at **09:25:11.489683**, contains **38 preliminary reports and one absent report** among 39 stations. The absent station is SAN, outside this study. All 15 study stations have `status="preliminary"`, `isOfficial=false`, an empty `issueTime`, and report date September 29.

The unchanged primary parser therefore returns **unknown: `unsupported_nonofficial_report`** for all 15. A separate descriptive application of the unchanged market-lifecycle helper finds **all 90 corresponding market/book components closed**. That check neither overrides the official-report requirement nor reclassifies the primary unknown observations as absence. The primary analysis exits at its source gate and does not report source/book pairing lag for this snapshot.

This observed receipt is not a first-publication timestamp, an official settlement source, or evidence of an executable opportunity. No later official value, hourly maximum or refreshed response replaces the preserved preliminary body. Opportunities between checkpoints remain unobserved.

## Complete inventory and fixed denominator

The complete inventory page was received at **September 30, 18:10:07.844143 UTC**; all original-attempt metadata responses were received by **18:10:16.490108**. All **16 runs** are terminal, original attempts and scheduled jobs. Ten produced eligible snapshots; six late jobs skipped collection without source requests.

The development evaluation at **18:10:22.576688 UTC** reports:

| Contribution | Station checkpoints |
|---|---:|
| Observed overlap | 0 |
| Observed absence across ten selected snapshots | 123 |
| Unknown components within selected snapshots | 27 |
| Sixteen closed slots without an eligible invocation | 240 |
| Fifty-eight future slots, pending | 870 |
| Total | **1,260** |

Machine totals are **0 overlap / 123 absence / 1,137 unknown**. The 27 selected unknowns comprise twelve close-bound conflicts across three 05:15 snapshots and fifteen unsupported preliminary reports. The sixteen unobserved closed slots comprise ten missed before deployment and six late jobs. Pending and missing checkpoints are not observed absences. Repeated stations and checkpoints are not independent return observations.

Reserved evaluation at **18:10:22.513133 UTC** remains **locked**, without source-comparison counts or reserved body inspection. The original development/validation dates, analysis freeze before October 10 at 00:00 UTC, October 24 at 09:30 earliest release, and complete terminal-inventory requirement remain unchanged. A separately frozen trading candidate and fee-inclusive payout comparison remain undefined.

The next scheduled dispatch after this inventory is September 30 at **21:05 UTC**, for the unchanged **21:15 nominal slot**. This report makes no claim about that job's delivery.

## Portable evidence and reproduction

The [portable bundle](2026-09-30-hosted-source-evidence.json.gz) preserves **104 files**: sixteen original artifact ZIPs, their metadata/download provenance, complete inventory and raw status/API responses, both analysis outputs, and supporting integrity, dispatch and preliminary-report checks. The earlier eleven archives retain their original download provenance. All sixteen ZIP digests match preserved GitHub metadata; all **1,320 extracted files** match the original archives. Across ten collected snapshots, all **1,220 responses** pass exact length, SHA-256, HTTP-200 and JSON checks.

Bundle size is **2,151,002 compressed bytes / 9,607,660 uncompressed bytes**. Uncompressed SHA-256:

`150c9970597027418f4d129a29c8a7c01adafd4dc09fcba6e54c060b1a52a805`

Analysis revision is `fccf3165e375ceb5917f4a98a939b5594e17341f`. The bundle retains per-file lengths and hashes. `inventory/artifact-integrity-summary.json` records each ZIP, manifest and daily-response hash; `inventory/postrepair-dispatch-check.json` records the five observed jobs and their workflow revisions. `inventory/preliminary-report-descriptive-check.json` preserves the separate descriptive check without changing primary classifications.

The [existing offline reproduction procedure](2026-09-28-hosted-source-study.md#portable-evidence-and-reproduction) applies: verify/decode every file, extract each original ZIP into its named artifact directory, and rebase only local artifact directory mappings in a separate inventory copy. With the preserved Fahrenheit authority bundle and each output's historical evaluation time, both outputs reproduce exactly from a fresh temporary directory. Reproduction does not refresh source endpoints or release reserved evidence.

No source collector, protocol, evaluator, trading rule, risk limit or live setting changed in this update. The separate [paper audit](2026-09-30-forward-update.md) records cumulative losses and admission status; source availability does not establish fills or profitability.
