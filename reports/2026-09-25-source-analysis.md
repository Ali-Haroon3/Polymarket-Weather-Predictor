# Prospective source study: inventory and offline evaluation

**The availability evaluator is implemented; the prospective study has not started.** A real, read-only GitHub inventory request at September 25, 18:53:08 UTC found zero scheduled invocations in the fixed study window. The development check returned `not_evaluated`, with all 1,260 station-checkpoints pending. The reserved check remained locked and emitted no source-comparison counts. No availability rate, return estimate or alpha claim follows from these checks.

This builds on the [parser support definition](2026-09-25-source-parser-design.md) and preserves the original [collection protocol](2026-09-25-source-collection-protocol.md), its dates and its collector bytes. The proposed [schedule](2026-09-25-source-schedule.md) is still in a draft pull request. A tested evaluator does not establish successful deployment or future collection.

## Inventory before observations

`scripts/weather_source_inventory_capture.py` preserves the exact JSON body bytes emitted by `gh api` for the fixed repository/workflow and full study creation window. It uses GET only, keeps failures, checkpoints after requests, permits at most 512 requests with a 30-second timeout each and no automatic retries, and refuses an existing output directory. Response headers are restricted to a safe allowlist; credentials, OAuth headers and command stderr are not preserved.

The exporter obtains every count-complete page and every attempt of each discovered run, including failed attempts and reruns. It accepts fewer than 1,000 listed runs; reaching the API search cap cannot establish completeness. The relevant read endpoints and limits are documented in [GitHub's workflow-run API reference](https://docs.github.com/en/rest/actions/workflow-runs). It does not dispatch jobs, download weather/quote bodies, place orders or modify repository data.

An optional local artifact root is checked only at the exact known `weather-source-RUN_ID-ATTEMPT/run/run-status.json` paths. Missing or unreadable status leaves the affected invocation's timing unknown. Source bodies are not inspected by this inventory step. Raw metadata and status files are hashed, and the evaluator verifies those bytes against the decoded inventory before using it.

`scripts/weather_source_inventory.py` requires exact query identity, consistent contiguous pagination, unique run identities, all original/rerun attempts, explicit timezone-aware chronology, and agreement between each saved coordinator status and its GitHub run. A bare `inventory_complete` assertion cannot establish completeness. This is completeness within preserved GitHub metadata, not a cryptographic authentication of server history.

## Primary selection and fixed coverage

Each phase has the original 84 slots and 1,260 station-checkpoints. Selection uses the earliest coordinator start, never completion, source values, price or success. The first failed invocation remains primary when its timing is known; individually intact partial responses can still support observations. A later successful duplicate cannot replace it. Missing start evidence, tied starts, a nonterminal primary, inconsistent timestamps, an off-window collection or inventory fetched before the slot window closed prevents selection.

Every unselected or unavailable observation remains in the fixed denominator. Pending future slots are separately identified; they are not described as completed failures. Metadata/body retrieval failure is unknown evidence. Engineering targets and reruns never enter the primary scheduled sample, and no window is extended to replace missed observations.

Before a selected capture is read, its local status bytes must match both recorded raw and canonical hashes. The capture manifest must match the coordinator's byte length/hash, exact target, collector start and finish bound. All later source and book response checks still apply. Reusing another invocation's capture cannot satisfy this binding.

## Station-level interpretation

`scripts/weather_source_analysis.py` combines source, rules, lifecycle, timing and book checks. It produces separate `overlap`, `absence` and `unknown` counts by city, target date and city/target pair, retaining all planned observations. Repeated checkpoints are not treated as independent outcomes, and no confidence interval or trading return is generated.

An overlap requires at least one verified official daily report and a report-consistent displayed offer that passes every relevant identity, lifecycle, unit, depth and pairing check. Other failed components remain disclosed through `complete_components=false`; one verified pair establishes an observed occurrence, not full coverage or a fill.

An absence requires all station components to be known and no qualifying pair. For example, an explicit `no_report` with otherwise complete eligible evidence establishes absence at that checkpoint. The same source record plus a missing book remains unknown. Unsupported fee metadata also prevents a complete-absence claim, while a verified displayed overlap can still be recorded with fee comparison unavailable. Conflicting close times and unsupported rules stay unknown instead of becoming favorable observations.

There is no CLI flag to override reserved release. Full reserved evaluation requires the fixed release time and count-complete metadata fetched after that time, with every discovered attempt terminal. Before then, only operational inventory information is emitted: no source values, quotes, inferred outcomes, availability counts or payout headroom. Missing artifacts after release remain unknown, rather than being backfilled or replaced.

## Preserved unit authority and fee limitation

The previously cited client asset now returns HTTP 404. A newly discovered [public client asset](https://weather.com/vc-ap-7f3f87/_next/static/chunks/16tzq6chnik5a.js?dpl=dpl_6EuN2eXkZ12qptNwuwESajgPCKR5) explicitly describes daily climate figures as Fahrenheit and limits the unit toggle to hourly observations. Its new receipt was **September 25, 18:46:54.998144 UTC**; its exact 746,239 bytes have SHA-256 `cf82c9a3d87111091a12054bb904be96f4d6c587606c54629f0b4f5c5cff7a5d`.

The [authority evidence bundle](2026-09-25-source-authority-evidence.json.gz) preserves that asset, its public discovery chain, the expired-asset response and failed direct fee-document retrievals. Its uncompressed SHA-256 is `3ce6cdd4823bd0067d9ac9c52f4e5149ae4c0b16a8426f529f139bddacb5c492`. The evaluator requires the reviewed exact URL/hash, successful complete body, matching bytes and ordered real request/receipt/finish times. A caller-supplied unit boolean is not the study entry point's evidence basis. This new asset does not reconstruct the missing old receipt.

Exact fee rounding remains unresolved. The saved series metadata lacks coefficient/rounding detail; the [official fee document](https://kalshi.com/docs/kalshi-fee-schedule.pdf)'s extracted prose and examples require reconciliation. The evaluator therefore continues to emit `payout_comparison_available=false`. It does not equate a price below binary payout with positive expected return or reuse paper-fee assumptions as authoritative transaction costs.

## Verification and use

All **86 new offline checks pass**: 28 inventory/selection tests, 14 exporter tests, 20 snapshot/unit tests and 24 study/binding tests. They cover incomplete pagination, reruns, failed earliest invocations, missing timing, stale inventory, preserved failures, raw/decoded tampering, artifact binding, exact denominators, unit evidence, partial overlap, missing-book ambiguity and the reserved gate. Existing collector/protocol and canonical capture/ledger hashes are unchanged.

The [real inventory-check bundle](2026-09-25-source-inventory-evidence.json.gz) preserves the zero-run response, exact metadata, hashes and operational test outputs. The single GET returned HTTP 200 with a 36-byte body, SHA-256 `a2790a384d7d281e7395679000c35d27768d89dbd7052f725b8f4688beb59915`. This verifies the integration before the study starts; it is not a zero-opportunity finding.

The inventory bundle's uncompressed SHA-256 is `79da3dae98a29a16a17ea19144ef458710a48758f6e85eb18138585019b0b960`.

After collecting and preserving actual artifacts, use fresh, unique output paths:

```bash
python3 scripts/weather_source_inventory_capture.py \
  --output-dir data/raw/NEW-INVENTORY \
  --artifacts-root data/raw/DOWNLOADED-ARTIFACTS

python3 scripts/weather_source_analysis.py \
  --inventory data/raw/NEW-INVENTORY/inventory.json \
  --phase development \
  --authority-evidence reports/2026-09-25-source-authority-evidence.json.gz \
  --output data/raw/NEW-DEVELOPMENT-REPORT.json
```

The artifact root must contain the original uniquely named artifact directories, each with its `run` directory. The exporter does not repair missing downloads. Replace placeholders explicitly; commands refuse to overwrite previous inventory directories or analysis outputs. Preserve every vintage and operational repair.

The first planned observation is September 26 at 13:15 UTC. Independent source/price evidence and a separately frozen trading candidate are still missing. This implementation addresses evaluation integrity; it does not fulfill the broader goal of establishing alpha.

This commit records the availability-analysis definition before development observations. Any changes must preserve version history and obey the original pre-validation freeze deadline. It does not freeze a trading candidate or convert subsequently inspected data into an untouched strategy holdout.
