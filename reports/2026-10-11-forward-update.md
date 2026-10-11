# October 11: hosted loss guard verified; challenger pending

**The original pilot remains at −$89.53 settled paper P&L, unchanged since October 2.** All four October 8–11 primary hosted runs stopped new entries at the full-history drawdown guard. The separate prospective scale-only challenger has ten saved selections, two settled for **−$2.44 including modeled fees**, and eight unresolved with **$118.75 fee-inclusive paper cost**. There is no validated alpha or live authorization.

This audit uses canonical commit `15440665ca4d65c2aeb658e594df3fd2ad3ff780`. Four daily capture commits were pulled without altering the original checkout's dirty capture/dashboard files. The [machine-readable update](2026-10-11-forward-update.json) and exact evidence bundles below preserve the findings. No strategy, registration, risk limit or live setting changed.

## Original pilot: no new losses or entries

The canonical ledger still contains 7,985 paper rows. Market-shape accounting has 84 settled orders, 43 wins, no open intentions, $1,240.69 principal, $45.84 modeled fees and −$89.53 net. The reported −7.22% ROI uses principal as its denominator. That cost definition differs from the challenger's fee-inclusive cost. The live-admission gate remains NO-GO at 84/100 settlements and negative ROI. These are paper results, not verified account returns.

The full-history curve has a +$78.34 peak and **$167.87 current drawdown**. Each primary run logged that exact drawdown and `STAND DOWN`; each backup logged that the day's capture was already committed and skipped both pilot and challenger.

| Capture date | Primary run | Backup run | Pilot daily P&L change | Saved primary challenger selections |
|---|---|---|---:|---:|
| October 8 | [37796559005](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/37796559005) | [37802096209](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/37802096209) | $0.00 | 1 |
| October 9 | [37948041385](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/37948041385) | [37952673730](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/37952673730) | $0.00 | 1 |
| October 10 | [38061414178](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/38061414178) | [38064061095](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/38064061095) | $0.00 | 3 |
| October 11 | [38149064655](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/38149064655) | [38151753680](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/38151753680) | $0.00 | 5 |

October 8's trailing weekly result had already improved mechanically to approximately −$6.92 as older losses aged out. The new full-history guard nevertheless blocked entries. This verifies the deployed protection; it does not show strategy recovery or guarantee a $50 maximum account loss. All four primary logs have an empty `PILOT_LIVE`; the repository variables `PILOT_LIVE`, `PILOT_DISABLE` and `KALSHI_BASE_URL` were unset at audit time.

## Prospective challenger: two outcomes are insufficient

The [frozen registration](2026-10-07-challenger-preregistration.md) names scale-only as the sole primary family. The scorer was committed and synthetically tested in [PR #58](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/pull/58) before this first prospective performance inspection. Its original saved selections were scored without refitting or reranking.

Assessment time is **October 11, 21:47:59.852506 UTC**. Eight original scheduled runs supply four decision archives and four verified backup skips. All four archive digests match GitHub metadata, and each capture input matches its dated Git commit. Evidence for October 8–11 is valid with no errors; future dates remain pending. Valid provenance is not complete trial coverage.

| Selected target date | Orders | Settled net | Unresolved fee-inclusive paper cost |
|---|---:|---:|---:|
| October 9 | 1 | +$12.48 | $0.00 |
| October 10 | 1 | −$14.92 | $0.00 |
| October 11 | 3 | unavailable | $44.68 |
| October 12 | 5 | unavailable | $74.07 |

Dallas October 9 YES won: 27 contracts at $0.52 plus $0.48 fee, $14.52 total cost and +$12.48 net. Atlanta October 10 YES lost: 49 contracts at $0.29 plus $0.71 fee, $14.92 cost and −$14.92 net. Combined settled cost is $29.44 and settled net is −$2.44. All ten selections cost $148.19 including $4.71 modeled fees. If all eight unresolved selections lose, total net would be −$121.19; this is a conditional downside calculation, not a forecast.

The frozen verdict is **pending**. Full-sample net, ROI, bootstrap, drawdown and best-city exclusion remain null because outcomes and the trial window are incomplete. The trial retains October 8–December 6 capture dates and an exclusive December 22 00:00 UTC outcome-receipt cutoff. No early success, family substitution, date extension or pooling is allowed. The minimum 100 settlements, 30 traded target dates and two negative target-day groups are only part of the fixed criteria; no passing result would authorize live trading.

## Separate source study: preserve evidence, keep reserved comparisons locked

The latest operational listing contains 83 original scheduled runs. This audit preserves **24 new original ZIPs** beyond the 59 already retained: 20 complete collection manifests and four `creation_off_schedule` skips. All 4,082,213 ZIP bytes match their preserved GitHub metadata and receipts. Only bootstrap, coordinator and manifest metadata were inspected; source/quote ZIP members remain unread. These counts describe collection operations, not eligible snapshots or opportunities.

The October 7 inventory blocker persists: run `37470401551` has attempt creation `2026-10-06T13:23:24Z`, but start and listing/coordinator creation `13:23:23Z`. The newly retained run `37888372249` has the same contradiction on October 9: attempt creation `05:23:50Z`, start and listing/coordinator creation `05:23:49Z`. Two other new runs differ in `updated_at` only. All original values remain preserved. The separate source protocol's strict validation rules are unchanged; the challenger's different preregistered metadata contract does not amend them.

No full source inventory or evaluation was rerun. The previous development verdict remains **not evaluated**, with all 1,260 checkpoints unknown; unknown does not mean zero observed opportunities. Reserved comparison stays locked until **October 24, 09:30 UTC**, and also requires a fresh valid terminal inventory. Nine source protocol/code/workflow files match the October 9 pre-freeze commit `2dfb1ff`. No source trading candidate was found frozen in committed main before October 10. Reserved availability alone therefore cannot establish independent strategy validation; the separate challenger registration supplies no source candidate retroactively.

Four progress-print errors occurred after skipped-run archives were saved because their slot field is null. Original diagnostics remain preserved. Separate offline verification confirmed every expected archive, receipt and operational member; there were no retries or substitutions.

## Evidence and reproduction

| Portable bundle | Files | Compressed bytes | SHA-256 |
|---|---:|---:|---|
| [Pilot guard](2026-10-11-pilot-guard-evidence.json.gz) | 34 | 125,861 | `7b5164fffcb1948fb859b7837d0ca79891c25b54976bbd739524029822203f6d` |
| [Challenger](2026-10-11-challenger-evidence.json.gz) | 158 | 8,403,281 | `5a94492a56b5bf969a21c0da421201ef1599054e50b6ffbfd3df51eb7c866008` |
| [Source operational update](2026-10-11-source-operational-evidence.json.gz) | 226 | 4,701,405 | `4aaef07af6291e30ae34ef6c74b92b7cf97627e9d93bd29269a2798ba2159513` |

Each gzip contains JSON with a `files` array of relative `path`, `bytes`, `sha256` and `base64`. Extract only into a fresh directory, reject absolute/traversing/duplicate paths, and verify every decoded length and SHA-256 before use. Source ZIPs must remain opaque during the embargo. The source update is incremental, not a complete terminal study inventory. Hashes establish byte consistency, not independent server authentication.

All bundles were decoded into fresh ignored directories and verified. The frozen scorer reproduced the original challenger assessment byte-for-byte after extraction:

```sh
python3 scripts/weather_challenger_forward.py \
  --evidence EXTRACTED/evidence \
  --as-of 2026-10-11T21:47:59.852506+00:00 \
  --output EXTRACTED/reproduced-assessment.json
```

Expected assessment SHA-256 is `68ffb54330a009db31d8a61158038ecce013e872d301ff4f58a0413d2338769a`; exit zero means a report was produced, not admission. The pilot audit and admission gate reproduce with `python3 scripts/pilot_alpha_audit.py` and `python3 scripts/go_live_gate.py --enforce --json` (expected NO-GO exit 1). Canonical capture SHA-256 is `3bcb43e261a9ebbc7b0a68a0288de86e99d5a058f0c58da7e6aa7d5792d50205`; ledger SHA-256 is `8945f1bcf8d87c07cf4bd2f3e495c360ecdb441bab106518043751588655b2e8`.

Next evidence comes from the existing scheduled paper captures and subsequent outcome receipts. The pilot remains under the full-history guard. No candidate or control should be promoted from this partial sample.
