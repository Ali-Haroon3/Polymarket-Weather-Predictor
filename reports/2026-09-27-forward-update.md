# September 27 capture: two YES losses and the first prospective NO settlement

**Cumulative settled paper P&L is +$28.36 after modeled fees across 71 orders.** The September 26–27 captures added two YES losses totaling −$31.67 and one NO win of +$1.49, a net decline of **$30.18** from the September 25 balance. The first frozen prospective NO-shadow settlement won; one observation does not establish alpha. Admission remains **NO-GO**.

This audit uses canonical commit `151b15ef5bf58cbd8d02c586ca5426fc9ecde330`, pulled into the research branch without changing selection, fee modeling, risk limits or frozen experiments. The audit and research-inventory check ran on September 28 UTC (September 27 in America/Denver). All P&L below is paper accounting, not verified live execution.

## Settlement bridge

| Capture date | Newly reflected settlement | Net after modeled fee | Cumulative net |
|---|---|---:|---:|
| September 25 | Previous audited balance | — | +$58.54 |
| September 26 | Atlanta YES, September 25 target | −$15.93 | |
| September 26 | Austin YES, September 25 target | −$15.74 | +$26.87 |
| September 27 | Seattle NO, September 26 target | +$1.49 | **+$28.36** |

Atlanta `KXHIGHTATL-26SEP25-B80.5` lost its $15.00 principal plus $0.93 modeled fee. Austin `KXHIGHAUS-26SEP25-B99.5` lost $14.85 plus $0.89. Seattle `KXHIGHTSEA-26SEP26-B59.5` paid $16 against $14.40 principal and $0.11 fee. Earlier settlement labels are unchanged.

The full sample has 39 wins, $1,048.59 principal and $38.05 modeled fees: **+2.70% ROI**. The target-day bootstrap 95% interval is **−23.22% to +33.86%**. Holding selected contracts fixed and worsening every entry by one cent leaves only **+$0.82** under the existing sensitivity calculation; this does not reprice fees or establish fillability.

The original roughly −$40 reading was the **September 22 cumulative balance of −$42.75**. The later positive balance does not erase that loss interval or demonstrate positive expected returns.

## Frozen NO shadow and open paper exposure

The September 21 rule keeps NO orders from the existing pilot's selected top five, with no replacement for excluded YES orders. It now has **two prospective selections: one settled win (+$1.49) and one open**. The open selection is Seattle `KXHIGHTSEA-26SEP27-B61.5`, entered September 26 at $0.90 for 16 NO contracts. The scorer's `shadow_rank_then_no.forward` section reports settlements only.

Historical selected NO totals +$64.61 on 22 settlements; YES totals −$36.25 on 49. This side attribution includes the data used to choose the shadow. Its apparent historical advantage is exploratory and cannot substitute for a new prospective sample. No side filter is promoted.

Six intents remain open in the latest canonical ledger, with **$87.97 principal**:

| Target | City / side | Contracts × price | Principal |
|---|---|---:|---:|
| September 27 | Seattle NO | 16 × $0.90 | $14.40 |
| September 27 | Philadelphia YES | 49 × $0.30 | $14.70 |
| September 28 | Boston YES | 51 × $0.29 | $14.79 |
| September 28 | Austin YES | 27 × $0.54 | $14.58 |
| September 28 | Philadelphia YES | 49 × $0.30 | $14.70 |
| September 28 | Atlanta YES | 37 × $0.40 | $14.80 |

These are unresolved paper intents at the September 27 capture, not fill records or claims about outcomes observed later. Their fees are not included in settled P&L yet.

## Rolling weekly risk

The current target-date window is **September 20–26**, inclusive: 16 settled orders, eight wins, **+$13.24** after rounded modeled fees. The Rust pilot's unrounded approximation gives **+$13.32405**, displayed as +$13.32.

The weekly result improved primarily because losses aged out, not because new orders earned $42.95:

- September 26: `−6.38 − (−8.34) − 31.67 = −29.71`; September 18 losses left the window while Atlanta and Austin settled.
- September 27: `−29.71 − (−41.46) + 1.49 = +13.24`; September 19 losses left the window and Seattle settled.

The existing −$50 breaker is clear without an override. The unchanged enforced admission gate exits 1 with **NO-GO at 71/100 settlements**. Its other existing criteria pass; no live orders or unverified live liabilities are recorded. Passing a future sample-count threshold alone would not prove the broader alpha claim.

## Input integrity and scheduled runs

Captures grew from 10,071 to 10,188 on September 26 (+90 Kalshi, +27 Polymarket), then to 10,313 on September 27 (+90 Kalshi, +35 Polymarket). The respective updates resolved 132 and 126 formerly null outcomes. There were no deletions, duplicate identities, known-label revisions, field loss or historical metadata changes. Other historical edits were only floating-point serialization differences, at most approximately `7.11e-15`.

Each new Kalshi cohort contains 15 complete six-leg ladders for the next target day. All 360 metadata-bearing rows retain valid raw-rule hashes. The ledger preserves its prior byte prefix and appends 179 then 178 dry decisions, including two then four orders; all 7,453 rows remain marked dry. The [September 26 primary run](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/36250005374) and [September 27 primary run](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/36327512945) succeeded with `PILOT_LIVE` unset; both backups stood down after finding existing daily captures.

A new Polymarket Hong Kong row, `4900642`, was captured September 27 for a September 26 target. It remains unresolved and is excluded from forecast shrinkage calibration and this Kalshi pilot/study. The separate generic dashboard replay permits explicitly labeled post-day observations; its aggregate is not the prospective pilot sample scored here. The new row contributes no realized P&L at this audit.

## Source study: missing observations remain missing

The proposed research workflow remains an open draft in [PR #50](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/pull/50), absent from the default branch. A fresh read-only inventory query received at **September 28, 04:23:52.831400 UTC** returned HTTP 200 and zero scheduled runs in the fixed study window.

At the offline check at 04:24:09 UTC, **10 development slots had closed without an invocation**: all six September 26-target slots and the first four September 27-target slots. They represent **150 missing station-checkpoints**. Another **74 slots / 1,110 station-checkpoints** remained pending, within the unchanged **84-slot / 1,260-observation development denominator**. Zero observed overlaps and absences here reflect no evaluated snapshots, not evidence of no opportunity. The development state remains `not_evaluated`; reserved validation stays `locked` and emits no source-comparison counts.

The next nominal slot after that check is September 28 at 05:15 UTC, for the September 27 target. Collection requires the prepared workflow to reach the default branch; this report does not claim that deployment or any future run occurred. Missed slots cannot be recovered with a later response, rerun, shifted window or favorable replacement. The original October 10 freeze and October 24 release remain unchanged.

The [preserved metadata bundle](2026-09-28-source-inventory-evidence.json.gz) contains exact response bytes, request/receipt times, inventory and operational evaluation summaries. Its uncompressed SHA-256 is `72563ca3eddb0a8e71e59473c7f2e11da354fe3ed4a6776ed82e8b374c3ae865`. The response body is 36 bytes with SHA-256 `a2790a384d7d281e7395679000c35d27768d89dbd7052f725b8f4688beb59915`. No source or book observations were fetched for this check.

## Reproduction

The [machine-readable score](2026-09-27-forward-update.json) is produced by unchanged `scripts/pilot_alpha_audit.py`. Inputs match canonical commit `151b15e` exactly:

- Captures: `720b974a02145e97030c66b0c63569c4adc1e71ca9c664a2177640cd5356a02b`
- Ledger: `e6a2e7a9a2a1dd69649debfcf4b8b39bc8a6983d2dd28ae87130db773483036c`

Run `python3 scripts/pilot_alpha_audit.py` and `python3 scripts/go_live_gate.py --enforce --json` against those inputs. The frozen protocol, source collector, strategy scorer and admission script are unchanged. The current evidence supports continued paper research; it does not establish validated alpha or authorize live promotion.
