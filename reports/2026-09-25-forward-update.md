# September 25 capture: first prospective NO selection, no new pilot settlements

**Cumulative paper P&L remains +$58.54 after modeled fees across 68 settlements.** The new Seattle NO intent is the first selection in the frozen prospective NO shadow, but it remains open. One prospective selection is evidence that the frozen rule is being exercised, not evidence of profitability. Admission remains **NO-GO**.

This scores canonical commit `740fe91ca76be63e1effc0fc93f7a35b2d37e03f` with unchanged scripts and selection rules. [PR #49](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/pull/49) was merged before the [September 25 scheduled capture](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/36150517381). That run used implementation commit `822c8b6bc123fbaa0291df6b6ac452951d6854cd` and was ledger-only dry run, with empty live and trade-credential settings.

## Settled score and open decisions

The settled score is identical to the [September 24 audit](2026-09-24-forward-update.md): 68 orders, 38 wins, $1,004.34 purchase cost, $36.12 modeled rounded fees and **+$58.54 net (+5.83% ROI)**. There are no newly settled pilot orders or revised pilot outcomes. The 15-target-day bootstrap 95% ROI interval remains **−20.77% to +39.12%**. The original simplified same-contract one-cent adverse-price sensitivity remains +$33.40; it is not the archive evaluator's repriced-fee stress.

| Contract | Entry date | Target | Side | Contracts | Price | Principal |
| --- | --- | --- | --- | ---: | ---: | ---: |
| KXHIGHTATL-26SEP25-B80.5 | September 24 | September 25 | YES | 125 | $0.12 | $15.00 |
| KXHIGHAUS-26SEP25-B99.5 | September 24 | September 25 | YES | 99 | $0.15 | $14.85 |
| KXHIGHTSEA-26SEP26-B59.5 | September 25 | September 26 | NO | 16 | $0.90 | $14.40 |

All three remain open, with **$44.25 principal**. Seattle was selected by the ordinary parent pilot at 14:56:23 UTC; filtering the parent's selected orders retains it without admitting replacement orders. Its entry date is strictly after the September 21 freeze. Therefore the prospective NO shadow now has **one selection, one open intent and zero settlements**.

The machine-readable scorer's `shadow_rank_then_no.forward` section summarizes settled orders only. Its zero settled count must not be described as zero selections. Neither the strategy nor that frozen scorer was changed to add this reporting distinction. Historical selected NO remains +$63.12 on 21 settled orders and YES −$4.58 on 47; that earlier side split remains exploratory.

Seattle's modeled rounded fee is $0.11, which is not included in settled fees yet. Under the current ordinary binary paper accounting, a winning NO would net $1.49 and a losing NO would net −$14.51. This is an unresolved dry-run intent, not an `unfilled` execution record or proof of a live fill.

## Rolling weekly risk and admission

The current rolling target-date window is **September 18–24**, inclusive. It contains 23 settled paper orders, including 11 wins, with $339.69 principal and $6.31 gross profit. The Rust pilot's unrounded fee approximation is $12.572777, giving **−$6.262777**, matching its logged **−$6.26**. Rounded per-order modeled fees total $12.69, giving **−$6.38**.

Five September 17-target orders with net +$5.36 aged out; there were no pilot orders targeting September 24. The rounded-fee bridge is **−1.02 − 5.36 = −6.38**. Both fee conventions remain above the existing −$50 stop threshold. No limit or breaker was overridden.

The unchanged enforced gate returns **NO-GO**, exit 1, at **68/100** settled orders. The other existing criteria pass, and no live orders are recorded. The inherited daily fit updated to Normal μ +0.0°C, σ ×0.95 on the newest 300 eligible ladders; its trailing 30-day replay was +4.9% on 621 trades. These are ordinary updates under the existing algorithm, not a newly validated model or a changed trading rule.

## Input integrity and local capture behavior

Captures grew from 9,934 to 10,071: 137 new rows, comprising 90 Kalshi rows in 15 complete September 26 ladders and 47 Polymarket rows. None were removed. There are 158 null-to-binary outcome updates: 90 Kalshi and 67 Polymarket results for September 24, plus one legacy Polymarket row. No known labels were revised. Other historical changes are 144 numeric reserializations across 93 rows, at most approximately `3.55e-15`; historical fields remain intact.

The 90 new Kalshi rows all preserve exact primary/secondary rules and valid hashes identifying The Weather Company. Total metadata-bearing rows are now 180. Historical metadata is unchanged, with no backfill. No duplicates, invalid quotes or chronology reversals were found.

The ledger preserves its old 6,918-row prefix exactly and appends 178 dry-run decisions: one order, 44 price-floor skips, 45 edge skips and 88 lead-zero skips, for 7,096 rows total. The workflow prints its gate and attribution before pilot selection, so its earlier two-open count precedes Seattle; the final ledger correctly has three open intents.

Two capture-quality caveats remain: legacy Polymarket `2047447` is a Seattle Storm basketball question mislabeled as a storm market; new Seattle rain `4905285` has a September 25 title but September 26 target. Both have null model estimates. Independent code tracing confirms that neither contributes to current dashboard strategy P&L, open positions, calibration or pilot scoring. They affect only the unfiltered raw capture counts: one resolved row and one pending row. Neither was used to claim alpha.

The local guarded wrapper is now observed running at **September 25, 09:00:23 MDT**. Its release build succeeded; the subsequent Polymarket request failed before capture wrote data. The wrapper still rendered the dashboard from the 10,071 existing canonical captures. This verifies the rebuilt wrapper and last-good-data fallback ran; it does not prove a successful local capture or a resolution of the upstream fetch failure. Canonical hashes match the cloud commit, with no sign of the earlier obsolete-binary field stripping.

## Reproduction and next evidence

The [machine-readable score](2026-09-25-forward-update.json) was independently reproduced byte for byte from unchanged `scripts/pilot_alpha_audit.py` against inputs extracted from commit `740fe91` into a temporary directory.

- Capture SHA-256: `d85362eb5c906cfa47b14e41b73894261de5da2b23c4f0a322c56957b56914d2`
- Ledger SHA-256: `388cb0b6827dbead8fa0c4cf1de5c05b0c24fb3e32abb45bc73e0b047bf6f85c`

Run `python3 scripts/pilot_alpha_audit.py` and `python3 scripts/go_live_gate.py --enforce --json` against those inputs. No frozen strategy or threshold changed. The next NO-shadow performance observation requires an actual venue outcome for the September 26 Seattle target.

Separately, the source-availability study needs prospective source/book observations at its preregistered times. The [schedule addendum](2026-09-25-source-schedule.md) prepares that collection without changing the existing protocol or analyzing reserved data. A proposed workflow is not evidence that future jobs have run or that a profitable strategy exists.
