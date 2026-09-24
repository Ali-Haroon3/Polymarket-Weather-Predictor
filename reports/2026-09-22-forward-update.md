# September 22 capture: paper losses increase, shadow remains untested

**The current paper pilot is now −$42.75 after modeled fees on 65 settled orders.** Five newly resolved orders added **−$41.44**. No replacement strategy is promoted, and the enforced admission gate remains **NO-GO**.

This update uses upstream capture commit `20e0a9fcb36a89e8aafcbe6c2e36b4a6c370b362`, merged without changing any strategy rule. The [September 22 workflow](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/35743496644) completed successfully. Its log confirms dry mode, no trading credentials, and one new paper intent. The previous [September 21 audit](2026-09-21-alpha-audit.md) remains a dated historical snapshot.

## Fixed-rule score

| Metric | Through September 21 capture | Through September 22 capture |
| --- | ---: | ---: |
| Settled paper orders | 60 | 65 |
| Wins | 33 | 35 |
| Purchase cost, excluding fees | $885.79 | $959.80 |
| Modeled rounded fees | $31.52 | $33.95 |
| Net profit | −$1.31 | **−$42.75** |
| ROI after fees | −0.15% | **−4.45%** |
| Unresolved orders | 7 | 3 |

The full pilot's target-day bootstrap 95% ROI interval is **−28.36% to +23.09%** across 13 target days. The losses are observed; this interval does not establish negative expected returns statistically. The existing same-contract one-cent price sensitivity is −$66.41; it is the original simplified attribution sensitivity, not the archive evaluator's separately recomputed-fee stress.

## Newly resolved orders

All five entered on **September 20** for **September 21** targets. They became labeled in today's capture; they are not newly entered prospective shadow decisions.

| Contract | Side | Contracts | Price | Net after modeled fees |
| --- | --- | ---: | ---: | ---: |
| KXHIGHTLV-26SEP21-T90 | NO | 17 | $0.87 | +$2.07 |
| KXHIGHDEN-26SEP21-B76.5 | YES | 48 | $0.31 | −$15.60 |
| KXHIGHTATL-26SEP21-B93.5 | YES | 57 | $0.26 | −$15.59 |
| KXHIGHNY-26SEP21-B66.5 | NO | 18 | $0.82 | +$3.05 |
| KXHIGHTPHX-26SEP21-B103.5 | YES | 36 | $0.41 | −$15.37 |
| **Increment** | | | | **−$41.44** |

The three YES losses contribute −$46.56; the two NO wins contribute +$5.12. Previously settled orders retain the same outcomes and economics. These five trades share one target day, so treating them as five independent validation days would overstate evidence.

## Prospective NO shadow

The frozen shadow retains NO orders from the parent's selected top five and never fills freed slots with replacements. Entry date must be **after September 21**, regardless of when settlement is observed.

Today's 178 ledger decisions include exactly one new order: **62 Boston YES contracts at $0.24**, $14.88 principal, targeting September 23 (`KXHIGHTBOS-26SEP23-B64.5`). It remains open and is excluded from the NO shadow. Therefore the shadow has **zero prospective NO selections, zero settlements, and no estimable ROI**.

The broader selected-NO historical slice is now +$63.12 on 21 settled orders, while YES is −$105.87 on 44. Those historical slices remain exploratory. Counting today's two pre-freeze NO winners as shadow results would introduce selection bias. The failed [frozen archive test](2026-09-22-archive-training-extension.md) is unchanged.

## Data integrity and deployment scope

The new commit adds 158 captures: 90 Kalshi rows forming 15 complete September 23 temperature ladders, plus 68 Polymarket rows. It removes no captures and fills 123 previously null outcomes. Known outcomes are not revised. Some existing floating-point values reserialize with differences no larger than approximately `7.1e-15`; no text or schema fields change. The prior ledger is an exact prefix of the new ledger. There are no duplicate appended identities, invalid/crossed new quotes, or new live orders.

Two non-temperature Polymarket rows have a title/target-date mismatch: Seattle rain `4754827` and tornado market `4788967` mention September 21 while carrying September 22 targets. They do not enter Kalshi market-shape scoring. This update does not validate unrelated non-temperature modeling.

The scheduled workflow still runs `main`, before the pending PR's admission/provenance changes. Accordingly these new captures have no settlement metadata. The draft PR now contains the merged current data; it does not imply those code changes are deployed. No live settings changed.

## Reproduction

[Machine-readable frozen-rule output](2026-09-22-forward-update.json) reproduces byte for byte using the capture and ledger from commit `20e0a9fcb36a89e8aafcbe6c2e36b4a6c370b362` with the unchanged `scripts/pilot_alpha_audit.py`.

- Capture SHA-256: `6fa869478563eb6ba09210278d2526c3db313b75a4708dc32130995862e8eedd`
- Ledger SHA-256: `cad692b0388df8bc36bac343fa1752a563706a1de4eb05522186929f9b041566`

Run `python3 scripts/pilot_alpha_audit.py` and `python3 scripts/go_live_gate.py --enforce --json` against those inputs. The latter exits 1: 65/100 settled orders and negative net ROI fail admission. Successful ingestion and tests do not establish alpha; new prospective selections and settled outcomes are still needed.
