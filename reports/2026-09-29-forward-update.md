# September 29 capture: paper P&L turns negative after a $34.44 settlement loss

**Cumulative settled paper P&L is −$20.02 after modeled fees across 77 orders.** Four September 28 target orders newly resolved in the September 29 capture contributed **−$34.44**, taking the previous +$14.42 balance below zero. The decline since the September 25 balance of +$58.54 is **$78.56**. Alpha remains unproven; admission remains **NO-GO**, now failing both the minimum sample and positive-return requirements.

This audit uses canonical commit `53c9f30eaaed2593240c58594a1f1e3d6874ecfe`, compared with `0edf4c348388062feabc04c65a12c6625214f424`. The selection rule, fee model, risk limits and frozen scorer are unchanged. These are paper results from recorded intentions and captured settlement labels, not verified live execution. Capture dates describe when the repository reflected outcomes; target dates identify the weather events.

## Settlement bridge and uncertainty

| September 28 target | Intended position | Payout | Modeled fee | Net |
|---|---|---:|---:|---:|
| Boston `KXHIGHTBOS-26SEP28-B61.5` | 51 YES × $0.29; $14.79 principal | $0.00 | $0.74 | −$15.53 |
| Austin `KXHIGHAUS-26SEP28-B97.5` | 27 YES × $0.54; $14.58 principal | $27.00 | $0.47 | +$11.95 |
| Philadelphia `KXHIGHPHIL-26SEP28-B67.5` | 49 YES × $0.30; $14.70 principal | $0.00 | $0.73 | −$15.43 |
| Atlanta `KXHIGHTATL-26SEP28-B82.5` | 37 YES × $0.40; $14.80 principal | $0.00 | $0.63 | −$15.43 |
| Total | $58.87 principal | $27.00 | $2.57 | **−$34.44** |

All previously settled orders retain their settlement economics. The full sample has **41 wins in 77 settlements across 19 target days**, $1,136.56 principal and $41.46 modeled fees. Fee-inclusive ROI is **−1.76%**, with a target-day bootstrap 95% interval of **−27.06% to +27.20%**. Independent Decimal accounting reproduced the totals to the recorded floating-point precision.

Holding selected contracts fixed and worsening each entry by one cent gives **−$49.85** under the existing sensitivity calculation. That calculation does not reprice fees or establish that the intentions could have filled.

| Capture date | Cumulative settled paper P&L |
|---|---:|
| September 25 | +$58.54 |
| September 27 | +$28.36 |
| September 28 | +$14.42 |
| September 29 | **−$20.02** |

The [September 28 audit](2026-09-28-forward-update.md) documents the earlier bridges and the distinct −$42.75 cumulative reading from September 22.

## Frozen NO shadow

The September 21 rule keeps NO orders from the existing pilot's selected top five, without replacing excluded YES orders. It remains at **two prospective selections, both settled wins, +$2.98 total, and no open selections**. Both were Seattle orders on consecutive target dates. No new NO observation arrived.

The reported bootstrap interval remains degenerate at **+10.3472% ROI** because both observed target-day returns have identical economics. Two same-city observations do not establish negligible uncertainty or validated alpha. Historical side attribution is **NO +$66.10 on 23 settlements** versus **YES −$86.12 on 54**; it includes the observations used to select the shadow and remains exploratory. No filter is promoted from these totals.

## Open paper exposure

Four paper intentions remain unresolved, with **$59.46 principal**:

| Target | City / side | Contracts × price | Principal |
|---|---|---:|---:|
| September 29 | Denver YES | 93 × $0.16 | $14.88 |
| September 30 | Austin YES | 83 × $0.18 | $14.94 |
| September 30 | Chicago YES | 37 × $0.40 | $14.80 |
| September 30 | NYC YES | 53 × $0.28 | $14.84 |

The three new September 30 orders total **$44.58** and were recorded at approximately `2026-09-29T14:57:20.4096Z`. Their tickers are `KXHIGHAUS-26SEP30-B93.5`, `KXHIGHCHI-26SEP30-T69` and `KXHIGHNY-26SEP30-B72.5`. Open-intent fees are not included in settled P&L; the intentions are not fill evidence.

## Rolling weekly risk and admission

The current target-date window is **September 22–28 inclusive**: 12 settled orders, six wins, $176.76 principal and $7.51 modeled fees, giving **+$22.73 net**. The Rust pilot's unrounded fee approximation gives **+$22.812468**, displayed as +$22.81.

The weekly total rose because a larger losing cohort aged out:

`+$15.73 previous week − (−$41.44 September 21 cohort) − $34.44 new settlements = +$22.73`.

Thus the existing −$50 weekly breaker remains clear despite the cumulative drawdown. The unchanged enforced admission gate exits 1 with **NO-GO: 77/100 settlements and negative fee-inclusive ROI**. Other existing criteria pass: Austin accounts for 13.85% of realized losses, structural eligibility is reported satisfied, and zero live orders await verification. There are zero live settlements.

The [primary workflow run](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/36586239047) had `PILOT_LIVE` empty and recorded the new orders as dry runs. Its gate output preceded those three new intentions, so it showed one open order; the final ledger has four.

## Input integrity and source research

Captures grew from 10,425 to **10,544**, adding 90 Kalshi rows for September 30 (six per study city) and 29 Polymarket rows. The update resolved 125 previously null outcomes: 90 Kalshi September 28 rows, 32 Polymarket September 28 rows and three Polymarket September 29 rows. There were no deletions, duplicate identities, known-label revisions, field loss or settlement-metadata revisions. Other historical changes were 175 floating-point serialization differences across 94 rows, at most approximately `3.55e-15`. All **540 raw rule hashes** verify.

The ledger retains the complete previous byte prefix, adding 179 decisions: three orders, 41 edge-below-cost skips, 46 price-floor skips and 89 lead-zero skips. All **7,808 ledger rows remain marked dry**.

The separate [September 29 hosted source study audit](2026-09-29-hosted-source-study.md) records prospective source evidence. Those observations do not change this ledger, validate intended fills or authorize live promotion.

## Reproduction

The [machine-readable score](2026-09-29-forward-update.json) exactly reproduces with the unchanged `scripts/pilot_alpha_audit.py`. Workspace canonical inputs match commit `53c9f30` byte for byte:

- Captures SHA-256: `ed515ff9ca7dd145ecd20c41208943c3cb10fe230ecf45b6c78cff2e56babc52`
- Ledger SHA-256: `70301d5f4800072014600f4508d6faa5159f6a8419368e550a96425595618e37`

From the repository root:

```sh
python3 scripts/pilot_alpha_audit.py
python3 scripts/go_live_gate.py --enforce --json
```

The first command reproduces the score; the second returns expected NO-GO exit code 1. Frozen scorer hashes match both canonical commits and the workspace:

- `pilot_alpha_audit.py`: `615df82678438722b1c6ffc8ff08a4fd479485aa0b47d186c6eb7c5c5c5343fc`
- `market_shape_alpha.py`: `2c309132a39db89c55dda8daeab995a5dde12f0cfdde3f6a20056129aecf12e7`
- `go_live_gate.py`: `061692e86b19748c878458e3ed303cc563f854fb77b871ec269958e01d396da4`
