# September 28 capture: another YES loss reduces cumulative paper P&L to +$14.42

**Cumulative settled paper P&L is +$14.42 after modeled fees across 73 orders.** The September 28 capture added a Seattle NO win of +$1.49 and a Philadelphia YES loss of −$15.43, both for September 27 targets. Their net **−$13.94** reduces the September 27 capture balance of +$28.36. The cumulative decline since the September 25 balance of +$58.54 is **$44.12**. Alpha remains unproven and admission remains **NO-GO**.

This audit uses canonical commit `0edf4c348388062feabc04c65a12c6625214f424`, compared with the September 27 canonical commit `151b15ef5bf58cbd8d02c586ca5426fc9ecde330`. The selection rule, fee model, risk limits and frozen scorer are unchanged. These figures describe paper accounting from recorded intentions and captured settlement labels, not verified live execution.

## What the roughly −$40 reading meant

The verified roughly −$40 reading was **−$42.75 cumulative paper P&L in the September 22 capture**. It was not a verified claim that September 27 alone lost $40. The September 27 capture showed **+$28.36 cumulative**; the September 27 target orders newly settled in the September 28 capture contributed **−$13.94**, leaving **+$14.42 cumulative**. Capture dates describe when the repository reflected the outcomes; they are not interchangeable with target dates or a complete calendar-day equity statement.

| Capture date | Cumulative settled paper P&L | Change from the preceding listed capture |
|---|---:|---:|
| September 22 | −$42.75 | — |
| September 25 | +$58.54 | +$101.29 |
| September 27 | +$28.36 | −$30.18 |
| September 28 | **+$14.42** | **−$13.94** |

The September 25–27 bridge is documented in the [previous audit](2026-09-27-forward-update.md). A positive cumulative balance does not erase the subsequent loss interval or establish positive expected returns.

## Newly reflected settlements and uncertainty

| September 27 target | Intended position | Payout | Modeled fee | Net |
|---|---|---:|---:|---:|
| Seattle `KXHIGHTSEA-26SEP27-B61.5` | 16 NO × $0.90; $14.40 principal | $16.00 | $0.11 | +$1.49 |
| Philadelphia `KXHIGHPHIL-26SEP27-B65.5` | 49 YES × $0.30; $14.70 principal | $0.00 | $0.73 | −$15.43 |
| Total | $29.10 principal | $16.00 | $0.84 | **−$13.94** |

All previously settled orders retain the same settlement economics. The complete sample has **40 wins in 73 settlements across 18 target days**, $1,077.69 principal and $38.89 modeled fees. Fee-inclusive ROI is **+1.34%**, with a target-day bootstrap 95% interval of **−24.21% to +32.15%**.

Holding all selected contracts fixed and worsening each entry by one cent produces **−$13.77** under the existing sensitivity calculation. That calculation does not reprice fees or establish that the intended positions could have filled. The broad uncertainty interval and sensitivity to entry prices do not support an alpha claim.

## Frozen NO shadow

The September 21 rule keeps NO orders from the existing pilot's selected top five, without replacing excluded YES orders. It now has **two prospective selections, both settled wins, totaling +$2.98**, with no open selections. Both are Seattle orders on consecutive target dates, September 26 and 27, each for 16 NO contracts at $0.90 with a $0.11 modeled fee.

The scorer reports a bootstrap interval whose two endpoints both equal **+10.3472% ROI**. This degeneracy occurs because the two observed target-day returns have identical economics: resampling them cannot generate a different return. It is not evidence of negligible uncertainty, robustness across cities, or validated alpha. The sample remains two same-city observations.

Historical side attribution now shows **NO +$66.10 on 23 settlements** and **YES −$51.68 on 50**. These totals include the observations used to choose the shadow and remain exploratory. No side filter is promoted and no replacement strategy is inferred from this attribution.

## Open paper exposure

Five paper intents remain unresolved in the latest capture, with **$73.75 principal**:

| Target | City / side | Contracts × price | Principal |
|---|---|---:|---:|
| September 28 | Boston YES | 51 × $0.29 | $14.79 |
| September 28 | Austin YES | 27 × $0.54 | $14.58 |
| September 28 | Philadelphia YES | 49 × $0.30 | $14.70 |
| September 28 | Atlanta YES | 37 × $0.40 | $14.80 |
| September 29 | Denver YES | 93 × $0.16 | $14.88 |

Denver `KXHIGHDEN-26SEP29-B65.5`, recorded at `2026-09-28T14:58:02.407392717Z`, is the only newly selected order. Open-intent fees are not yet included in settled P&L. These are recorded paper intentions, not fill evidence or claims about subsequently observed outcomes.

## Rolling weekly risk and admission

The current target-date window is **September 21–27 inclusive**: 13 settled orders, seven wins, $191.90 principal and $7.37 modeled fees, for **+$15.73 net**. The Rust pilot's unrounded fee approximation gives **+$15.802374**, displayed as +$15.80.

The weekly total improved despite the new losses because a losing cohort aged out:

`+$13.24 previous week − (−$16.43 September 20 cohort) − $13.94 new settlements = +$15.73`.

The existing −$50 weekly breaker remains clear without an override. The unchanged enforced admission gate exits 1 with **NO-GO at 73/100 settlements**. Its other existing criteria pass, including zero unverified live orders. There are zero live settlements. A future pass of the sample-count threshold alone would not establish the broader alpha claim.

## Input integrity and prospective source research

Captures grew from 10,313 to **10,425**, adding 90 Kalshi and 22 Polymarket rows. The update resolved 138 previously null outcomes: 90 Kalshi September 27 rows and 48 Polymarket rows across September 25–27. There were no deletions, duplicate capture identities, known-label revisions or field loss. Other historical changes were 137 floating-point serialization differences across 91 rows, at most approximately `3.55e-15`.

The ledger preserves its prior byte prefix and adds 176 decisions: one order, 51 edge-below-cost skips, 38 price-floor skips and 86 lead-zero skips. All **7,629 ledger rows remain marked dry**.

The separate [hosted source study audit](2026-09-28-hosted-source-study.md) documents prospective collection evidence and its limitations. Source-study observations are separate from this pilot ledger; they do not change these settlement totals, validate intended fills, or authorize live promotion.

## Reproduction

The [machine-readable score](2026-09-28-forward-update.json) is produced by unchanged `scripts/pilot_alpha_audit.py`. Canonical input SHA-256 hashes are:

- Captures: `da46f58839ae37d55338c1256076d7113a22d7e177db2cc71a23af8a8a8cea19`
- Ledger: `5df886129bcc3c4852f021cd222073f541bf8daf7560754082a073da98e07681`

Run these commands from the repository root against the canonical inputs:

```sh
python3 scripts/pilot_alpha_audit.py
python3 scripts/go_live_gate.py --enforce --json
```

The first command reproduces the score; the second returns the expected NO-GO exit code 1. Verified frozen scorer hashes match both canonical commits and the audited workspace:

- `pilot_alpha_audit.py`: `615df82678438722b1c6ffc8ff08a4fd479485aa0b47d186c6eb7c5c5c5343fc`
- `market_shape_alpha.py`: `2c309132a39db89c55dda8daeab995a5dde12f0cfdde3f6a20056129aecf12e7`
- `go_live_gate.py`: `061692e86b19748c878458e3ed303cc563f854fb77b871ec269958e01d396da4`

No selection or admission rule was changed to accommodate the new losses or the two NO wins.
