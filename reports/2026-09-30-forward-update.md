# September 30 capture: another $15.76 paper loss brings cumulative P&L to −$35.78

**Cumulative settled paper P&L is −$35.78 after modeled fees across 78 orders.** The September 29 Denver YES position newly resolved as a loss of **$15.76**, reducing the previous −$20.02 balance. The decline since the September 25 balance of +$58.54 is **$94.32**. The rolling weekly result is now **−$48.00 under rounded accounting**, close to the existing −$50 breaker. Alpha remains unproven and admission remains **NO-GO**.

This audit uses canonical commit `fccf3165e375ceb5917f4a98a939b5594e17341f`, compared with `53c9f30eaaed2593240c58594a1f1e3d6874ecfe`. The selection rule, fee model, risk limits and frozen scorer are unchanged. These figures describe paper intentions joined to captured outcomes, not verified live execution. Capture dates describe when the repository reflected outcomes; target dates identify the weather events.

## Settlement bridge and uncertainty

| September 29 target | Intended position | Payout | Modeled fee | Net |
|---|---|---:|---:|---:|
| Denver `KXHIGHDEN-26SEP29-B65.5` | 93 YES × $0.16; $14.88 principal | $0.00 | $0.88 | **−$15.76** |

All previously settled orders retain their settlement economics. The full sample has **41 wins in 78 settlements across 20 target days**, $1,151.44 principal and $42.34 modeled fees. Fee-inclusive ROI is **−3.11%**, with a target-day bootstrap 95% interval of **−28.76% to +25.47%**. Independent Decimal accounting reproduces the totals to the recorded floating-point precision.

Holding selected contracts fixed and worsening every entry by one cent produces **−$66.54** under the existing sensitivity calculation. That calculation does not reprice fees or establish that the intentions could have filled.

| Capture date | Cumulative settled paper P&L |
|---|---:|
| September 25 | +$58.54 |
| September 28 | +$14.42 |
| September 29 | −$20.02 |
| September 30 | **−$35.78** |

The [September 29 audit](2026-09-29-forward-update.md) documents the preceding −$34.44 settlement bridge. These capture-to-capture changes are not complete calendar-day equity statements.

## Frozen NO shadow

The September 21 rule keeps NO orders from the existing pilot's selected top five, without replacing excluded YES orders. There are now **four prospective selections: two settled wins totaling +$2.98 and two open selections**. The new October 1 positions are Philadelphia NO and Austin NO, totaling **$29.85 principal**. They supply no settled return yet.

The two settled observations remain consecutive Seattle target dates with identical economics. Their reported bootstrap interval is consequently degenerate at **+10.3472% ROI**. This does not establish negligible uncertainty or validated alpha. Historical side attribution is **NO +$66.10 on 23 settlements** versus **YES −$101.88 on 55**; it includes the observations used to select the shadow and remains exploratory. No side filter is promoted from these totals.

## Open paper exposure

Six paper intentions remain unresolved, with **$89.25 principal**:

| Target | City / side | Contracts × price | Principal |
|---|---|---:|---:|
| September 30 | Austin YES | 83 × $0.18 | $14.94 |
| September 30 | Chicago YES | 37 × $0.40 | $14.80 |
| September 30 | NYC YES | 53 × $0.28 | $14.84 |
| October 1 | Philadelphia NO | 21 × $0.71 | $14.91 |
| October 1 | Boston YES | 57 × $0.26 | $14.82 |
| October 1 | Austin NO | 18 × $0.83 | $14.94 |

The three new October 1 orders total **$44.67** and were recorded at approximately `2026-09-30T14:59:32.4256Z`. Their tickers are `KXHIGHPHIL-26OCT01-B84.5`, `KXHIGHTBOS-26OCT01-B77.5` and `KXHIGHAUS-26OCT01-B89.5`. Open-intent fees are not included in settled P&L; intentions are not fill evidence.

## Rolling weekly risk and admission

The current target-date window is **September 23–29 inclusive**: 11 settled orders, four wins, $161.98 principal and $7.02 modeled fees, for **−$48.00 net**. The Rust pilot uses an unrounded fee approximation, giving **−$47.922474**, displayed as −$47.92.

Two September 22 winners totaling +$54.97 aged out, so the weekly decline exceeds the new settlement loss:

`+$22.73 previous week − $54.97 September 22 cohort − $15.76 new loss = −$48.00`.

The existing breaker trips when its Rust weekly estimate falls below −$50; this capture remains approximately **$2.08 above that threshold**. The breaker was not overridden. Its weekly window is distinct from the cumulative decline since September 25.

The unchanged enforced admission gate exits 1 with **NO-GO: 78/100 settlements and negative fee-inclusive ROI**. Other existing criteria pass: Austin accounts for 13.47% of realized losses, structural eligibility is reported satisfied, and zero live orders await verification. There are zero live settlements.

The [primary workflow run](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/36732999108) had `PILOT_LIVE` empty, reported −$47.92 over 11 dry settled orders, and recorded three new dry intentions totaling $44.67. Its gate output preceded those intentions, so it showed three open orders; the final ledger has six.

## Input integrity and source research

Captures grew from 10,544 to **10,653**, adding 90 Kalshi rows for October 1 (six per study city) and 19 Polymarket rows. The update resolved 116 previously null outcomes: 90 Kalshi and 26 Polymarket rows, all for September 29. There were no deletions, duplicate identities, known-label revisions, field loss or settlement-metadata revisions. Other historical changes were 127 floating-point serialization differences across 79 rows, at most approximately `3.55e-15`. All **630 raw rule hashes** verify.

The ledger preserves its complete previous byte prefix and adds 177 decisions: three orders, 46 edge-below-cost skips, 41 price-floor skips and 87 lead-zero skips. All **7,985 ledger rows remain marked dry**.

The separate [September 30 hosted source study audit](2026-09-30-hosted-source-study.md) records prospective source evidence. Those observations do not change this paper ledger, validate intended fills or authorize live promotion.

### Preserved local capture

The original checkout retains a separate, uncommitted later capture and dashboard. Its guarded wrapper started at 15:00 UTC; the capture and dashboard were written at approximately 15:01:29 and 15:01:32, after the canonical capture commit at 14:59:34. The local file's first 10,544 rows match the canonical file byte for byte. It adds 111 rows rather than the canonical 109, including two additional Polymarket rows. Tokyo and Seoul crossed local midnight between the captures, changing forecast phases and the associated bias/sigma values. This is a separate observation vintage, not historical field loss.

Both original files remain intact and excluded from the canonical totals above:

- Local `data/captures.jsonl` SHA-256: `aa15bc19bf958013e2b90b212e61cd0b0855b4726166bf49376a92e7fa59d812`
- Local `dashboard.html` SHA-256: `03216b235dcb0cc20d5ff1b0eb7fa759234f95d2285a5aedcb65f4fccdf51dcd`

## Reproduction

The [machine-readable score](2026-09-30-forward-update.json) exactly reproduces with unchanged `scripts/pilot_alpha_audit.py`. The audited inputs match canonical commit `fccf316` byte for byte; uncommitted captures from another checkout are not included:

- Captures SHA-256: `0f0f14fb640c87794e44766d41e96809afc031ef242bf6cdf80e2d94c567ca91`
- Ledger SHA-256: `8945f1bcf8d87c07cf4bd2f3e495c360ecdb441bab106518043751588655b2e8`

From the repository root:

```sh
python3 scripts/pilot_alpha_audit.py
python3 scripts/go_live_gate.py --enforce --json
```

The first command reproduces the score; the second returns expected NO-GO exit code 1. Frozen scorer hashes match both canonical commits and the audited workspace:

- `pilot_alpha_audit.py`: `615df82678438722b1c6ffc8ff08a4fd479485aa0b47d186c6eb7c5c5c5343fc`
- `market_shape_alpha.py`: `2c309132a39db89c55dda8daeab995a5dde12f0cfdde3f6a20056129aecf12e7`
- `go_live_gate.py`: `061692e86b19748c878458e3ed303cc563f854fb77b871ec269958e01d396da4`
