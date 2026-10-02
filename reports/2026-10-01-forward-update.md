# October 1 capture: $46.82 further paper losses trip the weekly breaker

**Cumulative settled paper P&L is −$82.60 after modeled fees across 81 orders.** Three September 30 YES positions newly resolved as losses totaling **$46.82**, reducing the previous −$35.78 balance. The decline since the September 25 balance of +$58.54 is **$141.14**. The primary October 1 workflow stood down at the existing weekly loss breaker and placed no new paper orders. Alpha remains unproven and live admission remains **NO-GO**.

This audit uses canonical commit `096fa88584a5e79509b05aa8bcee6f9d89d237d3`, compared with `fccf3165e375ceb5917f4a98a939b5594e17341f`. The selection rule, fee model, risk limits and frozen scorer are unchanged. These figures describe recorded paper intentions joined to captured outcomes, not verified live execution. Capture dates describe when the repository reflected outcomes; target dates identify the weather events. Later observations outside this canonical capture are excluded.

## Settlement bridge and uncertainty

| September 30 target | Intended position | Payout | Modeled fee | Net |
|---|---|---:|---:|---:|
| Austin `KXHIGHAUS-26SEP30-B93.5` | 83 YES × $0.18; $14.94 principal | $0.00 | $0.86 | −$15.80 |
| Chicago `KXHIGHCHI-26SEP30-T69` | 37 YES × $0.40; $14.80 principal | $0.00 | $0.63 | −$15.43 |
| NYC `KXHIGHNY-26SEP30-B72.5` | 53 YES × $0.28; $14.84 principal | $0.00 | $0.75 | −$15.59 |
| Total | $44.58 principal | $0.00 | $2.24 | **−$46.82** |

All previously settled orders retain their settlement economics. The full sample has **41 wins in 81 settlements across 21 target days**, $1,196.02 principal and $44.58 modeled fees. Fee-inclusive ROI is **−6.91%**, with a target-day bootstrap 95% interval of **−32.18% to +21.16%**. Independent Decimal accounting reproduces the totals to the recorded floating-point precision.

Holding selected contracts fixed and worsening each entry by one cent produces **−$115.09** under the existing sensitivity calculation. That calculation does not reprice fees or establish that the intentions could have filled.

| Capture date | Cumulative settled paper P&L |
|---|---:|
| September 25 | +$58.54 |
| September 28 | +$14.42 |
| September 29 | −$20.02 |
| September 30 | −$35.78 |
| October 1 | **−$82.60** |

The [September 30 audit](2026-09-30-forward-update.md) documents the preceding Denver loss. These capture-to-capture changes are not complete calendar-day equity statements.

## Weekly breaker and admission

The October 1 run's target-date window is **September 24–30 inclusive**: 13 settled orders, three wins, $191.68 principal and $8.46 modeled fees, for **−$141.14 net**. The Rust pilot uses an unrounded fee approximation, giving **−$141.057950**, displayed as −$141.06.

A September 23 winner worth +$46.32 aged out of the window, so the weekly deterioration is larger than the new losses:

`−$48.00 previous week − $46.32 September 23 winner − $46.82 new losses = −$141.14`.

The [primary workflow run](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/36880033609) logged **STAND DOWN** at `2026-10-01T14:56:07.5438073Z` because its −$141.06 weekly estimate breached the existing −$50 limit. `PILOT_LIVE` was empty, the run used dry ledger-only mode without trading credentials, and it created no orders. The ledger remains byte-for-byte unchanged from the September 30 canonical capture.

The canonical capture commit was recorded at `2026-10-01T14:56:09Z`. At `2026-10-01T15:34:36.384592Z`, the [backup workflow](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/36885311301) explicitly skipped its remaining steps because that day's capture was already committed. It did not rerun the pilot.

This is the observed run's automatic stand-down, not a permanent disable or a latched human pause. The unchanged rolling calculation can clear as losses age out; no threshold override or strategy change was applied. The weekly figure happens to equal the cumulative decline since September 25 under rounded accounting, but the two measures use different definitions.

The separately enforced admission gate exits 1 with **NO-GO: 81/100 settlements and negative fee-inclusive ROI**. Other existing criteria pass: Austin accounts for 15.00% of realized losses, structural eligibility is reported satisfied, and zero live orders await verification. There are zero live settlements. The positive trailing replay reported by the pilot does not override these failed prospective-return and risk checks or establish alpha.

## Frozen NO shadow and remaining exposure

The September 21 shadow keeps NO orders from the existing pilot's selected top five, without replacing excluded YES orders. It remains at **four prospective selections: two settled Seattle wins totaling +$2.98 and two open October 1 NO positions**. No new shadow selection or settlement arrived.

Both settled observations have identical economics on consecutive Seattle target dates. Their reported bootstrap interval is therefore degenerate at **+10.3472% ROI**; that is not evidence of negligible uncertainty or validated alpha. Historical side attribution is **NO +$66.10 on 23 settlements** versus **YES −$148.70 on 58**. It includes the observations used to select the shadow and remains exploratory; no side filter is promoted.

Three paper intentions remain unresolved in this canonical capture, totaling **$44.67 principal**:

| October 1 target | Side | Contracts × price | Principal |
|---|---|---:|---:|
| Philadelphia `KXHIGHPHIL-26OCT01-B84.5` | NO | 21 × $0.71 | $14.91 |
| Boston `KXHIGHTBOS-26OCT01-B77.5` | YES | 57 × $0.26 | $14.82 |
| Austin `KXHIGHAUS-26OCT01-B89.5` | NO | 18 × $0.83 | $14.94 |

All were recorded on September 30. The two NO intentions account for **$29.85** of open principal. Open-intent fees are not included in settled P&L; intentions do not establish executed positions or future returns.

The separate [October 1 hosted source study audit](2026-10-01-hosted-source-study.md) records prospective source evidence. Those observations do not change this ledger, validate intended fills or authorize live promotion.

## Input integrity and local preservation

Canonical captures grew from 10,653 to **10,767**, adding 90 Kalshi rows for October 2 and 24 Polymarket rows. The update resolved 106 previously null outcomes, all for September 30: 90 Kalshi and 16 Polymarket. Existing capture identities remain in their original order. There were no deletions, duplicate identities, known-label revisions, field loss or settlement-metadata revisions. Other historical changes were 142 floating-point serialization differences across 84 rows, at most approximately `3.55e-15`. All **720 raw rule hashes** verify. The ledger remains at **7,985 dry rows**, with unchanged bytes and no appended decisions.

The original checkout's separate, uncommitted September 30 capture and dashboard remain intact at the hashes recorded in the [September 30 preservation audit](2026-09-30-forward-update.md#preserved-local-capture): captures `aa15bc19bf958013e2b90b212e61cd0b0855b4726166bf49376a92e7fa59d812` and dashboard `03216b235dcb0cc20d5ff1b0eb7fa759234f95d2285a5aedcb65f4fccdf51dcd`. They are excluded from the canonical totals. Preserving those files does not establish that a local October 1 capture occurred.

## Reproduction

The [machine-readable score](2026-10-01-forward-update.json) exactly reproduces with unchanged `scripts/pilot_alpha_audit.py`. The audited inputs match canonical commit `096fa885` byte for byte:

- Captures SHA-256: `5abe243d99b09d1da15a31fdb1da2689e12b13e9b18a17e63fb973a12a39533a`
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
