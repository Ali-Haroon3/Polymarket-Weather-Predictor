# September 24 capture: Boston win lifts paper P&L; alpha remains unproven

**The paper pilot is now +$58.54 after modeled fees across 68 settled orders.** Boston's September 23 YES result added **+$46.32**. The rolling weekly result improved enough for the existing paper pilot to resume selection, producing two new YES intents. Admission remains **NO-GO**, and no live orders are recorded.

This update scores capture commit `229e30ad82572a82193051a93d4105abcf3bb0b8` with unchanged research rules. The [September 24 scheduled run](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/36016199642) used merged implementation commit `450060736ca7f8b3d6b75f63c0979629e052c70e`. No strategy, risk limit, trading setting, or canonical historical observation was changed by this audit.

## Fixed-rule score

| Metric | September 23 capture | September 24 capture |
| --- | ---: | ---: |
| Settled paper orders | 67 | 68 |
| Wins | 37 | 38 |
| Purchase cost, excluding fees | $989.46 | $1,004.34 |
| Modeled rounded fees | $35.32 | $36.12 |
| Net profit | +$12.22 | **+$58.54** |
| ROI after fees | +1.24% | **+5.83%** |
| Unresolved orders after the pilot | 1 | 2 |

The target-day bootstrap 95% ROI interval is **−20.77% to +39.12%** across 15 target days. The existing same-contract one-cent adverse-price sensitivity is **+$33.40**. This remains the original simplified attribution sensitivity, not the archive evaluator's separately recomputed-fee stress. Paper intents and modeled fees are not proof of executable fills or a reliable edge.

The roughly −$40 reading was **−$42.75 at the September 22 capture**. Two September 22-target wins raised that to +$12.22 in the September 23 capture; the Boston settlement below explains today's further increase. These are cumulative settled-paper balances, not individual daily losses or gains.

## New settlement and open paper intents

The one new settlement is `KXHIGHTBOS-26SEP23-B64.5`, a YES intent entered September 22 for September 23: 62 contracts at $0.24, $14.88 principal and $0.80 modeled rounded fee. The winning payout produces **62 − 14.88 − 0.80 = +$46.32**. The previous 67 settlements are unchanged.

The [source reconciliation](2026-09-24-weather-source-reconciliation.md) independently finds a later official Boston daily maximum of 64°F and venue settlement YES at 64.00. It also explains why the earlier incomplete hourly observations could not be substituted for the daily settlement value.

Two new paper intents entered September 24 for September 25 remain open:

| Contract | Side | Contracts | Price | Principal |
| --- | --- | ---: | ---: | ---: |
| KXHIGHTATL-26SEP25-B80.5 | YES | 125 | $0.12 | $15.00 |
| KXHIGHAUS-26SEP25-B99.5 | YES | 99 | $0.15 | $14.85 |

Historical selected NO remains +$63.12 on 21 settled orders; YES is now −$4.58 on 47. These are exploratory attribution slices. The frozen rank-first NO shadow retains only actual parent-selected NO orders entered strictly after September 21, without replacing YES selections. It still has **zero prospective selections and zero settlements**. Skip rows whose side defaults to NO do not constitute selections.

## Why the weekly breaker cleared

The September 24 rolling window contains 28 settled paper orders with target dates **September 17–23**, inclusive. Their $413.81 principal produces $14.19 gross profit. The Rust pilot's existing unrounded fee approximation totals $15.061641, giving **−$0.871641**, matching its logged **−$0.87**. Rounded per-order modeled fees total $15.21, giving **−$1.02**.

The previous window was −$66.12 under rounded fees. Five September 16-target orders with net P&L of −$18.78 aged out, while Boston added +$46.32: **−66.12 + 18.78 + 46.32 = −1.02**. Both current fee conventions are above the existing −$50 stop threshold. The rolling breaker cleared automatically; no override or risk-limit change occurred. It is not a latched human pause.

The unchanged admission gate still returns **NO-GO**, exit 1: 68 settled orders do not meet the 100-order requirement. Positive cumulative ROI satisfies only the profitability portion of that combined criterion. The city-loss and structural checks pass; there are no recorded live orders or unresolved live-fill evidence. These facts do not establish that a funded strategy would be profitable.

The inherited daily fit updated to Normal μ +0.1°C, σ ×1.00 using its existing newest-300-ladder procedure. The scheduled run's trailing 30-day replay was +4.8% on 650 trades. That is an ordinary parameter update under the existing algorithm, not a new rule or independent alpha validation. The failed frozen archive experiment and unselected NO shadow remain unchanged.

## Capture integrity and deployment evidence

Canonical captures grew from 9,790 to 9,934: 144 new rows, comprising 90 Kalshi rows forming 15 complete September 25 ladders and 54 Polymarket rows. No rows were removed. There are 146 null-to-binary outcome updates for September 23 targets: 90 Kalshi and 56 Polymarket. No previously known outcomes were revised. Other historical changes are numeric reserialization in 158 fields across 94 rows, with maximum absolute difference approximately `3.55e-15`; historical fields were preserved.

Exactly the 90 new Kalshi rows carry settlement metadata. All explicitly identify The Weather Company in the primary rule, and all hashes recompute from the exact raw rule pair. There is no historical provenance backfill. The new unpriced Polymarket wind market remains outside Kalshi temperature scoring.

The ledger preserves its previous 6,738 rows exactly and appends 180 dry-run decisions: two orders, 39 price-floor skips, 49 edge skips and 90 lead-zero skips. It now has 6,918 rows. No duplicate identities, invalid quotes, chronology reversals, or live order IDs were found.

[PR #48](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/pull/48) was merged before this scheduled run. The run executed the new code and preserved source metadata, but its live setting and trade credentials were empty; its pilot was ledger-only dry run. Live-entry enforcement was not exercised by this run. The workflow scores the gate and attribution before running the pilot, so its audit log's zero open orders precedes the two new intents; the report here scores the final committed ledger and correctly shows two open orders.

The local capture-only cron remains routed through the guarded build wrapper described in the [September 23 repair](2026-09-23-forward-update.md). At this audit, the local capture log still ends September 23. The cloud capture proves deployed metadata preservation; it does not prove that the repaired local job has subsequently executed. Preserved local stashes and raw observations remain separate from canonical scoring.

## Reproduction

The [machine-readable output](2026-09-24-forward-update.json) reproduces byte for byte from the capture and ledger in commit `229e30a`, using unchanged `scripts/pilot_alpha_audit.py`.

- Capture SHA-256: `fd3c379a68491dd59317c8bf7d1608d7bd750a643b80b8bc529be588a44d9a9c`
- Ledger SHA-256: `b775e763168da04c7723742fa06d54412c795fb5e5891babacb0692cb1d26088`

Run `python3 scripts/pilot_alpha_audit.py` and `python3 scripts/go_live_gate.py --enforce --json` against these inputs. Verification covered exact score reproduction, input integrity, scheduled-run logs, and source-response hashes. This update changes reports and documentation only; it does not rerun or relabel earlier code tests as new validation.
