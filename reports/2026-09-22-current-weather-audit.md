# Current contract source and full-ladder price audit

**No profitable full-ladder basket appeared in this snapshot. A separate data defect was confirmed: the repository did not retain the settlement source named in each contract's rules.** This audit does not establish a new alpha signal or explain the pilot's losses by itself.

## Contract rules differ from the repository's source assumption

The public snapshot collected 2026-09-22 from 07:18:53 to 07:19:23 UTC includes all six daily-high contracts for each of the existing 15 city series, targeting September 22. All **90 contracts** explicitly name **The Weather Company** in `rules_primary`; the secondary rules direct readers to `weather.com/kalshi` and describe preliminary-data and correction risks.

An outcome-independent check of three adjacent NYC events found:

| Event | Contracts | Source named in primary rules |
| --- | ---: | --- |
| KXHIGHNY-26AUG13 | 6 | National Weather Service |
| KXHIGHNY-26AUG14 | 6 | The Weather Company |
| KXHIGHNY-26AUG15 | 6 | The Weather Company |

This establishes an observed source-label boundary for these NYC contracts. It does not establish the exact cutover for every city or prove that the underlying station measurements, rounding or daily window changed. The separate March–June archive contains 1,290 rows whose primary rules name NWS; those older observations should not be assumed to share every contractual detail with current markets.

Primary references are the [current NYC market metadata](https://external-api.kalshi.com/trade-api/v2/markets?event_ticker=KXHIGHNY-26SEP22&limit=100), [August 13 metadata](https://external-api.kalshi.com/trade-api/v2/markets?event_ticker=KXHIGHNY-26AUG13&limit=100), and [August 14 metadata](https://external-api.kalshi.com/trade-api/v2/markets?event_ticker=KXHIGHNY-26AUG14&limit=100). Kalshi's [August 27 partnership announcement](https://news.kalshi.com/p/kalshi-weather-company-partnership) confirms The Weather Company's role in outcome verification, but the announcement date alone is not a contract cutover date. The older [weather help page](https://help.kalshi.com/en/articles/13823837-weather-markets) still describes daily contracts as NWS-based; the specific contract rules are the relevant evidence here.

## Impact on the current analysis

Canonical Kalshi outcomes and paper P&L use the venue's reported YES/NO result, not an NWS-imputed outcome. There is no demonstrated error in those labels or the measured −$1.31 paper result. The default market-shape strategy learns from market ladders and venue outcomes, without calling the weather forecaster. Its historical calibration currently has no recorded contract-source boundary, so source-sensitive transfer cannot be checked from canonical capture fields alone.

The September 21 canonical fit uses 300 eligible ladders with targets September 1–20. The observed NYC August boundary is therefore insufficient to claim that the current fit mixes old and new source contracts. Its loss mechanism remains an empirical question; source wording alone does not answer it.

The retired `model-shrunk` path does call the station pricer. Its Kalshi station constants were fitted against CLI data; current contract wording alone does not verify that those constants remain calibrated to The Weather Company's contractual truth. Treat the older NWS mechanism explanation as a hypothesis, not proof that today's warm bias is structural.

The forward data correction preserves raw primary/secondary rules, a conservative observed-source tag, and a deterministic rules hash in new captures. Legacy missing metadata stays missing. The hash fingerprints exact rule text, including date/strike wording; it is not a stable cross-contract regime identifier. It changes provenance only: no historical outcome, model probability, fitted coefficient, selection rule or trading setting is rewritten. Source metadata is not a new admission criterion or evidence that the current source has been calibrated.

## Current full-ladder basket check

All 15 current events formed complete, mutually exclusive temperature partitions. Ninety public top-of-book requests supplied current bid levels and displayed quantities. The existing basket functions checked equal integer contract quantities, total budgets of **$15 and $150 including fees**, rounded general taker fees per leg, and the original one-leg adverse-cent sensitivity. This is the same outcome-independent basket identity as the historical audit, applied directly to today's events; no outcome, forecast or fitted correction enters the decision.

| Measure | Result |
| --- | ---: |
| Complete event partitions | 15 |
| Basket sides with every required quote present | 19 of 30 |
| Budget/basket comparisons | 38 |
| Gross-positive basket sides before fees | 2 |
| Positive baskets after modeled fees | 0 |

Only Chicago NO and Seattle NO had a positive gross basket spread: **2¢ and 1¢ per complete basket**, respectively. Fees erase both even before rounding; after rounding the best permitted net is **−6¢ and −7¢**, respectively, at either tested budget. The [Chicago series metadata](https://external-api.kalshi.com/trade-api/v2/series/KXHIGHCHI) and [Seattle series metadata](https://external-api.kalshi.com/trade-api/v2/series/KXHIGHTSEA), retrieved separately after the quote snapshot with the final receipt at **07:20:54 UTC**, both report `fee_type=quadratic` and `fee_multiplier=1`. All other quoted basket sides fail even before a nonnegative fee. Eleven basket sides lacked at least one required quote and remain unevaluated.

The screen initially allows every budget-affordable quantity at the displayed best prices, without restricting quantity by depth. That is optimistic: deeper fills can only cost more. Since no candidate is positive even under those prices, applying depth limits cannot create a positive candidate. The saved output records minimum top-level depth and whether each computed best quantity fits it.

These were sequential requests, not simultaneous executable baskets. Quotes can change and multi-leg partial fills leave directional exposure. This snapshot rejects an immediate candidate at observed prices; it does not establish that no opportunity can ever appear. No orders were placed.

## Evidence and limits

The [compressed evidence bundle](2026-09-22-current-weather-evidence.json.gz) contains the snapshot report, all **110** raw public responses, per-response receipt times and SHA-256 hashes, and the exact one-off snapshot script. It includes 18 event-list requests, 90 orderbooks, and two series fee checks. All response hashes were checked before packaging. Uncompressed bundle SHA-256:

`a254017289889e95642f767720dd7693d7f5d4667b99ffef94d99f934a20787e`

For orderbook semantics, Kalshi's [official documentation](https://docs.kalshi.com/api-reference/market/get-market-orderbook) specifies YES and NO bids; the opposite ask is obtained by complementing the bid. Canonical captures and the previously frozen archive experiments remain separate and unchanged.

No fully specified, distinct strategy with a demonstrably unused validation sample was found in the remaining repository inventory. New forward evidence is still required for the alpha goal. This source audit improves what future evidence can establish; it does not substitute for a profitable, validated strategy.
