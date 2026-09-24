# September 23 ladders: no fee-positive basket in the forward quote screen

**No positive full-ladder taker basket appeared in the September 23 contracts at the observed prices.** This extends the current-price coverage to the pilot's next-day horizon; it does not extend or retune the completed historical experiment. The paper pilot remains at −$42.75 on 65 settled orders, and no strategy is promoted.

## Fixed screen and coverage

Before retrieval, the screen fixed September 23 as the target, all 15 existing city series, both full YES and full NO baskets, and $15/$150 budgets including fees. It uses the unchanged `ladder_arbitrage_audit.py` functions: complete contiguous partitions, opposite-bid complements for asks, equal integer contract quantities, exhaustive affordable sizing, per-leg rounded taker fees, and the worst single-leg one-cent adverse-price sensitivity. No outcome, forecast, fitted correction, or historical profit enters selection.

The sequential public snapshot ran on **September 22, 16:12:38–16:13:10 UTC**. Its 120 requests comprise 15 event listings, 15 series fee schedules, and 90 orderbooks. Every series reports `fee_type=quadratic` and `fee_multiplier=1`. All 90 markets are active, belong to their requested event, and form 15 complete six-leg temperature partitions.

| Measure | Result |
| --- | ---: |
| Complete event partitions | 15 |
| Basket sides with every required quote | 23 of 30 |
| Budget/basket comparisons | 46 |
| Gross-positive basket sides before fees | 1 |
| Positive baskets after modeled fees | **0** |

All 15 YES baskets and eight NO baskets have every required quote. Seven NO basket sides lack at least one required quote and remain unevaluated. Missing quotes are never replaced by last prices.

## The apparent spread disappears after fees

NYC NO is the only gross-positive basket: its six quoted asks total $4.99 against the ordinary-settlement payout of $5.00, a one-cent gross spread. At one contract per leg, rounded modeled fees total eight cents, leaving **−$0.07**. Exhaustive sizing produces no positive quantity at either budget. Its minimum displayed top-level depth is only **0.1 contract**, so even that one-contract basket does not fit the observed best-level depth.

The screen optimistically allows every budget-affordable integer quantity at the best prices without limiting it by depth. Since the unconstrained maximum is nonpositive for every quoted basket, restricting size to displayed depth cannot create a positive result. Deeper prices can only worsen this bound. This reasoning does not establish that the negative baskets could actually be filled.

## Settlement and execution limits

Within each event, the contract rules refer to the same city's maximum temperature in Fahrenheit on September 23, reported by The Weather Company, with consistent secondary rules. The usual basket identity—one winning YES leg or five winning NO legs—assumes ordinary binary settlement of that shared temperature partition.

The secondary rules also permit Kalshi to assign last fair prices if settlement data never becomes available. Therefore the ordinary basket payout is **not an unconditional contractual guarantee**. The report's arithmetic uses ordinary settlement, not a modeled recovery value for that exception.

Books were requested sequentially over approximately 32 seconds, not atomically. Prices can change, partial fills leave exposure, and fees assume one modeled fill per leg. This snapshot rejects a candidate at its observed prices; it does not prove that profitable quotes never appear. No orders were placed, and the pilot's selection rules and canonical data were not changed.

## Evidence

The [compressed evidence bundle](2026-09-22-next-day-basket-evidence.json.gz) contains the full report, all 120 raw response bodies, their receipt times and SHA-256 hashes, and the exact snapshot script. Uncompressed bundle SHA-256:

`f393959e95f1cca7b54af9c830f19b5abacbc7737dcfa34ade82035f3417cfe4`

The report records hashes of the unchanged basket, fee, and partition code. Independent Decimal enumeration verified all 46 budget results and adverse-price stresses, including all integer quantities that fit displayed depth. All 120 response hashes, 90 contract identities and rule sets, and 15 fee schedules were checked. The best depth-feasible result is Chicago YES at one contract per leg, also negative at −$0.08.

Raw bodies permit reproduction without another network request. The earlier [same-day source and basket audit](2026-09-22-current-weather-audit.md) and [latest paper accounting](2026-09-22-forward-update.md) remain separate dated observations.

Primary API references: [NYC September 23 market metadata](https://external-api.kalshi.com/trade-api/v2/markets?event_ticker=KXHIGHNY-26SEP23&limit=100), [NYC series fee metadata](https://external-api.kalshi.com/trade-api/v2/series/KXHIGHNY), and [official orderbook semantics](https://docs.kalshi.com/api-reference/market/get-market-orderbook). The bundled raw responses preserve what was observed at the stated time; these endpoints are live.
