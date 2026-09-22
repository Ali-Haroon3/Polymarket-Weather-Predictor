# Independent archive validation: rules frozen before retrieval

Recorded 2026-09-22 at 06:02 UTC, before downloading the evaluation sample. This is a
historical period outside the existing capture dataset, not a prospective experiment or
evidence of actual fills. The purpose is to test already identified candidates without
choosing dates, cities, parameters, or execution prices from their historical profits.

## Data and timing

- Retrieve target dates **2026-05-01 through 2026-06-28**, inclusive, from Kalshi's public
  historical market and minute-candle endpoints. Keep this separate from canonical captures.
- Use all 15 existing series: KXHIGHNY, KXHIGHCHI, KXHIGHAUS, KXHIGHDEN, KXHIGHLAX,
  KXHIGHMIA, KXHIGHPHIL, KXHIGHTDAL, KXHIGHTSEA, KXHIGHTATL, KXHIGHTBOS,
  KXHIGHTPHX, KXHIGHTLV, KXHIGHTDC, KXHIGHTHOU. Missing markets are missing coverage,
  not permission to substitute another city or date.
- Entry is the **one-minute close ending exactly at 15:00 UTC on the prior calendar day**.
  Do not use intraminute highs/lows, later candles, trade prices, or current/final books.
- Require every event market to have opened by entry, an exact contiguous partition with
  both tails, valid unique identifiers and consistent settlement data. Require every leg's
  exact candle. Preserve legitimate missing quote sides, but reject crossed, malformed,
  nonfinite and out-of-range quotes. A leg with neither usable side makes the ladder unusable.
- **Exclude NYC target June 15 entirely, including training.** Its outcome and one quote
  were seen during the preliminary archive-access probe. Other smoke checks on that event
  are permitted solely to verify extraction.
- May supplies initial history. The fixed evaluation window is entry dates **June 1 through
  June 27** (target June 2 through June 28). Target June 1 may enter later training only.
- For each entry, use only complete earlier-target ladders whose **every market settlement
  timestamp is at or before entry**. Include earlier evaluation outcomes once actually
  settled. Use the existing newest-300-ladder history and minimum of 60 ladders. Never use
  June-through-September canonical outcomes to initialize this earlier test.

## Four fixed policies

1. Joint bias/scale fit, both sides, inherited pilot selection.
2. The same joint fit and parent selection, retaining only its selected NO orders without
   backfilling omitted YES orders.
3. Independently refitted bias-only `(b, 1)`, inherited pilot selection.
4. Independently refitted scale-only `(0, k)`, inherited pilot selection.

Use existing fit grids, completeness probability-sum limits, side precedence and quote
floors from `market_shape_alpha.py`. Use the existing **4% after-fee hurdle**, integer
**$15 stakes**, **top five per day**, **$30 per city/target** cap, **$150 per day** cap and
ticker deduplication from `pilot_alpha_audit.py`. No additional variants, exclusions, threshold
searches or date changes. Commit selection before inspecting each selected outcome. Fees
are the existing rounded per-order taker model; no rebates or assumed maker fills.

Frozen shared-code SHA-256:

| File | SHA-256 |
|---|---|
| scripts/market_shape_alpha.py | `2c309132a39db89c55dda8daeab995a5dde12f0cfdde3f6a20056129aecf12e7` |
| scripts/pilot_alpha_audit.py | `615df82678438722b1c6ffc8ff08a4fd479485aa0b47d186c6eb7c5c5c5343fc` |
| scripts/go_live_gate.py | `061692e86b19748c878458e3ed303cc563f854fb77b871ec269958e01d396da4` |

## Evaluation and decision

Report all four policies, including losses, missing coverage, number of independent days,
fit paths, fees, profit, capital risked, sides and cities. Resample whole calendar target
days, including zero-trade days in the fixed evaluation window. Report nominal 95% intervals
and Bonferroni-adjusted **98.75% individual intervals** for the four-policy family. Also
report fixed seven-day moving-block resampling as a serial-dependence sensitivity. Bootstrap
intervals remain approximate and do not establish a live-performance guarantee.

A credible candidate for further prospective testing must have positive lower bounds under
both simultaneous day and seven-day block intervals, positive profit with every selected
contract priced one cent worse and fees recomputed, and positive profit after removing its
two most profitable cities. The stress keeps contract counts fixed and explicitly reports
extra capital required; it is not a resized alternative strategy. Impossible stressed prices
must be reported, not silently removed. These are research criteria, not a live-admission
override. Report failures directly and do not revise criteria after seeing results.

Archive candles do not provide depth, quote-change timestamps, queue position, or actual
fills. Historical weather regimes and universe availability differ from September. A positive
result would justify a frozen prospective paper test, not automatic live promotion. Raw
responses, retrieval times, normalized data hashes and extraction failures must be retained.

Sources: [historical candles](https://docs.kalshi.com/api-reference/historical/get-historical-market-candlesticks),
[archive routing](https://docs.kalshi.com/getting_started/historical_data).
