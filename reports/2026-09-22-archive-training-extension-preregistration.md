# Training-history extension registered before retrieval

Recorded 2026-09-22 at 06:40 UTC. This is a bounded follow-up to the
[original archive experiment](2026-09-22-archive-validation.md), which selected no orders
because every entry lacked the required 60 earlier settled ladders. Its original result
remains unchanged. The June archive has already been processed for coverage, so this is a
registered extension and reanalysis, not a newly collected pristine holdout.

## The single change

Retrieve **target dates March 1 through April 30, 2026, inclusive**, for the same 15 city
series, and append accepted ladders to training history only. These 915 requested events
use the existing exact one-minute candle ending at **15:00 UTC on the prior calendar day**,
full-partition, identity, quote and lifecycle checks. Missing exact candles remain missing;
no previous/later candle, current book, last trade or alternative clock time may replace them.

The extension ends at April 30 regardless of coverage or results. Do not add further months
or lower the history minimum to make this experiment pass. Transport may use the existing
public cache, up to 12 workers, a shared maximum of five GETs per second, and bounded retries.
Transport failures are recoverable from cache; they are not permission to drop events.

## What remains fixed

- The original May–June normalized input bytes and SHA-256 must remain
  `094dafd21aee0c78f6e0912a6caca5dcc8f5b86020855a1ac56d35b23224fe26`.
- June entry dates remain **June 1–27**, with targets June 2–28. March and April may never
  supply evaluation candidates, trades, or additional return comparisons.
- Training requires an earlier target and every leg's settlement timestamp at or before
  entry; use the newest **300** qualifying ladders with a minimum of **60**. Earlier June
  settlements may join only when actually available, as in the original experiment.
- Preserve the NYC June 15 probe exclusion from all training and evaluation.
- Preserve the same four policies, fit grids, probability-sum completeness limits, side
  precedence, quote floors, **4% net-edge hurdle**, integer **$15 stakes**, **five selections
  per entry**, **$30 city/target** and **$150 daily** caps, and ticker deduplication.
- The NO policy retains only the joint parent's selected NO orders, without replacements.
- Preserve rounded modeled fees, outcome-blind selection, fixed-calendar day and seven-day
  block bootstraps, 10,000 draws with seed 42, four-policy Bonferroni intervals, recomputed
  one-cent adverse-price stress, and removal of the two best cities. No additional variants.
- Preserve the original credibility criteria: positive simultaneous lower bounds for both
  day and block resampling, positive stressed total profit, positive profit after removing
  the two best cities, and known outcomes for all selected orders. No live promotion.

The existing shared research-code hashes in the original preregistration remain fixed.
Base importer SHA-256:
`451337fa0fc3760ee71e50d10666eb7ed1f06f24b5c46fe6163603e2640f4363`.
The optional warmup evaluator must reproduce the original no-warmup JSON byte for byte.
Commit its extension implementation before evaluating the combined sample; record its hash,
both input hashes, coverage, causal fit paths, all four results and all failed criteria.

## Interpretation and stopping rule

Adding earlier outcomes may legitimately change fitted parameters and June decisions. It
must not change June quote/outcome eligibility or be selected using June profitability.
Report the result even if it is unprofitable or still has too little training data.

Sparse quote coverage is a selective subset of the universe; older weather regimes may
differ. Exact candles still lack depth and actual execution evidence. A passing result
would identify a candidate for further prospective paper validation, not establish current
tradable alpha or authorize live trading. If no candidate passes, record that outcome and
do not tune or further extend this experiment after inspecting its returns.
