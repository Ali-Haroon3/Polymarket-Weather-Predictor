# March–April training extension: June policy validation

**All four policies lost money after modeled fees; none passed the preregistered criteria.** Earlier history supplies the required training on all 27 June entry days, but does not produce a qualifying candidate. These are historical quote replays, not evidence of actual fills.

This follows the [original inconclusive test](2026-09-22-archive-validation.md). The [extension preregistration](2026-09-22-archive-training-extension-preregistration.md) was committed and pushed as `f534598c21615f5cb97b61933f8927844a9217a0` before retrieving March–April data. The optional evaluator was committed as `366a825640d0975e25022f1f037f4fd4447a2b9c` before evaluation. June had already been processed for coverage, so this is a bounded reanalysis, not a new untouched holdout.

## Coverage and causal training

| Measure | Result |
| --- | --- |
| March 1–April 30 requested events, all 15 cities | 915 |
| Additional accepted events / normalized rows | 154 / 924 |
| Additional rejected events | 761 |
| Combined March–June normalized events | 215 |
| Combined events failing probability-sum completeness | 2 |
| Earlier settled training ladders per June entry | 182–208 |
| Entry days with at least 60 earlier settled ladders | 27 / 27 |
| Usable June holdout events | 30 / 404 |
| June entry days with at least one usable event | 16 / 27 |

Rejection reasons: 760 exact 15:00 UTC candle missing; 1 market has no final binary settlement. Missing exact candles remain missing; no quote is carried forward. Before the March daylight-saving change, the fixed clock also coincides with some event openings. The entry clock was not moved to increase coverage.

Only earlier targets with every contract settled by entry enter training. The newest 300 qualifying ladders are retained. March–April supplies no evaluation candidates or return comparisons. The original June quotes and outcomes, 4% edge hurdle, integer $15 stakes, top-five selection, exposure caps, four policies and NYC probe exclusion are unchanged.

## All four frozen policies

| Policy | Settled / selected | Stake | Fees | Net | ROI |
| --- | --- | --- | --- | --- | --- |
| Joint bias/scale | 14 / 14 | $207.79 | $9.57 | −$29.36 | −14.13% |
| Parent-selected NO, no replacements | 4 / 4 | $58.63 | $1.07 | −$0.70 | −1.19% |
| Independently refitted bias only | 12 / 12 | $177.67 | $7.78 | −$17.45 | −9.82% |
| Independently refitted scale only | 14 / 14 | $208.24 | $10.28 | −$16.52 | −7.93% |

Profit includes modeled rounded order fees. ROI divides that net profit by contract purchase cost, excluding fees from the denominator; an all-loss resample can therefore fall below −100%. The NO policy removes YES orders from the joint parent selection without backfilling freed slots. The other two families are independently refitted, not produced by modifying joint-fit coefficients.

## Uncertainty and preregistered checks

Intervals resample all 27 calendar days, including days without trades, with 10,000 draws and seed 42. The moving-block variant uses seven-day blocks. The 98.75% intervals implement the preregistered four-policy Bonferroni adjustment; they remain approximate given the short, selectively observed sample.

| Policy | Day 95% ROI | Day 98.75% ROI | Block 95% ROI | Block 98.75% ROI |
| --- | --- | --- | --- | --- |
| Joint bias/scale | -105.08% to +76.79% | -105.83% to +100.11% | -105.42% to +6.25% | -105.80% to +36.72% |
| Parent-selected NO, no replacements | -102.08% to +58.60% | -102.08% to +58.60% | -43.99% to +58.60% | -43.99% to +58.60% |
| Independently refitted bias only | -104.34% to +70.96% | -105.38% to +101.75% | -104.51% to +17.47% | -104.73% to +35.19% |
| Independently refitted scale only | -105.86% to +87.28% | -106.02% to +110.43% | -105.42% to +14.40% | -105.80% to +42.54% |

| Policy | Net with 1¢ adverse fill | Removed best two cities | Net after removal | Passes all criteria |
| --- | --- | --- | --- | --- |
| Joint bias/scale | −$38.73 | Atlanta, Austin | −$83.39 | No |
| Parent-selected NO, no replacements | −$1.48 | Washington, Austin | −$12.99 | No |
| Independently refitted bias only | −$24.89 | Atlanta, Austin | −$61.59 | No |
| Independently refitted scale only | −$26.14 | Austin, Atlanta | −$61.68 | No |

The adverse-price stress keeps contract counts fixed, pays one cent more per contract, and recomputes rounded fees. It can require more than the original stake or city cap; it is a sensitivity check, not an executable resized portfolio. Every policy must have known selected outcomes, positive adjusted lower bounds under both bootstraps, positive stressed net profit and positive profit after removing its two best cities.

Failed criteria by policy:

- **Joint bias/scale:** positive simultaneous day lower, positive simultaneous block lower, positive repriced stress net, positive after removing best two cities.
- **Parent-selected NO, no replacements:** positive simultaneous day lower, positive simultaneous block lower, positive repriced stress net, positive after removing best two cities.
- **Independently refitted bias only:** positive simultaneous day lower, positive simultaneous block lower, positive repriced stress net, positive after removing best two cities.
- **Independently refitted scale only:** positive simultaneous day lower, positive simultaneous block lower, positive repriced stress net, positive after removing best two cities.

## Interpretation

Each policy trades on only four to seven target days. Every confidence interval spans both loss and gain, so these estimates do not establish negative expected returns. They provide no basis to promote any policy. Positive side or city slices discovered within a failed policy are exploratory and cannot become additional candidates after inspecting this result.

The extension stops at April 30 as registered. No further months, lower training minimum, altered quote time or new policy variant are used to make this experiment pass. Sparse exact-minute coverage can select a different population from the full market; March–April training and June testing also precede the September regime. Historical candles provide neither depth nor queue position nor proof of executable fills. No trading rule or live setting is promoted.

The [current-period audit](2026-09-21-alpha-validation.md) remains necessary context: the old warm correction weakened after September 7, and recent replacement profits were not robust. A historical criterion pass would justify additional prospective paper validation only; a failure is recorded without tuning this test.

## Reproduction and provenance

- [Machine-readable policies and fit paths](2026-09-22-archive-training-extension.json).
- [March–April normalized captures](2026-09-22-archive-training-captures.jsonl.gz).
- [All 915 event coverage records and provenance](2026-09-22-archive-training-coverage.json.gz).
- [Unchanged original May–June captures](2026-09-22-archive-captures.jsonl.gz).

March–April capture SHA-256: `bff771f656c9d1855bf16cc01cceb4373cdfb7e40f8ff878e5a2f7949b5d0e6f`.

Original May–June capture SHA-256: `094dafd21aee0c78f6e0912a6caca5dcc8f5b86020855a1ac56d35b23224fe26`.

Evaluator SHA-256: `8c0b58914a240c2e32147a9006dd1aa05f395bf1854c39824ba5c01e6924823c`.

Training wrapper SHA-256: `8c4d3c8a1f9c285e2132c84a42db95bc99e679b366acf73defb564122cfd2f73`.

Base importer SHA-256: `451337fa0fc3760ee71e50d10666eb7ed1f06f24b5c46fe6163603e2640f4363`.

Reproduce offline with the committed compressed inputs:

```bash
mkdir -p data/raw/kalshi_archive_reproduction
gzip -dc reports/2026-09-22-archive-captures.jsonl.gz \
  > data/raw/kalshi_archive_reproduction/captures.jsonl
gzip -dc reports/2026-09-22-archive-training-captures.jsonl.gz \
  > data/raw/kalshi_archive_reproduction/warmup.jsonl
python3 scripts/archive_alpha_validation.py \
  --captures data/raw/kalshi_archive_reproduction/captures.jsonl \
  --warmup data/raw/kalshi_archive_reproduction/warmup.jsonl
```
