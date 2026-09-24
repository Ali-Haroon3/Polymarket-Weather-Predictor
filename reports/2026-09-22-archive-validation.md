# Independent archive test: insufficient usable history

**No candidate could be evaluated under the frozen rules.** This is a data-coverage failure,
not evidence that any strategy returned zero or lost money. The four-policy definition,
entry time, fees, history minimum and decision criteria were not changed after retrieval.

The [preregistration](2026-09-22-archive-preregistration.md) was committed and pushed as
`e7c78f46121e4bfa9915c91c2165fba9377de118` before full retrieval. The evaluator was committed
as `5f5370c40128aeda796c00bccf229c3f368f03a4` before evaluation; its bytes and frozen shared-code
hashes were verified unchanged. Evaluation completed at 2026-09-22 06:34:16 UTC.

## Coverage and causal availability

| Measure | Result |
|---|---:|
| Requested events, May 1–June 28, all 15 cities | 885 |
| Events with every required exact 15:00 UTC candle | 61 |
| Normalized market rows | 366 |
| Events rejected for at least one missing exact candle | 824 |
| Events passing the existing probability-sum completeness rule | 60 |
| Usable holdout events, June 2–28 | 30 of 404 |
| Eligible earlier settled ladders per entry | 29–55 |
| Required earlier settled ladders | 60 |
| Entry days meeting that minimum | 0 of 27 |

The 60 usable ladders across the entire archive cannot initialize earlier decisions: some
targets had not happened or settled at those entries. Even the final entry had only 55
eligible earlier ladders. The deliberately excluded NYC June 15 probe also lacked exact
candles, so it is already among the 824 rejections. Excluding that known probe leaves 823
missing events out of the 884-event inference universe.

| Fixed policy | Selected orders | Verdict |
|---|---:|---|
| Joint bias/scale, inherited pilot selection | 0 | Not evaluated |
| Joint parent-selected NO, without backfilling | 0 | Not evaluated |
| Independently refitted bias only | 0 | Not evaluated |
| Independently refitted scale only | 0 | Not evaluated |

ROI and meaningful confidence intervals cannot be estimated. The machine report's `[0, 0]`
bootstrap arrays and zero-dollar stress values are empty-sample defaults, not confidence
bounds on a measured return. No candidate met the preregistered criteria, and no trading rule
or live setting was promoted.

## Why exact-candle coverage is sparse

Missing candles do not establish missing liquidity. On the excluded NYC June 15 event,
the T78 contract had identical 28¢/29¢ quotes at 14:58 and 15:01 UTC on June 14, but no 15:00
candle. B78.5 had identical 27¢/29¢ quotes at 14:56 and 15:02, again with no 15:00 candle.
Those surrounding candles had zero trade volume. They suggest inactivity, but do not prove
an executable quote throughout the gap; no quote was carried forward to repair the test.

The [historical candle documentation](https://docs.kalshi.com/api-reference/historical/get-historical-market-candlesticks)
defines inclusive time boundaries without specifying omitted-minute semantics. Historical
candles also lack depth, queue position and fill evidence. This retrieval therefore does not
provide the independent executable-price validation needed to promote a replacement strategy.

## Reproduction and evidence

- [Full policy/coverage output](2026-09-22-archive-validation.json).
- [Normalized independent captures](2026-09-22-archive-captures.jsonl.gz), compressed separately
  from canonical data. Uncompressed SHA-256:
  `094dafd21aee0c78f6e0912a6caca5dcc8f5b86020855a1ac56d35b23224fe26`.
- [Event coverage and extraction provenance](2026-09-22-archive-coverage.json.gz), including
  every rejection and the full local manifest's hash. Raw response bodies and their full
  URL/hash index remain in ignored `data/raw/kalshi_archive_http/` and the download manifest.
- Importer SHA-256:
  `451337fa0fc3760ee71e50d10666eb7ed1f06f24b5c46fe6163603e2640f4363`.

Reproduce the unchanged evaluation without more network calls:

```bash
mkdir -p data/raw/kalshi_archive_reproduction
gzip -dc reports/2026-09-22-archive-captures.jsonl.gz \
  > data/raw/kalshi_archive_reproduction/captures.jsonl
python3 scripts/archive_alpha_validation.py \
  --captures data/raw/kalshi_archive_reproduction/captures.jsonl
```

The [September validation](2026-09-21-alpha-validation.md) remains the substantive evidence
about current losses: recent data no longer supports the old warm correction, and apparent
replacement profits remain uncertain. Reliable forward alpha has not been established.
