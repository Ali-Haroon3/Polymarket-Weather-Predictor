# October 16: block trades do not change the favorite-longshot results

Addendum to `reports/2026-10-12-favorite-longshot-study.md`, written after CodeRabbit's review of
[PR #61](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/pull/61). The frozen report
is unchanged; its recorded protocol hashes still match. The forward test (section 4) is
unaffected, because it reads captured quotes rather than the trade tape.

**Result: both verdicts stand, and every number is the same.** Re-pulling every tape with block
trades excluded produced identical trades, identical orders and identical P&L.

## Block trades

Kalshi's trade endpoints return block trades unless the request passes `is_block_trade=false`.
The October 12 pulls omitted the filter, so block prints could in principle have set a signal or
a fill. `scripts/favorite_longshot_study.py` now passes the filter in both tape fetchers, and
every tape was re-pulled through the committed script:

| Tape | Markets | Trades, Oct 12 pull | Trades, filtered | Orders changed |
|---|---:|---:|---:|---:|
| Holdout decision windows | 49,894 | 432,568 | 432,568 | 0 of 509 |
| Replication decision windows | 37,654 | 228,514 | 228,514 | 0 of 741 |
| In-sample markets, all trades after capture (block-only query) | 6,492 | — | 0 block trades | — |

| Test | Orders | Net after fees | ROI | 1¢ stress | Day-bootstrap 95% | Verdict |
|---|---:|---:|---:|---:|---|---|
| Holdout (§5) | 509 | +$520.19 | +6.95% | +4.51% | −2.56% to +16.49% | CONSISTENT, unchanged |
| Replication (§6) | 741 | +$1,090.49 | +10.03% | +7.46% | +1.48% to +18.88% | CONFIRMED, unchanged |

The replication interval moved from +1.32%–+18.73% to +1.48%–+18.88%, even though the trades are
identical. The shared `weather_challenger.summarize` bootstrap resamples day groups in the order
trades arrive, and the committed fetcher returns markets in a different order. This is
resampling noise, not a data change, and the lower bound stays above zero under both orders.
`weather_challenger.py` is hash-pinned by the challenger registration, so it is left unchanged.

The committed market lister drops 18 malformed legacy tickers that the October 12 scratch lister
had kept (`HIGHAUS--23MAY11-*`, `HIGHMIA--23MAY11-*`, `HIGHCHI-2-24FEB28-*`). Their decision
windows hold no trades, so they never produced an order.

## Other review items

- **Equal-timestamp ordering: rejected, with evidence.** CodeRabbit suggested sorting by time
  alone so ties keep tape order. Ties are common: 124,747 tied YES timestamps across the two
  tapes. Each tie is one taker order sweeping several price levels, and the API lists those fills
  in no fixed order: 10,469 groups ascending in price, 10,288 descending, 13,192 mixed and 25,768
  single-price. A YES taker fills the cheapest ask first, so the existing (time, price) sort is
  execution order. Tape order would make the fill arbitrary. The code now carries a comment
  saying so.
- **Fee formula wording: clarified here.** Section 4's `ceil(0.07·C·P·(1−P))` means rounding up
  to the **cent**: `ceil(100·0.07·C·P·(1−P) − 1e−9)/100`, as in `go_live_gate.kalshi_fee`, which
  the frozen code has always used. The rule did not change.
- **Maker simulation start time: fixed.** A resting order now never starts before its capture
  time. Canonical captures are date-only, so no current number moves.
- **Drawdown-latch wording: fixed.** CLAUDE.md and the `kalshi_pilot.rs` module doc now say the
  guard latches in practice once it trips with nothing open. The dated October 1 report is left
  as written, and its wording is flagged in CLAUDE.md.

## Reproduce

```sh
python3 scripts/favorite_longshot_study.py --fetch-holdout research-output/holdout-tape.jsonl
python3 scripts/favorite_longshot_study.py --fetch-replication research-output/replication-tape.jsonl
python3 scripts/favorite_longshot_study.py --holdout research-output/holdout-tape.jsonl
python3 scripts/favorite_longshot_study.py --holdout research-output/replication-tape.jsonl
```

Each fetch takes about 1.5–2 hours at the script's throttle. Keep the tapes out of `data/`.
