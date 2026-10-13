# October 12: Codex changes review and a favorite-longshot study

**Main finding.** The pilot's entire loss comes from buying YES below 50¢. Kalshi's weather ladders
show a favorite-longshot bias: YES contracts priced below 50¢ lose money for a taker, and
favorites priced at 50¢ or more win more often than their price implies.

On the past data, one rule with no fitted parameters made **+10.3% after fees on 235 orders**:
buy YES at the ask whenever the ask is 50¢ or more. That result is in-sample, and its interval
touches zero.

The rule was then tested on a **holdout** whose protocol was fixed before any of its data was
fetched: 49,912 Kalshi weather markets that settled before captures began (2021 to July 4, 2026),
priced from the public trade tape. It made **+6.95% after fees on 509 orders over 256 days**, and
+4.5% with every entry 1¢ worse. That is a **CONSISTENT** verdict. It is not CONFIRMED, because
the day-bootstrap interval is −2.6% to +16.5%. Pooling both samples (disjoint dates, descriptive
only) gives +8.0% on 744 orders, with an interval of +0.9% to +15.6%.

A second preregistered test then ran on **85 Kalshi daily high/low temperature series that no
analysis in this repo had read**: 37,654 markets. It is **CONFIRMED**: **+10.0% after fees on 741
orders over 194 days**, +7.5% with every entry 1¢ worse, and a day-bootstrap 95% interval of
**+1.3% to +18.7%**.

This is the first edge in the repo to survive untouched out-of-sample tests. It is modest: about
$1 per $15 order, roughly 2.6 orders a day on the tracked ladders. A frozen forward paper test on
captured quotes starts with the October 13 capture. There is no live authority.

Canonical inputs: main `83d46ea`, 11,817 capture rows (6,312 resolved lead-1 Kalshi rows), and the
unchanged pilot ledger. Reproduce with `python3 scripts/favorite_longshot_study.py`.

## 1. Codex changes since `0af7996`: sound, with one wrong claim

- **Tests pass.** `cargo test` passes in every binary, and all 331 Python unittests pass.
- **The challenger shows no look-ahead.** I checked the following in `scripts/weather_challenger.py`:
  - outcome availability, a strict day before decision;
  - history windows;
  - executable prices;
  - rounded fee inside the $15 budget;
  - per-event and per-day caps;
  - strict-threshold ladder bounds.

  Every number in the October 1 family table reproduces exactly from `096fa88`.
- **The forward scorer and dashboard wrapper are correct.** They reproduce:
  - 84 settled orders, −$89.53, −7.22%;
  - October 11's 10 selections, two of them settled, at −$2.44.
- **Wrong: the drawdown guard is a permanent latch in practice.** CLAUDE.md, the
  strategy-recovery report and the `kalshi_pilot.rs` module doc all say "outstanding settlements
  can recover the curve, so this is not a permanent latch".
  - With zero open orders and new entries blocked, nothing can ever settle, so the $167.87
    drawdown can never shrink.
  - Even on October 1 the three open orders could recover at most $50.07 of the $160.94 drawdown.
  - The market-shape paper sample is therefore frozen at 84/100 settlements, and gate criterion 1
    can never be reached.

  Stopping a losing strategy is defensible, but the docs should say it is a latch that only a
  human can clear.
- **Overstated: the "causal training" framing.** All 911 events in the October 1 study used the
  target+2 availability approximation. None used an observed outcome-receipt time.
- **Fragile: the preregistered challenger trial.** A single closed day without a valid primary
  artifact makes the whole 60-day trial inconclusive. That includes a day where the
  continue-on-error challenger step fails and only the backup run succeeds.

## 2. What does not work

**Every challenger family loses on the newest data.** I reran study mode on captures through
October 12 and sliced selections to capture dates October 2–11, which were unseen when scale-only
was chosen. This is descriptive only and is not the registered sample.

| Family | Orders | Net | ROI |
|---|---:|---:|---:|
| Market normal | 5 | −$26.28 | −35.9% |
| Joint bias+scale | 43 | −$145.67 | −23.0% |
| Bias only | 28 | −$165.06 | −39.8% |
| Scale only | 11 | −$29.58 | −18.4% |
| Weather blend | 12 | −$49.53 | −28.1% |

**There is no ladder arbitrage.** Across 1,112 complete books, the sum of YES asks plus fees
exceeds $1, and only one book was above break-even, by 0.2¢. The median overround is about 12¢.

**Making a market in longshots earns nothing.** This is new evidence from Kalshi's public trade tape: 6,312 markets and
about 5.5M trades, retrieved anonymously after each capture. The test rests an order 1¢ inside the
spread from 15:30 UTC on the capture day and fills it only when a later taker trade crosses it.

- Selling YES on longshots looks profitable on snapshots, because realized frequency sits below
  the ask (+2% to +6% per dollar).
- On the tape it nets about zero: −$4 on 829 fills at 2–10¢, and +$12 on 408 fills at 10–20¢.
- The reason is adverse selection. The orders that fill are the ones the market moves toward.
- The weather series report `fee_type: quadratic` (not `quadratic_with_maker_fees`), so fees are
  not what kills the edge.

## 3. What does work: favorites are underpriced and longshots are overpriced

**Taker returns by executable price.** These are lead-1 captures, after Kalshi's rounded fee,
for $15 orders:

| Price paid | Buy YES at ask | Buy NO at 1−bid |
|---|---:|---:|
| 2–10¢ | −55.5% (n=2,736) | — |
| 10–20¢ | −33.7% (n=964) | — |
| 20–35¢ | −11.1% (n=1,232) | — |
| 35–50¢ | −0.8% (n=947) | −26.1% (n=124) |
| 50–65¢ | **+10.4% (n=218)** | −13.2% (n=711) |
| 65–80¢ | +6.9% (n=13) | −5.3% (n=1,301) |
| 80–98¢ | — | ≈0% (n=2,695) |

This matches Bürgi, Deng & Whelan's study of more than 300,000 Kalshi contracts. Low-priced
contracts win less often than their prices require, and high-priced contracts earn small positive
returns. That external result is the prior here; it was not mined from this sample.

**The pilot's actual paper ledger fits the same pattern.** Market-shape's YES side bought the
under-50¢ neighbours of the ladder mode, which is exactly the overpriced region.

| Pilot orders | Orders | Wins | Net | ROI |
|---|---:|---:|---:|---:|
| YES below 50¢ | 52 | 15 | **−$170.51** | −21.1% |
| YES at or above 50¢ | 7 | 4 | +$6.22 | +5.9% |
| NO | 25 | 24 | +$74.76 | +20.1% |

**The favorite rule:** buy YES at the captured ask when 0.50 ≤ ask < 0.99, sized at $15 including
the fee. Results on all in-sample data (captures July 5–October 12):

- 235 orders over 79 target days, 64% won.
- +$354.83 net on $3,437.17 cost, including $103.27 fees: **+10.3% ROI**.
- Target-day bootstrap 95% interval is **−1.3% to +21.2%**.
- Maximum target-day drawdown is $145.

| Check | Result |
|---|---|
| Positive every month? | July +7.0% (36), August +9.9% (63), September +4.4% (92), October +26.0% (44) |
| Adverse entry price | +1¢: +8.5%; +2¢: +6.5% |
| Without best city (Vegas) | +$263.62, +9.1% |
| Cities positive | 10 of 14 (Miami, Atlanta, Phoenix and Philadelphia negative) |
| Threshold sensitivity | 40¢ +4.4%, 45¢ +7.6%, 50¢ +10.3%, 55¢ +7.6%, 60¢ −2.6% (n=53) |
| Quote executable? | 94% had taker-YES prints at or below the captured ask within 14:30–16:30 UTC; median 59 contracts against roughly 28 needed |
| Maker version (bid 1¢ inside) | 97 fills at 50–65¢, +11.6% |

**Why this is not proof:**

- I chose the 50¢ cut after seeing the band table, among roughly 20 side/band/maker cells.
- The interval includes zero, and dates are correlated.
- The sample covers one summer and early autumn.
- The rule's whole benefit could shrink once other traders notice it.

## 4. Frozen forward test (paper only)

The rule and its evaluation are fixed by this commit, in `scripts/favorite_longshot_study.py`:

- **Universe:** Kalshi daily-high rows in canonical `data/captures.jsonl` with
  `captured_at = target − 1`.
- **Rule:** buy YES at `best_ask` when 0.50 ≤ ask < 0.99. Use the largest integer contract count
  whose principal plus `ceil(0.07·C·P·(1−P))` fits within $15. No caps, filters or fitted values.
- **Window:**
  - Capture dates **2026-10-13 through 2026-12-11** inclusive. A missing capture day stays
    missing; no backfill, extension or substitution.
  - Score once, after every window outcome has resolved or on **2026-12-22**, whichever comes
    first. Unresolved orders at that point are reported, not counted.
- **Verdict:**
  - **Consistent** requires all of the following:
    - at least 100 settled orders over at least 30 target days;
    - net ROI above 0;
    - 1¢-stress ROI above 0.
  - **Confirmed** additionally requires the target-day bootstrap 95% lower bound to be above 0.
  - Anything else is **not supported**. Fewer than 100 orders is **inconclusive**.

  At the in-sample rate of about 2.6 orders a day, the window should yield about 150 orders. The
  confirmed tier is deliberately hard to reach at that size.
- **Separation:**
  - This test is separate from the original pilot and from the preregistered scale-only
    challenger, and it does not amend either one.
  - It grants no live authority.
  - Fills are quote-based. The tape check above is the only execution evidence.

## 5. Holdout test on earlier markets (protocol fixed before any of this data was fetched)

The rule needs only a price and an outcome; it never parses the ladder. So it can be tested on
Kalshi daily-high markets that settled **before the first capture**, using the public trade tape
in place of captured quotes. None of these markets were used to choose the rule.

Protocol (written at 2026-10-12, before any pre-July market or tape was retrieved):

- **Universe:** every finalized market in the 15 `WEATHER_SERIES` whose ticker target date is
  **before 2026-07-05**, as far back as Kalshi's historical API returns.
- **Decision time:** T = 15:00 UTC on target − 1, the same timing as the capture.
- **Signal:** the last taker-YES print in [T − 6h, T). If there is none, skip the market.
- **Entry:** when 0.50 ≤ signal < 0.99, buy at the **first taker-YES print in [T, T + 2h)**, the
  price an actual buyer paid after the decision. If there is none, skip.
  - Sizing is $15 including the rounded fee, as in the forward test.
  - Outcome is the market's `result`.
- **Verdict:** same tiers as section 4. **Consistent** needs at least 100 orders, at least 30
  target days, net ROI above 0 and 1¢-stress ROI above 0. **Confirmed** additionally needs the
  target-day bootstrap lower bound above 0. One run, reported in full, with no slicing to rescue it.
- **Proxy check (reported only, not part of the verdict):** run the same tape proxy on the
  in-sample capture period, to show whether it tracks the capture-based result.

### Holdout result: **CONSISTENT** (not confirmed)

The protocol above was hashed at 20:52 UTC (SHA-256 `068bbb47…f210a`) before any holdout data was
retrieved. The pull covered **49,912** finalized pre-July-5 markets back to August 2021, with zero
fetch failures.

| | Orders | Target days | Win | Net after fees | ROI | 1¢ stress | Day-bootstrap 95% |
|---|---:|---:|---:|---:|---:|---:|---|
| **Holdout 2021 to Jul 4 2026** | **509** | **256** | 55% | **+$520.19** | **+6.95%** | **+4.5%** | −2.6% to +16.5% |
| In-sample Jul 5 to Oct 12 2026 (§3) | 235 | 79 | 64% | +$354.83 | +10.3% | +8.5% | −1.3% to +21.2% |

- **Verdict:** at least 100 orders, at least 30 days, ROI above 0 and 1¢-stress ROI above 0 all
  hold, so the result is consistent. The bootstrap lower bound is negative, so it is **not
  confirmed**.
- **By year:**

  | Year | Orders | ROI |
  |---|---:|---:|
  | 2021 | 7 | −5.3% |
  | 2022 | 22 | −0.5% |
  | 2023 | 7 | −0.3% |
  | 2024 | 53 | +3.7% |
  | 2025 | 86 | +20.8% |
  | 2026 H1 | 334 | +4.8% |

  The early years are tiny, so this is not evidence of decay.
- **Without the best city (Chicago):** +$315.16.
- **Cities:** 10 of 15 positive. Seattle, DC, Atlanta, Denver and Phoenix are negative.
- **Where the fills landed:** the protocol fills at the first post-decision print, which is
  sometimes far below a stale signal. Fills below 40¢ (n=99) returned −2.5%. Fills at 50–65¢
  (n=270) returned **+10.0%**, the same as the in-sample 50–65¢ band (+10.4%). This breakdown is
  descriptive and does not change the verdict.
- **Pooled in-sample plus holdout:** the two samples cover disjoint dates. Together they make
  744 orders, +$875.02 and **+8.0% ROI**, with a day-bootstrap interval of **+0.9% to +15.6%**
  and +5.8% under 1¢ stress. This pooling was not preregistered, and the in-sample half carries
  the threshold-selection bias.

- **Proxy check:** the same tape proxy over the in-sample capture period gives 219 orders,
  **+14.6%** and +12.5% under 1¢ stress. The captured-quote rule over the same period gave 235
  orders and +10.3%. So the proxy tracks the sign, but it may read about 4 points high. If that gap
  carries over, the holdout's quote-equivalent return could be closer to **+3%**. This is the
  largest unresolved caveat, and the forward test, which uses captured quotes, is what settles it.

**Reading.** An independent sample of more than twice the in-sample size reproduces the sign, and
roughly the size, of the favorite effect after fees and adverse-price stress. The effect also
matches published Kalshi-wide evidence. That makes the favorite-longshot edge the first candidate
in this repo to survive an untouched holdout.

It is modest: about 5–10% per order, roughly $1 per $15 order, at about 2.6 orders a day. It is
not yet confirmed at the 95% level on the holdout alone. The section 4 forward paper test remains
the gate before any money is involved.

## 6. Replication on untouched series (protocol fixed before any of this data was fetched)

The section 5 holdout is consistent but not confirmed. Rather than reinterpret it, the same rule
gets one more independent test, on Kalshi daily temperature series that no analysis in this repo
has ever read.

- **Universe:** every finalized market, with target date on or before 2026-10-11, in Kalshi's
  "Climate and Weather" series whose title names a daily highest/high or lowest/low temperature.
- **Exclusions:**
  - the 15 `WEATHER_SERIES`;
  - high-temperature aliases of their cities: `HIGHAUS`, `HIGHCHI`, `HIGHMIA`, `HIGHNY`,
    `KXHIGHHOU`, `KXHIGHOU`, `KXHOUHIGH`, `KXDENHIGH`, `KXHIGHTEMPDEN`, `KXPHILHIGH`;
  - national markets: `HIGHUS`, `KXHIGHUS`;
  - non-daily markets: `KXHIGHNYD`, `KXSJCLOWT`.
- **Rule and verdict:** identical to section 5. That means T = 15:00 UTC on target − 1, signal =
  last taker-YES print in [T − 6h, T), entry when 0.50 ≤ signal < 0.99, and fill = first taker-YES
  print in [T, T + 2h), at $15 including fee. The tiers are the same: consistent and confirmed.
- **Primary result:** the pooled replication universe. Highs and lows are reported separately as
  descriptive.
- One run, reported in full.

### Replication result: **CONFIRMED**

The protocol above was hashed at 22:24 UTC (SHA-256 `7b51af73…76687`) before the series were
listed or any tape retrieved.

- **Universe:** 85 series qualified and **37,654** markets were pulled, with zero fetch failures.
  All are 2026 markets, because these series launched in 2026.

| | Orders | Target days | Win | Net after fees | ROI | 1¢ stress | Day-bootstrap 95% |
|---|---:|---:|---:|---:|---:|---:|---|
| **Replication (primary)** | **741** | **194** | 62% | **+$1,090.49** | **+10.0%** | **+7.5%** | **+1.3% to +18.7%** |
| Daily highs (descriptive) | 103 | 71 | | +$438.80 | +28.9% | +23.8% | −4.1% to +68.9% |
| Daily lows (descriptive) | 638 | 189 | | +$651.69 | +7.0% | +4.8% | −0.8% to +14.9% |

- **Verdict:** at least 100 orders, at least 30 days, ROI above 0, 1¢-stress ROI above 0, and the
  bootstrap lower bound above 0 all hold, so the result is **confirmed**.
- **Concentration:** without the best series, +$927.58. 21 of 31 trading series are positive. The
  largest series takes 12.6% of capital. Maximum target-day drawdown is $297.
- **By month:** positive in 7 of 8 months.

  | Month | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | ROI | +5.8% | −4.7% | +10.4% | +16.4% | +12.0% | +9.7% | +18.3% | +0.3% |
- **By fill price (descriptive):** below 40¢, +23.7% (n=143); 40–50¢, +5.8%; 50–65¢, +8.7%
  (n=272); 65¢ and above, +4.8% (n=243). Every fill band is positive.

**Overall reading.** Three tests point the same way:

1. The rule was chosen in-sample: +10.3%.
2. A preregistered holdout on earlier markets came out consistent: +6.95%.
3. A preregistered replication on 85 untouched series came out confirmed: +10.0%, with the 95%
   interval above zero.

That is a validated favorite-longshot edge on Kalshi daily temperature markets, at modest size
after taker fees.

Remaining caveats:

- The tape proxy may read about 4 points high (section 5). Even after that haircut, the
  replication stays positive.
- Executable depth at size is only checked for the in-sample asks.
- These samples share calendar days, so weather regimes are not independent across them, even
  though the contracts are.

The section 4 forward paper test on captured quotes is still the step before any real money.

## Reproduce

```sh
python3 scripts/favorite_longshot_study.py                       # bands, rule, pilot ledger split
python3 scripts/favorite_longshot_study.py --fetch-tape research-output/kalshi-tape.jsonl   # ~35 min, ~425 MB
python3 scripts/favorite_longshot_study.py --tape research-output/kalshi-tape.jsonl         # + maker simulation
python3 scripts/favorite_longshot_study.py --fetch-holdout research-output/holdout-tape.jsonl  # ~1 h, ~50k markets
python3 scripts/favorite_longshot_study.py --holdout research-output/holdout-tape.jsonl       # section 5 verdict
python3 scripts/favorite_longshot_study.py --fetch-replication research-output/replication-tape.jsonl  # ~1.5 h, ~38k markets
python3 scripts/favorite_longshot_study.py --holdout research-output/replication-tape.jsonl  # section 6 verdict
python3 -m unittest tests/test_favorite_longshot_study.py
```

The tape is research data. Keep it out of `data/` and out of canonical captures.

Sources: [Bürgi, Deng & Whelan, *Makers and Takers: The Economics of the Kalshi Prediction Market*
(CEPR DP20631)](https://cepr.org/publications/dp20631); [VoxEU summary](https://cepr.org/voxeu/columns/economics-kalshi-prediction-market).
