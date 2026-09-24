# Loss audit: captures through September 21, 2026

The current strategy has not established a tradable edge. Keep real-money admission closed.
The only promising lead in the actual pilot sample is the NO side of orders already selected
by the existing top-five ranking. Switching to a NO-only strategy and filling the vacated
slots is a different rule, and its forward replay loses.

## Inputs and operational state

- Main updated by 11 commits to `e74b74c` (daily capture 2026-09-21).
- 9,482 canonical captures. Capture SHA-256:
  `99bc88df064f7151ad0db1d725db0f4bb25994456aeaa5930b4d5feaca47ac16`.
- Pilot ledger SHA-256:
  `84c42fd65b8829bf85c58b43ecd33a79536c35258cd1b7746211f343d4ddb6f4`.
- Every recorded order is a dry run; no live fills appear in this ledger. `gh variable list`
  returned no repository variables, including no `PILOT_LIVE`. This does not establish the
  state of any separately operated account or bot.
- Pre-existing local dashboard/capture changes were preserved in the named Git stash
  `pre-alpha-audit-2026-09-21 local capture and dashboard`. They were not merged into the
  canonical research data: 8,418 shared rows differ and 15 local rows have no upstream match.
  Many differences remove stored research fields. Mixing versions would destroy reproducibility.

## Actual pilot decisions, after modeled fees

| Sample | Settled | Wins | Stake | Net P&L | ROI |
|---|---:|---:|---:|---:|---:|
| Current market-shape strategy | 60 | 33 | $885.79 | −$1.31 | −0.15% |
| Its BUY YES orders | 41 | 15 | $607.14 | −$59.31 | −9.77% |
| Its BUY NO orders | 19 | 18 | $278.65 | +$58.00 | +20.81% |
| Retired model-shrunk strategy | 23 | 12 | $338.74 | −$13.27 | −3.92% |

Seven current orders are still open and excluded from realized P&L. The current strategy made
$30.21 gross, then incurred $31.52 of modeled fees. Fees consume the entire observed gross edge;
that is a diagnosis of this prospective sample, not a reason to assume maker fills are free.

The current sample spans only 12 target days. A deterministic 10,000-draw bootstrap resampling
whole target days gives an approximate 95% ROI interval of **−25.15% to +29.07%**. NO orders
span 11 target days; their interval is +5.30% to +36.06%, but this exploratory slice was selected
after seeing performance. The interval does not correct for selection, serial dependence
across dates, or unobserved rare tail losses. Eighteen wins in 19 bets is not proof of safety.

A one-cent adverse fill on each selected contract, holding quantities fixed, takes the current
strategy to **−$23.21**. The selected NO subset remains +$54.38 under the same stress. This is a
simple execution sensitivity, not a simulation of available depth, queue priority, or fees at
the new price.

The fee model uses the published general taker curve, `0.07 × C × P × (1−P)`, rounded upward
to cents per order. [Kalshi's July 7, 2026 fee schedule](https://kalshi.com/docs/kalshi-fee-schedule.pdf)
lists this curve with a series multiplier, default 1, and whole-cent worked examples. Its
rounding prose mentions centicents; this audit retains the repository's conservative whole-cent
model. Actual per-fill fees and series-specific multipliers must be reconciled before scaling.

## Why the obvious strategy switch fails

Replayed capture-time candidates **after the September 7 freeze**, fitting each day's parameters
only on complete ladders with target dates before that day. Used the pilot's actual 3% edge
plus 1% buffer after fees, integer $15 stakes, top-five ranking, $30 per city/target cap, and
ticker deduplication. Open candidates consume capacity before any outcomes are inspected.

| Capture-time policy | Settled | Open | Net P&L | ROI | Approx. 95% day-bootstrap ROI |
|---|---:|---:|---:|---:|---:|
| Both sides, top five | 60 | 7 | +$2.24 | +0.25% | −24.31% to +27.29% |
| Filter NO first, then take top five | 57 | 2 | −$9.55 | −1.14% | −14.16% to +10.11% |

The first row differs from the actual ledger because the capture and subsequent pilot API scan
see different quotes. Neither row proves fills. This research comparison does not reproduce
the pilot's dynamic replay/weekly breakers or account-wide held positions. Captures also lack
historical settlement publication timestamps: using prior target dates is the repository's
availability approximation, not proof that every outcome had been published at entry time.

The NO-only dashboard row was frozen September 7. Its forward Kelly-weighted result through
this capture is just +0.5% on 134 cells. It does not support replacing the current pilot with
a broad NO-only rule. Higher sample count there reflects a different sizing/selection policy.

## Prospective candidate, frozen September 21

**Rank first, then retain NO orders, without replacement.** Observe only NO orders that the
unchanged parent pilot actually selects under its normal ranking, caps, and breakers. Do not
backfill skipped YES capacity with lower-ranked NO bets. `pilot_alpha_audit.py` reports the new
sample strictly after this freeze separately from the exploratory 19 orders.

This is a shadow attribution candidate, not an executable new strategy, and its forward sample
is initially empty. Its paper evidence must accrue independently; it cannot inherit the parent
strategy's admission count. No signal thresholds, city blacklist, or live allocation changed.

## Implemented loss controls

Previously the workflow printed NO-GO but `PILOT_LIVE=1` could still place new orders. Now the
Rust pilot embeds the same Python gate and requires it to pass on every `--live` run, including
demo. Python 3 is therefore required at runtime for live admission. Missing, malformed,
duplicate, or conflicting evidence fails closed.

Reconciliation and expiry cancellation run before admission and strategy breakers. Outstanding
live orders without a fill/expiry verdict block new exposure, including orders from retired
strategies. Unreconciled intended winners no longer inflate reported profit; zero-fill orders
now reduce the reported fill ratio. The existing 100-settled-order, positive-ROI and loss-share
thresholds remain unchanged. Paper collection continues, with the new audit printed daily.

## Reproduce

```bash
python3 scripts/go_live_gate.py
python3 scripts/go_live_gate.py --strategy model-shrunk
python3 scripts/pilot_alpha_audit.py --replay
python3 -m unittest discover -s tests -p 'test_*.py'
cargo test --bin kalshi_pilot
```

`python3 scripts/go_live_gate.py --enforce` must exit 1 on the captured sample above. The audit
prints input hashes, settlement-day uncertainty, side/mode attribution, the empty prospective
candidate, and (with `--replay`) the replacement-policy comparison. All work is offline; none
of these commands places orders.
