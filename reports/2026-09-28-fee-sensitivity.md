# Fee rounding sensitivity: pennies do not explain the recent paper losses

**Under the stated single-fill assumptions, changing balance precision explains only $0.0216 of the latest $30.18 paper decline.** It does not turn those new settlements profitable. Frozen paper accounting, strategy selection and live admission remain unchanged.

## Primary documentation and limits

The current [official rounding documentation](https://docs.kalshi.com/getting_started/fee_rounding) distinguishes $0.0001 direct-member balances from $0.01 non-direct-member balances. Model fees first round upward to six dollar decimals; another adjustment aligns the signed cash change to the relevant grid. An order carries excess rounding between fills and can receive rebates, including across taker/maker transitions. Thus a whole-cent display table alone is insufficient to infer an actual fee.

The [general fee schedule](https://kalshi.com/docs/kalshi-fee-schedule.pdf) supplies the standard quadratic taker coefficient 0.07 and default multiplier 1, subject to product exceptions. Its text was readable through the web tool. Direct PDF retrieval returned HTTP 429, and no usable page image or PDF body was obtained; visual verification is not claimed. The [fixed-point documentation](https://docs.kalshi.com/getting_started/fixed_point_migration) describes fractional quantities and subcent prices, which can make cash alignment relevant beyond the fee alone.

This documents current mechanics, not the user's membership category, historical effective terms, a market-specific override or actual execution costs. It does not authorize live trading or unlock the source evaluator's fee-inclusive comparisons.

## Conditional paper comparison

The [machine-readable calculation](2026-09-28-fee-sensitivity.json) holds the existing selected contracts, prices and outcomes fixed. It assumes one taker buy fill per intent, coefficient 0.07, multiplier 1, and an empty order rounding accumulator. Every settled price is within $0.0000000000000001 of a whole cent; the calculation explicitly normalizes that floating-point serialization noise using Decimal arithmetic. These assumptions are a sensitivity, not evidence of fills.

| September 27 sample: 71 settlements | Modeled fees | Cumulative net | Change since September 25 |
|---|---:|---:|---:|
| Frozen paper scorer | $38.0500 | +$28.3600 | −$30.1800 |
| Single-fill, non-direct precision | $38.0500 | +$28.3600 | −$30.1800 |
| Single-fill, direct precision | $37.6929 | +$28.7171 | −$30.1584 |

The two new losing YES intents and winning NO intent generated **−$28.25 before fees**. Their modeled fees are $1.93 under the frozen scorer and $1.9084 under direct precision. The loss remains substantial under either assumption; rounding is not its cause.

Across all 71 settlements, direct precision changes cumulative net by only $0.3571. That small change does not resolve the wide return interval, thin prospective NO sample, or one-cent execution sensitivity in the [September 27 audit](2026-09-27-forward-update.md). No new alpha follows from this comparison.

## Why this is not an actual-fill fee model

The scorer applies its formula once to each order's quantity and price. Reconciled live orders use an average execution price. The quadratic formula is nonlinear: an illustrative order filled 10 contracts at $0.20 and 10 at $0.80 has $0.2240 raw model fees, whereas applying the same coefficient to 20 contracts at the $0.50 average gives $0.3500. Fill-level information matters independently of account precision.

The repository does not retain actual exchange fee/rebate components in its live-fill accounting. The new source evaluator preserves decimal book levels and series metadata, but a displayed book cannot establish actual fills, future rebates or the account's applicable precision. Its `payout_comparison_available=false` remains appropriate. Any future execution-aware implementation must retain authoritative fill-level costs and separately handle unsupported metadata; this report does not silently substitute a more favorable fee model.

## Evidence and reproduction

Exact documentation HTML was retrieved anonymously and retained under ignored `data/raw/fee-authority-20260928T0429Z/`:

- Rounding: received September 28 at 04:29:31.220998 UTC; 263,525 bytes; SHA-256 `9cd3b54bc3c6159ae70e820197026b3194601b2b95d5a5875360b91d995c6349`.
- Fixed-point representation: received at 04:29:31.465115 UTC; 273,713 bytes; SHA-256 `fed4f6236cd22927ad7434c9ef6c9499f067c35f075764649c1f30c0e4f11e32`.

The tracked calculation includes response metadata, immutable input commits/hashes and every settlement's conditional fee amounts. It does not bundle the documentation text. This check's failed PDF request is retained separately; it was not retried or replaced with another vintage.

For each normalized price `p` and whole-contract quantity `n`, the calculation is:

```python
from decimal import Decimal as D, ROUND_CEILING

cost = n * p
raw_model_fee = D("0.07") * n * p * (1 - p)
six_decimal_fee = raw_model_fee.quantize(D("0.000001"), rounding=ROUND_CEILING)
# Single buy fill, no pre-existing accumulator; grid is 0.0001 or 0.01 dollars.
fee = (cost + six_decimal_fee).quantize(grid, rounding=ROUND_CEILING) - cost
net = (n if won else D(0)) - cost - fee
```

Independent calculation matched all three commit-level summaries and the incremental result. Each non-direct conditional fee also matched the existing frozen per-order fee. Original capture/ledger, protocol, collector, scorer and gate bytes were not changed.
