# Prospective paper evaluation of the unchanged scale-only challenger

**Register one primary candidate and one fixed evaluation before its first prospective decision.**
This is a paper-research protocol, not a claim of profitable alpha or permission to trade. The
[October 1 study](2026-10-01-strategy-recovery.md) already preferred scale-only calibration;
this protocol does not choose another family, retune the policy, or rehabilitate historical returns.
The exact machine-readable rules are in
[the accompanying specification](2026-10-07-challenger-preregistration.json). Both files must be
committed before **October 8 at 00:00 UTC**, the first eligible capture date. If that deadline is missed, this registration
cannot be claimed retroactively; a new future study must be registered instead.

## Candidate and scope

The sole primary family is **`scale_only`**. `market_normal`, `joint_shape`, `bias_only` and
`weather_blend` remain descriptive comparisons, with no primary success claim or fallback winner.
A later choice among them requires a new future study. The original market-shape ledger, frozen
NO shadow, historical replays and archive studies cannot contribute orders to this sample.

The deployed decision policy remains `weather-challenger-v1`, from reviewed base commit
`269b42de66e13a42e787920ef584ecaf49997f4c`, with policy SHA-256:

`d7509940235a42ab998357cf067b868bd3689e27e7686a069299ac24d2a95df1`

The JSON pins the complete policy specification and all three implementation hashes. Preserve
its existing earliest complete lead-at-least-one ladder, independently fitted scale, history rules,
price/edge limits, one position per city/target and five positions per day. Each order uses integer
contracts and at most **$15 including rounded modeled fees**. Daily fitting under this unchanged
algorithm is allowed; changing the algorithm or selecting parameters using evaluation returns is not.
A changed policy hash makes this original study incomplete. Preserve its data and register a new
future study; do not pool versions or move these dates.

## Calendar and stopping

Eligible capture dates are **October 8 through December 6, 2026**, inclusive: exactly **60 UTC days**.
The fixed outcome-receipt cutoff is **December 22, 2026 at 00:00 UTC, exclusive**. Final assessment
cannot occur earlier. This allows fifteen full calendar days after the last capture date; it does
not assume every market will settle within that time. Unresolved primary selections prevent a
complete passing verdict. Never extend the window to acquire more winners, more observations or
a more favorable confidence interval. No interim result can authorize early success or live orders.

Operational checks may inspect artifact delivery, policy identity and completeness. Commit and
synthetically test the offline evaluator **before inspecting prospective performance**. After that,
interim descriptions may be produced, but final thresholds and dates remain fixed. This study is
prospective, not blinded: routine market data and other audit results remain visible. Any necessary
code repair must be versioned and cannot silently change this sample's decision policy.

## Required evidence and duplicate handling

Use scheduled original attempts of `.github/workflows/daily-capture.yml` in
`Ali-Haroon3/Polymarket-Weather-Predictor`. Preserve complete paginated run listings, every original
attempt, every run's complete artifact listing, failed/missing/expired records and exact retrieval
bytes, safe response headers, times and hashes. Local reconstructions, manually dispatched jobs,
rerun attempts and a later `study` replay cannot establish a prospective decision.

For each day, require the original `weather-recovery-{run_id}-1` ZIP and its `weather-shadow.json`.
The saved mode must be `shadow`; `latest_capture` and `required_capture_date` must equal that UTC
day, and policy/specification hashes must match this registration. Archive the exact captured input
bytes matching `capture_sha256` as well as archive/member hashes and GitHub's archive digest when
provided. Missing input linkage is unknown evidence, not permission to regenerate selections.

The conservative publication bound is the maximum of artifact `created_at`, artifact `updated_at`
and the terminal original attempt's `updated_at`. Both that bound and run creation must fall on
the capture UTC day, and the bound must precede every selected target date at 00:00 UTC. Metadata
must establish repository, workflow, run/attempt identity and artifact binding. Named provider
creation/start/update fields need not be identical to one another; preserve discrepancies and require
unambiguous same-day/before-target qualification. This new protocol does not alter the older
source-study's frozen timestamp checks.

Retain all same-date artifacts. Count one only when their capture hash, required/latest dates,
policy hashes/specification and all five saved family selections agree exactly; choose the earliest
publication bound, then lowest run ID. Conflicting candidate artifacts, or an earlier decision that
may have been produced but is unavailable, make that date unknown. A missing-artifact attempt can
be excluded as a harmless duplicate only when preserved jobs/step evidence proves its challenger
step was skipped by the already-captured guard **and a complete same-date candidate artifact exists**.
Do not substitute a successful backup for an unknown earlier decision.

**All 60 dates require valid decision evidence**, including explicit zero-order artifacts. A missing
or failed day is not a zero-order day. Report every missing date; incomplete coverage is inconclusive,
not a reduced sample that can pass. This strict criterion may leave the experiment inconclusive even
if its observed subset is profitable. It is an advance evidence requirement, not a discovered result.

## Settlement and costs

Settle the saved selections directly, without fitting, ranking, backfilling or replacing orders.
Validate ticker, target date and city against Kalshi capture evidence. Binary outcomes require a
recorded `outcome_observed_at` strictly after the publication bound and strictly before the fixed
cutoff. Receipt means first capture-process observation, not exchange settlement or source publication.
There is no legacy target-plus-two-day fallback for evaluation outcomes. Legacy approximations remain
part of the unchanged model's historical training and must remain disclosed.

Conflicting outcomes or unresolved orders prevent passing. Retain every selection, its full saved
cost and unresolved worst-case exposure; never discard unsettled losers or substitute a later label.
Post-cutoff resolutions/revisions may be shown separately as sensitivity, never to rescue the primary
verdict. Preserve each original version and receipt.

For each saved order, net is winning contracts minus saved fee-inclusive cost. Primary ROI is
summed net divided by summed fee-inclusive cost, **not account return**. Validate the existing
rounded fee model `ceil(100 × 0.07 × contracts × price × (1 − price) − 1e−9) / 100` and saved integer
sizing. These are modeled fees; current account-specific fees, depth and fills remain unverified.
The one-cent stress keeps ticker and side, adds $0.01 to paid price, resizes within the same $15
budget and recalculates rounded fees using the unchanged sizing helper. No reselection is permitted.
An unpriceable stress order prevents passing that criterion.

## Primary endpoint and uncertainty

Group all primary contracts and cities by **target date**. Define the inclusive calendar grid from
the minimum to maximum target date of all saved primary selections **without consulting outcomes**;
unresolved selections still define its bounds. Include internal dates without orders as zero net/zero
cost groups. Do not append zero-return dates after the final selected target or erase internal gaps.

Use a circular stationary bootstrap of joint daily `(net, fee-inclusive cost)` pairs: geometric block
lengths with restart probability **1/7** (mean seven days), **20,000 replicates**, seed **20261007**.
Each replicate starts at a uniformly chosen grid index; each subsequent draw restarts uniformly with
probability 1/7, otherwise advances one calendar day circularly. Take exactly the grid's length in
each replicate. The JSON fixes random calls and percentile indices. Keep zero-cost replicates with
conservative ROI zero and report their count. Require the **2.5th percentile** ROI to exceed zero.

This method is approximate and relies on weak dependence and stationarity; it does not guarantee
finite-sample coverage or protect against an unseen loss state or regime change. Random geometric
blocks distinguish it from fixed-length block resampling. See the primary
[Politis–Romano stationary-bootstrap paper](https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476870).
Sixty calendar days are an operationally fixed trial, not a demonstrated power calculation.

## Final decision rule

A complete assessment requires **at least 100 settled primary orders**, **at least 30 distinct target
dates with orders**, and **at least two strictly negative target-day net groups**. These are conservative administrative
minimum-information rules, not theorems. Fewer than two negative groups yields insufficient tail
evidence even if every observed selection wins; two losses still cannot represent all unseen tails.

Only after those evidence and information requirements pass can the candidate meet the preregistered
paper-research criteria. All five following conditions must hold:

1. Total modeled net is strictly positive.
2. Stationary-bootstrap 2.5th-percentile ROI is strictly positive.
3. One-cent-stressed modeled net is strictly positive, with every stress order priceable.
4. Removing the city with the highest total net still leaves strictly positive net; at least two cities exist.
5. Maximum observed target-day realized drawdown is at most **$50**.

For the last condition, begin cumulative net and peak at zero, update the peak after each target-day
total, and take the largest peak-minus-cumulative value. This is a **research risk-compatibility
condition**, not an account-loss cap or a replay of the running guard. It does not reserve unresolved
exposure or justify assuming all simultaneous quotes would fill. The $50 level matches the existing
risk scale; its acceptability has not been learned from prospective results.

Before cutoff, verdict is pending. Invalid/incomplete evidence is inconclusive; inadequate sample or
tail information is insufficient. Complete evidence failing any performance/risk condition does not
meet the criteria. Passing all conditions means only **“meets preregistered paper-research criteria.”**
It never means guaranteed positive expected return or automatic live admission. Actual execution,
fee schedules, depth, account exposure and a separately authorized deployment still need validation.

## Implementation boundary

At registration, no prospective scorer exists yet. The next implementation is a small offline evaluator over saved
artifacts and preserved outcomes, with synthetic tests for missing/duplicate/late evidence, sizing,
conflicting/unresolved settlements, fixed dates, grouping and verdicts. It must reproduce this
specification without changing the decision policy or inheriting the original pilot's order count.
Any ambiguity must be resolved and committed before prospective performance inspection; material
post-start protocol changes require a new future registration rather than silent amendments.

This protocol leaves the separate September 25 source-availability study, October 10 freeze,
October 10–23 reserved embargo and October 24 release gate unchanged. Its reserved observations
are not inputs here. The losing parent pilot remains subject to deployed drawdown/admission guards.
