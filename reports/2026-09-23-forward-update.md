# September 23 capture: positive cumulative paper P&L, weekly breaker stops new orders

**The paper pilot is now +$12.22 after modeled fees across 67 settled orders.** Two newly resolved YES winners added **+$54.97**. This does not validate alpha: the day-bootstrap interval spans substantial losses and gains, the one-cent adverse-price sensitivity remains negative, and admission still fails at 67/100 settled orders.

This update scores upstream capture commit `ffdb7929a045c6c6f1bc4b5bcb51303403bc4c44` with unchanged research rules. The [September 23 scheduled run](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/actions/runs/35877522658) reports that the existing weekly loss breaker stopped new orders. The ledger is byte-identical to September 22. No strategy, risk limit, trading setting, or historical outcome was changed by this audit.

## Fixed-rule score

| Metric | September 22 capture | September 23 capture |
| --- | ---: | ---: |
| Settled paper orders | 65 | 67 |
| Wins | 35 | 37 |
| Purchase cost, excluding fees | $959.80 | $989.46 |
| Modeled rounded fees | $33.95 | $35.32 |
| Net profit | −$42.75 | **+$12.22** |
| ROI after fees | −4.45% | **+1.24%** |
| Unresolved orders | 3 | 1 |

The full pilot's target-day bootstrap 95% ROI interval is **−24.40% to +31.72%** across 14 target days. The existing same-contract one-cent sensitivity is **−$12.30**. This is the original simplified attribution sensitivity, not the archive evaluator's separately recomputed-fee stress. Positive cumulative paper P&L alone is insufficient evidence of a reliable edge.

## Newly resolved orders

Both orders entered on **September 21** for **September 22** targets. They share one target day and must not be treated as two independent validation days.

| Contract | Side | Contracts | Price | Rounded fee | Net |
| --- | --- | ---: | ---: | ---: | ---: |
| KXHIGHTBOS-26SEP22-B62.5 | YES | 42 | $0.35 | $0.67 | +$26.63 |
| KXHIGHDEN-26SEP22-B83.5 | YES | 44 | $0.34 | $0.70 | +$28.34 |
| **Increment** | | | | **$1.37** | **+$54.97** |

Historical selected NO remains +$63.12 on 21 orders; YES is now −$50.90 on 46. These exploratory side slices are not promoted. The frozen NO shadow requires entry dates strictly after September 21 and still has **zero prospective selections and zero settlements**. Today's pipeline did not reach new selection, and the newly resolved orders were YES entries made on September 21, outside the shadow's eligible window.

The one unresolved paper order is the September 22 Boston YES intent for September 23, `KXHIGHTBOS-26SEP23-B64.5`: 62 contracts at $0.24. This is an existing intent, not a new September 23 order.

## Why a winning day still tripped the weekly loss breaker

The Rust risk calculation uses settled paper orders with target dates **September 16–22**, inclusive. That window contains 32 orders. Its existing unrounded per-contract fee approximation produces **−$65.952922**, matching the logged **−$65.95**. Applying the audit's rounded per-order modeled fees to the same orders gives **−$66.12** on $472.58 principal, including $17.54 fees. Both values breach the existing −$50 limit; the accounting difference does not change the decision.

Five September 15 target orders worth **+$73.32** after rounded fees aged out of the window. The new September 22 winners added **+$54.97**, so the rolling result worsened even though cumulative P&L improved. The breaker evaluates a rolling window; it is not a permanently latched manual pause. No limit was overridden.

The workflow had empty live and credential settings and recorded no new paper or live intents. The enforced admission scorer in this PR independently returns **NO-GO**, exit 1, because the 100-settled-order requirement is unmet. A positive sign now satisfies only the profitability part of that combined criterion. No live orders are recorded.

## Capture integrity and preserved local edits

Upstream adds 150 rows: 90 Kalshi rows forming 15 complete September 24 ladders and 60 Polymarket rows. It removes no rows and fills 138 previously null outcomes: 90 Kalshi and 48 Polymarket results for September 22. Previously known outcomes remain unchanged. Tiny numeric reserialization changes affect 136 fields in 94 older rows, with maximum absolute difference approximately `3.55e-15`. No invalid quotes, duplicate identities, or capture-order reversals were found. Canonical captures now total 9,790; the ledger remains unchanged at 6,738 rows.

Separately, uncommitted local capture/dashboard edits were preserved in stash commit `3c5abd721b5a49c4c7165735101d5befa9c98172` before merging upstream. That local capture rewrote all 9,640 prior rows, removing 14 fields per row and 74,746 previously non-null historical values. It must not replace canonical captures. Its 48 outcome updates are already present upstream.

The local file's 63 appended Polymarket observations are additionally preserved under ignored `data/raw/pre-alpha-update-20260923-local-captures.jsonl`, SHA-256 `a9c4e2b8fe0df5cf4f6f961152332d25755dbfb4aeecddb43da944d9aabeece9`. Fifty-nine overlap upstream, with some quote and forecast differences and insufficient timestamp precision to establish interchangeability. Four are local-only: London `4782084` and Miami `4834104` are temperature observations worth retaining separately; Seoul `4027993` is monthly precipitation misparsed with a December 8 target, and Moscow `4868542` is a non-weather question misclassified as precipitation. These observations remain outside the frozen canonical score. Both snapshots also contain non-temperature Boston/Miami rain title/target-date mismatches, outside Kalshi shape scoring.

## Preventing another obsolete local rewrite

The local 09:00 cron entry invoked `target/release/capture_prices` and `weather_dashboard` directly. Their modification times were July 6 and July 5, respectively. The September 23 09:00 capture log matches the preserved rewrite: zero Kalshi, 63 new Polymarket observations, 9,703 total rows, followed by a dashboard write. This strongly identifies the writer; the precise source commit compiled into those binaries is unknown.

`scripts/daily_capture.sh` now locates Cargo under a minimal cron/launchd PATH, builds both current release binaries, and refuses to run either binary if the build fails. A transient capture runtime failure still permits rendering from the last-good data. Four isolated stub tests verify these behaviors, and the actual release build of both binaries succeeded. Neither capture nor dashboard was executed during this repair.

The one capture-only cron entry now calls this guarded wrapper. Its 09:00 schedule and every other cron entry were preserved, and the installed table was read back and compared. The previous table is backed up locally, with restricted permissions, at `data/raw/local-capture-crontab-20260923.before.txt`, SHA-256 `c2deb5d63189d0bfff03376caab2f2c956f0023d30d4054abbda6760a05969e6`. This fixes a local data-writing path; it does not establish alpha, change trading settings, or deploy the draft PR to GitHub's scheduled runner.

## Reproduction and deployment scope

The [machine-readable output](2026-09-23-forward-update.json) reproduces byte for byte from the capture and ledger in upstream commit `ffdb792` using unchanged `scripts/pilot_alpha_audit.py`.

- Capture SHA-256: `2aed2bf5d7250a6f4a17d315984b2385043edc4ccba664014d2cd4037edc437e`
- Ledger SHA-256: `cad692b0388df8bc36bac343fa1752a563706a1de4eb05522186929f9b041566`

Run `python3 scripts/pilot_alpha_audit.py` and `python3 scripts/go_live_gate.py --enforce --json` against those inputs. The earlier [September 22 accounting](2026-09-22-forward-update.md), [failed frozen archive experiment](2026-09-22-archive-training-extension.md), and [next-day basket screen](2026-09-22-next-day-basket-audit.md) remain dated observations. This update does not revise their inputs or results.

The scheduled run still used `main` before this draft PR's admission and provenance changes. Today's loss stop is the existing weekly breaker, not deployment of the PR. Upstream data were merged into the draft branch. GitHub workflow settings and trading settings were not changed; the local capture-only cron correction is described above.
