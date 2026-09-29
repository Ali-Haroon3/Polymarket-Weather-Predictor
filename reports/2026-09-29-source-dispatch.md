# Dispatch research jobs earlier within the original observation windows

**Five of the first eleven scheduled jobs were created too late to collect.** The proposed repair launches each job ten minutes before its nominal checkpoint, giving dispatch and initialization more time while retaining the exact frozen observation windows. The repair is prepared in the research branch and is not deployed at this report.

The [September 29 artifact audit](2026-09-29-hosted-source-study.md) preserves all eleven original runs. Their five skipped creation delays relative to the preceding nominal slot were 23m11s, 17m27s, 24m49s, 15m08s and 18m42s. Each correctly returned `creation_off_schedule` without source requests. GitHub [documents that scheduled runs may be delayed or dropped](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule); no cron adjustment guarantees timely delivery.

## Proposed operational change

The five existing calendar expressions dispatch at minute **05** instead of **15**, once per original checkpoint. Nominal slots remain **13:15, 17:15 and 21:15 UTC on target date D**, followed by **01:15, 05:15 and 09:15 UTC on D+1**. All actual timestamps must still fall inside their original ±15-minute windows. No extra invocation, retry, manual catch-up, shifted target date or wider tolerance is introduced.

An immediate :05 start is already within the frozen :00–:30 window; a delayed job must still finish initialization and pass the pre-request check by :30. From a scheduled :05 dispatch, that allows up to 25 minutes for dispatch/setup rather than 15. The workflow avoids a top-of-hour trigger. The Python coordinator, inventory selector, evaluator, protocol and collector bytes are unchanged.

All 168 dispatches map one-to-one to the original slots, preserving the development/reserved split. The calendar's first possible dispatch becomes September 26 at 13:05, and its last October 24 at 09:05; these remain within the original inventory-query bounds of September 26 13:00 through October 24 09:30. Those historical calendar entries are not rerun. The unchanged explicit year guard prevents source requests outside 2026.

This is an operational response to observed delivery failures, prepared before the October 10 analysis freeze. It does not select source values or quotes. Earlier successful starts can nevertheless observe a different source vintage within the allowed window. Preserve deployment SHA/time and distinguish dispatch regimes when interpreting observations; improved collection coverage is not itself evidence of alpha.

## Preserved failures and deployment boundary

The current five late runs and ten earlier missed slots remain unknown in the original denominator. Subtracting ten minutes from historical timestamps is not evidence of successful collection. In particular, the slowest coordinator would have only about 4.7 seconds left for remaining initialization under an identical-delay hypothetical; future platform delays may differ.

All analyzed runs through September 29 at 21:15 used the original schedule. The new policy becomes operative only when this workflow revision reaches the default branch. Record that merge revision and timestamp and verify subsequent run metadata/artifacts before claiming deployment or improved delivery. If old/new cron events both arrive during transition, retain both: the original earliest-invocation rule selects the primary even if it failed. A later successful duplicate cannot replace it.

Reserved bodies remain uninspected until the original release requirements are satisfied. No paper-selection rule, live admission setting, source endpoint, request limit or trading credential changes.

## Validation

The calendar test enumerates every dispatch and verifies its unchanged nominal slot and target. New fixture coverage checks an immediate :05 start, a 23-minute dispatch delay, the inclusive :30 cutoff, rejection one microsecond beyond it, and initialization that crosses the cutoff without sending source requests. A separate inventory regression checks that a failed :05 invocation on one workflow revision remains primary ahead of a successful :15 duplicate on another, independent of inventory order.

Focused scheduler/inventory suites pass **42 tests**. They also retain year/date, rerun, timing/chronology, frozen-input, duplicate and partial-failure coverage. All fixtures are offline and no workflow was dispatched. Actionlint 1.7.12 passes with no diagnostics; its official Darwin/arm64 archive matches published SHA-256 `aba9ced2dee8d27fecca3dc7feb1a7f9a52caefa1eb46f3271ea66b6e0e6953f`. ShellCheck was unavailable and disabled; that check is not claimed.

The original protocol SHA-256 remains `57c61d4c6495251e8b09f00897975893c4727dd25f33d16d3162e837fd2cd98c`, and collector SHA-256 remains `652fb66e90caf065a3324e7d0d83ddb23a917c5e0859bbadf1704eb1a46a3ab6`. These checks establish local correctness of the repair, not hosted delivery or profitability.
