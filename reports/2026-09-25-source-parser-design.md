# Offline eligibility checks for preserved source observations

These helpers prepare evaluation of the [fixed prospective study](2026-09-25-source-collection-protocol.md). They do not select a trading strategy, produce a study-wide availability rate, or estimate profit. They were developed from synthetic cases and the already-inspected, excluded September 23/24 evidence. No development or reserved observation was retrieved or inspected for this change.

The original protocol and collector remain byte-identical. The [scheduled collector](2026-09-25-source-schedule.md) remains proposed in a draft pull request, with no qualifying collection yet. This document records current parser support and limitations; it is not a claim that the complete analysis or a trading candidate has been frozen.

## Evidence integrity and causal pairing

`scripts/weather_source_evidence.py` reads the original manifest, copied protocol, and response bytes. It requires the frozen collector/protocol hashes, unique request IDs, exact target-specific URLs, successful complete responses, matching lengths/hashes, and ordered timezone-aware timestamps. Missing or unreadable files, duplicate JSON keys, unsupported identities, changed bytes, partial bodies, redirects and inconsistent metadata raise `EvidenceError`; an evaluator must retain these as unknown observations.

JSON decimal numbers are preserved as `Decimal`. Binary floating-point decoding can otherwise turn an unsupported fractional daily value such as `64.0000000000000001` into an apparently official whole-degree `64`. A regression test reads that exact raw representation through both the evidence reader and daily parser and requires unknown eligibility.

A partial capture may still contain an individually verified response; the manifest's `complete` flag alone neither validates nor invalidates a particular source/book pair. Every pair must come from the same saved manifest, with source receipt strictly before book dispatch, at most 120 seconds from source receipt to book receipt, and market metadata received before book dispatch. Metadata/book receipt at or after the explicit close is excluded. Saved cache headers travel with the result. These checks do not establish first publication, atomic execution, fill probability or revision immunity.

Reserved response bodies remain locked until October 24 at 09:30 UTC and an explicit caller assertion that final-slot attempts have been reconciled as terminal. The reader does not itself obtain workflow history or prove that assertion. It reads only operational manifest/protocol information before that gate. No release flag may substitute for the required inventory reconciliation.

## Daily report support

`scripts/weather_daily_observation.py` requires an exact root date, exact unique CLI station, internally consistent station/status counts, matching official report date and station ID, `status="official"`, and boolean `isOfficial=true`. It never substitutes averages or hourly values. Recognized `no_report` with null data means an observed missing report at that checkpoint; absent rows, malformed data and unsupported preliminary/revised semantics mean unknown.

Official numeric interpretation requires separately verified Fahrenheit evidence. The saved daily JSON contains no unit field. The earlier [source reconciliation](2026-09-24-weather-source-reconciliation.md) documents public-client unit evidence, but a caller must establish the preserved evidence basis rather than set the flag because a city or endpoint looks familiar. Explicit conflicting unit fields fail eligibility.

The supported daily maximum is an integral JSON number from −150 through +150 °F, without rounding. This is a conservative parser support range, not a learned threshold, an API sentinel definition, or a claim that other values are impossible. Numeric strings, nonfinite values, fractions and values outside that range remain unknown. The observed data do not document numeric missing-value sentinels. Issue-time text is preserved but never converted into a known publication timestamp; all inspected official issue-time fields were empty.

## Market rules, clocks and displayed depth

`scripts/weather_market_observation.py` requires all six exact market identities and a disjoint, exhaustive partition of integer temperatures. Primary rules must match the target date, explicit CLI station, Fahrenheit unit, Weather Company source and numeric condition. Known city-series mappings include Chicago Midway and Houston Hobby. The supported secondary-rule SHA-256 is `c1f11eaf372267f2e69ca8ba131916928052ec5e92138e45c16e197f2d4bb507`, shared by all 90 excluded engineering markets; changed secondary text remains unsupported. This preserves the relevance of correction and exceptional-settlement clauses instead of silently accepting arbitrary nonempty text.

The excluded market responses contain a material clock discrepancy: their `close_time` differs from the text's 11:59 PM local last-trading cutoff. Eligibility requires observation before both bounds. A book between conflicting bounds remains unknown; the parser does not choose the later time to manufacture availability. Explicit non-active lifecycle evidence is retained separately. A market observed active is not proof that it continuously accepted orders between requests.

Books contain no ticker. Their identity must come from the verified request URL and manifest ticker. A YES bid implies a NO offer; a NO bid implies a YES offer. Both sides are checked for malformed, duplicated or crossed prices. Only interior prices count. Fractional displayed depth may accumulate across levels to at least one whole contract; the returned price is the worst included offer, a limit-price bound rather than an average fill. Fewer than one contract in the saved depth is observed insufficient depth, with deeper and later liquidity still unknown. Decimal arithmetic prevents fractional depth from rounding up to a whole contract.

Saved series metadata identifies `quadratic` with multiplier `1`, but it does not specify the coefficient and rounding rule. The helper therefore reports fee metadata support with `payout_comparison_available=false`. It does not silently reuse the ordinary paper pilot's fee approximation. A complete fee-inclusive comparison requires separately preserved authoritative fee-rule evidence.

## Validation

All 46 focused offline tests pass: 16 daily-report checks, 20 market/book checks and 10 evidence-reader checks. They include exact raw decimal preservation, unsupported source states, contradictory rules/clocks, fractional depth, corruption and missing files, cross-invocation pairing, and the reserved-body gate. The excluded engineering bundle passes identity/byte verification for all 122 responses and schema checks for all 15 events. Those engineering results are not prospective availability observations or profit estimates.

Run `PYTHONPATH=tests python3 -m unittest test_weather_daily_observation test_weather_market_observation test_weather_source_evidence`. The tests issue no network requests. Original collector/protocol hashes and canonical capture/ledger files are unchanged.

## Remaining integration and interpretation

Before producing any study result, an evaluator still must reconcile the complete workflow/artifact inventory, retain failed and missing slots in the fixed denominator, select the earliest-started invocation even if it failed, and treat unresolved timing ties as unknown. It must then combine these helpers into station/checkpoint results, distinguishing overlap, observed absence and unknown evidence, and summarize by target date without treating repeated checkpoints as independent outcomes. The helper outputs alone cannot establish any of those study-wide claims.

The parsing and full eligibility analysis must be frozen before the protocol deadline. Any trading candidate requires its own earlier definition and untouched reserved evidence under the original rules. No candidate, fee-inclusive opportunity, availability rate or validated alpha is claimed here.
