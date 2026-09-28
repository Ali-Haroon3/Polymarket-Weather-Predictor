# Source research activated; first exploratory snapshot finds no official report

**The scheduled collector is now deployed, but no scheduled observation had run at this check.** [PR #50](https://github.com/Ali-Haroon3/Polymarket-Weather-Predictor/pull/50) merged at September 28, 04:33:29 UTC as `dccd89dd4d0f1e4b3122c0dab53791fb6a6ec327`. GitHub reports workflow `367208409`, `weather-source-research`, active on the default branch. The merged protocol and collector hashes match their frozen values. Registration does not prove schedule delivery, successful collection or artifact upload.

The first remaining nominal slot is September 28 at **05:15 UTC**, for the September 27 target. The earlier [inventory check](2026-09-27-forward-update.md) recorded ten missed development slots; they remain missing. No schedule, tolerance, target window or analysis definition was changed after deployment.

## Separate off-schedule development observation

While the merged code review was running, one anonymous, on-demand development capture checked whether an official daily report was presently available alongside market evidence. It ran **04:39:54.310521–04:40:55.738957 UTC**, September 28, targeting September 27. The original frozen collector issued its full, unfiltered 122-request sequence once, without retries. All responses returned HTTP 200 and complete, valid JSON; byte lengths, hashes, identities and chronology independently verify.

This invocation began outside every fixed slot's ±15-minute tolerance and was not a scheduled workflow run. It is **excluded from the primary scheduled sample and from independent validation**. It does not replace a missed slot or repair study coverage. Any candidate informed by this observation must treat it as development evidence. It is also not the earlier engineering sample, which used a target before September 26.

The daily response arrived at **04:39:54.726959 UTC**. All 39 source rows reported `no_report` with null data, including all 15 study stations. Thus **no official-report/quote overlap was observed** at this receipt. That statement does not establish source publication time, absence throughout the day, or the impossibility of an opportunity at another checkpoint.

Using the unchanged snapshot eligibility logic:

| Snapshot classification | Cities | Explanation |
|---|---:|---|
| Observed overlap | 0 | No official daily report was available. |
| Observed absence with complete components | 9 | Chicago, Austin, Denver, Los Angeles, Dallas, Seattle, Phoenix, Las Vegas and Houston had eligible market evidence but explicit report absence. |
| Unknown | 6 | New York City, Miami, Philadelphia, Atlanta, Boston and Washington DC had conflicting market close bounds. |

For the six unknown cities, the 36 books were observed after the textual cutoff at **03:59 UTC** but before the API `close_time` at **05:00 UTC**. The analysis preserves that conflict as unknown; it does not choose the later bound to manufacture tradability. Across all 90 contracts, market partitions and station/date identities validated, both book sides parsed, and source/book receipt lags were at most **61.004536 seconds**. Those technical checks do not turn missing official reports into a weather signal.

Hourly data was preserved by the collector but was not substituted for official daily maxima. No fee-inclusive payout comparison, hypothetical return, order or alpha claim was produced. The primary development denominator remains 84 slots / 1,260 station-checkpoints; these 15 exploratory observations are not added to it. Reserved-period bodies were not retrieved or inspected.

## Preserved evidence and reproduction

The [portable evidence bundle](2026-09-28-offschedule-source-evidence.json.gz) preserves the exact manifest bytes, copied frozen protocol, all 122 raw response bodies, snapshot output and selected deployment-status fields. It is 307,382 compressed bytes, 4,629,951 uncompressed bytes; uncompressed SHA-256:

`f3bd0d670ef24e2a69768f577cd4e089f8e84bb5dd82d57851f67b88242ac833`

Manifest SHA-256: `80847cd7deaca3c7b0df06478452920eb7da7e0e57a224afa38eb8821e64f402`.

Daily body: 9,489 bytes; SHA-256 `0e619851b1e101b617b02f49d47250181a61d53cb82a827c35d04b201c6ebd96`.

Original files remain under ignored `data/raw/source-development-offschedule-20260928T0440Z/`. The folder label is an approximate identifier; the manifest contains actual timestamps. Offline snapshot analysis used implementation `dccd89d` and the already preserved Fahrenheit authority. Independent review verified every response and the 9-absence / 6-unknown result. Extracting the bundle into a temporary directory reproduces that snapshot output without refreshing endpoints. No scheduled-run inventory was fabricated for this local invocation.

Canonical paper inputs and frozen trading/admission code are unchanged. The latest paper score remains +$28.36 across 71 settlements, with live admission NO-GO. This observation advances development evidence, not validation of alpha. Actual hosted collection and a prospective sample are still required.
