# September 23 weather sources: settlement agrees, hourly samples are not daily maxima

**Boston's later official daily report agrees with its venue settlement. All 15 matched daily reports also agree with the 90 September 23 contract outcomes in the canonical captures.** The earlier hourly snapshot is a different measurement: even after ordinary integer rounding, its available settled hourly maximum differs from the later daily maximum at 11 of the 15 stations. This is a post-settlement source-consistency diagnostic, not an alpha strategy or a performance backtest.

The comparison uses two preserved snapshots of public responses. No model fitting, buffer selection, order placement, or changes to canonical outcomes were needed. The separate public client asset described below establishes the daily display's Fahrenheit semantics.

## Boston before and after

The first requests were received on **September 24 at 02:15:51.644–02:15:51.757 UTC**. The daily response for September 23 contained no report for Boston: `results[2].status=no_report` and `data=null`. Its station metadata identified `cliId=BOS`, `icao=KBOS`, and `timezone=America/New_York`. The reported `avgTemp=57.3` was an average, not a daily maximum.

The saved hourly response had `fetchedAt=2026-09-24T02:14:50.612Z`. Boston's September 23 records covered local hours 00–22. Hours 00–21 were marked `settled`; hour 22 was `pending`. The largest eligible hourly value was **64°F**, at local hours 11, 12, and 13. These are portal observation statuses, not venue settlement decisions.

At that time, `KXHIGHTBOS-26SEP23-B64.5` was `active`, with empty `result` and `expiration_value`. Its saved book contained a YES bid of $0.99 for 5,920.80 contracts, equivalent to a NO ask of $0.01. The NO bid side was empty: there was **no executable YES ask**. The market metadata's $1.00 YES ask had zero size and was not available liquidity.

The later daily and market responses arrived on **September 24 at 23:55:09.403 and 23:55:09.471 UTC**, respectively. All 39 daily reports were then `official`. Boston's record had `stationId=BOS`, `reportDate=2026-09-23`, `maxTemp=64`, and `isOfficial=true`. The venue market was `finalized`, with `result=yes`, `expiration_value=64.00`, `settlement_value_dollars=1.0000`, and `settlement_ts=2026-09-24T11:21:18.096846Z`. This agrees with the contract's inclusive 64–65°F condition and its canonical outcome of 1.

## All 15 station comparisons

Each of the 90 original contract rules explicitly names one CLI identifier, September 23, 2026, Fahrenheit, and The Weather Company. Within each six-contract event the identifier is identical. Removing the `CLI` product prefix gives a unique match to both the daily record's `station.cliId` and `data.stationId`; hourly records match the corresponding explicit ICAO identifier. Chicago therefore means **Midway**, and Houston means **Hobby**. No city-name fallback is used.

Only hourly records present in the saved response, with `localDate=2026-09-23`, `status=settled`, and valid `reportTimeUTC` no later than its **02:15:51.735232 UTC receipt**, enter the maximum. UTC timestamps also agree with the stations' local dates and hours. All 327 records pass those timestamp checks; 312 are eligible and 15 pending records are excluded.

In the table, H is the eligible hourly maximum and D the later official daily maximum, in Fahrenheit. Differences are D minus H. `round(H)` means nearest integer, with exact half-degree ties rounded upward; no ties occurred.

| City / CLI suffix / ICAO | Eligible local hours | H | D | D − H | D − round(H) |
| --- | --- | ---: | ---: | ---: | ---: |
| NYC / NYC / KNYC | 00–21 | 66 | 67 | +1 | +1 |
| Chicago Midway / MDW / KMDW | 00–20 | 64.9 | 65 | +0.1 | 0 |
| Austin / AUS / KAUS | 00–20 | 95 | 97 | +2 | +2 |
| Denver / DEN / KDEN | 00–19 | 72 | 72 | 0 | 0 |
| Los Angeles / LAX / KLAX | 00–18 | 77 | 78 | +1 | +1 |
| Miami / MIA / KMIA | 00–21 | 82 | 83 | +1 | +1 |
| Philadelphia / PHL / KPHL | 00–21 | 69.1 | 70 | +0.9 | +1 |
| Dallas / DFW / KDFW | 00–20 | 97 | 98 | +1 | +1 |
| Seattle / SEA / KSEA | 00–18 | 66 | 67 | +1 | +1 |
| Atlanta / ATL / KATL | 00–21 | 80.1 | 82 | +1.9 | +2 |
| Boston / BOS / KBOS | 00–21 | 64 | 64 | 0 | 0 |
| Phoenix / PHX / KPHX | 00–18 | 98.1 | 99 | +0.9 | +1 |
| Las Vegas / LAS / KLAS | 00–18 | 91.9 | 93 | +1.1 | +1 |
| Washington DC / DCA / KDCA | 00–21 | 66 | 66 | 0 | 0 |
| Houston Hobby / HOU / KHOU | 00–20 | 93.9 | 96 | +2.1 | +2 |

Four stations agree after rounding; eight differ by 1°F and three by 2°F. Chicago's 0.1°F raw difference disappears after rounding and is not treated as a contradiction.

**None of the 15 stations had a complete day of settled hourly observations.** Eligible records number 19–22 per station and end at 01:00 UTC; the next 02:00 UTC observation was pending at every station. The snapshot was approximately 22:15 Eastern, 21:15 Central, 20:15 Mountain, and 19:15 Pacific/Phoenix local time. Missing later observations, peaks between hourly samples, and subsequent corrections cannot be separated by this comparison. It provides no basis for selecting a temperature buffer or asserting that a sampled hourly maximum is an immutable daily result.

Applying the later daily value to each explicitly stated contract condition produces exactly one winning bucket per event and agrees with **all 90 stored canonical outcomes**: no missing mappings, missing outcomes, or mismatches. Between conditions are inclusive; less-than and greater-than conditions are strict. This check uses the original six-leg event metadata in the [September 22 next-day basket evidence](2026-09-22-next-day-basket-evidence.json.gz), not a newly selected city or contract sample. The [original basket report](2026-09-22-next-day-basket-audit.md) remains an independent, dated price screen.

## Units, timing, and limits

The daily JSON itself contains no explicit unit field. The [public portal client](https://weather.com/vc-ap-7f3f87/_next/static/chunks/072jb9-wdzll8.js?dpl=dpl_5vSuvkcPQYd5LxkiBDkjVPLhYcJh) states: **“Daily values are official Fahrenheit climate-report figures”**. Its explanation limits the Fahrenheit/Celsius toggle to hourly observations and identifies NWS/NOAA daily climate reports, with CF6 as backup. Thus the daily Fahrenheit interpretation has a public source, but is not self-describing within the daily JSON. The saved hourly fields already distinguish `tempF` and `tempC`; their displayed precision does not establish sensor precision.

All 15 matched daily `issueTime` fields are empty. Receipt time bounds when these responses were observed; it does not establish the first publication time of an official daily report. The venue settlement timestamp is a separate event. Neither the difference between receipts nor the difference from settlement measures an exploitable publication delay. Station timezones and hourly coverage also do not independently specify the official climate-report measurement window.

Requests were not atomic. The contract's correction provisions remain applicable, and even the portal's final-status label does not guarantee immunity from revisions. Its exceptional last-fair-price settlement provision also remains distinct from ordinary binary settlement. This comparison does not turn the earlier hourly agreement in Boston into a prospective settlement guarantee, executable trade, or evidence of alpha.

## Reproducible evidence

The [source evidence bundle](2026-09-24-weather-source-evidence.json.gz) contains the four before responses, two after responses, and their manifests with request URLs, UTC receipt times, headers, byte lengths, and SHA-256 hashes. All six response lengths and hashes match. Its uncompressed SHA-256 is:

`775df54990efb4c4539419d74e51b9cc3d2de73fed4de7930358d1acfa18d523`

The existing [basket evidence bundle](2026-09-22-next-day-basket-evidence.json.gz) supplies the exact station, date, unit, and bucket rules. All 120 embedded response hashes match. Its uncompressed SHA-256 is:

`f393959e95f1cca7b54af9c830f19b5abacbc7737dcfa34ade82035f3417cfe4`

The separately retrieved public client asset is 726,416 bytes, SHA-256:

`79b37eb8ec037d8ce0299ad88d503e2735311856b3e3940b1ce1b21ecf4575f9`

Canonical comparison: repository HEAD `229e30ad82572a82193051a93d4105abcf3bb0b8`; `data/captures.jsonl` SHA-256 `fd3c379a68491dd59317c8bf7d1608d7bd750a643b80b8bc529be588a44d9a9c`. The matched rows retain their September 22 capture date, which describes the earlier quote capture, not the later outcome's publication time.

Extraction pointers are `snapshots[*].raw_response_bodies` in the source bundle; daily `results[*].station` and `.data`; hourly `stations[*].byDate["2026-09-23"]`; and venue `market`. Boston occupies daily `results[2]` and hourly `stations[2]` in these saved responses. The basket bundle's `report.requests` links each event URL to its body in `raw_response_bodies`.
