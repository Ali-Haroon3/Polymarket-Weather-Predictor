# Engineering check: prospective source and quote evidence capture

**The new collector completed one real public-data run: 122 successful responses across the fixed 15-city universe.** This closes a data-preservation gap for future research. It does not establish source-publication latency, a trading opportunity or alpha.

The [collection protocol](2026-09-25-source-collection-protocol.md), collector and tests were committed and pushed as `d3837dfc95e3deccfb5d9578cacd62a47dada9a9` before the requests began. The engineering target is **September 24, 2026**, which is explicitly excluded from both prospective windows. No temperatures, buffers, strategy parameters or profit thresholds were selected from this run.

## Observed engineering results

The run started **September 25 at 00:17:21.748443 UTC** and finished at **00:18:24.548673 UTC**, taking 62.80 seconds. Every HTTP response was 200, complete at the transport layer and valid JSON. All 15 event responses contained six distinct binary market identities for their requested series/date. This identity check is not a mathematical partition proof or a guarantee of executable orders.

| Response type | Count |
| --- | ---: |
| Daily climate report response | 1 |
| Weekly hourly observation response | 1 |
| Event market lists with raw rules and lifecycle fields | 15 |
| Series metadata including fee fields | 15 |
| Order books | 90 |
| **Total** | **122** |

The minimum interval between client dispatches was 0.500122 seconds. The longest recorded request attempt took 0.915203 seconds. The last book was observed 62.343308 seconds after the daily response, within the protocol's 120-second pairing limit. These timestamps bound what this collector observed; they do not identify when the source first published information or when a particular quote first became available. Sequential requests are not simultaneous fills.

All saved response byte lengths and SHA-256 hashes verify, as do the copied protocol and collector hashes. Canonical capture and pilot-ledger hashes remain unchanged from the [September 24 audit](2026-09-24-forward-update.md). No trading setting, scheduled job, order or research-model parameter was changed.

Independent structural review also matched all 15 station/date/rule identities and fee-series responses. The daily response contained 39 `no_report` records with null data: transport success does not mean an official daily value was available. Order-book bodies do not echo their ticker, so their identity is retained through the exact request URL and manifest pairing. No source/quote availability rate or profit was computed from this engineering sample.

## Implementation verification

Nine focused, offline tests pass. They exercise the fixed request universe and rate limit, checkpoint persistence, exact byte preservation, protocol copying, overwrite refusal, missing protocol rejection, HTTP errors, no-response timestamps, invalid/paginated event identities, partial valid JSON, truncated and oversized bodies, deadlines, and worker cleanup after timeout or interruption. CLI help and whitespace checks also pass. Existing strategy tests were not rerun or relabeled as new validation.

The collector uses a separate anonymous HTTP worker with no inherited environment, proxy configuration, cookies or trading credentials. Request timeouts, the overall request deadline, a body-size limit and a fixed endpoint allowlist bound the collection. A partial response remains explicitly incomplete even if its saved prefix parses as JSON. `complete` describes request coverage and parsing only; subsequent analysis must separately validate source semantics, station/date/unit matching, lifecycle state, fees and available depth.

There is **no installed collection schedule**. The future dates and slots in the protocol define eligibility, not a claim that observations will automatically be gathered. No candidate trading strategy has been frozen for the reserved window. Source values or quotes inspected while selecting a candidate cannot later serve as independent validation of that candidate.

## Portable evidence

The [engineering evidence bundle](2026-09-25-source-engineering-evidence.json.gz) preserves the manifest, exact protocol bytes and all 122 exact response bodies. Binary-safe base64 avoids altering error bodies or whitespace. Extract `protocol_base64` and entries of `raw_response_bodies_base64`; verify each decoded response against its manifest record before use.

- Uncompressed bundle SHA-256: `cdc77bef1ce57629f192a1e4b1979d43b11058f54de165d2b379b0182c45e2cb`
- Collector SHA-256: `652fb66e90caf065a3324e7d0d83ddb23a917c5e0859bbadf1704eb1a46a3ab6`
- Protocol SHA-256: `57c61d4c6495251e8b09f00897975893c4727dd25f33d16d3162e837fd2cd98c`

The original local run is preserved under ignored `data/raw/source-engineering-20260925T001718Z`. Its directory cannot be reused by the collector. Inspecting this engineering evidence does not consume the protocol's future validation sample, but it also cannot establish performance on that future sample.
