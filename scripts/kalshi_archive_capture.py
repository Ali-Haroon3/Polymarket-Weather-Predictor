#!/usr/bin/env python3
"""Reconstruct the frozen May 1–June 28, 2026 Kalshi sample from public archives.

One quote per market: the exact one-minute candle CLOSE at 15:00 UTC on the
calendar day before its target. No current book, trade-price fallback, or nearest
candle substitution. Archives lack historical depth and do not prove fills.

The full sample requires --download; --smoke fetches only NYC June 15. Full output
and raw response caches stay under ignored data/raw/ or a system temporary path,
never canonical captures. The NYC June 15 probe must be excluded from subsequent
training and validation because its outcome was inspected before preregistration.
Only public GETs are implemented; no credentials are read or orders placed.

Historical candles have no documented batch endpoint, so requests default to four
workers (at most twelve) and ten requests per second, with bounded retries and a request cap. Every successful
raw response is cached and hashed for reproducibility and restartability.
"""
import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from ladder_arbitrage_audit import full_partition

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
SAMPLE_START = dt.date(2026, 5, 1)
SAMPLE_END = dt.date(2026, 6, 28)
SERIES = {
    "KXHIGHNY": "NYC", "KXHIGHCHI": "Chicago", "KXHIGHAUS": "Austin",
    "KXHIGHDEN": "Denver", "KXHIGHLAX": "LA", "KXHIGHMIA": "Miami",
    "KXHIGHPHIL": "Philadelphia", "KXHIGHTDAL": "Dallas", "KXHIGHTSEA": "Seattle",
    "KXHIGHTATL": "Atlanta", "KXHIGHTBOS": "Boston", "KXHIGHTPHX": "Phoenix",
    "KXHIGHTLV": "Vegas", "KXHIGHTDC": "Washington", "KXHIGHTHOU": "Houston",
}
PROBE_EXCLUSION = {"city": "NYC", "target_date": "2026-06-15"}
UTC = dt.timezone.utc
REPO_ROOT = Path(__file__).resolve().parents[1]


class InvalidArchive(ValueError):
    """Data cannot support the frozen contemporaneous snapshot."""


def strict_json(text):
    def invalid_constant(value):
        raise InvalidArchive(f"nonfinite JSON constant: {value}")

    def unique_keys(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise InvalidArchive(f"duplicate JSON key: {key}")
            obj[key] = value
        return obj

    try:
        return json.loads(text, parse_constant=invalid_constant, object_pairs_hook=unique_keys)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise InvalidArchive("malformed JSON response") from exc


def utc_string(timestamp):
    return dt.datetime.fromtimestamp(timestamp, UTC).isoformat().replace("+00:00", "Z")


def timestamp(value, field):
    """Require an explicit timezone; never interpret an ambiguous local date."""
    if not isinstance(value, str):
        raise InvalidArchive(f"missing or invalid {field}")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InvalidArchive(f"invalid {field}") from exc
    if parsed.tzinfo is None:
        raise InvalidArchive(f"timezone missing in {field}")
    return parsed.timestamp()


def number(value, field):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise InvalidArchive(f"invalid {field}")
    try:
        result = float(value)
    except ValueError as exc:
        raise InvalidArchive(f"invalid {field}") from exc
    if not math.isfinite(result):
        raise InvalidArchive(f"nonfinite {field}")
    return result


def entry_timestamp(target):
    return int(dt.datetime.combine(target - dt.timedelta(days=1), dt.time(15), UTC).timestamp())


def event_ticker(series, target):
    if series not in SERIES:
        raise InvalidArchive("series outside frozen universe")
    months = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
    return f"{series}-{target.year % 100:02d}{months[target.month - 1]}{target.day:02d}"


def frozen_events(smoke=False):
    if smoke:
        return [("KXHIGHNY", dt.date(2026, 6, 15))]
    return [(series, SAMPLE_START + dt.timedelta(days=day))
            for day in range((SAMPLE_END - SAMPLE_START).days + 1) for series in SERIES]


def guarded_directory(path):
    resolved = Path(path).resolve()
    roots = [(REPO_ROOT / "data/raw").resolve(), Path("/tmp").resolve(),
             Path(tempfile.gettempdir()).resolve()]
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise ValueError("archive output and cache must remain under data/raw/ or a temporary directory")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def atomic_text(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


class PublicArchiveClient:
    def __init__(self, cache_dir, max_requests=8000, interval=0.10, retries=4,
                 opener=None, sleeper=None, clock=None):
        if max_requests < 1 or interval < 0.10 or not 0 <= retries <= 6:
            raise ValueError("invalid request/rate/retry bounds")
        self.cache_dir = guarded_directory(cache_dir)
        self.max_requests, self.interval, self.retries = max_requests, interval, retries
        self.opener = opener or urllib.request.urlopen
        self.sleep, self.clock = sleeper or time.sleep, clock or time.monotonic
        self.last_request, self.requests, self.cache_hits = None, 0, 0
        self.responses = {}
        self.lock, self.blocked_until = threading.Lock(), 0.0

    def request_slot(self):
        # One shared limiter covers all workers, including retries. Holding this
        # lock during the bounded delay prevents a simultaneous wake-up burst.
        with self.lock:
            if self.requests >= self.max_requests:
                raise RuntimeError("public request cap reached; rerun using the response cache")
            allowed = max(self.blocked_until, (self.last_request + self.interval)
                          if self.last_request is not None else 0.0)
            self.sleep(max(0.0, allowed - self.clock()))
            self.last_request = self.clock()
            self.requests += 1

    def backoff(self, delay):
        with self.lock:
            self.blocked_until = max(self.blocked_until, self.clock() + delay)

    def get(self, path, params=None):
        if not path.startswith("/historical/") or "?" in path:
            raise ValueError("only public historical GET paths are supported")
        url = BASE_URL + path
        if params:
            url += "?" + urllib.parse.urlencode(sorted(params.items()))
        cache_file = self.cache_dir / (hashlib.sha256(url.encode()).hexdigest() + ".json")
        if cache_file.exists():
            record = strict_json(cache_file.read_text(encoding="utf-8"))
            if (not isinstance(record, dict) or record.get("url") != url
                    or not isinstance(record.get("body"), str)
                    or record.get("status") != 200
                    or hashlib.sha256(record["body"].encode()).hexdigest() != record.get("sha256")):
                raise InvalidArchive(f"invalid cached response: {cache_file}")
            timestamp(record.get("fetched_at"), "cached fetched_at")
            payload = strict_json(record["body"])
            with self.lock:
                self.cache_hits += 1
        else:
            for attempt in range(self.retries + 1):
                self.request_slot()
                request = urllib.request.Request(url, headers={"Accept": "application/json",
                                                               "User-Agent": "weather-archive-research/1"})
                try:
                    with self.opener(request, timeout=25) as response:
                        if response.status != 200:
                            raise InvalidArchive(f"unexpected HTTP status {response.status}")
                        raw = response.read(2_000_001)
                        if len(raw) > 2_000_000:
                            raise InvalidArchive("response exceeds 2MB bound")
                        body = raw.decode("utf-8")
                        payload = strict_json(body)
                        record = dict(url=url, status=200, body=body,
                                      fetched_at=utc_string(time.time()),
                                      sha256=hashlib.sha256(raw).hexdigest())
                    atomic_text(cache_file, json.dumps(record, sort_keys=True, allow_nan=False))
                    break
                except urllib.error.HTTPError as exc:
                    if (exc.code != 429 and not 500 <= exc.code <= 599) or attempt == self.retries:
                        raise
                    retry_after = exc.headers.get("Retry-After", "0")
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = 0.0
                    self.backoff(min(60.0, max(2.0 ** attempt, delay)))
                except (urllib.error.URLError, TimeoutError):
                    if attempt == self.retries:
                        raise
                    self.backoff(min(30.0, 2.0 ** attempt))
        if not isinstance(payload, dict):
            raise InvalidArchive("API response must be an object")
        provenance = {key: record[key] for key in ("url", "sha256", "fetched_at")}
        provenance["cache_file"] = str(cache_file)
        with self.lock:
            self.responses[url] = provenance
        return payload, provenance


def load_event(client, series, target):
    event = event_ticker(series, target)
    markets, records, cursors = [], [], set()
    params = {"event_ticker": event, "limit": 100}
    for _ in range(10):
        payload, provenance = client.get("/historical/markets", params)
        page, cursor = payload.get("markets"), payload.get("cursor")
        if not isinstance(page, list) or not isinstance(cursor, str):
            raise InvalidArchive("malformed event metadata page")
        if any(not isinstance(market, dict) for market in page):
            raise InvalidArchive("malformed market metadata")
        markets.extend(page)
        records.extend([provenance] * len(page))
        if not cursor:
            if not markets:
                raise InvalidArchive("event has no archived markets")
            return markets, records
        if cursor in cursors or not page:
            raise InvalidArchive("invalid/repeated event pagination cursor")
        cursors.add(cursor)
        params = dict(params, cursor=cursor)
    raise InvalidArchive("event metadata exceeds bounded pagination")


def shape_fields(market):
    strike = market.get("strike_type")
    def whole(field):
        result = number(market.get(field), field)
        if result != math.floor(result):
            raise InvalidArchive("temperature strike must be a whole degree")
        return result
    if strike in ("greater", "greater_or_equal"):
        return "temp_at_least", whole("floor_strike") + (strike == "greater"), None
    if strike in ("less", "less_or_equal"):
        field = "cap_strike" if market.get("cap_strike") is not None else "floor_strike"
        return "temp_at_most", whole(field) - (strike == "less"), None
    if strike == "between":
        lower, upper = whole("floor_strike"), whole("cap_strike")
        if lower > upper:
            raise InvalidArchive("reversed temperature bucket")
        return "temp_bucket", lower, upper
    raise InvalidArchive("unsupported temperature strike type")


def close_quote(candle, side):
    book = candle.get("yes_" + side)
    if not isinstance(book, dict) or "close" not in book:
        raise InvalidArchive(f"missing candle yes_{side}.close")
    raw = book["close"]
    if raw is None:
        return None
    price = number(raw, f"yes_{side}.close")
    if (side == "bid" and price == 0) or (side == "ask" and price == 1):
        return None
    if not 0 < price < 1:
        raise InvalidArchive(f"invalid candle yes_{side}.close")
    return price


def exact_candle(payload, ticker, entry_ts):
    if payload.get("ticker") != ticker or not isinstance(payload.get("candlesticks"), list):
        raise InvalidArchive("candle response ticker/schema mismatch")
    selected, seen = None, set()
    for candle in payload["candlesticks"]:
        if not isinstance(candle, dict):
            raise InvalidArchive("malformed candle")
        end = candle.get("end_period_ts")
        if isinstance(end, bool) or not isinstance(end, int) or end % 60 or end in seen:
            raise InvalidArchive("invalid/duplicate candle boundary")
        seen.add(end)
        if end == entry_ts:
            selected = candle
    if selected is None:
        raise InvalidArchive("exact 15:00 UTC candle missing")
    return selected


def reconstruct_event(client, series, target):
    markets, metadata_records = load_event(client, series, target)
    event, entry = event_ticker(series, target), entry_timestamp(target)
    rows = []
    for market, metadata in zip(markets, metadata_records):
        ticker = market.get("ticker")
        if (not isinstance(ticker, str) or not ticker.startswith(event + "-")
                or market.get("event_ticker") != event or market.get("market_type") != "binary"):
            raise InvalidArchive("market identity/type differs from requested event")
        created = timestamp(market.get("created_time"), "created_time")
        opened = timestamp(market.get("open_time"), "open_time")
        closed = timestamp(market.get("close_time"), "close_time")
        settled = timestamp(market.get("settlement_ts"), "settlement_ts")
        if not created <= opened <= entry < closed <= settled:
            raise InvalidArchive("market was not open at entry or lifecycle timestamps are invalid")
        if market.get("status") not in ("settled", "finalized") or market.get("result") not in ("yes", "no"):
            raise InvalidArchive("market has no final binary settlement")
        kind, threshold, upper = shape_fields(market)
        rows.append(dict(
            source="kalshi", city=SERIES[series], target_date=target.isoformat(),
            captured_at=utc_string(entry)[:10], captured_at_utc=utc_string(entry),
            entry_ts=entry, quote_close_ts=entry, market_id=ticker,
            market_title=market.get("title") or market.get("yes_sub_title") or ticker,
            market_type=kind, threshold=threshold, threshold_upper=upper, unit="F",
            outcome=float(market["result"] == "yes"), settlement_ts=int(math.ceil(settled)),
            settlement_time_utc=utc_string(settled), open_time=utc_string(opened),
            created_time=utc_string(created), close_time=utc_string(closed),
            event_ticker=event, archive_method="historical_1m_bid_ask_close",
            metadata_url=metadata["url"], metadata_response_sha256=metadata["sha256"],
            metadata_fetched_at=metadata["fetched_at"],
            rules_primary=market.get("rules_primary"), rules_secondary=market.get("rules_secondary"),
            model_estimate=None, forecast_high=None, forecast_sigma=None,
        ))
    ordered = full_partition(rows)
    if ordered is None:
        raise InvalidArchive("event is not a complete unique contiguous temperature partition")
    if sum(row["outcome"] for row in ordered) != 1:
        raise InvalidArchive("partition does not have exactly one winning settlement")
    candle_responses = []
    for row in ordered:
        path = "/historical/markets/" + urllib.parse.quote(row["market_id"], safe="") + "/candlesticks"
        payload, provenance = client.get(path, {"start_ts": entry - 60, "end_ts": entry,
                                              "period_interval": 1})
        candle_responses.append((payload, provenance))
    # Retain every leg's response even when one is missing. This exposes complete
    # coverage diagnostics without relaxing the all-legs requirement.
    for row, (payload, provenance) in zip(ordered, candle_responses):
        candle = exact_candle(payload, row["market_id"], entry)
        bid, ask = close_quote(candle, "bid"), close_quote(candle, "ask")
        if bid is None and ask is None:
            raise InvalidArchive("both quote sides missing at entry")
        if bid is not None and ask is not None and bid > ask:
            raise InvalidArchive("crossed quote at entry")
        # Canonical reference-price rule; no use of final market prices or last trades.
        reference = (bid + ask) / 2 if bid is not None and ask is not None else bid if bid is not None else ask
        row.update(best_bid=bid, best_ask=ask, entry_price=reference,
                   candle_url=provenance["url"], candle_response_sha256=provenance["sha256"],
                   candle_fetched_at=provenance["fetched_at"],
                   volume=None, open_interest=None, volume_24h=None, liquidity=None)
        for source, destination in (("volume", "volume"), ("open_interest", "open_interest")):
            value = candle.get(source)
            if value is not None:
                parsed = number(value, source)
                if parsed < 0:
                    raise InvalidArchive(f"negative candle {source}")
                row[destination] = parsed
    return ordered


def download_sample(client, events, workers=4):
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 12:
        raise ValueError("archive workers must be an integer between one and twelve")
    cutoff, cutoff_provenance = client.get("/historical/cutoff")
    timestamp(cutoff.get("market_settled_ts"), "market_settled_ts cutoff")
    rows, coverage = [], []
    def load_one(event_spec):
        series, target = event_spec
        event = event_ticker(series, target)
        try:
            captured = reconstruct_event(client, series, target)
            return captured, dict(event_ticker=event, city=SERIES[series], target_date=target.isoformat(),
                                  status="accepted", rows=len(captured))
        except InvalidArchive as exc:
            return [], dict(event_ticker=event, city=SERIES[series], target_date=target.isoformat(),
                            status="rejected", reason=str(exc))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        pending = [executor.submit(load_one, event) for event in events]
        # Transport errors abort the run rather than silently changing sample composition.
        try:
            for index, future in enumerate(pending, 1):
                captured, status = future.result()
                rows.extend(captured)
                coverage.append(status)
                if index % 15 == 0 or index == len(events):
                    print(f"Archive events {index}/{len(events)}; rows {len(rows)}; "
                          f"requests {client.requests}; cache hits {client.cache_hits}", flush=True)
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    manifest = dict(
        schema_version=1, sample_start=SAMPLE_START.isoformat(), sample_end=SAMPLE_END.isoformat(),
        universe=SERIES, snapshot_rule="Exact prior-calendar-day 15:00 UTC one-minute bid/ask close",
        expected_events=len(events), accepted_events=sum(x["status"] == "accepted" for x in coverage),
        rows=len(rows), probe_exclusion=PROBE_EXCLUSION, cutoff=cutoff,
        cutoff_provenance=cutoff_provenance, event_coverage=coverage,
        raw_responses=[client.responses[url] for url in sorted(client.responses)],
        requests=client.requests, cache_hits=client.cache_hits, workers=workers,
        limitations="Historical candles lack depth and fill evidence; this is separate research data, not live admission.",
    )
    return rows, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--download", action="store_true", help="download the full preregistered sample")
    mode.add_argument("--smoke", action="store_true", help="only NYC June 15, already excluded from inference")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data/raw/kalshi_archive_http")
    parser.add_argument("--max-requests", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 13))
    parser.add_argument("--preregistration-commit", default=None)
    args = parser.parse_args()
    if args.download and (not args.preregistration_commit or len(args.preregistration_commit) != 40
                          or any(c not in "0123456789abcdef" for c in args.preregistration_commit)):
        parser.error("--download requires the full SHA of --preregistration-commit")
    default = Path("/tmp/kalshi_archive_smoke") if args.smoke else REPO_ROOT / "data/raw/kalshi_archive_may_june_2026"
    out_dir = guarded_directory(args.out_dir or default)
    client = PublicArchiveClient(args.cache_dir, max_requests=args.max_requests)
    rows, manifest = download_sample(client, frozen_events(args.smoke), workers=args.workers)
    content = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    manifest.update(mode="smoke" if args.smoke else "frozen_full_sample",
                    preregistration_commit=args.preregistration_commit,
                    captures_sha256=hashlib.sha256(content.encode()).hexdigest(),
                    importer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    generated_at=utc_string(time.time()))
    atomic_text(out_dir / "captures.jsonl", content)
    atomic_text(out_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"Wrote {len(rows)} rows; {manifest['accepted_events']}/{manifest['expected_events']} "
          f"events accepted; capture SHA-256 {manifest['captures_sha256']}; output {out_dir}")


if __name__ == "__main__":
    main()
