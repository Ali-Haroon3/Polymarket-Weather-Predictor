#!/usr/bin/env python3
"""Preserve a bounded public weather-source and Kalshi book observation.

This is an availability collector, not a strategy, backfill, or proof of what was
known at an earlier decision time. Requests are serial and are not simultaneous.
Only the fixed public GET endpoints below are used; no credentials are read.
"""
import argparse
from dataclasses import dataclass
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request


# Same fixed universe as kalshi_archive_capture.py; no archive behavior imported.
SERIES = {
    "KXHIGHNY": "NYC", "KXHIGHCHI": "Chicago", "KXHIGHAUS": "Austin",
    "KXHIGHDEN": "Denver", "KXHIGHLAX": "LA", "KXHIGHMIA": "Miami",
    "KXHIGHPHIL": "Philadelphia", "KXHIGHTDAL": "Dallas", "KXHIGHTSEA": "Seattle",
    "KXHIGHTATL": "Atlanta", "KXHIGHTBOS": "Boston", "KXHIGHTPHX": "Phoenix",
    "KXHIGHTLV": "Vegas", "KXHIGHTDC": "Washington", "KXHIGHTHOU": "Houston",
}
KALSHI_BASE = "https://external-api.kalshi.com/trade-api/v2"
WEATHER_BASE = "https://weather.com/kalshi/api"
MAX_REQUESTS = 122
MIN_INTERVAL_SECONDS = 0.5
REQUEST_TIMEOUT_SECONDS = 15.0
DEADLINE_SECONDS = 300.0
MAX_BODY_BYTES = 8 * 1024 * 1024
SAFE_HEADERS = {
    "date", "age", "cache-control", "expires", "etag", "last-modified", "vary",
    "content-type", "content-length", "content-encoding", "content-language",
    "x-cache", "x-cache-hits", "cf-cache-status",
}
UTC = dt.timezone.utc


def utc_now():
    return dt.datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def target_date(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("target date must be YYYY-MM-DD")
    result = dt.date.fromisoformat(value)
    if not 2000 <= result.year <= 2099:
        raise ValueError("target date must have an unambiguous 2000-2099 Kalshi ticker year")
    return result


def event_ticker(series, target):
    months = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
    return f"{series}-{target.year % 100:02d}{months[target.month - 1]}{target.day:02d}"


def safe_headers(headers):
    return {str(k).lower(): str(v) for k, v in headers.items() if str(k).lower() in SAFE_HEADERS}


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def strict_json(body):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(body, object_pairs_hook=unique, parse_constant=invalid)


def market_identities(payload, expected_event):
    """Validate identities only, never infer a mathematical partition or liquidity."""
    if not isinstance(payload, dict):
        raise ValueError("market response is not an object")
    if payload.get("cursor") not in (None, "") or payload.get("next_cursor") not in (None, ""):
        raise ValueError("market response is paginated")
    markets = payload.get("markets")
    if not isinstance(markets, list) or len(markets) != 6:
        raise ValueError("expected exactly six markets")
    tickers = []
    pattern = re.escape(expected_event) + r"-[BT]-?\d+(?:\.\d+)?"
    for market in markets:
        if not isinstance(market, dict):
            raise ValueError("invalid market object")
        ticker = market.get("ticker")
        if (not isinstance(ticker, str) or not re.fullmatch(pattern, ticker)
                or market.get("event_ticker") != expected_event
                or market.get("market_type") != "binary"):
            raise ValueError("invalid market ticker, event/date, or market type")
        tickers.append(ticker)
    if len(set(tickers)) != 6:
        raise ValueError("duplicate market ticker")
    return sorted(tickers)


@dataclass
class Response:
    status: int | None = None
    final_url: str | None = None
    headers: dict | None = None
    body: bytes | None = None
    body_complete: bool = False
    error: str | None = None


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def allowed_url(url):
    """Keep even the private HTTP worker restricted to these anonymous endpoints."""
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.username or parsed.password or parsed.port
            or parsed.fragment):
        return False
    query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    if parsed.netloc == "weather.com":
        return ((parsed.path == "/kalshi/api/climate/primary" and set(query) == {"date"}
                 and len(query["date"]) == 1)
                or (parsed.path == "/kalshi/api/metar" and set(query) == {"primary", "weekStart"}
                    and query["primary"] == ["true"] and len(query["weekStart"]) == 1))
    if parsed.netloc != "external-api.kalshi.com":
        return False
    if parsed.path == "/trade-api/v2/markets":
        return set(query) == {"event_ticker", "limit"} and query["limit"] == ["100"]
    if parsed.path in {f"/trade-api/v2/series/{series}" for series in SERIES}:
        return not query
    return (bool(re.fullmatch(r"/trade-api/v2/markets/[A-Z0-9.\-]+/orderbook", parsed.path))
            and query == {"depth": ["100"]})


def http_worker(url, timeout, metadata_path, body_path):
    """One GET, no redirects/retries/proxy/auth; parent enforces total wall time."""
    metadata = dict(status=None, final_url=None, headers={}, body_complete=False, error=None)
    response = None
    try:
        if not allowed_url(url):
            raise ValueError("URL is outside fixed public endpoint allowlist")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        request = urllib.request.Request(url, method="GET", headers={
            "User-Agent": "weather-source-evidence/1.0",
            "Accept": "application/json", "Accept-Encoding": "identity",
        })
        try:
            response = opener.open(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            # HTTP errors are responses too; preserve their status, safe headers, and raw body.
            response = exc
        metadata.update(status=response.getcode(), final_url=response.geturl(),
                        headers=safe_headers(response.headers))
        atomic_json(metadata_path, metadata)
        request_deadline = time.monotonic() + timeout
        read = getattr(response, "read1", response.read)
        with body_path.open("wb") as output:
            size = 0
            while True:
                if time.monotonic() >= request_deadline:
                    raise TimeoutError("response body deadline exceeded")
                chunk = read(min(65536, MAX_BODY_BYTES - size + 1))
                if not chunk:
                    length = metadata["headers"].get("content-length", "").strip()
                    encoding = metadata["headers"].get("content-encoding", "identity").lower()
                    if encoding == "identity" and re.fullmatch(r"\d+", length) and int(length) != size:
                        metadata["error"] = f"Content-Length mismatch: expected {length}, received {size}"
                    else:
                        metadata["body_complete"] = True
                    break
                remaining = MAX_BODY_BYTES - size
                output.write(chunk[:remaining])
                output.flush()
                size += min(len(chunk), remaining)
                if len(chunk) > remaining:
                    metadata["error"] = "response exceeds 8 MiB; exact prefix only"
                    break
    except Exception as exc:
        metadata["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
    finally:
        if response is not None:
            response.close()
        atomic_json(metadata_path, metadata)


class PublicTransport:
    def __call__(self, url, timeout):
        # Socket timeouts alone do not bound DNS or trickle streams. The isolated worker
        # writes partial evidence before the parent terminates it at the wall-clock limit.
        deadline = time.monotonic() + timeout
        with tempfile.TemporaryDirectory(prefix="weather-source-http-") as directory:
            root = Path(directory)
            metadata_path, body_path = root / "response.json", root / "response.body"
            process = subprocess.Popen(
                [sys.executable, "-I", str(Path(__file__).resolve()), "--_http-worker", url,
                 str(timeout), str(metadata_path), str(body_path)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env={},
            )
            timed_out = False
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                process.wait()
            except BaseException:
                # An interrupted attempt remains checkpointed, without claiming receipt.
                # Stop our worker before removing the directory it could still write to.
                process.kill()
                process.wait()
                raise
            metadata = strict_json(metadata_path.read_bytes()) if metadata_path.exists() else {}
            if timed_out:
                metadata.update(error="request wall-clock timeout", body_complete=False)
            elif process.returncode:
                metadata.update(error=f"HTTP worker exited {process.returncode}", body_complete=False)
            return Response(
                status=metadata.get("status"), final_url=metadata.get("final_url"),
                headers=metadata.get("headers", {}),
                body=body_path.read_bytes() if body_path.exists() else None,
                body_complete=metadata.get("body_complete", False), error=metadata.get("error"),
            )


class CollectionStopped(Exception):
    pass


class Collector:
    def __init__(self, target, output_dir, protocol_file, *, transport=None,
                 monotonic=None, sleep=None, now=None):
        self.target = target_date(target)
        protocol_path = Path(protocol_file).resolve(strict=True)
        protocol = protocol_path.read_bytes()  # Fail before directory creation or network.
        self.output = Path(output_dir).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "responses").mkdir()
        (self.output / "protocol.md").write_bytes(protocol)
        self.transport = transport or PublicTransport()
        self.monotonic, self.sleep, self.now = monotonic or time.monotonic, sleep or time.sleep, now or utc_now
        self.started = self.monotonic()
        self.deadline = self.started + DEADLINE_SECONDS
        self.last_request = None
        self.manifest = {
            "schema_version": 1, "purpose": "public_source_and_quote_availability_only",
            "target_date": target, "week_start": (self.target - dt.timedelta(days=self.target.weekday())).isoformat(),
            "started_at_utc": self.now(), "finished_at_utc": None, "complete": False,
            "stop_reason": None, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "protocol": {"original_path": str(protocol_path), "file": "protocol.md",
                         "bytes": len(protocol), "sha256": hashlib.sha256(protocol).hexdigest()},
            "bounds": {"http_requests": MAX_REQUESTS, "minimum_request_interval_seconds": MIN_INTERVAL_SECONDS,
                       "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS, "deadline_seconds": DEADLINE_SECONDS,
                       "response_body_bytes": MAX_BODY_BYTES, "retries": 0, "redirects": False},
            "limitations": [
                "Sequential observations are not simultaneous or guaranteed executable quotes.",
                "No snapshot or later backfill establishes availability before its recorded receipt time.",
                "Complete means request coverage and JSON parsing, not source completeness, partition proof, depth, or alpha.",
                "Missing/invalid event identities suppress books; errors and partial bodies remain evidence only.",
            ],
            "timestamp_semantics": {
                "request_at_utc": "Client dispatch before HTTP worker launch; not wire first-byte time.",
                "receipt_at_utc": "Client observation after HTTP worker returns with a response, including partial/error responses; not first-byte or publication time. Null if no HTTP response was observed.",
                "attempt_finished_at_utc": "Client completion of the attempt, including parsing and persistence.",
            },
            "schema_validation": "event_market_identity_only; other JSON schemas are not validated",
            "requests": [],
            "events": [{"series": series, "city": city, "event_ticker": event_ticker(series, self.target),
                        "identity_status": "not_requested", "identity_error": None,
                        "market_tickers": [], "market_request_id": None,
                        "series_request_id": None, "book_request_ids": []}
                       for series, city in SERIES.items()],
        }
        self.checkpoint()

    def checkpoint(self):
        atomic_json(self.output / "manifest.json", self.manifest)

    def get(self, url, role, **identity):
        if len(self.manifest["requests"]) >= MAX_REQUESTS:
            raise CollectionStopped("request_limit")
        wait = 0.0 if self.last_request is None else max(
            0.0, MIN_INTERVAL_SECONDS - (self.monotonic() - self.last_request))
        if self.monotonic() + wait >= self.deadline:
            raise CollectionStopped("deadline")
        if wait:
            self.sleep(wait)
        remaining = self.deadline - self.monotonic()
        if remaining <= 0:
            raise CollectionStopped("deadline")
        self.last_request = self.monotonic()
        number = len(self.manifest["requests"]) + 1
        record = dict(id=number, role=role, **identity, original_url=url, final_url=None,
                      request_at_utc=self.now(), receipt_at_utc=None, attempt_finished_at_utc=None,
                      status=None, headers={},
                      raw_file=None, bytes=0, sha256=None, body_complete=False,
                      json_valid=False, error=None, timeout_seconds=min(REQUEST_TIMEOUT_SECONDS, remaining))
        self.manifest["requests"].append(record)
        self.checkpoint()
        payload = None
        try:
            response = self.transport(url, record["timeout_seconds"])
            if response.status is not None:
                record["receipt_at_utc"] = self.now()
            record.update(status=response.status, final_url=response.final_url,
                          headers=safe_headers(response.headers or {}),
                          body_complete=response.body_complete, error=response.error)
            if response.body is not None:
                relative = f"responses/{number:03d}-{role}.body"
                (self.output / relative).write_bytes(response.body)
                record.update(raw_file=relative, bytes=len(response.body),
                              sha256=hashlib.sha256(response.body).hexdigest())
                try:
                    payload = strict_json(response.body)
                    record["json_valid"] = True
                except (ValueError, UnicodeError) as exc:
                    record["error"] = record["error"] or f"invalid JSON: {exc}"
            if response.status != 200:
                record["error"] = record["error"] or f"HTTP status {response.status}"
            if not response.body_complete:
                record["error"] = record["error"] or "incomplete response body"
            if not record["json_valid"]:
                record["error"] = record["error"] or "missing JSON response"
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        except KeyboardInterrupt:
            record["error"] = "interrupted"
            raise
        finally:
            record["attempt_finished_at_utc"] = self.now()
            record["elapsed_seconds"] = max(0.0, self.monotonic() - self.last_request)
            self.checkpoint()
        return payload if record["error"] is None else None, record

    def run(self):
        try:
            self.get(f"{WEATHER_BASE}/climate/primary?date={self.target.isoformat()}", "climate")
            self.get(f"{WEATHER_BASE}/metar?primary=true&weekStart={self.manifest['week_start']}", "metar")
            for event in self.manifest["events"]:
                query = urllib.parse.urlencode({"event_ticker": event["event_ticker"], "limit": 100})
                payload, record = self.get(f"{KALSHI_BASE}/markets?{query}", "markets", series=event["series"])
                event["market_request_id"] = record["id"]
                try:
                    if payload is None:
                        raise ValueError("market request failed")
                    event["market_tickers"] = market_identities(payload, event["event_ticker"])
                    event["identity_status"] = "six_unique_target_markets"
                except ValueError as exc:
                    event["identity_status"], event["identity_error"] = "invalid", str(exc)
                self.checkpoint()
                _, record = self.get(f"{KALSHI_BASE}/series/{event['series']}", "series", series=event["series"])
                event["series_request_id"] = record["id"]
                self.checkpoint()
                for ticker in event["market_tickers"]:
                    _, record = self.get(f"{KALSHI_BASE}/markets/{ticker}/orderbook?depth=100",
                                         "orderbook", series=event["series"], ticker=ticker)
                    event["book_request_ids"].append(record["id"])
                    self.checkpoint()
            self.manifest["complete"] = (
                self.monotonic() < self.deadline
                and len(self.manifest["requests"]) == MAX_REQUESTS
                and all(record["error"] is None for record in self.manifest["requests"])
                and all(len(event["book_request_ids"]) == 6 for event in self.manifest["events"])
            )
            self.manifest["stop_reason"] = (
                "finished" if self.manifest["complete"] else
                "deadline" if self.monotonic() >= self.deadline else "incomplete_responses_or_identities")
        except CollectionStopped as exc:
            self.manifest["stop_reason"] = str(exc)
        except KeyboardInterrupt:
            self.manifest["stop_reason"] = "interrupted"
            raise
        finally:
            self.manifest["finished_at_utc"] = self.now()
            self.manifest["elapsed_seconds"] = max(0.0, self.monotonic() - self.started)
            self.checkpoint()
        return self.manifest


def main():
    if len(sys.argv) == 6 and sys.argv[1] == "--_http-worker":
        http_worker(sys.argv[2], float(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol-file", required=True)
    args = parser.parse_args()
    try:
        collector = Collector(args.target_date, args.output_dir, args.protocol_file)
        manifest = collector.run()
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps({"output_dir": str(collector.output), "requests": len(manifest["requests"]),
                      "complete": manifest["complete"], "stop_reason": manifest["stop_reason"]}))
    return 0 if manifest["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
