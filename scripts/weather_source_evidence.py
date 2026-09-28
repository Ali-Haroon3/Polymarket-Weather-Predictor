#!/usr/bin/env python3
"""Read preserved source evidence without replacing missing or damaged vintages.

This module does not select primary invocations, certify artifact inventory, or
evaluate a strategy. Reserved bodies require both the frozen release time and an
explicit assertion that final-slot attempts have been reconciled as terminal.
"""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import datetime as dt
import hashlib
from functools import wraps
import json
from pathlib import Path
import re
from urllib.parse import urlencode

from weather_source_schedule import COLLECTOR_SHA256, PROTOCOL_SHA256

UTC = dt.timezone.utc
RESERVED_START = dt.date(2026, 10, 10)
RESERVED_END = dt.date(2026, 10, 23)
RELEASE_TIME = dt.datetime(2026, 10, 24, 9, 30, tzinfo=UTC)
BASE = "https://external-api.kalshi.com/trade-api/v2"
SERIES = frozenset(("KXHIGHNY", "KXHIGHCHI", "KXHIGHAUS", "KXHIGHDEN", "KXHIGHLAX",
                    "KXHIGHMIA", "KXHIGHPHIL", "KXHIGHTDAL", "KXHIGHTSEA", "KXHIGHTATL",
                    "KXHIGHTBOS", "KXHIGHTPHX", "KXHIGHTLV", "KXHIGHTDC", "KXHIGHTHOU"))


class EvidenceError(ValueError):
    """Evidence is unavailable or cannot support the requested inference."""


def unavailable_is_unknown(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except OSError as exc:
            raise EvidenceError("evidence_file_unavailable") from exc
    return wrapped


def strict_json(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise EvidenceError("duplicate_json_key")
            value[key] = item
        return value

    def invalid(_):
        raise EvidenceError("nonfinite_json_constant")

    try:
        return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid, parse_float=Decimal)
    except (ValueError, UnicodeError, InvalidOperation) as exc:
        raise EvidenceError("invalid_json") from exc


def timestamp(value):
    if not isinstance(value, str):
        raise EvidenceError("missing_timestamp")
    try:
        result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidenceError("invalid_timestamp") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise EvidenceError("timestamp_timezone_missing")
    return result.astimezone(UTC)


def event_ticker(series, target):
    if series not in SERIES:
        raise EvidenceError("series_outside_frozen_universe")
    months = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
    return f"{series}-{target.year % 100:02}{months[target.month - 1]}{target.day:02}"


def expected_url(target, role, series=None, ticker=None):
    if role == "climate" and series is None and ticker is None:
        return f"https://weather.com/kalshi/api/climate/primary?date={target.isoformat()}"
    if role == "metar" and series is None and ticker is None:
        week = target - dt.timedelta(days=target.weekday())
        return f"https://weather.com/kalshi/api/metar?primary=true&weekStart={week.isoformat()}"
    event = event_ticker(series, target)
    if role == "markets" and ticker is None:
        return BASE + "/markets?" + urlencode({"event_ticker": event, "limit": 100})
    if role == "series" and ticker is None:
        return f"{BASE}/series/{series}"
    if (role == "orderbook" and isinstance(ticker, str)
            and re.fullmatch(re.escape(event) + r"-[BT]-?\d+(?:\.\d+)?", ticker)):
        return f"{BASE}/markets/{ticker}/orderbook?depth=100"
    raise EvidenceError("unsupported_request_identity")


@dataclass(frozen=True)
class ResponseEvidence:
    target_date: str
    role: str
    series: str | None
    ticker: str | None
    request_at: dt.datetime
    receipt_at: dt.datetime
    payload: object
    provenance: dict


class CaptureEvidence:
    @unavailable_is_unknown
    def __init__(self, directory):
        self.directory = Path(directory).resolve(strict=True)
        manifest_bytes = (self.directory / "manifest.json").read_bytes()
        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        self.manifest = strict_json(manifest_bytes)
        m = self.manifest
        if (not isinstance(m, dict) or type(m.get("schema_version")) is not int
                or m["schema_version"] != 1
                or m.get("purpose") != "public_source_and_quote_availability_only"
                or m.get("script_sha256") != COLLECTOR_SHA256):
            raise EvidenceError("unsupported_manifest_or_collector")
        target = m.get("target_date")
        if not isinstance(target, str) or not re.fullmatch(r"2026-\d{2}-\d{2}", target):
            raise EvidenceError("unsupported_target_date")
        try:
            self.target = dt.date.fromisoformat(target)
        except ValueError as exc:
            raise EvidenceError("invalid_target_date") from exc
        protocol = m.get("protocol")
        if (not isinstance(protocol, dict) or protocol.get("file") != "protocol.md"
                or protocol.get("sha256") != PROTOCOL_SHA256):
            raise EvidenceError("protocol_identity_mismatch")
        raw_protocol = self.safe_file("protocol.md").read_bytes()
        if (type(protocol.get("bytes")) is not int or protocol["bytes"] != len(raw_protocol)
                or hashlib.sha256(raw_protocol).hexdigest() != PROTOCOL_SHA256):
            raise EvidenceError("protocol_bytes_mismatch")
        self.started = timestamp(m.get("started_at_utc"))
        self.finished = timestamp(m.get("finished_at_utc"))
        if self.finished < self.started:
            raise EvidenceError("manifest_timestamp_order_invalid")
        requests = m.get("requests")
        if not isinstance(requests, list) or len(requests) > 122:
            raise EvidenceError("invalid_request_inventory")
        ids = [row.get("id") if isinstance(row, dict) else None for row in requests]
        if any(type(i) is not int for i in ids) or ids != list(range(1, len(ids) + 1)):
            raise EvidenceError("ambiguous_request_inventory")

    @unavailable_is_unknown
    def safe_file(self, relative):
        if not isinstance(relative, str):
            raise EvidenceError("missing_evidence_file")
        path = (self.directory / relative).resolve(strict=True)
        if not path.is_relative_to(self.directory) or not path.is_file():
            raise EvidenceError("evidence_path_outside_capture")
        return path

    @unavailable_is_unknown
    def response(self, role, *, series=None, ticker=None, validation_released=False, now=None):
        # Enforce embargo before touching any response body, even an engineering fixture.
        if RESERVED_START <= self.target <= RESERVED_END:
            current = now if now is not None else dt.datetime.now(UTC)
            if current.tzinfo is None or current.utcoffset() is None:
                raise EvidenceError("release_clock_timezone_missing")
            if current.astimezone(UTC) < RELEASE_TIME or validation_released is not True:
                raise EvidenceError("reserved_bodies_locked")
        url = expected_url(self.target, role, series, ticker)
        rows = [r for r in self.manifest["requests"] if r.get("role") == role
                and r.get("series") == series and r.get("ticker") == ticker]
        if len(rows) != 1:
            raise EvidenceError("missing_or_duplicate_response")
        row = rows[0]
        if row.get("original_url") != url or row.get("final_url") != url:
            raise EvidenceError("response_identity_mismatch")
        if (type(row.get("status")) is not int or row["status"] != 200
                or row.get("body_complete") is not True or row.get("json_valid") is not True
                or row.get("error") is not None):
            raise EvidenceError("response_unsuccessful_or_incomplete")
        requested = timestamp(row.get("request_at_utc"))
        received = timestamp(row.get("receipt_at_utc"))
        ended = timestamp(row.get("attempt_finished_at_utc"))
        if not self.started <= requested <= received <= ended <= self.finished:
            raise EvidenceError("response_timestamp_order_invalid")
        expected_file = f"responses/{row['id']:03d}-{role}.body"
        if row.get("raw_file") != expected_file:
            raise EvidenceError("unexpected_response_file")
        body_file = self.safe_file(expected_file)
        if body_file.stat().st_size > 8 * 1024 * 1024:
            raise EvidenceError("response_bytes_mismatch")
        raw = body_file.read_bytes()
        if (type(row.get("bytes")) is not int or len(raw) != row["bytes"]
                or len(raw) > 8 * 1024 * 1024
                or hashlib.sha256(raw).hexdigest() != row.get("sha256")):
            raise EvidenceError("response_bytes_mismatch")
        headers = row.get("headers")
        if (not isinstance(headers, dict)
                or any(not isinstance(k, str) or not isinstance(v, str) for k, v in headers.items())):
            raise EvidenceError("invalid_response_headers")
        length = headers.get("content-length")
        if (headers.get("content-encoding", "identity").lower() == "identity" and length is not None
                and (not re.fullmatch(r"\d+", length) or int(length) != len(raw))):
            raise EvidenceError("response_content_length_mismatch")
        provenance = {key: row.get(key) for key in (
            "id", "original_url", "request_at_utc", "receipt_at_utc", "attempt_finished_at_utc",
            "headers", "bytes", "sha256", "raw_file")}
        provenance["manifest_sha256"] = self.manifest_sha256
        return ResponseEvidence(self.target.isoformat(), role, series, ticker, requested, received,
                                strict_json(raw), provenance)


def pair_source_book(daily, markets, book, close_time):
    """Transport/lifecycle timing only; callers must validate report and market semantics."""
    reasons = []
    if (daily.role != "climate" or markets.role != "markets" or book.role != "orderbook"
            or daily.target_date != book.target_date or markets.target_date != book.target_date
            or markets.series != book.series
            or len({item.provenance.get("manifest_sha256") for item in (daily, markets, book)}) != 1
            or daily.provenance.get("manifest_sha256") is None):
        reasons.append("pair_identity_mismatch")
    if not daily.receipt_at < book.request_at:
        reasons.append("source_not_received_before_book_request")
    lag = (book.receipt_at - daily.receipt_at).total_seconds()
    if not 0 <= lag <= 120:
        reasons.append("source_book_pair_exceeds_120_seconds")
    if not markets.receipt_at < book.request_at:
        reasons.append("market_metadata_not_received_before_book_request")
    try:
        close = timestamp(close_time)
        if markets.receipt_at >= close or book.receipt_at >= close:
            reasons.append("market_or_book_received_at_or_after_close")
    except EvidenceError:
        reasons.append("close_time_unknown")
    return {"state": "eligible" if not reasons else "unknown", "reasons": reasons,
            "source_to_book_receipt_seconds": lag,
            "cache_headers": {"source": daily.provenance["headers"],
                              "markets": markets.provenance["headers"],
                              "book": book.provenance["headers"]},
            "limitations": ["Receipt times are not source publication times.",
                            "Sequential public requests do not prove an executable fill."]}
