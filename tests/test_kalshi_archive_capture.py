"""Archive fixtures test contemporaneous quotes, causal metadata, and safe retrieval."""
import copy
import datetime as dt
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import kalshi_archive_capture as archive

TARGET = dt.date(2026, 6, 15)
ENTRY = archive.entry_timestamp(TARGET)
EVENT = "KXHIGHNY-26JUN15"


def fixture():
    common = dict(event_ticker=EVENT, market_type="binary", status="finalized",
                  created_time="2026-06-14T09:30:00Z", open_time="2026-06-14T14:00:00Z",
                  close_time="2026-06-16T04:59:00Z", settlement_ts="2026-06-16T12:01:43.4Z",
                  result="no", yes_bid_dollars="0.9900", yes_ask_dollars="1.0000",
                  last_price_dollars="0.9900")
    markets = [dict(common, ticker=EVENT + "-T78", strike_type="less", cap_strike=78),
               dict(common, ticker=EVENT + "-T85", strike_type="greater", floor_strike=85)]
    for floor in (78, 80, 82, 84):
        markets.append(dict(common, ticker=EVENT + f"-B{floor}.5", strike_type="between",
                            floor_strike=floor, cap_strike=floor + 1))
    markets[0]["result"] = "yes"
    candles = {}
    for market in markets:
        ticker = market["ticker"]
        candle = dict(end_period_ts=ENTRY, yes_bid={"close": "0.1200"},
                      yes_ask={"close": "0.2200"}, price={"close": "0.9900"},
                      volume="0.00", open_interest="71.80")
        candles[ticker] = dict(ticker=ticker, candlesticks=[candle])
    return markets, candles


class FixtureClient:
    def __init__(self):
        self.markets, self.candles = fixture()
        self.calls = []

    def get(self, path, params=None):
        self.calls.append((path, params))
        if path == "/historical/markets":
            payload = dict(markets=self.markets, cursor="")
        else:
            ticker = path.split("/")[-2]
            payload = self.candles[ticker]
            assert params == dict(start_ts=ENTRY - 60, end_ts=ENTRY, period_interval=1)
        return copy.deepcopy(payload), dict(url="https://example.test" + path,
                                            sha256="a" * 64, fetched_at="2026-09-22T05:00:00Z")


class FakeResponse(io.BytesIO):
    status = 200


class ArchiveCaptureTests(unittest.TestCase):
    def test_worker_bounds_allow_twelve_without_changing_frozen_sample(self):
        for invalid in (0, 13, 4.5, True):
            with self.assertRaisesRegex(ValueError, "between one and twelve"):
                archive.download_sample(None, [], workers=invalid)
        client = Mock(requests=0, cache_hits=0, responses={})
        client.get.return_value = ({"market_settled_ts": "2026-07-23T00:00:00Z"}, {})
        rows, manifest = archive.download_sample(client, [], workers=12)
        self.assertEqual(rows, [])
        self.assertEqual(manifest["workers"], 12)
        self.assertEqual(manifest["sample_start"], "2026-05-01")
        self.assertEqual(manifest["sample_end"], "2026-06-28")

    def test_frozen_universe_has_exact_dates_and_all_fifteen_series(self):
        events = archive.frozen_events()
        self.assertEqual(len(events), 59 * 15)
        self.assertEqual(len(set(events)), len(events))
        self.assertEqual(min(target for _, target in events), dt.date(2026, 5, 1))
        self.assertEqual(max(target for _, target in events), dt.date(2026, 6, 28))
        self.assertEqual(len(set(series for series, _ in events)), 15)
        self.assertEqual(archive.utc_string(ENTRY), "2026-06-14T15:00:00Z")

    def test_full_ladder_uses_exact_candle_and_preserves_provenance(self):
        client = FixtureClient()
        rows = archive.reconstruct_event(client, "KXHIGHNY", TARGET)
        self.assertEqual(len(rows), 6)
        self.assertEqual(len(client.calls), 7)
        self.assertEqual(sum(row["outcome"] for row in rows), 1)
        for row in rows:
            self.assertEqual(row["best_bid"], 0.12)
            self.assertEqual(row["best_ask"], 0.22)
            self.assertAlmostEqual(row["entry_price"], 0.17)
            self.assertEqual(row["entry_ts"], ENTRY)
            self.assertEqual(row["quote_close_ts"], ENTRY)
            self.assertEqual(row["captured_at"], "2026-06-14")
            self.assertEqual(row["event_ticker"], EVENT)
            self.assertEqual(row["metadata_response_sha256"], "a" * 64)
            self.assertEqual(row["candle_response_sha256"], "a" * 64)
            # Round settlement availability upward, never before the supplied fractional second.
            self.assertEqual(archive.utc_string(row["settlement_ts"]), "2026-06-16T12:01:44Z")
            self.assertEqual(row["volume"], 0.0)

    def test_prior_or_later_candle_cannot_replace_exact_boundary(self):
        for offset in (-60, 60):
            client = FixtureClient()
            next(iter(client.candles.values()))["candlesticks"][0]["end_period_ts"] += offset
            with self.assertRaisesRegex(archive.InvalidArchive, "exact 15:00"):
                archive.reconstruct_event(client, "KXHIGHNY", TARGET)

    def test_missing_or_malformed_quotes_never_use_trade_or_final_market_price(self):
        for invalid in (None, "NaN", "-0.01", "1.01"):
            client = FixtureClient()
            candle = next(iter(client.candles.values()))["candlesticks"][0]
            candle["yes_bid"]["close"] = invalid
            candle["yes_ask"]["close"] = invalid
            with self.assertRaises(archive.InvalidArchive):
                archive.reconstruct_event(client, "KXHIGHNY", TARGET)
        client = FixtureClient()
        del next(iter(client.candles.values()))["candlesticks"][0]["yes_bid"]["close"]
        with self.assertRaisesRegex(archive.InvalidArchive, "missing candle"):
            archive.reconstruct_event(client, "KXHIGHNY", TARGET)

    def test_real_one_sided_book_is_preserved_without_phantom_quote(self):
        for side, absent in (("yes_bid", None), ("yes_bid", "0.0000"),
                             ("yes_ask", None), ("yes_ask", "1.0000")):
            client = FixtureClient()
            first_ticker = next(iter(client.candles))
            client.candles[first_ticker]["candlesticks"][0][side]["close"] = absent
            rows = archive.reconstruct_event(client, "KXHIGHNY", TARGET)
            row = next(row for row in rows if row["market_id"] == first_ticker)
            missing = "best_bid" if side == "yes_bid" else "best_ask"
            present = "best_ask" if side == "yes_bid" else "best_bid"
            self.assertIsNone(row[missing])
            self.assertEqual(row["entry_price"], row[present])

    def test_whole_event_rejected_for_missing_duplicate_gap_or_overlap_leg(self):
        for operation in ("missing", "duplicate", "gap", "overlap"):
            client = FixtureClient()
            if operation == "missing":
                client.markets.pop()
            elif operation == "duplicate":
                client.markets.append(copy.deepcopy(client.markets[0]))
            else:
                client.markets[1]["floor_strike"] += 1 if operation == "gap" else -1
            with self.assertRaisesRegex(archive.InvalidArchive, "partition"):
                archive.reconstruct_event(client, "KXHIGHNY", TARGET)
            self.assertEqual(len(client.calls), 1)  # Quarantine before requesting candles.

    def test_lifecycle_and_identity_must_be_known_at_entry(self):
        changes = [dict(open_time="2026-06-14T15:01:00Z"),
                   dict(created_time="2026-06-14T16:00:00Z"),
                   dict(settlement_ts="2026-06-14T14:00:00Z"),
                   dict(open_time="2026-06-14T14:00:00"),
                   dict(event_ticker="KXHIGHNY-26JUN16"), dict(result="scalar")]
        for change in changes:
            client = FixtureClient()
            client.markets[0].update(change)
            with self.assertRaises(archive.InvalidArchive):
                archive.reconstruct_event(client, "KXHIGHNY", TARGET)

    def test_duplicate_candle_and_wrong_ticker_are_rejected(self):
        client = FixtureClient()
        payload = next(iter(client.candles.values()))
        payload["candlesticks"].append(copy.deepcopy(payload["candlesticks"][0]))
        with self.assertRaisesRegex(archive.InvalidArchive, "duplicate candle"):
            archive.reconstruct_event(client, "KXHIGHNY", TARGET)
        client = FixtureClient()
        next(iter(client.candles.values()))["ticker"] = "WRONG"
        with self.assertRaisesRegex(archive.InvalidArchive, "ticker/schema"):
            archive.reconstruct_event(client, "KXHIGHNY", TARGET)

    def test_http_cache_preserves_raw_hash_and_sends_no_credentials(self):
        requests = []
        def opener(request, timeout):
            requests.append(request)
            return FakeResponse(b'{"market_settled_ts":"2026-07-23T00:00:00Z"}')
        with tempfile.TemporaryDirectory() as directory:
            client = archive.PublicArchiveClient(directory, opener=opener)
            first = client.get("/historical/cutoff")
            second = client.get("/historical/cutoff")
            self.assertEqual(first, second)
            self.assertEqual(len(requests), 1)
            self.assertEqual(client.cache_hits, 1)
            self.assertFalse(any("kalshi-access" in name.lower() for name in requests[0].headers))
            cache_file = next(Path(directory).glob("*.json"))
            record = json.loads(cache_file.read_text())
            record["body"] = "{}"
            cache_file.write_text(json.dumps(record))
            with self.assertRaisesRegex(archive.InvalidArchive, "invalid cached"):
                client.get("/historical/cutoff")

    def test_retry_and_request_cap_are_bounded(self):
        attempts, sleeps = [], []
        def opener(request, timeout):
            attempts.append(request)
            if len(attempts) < 3:
                raise urllib.error.HTTPError(request.full_url, 429 if len(attempts) == 1 else 503,
                                             "retry", {"Retry-After": "1"}, None)
            return FakeResponse(b'{}')
        with tempfile.TemporaryDirectory() as directory:
            client = archive.PublicArchiveClient(directory, max_requests=3, opener=opener,
                                                 sleeper=sleeps.append, clock=lambda: 0.0)
            client.get("/historical/cutoff")
            self.assertEqual(len(attempts), 3)
            self.assertIn(1.0, sleeps)
            self.assertIn(2.0, sleeps)
            with self.assertRaisesRegex(RuntimeError, "request cap"):
                client.get("/historical/markets", {"event_ticker": EVENT})

    def test_malformed_json_and_canonical_output_are_rejected(self):
        for raw in ('{"a":NaN}', '{"a":1,"a":2}', 'broken'):
            with self.assertRaises(archive.InvalidArchive):
                archive.strict_json(raw)
        with self.assertRaisesRegex(ValueError, "data/raw"):
            archive.guarded_directory(archive.REPO_ROOT / "data")


if __name__ == "__main__":
    unittest.main()
