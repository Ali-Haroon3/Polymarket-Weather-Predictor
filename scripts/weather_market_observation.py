#!/usr/bin/env python3
"""Pure parsing of saved weather-market evidence; never fetch, trade, or score profit.

These helpers assume the caller has separately verified exact response bytes,
request identities, completeness, causal pairing and primary-slot selection.
Unknown or inconsistent schemas are not evidence of absence. No daily source
value, market outcome, or fee coefficient is inferred here.
"""
import datetime as dt
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


STATIONS = {
    "KXHIGHNY": ("New York City", "NYC", "KNYC", "America/New_York"),
    "KXHIGHCHI": ("Chicago", "MDW", "KMDW", "America/Chicago"),
    "KXHIGHAUS": ("Austin", "AUS", "KAUS", "America/Chicago"),
    "KXHIGHDEN": ("Denver", "DEN", "KDEN", "America/Denver"),
    "KXHIGHLAX": ("Los Angeles", "LAX", "KLAX", "America/Los_Angeles"),
    "KXHIGHMIA": ("Miami", "MIA", "KMIA", "America/New_York"),
    "KXHIGHPHIL": ("Philadelphia", "PHL", "KPHL", "America/New_York"),
    "KXHIGHTDAL": ("Dallas", "DFW", "KDFW", "America/Chicago"),
    "KXHIGHTSEA": ("Seattle", "SEA", "KSEA", "America/Los_Angeles"),
    "KXHIGHTATL": ("Atlanta", "ATL", "KATL", "America/New_York"),
    "KXHIGHTBOS": ("Boston", "BOS", "KBOS", "America/New_York"),
    "KXHIGHTPHX": ("Phoenix", "PHX", "KPHX", "America/Phoenix"),
    "KXHIGHTLV": ("Las Vegas", "LAS", "KLAS", "America/Los_Angeles"),
    "KXHIGHTDC": ("Washington DC", "DCA", "KDCA", "America/New_York"),
    "KXHIGHTHOU": ("Houston", "HOU", "KHOU", "America/Chicago"),
}
MONTHS = ("January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December")
# Parser allocation bound, not a weather prediction or trading threshold.
MAX_SUPPORTED_WHOLE_MAGNITUDE = Decimal("1000000")
# Exact supported correction/exceptional-settlement text in the excluded,
# hash-verified engineering evidence. Changed wording requires explicit review.
SECONDARY_RULE_SHA256 = "c1f11eaf372267f2e69ca8ba131916928052ec5e92138e45c16e197f2d4bb507"


def _result(state, reason=None, **fields):
    return dict(state=state, reasons=[] if reason is None else [reason], **fields)


def _date(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("invalid target date")
    parsed = dt.date.fromisoformat(value)
    if not 2000 <= parsed.year <= 2099:
        raise ValueError("ambiguous event year")
    return parsed


def _decimal(value, *, strings_only=False):
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError("invalid decimal")
    if strings_only and (not isinstance(value, str) or len(value) > 128 or not re.fullmatch(
            r"(?:0|[1-9]\d*)(?:\.\d+)?", value)):
        raise ValueError("fixed-point value must be a nonnegative decimal string")
    number = Decimal(str(value))
    if not number.is_finite():
        raise ValueError("nonfinite decimal")
    return number


def _whole(value):
    number = _decimal(value)
    if number.copy_abs() > MAX_SUPPORTED_WHOLE_MAGNITUDE:
        raise ValueError("whole-degree value exceeds parser support/size limit")
    if number != number.to_integral_value():
        raise ValueError("temperature must be a whole degree")
    return int(number)


def _time(value):
    if not isinstance(value, str):
        raise ValueError("missing timestamp")
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp has no timezone")
    return parsed.astimezone(dt.timezone.utc)


def validate_event(payload, series, target_date):
    """Require six exact rules and a disjoint exhaustive partition of integers.

    raw_market retains the caller's parsed object for subsequent lifecycle checks;
    it can contain Decimal and is not a newly serialized evidence artifact.
    """
    try:
        target = _date(target_date)
        city, cli, icao, timezone = STATIONS[series]
        expected = f"{series}-{target.year % 100:02d}{MONTHS[target.month-1][:3].upper()}{target.day:02d}"
        if not isinstance(payload, dict) or any(payload.get(k) not in (None, "")
                                              for k in ("cursor", "next_cursor")):
            raise ValueError("invalid or paginated event response")
        rows = payload.get("markets")
        if not isinstance(rows, list) or len(rows) != 6:
            raise ValueError("event must contain six markets")
        normalized, seen = [], set()
        day = f"{MONTHS[target.month-1][:3]} {target.day}, {target.year}"
        prefix = f"If the maximum temperature recorded at {city} (CLI{cli}) for {day}, is "
        suffix = "° fahrenheit according to The Weather Company, then the market resolves to Yes."
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("malformed market")
            ticker = row.get("ticker")
            match = re.fullmatch(re.escape(expected) + r"-([BT])(-?\d+(?:\.\d+)?)",
                                 ticker) if isinstance(ticker, str) else None
            if (match is None or ticker in seen or row.get("event_ticker") != expected
                    or row.get("market_type") != "binary"):
                raise ValueError("market identity mismatch or duplicate")
            seen.add(ticker)
            if _decimal(row.get("notional_value_dollars")) != 1:
                raise ValueError("unsupported binary notional")
            strike = row.get("strike_type")
            if strike == "between":
                lower, upper = _whole(row.get("floor_strike")), _whole(row.get("cap_strike"))
                if lower > upper or match[1] != "B" or _decimal(match[2]) != Decimal(lower + upper) / 2:
                    raise ValueError("bucket or ticker strike mismatch")
                condition = f"between {lower}-{upper}"
            elif strike == "less":
                cap = _whole(row.get("cap_strike"))
                if row.get("floor_strike") is not None or match[1] != "T" or _decimal(match[2]) != cap:
                    raise ValueError("lower-tail strike mismatch")
                lower, upper, condition = None, cap - 1, f"less than {cap}"
            elif strike == "greater":
                floor = _whole(row.get("floor_strike"))
                if row.get("cap_strike") is not None or match[1] != "T" or _decimal(match[2]) != floor:
                    raise ValueError("upper-tail strike mismatch")
                lower, upper, condition = floor + 1, None, f"greater than {floor}"
            else:
                raise ValueError("unsupported strike type")
            if row.get("rules_primary") != prefix + condition + suffix:
                raise ValueError("rule station/date/unit/source/strike mismatch")
            if not isinstance(row.get("rules_secondary"), str) or not row["rules_secondary"].strip():
                raise ValueError("missing correction or exceptional-settlement rules")
            if hashlib.sha256(row["rules_secondary"].encode("utf-8")).hexdigest() != SECONDARY_RULE_SHA256:
                raise ValueError("unsupported correction or exceptional-settlement rules")
            normalized.append(dict(ticker=ticker, lower=lower, upper=upper, raw_market=row))
        normalized.sort(key=lambda row: (row["lower"] is not None, row["lower"] or 0))
        if (normalized[0]["lower"] is not None or normalized[-1]["upper"] is not None
                or any(left["upper"] is None or right["lower"] is None or
                       left["upper"] + 1 != right["lower"]
                       for left, right in zip(normalized, normalized[1:]))):
            raise ValueError("strikes are not an exhaustive disjoint integer partition")
        return _result("valid", event_ticker=expected, target_date=target_date,
                       station=dict(cli_id=cli, icao=icao, timezone=timezone), markets=normalized)
    except (ValueError, TypeError, KeyError, InvalidOperation, OverflowError) as exc:
        return _result("unknown", str(exc))


def report_consistent_side(normalized_market, whole_degrees_f):
    """Map a verified official whole-degree value; do not infer eventual settlement."""
    try:
        value = _whole(whole_degrees_f)
        lower, upper = normalized_market["lower"], normalized_market["upper"]
        if (lower is not None and _whole(lower) != lower or
                upper is not None and _whole(upper) != upper or
                lower is None and upper is None or
                lower is not None and upper is not None and lower > upper):
            raise ValueError("invalid normalized strike bounds")
        yes = (lower is None or value >= lower) and (upper is None or value <= upper)
        return _result("valid", side="yes" if yes else "no")
    except (ValueError, TypeError, KeyError, InvalidOperation, OverflowError) as exc:
        return _result("unknown", str(exc))


def market_lifecycle(market, metadata_receipt_utc, book_request_utc, book_receipt_utc,
                     *, target_date, timezone):
    """Use both explicit close bounds; disagreement cannot manufacture absence."""
    try:
        target = _date(target_date)
        received, requested, booked = map(_time, (metadata_receipt_utc, book_request_utc,
                                                 book_receipt_utc))
        opened, closed = _time(market.get("open_time")), _time(market.get("close_time"))
        if not received <= requested <= booked or opened >= closed:
            raise ValueError("invalid market/book chronology")
        text = market.get("early_close_condition")
        day = f"{MONTHS[target.month-1]} {target.day}, {target.year}"
        expected = (f"The Last Trading Time will be 11:59 PM local time on {day} "
                    "regardless of any data releases or events occurring. Expiration will occur ")
        if not isinstance(text, str) or not text.startswith(expected):
            raise ValueError("unsupported or missing textual last-trading-time rule")
        zone = ZoneInfo(timezone)
        early = dt.datetime.combine(target, dt.time(23, 59), zone).astimezone(dt.timezone.utc)
        if opened >= early:
            raise ValueError("market opens after textual cutoff")
        bounds = dict(close_time=closed.isoformat(), textual_cutoff=early.isoformat(),
                      close_bounds_disagree=closed != early,
                      limitation="Separate saved observations cannot prove continuous order acceptance.")
        status = market.get("status")
        if status not in ("active", "closed", "determined", "disputed", "amended", "finalized", "initialized", "inactive"):
            return _result("unknown", "unsupported or missing lifecycle status", **bounds)
        if status in ("closed", "determined", "disputed", "amended", "finalized"):
            return _result("closed", "market response explicitly not active", **bounds)
        if status != "active" or received < opened:
            return _result("unknown", "market not observed active and open", **bounds)
        if any(market.get(key) not in (None, "") for key in ("result", "settlement_ts", "settlement_time")):
            return _result("unknown", "active state conflicts with settlement evidence", **bounds)
        if booked >= max(closed, early):
            return _result("closed", "book received at or after both stated cutoffs", **bounds)
        if booked >= min(closed, early):
            return _result("unknown", "book falls within conflicting close bounds", **bounds)
        return _result("active", **bounds)
    except (ValueError, TypeError, AttributeError, OverflowError, ZoneInfoNotFoundError) as exc:
        return _result("unknown", str(exc))


def displayed_offer(payload, side):
    """Return a one-contract limit bound from opposite bids, not an actual fill."""
    # Inputs are at most 128 decimal characters and 100 levels. This precision
    # retains every input digit plus carry digits, independent of caller context.
    with localcontext() as context:
        context.prec = 256
        return _displayed_offer(payload, side)


def _displayed_offer(payload, side):
    try:
        if side not in ("yes", "no") or not isinstance(payload, dict):
            raise ValueError("invalid requested side or book object")
        book = payload.get("orderbook_fp")
        if not isinstance(book, dict):
            raise ValueError("missing fixed-point orderbook")
        parsed = {}
        for key in ("yes_dollars", "no_dollars"):
            levels = book.get(key)
            if not isinstance(levels, list) or len(levels) > 100:
                raise ValueError("missing or oversized bid levels")
            prices, selected = set(), []
            for level in levels:
                if not isinstance(level, list) or len(level) != 2:
                    raise ValueError("malformed price/depth pair")
                price, depth = (_decimal(value, strings_only=True) for value in level)
                if not 0 <= price <= 1 or depth < 0 or price in prices:
                    raise ValueError("invalid or duplicated price/depth")
                prices.add(price)
                if depth > 0:
                    selected.append((price, depth))
            parsed[key] = sorted(selected, reverse=True)
        if (all(parsed.values()) and
                parsed["yes_dollars"][0][0] + parsed["no_dollars"][0][0] > 1):
            raise ValueError("crossed positive-depth bids")
        opposing = parsed["no_dollars" if side == "yes" else "yes_dollars"]
        cumulative, included = Decimal(0), []
        for bid, depth in opposing:
            offer = 1 - bid
            if not 0 < offer < 1:
                continue
            cumulative += depth
            included.append(dict(offer_price=str(offer), displayed_depth=str(depth)))
            if cumulative >= 1:
                return _result("available", side=side, price=str(offer), depth=str(cumulative),
                               levels=included, price_is_limit_upper_bound=True,
                               limitation="Displayed cumulative depth; no atomic fill or fee claim.")
        return _result("absent", "fewer than one whole contract at interior offers in saved levels",
                       side=side, depth=str(cumulative), levels=included,
                       limitation="Finite saved depth only; deeper or later liquidity is unknown.")
    except (ValueError, TypeError, InvalidOperation, OverflowError) as exc:
        return _result("unknown", str(exc))


def fee_terms(payload, expected_series):
    """Identify supported saved metadata, without inventing its missing coefficient."""
    try:
        if expected_series not in STATIONS or not isinstance(payload, dict):
            raise ValueError("invalid series")
        series = payload.get("series")
        if not isinstance(series, dict) or series.get("ticker") != expected_series:
            raise ValueError("fee series identity mismatch")
        multiplier = _decimal(series.get("fee_multiplier"))
        if series.get("fee_type") != "quadratic" or multiplier != 1:
            raise ValueError("unsupported or missing saved fee rule")
        if series.get("settlement_sources") != [{"name": "The Weather Company",
                                                "url": "https://weather.com/kalshi"}]:
            raise ValueError("series settlement source mismatch")
        return _result("supported_metadata", fee_type="quadratic", fee_multiplier=str(multiplier),
                       payout_comparison_available=False,
                       limitation="Saved metadata does not specify coefficient or rounding rule.")
    except (ValueError, TypeError, InvalidOperation) as exc:
        return _result("unknown", str(exc), payout_comparison_available=False)
