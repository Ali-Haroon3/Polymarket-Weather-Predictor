"""Parse an already-preserved daily report without inferring missing observations.

This module performs no I/O. The public daily payload is not self-describing as
to units: callers must separately verify preserved Fahrenheit evidence before
passing ``fahrenheit_basis_verified=True``. That flag is not a substitute for
checking the saved market rules, source transport, or receipt chronology.

Only the observed official/no_report schema is supported. Revised and
preliminary report semantics have not been verified and remain unknown. The
supported whole-degree domain below is a conservative parser restriction, not
a claim about source sentinel definitions or a fitted weather parameter.
"""

import datetime as dt
from decimal import Decimal, InvalidOperation
import re


SUPPORTED_MIN_F = Decimal("-150")
SUPPORTED_MAX_F = Decimal("150")
COUNTERS = {"official": "officialReports", "revised": "revisedReports",
            "preliminary": "preliminaryReports", "no_report": "noReports"}
UNIT_FIELDS = ("unit", "units", "temperatureUnit", "tempUnit", "maxTempUnit")
FAHRENHEIT_LABELS = {"F", "°F", "fahrenheit", "Fahrenheit"}


def _date(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return False
    try:
        return dt.date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def _number(value):
    # bool is an int in Python; numeric strings are not JSON weather numbers.
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() else None


def parse_daily(payload, target_date, cli_id, *, fahrenheit_basis_verified=False):
    """Return a JSON-safe official/absent/unknown observation for one exact CLI.

    ``absent`` means a valid source response explicitly reported ``no_report``;
    it does not mean the source was absent between scheduled observations.
    Missing rows, malformed reports and unsupported schema remain ``unknown``.
    Issue-time text is preserved but never represented as a known publication
    time: the inspected evidence does not establish its timestamp semantics.
    """
    result = {
        "state": "unknown", "reason": None, "max_temp_f": None,
        "station_id": cli_id if isinstance(cli_id, str) else None,
        "report_date": target_date if isinstance(target_date, str) else None,
        "source_status": None, "is_official": None, "issue_time": None,
        "publication_time_known": False,
        "fahrenheit_basis_verified": fahrenheit_basis_verified is True,
        "disclosures": [
            "Daily JSON does not establish its own Fahrenheit unit basis.",
            "Receipt time is not source publication time.",
            "An official report may later be revised.",
        ],
    }

    def finish(reason, state="unknown", maximum=None):
        result.update(state=state, reason=reason, max_temp_f=maximum)
        return result

    if not _date(target_date) or not isinstance(cli_id, str) or not re.fullmatch(r"[A-Z]{3}", cli_id):
        return finish("invalid_requested_identity")
    if not isinstance(payload, dict) or not _date(payload.get("date")):
        return finish("invalid_daily_envelope")
    if payload["date"] != target_date:
        return finish("daily_date_mismatch")
    rows = payload.get("results")
    if not isinstance(rows, list):
        return finish("invalid_results")
    if type(payload.get("totalStations")) is not int or payload["totalStations"] != len(rows):
        return finish("station_count_conflict")
    counts = dict.fromkeys(COUNTERS, 0)
    matches = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("station"), dict):
            return finish("invalid_station_row")
        station = row["station"]
        if not isinstance(station.get("cliId"), str) or not re.fullmatch(r"[A-Z]{3}", station["cliId"]):
            return finish("invalid_station_identifier")
        status = row.get("status")
        if not isinstance(status, str) or status not in COUNTERS:
            return finish("unsupported_daily_status")
        counts[status] += 1
        if station["cliId"] == cli_id:
            matches.append(row)
    for status, field in COUNTERS.items():
        if type(payload.get(field)) is not int or payload[field] != counts[status]:
            return finish("status_count_conflict")
    if not matches:
        return finish("station_missing")
    if len(matches) != 1:
        return finish("duplicate_station")
    row = matches[0]
    status = row["status"]
    result["source_status"] = status
    if "data" not in row:
        return finish("missing_report_data")
    data = row["data"]
    if status == "no_report":
        if data is not None:
            return finish("no_report_data_conflict")
        # No numeric weather interpretation is needed to observe explicit absence.
        return finish("source_explicitly_reports_no_report", "absent")
    if not isinstance(data, dict):
        return finish("invalid_report_data")
    if isinstance(data.get("issueTime"), str):
        result["issue_time"] = data["issueTime"]
    elif "issueTime" in data:
        return finish("invalid_issue_time")
    else:
        result["disclosures"].append("Source issue-time field is missing.")
    if type(data.get("isOfficial")) is bool:
        result["is_official"] = data["isOfficial"]
    if status != "official":
        return finish("unsupported_nonofficial_report")
    if data.get("isOfficial") is not True:
        return finish("official_status_conflict")
    for flag in ("isRevised", "isPreliminary"):
        if flag in data and data[flag] is not False:
            return finish("unsupported_revision_or_preliminary_flag")
    if data.get("stationId") != cli_id:
        return finish("report_station_mismatch")
    if data.get("reportDate") != target_date:
        return finish("report_date_mismatch")
    for obj in (payload, row, row["station"], data):
        for field in UNIT_FIELDS:
            if field in obj and (not isinstance(obj[field], str) or obj[field] not in FAHRENHEIT_LABELS):
                return finish("conflicting_or_unsupported_unit")
    if fahrenheit_basis_verified is not True:
        return finish("fahrenheit_basis_unverified")
    maximum = _number(data.get("maxTemp"))
    if maximum is None:
        return finish("invalid_daily_maximum")
    if maximum != maximum.to_integral_value():
        return finish("unsupported_fractional_daily_maximum")
    if not SUPPORTED_MIN_F <= maximum <= SUPPORTED_MAX_F:
        return finish("unsupported_numeric_domain")
    minimum = _number(data.get("minTemp"))
    if minimum is not None and maximum < minimum:
        return finish("maximum_below_minimum")
    if result["issue_time"]:
        result["disclosures"].append("Nonempty issue-time text has unverified publication semantics.")
    return finish("matched_official_daily_maximum", "official", str(int(maximum)))
