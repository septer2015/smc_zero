"""Time helpers: the fixed MSK offset, the EU/US DST seasons and the broker clock.

``data_loader`` owns the pipeline convention that every stamp is
``datetime64[ns, UTC]``, so this module supplies the three pieces the layer
around it needs (SPEC_SMC.md, C1 and §7.26):

* :func:`to_msk_hour` - Moscow is UTC+3 all year (no DST since 2014), so the
  conversion is a constant shift and never a season lookup;
* :func:`eu_is_summer` / :func:`us_is_summer` - the DST season of a *date*,
  computed from the calendar (nth / last Sunday of a month) instead of a table of
  hard-coded years, so every year - inside the 2022-2026 data range and far
  outside it - follows the same rule;
* :func:`broker_server_offset_hours` - how far the clock of the *broker server*
  runs ahead of UTC on a raw stamp.  The tapes of ``./data`` and of the
  MetaTrader 5 export carry server time, not UTC (Э17), and the audit of that
  subtask fixed its rule: GMT+3 while the **US** daylight-saving season is in
  force and GMT+2 outside it - not MSK, which never moves, and not the European
  calendar, which switches three weeks apart from the American one.

Season boundaries are inclusive on both Sundays ("from the last Sunday of March
to the last Sunday of October"), i.e. the switching day itself belongs to
summer.  The actual clock change happens at 01:00 UTC (EU) and 06:00-07:00 UTC
(US), so on those two Sundays a one-hour slice of bars still belongs to the
neighbouring season; that is inherent to a date-level rule and is documented
here rather than hidden.  The broker server follows the US dates and is closed
while its clocks move, so on a switching Sunday there is no bar to misread.
"""

from __future__ import annotations

import calendar
from datetime import date, datetime

# Moscow is UTC+3 the whole year round (Russia stopped observing DST in 2014).
MSK_UTC_OFFSET_HOURS = 3

#: The offset of the broker *server* clock from UTC (Э17): GMT+3 in the US summer
#: season and GMT+2 outside it.  Alfa-Forex midnight is 21:00 UTC in summer and
#: 22:00 UTC in winter, which is why a tape read as UTC is shifted by two to three
#: hours - the defect the Э17 audit found and §7.26 fixes.
BROKER_SERVER_SUMMER_OFFSET_HOURS = 3
BROKER_SERVER_WINTER_OFFSET_HOURS = 2

SUNDAY = 6  # ``datetime.date.weekday()`` counts Monday as 0.


def to_msk_hour(hour_utc: int) -> int:
    """Return the MSK hour of day for a UTC hour of day (wrap-around included)."""
    return (hour_utc + MSK_UTC_OFFSET_HOURS) % 24


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    """Return the ``occurrence``-th ``weekday`` of ``year``-``month`` (1-based)."""
    if occurrence < 1:
        raise ValueError("occurrence must be >= 1")
    first_weekday = date(year, month, 1).weekday()
    day = 1 + (weekday - first_weekday) % 7 + 7 * (occurrence - 1)
    if day > calendar.monthrange(year, month)[1]:
        raise ValueError(f"{year}-{month:02d} has no {weekday=} occurrence {occurrence}")
    return date(year, month, day)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Return the last ``weekday`` of ``year``-``month``."""
    last_day = calendar.monthrange(year, month)[1]
    last_weekday = date(year, month, last_day).weekday()
    return date(year, month, last_day - (last_weekday - weekday) % 7)


def eu_is_summer(day: date) -> bool:
    """Return ``True`` when ``day`` is inside the EU summer-time season.

    Summer runs from the last Sunday of March to the last Sunday of October,
    both endpoints included.
    """
    return _last_weekday(day.year, 3, SUNDAY) <= day <= _last_weekday(day.year, 10, SUNDAY)


def us_is_summer(day: date) -> bool:
    """Return ``True`` when ``day`` is inside the US daylight-saving season.

    Summer runs from the second Sunday of March to the first Sunday of November,
    both endpoints included.
    """
    return _nth_weekday(day.year, 3, SUNDAY, 2) <= day <= _nth_weekday(day.year, 11, SUNDAY, 1)


def broker_server_offset_hours(stamp: date | datetime) -> int:
    """Return how far the broker *server* clock runs ahead of UTC on ``stamp``.

    ``3`` inside the US daylight-saving season and ``2`` outside it: the server
    clock of the tapes the project reads follows the **American** dates, as the Э17
    audit established (GMT+3 on 2026-10-11, GMT+2 after the first Sunday of
    November), and the season is therefore the ordinary :func:`us_is_summer` rule of
    this module - a calendar, never a table of years and never MSK, which is UTC+3
    all year round and would be wrong every winter.

    The argument is the **raw server stamp itself** (a ``date`` or a naive
    ``datetime``, a :class:`pandas.Timestamp` included) and not the UTC instant it
    stands for: one reads a server stamp to decide how to turn it into UTC.  A
    stamp that carries a time zone is read in its own zone.
    """
    day = stamp.date() if isinstance(stamp, datetime) else stamp
    if us_is_summer(day):
        return BROKER_SERVER_SUMMER_OFFSET_HOURS
    return BROKER_SERVER_WINTER_OFFSET_HOURS
