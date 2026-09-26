"""Time helpers: the fixed MSK offset and the EU/US DST season rules.

``data_loader`` owns the pipeline convention that every stamp is
``datetime64[ns, UTC]``, so this module only supplies the two pieces the
killzone gate needs (SPEC_SMC.md, C1):

* :func:`to_msk_hour` - Moscow is UTC+3 all year (no DST since 2014), so the
  conversion is a constant shift and never a season lookup;
* :func:`eu_is_summer` / :func:`us_is_summer` - the DST season of a *date*,
  computed from the calendar (nth / last Sunday of a month) instead of a table of
  hard-coded years, so every year - inside the 2022-2026 data range and far
  outside it - follows the same rule.

Season boundaries are inclusive on both Sundays ("from the last Sunday of March
to the last Sunday of October"), i.e. the switching day itself belongs to
summer.  The actual clock change happens at 01:00 UTC (EU) and 06:00-07:00 UTC
(US), so on those two Sundays a one-hour slice of bars still belongs to the
neighbouring season; that is inherent to a date-level rule and is documented
here rather than hidden.
"""

from __future__ import annotations

import calendar
from datetime import date

# Moscow is UTC+3 the whole year round (Russia stopped observing DST in 2014).
MSK_UTC_OFFSET_HOURS = 3

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
