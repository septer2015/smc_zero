"""Tests for :mod:`smc_zero.utils.time`: MSK shift and EU/US DST seasons.

The season must be a *rule*, not a table: the boundary blocks below cover
2022-2027 plus a year far outside the data range, so replacing the calendar
arithmetic with a table of 2026 seasons goes red on the 2027 dates - exactly the
mutation required by the Э1'.0 acceptance gate.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from smc_zero.utils.time import MSK_UTC_OFFSET_HOURS, eu_is_summer, to_msk_hour, us_is_summer

ONE_DAY = timedelta(days=1)

# (year, EU summer start, EU summer end, US summer start, US summer end);
# every transition Sunday is also cross-checked against the zoneinfo DST offsets
# of Europe/London and America/New_York during review.
SEASON_BOUNDARIES: list[tuple[int, date, date, date, date]] = [
    (2022, date(2022, 3, 27), date(2022, 10, 30), date(2022, 3, 13), date(2022, 11, 6)),
    (2023, date(2023, 3, 26), date(2023, 10, 29), date(2023, 3, 12), date(2023, 11, 5)),
    (2024, date(2024, 3, 31), date(2024, 10, 27), date(2024, 3, 10), date(2024, 11, 3)),
    (2025, date(2025, 3, 30), date(2025, 10, 26), date(2025, 3, 9), date(2025, 11, 2)),
    (2026, date(2026, 3, 29), date(2026, 10, 25), date(2026, 3, 8), date(2026, 11, 1)),
    (2027, date(2027, 3, 28), date(2027, 10, 31), date(2027, 3, 14), date(2027, 11, 7)),
]

# Dates where the EU and the US calendars disagree: the shared mutation target of
# Э1'.0 and Э1'.1 (an EU flag applied to New York, or a US flag applied to
# London, is invisible on every other day of the year).
ASYMMETRY_DATES: list[date] = [
    date(2026, 3, 15),  # US summer already, EU winter still
    date(2026, 10, 28),  # EU winter already, US summer still
    date(2027, 3, 20),  # same asymmetry one year later (no 2026 table)
    date(2022, 3, 20),  # ... and one year earlier
]


@pytest.mark.parametrize(
    ("hour_utc", "expected"),
    [(0, 3), (3, 6), (12, 15), (20, 23), (21, 0), (23, 2)],
)
def test_to_msk_hour_is_the_fixed_utc_plus_three(hour_utc: int, expected: int) -> None:
    assert MSK_UTC_OFFSET_HOURS == 3
    assert to_msk_hour(hour_utc) == expected


@pytest.mark.parametrize(("year", "eu_start", "eu_end", "us_start", "us_end"), SEASON_BOUNDARIES)
def test_season_boundaries_are_inclusive_and_calendar_driven(
    year: int, eu_start: date, eu_end: date, us_start: date, us_end: date
) -> None:
    assert year  # the year only names the row; it is never looked up
    assert not eu_is_summer(eu_start - ONE_DAY)
    assert eu_is_summer(eu_start)
    assert eu_is_summer(eu_start + ONE_DAY)
    assert eu_is_summer(eu_end - ONE_DAY)
    assert eu_is_summer(eu_end)
    assert not eu_is_summer(eu_end + ONE_DAY)
    assert not us_is_summer(us_start - ONE_DAY)
    assert us_is_summer(us_start)
    assert us_is_summer(us_start + ONE_DAY)
    assert us_is_summer(us_end - ONE_DAY)
    assert us_is_summer(us_end)
    assert not us_is_summer(us_end + ONE_DAY)


def test_2026_boundary_dates_are_exact() -> None:
    # EU: last Sunday of March (29) and last Sunday of October (25)
    assert not eu_is_summer(date(2026, 3, 28))
    assert eu_is_summer(date(2026, 3, 29))
    assert eu_is_summer(date(2026, 3, 30))
    assert eu_is_summer(date(2026, 10, 25))
    assert not eu_is_summer(date(2026, 10, 26))
    # US: second Sunday of March (8) and first Sunday of November (1)
    assert not us_is_summer(date(2026, 3, 7))
    assert us_is_summer(date(2026, 3, 8))
    assert us_is_summer(date(2026, 3, 9))
    assert us_is_summer(date(2026, 11, 1))
    assert not us_is_summer(date(2026, 11, 2))


@pytest.mark.parametrize("day", ASYMMETRY_DATES)
def test_asymmetry_dates_are_eu_winter_and_us_summer(day: date) -> None:
    assert not eu_is_summer(day)
    assert us_is_summer(day)


def test_2027_dates_are_computed_not_tabulated() -> None:
    """A hard-coded 2026 season table must fail here (the Э1'.0 mutation)."""
    assert not eu_is_summer(date(2027, 3, 27))
    assert eu_is_summer(date(2027, 3, 28))
    assert eu_is_summer(date(2027, 10, 31))
    assert not eu_is_summer(date(2027, 11, 1))
    assert not us_is_summer(date(2027, 3, 13))
    assert us_is_summer(date(2027, 3, 14))
    assert us_is_summer(date(2027, 11, 7))
    assert not us_is_summer(date(2027, 11, 8))


def test_years_far_outside_the_data_range_follow_the_same_rule() -> None:
    assert not eu_is_summer(date(2030, 3, 30))
    assert eu_is_summer(date(2030, 3, 31))
    assert eu_is_summer(date(2030, 10, 27))
    assert not eu_is_summer(date(2030, 10, 28))
    assert not us_is_summer(date(2030, 3, 9))
    assert us_is_summer(date(2030, 3, 10))
    assert us_is_summer(date(2030, 11, 3))
    assert not us_is_summer(date(2030, 11, 4))
    # a mid-winter and a mid-summer sanity pair, also outside 2022-2026
    assert not eu_is_summer(date(2031, 1, 15))
    assert eu_is_summer(date(2031, 7, 15))
    assert not us_is_summer(date(2031, 1, 15))
    assert us_is_summer(date(2031, 7, 15))
