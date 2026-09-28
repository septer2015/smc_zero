"""Tests for :mod:`smc_zero.indicators.sessions`: the killzone gate and the C6 hours.

The gate is the one place where C1's seasonality lives, so it is pinned in three
directions: the MSK windows of the config, their UTC re-expression (which the Э3'
session maps must reuse) and the actual bar-level gate on dates where the EU and
US calendars disagree.  The same module owns the broker's *week* (C6: Monday 02:00
MSK - Friday 23:55 MSK), whose boundaries are read from the same config object.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone

import pandas as pd
import pytest

from smc_zero.config import SessionConfig
from smc_zero.indicators.sessions import (
    LONDON,
    NY,
    PRE_LONDON,
    SUMMER,
    WINTER,
    alfa_trading_mask,
    in_killzone,
    in_trading_hours,
    killzone_mask,
    kz_window_utc_hours,
    kz_windows_msk,
    season_for,
)

MSK = timezone(timedelta(hours=3))

SUMMER_DAY = date(2026, 7, 15)
WINTER_DAY = date(2026, 1, 15)

# Days on which the EU and the US calendars disagree: one session is already (or
# still) on winter hours while the other is still (or already) on summer hours.
# Applying a single flag to both sessions is invisible on every other date.
SPLIT_DAYS = [date(2026, 3, 15), date(2026, 10, 28), date(2027, 3, 20)]

# Prod's UTC-hour windows that this port reproduces (reference behaviour):
# prod London ``hour >= 6 and hour < 9`` == London summer, prod NY
# ``hour >= 12 and hour < 15`` == New York winter.
PROD_LONDON_SUMMER_UTC = (6, 9)
PROD_NY_WINTER_UTC = (12, 15)


def utc_stamp(day: date, hour_msk: int) -> pd.Timestamp:
    """Return the UTC stamp of the bar opening at ``hour_msk`` Moscow time."""
    local = datetime(day.year, day.month, day.day, hour_msk, tzinfo=MSK)
    return pd.Timestamp(local.astimezone(UTC))


def test_season_is_taken_per_session() -> None:
    assert season_for(LONDON, SUMMER_DAY) == SUMMER
    assert season_for(NY, SUMMER_DAY) == SUMMER
    assert season_for(LONDON, WINTER_DAY) == WINTER
    assert season_for(NY, WINTER_DAY) == WINTER
    assert season_for(PRE_LONDON, SUMMER_DAY) == SUMMER  # fixed window, EU label


@pytest.mark.parametrize("day", SPLIT_DAYS)
def test_split_days_give_london_winter_and_ny_summer(day: date) -> None:
    assert season_for(LONDON, day) == WINTER
    assert season_for(NY, day) == SUMMER
    # London winter hours already apply, London summer hours no longer do ...
    assert not in_killzone(utc_stamp(day, 9))
    assert in_killzone(utc_stamp(day, 10))
    # ... while New York is still/ already on summer hours.
    assert in_killzone(utc_stamp(day, 14))
    assert in_killzone(utc_stamp(day, 16))
    assert not in_killzone(utc_stamp(day, 17))
    assert not in_killzone(utc_stamp(day, 13))


def test_msk_windows_match_the_config() -> None:
    assert kz_windows_msk(LONDON, SUMMER) == (9, 12)
    assert kz_windows_msk(LONDON, WINTER) == (10, 13)
    assert kz_windows_msk(NY, SUMMER) == (14, 17)
    assert kz_windows_msk(NY, WINTER) == (15, 18)
    assert kz_windows_msk(PRE_LONDON, SUMMER) == (7, 9)
    assert kz_windows_msk(PRE_LONDON, WINTER) == (7, 9)  # fixed all year


def test_utc_windows_are_the_same_table_shifted_by_three_hours() -> None:
    """The Э3' session maps must read their UTC hours from here, not re-declare them."""
    assert kz_window_utc_hours(LONDON, SUMMER_DAY) == PROD_LONDON_SUMMER_UTC
    assert kz_window_utc_hours(NY, WINTER_DAY) == PROD_NY_WINTER_UTC
    assert kz_window_utc_hours(LONDON, WINTER_DAY) == (7, 10)
    assert kz_window_utc_hours(NY, SUMMER_DAY) == (11, 14)
    assert kz_window_utc_hours(PRE_LONDON, WINTER_DAY) == (4, 6)
    config = SessionConfig(london_summer_msk=(8, 11))
    assert kz_window_utc_hours(LONDON, SUMMER_DAY, config) == (5, 8)


@pytest.mark.parametrize("msk_hour", [9, 10, 11])
def test_london_summer_hours_are_gated_in_summer(msk_hour: int) -> None:
    assert in_killzone(utc_stamp(SUMMER_DAY, msk_hour))


@pytest.mark.parametrize("msk_hour", [8, 12, 13, 18])
def test_london_summer_hours_are_rejected_in_summer(msk_hour: int) -> None:
    # 12/13 fall between the London and the NY windows, 8 is before London, 18 after NY.
    assert not in_killzone(utc_stamp(SUMMER_DAY, msk_hour))


@pytest.mark.parametrize("msk_hour", [10, 11, 12])
def test_london_winter_hours_are_gated_in_winter(msk_hour: int) -> None:
    assert in_killzone(utc_stamp(WINTER_DAY, msk_hour))


@pytest.mark.parametrize("msk_hour", [9, 13, 14])
def test_london_summer_hours_are_rejected_in_winter(msk_hour: int) -> None:
    """A gate that always used the summer windows fails here (the Э1'.1 mutation)."""
    assert not in_killzone(utc_stamp(WINTER_DAY, msk_hour))


@pytest.mark.parametrize("msk_hour", [14, 15, 16])
def test_ny_summer_hours_are_gated_in_summer(msk_hour: int) -> None:
    assert in_killzone(utc_stamp(SUMMER_DAY, msk_hour))


@pytest.mark.parametrize("msk_hour", [13, 17, 18])
def test_ny_summer_hours_are_rejected_in_summer(msk_hour: int) -> None:
    assert not in_killzone(utc_stamp(SUMMER_DAY, msk_hour))


@pytest.mark.parametrize("msk_hour", [15, 16, 17])
def test_ny_winter_hours_are_gated_in_winter(msk_hour: int) -> None:
    assert in_killzone(utc_stamp(WINTER_DAY, msk_hour))


@pytest.mark.parametrize("msk_hour", [14, 18])
def test_ny_summer_hours_are_rejected_in_winter(msk_hour: int) -> None:
    """A gate that always used the summer windows fails here too."""
    assert not in_killzone(utc_stamp(WINTER_DAY, msk_hour))


@pytest.mark.parametrize("msk_hour", range(24))
def test_use_kz_false_lets_every_bar_through(msk_hour: int) -> None:
    disabled = SessionConfig(use_kz=False)
    assert not in_killzone(utc_stamp(SUMMER_DAY, msk_hour), disabled)
    assert not in_killzone(utc_stamp(WINTER_DAY, msk_hour), disabled)


def test_prelondon_is_off_by_default_and_opt_in() -> None:
    assert not in_killzone(utc_stamp(SUMMER_DAY, 7))
    assert not in_killzone(utc_stamp(SUMMER_DAY, 8))
    opted_in = SessionConfig(prelondon=True)
    assert in_killzone(utc_stamp(SUMMER_DAY, 7), opted_in)
    assert in_killzone(utc_stamp(SUMMER_DAY, 8), opted_in)
    assert in_killzone(utc_stamp(WINTER_DAY, 8), opted_in)  # fixed MSK window
    assert not in_killzone(utc_stamp(SUMMER_DAY, 6), opted_in)
    assert not in_killzone(utc_stamp(WINTER_DAY, 9), opted_in)  # London winter starts at 10


def test_non_utc_stamps_are_converted_before_the_gate() -> None:
    msk_stamp = pd.Timestamp(datetime(2026, 7, 15, 9, tzinfo=MSK))
    assert in_killzone(msk_stamp)
    assert in_killzone(msk_stamp.tz_convert("UTC"))
    assert not in_killzone(pd.Timestamp(datetime(2026, 7, 15, 8, tzinfo=MSK)))


def test_unknown_killzone_or_season_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported killzone"):
        season_for("asia", SUMMER_DAY)
    with pytest.raises(ValueError, match="unsupported killzone"):
        kz_windows_msk("asia", SUMMER)
    with pytest.raises(ValueError, match="unsupported season"):
        kz_windows_msk(LONDON, "monsoon")


def test_session_config_rejects_malformed_windows() -> None:
    with pytest.raises(ValueError, match="london_summer_msk"):
        SessionConfig(london_summer_msk=(12, 9))
    with pytest.raises(ValueError, match="ny_winter_msk"):
        SessionConfig(ny_winter_msk=(18, 15))
    with pytest.raises(ValueError, match="prelondon_msk"):
        SessionConfig(prelondon_msk=(9, 9))


# --- the vector form of the killzone gate (the Э4' entry chain) ---------------
# Ranges that straddle the EU/US DST switches of 2026 (EU: last Sunday of March /
# October, US: second Sunday of March / first Sunday of November), so a window taken
# from these dates always contains at least two seasons.
SEASON_SWITCH_WINDOWS = [
    (date(2026, 3, 6), date(2026, 3, 31)),
    (date(2026, 10, 22), date(2026, 11, 3)),
]


def day_bars(day: date) -> pd.Series:
    """All 96 M15 open times of one UTC day, as an entry frame would hand them over."""
    start = pd.Timestamp(datetime(day.year, day.month, day.day, tzinfo=UTC))
    return pd.Series(pd.date_range(start, periods=96, freq="15min"))


def test_killzone_mask_matches_the_scalar_gate() -> None:
    """Both forms of the gate read one table: they may never disagree on any bar.

    The window straddles the real DST switches - EU summer runs from the last Sunday
    of March to the last Sunday of October, US summer from the second Sunday of March
    to the first Sunday of November - because inside a single season a "one season
    for the whole frame" bug is invisible.
    """
    for first_day, last_day in SEASON_SWITCH_WINDOWS:
        days = [first_day + timedelta(days=shift) for shift in range((last_day - first_day).days + 1)]
        window = pd.concat([day_bars(day) for day in days])
        for cfg in (SessionConfig(), SessionConfig(prelondon=True)):
            mask = killzone_mask(window, cfg)
            assert mask.tolist() == [in_killzone(stamp, cfg) for stamp in window]


def test_killzone_mask_keeps_only_the_session_windows() -> None:
    """Summer: London 09-12 MSK = 06-09 UTC and NY 14-17 MSK = 11-14 UTC."""
    stamps = day_bars(SUMMER_DAY)
    mask = killzone_mask(stamps)
    kept_hours = {stamp.hour for stamp, kept in zip(stamps, mask) if kept}
    assert kept_hours == {6, 7, 8, 11, 12, 13}
    assert not mask.iloc[6 * 4 - 1]  # 05:45 UTC, before the London window
    assert not mask.iloc[9 * 4]  # 09:00 UTC: windows are half-open, London is over


def test_killzone_mask_rejects_the_neighbouring_season_window() -> None:
    """Winter: London 10-13 MSK = 07-10 UTC and NY 15-18 MSK = 12-15 UTC."""
    stamps = day_bars(WINTER_DAY)
    mask = killzone_mask(stamps)
    kept_hours = {stamp.hour for stamp, kept in zip(stamps, mask) if kept}
    assert kept_hours == {7, 8, 9, 12, 13, 14}
    assert not mask.iloc[6 * 4]  # 06:00 UTC is summer London: the season must be read


def test_killzone_mask_is_empty_when_the_gate_is_off() -> None:
    stamps = day_bars(SUMMER_DAY)
    mask = killzone_mask(stamps, SessionConfig(use_kz=False))
    assert not mask.any()
    assert mask.index.equals(stamps.index)


def test_killzone_mask_keeps_the_input_index_and_dtype() -> None:
    stamps = day_bars(SUMMER_DAY)
    stamps.index = pd.RangeIndex(1000, 1000 + len(stamps))
    mask = killzone_mask(stamps)
    assert mask.dtype == bool
    assert mask.index.equals(stamps.index)


# --- C6 trading hours: the broker's week, not a killzone ----------------------
#
# The mask is built from the MSK week, so the dates below are derived from a known
# weekday instead of being written out: ``MONDAY`` is always a Monday whatever the
# calendar does.
MONDAY = SUMMER_DAY - timedelta(days=SUMMER_DAY.weekday())
TUESDAY = MONDAY + timedelta(days=1)
FRIDAY = MONDAY + timedelta(days=4)
SATURDAY = MONDAY + timedelta(days=5)
SUNDAY = MONDAY + timedelta(days=6)


def msk_stamp(day: date, hour_msk: int, minute_msk: int = 0) -> pd.Timestamp:
    """Return the UTC stamp of the bar opening at ``hour_msk:minute_msk`` Moscow time."""
    local = datetime(day.year, day.month, day.day, hour_msk, minute_msk, tzinfo=MSK)
    return pd.Timestamp(local.astimezone(UTC))


def test_trading_hours_open_monday_close_friday() -> None:
    """C6: 02:00 MSK Monday opens the week, 23:55 MSK Friday closes it.

    The Friday close is the boundary a weekend-blind mask gets wrong: it would keep
    23:55-23:59 MSK Friday (and, worse, the whole of Saturday night) tradable.
    """
    assert not in_trading_hours(msk_stamp(MONDAY, 1, 45))
    assert in_trading_hours(msk_stamp(MONDAY, 2, 0))
    assert in_trading_hours(msk_stamp(FRIDAY, 23, 45))
    assert not in_trading_hours(msk_stamp(FRIDAY, 23, 55))
    assert not in_trading_hours(msk_stamp(FRIDAY, 23, 59))
    assert not in_trading_hours(msk_stamp(SUNDAY, 23, 59))
    assert not in_trading_hours(msk_stamp(MONDAY, 0, 0))


@pytest.mark.parametrize("day", [SATURDAY, SUNDAY])
@pytest.mark.parametrize("hour_msk", [0, 6, 12, 23])
def test_weekend_is_closed(day: date, hour_msk: int) -> None:
    assert not in_trading_hours(msk_stamp(day, hour_msk, 55))


@pytest.mark.parametrize("hour_msk", [0, 2, 12, 23])
def test_midweek_hours_are_open(hour_msk: int) -> None:
    """Only the Monday morning and the Friday evening are part of the week off."""
    assert in_trading_hours(msk_stamp(TUESDAY, hour_msk, 0))


def test_mask_agrees_with_the_scalar_gate_over_a_whole_week() -> None:
    """The vectorised mask is not a second implementation of the rule."""
    stamps = pd.Series(pd.date_range(msk_stamp(MONDAY, 0, 0), periods=7 * 24 * 4, freq="15min"))
    mask = alfa_trading_mask(stamps)
    assert mask.index.equals(stamps.index)
    assert mask.tolist() == [in_trading_hours(stamp) for stamp in stamps]


def test_mask_of_an_empty_frame_is_empty() -> None:
    """The entry frame of a day off is empty and must not raise."""
    empty = pd.Series([], dtype="datetime64[ns, UTC]")
    mask = alfa_trading_mask(empty)
    assert mask.empty
    assert mask.dtype == bool


def test_mask_follows_the_configured_boundaries() -> None:
    early = SessionConfig(session_open_msk=(1, 0), session_close_msk=(22, 0))
    assert in_trading_hours(msk_stamp(MONDAY, 1, 0), early)
    assert not in_trading_hours(msk_stamp(MONDAY, 0, 45), early)
    assert in_trading_hours(msk_stamp(FRIDAY, 21, 45), early)
    assert not in_trading_hours(msk_stamp(FRIDAY, 22, 0), early)


def test_session_config_rejects_malformed_trading_hours() -> None:
    with pytest.raises(ValueError, match="session_open_msk hour"):
        SessionConfig(session_open_msk=(24, 0))
    with pytest.raises(ValueError, match="session_close_msk minute"):
        SessionConfig(session_close_msk=(23, 60))
