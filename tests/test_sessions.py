"""Tests for :mod:`smc_zero.indicators.sessions`: the seasonal killzone gate.

The gate is the one place where C1's seasonality lives, so it is pinned in three
directions: the MSK windows of the config, their UTC re-expression (which the Э3'
session maps must reuse) and the actual bar-level gate on dates where the EU and
US calendars disagree.
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
    in_killzone,
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
