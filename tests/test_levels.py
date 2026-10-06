"""Level tests: period ranges, seasonal session maps, availability gates, lifecycle, leakage.

Synthetic OHLCV only (rules 1 and 3).  Every day helper builds one UTC day of 96 M15 bars
(``00:00`` .. ``23:45``), so the windows that matter are always fully present in the frame:

* :func:`_flat_day` keeps one high/low for the whole day - the shape the period-range tests
  need, because then ``PDH`` of a day is simply the previous day's constant high;
* :func:`_hourly_day` makes high and low grow with the UTC hour (``high = 100 + h``), so the
  extreme of a session window is fixed by the window's first and last hour.  That is what makes
  a wrong season table *and* an off-by-one window visible instead of a coincidence.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from smc_zero.config import LevelConfig, SessionConfig
from smc_zero.data_loader import IS_CLOSED_COLUMN, TIMESTAMP_COLUMN
from smc_zero.indicators.levels import (
    BROKEN_AT_COLUMN,
    IDH_DYN_COLUMN,
    IDL_DYN_COLUMN,
    LEVEL_AVAILABLE_AT_COLUMN,
    LEVEL_COLUMNS,
    LEVEL_DATE_COLUMN,
    LEVEL_IS_UPPER_COLUMN,
    LEVEL_NAME_COLUMN,
    LEVEL_NAMES,
    LEVEL_PRICE_COLUMN,
    LEVEL_SOURCE_WINDOW_COLUMN,
    dynamic_idl_idh,
    fresh_at,
    level_lifecycle,
    static_levels,
)
from smc_zero.utils.time import MSK_UTC_OFFSET_HOURS

M15_FREQ = "15min"
BARS_PER_DAY = 96
BASE = 100.0  # high of hour 0 in _hourly_day; the low of that hour is BASE - SPREAD
SPREAD = 10.0


def _bars(
    stamps: pd.DatetimeIndex,
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float] | None = None,
) -> pd.DataFrame:
    """One M15 frame; ``close`` defaults to the middle of every bar's range."""
    high_array = np.asarray(high, dtype="float64")
    low_array = np.asarray(low, dtype="float64")
    close_array = (
        (high_array + low_array) / 2 if close is None else np.asarray(close, dtype="float64")
    )
    # the dict constructor places ``stamps`` positionally and keeps its UTC dtype; a
    # ``DataFrame.insert`` of a Series would align on the index instead and yield NaT here
    return pd.DataFrame(
        {
            TIMESTAMP_COLUMN: stamps,
            "open": close_array,
            "high": high_array,
            "low": low_array,
            "close": close_array,
        }
    )


def _day_stamps(day: str) -> pd.DatetimeIndex:
    """The 96 M15 labels of one UTC day."""
    return pd.date_range(f"{day} 00:00", periods=BARS_PER_DAY, freq=M15_FREQ, tz="UTC")


def _flat_day(day: str, high: float, low: float) -> pd.DataFrame:
    """M15 bars of one day carrying the same range on every bar."""
    stamps = _day_stamps(day)
    return _bars(stamps, [high] * BARS_PER_DAY, [low] * BARS_PER_DAY)


def _hourly_day(day: str) -> pd.DataFrame:
    """M15 bars whose high/low grow by one per UTC hour."""
    hours = np.repeat(np.arange(24), 4)
    return _bars(_day_stamps(day), BASE + hours, BASE - SPREAD + hours)


def _frame(*days: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(days, ignore_index=True)


def _stamp(value: str) -> pd.Timestamp:
    """A UTC stamp from a naive ``"YYYY-MM-DD HH:MM"`` literal."""
    return pd.Timestamp(value, tz="UTC")


def _single_level(name: str, price: float, is_upper: bool, available_at: str) -> pd.DataFrame:
    """One hand-written level instance - a map a strategy could hand to the lifecycle."""
    return pd.DataFrame(
        {
            LEVEL_NAME_COLUMN: [name],
            LEVEL_DATE_COLUMN: [pd.Timestamp("2026-01-05")],
            LEVEL_PRICE_COLUMN: [price],
            LEVEL_IS_UPPER_COLUMN: [is_upper],
            LEVEL_AVAILABLE_AT_COLUMN: [_stamp(available_at)],
            LEVEL_SOURCE_WINDOW_COLUMN: ["manual"],
        }
    )


def _row(levels: pd.DataFrame, name: str, day: str) -> pd.Series:
    """The single ``name`` instance dated ``day`` (fails loudly on 0 or many rows)."""
    selected = levels[
        (levels[LEVEL_NAME_COLUMN] == name) & (levels[LEVEL_DATE_COLUMN] == pd.Timestamp(day))
    ]
    assert len(selected) == 1, f"expected one {name} row for {day}, got {len(selected)}"
    return selected.iloc[0]


def _visible(levels: pd.DataFrame, t: str) -> pd.DataFrame:
    """The rows a consumer may use at instant ``t`` (the level exists by then)."""
    visible = levels[levels[LEVEL_AVAILABLE_AT_COLUMN] <= _stamp(t)]
    return visible.sort_values([LEVEL_NAME_COLUMN, LEVEL_DATE_COLUMN]).reset_index(drop=True)


def _has(levels: pd.DataFrame, name: str, day: str) -> bool:
    """Whether ``name`` dated ``day`` is part of ``levels``."""
    return bool(
        ((levels[LEVEL_NAME_COLUMN] == name) & (levels[LEVEL_DATE_COLUMN] == pd.Timestamp(day))).any()
    )


def _is_visible(levels: pd.DataFrame, name: str, day: str, t: str) -> bool:
    """Whether ``name`` of ``day`` exists for a consumer at instant ``t``."""
    return _has(_visible(levels, t), name, day)


def _one_bar_before(t: str) -> str:
    """The label of the M15 bar right before ``t``, as a naive literal."""
    return (_stamp(t) - pd.Timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M")


def _window_prices(window_msk: tuple[int, int]) -> tuple[float, float, str]:
    """Expected (high, low, closing ``HH:MM``) of an hourly ramp inside an MSK window."""
    start_utc = window_msk[0] - MSK_UTC_OFFSET_HOURS
    end_utc = window_msk[1] - MSK_UTC_OFFSET_HOURS
    return BASE + end_utc - 1, BASE - SPREAD + start_utc, f"{end_utc:02d}:00"


def _assert_window_levels(
    levels: pd.DataFrame, day: str, prefix: str, window_msk: tuple[int, int]
) -> None:
    """Pin both levels of a session window: price, side, gate instant and window text."""
    high_price, low_price, closing = _window_prices(window_msk)
    for name, price, is_upper in ((f"{prefix}H", high_price, True), (f"{prefix}L", low_price, False)):
        row = _row(levels, name, day)
        assert row[LEVEL_PRICE_COLUMN] == pytest.approx(price)
        assert bool(row[LEVEL_IS_UPPER_COLUMN]) is is_upper
        assert row[LEVEL_AVAILABLE_AT_COLUMN] == _stamp(f"{day} {closing}")
        assert row[LEVEL_DATE_COLUMN] == pd.Timestamp(day)
        assert row[LEVEL_SOURCE_WINDOW_COLUMN] == (
            f"{prefix.lower()}_msk[{window_msk[0]},{window_msk[1]})"
        )


# --------------------------------------------------------------------------------------
# (a) previous-period ranges: PDH/PDL, PWH/PWL, PMH/PML
# --------------------------------------------------------------------------------------

# Week 2025-52 ends on Wednesday 2025-12-31; the frame then skips the New Year holidays and
# restarts in week 2026-00 on Monday 2026-01-05.  For that Monday the previous day, week and
# month are all 2025-12-31 (three calendar days back), while the current periods' high (120)
# differs from theirs (112) - so any range leaking from the current period is visible.
GAP_DAYS = ("2025-12-29", "2025-12-30", "2025-12-31")
GAP_HIGHS = (110.0, 111.0, 112.0)
GAP_LOWS = (90.0, 91.0, 92.0)
RESTART_DAYS = ("2026-01-05", "2026-01-06", "2026-01-07")
PERIOD_LEVELS = ("PDH", "PDL", "PWH", "PWL", "PMH", "PML")


def _gap_frame() -> pd.DataFrame:
    """Six trading days: a week and a month boundary plus a multi-day holiday gap."""
    return _frame(
        *[
            _flat_day(day, high, low)
            for day, high, low in zip(GAP_DAYS, GAP_HIGHS, GAP_LOWS, strict=True)
        ],
        *[_flat_day(day, 120.0, 100.0) for day in RESTART_DAYS],
    )


def test_period_ranges_take_the_previous_period_only() -> None:
    frame = _gap_frame()
    levels = static_levels(frame)

    # long format, one row per (name, date), and every name of the map is produced
    assert list(levels.columns) == list(LEVEL_COLUMNS)
    assert not levels.duplicated([LEVEL_NAME_COLUMN, LEVEL_DATE_COLUMN]).any()
    assert set(levels[LEVEL_NAME_COLUMN]) == set(LEVEL_NAMES)

    # the first day of the frame has no earlier day, week or month at all
    for name in PERIOD_LEVELS:
        assert not _has(levels, name, "2025-12-29")

    # PDH/PDL: the maximal earlier day *present in the data*, no calendar inference
    assert _row(levels, "PDH", "2025-12-30")[LEVEL_PRICE_COLUMN] == 110.0
    assert _row(levels, "PDL", "2025-12-30")[LEVEL_PRICE_COLUMN] == 90.0
    monday_high = _row(levels, "PDH", "2026-01-05")
    assert monday_high[LEVEL_PRICE_COLUMN] == 112.0  # 2025-12-31, three calendar days back
    assert monday_high[LEVEL_PRICE_COLUMN] != 120.0  # not the current day's own high
    assert monday_high[LEVEL_SOURCE_WINDOW_COLUMN] == "prev_day=2025-12-31"
    assert monday_high[LEVEL_AVAILABLE_AT_COLUMN] == _stamp("2026-01-05 00:00")
    assert monday_high[LEVEL_DATE_COLUMN] == pd.Timestamp("2026-01-05")
    assert bool(monday_high[LEVEL_IS_UPPER_COLUMN])
    assert not bool(_row(levels, "PDL", "2026-01-05")[LEVEL_IS_UPPER_COLUMN])
    assert _row(levels, "PDL", "2026-01-05")[LEVEL_PRICE_COLUMN] == 92.0

    # PWH/PWL: one instance per week, fed by the previous week only
    assert len(levels[levels[LEVEL_NAME_COLUMN] == "PWH"]) == 1
    week_high = _row(levels, "PWH", "2026-01-05")
    assert week_high[LEVEL_PRICE_COLUMN] == 112.0  # not the current week's 120
    assert _row(levels, "PWL", "2026-01-05")[LEVEL_PRICE_COLUMN] == 90.0
    assert week_high[LEVEL_SOURCE_WINDOW_COLUMN] == (
        f"prev_week={pd.Timestamp('2025-12-29').strftime('%Y-%W')}"
    )
    assert week_high[LEVEL_AVAILABLE_AT_COLUMN] == _stamp("2026-01-05 00:00")

    # PMH/PML: one instance per month, fed by the previous month only
    assert len(levels[levels[LEVEL_NAME_COLUMN] == "PMH"]) == 1
    month_high = _row(levels, "PMH", "2026-01-05")
    assert month_high[LEVEL_PRICE_COLUMN] == 112.0
    assert _row(levels, "PML", "2026-01-05")[LEVEL_PRICE_COLUMN] == 90.0
    assert month_high[LEVEL_SOURCE_WINDOW_COLUMN] == "prev_month=2025-12"

    # every gate is a bar open that exists in the entry frame
    assert set(levels[LEVEL_AVAILABLE_AT_COLUMN]) <= set(frame[TIMESTAMP_COLUMN])


# --------------------------------------------------------------------------------------
# (b) seasonal session windows
# --------------------------------------------------------------------------------------

# MSK windows expected from the season table (C1): London follows the EU DST calendar, New York
# the US one.  On 2026-03-16 (US summer since 2026-03-08, EU winter until 2026-03-29) and on
# 2026-10-28 (EU winter since 2026-10-25, US summer until 2026-11-01) the two sessions are on
# different seasons, so a single shared flag cannot reproduce both rows.
SESSION_CASES = (
    ("2026-01-15", (10, 13), (15, 18)),  # both sessions on winter hours
    ("2026-07-15", (9, 12), (14, 17)),  # both on summer hours
    ("2026-03-16", (10, 13), (14, 17)),  # EU winter, US already summer
    ("2026-10-28", (10, 13), (14, 17)),  # EU winter again, US still summer
)
SESSION_DAYS = tuple(case[0] for case in SESSION_CASES)


@pytest.mark.parametrize(("day", "london", "ny"), SESSION_CASES)
def test_session_maps_follow_the_season_of_their_own_calendar(
    day: str, london: tuple[int, int], ny: tuple[int, int]
) -> None:
    levels = static_levels(_hourly_day(day))
    _assert_window_levels(levels, day, "London", london)
    _assert_window_levels(levels, day, "NY", ny)


@pytest.mark.parametrize("day", SESSION_DAYS)
def test_asian_map_is_a_fixed_utc_window(day: str) -> None:
    """Asia does not switch clocks, so its range is the same on every date."""
    levels = static_levels(_hourly_day(day))
    assert _row(levels, "AsianH", day)[LEVEL_PRICE_COLUMN] == BASE + 7.0  # hours 00..07 UTC
    assert _row(levels, "AsianL", day)[LEVEL_PRICE_COLUMN] == BASE - SPREAD
    assert _row(levels, "AsianH", day)[LEVEL_AVAILABLE_AT_COLUMN] == _stamp(f"{day} 08:00")
    assert _row(levels, "AsianH", day)[LEVEL_SOURCE_WINDOW_COLUMN] == "asia_utc[0,8)"


@pytest.mark.parametrize("window", [(0, 8), (1, 7)])
def test_asian_window_comes_from_the_config(window: tuple[int, int]) -> None:
    """Widening or shifting the Asian window moves the levels with it."""
    levels = static_levels(_hourly_day("2026-01-15"), LevelConfig(asian_window_utc=window))
    start, end = window
    assert _row(levels, "AsianH", "2026-01-15")[LEVEL_PRICE_COLUMN] == BASE + end - 1
    assert _row(levels, "AsianL", "2026-01-15")[LEVEL_PRICE_COLUMN] == BASE - SPREAD + start
    assert _row(levels, "AsianH", "2026-01-15")[LEVEL_SOURCE_WINDOW_COLUMN] == (
        f"asia_utc[{start},{end})"
    )


def test_session_maps_read_the_shared_killzone_table() -> None:
    """A customised gate table has to move the maps too - there is no second window list."""
    custom = SessionConfig(london_summer_msk=(8, 11), london_winter_msk=(11, 14))
    levels = static_levels(_hourly_day("2026-01-15"), session_cfg=custom)
    high_price, low_price, closing = _window_prices((11, 14))  # 11-14 MSK == 08-11 UTC
    assert (high_price, low_price, closing) == (110.0, 98.0, "11:00")
    london_high = _row(levels, "LondonH", "2026-01-15")
    assert london_high[LEVEL_PRICE_COLUMN] == pytest.approx(high_price)
    assert _row(levels, "LondonL", "2026-01-15")[LEVEL_PRICE_COLUMN] == pytest.approx(low_price)
    assert london_high[LEVEL_SOURCE_WINDOW_COLUMN] == "london_msk[11,14)"
    assert london_high[LEVEL_AVAILABLE_AT_COLUMN] == _stamp("2026-01-15 11:00")
    # the untouched New York entry still comes from the same table
    assert _row(levels, "NYH", "2026-01-15")[LEVEL_SOURCE_WINDOW_COLUMN] == "ny_msk[15,18)"


# --------------------------------------------------------------------------------------
# (c) availability gates
# --------------------------------------------------------------------------------------


def test_session_levels_are_invisible_until_their_window_closes() -> None:
    levels = static_levels(_frame(_hourly_day("2026-01-15"), _hourly_day("2026-01-16")))

    # 07:45 is the last M15 label inside the London winter window (UTC 07..10)
    assert not _is_visible(levels, "LondonH", "2026-01-15", "2026-01-15 07:45")
    assert not _is_visible(levels, "LondonL", "2026-01-15", "2026-01-15 07:45")
    # the next day's London range does not exist yet at the end of the previous day
    assert not _is_visible(levels, "LondonH", "2026-01-16", "2026-01-15 23:45")
    # from the window's close the range is a fact
    assert _is_visible(levels, "LondonH", "2026-01-15", "2026-01-15 10:00")
    assert (
        _row(_visible(levels, "2026-01-15 10:00"), "LondonH", "2026-01-15")[LEVEL_PRICE_COLUMN]
        == 109.0
    )

    # every session gate is the first bar *after* its window, not the window's last bar
    gates = (
        ("AsianH", "08:00"),
        ("AsianL", "08:00"),
        ("LondonH", "10:00"),
        ("LondonL", "10:00"),
        ("NYH", "15:00"),
        ("NYL", "15:00"),
    )
    for name, gate in gates:
        assert _row(levels, name, "2026-01-15")[LEVEL_AVAILABLE_AT_COLUMN] == _stamp(
            f"2026-01-15 {gate}"
        )
        assert _is_visible(levels, name, "2026-01-15", f"2026-01-15 {gate}")
        assert not _is_visible(levels, name, "2026-01-15", _one_bar_before(f"2026-01-15 {gate}"))


def test_previous_period_levels_appear_with_the_first_bar_of_their_period() -> None:
    frame = _frame(_flat_day("2025-12-31", 112.0, 92.0), _flat_day("2026-01-05", 120.0, 100.0))
    levels = static_levels(frame)
    first_bar = "2026-01-05 00:00"
    for name in PERIOD_LEVELS:
        assert _row(levels, name, "2026-01-05")[LEVEL_AVAILABLE_AT_COLUMN] == _stamp(first_bar)
        assert _is_visible(levels, name, "2026-01-05", first_bar)
        # one bar earlier the new period does not exist yet
        assert not _is_visible(levels, name, "2026-01-05", "2025-12-31 23:45")
    # ... and the session levels of that day follow the same rule
    assert not _is_visible(levels, "AsianH", "2026-01-05", "2026-01-05 07:45")
    assert _is_visible(levels, "AsianH", "2026-01-05", "2026-01-05 08:00")


# --------------------------------------------------------------------------------------
# (d) intraday IDL/IDH
# --------------------------------------------------------------------------------------


def _id_frame() -> pd.DataFrame:
    """Two days of four and three M15 bars, with falling lows and rising highs."""
    stamps = pd.DatetimeIndex(
        pd.to_datetime(
            [
                "2026-01-05 00:00",
                "2026-01-05 00:15",
                "2026-01-05 00:30",
                "2026-01-05 00:45",
                "2026-01-06 00:00",
                "2026-01-06 00:15",
                "2026-01-06 00:30",
            ],
            utc=True,
        )
    )
    return _bars(
        stamps,
        high=[10.0, 12.0, 11.0, 13.0, 20.0, 22.0, 21.0],
        low=[5.0, 3.0, 4.0, 2.0, 9.0, 7.0, 8.0],
    )


def test_idl_idh_lag_by_one_bar_and_reset_with_every_new_day() -> None:
    frame = _id_frame()
    marked = dynamic_idl_idh(frame)

    # the first bar of a day has no earlier bar, so it carries no intraday extreme
    assert pd.isna(marked[IDL_DYN_COLUMN].iloc[0])
    assert pd.isna(marked[IDH_DYN_COLUMN].iloc[4])
    # afterwards the running extreme of *earlier* bars only: the low of bar 2 (4.0) and the
    # own low of bar 3 (2.0) never enter the value of their own bar
    assert marked[IDL_DYN_COLUMN].iloc[1:4].tolist() == [5.0, 3.0, 3.0]
    assert marked[IDH_DYN_COLUMN].iloc[1:4].tolist() == [10.0, 12.0, 12.0]
    assert marked[IDL_DYN_COLUMN].iloc[5:].tolist() == [9.0, 7.0]
    assert marked[IDH_DYN_COLUMN].iloc[5:].tolist() == [20.0, 22.0]
    # the previous day's extreme (2.0) is not carried over the day boundary
    assert pd.isna(marked[IDL_DYN_COLUMN].iloc[4])
    # the caller's frame is left alone
    assert IDL_DYN_COLUMN not in frame.columns and IDH_DYN_COLUMN not in frame.columns


def test_bars_flagged_as_still_forming_are_never_used() -> None:
    """Rule 2b: a bar the loader marked as unclosed is dropped, never silently included."""
    tail = _id_frame()
    tail[IS_CLOSED_COLUMN] = True
    tail.loc[tail.index[-1], IS_CLOSED_COLUMN] = False
    assert len(dynamic_idl_idh(tail)) == 6

    # a spike on a forming bar of a session window may not become that window's high
    frame = _hourly_day("2026-01-15")
    frame[IS_CLOSED_COLUMN] = True
    spike = frame[TIMESTAMP_COLUMN] == _stamp("2026-01-15 09:45")
    frame.loc[spike, IS_CLOSED_COLUMN] = False
    frame.loc[spike, "high"] = 9_999.0
    forming = static_levels(frame)
    assert _row(forming, "LondonH", "2026-01-15")[LEVEL_PRICE_COLUMN] == 109.0
    # with the same bar closed the spike *is* used - the flag is the whole protection
    closed = static_levels(frame.assign(**{IS_CLOSED_COLUMN: True}))
    assert _row(closed, "LondonH", "2026-01-15")[LEVEL_PRICE_COLUMN] == 9_999.0


# --------------------------------------------------------------------------------------
# (e) fresh / broken lifecycle
# --------------------------------------------------------------------------------------


def _lifecycle_frame() -> pd.DataFrame:
    """Eight M15 bars closing at 99.0, 99.5, 99.9, 97.5, 103.0 and a quiet tail."""
    stamps = pd.date_range("2026-01-05 00:00", periods=8, freq=M15_FREQ, tz="UTC")
    close = [99.0, 99.5, 99.9, 97.5, 103.0, 100.0, 100.0, 100.0]
    return _bars(
        stamps,
        high=[max(value, 100.0) for value in close],
        low=[min(value, 97.0) for value in close],
        close=close,
    )


def _one_hundred_pair() -> pd.DataFrame:
    """An upper and a lower level at 100.0, both known from the second bar on."""
    return pd.concat(
        [
            _single_level("PDH", 100.0, True, "2026-01-05 00:15"),
            _single_level("PDL", 100.0, False, "2026-01-05 00:15"),
        ],
        ignore_index=True,
    )


def test_lifecycle_breaks_on_the_first_close_beyond_the_buffer() -> None:
    frame = _lifecycle_frame()
    lifecycle = level_lifecycle(_one_hundred_pair(), frame)

    # 99.0 / 99.5 / 99.9 do not clear the 2.0 buffer while 103.0 does: the bar opening at
    # 01:00 is the first one, so the break is known at its *close time* 01:15
    assert _row(lifecycle, "PDH", "2026-01-05")[BROKEN_AT_COLUMN] == _stamp("2026-01-05 01:15")
    # the lower level is swept by the 97.5 close of the bar opening at 00:45
    assert _row(lifecycle, "PDL", "2026-01-05")[BROKEN_AT_COLUMN] == _stamp("2026-01-05 01:00")

    # a level that is never cleared stays unbroken
    never = level_lifecycle(_single_level("PWL", 60.0, False, "2026-01-05 00:15"), frame)
    assert pd.isna(never[BROKEN_AT_COLUMN].iloc[0])
    # the tail bars close exactly at 100.0: touching a level is not breaking it
    touching = level_lifecycle(_single_level("PDH", 100.0, True, "2026-01-05 02:00"), frame)
    assert pd.isna(touching[BROKEN_AT_COLUMN].iloc[0])

    # a zero buffer breaks one bar earlier, but never on a bar before the level exists
    loose = level_lifecycle(_one_hundred_pair(), frame, LevelConfig(break_buffer_pip=0.0))
    assert _row(loose, "PDL", "2026-01-05")[BROKEN_AT_COLUMN] == _stamp("2026-01-05 00:30")
    assert _row(loose, "PDH", "2026-01-05")[BROKEN_AT_COLUMN] == _stamp("2026-01-05 01:15")


def test_lifecycle_ignores_bars_that_precede_the_level() -> None:
    frame = _lifecycle_frame()
    # the same lower level, but only known from 01:00: the 97.5 close of 00:45 lies before it
    # and the later closes (103.0, then 100.0) are not low enough, so nothing breaks it
    late = level_lifecycle(_single_level("PDL", 100.0, False, "2026-01-05 01:00"), frame)
    assert pd.isna(late[BROKEN_AT_COLUMN].iloc[0])


def test_fresh_at_ends_when_the_break_becomes_known() -> None:
    frame = _lifecycle_frame()
    row = level_lifecycle(_single_level("PDH", 100.0, True, "2026-01-05 00:15"), frame).iloc[0]

    def fresh(instant: str) -> bool:
        return fresh_at(row, _stamp(instant))

    assert not fresh("2026-01-05 00:00")  # the level does not exist yet
    assert fresh("2026-01-05 00:15")
    assert fresh("2026-01-05 01:00")  # the breaking bar itself: its close is not known yet
    assert not fresh("2026-01-05 01:15")  # from the break's close time on it is stale
    assert not fresh("2026-01-05 23:45")

    # an unbroken level stays fresh forever, and so does a row without the column at all
    unbroken = level_lifecycle(_single_level("PWL", 60.0, False, "2026-01-05 00:15"), frame).iloc[0]
    assert fresh_at(unbroken, _stamp("2026-01-05 23:45"))
    plain = {LEVEL_AVAILABLE_AT_COLUMN: _stamp("2026-01-05 00:15")}
    assert fresh_at(plain, _stamp("2026-01-05 00:30"))
    assert not fresh_at(plain, _stamp("2026-01-05 00:00"))


# --------------------------------------------------------------------------------------
# (f) look-ahead: a tamper may move only what is not visible yet
# --------------------------------------------------------------------------------------


def test_a_tamper_inside_a_forming_window_stays_invisible() -> None:
    """Mutation: raise a high *inside* the current London window before it has closed."""
    frame = _frame(_hourly_day("2026-01-15"), _hourly_day("2026-01-16"))
    tampered = frame.copy()
    inside = tampered[TIMESTAMP_COLUMN] == _stamp("2026-01-15 09:00")
    tampered.loc[inside, "high"] = 9_999.0
    assert bool(inside.any())  # the tampered bar really is in the frame

    # at 07:45 the window is still running, so the tampered bar changes nothing visible
    at_bar = "2026-01-15 07:45"
    pd.testing.assert_frame_equal(
        _visible(static_levels(frame), at_bar), _visible(static_levels(tampered), at_bar)
    )
    # the tamper does move that day's London high - but only once the window has closed
    assert _row(static_levels(tampered), "LondonH", "2026-01-15")[LEVEL_PRICE_COLUMN] == 9_999.0
    before = _visible(static_levels(frame), "2026-01-15 10:00")
    after = _visible(static_levels(tampered), "2026-01-15 10:00")
    assert _row(before, "LondonH", "2026-01-15")[LEVEL_PRICE_COLUMN] == 109.0
    assert _row(after, "LondonH", "2026-01-15")[LEVEL_PRICE_COLUMN] == 9_999.0


def test_a_tamper_of_the_current_day_does_not_move_its_own_pdh() -> None:
    frame = _frame(
        _flat_day("2026-01-05", 110.0, 90.0),
        _flat_day("2026-01-06", 120.0, 100.0),
        _flat_day("2026-01-07", 130.0, 105.0),
    )
    tampered = frame.copy()
    tampered.loc[tampered[TIMESTAMP_COLUMN] == _stamp("2026-01-06 20:00"), "high"] = 9_999.0

    at_bar = "2026-01-06 12:00"
    pd.testing.assert_frame_equal(
        _visible(static_levels(frame), at_bar), _visible(static_levels(tampered), at_bar)
    )
    # the day's own high is never its own PDH - before and after the tamper
    assert _row(static_levels(tampered), "PDH", "2026-01-06")[LEVEL_PRICE_COLUMN] == 110.0
    # the tamper is not swallowed either: it becomes the *next* day's PDH ...
    assert _row(static_levels(frame), "PDH", "2026-01-07")[LEVEL_PRICE_COLUMN] == 120.0
    assert _row(static_levels(tampered), "PDH", "2026-01-07")[LEVEL_PRICE_COLUMN] == 9_999.0
    # ... and only from that day's first bar on
    assert not _is_visible(static_levels(tampered), "PDH", "2026-01-07", "2026-01-06 23:45")
    assert _is_visible(static_levels(tampered), "PDH", "2026-01-07", "2026-01-07 00:00")


def test_a_tamper_after_the_break_cannot_move_broken_at() -> None:
    level = _single_level("PDH", 100.0, True, "2026-01-05 00:15")
    base = level_lifecycle(level, _lifecycle_frame())
    late = _lifecycle_frame()
    late.loc[late[TIMESTAMP_COLUMN] == _stamp("2026-01-05 01:30"), "close"] = 500.0
    after = level_lifecycle(level, late)
    assert base[BROKEN_AT_COLUMN].iloc[0] == _stamp("2026-01-05 01:15")
    assert after[BROKEN_AT_COLUMN].iloc[0] == base[BROKEN_AT_COLUMN].iloc[0]


# --------------------------------------------------------------------------------------
# (g) configuration, schema and grid errors
# --------------------------------------------------------------------------------------


def test_level_config_rejects_malformed_values() -> None:
    with pytest.raises(ValueError, match="break_buffer_pip"):
        LevelConfig(break_buffer_pip=-0.1)
    with pytest.raises(ValueError, match="asian_window_utc"):
        LevelConfig(asian_window_utc=(8, 8))
    with pytest.raises(ValueError, match="week_convention"):
        LevelConfig(week_convention="%W")


def test_levels_require_the_loader_schema() -> None:
    frame = _flat_day("2026-01-05", 110.0, 90.0)
    with pytest.raises(ValueError, match="timestamp"):
        static_levels(frame.drop(columns=[TIMESTAMP_COLUMN]))
    with pytest.raises(ValueError, match="'high'"):
        static_levels(frame.drop(columns=["high"]))
    with pytest.raises(ValueError, match="'low'"):
        static_levels(frame.drop(columns=["low"]))
    with pytest.raises(ValueError, match="'close'"):
        level_lifecycle(
            _single_level("PDH", 100.0, True, "2026-01-05 00:00"), frame.drop(columns=["close"])
        )


def test_lifecycle_validates_the_level_map() -> None:
    frame = _flat_day("2026-01-05", 110.0, 90.0)
    incomplete = _single_level("PDH", 100.0, True, "2026-01-05 00:00").drop(
        columns=[LEVEL_PRICE_COLUMN]
    )
    with pytest.raises(ValueError, match="missing"):
        level_lifecycle(incomplete, frame)


def test_lifecycle_refuses_a_frame_that_is_not_the_entry_timeframe() -> None:
    """A foreign grid would mis-date every broken_at, so it is an error instead of a guess."""
    hourly = _flat_day("2026-01-05", 110.0, 90.0).iloc[::4]
    with pytest.raises(ValueError, match="entry frame"):
        level_lifecycle(_single_level("PDH", 100.0, True, "2026-01-05 00:00"), hourly)


def test_the_lifecycle_of_an_m5_tape_dates_the_break_in_m5_bars() -> None:
    """ОМ-2: the period of ``broken_at`` follows the named entry timeframe, not a hard-coded M15.

    The same twelve five minute bars answer two different questions: named ``"M5"`` they date the
    break on their own grid, and left unnamed they are refused, because the default M15 grid would
    put the break 10 minutes late.  A silent fallback here would mis-date every lifecycle date of
    the H4 -> M15 -> M5 hierarchy.
    """
    stamps = pd.date_range("2026-01-05 00:00", periods=12, freq="5min", tz="UTC")
    close = np.array([99.0, 99.5, 99.9, 103.0, 100.0, 100.0] + [100.0] * 6)
    frame = _bars(stamps, high=np.maximum(close, 100.0), low=np.minimum(close, 97.0), close=close)
    level = _single_level("PDH", 100.0, True, "2026-01-05 00:05")

    on_grid = level_lifecycle(level, frame, timeframe="M5")

    # the bar opening at 00:15 is the first close beyond 102.0, so the break is known at 00:20
    assert on_grid[BROKEN_AT_COLUMN].iloc[0] == _stamp("2026-01-05 00:20")
    with pytest.raises(ValueError, match="entry frame"):
        level_lifecycle(level, frame)


def test_empty_frames_yield_empty_results() -> None:
    empty = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: pd.Series(dtype="datetime64[ns, UTC]"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
        }
    )
    levels = static_levels(empty)
    assert levels.empty and list(levels.columns) == list(LEVEL_COLUMNS)
    marked = dynamic_idl_idh(empty)
    assert marked.empty and IDL_DYN_COLUMN in marked.columns and IDH_DYN_COLUMN in marked.columns
    lifecycle = level_lifecycle(levels, empty)
    assert lifecycle.empty and list(lifecycle.columns) == [*LEVEL_COLUMNS, BROKEN_AT_COLUMN]


