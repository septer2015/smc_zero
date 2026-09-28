"""Liquidity level maps and their lifecycle (SPEC_SMC.md п.3-п.5, Э3').

Three pure functions turn a closed-bar M15 frame into liquidity facts:

* :func:`static_levels` - the long-format map of the daily, weekly and monthly ranges
  (``PDH/PDL``, ``PWH/PWL``, ``PMH/PML``) plus the Asian, London and New York session ranges;
* :func:`dynamic_idl_idh` - prod's intraday running low/high (``idl_dyn`` / ``idh_dyn``);
* :func:`level_lifecycle` - when a level instance got broken, plus :func:`fresh_at`, i.e.
  whether it is still fresh at a given decision instant.

Conventions that shape all of them:

* **Long format.**  A level instance is one row - ``name``, ``date``, ``price``, ``is_upper``,
  ``available_at``, ``source_window`` - so ``(name, date)`` identifies it and a consumer filters
  by instant instead of picking DataFrame columns per level name.
* **The clock is the loader's.**  ``timestamp`` is the bar's ``open_time`` in UTC and the
  presumed still-forming tail bar is dropped first (rule 2b): a forming bar can neither set a
  level nor break one.  ``available_at`` is the instant the level became a fact, and no bar of
  the level's own window can precede it - that is what makes the maps leak free.
* **Session windows live in** :mod:`smc_zero.indicators.sessions`, never in a second table: the
  MSK window comes from ``kz_windows_msk`` / ``kz_window_utc_hours`` (with ``season_for``
  picking the season), so the level map and the killzone gate can never disagree (C1).  Asia is
  the documented exception - a fixed UTC window (``LevelConfig.asian_window_utc``, prod's
  ``hour < 8``) because Asia does not switch clocks.
* **A break needs a close.**  ``broken_at`` is the ``close_time`` of the first bar whose close
  crossed the level by more than ``LevelConfig.break_buffer_pip`` - the instant the break became
  *knowable*, following the same confirmation-instant convention as ``known_at`` /
  ``close_time`` elsewhere.  A wick through the level is not a break, and the breaker/retest
  setup of spec п.4 is a separate setup left out of v1 (SPEC_SMC.md §7.7).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from smc_zero.config import HourWindow, Killzone, LevelConfig, SessionConfig, TimeframeConfig
from smc_zero.data_loader import TIMESTAMP_COLUMN, drop_unclosed, period_for
from smc_zero.indicators.sessions import (
    LONDON,
    NY,
    kz_window_utc_hours,
    kz_windows_msk,
    season_for,
)

LEVEL_NAME_COLUMN = "name"
LEVEL_DATE_COLUMN = "date"
LEVEL_PRICE_COLUMN = "price"
LEVEL_IS_UPPER_COLUMN = "is_upper"
LEVEL_AVAILABLE_AT_COLUMN = "available_at"
LEVEL_SOURCE_WINDOW_COLUMN = "source_window"
LEVEL_COLUMNS: tuple[str, ...] = (
    LEVEL_NAME_COLUMN,
    LEVEL_DATE_COLUMN,
    LEVEL_PRICE_COLUMN,
    LEVEL_IS_UPPER_COLUMN,
    LEVEL_AVAILABLE_AT_COLUMN,
    LEVEL_SOURCE_WINDOW_COLUMN,
)
BROKEN_AT_COLUMN = "broken_at"
IDL_DYN_COLUMN = "idl_dyn"
IDH_DYN_COLUMN = "idh_dyn"

PDH, PDL = "PDH", "PDL"
PWH, PWL = "PWH", "PWL"
PMH, PML = "PMH", "PML"
ASIAN_HIGH, ASIAN_LOW = "AsianH", "AsianL"
LONDON_HIGH, LONDON_LOW = "LondonH", "LondonL"
NY_HIGH, NY_LOW = "NYH", "NYL"
LEVEL_NAMES: tuple[str, ...] = (
    PDH,
    PDL,
    PWH,
    PWL,
    PMH,
    PML,
    ASIAN_HIGH,
    ASIAN_LOW,
    LONDON_HIGH,
    LONDON_LOW,
    NY_HIGH,
    NY_LOW,
)

# Which killzone owns which session-range pair.  The windows themselves are *not* listed here:
# they are read from ``sessions`` (C1), this mapping only pairs a killzone with its two names.
SESSION_LEVELS: dict[Killzone, tuple[str, str]] = {
    LONDON: (LONDON_HIGH, LONDON_LOW),
    NY: (NY_HIGH, NY_LOW),
}

# Period keys of the daily/weekly/monthly ranges.  All three are *strings* of fixed width, so
# lexicographic order is temporal order (``"%Y-%W"`` is zero-padded) and no calendar stepping
# over weekends or holidays is needed - a missing period is simply an absent key.
DAY_KEY_FORMAT = "%Y-%m-%d"
MONTH_KEY_FORMAT = "%Y-%m"


def _utc_stamps(frame: pd.DataFrame) -> pd.Series:
    """Return ``frame["timestamp"]`` as UTC-aware ``datetime64[ns, UTC]``.

    The loader already stores open_time stamps in that dtype; naive input is localized to UTC
    like :func:`smc_zero.data_loader._to_utc`, and a non-datetime column is a hard error
    (guessing the unit of an integer stamp would hide a broken pipeline).
    """
    if TIMESTAMP_COLUMN not in frame.columns:
        raise ValueError(f"levels need a {TIMESTAMP_COLUMN!r} column (loader schema)")
    stamps = frame[TIMESTAMP_COLUMN]
    if not pd.api.types.is_datetime64_any_dtype(stamps.dtype):
        raise ValueError(
            f"{TIMESTAMP_COLUMN!r} must be a datetime column, got dtype {stamps.dtype}"
        )
    if not isinstance(stamps.dtype, pd.DatetimeTZDtype):
        stamps = stamps.dt.tz_localize("UTC")
    return stamps.dt.tz_convert("UTC").astype("datetime64[ns, UTC]")


def _closed_bars(frame: pd.DataFrame, required: tuple[str, ...]) -> pd.DataFrame:
    """Drop the presumed still-forming tail bar and check the required columns.

    Rule 2b: the last bar of the frame is treated as still forming, so a partially printed
    candle can never set a level range or break one.  The returned frame is a copy whose
    ``timestamp`` column is normalized to ``datetime64[ns, UTC]``.
    """
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"levels need the {missing} column(s) (loader schema)")
    closed = drop_unclosed(frame)
    if closed.empty:
        return closed.copy()
    closed = closed.copy()
    closed[TIMESTAMP_COLUMN] = _utc_stamps(closed)
    return closed


def _calendar_date(stamp: pd.Timestamp) -> pd.Timestamp:
    """Return the naive midnight of a UTC stamp: the date a level instance belongs to."""
    return pd.Timestamp(stamp).tz_convert("UTC").tz_localize(None).normalize()


def _levels_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Assemble the long-format level frame with fixed dtypes (empty-frame safe)."""
    frame = pd.DataFrame(rows, columns=list(LEVEL_COLUMNS))
    frame[LEVEL_PRICE_COLUMN] = frame[LEVEL_PRICE_COLUMN].astype("float64")
    frame[LEVEL_IS_UPPER_COLUMN] = frame[LEVEL_IS_UPPER_COLUMN].astype(bool)
    frame[LEVEL_DATE_COLUMN] = pd.to_datetime(frame[LEVEL_DATE_COLUMN]).astype(
        "datetime64[ns]"
    )
    frame[LEVEL_AVAILABLE_AT_COLUMN] = pd.to_datetime(
        frame[LEVEL_AVAILABLE_AT_COLUMN], utc=True
    ).astype("datetime64[ns, UTC]")
    return frame.sort_values([LEVEL_NAME_COLUMN, LEVEL_DATE_COLUMN], kind="stable").reset_index(
        drop=True
    )


def _day_stamps(frame: pd.DataFrame) -> pd.DatetimeIndex:
    """Return the sorted unique calendar dates (naive midnights) of the bar labels.

    The trading day is the date carried by the bar label itself (what the D1 series of the
    loader groups by), so a level can never be attributed to a neighbouring day by a timezone
    change of the machine or by a DST rule.
    """
    naive = frame[TIMESTAMP_COLUMN].dt.tz_localize(None).dt.normalize()
    return pd.DatetimeIndex(np.unique(naive.to_numpy(dtype="datetime64[ns]")))


def _previous_period_rows(
    frame: pd.DataFrame,
    keys: np.ndarray,
    high_name: str,
    low_name: str,
    label: str,
) -> list[dict[str, object]]:
    """Rows of a PDH/PDL-style pair over the period grouping ``keys``.

    A level instance belongs to the *current* period and carries the range of the previous
    period *present in the frame* - the maximal earlier key, no calendar stepping, so weekends
    and holidays are simply absent keys.  Both prices are known from the first bar of the
    current period on, hence ``available_at`` is that bar's ``open_time`` and ``date`` is its
    calendar date.  ``keys`` is passed positionally (an array, not a Series) so the grouping can
    never be silently misaligned with the frame, and each period is aggregated on its own bars,
    so the range of the current period can not leak into the level of a later period.
    """
    aggregated = frame.groupby(keys, sort=True).agg(
        high=("high", "max"),
        low=("low", "min"),
        first=(TIMESTAMP_COLUMN, "min"),
    )
    rows: list[dict[str, object]] = []
    for position in range(1, len(aggregated)):
        current = aggregated.iloc[position]
        previous = aggregated.iloc[position - 1]
        available_at = pd.Timestamp(current["first"])
        common: dict[str, object] = {
            LEVEL_DATE_COLUMN: _calendar_date(available_at),
            LEVEL_AVAILABLE_AT_COLUMN: available_at,
            LEVEL_SOURCE_WINDOW_COLUMN: f"{label}={previous.name}",
        }
        rows.append(
            {
                **common,
                LEVEL_NAME_COLUMN: high_name,
                LEVEL_PRICE_COLUMN: float(previous["high"]),
                LEVEL_IS_UPPER_COLUMN: True,
            }
        )
        rows.append(
            {
                **common,
                LEVEL_NAME_COLUMN: low_name,
                LEVEL_PRICE_COLUMN: float(previous["low"]),
                LEVEL_IS_UPPER_COLUMN: False,
            }
        )
    return rows


def _fixed_windows(
    days: pd.DatetimeIndex, window: HourWindow, label: str
) -> list[tuple[pd.Timestamp, HourWindow, str]]:
    """Return the same fixed UTC hour window for every day (Asia: no clock change)."""
    start, end = window
    return [(day, (start, end), f"{label}[{start},{end})") for day in days]


def _session_windows(
    days: pd.DatetimeIndex, killzone: Killzone, session_cfg: SessionConfig | None
) -> list[tuple[pd.Timestamp, HourWindow, str]]:
    """Return the UTC window of ``killzone`` for every day, straight from ``sessions``.

    ``kz_window_utc_hours`` is the single table the killzone gate uses (it applies
    ``season_for`` + ``kz_windows_msk`` and the fixed MSK shift, C1); the MSK window put into
    ``source_window`` is the very same table entry, so the reported window can never disagree
    with the bars the price came from.  ``day`` is handed over as the bar's calendar date, which
    is also its MSK date for every shipped window (they all lie inside 04:00-15:00 UTC); a
    custom window reaching before 00:00 UTC would break that identity and is rejected here
    instead of being mapped onto the wrong day.
    """
    windows: list[tuple[pd.Timestamp, HourWindow, str]] = []
    for day in days:
        start_msk, end_msk = kz_windows_msk(killzone, season_for(killzone, day.date()), session_cfg)
        start_utc, end_utc = kz_window_utc_hours(killzone, day.date(), session_cfg)
        if start_utc < 0 or end_utc > 24:
            raise ValueError(
                f"{killzone} window {start_msk}-{end_msk} MSK is not inside one UTC day"
            )
        windows.append((day, (start_utc, end_utc), f"{killzone}_msk[{start_msk},{end_msk})"))
    return windows


def _window_rows(
    frame: pd.DataFrame,
    windows: list[tuple[pd.Timestamp, HourWindow, str]],
    name_high: str,
    name_low: str,
) -> list[dict[str, object]]:
    """Rows of a session range: the extreme of the bars that *open* inside each window.

    ``windows`` holds ``(day, (start_hour_utc, end_hour_utc), text)``; a window without a
    single bar emits nothing (holidays, data gaps).  ``available_at`` is the window's closing
    instant - exactly one M15 step after the last bar of the window opens - so every bar of the
    window is closed before the range becomes a fact, and the still-forming part of a window
    can not be read at an earlier bar.
    """
    stamps = frame[TIMESTAMP_COLUMN].dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    high = frame["high"].to_numpy(dtype="float64")
    low = frame["low"].to_numpy(dtype="float64")
    rows: list[dict[str, object]] = []
    for day, (start_hour, end_hour), text in windows:
        left = int(
            stamps.searchsorted(np.datetime64(day + pd.Timedelta(hours=start_hour)), "left")
        )
        right = int(stamps.searchsorted(np.datetime64(day + pd.Timedelta(hours=end_hour)), "left"))
        if right <= left:
            continue
        available_at = pd.Timestamp(day).tz_localize("UTC") + pd.Timedelta(hours=end_hour)
        common: dict[str, object] = {
            LEVEL_DATE_COLUMN: day,
            LEVEL_AVAILABLE_AT_COLUMN: available_at,
            LEVEL_SOURCE_WINDOW_COLUMN: text,
        }
        rows.append(
            {
                **common,
                LEVEL_NAME_COLUMN: name_high,
                LEVEL_PRICE_COLUMN: float(high[left:right].max()),
                LEVEL_IS_UPPER_COLUMN: True,
            }
        )
        rows.append(
            {
                **common,
                LEVEL_NAME_COLUMN: name_low,
                LEVEL_PRICE_COLUMN: float(low[left:right].min()),
                LEVEL_IS_UPPER_COLUMN: False,
            }
        )
    return rows


def static_levels(
    df: pd.DataFrame,
    cfg: LevelConfig | None = None,
    *,
    session_cfg: SessionConfig | None = None,
) -> pd.DataFrame:
    """Return the long-format map of every liquidity level instance visible in ``df``.

    One row per instance with the columns of :data:`LEVEL_COLUMNS`: ``name`` is one of
    :data:`LEVEL_NAMES`, ``date`` the calendar date of the period the instance belongs to,
    ``is_upper`` distinguishes a high from a low, ``available_at`` is the first instant the price
    is a fact, and ``source_window`` records where the range came from
    (``prev_day=2026-01-05``, ``asia_utc[0,8)``, ``london_msk[10,13)``) so any row can be traced
    back without re-deriving the window.  ``(name, date)`` is unique.

    Daily ranges yield one instance per trading day, weekly and monthly ones one instance per
    week / month (dated by that period's first trading day and known from its first bar), and
    session ranges one instance per day - so a consumer never sees the same fact twice.

    ``df`` is the M15 entry frame: only ``timestamp``, ``high`` and ``low`` are read, and the
    presumed still-forming tail bar is ignored (rule 2b).  ``session_cfg`` is the killzone table
    shared with the gate; leaving it ``None`` uses the default table, i.e. exactly what
    :func:`smc_zero.indicators.sessions.in_killzone` uses by default.
    """
    config = LevelConfig() if cfg is None else cfg
    closed = _closed_bars(df, ("high", "low"))
    if closed.empty:
        return _levels_frame([])
    stamps = closed[TIMESTAMP_COLUMN]
    rows: list[dict[str, object]] = []
    rows += _previous_period_rows(
        closed, stamps.dt.strftime(DAY_KEY_FORMAT).to_numpy(), PDH, PDL, "prev_day"
    )
    rows += _previous_period_rows(
        closed, stamps.dt.strftime(config.week_convention).to_numpy(), PWH, PWL, "prev_week"
    )
    rows += _previous_period_rows(
        closed, stamps.dt.strftime(MONTH_KEY_FORMAT).to_numpy(), PMH, PML, "prev_month"
    )
    days = _day_stamps(closed)
    rows += _window_rows(
        closed, _fixed_windows(days, config.asian_window_utc, "asia_utc"), ASIAN_HIGH, ASIAN_LOW
    )
    for killzone, (name_high, name_low) in SESSION_LEVELS.items():
        rows += _window_rows(
            closed, _session_windows(days, killzone, session_cfg), name_high, name_low
        )
    return _levels_frame(rows)


def dynamic_idl_idh(df: pd.DataFrame) -> pd.DataFrame:
    """Add prod's intraday running low/high as ``idl_dyn`` / ``idh_dyn`` columns.

    At bar ``i`` the value is the extreme of the bars of the same trading day *strictly before*
    ``i`` - prod's ``cummin`` / ``cummax`` over the day followed by ``shift(1)``.  Hence the first
    bar of every day is ``NaN``, a bar never contributes to its own value, and the previous day's
    extremes never bleed into the new day.  The presumed still-forming tail bar is dropped like
    everywhere else (rule 2b) and a copy is returned, so the input frame stays untouched.
    """
    closed = _closed_bars(df, ("high", "low"))
    frame = closed.copy()
    if frame.empty:
        for column in (IDL_DYN_COLUMN, IDH_DYN_COLUMN):
            frame[column] = pd.Series(dtype="float64", index=frame.index)
        return frame
    stamps = frame[TIMESTAMP_COLUMN]
    day = stamps.dt.tz_localize(None).dt.normalize().to_numpy(dtype="datetime64[ns]")
    frame[IDL_DYN_COLUMN] = frame["low"].groupby(day).cummin().groupby(day).shift(1)
    frame[IDH_DYN_COLUMN] = frame["high"].groupby(day).cummax().groupby(day).shift(1)
    return frame


def _ltf_period(stamps: pd.Series) -> pd.Timedelta:
    """Return the entry timeframe's bar period after checking the frame's own grid.

    ``close_time`` of a breaking bar is ``timestamp + period``, and that period is the pipeline's
    entry timeframe (:attr:`smc_zero.config.TimeframeConfig.ltf`, C5).  A frame that is *not* on
    that grid would silently mis-date every ``broken_at``, so a mismatch is an error instead of a
    silently inferred period.
    """
    timeframe = TimeframeConfig().ltf
    period = period_for(timeframe)
    if len(stamps) > 1:
        step = stamps.diff().dropna().min()
        if step != period:
            raise ValueError(
                f"level_lifecycle expects the {timeframe} entry frame ({period} bars), "
                f"got a {step} grid"
            )
    return period


def level_lifecycle(
    levels: pd.DataFrame, df: pd.DataFrame, cfg: LevelConfig | None = None
) -> pd.DataFrame:
    """Return ``levels`` with a ``broken_at`` column attached to every instance.

    A level is broken by the first *closed* bar not earlier than ``available_at`` whose close
    went beyond the level by more than the buffer: ``close > price + break_buffer_pip`` for an
    upper level, ``close < price - break_buffer_pip`` for a lower one.  ``broken_at`` is that
    bar's ``close_time`` - the instant the break became knowable, like ``known_at`` everywhere
    else - and ``NaT`` when no bar of ``df`` ever broke the level.  A wick through the level is
    not a break, and the breaker / retest setup of spec п.4 is a separate setup left out of v1
    (SPEC_SMC.md §7.7).

    Combine the result with :func:`fresh_at` to ask what a strategy asks: is this level still
    tradable at bar ``i``?
    """
    config = LevelConfig() if cfg is None else cfg
    missing = [column for column in LEVEL_COLUMNS if column not in levels.columns]
    if missing:
        raise ValueError(f"levels are missing the {missing} column(s)")
    lifecycle = levels.copy()
    closed = _closed_bars(df, ("close",))
    if closed.empty:
        lifecycle[BROKEN_AT_COLUMN] = pd.Series(
            pd.NaT, index=lifecycle.index, dtype="datetime64[ns, UTC]"
        )
        return lifecycle
    stamps = closed[TIMESTAMP_COLUMN]
    period = _ltf_period(stamps)
    stamps_np = stamps.dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    close = closed["close"].to_numpy(dtype="float64")
    buffer = config.break_buffer_pip
    broken: list[pd.Timestamp | None] = []
    for price, is_upper, available_at in zip(
        levels[LEVEL_PRICE_COLUMN].to_numpy(dtype="float64"),
        levels[LEVEL_IS_UPPER_COLUMN].to_numpy(dtype=bool),
        levels[LEVEL_AVAILABLE_AT_COLUMN],
        strict=True,
    ):
        available = pd.Timestamp(available_at).tz_convert("UTC").tz_localize(None)
        start = int(stamps_np.searchsorted(np.datetime64(available), "left"))
        threshold = price + buffer if is_upper else price - buffer
        window = close[start:]
        if is_upper:
            hits = np.flatnonzero(window > threshold)
        else:
            hits = np.flatnonzero(window < threshold)
        if hits.size == 0:
            broken.append(None)
            continue
        broken.append(pd.Timestamp(stamps_np[start + int(hits[0])]).tz_localize("UTC") + period)
    lifecycle[BROKEN_AT_COLUMN] = pd.to_datetime(
        pd.Series(broken, index=lifecycle.index), utc=True
    ).astype("datetime64[ns, UTC]")
    return lifecycle


def fresh_at(row: Mapping[str, object] | pd.Series, t: pd.Timestamp) -> bool:
    """Return whether the level instance in ``row`` is still fresh at instant ``t``.

    ``t`` is the decision instant of an M15 bar - its ``open_time``, the key
    :func:`smc_zero.indicators.bias.bias_frames` hands to consumers.  A level is fresh while it
    exists (``available_at <= t``) and its break is not known yet (``t < broken_at``); a row with
    a ``NaT`` break stays fresh forever, and a row without the ``broken_at`` column is treated as
    unbroken.  Since ``broken_at`` is the breaking bar's close time, the breaking bar itself is
    still fresh at its own open - its close is not known at that instant.
    """
    available_at = pd.Timestamp(row[LEVEL_AVAILABLE_AT_COLUMN])
    broken_at = row.get(BROKEN_AT_COLUMN)
    if broken_at is None or pd.isna(broken_at):
        return bool(available_at <= t)
    return bool(available_at <= t < pd.Timestamp(broken_at))
