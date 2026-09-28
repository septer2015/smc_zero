"""Session clocks: the seasonal killzone table and the broker's trading hours.

SPEC_SMC.md, C1 splits seasonality across two calendars: London follows the EU
DST rule, New York the US one, while pre-London is a fixed MSK window (it is not
a foreign session).  That decision lives here and *only* here - the Э3' session
level maps (``london_*`` / ``ny_*``) must import :func:`kz_windows_msk` /
:func:`kz_window_utc_hours` from this module rather than re-declaring windows,
otherwise the gate and the level maps drift apart.

The same module owns the *second*, coarser clock: C6's broker week (Alfa opens
Monday 02:00 MSK and closes Friday 23:55 MSK).  :func:`alfa_trading_mask` /
:func:`in_trading_hours` answer "may an order exist at all at this instant?" while
:func:`killzone_mask` / :func:`in_killzone` answer "is this bar inside a session we
trade?" - an outer and an inner gate, both fed by
:class:`~smc_zero.config.SessionConfig`.  Each gate has a vector form (a mask over a
whole entry frame, what the Э4' chain consumes) and a scalar form for a single
decision instant; both forms of a gate call the *same* table and the same window
helper, so they can never disagree.

All windows are half-open, ``start <= hour < end``, expressed in MSK hours.
Moscow is UTC+3 all year (no DST since 2014), so the same table converts to UTC
hours by a constant shift - no DST arithmetic is repeated anywhere else.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime

import numpy as np
import pandas as pd

from smc_zero.config import HourWindow, Killzone, Season, SessionConfig
from smc_zero.utils.time import MSK_UTC_OFFSET_HOURS, eu_is_summer, to_msk_hour, us_is_summer

PRE_LONDON: Killzone = "prelondon"
LONDON: Killzone = "london"
NY: Killzone = "ny"

SUMMER: Season = "summer"
WINTER: Season = "winter"

# Which DST calendar owns a killzone.  Pre-London is listed with the EU rule
# because it is a *fixed* window: its season lookup is never used, but keeping
# the key present makes every killzone name valid for :func:`season_for`.
SEASON_RULE: dict[str, Callable[[date], bool]] = {
    PRE_LONDON: eu_is_summer,
    LONDON: eu_is_summer,
    NY: us_is_summer,
}


def _check_killzone(killzone: str) -> None:
    if killzone not in SEASON_RULE:
        raise ValueError(
            f"unsupported killzone {killzone!r}; expected one of {sorted(SEASON_RULE)}"
        )


def _check_season(season: str) -> None:
    if season not in (SUMMER, WINTER):
        raise ValueError(f"unsupported season {season!r}; expected one of {(SUMMER, WINTER)}")


def season_for(killzone: Killzone, day: date) -> Season:
    """Return the season that owns ``day`` for ``killzone``.

    ``day`` is the *MSK* date of the bar (C1 defines the windows per bar date in
    MSK hours).  London resolves against the EU calendar, New York against the US
    one - the same date can therefore be summer for one and winter for the other.
    """
    _check_killzone(killzone)
    return SUMMER if SEASON_RULE[killzone](day) else WINTER


def kz_windows_msk(
    killzone: Killzone,
    season: Season,
    cfg: SessionConfig | None = None,
) -> HourWindow:
    """Return the half-open MSK window of ``killzone`` in ``season``."""
    _check_killzone(killzone)
    _check_season(season)
    config = SessionConfig() if cfg is None else cfg
    if killzone == PRE_LONDON:  # fixed: no DST
        return config.prelondon_msk
    if killzone == LONDON:
        return config.london_summer_msk if season == SUMMER else config.london_winter_msk
    return config.ny_summer_msk if season == SUMMER else config.ny_winter_msk


def kz_window_utc_hours(
    killzone: Killzone,
    day: date,
    cfg: SessionConfig | None = None,
) -> HourWindow:
    """Return the killzone window of ``day`` re-expressed as UTC hours.

    The Э3' session maps are built on UTC hours (``hour >= 6 and hour < 9`` for
    London summer, ``15 <= hour < 18`` for New York winter); they must call this
    function so both levels share one table.  Because MSK is a constant UTC+3,
    the shift cannot produce a negative hour for the shipped windows
    (``prelondon_msk`` starts at 07 MSK = 04 UTC).
    """
    start, end = kz_windows_msk(killzone, season_for(killzone, day), cfg)
    return (start - MSK_UTC_OFFSET_HOURS, end - MSK_UTC_OFFSET_HOURS)


def _killzone_windows(day: date, cfg: SessionConfig) -> tuple[HourWindow, ...]:
    """Return the enabled MSK windows of ``day`` - the one place that picks them.

    The killzone *set* (London + New York, plus the fixed pre-London window when
    ``cfg.prelondon``) and the seasonal lookup are stated here once, so the scalar
    gate :func:`in_killzone` and the vector gate :func:`killzone_mask` can never
    drift apart.
    """
    killzones: tuple[Killzone, ...] = (LONDON, NY)
    if cfg.prelondon:
        killzones = (PRE_LONDON, *killzones)
    return tuple(kz_windows_msk(zone, season_for(zone, day), cfg) for zone in killzones)


def in_killzone(ts_utc: pd.Timestamp | datetime, cfg: SessionConfig | None = None) -> bool:
    """Return whether the UTC stamp ``ts_utc`` falls inside an enabled killzone.

    Outside every enabled window the answer is ``False``: such a bar carries no
    signal at all (``no_trade``).  The season is resolved from the MSK date of
    the bar; the DST switch happens at 01:00 UTC (EU) / 06:00-07:00 UTC (US), so
    on those Sundays a few bars use the neighbouring season's window - inherent
    to a date-level rule and documented in :mod:`smc_zero.utils.time`.

    ``cfg.use_kz=False`` disables the gate for the whole run (no bar is in a
    killzone); ``cfg.prelondon`` adds the fixed pre-London window.  The scalar
    counterpart of :func:`killzone_mask`, sharing :func:`_killzone_windows` with it.
    """
    config = SessionConfig() if cfg is None else cfg
    if not config.use_kz:
        return False
    stamp = pd.Timestamp(ts_utc).tz_convert("UTC")
    msk_day = (stamp + pd.Timedelta(hours=MSK_UTC_OFFSET_HOURS)).date()
    hour_msk = to_msk_hour(stamp.hour)
    return any(start <= hour_msk < end for start, end in _killzone_windows(msk_day, config))


def killzone_mask(stamps: pd.Series, cfg: SessionConfig | None = None) -> pd.Series:
    """Return the killzone gate of a whole entry frame as a boolean mask.

    Vector form of :func:`in_killzone`: bar ``i`` is ``True`` when its MSK
    hour falls inside one of the enabled killzone windows of its own MSK date.
    The season is looked up once per MSK date (not once per bar) - the windows
    themselves come from :func:`_killzone_windows`, the same helper the scalar gate
    uses, so the two can not disagree.

    ``cfg.use_kz=False`` yields an all-``False`` mask (the gate is off, no bar is
    tradable), ``cfg.prelondon`` adds the fixed pre-London window.  This is the
    *inner* gate of the Э4' entry chain: :func:`alfa_trading_mask` already removed
    the weekend, this one keeps only the sessions the strategy trades.
    """
    config = SessionConfig() if cfg is None else cfg
    index = (stamps if isinstance(stamps, pd.Series) else pd.Series(stamps)).index
    if not config.use_kz:
        return pd.Series(False, index=index, dtype=bool)
    msk = pd.to_datetime(stamps, utc=True) + pd.Timedelta(hours=MSK_UTC_OFFSET_HOURS)
    minute = (msk.dt.hour * 60 + msk.dt.minute).to_numpy()
    # The MSK date of the bar drives the seasonal lookup, exactly like the scalar
    # gate; ``days`` is compared as ``datetime64`` so the grouping needs no groupby.
    days = msk.dt.tz_localize(None).dt.normalize().to_numpy(dtype="datetime64[ns]")
    mask = np.zeros(minute.size, dtype=bool)
    for day_stamp in np.unique(days):
        positions = days == day_stamp
        msk_day = pd.Timestamp(day_stamp).date()
        for start, end in _killzone_windows(msk_day, config):
            mask[positions] |= (minute[positions] >= start * 60) & (minute[positions] < end * 60)
    return pd.Series(mask, index=index, dtype=bool)


def _msk_weekday_minute(stamps: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return ``(weekday, minute of day)`` of UTC ``stamps`` in MSK.

    Moscow is a constant UTC+3, so the shift is applied to the whole series at once
    and the minute is unchanged; the weekday is derived from the shifted stamp, not
    from the UTC one - 23:55 MSK Friday is 20:55 UTC Friday, while 00:10 MSK
    Saturday is already a weekend bar (this is the trap the mask exists for).
    """
    msk = pd.to_datetime(stamps, utc=True) + pd.Timedelta(hours=MSK_UTC_OFFSET_HOURS)
    return msk.dt.weekday, msk.dt.hour * 60 + msk.dt.minute


def alfa_trading_mask(stamps: pd.Series, cfg: SessionConfig | None = None) -> pd.Series:
    """Return the broker's trading-hours mask for the UTC ``stamps`` (C6).

    Alfa accepts orders from Monday 02:00 MSK to Friday 23:55 MSK and is shut over
    the weekend, so the mask closes three windows: the weekend itself, the
    Monday-before-open part of the day and the Friday-after-close part.  The
    boundaries are read from ``SessionConfig.session_open_msk`` /
    ``session_close_msk`` (C6 numbers, no literals here); the Monday part is
    ``< open`` and the Friday part is ``>= close``, so 23:55 MSK is already closed.

    This is the *outer* gate of the strategy: a bar outside the mask carries no
    signal at all (SPEC_SMC.md §7.8), while :func:`in_killzone` keeps deciding which
    sessions a tradable bar may belong to.  The mask is vectorised over a whole
    entry frame - the Э4' chain attaches it as a column - and
    :func:`in_trading_hours` is the scalar form for a single decision instant.
    """
    config = SessionConfig() if cfg is None else cfg
    open_hour, open_minute = config.session_open_msk
    close_hour, close_minute = config.session_close_msk
    weekday, minute = _msk_weekday_minute(stamps)
    before_open = (weekday == 0) & (minute < open_hour * 60 + open_minute)
    after_close = (weekday == 4) & (minute >= close_hour * 60 + close_minute)
    # The parentheses around the comparison are load bearing: ``&`` binds tighter
    # than ``<`` in Python, so ``weekday < 5 & mask`` would AND the integers first.
    return (weekday < 5) & ~before_open & ~after_close


def in_trading_hours(ts_utc: pd.Timestamp | datetime, cfg: SessionConfig | None = None) -> bool:
    """Return whether the single UTC instant ``ts_utc`` is inside Alfa's hours (C6).

    Scalar counterpart of :func:`alfa_trading_mask`, sharing its only rule: the
    opening/closing boundaries live in :class:`~smc_zero.config.SessionConfig`.
    """
    mask = alfa_trading_mask(pd.Series([pd.Timestamp(ts_utc)]), cfg)
    return bool(mask.iloc[0])
