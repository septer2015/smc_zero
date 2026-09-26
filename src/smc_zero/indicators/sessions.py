"""Killzone gate: the single seasonal source of truth for session windows.

SPEC_SMC.md, C1 splits seasonality across two calendars: London follows the EU
DST rule, New York the US one, while pre-London is a fixed MSK window (it is not
a foreign session).  That decision lives here and *only* here - the Э3' session
level maps (``london_*`` / ``ny_*``) must import :func:`kz_windows_msk` /
:func:`kz_window_utc_hours` from this module rather than re-declaring windows,
otherwise the gate and the level maps drift apart.

All windows are half-open, ``start <= hour < end``, expressed in MSK hours.
Moscow is UTC+3 all year (no DST since 2014), so the same table converts to UTC
hours by a constant shift - no DST arithmetic is repeated anywhere else.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime

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


def in_killzone(ts_utc: pd.Timestamp | datetime, cfg: SessionConfig | None = None) -> bool:
    """Return whether the UTC stamp ``ts_utc`` falls inside an enabled killzone.

    Outside every enabled window the answer is ``False``: such a bar carries no
    signal at all (``no_trade``).  The season is resolved from the MSK date of
    the bar; the DST switch happens at 01:00 UTC (EU) / 06:00-07:00 UTC (US), so
    on those Sundays a few bars use the neighbouring season's window - inherent
    to a date-level rule and documented in :mod:`smc_zero.utils.time`.

    ``cfg.use_kz=False`` disables the gate for the whole run (no bar is in a
    killzone); ``cfg.prelondon`` adds the fixed pre-London window.
    """
    config = SessionConfig() if cfg is None else cfg
    if not config.use_kz:
        return False
    stamp = pd.Timestamp(ts_utc).tz_convert("UTC")
    msk_day = (stamp + pd.Timedelta(hours=MSK_UTC_OFFSET_HOURS)).date()
    hour_msk = to_msk_hour(stamp.hour)
    killzones: tuple[Killzone, ...] = (LONDON, NY)
    if config.prelondon:
        killzones = (PRE_LONDON, *killzones)
    windows = (kz_windows_msk(zone, season_for(zone, msk_day), config) for zone in killzones)
    return any(start <= hour_msk < end for start, end in windows)
