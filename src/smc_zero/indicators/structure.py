"""Market structure: swings, break signal, trend automaton (SPEC_SMC.md, C4).

Semantics ported from the prod source ``core.py`` (semantics only, no code reuse):

* a swing is a *strict* N-bar fractal - ``high[i] > high[i +/- k]`` for every
  ``k <= N`` (prod's formula, core.py lines 339-343, is exactly the ``N = 1``
  case).  It is therefore known only at bar ``i + N``, once the right candles have
  closed, and every markup row carries that position in ``swing_*_known_at``
  (rule 2: no look-ahead);
* ``break_dir`` reproduces prod's ``choch_signal`` (core.py lines 347-365):
  ``+1`` while the close sits beyond the last known swing high, ``-1`` while it
  sits below the last known swing low, ``0`` in between.  A level is *not*
  consumed by a break, so the signal repeats while price stays beyond it (prod
  parity) - downstream code that needs "the first bar of the break" must dedupe
  itself;
* ``structure_event`` / ``trend`` add the automaton the constitution defines on
  top of the raw signal: a break *with* the trend is a BOS, a break *against* it
  is a CHoCH that flips the trend.  The very first break has no trend to agree
  with and is labelled ``bos`` (trend initialisation, agreed with the project
  owner); ``trend`` stays ``0`` until that first break.

Every function is pure (``DataFrame`` + :class:`~smc_zero.config.StructureConfig`
in, markup out), preserves the input index and never mutates the argument
(SPEC_SMC.md, section 0.7: prod's ``_PRECOMPUTE_CACHE`` is not ported).  All
markup positions are *positional* bar numbers; use ``df.index[position]`` for the
timestamp.  ``structure_event`` uses ``None`` for "no event".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smc_zero.config import BreakEvent, StructureConfig


def _fractal_mask(values: np.ndarray, lookback: int, *, higher: bool) -> np.ndarray:
    """Return a mask of strict N-bar fractal extremes of ``values``.

    ``higher`` selects highs (fractal high) or lows (fractal low).  The mask has
    one ``True`` per extreme bar; the first and last ``lookback`` bars can never
    qualify, which is what makes the confirmation lag explicit.
    """
    n = values.size
    mask = np.zeros(n, dtype=bool)
    if n <= 2 * lookback:
        return mask
    centre = values[lookback : n - lookback]
    strict = np.ones(centre.shape, dtype=bool)
    for shift in range(1, lookback + 1):
        left = values[lookback - shift : n - lookback - shift]
        right = values[lookback + shift : n - lookback + shift]
        if higher:
            strict &= (centre > left) & (centre > right)
        else:
            strict &= (centre < left) & (centre < right)
    mask[lookback : n - lookback] = strict
    return mask


def _known_positions(swing: np.ndarray, lookback: int, index: pd.Index) -> pd.Series:
    """Return the confirming bar position (``i + lookback``) of each swing, NA elsewhere.

    The result is built on the *input* index: pandas aligns on labels when a Series
    is put into a DataFrame constructor, so a freshly built ``RangeIndex`` series
    would silently turn into NA for a DatetimeIndex frame.
    """
    positions = np.arange(swing.size)
    values = np.where(swing, positions + lookback, np.nan)
    return pd.Series(values, index=index).astype("Int64")


def swing_points(df: pd.DataFrame, cfg: StructureConfig | None = None) -> pd.DataFrame:
    """Return swing flags plus the position at which each swing becomes known.

    Columns (same index as ``df``): ``swing_high`` / ``swing_low`` (``bool``,
    ``True`` on the extreme bar itself), ``swing_high_price`` / ``swing_low_price``
    (``float``, NaN elsewhere, the extreme price) and ``swing_high_known_at`` /
    ``swing_low_known_at`` (``Int64``, ``i + swing_lookback``, NA elsewhere).

    With the default ``swing_lookback = 1`` a swing at bar ``k`` is visible from
    bar ``k + 1`` on, i.e. exactly prod's ``is_swing_high[i - 1]`` read
    (core.py lines 355-360).
    """
    config = StructureConfig() if cfg is None else cfg
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    lookback = config.swing_lookback
    swing_high = _fractal_mask(high, lookback, higher=True)
    swing_low = _fractal_mask(low, lookback, higher=False)
    return pd.DataFrame(
        {
            "swing_high": swing_high,
            "swing_low": swing_low,
            "swing_high_price": np.where(swing_high, high, np.nan),
            "swing_low_price": np.where(swing_low, low, np.nan),
            "swing_high_known_at": _known_positions(swing_high, lookback, df.index),
            "swing_low_known_at": _known_positions(swing_low, lookback, df.index),
        },
        index=df.index,
    )



def known_swing_levels(df: pd.DataFrame, cfg: StructureConfig | None = None) -> pd.DataFrame:
    """Return the last swing levels *as they are visible at each bar*.

    ``last_swing_high`` / ``last_swing_low`` (same index as ``df``) hold the extreme
    of the most recent swing whose confirmation bar is not later than the current
    one: a swing at bar ``k`` enters at bar ``k + swing_lookback`` (``shift`` +
    ``ffill``).  Before the first confirmed swing the value is NaN, and NaN never
    compares as a break.
    """
    config = StructureConfig() if cfg is None else cfg
    points = swing_points(df, config)
    lookback = config.swing_lookback
    return pd.DataFrame(
        {
            "last_swing_high": points["swing_high_price"].shift(lookback).ffill(),
            "last_swing_low": points["swing_low_price"].shift(lookback).ffill(),
        },
        index=df.index,
    )


def _run_trend_automaton(break_dir: np.ndarray) -> tuple[list[BreakEvent | None], np.ndarray]:
    """Return the per-bar BOS/CHoCH label and the trend state *after* each bar.

    The automaton is inherently sequential (each bar depends on the previous
    trend), so this is one of the few loops in the indicator layer; it is O(n) over
    int8 input.  A break with the trend (or the initialising first break) is a
    ``bos``; a break against it is a ``choch`` and flips the trend.
    """
    n = break_dir.size
    events: list[BreakEvent | None] = [None] * n
    trends = np.zeros(n, dtype=np.int8)
    trend = 0
    for i in range(n):
        direction = int(break_dir[i])
        if direction != 0:
            events[i] = "choch" if trend != 0 and direction != trend else "bos"
            trend = direction
        trends[i] = trend
    return events, trends


def structure_breaks(df: pd.DataFrame, cfg: StructureConfig | None = None) -> pd.DataFrame:
    """Return the prod break signal, the BOS/CHoCH label and the trend state.

    Columns (same index as ``df``):

    * ``break_dir`` - ``int8`` prod parity (core.py lines 362-365): ``+1`` when
      the close (or the high when ``confirmation="wick"``) is beyond the last
      *known* swing high, ``-1`` for the mirrored case, ``0`` otherwise.  The
      upward probe wins when both fire, exactly like prod's ``if / elif``.
    * ``structure_event`` - ``"bos"`` / ``"choch"`` / ``None`` for that same bar.
    * ``trend`` - ``int8`` trend state after the bar: ``+1``, ``-1`` or ``0`` while
      no break has happened yet.

    The break is always measured against a trailing swing, never against the swing
    of the current bar, so substituting a future candle cannot change past rows.
    """
    config = StructureConfig() if cfg is None else cfg
    levels = known_swing_levels(df, config)
    last_high = levels["last_swing_high"].to_numpy(dtype=float)
    last_low = levels["last_swing_low"].to_numpy(dtype=float)
    if config.confirmation == "close":
        probe_up = df["close"].to_numpy(dtype=float)
        probe_down = probe_up
    else:  # "wick": a touch of the level already counts as a break
        probe_up = df["high"].to_numpy(dtype=float)
        probe_down = df["low"].to_numpy(dtype=float)
    up = last_high < probe_up  # NaN levels compare False -> no break before a swing exists
    down = last_low > probe_down
    break_dir = np.where(up, 1, np.where(down, -1, 0)).astype(np.int8)
    events, trends = _run_trend_automaton(break_dir)
    return pd.DataFrame(
        {
            "break_dir": break_dir,
            "structure_event": pd.Series(events, index=df.index, dtype="object"),
            "trend": trends,
        },
        index=df.index,
    )

