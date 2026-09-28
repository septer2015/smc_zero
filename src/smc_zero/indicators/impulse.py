"""Displacement gate: the formal impulse filter of п.7 / C3 (SPEC_SMC.md §7.5).

Prod never measured the impulse - ``BOS_MIN_BREAK_PIP`` is a break distance in pips
and no ATR exists in the prod sources - so nothing here is a parity port: the
definition is pinned by tests and mutation gates only.

* :func:`atr_wilder` - Wilder's ATR.  ``ATR[c]`` reads bars ``<= c`` only (the true
  range of bar ``c`` is built from its own high/low and the *previous* close), so a
  future candle can never change it;
* the gate is evaluated on confirming bars only (``breaks["break_dir"] != 0``) and
  takes the broken level from ``breaks["break_level"]``; the structure is never
  recomputed inside this module;
* ``disp_atr`` = ``|close[c] - close[c - leg_bars]| / ATR[c]`` - a zero or missing ATR
  makes the condition false instead of exploding;
* ``disp_body_frac`` = ``sum|close - open| / sum(high - low)`` over the leg bars
  ``(c - leg_bars + 1 .. c)`` - a zero leg range makes the condition false;
* ``disp_close_beyond`` - the confirming bar closed beyond the broken level;
* ``disp_no_fast_return`` - no return event during the ``no_return_bars`` bars after ``c``
  (§7.5 п.20): for an up break a return is a close *below* the broken level, for a down
  break a close *above* it - continuing in the direction of the break is not a return.  A
  close exactly *on* the level is not a return either (non-strict ``>=`` / ``<=`` against
  the level);
* ``disp_ok`` - all of the above.  **Unknown counts as false**: while the return
  window still reaches past the last bar of the frame the gate is not satisfied
  (``disp_known_at`` already points beyond the frame, so a consumer filtering on
  ``known_at <= i`` never sees such a row);
* every gate column carries ``disp_known_at = c + no_return_bars`` - the bar at which
  the gate may first be read.  Rows without a break carry no gate at all
  (NaN/False/NA).

All functions are pure: neither ``df`` nor ``breaks`` is modified, and nothing is
cached.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smc_zero.config import DisplacementConfig
from smc_zero.indicators._markup import known_at

BREAK_COLUMNS = ("break_dir", "break_level")


def atr_wilder(df: pd.DataFrame, period: int) -> pd.Series:
    """Return Wilder's ATR, whose value at bar ``c`` uses bars up to ``c`` only.

    The first ``period - 1`` bars are NaN (no full window yet), bar ``period - 1``
    carries the mean true range of the first window, and every later bar is Wilder's
    recursion ``ATR[c] = (ATR[c - 1] * (period - 1) + TR[c]) / period``.  The recursion
    is expressed as an ``ewm(alpha=1/period, adjust=False)`` seeded with that mean, so
    it stays vectorised over the M15 series (equality with a naive loop is pinned by
    ``tests/test_impulse.py``).
    """
    if period < 1:
        raise ValueError("period must be >= 1")
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    n = close.size
    atr = pd.Series(np.nan, index=df.index, dtype=float)
    if n < period:
        return atr
    previous_close = np.empty(n, dtype=float)
    previous_close[0] = np.nan
    previous_close[1:] = close[:-1]
    true_range = np.maximum(
        high - low, np.maximum(np.abs(high - previous_close), np.abs(low - previous_close))
    )
    true_range[0] = high[0] - low[0]  # no previous close on the first bar
    seed = float(true_range[:period].mean())
    smoothed = pd.Series(true_range, index=df.index)
    smoothed.iloc[: period - 1] = np.nan
    smoothed.iloc[period - 1] = seed
    return smoothed.ewm(alpha=1.0 / period, adjust=False).mean()


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Return ``numerator / denominator``, NaN wherever the denominator is not positive."""
    out = np.full(numerator.shape, np.nan, dtype=float)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def _no_fast_return(
    close: np.ndarray,
    level: np.ndarray,
    up_rows: np.ndarray,
    down_rows: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (no return event happened, the return window has fully closed).

    A *return event* (§7.5 п.20) is a close back beyond the broken level: for an up break
    any close *below* the level ``L`` inside ``c + 1 .. c + window``, for a down break any
    close *above* ``L``.  Continuing in the direction of the break is not a return, and a
    close exactly on ``L`` is not a return either - so the window must stay ``>= L`` for an
    up break and ``<= L`` for a down break (non-strict comparisons).  A window that reaches
    past the frame end is not mature, so it counts as "returned" for
    :func:`displacement_gate` - an unknown gate must never look satisfied.
    """
    n = close.size
    no_return = np.zeros(n, dtype=bool)
    if window == 0:  # empty window: nothing can violate and the gate is known at c
        no_return[:] = True
        return no_return, np.ones(n, dtype=bool)
    series = pd.Series(close)
    rolling = series.rolling(window, min_periods=window)
    # The rolling window at bar j covers j-window+1..j; shifting by -window lands on c, so at
    # c these two series hold the low/high of the future window c+1..c+window.  Both are NaN
    # while that window is not fully inside the frame.
    window_low = rolling.min().shift(-window).to_numpy()
    window_high = rolling.max().shift(-window).to_numpy()
    mature = ~np.isnan(window_high)
    no_return[up_rows] = mature[up_rows] & (window_low[up_rows] >= level[up_rows])
    no_return[down_rows] = mature[down_rows] & (window_high[down_rows] <= level[down_rows])
    return no_return, mature


def displacement_gate(
    df: pd.DataFrame,
    breaks: pd.DataFrame,
    cfg: DisplacementConfig | None = None,
) -> pd.DataFrame:
    """Return the impulse gate columns for the confirming bars of ``breaks``.

    Columns (same index as ``df``): ``disp_atr``, ``disp_body_frac``,
    ``disp_close_beyond``, ``disp_no_fast_return``, ``disp_ok`` and ``disp_known_at``
    (``Int64`` position ``c + no_return_bars``).  ``breaks`` must be the output of
    :func:`smc_zero.indicators.structure.structure_breaks` (or any frame carrying the
    same two columns) and must line up with ``df`` row for row.
    """
    config = DisplacementConfig() if cfg is None else cfg
    missing = [name for name in BREAK_COLUMNS if name not in breaks.columns]
    if missing:
        raise ValueError(f"breaks must provide {BREAK_COLUMNS}; missing {missing}")
    if len(breaks) != len(df):
        raise ValueError(f"breaks has {len(breaks)} rows but the frame has {len(df)}")

    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    n = close.size
    break_dir = breaks["break_dir"].to_numpy(dtype=np.int8)
    level = breaks["break_level"].to_numpy(dtype=float)
    candidates = break_dir != 0

    atr = atr_wilder(df, config.atr_period).to_numpy(dtype=float)
    lag = config.leg_bars
    leg_move = np.full(n, np.nan, dtype=float)
    if n > lag:
        leg_move[lag:] = np.abs(close[lag:] - close[:-lag])
    disp_atr = _safe_ratio(leg_move, atr)

    body_sum = pd.Series(np.abs(close - open_), index=df.index).rolling(lag).sum().to_numpy()
    range_sum = pd.Series(high - low, index=df.index).rolling(lag).sum().to_numpy()
    disp_body_frac = _safe_ratio(body_sum, range_sum)

    up_rows = break_dir == 1
    down_rows = break_dir == -1
    beyond = np.zeros(n, dtype=bool)
    beyond[up_rows] = close[up_rows] > level[up_rows]
    beyond[down_rows] = close[down_rows] < level[down_rows]

    no_return, _ = _no_fast_return(close, level, up_rows, down_rows, config.no_return_bars)

    ok = (
        candidates
        & (disp_atr >= config.atr_mult_min)
        & (disp_body_frac >= config.body_frac_min)
        & beyond
        & no_return
    )
    return pd.DataFrame(
        {
            "disp_atr": disp_atr,
            "disp_body_frac": disp_body_frac,
            "disp_close_beyond": beyond & candidates,
            "disp_no_fast_return": no_return & candidates,
            "disp_ok": ok,
            "disp_known_at": known_at(candidates, config.no_return_bars, df.index),
        },
        index=df.index,
    )
