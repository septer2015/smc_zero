"""Liquidity levels and sweeps.

Э1'.4 ports the prod sweep detector ``find_sweep_arr`` (core.py lines 507-527):

* the search window is ``[current_idx - sweep_lookback, current_idx]`` - the evaluated
  bar is *included*, because prod reads the signal on the close of that bar;
* an upper level is swept when a high pierces ``level + sweep_buffer`` and the close
  comes back inside, i.e. below that very threshold.  Prod compares against the
  *buffered* threshold, not the raw level, so a close that sits between the level and
  the threshold still counts as "back inside"; the mirror rule holds for a lower level
  (``level - sweep_buffer``, a low below it, a close above it);
* ``sweep_buffer`` is in price units, like ``RiskConfig.spread``; prod feeds it as
  ``sweep_buffer_pip * pip_size`` (core.py line 1321);
* the index returned is the *most extreme* qualifying bar in the window (highest high
  for an upper sweep, lowest low for a lower one) and the earliest one on a tie -
  prod's strict ``>``/``<`` update rule.  ``None`` means "no sweep", which the
  strategy turns into "no setup".

``LiquidityConfig.sweep_mode`` keeps prod's rule (``"wick_close_inside"``, the default,
which is also spec п.5: shadow beyond the level plus a close back inside) or relaxes
it to a bare pierce (``"wick_only"``).

Still out of scope here (Э3'): equal high/low clustering with ``equal_tol`` and the
PDH/PDL + session level maps built from :mod:`smc_zero.indicators.sessions`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smc_zero.config import LiquidityConfig


def sweep_index(
    df: pd.DataFrame,
    *,
    level: float,
    upper: bool,
    current_idx: int,
    cfg: LiquidityConfig | None = None,
) -> int | None:
    """Return the positional bar that swept ``level`` within the last ``sweep_lookback`` bars.

    ``upper`` selects the direction (an upper level is swept upwards), ``current_idx``
    is the bar whose close is being evaluated.  Nothing after ``current_idx`` is ever
    read, so the answer cannot change when a future candle is substituted.
    """
    config = LiquidityConfig() if cfg is None else cfg
    if len(df) == 0:
        return None
    if not 0 <= current_idx < len(df):
        raise IndexError(f"current_idx {current_idx} outside the frame of {len(df)} bars")
    window = slice(max(0, current_idx - config.sweep_lookback), current_idx + 1)
    high = df["high"].to_numpy(dtype=float)[window]
    low = df["low"].to_numpy(dtype=float)[window]
    close = df["close"].to_numpy(dtype=float)[window]
    if upper:
        threshold = level + config.sweep_buffer
        pierced, extreme = high > threshold, high
    else:
        threshold = level - config.sweep_buffer
        pierced, extreme = low < threshold, low
    if config.sweep_mode == "wick_close_inside":
        back_inside = close < threshold if upper else close > threshold
        pierced = pierced & back_inside
    if not pierced.any():
        return None
    candidates = np.where(pierced, extreme, -np.inf if upper else np.inf)
    offset = int(np.argmax(candidates) if upper else np.argmin(candidates))
    return window.start + offset

