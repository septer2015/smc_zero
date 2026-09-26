"""Fair value gaps: three-candle imbalance with an explicit visibility delay.

SPEC_SMC.md п.8: the gap is defined by the triple ``(k - 1, k, k + 1)`` and becomes
available only from bar ``k + 1`` on (prod enforces the same lag by capping its
search at ``current_idx - 1``, core.py line 471), so every row carries
``*_known_at = k + 1`` - reading the flag on bar ``k`` itself is forbidden.

Bullish (demand) gap, prod's ``fvg_*_lower`` arrays, ``high[k - 1] < low[k + 1]``:
the zone is ``[high[k - 1], low[k + 1]]`` and is bought on the way down.  Bearish
(supply) gap, prod's ``fvg_*_upper`` arrays, ``low[k - 1] > high[k + 1]``: the zone
is ``[high[k + 1], low[k - 1]]`` and is sold on the way up.  Both are the prod wick
formulas (core.py lines 398-413) and stay strict, so touching wicks produce no gap.

Gap *fills* are deliberately not tracked here: prod consumes them in its order layer
(Э2'), while ``FVGConfig.min_gap_size`` filters at detection time, exactly like
prod's ``size >= min_size`` lookup filter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smc_zero.config import FVGConfig, FVGEntryMode
from smc_zero.indicators._markup import known_at

ENTRY_MODES: tuple[FVGEntryMode, ...] = ("proximal", "mid")


def fair_value_gaps(df: pd.DataFrame, cfg: FVGConfig | None = None) -> pd.DataFrame:
    """Return the bullish/bearish three-candle gaps of ``df``.

    Columns (same index as ``df``, one row per bar); the flag and both bounds live on
    the *middle* bar ``k``:

    * ``bullish`` / ``bearish`` - ``bool``;
    * ``bullish_bottom`` / ``bullish_top`` / ``bullish_size`` and the bearish twins -
      zone bounds and width (NaN/NaN/0.0 without a gap, as prod initialises its
      arrays);
    * ``bullish_known_at`` / ``bearish_known_at`` - ``Int64`` position ``k + 1`` of the
      bar that confirms the gap, NA elsewhere.

    The first and last bars can never be middle candles, so the tail of the frame
    carries no markup (the right candle has to close first).
    """
    config = FVGConfig() if cfg is None else cfg
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n = high.size
    bullish = np.zeros(n, dtype=bool)
    bearish = np.zeros(n, dtype=bool)
    bull_bottom = np.full(n, np.nan, dtype=float)
    bull_top = np.full(n, np.nan, dtype=float)
    bull_size = np.zeros(n, dtype=float)
    bear_bottom = np.full(n, np.nan, dtype=float)
    bear_top = np.full(n, np.nan, dtype=float)
    bear_size = np.zeros(n, dtype=float)
    if n >= 3:
        left_high, left_low = high[:-2], low[:-2]  # bars k - 1
        right_high, right_low = high[2:], low[2:]  # bars k + 1
        bull_width = right_low - left_high
        bear_width = left_low - right_high
        bullish[1:-1] = (right_low > left_high) & (bull_width >= config.min_gap_size)
        bearish[1:-1] = (left_low > right_high) & (bear_width >= config.min_gap_size)
        bull_bottom[1:-1] = np.where(bullish[1:-1], left_high, np.nan)
        bull_top[1:-1] = np.where(bullish[1:-1], right_low, np.nan)
        bull_size[1:-1] = np.where(bullish[1:-1], bull_width, 0.0)
        bear_bottom[1:-1] = np.where(bearish[1:-1], right_high, np.nan)
        bear_top[1:-1] = np.where(bearish[1:-1], left_low, np.nan)
        bear_size[1:-1] = np.where(bearish[1:-1], bear_width, 0.0)
    markup = pd.DataFrame(
        {
            "bullish": bullish,
            "bearish": bearish,
            "bullish_bottom": bull_bottom,
            "bullish_top": bull_top,
            "bullish_size": bull_size,
            "bearish_bottom": bear_bottom,
            "bearish_top": bear_top,
            "bearish_size": bear_size,
            "bullish_known_at": known_at(bullish, 1, df.index),
            "bearish_known_at": known_at(bearish, 1, df.index),
        },
        index=df.index,
    )
    return markup


def entry_level(top: float, bottom: float, mode: FVGEntryMode, *, bullish: bool) -> float:
    """Return the limit price inside a gap for the ``proximal``/``mid`` entry mode.

    ``proximal`` is the edge price reaches first - the top of a demand gap (bought on
    the way down) or the bottom of a supply gap (sold on the way up), matching prod's
    ``fvg[1] if is_upper else fvg[0]`` (core.py lines 705-706); ``mid`` is the centre.
    Prod additionally knows a ``distal`` edge (core.py lines 707-708) which п.8 keeps
    out of v1 scope.
    """
    if mode not in ENTRY_MODES:
        raise ValueError(f"unsupported entry mode {mode!r}; expected one of {ENTRY_MODES}")
    if mode == "mid":
        return (top + bottom) / 2.0
    return top if bullish else bottom

