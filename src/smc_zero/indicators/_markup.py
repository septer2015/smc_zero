"""Shared helpers for indicator markup columns (private module).

The only shared concern of the indicator layer is how a markup fact records the
bar at which it became known, so that one implementation carries the pandas
alignment trap documented in :func:`known_at`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def known_at(mask: np.ndarray, offset: int, index: pd.Index) -> pd.Series:
    """Return ``position + offset`` for every ``True`` in ``mask``, NA elsewhere.

    ``offset`` is the confirmation lag: N-bar swing at bar ``i`` -> ``i + N``,
    three-candle FVG with middle candle ``k`` -> ``k + 1``.  The result is built on
    the *input* index on purpose - pandas aligns on labels when a Series is put
    into a DataFrame constructor, so a freshly built ``RangeIndex`` series would
    silently become NA for a DatetimeIndex frame.
    """
    positions = np.arange(mask.size)
    values = np.where(mask, positions + offset, np.nan)
    return pd.Series(values, index=index).astype("Int64")
