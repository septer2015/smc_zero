"""Fair value gaps / three-candle imbalance (stub, no logic yet).

Planned contract: bull FVG when ``low[i + 1] > high[i - 1]``, bear FVG when
``high[i + 1] < low[i - 1]``; only closed candles participate, the gap is known
at ``i + 1`` and must be at least ``FVGConfig.min_gap_size`` wide.  Gap fill
tracking must use only bars that close after the gap became known.
"""

from __future__ import annotations

# TODO(phase-indicators): vectorised three-candle gap detection with min_gap_size.
# TODO(phase-indicators): track partial/full fills without look-ahead.
# TODO(tests): synthetic OHLCV test + leak test.
