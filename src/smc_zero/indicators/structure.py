"""Market structure: swings, BOS and CHoCH (stub, no logic yet).

Planned contract (constitution "Определения SMC" + :class:`smc_zero.config.StructureConfig`):

* ``swing_high``/``swing_low`` = local extreme with ``swing_lookback`` candles on
  both sides, confirmed by the close (or wick, per ``confirmation``) of the right
  candle, hence known only at bar ``i + swing_lookback``.
* BOS = close beyond the last swing *with* the trend (continuation).
* CHoCH = close *against* the trend, breaking the last counter-swing.
* Vectorised pandas/numpy only; every output column must carry the bar index at
  which the fact became known (``known_at``) to make the look-ahead test easy.
"""

from __future__ import annotations

# TODO(phase-indicators): implement swing detection with explicit confirmation lag.
# TODO(phase-indicators): implement trend state, BOS and CHoCH label columns.
# TODO(tests): synthetic OHLCV test + future-candle substitution leak test.
