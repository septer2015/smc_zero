"""Liquidity levels and sweeps (stub, no logic yet).

Planned contract: equal highs/lows within ``LiquidityConfig.equal_tol``, plus
PDH/PDL (previous day high/low) and session highs/lows.  A sweep is a wick beyond
such a level followed by a close *back inside* (``sweep_mode``), so the event is
only known once that candle closes.
"""

from __future__ import annotations

# TODO(phase-indicators): equal high/low level clustering with equal_tol.
# TODO(phase-indicators): PDH/PDL and session levels taken from closed HTF bars.
# TODO(phase-indicators): sweep events per sweep_mode.
# TODO(tests): synthetic OHLCV test + leak test.
