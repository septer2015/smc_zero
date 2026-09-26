"""Order blocks (stub, no logic yet).

Planned contract: the last opposite-coloured candle before the impulse that
caused a BOS/CHoCH, valid when the impulse shows displacement (body larger than
``OBConfig.displacement_threshold``) and/or left an FVG behind.  Zone boundaries
plus the bar index at which the block became known must be returned, so the
strategy can only act on closed information.
"""

from __future__ import annotations

# TODO(phase-indicators): derive displacement from the structure break events.
# TODO(phase-indicators): return (zone_top, zone_bottom, created_at, mitigated_at).
# TODO(tests): synthetic OHLCV test + leak test.
