"""Time helpers (stub, no logic yet).

Planned contract: session/timezone utilities that build on the loader's UTC
normalisation - converting active session windows given a broker offset,
generating bar ranges for a timeframe, and merging overlapping trading calendars
(weekend gaps in FX) without ever shifting the bar labels off their open stamp.
"""

from __future__ import annotations

# TODO(phase-utils): session window helpers on UTC stamps.
# TODO(phase-utils): timeframe grid helpers reusing data_loader.period_for.
# TODO(phase-utils): weekend/holiday gap detection for FX calendars.
