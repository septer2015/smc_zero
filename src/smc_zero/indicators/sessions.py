"""Trading sessions (stub, no logic yet).

Planned contract: session labels for the H1 working timeframe (Asia / London /
New York) derived from UTC stamps *after* the loader has localized them, plus the
session open/close boundaries used by the liquidity module.  Session boundaries
are configuration, not constants in code.
"""

from __future__ import annotations

# TODO(phase-indicators): session label column from TimeframeConfig-compatible config.
# TODO(phase-indicators): expose session open/close ranges for liquidity levels.
# TODO(tests): synthetic OHLCV test on UTC boundary stamps.
