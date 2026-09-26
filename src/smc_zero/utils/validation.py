"""Reusable validation helpers (stub, no logic yet).

The OHLCV checks that guard ``./data`` already live in
:func:`smc_zero.data_loader.validate_ohlcv` because they run inside the loader,
so this module is reserved for checks shared by later phases: asserting that a
frame carries ``is_closed``, that HTF columns were stitched by
:func:`smc_zero.data_loader.align_htf_to_ltf`, and that ``close_time`` is
consistent with ``timestamp`` plus the timeframe period.
"""

from __future__ import annotations

# TODO(phase-utils): assert_closed(frame) guard for strategy inputs.
# TODO(phase-utils): assert_no_lookahead helpers reused by indicator tests.
# TODO(phase-utils): assert_close_time_consistent(frame, period).
