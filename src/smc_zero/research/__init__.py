"""Research-only entry points of the project (Э17.z++).

Nothing in this package belongs to a live run.  It collects the switches that exist to *measure*
one part of the strategy by comparing the run with and without that part - the same idea as
``SessionConfig.research_ignore_sessions`` of Э17.z, which measures the hour filter by turning it
off.  Every entry point keeps the shipped behaviour as its default, so a caller that names nothing
gets exactly the run of the project.
"""

from __future__ import annotations

from smc_zero.research.bias_gate import (
    BIAS_GATE_MODES,
    DEFAULT_BIAS_GATE_MODE,
    defined_majority,
    direction_frame,
)

__all__ = [
    "BIAS_GATE_MODES",
    "DEFAULT_BIAS_GATE_MODE",
    "defined_majority",
    "direction_frame",
]
