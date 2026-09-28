"""SMC backtest toolkit: D1 bias -> H1 structure -> M15 entries.

Only configuration objects are re-exported here so that ``import smc_zero``
stays cheap; the data pipeline is imported explicitly from
:mod:`smc_zero.data_loader`.
"""

from __future__ import annotations

from smc_zero.config import (
    BacktestConfig,
    DisplacementConfig,
    FVGConfig,
    LiquidityConfig,
    OBConfig,
    RiskConfig,
    SessionConfig,
    StructureConfig,
    TimeframeConfig,
    WalkForwardConfig,
)

__version__ = "0.1.0"

__all__ = [
    "BacktestConfig",
    "DisplacementConfig",
    "FVGConfig",
    "LiquidityConfig",
    "OBConfig",
    "RiskConfig",
    "SessionConfig",
    "StructureConfig",
    "TimeframeConfig",
    "WalkForwardConfig",
    "__version__",
]
