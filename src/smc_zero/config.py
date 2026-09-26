"""Typed configuration objects for the SMC backtest project.

Every threshold used by indicators, the strategy and the backtester lives in a
dataclass here: strategy code must not contain magic numbers (constitution rule
5).  Defaults are intentionally inert (zero tolerances, zero costs) wherever a
value is project specific, so a concrete run has to set them explicitly - in
particular :class:`RiskConfig` must carry non-zero costs before any result is
called profitable (rule 4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

# Timeframes supported by the pipeline: M15 entry, H1 structure, D1 bias.
Timeframe: TypeAlias = Literal["M15", "H1", "D1"]
Confirmation: TypeAlias = Literal["close", "wick"]
SweepMode: TypeAlias = Literal["wick_close_inside", "wick_only"]
Season: TypeAlias = Literal["summer", "winter"]
Killzone: TypeAlias = Literal["prelondon", "london", "ny"]
# Half-open MSK hour window: ``start <= hour < end``.
HourWindow: TypeAlias = tuple[int, int]


@dataclass(frozen=True, slots=True)
class TimeframeConfig:
    """Working timeframe hierarchy (M5 and H4 are deliberately absent)."""

    ltf: Timeframe = "M15"
    mtf: Timeframe = "H1"
    htf: Timeframe = "D1"


@dataclass(frozen=True, slots=True)
class SessionConfig:
    """Killzone gate: seasonal session windows in MSK hours (SPEC_SMC.md, C1).

    Moscow is UTC+3 all year round, so the boundaries are MSK hours and the
    season only selects *which* pair applies: London follows the EU DST calendar,
    New York the US one.  Both calendars live in :mod:`smc_zero.utils.time` and
    are consumed only by :mod:`smc_zero.indicators.sessions`, so the Э3' session
    level maps (``london_*`` / ``ny_*``) can reuse this very table instead of
    re-declaring windows.

    ``use_kz`` is ``True`` by default (prod's gate defaulted to *off*).
    ``prelondon`` is off by default: C1 leaves open whether prod's 07-09 MSK
    window survives.  Windows are half-open in MSK hours.
    """

    use_kz: bool = True
    prelondon: bool = False
    prelondon_msk: HourWindow = (7, 9)
    london_summer_msk: HourWindow = (9, 12)
    london_winter_msk: HourWindow = (10, 13)
    ny_summer_msk: HourWindow = (14, 17)
    ny_winter_msk: HourWindow = (15, 18)

    def __post_init__(self) -> None:
        windows: dict[str, HourWindow] = {
            "prelondon_msk": self.prelondon_msk,
            "london_summer_msk": self.london_summer_msk,
            "london_winter_msk": self.london_winter_msk,
            "ny_summer_msk": self.ny_summer_msk,
            "ny_winter_msk": self.ny_winter_msk,
        }
        for name, window in windows.items():
            start, end = window
            if not 0 <= start < end <= 24:
                raise ValueError(f"{name} must satisfy 0 <= start < end <= 24, got {window}")


@dataclass(frozen=True, slots=True)
class StructureConfig:
    """Swing / BOS / CHoCH detection parameters.

    ``swing_lookback`` is the symmetric number of candles on both sides of a
    swing (the swing only becomes known after the right candle closes, i.e. at
    bar ``i + swing_lookback``).
    """

    swing_lookback: int = 3
    confirmation: Confirmation = "close"

    def __post_init__(self) -> None:
        if self.swing_lookback < 1:
            raise ValueError("swing_lookback must be >= 1")


@dataclass(frozen=True, slots=True)
class FVGConfig:
    """Three-candle fair value gap detection parameters."""

    min_gap_size: float = 0.0

    def __post_init__(self) -> None:
        if self.min_gap_size < 0:
            raise ValueError("min_gap_size must be >= 0")


@dataclass(frozen=True, slots=True)
class OBConfig:
    """Order block detection parameters.

    ``displacement_threshold`` is expressed in price units (it becomes an ATR
    multiple once the displacement indicator defines its unit) and is compared
    against the impulse candle body size.
    """

    displacement_threshold: float = 0.0
    require_structure_break: bool = True

    def __post_init__(self) -> None:
        if self.displacement_threshold < 0:
            raise ValueError("displacement_threshold must be >= 0")


@dataclass(frozen=True, slots=True)
class LiquidityConfig:
    """Liquidity sweep and equal high/low tolerance parameters."""

    equal_tol: float = 0.0
    sweep_mode: SweepMode = "wick_close_inside"

    def __post_init__(self) -> None:
        if self.equal_tol < 0:
            raise ValueError("equal_tol must be >= 0")


@dataclass(frozen=True, slots=True)
class RiskConfig:
    """Position sizing and trading costs.

    ``commission`` is charged per trade in account currency, ``spread`` and
    ``slippage`` are expressed in price units.  :attr:`has_costs` is the guard
    used by the reporting layer: without costs a positive equity curve must not
    be presented as a profit.
    """

    risk_pct: float = 1.0
    rr: float = 2.0
    commission: float = 0.0
    spread: float = 0.0
    slippage: float = 0.0

    def __post_init__(self) -> None:
        if self.risk_pct <= 0:
            raise ValueError("risk_pct must be > 0")
        if self.rr <= 0:
            raise ValueError("rr must be > 0")
        if min(self.commission, self.spread, self.slippage) < 0:
            raise ValueError("commission, spread and slippage must be >= 0")

    @property
    def has_costs(self) -> bool:
        """``True`` when commission, spread and slippage are all set."""
        return self.commission > 0 and self.spread > 0 and self.slippage > 0


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Backtest wiring: capital, costs and the unclosed-tail protection."""

    initial_capital: float = 10_000.0
    drop_unclosed: bool = True
    risk: RiskConfig = field(default_factory=RiskConfig)
    timeframes: TimeframeConfig = field(default_factory=TimeframeConfig)

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be > 0")
