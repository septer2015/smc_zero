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
BreakEvent: TypeAlias = Literal["bos", "choch"]
# Where a limit order sits inside an entry gap: ``proximal`` = the edge price
# reaches first (top of a bullish gap, bottom of a bearish one), ``mid`` = centre.
FVGEntryMode: TypeAlias = Literal["proximal", "mid"]
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
    """Swing / BOS / CHoCH detection parameters (SPEC_SMC.md, C4).

    ``swing_lookback`` is the symmetric number of candles on both sides of a
    swing: the swing is a strict N-bar fractal and only becomes known after the
    right candles close, i.e. at bar ``i + swing_lookback``.  The default is ``1``
    for prod-fractal compatibility (prod's formula, core.py lines 339-343, is
    exactly the 1-bar case); tests must cover ``N > 1`` as well.

    ``confirmation`` decides *what must cross* a swing level to count as a break:
    ``"close"`` is prod parity and the spec default (a break is a close beyond the
    level, never a wick), ``"wick"`` makes a touch count.  Swing detection itself
    is always the strict wick fractal - the two are independent, and the
    interpretation is recorded in the Э1'.2 report as a spec clarification.
    """

    swing_lookback: int = 1
    confirmation: Confirmation = "close"

    def __post_init__(self) -> None:
        if self.swing_lookback < 1:
            raise ValueError("swing_lookback must be >= 1")


@dataclass(frozen=True, slots=True)
class DisplacementConfig:
    """Formal impulse (displacement) thresholds - spec п.7, conflict C3.

    Prod never measured the impulse: its only relative threshold,
    ``BOS_MIN_BREAK_PIP``, is a break *distance in pips*, not a body/ATR ratio, and
    no ATR is computed in the prod sources.  These thresholds therefore have no prod
    counterpart to compare against and are validated by tests plus mutation gates
    (SPEC_SMC.md §7.5).  Defaults are inert: with ``atr_mult_min = 0``,
    ``body_frac_min = 0`` and ``no_return_bars = 0`` the gate blocks nothing, so a
    real run must set them explicitly.

    * ``atr_period`` - Wilder ATR period the impulse leg is normalised by;
    * ``atr_mult_min`` - minimum ``|close[c] - close[c - leg_bars]| / ATR[c]``;
    * ``body_frac_min`` - minimum share of the leg range covered by candle bodies;
    * ``no_return_bars`` - bars *after* the confirming bar during which no close may
      come back beyond the broken level (``0`` = no waiting, the gate is known at the
      confirming bar itself);
    * ``leg_bars`` - length of the impulse leg in bars.
    """

    atr_period: int = 14
    atr_mult_min: float = 0.0
    body_frac_min: float = 0.0
    no_return_bars: int = 0
    leg_bars: int = 1

    def __post_init__(self) -> None:
        if self.atr_period < 1:
            raise ValueError("atr_period must be >= 1")
        if self.leg_bars < 1:
            raise ValueError("leg_bars must be >= 1")
        if self.no_return_bars < 0:
            raise ValueError("no_return_bars must be >= 0")
        if min(self.atr_mult_min, self.body_frac_min) < 0:
            raise ValueError("atr_mult_min and body_frac_min must be >= 0")


@dataclass(frozen=True, slots=True)
class FVGConfig:
    """Three-candle fair value gap detection parameters (SPEC_SMC.md, п.8).

    The gap is the strict wick imbalance of the triple ``(k - 1, k, k + 1)`` and is
    only visible from bar ``k + 1`` on.  ``min_gap_size`` is applied at detection
    (``size >= min_gap_size``, inclusive, like prod's lookup filter) - narrower gaps
    are not marked at all.  Where the limit order sits inside the gap is *not* a
    detection parameter and lives in ``EntryConfig.fvg_entry_mode`` (Э4').
    """

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
    """Liquidity levels and sweep detection parameters.

    ``equal_tol`` clusters equal highs/lows (Э3').  ``sweep_buffer`` is the price
    distance a wick must exceed a level by, in *price units* like
    ``RiskConfig.spread`` (prod feeds ``sweep_buffer_pip * pip_size``);
    ``sweep_lookback`` is the prod search window (``sweep_lookback``, default 48),
    counted backwards from the evaluated bar *including* it.  ``sweep_mode`` keeps
    prod's rule (a wick beyond the buffered threshold plus a close back inside it)
    or relaxes it to a bare pierce.
    """

    equal_tol: float = 0.0
    sweep_mode: SweepMode = "wick_close_inside"
    sweep_buffer: float = 0.0
    sweep_lookback: int = 48

    def __post_init__(self) -> None:
        if self.equal_tol < 0:
            raise ValueError("equal_tol must be >= 0")
        if self.sweep_buffer < 0:
            raise ValueError("sweep_buffer must be >= 0")
        if self.sweep_lookback < 1:
            raise ValueError("sweep_lookback must be >= 1")


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
