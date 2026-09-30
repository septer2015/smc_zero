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

# Timeframes supported by the pipeline: M15 entry, H1 structure, H4/D1 bias.
Timeframe: TypeAlias = Literal["M15", "H1", "H4", "D1"]
Confirmation: TypeAlias = Literal["close", "wick"]
SweepMode: TypeAlias = Literal["wick_close_inside", "wick_only"]
Season: TypeAlias = Literal["summer", "winter"]
Killzone: TypeAlias = Literal["prelondon", "london", "ny"]
BreakEvent: TypeAlias = Literal["bos", "choch"]
# What ``bias_dir`` does when the bias timeframes disagree: ``no_trade`` zeroes it,
# ``reduced_risk`` is a deferred decision (SPEC_SMC.md §7.6) and is not implemented
# by v1 - :func:`smc_zero.indicators.bias.bias_frames` raises for it.
ConflictPolicy: TypeAlias = Literal["no_trade", "reduced_risk"]
# How many timeframes have to point the same way for a direction: ``unanimous`` is the
# owner ruling of v1 (SPEC_SMC.md §7.6 п.24 - a 2-1 split is a conflict, not a signal),
# ``majority`` accepts a *strict* majority of a 2-1 split as well; it is the A/B factor
# of Э7' (SPEC_SMC.md §7.11 п.70) and changes nothing else about the bias.
Agreement: TypeAlias = Literal["unanimous", "majority"]
#: The agreement modes as a value, for runtime validation and for the optimizer's range.
AGREEMENTS: tuple[Agreement, ...] = ("unanimous", "majority")
# Per-bar verdict of the multi-timeframe bias: every timeframe agrees long / short
# (``agree_*``), a strict majority does (``majority_*``, reachable under ``majority``
# only), they disagree (``conflict``) or at least one has no trend yet (``undefined``).
BiasState: TypeAlias = Literal[
    "agree_long", "agree_short", "majority_long", "majority_short", "conflict", "undefined"
]
# Headline metric of the out-of-sample folds the optimizer maximises (SPEC_SMC.md §7.11
# п.69): the fold mean of the Sharpe ratio or of the profit percentage of a fold.
ScoreMetric: TypeAlias = Literal["sharpe", "profit"]
#: The two score metrics as a value, for runtime validation of :class:`OptunaConfig`.
SCORE_METRICS: tuple[ScoreMetric, ...] = ("sharpe", "profit")
# Where a limit order sits inside an entry gap: ``proximal`` = the edge price
# reaches first (top of a bullish gap, bottom of a bearish one), ``mid`` = centre.
FVGEntryMode: TypeAlias = Literal["proximal", "mid"]
# Half-open MSK hour window: ``start <= hour < end``.
HourWindow: TypeAlias = tuple[int, int]
# MSK time of day as ``(hour, minute)``: the broker's session boundaries (C6).
TimeOfDay: TypeAlias = tuple[int, int]
# Where a take-profit may come from (SPEC_SMC.md, C2): the nearest counter-side
# liquidity level visible at the entry bar, the RR fallback only, or the level with
# the RR fallback as a second chance - the C2 default.  ``"liquidity"`` and ``"rr"``
# need a ruling about the empty-book / RR-grid cases and are not implemented by v1
# (SPEC_SMC.md §7.8 п.36).
TPMode: TypeAlias = Literal["liquidity", "rr", "liquidity_with_rr_fallback"]
# How a gap is chosen among the gaps inside the lookback window: prod's ``"first"``
# (parity, default) or ``"biggest"`` (prod's alternative branch, deferred by §7.8).
FVGSelect: TypeAlias = Literal["first", "biggest"]
# Which level state may carry a setup.  v1 trades fresh levels only: the breaker /
# retest lifecycle of spec п.4 is a separate setup left out of v1 (§7.8 п.33).
SetupType: TypeAlias = Literal["fresh", "breaker"]
# The SL anchor of an intent: beyond the swept extreme of the signal, or beyond the
# gap edge (prod's fallback when the swept extreme sits too close to the entry).
SLType: TypeAlias = Literal["sweep_extreme", "fvg_edge"]
# Where an intent's take-profit came from (C2): a liquidity level or the RR fallback.
TPSource: TypeAlias = Literal["liquidity", "rr"]
# Direction of an intent (SPEC_SMC.md §7.8 п.35).  The entry chain derives it from the
# level it trades: a level *above* the price is swept upwards and sold (``short``), a
# level below it is swept downwards and bought (``long``).
Side: TypeAlias = Literal["long", "short"]
# Which entry flavour of prod's ``ENTRY_TYPE`` v1 implements: the FVG limit only, or
# one of the sweep-midpoint branches (core.py lines 712-734).  The latter two are
# deferred by SPEC_SMC.md §7.8 п.33 and raise from the strategy instead of guessing.
EntryType: TypeAlias = Literal["fvg", "sweep50", "fvg_or_sweep50"]
# The FX pairs of the Alfa-Forex price list (C6).  BTCUSD / ETHUSD carry no spread or
# swap line there, and §5 п.9 (which symbols v1 trades) is still open, so v1 ships the
# two pairs whose C6 profile is actually known.
Symbol: TypeAlias = Literal["EURUSD", "GBPUSD"]


@dataclass(frozen=True, slots=True)
class TimeframeConfig:
    """Working timeframe hierarchy (M5 is deliberately absent).

    ``htf`` stays ``"D1"`` for the global bias context; the full bias hierarchy is
    ``BiasConfig.timeframes`` (H1 + H4 + D1, SPEC_SMC.md C5).
    """

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

    ``session_open_msk`` / ``session_close_msk`` are the broker's weekly trading
    hours (C6) and are *not* killzones: the killzone gate filters the sessions
    inside the day, the mask built by :func:`smc_zero.indicators.sessions.alfa_trading_mask`
    filters whole days (Monday opening, Friday closing, weekend off).
    """

    use_kz: bool = True
    prelondon: bool = False
    prelondon_msk: HourWindow = (7, 9)
    london_summer_msk: HourWindow = (9, 12)
    london_winter_msk: HourWindow = (10, 13)
    ny_summer_msk: HourWindow = (14, 17)
    ny_winter_msk: HourWindow = (15, 18)
    # Broker trading hours (C6): Alfa opens Monday 02:00 MSK and closes Friday
    # 23:55 MSK, the market is shut over the weekend.  These are MSK *times of day*,
    # not half-open windows, and they are consumed by ``sessions.alfa_trading_mask``.
    # The pair lives here while v1 has no broker profile object (``BrokerSpec``, C6,
    # Э5'), because the mask must reuse this module's single MSK definition.
    session_open_msk: TimeOfDay = (2, 0)
    session_close_msk: TimeOfDay = (23, 55)

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
        for name, time_of_day in (
            ("session_open_msk", self.session_open_msk),
            ("session_close_msk", self.session_close_msk),
        ):
            hour, minute = time_of_day
            if not 0 <= hour <= 23:
                raise ValueError(f"{name} hour must satisfy 0 <= hour <= 23, got {time_of_day}")
            if not 0 <= minute <= 59:
                raise ValueError(f"{name} minute must satisfy 0 <= minute <= 59, got {time_of_day}")


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
class BiasConfig:
    """Multi-timeframe bias hierarchy and its conflict policy (SPEC_SMC.md, C5).

    ``timeframes`` lists the frames whose *closed* trends decide the bias; the tuple is
    also the order in which the trend columns are attached (``trend_h1``, ``trend_h4``,
    ``trend_d1``) and must therefore be non-empty and duplicate free.

    ``agreement`` decides how many of them have to point the same way.  ``unanimous``
    (the v1 default, the owner ruling of SPEC_SMC.md §7.6 п.24) needs every frame, so a
    2-1 split is a conflict and never a direction; ``majority`` accepts a *strict*
    majority as well and reports it as ``majority_long`` / ``majority_short``, so the
    weaker verdict stays visible in the markup.  Both modes rank the verdicts the same
    way: an undefined trend (``NaN`` or ``0``) outranks everything, a tie of defined
    trends is ``conflict``.  The A/B factor of Э7' (SPEC_SMC.md §7.11 п.70) is the
    categorical parameter that switches between the two, and it is the only bias field an
    optimization trial may vary - the trends behind it are cached for the whole run.

    ``on_conflict`` decides what ``bias_dir`` does when the timeframes disagree:
    ``no_trade`` (default, the conservative SMC ruling) forces ``bias_dir = 0`` while
    ``bias_state`` still reports ``"conflict"``; ``reduced_risk`` awaits a decision
    about what "reduced" means (lot fraction or a wider SL) and raises
    ``NotImplementedError`` from the indicator instead of guessing.

    ``structure`` is the swing / BOS / CHoCH configuration used to derive the trend
    of every HTF frame, so the bias layer never falls back on a hidden default.
    """

    timeframes: tuple[Timeframe, ...] = ("H1", "H4", "D1")
    on_conflict: ConflictPolicy = "no_trade"
    agreement: Agreement = "unanimous"
    structure: StructureConfig = field(default_factory=StructureConfig)

    def __post_init__(self) -> None:
        if not self.timeframes:
            raise ValueError("timeframes must not be empty")
        if len(set(self.timeframes)) != len(self.timeframes):
            raise ValueError(f"timeframes must be unique, got {self.timeframes}")
        if self.agreement not in AGREEMENTS:
            raise ValueError(f"agreement must be one of {AGREEMENTS}, got {self.agreement!r}")


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
    detection parameter and lives in ``EntryConfig.edge`` (Э4').
    """

    min_gap_size: float = 0.0

    def __post_init__(self) -> None:
        if self.min_gap_size < 0:
            raise ValueError("min_gap_size must be >= 0")


@dataclass(frozen=True, slots=True)
class EntryConfig:
    """Where the pending limit order sits inside the entry gap (SPEC_SMC.md, п.8).

    ``edge`` keeps prod's ``fvg_entry_edge`` (C6): ``"proximal"`` is the gap edge
    price reaches first, ``"mid"`` the centre of the gap.  v1 rules for ``"mid"``
    (prod's own default ``DEFAULT_FVG_ENTRY_EDGE``); ``"distal"`` was never used by
    the prod run and is deferred by §7.8 п.33 - the value itself is validated by
    :func:`smc_zero.indicators.fvg.entry_level`, not here, so that the deferral
    message lives next to the code that would have to implement it.

    ``limit_stop_pip`` is prod's ``limit_stop_level_pip`` (0.7 pip for EURUSD, C6):
    the minimal distance a pending order - and hence the SL - must keep from the
    close of the signal bar.  Like ``LiquidityConfig.sweep_buffer`` and
    ``LevelConfig.break_buffer_pip`` it is a *pip* value converted at the call site
    with ``StrategyConfig.pip_size``, so the strategy layer never re-defines a pip.

    ``type`` is prod's ``ENTRY_TYPE`` switch.  ``"fvg"`` is the only flavour v1
    trades; ``"sweep50"`` / ``"fvg_or_sweep50"`` (core.py lines 712-734) are
    deferred by SPEC_SMC.md §7.8 п.33 and raise ``NotImplementedError`` from
    ``build_intents`` instead of silently falling back on the FVG entry.
    """

    type: EntryType = "fvg"
    edge: FVGEntryMode = "mid"
    limit_stop_pip: float = 0.7

    def __post_init__(self) -> None:
        if self.limit_stop_pip < 0:
            raise ValueError("limit_stop_pip must be >= 0")


@dataclass(frozen=True, slots=True)
class TPConfig:
    """Take-profit policy (SPEC_SMC.md, C2 / п.9).

    ``mode = "liquidity_with_rr_fallback"`` (the C2 default) aims at the nearest
    counter-side liquidity level that is *visible* at the entry bar and clears
    ``min_tp_rr``; when no such level exists the target falls back to prod's
    formula ``entry -/+ sl_size * rr_fallback`` (core.py line 783).  A level that is
    farther than ``min_tp_rr`` is taken as it is - spec п.9 wants the level, not a
    multiple - while a level that is closer is skipped in favour of the next one.
    ``min_tp_rr = 1.0`` is the ruling of §7.8 п.36; ``rr_fallback = 2.0`` is prod's
    ``params["rr"]`` default.

    ``"liquidity"`` and ``"rr"`` are recognised names whose edge cases (empty level
    book, which RR grid replaces static levels) are not decided yet: the take-profit
    builder raises ``NotImplementedError`` for them instead of guessing.
    """

    mode: TPMode = "liquidity_with_rr_fallback"
    min_tp_rr: float = 1.0
    rr_fallback: float = 2.0

    def __post_init__(self) -> None:
        if self.min_tp_rr <= 0:
            raise ValueError("min_tp_rr must be > 0")
        if self.rr_fallback <= 0:
            raise ValueError("rr_fallback must be > 0")


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

    ``equal_tol`` clusters equal highs/lows; the clustering itself is deferred with
    the breaker setup (SPEC_SMC.md §7.7), so ``equal_tol`` is inert in v1 - only the
    level maps below exist so far.  ``sweep_buffer`` is the price
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
class LevelConfig:
    """Liquidity level-map parameters (SPEC_SMC.md, п.3-п.4).

    ``break_buffer_pip`` is the distance a *close* has to clear a level by before the
    level counts as broken (``close > level + buffer`` / ``close < level - buffer``).
    It keeps the prod name and default of ``BOS_MIN_BREAK_PIP`` for traceability and,
    exactly like ``LiquidityConfig.sweep_buffer``, is compared in *price units*: prod
    multiplies the pip value by ``pip_size`` at its call site, so that conversion
    belongs to the broker layer (``BrokerSpec``, C6) and the indicator never needs to
    know what a pip is.  A wick through the level is *not* a break - п.5 keeps the
    wick rule for sweeps, while п.4 wants the decisive close.

    ``asian_window_utc`` is the fixed UTC window of the Asian range (prod's
    ``hour < 8``).  Asia does not switch clocks, so the season table of
    :mod:`smc_zero.indicators.sessions` is deliberately *not* applied to it, while the
    London and New York maps take their MSK windows from that very table and own no
    window of their own (C1).

    ``week_convention`` is prod's week key (``"%Y-%W"``) used to group bars into
    weeks.  It has to stay year-qualified (``"%Y"``): the weekly grouping sorts the
    labels and relies on ``"%Y-%W"`` being zero-padded, so a year-less format would
    merge the same week number of different years into one group.
    """

    break_buffer_pip: float = 2.0
    asian_window_utc: HourWindow = (0, 8)
    week_convention: str = "%Y-%W"

    def __post_init__(self) -> None:
        if self.break_buffer_pip < 0:
            raise ValueError("break_buffer_pip must be >= 0")
        start, end = self.asian_window_utc
        if not 0 <= start < end <= 24:
            raise ValueError(
                f"asian_window_utc must satisfy 0 <= start < end <= 24, got {self.asian_window_utc}"
            )
        if "%Y" not in self.week_convention:
            raise ValueError(
                "week_convention must be year-qualified (contain '%Y'), "
                f"got {self.week_convention!r}"
            )


@dataclass(frozen=True, slots=True)
class RiskConfig:
    """Position sizing, trading costs and the C7 risk profile.

    ``risk_pct`` is the per-trade risk budget in percent of equity; ``rr`` keeps
    prod's ``params["rr"]`` (the 1.5 / 2.0 / 2.5 grid built from ``MIN_RR`` /
    ``MAX_RR`` / ``RR_STEP``), which v1 uses only as the TP fallback
    (``TPConfig.rr_fallback``) - the strict RR mode is deferred.

    ``commission`` is charged per trade in account currency, ``spread`` and
    ``slippage`` are expressed in price units.  :attr:`has_costs` is the guard
    used by the reporting layer: without costs a positive equity curve must not
    be presented as a profit.

    ``lot``, ``leverage``, ``contract_size``, ``pip_size`` and ``deposit`` are C7's
    fixed risk profile, and it is the *one* copy of every broker number: the risk
    gate of :mod:`smc_zero.strategy.risk_gate` prices an order from this object
    alone (``apply_risk_gate(intents, cfg, equity)``).  The lot is *given*, not
    derived from ``risk_pct`` (``DEFAULT_LOT = 0.1`` with ``contract_size =
    100 000`` is $1/pip on EURUSD, so a 20-60 pip SL risks 2-6 % of a $1000
    deposit - C7's own arithmetic), and an order has to pass a margin check before
    it is placed.  ``warning_risk_pct`` is C7's reporting threshold: a single trade
    above it is still allowed but flagged, and the backtester must print the
    median risk per trade.

    ``deposit`` is the C7 answer of SPEC_SMC.md §7.8 п.37 (ruling R2): the
    *denominator* of the reported ``risk_pct`` column, 1000 as the owner ruled.
    It is deliberately not ``BacktestConfig.initial_capital`` - that field is the
    backtester's own capital (prod's 10000, SPEC_SMC.md §5 п.10) and the margin
    gate measures against the *current* equity it is handed, so the two numbers
    answer two different questions until §5 п.10 is answered.  ``pip_size`` and
    ``contract_size`` are the FX pair of the price list (0.0001 pip, 100 000 a
    lot); ``BrokerSpec`` (C6, Э5') has to *replace* this trio, not duplicate it.
    """

    risk_pct: float = 1.0
    rr: float = 2.0
    commission: float = 0.0
    spread: float = 0.0
    slippage: float = 0.0
    lot: float = 0.1
    leverage: float = 40.0
    contract_size: float = 100_000.0
    pip_size: float = 0.0001
    deposit: float = 1000.0
    warning_risk_pct: float = 2.0

    def __post_init__(self) -> None:
        if self.risk_pct <= 0:
            raise ValueError("risk_pct must be > 0")
        if self.rr <= 0:
            raise ValueError("rr must be > 0")
        if min(self.commission, self.spread, self.slippage) < 0:
            raise ValueError("commission, spread and slippage must be >= 0")
        if self.lot <= 0:
            raise ValueError("lot must be > 0")
        if self.leverage < 1:
            raise ValueError("leverage must be >= 1")
        if self.contract_size <= 0:
            raise ValueError("contract_size must be > 0")
        if self.pip_size <= 0:
            raise ValueError("pip_size must be > 0")
        if self.deposit <= 0:
            raise ValueError("deposit must be > 0")
        if not 0 < self.warning_risk_pct <= 100:
            raise ValueError("warning_risk_pct must satisfy 0 < warning_risk_pct <= 100")

    @property
    def has_costs(self) -> bool:
        """``True`` when commission, spread and slippage are all set."""
        return self.commission > 0 and self.spread > 0 and self.slippage > 0


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    """Entry-chain thresholds of the Э4' strategy layer (SPEC_SMC.md §7.8).

    Every window is counted in *bars of the entry timeframe* and every number has a
    prod counterpart, because C6/C7 forbid inventing broker or strategy numbers:

    * ``setup_type`` - ``"fresh"`` trades levels still unbroken at the entry bar;
      ``"breaker"`` (spec п.4) is deferred beyond v1 and raises
      ``NotImplementedError`` instead of silently trading fresh levels;
    * ``signal_max_age_bars`` - ``DEFAULT_SIGNAL_MAX_AGE_BARS = 60``: how old the
      sweep signal may be at the entry bar (prod tests ``i - source_idx > age``);
    * ``sweep_buffer_pip`` / ``min_fvg_pip`` - prod has no constant for either: the
      optimizer grids are 1-3 (step 1) and 1-5 (step 1) and the diagnostic
      fall-throughs are 1, which is the v1 value;
    * ``fvg_lookback`` - prod's ``fvg_lookback`` (grid 10-30 step 5, diagnostic
      default 20): how far after the CHoCH a gap is still accepted;
    * ``max_fvg_age_bars`` - ``DEFAULT_MAX_FVG_AGE_BARS = 12``: gap age at entry;
    * ``choch_wait_bars`` - ``CHOCH_WAIT_BARS = 20``: prod's CHoCH window, which
      starts at the sweep bar and runs forward;
    * ``fvg_select`` - ``"first"`` is prod's ``DEFAULT_FVG_SELECT``;
    * ``max_setups_per_level_per_day`` - v1 ruling (§7.8): one setup per level
      *instance* per MSK day, replacing prod's ``MAX_USES_PER_LEVEL = 2`` (which
      counted uses of one *price*), while prod's ``MAX_ORDERS_PER_ENTRY`` /
      ``LIMIT_VALID_BARS`` / ``MAX_SL_PER_DAY`` counters belong to the order
      lifecycle of the backtester (Э5');
    * ``sl_buffer_pip`` - prod's ``SL_BUFFER_PIP_RANGE = (5, 20)``, v1 takes the
      lower end (5) so the SL sits just beyond the swept extreme; ``min_sl_pip`` -
      ``MIN_SL_PIP_RANGE = (5, 12)``, v1 takes 5;
    * ``min_sl_realistic_pip`` / ``max_sl_realistic_pip`` -
      ``MIN_SL_REALISTIC_PIP = 20`` / ``MAX_SL_REALISTIC_PIP = 60``: the realistic
      SL band C7's arithmetic is built on (20-60 pips = 2-6 % of $1000 at 0.1 lot).

    ``pip_size`` and ``contract_size`` are *not* fields here: they are the C7
    numbers of :class:`RiskConfig`, and this class reads them through the
    read-only properties of the same name (one copy of every broker number, so
    that the risk gate can price an order from the risk profile alone).
    ``pip_size`` converts the pip-denominated prod parameters at the *strategy*
    boundary - indicators never see a pip - and ``contract_size`` feeds the
    margin/risk arithmetic of :mod:`smc_zero.strategy.risk_gate`.

    ``displacement`` is the formal impulse gate of §7.5 that the chain applies to
    the CHoCH bar (§7.8 п.40), and ``use_displacement`` switches it off for
    experiments.  The defaults of :class:`DisplacementConfig` are inert by design,
    so a run that wants the gate to bite sets its thresholds explicitly.

    ``risk`` is the C7 profile used by the risk gate (lot, leverage, warning
    threshold); the cost fields of the same class are the backtester's business, so
    both may be wired from one instance.

    ``bias`` is the C5 hierarchy (H1 + H4 + D1) the entry chain asks for a
    direction, and its ``structure`` field is also the swing / CHoCH configuration
    of the M15 entry frame - prod had a single global structure parameter set, so
    the strategy does not keep a second copy of it.  ``max_spread_pct_of_sl`` is
    prod's ``DEFAULT_MAX_SPREAD_PCT_OF_SL``: the spread may not exceed this
    fraction of the stop distance (it is inert while ``RiskConfig.spread`` is zero,
    which is the Э3' default - costs arrive with Э5').
    """

    setup_type: SetupType = "fresh"
    use_displacement: bool = True
    displacement: DisplacementConfig = field(default_factory=DisplacementConfig)
    signal_max_age_bars: int = 60
    sweep_buffer_pip: float = 1.0
    min_fvg_pip: float = 1.0
    fvg_lookback: int = 20
    max_fvg_age_bars: int = 12
    choch_wait_bars: int = 20
    fvg_select: FVGSelect = "first"
    max_setups_per_level_per_day: int = 1
    sl_buffer_pip: float = 5.0
    min_sl_pip: float = 5.0
    min_sl_realistic_pip: float = 20.0
    max_sl_realistic_pip: float = 60.0
    entry: EntryConfig = field(default_factory=EntryConfig)
    take_profit: TPConfig = field(default_factory=TPConfig)
    bias: BiasConfig = field(default_factory=BiasConfig)
    fvg: FVGConfig = field(default_factory=FVGConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    liquidity: LiquidityConfig = field(default_factory=LiquidityConfig)
    levels: LevelConfig = field(default_factory=LevelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    max_spread_pct_of_sl: float = 0.12

    @property
    def pip_size(self) -> float:
        """The FX pip of the traded pair, read from the C7 profile (one copy)."""
        return self.risk.pip_size

    @property
    def contract_size(self) -> float:
        """The contract size of the traded pair, read from the C7 profile."""
        return self.risk.contract_size

    def __post_init__(self) -> None:
        if self.signal_max_age_bars < 0:
            raise ValueError("signal_max_age_bars must be >= 0")
        if self.fvg_lookback < 1:
            raise ValueError("fvg_lookback must be >= 1")
        if self.max_fvg_age_bars < 0:
            raise ValueError("max_fvg_age_bars must be >= 0")
        if self.choch_wait_bars < 1:
            raise ValueError("choch_wait_bars must be >= 1")
        if self.max_setups_per_level_per_day < 1:
            raise ValueError("max_setups_per_level_per_day must be >= 1")
        if min(self.sweep_buffer_pip, self.min_fvg_pip, self.sl_buffer_pip, self.min_sl_pip) < 0:
            raise ValueError(
                "sweep_buffer_pip, min_fvg_pip, sl_buffer_pip and min_sl_pip must be >= 0"
            )
        if self.min_sl_realistic_pip <= 0:
            raise ValueError("min_sl_realistic_pip must be > 0")
        if self.max_spread_pct_of_sl < 0:
            raise ValueError("max_spread_pct_of_sl must be >= 0")
        if self.max_sl_realistic_pip <= self.min_sl_realistic_pip:
            raise ValueError(
                "max_sl_realistic_pip must be > min_sl_realistic_pip, "
                f"got {self.max_sl_realistic_pip} <= {self.min_sl_realistic_pip}"
            )


@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    """The Alfa-Forex price list of one symbol (C6): the costs the engine charges.

    A row of prod's ``ALFAFOREX_SPECS`` as a dataclass (Э5', SPEC_SMC.md §7.9).
    ``spread_pip`` is the round-trip spread of the price list, ``limit_stop_level_pip``
    the broker's minimum distance between the market and a pending order,
    ``swap_long_pip`` / ``swap_short_pip`` the overnight rates in *pips per night*
    (negative = charged to the trader) and ``contract_size`` the units of one lot;
    ``pip_size`` is the pip of the symbol (prod's ``PIP_SIZE``: 0.0001 for FX).

    The helpers keep the FX arithmetic in one place: :attr:`spread_abs` and
    :meth:`swap_abs` are price units (exactly what the engine multiplies by the lot with
    :meth:`money`), and the lot itself comes from :class:`RiskConfig` - a price list does
    not know the position size.  ``pip_size`` / ``contract_size`` are deliberately *also*
    fields of that risk profile: C7 prices a stop in pips, C6 quotes a spread in pips, and
    Э5' hands the instrument to the engine *beside* the risk profile instead of letting the
    engine guess which copy it means (§7.9 keeps the duplication on the table until C6
    replaces the C7 trio).
    """

    symbol: Symbol = "EURUSD"
    pip_size: float = 0.0001
    contract_size: float = 100_000.0
    spread_pip: float = 1.4
    limit_stop_level_pip: float = 0.7
    swap_long_pip: float = -0.70
    swap_short_pip: float = 0.0

    def __post_init__(self) -> None:
        if self.pip_size <= 0:
            raise ValueError("pip_size must be > 0")
        if self.contract_size <= 0:
            raise ValueError("contract_size must be > 0")
        if self.spread_pip < 0:
            raise ValueError("spread_pip must be >= 0")
        if self.limit_stop_level_pip < 0:
            raise ValueError("limit_stop_level_pip must be >= 0")

    @property
    def spread_abs(self) -> float:
        """The round-trip spread in price units (prod: ``spread_pip * pip_size``)."""
        return self.spread_pip * self.pip_size

    def swap_pip(self, side: Side) -> float:
        """The overnight rate of ``side`` in pips per night (one leg of the pair)."""
        return self.swap_long_pip if side == "long" else self.swap_short_pip

    def swap_abs(self, side: Side, days_held: int) -> float:
        """The overnight result of ``side`` in price units for ``days_held`` nights."""
        return self.swap_pip(side) * days_held * self.pip_size

    def money(self, move_abs: float, lot: float) -> float:
        """Convert a price distance into account currency: ``move * contract * lot``."""
        return move_abs * self.contract_size * lot


#: The price-list rows v1 ships: the two FX pairs of prod's ``ALFAFOREX_SPECS``.
ALFAFOREX_SPECS: dict[str, InstrumentSpec] = {
    "EURUSD": InstrumentSpec(),
    "GBPUSD": InstrumentSpec(
        symbol="GBPUSD",
        spread_pip=2.1,
        limit_stop_level_pip=1.1,
        swap_long_pip=-0.55,
        swap_short_pip=-0.25,
    ),
}


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Backtest wiring: capital, the order lifecycle and the unclosed-tail protection.

    ``initial_capital`` is the *backtester's* capital (prod's 10000, SPEC_SMC.md §5 п.10)
    and the denominator of ``return_pct``; ``RiskConfig.deposit`` (1000) is the C7
    denominator of the reported ``risk_pct``.  The two are different questions - the
    deposit prices a stop, the capital is what the equity curve starts from - and mixing
    them is the C7 trap the metric layer guards against (§7.9).

    The order lifecycle is prod's, one field per counter:

    * ``limit_valid_bars`` - ``LIMIT_VALID_BARS = 10``: how many M15 bars a pending limit
      stays alive after the signal bar (fills are searched from the *next* bar, never from
      the signal bar itself);
    * ``max_orders_day`` - prod's ``params["max_orders_day"]`` (diagnostic default 5): how
      many orders the day may *place* (``0`` refuses every order) - prod's counter is spent by
      a placement, not by a fill, and §7.9 records it;
    * ``max_sl_per_day`` - ``MAX_SL_PER_DAY = 2``: after this many stops no new order is
      placed for the rest of the day (prod's counter - its blocking branch is dead code,
      see §7.9);
    * ``force_close_eod`` - ``DEFAULT_FORCE_CLOSE_EOD = False``: close an open trade at the
      last close of its MSK day instead of carrying it over the date change;
    * ``max_bars_per_trade`` - prod's literal ``500`` inside ``simulate_trade_idx``: the
      window in which SL/TP may resolve, after which the trade is dropped as
      ``no_result``.

    ``pf_cap`` (prod's ``PF_CAP = 5.0``) and ``sharpe_bars_per_day`` (prod scales the
    bar-to-bar ratio by ``sqrt(96)`` - the M15 bars of a day) are the two metric constants
    and live here so no reporting number is a literal.  ``risk`` / ``timeframes`` /
    ``session`` are the profiles the engine reads, and ``drop_unclosed`` is the
    constitution's protection: the still-forming tail of the tape is dropped before
    anything is simulated (rule 2b).
    """

    initial_capital: float = 10_000.0
    limit_valid_bars: int = 10
    max_orders_day: int = 5
    max_sl_per_day: int = 2
    force_close_eod: bool = False
    max_bars_per_trade: int = 500
    pf_cap: float = 5.0
    sharpe_bars_per_day: int = 96
    drop_unclosed: bool = True
    risk: RiskConfig = field(default_factory=RiskConfig)
    timeframes: TimeframeConfig = field(default_factory=TimeframeConfig)
    session: SessionConfig = field(default_factory=SessionConfig)

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be > 0")
        if self.limit_valid_bars < 0:
            raise ValueError("limit_valid_bars must be >= 0")
        if self.max_orders_day < 0:
            raise ValueError("max_orders_day must be >= 0")
        if self.max_sl_per_day < 0:
            raise ValueError("max_sl_per_day must be >= 0")
        if self.max_bars_per_trade < 1:
            raise ValueError("max_bars_per_trade must be >= 1")
        if self.pf_cap <= 0:
            raise ValueError("pf_cap must be > 0")
        if self.sharpe_bars_per_day < 1:
            raise ValueError("sharpe_bars_per_day must be >= 1")


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    """Walk-forward splitting of a tape: the fit window and the out-of-sample window (Э6').

    A fold is a pair ``(train, test)`` of *adjacent* slices of one tape: ``train`` is the
    information the optimizer of Э7' may look at, ``test`` the bars that are simulated with
    the parameters it chose - the walk-forward of SPEC_SMC.md §7.10 never lets a fold see its
    own future.  ``anchored`` picks between the two classic schemes:

    * ``anchored=True`` (the default, an expanding window): ``train`` always starts at the
      tape's first bar and grows by ``test_period_bars`` every fold, so the first fold's train
      window is exactly ``min_train_bars`` bars long and the k-th one is
      ``min_train_bars + k * test_period_bars``;
    * ``anchored=False`` (a rolling window): ``train`` is a fixed ``train_period_bars`` window
      that slides with the test window - it starts at ``test_start - train_period_bars``,
      clipped at the tape's first bar, so the early folds of a tape carry a shorter warm-up
      than ``train_period_bars``.

    ``test_start`` is always the end of the train window and the test window is always
    ``test_period_bars`` long, so the two windows never share a bar and the fold count of a
    tape of ``n`` bars is ``max(0, (n - min_train_bars) // test_period_bars)`` - the same
    number for both schemes, only the train window differs between them.

    The defaults are the M15 units of the project (96 bars a day): a 120 day warm-up, a
    **60 day** out-of-sample window and a 120 day rolling window.  The 60 day window is the
    unit an optimization fold is read through (SPEC_SMC.md §7.11 п.69), so a study sees 15
    folds on the 100 111 M15 bars of ``./data/EURUSD_M15.csv`` (4.1 years, 2022-08-15 ..
    2026-09-22); SPEC_SMC.md §7.10 п.62 records the arithmetic of both.  The fold count is
    a decision about ``min_train_bars`` - the window lengths below are what a run is about.
    """

    anchored: bool = True
    test_period_bars: int = 96 * 60
    min_train_bars: int = 96 * 120
    train_period_bars: int = 96 * 120

    def __post_init__(self) -> None:
        if self.test_period_bars < 1:
            raise ValueError("test_period_bars must be >= 1")
        if self.min_train_bars < 1:
            raise ValueError("min_train_bars must be >= 1")
        if self.train_period_bars < 1:
            raise ValueError("train_period_bars must be >= 1")


@dataclass(frozen=True, slots=True)
class OptunaConfig:
    """Search budget and score of the Э7' optimizer (SPEC_SMC.md §7.11).

    ``n_trials`` is how many parameter sets the study evaluates; ``n_jobs`` how many of
    them run at once (optuna threads - the default ``1`` is the honest setting for the
    TPE sampler, which is otherwise asked to guess at trials it cannot see yet); ``seed``
    seeds that sampler, so a run is reproducible.

    ``score_metric`` names the headline metric of the *out-of-sample* half of the folds
    that the study maximises - ``sharpe`` (the default) or ``profit`` - and
    ``penalty_power`` weighs the last factor of the score: the train -> test degradation
    (:func:`smc_zero.optimizer.score.score_from_aggregates`).  ``0.0`` (the default)
    switches that gate off, so the score is the pure out-of-sample reading of the test
    window; ``1.0`` is the plain product of SPEC_SMC.md §7.11 п.69, and a larger power
    punishes a trial that only looks good in sample harder, i.e. pulls the study towards
    parameters whose in-sample edge survives out of sample.  A test window that made no
    money (``profit_mean(test) <= 0``) scores a flat ``0.0`` whatever the rest of its
    numbers say, so no weight here can let a losing parameter set outrank a profitable one.

    The defaults are sized for a first real run over four years of M15 with the 60 day
    out-of-sample window of :class:`WalkForwardConfig` - 15 folds of
    ``./data/EURUSD_M15.csv``; what such a study costs is measured on the run itself and
    never estimated here, because the test suite runs on synthetic tapes and never spends
    minutes on a full study.
    """

    n_trials: int = 100
    n_jobs: int = 1
    seed: int = 42
    score_metric: ScoreMetric = "sharpe"
    penalty_power: float = 0.0

    def __post_init__(self) -> None:
        if self.n_trials < 1:
            raise ValueError("n_trials must be >= 1")
        if self.n_jobs < 1:
            raise ValueError("n_jobs must be >= 1")
        if self.score_metric not in SCORE_METRICS:
            raise ValueError(
                f"score_metric must be one of {SCORE_METRICS}, got {self.score_metric!r}"
            )
        if self.penalty_power < 0:
            raise ValueError("penalty_power must be >= 0 (0 switches the decay gate off)")
