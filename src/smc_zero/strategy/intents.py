"""The M15 entry chain: liquidity sweep -> CHoCH -> gap limit (SPEC_SMC.md §7.8 п.35).

:func:`build_intents` walks the level instances of a lifecycled book and every closed M15 bar
and turns each attempt into either a :class:`~smc_zero.strategy.base.TradeIntent` or one row of
a rejection ledger.  Gate order and reason names are the spec's; the ledger is the test surface,
so no attempt may vanish silently.

Two readings of п.35 shape the code.  First, *an attempt exists where and only where a sweep was
found*: a bar that merely pierced a level leaves no record, which is what makes the rolling
max/min prefilter legal - п.35 calls it an optimization and keeps
:func:`smc_zero.indicators.liquidity.sweep_index` the single implementation of the rule.
Second, the **bar gates** filter the *bar* (``alfa_trading_mask`` and ``in_killzone``: outside
them a signal does not exist, it is not "rejected"), while the **attempt gates** filter the
*attempt* and always leave exactly one ledger row.

===  =============================  ===========================================================
(1)  ``level_not_available``        ``t < available_at`` - the price is not a fact yet
(2)  ``level_broken``               the instance does not own its price at ``t``: broken,
                                    replaced by its successor (``retired_at``) or outranked by a
                                    same-price / same-side instance (``deduplicate_levels``)
(3)  ``level_used_today``           the instance already produced its daily cap of setups
(4)  ``bias_skip``                  the side is against the closed HTF ``bias_dir``
(5)  ``stale_signal``               ``i - sweep_bar > signal_max_age_bars``
(6)  ``choch_not_found``            no counter-trend break in ``[sweep_bar, sweep_bar + wait]``
(7)  ``displacement_skip``          the CHoCH break is not a formal impulse, or its outcome is
                                    not knowable at the decision bar (§7.5, §7.8 п.40)
(8)  ``fvg_not_ready``              ``i - choch_bar < 2`` (prod's lag)
(9)  ``fvg_not_found``              no gap inside ``(choch_bar, choch_bar + fvg_lookback]``
                                    younger than the decision bar
(10) ``fvg_too_old``                ``i - fvg_bar > max_fvg_age_bars``
(11) ``wrong_side_limit``           the limit sits closer to the close than ``limit_stop_pip``,
                                    or on the wrong side of it
(12) ``sl_rejected_all`` /          the 20-60 pip band decided the anchor: the swept extreme
     ``sl_rejected_wide``           (after prod's fallback to the gap edge) does not fit, or is
                                    wider than ``max_sl_realistic_pip``
(13) ``min_sl_skip``                stop narrower than ``min_sl_pip``
(14) ``spread_pct_skip``            the spread exceeds ``max_spread_pct_of_sl`` of the stop
===  =============================  ===========================================================

Gate (15) ``limit_stop_violation`` of п.35 is prod's second distance test (core.py lines 802-804).
After (11) the distance to the close is already at least ``limit_stop_pip`` on the correct side,
so it can never fire - п.35 says so itself and the branch is not ported.

Gate (7) is the gate prod did not have: :func:`smc_zero.indicators.impulse.displacement_gate`
measures the break on the CHoCH bar and the chain reads it only from ``disp_known_at`` on
(§7.8 п.40); ``StrategyConfig.use_displacement = False`` switches it off for experiments.

Three consequences of the defaults are documented in the Э4' report because they look like bugs
from the outside:

* ``stale_signal`` is unreachable while ``signal_max_age_bars >= sweep_lookback`` (60 vs 48 by
  default) - the sweep window caps the age by itself.  Prod needed the gate for its
  ``pending_sweeps`` state machine, which v1 does not have; it is kept as a guard for a
  non-default config (``signal_max_age_bars < sweep_lookback``), which is how it is tested.
* prod's SL fallback to the gap edge is *reachable but narrow*, and the port keeps it exactly as
  prod wrote it (core.py lines 765-781).  The fallback is entered whenever the swept extreme sits
  so close to the entry that ``sl_buffer_pip`` cannot lift the stop into the band; it then helps
  only if the gap's far edge is *beyond* that extreme (``top > extreme`` for a short, mirrored for
  a long), i.e. when price ran away from the sweep before the gap formed.  In the common geometry
  the sweep extreme *is* the window's highest high, so it sits at or above the gap top and the
  fallback arm collapses into ``sl_rejected_all``; the tests build the running-away scenario
  explicitly instead of asserting that the branch is dead.
* with a ``mid`` entry the stop is measured to the far gap edge, so ``sl_size`` grows with the
  gap: a narrow gap can never fail the band from the "too wide" side and a wide one never from
  the "too narrow" side.

The pip-denominated thresholds live on :class:`~smc_zero.config.StrategyConfig` /
``EntryConfig`` (прod's parameter names) while the indicators keep their price-unit configs:
the chain is the one place that knows ``pip_size``, so it converts once,
:func:`dataclasses.replace`-ing ``LiquidityConfig.sweep_buffer`` and ``FVGConfig.min_gap_size``
instead of letting a second threshold table appear.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace
from typing import NamedTuple

import numpy as np
import pandas as pd

from smc_zero.config import Side, SLType, StrategyConfig
from smc_zero.data_loader import TIMESTAMP_COLUMN, drop_unclosed
from smc_zero.indicators.bias import BIAS_DIR_COLUMN
from smc_zero.indicators.fvg import entry_level, fair_value_gaps
from smc_zero.indicators.impulse import displacement_gate
from smc_zero.indicators.levels import (
    ASIAN_HIGH,
    ASIAN_LOW,
    BROKEN_AT_COLUMN,
    LEVEL_AVAILABLE_AT_COLUMN,
    LEVEL_COLUMNS,
    LEVEL_DATE_COLUMN,
    LEVEL_IS_UPPER_COLUMN,
    LEVEL_NAME_COLUMN,
    LEVEL_PRICE_COLUMN,
    LEVEL_RETIRED_AT_COLUMN,
    LONDON_HIGH,
    LONDON_LOW,
    NY_HIGH,
    NY_LOW,
    PDH,
    PDL,
    PMH,
    PML,
    PWH,
    PWL,
    fresh_at,
    retired_at,
)
from smc_zero.indicators.liquidity import sweep_index
from smc_zero.indicators.sessions import alfa_trading_mask, killzone_mask
from smc_zero.indicators.structure import structure_breaks
from smc_zero.strategy.base import TradeIntent, utc_stamps
from smc_zero.strategy.take_profit import take_profit_for

REASON_LEVEL_NOT_AVAILABLE = "level_not_available"
REASON_LEVEL_BROKEN = "level_broken"
REASON_LEVEL_USED_TODAY = "level_used_today"
REASON_BIAS_SKIP = "bias_skip"
REASON_STALE_SIGNAL = "stale_signal"
REASON_CHOCH_NOT_FOUND = "choch_not_found"
REASON_DISPLACEMENT_SKIP = "displacement_skip"
REASON_FVG_NOT_READY = "fvg_not_ready"
REASON_FVG_NOT_FOUND = "fvg_not_found"
REASON_FVG_TOO_OLD = "fvg_too_old"
REASON_WRONG_SIDE_LIMIT = "wrong_side_limit"
REASON_SL_REJECTED_ALL = "sl_rejected_all"
REASON_SL_REJECTED_WIDE = "sl_rejected_wide"
REASON_MIN_SL_SKIP = "min_sl_skip"
REASON_SPREAD_PCT_SKIP = "spread_pct_skip"

#: Every reason the ledger may carry, in gate order (п.35).
REJECTION_REASONS: tuple[str, ...] = (
    REASON_LEVEL_NOT_AVAILABLE,
    REASON_LEVEL_BROKEN,
    REASON_LEVEL_USED_TODAY,
    REASON_BIAS_SKIP,
    REASON_STALE_SIGNAL,
    REASON_CHOCH_NOT_FOUND,
    REASON_DISPLACEMENT_SKIP,
    REASON_FVG_NOT_READY,
    REASON_FVG_NOT_FOUND,
    REASON_FVG_TOO_OLD,
    REASON_WRONG_SIDE_LIMIT,
    REASON_SL_REJECTED_ALL,
    REASON_SL_REJECTED_WIDE,
    REASON_MIN_SL_SKIP,
    REASON_SPREAD_PCT_SKIP,
)
#: Ledger columns: the decision bar, the level instance and the verdict (п.35).
REJECTION_COLUMNS: tuple[str, ...] = (
    "bar",
    "open_time",
    "name",
    "date",
    "price",
    "side",
    "reason",
)

# prod's ``LEVEL_PRIORITY`` / ``get_level_priority`` (core.py lines 122-138): the smaller the
# number, the more significant the level.  The ``BOS_*`` entries (10) are out of v1
# (SPEC_SMC.md §7.8 п.33), so only the table and prod's 99 fallback for an unknown name travel.
LEVEL_PRIORITY: Mapping[str, int] = {
    PDH: 1,
    PDL: 1,
    PWH: 2,
    PWL: 2,
    PMH: 3,
    PML: 3,
    ASIAN_HIGH: 4,
    ASIAN_LOW: 4,
    LONDON_HIGH: 5,
    LONDON_LOW: 5,
    NY_HIGH: 6,
    NY_LOW: 6,
}
UNKNOWN_LEVEL_PRIORITY = 99

# prod's ``i - choch_idx < 2`` (core.py line 680): a gap needs the bar after its middle candle
# before it exists (``fair_value_gaps`` marks it from ``k + 1``), so two bars after the CHoCH are
# the earliest a lookup can succeed - prod rejects earlier attempts instead of looking.
FVG_READY_BARS = 2


class EntryChain(NamedTuple):
    """Verdict of the chain: the accepted intents and the rejection ledger.

    ``intents`` are in the order they were accepted (level instance by level instance, bar by
    bar), and ``rejections`` has one row per filtered attempt with the columns of
    :data:`REJECTION_COLUMNS`.  Attempts dropped by the two *bar* gates leave no row: outside
    trading hours or outside a killzone a signal does not exist (п.35).
    """

    intents: tuple[TradeIntent, ...]
    rejections: pd.DataFrame


def _utc_stamps(frame: pd.DataFrame, *, source: str) -> pd.Series:
    """Return ``frame['timestamp']`` as UTC-aware ``datetime64[ns, UTC]``.

    The loader already stores open_time stamps in that dtype; a frame without the column is a
    hard error, exactly like the private helper of :mod:`smc_zero.indicators.levels` (the chain
    may not guess the unit of a broken pipeline).  The dtype rule itself lives in
    :func:`smc_zero.strategy.base.utc_stamps`, which the layer's wrappers share.
    """
    if TIMESTAMP_COLUMN not in frame.columns:
        raise ValueError(f"{source} needs a {TIMESTAMP_COLUMN!r} column (loader schema)")
    return utc_stamps(frame[TIMESTAMP_COLUMN], source=source)


def _entry_bars(ltf: pd.DataFrame) -> pd.DataFrame:
    """Return the closed entry bars with normalized stamps and a positional index.

    Rule 2b: the last bar of the frame is presumed still forming and is dropped, so the chain
    can never trade the live tail.  ``is_closed`` is the loader's flag; a frame without it is
    used as it is (the caller then owns the decision, e.g. a synthetic test frame).
    """
    required = (TIMESTAMP_COLUMN, "high", "low", "close")
    missing = [column for column in required if column not in ltf.columns]
    if missing:
        raise ValueError(f"the entry frame needs the {missing} column(s) (loader schema)")
    bars = drop_unclosed(ltf)
    if bars.empty:
        return bars
    bars = bars.copy()
    bars[TIMESTAMP_COLUMN] = _utc_stamps(bars, source="the entry frame")
    return bars.reset_index(drop=True)


def _bias_direction(bars: pd.DataFrame, bias: pd.DataFrame) -> np.ndarray:
    """Return the closed-HTF bias direction of every entry bar as ``int8``.

    The direction is read from the ``bias_frames`` markup, i.e. from the HTF bar that had *closed*
    before the entry bar - the alignment of Э3' accounts for that and the chain only joins on
    ``timestamp``.  A missing row is a wiring error, not a "no bias" verdict: the point of the
    gate is that the direction is known, so a gap raises instead of silently trading.
    """
    required = (TIMESTAMP_COLUMN, BIAS_DIR_COLUMN)
    missing = [column for column in required if column not in bias.columns]
    if missing:
        raise ValueError(
            f"the bias frame needs the {missing} column(s); build it with "
            "bias_frames(ltf, htf_frames)"
        )
    right = bias.loc[:, list(required)].copy()
    right[TIMESTAMP_COLUMN] = _utc_stamps(bias, source="the bias frame")
    merged = bars.loc[:, [TIMESTAMP_COLUMN]].merge(
        right, on=TIMESTAMP_COLUMN, how="left", validate="many_to_one"
    )
    direction = merged[BIAS_DIR_COLUMN].to_numpy(dtype="float64")
    uncovered = int(np.isnan(direction).sum())
    if uncovered:
        raise ValueError(f"the bias frame does not cover {uncovered} entry bar(s)")
    return direction.astype("int8")


def _level_book(levels: pd.DataFrame) -> pd.DataFrame:
    """Return the level book with ``retired_at`` attached and the lifecycle columns checked.

    ``retired_at`` is derived here instead of inside Э3' because replacement by a successor is
    only a fact *for a consumer walking the whole book* (п.35): the chain is that consumer.
    """
    required = (*LEVEL_COLUMNS, BROKEN_AT_COLUMN)
    missing = [column for column in required if column not in levels.columns]
    if missing:
        raise ValueError(
            f"the level book needs the {missing} column(s); build it with "
            "level_lifecycle(static_levels(ltf, cfg.levels, "
            "session_cfg=cfg.session), ltf, cfg.levels)"
        )
    book = levels.reset_index(drop=True)
    book[LEVEL_RETIRED_AT_COLUMN] = retired_at(book)
    return book


def _price_groups(book: pd.DataFrame, pip_size: float) -> dict[tuple[int, bool], list[int]]:
    """Group book positions by ``deduplicate_levels``' price key ``(round(price / pip), upper)``.

    Ported from prod's ``deduplicate_levels`` (core.py lines 616-635), minus the clustering
    tolerance of the deferred breaker setup (``equal_tol``, §7.7): two instances share a price
    when they round to the same whole pip.
    """
    groups: dict[tuple[int, bool], list[int]] = defaultdict(list)
    prices = book[LEVEL_PRICE_COLUMN].to_numpy(dtype="float64")
    uppers = book[LEVEL_IS_UPPER_COLUMN].to_numpy(dtype=bool)
    for position in range(len(book)):
        key = (round(float(prices[position]) / pip_size), bool(uppers[position]))
        groups[key].append(position)
    return groups


def _owns_price(
    rows: list[dict[str, object]],
    position: int,
    t: pd.Timestamp,
    groups: dict[tuple[int, bool], list[int]],
    pip_size: float,
) -> bool:
    """Return whether the instance at ``position`` owns its price at ``t`` (gate 2).

    The instance has to be fresh *and* outrank every other fresh instance of the same price and
    side - prod's priority table applied to its active list, with a stable sort, so the earlier
    book row wins a priority tie.  Without this one sweep would produce two orders at one price,
    which is what ``deduplicate_levels`` exists to prevent.
    """
    row = rows[position]
    if not fresh_at(row, t):
        return False
    key = (round(float(row[LEVEL_PRICE_COLUMN]) / pip_size), bool(row[LEVEL_IS_UPPER_COLUMN]))
    own = _priority_key(rows, position)
    for member in groups[key]:
        if member == position:
            continue
        if not fresh_at(rows[member], t):
            continue
        if _priority_key(rows, member) < own:
            return False
    return True


def _priority_key(rows: list[dict[str, object]], position: int) -> tuple[int, int]:
    """Rank an instance against the rest of its price: priority first, then the book order.

    ``LEVEL_PRIORITY`` alone cannot break a tie, and a tie must still leave exactly one owner of
    the price (prod sorted its active list once and kept the head), so the position in the book
    - the instance the Э3' build emitted first - is the second key.
    """
    row = rows[position]
    return LEVEL_PRIORITY.get(str(row[LEVEL_NAME_COLUMN]), UNKNOWN_LEVEL_PRIORITY), position


def _choch_bar(break_dir: np.ndarray, sweep_bar: int, i: int, wait: int, upper: bool) -> int | None:
    """Return the first counter-trend break in ``[sweep_bar, sweep_bar + wait]`` up to ``i``.

    A swept *upper* level needs a downward break (``-1``) and vice versa, prod allows the CHoCH
    on the sweep bar and on the decision bar itself, hence the inclusive window
    (``find_choch_in_window_pre``, core.py lines 660-680).  ``i`` caps the right end because the
    break of the decision bar is only known once that bar has closed.
    """
    want = -1 if upper else 1
    stop = min(i, sweep_bar + wait)
    window = break_dir[sweep_bar : stop + 1]
    hits = np.flatnonzero(window == want)
    if hits.size == 0:
        return None
    return sweep_bar + int(hits[0])


def _displacement_ok(
    ok: np.ndarray, known_at: np.ndarray, choch_bar: int, i: int
) -> bool:
    """Return whether the CHoCH break is a formal impulse *knowable* at the decision bar.

    ``displacement_gate`` marks the break with ``disp_known_at = choch_bar + no_return_bars``
    (§7.5 п.20): while the no-return window still reaches past the decision bar the outcome is
    not a fact yet, so the gate answers ``False`` and the attempt leaves ``displacement_skip``.
    Reading ``disp_ok`` without that test would let a bar decide on a return it cannot see yet -
    the mutation m2 of §7.8 п.40 - which is why the leak test walks this window.
    """
    known = known_at[choch_bar]
    return bool(ok[choch_bar] and np.isfinite(known) and known <= i)


def _gap_bar(gaps: pd.DataFrame, choch_bar: int, i: int, lookback: int, upper: bool) -> int | None:
    """Return the first gap inside ``(choch_bar, choch_bar + lookback]``, i.e. the order host.

    The window is capped at ``i - 1``: a gap is confirmed by the bar after its middle candle
    (``fair_value_gaps``), so the newest usable gap is the one marked on ``i - 1``.  An upper
    level is sold into a *bearish* gap and a lower level bought in a bullish one; ``first`` is
    prod's ``DEFAULT_FVG_SELECT`` and the only value v1 implements (§7.8 п.33).
    """
    flags = gaps["bearish" if upper else "bullish"].to_numpy(dtype=bool)
    start = choch_bar + 1
    stop = min(choch_bar + lookback, i - 1)
    if stop < start:
        return None
    hits = np.flatnonzero(flags[start : stop + 1])
    if hits.size == 0:
        return None
    return start + int(hits[0])


def _sl_geometry(
    entry: float,
    extreme: float,
    top: float,
    bottom: float,
    *,
    upper: bool,
    sl_buffer: float,
    min_sl_realistic: float,
    max_sl_realistic: float,
) -> tuple[float, float, SLType] | str:
    """Return ``(sl, sl_size, sl_source)`` or the rejection reason of gate (11).

    Ported from prod's ``try_open_trade`` (core.py lines 748-781): the stop sits ``sl_buffer_pip``
    beyond the swept extreme, and when that distance is narrower than ``min_sl_realistic_pip``
    prod retries with the far edge of the entry gap.  The 20-60 pip band is applied *after* that
    fallback, so an attempt outside it leaves ``sl_rejected_all`` (neither anchor fits) or
    ``sl_rejected_wide`` (the swept extreme alone is too wide).

    The fallback only helps when the gap's far edge lies *beyond* the swept extreme
    (``edge_size > source_size``), which is exactly the geometry the module docstring describes;
    otherwise a too-narrow source is rejected.  Both outcomes are exercised by the tests.
    """
    source = extreme + sl_buffer if upper else extreme - sl_buffer
    edge = top + sl_buffer if upper else bottom - sl_buffer
    source_size = source - entry if upper else entry - source
    edge_size = edge - entry if upper else entry - edge
    if source_size < min_sl_realistic:
        if min_sl_realistic <= edge_size <= max_sl_realistic:
            return edge, edge_size, "fvg_edge"
        return REASON_SL_REJECTED_ALL
    if source_size > max_sl_realistic:
        return REASON_SL_REJECTED_WIDE
    return source, source_size, "sweep_extreme"


def _check_deferred(cfg: StrategyConfig) -> None:
    """Raise for every config value v1 declares but does not implement (§7.8 п.33).

    The literals exist so a config can *name* what prod knew; v1 implements exactly one value of
    each and refuses the rest instead of silently substituting its own choice.  This runs before
    the walk, so a misconfigured run fails even when its ledger would happen to be empty.
    """
    deferred = (
        ("setup_type", cfg.setup_type, "fresh", "breaker / retest setups are out of v1"),
        ("entry.type", cfg.entry.type, "fvg", "the sweep50 entries are out of v1"),
        ("fvg_select", cfg.fvg_select, "first", "prod's biggest-gap branch is out of v1"),
        (
            "take_profit.mode",
            cfg.take_profit.mode,
            "liquidity_with_rr_fallback",
            'the strict "liquidity" / "rr" modes are out of v1',
        ),
    )
    for field, value, implemented, why in deferred:
        if value != implemented:
            raise NotImplementedError(
                f"{field}={value!r} is declared by SPEC_SMC.md §7.8 п.33 but not implemented by "
                f"v1 ({why}; v1 implements {implemented!r})"
            )


def _rejection(
    bar: int,
    open_time: pd.Timestamp,
    name: str,
    level_date: pd.Timestamp,
    price: float,
    side: Side,
    reason: str,
) -> dict[str, object]:
    """One ledger row: the decision bar, the level instance and the reason (§7.8 п.38)."""
    return {
        "bar": int(bar),
        "open_time": open_time,
        "name": name,
        "date": pd.Timestamp(level_date),
        "price": float(price),
        "side": side,
        "reason": reason,
    }


def _ledger(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Return the rejection ledger, sorted by ``(bar, name, date)`` and typed.

    The order is part of the contract: instance-major acceptance order is not, but a reader (and a
    test) has to get the same frame twice, so ties break on the level instance.
    """
    frame = pd.DataFrame(rows, columns=list(REJECTION_COLUMNS))
    frame["bar"] = frame["bar"].astype("int64")
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True).astype("datetime64[ns, UTC]")
    frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
    frame["price"] = frame["price"].astype("float64")
    if len(frame):
        frame = frame.sort_values(["bar", "name", "date"], kind="stable").reset_index(drop=True)
    return frame


def build_intents(
    ltf: pd.DataFrame,
    bias: pd.DataFrame,
    levels: pd.DataFrame,
    cfg: StrategyConfig | None = None,
) -> EntryChain:
    """Turn a closed M15 frame, the HTF bias and a level book into intents and a ledger.

    ``ltf`` is the entry frame of the loader (its presumed unclosed tail is dropped), ``bias`` the
    markup of :func:`smc_zero.indicators.bias.bias_frames` built from *already aligned* HTF frames
    (the chain only joins on ``open_time``, so alignment stays the loader's business), and
    ``levels`` a book carrying ``broken_at`` (:func:`smc_zero.indicators.levels.level_lifecycle`);
    ``retired_at`` is derived here.  The result is an :class:`EntryChain` - one
    :class:`~smc_zero.strategy.base.TradeIntent` per accepted attempt and exactly one ledger row
    per rejected one (п.35).

    The walk is instance-major (level instance by level instance, bar by bar), and the rolling
    max/min test in front of the sweep search is the optimization п.35 allows: a window that never
    pierced the buffered price cannot contain a sweep, so skipping it loses no attempt.
    """
    config = StrategyConfig() if cfg is None else cfg
    _check_deferred(config)
    pip = config.pip_size
    sweep_buffer = config.sweep_buffer_pip * pip
    sl_buffer = config.sl_buffer_pip * pip
    limit_stop = config.entry.limit_stop_pip * pip
    min_sl = config.min_sl_pip * pip
    min_sl_realistic = config.min_sl_realistic_pip * pip
    max_sl_realistic = config.max_sl_realistic_pip * pip
    spread = config.risk.spread
    max_spread_pct = config.max_spread_pct_of_sl

    bars = _entry_bars(ltf)
    book = _level_book(levels)
    if bars.empty or book.empty:
        return EntryChain((), _ledger([]))
    direction = _bias_direction(bars, bias)
    high = bars["high"].to_numpy(dtype="float64")
    low = bars["low"].to_numpy(dtype="float64")
    close = bars["close"].to_numpy(dtype="float64")
    stamps = bars[TIMESTAMP_COLUMN]
    tradable = alfa_trading_mask(stamps, config.session).to_numpy(dtype=bool) & killzone_mask(
        stamps, config.session
    ).to_numpy(dtype=bool)
    breaks = structure_breaks(bars, config.bias.structure)
    break_dir = breaks["break_dir"].to_numpy(dtype="int8")
    # Gate (7): the impulse of the CHoCH break, read from its own ``disp_known_at`` (§7.8 п.40).
    impulse = displacement_gate(bars, breaks, config.displacement)
    disp_ok = impulse["disp_ok"].to_numpy(dtype=bool)
    disp_known = impulse["disp_known_at"].to_numpy(dtype="float64", na_value=np.nan)
    gaps = fair_value_gaps(bars, replace(config.fvg, min_gap_size=config.min_fvg_pip * pip))
    bearish_top = gaps["bearish_top"].to_numpy(dtype="float64")
    bearish_bottom = gaps["bearish_bottom"].to_numpy(dtype="float64")
    bullish_top = gaps["bullish_top"].to_numpy(dtype="float64")
    bullish_bottom = gaps["bullish_bottom"].to_numpy(dtype="float64")
    liquidity_cfg = replace(config.liquidity, sweep_buffer=sweep_buffer)
    roll_max = (
        pd.Series(high).rolling(liquidity_cfg.sweep_lookback + 1, min_periods=1).max().to_numpy()
    )
    roll_min = (
        pd.Series(low).rolling(liquidity_cfg.sweep_lookback + 1, min_periods=1).min().to_numpy()
    )

    rows = book.to_dict("records")
    groups = _price_groups(book, pip)
    names = book[LEVEL_NAME_COLUMN].to_list()
    prices = book[LEVEL_PRICE_COLUMN].to_numpy(dtype="float64")
    uppers = book[LEVEL_IS_UPPER_COLUMN].to_numpy(dtype=bool)
    dates = book[LEVEL_DATE_COLUMN].to_list()
    available = book[LEVEL_AVAILABLE_AT_COLUMN].to_list()
    intents: list[TradeIntent] = []
    rejections: list[dict[str, object]] = []
    # п.38: the cap counts setups per (name, date) instance, not per price like prod did.
    used: defaultdict[tuple[str, pd.Timestamp], int] = defaultdict(int)

    for position in range(len(book)):
        upper = bool(uppers[position])
        price = float(prices[position])
        name = str(names[position])
        level_date = pd.Timestamp(dates[position])
        available_at = available[position]
        side: Side = "short" if upper else "long"
        # п.35's prefilter: only bars whose sweep window pierced the buffered level.
        if upper:
            candidates = np.flatnonzero(roll_max > price + sweep_buffer)
        else:
            candidates = np.flatnonzero(roll_min < price - sweep_buffer)
        for i in candidates.tolist():
            if not tradable[i]:
                continue  # bar gate: outside Alfa hours / killzone no attempt exists at all
            sweep_bar = sweep_index(
                bars, level=price, upper=upper, current_idx=i, cfg=liquidity_cfg
            )
            if sweep_bar is None:
                continue  # no sweep found -> no attempt, hence no ledger row
            t = stamps.iloc[i]
            if t < available_at:  # (1)
                rejections.append(
                    _rejection(i, t, name, level_date, price, side, REASON_LEVEL_NOT_AVAILABLE)
                )
                continue
            if not _owns_price(rows, position, t, groups, pip):  # (2)
                rejections.append(
                    _rejection(i, t, name, level_date, price, side, REASON_LEVEL_BROKEN)
                )
                continue
            if used[(name, level_date)] >= config.max_setups_per_level_per_day:  # (3)
                rejections.append(
                    _rejection(i, t, name, level_date, price, side, REASON_LEVEL_USED_TODAY)
                )
                continue
            if direction[i] != (-1 if upper else 1):  # (4)
                rejections.append(_rejection(i, t, name, level_date, price, side, REASON_BIAS_SKIP))
                continue
            if i - sweep_bar > config.signal_max_age_bars:  # (5)
                rejections.append(
                    _rejection(i, t, name, level_date, price, side, REASON_STALE_SIGNAL)
                )
                continue
            choch_bar = _choch_bar(break_dir, sweep_bar, i, config.choch_wait_bars, upper)  # (6)
            if choch_bar is None:
                rejections.append(
                    _rejection(i, t, name, level_date, price, side, REASON_CHOCH_NOT_FOUND)
                )
                continue
            if config.use_displacement and not _displacement_ok(  # (7)
                disp_ok, disp_known, choch_bar, i
            ):
                rejections.append(
                    _rejection(i, t, name, level_date, price, side, REASON_DISPLACEMENT_SKIP)
                )
                continue
            if i - choch_bar < FVG_READY_BARS:  # (8)
                rejections.append(
                    _rejection(i, t, name, level_date, price, side, REASON_FVG_NOT_READY)
                )
                continue
            fvg_bar = _gap_bar(gaps, choch_bar, i, config.fvg_lookback, upper)  # (9)
            if fvg_bar is None:
                rejections.append(
                    _rejection(i, t, name, level_date, price, side, REASON_FVG_NOT_FOUND)
                )
                continue
            if i - fvg_bar > config.max_fvg_age_bars:  # (10)
                rejections.append(
                    _rejection(i, t, name, level_date, price, side, REASON_FVG_TOO_OLD)
                )
                continue
            if upper:
                top = float(bearish_top[fvg_bar])
                bottom = float(bearish_bottom[fvg_bar])
            else:
                top = float(bullish_top[fvg_bar])
                bottom = float(bullish_bottom[fvg_bar])
            entry = entry_level(top, bottom, config.entry.edge, bullish=not upper)
            # (11) prod lines 739-746: the limit may not hug the close of the signal bar.
            wrong_side = entry < close[i] + limit_stop if upper else entry > close[i] - limit_stop
            if wrong_side:
                rejections.append(
                    _rejection(i, t, name, level_date, price, side, REASON_WRONG_SIDE_LIMIT)
                )
                continue
            # (12) prod's ``source_extreme``: the extreme of the sweep bar sweep_index picked.
            extreme = float(high[sweep_bar]) if upper else float(low[sweep_bar])
            geometry = _sl_geometry(
                entry,
                extreme,
                top,
                bottom,
                upper=upper,
                sl_buffer=sl_buffer,
                min_sl_realistic=min_sl_realistic,
                max_sl_realistic=max_sl_realistic,
            )
            if isinstance(geometry, str):
                rejections.append(_rejection(i, t, name, level_date, price, side, geometry))
                continue
            sl, sl_size, sl_source = geometry
            if sl_size <= 0 or sl_size < min_sl:  # (13) prod lines 785-787
                rejections.append(
                    _rejection(i, t, name, level_date, price, side, REASON_MIN_SL_SKIP)
                )
                continue
            # (14) prod lines 797-800: inert while RiskConfig.spread is zero (Э3' default).
            if max_spread_pct > 0 and spread > sl_size * max_spread_pct:
                rejections.append(
                    _rejection(i, t, name, level_date, price, side, REASON_SPREAD_PCT_SKIP)
                )
                continue
            tp, tp_source, rr = take_profit_for(entry, sl, side, book, t, config.take_profit)
            intents.append(
                TradeIntent(
                    open_time=t,
                    bar=i,
                    side=side,
                    entry=float(entry),
                    sl=float(sl),
                    tp=float(tp),
                    tp_source=tp_source,
                    sl_source=sl_source,
                    sl_pips=float(sl_size / pip),
                    rr=float(rr),
                    level_name=name,
                    level_date=level_date,
                    level_price=price,
                    setup_type=config.setup_type,
                    sweep_bar=int(sweep_bar),
                    choch_bar=int(choch_bar),
                    fvg_bar=int(fvg_bar),
                )
            )
            used[(name, level_date)] += 1
    return EntryChain(tuple(intents), _ledger(rejections))
