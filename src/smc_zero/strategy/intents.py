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

Third, **one bar carries at most one intent**: prod left its loop over the level instances of a
bar at the first accepted setup, so the setups of the levels it had not reached yet were never
evaluated.  v1 places all of them (п.51) and keeps the one prod would have reached first - the
instance standing higher in :data:`LEVEL_PRIORITY` (PDH/PDL > PWH/PWL > PMH/PML > Asian > London >
NY), the earliest instance of the walk winning a tie.  A setup dropped there is *not* a refused
attempt (prod never evaluated it either), so it leaves no ledger row; every attempt that reaches
a gate still does.

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
from functools import partial
from typing import NamedTuple

import numpy as np
import pandas as pd

from smc_zero.config import FVGEntryMode, Side, StrategyConfig
from smc_zero.data_loader import TIMESTAMP_COLUMN, drop_unclosed
from smc_zero.indicators.bias import BIAS_DIR_COLUMN
from smc_zero.indicators.fvg import ENTRY_MODES, fair_value_gaps
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
    retired_at,
)
from smc_zero.indicators.sessions import alfa_trading_mask, killzone_mask
from smc_zero.indicators.structure import STRUCTURE_LAYER_COLUMNS, structure_breaks
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

# prod's ``i - choch_idx < 2`` (core.py line 680) now lives in ``StrategyConfig.fvg_ready_bars``: a
# gap needs the bar after its own middle candle before it exists (``fair_value_gaps`` marks it from
# ``k + 1``), so the earliest a lookup can succeed is that many bars after the CHoCH - prod rejects
# earlier attempts instead of looking, and the M5 hierarchy of §7.20 raises the number.
#: What a missing stamp becomes on the integer clock of Э9': ``NaT`` is ``iNaT``, the smallest
#: ``int64``, so one comparison recognises a missing ``available_at`` (never fresh) and a missing
#: ``broken_at`` / ``retired_at`` (no limit) without a second ``isna`` pass over the column.
MISSING_STAMP = np.iinfo("int64").min


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


def _layer_arrays(
    layer: pd.DataFrame, bars: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the three arrays of a recorded structure layer, checked against the entry bars.

    ``layer`` is what :func:`smc_zero.indicators.structure.structure_layer` returns - the working
    frame's break, its impulse and the *entry* bar the impulse becomes knowable at.  The chain
    addresses ``bars`` positionally, so another length or a missing column would silently
    mis-address every gate; both are refused instead of guessed.
    """
    missing = [column for column in STRUCTURE_LAYER_COLUMNS if column not in layer.columns]
    if missing:
        raise ValueError(f"structure is missing the {missing} column(s)")
    if len(layer) != len(bars):
        raise ValueError(
            f"structure has {len(layer)} rows but the entry frame has {len(bars)}: "
            "pass the layer built for this very frame (structure_layer)"
        )
    return (
        layer["break_dir"].to_numpy(dtype="int8"),
        layer["disp_ok"].to_numpy(dtype=bool),
        layer["disp_known_at"].to_numpy(dtype="float64", na_value=np.nan),
    )


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


def _first_in_window(events: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Return the first event inside ``[lower, upper]`` for every element, or ``-1``.

    ``events`` holds ascending bar positions (the break bars of one direction, the gaps of one
    side), so "the first one in a window" is a pair of ``searchsorted`` calls instead of the
    scan the Э4' walk ran per attempt.  An empty window (``upper < lower``) and a window without
    an event answer the same ``-1``, which is how the scalar rules of that walk spelled ``None``
    (§7.8 п.35, gates 6 and 9).
    """
    if events.size == 0:
        return np.full(lower.shape, -1, dtype="int64")
    position = np.searchsorted(events, lower, side="left")
    candidate = events[np.minimum(position, events.size - 1)]
    found = (position < events.size) & (candidate <= upper)
    return np.where(found, candidate, -1)


def _sparse_tables(values: np.ndarray, keys: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build the range-query tables of ``(value, key)`` maxima over ascending ``keys``.

    Table ``t`` holds, for every block of ``2**t`` events, the best value of the block and the key
    carrying it; a tie stays on the left, and the left block always holds the smaller keys.  This
    is the structure that makes "the most extreme qualifying bar of the window" a constant number
    of lookups, i.e. it is what removes the per-bar rescan of the Э4' sweep search.  The caller
    negates its values to ask for a *minimum*; negation is exact and does not touch the tie rule.
    """
    tables = [(values, keys)]
    step = 1
    while 2 * step <= values.size:
        previous_values, previous_keys = tables[-1]
        length = previous_values.size - step
        left_values, left_keys = previous_values[:length], previous_keys[:length]
        right_values = previous_values[step : step + length]
        right_keys = previous_keys[step : step + length]
        left_wins = left_values >= right_values
        tables.append(
            (np.where(left_wins, left_values, right_values), np.where(left_wins, left_keys, right_keys))
        )
        step *= 2
    return tables


def _range_best(
    tables: list[tuple[np.ndarray, np.ndarray]], lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Return the key of the best event in ``[lower, upper]`` (event positions, inclusive).

    Two overlapping blocks cover every window, so one lookup per side answers the query: the left
    block wins a tie because its keys are the smaller ones, which is the "earliest on a tie" rule
    of :func:`smc_zero.indicators.liquidity.sweep_index`.  ``-1`` marks a window without an event.
    """
    keys = np.full(lower.shape, -1, dtype="int64")
    valid = upper >= lower
    if not valid.any():
        return keys
    span = np.where(valid, upper - lower + 1, 1)
    level = np.minimum(np.floor(np.log2(span)).astype("int64"), len(tables) - 1)
    for depth, (table_values, table_keys) in enumerate(tables):
        selected = valid & (level == depth)
        if not selected.any():
            continue
        left = lower[selected]
        right = upper[selected] - ((1 << depth) - 1)
        left_values, left_keys = table_values[left], table_keys[left]
        right_values, right_keys = table_values[right], table_keys[right]
        left_wins = left_values >= right_values
        keys[selected] = np.where(left_wins, left_keys, right_keys)
    return keys


def _attempt_bars(events: np.ndarray, lookback: int, bars: int) -> np.ndarray:
    """Return the bars whose sweep window holds a qualifying event, ascending.

    An attempt exists exactly where the Э4' walk found a sweep, and that is where the window
    ``[bar - lookback, bar]`` contains a bar that pierced the buffered level *and* closed back
    inside: the windows of those events cover the attempt set and nothing else.  The bars that
    only pierced the level - the ones ``sweep_index`` answers ``None`` for - are outside it, so
    the ledger stays free of them (§7.8 п.35).
    """
    if events.size == 0:
        return np.empty(0, dtype="int64")
    ends = np.minimum(events + lookback, bars - 1)
    # Two consecutive windows are one interval exactly while the next event starts inside the
    # previous window, so the runs of the union are found without touching the bars themselves.
    new_run = events[1:] > ends[:-1]
    starts = np.concatenate(([events[0]], events[1:][new_run]))
    stops = np.concatenate((ends[:-1][new_run], [ends[-1]]))
    lengths = stops - starts + 1
    repeated_starts = np.repeat(starts, lengths)
    completed = np.repeat(np.cumsum(lengths) - lengths, lengths)
    return repeated_starts + np.arange(int(lengths.sum()), dtype="int64") - completed


def _sweep_bars(
    high: np.ndarray,
    low: np.ndarray,
    events: np.ndarray,
    attempts: np.ndarray,
    lookback: int,
    upper: bool,
) -> np.ndarray:
    """Return the sweep bar of every attempt: the most extreme qualifying bar of its window.

    ``events`` are the ascending bars at which the level was swept - the pierce of the buffered
    price plus the close back inside - and the window of an attempt is
    ``[attempt - lookback, attempt]``.  Inside it the rule is the one of
    :func:`smc_zero.indicators.liquidity.sweep_index`: the highest high (the lowest low) among
    those bars, the earliest bar winning a tie.  The event table answers every attempt of the
    instance at once, which is the change Э9' is built on.
    """
    values = high[events] if upper else -low[events]
    tables = _sparse_tables(values, events)
    lower = np.searchsorted(events, np.maximum(attempts - lookback, 0), side="left")
    upper_position = np.searchsorted(events, attempts, side="right") - 1
    return _range_best(tables, lower, upper_position)


def _stamp_ns(stamps: pd.Series, *, source: str) -> np.ndarray:
    """Return a stamp column as UTC nanoseconds: the integer clock the walk compares on.

    Both the bars and the level book are localized to UTC first (the convention of
    :func:`smc_zero.strategy.base.utc_stamps`, which the level layer follows as well), so
    freshness and age become integer comparisons instead of Timestamp arithmetic - that is what
    lets gates (1), (2) and (5) answer a whole instance at once.
    """
    normalized = utc_stamps(stamps, source=source).dt.tz_localize(None)
    return normalized.to_numpy(dtype="datetime64[ns]").astype("int64")


def _fresh_limits(book: pd.DataFrame) -> np.ndarray:
    """Return the instant every instance stops being fresh, ``+inf`` when nothing limits it.

    Mirrors :func:`smc_zero.indicators.levels.fresh_at`: an instance is fresh until the earlier of
    its two limits, and a missing limit is no limit - so ``NaT`` in ``broken_at`` / ``retired_at``
    leaves the corresponding check out instead of retiring the instance at once.
    """
    infinite = np.iinfo("int64").max
    limits = np.full(len(book), infinite, dtype="int64")
    for column in (BROKEN_AT_COLUMN, LEVEL_RETIRED_AT_COLUMN):
        stamps = _stamp_ns(book[column], source="the level book")
        limits = np.minimum(limits, np.where(stamps == MISSING_STAMP, infinite, stamps))
    return limits


def _entry_prices(
    top: np.ndarray, bottom: np.ndarray, mode: FVGEntryMode, *, upper: bool
) -> np.ndarray:
    """Return the limit price of every attempt that reaches gate (11): prod's gap edge or the mid.

    The refusal of :func:`smc_zero.indicators.fvg.entry_level` travels with the rule: a mode v1
    does not implement raises instead of being read as another one, and the chain asks only when
    an attempt is actually at the gate - exactly where the Э4' walk called the scalar rule.
    """
    if mode not in ENTRY_MODES:
        raise ValueError(f"unsupported entry mode {mode!r}; expected one of {ENTRY_MODES}")
    if mode == "mid":
        return (top + bottom) / 2.0
    return bottom if upper else top


def _sl_arrays(
    entry: np.ndarray,
    extreme: np.ndarray,
    top: np.ndarray,
    bottom: np.ndarray,
    *,
    upper: bool,
    sl_buffer: float,
    min_sl_realistic: float,
    max_sl_realistic: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(sl, sl_size, from_edge, rejected_all, rejected_wide)`` of gate (12), per attempt.

    Ported expression by expression from prod's ``try_open_trade`` (core.py lines 748-781): the
    stop sits ``sl_buffer_pip`` beyond the swept extreme, and when that distance is narrower than
    ``min_sl_realistic_pip`` prod retries with the far edge of the entry gap.  The 20-60 pip band
    is applied *after* that fallback, so an attempt outside it is ``sl_rejected_all`` (neither
    anchor fits) or ``sl_rejected_wide`` (the swept extreme alone is too wide).
    """
    source = extreme + sl_buffer if upper else extreme - sl_buffer
    edge = top + sl_buffer if upper else bottom - sl_buffer
    source_size = source - entry if upper else entry - source
    edge_size = edge - entry if upper else entry - edge
    narrow = source_size < min_sl_realistic
    edge_fits = (min_sl_realistic <= edge_size) & (edge_size <= max_sl_realistic)
    from_edge = narrow & edge_fits
    return (
        np.where(from_edge, edge, source),
        np.where(from_edge, edge_size, source_size),
        from_edge,
        narrow & ~edge_fits,
        ~narrow & (source_size > max_sl_realistic),
    )



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


def _reject(reason: np.ndarray, pending: np.ndarray, mask: np.ndarray, code: str) -> None:
    """Write ``code`` on the pending attempts the mask selects and clear them (§7.8 п.35).

    The order of the gates is the order of the calls: an attempt already answered by an earlier
    gate keeps its first reason, which is what the ledger of the walk records.
    """
    hit = pending & mask
    reason[hit] = code
    pending[hit] = False


def _finalize_ledger(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a ledger frame typed and ordered: the contract of §7.8 п.38.

    The columns, the dtypes and the sort live here so that the row-at-a-time ledger of Э4' and the
    column-wise one of Э9' cannot drift apart on any of them.
    """
    frame = frame.loc[:, list(REJECTION_COLUMNS)].copy()
    frame["bar"] = frame["bar"].astype("int64")
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True).astype("datetime64[ns, UTC]")
    frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
    frame["price"] = frame["price"].astype("float64")
    if len(frame):
        frame = frame.sort_values(["bar", "name", "date"], kind="stable").reset_index(drop=True)
    return frame


def _ledger(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Return the ledger of a list of row dicts - the shape the refusing paths answer with."""
    return _finalize_ledger(pd.DataFrame(rows, columns=list(REJECTION_COLUMNS)))


def _ledger_frame(fragments: dict[str, list[np.ndarray]]) -> pd.DataFrame:
    """Return the ledger of column-wise fragments: one concatenation per column.

    The walk of Э9' emits the rows of an instance as a handful of small arrays (the bars, their
    stamps expanded as nanoseconds and the constant level fields), so the ledger is built by
    concatenating each column once instead of materialising a dict per row.
    """
    return _finalize_ledger(
        pd.DataFrame(
            {
                "bar": np.concatenate(fragments["bar"]),
                "open_time": pd.to_datetime(np.concatenate(fragments["open_time"]), utc=True),
                "name": np.concatenate(fragments["name"]),
                "date": pd.to_datetime(np.concatenate(fragments["date"])),
                "price": np.concatenate(fragments["price"]),
                "side": np.concatenate(fragments["side"]),
                "reason": np.concatenate(fragments["reason"]),
            }
        )
    )


def _one_intent_per_bar(intents: list[TradeIntent]) -> tuple[TradeIntent, ...]:
    """Return the accepted intents of the walk with the same-bar duplicates dropped (§7.8 п.35).

    prod left its loop over the level instances of a bar at the first accepted setup (``break`` in
    ``core.py``), so a bar never carried two orders and the setups of the levels it had not reached
    yet were never evaluated.  v1 places all of them (§7.8 п.51) and keeps here exactly the one
    prod would have reached first: the instance whose name stands higher in :data:`LEVEL_PRIORITY`
    (PDH/PDL > PWH/PWL > PMH/PML > Asian > London > NY), the instance coming first in the walk
    winning a tie.  The survivors keep the order they were accepted in - the order the engine is
    handed them in.

    A setup dropped here leaves no ledger row: it is not an attempt this chain *refused* (prod
    never evaluated it), so the ledger's promise - one row per rejected attempt - stays intact.
    """
    winner: dict[int, int] = {}
    rank: dict[int, int] = {}
    for index, intent in enumerate(intents):
        priority = LEVEL_PRIORITY.get(str(intent.level_name), UNKNOWN_LEVEL_PRIORITY)
        if intent.bar not in winner or priority < rank[intent.bar]:
            winner[intent.bar] = index
            rank[intent.bar] = priority
    return tuple(intent for index, intent in enumerate(intents) if winner[intent.bar] == index)


def build_intents(
    ltf: pd.DataFrame,
    bias: pd.DataFrame,
    levels: pd.DataFrame,
    cfg: StrategyConfig | None = None,
    *,
    structure: pd.DataFrame | None = None,
) -> EntryChain:
    """Turn a closed M15 frame, the HTF bias and a level book into intents and a ledger.

    ``ltf`` is the entry frame of the loader (its presumed unclosed tail is dropped), ``bias`` the
    markup of :func:`smc_zero.indicators.bias.bias_frames` built from *already aligned* HTF frames
    (the chain only joins on ``open_time``, so alignment stays the loader's business), and
    ``levels`` a book carrying ``broken_at`` (:func:`smc_zero.indicators.levels.level_lifecycle`);
    ``retired_at`` is derived here.  The result is an :class:`EntryChain` - one
    :class:`~smc_zero.strategy.base.TradeIntent` per accepted *bar* (:func:`_one_intent_per_bar`
    keeps the most significant level of a bar, п.35) and exactly one ledger row per rejected one
    (п.35).

    The walk stays instance-major - level instance by level instance, as in Э4' - but every gate of
    one instance answers *all* of its decision bars at once (Э9', SPEC_SMC.md §7.13).  An attempt
    exists where a sweep exists, and a sweep is a bar that pierced the buffered level and closed
    back inside (п.5, п.35), so the attempt set of an instance is the union of the
    ``[event, event + sweep_lookback]`` windows of those bars: bars that only pierced the level
    produce no attempt at all and no ledger row, exactly as the skipping prefilter of Э4' made
    sure.  The sweep bar of an attempt, the counter-trend break, the gap and the stop are then
    range lookups and arithmetic over the whole instance.

    ``structure`` is the optional layer of a working frame
    (:func:`smc_zero.indicators.structure.structure_layer`, the H4 -> M15 -> M5 hierarchy of §7.20):
    ``break_dir``, ``disp_ok`` and ``disp_known_at`` come from it instead of being measured on
    ``ltf`` itself, and it must be the layer of *this very frame* - one row per entry bar, in the
    entry order - or a ``ValueError``.  ``None`` keeps the v1 hierarchy, where the structure and the
    entry share one tape.
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
    # Gate (14) compares the spread of the profile with the stop in *price units*, which is what
    # ``broker.spread_abs`` answers (Э10': the profile speaks pips, the levels speak prices).
    spread = config.risk.broker.spread_abs
    max_spread_pct = config.max_spread_pct_of_sl
    lookback = config.liquidity.sweep_lookback
    cap = config.max_setups_per_level_per_day
    close_back_inside = config.liquidity.sweep_mode == "wick_close_inside"

    bars = _entry_bars(ltf)
    book = _level_book(levels)
    if bars.empty or book.empty:
        return EntryChain((), _ledger([]))
    direction = _bias_direction(bars, bias)
    high = bars["high"].to_numpy(dtype="float64")
    low = bars["low"].to_numpy(dtype="float64")
    close = bars["close"].to_numpy(dtype="float64")
    stamps = bars[TIMESTAMP_COLUMN]
    stamp_ns = _stamp_ns(stamps, source="the entry frame")
    tradable = alfa_trading_mask(stamps, config.session).to_numpy(dtype=bool) & killzone_mask(
        stamps, config.session
    ).to_numpy(dtype=bool)
    # Gates (6) and (7) read the structure of the working frame.  A caller that hands in a recorded
    # layer (:func:`smc_zero.indicators.structure.structure_layer`, §7.20) owns that choice: the
    # layer already speaks in entry bars, so the chain only takes its three columns.  Without one
    # the chain measures the entry frame itself - the v1 reading, bit for bit.
    if structure is None:
        breaks = structure_breaks(bars, config.bias.structure)
        break_dir = breaks["break_dir"].to_numpy(dtype="int8")
        impulse = displacement_gate(bars, breaks, config.displacement)
        disp_ok = impulse["disp_ok"].to_numpy(dtype=bool)
        disp_known = impulse["disp_known_at"].to_numpy(dtype="float64", na_value=np.nan)
    else:
        break_dir, disp_ok, disp_known = _layer_arrays(structure, bars)
    # Gate (6) reads the break of one direction only: an upper level needs a downward break.
    down_breaks = np.flatnonzero(break_dir == -1)
    up_breaks = np.flatnonzero(break_dir == 1)
    gaps = fair_value_gaps(bars, replace(config.fvg, min_gap_size=config.min_fvg_pip * pip))
    bearish_gaps = np.flatnonzero(gaps["bearish"].to_numpy(dtype=bool))
    bullish_gaps = np.flatnonzero(gaps["bullish"].to_numpy(dtype=bool))
    bearish_top = gaps["bearish_top"].to_numpy(dtype="float64")
    bearish_bottom = gaps["bearish_bottom"].to_numpy(dtype="float64")
    bullish_top = gaps["bullish_top"].to_numpy(dtype="float64")
    bullish_bottom = gaps["bullish_bottom"].to_numpy(dtype="float64")

    # The book as arrays: every gate of the walk compares columns instead of rows (Э9').
    groups = _price_groups(book, pip)
    names = book[LEVEL_NAME_COLUMN].to_list()
    prices = book[LEVEL_PRICE_COLUMN].to_numpy(dtype="float64")
    uppers = book[LEVEL_IS_UPPER_COLUMN].to_numpy(dtype=bool)
    dates = book[LEVEL_DATE_COLUMN].to_list()
    available = _stamp_ns(book[LEVEL_AVAILABLE_AT_COLUMN], source="the level book")
    limits = _fresh_limits(book)
    priorities = np.array(
        [LEVEL_PRIORITY.get(str(name), UNKNOWN_LEVEL_PRIORITY) for name in names], dtype="int64"
    )
    intents: list[TradeIntent] = []
    # п.38: the cap counts setups per (name, date) instance, not per price like prod did.
    ledger: dict[str, list[np.ndarray]] = {column: [] for column in REJECTION_COLUMNS}

    for position in range(len(book)):
        upper = bool(uppers[position])
        price = float(prices[position])
        name = str(names[position])
        level_date = dates[position]
        side: Side = "short" if upper else "long"
        threshold = price + sweep_buffer if upper else price - sweep_buffer
        # The swept bars of this instance: the pierce of the buffered level and the close back
        # inside, i.e. the two conditions ``sweep_index`` reads inside its window (§7.8 п.35).
        # The cheap half of the pair runs over the whole frame, the close test only over the
        # pierces - the same two passes per level the Э4' walk spent per (level, bar) pair.
        pierced = high > threshold if upper else low < threshold
        events = np.flatnonzero(pierced)
        if events.size and close_back_inside:
            inside = close[events] < threshold if upper else close[events] > threshold
            events = events[inside]
        attempts = _attempt_bars(events, lookback, len(bars))
        if attempts.size == 0:
            continue  # no sweep anywhere -> no attempt, hence no ledger row either
        attempts = attempts[tradable[attempts]]
        if attempts.size == 0:
            continue  # bar gate: outside Alfa hours / killzone no attempt exists at all
        sweep = _sweep_bars(high, low, events, attempts, lookback, upper)
        t_ns = stamp_ns[attempts]
        reason = np.empty(attempts.size, dtype=object)
        pending = np.ones(attempts.size, dtype=bool)
        refuse = partial(_reject, reason, pending)
        # (1) the price is not a fact yet.  ``NaT`` in ``available_at`` fails this test like it
        # fails the comparison of ``fresh_at``: such an instance is never fresh, not never known.
        known = (available[position] != MISSING_STAMP) & (t_ns < available[position])
        refuse(known, REASON_LEVEL_NOT_AVAILABLE)
        # (2) the instance owns its price: it has to be fresh itself - available, not broken, not
        # replaced by its successor - and outrank every fresh same-price / same-side member of the
        # book, priorities first and the book order as the tie-break.
        fresh = (t_ns >= available[position]) & (t_ns < limits[position])
        fresh &= available[position] != MISSING_STAMP
        shadow = np.zeros(attempts.size, dtype=bool)
        for member in groups[(round(price / pip), upper)]:
            if member == position or (priorities[member], member) >= (priorities[position], position):
                continue
            rival = (t_ns >= available[member]) & (t_ns < limits[member])
            shadow |= rival & (available[member] != MISSING_STAMP)
        broken = ~fresh | shadow
        refuse(broken, REASON_LEVEL_BROKEN)
        live = ~known & ~broken  # the attempts that reach gate (3) and can still be traded
        # (4) the side is against the closed HTF bias.  (5) the sweep is older than the age cap.
        refuse(direction[attempts] != (-1 if upper else 1), REASON_BIAS_SKIP)
        refuse(attempts - sweep > config.signal_max_age_bars, REASON_STALE_SIGNAL)
        # (6) the counter-trend break inside ``[sweep, sweep + choch_wait]``, nowhere after ``i``
        choch = _first_in_window(
            down_breaks if upper else up_breaks,
            sweep,
            np.minimum(attempts, sweep + config.choch_wait_bars),
        )
        refuse(choch < 0, REASON_CHOCH_NOT_FOUND)
        # (7) the impulse of that break, readable only from its own ``disp_known_at`` (§7.8 п.40)
        if config.use_displacement:
            reachable = np.where(choch >= 0, choch, 0)
            impulse_ok = (
                disp_ok[reachable]
                & np.isfinite(disp_known[reachable])
                & (disp_known[reachable] <= attempts)
            )
            refuse(~impulse_ok, REASON_DISPLACEMENT_SKIP)
        # (8) a gap needs the bar after its middle candle before it can exist at all
        refuse(attempts - choch < config.fvg_ready_bars, REASON_FVG_NOT_READY)
        # (9) the first gap inside ``(choch, choch + fvg_lookback]``, nowhere later than ``i - 1``
        fvg = _first_in_window(
            bearish_gaps if upper else bullish_gaps,
            np.maximum(choch + 1, 0),
            np.minimum(choch + config.fvg_lookback, attempts - 1),
        )
        refuse(fvg < 0, REASON_FVG_NOT_FOUND)
        # (10) the gap is older than ``max_fvg_age_bars``
        refuse(attempts - fvg > config.max_fvg_age_bars, REASON_FVG_TOO_OLD)
        # (11)..(14) and the acceptance: the limit, the stop band, the minimum stop, the spread
        if pending.any():
            if upper:
                top = bearish_top[fvg]
                bottom = bearish_bottom[fvg]
            else:
                top = bullish_top[fvg]
                bottom = bullish_bottom[fvg]
            entry = _entry_prices(top, bottom, config.entry.edge, upper=upper)
            # (11) prod lines 739-746: the limit may not hug the close of the signal bar.
            wrong_side = (
                entry < close[attempts] + limit_stop
                if upper
                else entry > close[attempts] - limit_stop
            )
            refuse(wrong_side, REASON_WRONG_SIDE_LIMIT)
            # (12) prod's ``source_extreme``: the extreme of the bar the sweep search picked.
            extreme = high[sweep] if upper else low[sweep]
            sl, sl_size, from_edge, rejected_all, rejected_wide = _sl_arrays(
                entry,
                extreme,
                top,
                bottom,
                upper=upper,
                sl_buffer=sl_buffer,
                min_sl_realistic=min_sl_realistic,
                max_sl_realistic=max_sl_realistic,
            )
            refuse(rejected_all, REASON_SL_REJECTED_ALL)
            refuse(rejected_wide, REASON_SL_REJECTED_WIDE)
            refuse((sl_size <= 0) | (sl_size < min_sl), REASON_MIN_SL_SKIP)  # (13)
            if max_spread_pct > 0:  # (14) the spread of the profile against the stop of the setup
                refuse(spread > sl_size * max_spread_pct, REASON_SPREAD_PCT_SKIP)
        # (3) the cap of §7.8 п.38: the counter is per instance, so once the cap-th setup is taken
        # gate (3) answers before every later gate and a bar past that setup keeps that very reason.
        accepted = np.flatnonzero(pending)
        if accepted.size >= cap:
            cutoff = attempts[accepted[cap - 1]]
            reached = live & (attempts > cutoff)
            reason[reached] = REASON_LEVEL_USED_TODAY
            pending[reached] = False
            accepted = accepted[:cap]
        for index in accepted:
            bar = int(attempts[index])
            t = stamps.iloc[bar]
            tp, tp_source, rr = take_profit_for(
                float(entry[index]), float(sl[index]), side, book, t, config.take_profit
            )
            intents.append(
                TradeIntent(
                    open_time=t,
                    bar=bar,
                    side=side,
                    entry=float(entry[index]),
                    sl=float(sl[index]),
                    tp=float(tp),
                    tp_source=tp_source,
                    sl_source="fvg_edge" if from_edge[index] else "sweep_extreme",
                    sl_pips=float(sl_size[index] / pip),
                    rr=float(rr),
                    level_name=name,
                    level_date=pd.Timestamp(level_date),
                    level_price=price,
                    setup_type=config.setup_type,
                    sweep_bar=int(sweep[index]),
                    choch_bar=int(choch[index]),
                    fvg_bar=int(fvg[index]),
                )
            )
        # The ledger rows of the instance, column by column: one array per column per instance.
        refused = ~pending
        if refused.any():
            rows_at = attempts[refused]
            ledger["bar"].append(rows_at)
            ledger["open_time"].append(stamp_ns[rows_at])
            ledger["name"].append(np.full(rows_at.size, name, dtype=object))
            ledger["date"].append(np.full(rows_at.size, np.int64(pd.Timestamp(level_date).value)))
            ledger["price"].append(np.full(rows_at.size, price, dtype="float64"))
            ledger["side"].append(np.full(rows_at.size, side, dtype=object))
            ledger["reason"].append(reason[refused])
    return EntryChain(
        _one_intent_per_bar(intents),
        _ledger_frame(ledger) if ledger["bar"] else _ledger([]),
    )
