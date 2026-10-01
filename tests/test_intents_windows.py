"""Window boundaries and ties of the vectorised chain: the Э4' windows, measured (Э9').

The rewrite of Э9' replaced three scans of the Э4' walk with range lookups over event arrays
(:func:`smc_zero.strategy.intents._first_in_window` for the first counter-trend break and the first
gap, :func:`~smc_zero.strategy.intents._attempt_bars` plus
:func:`~smc_zero.strategy.intents._sweep_bars` for the sweep), and a lookup is exactly where an
off-by-one hides: a ``searchsorted`` side, a clipped edge or a tie rule one letter away from the
scalar expression it replaces.

What is pinned here:

* the *boundaries* of every window, kept from §7.8 п.35: the sweep window includes its left edge and
  the counter-trend break may sit on the sweep bar or on the decision bar itself, while the gap
  window starts strictly after the CHoCH;
* the *tie rule* of the sweep search: the most extreme qualifying bar of the window, and the
  **earliest** one when two bars are equally extreme - prod's strict ``>``/``<`` update
  (``sweep_index``) - which is the mutation m2 of §7.13;
* the *equivalence* of the vectorised sweep with the scalar rule that is still shipped in
  :mod:`smc_zero.indicators.liquidity`: on random tapes every bar is asked both ways, and the
  attempt set has to be exactly "the bars where ``sweep_index`` answers a bar";
* the *same-bar rule* of п.35 (Э9''.1): two setups that reach the same bar leave one intent - the
  one whose level stands higher in ``LEVEL_PRIORITY`` - and the dropped setup writes no ledger row,
  because prod never evaluated it either.  Its mutation m3 of Э9''.1 ("keep every accepted intent",
  i.e. no dedup at all) turns
  :func:`test_one_bar_carries_the_most_significant_level_of_two_setups` red.

The scalar rule is the reference implementation of the sweep *window* (it stays the module's single
statement of the rule), and the random comparison is what makes the vectorised replacement of it
answerable in this repository and not only through the oracle of ``tests/test_intents_oracle.py``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from smc_zero.config import BrokerSpec, LiquidityConfig, RiskConfig, StrategyConfig
from smc_zero.data_loader import IS_CLOSED_COLUMN, TIMESTAMP_COLUMN
from smc_zero.indicators.bias import BIAS_DIR_COLUMN
from smc_zero.indicators.levels import (
    BROKEN_AT_COLUMN,
    LEVEL_AVAILABLE_AT_COLUMN,
    LEVEL_COLUMNS,
    LEVEL_DATE_COLUMN,
    LEVEL_IS_UPPER_COLUMN,
    LEVEL_NAME_COLUMN,
    LEVEL_PRICE_COLUMN,
    LEVEL_SOURCE_WINDOW_COLUMN,
    LONDON_HIGH,
    PDH,
)
from smc_zero.indicators.liquidity import sweep_index
from smc_zero.strategy.intents import (
    EntryChain,
    _attempt_bars,
    _first_in_window,
    _sweep_bars,
    build_intents,
)

DAY = "2026-06-10"  # Wednesday
PIP = 0.01  # a JPY-quote pip: every distance below reads as whole pips
BARS = 36  # 00:00 .. 08:45 UTC
HEAD = 24  # 06:00 UTC: the first London killzone bar of the day
LEVEL = 100.50
LOOKBACK = 48  # LiquidityConfig().sweep_lookback: the window the scenarios are read against


def _frame(rows: list[tuple[float, float, float]], *, periods: int = BARS) -> pd.DataFrame:
    """One M15 day: a flat head up to the killzone, ``rows``, then a flat tail."""
    flat_head = (100.40, 100.30, 100.35)
    flat_tail = (100.05, 99.95, 100.00)
    path = ([flat_head] * HEAD + list(rows))
    path = (path + [flat_tail] * max(0, periods - len(path)))[:periods]
    stamps = pd.date_range(f"{DAY} 00:00", periods=periods, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            TIMESTAMP_COLUMN: stamps,
            "open": [row[2] for row in path],
            "high": [row[0] for row in path],
            "low": [row[1] for row in path],
            "close": [row[2] for row in path],
            "volume": 1,
            IS_CLOSED_COLUMN: True,
        }
    )


def _bias(frame: pd.DataFrame, value: int = -1) -> pd.DataFrame:
    """The HTF markup of a short bias, one row per entry bar."""
    return pd.DataFrame({TIMESTAMP_COLUMN: frame[TIMESTAMP_COLUMN], BIAS_DIR_COLUMN: value})


def _book(*rows: tuple[str, float, bool, str | None, str | None]) -> pd.DataFrame:
    """A level book from ``(name, price, is_upper, available_at, broken_at)`` rows."""
    frame = pd.DataFrame(
        {
            LEVEL_NAME_COLUMN: [row[0] for row in rows],
            LEVEL_DATE_COLUMN: pd.to_datetime([DAY] * len(rows)),
            LEVEL_PRICE_COLUMN: [row[1] for row in rows],
            LEVEL_IS_UPPER_COLUMN: [row[2] for row in rows],
            LEVEL_AVAILABLE_AT_COLUMN: pd.to_datetime(
                [f"{DAY} {row[3]}" if row[3] else f"{DAY} 00:00" for row in rows], utc=True
            ),
            BROKEN_AT_COLUMN: pd.to_datetime(
                [f"{DAY} {row[4]}" if row[4] else None for row in rows], utc=True
            ),
            LEVEL_SOURCE_WINDOW_COLUMN: [row[0] for row in rows],
        }
    )
    return frame.loc[:, [*LEVEL_COLUMNS, BROKEN_AT_COLUMN]]


# The short scenario of the Э4' suite: a sweep of ``PDH = 100.50`` on bar 27, a CHoCH on bar 29 and
# the bearish gap of bar 31 (zone ``100.05 .. 100.25``), accepted on bar 32.
_REVERSAL = [
    (100.45, 100.25, 100.40),  # k=0
    (100.50, 100.15, 100.20),  # k=1  swing low 100.15, known on bar 26
    (100.55, 100.35, 100.50),  # k=2  pierced level (high 100.55) with a close back inside
    (100.62, 100.45, 100.48),  # k=3  bar 27: the swept extreme
    (100.50, 100.30, 100.35),  # k=4  bar 28
    (100.28, 99.95, 100.05),  # k=5  bar 29: close breaks the swing low -> CHoCH
    (100.40, 100.25, 100.35),  # k=6  bar 30: the gap's left candle (low 100.25)
    (100.20, 99.70, 99.80),  # k=7  bar 31: the gap's middle candle
    (100.05, 99.50, 99.60),  # k=8  bar 32: the decision bar
    (100.00, 99.50, 99.70),  # k=9
    (99.90, 99.55, 99.75),  # k=10
    (99.80, 99.50, 99.60),  # k=11
]
# The same scenario with the high of bar 28 raised to the swept extreme of bar 27: the two bars are
# then equally extreme, the earliest one still owns the sweep, and nothing else moves - the entry,
# the stop and the gap of the setup are all below them.
_TIED_EXTREMES = [*_REVERSAL[:4], (100.62, 100.30, 100.35), *_REVERSAL[5:]]


def _chain(
    rows: list[tuple[float, float, float]],
    *,
    levels: pd.DataFrame | None = None,
    **cfg_kw: object,
) -> EntryChain:
    """Run the chain on one scenario with the JPY pip of this file pinned (the Э4' fixture rule).

    Э10': the pip is a field of ``BrokerSpec`` now, so it is pinned *inside* the profile.
    """
    bars = _frame(rows)
    book = _book((PDH, LEVEL, True, None, None)) if levels is None else levels
    risk = replace(RiskConfig(), broker=replace(BrokerSpec(), pip_size=PIP))
    return build_intents(bars, _bias(bars), book, StrategyConfig(risk=risk, **cfg_kw))


def test_the_chain_reads_the_earliest_of_two_equal_extremes() -> None:
    """m2 of §7.13 through the public API: two equal highs, one sweep - the earlier bar.

    The scenario is the Э4' reversal with the height of bar 28 pushed to the extreme of bar 27, so
    the accepted intent is the same setup with the same geometry (the extreme is the same price) and
    the only field that can move is ``sweep_bar``: 27 while the earliest bar wins, 28 while the last
    one does - and a stop built from the later bar would be a different trade on a different bar.
    """
    accepted = _chain(_TIED_EXTREMES)
    assert len(accepted.intents) == 1
    intent = accepted.intents[0]
    assert intent.bar == 32
    assert intent.entry == pytest.approx(100.15)  # the mid of the gap 100.05 .. 100.25
    assert intent.sl == pytest.approx(100.67)  # 5 pips beyond the swept extreme 100.62
    assert intent.sl_source == "sweep_extreme"
    assert (intent.sweep_bar, intent.choch_bar, intent.fvg_bar) == (27, 29, 31)


def test_the_cap_of_an_instance_answers_for_every_later_bar() -> None:
    """The cap is reached by the *last* accepted setup, not only by a setup after it (§7.8 п.38).

    The scenario is the reversal with ``max_fvg_age_bars = 1``: the setup of bar 32 is accepted and
    the gap is too old for every later bar, so the three bars after it fail gate (10) instead of
    being accepted.  Gate (3) of the *instance* has already counted its cap-th setup at bar 32, and
    it answers before every later gate - so those bars are ``level_used_today`` and not
    ``fvg_too_old``.  This is the boundary the oracle found on the real window of Э9': a version
    that only reacts when a *further* setup would be accepted carried the wrong reason there.
    """
    accepted = _chain(_REVERSAL, max_fvg_age_bars=1)
    assert tuple(intent.bar for intent in accepted.intents) == (32,)
    pairs = list(zip(accepted.rejections["bar"].tolist(), accepted.rejections["reason"], strict=True))
    assert pairs == [
        *[(bar, "choch_not_found") for bar in (26, 27, 28)],
        *[(bar, "fvg_not_ready") for bar in (29, 30)],
        (31, "fvg_not_found"),
        *[(bar, "level_used_today") for bar in (33, 34, 35)],
    ]


def test_the_attempt_set_is_the_union_of_the_sweep_windows() -> None:
    """A sweep is a fact for its window: the swept bar and the ``sweep_lookback`` bars after it."""
    events = np.array([10, 200], dtype="int64")
    attempts = _attempt_bars(events, LOOKBACK, 300)
    assert attempts.tolist() == [
        *range(10, 10 + LOOKBACK + 1),
        *range(200, 200 + LOOKBACK + 1),
    ]
    assert 9 not in attempts.tolist(), "the bar before the sweep is not an attempt"
    assert 10 + LOOKBACK in attempts.tolist(), "the far edge of the window still is"
    # Overlapping windows are one run, and the right edge of the frame clips the last one.
    assert _attempt_bars(np.array([10, 20]), LOOKBACK, 300).tolist() == list(range(10, 20 + LOOKBACK + 1))
    assert _attempt_bars(np.array([295]), LOOKBACK, 300).tolist() == [295, 296, 297, 298, 299]
    assert _attempt_bars(np.empty(0, dtype="int64"), LOOKBACK, 300).size == 0


def test_the_first_in_window_lookup_pins_both_edges() -> None:
    """Both ends of a lookup window are inclusive and one bar outside them is not (m1).

    The window of the counter-trend break is ``[sweep, sweep + choch_wait]`` capped by the decision
    bar, so an off-by-one on either ``searchsorted`` edge would move a whole ledger column.
    """
    events = np.array([5, 9], dtype="int64")
    assert _first_in_window(events, np.array([5]), np.array([5])).tolist() == [5]
    assert _first_in_window(events, np.array([5]), np.array([9])).tolist() == [5]
    assert _first_in_window(events, np.array([6]), np.array([8])).tolist() == [-1]
    assert _first_in_window(events, np.array([9]), np.array([9])).tolist() == [9]
    assert _first_in_window(events, np.array([10]), np.array([12])).tolist() == [-1]
    assert _first_in_window(events, np.array([9]), np.array([8])).tolist() == [-1]
    assert _first_in_window(np.empty(0, dtype="int64"), np.array([0]), np.array([9])).tolist() == [-1]


def test_the_sweep_window_takes_the_most_extreme_bar_and_the_earliest_of_a_tie() -> None:
    """m2 of §7.13: an equally extreme pair resolves to the *earlier* bar, as prod's rule does."""
    high = np.array([1.10500, 1.10400, 1.10500, 1.10300], dtype=float)
    low = np.full(4, 1.00000, dtype=float)
    events = np.array([0, 2], dtype="int64")  # the same high on two bars of the window
    attempts = np.array([2, 3], dtype="int64")
    assert _sweep_bars(high, low, events, attempts, LOOKBACK, True).tolist() == [0, 0]
    # The extreme decides first: a higher bar of the window wins wherever it sits.
    assert _sweep_bars(high, low, events, np.array([1]), LOOKBACK, True).tolist() == [0]
    taller = np.array([1.10400, 1.10500, 1.10300], dtype=float)
    assert _sweep_bars(taller, low, np.array([0, 1]), np.array([1]), LOOKBACK, True).tolist() == [1]
    # The mirror case of a lower level: the earliest of two equal lows wins as well.
    lows = np.array([1.10000, 1.10000, 1.10100], dtype=float)
    flat = np.full(3, 1.20000, dtype=float)
    assert _sweep_bars(flat, lows, np.array([0, 1]), np.array([1]), LOOKBACK, False).tolist() == [0]


@pytest.mark.parametrize("upper", [True, False])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_the_vectorized_sweep_answers_exactly_what_the_scalar_rule_answers(upper: bool, seed: int) -> None:
    """The whole window rule, bar by bar, against the scalar ``sweep_index`` that is still shipped.

    A random walk is built *around* the level, so the tape carries sweeps, non-sweeps and bars that
    pierce without closing back inside - the three answers the window has to get right.  The attempt
    set of the vectorised chain must be exactly the bars the scalar rule answers a bar for, and the
    sweep bar of each of them must be the same index (ties included).  This is the mutation m1/m2 net
    inside this repository, next to the oracle of ``tests/test_intents_oracle.py``.
    """
    bars = 400
    level = 1.10300
    buffer = 0.00020
    rng = np.random.default_rng(seed)
    close = np.round(level + rng.normal(0.0, 0.00060, bars), 5)
    wick = np.round(rng.uniform(0.0, 0.00080, bars), 5)
    frame = pd.DataFrame({"high": close + wick, "low": close - wick, "close": close})
    cfg = replace(LiquidityConfig(), sweep_lookback=LOOKBACK, sweep_buffer=buffer)
    threshold = level + buffer if upper else level - buffer
    pierced = frame["high"].to_numpy(dtype=float) > threshold if upper else (
        frame["low"].to_numpy(dtype=float) < threshold
    )
    events = np.flatnonzero(pierced)
    inside = frame["close"].to_numpy(dtype=float)[events] < threshold if upper else (
        frame["close"].to_numpy(dtype=float)[events] > threshold
    )
    events = events[inside]
    assert events.size >= 5, "the fixture has to sweep the level more than once"
    attempts = _attempt_bars(events, cfg.sweep_lookback, bars)
    sweep = _sweep_bars(
        frame["high"].to_numpy(dtype=float),
        frame["low"].to_numpy(dtype=float),
        events,
        attempts,
        cfg.sweep_lookback,
        upper,
    )
    scalar = np.array(
        [
            -1 if (found := sweep_index(frame, level=level, upper=upper, current_idx=i, cfg=cfg)) is None
            else found
            for i in range(bars)
        ],
        dtype="int64",
    )
    assert attempts.tolist() == np.flatnonzero(scalar >= 0).tolist()
    assert sweep.tolist() == scalar[attempts].tolist()


def test_one_bar_carries_the_most_significant_level_of_two_setups() -> None:
    """Two levels swept into one bar leave one intent, and the dropped one is no rejection (m3).

    ``PDH = 100.50`` and ``LondonH = 100.30`` are both swept, both are broken by the CHoCH of bar 29
    and both reach the same gap on bar 32 - so the Э4' walk placed two orders on that bar, and the
    rule of п.35 keeps prod's one: the level standing higher in ``LEVEL_PRIORITY`` (PDH, 1) outranks
    the session range (LondonH, 5), whatever the order of the book is.  The dropped setup is *not* a
    refusal - prod never evaluated it - so it leaves no ledger row, while every refused attempt of
    both instances is still written down.
    """
    for book in (
        _book((LONDON_HIGH, 100.30, True, None, None), (PDH, LEVEL, True, None, None)),
        _book((PDH, LEVEL, True, None, None), (LONDON_HIGH, 100.30, True, None, None)),
    ):
        chain = _chain(_REVERSAL, levels=book)

        assert [(intent.bar, intent.level_name) for intent in chain.intents] == [(32, PDH)]
        assert 32 not in set(chain.rejections["bar"].tolist()), "the dropped setup is no refusal"
        assert set(chain.rejections["name"]) == {PDH, LONDON_HIGH}
