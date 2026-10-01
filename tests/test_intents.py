"""Entry-chain tests: the instance walk of SPEC_SMC.md §7.8 п.35-38 on synthetic OHLCV.

One UTC day of M15 bars (``00:00`` .. ``08:45``) carries two hand-built short setups for a level
above the market, so every expectation is arithmetic a reader can redo on paper:

* ``_REVERSAL`` - a sweep of ``PDH = 100.50`` on bar 27, a CHoCH on bar 29 and a bearish gap on
  bar 31 (zone ``100.05 .. 100.25``).  The gap is a *far* one: the swept extreme sits 13 pips
  above the gap mid, so the stop has to fall back on the gap edge and the 20-60 pip band decides;
* ``_SAWTOOTH`` - the same sweep and CHoCH, but the market runs away before printing the gap:
  the gap of bar 33 lies 13 pips above the swept extreme, so prod's ``fvg_edge`` fallback is the
  only stop that fits the band, while ``_NARROW`` (a 17.5 pip edge) and ``_WIDE`` (a 75 pip
  source) are the two rejects of the same gate.

Bar indices are absolute (``HEAD + k``), because the chain reports positions, not rows: the
scenario starts on bar 24, the first London killzone bar of the day, and the frame ends at bar
35, so the shots that matter stay inside one killzone window.  The level book is built by hand
(:func:`_book`), not by :mod:`smc_zero.indicators.levels`, because these tests are about the
*consumer* of the book; the Э3' suite owns the producer.
"""

from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest

from smc_zero.config import (
    BrokerSpec,
    DisplacementConfig,
    EntryConfig,
    RiskConfig,
    StrategyConfig,
    TPConfig,
)
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
    PDL,
)
from smc_zero.strategy.base import INTENT_COLUMNS, intents_frame
from smc_zero.strategy.intents import (
    REASON_BIAS_SKIP,
    REASON_CHOCH_NOT_FOUND,
    REASON_DISPLACEMENT_SKIP,
    REASON_FVG_NOT_FOUND,
    REASON_FVG_NOT_READY,
    REASON_FVG_TOO_OLD,
    REASON_LEVEL_BROKEN,
    REASON_LEVEL_NOT_AVAILABLE,
    REASON_LEVEL_USED_TODAY,
    REASON_MIN_SL_SKIP,
    REASON_SL_REJECTED_ALL,
    REASON_SL_REJECTED_WIDE,
    REASON_SPREAD_PCT_SKIP,
    REASON_STALE_SIGNAL,
    REASON_WRONG_SIDE_LIMIT,
    REJECTION_COLUMNS,
    REJECTION_REASONS,
    EntryChain,
    build_intents,
)
from smc_zero.strategy.risk_gate import RISK_PCT_COLUMN, RISK_WARNING_COLUMN, apply_risk_gate
from smc_zero.strategy.take_profit import build_take_profit

DAY = "2026-06-10"  # Wednesday
SATURDAY = "2026-06-13"
PIP = 0.01  # a JPY-quote pip, so every distance below reads as whole pips
BARS = 36  # 00:00 .. 08:45 UTC
HEAD = 24  # 06:00 UTC: the first London killzone bar of the day
FLAT_HEAD = (100.40, 100.30, 100.35)
FLAT_TAIL = (100.05, 99.95, 100.00)
LEVEL = 100.50

# (high, low, close) of bars 24 + k; the open is the previous close, which no gate reads.
_REVERSAL = [
    (100.45, 100.25, 100.40),  # k=0
    (100.50, 100.15, 100.20),  # k=1  swing low 100.15, known on bar 26
    (100.55, 100.35, 100.50),  # k=2  pierced level (high 100.55) with a close back inside
    (100.62, 100.45, 100.48),  # k=3  bar 27: the swept extreme
    (100.50, 100.30, 100.35),  # k=4
    (100.28, 99.95, 100.05),  # k=5  bar 29: close breaks the swing low -> CHoCH
    (100.40, 100.25, 100.35),  # k=6  gap left candle (low 100.25)
    (100.20, 99.70, 99.80),  # k=7  gap middle  (right candle: high 100.05)
    (100.05, 99.50, 99.60),  # k=8  bar 32: the decision bar of the far gap
    (100.00, 99.50, 99.70),  # k=9
    (99.90, 99.55, 99.75),  # k=10
    (99.80, 99.50, 99.60),  # k=11
]
# Same sweep and CHoCH, then a run-away rally that leaves the gap of bar 33 above the extreme.
_SAWTOOTH = [
    (100.45, 100.25, 100.40),
    (100.50, 100.15, 100.20),
    (100.55, 100.35, 100.50),
    (100.62, 100.45, 100.48),
    (100.50, 100.30, 100.35),
    (100.28, 99.95, 100.05),
    (100.40, 100.30, 100.38),
    (100.75, 100.35, 100.70),  # k=7  bar 31
    (101.20, 101.00, 101.15),  # k=8  gap left candle (low 101.00)
    (101.00, 100.60, 100.85),  # k=9  gap middle  (right candle: high 100.60)
    (100.60, 100.20, 100.25),  # k=10 bar 34: the decision bar
    (100.30, 100.10, 100.15),  # k=11
]
# The runaway gap pushed 1.5 pips closer: the edge fallback lands under min_sl_realistic_pip.
_NARROW = [*_SAWTOOTH[:8], (101.05, 100.85, 101.00), (100.85, 100.55, 100.70), *_SAWTOOTH[10:]]
# The sweep pierces 88 pips above the level: even the source stop is wider than the band.
_WIDE = [*_SAWTOOTH[:3], (101.50, 100.45, 100.48), *_SAWTOOTH[4:]]


def _frame(
    rows: list[tuple[float, float, float]],
    *,
    day: str = DAY,
    head: bool = True,
    periods: int = BARS,
    unclosed: tuple[int, ...] = (),
) -> pd.DataFrame:
    """One M15 day: ``rows`` right after the flat head, flat tail up to ``periods`` bars."""
    start = HEAD if head else 0
    path = [FLAT_HEAD] * start + list(rows)
    path = (path + [FLAT_TAIL] * max(0, periods - len(path)))[:periods]
    stamps = pd.date_range(f"{day} 00:00", periods=periods, freq="15min", tz="UTC")
    frame = pd.DataFrame(
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
    for index in unclosed:
        frame.loc[index, IS_CLOSED_COLUMN] = False
    return frame


def _bias(frame: pd.DataFrame, value: int = -1) -> pd.DataFrame:
    """The HTF markup of a short bias, one row per entry bar (the Э3' join is not under test)."""
    return pd.DataFrame({TIMESTAMP_COLUMN: frame[TIMESTAMP_COLUMN], BIAS_DIR_COLUMN: value})


def _book(*rows: tuple[str, float, bool, str | None, str | None]) -> pd.DataFrame:
    """A level book from ``(name, price, is_upper, available_at, broken_at)`` rows.

    ``None`` means the prices of the level were known from the day open (``available_at``) or
    that the level was never broken (``broken_at`` is ``NaT``).
    """
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


PDH_LEVEL = _book((PDH, LEVEL, True, None, None))
PDH_WITH_PDL = _book((PDH, LEVEL, True, None, None), (PDL, 99.20, False, None, None))


def _chain(
    rows: list[tuple[float, float, float]] = _REVERSAL,
    levels: pd.DataFrame | None = None,
    frame: pd.DataFrame | None = None,
    bias_frame: pd.DataFrame | None = None,
    bias_value: int = -1,
    **cfg_kw,
) -> EntryChain:
    """Run the chain on the scenario with ``pip_size`` pinned and the kwargs overridden.

    The FX pip is part of the C7 risk profile (§7.8 п.37), so it is pinned there - one copy of
    the broker numbers (Э10': inside ``BrokerSpec``, never beside it), and a ``risk`` override of
    the caller keeps it.
    """
    bars = _frame(rows) if frame is None else frame
    book = PDH_LEVEL if levels is None else levels
    markup = _bias(bars, bias_value) if bias_frame is None else bias_frame
    base_risk = cfg_kw.pop("risk", RiskConfig())
    risk = replace(base_risk, broker=replace(base_risk.broker, pip_size=PIP))
    return build_intents(bars, markup, book, StrategyConfig(risk=risk, **cfg_kw))


def _pairs(chain: EntryChain) -> list[tuple[int, str]]:
    """The ledger as ``(bar, reason)`` pairs; the chain's own sort makes this a contract."""
    return list(
        zip(chain.rejections["bar"].tolist(), chain.rejections["reason"].tolist(), strict=True)
    )


def _ledger(*segments: tuple[tuple[int, ...], str]) -> list[tuple[int, str]]:
    """Expand ``((bars,), reason)`` segments into the expected ``(bar, reason)`` pairs."""
    return [(bar, reason) for bars, reason in segments for bar in bars]


# The happy path of the reversal scenario, minus the acceptance itself: the three CHoCH-gate
# attempts, the two "gap not confirmable yet" bars, the one bar whose gap window missed and the
# three bars the per-level cap swallows.
_REVERSAL_LEDGER = _ledger(
    ((26, 27, 28), REASON_CHOCH_NOT_FOUND),
    ((29, 30), REASON_FVG_NOT_READY),
    ((31,), REASON_FVG_NOT_FOUND),
    ((33, 34, 35), REASON_LEVEL_USED_TODAY),
)


def test_the_reversal_setup_is_accepted_once_with_the_core_geometry() -> None:
    chain = _chain()
    assert len(chain.intents) == 1
    intent = chain.intents[0]
    assert intent.bar == 32
    assert intent.open_time == pd.Timestamp(f"{DAY} 08:00", tz="UTC")
    assert intent.side == "short"
    # the gap of bar 31 spans 100.05..100.25, so its mid is the limit of the entry mode "mid"
    assert intent.entry == pytest.approx(100.15)
    # 5 pips beyond the swept extreme of bar 27 (100.62): 52 pips, inside the 20-60 band
    assert intent.sl == pytest.approx(100.67)
    assert intent.sl_source == "sweep_extreme"
    assert intent.sl_pips == pytest.approx(52.0)
    assert intent.tp == pytest.approx(99.11)  # no counter-side level -> prod's 2R fallback
    assert intent.tp_source == "rr"
    assert intent.rr == pytest.approx(2.0)
    assert intent.level_name == PDH
    assert intent.level_date == pd.Timestamp(DAY)
    assert intent.level_price == pytest.approx(LEVEL)
    assert intent.setup_type == "fresh"
    assert (intent.sweep_bar, intent.choch_bar, intent.fvg_bar) == (27, 29, 31)
    assert _pairs(chain) == _REVERSAL_LEDGER


def test_the_ledger_carries_one_typed_row_per_filtered_attempt() -> None:
    ledger = _chain().rejections
    assert list(ledger.columns) == list(REJECTION_COLUMNS)
    assert ledger["bar"].dtype == np.dtype("int64")
    assert ledger["open_time"].dtype == pd.DatetimeTZDtype(tz="UTC")
    assert ledger["date"].dtype == np.dtype("datetime64[ns]")
    assert ledger["price"].dtype == np.dtype("float64")
    assert ledger["side"].unique().tolist() == ["short"]
    assert ledger["date"].eq(pd.Timestamp(DAY)).all()
    # every reason the chain may write belongs to the vocabulary of §7.8 п.35
    assert set(ledger["reason"]).issubset(REJECTION_REASONS)
    # the ledger is sorted by (bar, name, date): instance-major acceptance is not a contract
    assert ledger["bar"].tolist() == sorted(ledger["bar"].tolist())


def test_no_attempt_exists_outside_the_killzone() -> None:
    """The two bar gates write no row: before 06:00 UTC a signal does not exist (§7.8 п.35)."""
    early = _chain(frame=_frame(_REVERSAL, head=False, periods=20))
    assert early.intents == ()
    assert early.rejections.empty
    assert list(early.rejections.columns) == list(REJECTION_COLUMNS)


def test_the_weekend_has_no_alfa_hours() -> None:
    weekend = _chain(frame=_frame(_REVERSAL, day=SATURDAY))
    assert weekend.intents == ()
    assert weekend.rejections.empty


def test_an_empty_book_or_a_too_short_frame_is_not_an_error() -> None:
    for chain in (_chain(levels=_book()), _chain(frame=_frame(_REVERSAL, periods=3))):
        assert chain.intents == ()
        assert chain.rejections.empty
        assert list(chain.rejections.columns) == list(REJECTION_COLUMNS)


class _GateCase(NamedTuple):
    """One scenario of the gate walk: the inputs and the ledger those inputs must produce."""

    id: str
    levels: pd.DataFrame
    rows: list[tuple[float, float, float]]
    bias: int
    cfg_kw: dict[str, object]
    ledger: list[tuple[int, str]]
    accepted: tuple[int, ...]


# Every reason of §7.8 п.35 has a case in which it is the verdict of at least one bar, and each
# case pins the whole ledger - the order of the gates is what makes one reason win over another
# (e.g. a broken level answers on bar 30 before the FVG gate of the same bar is reached).
_GATE_CASES: tuple[_GateCase, ...] = (
    _GateCase("happy_path", PDH_LEVEL, _REVERSAL, -1, {}, _REVERSAL_LEDGER, (32,)),
    _GateCase(
        "level_not_available",
        _book((PDH, LEVEL, True, "08:30", None)),
        _REVERSAL,
        -1,
        {},
        _ledger(
            (tuple(range(26, 34)), REASON_LEVEL_NOT_AVAILABLE), ((35,), REASON_LEVEL_USED_TODAY)
        ),
        (34,),
    ),
    _GateCase(
        "level_broken",
        _book((PDH, LEVEL, True, None, "07:30")),
        _REVERSAL,
        -1,
        {},
        _ledger(
            ((26, 27, 28), REASON_CHOCH_NOT_FOUND),
            ((29,), REASON_FVG_NOT_READY),
            (tuple(range(30, 36)), REASON_LEVEL_BROKEN),
        ),
        (),
    ),
    _GateCase(
        "bias_skip",
        PDH_LEVEL,
        _REVERSAL,
        1,
        {},
        _ledger((tuple(range(26, 36)), REASON_BIAS_SKIP)),
        (),
    ),
    _GateCase(
        "stale_signal",
        PDH_LEVEL,
        _REVERSAL,
        -1,
        {"signal_max_age_bars": 0},
        _ledger(((26, 27), REASON_CHOCH_NOT_FOUND), (tuple(range(28, 36)), REASON_STALE_SIGNAL)),
        (),
    ),
    _GateCase(
        "choch_not_found",
        PDH_LEVEL,
        _REVERSAL,
        -1,
        {"choch_wait_bars": 1},
        _ledger((tuple(range(26, 36)), REASON_CHOCH_NOT_FOUND)),
        (),
    ),
    _GateCase(
        "displacement_skip",
        PDH_LEVEL,
        _REVERSAL,
        -1,
        {"displacement": DisplacementConfig(atr_mult_min=100.0)},
        _ledger(
            ((26, 27, 28), REASON_CHOCH_NOT_FOUND),
            (tuple(range(29, 36)), REASON_DISPLACEMENT_SKIP),
        ),
        (),
    ),
    _GateCase(
        "displacement_off",
        PDH_LEVEL,
        _REVERSAL,
        -1,
        {"use_displacement": False, "displacement": DisplacementConfig(atr_mult_min=100.0)},
        _REVERSAL_LEDGER,
        (32,),
    ),
    _GateCase(
        "fvg_not_found",
        PDH_LEVEL,
        _REVERSAL,
        -1,
        {"fvg_lookback": 1},
        _ledger(
            ((26, 27, 28), REASON_CHOCH_NOT_FOUND),
            ((29, 30), REASON_FVG_NOT_READY),
            (tuple(range(31, 36)), REASON_FVG_NOT_FOUND),
        ),
        (),
    ),
    _GateCase(
        "fvg_too_old",
        PDH_LEVEL,
        _REVERSAL,
        -1,
        {"max_fvg_age_bars": 0},
        _ledger(
            ((26, 27, 28), REASON_CHOCH_NOT_FOUND),
            ((29, 30), REASON_FVG_NOT_READY),
            ((31,), REASON_FVG_NOT_FOUND),
            (tuple(range(32, 36)), REASON_FVG_TOO_OLD),
        ),
        (),
    ),
    _GateCase(
        "wrong_side_limit",
        PDH_LEVEL,
        _REVERSAL,
        -1,
        {"entry": EntryConfig(limit_stop_pip=100.0)},
        _ledger(
            ((26, 27, 28), REASON_CHOCH_NOT_FOUND),
            ((29, 30), REASON_FVG_NOT_READY),
            ((31,), REASON_FVG_NOT_FOUND),
            (tuple(range(32, 36)), REASON_WRONG_SIDE_LIMIT),
        ),
        (),
    ),
    _GateCase(
        "sl_rejected_all",
        PDH_LEVEL,
        _NARROW,
        -1,
        {},
        _ledger(
            ((26, 27, 28), REASON_CHOCH_NOT_FOUND),
            ((29, 30), REASON_FVG_NOT_READY),
            ((31, 32, 33), REASON_FVG_NOT_FOUND),
            ((34, 35), REASON_SL_REJECTED_ALL),
        ),
        (),
    ),
    _GateCase(
        "sl_rejected_wide",
        PDH_LEVEL,
        _WIDE,
        -1,
        {},
        _ledger(
            ((26, 27, 28), REASON_CHOCH_NOT_FOUND),
            ((29, 30), REASON_FVG_NOT_READY),
            ((31, 32, 33), REASON_FVG_NOT_FOUND),
            ((34, 35), REASON_SL_REJECTED_WIDE),
        ),
        (),
    ),
    _GateCase(
        "min_sl_skip",
        PDH_LEVEL,
        _REVERSAL,
        -1,
        {"min_sl_pip": 60.0},
        _ledger(
            ((26, 27, 28), REASON_CHOCH_NOT_FOUND),
            ((29, 30), REASON_FVG_NOT_READY),
            ((31,), REASON_FVG_NOT_FOUND),
            (tuple(range(32, 36)), REASON_MIN_SL_SKIP),
        ),
        (),
    ),
    _GateCase(
        "spread_pct_skip",
        PDH_LEVEL,
        _REVERSAL,
        -1,
        {"risk": RiskConfig(broker=replace(BrokerSpec(), spread_pip=10.0))},
        # Э10': the Э5' ``spread=0.10`` price units *of the JPY pip* - 10 pips - stated in the
        # profile's own unit.  The Э5' keyword converts by the *default* FX pip instead, a 100x
        # wider spread that would hide the gate's own arithmetic behind the pin of :func:`_chain`.
        _ledger(
            ((26, 27, 28), REASON_CHOCH_NOT_FOUND),
            ((29, 30), REASON_FVG_NOT_READY),
            ((31,), REASON_FVG_NOT_FOUND),
            (tuple(range(32, 36)), REASON_SPREAD_PCT_SKIP),
        ),
        (),
    ),
)


@pytest.mark.parametrize("case", _GATE_CASES, ids=lambda case: case.id)
def test_every_gate_rejects_with_its_own_reason(case: _GateCase) -> None:
    chain = _chain(case.rows, case.levels, bias_value=case.bias, **case.cfg_kw)
    assert tuple(intent.bar for intent in chain.intents) == case.accepted
    assert _pairs(chain) == case.ledger


def test_the_nearest_visible_liquidity_level_is_the_target() -> None:
    intent = _chain(levels=PDH_WITH_PDL).intents[0]
    assert intent.tp == pytest.approx(99.20)
    assert intent.tp_source == "liquidity"
    assert intent.rr == pytest.approx((100.15 - 99.20) / 0.52)


def test_a_target_under_the_rr_floor_falls_back_to_the_multiple() -> None:
    """``min_tp_rr`` is a floor, not a filter: 100.55 is exactly 1R on the sawtooth geometry."""
    at_the_floor = _book((PDH, LEVEL, True, None, None), (PDL, 100.55, False, None, None))
    intent = _chain(_SAWTOOTH, at_the_floor).intents[0]
    assert intent.sl == pytest.approx(101.05)  # the fvg_edge fallback of the run-away gap
    assert intent.tp == pytest.approx(100.55)
    assert intent.tp_source == "liquidity"
    assert intent.rr == pytest.approx(1.0)
    under_the_floor = _book((PDH, LEVEL, True, None, None), (PDL, 100.56, False, None, None))
    intent = _chain(_SAWTOOTH, under_the_floor).intents[0]
    assert intent.tp == pytest.approx(100.30)  # 100.80 - 2 * 0.25
    assert intent.tp_source == "rr"
    assert intent.rr == pytest.approx(2.0)


def test_a_target_invisible_at_the_entry_bar_is_not_used() -> None:
    hidden = _book((PDH, LEVEL, True, None, None), (PDL, 100.55, False, "09:00", None))
    chain = _chain(_SAWTOOTH, hidden)
    assert chain.intents[0].tp_source == "rr"
    # the counter-side instance is walked as well, and its own gate 1 answers for it
    assert (35, REASON_LEVEL_NOT_AVAILABLE) in _pairs(chain)


def test_the_stop_falls_back_on_the_gap_edge_when_the_sweep_is_too_close() -> None:
    intent = _chain(_SAWTOOTH).intents[0]
    assert (intent.entry, intent.sl) == (pytest.approx(100.80), pytest.approx(101.05))
    assert intent.sl_source == "fvg_edge"
    assert intent.sl_pips == pytest.approx(25.0)
    assert intent.bar == 34


def test_the_same_price_is_traded_by_one_instance_only() -> None:
    """``deduplicate_levels``: two instances share a price, so the higher priority one owns it."""
    twin = _book((PDH, LEVEL, True, None, None), (LONDON_HIGH, LEVEL, True, None, None))
    chain = _chain(levels=twin)
    assert len(chain.intents) == 1
    assert chain.intents[0].level_name == PDH
    assert set(chain.rejections.loc[chain.rejections["name"] == LONDON_HIGH, "reason"]) == {
        REASON_LEVEL_BROKEN
    }


def test_a_duplicate_row_on_one_price_keeps_a_single_owner() -> None:
    """A successor row of the same name retires the earlier one, so a price has one owner.

    Both rows sit on ``PDH`` and the earlier one is retired exactly at the successor's
    ``available_at`` (the book's own day open), so gate 2 answers ``level_broken`` for it on every
    bar while the live instance trades the price.
    """
    duplicate = _book((PDH, LEVEL, True, None, None), (PDH, LEVEL, True, None, None))
    chain = _chain(levels=duplicate)
    assert len(chain.intents) == 1
    assert chain.intents[0].bar == 32
    retired = chain.rejections.loc[chain.rejections["reason"] == REASON_LEVEL_BROKEN]
    assert retired["bar"].tolist() == list(range(26, 36))
    assert _pairs(chain)[:1] == [(26, REASON_LEVEL_BROKEN)]
    combined = _ledger((tuple(range(26, 36)), REASON_LEVEL_BROKEN)) + _REVERSAL_LEDGER
    assert sorted(_pairs(chain)) == sorted(combined)


def test_the_per_level_cap_counts_setups_per_instance_per_day() -> None:
    chain = _chain(max_setups_per_level_per_day=2)
    assert tuple(intent.bar for intent in chain.intents) == (32, 33)
    assert _pairs(chain) == _ledger(
        ((26, 27, 28), REASON_CHOCH_NOT_FOUND),
        ((29, 30), REASON_FVG_NOT_READY),
        ((31,), REASON_FVG_NOT_FOUND),
        ((34, 35), REASON_LEVEL_USED_TODAY),
    )


def test_the_unclosed_tail_cannot_carry_a_signal() -> None:
    """Rule 2b: the last bar of the frame is presumed live, so it is not a decision bar."""
    closed = _chain(frame=_frame(_SAWTOOTH, periods=35))
    assert tuple(intent.bar for intent in closed.intents) == (34,)
    unclosed = _chain(frame=_frame(_SAWTOOTH, periods=35, unclosed=(34,)))
    assert unclosed.intents == ()
    assert _pairs(unclosed) == _ledger(
        ((26, 27, 28), REASON_CHOCH_NOT_FOUND),
        ((29, 30), REASON_FVG_NOT_READY),
        ((31, 32, 33), REASON_FVG_NOT_FOUND),
    )


def _core(chain: EntryChain) -> tuple[tuple[object, ...], ...]:
    """The accepted intents as comparable tuples: the fields a leak would move."""
    return tuple(
        (i.bar, i.entry, i.sl, i.tp, i.rr, i.sweep_bar, i.choch_bar, i.fvg_bar)
        for i in chain.intents
    )


def _past(chain: EntryChain, bar: int = 32) -> list[tuple[int, str]]:
    """The ledger rows up to ``bar``: the part no future candle may rewrite."""
    return [(row_bar, reason) for row_bar, reason in _pairs(chain) if row_bar <= bar]


def test_a_future_bar_cannot_change_the_past() -> None:
    """Rule 2: nothing that happens after the decision bar may move it."""
    reference = _chain()
    late_high = _frame(_REVERSAL)
    late_high.loc[35, "high"] = 999.0
    late_low = _frame(_REVERSAL)
    late_low.loc[33:35, "low"] = 1.0
    early = _frame(_REVERSAL)
    early.loc[27, "high"] = 999.0  # the sweep bar itself: bars before it must not move
    future_bias = _bias(_frame(_REVERSAL))
    future_bias.loc[33:, BIAS_DIR_COLUMN] = 1
    for frame, markup in ((late_high, None), (late_low, None), (None, future_bias)):
        other = _chain(frame=frame, bias_frame=markup)
        assert _core(other) == _core(reference)
        assert _past(other) == _past(reference)
    swept = _chain(frame=early)
    assert _past(swept, 26) == _past(reference, 26)
    assert _core(swept) != _core(reference)  # the sweep bar is a signal, not noise


def test_naive_stamps_are_read_as_utc() -> None:
    naive = _frame(_REVERSAL)
    naive[TIMESTAMP_COLUMN] = naive[TIMESTAMP_COLUMN].dt.tz_localize(None)
    chain = _chain(frame=naive)
    assert tuple(intent.bar for intent in chain.intents) == (32,)
    assert _pairs(chain) == _REVERSAL_LEDGER


def test_an_incomplete_entry_frame_is_a_hard_error() -> None:
    with pytest.raises(ValueError, match="needs the"):
        _chain(frame=_frame(_REVERSAL).drop(columns=["close"]))


def test_an_incomplete_bias_frame_is_a_hard_error() -> None:
    bars = _frame(_REVERSAL)
    with pytest.raises(ValueError, match="bias frame needs"):
        _chain(frame=bars, bias_frame=_bias(bars).drop(columns=[BIAS_DIR_COLUMN]))
    with pytest.raises(ValueError, match="does not cover"):
        _chain(frame=bars, bias_frame=_bias(bars).iloc[:-1])


def test_an_incomplete_level_book_is_a_hard_error() -> None:
    with pytest.raises(ValueError, match="level book needs"):
        _chain(levels=PDH_LEVEL.drop(columns=[BROKEN_AT_COLUMN]))


@pytest.mark.parametrize(
    "cfg_kw",
    [
        {"setup_type": "breaker"},
        {"entry": EntryConfig(type="sweep50")},
        {"fvg_select": "biggest"},
        {"take_profit": TPConfig(mode="rr")},
    ],
    ids=["setup_type", "entry.type", "fvg_select", "take_profit.mode"],
)
def test_a_deferred_config_value_is_refused(cfg_kw: dict[str, object]) -> None:
    """§7.8 п.33: a name v1 does not implement raises instead of trading something else."""
    with pytest.raises(NotImplementedError, match="not implemented by v1"):
        _chain(**cfg_kw)
# --------------------------------------------------------------------------------------------
# Gate (7), the displacement of the CHoCH break: its own tests, because prod never measured an
# impulse (SPEC_SMC.md §7.5) - the rule is pinned here and by the mutations of §7.8 п.40.
# --------------------------------------------------------------------------------------------


def _quiet_impulse_frame() -> pd.DataFrame:
    """``_REVERSAL`` with the pullback of bar 30 removed, so the CHoCH break survives.

    Bar 30 is the bar right after the CHoCH of bar 29, and in the scenario it closes back at
    ``100.35`` - above the broken swing low of ``100.15``.  Pushing that close down to ``100.10``
    keeps every price the setup is built from (the swept extreme of bar 27 and the wicks of the
    gap of bar 31) untouched while making the impulse survive its no-return window.
    """
    quiet = _frame(_REVERSAL)
    quiet.loc[30, ["open", "close"]] = 100.10
    return quiet


def test_a_weak_impulse_does_not_displace() -> None:
    """With §7.5 thresholds armed the doji scenario cannot pass the gate (m1 of п.40)."""
    for displacement in (
        DisplacementConfig(atr_mult_min=100.0),
        DisplacementConfig(body_frac_min=1.0),
    ):
        chain = _chain(displacement=displacement)
        assert chain.intents == ()
        assert _pairs(chain) == _ledger(
            ((26, 27, 28), REASON_CHOCH_NOT_FOUND),
            (tuple(range(29, 36)), REASON_DISPLACEMENT_SKIP),
        )


def test_the_pullback_after_the_choch_voids_the_impulse() -> None:
    """§7.5 п.20: a close back beyond the broken level inside the window kills the break (m1)."""
    chain = _chain(displacement=DisplacementConfig(no_return_bars=1))
    assert chain.intents == ()
    assert _pairs(chain) == _ledger(
        ((26, 27, 28), REASON_CHOCH_NOT_FOUND),
        (tuple(range(29, 36)), REASON_DISPLACEMENT_SKIP),
    )


def test_the_impulse_is_read_only_once_its_window_has_closed() -> None:
    """Gate (7) waits for ``disp_known_at``: bar 29 may not decide on bar 30 (m2 of п.40)."""
    quiet = _chain(frame=_quiet_impulse_frame(), displacement=DisplacementConfig(no_return_bars=1))
    assert _core(quiet) == _core(_chain())  # the accepted setup is the happy path
    assert _pairs(quiet) == _ledger(
        ((26, 27, 28), REASON_CHOCH_NOT_FOUND),
        ((29,), REASON_DISPLACEMENT_SKIP),
        ((30,), REASON_FVG_NOT_READY),
        ((31,), REASON_FVG_NOT_FOUND),
        ((33, 34, 35), REASON_LEVEL_USED_TODAY),
    )


def test_a_future_bar_cannot_make_the_impulse_known_earlier() -> None:
    """Leak test of §7.8 п.40 (m2): the verdict of bar 29 may not move with bar 30's close.

    In one frame the impulse survives its window, in the other bar 30 closes back beyond the
    broken level.  Up to bar 29 - the last bar whose verdict cannot depend on bar 30 - the two
    ledgers have to be identical; a gate that reads ``disp_ok`` without ``disp_known_at`` trades
    the first frame's bar 29 on an unseen return and breaks exactly here.
    """
    quiet = _chain(frame=_quiet_impulse_frame(), displacement=DisplacementConfig(no_return_bars=1))
    returned = _chain(displacement=DisplacementConfig(no_return_bars=1))
    assert _past(quiet, 29) == _past(returned, 29)
    assert (29, REASON_DISPLACEMENT_SKIP) in _past(quiet, 29)
    assert _core(quiet) != _core(returned)  # from bar 30 on the return is a fact


def test_the_displacement_gate_can_be_switched_off() -> None:
    """``use_displacement = False`` is the experiment switch of §7.8 п.40."""
    chain = _chain(
        use_displacement=False,
        displacement=DisplacementConfig(atr_mult_min=100.0, no_return_bars=1),
    )
    assert tuple(intent.bar for intent in chain.intents) == (32,)
    assert _pairs(chain) == _REVERSAL_LEDGER


def test_the_gate_order_decides_which_verdict_is_recorded() -> None:
    """Ruling R5: prod's order of §7.8 п.35 - level (2), then bias (4), then impulse (7).

    All three gates answer "no" for these attempts, so the ledger is the only place the order is
    visible; the accepted intents of a conjunctive chain are the same in any order.  The level is
    broken before the first attempt of the scenario (``06:15``, bar 25), so gate (2) owns every
    bar the sweep window produces.
    """
    broken_and_against = _chain(levels=_book((PDH, LEVEL, True, None, "06:15")), bias_value=1)
    assert broken_and_against.intents == ()
    assert _pairs(broken_and_against) == _ledger((tuple(range(26, 36)), REASON_LEVEL_BROKEN))
    against = _chain(bias_value=1, displacement=DisplacementConfig(atr_mult_min=100.0))
    assert against.intents == ()
    assert _pairs(against) == _ledger((tuple(range(26, 36)), REASON_BIAS_SKIP))


def test_the_chain_feeds_the_take_profit_and_the_risk_gate() -> None:
    """§7.8 п.41: the batch wrappers answer exactly what the chain built, row by row.

    ``PIP`` is a *JPY-quote* pip here (0.01) so that distances read as whole pips, so the risk
    percentage of this scenario is not the EURUSD one - the C7 arithmetic on EURUSD numbers (30
    pips at 0.1 lot = 3 % of the ruled deposit) lives in ``tests/test_risk_gate.py``.  What this
    test pins is the parity of the three faces: chain, take-profit builder and risk gate.
    """
    chain = _chain(levels=PDH_WITH_PDL)
    intent = chain.intents[0]
    priced = build_take_profit(intents_frame(chain.intents), PDH_WITH_PDL, TPConfig())
    assert priced.loc[0, "tp"] == pytest.approx(intent.tp)
    assert priced.loc[0, "tp_source"] == intent.tp_source
    assert priced.loc[0, "rr"] == pytest.approx(intent.rr)
    cfg = RiskConfig()
    # The chain measures its stop in *its* pips (0.01 in this fixture), the C7 profile prices in
    # EURUSD pips (0.0001): both spell the same 0.52 price distance, and that distance is what the
    # money at risk is built from - 52 * 0.0001 * 100 000 * 0.1 = $52, i.e. 5.2 % of the deposit.
    assert intent.sl_pips * PIP == pytest.approx(abs(intent.entry - intent.sl))
    # The prices of the scenario are JPY-scale, so the EURUSD profile of 100 000 a lot needs a
    # large account; the risk percentage is measured against the deposit, not that equity (R2).
    margin_needed = cfg.lot * cfg.contract_size * intent.entry / cfg.leverage
    kept, rejects = apply_risk_gate(priced, cfg, current_equity=1_000_000.0)
    assert rejects.empty
    assert list(kept.columns) == [*INTENT_COLUMNS, RISK_PCT_COLUMN, RISK_WARNING_COLUMN]
    expected = intent.sl_pips * cfg.pip_size * cfg.contract_size * cfg.lot / cfg.deposit * 100.0
    assert expected == pytest.approx(5.2)
    assert kept.loc[0, RISK_PCT_COLUMN] == pytest.approx(expected)
    assert bool(kept.loc[0, RISK_WARNING_COLUMN]) is (expected > cfg.warning_risk_pct)
    # the same intent on a $1000 account is refused, and the ledger carries the C7 margin
    refused, ledger = apply_risk_gate(priced, cfg, current_equity=1000.0)
    assert refused.empty
    assert ledger["reason"].tolist() == ["no_margin"]
    assert ledger["required_margin"].tolist() == [pytest.approx(margin_needed)]
