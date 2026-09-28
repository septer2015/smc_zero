"""Take-profit tests: the liquidity target, its RR floor, the batch wrapper (§7.8 п.36, п.41).

Every expectation is hand arithmetic on a two-level book: a short at ``100.00`` with a 50 pip stop
makes the reward of a level its ``distance / 0.50``, so "nearest", "under the floor" and "fallback"
are readable without running anything.

The two mutations the module has to survive are the two rules the code could drop silently:

* m1 "ignore ``available_at``" - a level that is not a fact yet becomes the target; it must break
  :func:`test_a_level_that_is_not_a_fact_yet_is_not_a_target`;
* m2 "ignore ``min_tp_rr``" - a level under the floor becomes the target; it must break
  :func:`test_a_target_under_the_rr_floor_is_skipped_for_the_next_one`.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from smc_zero.config import TPConfig
from smc_zero.indicators.levels import (
    LEVEL_AVAILABLE_AT_COLUMN,
    LEVEL_DATE_COLUMN,
    LEVEL_IS_UPPER_COLUMN,
    LEVEL_NAME_COLUMN,
    LEVEL_PRICE_COLUMN,
    LEVEL_SOURCE_WINDOW_COLUMN,
    PDH,
    PDL,
    PWL,
)
from smc_zero.strategy.base import INTENT_COLUMNS, TradeIntent, intents_frame
from smc_zero.strategy.take_profit import TP_COLUMNS, build_take_profit, take_profit_for

DAY = "2026-06-10"  # Wednesday
ENTRY_TIME = pd.Timestamp(f"{DAY} 08:00", tz="UTC")
DAY_OPEN = f"{DAY} 00:00"


def _book(*rows: tuple[str, float, bool, str | None]) -> pd.DataFrame:
    """A level book from ``(name, price, is_upper, available_at)`` rows (Э3' columns).

    ``None`` as the availability means the prices of the level were known from the day open, and
    the rows are typed like :func:`smc_zero.indicators.levels.static_levels` hands them over.
    """
    return pd.DataFrame(
        {
            LEVEL_NAME_COLUMN: [row[0] for row in rows],
            LEVEL_DATE_COLUMN: pd.to_datetime([DAY] * len(rows)),
            LEVEL_PRICE_COLUMN: [row[1] for row in rows],
            LEVEL_IS_UPPER_COLUMN: [row[2] for row in rows],
            LEVEL_AVAILABLE_AT_COLUMN: pd.to_datetime(
                [f"{DAY} {row[3]}" if row[3] else DAY_OPEN for row in rows], utc=True
            ),
            LEVEL_SOURCE_WINDOW_COLUMN: [row[0] for row in rows],
        }
    )


def _intent(**overrides: object) -> TradeIntent:
    """One accepted intent; only the fields a test varies are passed in."""
    fields: dict[str, object] = {
        "open_time": ENTRY_TIME,
        "bar": 32,
        "side": "short",
        "entry": 100.00,
        "sl": 100.50,
        "tp": 0.00,
        "tp_source": "rr",
        "sl_source": "sweep_extreme",
        "sl_pips": 50.0,
        "rr": 2.0,
        "level_name": PDH,
        "level_date": pd.Timestamp(DAY),
        "level_price": 100.50,
        "setup_type": "fresh",
        "sweep_bar": 27,
        "choch_bar": 29,
        "fvg_bar": 31,
    }
    fields.update(overrides)
    return TradeIntent(**fields)  # type: ignore[arg-type]


def test_the_nearest_clearing_level_is_the_target() -> None:
    """A short falls to the *nearest* lower level, not to the farthest one."""
    book = _book((PDL, 99.40, False, None), (PWL, 99.00, False, None))
    tp, source, rr = take_profit_for(100.00, 100.50, "short", book, ENTRY_TIME, TPConfig())
    assert (tp, source) == (pytest.approx(99.40), "liquidity")
    assert rr == pytest.approx(1.2)  # 0.60 / 0.50, above the 1.0 floor


def test_a_long_targets_the_upper_side_only() -> None:
    """A long takes the upper level it can reach: the lower one is not a target at all."""
    book = _book((PDH, 100.60, True, None), (PDL, 99.60, False, None))
    tp, source, rr = take_profit_for(100.00, 99.50, "long", book, ENTRY_TIME, TPConfig())
    assert (tp, source) == (pytest.approx(100.60), "liquidity")
    assert rr == pytest.approx(1.2)

def test_a_target_under_the_rr_floor_is_skipped_for_the_next_one() -> None:
    """``min_tp_rr`` is a floor, not a filter: the 1.6R level wins over the 0.8R one (m2)."""
    book = _book((PDL, 99.60, False, None), (PWL, 99.20, False, None))
    tp, source, rr = take_profit_for(100.00, 100.50, "short", book, ENTRY_TIME, TPConfig())
    assert (tp, source) == (pytest.approx(99.20), "liquidity")
    assert rr == pytest.approx(1.6)  # 0.80 / 0.50
    loose = TPConfig(min_tp_rr=0.5)  # with a lower floor the nearer 0.8R level is the target
    tp, source, rr = take_profit_for(100.00, 100.50, "short", book, ENTRY_TIME, loose)
    assert (tp, source) == (pytest.approx(99.60), "liquidity")
    assert rr == pytest.approx(0.8)


def test_the_rr_fallback_is_prod_multiple_when_no_level_clears() -> None:
    """Without a level that clears the floor the target is ``entry ∓ sl * rr_fallback``."""
    under_the_floor = _book((PDL, 99.60, False, None))  # 0.8R
    for levels in (under_the_floor, _book()):
        tp, source, rr = take_profit_for(100.00, 100.50, "short", levels, ENTRY_TIME, TPConfig())
        assert (tp, source) == (pytest.approx(99.00), "rr")  # 100.00 - 2 * 0.50
        assert rr == pytest.approx(2.0)  # TPConfig.rr_fallback, prod's params["rr"]
    at_the_entry = _book((PDH, 100.00, True, None))  # *at* the entry is not counter-side either
    tp, source, _ = take_profit_for(100.00, 99.50, "long", at_the_entry, ENTRY_TIME, TPConfig())
    assert (tp, source) == (pytest.approx(101.00), "rr")


def test_a_level_that_is_not_a_fact_yet_is_not_a_target() -> None:
    """C2 п.2: only ``available_at <= t`` may become the target (m1)."""
    hidden = _book((PDL, 99.20, False, "09:00"))
    tp, source, _ = take_profit_for(100.00, 100.50, "short", hidden, ENTRY_TIME, TPConfig())
    assert (tp, source) == (pytest.approx(99.00), "rr")  # the level is not a fact yet
    visible = _book((PDL, 99.20, False, "07:00"))
    tp, source, _ = take_profit_for(100.00, 100.50, "short", visible, ENTRY_TIME, TPConfig())
    assert (tp, source) == (pytest.approx(99.20), "liquidity")
    # the boundary is inclusive: a level that became a fact at the decision bar is usable
    at_the_bar = _book((PDL, 99.20, False, "08:00"))
    tp, source, _ = take_profit_for(100.00, 100.50, "short", at_the_bar, ENTRY_TIME, TPConfig())
    assert (tp, source) == (pytest.approx(99.20), "liquidity")


def test_a_level_on_the_loss_side_of_the_entry_is_not_a_target() -> None:
    """A short cannot aim at a *higher* level: the counter-side filter is per side."""
    book = _book((PDH, 100.60, True, None), (PWL, 99.00, False, None))
    tp, source, _ = take_profit_for(100.00, 100.50, "short", book, ENTRY_TIME, TPConfig())
    assert (tp, source) == (pytest.approx(99.00), "liquidity")  # the 99.00 level, never 100.60

def test_a_take_profit_needs_a_book_and_a_stop() -> None:
    """A broken input is a hard error, not a silent fallback."""
    book = _book((PDL, 99.20, False, None))
    without_price = book.drop(columns=[LEVEL_PRICE_COLUMN])
    with pytest.raises(ValueError, match="needs a level book"):
        take_profit_for(100.00, 100.50, "short", without_price, ENTRY_TIME, TPConfig())
    with pytest.raises(ValueError, match="positive stop distance"):
        take_profit_for(100.00, 100.00, "short", book, ENTRY_TIME, TPConfig())
    undated = book.assign(**{LEVEL_AVAILABLE_AT_COLUMN: DAY_OPEN})
    with pytest.raises(ValueError, match="datetime column"):
        take_profit_for(100.00, 100.50, "short", undated, ENTRY_TIME, TPConfig())


def test_the_deferred_modes_are_refused() -> None:
    """§7.8 п.33: ``"liquidity"`` and ``"rr"`` are names prod knew, not implementations."""
    book = _book((PDL, 99.20, False, None))
    with pytest.raises(NotImplementedError, match="not implemented by v1"):
        take_profit_for(100.00, 100.50, "short", book, ENTRY_TIME, TPConfig(mode="rr"))
    with pytest.raises(NotImplementedError, match="not implemented by v1"):
        build_take_profit(intents_frame([]), book, TPConfig(mode="liquidity"))


def test_the_batch_wrapper_answers_what_the_scalar_answers() -> None:
    """§7.8 п.41: the wrapper rebuilds ``tp`` / ``tp_source`` / ``rr`` row by row."""
    book = _book((PDH, 100.60, True, None), (PDL, 99.40, False, None))
    intents = (
        _intent(),
        _intent(side="long", entry=100.00, sl=99.50, sl_pips=50.0, bar=33, level_name=PDL),
    )
    priced = build_take_profit(intents_frame(intents), book, TPConfig())
    assert list(priced.columns) == list(INTENT_COLUMNS)
    for position, intent in enumerate(intents):
        tp, source, rr = take_profit_for(
            intent.entry, intent.sl, intent.side, book, intent.open_time, TPConfig()
        )
        assert priced.loc[position, "tp"] == pytest.approx(tp)
        assert priced.loc[position, "tp_source"] == source
        assert priced.loc[position, "rr"] == pytest.approx(rr)
    assert priced["tp_source"].tolist() == ["liquidity", "liquidity"]
    assert priced["tp"].tolist() == [pytest.approx(99.40), pytest.approx(100.60)]
    assert priced.loc[:, list(TP_COLUMNS)].columns.tolist() == list(TP_COLUMNS)


def test_the_batch_wrapper_keeps_an_empty_frame_empty() -> None:
    """``intents_frame([])`` round-trips: no intents in, no intents out, columns intact."""
    empty = build_take_profit(intents_frame([]), _book(), TPConfig())
    assert list(empty.columns) == list(INTENT_COLUMNS)
    assert empty.empty


def test_the_batch_wrapper_validates_the_intents_frame() -> None:
    """A frame without the intent columns is a wiring error, not an empty run."""
    incomplete = intents_frame([_intent()]).drop(columns=["sl_pips"])
    with pytest.raises(ValueError, match="needs the"):
        build_take_profit(incomplete, _book(), TPConfig())


def test_the_rebuilt_intent_carries_the_new_source() -> None:
    """The wrapper returns a *new* value object: the input intent is not mutated."""
    intent = _intent()
    priced = build_take_profit(
        intents_frame([intent]), _book((PDL, 99.40, False, None)), TPConfig()
    )
    assert priced.loc[0, "tp_source"] == "liquidity"
    assert intent.tp_source == "rr"  # frozen and untouched
    assert replace(intent, tp=float(priced.loc[0, "tp"])).tp == pytest.approx(99.40)
