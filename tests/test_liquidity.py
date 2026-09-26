"""Liquidity sweep tests: prod ``find_sweep_arr`` semantics, window and look-ahead.

The "no sweep" test (a wick that piershed but closed *outside* the threshold) is the
one the mandated mutation - dropping the close-back-inside condition - must break.
"""

from __future__ import annotations

import pandas as pd
import pytest

from smc_zero.config import LiquidityConfig
from smc_zero.indicators.liquidity import sweep_index

LEVEL = 100.0
BUFFER = 0.2  # threshold for the upper case: 100.2, for the lower one: 99.8


def make_frame(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Build an OHLCV frame with a UTC M15 index from ``(open, high, low, close)`` rows."""
    index = pd.date_range("2026-01-05 09:00", periods=len(rows), freq="15min", tz="UTC")
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index)
    frame["volume"] = 1.0
    return frame


def upper_cfg(**overrides: object) -> LiquidityConfig:
    return LiquidityConfig(sweep_buffer=BUFFER, **overrides)  # type: ignore[arg-type]


# Every bar below pierces 100.2 except bar 1; bar 3 closes back *outside* the
# threshold (100.5 > 100.2), so it is not a sweep.  Bar 4 closes between the level
# (100.0) and the threshold (100.2), which prod still counts as "back inside".
UPPER_ROWS = [
    (99.00, 100.50, 98.90, 99.90),
    (99.90, 100.10, 99.80, 100.00),
    (100.00, 100.90, 99.00, 100.00),
    (100.00, 100.90, 99.50, 100.50),
    (100.50, 101.50, 100.00, 100.10),
]

LOWER_ROWS = [
    (101.00, 101.10, 99.50, 100.50),
    (100.50, 100.60, 99.70, 99.90),
    (99.90, 100.00, 98.90, 100.00),
    (100.00, 100.10, 99.00, 99.50),
]


def test_upper_sweep_returns_the_most_extreme_bar() -> None:
    frame = make_frame(UPPER_ROWS)
    assert sweep_index(frame, level=LEVEL, upper=True, current_idx=2, cfg=upper_cfg()) == 2
    # Bar 4 pierces with the highest high of the window (101.5) -> it wins over bar 2
    # even though bar 3 is skipped (its close stayed outside the threshold).
    assert sweep_index(frame, level=LEVEL, upper=True, current_idx=4, cfg=upper_cfg()) == 4
    assert sweep_index(frame, level=LEVEL, upper=True, current_idx=3, cfg=upper_cfg()) == 2


def test_lower_sweep_returns_the_most_extreme_bar() -> None:
    frame = make_frame(LOWER_ROWS)
    assert sweep_index(frame, level=LEVEL, upper=False, current_idx=1, cfg=upper_cfg()) == 0
    assert sweep_index(frame, level=LEVEL, upper=False, current_idx=2, cfg=upper_cfg()) == 2
    # Bar 3 made the lowest low of the window but closed outside the threshold (99.5 <
    # 99.8), so bar 2 stays the answer.
    assert sweep_index(frame, level=LEVEL, upper=False, current_idx=3, cfg=upper_cfg()) == 2


def test_pierce_without_close_back_inside_is_not_a_sweep() -> None:
    """The mandatory mutation "drop the close-back-inside condition" must break this."""
    upper = make_frame(
        [
            (100.00, 100.50, 99.90, 100.40),
            (100.40, 100.60, 100.30, 100.55),
        ]
    )
    lower = make_frame(
        [
            (100.00, 100.10, 99.50, 99.60),
            (99.60, 99.70, 99.40, 99.45),
        ]
    )
    assert sweep_index(upper, level=LEVEL, upper=True, current_idx=1) is None
    assert sweep_index(lower, level=LEVEL, upper=False, current_idx=1) is None
    # The relaxed mode is a documented alternative, not the default: the very same bars
    # do count as a bare pierce, and the most extreme one wins (bar 1 in both frames).
    relaxed = LiquidityConfig(sweep_mode="wick_only")
    assert sweep_index(upper, level=LEVEL, upper=True, current_idx=1, cfg=relaxed) == 1
    assert sweep_index(lower, level=LEVEL, upper=False, current_idx=1, cfg=relaxed) == 1


def test_window_is_lookback_bars_back_inclusive_of_the_current_one() -> None:
    frame = make_frame(
        [
            (99.00, 100.50, 98.90, 99.50),  # the only sweep, made by bar 0
            (99.50, 100.10, 99.40, 99.90),
            (99.90, 100.05, 99.80, 99.95),
            (99.95, 100.00, 99.90, 99.95),
            (99.95, 100.10, 99.90, 99.90),
        ]
    )
    assert sweep_index(frame, level=LEVEL, upper=True, current_idx=4, cfg=upper_cfg()) == 0
    assert sweep_index(frame, level=LEVEL, upper=True, current_idx=0, cfg=upper_cfg()) == 0
    # i - lookback is inside the window: 4 - 4 = 0 still sees the sweep ...
    assert (
        sweep_index(frame, level=LEVEL, upper=True, current_idx=4, cfg=upper_cfg(sweep_lookback=4))
        == 0
    )
    # ... while a shorter window drops it.
    assert (
        sweep_index(frame, level=LEVEL, upper=True, current_idx=4, cfg=upper_cfg(sweep_lookback=2))
        is None
    )


def test_zero_buffer_keeps_the_pierce_strict() -> None:
    frame = make_frame(
        [
            (99.50, 100.00, 99.40, 99.90),  # high exactly on the level: still no pierce
            (99.90, 100.01, 99.80, 99.95),
        ]
    )
    assert sweep_index(frame, level=LEVEL, upper=True, current_idx=0) is None
    assert sweep_index(frame, level=LEVEL, upper=True, current_idx=1) == 1
    mirrored = make_frame(
        [
            (100.50, 100.60, 100.00, 100.10),
            (100.10, 100.20, 99.99, 100.05),
        ]
    )
    assert sweep_index(mirrored, level=LEVEL, upper=False, current_idx=0) is None
    assert sweep_index(mirrored, level=LEVEL, upper=False, current_idx=1) == 1


def test_out_of_range_index_and_config_validation() -> None:
    frame = make_frame(UPPER_ROWS)
    assert sweep_index(make_frame([]), level=LEVEL, upper=True, current_idx=0) is None
    with pytest.raises(IndexError, match="outside the frame"):
        sweep_index(frame, level=LEVEL, upper=True, current_idx=len(frame))
    with pytest.raises(IndexError, match="outside the frame"):
        sweep_index(frame, level=LEVEL, upper=True, current_idx=-1)
    with pytest.raises(ValueError, match="sweep_buffer"):
        LiquidityConfig(sweep_buffer=-0.1)
    with pytest.raises(ValueError, match="sweep_lookback"):
        LiquidityConfig(sweep_lookback=0)


def test_future_candle_cannot_change_the_sweep() -> None:
    """Rule 2: for every evaluated bar, a tampered *later* candle leaves the answer alone."""
    frame = make_frame([*UPPER_ROWS, (99.90, 100.00, 99.80, 99.90)])
    fields = ["open", "high", "low", "close"]
    for current_idx in range(4):
        baseline = sweep_index(
            frame, level=LEVEL, upper=True, current_idx=current_idx, cfg=upper_cfg()
        )
        mutated = frame.copy(deep=True)
        mutated.loc[mutated.index[5], fields] = [500.0, 501.0, 100.0, 100.5]
        assert (
            sweep_index(mutated, level=LEVEL, upper=True, current_idx=current_idx, cfg=upper_cfg())
            == baseline
        )
    # Positive control: the same edit *inside* the window does move the answer, so the
    # assertions above are not vacuous.
    inside = frame.copy(deep=True)
    inside.loc[inside.index[0], fields] = [99.0, 200.0, 98.0, 99.0]
    assert sweep_index(frame, level=LEVEL, upper=True, current_idx=4, cfg=upper_cfg()) == 4
    assert sweep_index(inside, level=LEVEL, upper=True, current_idx=4, cfg=upper_cfg()) == 0


def test_input_frame_is_not_mutated() -> None:
    frame = make_frame(UPPER_ROWS)
    before = frame.copy(deep=True)
    sweep_index(frame, level=LEVEL, upper=True, current_idx=4, cfg=upper_cfg())
    pd.testing.assert_frame_equal(frame, before)



def test_tie_goes_to_the_earliest_bar() -> None:
    frame = make_frame(
        [
            (99.00, 100.50, 98.90, 99.90),
            (99.90, 100.50, 99.80, 99.95),
            (99.95, 100.10, 99.70, 99.90),
        ]
    )
    assert sweep_index(frame, level=LEVEL, upper=True, current_idx=2, cfg=upper_cfg()) == 0
