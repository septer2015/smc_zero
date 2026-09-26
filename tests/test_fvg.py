"""FVG tests: three-candle imbalance, visibility lag (``known_at = k + 1``), entry edge.

The visibility tests are the ones the п.8 mutation ("let the gap of bar ``k`` be
visible on bar ``k``") must break; the detection tests pin the prod wick formulas
and the ``min_gap_size`` boundary.
"""

from __future__ import annotations

import pandas as pd
import pytest

from smc_zero.config import FVGConfig
from smc_zero.indicators.fvg import entry_level, fair_value_gaps


def make_frame(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Build an OHLCV frame with a UTC M15 index from ``(open, high, low, close)`` rows."""
    index = pd.date_range("2026-01-05 09:00", periods=len(rows), freq="15min", tz="UTC")
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index)
    frame["volume"] = 1.0
    return frame


# Bullish (demand) gap with middle candle k=1: high[0] = 10.5 < low[2] = 11.0,
# so the zone is [10.5, 11.0] and it is known at bar 2.
BULL_ROWS = [
    (10.00, 10.50, 10.00, 10.40),
    (10.40, 12.00, 10.30, 11.80),
    (11.80, 11.90, 11.00, 11.50),
    (11.50, 11.60, 11.20, 11.40),
    (11.40, 11.45, 11.10, 11.20),
]

# Bearish (supply) gaps with middle candles k=2 (low[1] = 10.8 > high[3] = 10.6) and
# k=3 (low[2] = 10.4 > high[4] = 10.35).
BEAR_ROWS = [
    (10.90, 11.00, 10.70, 10.95),
    (11.00, 11.20, 10.80, 11.00),
    (11.00, 11.10, 10.40, 10.50),
    (10.50, 10.60, 10.20, 10.30),
    (10.30, 10.35, 10.10, 10.20),
]


def test_bullish_gap_bounds_size_and_middle_bar() -> None:
    frame = make_frame(BULL_ROWS)
    markup = fair_value_gaps(frame)
    assert markup.index.equals(frame.index)
    assert markup["bullish"].tolist() == [False, True, False, False, False]
    assert markup["bearish"].tolist() == [False] * 5
    middle = frame.index[1]
    assert markup.loc[middle, "bullish_bottom"] == 10.5  # high[k - 1]
    assert markup.loc[middle, "bullish_top"] == 11.0  # low[k + 1]
    assert markup.loc[middle, "bullish_size"] == pytest.approx(0.5)
    assert markup["bullish"].dtype == bool
    # No gap -> NaN bounds and a zero size, like prod initialises its arrays.
    assert pd.isna(markup.loc[frame.index[0], "bullish_bottom"])
    assert markup.loc[frame.index[0], "bullish_size"] == 0.0
    assert markup.loc[frame.index[3], "bearish_size"] == 0.0


def test_bearish_gap_bounds_size_and_middle_bar() -> None:
    frame = make_frame(BEAR_ROWS)
    markup = fair_value_gaps(frame)
    assert markup["bullish"].tolist() == [False] * 5
    assert markup["bearish"].tolist() == [False, False, True, True, False]
    assert markup.loc[frame.index[2], "bearish_top"] == 10.8  # low[k - 1]
    assert markup.loc[frame.index[2], "bearish_bottom"] == 10.6  # high[k + 1]
    assert markup.loc[frame.index[2], "bearish_size"] == pytest.approx(0.2)
    assert markup.loc[frame.index[3], "bearish_size"] == pytest.approx(0.05)


def test_touching_wicks_are_not_a_gap() -> None:
    """The formulas are strict: ``low[k + 1] == high[k - 1]`` is a touch, not a gap."""
    frame = make_frame(
        [
            (10.00, 10.50, 9.90, 10.05),
            (10.05, 10.60, 9.95, 10.20),
            (10.55, 10.60, 10.50, 10.58),
        ]
    )
    markup = fair_value_gaps(frame)
    assert markup["bullish"].tolist() == [False, False, False]
    assert markup["bearish"].tolist() == [False, False, False]


def test_gap_is_known_only_at_k_plus_one() -> None:
    """The п.8 mutation ``known_at = k`` must break this test."""
    frame = make_frame(BULL_ROWS)
    markup = fair_value_gaps(frame)
    known = markup["bullish_known_at"]
    assert known.dropna().tolist() == [2]
    assert known.dropna().index.tolist() == [frame.index[1]]
    assert str(known.dtype) == "Int64"
    assert markup["bearish_known_at"].isna().all()
    # A consumer that respects known_at sees the gap from bar 2 on, and nothing at
    # bars 0-1 (its own middle bar 1 included).
    visible = (known <= 1).sum()
    assert visible == 0
    assert (known <= 2).sum() == 1



def test_min_gap_size_filter_is_inclusive() -> None:
    frame = make_frame(BULL_ROWS)  # the gap is exactly 0.5 wide
    kept = fair_value_gaps(frame, FVGConfig(min_gap_size=0.5))
    assert kept["bullish"].tolist() == [False, True, False, False, False]
    filtered = fair_value_gaps(frame, FVGConfig(min_gap_size=0.5000001))
    assert filtered["bullish"].tolist() == [False] * 5
    assert filtered["bullish_size"].tolist() == [0.0] * 5
    assert filtered["bullish_known_at"].isna().all()


def test_entry_level_proximal_and_mid_for_both_directions() -> None:
    assert entry_level(11.0, 10.5, "proximal", bullish=True) == 11.0  # demand: top first
    assert entry_level(11.0, 10.5, "mid", bullish=True) == pytest.approx(10.75)
    assert entry_level(10.8, 10.6, "proximal", bullish=False) == 10.6  # supply: bottom first
    assert entry_level(10.8, 10.6, "mid", bullish=False) == pytest.approx(10.7)
    with pytest.raises(ValueError, match="unsupported entry mode"):
        entry_level(10.8, 10.6, "distal", bullish=False)  # type: ignore[arg-type]


def test_short_frames_are_markup_free_without_raising() -> None:
    short = [
        [],
        [(10.0, 10.1, 9.9, 10.0)],
        [(10.0, 10.1, 9.9, 10.0), (10.0, 10.2, 9.8, 9.9)],
    ]
    for rows in short:
        markup = fair_value_gaps(make_frame(rows))
        assert len(markup) == len(rows)
        assert not markup["bullish"].any()
        assert not markup["bearish"].any()


def test_future_candle_cannot_change_past_markup() -> None:
    """Rule 2 leak test: a tampered *future* bar leaves earlier rows untouched."""
    frame = make_frame(BEAR_ROWS)
    markup = fair_value_gaps(frame)
    for tampered in (3, 4):
        mutated = frame.copy(deep=True)
        fields = ["open", "high", "low", "close"]
        mutated.loc[mutated.index[tampered], fields] = [500.0, 501.0, 100.0, 100.5]
        # Bar k = tampered - 1 is the first row that may legitimately change: its gap
        # triple includes the tampered candle.  Everything before it must not move.
        pd.testing.assert_frame_equal(
            markup.iloc[: tampered - 1], fair_value_gaps(mutated).iloc[: tampered - 1]
        )


def test_input_frame_is_not_mutated_and_index_is_positional() -> None:
    frame = make_frame(BULL_ROWS)
    before = frame.copy(deep=True)
    fair_value_gaps(frame)
    pd.testing.assert_frame_equal(frame, before)
    bare = fair_value_gaps(make_frame(BULL_ROWS).reset_index(drop=True))
    assert bare["bullish_known_at"].dropna().tolist() == [2]
