"""Structure tests: N-bar swings, known-at lag, prod break parity, trend automaton.

Every test uses synthetic OHLCV (constitution rule 3) and the mandatory
look-ahead test tampers with a *future* candle to prove that past markup cannot
move.  The known-at tests are the ones the C4 mutation must break
(``known_at = i`` instead of ``i + N``).
"""

from __future__ import annotations

import pandas as pd

from smc_zero.config import StructureConfig
from smc_zero.indicators.structure import (
    known_swing_levels,
    structure_breaks,
    swing_points,
)


def make_frame(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Build an OHLCV frame with a UTC M15 index from ``(open, high, low, close)`` rows."""
    index = pd.date_range("2026-01-05 09:00", periods=len(rows), freq="15min", tz="UTC")
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index)
    frame["volume"] = 1.0
    return frame


# A swing high at bar 1 (11.0) and a swing low at bar 2 (9.5); closes then hold
# above 11.0 for two bars, which is prod's repeating break signal.
UP_ROWS = [
    (10.0, 10.5, 9.8, 10.0),
    (10.0, 11.0, 9.9, 10.2),
    (10.2, 10.6, 9.5, 9.8),
    (9.8, 11.4, 9.7, 11.2),
    (11.2, 11.5, 11.0, 11.3),
]

# Swing low at bar 1 (9.0) -> close below it (down break), then a swing high at
# bar 5 (9.9) -> close above it (CHoCH): the automaton has to flip the trend.
DOWN_THEN_CHOCH_ROWS = [
    (10.0, 10.5, 9.9, 10.4),
    (10.4, 10.6, 9.0, 9.4),
    (9.4, 9.5, 9.2, 9.45),
    (9.45, 9.5, 8.5, 8.6),
    (8.6, 8.7, 8.4, 8.65),
    (8.65, 9.9, 8.6, 9.85),
    (9.85, 9.8, 9.5, 9.6),
    (9.6, 10.1, 9.55, 10.05),
    (10.05, 10.2, 10.0, 10.15),
]


def test_one_bar_fractal_flags_and_prices() -> None:
    frame = make_frame(UP_ROWS)
    points = swing_points(frame)
    assert points.index.equals(frame.index)  # input index preserved
    assert points["swing_high"].tolist() == [False, True, False, False, False]
    assert points["swing_low"].tolist() == [False, False, True, False, False]
    assert points.loc[frame.index[1], "swing_high_price"] == 11.0
    assert points.loc[frame.index[2], "swing_low_price"] == 9.5
    assert pd.isna(points.loc[frame.index[0], "swing_high_price"])
    assert points["swing_high"].dtype == bool


def test_swing_knows_its_confirmation_bar_one_later() -> None:
    """N=1: the swing at bar 1 is known at bar 2, exactly prod's ``is_swing_high[i-1]``."""
    frame = make_frame(UP_ROWS)
    points = swing_points(frame)
    high_known = points["swing_high_known_at"]
    assert high_known.isna().tolist() == [True, False, True, True, True]
    assert high_known.dropna().tolist() == [2]
    assert high_known.dropna().index.tolist() == [frame.index[1]]
    assert points["swing_low_known_at"].dropna().tolist() == [3]
    assert str(high_known.dtype) == "Int64"


def test_plateau_is_not_a_swing_and_neighbourhood_is_strict() -> None:
    """Equal highs and a higher right neighbour must not produce a swing (prod uses ``>``)."""
    plateau = make_frame(
        [
            (10.0, 10.0, 9.0, 9.5),
            (9.5, 10.0, 9.1, 9.6),
            (9.6, 9.9, 9.2, 9.7),
        ]
    )
    assert not swing_points(plateau)["swing_high"].any()
    assert not swing_points(plateau)["swing_low"].any()


def test_lookback_two_needs_two_candles_on_both_sides() -> None:
    """A 1-bar spike is a swing for N=1 but never for N=2 (mask starts at bar N)."""
    frame = make_frame(
        [
            (10.0, 10.5, 9.5, 10.0),
            (10.0, 12.0, 9.8, 10.4),
            (10.4, 10.8, 10.0, 10.5),
            (10.5, 10.6, 10.1, 10.2),
            (10.2, 10.4, 9.9, 10.0),
        ]
    )
    assert swing_points(frame)["swing_high"].tolist() == [False, True, False, False, False]
    assert not swing_points(frame, StructureConfig(swing_lookback=2))["swing_high"].any()



def test_levels_enter_the_view_only_at_i_plus_lookback() -> None:
    """The C4 mutation ``known_at = i`` must break this test."""
    frame = make_frame(
        [
            (10.0, 10.0, 9.0, 9.5),
            (9.5, 10.1, 9.1, 9.8),
            (9.8, 12.0, 9.7, 11.0),
            (11.0, 11.2, 10.5, 10.8),
            (10.8, 10.9, 10.2, 10.4),
            (10.4, 10.6, 10.0, 10.2),
        ]
    )
    points = swing_points(frame, StructureConfig(swing_lookback=2))
    assert points["swing_high"].tolist() == [False, False, True, False, False, False]
    known = points["swing_high_known_at"]
    assert known.isna().tolist() == [True, True, False, True, True, True]
    assert known.dropna().tolist() == [4]
    levels = known_swing_levels(frame, StructureConfig(swing_lookback=2))
    # The peak at bar 2 (12.0) is invisible at bars 2 and 3, appears at bar 4.
    assert levels["last_swing_high"].iloc[:4].isna().all()
    assert levels["last_swing_high"].iloc[4:].tolist() == [12.0, 12.0]
    # With N=1 the same peak is already visible one bar earlier.
    fast = known_swing_levels(frame, StructureConfig(swing_lookback=1))
    assert fast.loc[frame.index[3], "last_swing_high"] == 12.0
    assert pd.isna(fast.loc[frame.index[2], "last_swing_high"])


def test_break_dir_is_prod_parity_and_repeats_on_the_same_level() -> None:
    frame = make_frame(UP_ROWS)
    breaks = structure_breaks(frame)
    assert breaks.index.equals(frame.index)
    assert breaks["break_dir"].tolist() == [0, 0, 0, 1, 1]
    assert breaks["break_dir"].dtype.name == "int8"


def test_first_break_is_bos_and_choch_flips_the_trend() -> None:
    breaks = structure_breaks(make_frame(DOWN_THEN_CHOCH_ROWS))
    assert breaks["break_dir"].tolist() == [0, 0, 0, -1, -1, 0, 0, 1, 1]
    assert breaks["structure_event"].tolist() == [
        None,
        None,
        None,
        "bos",  # trend initialisation: the decision taken with the project owner
        "bos",  # continuation while price stays beyond the level
        None,
        None,
        "choch",  # up break against the down trend
        "bos",  # and the trend is now up
    ]
    assert breaks["trend"].tolist() == [0, 0, 0, -1, -1, -1, -1, 1, 1]
    assert breaks["trend"].dtype.name == "int8"


def test_trend_stays_zero_without_any_break() -> None:
    frame = make_frame(
        [
            (10.0, 10.5, 9.5, 10.0),
            (10.0, 10.4, 9.6, 10.1),
            (10.1, 10.6, 9.7, 10.2),
            (10.2, 10.5, 9.8, 10.3),
        ]
    )
    breaks = structure_breaks(frame)
    assert breaks["break_dir"].tolist() == [0, 0, 0, 0]
    assert breaks["structure_event"].tolist() == [None, None, None, None]
    assert breaks["trend"].tolist() == [0, 0, 0, 0]


def test_close_confirmation_ignores_a_wick_only_pierce() -> None:
    """The mandated mutation "BOS by eyes (high instead of close)" must break this test."""
    frame = make_frame(
        [
            (10.0, 10.5, 9.8, 10.0),
            (10.0, 11.0, 9.9, 10.5),
            (10.5, 10.9, 10.4, 10.6),
            (10.6, 11.5, 10.5, 10.8),
        ]
    )
    assert structure_breaks(frame)["break_dir"].tolist() == [0, 0, 0, 0]
    wick = structure_breaks(frame, StructureConfig(confirmation="wick"))
    assert wick["break_dir"].tolist() == [0, 0, 0, 1]
    assert wick["structure_event"].tolist() == [None, None, None, "bos"]
    assert wick["trend"].tolist() == [0, 0, 0, 1]


def test_short_frames_are_markup_free_without_raising() -> None:
    frame = make_frame(
        [(10.0, 10.1, 9.9, 10.0), (10.0, 10.2, 9.8, 10.1), (10.1, 10.3, 9.7, 9.9)]
    )
    config = StructureConfig(swing_lookback=2)
    assert not swing_points(frame, config)["swing_high"].any()
    assert known_swing_levels(frame, config)["last_swing_high"].isna().all()
    breaks = structure_breaks(frame, config)
    assert breaks["break_dir"].tolist() == [0, 0, 0]
    assert breaks["trend"].tolist() == [0, 0, 0]



def test_future_candle_cannot_change_past_markup() -> None:
    """Rule 2 leak test: a tampered *future* bar leaves earlier rows untouched."""
    frame = make_frame(DOWN_THEN_CHOCH_ROWS)
    for tampered in (4, 6):
        mutated = frame.copy(deep=True)
        fields = ["open", "high", "low", "close"]
        mutated.loc[mutated.index[tampered], fields] = [999.0, 1000.0, 990.0, 995.0]
        # The bar before the tampered one is the first row that may legitimately change
        # (its fractal neighbourhood includes the tampered candle).
        pd.testing.assert_frame_equal(
            swing_points(frame).iloc[: tampered - 1], swing_points(mutated).iloc[: tampered - 1]
        )
        pd.testing.assert_frame_equal(
            known_swing_levels(frame).iloc[:tampered], known_swing_levels(mutated).iloc[:tampered]
        )
        pd.testing.assert_frame_equal(
            structure_breaks(frame).iloc[:tampered], structure_breaks(mutated).iloc[:tampered]
        )


def test_input_frame_is_not_mutated() -> None:
    """Pure functions only (SPEC_SMC.md, section 0.7): the argument stays intact."""
    frame = make_frame(UP_ROWS)
    before = frame.copy(deep=True)
    swing_points(frame)
    known_swing_levels(frame)
    structure_breaks(frame)
    pd.testing.assert_frame_equal(frame, before)


def test_known_at_is_positional_for_any_index() -> None:
    points = swing_points(make_frame(UP_ROWS).reset_index(drop=True))
    known = points["swing_high_known_at"]
    assert known.dropna().tolist() == [2]
    assert known.isna().tolist() == [True, False, True, True, True]
