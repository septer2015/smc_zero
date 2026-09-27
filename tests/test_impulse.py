"""Impulse-gate tests: Wilder ATR causality, displacement conditions, known-at lag.

There is no prod oracle for this module (SPEC_SMC.md §7.5 п.14: prod never measured the
impulse - ``BOS_MIN_BREAK_PIP`` is a break distance in pips and no ATR exists in the prod
sources), so the definition is pinned by these tests plus the mutation gates of §7.5 п.19:
``known_at = c`` without the lag must break test (d), an ATR that reads future bars must
break test (e), and dropping the fast-return condition must break test (c).

The fast-return rule is taken verbatim from §7.5 п.15: inside the ``no_return_bars`` window
*after* the confirming bar ``c`` no close may come back *beyond* the broken level, i.e.
``close > level`` for an up break and ``close < level`` for a down break, while a close
exactly *on* the level is not a return (strict comparison).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smc_zero.config import DisplacementConfig
from smc_zero.indicators.impulse import atr_wilder, displacement_gate
from smc_zero.indicators.structure import structure_breaks

ATR_PERIOD = 3


def make_frame(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Build an OHLCV frame with a UTC M15 index from ``(open, high, low, close)`` rows."""
    index = pd.date_range("2026-01-05 09:00", periods=len(rows), freq="15min", tz="UTC")
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index)
    frame["volume"] = 1.0
    return frame


def naive_atr(frame: pd.DataFrame, period: int) -> np.ndarray:
    """The plain Wilder recurrence of §7.5 п.15, written as a loop - the oracle of (a)."""
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    true_range = np.zeros(close.size, dtype=float)
    for i in range(close.size):
        if i == 0:
            true_range[i] = high[i] - low[i]
        else:
            true_range[i] = max(
                high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])
            )
    values = np.full(close.size, np.nan, dtype=float)
    if close.size >= period:
        values[period - 1] = true_range[:period].mean()
        for i in range(period, close.size):
            values[i] = (values[i - 1] * (period - 1) + true_range[i]) / period
    return values


def make_breaks(index: pd.Index, dirs: list[int], levels: list[float]) -> pd.DataFrame:
    """Hand-built ``breaks`` input: the gate only needs ``break_dir`` and ``break_level``."""
    return pd.DataFrame(
        {
            "break_dir": np.array(dirs, dtype=np.int8),
            "break_level": np.array(levels, dtype=float),
        },
        index=index,
    )


def gate_cfg(**overrides: object) -> DisplacementConfig:
    """The calibrated test config: ATR(3), thresholds just under/over the test impulse."""
    defaults: dict[str, object] = {
        "atr_period": ATR_PERIOD,
        "atr_mult_min": 1.0,
        "body_frac_min": 0.5,
        "no_return_bars": 2,
        "leg_bars": 1,
    }
    return DisplacementConfig(**{**defaults, **overrides})  # type: ignore[arg-type]


# Break bars: row 1 breaks DOWN through 10.06 and row 3 breaks UP through 10.10 with a
# 0.55 body inside a 0.65 range (the impulse); the bars after row 3 close at or below the
# broken level.  ATR(3) of this frame is [nan, nan, 0.20, 0.35, ...], so row 3 measures
# disp_atr = 0.55 / 0.35 and disp_body_frac = 0.55 / 0.65.
IMPULSE_ROWS = [
    (10.00, 10.10, 9.90, 10.00),
    (10.00, 10.10, 9.90, 10.05),
    (10.05, 10.15, 9.95, 10.00),
    (10.00, 10.60, 9.95, 10.55),
    (10.55, 10.60, 10.00, 10.05),
    (10.05, 10.10, 9.95, 10.00),
    (10.00, 10.10, 9.95, 10.00),
    (10.00, 10.10, 9.90, 10.00),
]
BREAK_DIRS = [0, -1, 0, 1, 0, 0, 0, 0]
BREAK_LEVELS = [np.nan, 10.06, np.nan, 10.10, np.nan, np.nan, np.nan, np.nan]

# Row 4 closes at 10.30, i.e. back beyond the broken 10.10 level, on the first bar of row
# 3's return window: the case the "drop the fast-return condition" mutation must break.
RETURNED_ROWS = [
    *IMPULSE_ROWS[:4],
    (10.55, 10.60, 10.25, 10.30),
    (10.30, 10.35, 10.20, 10.25),
    *IMPULSE_ROWS[6:],
]

# Row 4 closes exactly on 10.10: not a return (strict comparison), so the gate still holds.
TOUCHING_ROWS = [
    *IMPULSE_ROWS[:4],
    (10.55, 10.60, 10.05, 10.10),
    (10.10, 10.15, 10.00, 10.05),
    *IMPULSE_ROWS[6:],
]

# Structure frames (as in tests/test_structure.py): an up break against the swing high
# 11.0, and a down break against 9.0 followed by a CHoCH back above the swing high 9.9.
UP_ROWS = [
    (10.0, 10.5, 9.8, 10.0),
    (10.0, 11.0, 9.9, 10.2),
    (10.2, 10.6, 9.5, 9.8),
    (9.8, 11.4, 9.7, 11.2),
    (11.2, 11.5, 11.0, 11.3),
]
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


def test_atr_matches_the_naive_wilder_loop() -> None:
    """(a) The vectorised ``ewm`` recursion must equal the plain Wilder loop of the spec."""
    frame = make_frame(IMPULSE_ROWS)
    atr = atr_wilder(frame, ATR_PERIOD)
    assert atr.index.equals(frame.index)
    np.testing.assert_allclose(atr.to_numpy(), naive_atr(frame, ATR_PERIOD), equal_nan=True)
    assert atr.iloc[: ATR_PERIOD - 1].isna().all()  # no full window yet
    assert atr.iloc[ATR_PERIOD - 1] == pytest.approx(0.20)  # first window mean
    assert atr.iloc[3] == pytest.approx(0.35)  # (0.20 * 2 + 0.65) / 3
    assert str(atr.dtype) == "float64"


def test_atr_needs_a_full_window_and_rejects_a_bad_period() -> None:
    short = make_frame(IMPULSE_ROWS[:2])
    assert atr_wilder(short, ATR_PERIOD).isna().all()
    with pytest.raises(ValueError, match="period"):
        atr_wilder(short, 0)


def test_strong_impulse_passes_and_weak_impulse_fails() -> None:
    """(b) ``disp_ok`` reacts to both ``atr_mult_min`` and ``body_frac_min``."""
    frame = make_frame(IMPULSE_ROWS)
    breaks = make_breaks(frame.index, BREAK_DIRS, BREAK_LEVELS)
    strong = displacement_gate(frame, breaks, gate_cfg())
    assert strong["disp_ok"].tolist() == [False, False, False, True, False, False, False, False]
    assert strong["disp_close_beyond"].iloc[3]
    assert strong["disp_atr"].iloc[3] == pytest.approx(0.55 / 0.35)
    assert strong["disp_body_frac"].iloc[3] == pytest.approx(0.55 / 0.65)
    # Row 1 breaks down but its ATR window is not full yet (bar 1 < atr_period - 1):
    # "unknown counts as false", so the gate stays off.
    assert strong["disp_close_beyond"].iloc[1]
    assert not strong["disp_ok"].iloc[1]
    # Either threshold above the measured impulse turns the gate off.
    assert not displacement_gate(frame, breaks, gate_cfg(atr_mult_min=1.6))["disp_ok"].iloc[3]
    assert not displacement_gate(frame, breaks, gate_cfg(body_frac_min=0.9))["disp_ok"].iloc[3]


def test_zero_thresholds_only_block_the_unmeasured_rows() -> None:
    """The documented inert defaults: thresholds 0 block nothing but the unknown rows."""
    frame = make_frame(IMPULSE_ROWS)
    breaks = make_breaks(frame.index, BREAK_DIRS, BREAK_LEVELS)
    inert = gate_cfg(atr_mult_min=0.0, body_frac_min=0.0, no_return_bars=0)
    gate = displacement_gate(frame, breaks, inert)
    assert gate["disp_ok"].tolist() == [False, False, False, True, False, False, False, False]
    assert np.isnan(gate["disp_atr"].iloc[1])  # no ATR window yet -> NaN, not 0
    assert gate["disp_no_fast_return"].iloc[1]  # window 0 can never be violated


def test_leg_bars_widen_the_measured_leg() -> None:
    frame = make_frame(IMPULSE_ROWS)
    breaks = make_breaks(frame.index, BREAK_DIRS, BREAK_LEVELS)
    two = displacement_gate(frame, breaks, gate_cfg(leg_bars=2))
    # leg 2 at row 3: |close[3] - close[1]| / ATR[3] and the bodies/ranges of bars 2..3.
    assert two["disp_atr"].iloc[3] == pytest.approx(0.50 / 0.35)
    assert two["disp_body_frac"].iloc[3] == pytest.approx(0.60 / 0.85)
    assert pd.isna(two["disp_body_frac"].iloc[0])  # no full leg yet -> unknown, not 0
    assert two["disp_ok"].iloc[3]


def test_fast_return_inside_the_window_blocks_the_gate() -> None:
    """(c) The mutation "drop the fast-return condition" must break the ``returned`` case."""
    frame = make_frame(IMPULSE_ROWS)
    breaks = make_breaks(frame.index, BREAK_DIRS, BREAK_LEVELS)
    respected = displacement_gate(frame, breaks, gate_cfg())
    assert respected["disp_no_fast_return"].iloc[3]
    assert respected["disp_ok"].iloc[3]

    returned = make_frame(RETURNED_ROWS)
    back = displacement_gate(
        returned, make_breaks(returned.index, BREAK_DIRS, BREAK_LEVELS), gate_cfg()
    )
    assert not back["disp_no_fast_return"].iloc[3]
    assert not back["disp_ok"].iloc[3]

    touching = make_frame(TOUCHING_ROWS)
    on_level = displacement_gate(
        touching, make_breaks(touching.index, BREAK_DIRS, BREAK_LEVELS), gate_cfg()
    )
    assert on_level["disp_no_fast_return"].iloc[3]  # exactly on the level is not a return
    assert on_level["disp_ok"].iloc[3]


def test_known_at_is_the_confirmation_bar_plus_no_return_bars() -> None:
    """(d) The gate physically cannot be known before ``c + no_return_bars``."""
    frame = make_frame(IMPULSE_ROWS)
    breaks = make_breaks(frame.index, BREAK_DIRS, BREAK_LEVELS)
    for window, expected in ((0, [1, 3]), (2, [3, 5]), (3, [4, 6])):
        gate = displacement_gate(frame, breaks, gate_cfg(no_return_bars=window))
        known = gate["disp_known_at"]
        assert str(known.dtype) == "Int64"
        # Rows 1 and 3 carry a break_dir; every other row has no gate at all.
        assert known.isna().tolist() == [True, False, True, False, True, True, True, True]
        assert known.dropna().index.equals(frame.index[[1, 3]])
        assert known.dropna().tolist() == expected
        assert (known.dropna() - [1, 3]).tolist() == [window, window]


def test_future_candles_cannot_change_the_atr_of_past_bars() -> None:
    """(e) The mutation "ATR reads future bars" must break this."""
    frame = make_frame(IMPULSE_ROWS)
    fields = ["open", "high", "low", "close"]
    for tampered in (4, 5):
        mutated = frame.copy(deep=True)
        mutated.loc[mutated.index[tampered], fields] = [900.0, 1000.0, 890.0, 995.0]
        # ATR of bar i reads bars <= i only, so rows before the tampered bar are identical.
        pd.testing.assert_series_equal(
            atr_wilder(frame, ATR_PERIOD).iloc[:tampered],
            atr_wilder(mutated, ATR_PERIOD).iloc[:tampered],
        )
        # Positive control: the tampered candle does move its own ATR.
        assert (
            atr_wilder(mutated, ATR_PERIOD).iloc[tampered]
            != atr_wilder(frame, ATR_PERIOD).iloc[tampered]
        )


def test_tampering_inside_the_return_window_becomes_visible_at_known_at() -> None:
    """(e) A consumer at bar i may only read gate rows with ``known_at <= i``."""
    frame = make_frame(IMPULSE_ROWS)
    breaks = make_breaks(frame.index, BREAK_DIRS, BREAK_LEVELS)
    tampered_bar = 4  # the first bar of row 3's return window (c + 1 .. c + no_return_bars)
    mutated = frame.copy(deep=True)
    mutated.loc[mutated.index[tampered_bar], ["open", "high", "low", "close"]] = [
        10.55,
        10.60,
        10.25,
        10.30,
    ]
    base = displacement_gate(frame, breaks, gate_cfg())
    changed = displacement_gate(mutated, breaks, gate_cfg())

    def known_view(gate: pd.DataFrame, bar: int) -> pd.DataFrame:
        known = gate["disp_known_at"]
        return gate[known.notna() & (known <= bar)]

    known_at = int(base["disp_known_at"].iloc[3])
    assert known_at == 5  # c + no_return_bars
    assert base["disp_ok"].iloc[3] and not changed["disp_ok"].iloc[3]
    moved = [
        p for p in range(len(base)) if base["disp_ok"].iloc[p] != changed["disp_ok"].iloc[p]
    ]
    assert moved == [3]
    # Every bar before row 3's known_at sees byte-identical gate rows.  Row 1 (known at 3)
    # is already readable there, so this equality is not vacuous.
    for bar in range(known_at):
        pd.testing.assert_frame_equal(known_view(base, bar), known_view(changed, bar))
    assert known_view(base, known_at - 1).index.tolist() == [frame.index[1]]
    # The change becomes readable only from bar c + no_return_bars on.
    assert known_view(base, known_at).index.tolist() == [frame.index[1], frame.index[3]]
    assert not known_view(base, known_at).equals(known_view(changed, known_at))
    pd.testing.assert_frame_equal(
        known_view(base, known_at).iloc[:1], known_view(changed, known_at).iloc[:1]
    )


def test_gate_and_atr_do_not_mutate_their_inputs() -> None:
    """(f) Pure functions on the caller's frames."""
    frame = make_frame(IMPULSE_ROWS)
    breaks = make_breaks(frame.index, BREAK_DIRS, BREAK_LEVELS)
    frame_before = frame.copy(deep=True)
    breaks_before = breaks.copy(deep=True)
    gate = displacement_gate(frame, breaks, gate_cfg())
    atr_wilder(frame, ATR_PERIOD)
    pd.testing.assert_frame_equal(frame, frame_before)
    pd.testing.assert_frame_equal(breaks, breaks_before)
    assert gate.index.equals(frame.index)


def test_known_at_is_positional_on_any_index() -> None:
    """(f) The known-at column is a bar position, so the index labels must not matter."""
    frame = make_frame(IMPULSE_ROWS)
    cfg = gate_cfg()
    by_datetime = displacement_gate(
        frame, make_breaks(frame.index, BREAK_DIRS, BREAK_LEVELS), cfg
    )
    assert by_datetime["disp_known_at"].dropna().tolist() == [3, 5]

    positional = frame.reset_index(drop=True)
    by_position = displacement_gate(
        positional, make_breaks(positional.index, BREAK_DIRS, BREAK_LEVELS), cfg
    )
    assert by_position["disp_known_at"].dropna().tolist() == [3, 5]
    pd.testing.assert_frame_equal(by_datetime.reset_index(drop=True), by_position)

    # A tz-naive index over the same rows must not change the gate either.
    naive = make_frame(IMPULSE_ROWS)
    naive.index = naive.index.tz_localize(None)
    by_naive = displacement_gate(naive, make_breaks(naive.index, BREAK_DIRS, BREAK_LEVELS), cfg)
    pd.testing.assert_frame_equal(
        by_datetime.reset_index(drop=True), by_naive.reset_index(drop=True)
    )


def test_break_level_is_the_last_confirmed_swing_of_the_broken_side() -> None:
    """(g) ``structure_breaks`` must name the broken swing in ``break_level``."""
    up = make_frame(UP_ROWS)
    up_breaks = structure_breaks(up)
    assert up_breaks["break_dir"].tolist() == [0, 0, 0, 1, 1]
    assert up_breaks["break_level"].isna().tolist() == [True, True, True, False, False]
    assert up_breaks["break_level"].iloc[3] == pytest.approx(11.0)  # swing high at bar 1
    assert up_breaks["break_level"].iloc[4] == pytest.approx(11.0)  # still the same swing

    down = make_frame(DOWN_THEN_CHOCH_ROWS)
    down_breaks = structure_breaks(down)
    assert down_breaks["break_dir"].tolist() == [0, 0, 0, -1, -1, 0, 0, 1, 1]
    assert down_breaks["break_level"].iloc[3] == pytest.approx(9.0)  # broken swing low
    assert down_breaks["break_level"].iloc[4] == pytest.approx(9.0)
    assert down_breaks["break_level"].iloc[7] == pytest.approx(9.9)  # CHoCH swing high
    assert down_breaks["break_level"].iloc[8] == pytest.approx(9.9)
    no_break = [0, 5, 6]
    assert down_breaks["break_level"].iloc[no_break].isna().all()


def test_gate_consumes_real_structure_breaks() -> None:
    """End to end: the real ``break_dir``/``break_level`` columns drive the known-at rows."""
    frame = make_frame(DOWN_THEN_CHOCH_ROWS)
    gate = displacement_gate(frame, structure_breaks(frame), gate_cfg(no_return_bars=2))
    break_rows = [True, True, True, False, False, True, True, False, False]
    assert gate["disp_known_at"].isna().tolist() == break_rows
    # Break rows 3, 4, 7 and 8 -> known_at = c + 2.
    assert gate["disp_known_at"].dropna().tolist() == [5, 6, 9, 10]
    # Rows 7 and 8 close beyond their level, but their return window reaches past the last
    # bar: unknown = false, while known_at already points past the frame.
    assert gate["disp_close_beyond"].iloc[7] and not gate["disp_ok"].iloc[7]
    assert gate["disp_known_at"].iloc[7] == len(frame)


def test_config_and_breaks_input_validation() -> None:
    frame = make_frame(IMPULSE_ROWS)
    breaks = make_breaks(frame.index, BREAK_DIRS, BREAK_LEVELS)

    with pytest.raises(ValueError, match="atr_period"):
        DisplacementConfig(atr_period=0)
    with pytest.raises(ValueError, match="leg_bars"):
        DisplacementConfig(leg_bars=0)
    with pytest.raises(ValueError, match="no_return_bars"):
        DisplacementConfig(no_return_bars=-1)
    with pytest.raises(ValueError, match="atr_mult_min"):
        DisplacementConfig(atr_mult_min=-0.1)
    with pytest.raises(ValueError, match="body_frac_min"):
        DisplacementConfig(body_frac_min=-0.1)
    with pytest.raises(ValueError, match="break_dir"):
        displacement_gate(frame, breaks.drop(columns=["break_dir"]), gate_cfg())
    with pytest.raises(ValueError, match="break_level"):
        displacement_gate(frame, breaks.drop(columns=["break_level"]), gate_cfg())
    with pytest.raises(ValueError, match="rows"):
        displacement_gate(frame, breaks.iloc[:3], gate_cfg())

    # The defaults are inert: nothing is filtered until Э4' sets real thresholds.
    default = DisplacementConfig()
    assert default.atr_period == 14
    assert default.atr_mult_min == 0.0
    assert default.body_frac_min == 0.0
    assert default.no_return_bars == 0
    assert default.leg_bars == 1
