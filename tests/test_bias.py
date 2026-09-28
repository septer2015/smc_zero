"""Bias tests: H1/H4/D1 agreement, conflict policy, undefined trends, look-ahead.

Synthetic OHLCV only (constitution rules 1 and 3).  The HTF builders below produce a
*deterministic* structure, so every test knows exactly on which bar a timeframe's
trend flips:

* the head of a frame puts a swing high (``101.0``) at bar 1 and a swing low
  (``98.8``) at bar 3, so both are known at bars 2 and 4 (``swing_lookback = 1``);
* bar 5 of the head closes beyond one of those levels and therefore flips the trend
  (``+1`` up, ``-1`` down) at a fixed index;
* the monotone tail afterwards creates no new swing at all, so the trend of the head
  simply persists (a break in the same direction just repeats).

That fixed index is what the visibility assertions of (d)/(e) need: a tampered bar is
allowed to move the bias from its own ``close_time`` on, and nowhere earlier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smc_zero.config import BiasConfig
from smc_zero.data_loader import IS_CLOSED_COLUMN, TIMESTAMP_COLUMN, period_for
from smc_zero.indicators.bias import (
    BIAS_DIR_COLUMN,
    BIAS_STATE_COLUMN,
    bias_frames,
    trend_column,
)

M15_START = "2026-01-05 00:00"
M15_FREQ = "15min"
M15_BARS = 672  # seven days: covers the late D1 break of the undefined test

H1_START = "2026-01-05 00:00"
H1_FREQ = "1h"
H1_BARS = 168
H4_START = "2026-01-05 00:00"
H4_FREQ = "4h"
H4_BARS = 42
D1_START = "2026-01-01 00:00"
D1_FREQ = "1D"
D1_BARS = 12
D1_LATE_START = "2026-01-05 00:00"
D1_LATE_BARS = 10

TREND_FLIP_BAR = 5  # head bar 5 is the first close beyond the head's swing level
TREND_COLUMNS = ("H1", "H4", "D1")
# A bar body whose close (97.6) is below the head's swing low (98.8): it turns any
# already-trending frame into a CHoCH, so a tamper with it is guaranteed to flip.
FLIP_ROW = (98.0, 98.2, 97.5, 97.6)


def _head(direction: int) -> list[tuple[float, float, float, float]]:
    """First six bars: swing high at bar 1, swing low at bar 3, trend flip at bar 5."""
    if direction > 0:
        return [
            (100.0, 100.4, 99.6, 100.0),  # 0
            (100.0, 101.0, 100.0, 100.4),  # 1 swing high 101.0
            (100.4, 100.6, 99.4, 99.8),  # 2 -> 101.0 is known here
            (99.8, 100.2, 98.8, 99.2),  # 3 swing low 98.8
            (99.2, 100.8, 99.6, 100.4),  # 4 -> 98.8 is known here
            (100.4, 101.8, 100.4, 101.6),  # 5 close 101.6 > 101.0 -> trend +1
        ]
    return [
        (100.4, 100.6, 99.4, 100.0),  # 0
        (100.0, 100.2, 99.0, 99.8),  # 1 swing low 99.0
        (99.8, 100.6, 99.4, 100.2),  # 2 -> 99.0 is known here
        (100.2, 101.0, 99.6, 100.8),  # 3 swing high 101.0
        (100.8, 100.4, 99.8, 100.0),  # 4 -> 101.0 is known here
        (100.0, 99.6, 98.4, 98.6),  # 5 close 98.6 < 99.0 -> trend -1
    ]


def _tail(direction: int, bars: int) -> list[tuple[float, float, float, float]]:
    """Monotone continuation: no new swing, so the head's trend keeps repeating."""
    step = 0.05
    rows: list[tuple[float, float, float, float]] = []
    for k in range(bars):
        if direction > 0:
            open_ = 101.6 + step * k
            close_ = open_ + 0.06
            rows.append((open_, close_ + 0.04, open_ - 0.03, close_))
        else:
            open_ = 98.6 - step * k
            close_ = open_ - 0.06
            rows.append((open_, open_ + 0.03, close_ - 0.04, close_))
    return rows


def htf_frame(start: str, freq: str, bars: int, direction: int) -> pd.DataFrame:
    """Deterministic HTF frame whose trend is ``direction`` from bar 5 on."""
    rows = _head(direction) + _tail(direction, bars - len(_head(direction)))
    assert len(rows) == bars
    return _frame(start, freq, rows)


def ltf_frame(bars: int = M15_BARS) -> pd.DataFrame:
    """Flat M15 frame: the bias reads HTF trends only, so its own prices are inert."""
    return _frame(M15_START, M15_FREQ, [(100.0, 100.1, 99.9, 100.0)] * bars)


def _frame(
    start: str,
    freq: str,
    rows: list[tuple[float, float, float, float]],
) -> pd.DataFrame:
    """Loader-style frame: ``timestamp`` column, ``RangeIndex``, UTC labels."""
    stamps = pd.date_range(start=start, periods=len(rows), freq=freq, tz="UTC")
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=stamps)
    frame["volume"] = 1.0
    frame.insert(0, TIMESTAMP_COLUMN, stamps)
    return frame.reset_index(drop=True)


def _scenario(
    h1: int,
    h4: int,
    d1: int,
    *,
    d1_start: str = D1_START,
    d1_bars: int = D1_BARS,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """M15 frame plus one HTF frame per timeframe, each with the requested trend."""
    frames = {
        "H1": htf_frame(H1_START, H1_FREQ, H1_BARS, h1),
        "H4": htf_frame(H4_START, H4_FREQ, H4_BARS, h4),
        "D1": htf_frame(d1_start, D1_FREQ, d1_bars, d1),
    }
    return ltf_frame(), frames


def _directional(markup: pd.DataFrame) -> pd.DataFrame:
    """Rows where every configured trend exists and is non-zero."""
    trends = markup[[trend_column(tf) for tf in TREND_COLUMNS]]
    return trends.ne(0.0) & trends.notna()


def _changed_against(base: pd.DataFrame, other: pd.DataFrame) -> np.ndarray:
    """NaN-aware mask of the M15 bars where any markup column differs.

    The whole frame is compared, not only ``bias_dir`` / ``bias_state``: a flipped H1
    trend can leave both of them at ``undefined`` (``0``) while still having reached
    the M15 row, and that reach is exactly what the look-ahead tests must catch.
    """
    changed = np.zeros(len(base), dtype=bool)
    for column in base.columns:
        if column == TIMESTAMP_COLUMN:
            continue
        left = base[column]
        right = other[column]
        differs = (left != right).to_numpy()
        both_missing = (left.isna() & right.isna()).to_numpy()
        changed |= differs & ~both_missing
    return changed


def _close_time(frames: dict[str, pd.DataFrame], timeframe: str, bar_index: int) -> pd.Timestamp:
    """``close_time`` of one HTF bar: the first instant its trend may be visible."""
    return frames[timeframe][TIMESTAMP_COLUMN].iloc[bar_index] + period_for(timeframe)


@pytest.mark.parametrize("direction", (1, -1), ids=("long", "short"))
def test_agreeing_trends_give_a_directional_bias(direction: int) -> None:
    """(a) H1 + H4 + D1 point the same way -> bias_dir = direction, state = agree_*."""
    ltf, frames = _scenario(direction, direction, direction)

    markup = bias_frames(ltf, frames)

    expected_state = "agree_long" if direction > 0 else "agree_short"
    agreeing = _directional(markup).all(axis=1) & markup[
        [trend_column(tf) for tf in TREND_COLUMNS]
    ].eq(direction).all(axis=1)
    assert bool(agreeing.any())  # the D1 trend must actually arrive inside the range
    assert markup.loc[agreeing, BIAS_DIR_COLUMN].eq(direction).all()
    assert markup.loc[agreeing, BIAS_STATE_COLUMN].eq(expected_state).all()
    # the first agreeing bar is D1's flipping close_time: that D1 bar opens on
    # 2026-01-06 00:00 and closes a whole day later, so Jan 7 00:00 - never the D1
    # open stamp - is the first agreeing M15 bar
    first_agree = markup.loc[agreeing, TIMESTAMP_COLUMN].min()
    assert first_agree == pd.Timestamp("2026-01-07 00:00", tz="UTC")


def test_conflicting_trends_zero_the_bias_and_report_conflict() -> None:
    """(b) H1 up, H4 down, D1 up -> bias_dir = 0 everywhere, state = conflict."""
    ltf, frames = _scenario(h1=1, h4=-1, d1=1)

    markup = bias_frames(ltf, frames)

    directional = _directional(markup).all(axis=1)
    assert bool(directional.any())
    # no_trade (the BiasConfig default): a conflict never becomes a trade direction
    assert markup[BIAS_DIR_COLUMN].eq(0).all()
    assert markup.loc[directional, BIAS_STATE_COLUMN].eq("conflict").all()
    # H1 alone must not be able to raise a long bias (this is what mutation m2 breaks)
    assert not markup[BIAS_DIR_COLUMN].eq(1).any()


@pytest.mark.parametrize(
    ("h1", "h4"),
    ((1, 1), (1, -1)),
    ids=("2-0-0", "1-1-0"),
)
def test_trends_that_are_not_established_yet_are_undefined(h1: int, h4: int) -> None:
    """(c) D1 has no trend yet (NaN head and 0 after) -> state undefined, dir 0.

    The undefined verdict outranks a disagreement (SPEC_SMC.md §7.6 п.24): with
    ``h1 = 1, h4 = -1`` the two *defined* trends already conflict, yet every row whose
    D1 trend is missing reports ``undefined``, never ``conflict``.
    """
    ltf, frames = _scenario(h1, h4, 1, d1_start=D1_LATE_START, d1_bars=D1_LATE_BARS)

    markup = bias_frames(ltf, frames)

    d1 = markup[trend_column("D1")]
    assert bool(d1.isna().any())  # before the first D1 close: nothing is attached
    assert bool(d1.eq(0.0).any())  # D1 bars closed, but its automaton saw no break
    undefined = d1.isna() | d1.eq(0.0)
    assert markup.loc[undefined, BIAS_STATE_COLUMN].eq("undefined").all()
    assert markup.loc[undefined, BIAS_DIR_COLUMN].eq(0).all()

    if h1 == h4:
        # once D1's flipping bar closes (Jan 10 00:00 + 1 day) the same scenario agrees,
        # so the "undefined" rows above are a matter of timing and not a blanket zero
        agree = markup[BIAS_STATE_COLUMN].eq("agree_long")
        assert bool(agree.any())
        first_agree = markup.loc[agree, TIMESTAMP_COLUMN].min()
        assert first_agree == pd.Timestamp("2026-01-11 00:00", tz="UTC")
    else:
        # ... while defined-but-disagreeing trends do raise a conflict of their own:
        # conflict needs *all* trends, so a missing D1 never counts as one
        conflict = markup[BIAS_STATE_COLUMN].eq("conflict")
        assert bool(conflict.any())
        assert markup.loc[conflict, trend_column("D1")].ne(0.0).all()
        assert not markup[BIAS_DIR_COLUMN].eq(1).any()


def _with_flipped_bar(frame: pd.DataFrame, bar_index: int) -> pd.DataFrame:
    """Copy of ``frame`` whose bar ``bar_index`` closes far below the head's swing low."""
    out = frame.copy()
    for column, value in zip(("open", "high", "low", "close"), FLIP_ROW, strict=True):
        out.loc[bar_index, column] = value
    return out


TAMPER_BAR_INDICES = (H1_BARS - 1, H1_BARS // 2, TREND_FLIP_BAR)


@pytest.mark.parametrize("bar_index", TAMPER_BAR_INDICES, ids=("last", "middle", "break"))
def test_an_htf_bar_moves_the_bias_only_from_its_close_time(bar_index: int) -> None:
    """(d) cross-lookahead: an H1 bar that is still open leaks into no M15 bar."""
    ltf, frames = _scenario(1, 1, 1)
    base = bias_frames(ltf, frames)

    tampered_frames = dict(frames)
    tampered_frames["H1"] = _with_flipped_bar(frames["H1"], bar_index)
    tampered = bias_frames(ltf, tampered_frames)

    close_time = _close_time(frames, "H1", bar_index)
    before = tampered[TIMESTAMP_COLUMN] < close_time
    window = (tampered[TIMESTAMP_COLUMN] >= close_time) & (
        tampered[TIMESTAMP_COLUMN] < close_time + period_for("H1")
    )
    assert bool(before.any())  # the "untouched" side must exist for the test to mean anything
    changed = _changed_against(base, tampered)

    # the tamper is meaningful: that H1 bar is short from its own close_time on
    at_close = tampered[TIMESTAMP_COLUMN] == close_time
    assert tampered.loc[at_close, trend_column("H1")].eq(-1).all()
    # nothing before the HTF bar closed may move; the shared NaN head is compared too
    pd.testing.assert_frame_equal(
        tampered.loc[before].reset_index(drop=True),
        base.loc[before].reset_index(drop=True),
    )
    if bool(window.any()):
        # ... and the flip shows up first exactly inside that bar's own window
        assert bool(changed[window.to_numpy()].any())
        assert tampered.loc[changed, TIMESTAMP_COLUMN].min() == close_time
    else:
        # a bar closing after the whole LTF range can never become visible at all
        assert bool(before.all())
        assert not bool(changed.any())


def test_a_flipped_htf_close_changes_the_bias_from_that_close_time() -> None:
    """(e) the first M15 bar that reacts is the flipping HTF bar's ``close_time``."""
    ltf, frames = _scenario(1, 1, 1)
    base = bias_frames(ltf, frames)

    bar_index = H4_BARS // 2
    tampered_frames = dict(frames)
    tampered_frames["H4"] = _with_flipped_bar(frames["H4"], bar_index)
    tampered = bias_frames(ltf, tampered_frames)

    close_time = _close_time(frames, "H4", bar_index)
    changed = _changed_against(base, tampered)
    assert bool(changed.any())

    # the M15 bar one step before the close_time still sees the untampered H4 trend
    before = tampered[TIMESTAMP_COLUMN] == close_time - period_for("M15")
    assert tampered.loc[before, trend_column("H4")].eq(1).all()
    # the bar opening exactly at close_time already sees the flipped H4 trend, and it
    # is the first moved bar: nothing earlier reacts (later bars may, until H4 recovers)
    at_close = tampered[TIMESTAMP_COLUMN] == close_time
    assert tampered.loc[at_close, trend_column("H4")].eq(-1).all()
    assert not bool(changed[tampered[TIMESTAMP_COLUMN] < close_time].any())
    assert tampered.loc[changed, TIMESTAMP_COLUMN].min() == close_time
    # H1 and D1 stay long, so the flipped H4 turns the agreement into a conflict
    assert tampered.loc[at_close, BIAS_STATE_COLUMN].eq("conflict").all()


def test_reduced_risk_conflict_policy_is_not_implemented() -> None:
    """(f) ``reduced_risk`` is a deferred decision: the indicator refuses to guess."""
    ltf, frames = _scenario(1, -1, 1)

    with pytest.raises(NotImplementedError, match="reduced_risk"):
        bias_frames(ltf, frames, BiasConfig(on_conflict="reduced_risk"))


def test_bias_frames_needs_every_configured_timeframe() -> None:
    ltf, frames = _scenario(1, 1, 1)

    with pytest.raises(ValueError, match="missing"):
        bias_frames(ltf, {"H1": frames["H1"], "D1": frames["D1"]})


def test_bias_config_rejects_a_duplicate_timeframe() -> None:
    with pytest.raises(ValueError, match="unique"):
        BiasConfig(timeframes=("H1", "H1"))


def test_bias_skips_the_unclosed_ltf_tail() -> None:
    """The presumed still-forming M15 bar gets no bias row (live-edge protection)."""
    ltf, frames = _scenario(1, 1, 1)
    ltf[IS_CLOSED_COLUMN] = True
    ltf.loc[ltf.index[-1], IS_CLOSED_COLUMN] = False

    markup = bias_frames(ltf, frames)

    assert len(markup) == len(ltf) - 1
    assert markup[TIMESTAMP_COLUMN].max() == ltf[TIMESTAMP_COLUMN].iloc[-2]
    assert str(markup[BIAS_DIR_COLUMN].dtype) == "int8"
