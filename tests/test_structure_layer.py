"""Tests for :func:`smc_zero.indicators.structure.structure_layer` on synthetic tapes (Э12).

The layer is the joint of the second hierarchy (H4 -> M15 -> M5, SPEC_SMC.md §7.20): the working
frame of the structure (M15) is joined onto the entry frame (M5) by ``close_time``, so an entry bar
can only ever see a working bar that has already closed.  Everything here is synthetic, positional
and deterministic (constitution rules 1 and 3).

The mutations this file is written to catch, and the test each one must break:

* m1 "join on the open_time of the working frame instead of its close_time" - the entry bar that
  opens *with* the working bar already sees the working bar's break; breaks
  :func:`test_the_working_frame_reaches_the_entry_only_after_its_bar_closes`;
* m2 "read ``disp_known_at`` without the period of the working frame" - the known entry bar lands
  one bar too early; breaks :func:`test_the_known_position_of_the_impulse_is_read_in_entry_bars`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smc_zero.config import DisplacementConfig, StructureLayerConfig
from smc_zero.data_loader import TIMESTAMP_COLUMN
from smc_zero.indicators.impulse import displacement_gate
from smc_zero.indicators.structure import (
    STRUCTURE_LAYER_COLUMNS,
    structure_breaks,
    structure_layer,
)

DAY = "2026-06-08"
#: The fixture tape is short, so the impulse gate is given an ATR period it can actually measure
#: (the shipped 14 needs 14 bars and would leave ``disp_atr`` NaN on this eight bar tape).  The
#: layer's own pass-through of the impulse columns is what these tests watch, not the ATR length.
LAYER = StructureLayerConfig(displacement=DisplacementConfig(atr_period=1))
#: Eight M15 bars: a swing high at 00:15 (known from 00:30), a swing low at 01:00 (known from
#: 01:15), an upward break at 01:30 and a downward one at 01:45.
M15_ROWS = [
    (100.5, 100.0, 100.2),
    (100.6, 100.1, 100.5),
    (100.4, 99.9, 100.2),
    (100.3, 99.8, 100.1),
    (100.2, 99.7, 99.9),
    (100.3, 99.8, 100.1),
    (100.9, 100.0, 100.8),
    (100.4, 99.4, 99.5),
]
#: The entry tape: two hours of quiet five minute bars, so nothing but the join is under test.
M5_ROWS = [(100.1, 99.9, 100.0)] * 24
#: The entry bar that sees the 01:30 M15 break: 01:45 is index 21 of the 24 five minute bars.
ENTRY_BAR_OF_THE_BREAK = 21


def _frame(rows: list[tuple[float, float, float]], freq: str) -> pd.DataFrame:
    """One tape with ``(high, low, close)`` rows, ``open`` equal to ``close`` and a UTC grid."""
    stamps = pd.date_range(f"{DAY} 00:00", periods=len(rows), freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            TIMESTAMP_COLUMN: stamps,
            "open": [row[2] for row in rows],
            "high": [row[0] for row in rows],
            "low": [row[1] for row in rows],
            "close": [row[2] for row in rows],
            "volume": 1,
        }
    )


def _stamp(hhmm: str) -> pd.Timestamp:
    """A UTC stamp of the fixture day."""
    return pd.Timestamp(f"{DAY} {hhmm}", tz="UTC")


def test_the_default_layer_is_the_structure_of_the_entry_frame() -> None:
    """``structure=None`` measures the entry frame itself: the v1 reading, bit for bit."""
    m15 = _frame(M15_ROWS, freq="15min")

    layer = structure_layer(m15)

    breaks = structure_breaks(m15)
    impulse = displacement_gate(m15, breaks)
    assert list(layer.columns) == [TIMESTAMP_COLUMN, *STRUCTURE_LAYER_COLUMNS]
    assert layer["break_dir"].tolist() == breaks["break_dir"].tolist()
    assert layer["break_dir"].tolist() == [0, 0, 0, 0, 0, 0, 1, -1]
    assert layer["disp_ok"].tolist() == impulse["disp_ok"].tolist()
    assert layer["disp_known_at"].tolist() == impulse["disp_known_at"].tolist()


def test_the_working_frame_reaches_the_entry_only_after_its_bar_closes() -> None:
    """The M15 break of 01:30 is visible on M5 from 01:45 on, and not one bar earlier (m1)."""
    m15 = _frame(M15_ROWS, freq="15min")
    m5 = _frame(M5_ROWS, freq="5min")

    layer = structure_layer(m5, m15, ltf_timeframe="M5", structure_timeframe="M15")

    visible = dict(zip(layer[TIMESTAMP_COLUMN], layer["break_dir"], strict=True))
    assert visible[_stamp("01:30")] == 0  # the M15 bar of 01:30 closes at 01:45
    assert visible[_stamp("01:40")] == 0
    assert visible[_stamp("01:45")] == 1
    # the downward break of the 01:45 M15 bar closes at 02:00, past the last entry bar of the frame
    assert (layer["break_dir"] == -1).sum() == 0


def test_the_known_position_of_the_impulse_is_read_in_entry_bars() -> None:
    """``disp_known_at`` is an entry bar, not a working one: the break's outcome lands on 01:45 (m2)."""
    m15 = _frame(M15_ROWS, freq="15min")
    m5 = _frame(M5_ROWS, freq="5min")

    layer = structure_layer(m5, m15, structure_timeframe="M15", cfg=LAYER)

    known = dict(zip(layer[TIMESTAMP_COLUMN], layer["disp_known_at"], strict=True))
    assert pd.isna(known[_stamp("01:30")])  # no confirming working bar is visible yet
    assert known[_stamp("01:45")] == ENTRY_BAR_OF_THE_BREAK
    assert bool(layer.loc[layer[TIMESTAMP_COLUMN] == _stamp("01:45"), "disp_ok"].iloc[0])
    # the dtype contract the chain reads: NA must stay NA and never become a number
    assert str(layer["disp_known_at"].dtype) == "Int64"
    assert str(layer["break_dir"].dtype) == "int8"
    assert str(layer["disp_ok"].dtype) == "bool"


def test_a_tamper_of_a_working_bar_touches_the_entry_only_after_its_close() -> None:
    """Rewriting the 01:30 M15 bar changes the layer from 01:45 on and not before it (rule 2)."""
    m5 = _frame(M5_ROWS, freq="5min")
    base = structure_layer(
        m5, _frame(M15_ROWS, freq="15min"), structure_timeframe="M15", cfg=LAYER
    )
    # the 01:30 M15 bar turns into a downward break instead of an upward one
    tampered = [*M15_ROWS[:6], (100.9, 99.0, 99.4), M15_ROWS[7]]
    late = structure_layer(
        m5, _frame(tampered, freq="15min"), structure_timeframe="M15", cfg=LAYER
    )

    pd.testing.assert_frame_equal(
        base.iloc[:ENTRY_BAR_OF_THE_BREAK].reset_index(drop=True),
        late.iloc[:ENTRY_BAR_OF_THE_BREAK].reset_index(drop=True),
    )
    assert late["break_dir"].iloc[ENTRY_BAR_OF_THE_BREAK] == -1
    assert base["break_dir"].iloc[ENTRY_BAR_OF_THE_BREAK] == 1


def test_the_layer_refuses_a_missing_period_a_foreign_grid_and_a_schema_gap() -> None:
    """Every way of wiring the join wrongly is a ``ValueError`` naming what is missing."""
    m15 = _frame(M15_ROWS, freq="15min")
    m5 = _frame(M5_ROWS, freq="5min")

    with pytest.raises(ValueError, match="structure_timeframe"):
        structure_layer(m5, m15)
    with pytest.raises(ValueError, match="expects the M5 entry frame"):
        structure_layer(m15, m15, ltf_timeframe="M5", structure_timeframe="M15")
    with pytest.raises(ValueError, match="column"):
        structure_layer(m5.drop(columns=["high"]), m15, structure_timeframe="M15")
    with pytest.raises(ValueError, match="unsupported timeframe"):
        structure_layer(m5, m15, structure_timeframe="M1")


def test_an_empty_entry_frame_yields_the_empty_layer() -> None:
    """A frame without closed bars is an empty layer with the documented columns, not a crash."""
    m15 = _frame(M15_ROWS, freq="15min")
    empty = _frame(M5_ROWS, freq="5min").iloc[:0]

    layer = structure_layer(empty, m15, structure_timeframe="M15")

    assert layer.empty
    assert list(layer.columns) == [TIMESTAMP_COLUMN, *STRUCTURE_LAYER_COLUMNS]
    assert str(layer["disp_known_at"].dtype) == "Int64"


def test_the_layer_keeps_the_entry_rows_it_was_given() -> None:
    """The join is a left join on the entry frame: one row per entry bar, in the entry order."""
    m15 = _frame(M15_ROWS, freq="15min")
    m5 = _frame(M5_ROWS, freq="5min")

    layer = structure_layer(m5, m15, structure_timeframe="M15")

    assert len(layer) == len(m5)
    assert np.array_equal(layer[TIMESTAMP_COLUMN].to_numpy(), m5[TIMESTAMP_COLUMN].to_numpy())
