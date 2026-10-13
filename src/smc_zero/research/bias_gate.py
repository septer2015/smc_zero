"""Research switch of Э17.z++: a *softer* direction gate, reached without touching the chain.

The direction gate of the entry chain (``bias_skip``, gate (4) of SPEC_SMC.md §7.8 п.35) compares
the side of an attempt with ``bias_dir``, and it lives in ``src/smc_zero/strategy/intents.py`` - a
protected file of Э17.z++.  The chain reads that direction as *data*, however: ``build_intents`` is
handed a bias frame, and the frame is the one place where a research run can answer "which way does
the closed HTF structure point at this bar" differently without editing the chain.

The module offers exactly one answer beside the shipped one:

* ``strict`` (the default) - the verdict of :func:`smc_zero.indicators.bias.classify_trends` under
  the agreement mode of the run, row for row the frame
  :meth:`smc_zero.optimizer.TapeMarks.bias_frame` hands the chain today.  A caller that names no
  mode gets this, so the gate is on by default;
* ``defined_majority`` - the same tally, but a timeframe *without* a trend (``NaN`` before its first
  closed bar, ``0`` while its automaton has seen no break) abstains instead of vetoing the row: the
  strict majority of the timeframes that do have a direction wins.

The softer rule only turns refusals into passes.  Every row ``strict`` defines keeps its direction
under ``defined_majority`` as well, and the two modes differ on the rows ``strict`` leaves at ``0``
alone (see :func:`defined_majority`).

What the switch deliberately cannot do is turn the gate off.  ``bias_dir`` is a property of the
*bar*, while the gate compares it with the side of an *attempt*, and one bar may carry attempts of
both sides (the sweep window of an upper level and of a lower one can overlap): no single frame can
let both through.  Switching the gate off is therefore a change of ``intents.py`` and not a research
frame - the audit of Э17.z++ records that as a block.

Neither mode looks at a future bar: both read one row of the trend matrix at a time, and that matrix
is already stitched onto the entry bars by ``close_time`` (rule 2).  Both are pure functions of their
input too - the frame is copied, never written through.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from smc_zero.config import AGREEMENTS, Agreement
from smc_zero.data_loader import TIMESTAMP_COLUMN
from smc_zero.indicators.bias import (
    BIAS_DIR_COLUMN,
    BIAS_STATE_COLUMN,
    classify_trends,
    trend_column,
)

#: The two modes of the switch: the shipped gate and the softer reading of the same trend matrix.
BIAS_GATE_MODES: tuple[str, ...] = ("strict", "defined_majority")
#: What a caller that names no mode gets: the gate as the project ships it.
DEFAULT_BIAS_GATE_MODE = "strict"


def defined_majority(trends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(bias_dir, bias_state)`` of a trend matrix where an undefined timeframe abstains.

    The tally is the one of :func:`~smc_zero.indicators.bias.classify_trends` - ``longs`` and
    ``shorts`` count the defined directions per row - and the single difference is the veto: a row is
    reported as ``agree_long`` / ``agree_short`` only when *every* timeframe is defined and agrees,
    and otherwise the strict majority of the defined timeframes decides.  A tie of the defined
    timeframes, and a row without a single defined one, stay ``0`` with the state ``conflict`` /
    ``undefined``: "no direction" is not a direction, exactly as it is in the shipped rule.

    ``trends`` is a ``(bars, timeframes)`` matrix: ``NaN`` marks "no HTF bar of this timeframe has
    closed yet" and ``0`` "this timeframe has no trend yet"; both abstain.
    """
    values = np.asarray(trends, dtype="float64")
    if values.ndim != 2:
        raise ValueError(f"trends must be a (bars, timeframes) matrix, got {values.shape}")
    defined = ~np.isnan(values) & (values != 0.0)
    total = values.shape[1]
    longs = (values == 1.0).sum(axis=1)
    shorts = (values == -1.0).sum(axis=1)
    agree_long = defined.all(axis=1) & (longs == total)
    agree_short = defined.all(axis=1) & (shorts == total)
    majority_long = ~agree_long & (longs > shorts)
    majority_short = ~agree_short & (shorts > longs)
    bias_dir = np.where(
        agree_long | majority_long, 1, np.where(agree_short | majority_short, -1, 0)
    ).astype("int8")
    bias_state = np.where(
        agree_long,
        "agree_long",
        np.where(
            agree_short,
            "agree_short",
            np.where(
                majority_long,
                "majority_long",
                np.where(
                    majority_short,
                    "majority_short",
                    np.where(longs + shorts > 0, "conflict", "undefined"),
                ),
            ),
        ),
    )
    return bias_dir, bias_state.astype(object)


def direction_frame(
    trends: pd.DataFrame,
    timeframes: Sequence[str],
    *,
    agreement: Agreement = "unanimous",
    mode: str = DEFAULT_BIAS_GATE_MODE,
) -> pd.DataFrame:
    """Return the bias frame of one gate mode: ``timestamp``, ``bias_dir`` and ``bias_state``.

    ``trends`` is the cached trend matrix of a run - the frame
    :func:`smc_zero.optimizer.build_tape_marks` stores in ``TapeMarks.trends``, i.e. the very frame
    :meth:`TapeMarks.bias_frame` re-classifies - and ``timeframes`` names the columns the verdict is
    asked of.  ``mode="strict"`` answers row for row what the chain is handed today (the tests pin it
    against ``TapeMarks.bias_frame``) and uses ``agreement``; ``mode="defined_majority"`` is the
    softer reading of :func:`defined_majority`, whose ``agreement`` plays no part because it *is* a
    majority rule.

    The result carries exactly the columns :func:`smc_zero.strategy.intents.build_intents` reads for
    its direction gate, and a ``ValueError`` names the missing column of a frame that cannot be one.
    """
    if mode not in BIAS_GATE_MODES:
        raise ValueError(f"mode must be one of {BIAS_GATE_MODES}, got {mode!r}")
    if mode == "strict" and agreement not in AGREEMENTS:
        raise ValueError(f"agreement must be one of {AGREEMENTS}, got {agreement!r}")
    columns = [trend_column(timeframe) for timeframe in timeframes]
    missing = [column for column in columns if column not in trends.columns]
    if missing:
        raise ValueError(f"the trend matrix has no column(s) {', '.join(missing)}")
    if TIMESTAMP_COLUMN not in trends.columns:
        raise ValueError(f"the trend matrix has no {TIMESTAMP_COLUMN!r} column")
    values = trends.loc[:, columns].to_numpy(dtype="float64")
    if mode == "strict":
        bias_dir, bias_state = classify_trends(values, agreement)
    else:
        bias_dir, bias_state = defined_majority(values)
    frame = trends.loc[:, [TIMESTAMP_COLUMN]].copy()
    frame[BIAS_DIR_COLUMN] = bias_dir
    frame[BIAS_STATE_COLUMN] = bias_state
    return frame
