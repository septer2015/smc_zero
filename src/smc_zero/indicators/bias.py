"""HTF bias markup: H1 / H4 / D1 trends stitched onto the M15 bars (SPEC_SMC.md, C5).

The module is a pure function layer: it reads OHLCV frames, never mutates them and
carries no state.  Two rules shape everything here.

* **The trend comes from the structure automaton on closed HTF bars.**  Every
  configured timeframe is passed through
  :func:`smc_zero.indicators.structure.structure_breaks` after its presumed
  still-forming tail bar has been dropped, so an open HTF bar can neither create nor
  flip a trend.
* **An HTF trend is visible only from its ``close_time``.**  The stitching is
  :func:`smc_zero.data_loader.align_htf_to_ltf` with the trend attached as an extra
  column, i.e. the same ``merge_asof(direction="backward")`` on ``close_time`` that
  the rest of the pipeline uses - never the HTF open stamp.  Before the first HTF bar
  closes the trend is ``NaN``.

``bias_dir`` is the direction the configured timeframes agree on: ``+1`` when they are
long, ``-1`` when they are short and ``0`` otherwise.  ``bias_state`` spells the reason
out - ``agree_long`` / ``agree_short`` when *every* timeframe points that way,
``majority_long`` / ``majority_short`` when only a strict majority does (reachable under
``agreement="majority"`` only), ``conflict`` when every timeframe has a direction but
they do not agree, and ``undefined`` when at least one timeframe has no trend yet
(``NaN`` before its first closed bar, or ``0`` while its automaton has not seen a
break).  A trend of ``0`` counts as undefined on purpose: "no direction" is not a
direction.

Unanimity is the *owner ruling* and the default (SPEC_SMC.md §7.6 п.24): a 2-1 split is
a conflict, not a majority direction, and any undefined trend outranks a conflict.  An
earlier spec wording said "majority agreement" (C5, §2 п.1); it was a paraphrase of the
owner's point 1 and is superseded.  The majority reading survives as the A/B parameter
``BiasConfig.agreement`` of Э7' (§7.11 п.70), which an optimization trial may suggest:
the two modes differ in nothing but the verdict on a 2-1 split, and the state keeps that
difference visible.  ``reduced_risk`` (which may later turn a 2-1 split into a smaller
trade) stays deferred - see §7.6 п.25.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from smc_zero.config import AGREEMENTS, Agreement, BiasConfig, StructureConfig
from smc_zero.data_loader import TIMESTAMP_COLUMN, align_htf_to_ltf, drop_unclosed
from smc_zero.indicators.structure import structure_breaks

BIAS_DIR_COLUMN = "bias_dir"
BIAS_STATE_COLUMN = "bias_state"
TREND_COLUMN = "trend"


def trend_column(timeframe: str) -> str:
    """Name of the attached trend column of a timeframe (``"H1"`` -> ``"trend_h1"``)."""
    return f"{TREND_COLUMN}_{timeframe.lower()}"


def _trend_markup(
    frame: pd.DataFrame,
    timeframe: str,
    structure_cfg: StructureConfig,
) -> pd.DataFrame:
    """Return ``timestamp`` and ``trend`` of one HTF frame, closed bars only."""
    if TIMESTAMP_COLUMN not in frame.columns:
        raise ValueError(f"bias_frames expects a 'timestamp' column in the {timeframe} frame")
    closed = drop_unclosed(frame)
    breaks = structure_breaks(closed, structure_cfg)
    return pd.DataFrame(
        {
            TIMESTAMP_COLUMN: closed[TIMESTAMP_COLUMN],
            TREND_COLUMN: breaks[TREND_COLUMN],
        }
    )


def _stitch_trends(
    ltf: pd.DataFrame,
    htf_frames: Mapping[str, pd.DataFrame],
    config: BiasConfig,
) -> pd.DataFrame:
    """Attach ``trend_<tf>`` of every configured HTF frame to the LTF bars."""
    missing = [timeframe for timeframe in config.timeframes if timeframe not in htf_frames]
    if missing:
        raise ValueError(f"htf_frames is missing {missing}; got {sorted(htf_frames)}")
    aligned = ltf
    for timeframe in config.timeframes:
        markup = _trend_markup(htf_frames[timeframe], timeframe, config.structure)
        aligned = align_htf_to_ltf(
            aligned,
            markup,
            htf_timeframe=timeframe,
            suffixes=("", f"_{timeframe.lower()}"),
            extra_columns=(TREND_COLUMN,),
            drop_unclosed=True,
        )
    return aligned


def classify_trends(
    trends: np.ndarray,
    agreement: Agreement = "unanimous",
) -> tuple[np.ndarray, np.ndarray]:
    """Turn per-timeframe trend values into ``bias_dir`` and ``bias_state``.

    ``trends`` is a ``(bars, timeframes)`` matrix where ``NaN`` marks "no HTF bar of
    this timeframe has closed yet" and ``0`` marks "this timeframe has no trend yet".
    Both count as undefined, so only defined directions are ever counted.

    ``agreement`` is the pass mark of a row.  ``"unanimous"`` wants every timeframe to
    point the same way (the v1 ruling); ``"majority"`` wants a *strict* majority, i.e.
    strictly more timeframes long than short or vice versa - an even split stays a
    conflict.  A row that passes unanimously keeps the strict verdict in both modes
    (``agree_long`` / ``agree_short``); a row that only passes the majority is reported
    as ``majority_long`` / ``majority_short``, so a consumer can tell the two apart.  A
    row of defined but tied trends is ``conflict`` (a 2-1 split under ``"unanimous"``),
    and a row with a single undefined trend is ``undefined`` - the undefined verdict
    outranks a conflict, exactly as it does in the markup.

    Rows of an empty matrix come back empty; the result is a tuple of the ``int8``
    ``bias_dir`` and its ``object``-dtype ``bias_state``, i.e. what
    :func:`bias_frames` writes into the markup.
    """
    values = np.asarray(trends, dtype="float64")
    if values.ndim != 2:
        raise ValueError(f"trends must be a (bars, timeframes) matrix, got {values.shape}")
    if agreement not in AGREEMENTS:
        raise ValueError(f"agreement must be one of {AGREEMENTS}, got {agreement!r}")

    defined = ~np.isnan(values) & (values != 0.0)
    all_defined = defined.all(axis=1)
    total = values.shape[1]
    longs = (values == 1.0).sum(axis=1)
    shorts = (values == -1.0).sum(axis=1)
    agree_long = all_defined & (longs == total)
    agree_short = all_defined & (shorts == total)
    empty = np.zeros(values.shape[0], dtype=bool)
    if agreement == "majority":
        majority_long = all_defined & ~agree_long & (longs > shorts)
        majority_short = all_defined & ~agree_short & (shorts > longs)
    else:
        majority_long = majority_short = empty

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
                    np.where(all_defined, "conflict", "undefined"),
                ),
            ),
        ),
    )
    return bias_dir, bias_state.astype(object)


def bias_frames(
    ltf: pd.DataFrame,
    htf_frames: Mapping[str, pd.DataFrame],
    cfg: BiasConfig | None = None,
) -> pd.DataFrame:
    """Return the M15 bias markup driven by the closed H1 / H4 / D1 trends.

    Parameters
    ----------
    ltf:
        M15 frame with a ``timestamp`` column (loader schema).  Rows flagged as
        unclosed are dropped, exactly like everywhere else in the pipeline.
    htf_frames:
        Mapping of timeframe label to its frame, e.g.
        ``{"H1": h1_df, "H4": h4_df, "D1": d1_df}``; every label listed in
        ``cfg.timeframes`` must be present (extra labels are ignored).
    cfg:
        :class:`smc_zero.config.BiasConfig`; defaults to the C5 hierarchy
        H1 + H4 + D1 with ``on_conflict="no_trade"``.

    Returns
    -------
    DataFrame with one row per retained M15 bar, indexed like
    :func:`smc_zero.data_loader.align_htf_to_ltf` (rows in ``open_time`` order):

    * ``timestamp`` - the M15 bar's open time (UTC), the audit key of the row;
    * ``trend_h1`` / ``trend_h4`` / ``trend_d1`` - the trend of the latest HTF bar
      closed at or before that M15 bar's open time, ``NaN`` before the first close and
      ``0`` while that timeframe's automaton has not seen a break;
    * ``bias_dir`` - ``int8``, ``+1`` / ``-1`` when the trends agree under
      ``cfg.agreement`` (unanimity by default: the owner ruling of SPEC_SMC.md §7.6
      п.24 makes a 2-1 split a conflict), else ``0``;
    * ``bias_state`` - ``"agree_long"`` / ``"agree_short"`` / ``"majority_long"`` /
      ``"majority_short"`` / ``"conflict"`` / ``"undefined"`` (see the module docstring).

    ``cfg.on_conflict = "reduced_risk"`` is a deferred decision and raises
    ``NotImplementedError``: v1 does not know what "reduced" means (SPEC_SMC.md §7.6).
    """
    config = BiasConfig() if cfg is None else cfg
    if config.on_conflict == "reduced_risk":
        raise NotImplementedError(
            "BiasConfig(on_conflict='reduced_risk') is not implemented in v1: what "
            "'reduced' means is still an open decision (SPEC_SMC.md §7.6). Use "
            "'no_trade', which forces bias_dir = 0 on a conflict instead."
        )
    aligned = _stitch_trends(ltf, htf_frames, config)
    columns = [trend_column(timeframe) for timeframe in config.timeframes]
    markup = aligned.loc[:, [TIMESTAMP_COLUMN, *columns]].copy()
    bias_dir, bias_state = classify_trends(
        markup.loc[:, columns].to_numpy(dtype="float64"), config.agreement
    )
    markup[BIAS_DIR_COLUMN] = bias_dir
    markup[BIAS_STATE_COLUMN] = bias_state
    return markup
