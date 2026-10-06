"""Market structure: swings, break signal, trend automaton (SPEC_SMC.md, C4).

Semantics ported from the prod source ``core.py`` (semantics only, no code reuse):

* a swing is a *strict* N-bar fractal - ``high[i] > high[i +/- k]`` for every
  ``k <= N`` (prod's formula, core.py lines 339-343, is exactly the ``N = 1``
  case).  It is therefore known only at bar ``i + N``, once the right candles have
  closed, and every markup row carries that position in ``swing_*_known_at``
  (rule 2: no look-ahead);
* ``break_dir`` reproduces prod's ``choch_signal`` (core.py lines 347-365):
  ``+1`` while the close sits beyond the last known swing high, ``-1`` while it
  sits below the last known swing low, ``0`` in between.  A level is *not*
  consumed by a break, so the signal repeats while price stays beyond it (prod
  parity) - downstream code that needs "the first bar of the break" must dedupe
  itself;
* ``structure_event`` / ``trend`` add the automaton the constitution defines on
  top of the raw signal: a break *with* the trend is a BOS, a break *against* it
  is a CHoCH that flips the trend.  The very first break has no trend to agree
  with and is labelled ``bos`` (trend initialisation, agreed with the project
  owner); ``trend`` stays ``0`` until that first break.

Every function is pure (``DataFrame`` + :class:`~smc_zero.config.StructureConfig`
in, markup out), preserves the input index and never mutates the argument
(SPEC_SMC.md, section 0.7: prod's ``_PRECOMPUTE_CACHE`` is not ported).  All
markup positions are *positional* bar numbers; use ``df.index[position]`` for the
timestamp.  ``structure_event`` uses ``None`` for "no event".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smc_zero.config import (
    BreakEvent,
    StructureConfig,
    StructureLayerConfig,
    Timeframe,
)
from smc_zero.data_loader import (
    CLOSE_TIME_COLUMN,
    TIMESTAMP_COLUMN,
    align_htf_to_ltf,
    attach_close_time,
    drop_unclosed,
    period_for,
)
from smc_zero.indicators._markup import known_at
from smc_zero.indicators.impulse import displacement_gate


def _fractal_mask(values: np.ndarray, lookback: int, *, higher: bool) -> np.ndarray:
    """Return a mask of strict N-bar fractal extremes of ``values``.

    ``higher`` selects highs (fractal high) or lows (fractal low).  The mask has
    one ``True`` per extreme bar; the first and last ``lookback`` bars can never
    qualify, which is what makes the confirmation lag explicit.
    """
    n = values.size
    mask = np.zeros(n, dtype=bool)
    if n <= 2 * lookback:
        return mask
    centre = values[lookback : n - lookback]
    strict = np.ones(centre.shape, dtype=bool)
    for shift in range(1, lookback + 1):
        left = values[lookback - shift : n - lookback - shift]
        right = values[lookback + shift : n - lookback + shift]
        if higher:
            strict &= (centre > left) & (centre > right)
        else:
            strict &= (centre < left) & (centre < right)
    mask[lookback : n - lookback] = strict
    return mask


def swing_points(df: pd.DataFrame, cfg: StructureConfig | None = None) -> pd.DataFrame:
    """Return swing flags plus the position at which each swing becomes known.

    Columns (same index as ``df``): ``swing_high`` / ``swing_low`` (``bool``,
    ``True`` on the extreme bar itself), ``swing_high_price`` / ``swing_low_price``
    (``float``, NaN elsewhere, the extreme price) and ``swing_high_known_at`` /
    ``swing_low_known_at`` (``Int64``, ``i + swing_lookback``, NA elsewhere).

    With the default ``swing_lookback = 1`` a swing at bar ``k`` is visible from
    bar ``k + 1`` on, i.e. exactly prod's ``is_swing_high[i - 1]`` read
    (core.py lines 355-360).
    """
    config = StructureConfig() if cfg is None else cfg
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    lookback = config.swing_lookback
    swing_high = _fractal_mask(high, lookback, higher=True)
    swing_low = _fractal_mask(low, lookback, higher=False)
    return pd.DataFrame(
        {
            "swing_high": swing_high,
            "swing_low": swing_low,
            "swing_high_price": np.where(swing_high, high, np.nan),
            "swing_low_price": np.where(swing_low, low, np.nan),
            "swing_high_known_at": known_at(swing_high, lookback, df.index),
            "swing_low_known_at": known_at(swing_low, lookback, df.index),
        },
        index=df.index,
    )



def known_swing_levels(df: pd.DataFrame, cfg: StructureConfig | None = None) -> pd.DataFrame:
    """Return the last swing levels *as they are visible at each bar*.

    ``last_swing_high`` / ``last_swing_low`` (same index as ``df``) hold the extreme
    of the most recent swing whose confirmation bar is not later than the current
    one: a swing at bar ``k`` enters at bar ``k + swing_lookback`` (``shift`` +
    ``ffill``).  Before the first confirmed swing the value is NaN, and NaN never
    compares as a break.
    """
    config = StructureConfig() if cfg is None else cfg
    points = swing_points(df, config)
    lookback = config.swing_lookback
    return pd.DataFrame(
        {
            "last_swing_high": points["swing_high_price"].shift(lookback).ffill(),
            "last_swing_low": points["swing_low_price"].shift(lookback).ffill(),
        },
        index=df.index,
    )


def _run_trend_automaton(break_dir: np.ndarray) -> tuple[list[BreakEvent | None], np.ndarray]:
    """Return the per-bar BOS/CHoCH label and the trend state *after* each bar.

    The automaton is inherently sequential (each bar depends on the previous
    trend), so this is one of the few loops in the indicator layer; it is O(n) over
    int8 input.  A break with the trend (or the initialising first break) is a
    ``bos``; a break against it is a ``choch`` and flips the trend.
    """
    n = break_dir.size
    events: list[BreakEvent | None] = [None] * n
    trends = np.zeros(n, dtype=np.int8)
    trend = 0
    for i in range(n):
        direction = int(break_dir[i])
        if direction != 0:
            events[i] = "choch" if trend != 0 and direction != trend else "bos"
            trend = direction
        trends[i] = trend
    return events, trends


def structure_breaks(df: pd.DataFrame, cfg: StructureConfig | None = None) -> pd.DataFrame:
    """Return the prod break signal, the BOS/CHoCH label and the trend state.

    Columns (same index as ``df``):

    * ``break_dir`` - ``int8`` prod parity (core.py lines 362-365): ``+1`` when
      the close (or the high when ``confirmation="wick"``) is beyond the last
      *known* swing high, ``-1`` for the mirrored case, ``0`` otherwise.  The
      upward probe wins when both fire, exactly like prod's ``if / elif``.
    * ``structure_event`` - ``"bos"`` / ``"choch"`` / ``None`` for that same bar.
    * ``trend`` - ``int8`` trend state after the bar: ``+1``, ``-1`` or ``0`` while
      no break has happened yet.
    * ``break_level`` - the swing level the break was measured against
      (``last_swing_high`` for ``+1``, ``last_swing_low`` for ``-1``), NaN without a
      break.  The impulse gate (Э1'.5) and the strategy read the level from here
      instead of re-deriving the structure.

    The break is always measured against a trailing swing, never against the swing
    of the current bar, so substituting a future candle cannot change past rows.
    """
    config = StructureConfig() if cfg is None else cfg
    levels = known_swing_levels(df, config)
    last_high = levels["last_swing_high"].to_numpy(dtype=float)
    last_low = levels["last_swing_low"].to_numpy(dtype=float)
    if config.confirmation == "close":
        probe_up = df["close"].to_numpy(dtype=float)
        probe_down = probe_up
    else:  # "wick": a touch of the level already counts as a break
        probe_up = df["high"].to_numpy(dtype=float)
        probe_down = df["low"].to_numpy(dtype=float)
    up = last_high < probe_up  # NaN levels compare False -> no break before a swing exists
    down = last_low > probe_down
    break_dir = np.where(up, 1, np.where(down, -1, 0)).astype(np.int8)
    break_level = np.where(up, last_high, np.where(down, last_low, np.nan))
    events, trends = _run_trend_automaton(break_dir)
    return pd.DataFrame(
        {
            "break_dir": break_dir,
            "structure_event": pd.Series(events, index=df.index, dtype="object"),
            "trend": trends,
            "break_level": break_level,
        },
        index=df.index,
    )


#: The columns the structure layer carries from a working frame onto the entry frame (§7.20).
STRUCTURE_LAYER_COLUMNS: tuple[str, ...] = ("break_dir", "disp_ok", "disp_known_at")
#: The suffix those columns wear while the two grids are joined by :func:`align_htf_to_ltf`.
STRUCTURE_SUFFIX = "_struct"


def _stamp_ns(values: pd.Series) -> np.ndarray:
    """Return a stamp column as UTC-naive ``datetime64[ns]`` - the comparison scale of the join."""
    stamps = (
        values.dt.tz_convert("UTC").dt.tz_localize(None)
        if isinstance(values.dtype, pd.DatetimeTZDtype)
        else values
    )
    return stamps.to_numpy(dtype="datetime64[ns]")


def _empty_structure_layer() -> pd.DataFrame:
    """Return an empty layer frame carrying the three columns and their documented dtypes."""
    return pd.DataFrame(
        {
            TIMESTAMP_COLUMN: pd.Series(dtype="datetime64[ns, UTC]"),
            "break_dir": pd.Series(dtype="int8"),
            "disp_ok": pd.Series(dtype=bool),
            "disp_known_at": pd.Series(dtype="Int64"),
        }
    )


def _entry_frame(ltf: pd.DataFrame, timeframe: Timeframe | None) -> pd.DataFrame:
    """Return the closed entry bars as a positional frame, checking the named grid when given.

    Rule 2b: the presumed still-forming tail bar is dropped, exactly as the entry chain does.  A
    ``timeframe`` names the grid the frame has to sit on - a working frame joined onto a foreign
    grid would mis-date every ``disp_known_at`` - and ``None`` leaves that check to the caller.
    """
    if TIMESTAMP_COLUMN not in ltf.columns:
        raise ValueError("structure_layer expects a 'timestamp' column (loader schema)")
    missing = [column for column in ("high", "low", "close") if column not in ltf.columns]
    if missing:
        raise ValueError(f"structure_layer needs the {missing} column(s) (loader schema)")
    bars = drop_unclosed(ltf).reset_index(drop=True)
    if timeframe is None or len(bars) < 2:
        return bars
    expected = period_for(timeframe)
    stamps = bars[TIMESTAMP_COLUMN]
    if isinstance(stamps.dtype, pd.DatetimeTZDtype):
        stamps = stamps.dt.tz_convert("UTC")
    step = stamps.diff().dropna().min()
    if step != expected:
        raise ValueError(
            f"structure_layer expects the {timeframe} entry frame ({expected} bars), "
            f"got a {step} grid"
        )
    return bars


def _layer_of_one_frame(frame: pd.DataFrame, cfg: StructureLayerConfig) -> pd.DataFrame:
    """Return the three layer columns of one frame: the break, the impulse and its known bar.

    The columns are the positional reading of :func:`structure_breaks` and
    :func:`~smc_zero.indicators.impulse.displacement_gate` - exactly what
    :func:`smc_zero.strategy.intents.build_intents` computes for itself when it owns the frame.
    """
    missing = [column for column in ("high", "low", "close") if column not in frame.columns]
    if missing:
        raise ValueError(f"structure_layer needs the {missing} column(s) (loader schema)")
    breaks = structure_breaks(frame, cfg.structure)
    impulse = displacement_gate(frame, breaks, cfg.displacement)
    return pd.DataFrame(
        {
            TIMESTAMP_COLUMN: frame[TIMESTAMP_COLUMN],
            "break_dir": breaks["break_dir"].to_numpy(dtype="int8"),
            "disp_ok": impulse["disp_ok"].to_numpy(dtype=bool),
            "disp_known_at": impulse["disp_known_at"],
        },
        index=frame.index,
    )


def _tidy_layer(joined: pd.DataFrame) -> pd.DataFrame:
    """Return the joined frame as the layer itself: three columns, documented dtypes, no index."""
    columns = {f"{column}{STRUCTURE_SUFFIX}": column for column in STRUCTURE_LAYER_COLUMNS}
    out = joined[[TIMESTAMP_COLUMN, *columns]].rename(columns=columns).copy()
    # an entry bar preceding the first close of the working frame has no break and no impulse yet
    out["break_dir"] = out["break_dir"].fillna(0).astype("int8")
    out["disp_ok"] = out["disp_ok"].fillna(False).astype(bool)
    out["disp_known_at"] = out["disp_known_at"].astype("Int64")
    return out.reset_index(drop=True)


def structure_layer(
    ltf: pd.DataFrame,
    structure: pd.DataFrame | None = None,
    *,
    ltf_timeframe: Timeframe | None = None,
    structure_timeframe: Timeframe | None = None,
    cfg: StructureLayerConfig | None = None,
) -> pd.DataFrame:
    """Stitch the working structure onto the entry bars, visible only once its bar has closed.

    ``ltf`` is the entry frame and ``structure`` the working frame whose swing / BOS / CHoCH markup
    is read (the M5 tape and the M15 tape of the H4 -> M15 -> M5 hierarchy of §7.20).  The result has
    the rows of the entry frame and three columns:

    * ``break_dir`` - the sign of the break as it is visible at that entry bar (``0`` before any
      working bar has closed);
    * ``disp_ok`` - the impulse verdict of the confirming working bar;
    * ``disp_known_at`` - the *entry* bar at which that impulse becomes knowable, i.e. the first
      entry bar opening at or after the close of the bar ``disp_known_at`` of the working frame.
      ``NA`` when that instant lies beyond the frame, so the gate reads "never visible" instead of a
      number it could compare - which is what makes a working bar visible only from the entry bar
      after its close (rule 2).

    ``structure=None`` keeps the v1 reading: the structure is measured on the entry frame itself and
    the positions of ``disp_known_at`` are its own, bit for bit.  ``structure_timeframe`` is
    mandatory as soon as a working frame is given - its bar period is what dates the join - and
    ``ltf_timeframe``, when passed, is checked against the grid of the entry frame.
    """
    config = StructureLayerConfig() if cfg is None else cfg
    bars = _entry_frame(ltf, ltf_timeframe)
    if structure is None:
        return _layer_of_one_frame(bars, config)
    if structure_timeframe is None:
        raise ValueError(
            "structure_layer needs structure_timeframe when a working frame is given: "
            "the bar period of that frame is what dates the visibility of the join"
        )
    period = period_for(structure_timeframe)
    working = drop_unclosed(structure).reset_index(drop=True)
    if bars.empty or working.empty:
        return _empty_structure_layer()

    layer = _layer_of_one_frame(working, config)
    known = layer["disp_known_at"].to_numpy(dtype="float64", na_value=np.nan)
    visible = np.full(known.shape, np.nan, dtype="float64")
    finite = np.isfinite(known)
    if finite.any():
        closes = attach_close_time(working, period)[CLOSE_TIME_COLUMN]
        entry_ns = _stamp_ns(bars[TIMESTAMP_COLUMN])
        found = np.searchsorted(entry_ns, _stamp_ns(closes)[finite], side="left")
        # an impulse knowable only after the last entry bar of the frame is never visible in it
        visible[finite] = np.where(found >= entry_ns.size, np.nan, found)

    source = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: working[TIMESTAMP_COLUMN],
            "break_dir": layer["break_dir"].to_numpy(dtype="int8"),
            "disp_ok": layer["disp_ok"].to_numpy(dtype=bool),
            "disp_known_at": visible,
        }
    )
    joined = align_htf_to_ltf(
        bars,
        source,
        htf_period=period,
        extra_columns=STRUCTURE_LAYER_COLUMNS,
        suffixes=("", STRUCTURE_SUFFIX),
    )
    return _tidy_layer(joined)

