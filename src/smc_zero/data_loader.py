"""CSV loading, bar closing and multi-timeframe stitching.

Reconnaissance of ``./data`` fixed the on-disk schema for every timeframe
(M15 / H1 / D1)::

    datetime,open,high,low,close,volume

Findings that drive this module:

* ``datetime`` is an ``open_time`` stamp - the label sits at the *start* of the
  interval (D1 at ``00:00:00``, M15 on the ``:00/:15/:30/:45`` grid, H1 on full
  hours), hence ``close_time = timestamp + period`` is built explicitly by
  :func:`attach_close_time`.
* the stamps are UTC-naive in the files, so they are *localized* to UTC - never
  shifted with ``tz_convert``; only relative alignment matters for the
  HTF -> LTF stitching.
* the last bar of every timeframe may still have been forming when the export
  was taken, hence :func:`mark_closed` and :func:`drop_unclosed`.
* pandas 3 infers coarse resolutions from strings (``datetime64[us, UTC]``), so
  every ``timestamp``/``close_time`` leaving this module is explicitly forced to
  ``datetime64[ns, UTC]`` - one time dtype for the whole pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from smc_zero.config import Timeframe

OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")
TIMESTAMP_COLUMN = "timestamp"
CLOSE_TIME_COLUMN = "close_time"
IS_CLOSED_COLUMN = "is_closed"
SOURCE_TIME_COLUMN = "datetime"

# Bar length for every supported timeframe label.
TIMEFRAME_PERIODS: dict[str, pd.Timedelta] = {
    "M15": pd.Timedelta(minutes=15),
    "H1": pd.Timedelta(hours=1),
    "D1": pd.Timedelta(days=1),
}

# Explicit resampling convention: buckets are labelled by their *open* and are
# left-closed, so a bucket labelled 15:00 covers [15:00, 16:00).
RESAMPLE_LABEL = "left"
RESAMPLE_CLOSED = "left"
RESAMPLE_ORIGIN = "start_day"

_AGGREGATION: dict[str, str] = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}

_LTF_KEY_COLUMN = "_smc_ltf_key"


def period_for(timeframe: str) -> pd.Timedelta:
    """Return the bar duration of a timeframe label such as ``"M15"``."""
    try:
        return TIMEFRAME_PERIODS[timeframe.upper()]
    except KeyError as exc:
        supported = ", ".join(sorted(TIMEFRAME_PERIODS))
        raise ValueError(f"unsupported timeframe {timeframe!r}; expected one of {supported}") from exc


def _coerce_period(period: pd.Timedelta | str) -> pd.Timedelta:
    """Accept a timeframe label or any ``pd.Timedelta``-compatible value."""
    if isinstance(period, str) and period.upper() in TIMEFRAME_PERIODS:
        delta = TIMEFRAME_PERIODS[period.upper()]
    else:
        delta = pd.Timedelta(period)
    if delta <= pd.Timedelta(0):
        raise ValueError(f"period must be positive, got {period!r}")
    return delta


def _to_utc(values: pd.Series) -> pd.Series:
    """Return ``values`` as a UTC-aware ``datetime64[ns, UTC]`` series.

    Naive input is localized (never shifted); aware input is converted.
    """
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        stamps = values.dt.tz_convert("UTC")
    elif pd.api.types.is_datetime64_any_dtype(values.dtype):
        stamps = values.dt.tz_localize("UTC")
    else:
        try:
            stamps = pd.to_datetime(values, format="%Y-%m-%d %H:%M:%S")
        except (TypeError, ValueError):
            stamps = pd.to_datetime(values)
        if stamps.dt.tz is None:
            stamps = stamps.dt.tz_localize("UTC")
        else:
            stamps = stamps.dt.tz_convert("UTC")
    return stamps.astype("datetime64[ns, UTC]")


def _as_utc_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    """Normalise a scalar timestamp to UTC (localize naive, convert aware)."""
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def _normalise(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Rename the source time column and coerce every column to its dtype."""
    data = frame.rename(columns={SOURCE_TIME_COLUMN: TIMESTAMP_COLUMN})
    missing = [column for column in (TIMESTAMP_COLUMN, *OHLCV_COLUMNS) if column not in data.columns]
    if missing:
        raise ValueError(f"{source}: missing required column(s) {missing}")

    normalised = pd.DataFrame(index=data.index)
    normalised[TIMESTAMP_COLUMN] = _to_utc(data[TIMESTAMP_COLUMN])
    for column in PRICE_COLUMNS:
        normalised[column] = pd.to_numeric(data[column], errors="raise").astype("float64")
    volume = pd.to_numeric(data["volume"], errors="raise")
    if bool(volume.isna().any()):
        raise ValueError(f"{source}: volume column contains NaN values")
    normalised["volume"] = volume.astype("int64")
    normalised[IS_CLOSED_COLUMN] = True
    return normalised


def validate_ohlcv(
    frame: pd.DataFrame,
    *,
    source: str = "<frame>",
    check_consistency: bool = True,
) -> pd.DataFrame:
    """Validate NaN, duplicate timestamps, sort order and OHLC sanity.

    Raises ``ValueError`` naming the offending rows so a broken export is never
    silently processed; returns ``frame`` unchanged otherwise.
    """
    nan_mask = frame[list(OHLCV_COLUMNS)].isna().any(axis=1)
    if bool(nan_mask.any()):
        rows = frame.index[nan_mask].tolist()[:5]
        raise ValueError(f"{source}: NaN values in OHLCV at rows {rows}")

    duplicated = frame[TIMESTAMP_COLUMN].duplicated()
    if bool(duplicated.any()):
        stamps = frame.loc[duplicated, TIMESTAMP_COLUMN].head(5).tolist()
        raise ValueError(f"{source}: duplicate timestamps {stamps}")

    if not frame[TIMESTAMP_COLUMN].is_monotonic_increasing:
        raise ValueError(f"{source}: timestamps are not sorted ascending")

    if check_consistency:
        broken = (
            (frame["high"] < frame["low"])
            | (frame["high"] < frame[["open", "close"]].max(axis=1))
            | (frame["low"] > frame[["open", "close"]].min(axis=1))
        )
        if bool(broken.any()):
            rows = frame.index[broken].tolist()[:5]
            raise ValueError(f"{source}: inconsistent OHLC at rows {rows}")
    return frame


def load_csv(path: str | Path, *, drop_unclosed: bool = True) -> pd.DataFrame:
    """Load a single timeframe CSV into the canonical project schema.

    Parameters
    ----------
    path:
        File following the ``datetime,open,high,low,close,volume`` schema.
    drop_unclosed:
        When ``True`` (default) the presumed still-forming last bar is removed.

    Returns
    -------
    DataFrame with UTC-aware ``timestamp``, ``open``/``high``/``low``/``close``
    (float64), ``volume`` (int64) and ``is_closed`` (bool), sorted by
    ``timestamp``.
    """
    frame = pd.read_csv(path)
    frame = _normalise(frame, source=str(path))
    frame = frame.sort_values(TIMESTAMP_COLUMN, kind="stable").reset_index(drop=True)
    frame = validate_ohlcv(frame, source=str(path))
    frame = mark_closed(frame)
    if drop_unclosed:
        frame = _drop_unclosed_bars(frame)
    return frame


def attach_close_time(df: pd.DataFrame, period: pd.Timedelta | str) -> pd.DataFrame:
    """Add ``close_time = timestamp + period`` (open_time convention).

    ``period`` may be a timeframe label (``"M15"``/``"H1"``/``"D1"``) or any
    ``pd.Timedelta``-compatible value.  This is the only place where a bar is
    converted from its open stamp to its close stamp, so the whole pipeline
    shares one definition of "when the bar became known".  ``timestamp`` is
    normalised too, so both columns leave as ``datetime64[ns, UTC]``.
    """
    delta = _coerce_period(period)
    out = df.copy()
    out[TIMESTAMP_COLUMN] = _to_utc(out[TIMESTAMP_COLUMN])
    out[CLOSE_TIME_COLUMN] = out[TIMESTAMP_COLUMN] + delta
    return out


def mark_closed(df: pd.DataFrame, now: pd.Timestamp | str | None = None) -> pd.DataFrame:
    """Flag unfinished bars in an ``is_closed`` column.

    ``now=None`` means "the live edge is unknown": the last bar is assumed to be
    still forming, which is the conservative default demanded by the project
    constitution.  With an explicit ``now`` a bar is closed only when its
    ``close_time`` is not later than ``now`` (requires ``close_time``).
    """
    out = df.copy()
    if now is None:
        flags = pd.Series(True, index=out.index)
        if len(flags) > 0:
            flags.iloc[-1] = False
    else:
        if CLOSE_TIME_COLUMN not in out.columns:
            raise ValueError("mark_closed(now=...) needs a close_time column; call attach_close_time")
        flags = out[CLOSE_TIME_COLUMN] <= _as_utc_timestamp(now)
    out[IS_CLOSED_COLUMN] = flags.astype(bool)
    return out


def drop_unclosed(df: pd.DataFrame) -> pd.DataFrame:
    """Return only bars flagged as closed (no-op when the flag is absent)."""
    if IS_CLOSED_COLUMN not in df.columns:
        return df.copy()
    return df.loc[df[IS_CLOSED_COLUMN].astype(bool)].copy()


# ``load_csv`` and ``align_htf_to_ltf`` expose a boolean ``drop_unclosed`` flag
# (public API), which would shadow the helper above inside those functions, so
# internal callers use this alias instead.
_drop_unclosed_bars = drop_unclosed


def resample_to_timeframe(src_df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Aggregate a finer timeframe into ``target`` with explicit label/closed.

    Buckets are labelled by their open (``label="left"``), left-closed and
    anchored to midnight (``origin="start_day"``), matching the open_time
    convention of the source files: a bucket labelled 15:00 covers
    ``[15:00, 16:00)`` and only aggregates candles of its own interval.  Bars
    flagged as unclosed are ignored when ``is_closed`` is present.
    """
    period = period_for(target)
    frame = _drop_unclosed_bars(src_df) if IS_CLOSED_COLUMN in src_df.columns else src_df.copy()
    if frame.empty:
        empty = pd.DataFrame({column: pd.Series(dtype="float64") for column in OHLCV_COLUMNS})
        empty.insert(0, TIMESTAMP_COLUMN, pd.Series(dtype="datetime64[ns, UTC]"))
        return empty

    frame[TIMESTAMP_COLUMN] = _to_utc(frame[TIMESTAMP_COLUMN])
    frame = frame.set_index(TIMESTAMP_COLUMN)
    aggregated = frame.resample(
        period,
        label=RESAMPLE_LABEL,
        closed=RESAMPLE_CLOSED,
        origin=RESAMPLE_ORIGIN,
    ).agg(_AGGREGATION)
    aggregated = aggregated.dropna(subset=list(PRICE_COLUMNS))
    aggregated["volume"] = aggregated["volume"].fillna(0).astype("int64")
    aggregated = aggregated[list(OHLCV_COLUMNS)]
    aggregated.index.name = TIMESTAMP_COLUMN
    return aggregated.reset_index()


def align_htf_to_ltf(
    ltf_df: pd.DataFrame,
    htf_df: pd.DataFrame,
    *,
    htf_period: pd.Timedelta | str | None = None,
    htf_timeframe: Timeframe | str | None = None,
    ltf_key: str = "open",
    ltf_period: pd.Timedelta | str | None = None,
    suffixes: tuple[str, str] = ("", "_htf"),
    drop_unclosed: bool = True,
) -> pd.DataFrame:
    """Attach the most recent *closed* HTF bar to every LTF bar.

    The HTF join key is always ``close_time`` (never the HTF open stamp) and the
    merge is ``direction="backward"``, so an LTF bar can only ever see HTF bars
    that have already closed.  HTF columns are suffixed (``open_htf`` and
    friends) while ``timestamp_htf`` records which HTF bar was used, which keeps
    the result auditable.

    ``ltf_key`` selects the decision instant of the LTF bar: ``"open"`` (default
    and most conservative - an M15 bar starting at 15:00 sees the H1 bar that
    closed at 15:00 or earlier) or ``"close"`` (the HTF bar closing exactly at
    the LTF bar's own close is visible too).
    """
    left = ltf_df.copy()
    right = htf_df.copy()
    if TIMESTAMP_COLUMN not in left.columns or TIMESTAMP_COLUMN not in right.columns:
        raise ValueError("align_htf_to_ltf expects a 'timestamp' column in both frames")
    left[TIMESTAMP_COLUMN] = _to_utc(left[TIMESTAMP_COLUMN])
    right[TIMESTAMP_COLUMN] = _to_utc(right[TIMESTAMP_COLUMN])

    if CLOSE_TIME_COLUMN not in right.columns:
        if htf_period is None:
            if htf_timeframe is None:
                raise ValueError("htf_df has no close_time: pass htf_period= or htf_timeframe=")
            htf_period = period_for(str(htf_timeframe))
        right = attach_close_time(right, htf_period)

    if drop_unclosed:
        left = _drop_unclosed_bars(left)
        right = _drop_unclosed_bars(right)

    if ltf_key == "open":
        key_values = _to_utc(left[TIMESTAMP_COLUMN])
    elif ltf_key == "close":
        if CLOSE_TIME_COLUMN not in left.columns:
            if ltf_period is None:
                raise ValueError("ltf_key='close' needs ltf_period= or a close_time column")
            left = attach_close_time(left, ltf_period)
        key_values = _to_utc(left[CLOSE_TIME_COLUMN])
    else:
        raise ValueError(f"unsupported ltf_key {ltf_key!r}; expected 'open' or 'close'")

    left = left.assign(**{_LTF_KEY_COLUMN: key_values})
    right = right.assign(**{CLOSE_TIME_COLUMN: _to_utc(right[CLOSE_TIME_COLUMN])})
    left = left.sort_values(_LTF_KEY_COLUMN, kind="stable").reset_index(drop=True)
    right = right.sort_values(CLOSE_TIME_COLUMN, kind="stable").reset_index(drop=True)

    merged = pd.merge_asof(
        left,
        right,
        left_on=_LTF_KEY_COLUMN,
        right_on=CLOSE_TIME_COLUMN,
        direction="backward",
        allow_exact_matches=True,
        suffixes=list(suffixes),
    )
    return merged.drop(columns=[_LTF_KEY_COLUMN])
