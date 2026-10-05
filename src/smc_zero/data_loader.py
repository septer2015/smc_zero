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

The second supported on-disk layout is the raw MetaTrader 5 "Bars" export
(Э11'.1)::

    symbol,timeframe,time,open,high,low,close,tick_volume,spread,real_volume

:func:`detect_format` tells the two apart by their columns, :func:`load_ohlcv`
is the single entry point over both and :func:`merge_ohlcv` extends a tape with
a second source.  The project layout keeps the older :func:`load_csv` contract
untouched, MT5 stamps are read with an explicit format list because D1 is a bare
date and intraday bars are full stamps.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from smc_zero.config import Timeframe

OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")
TIMESTAMP_COLUMN = "timestamp"
CLOSE_TIME_COLUMN = "close_time"
IS_CLOSED_COLUMN = "is_closed"
SOURCE_TIME_COLUMN = "datetime"

#: The two on-disk layouts and the value that lets the columns decide (Э11'.1).
PROJECT_FORMAT = "project"
MT5_FORMAT = "mt5"
AUTO_FORMAT = "auto"
SUPPORTED_FORMATS: tuple[str, ...] = (AUTO_FORMAT, PROJECT_FORMAT, MT5_FORMAT)

#: MetaTrader 5 "Bars" export: the columns that mark it and the raw OHLCV pair.
MT5_TIME_COLUMN = "time"
MT5_VOLUME_COLUMN = "tick_volume"
MT5_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"symbol", "timeframe", MT5_TIME_COLUMN, MT5_VOLUME_COLUMN}
)
#: MT5 writes D1 as a bare date and every intraday bar as a full stamp.
MT5_TIME_FORMATS: tuple[str, ...] = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d")

# Bar length for every supported timeframe label.
TIMEFRAME_PERIODS: dict[str, pd.Timedelta] = {
    "M15": pd.Timedelta(minutes=15),
    "H1": pd.Timedelta(hours=1),
    "H4": pd.Timedelta(hours=4),
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


def detect_format(columns: Iterable[str], *, source: str = "<frame>") -> str:
    """Tell the on-disk layout of ``columns``: ``"mt5"`` or ``"project"``.

    The MT5 marker wins when it is present, so a file carrying both the
    ``symbol,timeframe,time,tick_volume`` block and a ``datetime`` column is read
    as MT5.  A frame matching neither layout is refused with the file name: a
    guess here would silently mis-map prices onto the wrong column (rule 1).
    """
    names = {str(column) for column in columns}
    if MT5_REQUIRED_COLUMNS <= names:
        return MT5_FORMAT
    if {SOURCE_TIME_COLUMN, *OHLCV_COLUMNS} <= names:
        return PROJECT_FORMAT
    mt5 = ", ".join(sorted(MT5_REQUIRED_COLUMNS))
    project = f"{SOURCE_TIME_COLUMN},{','.join(OHLCV_COLUMNS)}"
    raise ValueError(
        f"{source}: cannot tell the layout; expected MT5 ({mt5}) or project ({project})"
    )


def _parse_mt5_time(values: pd.Series, *, source: str) -> pd.Series:
    """Read the MT5 ``time`` column; the two layouts MT5 writes are the contract.

    M15/H1 carry a full ``YYYY-MM-DD HH:MM:SS`` stamp and D1 a bare date, so the
    parse is an explicit list of patterns and not ``errors="coerce"``: a value
    matching neither is an export this loader does not know and it is refused,
    never turned into a silent ``NaT`` (rule 1).
    """
    for pattern in MT5_TIME_FORMATS:
        try:
            return pd.to_datetime(values, format=pattern)
        except (TypeError, ValueError):
            continue
    formats = ", ".join(MT5_TIME_FORMATS)
    raise ValueError(f"{source}: time column matches none of {formats}")


def _normalise_mt5(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Shed an MT5 export down to the project's raw schema.

    ``time`` becomes the source time column and ``tick_volume`` becomes
    ``volume``; ``symbol``, ``timeframe``, ``spread`` and ``real_volume`` carry
    nothing the pipeline reads and are dropped.  The result is the
    ``datetime,open,high,low,close,volume`` shape :func:`_normalise` already
    knows, so both layouts share one downstream path.
    """
    data = frame.copy()
    data[SOURCE_TIME_COLUMN] = _parse_mt5_time(data[MT5_TIME_COLUMN], source=source)
    data["volume"] = pd.to_numeric(data[MT5_VOLUME_COLUMN], errors="raise")
    return data[[SOURCE_TIME_COLUMN, *OHLCV_COLUMNS]]


def load_ohlcv(
    path: str | Path,
    *,
    format: str = AUTO_FORMAT,
    drop_unclosed: bool = True,
    dedup: bool | None = None,
) -> pd.DataFrame:
    """Load a CSV of either supported layout into the canonical project schema.

    ``format`` is ``"auto"`` (default), ``"project"`` or ``"mt5"``.  The explicit
    value overrides the columns; ``"auto"`` reads them through
    :func:`detect_format`.

    The MT5 layout is normalised first (:func:`_normalise_mt5`) and its duplicate
    timestamps are dropped by default - a repeated broker export often re-covers a
    range it was asked for again.  The project layout keeps the older contract of
    :func:`load_csv`, where a duplicate timestamp stays a ``ValueError``.  Pass
    ``dedup`` explicitly to override either default.

    Returns a DataFrame with UTC-aware ``timestamp``, ``open``/``high``/``low``/
    ``close`` (float64), ``volume`` (int64) and ``is_closed`` (bool), sorted by
    ``timestamp``.  A missing file is a :class:`FileNotFoundError` naming the file
    and the folder it was looked for in.
    """
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"no data file at {file}: expected {file.name} under {file.parent}")
    if format not in SUPPORTED_FORMATS:
        supported = ", ".join(SUPPORTED_FORMATS)
        raise ValueError(f"unsupported format {format!r}; expected one of {supported}")

    frame = pd.read_csv(file)
    layout = detect_format(frame.columns, source=str(file)) if format == AUTO_FORMAT else format
    if layout == MT5_FORMAT:
        frame = _normalise_mt5(frame, source=str(file))
        layout_dedup = True
    else:
        layout_dedup = False

    frame = _normalise(frame, source=str(file))
    frame = frame.sort_values(TIMESTAMP_COLUMN, kind="stable").reset_index(drop=True)
    if layout_dedup if dedup is None else dedup:
        frame = frame.drop_duplicates(subset=[TIMESTAMP_COLUMN], keep="last").reset_index(drop=True)
    frame = validate_ohlcv(frame, source=str(file))
    frame = mark_closed(frame)
    if drop_unclosed:
        frame = _drop_unclosed_bars(frame)
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

    This is a thin wrapper over :func:`load_ohlcv` with ``format="project"``: the
    schema, the type coercion, the unclosed-tail rule and the duplicate-timestamp
    ``ValueError`` are exactly the ones the loader has always had, so every
    existing caller keeps its behaviour (Э11'.1).
    """
    return load_ohlcv(path, format=PROJECT_FORMAT, drop_unclosed=drop_unclosed)


def merge_ohlcv(tapes: Iterable[pd.DataFrame], *, source: str = "<merge>") -> pd.DataFrame:
    """Concatenate canonical tapes into one, dropping duplicate timestamps.

    Every frame has to be the output of :func:`load_ohlcv` (or :func:`load_csv`):
    a UTC-aware ``timestamp`` plus the OHLCV columns.  The frames are joined in
    the given order, sorted by ``timestamp`` and de-duplicated with
    ``keep="last"`` - the later source wins a shared bar, so
    ``merge_ohlcv([project, fresh])`` lets the fresh export overwrite the old
    one.  The ``is_closed`` flag travels with its own row.

    This is the extension path of Э11'.1: four years of project data plus a
    longer MT5 export become one tape without a single manual conversion.
    """
    frames = list(tapes)
    if not frames:
        raise ValueError(f"{source}: nothing to merge")
    wanted = [TIMESTAMP_COLUMN, *OHLCV_COLUMNS]
    parts = []
    for frame in frames:
        missing = [column for column in wanted if column not in frame.columns]
        if missing:
            raise ValueError(f"{source}: a tape is missing {missing}")
        part = frame[wanted].copy()
        part[TIMESTAMP_COLUMN] = _to_utc(part[TIMESTAMP_COLUMN])
        if IS_CLOSED_COLUMN in frame.columns:
            part[IS_CLOSED_COLUMN] = frame[IS_CLOSED_COLUMN].astype(bool)
        else:
            part[IS_CLOSED_COLUMN] = True
        parts.append(part)
    merged = pd.concat(parts, ignore_index=True)
    merged = merged.sort_values(TIMESTAMP_COLUMN, kind="stable")
    merged = merged.drop_duplicates(subset=[TIMESTAMP_COLUMN], keep="last").reset_index(drop=True)
    merged = validate_ohlcv(merged, source=source)
    merged[TIMESTAMP_COLUMN] = merged[TIMESTAMP_COLUMN].astype("datetime64[ns, UTC]")
    return merged


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


def _extra_column_names(
    htf_df: pd.DataFrame,
    extra_columns: tuple[str, ...],
    suffix: str,
) -> dict[str, str]:
    """Map the requested ``extra_columns`` of an HTF frame onto their suffixed names.

    ``merge_asof`` suffixes only the names present in *both* frames, so an HTF-only
    column (``trend``) would keep its bare name and silently escape the suffix
    contract.  Renaming the HTF copy up front makes the merged name depend on
    ``suffix`` alone, whatever the LTF frame happens to carry.
    """
    if extra_columns and not suffix:
        raise ValueError("extra_columns needs a non-empty HTF suffix")
    names: dict[str, str] = {}
    for column in extra_columns:
        if column not in htf_df.columns:
            raise ValueError(f"htf_df has no extra column {column!r}")
        if column in names:
            raise ValueError(f"duplicate extra column {column!r}")
        names[column] = f"{column}{suffix}"
    return names


def align_htf_to_ltf(
    ltf_df: pd.DataFrame,
    htf_df: pd.DataFrame,
    *,
    htf_period: pd.Timedelta | str | None = None,
    htf_timeframe: Timeframe | str | None = None,
    ltf_key: str = "open",
    ltf_period: pd.Timedelta | str | None = None,
    suffixes: tuple[str, str] = ("", "_htf"),
    extra_columns: tuple[str, ...] = (),
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

    ``extra_columns`` lists HTF columns beyond OHLCV that must be carried along
    (e.g. ``("trend",)`` for the bias).  They travel through the very same
    ``close_time`` merge, so an extra value of an HTF bar that is still open stays
    invisible until that bar closes, and it appears as ``<column><htf suffix>``
    regardless of whether the LTF frame owns that name.
    """
    left = ltf_df.copy()
    right = htf_df.copy()
    if TIMESTAMP_COLUMN not in left.columns or TIMESTAMP_COLUMN not in right.columns:
        raise ValueError("align_htf_to_ltf expects a 'timestamp' column in both frames")
    extra_names = _extra_column_names(right, extra_columns, suffixes[1])
    clashing = [name for name in extra_names.values() if name in left.columns]
    if clashing:
        raise ValueError(f"ltf_df already has {clashing}: pick another suffix for extra_columns")
    right = right.rename(columns=extra_names)
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
