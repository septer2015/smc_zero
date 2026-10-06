"""Tests for :mod:`smc_zero.data_loader` on synthetic OHLCV data.

The synthetic frames never touch ``./data``: they are fully deterministic, so the
open_time convention, the unclosed-tail protection and the HTF -> LTF stitching
can be asserted exactly (including the look-ahead test).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smc_zero.data_loader import (
    CLOSE_TIME_COLUMN,
    IS_CLOSED_COLUMN,
    OHLCV_COLUMNS,
    SOURCE_TIME_COLUMN,
    TIMESTAMP_COLUMN,
    align_htf_to_ltf,
    attach_close_time,
    bars_per_day,
    drop_unclosed,
    load_csv,
    mark_closed,
    period_for,
    resample_to_timeframe,
)

LTF_PERIOD = pd.Timedelta(minutes=15)
MTF_PERIOD = pd.Timedelta(hours=1)
HTF_COLUMNS = tuple(f"{column}_htf" for column in OHLCV_COLUMNS)
HEADER = f"{SOURCE_TIME_COLUMN},{','.join(OHLCV_COLUMNS)}"


def _frame(start: str, periods: int, freq: str) -> pd.DataFrame:
    """Deterministic synthetic OHLCV frame with open_time UTC labels."""
    stamps = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")
    levels = 1.1 + np.arange(periods, dtype="float64") * 0.0001
    return pd.DataFrame(
        {
            TIMESTAMP_COLUMN: stamps,
            "open": levels,
            "high": levels + 0.0002,
            "low": levels - 0.0002,
            "close": levels + 0.0001,
            "volume": np.arange(100, 100 + periods, dtype="int64"),
        }
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    """Write a frame the way ``./data`` stores it: naive ``datetime`` strings."""
    naive = frame[TIMESTAMP_COLUMN].dt.tz_convert("UTC").dt.tz_localize(None)
    stamps = naive.dt.strftime("%Y-%m-%d %H:%M:%S")
    out = frame.drop(columns=[TIMESTAMP_COLUMN]).assign(**{SOURCE_TIME_COLUMN: stamps})
    out = out[[SOURCE_TIME_COLUMN, *OHLCV_COLUMNS]]
    out.to_csv(path, index=False)
    return path


def _write_text(path: Path, body: str) -> Path:
    path.write_text(f"{HEADER}\n{body}", encoding="utf-8")
    return path


def test_load_csv_schema_dtypes_and_unclosed_tail(tmp_path: Path) -> None:
    source = _frame("2022-01-03 00:00", periods=8, freq="15min")
    csv_path = _write_csv(tmp_path / "EURUSD_M15.csv", source)

    loaded = load_csv(csv_path)

    assert set(OHLCV_COLUMNS).issubset(loaded.columns)
    assert str(loaded[TIMESTAMP_COLUMN].dtype) == "datetime64[ns, UTC]"
    assert str(loaded["open"].dtype) == "float64"
    assert str(loaded["volume"].dtype) == "int64"
    assert loaded[TIMESTAMP_COLUMN].is_monotonic_increasing
    # the presumed still-forming last bar is dropped by default
    assert len(loaded) == len(source) - 1
    assert loaded[TIMESTAMP_COLUMN].max() == source[TIMESTAMP_COLUMN].iloc[-2]
    assert bool(loaded[IS_CLOSED_COLUMN].all())


def test_load_csv_keeps_unclosed_tail_when_asked(tmp_path: Path) -> None:
    source = _frame("2022-01-03 00:00", periods=5, freq="15min")
    csv_path = _write_csv(tmp_path / "EURUSD_M15.csv", source)

    loaded = load_csv(csv_path, drop_unclosed=False)

    assert len(loaded) == len(source)
    assert bool(loaded[IS_CLOSED_COLUMN].iloc[-1]) is False
    assert bool(loaded[IS_CLOSED_COLUMN].iloc[:-1].all())


def test_load_csv_localizes_naive_stamps_to_utc(tmp_path: Path) -> None:
    csv_path = _write_text(
        tmp_path / "naive.csv",
        "2022-01-03 00:00:00,1.1,1.2,1.0,1.15,10\n"
        "2022-01-03 00:15:00,1.1,1.2,1.0,1.15,11\n",
    )
    loaded = load_csv(csv_path)

    assert str(loaded[TIMESTAMP_COLUMN].dtype) == "datetime64[ns, UTC]"
    assert loaded[TIMESTAMP_COLUMN].iloc[0] == pd.Timestamp("2022-01-03 00:00:00", tz="UTC")


def test_load_csv_converts_aware_stamps_to_utc(tmp_path: Path) -> None:
    csv_path = _write_text(
        tmp_path / "aware.csv",
        "2022-01-03T00:00:00+02:00,1.1,1.2,1.0,1.15,10\n"
        "2022-01-03T00:15:00+02:00,1.1,1.2,1.0,1.15,11\n",
    )
    loaded = load_csv(csv_path)

    assert str(loaded[TIMESTAMP_COLUMN].dtype) == "datetime64[ns, UTC]"
    assert loaded[TIMESTAMP_COLUMN].iloc[0] == pd.Timestamp("2022-01-02 22:00:00", tz="UTC")


def test_load_csv_rejects_duplicate_timestamps(tmp_path: Path) -> None:
    csv_path = _write_text(
        tmp_path / "duplicates.csv",
        "2022-01-03 00:00:00,1.1,1.2,1.0,1.15,10\n"
        "2022-01-03 00:00:00,1.1,1.2,1.0,1.15,11\n"
        "2022-01-03 00:15:00,1.1,1.2,1.0,1.15,12\n",
    )
    with pytest.raises(ValueError, match="duplicate timestamps"):
        load_csv(csv_path)


def test_load_csv_rejects_nan(tmp_path: Path) -> None:
    csv_path = _write_text(
        tmp_path / "nan.csv",
        "2022-01-03 00:00:00,1.1,1.2,1.0,,10\n"
        "2022-01-03 00:15:00,1.1,1.2,1.0,1.15,11\n",
    )
    with pytest.raises(ValueError, match="NaN"):
        load_csv(csv_path)


def test_load_csv_rejects_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "broken.csv"
    csv_path.write_text("datetime,open,high,low,close\n2022-01-03 00:00:00,1.1,1.2,1.0,1.15\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required column"):
        load_csv(csv_path)


def test_load_csv_rejects_inconsistent_ohlc(tmp_path: Path) -> None:
    csv_path = _write_text(
        tmp_path / "broken_ohlc.csv",
        "2022-01-03 00:00:00,1.1,1.05,1.0,1.15,10\n"
        "2022-01-03 00:15:00,1.1,1.2,1.0,1.15,11\n",
    )
    with pytest.raises(ValueError, match="inconsistent OHLC"):
        load_csv(csv_path)


def test_attach_close_time_adds_the_bar_period(tmp_path: Path) -> None:
    loaded = load_csv(_write_csv(tmp_path / "ltf.csv", _frame("2022-01-03 00:00", 6, "15min")))

    with_close = attach_close_time(loaded, "M15")

    assert bool((with_close[CLOSE_TIME_COLUMN] == with_close[TIMESTAMP_COLUMN] + LTF_PERIOD).all())
    assert with_close[CLOSE_TIME_COLUMN].dtype == with_close[TIMESTAMP_COLUMN].dtype
    # the label sits at the open of the interval: 00:00 -> 00:15
    assert with_close[CLOSE_TIME_COLUMN].iloc[0] == pd.Timestamp("2022-01-03 00:15:00", tz="UTC")
    assert attach_close_time(loaded, pd.Timedelta(minutes=15))[CLOSE_TIME_COLUMN].equals(
        with_close[CLOSE_TIME_COLUMN]
    )


def test_attach_close_time_rejects_non_positive_period(tmp_path: Path) -> None:
    loaded = load_csv(_write_csv(tmp_path / "ltf.csv", _frame("2022-01-03 00:00", 3, "15min")))
    with pytest.raises(ValueError, match="period must be positive"):
        attach_close_time(loaded, pd.Timedelta(0))


def test_mark_closed_and_drop_unclosed(tmp_path: Path) -> None:
    loaded = load_csv(
        _write_csv(tmp_path / "ltf.csv", _frame("2022-01-03 00:00", 6, "15min")),
        drop_unclosed=False,
    )
    with_close = attach_close_time(loaded, "M15")

    marked = mark_closed(with_close)
    assert bool(marked[IS_CLOSED_COLUMN].iloc[:-1].all())
    assert bool(marked[IS_CLOSED_COLUMN].iloc[-1]) is False
    assert len(drop_unclosed(marked)) == len(loaded) - 1
    # without the flag nothing is dropped
    assert len(drop_unclosed(_frame("2022-01-03 00:00", 3, "15min"))) == 3

    # an explicit "now" closes exactly the bars whose close_time is not later
    closed_now = mark_closed(with_close, now="2022-01-03 00:30:00")
    assert bool(closed_now[IS_CLOSED_COLUMN].iloc[1]) is True
    assert bool(closed_now[IS_CLOSED_COLUMN].iloc[2]) is False


# Corrupted-bar positions for the look-ahead test: late / middle / early H1 bar.
# Corrupting only the last bar cannot catch an off-by-k shift in the alignment,
# because late bars have no LTF bars that must stay untouched *after* them.
LTF_BARS = 100
HTF_BARS = 24
CORRUPTED_HTF_BAR_INDICES = (HTF_BARS - 1, HTF_BARS // 2, 1)


@pytest.mark.parametrize(
    "corrupted_index",
    CORRUPTED_HTF_BAR_INDICES,
    ids=("last", "middle", "early"),
)
def test_align_htf_to_ltf_never_sees_an_open_htf_bar(corrupted_index: int) -> None:
    ltf = _frame("2022-01-03 00:00", periods=LTF_BARS, freq="15min")  # .. next day 00:45
    htf = _frame("2022-01-03 00:00", periods=HTF_BARS, freq="1h")  # 00:00 .. 23:00

    aligned = align_htf_to_ltf(ltf, htf, htf_period="H1")

    corrupted = htf.copy()
    corrupted.loc[corrupted.index[corrupted_index], ["open", "high", "low", "close"]] = 9.9999
    corrupted.loc[corrupted.index[corrupted_index], "volume"] = 999_999
    aligned_corrupted = align_htf_to_ltf(ltf, corrupted, htf_period="H1")

    # the corrupted bar can only be seen from its own close_time until the next H1
    # bar closes; ``before`` keeps the NaN head on purpose - LTF bars that have no
    # closed H1 bar yet must be covered by the comparison as well
    corrupted_open = htf[TIMESTAMP_COLUMN].iloc[corrupted_index]
    corrupted_close = corrupted_open + MTF_PERIOD
    visible_until = corrupted_close + MTF_PERIOD
    before = aligned[TIMESTAMP_COLUMN] < corrupted_close
    after = (aligned[TIMESTAMP_COLUMN] >= corrupted_close) & (
        aligned[TIMESTAMP_COLUMN] < visible_until
    )

    # both sides must be exercised, otherwise the test proves nothing
    assert bool(before.any()) and bool(after.any())
    # ``after`` is exactly the window where the corrupted bar is the latest closed one
    assert bool(aligned.loc[after, "timestamp_htf"].eq(corrupted_open).all())
    # only the LTF bars that precede the first HTF close have nothing to attach
    first_htf_close = htf[TIMESTAMP_COLUMN].iloc[0] + MTF_PERIOD
    assert int(aligned["close_htf"].isna().sum()) == int(
        (aligned[TIMESTAMP_COLUMN] < first_htf_close).sum()
    )
    assert not bool(aligned.loc[after, HTF_COLUMNS].isna().any().any())

    # a change inside a HTF bar that is still open for those LTF bars leaks nowhere
    pd.testing.assert_frame_equal(
        aligned.loc[before, HTF_COLUMNS],
        aligned_corrupted.loc[before, HTF_COLUMNS],
    )
    # ... while the LTF bars inside its visibility window do see the new value
    assert bool((aligned.loc[after, "close_htf"] != 9.9999).all())
    assert bool((aligned_corrupted.loc[after, "close_htf"] == 9.9999).all())
    # ... and nothing outside that window changed at all (catches an off-by-k shift);
    # the shared NaN head is unchanged by definition, so it is excluded explicitly
    both_nan = aligned["close_htf"].isna() & aligned_corrupted["close_htf"].isna()
    changed = aligned["close_htf"].ne(aligned_corrupted["close_htf"]) & ~both_nan
    assert not bool(changed.loc[~after].any())


def test_align_htf_to_ltf_drops_flagged_unclosed_htf_bar() -> None:
    ltf = _frame("2022-01-03 00:00", periods=100, freq="15min")
    htf = mark_closed(attach_close_time(_frame("2022-01-03 00:00", 24, "1h"), "H1"))
    htf.loc[htf.index[-1], "close"] = 9.9999

    aligned = align_htf_to_ltf(ltf, htf, htf_period="H1")

    assert bool((aligned["close_htf"] != 9.9999).all())
    assert aligned["timestamp_htf"].max() == htf[TIMESTAMP_COLUMN].iloc[-2]


# Extra HTF columns travel through the same close_time stitch as the OHLCV columns,
# so they are tested with the same tamper-over-positions pattern.  ``CORRUPTED_MARKER``
# is a value no HTF bar ever carries, so a leak cannot hide behind a coincidence.
EXTRA_COLUMN = "trend"
EXTRA_SUFFIX = "_h1"
CORRUPTED_MARKER = -1.0


@pytest.mark.parametrize(
    "corrupted_index",
    CORRUPTED_HTF_BAR_INDICES,
    ids=("last", "middle", "early"),
)
def test_align_extra_column_of_an_open_htf_bar_stays_invisible(corrupted_index: int) -> None:
    ltf = _frame("2022-01-03 00:00", periods=LTF_BARS, freq="15min")
    htf = _frame("2022-01-03 00:00", periods=HTF_BARS, freq="1h")
    htf[EXTRA_COLUMN] = np.arange(HTF_BARS, dtype="float64")
    attached = f"{EXTRA_COLUMN}{EXTRA_SUFFIX}"

    kwargs = {
        "htf_period": "H1",
        "suffixes": ("", EXTRA_SUFFIX),
        "extra_columns": (EXTRA_COLUMN,),
    }
    aligned = align_htf_to_ltf(ltf, htf, **kwargs)

    corrupted = htf.copy()
    corrupted.loc[corrupted.index[corrupted_index], EXTRA_COLUMN] = CORRUPTED_MARKER
    aligned_corrupted = align_htf_to_ltf(ltf, corrupted, **kwargs)

    bar_open = htf[TIMESTAMP_COLUMN].iloc[corrupted_index]
    bar_close = bar_open + MTF_PERIOD
    before = aligned[TIMESTAMP_COLUMN] < bar_close
    window = (aligned[TIMESTAMP_COLUMN] >= bar_close) & (
        aligned[TIMESTAMP_COLUMN] < bar_close + MTF_PERIOD
    )
    # both sides must be exercised, otherwise the test proves nothing
    assert bool(before.any()) and bool(window.any())

    # the extra value of an HTF bar that is still open is invisible before its close
    assert not bool(aligned_corrupted.loc[before, attached].eq(CORRUPTED_MARKER).any())
    # and it is exactly the attached bar inside its own visibility window
    assert bool(aligned_corrupted.loc[window, attached].eq(CORRUPTED_MARKER).all())
    assert bool(aligned.loc[window, attached].eq(corrupted_index).all())
    assert bool(aligned.loc[window, f"timestamp{EXTRA_SUFFIX}"].eq(bar_open).all())


def test_align_htf_to_ltf_rejects_broken_extra_columns() -> None:
    ltf = _frame("2022-01-03 00:00", periods=LTF_BARS, freq="15min")
    htf = _frame("2022-01-03 00:00", periods=HTF_BARS, freq="1h")
    htf[EXTRA_COLUMN] = np.arange(HTF_BARS, dtype="float64")

    with pytest.raises(ValueError, match="no extra column"):
        align_htf_to_ltf(ltf, htf, htf_period="H1", extra_columns=("missing",))
    with pytest.raises(ValueError, match="non-empty HTF suffix"):
        align_htf_to_ltf(ltf, htf, htf_period="H1", suffixes=("", ""), extra_columns=("trend",))
    with pytest.raises(ValueError, match="duplicate extra column"):
        align_htf_to_ltf(ltf, htf, htf_period="H1", extra_columns=("trend", "trend"))
    # an LTF frame owning the suffixed name would make merge_asof suffix both copies
    # and the requested column would silently vanish, so it is refused up front
    busy = ltf.assign(**{f"{EXTRA_COLUMN}{EXTRA_SUFFIX}": 0.0})
    with pytest.raises(ValueError, match="already has"):
        align_htf_to_ltf(
            busy,
            htf,
            htf_period="H1",
            suffixes=("", EXTRA_SUFFIX),
            extra_columns=(EXTRA_COLUMN,),
        )


def test_resample_to_timeframe_is_left_labelled_and_interval_local() -> None:
    ltf = _frame("2022-01-03 00:00", periods=12, freq="15min")

    hourly = resample_to_timeframe(ltf, "H1")

    first_hour = ltf[ltf[TIMESTAMP_COLUMN] < pd.Timestamp("2022-01-03 01:00", tz="UTC")]
    first = hourly.iloc[0]
    assert str(hourly[TIMESTAMP_COLUMN].dtype) == "datetime64[ns, UTC]"
    assert first[TIMESTAMP_COLUMN] == pd.Timestamp("2022-01-03 00:00", tz="UTC")
    assert first["open"] == first_hour["open"].iloc[0]
    assert first["high"] == first_hour["high"].max()
    assert first["low"] == first_hour["low"].min()
    assert first["close"] == first_hour["close"].iloc[-1]
    assert first["volume"] == first_hour["volume"].sum()
    # a resampled bar can never be stamped outside the source range
    assert hourly[TIMESTAMP_COLUMN].min() >= ltf[TIMESTAMP_COLUMN].min()
    assert hourly[TIMESTAMP_COLUMN].max() <= ltf[TIMESTAMP_COLUMN].max()
    assert bool((hourly["high"] <= ltf["high"].max()).all())


def test_resample_skips_unclosed_source_bars() -> None:
    marked = mark_closed(_frame("2022-01-03 00:00", periods=9, freq="15min"))  # 02:00 still open

    hourly = resample_to_timeframe(marked, "H1")

    assert hourly[TIMESTAMP_COLUMN].tolist() == [
        pd.Timestamp("2022-01-03 00:00", tz="UTC"),
        pd.Timestamp("2022-01-03 01:00", tz="UTC"),
    ]
    closed = marked.loc[marked[IS_CLOSED_COLUMN]]
    assert int(hourly["volume"].sum()) == int(closed["volume"].sum())


def test_the_m5_entry_timeframe_carries_its_bar_length() -> None:
    """``"M5"`` is a supported label, so the second hierarchy can name its entry bar.

    The bar length drives three things at once: ``close_time = timestamp + period`` of a
    loaded file, the visibility of a stitched structure bar and the lifecycle dates of a
    level.  A label the table does not hold must keep raising, never fall back silently.
    """
    assert period_for("M5") == pd.Timedelta(minutes=5)
    assert period_for("m5") == pd.Timedelta(minutes=5)
    with pytest.raises(ValueError, match="unsupported timeframe"):
        period_for("M1")


def test_the_bar_scale_of_a_timeframe_counts_one_day() -> None:
    """``bars_per_day`` is the scale of the run (Sharpe, fold windows): 96 on M15, 288 on M5."""
    assert bars_per_day("M15") == 96
    assert bars_per_day("M5") == 288
    assert bars_per_day("H1") == 24
    assert bars_per_day("H4") == 6
    assert bars_per_day("D1") == 1
    with pytest.raises(ValueError, match="unsupported timeframe"):
        bars_per_day("M1")
