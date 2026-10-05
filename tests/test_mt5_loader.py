"""Tests for the MetaTrader 5 adapter of :mod:`smc_zero.data_loader` (Э11'.1).

Everything here is synthetic and lives in ``tmp_path``: an MT5 "Bars" export, the
equivalent project CSV and the canonical frames both must produce.  The tests pin
what the adapter promises - a layout told apart by its columns, the two MT5 time
formats (full intraday stamps and the bare date of D1), a duplicate policy that
keeps the later export, and the fact that :func:`load_csv` still behaves exactly
as it did before the adapter existed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts import _common
from smc_zero.data_loader import (
    IS_CLOSED_COLUMN,
    MT5_FORMAT,
    OHLCV_COLUMNS,
    PROJECT_FORMAT,
    TIMESTAMP_COLUMN,
    detect_format,
    load_csv,
    load_ohlcv,
    merge_ohlcv,
)

MT5_HEADER = "symbol,timeframe,time,open,high,low,close,tick_volume,spread,real_volume"
PROJECT_HEADER = f"datetime,{','.join(OHLCV_COLUMNS)}"


def _intraday_rows(periods: int) -> list[tuple[str, float, int]]:
    """``periods`` M15 rows from 2022-01-03 00:00, one pip apart: (stamp, base, volume)."""
    stamps = pd.date_range("2022-01-03 00:00", periods=periods, freq="15min")
    return [
        (stamp.strftime("%Y-%m-%d %H:%M:%S"), 1.1 + index * 0.0001, 100 + index)
        for index, stamp in enumerate(stamps)
    ]


def _write_mt5(path: Path, rows: list[tuple[str, float, int]], *, timeframe: str = "M15") -> Path:
    """Write a MetaTrader 5 "Bars" export: the wide layout with its two marker columns."""
    body = "\n".join(
        f"EURUSD,{timeframe},{stamp},{base:.5f},{base + 0.0002:.5f},"
        f"{base - 0.0002:.5f},{base + 0.0001:.5f},{volume},20,0"
        for stamp, base, volume in rows
    )
    path.write_text(f"{MT5_HEADER}\n{body}\n", encoding="utf-8")
    return path


def _write_project(path: Path, rows: list[tuple[str, float, int]]) -> Path:
    """Write the same rows in the project layout: naive ``datetime`` strings, no extra columns."""
    body = "\n".join(
        f"{stamp},{base:.5f},{base + 0.0002:.5f},{base - 0.0002:.5f},{base + 0.0001:.5f},{volume}"
        for stamp, base, volume in rows
    )
    path.write_text(f"{PROJECT_HEADER}\n{body}\n", encoding="utf-8")
    return path


def test_detect_format_tells_the_two_layouts_apart() -> None:
    project_columns = PROJECT_HEADER.split(",")

    assert detect_format(MT5_HEADER.split(",")) == MT5_FORMAT
    assert detect_format(project_columns) == PROJECT_FORMAT
    # the MT5 marker wins when a file carries both, so prices never land in a wrong column
    assert detect_format([*project_columns, *MT5_HEADER.split(",")]) == MT5_FORMAT


def test_detect_format_names_the_file_it_cannot_read() -> None:
    with pytest.raises(ValueError) as error:
        detect_format(["foo", "bar"], source="mystery.csv")

    message = str(error.value)
    assert "cannot tell the layout" in message
    assert "mystery.csv" in message


def test_load_ohlcv_reads_mt5_intraday_with_the_canonical_schema(tmp_path: Path) -> None:
    rows = _intraday_rows(6)

    loaded = load_ohlcv(_write_mt5(tmp_path / "EURUSD_M15.csv", rows))

    assert list(loaded.columns) == [TIMESTAMP_COLUMN, *OHLCV_COLUMNS, IS_CLOSED_COLUMN]
    assert str(loaded[TIMESTAMP_COLUMN].dtype) == "datetime64[ns, UTC]"
    assert str(loaded["open"].dtype) == "float64"
    assert str(loaded["volume"].dtype) == "int64"
    assert loaded[TIMESTAMP_COLUMN].is_monotonic_increasing
    # ``tick_volume`` became ``volume`` and the presumed still-forming last bar is dropped
    assert len(loaded) == len(rows) - 1
    assert loaded["volume"].tolist() == [volume for _, _, volume in rows[:-1]]
    assert bool(loaded[IS_CLOSED_COLUMN].all())


def test_load_ohlcv_reads_a_date_only_d1_export(tmp_path: Path) -> None:
    rows = [("2022-01-03", 1.1, 10), ("2022-01-04", 1.2, 11), ("2022-01-05", 1.3, 12)]

    loaded = load_ohlcv(_write_mt5(tmp_path / "EURUSD_D1.csv", rows, timeframe="D1"))

    assert len(loaded) == 2  # the unclosed tail is dropped, two closed days remain
    assert loaded[TIMESTAMP_COLUMN].tolist() == [
        pd.Timestamp("2022-01-03", tz="UTC"),
        pd.Timestamp("2022-01-04", tz="UTC"),
    ]
    assert str(loaded[TIMESTAMP_COLUMN].dtype) == "datetime64[ns, UTC]"


def test_mt5_and_project_layouts_yield_the_same_canonical_frame(tmp_path: Path) -> None:
    rows = _intraday_rows(6)

    from_mt5 = load_ohlcv(_write_mt5(tmp_path / "mt5.csv", rows), format=MT5_FORMAT)
    from_project = load_csv(_write_project(tmp_path / "project.csv", rows))

    pd.testing.assert_frame_equal(from_mt5, from_project)


def test_load_ohlcv_drops_duplicate_mt5_timestamps_keeping_the_later_one(tmp_path: Path) -> None:
    rows = _intraday_rows(4)
    doubled = [rows[0], rows[1], (rows[1][0], 1.23456, 999), rows[2], rows[3]]

    loaded = load_ohlcv(_write_mt5(tmp_path / "dup.csv", doubled), drop_unclosed=False)

    assert len(loaded) == len(rows)
    repeated = loaded.loc[loaded[TIMESTAMP_COLUMN] == pd.Timestamp(rows[1][0], tz="UTC")]
    assert repeated["volume"].tolist() == [999]
    assert repeated["open"].round(5).tolist() == [1.23456]


def test_load_ohlcv_keeps_the_project_duplicate_rule(tmp_path: Path) -> None:
    stamp = "2022-01-03 00:00:00"
    path = _write_project(tmp_path / "dup_project.csv", [(stamp, 1.1, 10), (stamp, 1.1, 11)])

    with pytest.raises(ValueError, match="duplicate timestamps"):
        load_ohlcv(path)

    # an explicit switch turns the MT5 policy on for a project file as well
    assert len(load_ohlcv(path, dedup=True, drop_unclosed=False)) == 1


def test_load_ohlcv_missing_file_names_the_folder(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as error:
        load_ohlcv(tmp_path / "EURUSD_M15.csv")

    message = str(error.value)
    assert "EURUSD_M15.csv" in message
    assert str(tmp_path) in message


def test_load_ohlcv_rejects_an_unknown_format(tmp_path: Path) -> None:
    path = _write_mt5(tmp_path / "EURUSD_M15.csv", _intraday_rows(3))

    with pytest.raises(ValueError, match="unsupported format"):
        load_ohlcv(path, format="parquet")


def test_load_ohlcv_refuses_a_file_of_no_known_layout(tmp_path: Path) -> None:
    path = tmp_path / "mystery.csv"
    path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot tell the layout"):
        load_ohlcv(path)


def test_load_ohlcv_refuses_an_mt5_time_it_cannot_read(tmp_path: Path) -> None:
    path = tmp_path / "EURUSD_M15.csv"
    path.write_text(
        f"{MT5_HEADER}\nEURUSD,M15,29/08/2022 01:15,1.1,1.2,1.0,1.15,10,20,0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="matches none"):
        load_ohlcv(path)


def test_a_corrupted_future_mt5_bar_cannot_change_the_past(tmp_path: Path) -> None:
    rows = _intraday_rows(8)

    clean = load_ohlcv(_write_mt5(tmp_path / "clean.csv", rows), drop_unclosed=False)
    corrupted_rows = [*rows[:-1], (rows[-1][0], 9.9999, 999_999)]
    corrupted = load_ohlcv(
        _write_mt5(tmp_path / "corrupt.csv", corrupted_rows), drop_unclosed=False
    )

    # every bar before the future one is byte-identical: no value leaks backwards
    pd.testing.assert_frame_equal(clean.iloc[:-1], corrupted.iloc[:-1])
    assert clean["close"].iloc[-1] != corrupted["close"].iloc[-1]


def test_merge_ohlcv_later_source_wins_and_the_result_is_sorted(tmp_path: Path) -> None:
    rows = _intraday_rows(4)

    older = load_ohlcv(_write_mt5(tmp_path / "older.csv", rows))  # 3 closed bars
    fresh_rows = [(rows[0][0], 9.0, 500), (rows[1][0], 9.1, 501)]
    fresh = load_ohlcv(_write_mt5(tmp_path / "fresh.csv", fresh_rows))  # 1 closed bar

    merged = merge_ohlcv([older, fresh])

    assert merged[TIMESTAMP_COLUMN].is_monotonic_increasing
    assert not bool(merged[TIMESTAMP_COLUMN].duplicated().any())
    assert len(merged) == len(older)
    # the shared bar carries the values of the later frame, the rest stays the older one
    assert merged["volume"].tolist() == [500, *older["volume"].tolist()[1:]]
    assert merged["open"].iloc[0] == 9.0
    assert bool(merged[IS_CLOSED_COLUMN].all())


def test_merge_ohlcv_refuses_an_empty_list_and_a_broken_tape() -> None:
    with pytest.raises(ValueError, match="nothing to merge"):
        merge_ohlcv([])

    lonely = pd.DataFrame({TIMESTAMP_COLUMN: pd.to_datetime(["2022-01-03"], utc=True)})
    with pytest.raises(ValueError, match="a tape is missing"):
        merge_ohlcv([lonely], source="broken")


def test_the_default_source_is_the_project_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_common.SMC_DATA_DIR_ENV, str(tmp_path))

    assert _common.tape_path("eurusd", "m15") == Path("data") / "EURUSD_M15.csv"


def test_the_mt5_base_follows_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_common.SMC_DATA_DIR_ENV, str(tmp_path))
    assert _common.mt5_data_dir() == tmp_path
    assert _common.tape_path("eurusd", "m15", source=_common.MT5_SOURCE) == (
        tmp_path / "EURUSD_M15.csv"
    )

    monkeypatch.delenv(_common.SMC_DATA_DIR_ENV)
    assert _common.mt5_data_dir() == Path.home() / "_data" / "mt5"


def test_load_windowed_tape_reads_the_mt5_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_common.SMC_DATA_DIR_ENV, str(tmp_path))
    rows = _intraday_rows(8)
    _write_mt5(tmp_path / "EURUSD_M15.csv", rows)
    day = _common.read_day("2022-01-03")

    tape = _common.load_windowed_tape("EURUSD", "M15", day, day, source=_common.MT5_SOURCE)

    assert list(tape[TIMESTAMP_COLUMN]) == [
        pd.Timestamp(stamp, tz="UTC") for stamp, _, _ in rows[:-1]
    ]


def test_load_windowed_tape_names_the_mt5_base_when_the_tape_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_common.SMC_DATA_DIR_ENV, str(tmp_path))
    day = _common.read_day("2022-01-03")

    with pytest.raises(FileNotFoundError, match=_common.SMC_DATA_DIR_ENV):
        _common.load_windowed_tape("EURUSD", "M15", day, day, source=_common.MT5_SOURCE)
