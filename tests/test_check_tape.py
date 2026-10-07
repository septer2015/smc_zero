"""Тесты Э13.2: что ``smc-check-tape`` говорит о ленте, которую ему дали.

Синтетические ленты записываются в оба поддерживаемых формата (``project`` и ``mt5``), поэтому
проверяются обе ветки разведки: дубли и порядок читаются по сырому файлу, а сетка и покрытие -
по каноничной ленте загрузчика.

Мутации, от которых эти тесты обязаны краснеть:

* m1 "считать дубли по каноничной ленте" - загрузчик схлопывает дубли MT5-экспорта, и стык двух
  выгрузок выглядел бы чистым; краснит :func:`test_a_duplicated_seam_is_a_failure`;
* m2 "считать разрывом любую паузу длиннее порога" - выходные дали бы десятки пропусков на
  каждой ленте, и отчёт перестал бы отличать дефект от рынка; краснит
  :func:`test_a_weekend_pause_is_not_a_hole`;
* m3 "мерить покрытие по всем дням окна, а не по торговым" - календарная суббота без баров
  уронила бы покрытие чистой ленты; краснит :func:`test_a_clean_tape_is_reported_ok`.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

import scripts._common as common
import scripts.check_tape as checker
from smc_zero.data_loader import MT5_FORMAT, PROJECT_FORMAT

#: Понедельник: три дня такой ленты торговые, поэтому покрытие считается честно.
START = "2025-01-06 00:00"
#: 288 пятиминутных баров в сутках (24 * 60 / 5).
BARS_PER_DAY = 288
#: Пороги теста: пропуск длиннее двенадцати баров сетки и половина дня как граница тонкого дня.
THRESHOLDS = checker.TapeThresholds(max_gap_bars=12, thin_day_ratio=0.5)


def _tape(days: int = 3, *, start: str = START, freq: str = "5min") -> pd.DataFrame:
    """Return ``days`` whole UTC days of a flat tape, bars ``freq`` apart."""
    per_day = int(pd.Timedelta(days=1) / pd.Timedelta(freq))
    index = pd.date_range(start, periods=days * per_day, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": index,
            "open": 1.1,
            "high": 1.1005,
            "low": 1.0995,
            "close": 1.1,
            "volume": 1,
        }
    )


def _write(path: Path, frame: pd.DataFrame, layout: str = PROJECT_FORMAT) -> Path:
    """Write ``frame`` as the project layout or as a raw MetaTrader 5 export."""
    times = frame["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    if layout == MT5_FORMAT:
        table = pd.DataFrame(
            {
                "symbol": "EURUSD",
                "timeframe": "M5",
                "time": times,
                "open": frame["open"],
                "high": frame["high"],
                "low": frame["low"],
                "close": frame["close"],
                "tick_volume": frame["volume"],
                "spread": 10,
                "real_volume": 0,
            }
        )
    else:
        table = frame.assign(datetime=frame["timestamp"]).drop(columns=["timestamp"])
    table.to_csv(path, index=False)
    return path


def _args(path: Path, **overrides: object) -> Namespace:
    """Return the parsed arguments of a check over ``path``, as the console face makes them."""
    argv = [
        "--symbol",
        path.stem.split("_")[0],
        "--timeframe",
        "M5",
        "--min-coverage",
        str(overrides.get("min_coverage", checker.DEFAULT_THRESHOLDS.min_coverage)),
        "--max-gap-bars",
        str(overrides.get("max_gap_bars", checker.DEFAULT_THRESHOLDS.max_gap_bars)),
    ]
    if overrides.get("json"):
        argv.append("--json")
    return checker.build_parser().parse_args(argv)


def test_a_clean_tape_is_reported_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Three whole trading days of M5 are 288 bars each: coverage 1.0 and no defect."""
    monkeypatch.setattr(common, "DATA_DIR", tmp_path)
    path = _write(tmp_path / "EURUSD_M5.csv", _tape())

    report = checker.inspect_tape(path, timeframe="M5", thresholds=THRESHOLDS)

    assert report.rows == 3 * BARS_PER_DAY
    assert report.trading_days == 3
    assert report.expected_bars == 3 * BARS_PER_DAY
    assert report.coverage == pytest.approx(1.0)
    assert report.off_grid == 0
    assert report.duplicates == 0
    assert report.ordered
    assert report.intraday_gaps == 0
    assert report.failures_against(THRESHOLDS) == ()
    assert report.lines(THRESHOLDS)[-1] == "вердикт    : OK"

    # The console face agrees with the direct call and returns the code of the verdict.
    assert checker.run(_args(path)) == 0


def test_a_duplicated_seam_is_a_failure(tmp_path: Path) -> None:
    """A repeated stamp (the seam of two exports) is a defect, though the loader would hide it."""
    frame = _tape(days=1)
    seam = pd.concat([frame, frame.iloc[[100]]], ignore_index=True).sort_values("timestamp")
    path = _write(tmp_path / "EURUSD_M5.csv", seam, layout=MT5_FORMAT)

    report = checker.inspect_tape(path, timeframe="M5", thresholds=THRESHOLDS)

    assert report.duplicates == 1
    assert report.rows == BARS_PER_DAY  # the canonical loader dropped the copy
    assert any("дубли" in problem for problem in report.failures_against(THRESHOLDS))


def test_an_off_grid_bar_is_a_failure(tmp_path: Path) -> None:
    """A bar two minutes off the M5 grid is named, not silently traded."""
    frame = _tape(days=1)
    frame.loc[10, "timestamp"] = frame.loc[10, "timestamp"] + pd.Timedelta(minutes=2)
    path = _write(tmp_path / "EURUSD_M5.csv", frame.sort_values("timestamp"))

    report = checker.inspect_tape(path, timeframe="M5", thresholds=THRESHOLDS)

    assert report.off_grid == 1
    assert any("вне сетки" in problem for problem in report.failures_against(THRESHOLDS))


def test_a_short_tape_fails_on_coverage(tmp_path: Path) -> None:
    """Two days of bars inside a ten day window are a truncated export, not a study window."""
    path = _write(tmp_path / "EURUSD_M5.csv", _tape(days=2))
    start = pd.Timestamp("2025-01-06", tz="UTC")

    report = checker.inspect_tape(
        path, timeframe="M5", start=start, end=start + pd.Timedelta(days=9), thresholds=THRESHOLDS
    )

    assert report.trading_days == 8
    assert report.coverage < 0.5
    assert any("покрытие" in problem for problem in report.failures_against(THRESHOLDS))


def test_a_weekend_pause_is_not_a_hole(tmp_path: Path) -> None:
    """The Friday-to-Monday pause is the market, not the export: the report stays quiet."""
    frame = pd.concat(
        [
            _tape(days=1, start="2025-01-03 00:00"),
            _tape(days=1, start="2025-01-06 00:00"),
        ],
        ignore_index=True,
    )
    path = _write(tmp_path / "EURUSD_M5.csv", frame)

    report = checker.inspect_tape(path, timeframe="M5", thresholds=THRESHOLDS)

    assert report.gaps == ()
    assert report.intraday_gaps == 0
    assert report.failures_against(THRESHOLDS) == ()


def test_an_intraday_hole_is_reported(tmp_path: Path) -> None:
    """A three hour hole inside a Monday is reported with the bar it starts before."""
    keep = _tape(days=1).drop(index=range(100, 136))  # 36 five minute bars = three hours
    path = _write(tmp_path / "EURUSD_M5.csv", keep.reset_index(drop=True))

    report = checker.inspect_tape(path, timeframe="M5", thresholds=THRESHOLDS)

    assert report.intraday_gaps == 1
    stamp, hours = report.gaps[0]
    assert stamp == pd.Timestamp("2025-01-06 11:20", tz="UTC")
    assert hours == pytest.approx(3.0 + 5 / 60)
    assert any("пропуск" in line for line in report.lines(THRESHOLDS))


def test_a_thin_day_lands_in_the_report(tmp_path: Path) -> None:
    """A holiday Monday of ten bars is named as thin with the day it thins."""
    frame = _tape(days=2)
    thinned = pd.concat([frame.iloc[:10], frame.iloc[BARS_PER_DAY:]], ignore_index=True)
    path = _write(tmp_path / "EURUSD_M5.csv", thinned)

    report = checker.inspect_tape(path, timeframe="M5", thresholds=THRESHOLDS)

    assert report.thin_days[0][1] == 10
    assert report.thin_days[0][0] == pd.Timestamp("2025-01-06", tz="UTC")
    assert any("тонкий день 2025-01-06" in line for line in report.lines(THRESHOLDS))


def test_the_mt5_layout_is_read_like_the_project_one(tmp_path: Path) -> None:
    """The raw export feeds the same check, and the report names the layout it read."""
    frame = _tape()
    project = _write(tmp_path / "project.csv", frame, layout=PROJECT_FORMAT)
    raw = _write(tmp_path / "raw.csv", frame, layout=MT5_FORMAT)

    one = checker.inspect_tape(project, timeframe="M5", thresholds=THRESHOLDS)
    two = checker.inspect_tape(raw, timeframe="M5", thresholds=THRESHOLDS)

    assert (one.rows, one.first, one.last) == (two.rows, two.first, two.last)
    assert one.layout == PROJECT_FORMAT
    assert two.layout == MT5_FORMAT
    assert "layout=mt5" in two.lines(THRESHOLDS)[0]


def test_the_json_face_carries_the_verdict(tmp_path: Path) -> None:
    """``--json`` is the same report as data: a failing tape says so in the payload."""
    frame = _tape(days=1)
    frame.loc[10, "timestamp"] = frame.loc[10, "timestamp"] + pd.Timedelta(minutes=2)
    path = _write(tmp_path / "EURUSD_M5.csv", frame.sort_values("timestamp"))
    report = checker.inspect_tape(path, timeframe="M5", thresholds=THRESHOLDS)

    payload = json.loads(json.dumps(report.as_dict(THRESHOLDS), ensure_ascii=False))

    assert payload["verdict"] == "FAIL"
    assert payload["off_grid"] == 1
    assert payload["rows"] == report.rows
    assert payload["failures"]


def test_run_returns_two_for_a_missing_or_unreadable_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wrong request is refused with code 2, never reported as a clean tape (rule 4)."""
    monkeypatch.setattr(common, "DATA_DIR", tmp_path)
    assert checker.run(_args(tmp_path / "EURUSD_M5.csv")) == 2

    broken = tmp_path / "EURUSD_M5.csv"
    broken.write_text("open,high,low,close\n1,1,1,1\n", encoding="utf-8")
    assert checker.run(_args(broken)) == 2


def test_the_source_flag_picks_the_mt5_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--data-source mt5`` reads ``SMC_DATA_DIR``, the way a runner does (Э11'.1)."""
    _write(tmp_path / "EURUSD_M5.csv", _tape(days=1), layout=MT5_FORMAT)
    monkeypatch.setenv(common.SMC_DATA_DIR_ENV, str(tmp_path))
    args = _args(tmp_path / "EURUSD_M5.csv")
    args.data_source = common.MT5_SOURCE

    assert checker.run(args) == 0

    monkeypatch.setenv(common.SMC_DATA_DIR_ENV, str(tmp_path / "nowhere"))
    assert checker.run(args) == 2


def test_the_hole_threshold_is_counted_in_bars_not_hours(tmp_path: Path) -> None:
    """An H4 tape steps four hours a bar: a one-hour threshold would call every step a hole."""
    path = _write(tmp_path / "EURUSD_H4.csv", _tape(days=5, freq="4h"))

    report = checker.inspect_tape(path, timeframe="H4", thresholds=THRESHOLDS)

    assert report.intraday_gaps == 0
    assert report.coverage == pytest.approx(1.0)
    assert "сетки H4" in report.lines(THRESHOLDS)[4]


def test_the_grid_check_uses_the_period_of_the_timeframe(tmp_path: Path) -> None:
    """The same stamps sit on the M5 grid and are off the H1 one."""
    path = _write(tmp_path / "EURUSD_M5.csv", _tape(days=1).iloc[::2].reset_index(drop=True))

    every_ten = checker.inspect_tape(path, timeframe="M5", thresholds=THRESHOLDS)
    hourly = checker.inspect_tape(path, timeframe="H1", thresholds=THRESHOLDS)

    assert every_ten.off_grid == 0
    assert every_ten.rows == BARS_PER_DAY // 2
    assert hourly.rows == BARS_PER_DAY // 2
    assert hourly.off_grid == BARS_PER_DAY // 2 - 24
    assert hourly.expected_bars == 24
    assert hourly.coverage == pytest.approx(BARS_PER_DAY / 2 / 24)

