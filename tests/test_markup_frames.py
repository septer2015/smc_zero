"""Tests of the readable HTF tapes of the second hierarchy (§7.20, В2-B) on synthetic files.

The second hierarchy reads its H4 / D1 legs from their own CSVs over ``[start - warmup_days, end]``
instead of resampling them out of the entry tape; the v1 hierarchy keeps the resample.  The tests
here serve both tapes from memory and watch which files were asked for, so "loaded, not resampled"
is pinned by the loader calls themselves and not by a value that could coincide.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import scripts._common as common
from smc_zero.config import HIERARCHY_PRESETS, TimeframeConfig
from smc_zero.data_loader import TIMESTAMP_COLUMN

#: The fixture window: 60 days of warm-up before the first day of the run.
START = "2025-05-06"
WARMUP_DAYS = 60
#: The first day of both fixture tapes.
FIRST_BAR = "2025-01-01"
#: The working timeframes of the two hierarchies.
M5_PRESET = HIERARCHY_PRESETS["H4_M15_M5"].timeframes
V1_PRESET = TimeframeConfig()


def _tape(first: str, periods: int, *, freq: str, value: float) -> pd.DataFrame:
    """One synthetic tape: ``periods`` closed bars of ``freq`` from ``first``."""
    stamps = pd.date_range(first, periods=periods, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            TIMESTAMP_COLUMN: stamps,
            "open": value,
            "high": value + 0.0010,
            "low": value - 0.0010,
            "close": value,
            "volume": 1,
            "is_closed": True,
        }
    )


@pytest.fixture
def stubbed_htf(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Serve the H4 / D1 fixture tapes instead of reading ``./data``; return the asked names."""
    tapes = {
        "EURUSD_H4.csv": _tape(f"{FIRST_BAR} 00:00", 400 * 6, freq="4h", value=1.10),
        "EURUSD_D1.csv": _tape(f"{FIRST_BAR} 00:00", 400, freq="1D", value=1.10),
    }
    asked: list[str] = []

    def load_csv(path: Path, *, drop_unclosed: bool = True) -> pd.DataFrame:
        """Return the tape of the file the runner asked for, remembering its name."""
        asked.append(path.name)
        return tapes[path.name]

    monkeypatch.setattr(common, "load_csv", load_csv)
    return asked


def _window(start: str = START, end: str = "2025-08-01") -> common.RunWindow:
    """The window of the fixture: ``warmup_days`` of history before ``start``."""
    return common.RunWindow(
        start=common.read_day(start), end=common.read_day(end), warmup_days=WARMUP_DAYS
    )


def test_the_htf_files_are_loaded_not_resampled_for_h4_m15_m5(stubbed_htf: list[str]) -> None:
    """The second hierarchy reads H4 and D1 from their own files, and asks for both."""
    frames = common.markup_frames("EURUSD", M5_PRESET, _window())

    assert stubbed_htf == ["EURUSD_H4.csv", "EURUSD_D1.csv"]
    assert set(frames) == {"H4", "D1"}
    # the frames are the files: their own grid and their own value, nothing built here
    assert frames["H4"][TIMESTAMP_COLUMN].diff().dropna().min() == pd.Timedelta(hours=4)
    assert bool((frames["H4"]["close"] == 1.10).all())
    assert frames["D1"][TIMESTAMP_COLUMN].diff().dropna().min() == pd.Timedelta(days=1)


def test_v1_still_resamples_from_the_entry_frame(stubbed_htf: list[str]) -> None:
    """The v1 hierarchy answers no HTF frames: the cache keeps resampling them itself."""
    frames = common.markup_frames("EURUSD", V1_PRESET, _window())

    assert frames == {}
    assert stubbed_htf == []


def test_the_warmup_period_extends_the_loading_range(stubbed_htf: list[str]) -> None:
    """The tapes are read from ``start - warmup_days``: that is what makes the window warm (m5)."""
    window = _window()

    frames = common.markup_frames("EURUSD", M5_PRESET, window)

    assert window.loaded_from == common.read_day("2025-03-07")
    for frame in frames.values():
        assert frame[TIMESTAMP_COLUMN].min() == window.loaded_from
        # ``slice_window`` keeps the whole end day, exactly like every other window of the runners
        assert frame[TIMESTAMP_COLUMN].max() < window.end + pd.Timedelta(days=1)


def test_a_window_before_the_htf_tape_is_refused(stubbed_htf: list[str]) -> None:
    """A warm-up reaching past the first bar of a tape is refused with the earliest start (m6)."""
    window = _window(start="2025-01-15")

    with pytest.raises(ValueError, match=r"нужен --start >= 2025-03-02"):
        common.markup_frames("EURUSD", M5_PRESET, window)


def test_a_missing_htf_file_names_the_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A tape that is not there is a ``FileNotFoundError`` naming the file, not an empty frame."""
    monkeypatch.setattr(common, "DATA_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="EURUSD_H4.csv"):
        common.markup_frames("EURUSD", M5_PRESET, _window())
