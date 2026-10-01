"""Runner tests of the Э8' backtest face: the window, the report folder and the two files.

``load_csv`` of :mod:`scripts._common` is stubbed, so what is pinned here is the *wiring* of the
runner - the slice it cuts before the markup is built, the report folder it names from the
arguments, the files it writes there and the codes it returns - not the indicator layer, which has
tests of its own.  The tape is a seeded random walk of 100 closed M15 bars from a Monday 00:00, the
same recipe as the Э7' integration fixture: it carries no sweep -> CHoCH -> FVG setup, and the
tests never ask it for a profit.

The mutations the face is one line away from, and the test each one must break:

* m1 "raise instead of returning a code" - the console contract is gone and CI shows a traceback
  instead of a red test; breaks
  :func:`test_the_backtest_runner_returns_zero_and_writes_its_report`;
* m2 "build the markup and simulate the whole file, not the window" - the report covers bars the
  caller did not ask for; breaks :func:`test_the_report_holds_the_window_and_not_the_whole_file`;
* m3 "guess a price list for an unpriced symbol" - a run without C6 costs reads a curve as a
  result, exactly what rule 4 forbids; breaks
  :func:`test_a_symbol_without_a_price_list_is_refused`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts._common as common
import scripts.run_backtest as runner
from smc_zero.backtester.engine import TRADE_COLUMNS
from smc_zero.config import StrategyConfig

BARS = 100
#: The tape starts at a Monday 00:00 UTC, so its 100 bars reach into the next calendar day.
START = "2026-06-08"
NEXT_DAY = "2026-06-09"


def _tape(bars: int = BARS) -> pd.DataFrame:
    """Return the fixture tape: a 15 pip random walk of closed M15 bars, all flags set."""
    rng = np.random.default_rng(7)
    close = 1.1000 + np.cumsum(rng.normal(0.0, 0.0015, bars))
    open_ = np.concatenate(([1.1000], close[:-1]))
    frame = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + 0.0007,
            "low": np.minimum(open_, close) - 0.0007,
            "close": close,
        }
    )
    frame.insert(
        0, "timestamp", pd.date_range("2026-06-08 00:00", periods=bars, freq="15min", tz="UTC")
    )
    frame["volume"] = 1
    frame["is_closed"] = True
    return frame


def _argv(report_dir: Path, **overrides: str) -> list[str]:
    """Return a command line of the fixture, with the two window days of the tape."""
    argv = [
        "--symbol",
        "EURUSD",
        "--timeframe",
        "M15",
        "--start",
        START,
        "--end",
        NEXT_DAY,
        "--report-dir",
        str(report_dir),
    ]
    for name, value in overrides.items():
        argv += [f"--{name.replace('_', '-')}", value]
    return argv


@pytest.fixture
def stubbed_tape(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve the fixture tape to the runner instead of reading ``./data``."""

    def load_csv(path: Path, *, drop_unclosed: bool = True) -> pd.DataFrame:
        """Return the fixture tape whatever path was asked for."""
        assert path.name == "EURUSD_M15.csv"
        return _tape()

    monkeypatch.setattr(common, "load_csv", load_csv)


def test_the_backtest_runner_returns_zero_and_writes_its_report(
    tmp_path: Path, stubbed_tape: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run names its window, writes both files and returns 0."""
    report_dir = tmp_path / "reports"

    code = runner.main(_argv(report_dir))

    assert code == 0
    folder = report_dir / f"backtest_EURUSD_M15_{START}_{NEXT_DAY}"
    assert folder.is_dir()
    log = folder / "trades.csv"
    assert log.read_text(encoding="utf-8").splitlines()[0] == ",".join(TRADE_COLUMNS)
    assert (folder / "trades.parquet").is_file()
    text = (folder / "summary.txt").read_text(encoding="utf-8")
    assert "trades" in text
    assert "profit" in text
    printed = capsys.readouterr().out
    assert printed.splitlines()[0].startswith("SMC backtest: EURUSD M15")
    assert str(folder) in printed


def test_the_report_holds_the_window_and_not_the_whole_file(
    tmp_path: Path, stubbed_tape: None
) -> None:
    """A window of one day stops at its own last closed bar, not at the file's."""
    report_dir = tmp_path / "reports"

    code = runner.main(_argv(report_dir, end=START))

    assert code == 0
    text = (report_dir / f"backtest_EURUSD_M15_{START}_{START}" / "summary.txt").read_text(
        encoding="utf-8"
    )
    # The 96 bars of 2026-06-08 close at 00:00 of the 9th; the tape's own last bar closes at 01:00.
    assert ".. 2026-06-09 00:00 UTC" in text
    assert "2026-06-09 01:00" not in text


def test_the_summary_of_a_run_lists_the_costs_it_was_charged(
    tmp_path: Path, stubbed_tape: None
) -> None:
    """Rule 4 on the page: the cost assumptions of the run are printed, and a costed run is clear."""
    report_dir = tmp_path / "reports"

    assert runner.main(_argv(report_dir)) == 0

    text = (report_dir / f"backtest_EURUSD_M15_{START}_{NEXT_DAY}" / "summary.txt").read_text(
        encoding="utf-8"
    )
    assert "costs" in text
    # The runner trades the Alfa row and charges the shipped profile, so all three costs are on the
    # page and rule 4 has nothing to stamp (Э10').  The uncosted run is the one that builds a zero
    # profile on purpose, and that one is stamped (see tests/test_reports.py).
    assert StrategyConfig().risk.has_costs is True
    assert "spread 1.4 pip" in text
    assert "slippage 0.20 pip per market leg, commission 7.00 per lot" in text
    assert "RiskConfig.has_costs is False" not in text


def test_a_missing_tape_is_refused_without_creating_a_report(tmp_path: Path) -> None:
    """A symbol with no CSV stops the runner with a code and writes nothing."""
    report_dir = tmp_path / "reports"

    code = runner.main(_argv(report_dir, symbol="NZDUSD"))

    assert code == 2
    assert not report_dir.exists()


def test_an_empty_window_is_refused(tmp_path: Path, stubbed_tape: None) -> None:
    """A window outside the tape is an error, not a flat equity curve."""
    report_dir = tmp_path / "reports"

    code = runner.main(_argv(report_dir, start="2030-01-01", end="2030-01-02"))

    assert code == 2
    assert not report_dir.exists()


def test_a_symbol_without_a_price_list_is_refused(tmp_path: Path, stubbed_tape: None) -> None:
    """A symbol the C6 table does not price cannot be simulated for free (rule 4)."""
    report_dir = tmp_path / "reports"

    code = runner.main(_argv(report_dir, symbol="USDJPY"))

    assert code == 2
    assert not report_dir.exists()
