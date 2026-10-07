"""End-to-end tests of the Э8' console faces: real processes, real tapes, real exit codes.

Both runners are started as *processes* (``python -m scripts.run_backtest``), because what is tested
here is the command line itself: the argparse contract, the ``./data`` and ``./reports`` paths
relative to the working directory, the entry points of ``pyproject.toml`` and the code a shell
sees.  Two tapes are used:

* the shipped ``./data/EURUSD_M15.csv``, five days of it - the first command a user runs;
* a synthetic tape written into a temporary ``./data``: the same runs without the 5.7 MB file and
  with folds small enough to be paid in seconds (§7.10 п.61: a tape of ``n`` bars holds
  ``(n - min_train_bars) // test_period_bars`` folds).

The mutation this file is one line away from, and the test it must break:

* m1 "ask for an argument the runner does not define" - the command line is broken; both
  end-to-end tests go red, because argparse exits with code 2 where the shell expects 0.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BARS = 400
START = "2026-06-08"
END = "2026-06-15"


def _tape(bars: int = BARS) -> pd.DataFrame:
    """Return the fixture tape: a 15 pip random walk of closed M15 bars from a Monday 00:00."""
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
        0, "timestamp", pd.date_range(f"{START} 00:00", periods=bars, freq="15min", tz="UTC")
    )
    frame["volume"] = 1
    frame["is_closed"] = True
    return frame


def _environment() -> dict[str, str]:
    """Return the environment of a subprocess: ``src`` and the repository root on the path."""
    path = os.pathsep.join([str(REPO / "src"), str(REPO), os.environ.get("PYTHONPATH", "")])
    return {**os.environ, "PYTHONPATH": path}


def _run(module: str, argv: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run one runner as a process from ``cwd`` and return the completed call."""
    return subprocess.run(
        [sys.executable, "-m", module, *argv],
        cwd=str(cwd),
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )


def _write_tape(data_dir: Path, bars: int = BARS) -> Path:
    """Write the fixture tape into ``data_dir`` in the loader's CSV schema."""
    frame = _tape(bars)
    frame["datetime"] = frame["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "EURUSD_M15.csv"
    frame[["datetime", "open", "high", "low", "close", "volume"]].to_csv(path, index=False)
    return path


def test_the_backtest_script_runs_end_to_end_on_the_real_tape(tmp_path: Path) -> None:
    """The first command of a user: five days of the shipped tape, report written, code 0."""
    tape = REPO / "data" / "EURUSD_M15.csv"
    assert tape.is_file(), f"{tape} is part of the repository"
    report_dir = tmp_path / "reports"

    call = _run(
        "scripts.run_backtest",
        ["--symbol", "EURUSD", "--start", "2022-08-15", "--end", "2022-08-20",
         "--report-dir", str(report_dir)],
        cwd=REPO,
    )

    assert call.returncode == 0, call.stderr
    folder = report_dir / "backtest_EURUSD_M15_2022-08-15_2022-08-20"
    assert (folder / "trades.csv").is_file()
    assert (folder / "summary.txt").is_file()
    assert call.stdout.splitlines()[0].startswith("SMC backtest: EURUSD M15")


def test_the_backtest_script_runs_end_to_end_on_a_synthetic_tape(tmp_path: Path) -> None:
    """The same command over a temporary ``./data``, with nothing of the repository in it."""
    _write_tape(tmp_path / "data")
    report_dir = tmp_path / "reports"

    call = _run(
        "scripts.run_backtest",
        ["--symbol", "EURUSD", "--start", START, "--end", END, "--report-dir", str(report_dir)],
        cwd=tmp_path,
    )

    assert call.returncode == 0, call.stderr
    folder = report_dir / f"backtest_EURUSD_M15_{START}_{END}"
    assert (folder / "trades.csv").is_file()
    assert (folder / "summary.txt").is_file()


def test_the_optimization_script_runs_end_to_end_on_a_synthetic_tape(tmp_path: Path) -> None:
    """Two trials over a 400 bar tape: the four report files of the face, code 0."""
    _write_tape(tmp_path / "data")
    report_dir = tmp_path / "reports"

    call = _run(
        "scripts.run_optimization",
        ["--symbol", "EURUSD", "--start", START, "--end", END, "--n-trials", "2",
         "--min-train-bars", "100", "--test-period-bars", "100",
         "--report-dir", str(report_dir)],
        cwd=tmp_path,
    )

    assert call.returncode == 0, call.stderr
    folder = report_dir / f"optimization_EURUSD_M15_{START}_{END}_n2"
    for name in ("best_params.json", "trades.csv", "summary.txt", "fold_metrics.csv"):
        assert (folder / name).is_file(), name
    payload = json.loads((folder / "best_params.json").read_text(encoding="utf-8"))
    assert payload["n_trials"] == 2
    assert payload["folds"] >= 1
    assert payload["best_params"]


def test_the_entry_points_of_the_project_point_at_the_runners() -> None:
    """``pyproject.toml`` registers the console scripts of the project, all of them callable."""
    with (REPO / "pyproject.toml").open("rb") as handle:
        targets = tomllib.load(handle)["project"]["scripts"]

    assert targets == {
        "smc-backtest": "scripts.run_backtest:main",
        "smc-optimize": "scripts.run_optimization:main",
        "smc-check-tape": "scripts.check_tape:main",
    }
    for target in targets.values():
        module, _, attribute = target.partition(":")
        assert callable(getattr(importlib.import_module(module), attribute))

