"""Tests of the daily face of Э10'.4: the shell script a cron line calls at 22:00.

The script is started as a *process* by ``bash``, because its contract is with the shell: the
environment overrides it honours, the base of the tape it reads (Э11'.2), the report folder it
names, the exit code it forwards and the trades it prints.  Two tapes are used.  The 400 bar
synthetic walk of Э8' runs a real backtest over a temporary MT5 export folder (``SMC_DATA_DIR``),
while the work directory holds no ``./data`` at all: a run that drops ``--data-source mt5`` stops on
a missing tape.  A hand written ``trades.csv`` pins the filter itself, so the ``open_time >=``
boundary of the night is measured without waiting for a setup.

The mutations this file is one line away from, and the test each one must break (each one red on the
test named here and only there; m1-m5 measured on 2026-10-01, m6 and m7 on 2026-10-05):

* m1 "compare with ``>`` instead of ``>=``" - the trade opened exactly at the threshold is dropped,
  and a missed signal is invisible by construction; breaks
  :func:`test_the_check_prints_the_trades_from_the_night_threshold_on`;
* m2 "print the log without filtering" - four years of history arrive together with the signals of
  the day; breaks the same test;
* m3 "print the rows without the header of the log" - the printed list cannot be read against the
  columns of the journal; breaks
  :func:`test_the_daily_check_reads_the_mt5_export_and_prints_the_header`;
* m4 "leave the backtest in the background and stop the script" - the next step opens a report that
  is still being written; breaks the same test, which asserts the summary of the run and the count
  line, and both exist only after the run has finished;
* m5 "drop the executable bit" - the cron line of the guide calls the file directly and would fail
  before bash ever sees it; breaks :func:`test_the_script_is_executable_for_the_cron_line`;
* m6 "read the project tape base (drop ``--data-source mt5``)" - the work directory of the test has
  no ``./data``, so the run stops with code 2 and the message names ``data/EURUSD_M15.csv``; breaks
  :func:`test_the_daily_check_reads_the_mt5_export_and_prints_the_header` and
  :func:`test_the_check_forwards_the_refusal_code_of_the_runner`;
* m7 "assume the caller always sets ``SMC_DATA_DIR``" - a check without that variable (the run of a
  fresh machine, the suite) dies on the unbound variable under ``set -u`` before it prints anything;
  breaks every test that runs the script without a base, that is
  :func:`test_the_export_base_defaults_to_the_mt5_folder_of_the_home`,
  :func:`test_the_check_prints_the_trades_from_the_night_threshold_on` and
  :func:`test_the_threshold_of_the_night_defaults_to_22_00_msk`.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from smc_zero.backtester.engine import TRADE_COLUMNS

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "daily_signal_check.sh"
LIVE_CONFIG = REPO / "configs" / "live_eurusd_m15.yaml"
BARS = 400
START = "2026-06-08"
END = "2026-06-15"
#: The threshold of the hand written log: 19:00 UTC is 22:00 MSK of the same day.
NIGHT = "2026-06-09 19:00:00"


def _today() -> str:
    """Return the dated label of the report folder the script builds: the UTC day, ``YYYYMMDD``."""
    return time.strftime("%Y%m%d", time.gmtime())


def _yesterday_utc() -> str:
    """Return 19:00 UTC of yesterday - the ``yesterday 22:00 MSK`` default of the script."""
    stamp = datetime.now(UTC) - timedelta(days=1)
    return f"{stamp:%Y-%m-%d} 19:00:00"


def _write_tape(data_dir: Path, bars: int = BARS) -> None:
    """Write the 15 pip random walk of Э8' into ``data_dir``, in the schema of the loader."""
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
    frame.insert(0, "timestamp", pd.date_range(f"{START} 00:00", periods=bars, freq="15min", tz="UTC"))
    frame["datetime"] = frame["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    frame["volume"] = 1
    data_dir.mkdir(parents=True, exist_ok=True)
    frame[["datetime", "open", "high", "low", "close", "volume"]].to_csv(
        data_dir / "EURUSD_M15.csv", index=False
    )


def _write_trades(folder: Path, rows: list[tuple[str, str]]) -> Path:
    """Write a hand made trade log: the full column list of the engine and only the stamps filled.

    The stamps are written as text, exactly as ``trades.csv`` of a run carries them, so the filter
    reads the same bytes a live check reads.
    """
    folder.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({column: [""] * len(rows) for column in TRADE_COLUMNS})
    frame["open_time"] = [stamp for stamp, _side in rows]
    frame["side"] = [side for _stamp, side in rows]
    path = folder / "trades.csv"
    frame.to_csv(path, index=False)
    return path


def _check(*, drop: tuple[str, ...] = (), **overrides: str) -> subprocess.CompletedProcess[str]:
    """Run the script as a shell sees it: ``bash <script>`` with the override environment.

    ``drop`` removes variables the calling shell may carry (``SMC_DATA_DIR`` above all), so a test
    that measures a default cannot quietly read the export folder of the developer's machine.
    """
    env = {**os.environ, "SMC_PYTHON": sys.executable, **overrides}
    for name in drop:
        env.pop(name, None)
    return subprocess.run(
        ["bash", str(SCRIPT)], env=env, capture_output=True, text=True, check=False
    )


def test_the_daily_check_reads_the_mt5_export_and_prints_the_header(tmp_path: Path) -> None:
    """One run over the export: code 0, the dated report folder, the summary and the log header.

    The tape lies *only* in the export folder (``SMC_DATA_DIR``) and the work directory holds no
    ``./data``, so a run that reads the project base instead of ``--data-source mt5`` cannot start.
    """
    export = tmp_path / "export"
    _write_tape(export)
    reports = tmp_path / "reports"
    log = tmp_path / "daily.log"

    call = _check(
        SMC_WORKDIR=str(tmp_path),
        SMC_CONFIG=str(LIVE_CONFIG),
        SMC_DATA_DIR=str(export),
        SMC_REPORT_ROOT=str(reports),
        SMC_START=START,
        SMC_END=END,
        SMC_LOG=str(log),
        SMC_WAIT_TIMEOUT="120",
    )

    assert call.returncode == 0, call.stderr
    assert not (tmp_path / "data").exists()
    # The header of the run names the base it handed to the runner, so the log of the night can be
    # read on its own (rule 4: the data and the costs of a run come from named sources).
    assert f"  source   mt5 under {export}" in call.stdout
    dated = reports / f"daily_{_today()}"
    assert dated.is_dir(), call.stdout + call.stderr
    folder = dated / f"backtest_EURUSD_M15_{START}_{END}"
    assert (folder / "trades.csv").is_file()
    assert (folder / "summary.txt").is_file()
    # The list a human reads against the journal: the header of the log, printed by the script.
    assert ",".join(TRADE_COLUMNS) in call.stdout
    assert f"report   {folder}" in call.stdout
    assert "--- summary of the run ---" in call.stdout
    assert "signals:" in call.stdout
    # The run is backgrounded under nohup, so the pid and the two commands that follow it are shown.
    pid = next(line for line in call.stdout.splitlines() if line.startswith("  pid "))
    assert pid.split()[-1].isdigit()
    assert f"kill {pid.split()[-1]}" in call.stdout
    assert f"tail -5 {log}" in call.stdout
    assert "SMC backtest" in log.read_text(encoding="utf-8")


def test_the_check_prints_the_trades_from_the_night_threshold_on(tmp_path: Path) -> None:
    """Only the trades opened at or after the threshold are printed, and the boundary is included."""
    folder = tmp_path / "backtest_EURUSD_M15_2026-06-01_2026-06-15"
    _write_trades(
        folder,
        [
            ("2026-06-09 18:59:59+00:00", "long"),
            (f"{NIGHT}+00:00", "long"),
            ("2026-06-10 07:15:00+00:00", "short"),
        ],
    )

    call = _check(SMC_REPORT_FOLDER=str(folder), SMC_SINCE_UTC=NIGHT)

    assert call.returncode == 0, call.stderr
    assert ",".join(TRADE_COLUMNS) in call.stdout
    assert "2026-06-09 19:00:00+00:00" in call.stdout
    assert "2026-06-10 07:15:00+00:00" in call.stdout
    assert "2026-06-09 18:59:59+00:00" not in call.stdout
    assert "signals: 2" in call.stdout


def test_the_threshold_of_the_night_defaults_to_22_00_msk(tmp_path: Path) -> None:
    """Without an override the threshold is yesterday 22:00 MSK, and MSK is UTC+3 all year."""
    folder = tmp_path / "backtest_EURUSD_M15_2026-06-01_2026-06-15"
    _write_trades(folder, [(f"{_yesterday_utc()}+00:00", "long")])

    call = _check(SMC_REPORT_FOLDER=str(folder))

    assert call.returncode == 0, call.stderr
    assert f"open_time >= {_yesterday_utc()} UTC" in call.stdout
    assert "yesterday 22:00 MSK" in call.stdout
    assert "signals: 1" in call.stdout


def test_the_check_forwards_the_refusal_code_of_the_runner(tmp_path: Path) -> None:
    """A missing tape stops the run with its own code, and no report folder is made at all.

    The refusal names the export folder and not ``data/``: the message of the run shows which base
    the script asked for (Э11'.2).
    """
    export = tmp_path / "export"
    export.mkdir()
    reports = tmp_path / "reports"
    log = tmp_path / "daily.log"

    call = _check(
        SMC_WORKDIR=str(tmp_path),
        SMC_CONFIG=str(LIVE_CONFIG),
        SMC_DATA_DIR=str(export),
        SMC_REPORT_ROOT=str(reports),
        SMC_START=START,
        SMC_END=END,
        SMC_LOG=str(log),
        SMC_WAIT_TIMEOUT="120",
    )

    assert call.returncode == 2
    assert not reports.exists()
    refused = log.read_text(encoding="utf-8")
    assert f"no tape at {export}/EURUSD_M15.csv" in refused
    assert "point SMC_DATA_DIR at the export folder" in refused
    assert "error: the backtest exited with 2" in call.stderr


def test_the_export_base_defaults_to_the_mt5_folder_of_the_home(tmp_path: Path) -> None:
    """Without ``SMC_DATA_DIR`` the base is ``$HOME/_data/mt5`` - the default of the layer itself.

    The re-print mode runs no backtest, so the header of the script is the whole check here.
    """
    folder = tmp_path / "backtest_EURUSD_M15_2026-06-01_2026-06-15"
    _write_trades(folder, [(f"{NIGHT}+00:00", "long")])
    default = Path.home() / "_data" / "mt5"

    call = _check(drop=("SMC_DATA_DIR",), SMC_REPORT_FOLDER=str(folder), SMC_SINCE_UTC=NIGHT)

    assert call.returncode == 0, call.stderr
    assert f"  source   mt5 under {default}" in call.stdout


def test_the_script_is_executable_for_the_cron_line() -> None:
    """The cron line calls the file directly, so the executable bit and the shebang are the API."""
    assert SCRIPT.stat().st_mode & stat.S_IXUSR
    assert SCRIPT.read_text(encoding="utf-8").splitlines()[0] == "#!/usr/bin/env bash"
