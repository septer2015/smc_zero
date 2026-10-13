"""Tests of the daily face of Э10'.4: the shell script a cron line calls at 22:00.

The script is started as a *process* by ``bash``, because its contract is with the shell: the
environment overrides it honours, the base of the tape it reads (Э11'.2), the report folder it
names, the folder its log lands in (Э14.1, measured by Э15.5), the exit code it forwards and the
trades it prints.  The parallel paper mode of Э17.z++ is pinned here too: beside the live run the
same night is measured with the candidate config, and every reading gets a JSON snapshot of its own
(``base_<day>.json`` and ``candidate_<day>.json`` under ``SMC_PAPER_DIR``).  Two tapes are used.
The 400 bar synthetic walk of Э8' runs a real backtest over a temporary MT5 export folder
(``SMC_DATA_DIR``), while the work directory holds no ``./data`` at all: a run that drops
``--data-source mt5`` stops on a missing tape.  A hand written ``trades.csv`` pins the filter itself,
so the ``open_time >=`` boundary of the night is measured without waiting for a setup.

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
  :func:`test_the_threshold_of_the_night_defaults_to_22_00_msk`;
* m8 "write the log into the system ``/tmp`` again (drop the repository root of the default)" - the
  artifact of the project leaves the project and dies with the next reboot, and a cron run writes it
  beside whatever folder it started in; breaks
  :func:`test_the_log_of_a_run_goes_to_the_tmp_folder_of_the_project` (measured on 2026-10-12);
* m9 "read the tape without its clock" (drop the ``source_clock`` key of the live config, or ignore
  the ``--source-clock`` it produces) - the run of the night would read a server tape as UTC and
  shift every session window by two or three hours (Э17.B); breaks
  :func:`test_the_live_config_declares_the_clock_of_the_raw_tape` and
  :func:`test_the_clock_of_the_night_comes_from_the_config`;
* m10 "point the candidate at the dated root of the base" (``--report-dir "${DATED}"`` for the
  candidate too) - the role has no report folder of its own, the snapshot of the second reading is
  never written and the check stops with code 1; breaks
  :func:`test_the_parallel_paper_run_writes_a_snapshot_of_each_reading` and, because the parallel
  mode is part of one check, :func:`test_the_daily_check_reads_the_mt5_export_and_prints_the_header`;
* m11 "give the candidate snapshot the config of the base" - both files of ``./tmp/paper`` would
  then name the live config, and the journal would compare a strategy with itself; breaks
  :func:`test_the_parallel_paper_run_writes_a_snapshot_of_each_reading`, which pins the two names
  (m10 and m11 measured on 2026-10-13).
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts._common as common
import scripts.run_backtest as runner
from scripts import paper_snapshot
from smc_zero.backtester.engine import TRADE_COLUMNS
from smc_zero.config import SOURCE_CLOCK_BROKER_SERVER, SessionConfig

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "daily_signal_check.sh"
LIVE_CONFIG = REPO / "configs" / "live_eurusd_m15.yaml"
#: The research file of Э17.z++ the parallel paper run of the night measures (Э17.z++.1).
CANDIDATE_CONFIG = REPO / "configs" / "research_e17zp_candidate.yaml"
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


def _check(
    *, cwd: Path | None = None, drop: tuple[str, ...] = (), **overrides: str
) -> subprocess.CompletedProcess[str]:
    """Run the script as a shell sees it: ``bash <script>`` with the override environment.

    ``cwd`` starts the script from somewhere else than the repository, the way a cron run starts in
    the home folder.  ``drop`` removes variables the calling shell may carry (``SMC_DATA_DIR`` above
    all), so a test that measures a default cannot quietly read the export folder of the developer's
    machine; the overrides of the test are applied after it, so an explicit value of a dropped name
    still wins over the default of the script.
    """
    env = {**os.environ, "SMC_PYTHON": sys.executable}
    for name in drop:
        env.pop(name, None)
    env.update(overrides)
    return subprocess.run(
        ["bash", str(SCRIPT)], env=env, cwd=cwd, capture_output=True, text=True, check=False
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


def test_the_log_of_a_run_goes_to_the_tmp_folder_of_the_project(tmp_path: Path) -> None:
    """The Э14.1 convention reaches the script: the log lands in ``<repo>/tmp``, not in ``/tmp``.

    The run is started from another folder than the repository - that is what cron does - so the
    default path has to be built from the repository root of the script and never from the working
    directory.  The folder of the path is created on demand, and an explicit ``SMC_LOG`` wins.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    export = tmp_path / "export"
    _write_tape(export)
    runs = {
        "SMC_REPO": str(repo),
        "SMC_WORKDIR": str(repo),
        "SMC_CONFIG": str(LIVE_CONFIG),
        "SMC_DATA_DIR": str(export),
        "SMC_REPORT_ROOT": str(repo / "reports"),
        "SMC_START": START,
        "SMC_END": END,
        "SMC_WAIT_TIMEOUT": "120",
        # ``repo`` is a stand-in without ``configs/``: it carries no candidate file of Э17.z++, so
        # this test measures the log of the *live* run and not the parallel paper mode (the two
        # configs of the night have tests of their own).
        "SMC_PAPER": "0",
    }

    call = _check(cwd=tmp_path, drop=("SMC_LOG",), **runs)

    assert call.returncode == 0, call.stderr
    log = repo / "tmp" / f"daily_backtest_{_today()}.log"
    assert log.is_file()
    assert f"  log      {log}" in call.stdout
    assert "SMC backtest" in log.read_text(encoding="utf-8")
    # Nothing lands beside the folder the caller started in: the default is not a working-directory
    # path, which is exactly the defect of the system ``/tmp`` default (m8).
    assert not (tmp_path / "tmp").exists()

    chosen_log = tmp_path / "explicit.log"
    chosen = _check(cwd=tmp_path, drop=("SMC_LOG",), SMC_LOG=str(chosen_log), **runs)

    assert chosen.returncode == 0, chosen.stderr
    assert chosen_log.is_file(), chosen.stdout + chosen.stderr
    assert f"  log      {chosen_log}" in chosen.stdout


def test_the_live_config_declares_the_clock_of_the_raw_tape() -> None:
    """Режим A читает ленту брокерского сервера: конфиг объявляет это, а не угадывает (Э17.B).

    Проверка идёт по самому файлу конфига: раннер узнаёт о поясе оттуда, и потеря ключа в
    поставляемом YAML сдвинула бы окна сессий на два-три часа незаметно для прогона.
    """
    cfg = common.load_config(LIVE_CONFIG)
    args = runner.build_parser().parse_args(["--config-path", str(LIVE_CONFIG)])

    assert cfg[common.SOURCE_CLOCK_KEY] == SOURCE_CLOCK_BROKER_SERVER
    assert common.resolve_source_clock(args) == SOURCE_CLOCK_BROKER_SERVER

    # типизированный флаг перекрывает файл, а неизвестный пояс отвергается (правило 5)
    override = runner.build_parser().parse_args(
        ["--config-path", str(LIVE_CONFIG), "--source-clock", "utc"]
    )
    assert common.resolve_source_clock(override) == "utc"
    with pytest.raises(ValueError, match="source_clock"):
        common.resolve_source_clock(args, {"source_clock": "msk"})


def test_the_clock_of_the_night_comes_from_the_config(tmp_path: Path) -> None:
    """Штатный запуск печатает пояс из конфига, а ``SMC_SOURCE_CLOCK`` его перекрывает (Э17.B)."""
    folder = tmp_path / "backtest_EURUSD_M15_2026-06-01_2026-06-15"
    _write_trades(folder, [(f"{NIGHT}+00:00", "long")])

    live = _check(SMC_REPORT_FOLDER=str(folder), SMC_SINCE_UTC=NIGHT)

    assert live.returncode == 0, live.stderr
    assert "  clock    broker_server (source_clock конфига)" in live.stdout

    overridden = _check(
        SMC_REPORT_FOLDER=str(folder), SMC_SINCE_UTC=NIGHT, SMC_SOURCE_CLOCK="utc"
    )

    assert overridden.returncode == 0, overridden.stderr
    assert "  clock    utc (SMC_SOURCE_CLOCK, перекрывает конфиг)" in overridden.stdout


def test_the_candidate_config_is_the_live_config_with_the_two_shifts() -> None:
    """Файл кандидата Э17.z++ двигает ровно две вещи: окна сессий на −3 ч и суточный лимит уровня.

    Проверка идёт по двум файлам рядом: исследовательский конфиг обязан отличаться от боевого
    только ими, иначе бумажное чтение ночи сравнивало бы не кандидата, а третью стратегию.
    """
    live = common.load_config(LIVE_CONFIG)
    candidate = common.load_config(CANDIDATE_CONFIG)

    assert candidate["symbol"] == live["symbol"]
    assert candidate["timeframe"] == live["timeframe"]
    assert candidate["source_clock"] == live["source_clock"] == SOURCE_CLOCK_BROKER_SERVER
    assert candidate["broker"] == live["broker"]
    assert candidate["backtest"] == live["backtest"]

    # Блок стратегии отличается единственным ключом: суточный лимит настроек на уровень.
    candidate_strategy = dict(candidate["strategy"])
    assert candidate_strategy.pop("max_setups_per_level_per_day") == 3
    assert candidate_strategy == live["strategy"]
    # Боевой файл собственного значения не несёт: в коде остаётся поставленный дефолт 1.
    assert "max_setups_per_level_per_day" not in live["strategy"]
    assert "session" not in live

    # Сдвиг: каждое окно кандидата — поставленное окно минус три часа (полдень MSK и так далее).
    shipped = SessionConfig()
    assert set(candidate["session"]) == {
        "london_summer_msk",
        "london_winter_msk",
        "ny_summer_msk",
        "ny_winter_msk",
    }
    for name, window in candidate["session"].items():
        start, end = getattr(shipped, name)
        assert window == [start - 3, end - 3]


def test_the_parallel_paper_run_writes_a_snapshot_of_each_reading(tmp_path: Path) -> None:
    """Э17.z++: рядом с базовым прогоном ночь измеряет кандидат, и у каждого чтения свой снимок.

    Оба чтения идут по одной ленте, одному поясу и одному порогу ночи — различает их только
    конфиг. Снимок каждого ложится отдельным файлом в ``SMC_PAPER_DIR``: два чтения одной ночи
    нельзя записать в один файл, иначе журнал потеряет базовое.
    """
    export = tmp_path / "export"
    _write_tape(export)
    paper = tmp_path / "paper"
    base_json = paper / f"base_{_today()}.json"
    candidate_json = paper / f"candidate_{_today()}.json"

    call = _check(
        SMC_WORKDIR=str(tmp_path),
        SMC_CONFIG=str(LIVE_CONFIG),
        SMC_DATA_DIR=str(export),
        SMC_REPORT_ROOT=str(tmp_path / "reports"),
        SMC_START=START,
        SMC_END=END,
        SMC_LOG=str(tmp_path / "daily.log"),
        SMC_PAPER_DIR=str(paper),
        SMC_WAIT_TIMEOUT="600",
    )

    assert call.returncode == 0, call.stderr
    base = json.loads(base_json.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_json.read_text(encoding="utf-8"))

    assert base["role"] == "base"
    assert base["config"] == str(LIVE_CONFIG)
    assert candidate["role"] == "candidate"
    assert candidate["config"] == str(CANDIDATE_CONFIG)
    # Два прогона — две папки отчёта: сравнивать два чтения по одному отчёту невозможно.
    assert base["report"] != candidate["report"]
    assert (Path(candidate["report"]) / "trades.csv").is_file()
    # Одна лента, один пояс, один порог ночи и одно окно: различается только стратегия.
    assert candidate["since_utc"] == base["since_utc"]
    assert candidate["window"] == base["window"] == {"start": START, "end": END}
    assert "  paper    base + candidate=" in call.stdout
    assert f"  paper    candidate -> {candidate_json}" in call.stdout
    assert "--- paper journal: docs/PAPER_JOURNAL.md" in call.stdout


def test_the_parallel_run_is_off_with_smc_paper_zero(tmp_path: Path) -> None:
    """``SMC_PAPER=0`` оставляет проверку ровно тем, чем она была: один боевой прогон.

    Это сухой прогон живого конфига: ни кандидатского отчёта, ни папки снимков не появляется,
    и заголовок скрипта о параллельном чтении не сообщает.
    """
    export = tmp_path / "export"
    _write_tape(export)
    paper = tmp_path / "paper"

    call = _check(
        SMC_WORKDIR=str(tmp_path),
        SMC_CONFIG=str(LIVE_CONFIG),
        SMC_DATA_DIR=str(export),
        SMC_REPORT_ROOT=str(tmp_path / "reports"),
        SMC_START=START,
        SMC_END=END,
        SMC_LOG=str(tmp_path / "daily.log"),
        SMC_PAPER="0",
        SMC_PAPER_DIR=str(paper),
        SMC_WAIT_TIMEOUT="300",
    )

    assert call.returncode == 0, call.stderr
    assert not paper.exists()
    assert "  paper    " not in call.stdout


def test_the_paper_snapshot_maps_the_report_and_the_night_boundary(tmp_path: Path) -> None:
    """Читатель отчёта: порог ночи включается на границе, метрики берутся из ``summary.txt``."""
    folder = tmp_path / "backtest_EURUSD_M15_2026-06-01_2026-06-15"
    _write_trades(
        folder,
        [("2026-06-09 18:59:59+00:00", "long"), (f"{NIGHT}+00:00", "short")],
    )
    (folder / "summary.txt").write_text(
        "SMC backtest: EURUSD M15\n"
        "  trades                   2\n"
        "  win rate                 50.00 %\n"
        "  profit                   +12.34\n",
        encoding="utf-8",
    )
    out = tmp_path / "snapshot.json"

    code = paper_snapshot.main(
        [
            "--report",
            str(folder),
            "--role",
            "base",
            "--config",
            str(LIVE_CONFIG),
            "--since",
            NIGHT,
            "--date",
            "2026-06-10",
            "--out",
            str(out),
        ]
    )

    document = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0
    assert document["date"] == "2026-06-10"
    assert document["signal_count"] == 1
    assert [row["open_time"] for row in document["signals"]] == [f"{NIGHT}+00:00"]
    assert document["metrics"]["trades"] == 2.0
    assert document["metrics"]["profit"] == 12.34
    assert document["report"] == str(folder)


def test_the_snapshot_refuses_a_report_without_a_trade_log(tmp_path: Path) -> None:
    """Пустая папка — отказ с кодом 2, а не пустой снимок: молчание сходило бы за «нет сигнала»."""
    code = paper_snapshot.main(
        [
            "--report",
            str(tmp_path / "no_such_report"),
            "--role",
            "candidate",
            "--config",
            str(CANDIDATE_CONFIG),
            "--since",
            NIGHT,
            "--out",
            str(tmp_path / "snapshot.json"),
        ]
    )

    assert code == 2
    assert not (tmp_path / "snapshot.json").exists()

