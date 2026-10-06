"""Runner tests of the Э8' optimization face: the report of a *finished* study, not a study.

``run_optimization`` is stubbed with an :class:`~smc_zero.optimizer.OptunaResult` whose study,
parameters and fold tables are known by hand, so what is pinned here is the wiring of the runner:
the fold arithmetic it refuses a window for before the study starts, the four files it writes and
the trials it prints.  The real study is the business of ``tests/test_optimize*.py``.

The mutations the face is one line away from, and the test each one must break:

* m1 "write the fold tables but not ``best_params.json``" - the winner of a run is not auditable
  and its score cannot be reproduced; breaks
  :func:`test_the_optimization_runner_returns_zero_and_writes_its_report`;
* m2 "start the study on a window that cannot hold a fold" - every trial fails and the run burns
  ``--n-trials`` attempts to report nothing; breaks
  :func:`test_a_window_that_cannot_hold_a_fold_is_refused_before_the_study`;
* m3 "report the winner without re-running it" - the summary would show the study's score where a
  trade log belongs; breaks
  :func:`test_the_winner_is_reported_from_a_backtest_of_the_window`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import scripts._common as common
import scripts.run_optimization as runner
from smc_zero.backtester import FOLD_METRIC_FIELDS, aggregate_fold_metrics
from smc_zero.backtester.engine import TRADE_COLUMNS
from smc_zero.config import OptunaConfig, StrategyConfig, WalkForwardConfig
from smc_zero.optimizer import FoldEvaluation, OptunaResult

BARS = 300
START = "2026-06-08"
#: Three days of tape: the fold arithmetic of §7.10 п.61 needs more than one warm-up plus one test.
END = "2026-06-10"
#: The two folds of the stub: a quieter fit window and a weaker out-of-sample one.
PROFITS = ((1.0, 2.0), (0.5, 0.7))
#: What the stub study answers, and the score the runner has to print and store.
PARAMS: dict[str, Any] = {"sweep_buffer_pip": 3.0, "bias.agreement": "majority"}
SCORE = 1.25
#: The configuration the stub study's winner is: the searched value applied to the defaults.
WINNER = replace(StrategyConfig(), sweep_buffer_pip=float(PARAMS["sweep_buffer_pip"]))


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
    """Return a command line of the fixture: the tape's two days, two trials."""
    argv = [
        "--symbol",
        "EURUSD",
        "--timeframe",
        "M15",
        "--start",
        START,
        "--end",
        END,
        "--n-trials",
        "2",
        "--min-train-bars",
        "100",
        "--test-period-bars",
        "100",
        "--report-dir",
        str(report_dir),
    ]
    for name, value in overrides.items():
        argv += [f"--{name.replace('_', '-')}", value]
    return argv


def _table(profit: float) -> dict[str, float]:
    """Return one fold table shaped like :func:`~smc_zero.backtester.metrics.calc_metrics`."""
    return {
        "trades": 2.0,
        "profit": profit,
        "win_rate": 50.0,
        "pf": 1.1,
        "max_dd": 3.0,
        "sharpe": 0.9,
    }


@dataclass(frozen=True)
class _StubTrial:
    """One finished (or failed) trial of the stub study."""

    number: int
    value: float | None
    params: dict[str, Any]


@dataclass(frozen=True)
class _StubStudy:
    """The only face of a study the runner uses: its trials, in the order they ran."""

    trials: list[_StubTrial]


def _outcome(trials: list[_StubTrial] | None = None) -> OptunaResult:
    """Return the finished study the runner is stubbed with: two folds, both windows."""
    run = [_StubTrial(1, SCORE, dict(PARAMS))] if trials is None else trials
    train = [_table(profit) for profit in PROFITS[0]]
    test = [_table(profit) for profit in PROFITS[1]]
    return OptunaResult(
        study=_StubStudy(trials=run),
        best_params=dict(PARAMS),
        best_score=SCORE,
        best_trial_number=1,
        strategy=WINNER,
        best_evaluation=FoldEvaluation(
            fold_metrics_train=train,
            fold_metrics_test=test,
            train_aggregated=aggregate_fold_metrics(train),
            test_aggregated=aggregate_fold_metrics(test),
        ),
        config=OptunaConfig(n_trials=len(run)),
    )


def _folder(report_dir: Path) -> Path:
    """Return the report folder the fixture run must name."""
    return report_dir / f"optimization_EURUSD_M15_{START}_{END}_n2"


@pytest.fixture
def stubbed_study(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve the fixture tape and the finished study to the runner."""

    def load_csv(path: Path, *, drop_unclosed: bool = True) -> pd.DataFrame:
        """Return the fixture tape whatever path was asked for."""
        return _tape()

    monkeypatch.setattr(common, "load_csv", load_csv)
    monkeypatch.setattr(runner, "run_optimization", lambda *args, **kwargs: _outcome())



def test_the_optimization_runner_returns_zero_and_writes_its_report(
    tmp_path: Path, stubbed_study: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """A finished study becomes a report folder with the four files of the face."""
    report_dir = tmp_path / "reports"

    code = runner.main(_argv(report_dir))

    assert code == 0
    folder = _folder(report_dir)
    assert folder.is_dir()
    for name in ("best_params.json", "trades.csv", "trades.parquet", "summary.txt"):
        assert (folder / name).is_file(), name
    payload = json.loads((folder / "best_params.json").read_text(encoding="utf-8"))
    assert payload["best_params"] == PARAMS
    assert payload["best_score"] == pytest.approx(SCORE)
    assert payload["folds"] == 2
    # The report records how the score was computed: its metric and the weight of the decay gate.
    assert payload["score_metric"] == "sharpe"
    assert payload["penalty_power"] == pytest.approx(0.0)
    assert payload["test"]["profit_mean"] == pytest.approx(0.6)
    assert payload["train"]["profit_mean"] == pytest.approx(1.5)
    assert (folder / "summary.txt").read_text(encoding="utf-8").startswith("SMC backtest")
    printed = capsys.readouterr().out
    assert "trials: 1 run, 1 finished" in printed
    assert "best: trial #1, score +1.2500" in printed
    assert str(folder) in printed


def test_the_fold_table_of_the_report_holds_both_windows_and_their_means(
    tmp_path: Path, stubbed_study: None
) -> None:
    """One row per fold and window, plus the mean and sigma rows of each window."""
    report_dir = tmp_path / "reports"

    assert runner.main(_argv(report_dir)) == 0

    table = pd.read_csv(_folder(report_dir) / "fold_metrics.csv")
    assert {"fold", "window", *FOLD_METRIC_FIELDS} <= set(table.columns)
    assert set(table["window"]) == {"train", "test"}
    assert set(table["fold"]) == {"0", "1", "mean", "std"}
    means = table.loc[table["fold"] == "mean"]
    test_mean = float(means.loc[means["window"] == "test", "profit"].iloc[0])
    assert test_mean == pytest.approx(0.6)
    # The ``mean`` rows also carry the score inputs of §7.11 п.69: the aggregate profit of each
    # window and the ratio the decay gate weighs.
    for window in ("train", "test"):
        row = means.loc[means["window"] == window].iloc[0]
        assert float(row["train_profit_mean"]) == pytest.approx(1.5)
        assert float(row["test_profit_mean"]) == pytest.approx(0.6)
        assert float(row["degradation_ratio"]) == pytest.approx(0.4)
    # Every other row leaves them empty: a fold's own profit is its ``profit`` column, and the
    # spread rows are a reading of those columns, not of the run.
    other = table.loc[~table["fold"].eq("mean")]
    assert other["degradation_ratio"].isna().all()


def test_the_runner_prints_only_the_ten_best_finished_trials(
    tmp_path: Path,
    stubbed_study: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The console shows the head of the study: ten finished trials, the failed one left out."""
    trials = [_StubTrial(number, float(number), {"sweep_buffer_pip": 1.0}) for number in range(12)]
    trials.append(_StubTrial(99, None, {"sweep_buffer_pip": 2.0}))
    monkeypatch.setattr(runner, "run_optimization", lambda *args, **kwargs: _outcome(trials))

    assert runner.main(_argv(tmp_path / "reports")) == 0

    printed = capsys.readouterr().out
    lines = [line for line in printed.splitlines() if line.startswith("  #")]
    assert len(lines) == runner.TOP_TRIALS
    assert "trials: 13 run, 12 finished" in printed
    assert lines[0].startswith("  #11 ")
    assert " #99 " not in printed


def test_a_window_that_cannot_hold_a_fold_is_refused_before_the_study(
    tmp_path: Path, stubbed_study: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """100 bars cannot pay a 500 bar warm-up: the run stops instead of failing every trial."""
    calls: list[str] = []
    monkeypatch.setattr(runner, "run_optimization", lambda *args, **kwargs: calls.append("study"))

    code = runner.main(_argv(tmp_path / "reports", min_train_bars="500", test_period_bars="100"))

    assert code == 2
    assert calls == []


def test_an_optuna_less_environment_is_reported_not_traced(
    tmp_path: Path, stubbed_study: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the ``optimize`` extra the runner names the missing library."""

    def run_optimization(*args: Any, **kwargs: Any) -> Any:
        """Stand in for the Э7' entry point in an environment without optuna."""
        raise ImportError("No module named 'optuna'")

    monkeypatch.setattr(runner, "run_optimization", run_optimization)

    assert runner.main(_argv(tmp_path / "reports")) == 2


def test_the_winner_is_reported_from_a_backtest_of_the_window(
    tmp_path: Path, stubbed_study: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The summary is a trade log of the winner's configuration, not the study's score."""
    used: list[StrategyConfig] = []
    real = runner.build_intents

    def spy(
        ltf: pd.DataFrame,
        bias: pd.DataFrame,
        levels: pd.DataFrame,
        cfg: StrategyConfig | None = None,
    ) -> Any:
        """Record the configuration the report is armed with, then build it."""
        assert cfg is not None
        used.append(cfg)
        return real(ltf, bias, levels, cfg)

    monkeypatch.setattr(runner, "build_intents", spy)

    assert runner.main(_argv(tmp_path / "reports")) == 0

    assert used == [WINNER]
    folder = _folder(tmp_path / "reports")
    text = (folder / "summary.txt").read_text(encoding="utf-8")
    assert "initial capital" in text
    assert (folder / "trades.csv").read_text(encoding="utf-8").splitlines()[0] == ",".join(
        TRADE_COLUMNS
    )


def test_the_fold_windows_of_a_run_follow_its_entry_timeframe() -> None:
    """The M5 tape of the second hierarchy folds on the 60 / 30 day windows of §7.20.

    The M15 windows of §7.10 п.62 are 115 200 bars on a five minute tape - four times the bars of
    the M15 run they were reasoned about - and would leave a 1.4 year M5 tape with three folds
    instead of nine.  A count typed by the caller still wins over the windows of the timeframe.
    """
    m5 = runner.build_parser().parse_args(["--timeframe", "M5"])

    config = runner._walk_forward_config(m5, "M5")

    assert (config.min_train_bars, config.test_period_bars) == (288 * 60, 288 * 30)
    assert config.train_period_bars == 288 * 60
    # the M15 hierarchy keeps the windows it had
    assert runner._walk_forward_config(runner.build_parser().parse_args([]), "M15") == (
        WalkForwardConfig()
    )
    # a typed count outranks the windows of the timeframe, the rest keeps the timeframe's numbers
    typed = runner.build_parser().parse_args(["--min-train-bars", "500", "--test-period-bars", "100"])
    assert runner._walk_forward_config(typed, "M5") == WalkForwardConfig(
        min_train_bars=500, test_period_bars=100, train_period_bars=288 * 60
    )
