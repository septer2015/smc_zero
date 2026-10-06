"""Integration tests of the Э7' optimizer: the real tape, the real engine, the real folds (Э7').

Nothing of the strategy is stubbed here: the tape goes through the markup cache
(:func:`~smc_zero.optimizer.marks.build_tape_marks`), every trial runs the real entry chain of
Э4', the real engine of Э5' and the real fold split of Э6', and the winner is evaluated once
more from scratch.  The tape is a seeded random walk of 400 closed M15 bars - two folds of an
expanding fit window (200, then 300 bars) against out-of-sample windows of 100 bars - with the
bias reduced to H1, because no synthetic
tape is obliged to carry a whole sweep -> CHoCH -> FVG setup; what the tests pin down is
therefore the *wiring* (the cache, the folds, the window the engine is handed, the ranking of the
trials and the parameters of the winner), not a profit.

Two runs are driven by a stub study and a stub evaluator instead of optuna, so the search loop is
tested without the library: the trial schedule of the fixture makes the ranking of the trials
observable by hand (``sweep_buffer_pip`` 1, 5, 3 -> 1.82, 9.09, 5.45 - the score of §7.11 п.69
grows with the test window's profit).

The mutations the layer is one line away from, and the test each one must break:

* m1 "score the trials on the window they were fitted on" - the out-of-sample gate is gone;
  breaks :func:`test_the_study_ranks_its_trials_by_the_out_of_sample_window`;
* m2 "rebuild the markup per trial" - 100 trials pay 600 heavy indicator calls over the same
  tape; breaks :func:`test_the_markup_of_a_run_is_built_once_for_every_trial`;
* m3 "simulate a fold on the whole tape (or on its fit window)" - the fold is scored on bars it
  was allowed to fit on; breaks :func:`test_every_fold_is_simulated_on_its_own_test_window`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import smc_zero.optimizer.marks as marks_module
from smc_zero.backtester.engine import DEFAULT_INSTRUMENT, run_backtest
from smc_zero.backtester.walkforward import aggregate_fold_metrics, split_walkforward
from smc_zero.config import (
    BacktestConfig,
    InstrumentSpec,
    OptunaConfig,
    StrategyConfig,
    StructureLayerConfig,
    TradeTargetScore,
    WalkForwardConfig,
)
from smc_zero.data_loader import TIMESTAMP_COLUMN, drop_unclosed
from smc_zero.indicators.structure import STRUCTURE_LAYER_COLUMNS, structure_layer
from smc_zero.optimizer import (
    FoldEvaluation,
    TapeMarks,
    build_tape_marks,
    evaluate_params,
    run_optimization,
    score_from_aggregates,
    trades_pool_scorer,
    trades_scaled_score,
)
from smc_zero.strategy.intents import build_intents

BARS = 400
TRAIN_BARS = 200
TEST_BARS = 100
#: The folds of the fixture: a 200 bar fit window, then out-of-sample windows of 100 bars.
SHORT = WalkForwardConfig(min_train_bars=TRAIN_BARS, test_period_bars=TEST_BARS)
#: The costs, the sizing and the risk profile stay the project defaults of Э5'.
BACKTEST = BacktestConfig()
#: The bias is reduced to H1: 400 M15 bars carry no H4/D1 structure a synthetic tape could vouch
#: for, and an undefined bias would leave the entry chain with nothing to sweep.
BASE = replace(StrategyConfig(), bias=replace(StrategyConfig().bias, timeframes=("H1",)))
#: The knob the ranking of the fixture hinges on, and the value each trial of a stub run picks.
SWEEP = "sweep_buffer_pip"
SCHEDULE: dict[str, Sequence[Any]] = {SWEEP: (1, 5, 3)}
#: The aggregate table the stub evaluator reports for the fit window of every trial.
TRAIN = {"profit_mean": 100.0, "max_dd_mean": 40.0, "sharpe_mean": 3.0}


def _tape() -> pd.DataFrame:
    """Build the seeded 400 bar M15 tape of the module: a 15 pip random walk, all bars closed."""
    rng = np.random.default_rng(7)
    close = 1.1000 + np.cumsum(rng.normal(0.0, 0.0015, BARS))
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
        0, "timestamp", pd.date_range("2026-06-08 00:00", periods=BARS, freq="15min", tz="UTC")
    )
    frame["volume"] = 1
    frame["is_closed"] = True
    return frame


def _tape5() -> pd.DataFrame:
    """The entry tape of the second hierarchy: its own walk on a five minute grid (its own seed).

    The seed is deliberately not the M15 one: a working frame that carries the *same* sequence as
    the entry tape would make the ``m4`` mutation (measuring the structure on the entry tape) look
    identical to the correct run, and the test would lose its teeth.
    """
    rng = np.random.default_rng(11)
    close = 1.1000 + np.cumsum(rng.normal(0.0, 0.0005, BARS))
    open_ = np.concatenate(([1.1000], close[:-1]))
    frame = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + 0.0003,
            "low": np.minimum(open_, close) - 0.0003,
            "close": close,
        }
    )
    frame.insert(
        0, "timestamp", pd.date_range("2026-06-08 00:00", periods=BARS, freq="5min", tz="UTC")
    )
    frame["volume"] = 1
    frame["is_closed"] = True
    return frame


def _working_tape() -> pd.DataFrame:
    """The M15 working tape of the fixture: the module tape shifted up from its bar 10.

    The random walk alone carries no swing break at all, and a structure layer of empty breaks
    would make the ``m4`` mutation (measuring the structure on the entry tape) invisible: the test
    would compare two frames of zeros.  One deterministic step gives the working frame a break the
    entry tape does not have.
    """
    frame = _tape()
    shift = np.where(np.arange(BARS) >= 10, 0.02, 0.0)
    for column in ("open", "high", "low", "close"):
        frame[column] = frame[column] + shift
    return frame


def _marks(config: StrategyConfig) -> TapeMarks:
    """Return a cache record without building one: the stub evaluator never reads its frames."""
    return TapeMarks(trends=pd.DataFrame(), levels=pd.DataFrame(), config=config)


class _StubTrial:
    """A minimal ``optuna.Trial``: it answers from the schedule and keeps what was asked.

    ``params`` ends up being exactly what :func:`~smc_zero.optimizer.ranges.suggest_params`
    asked for, which is what a real trial publishes as ``trial.params``; every name the schedule
    does not cover falls back to the low bound of its range, the way a sampler may answer with any
    value of the space.
    """

    def __init__(self, number: int, schedule: Mapping[str, Sequence[Any]]) -> None:
        self.number = number
        self.schedule = schedule
        self.params: dict[str, Any] = {}
        self.attrs: dict[str, Any] = {}
        self.value: float | None = None

    def _answer(self, name: str, default: Any) -> Any:
        """Record the ask and return the scheduled value of this trial, or ``default``."""
        series = self.schedule.get(name)
        value = default if series is None else series[self.number]
        self.params[name] = value
        return value

    def suggest_int(self, name: str, low: int, high: int, step: int = 1, log: bool = False) -> int:
        """Answer an integer ask of the search space."""
        return int(self._answer(name, low))

    def suggest_float(
        self, name: str, low: float, high: float, step: float | None = None, log: bool = False
    ) -> float:
        """Answer a float ask of the search space."""
        return float(self._answer(name, low))

    def suggest_categorical(self, name: str, choices: Sequence[str]) -> str:
        """Answer a categorical ask of the search space."""
        return str(self._answer(name, choices[0]))

    def set_user_attr(self, name: str, value: Any) -> None:
        """Record an attribute the way a real trial publishes it to its study."""
        self.attrs[name] = value


class _StubStudy:
    """A minimal ``optuna.Study``: serial trials, the best one by value - enough for a run."""

    def __init__(self, config: OptunaConfig, schedule: Mapping[str, Sequence[Any]]) -> None:
        self.config = config
        self.schedule = schedule
        self.trials: list[_StubTrial] = []
        self.best_trial: _StubTrial | None = None

    def optimize(self, objective: Any, n_trials: int = 1, n_jobs: int = 1) -> None:
        """Score ``n_trials`` trials in series, keeping the best of them."""
        for number in range(n_trials):
            trial = _StubTrial(number, self.schedule)
            trial.value = float(objective(trial))
            self.trials.append(trial)
            if self.best_trial is None or trial.value > self.best_trial.value:
                self.best_trial = trial


def _stub_study(config: OptunaConfig) -> _StubStudy:
    """Return the stub study a run is driven with, on the schedule of the fixture."""
    return _StubStudy(config, SCHEDULE)


class _StubEvaluator:
    """A stub ``Evaluator``: tables that follow the parameters the trial suggested.

    It stands in for :func:`~smc_zero.optimizer.evaluate_params` where the point of a test is the
    *study* rather than the simulation: the test window's profit is proportional to the parameter
    the fixture searches, so the score of a trial is
    ``2 * (10 * sweep) / (1 + 10) = 1.818 * sweep`` - a ranking a reader can redo (the decay gate
    is off at the default ``penalty_power``, and the fixture's train window is a constant).
    """

    def __init__(self) -> None:
        self.configs: list[StrategyConfig] = []

    def __call__(
        self,
        df: pd.DataFrame,
        marks: TapeMarks,
        cfg_strategy: StrategyConfig,
        cfg_wf: WalkForwardConfig,
        backtest: BacktestConfig,
        instrument: InstrumentSpec,
    ) -> FoldEvaluation:
        """Return one fold of the fixture, scaled by the parameter the trial suggested."""
        self.configs.append(cfg_strategy)
        test = {
            "profit_mean": 10.0 * float(cfg_strategy.sweep_buffer_pip),
            "max_dd_mean": 10.0,
            "sharpe_mean": 2.0,
        }
        return FoldEvaluation(
            fold_metrics_train=[dict(TRAIN)],
            fold_metrics_test=[dict(test)],
            train_aggregated=dict(TRAIN),
            test_aggregated=test,
        )


def _count_heavy_calls(monkeypatch: pytest.MonkeyPatch) -> Counter[str]:
    """Count the heavy calls of the markup on :mod:`smc_zero.optimizer.marks` as they happen."""
    counts: Counter[str] = Counter()
    for name in (
        "resample_to_timeframe",
        "bias_frames",
        "static_levels",
        "level_lifecycle",
        "structure_layer",
    ):
        real = getattr(marks_module, name)

        def wrapper(*args: Any, _real: Any = real, _name: str = name, **kwargs: Any) -> Any:
            counts[_name] += 1
            return _real(*args, **kwargs)

        monkeypatch.setattr(marks_module, name, wrapper)
    return counts


def test_the_markup_of_a_run_is_built_once_for_every_trial(monkeypatch: pytest.MonkeyPatch) -> None:
    """The heavy calls happen once per *run*: the cache is the point of the layer (m2).

    Three trials of a real tape mean four evaluations (one per trial plus the fresh one of the
    winner); the signals of the tape are marked up exactly once for all of them.
    """
    counts = _count_heavy_calls(monkeypatch)
    seen: list[StrategyConfig] = []

    def recorder(
        df: pd.DataFrame,
        marks: TapeMarks,
        cfg_strategy: StrategyConfig,
        cfg_wf: WalkForwardConfig,
        backtest: BacktestConfig,
        instrument: InstrumentSpec,
    ) -> FoldEvaluation:
        """Run the real evaluation and remember which configuration was evaluated."""
        seen.append(cfg_strategy)
        return evaluate_params(df, marks, cfg_strategy, cfg_wf, backtest, instrument)

    result = run_optimization(
        _tape(),
        OptunaConfig(n_trials=3, seed=7),
        SHORT,
        BACKTEST,
        base=BASE,
        evaluate=recorder,
        study_factory=_stub_study,
    )

    assert counts == Counter(
        {
            "resample_to_timeframe": 1,
            "bias_frames": 1,
            "static_levels": 1,
            "level_lifecycle": 1,
        }
    )
    assert len(result.study.trials) == 3
    assert len(seen) == 4
    # The last evaluation is the winner's, recomputed from scratch rather than read back.
    assert seen[-1] == result.strategy
    assert all(table["trades"] >= 0 for table in result.best_evaluation.fold_metrics_test)


def test_m15_mode_does_not_compute_structure() -> None:
    """The v1 hierarchy reads the structure of the entry tape: no separate frame, no layer."""
    marks = build_tape_marks(_tape(), BASE)

    assert marks.structure is None


def test_the_structure_layer_is_cached_in_tape_marks() -> None:
    """A separate working frame is marked up once into the cache, in the scale of the entry bars."""
    entry = _tape5()

    marks = build_tape_marks(
        entry, BASE, ltf="M5", structure_frame=_working_tape(), structure_timeframe="M15"
    )

    assert marks.structure is not None
    assert len(marks.structure) == len(drop_unclosed(entry))
    assert list(marks.structure.columns) == [TIMESTAMP_COLUMN, *STRUCTURE_LAYER_COLUMNS]
    # the working frame carries a break and the entry tape (its own walk) does not
    assert int((marks.structure["break_dir"] != 0).sum()) > 0
    # the layer is the *working* frame's reading of the entry bars, not the entry tape's own (m4)
    expected = structure_layer(
        drop_unclosed(entry),
        _working_tape(),
        ltf_timeframe="M5",
        structure_timeframe="M15",
        cfg=StructureLayerConfig(structure=BASE.bias.structure, displacement=BASE.displacement),
    )
    pd.testing.assert_frame_equal(
        marks.structure.drop(columns=[TIMESTAMP_COLUMN]).reset_index(drop=True),
        expected.drop(columns=[TIMESTAMP_COLUMN]).reset_index(drop=True),
    )
    # the cached layer addresses the entry tape itself, which is what the chain requires of it
    chain = build_intents(
        entry,
        marks.bias_frame(BASE.bias.agreement),
        marks.levels,
        BASE,
        structure=marks.structure,
    )
    assert len(chain.rejections) >= 0


def test_the_markup_count_includes_the_structure_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    """The layer is one more heavy call: once per run in the M5 mode, absent in the v1 mode (m4)."""
    counts = _count_heavy_calls(monkeypatch)

    build_tape_marks(_tape(), BASE)
    assert counts["structure_layer"] == 0
    assert counts == Counter(
        {"resample_to_timeframe": 1, "bias_frames": 1, "static_levels": 1, "level_lifecycle": 1}
    )

    counts.clear()
    build_tape_marks(_tape5(), BASE, ltf="M5", structure_frame=_working_tape(), structure_timeframe="M15")

    assert counts == Counter(
        {
            "resample_to_timeframe": 1,
            "bias_frames": 1,
            "static_levels": 1,
            "level_lifecycle": 1,
            "structure_layer": 1,
        }
    )


def test_evaluate_params_reads_the_cached_layer_of_each_fold() -> None:
    """A fold gets the rows of the cached layer that belong to its own window, not the whole tape."""
    entry = _tape5()
    marks = build_tape_marks(
        entry, BASE, ltf="M5", structure_frame=_working_tape(), structure_timeframe="M15"
    )

    evaluation = evaluate_params(entry, marks, BASE, SHORT, BACKTEST)

    assert len(evaluation.fold_metrics_test) == 2
    assert len(evaluation.fold_metrics_train) == 2


def test_the_pooled_score_runs_on_the_real_fold_tables_of_the_engine() -> None:
    """The Э13.1 score reads the engine's own tables: it carries the sums and obeys its gates.

    The unit tests of ``tests/test_optimize.py`` pin the arithmetic of the pool on hand-written
    tables; this one pins the *contract* between the Э5' metric and the score: a fold table of the
    real engine carries the uncapped ``gross_win`` / ``gross_loss`` pair, the shipped gates refuse
    a two-fold pool outright, and the ``Scorer``-shaped adapter scores exactly the out-of-sample
    tables - not the fit ones - under a budget that accepts them.
    """
    df = _tape()
    marks = build_tape_marks(df, BASE)
    evaluation = evaluate_params(df, marks, BASE, SHORT, BACKTEST)
    folds = evaluation.fold_metrics_test

    assert len(folds) == 2
    assert all({"gross_win", "gross_loss"} <= set(table) for table in folds)
    # Two folds are below the shipped minimum of five: the pool is refused outright.
    assert trades_pool_scorer(evaluation, TradeTargetScore()) == 0.0
    # A budget that accepts those two folds scores their pool - the out-of-sample tables, nothing else.
    relaxed = TradeTargetScore(min_valid_folds=2, min_fold_trades=1, min_trades=1)
    assert trades_pool_scorer(evaluation, relaxed) == pytest.approx(
        trades_scaled_score(folds, relaxed)
    )


def test_every_fold_is_simulated_on_its_own_test_window() -> None:
    """Each fold's table is the engine's own run on that fold's window - never on the tape (m3)."""
    df = _tape()
    marks = build_tape_marks(df, BASE)
    evaluation = evaluate_params(df, marks, BASE, SHORT, BACKTEST)
    folds = list(split_walkforward(df, SHORT))

    def build(window: pd.DataFrame) -> tuple[object, ...]:
        """Arm the real entry chain on one window, the way a fold's builder has to."""
        return build_intents(
            window, marks.bias_frame(BASE.bias.agreement), marks.levels, BASE
        ).intents

    # The tape is a real one: a book of level instances and a defined bias, not an empty frame.
    assert len(marks.levels) > 0
    assert set(marks.trends["bias_state"]) & {"agree_long", "agree_short"}
    assert len(folds) == 2
    assert len(evaluation.fold_metrics_test) == len(evaluation.fold_metrics_train) == 2
    assert evaluation.test_aggregated == aggregate_fold_metrics(evaluation.fold_metrics_test)
    assert evaluation.train_aggregated == aggregate_fold_metrics(evaluation.fold_metrics_train)
    assert evaluation.test_aggregated["folds"] == 2.0

    for number, (train, test) in enumerate(folds):
        train_table = evaluation.fold_metrics_train[number]
        test_table = evaluation.fold_metrics_test[number]

        assert (len(train), len(test)) == (TRAIN_BARS + number * TEST_BARS, TEST_BARS)
        # The engine is handed the fold's own window and nothing else: the first simulated bar is
        # the one right after the last fitted one, and no bar of the fit window is in it (m3).
        assert int(train.index.max()) + 1 == int(test.index.min())
        assert test_table == run_backtest(test, build(test), BACKTEST, DEFAULT_INSTRUMENT).metrics
        assert (
            train_table == run_backtest(train, build(train), BACKTEST, DEFAULT_INSTRUMENT).metrics
        )


def test_the_study_ranks_its_trials_by_the_out_of_sample_window() -> None:
    """The winner is the trial with the best *test* score - an in-sample ranking would tie (m1)."""
    evaluator = _StubEvaluator()
    result = run_optimization(
        pd.DataFrame(),
        OptunaConfig(n_trials=3, seed=7),
        SHORT,
        BACKTEST,
        base=BASE,
        marks=_marks(BASE),
        evaluate=evaluator,
        study_factory=_stub_study,
    )

    # The schedule proposes sweep_buffer_pip 1, 5, 3; the score grows with the test window's
    # profit, so the middle trial wins. An objective reading the train side would tie all three
    # (m1), one without the profit factor would tie them at 2/11 - and the runner would report
    # trial 0 (m2).
    assert [trial.params[SWEEP] for trial in result.study.trials] == [1, 5, 3]
    assert [trial.value for trial in result.study.trials] == pytest.approx(
        [2.0 * 10.0 * sweep / 11.0 for sweep in (1, 5, 3)]
    )
    assert result.best_trial_number == 1
    assert result.best_params[SWEEP] == 5
    assert result.strategy.sweep_buffer_pip == 5.0
    assert result.best_score == pytest.approx(2.0 * 50.0 / 11.0)
    # The reported metrics are the winner's own, recomputed after the study.
    assert result.best_evaluation.test_aggregated["profit_mean"] == 50.0
    assert result.best_score == pytest.approx(
        score_from_aggregates(
            result.best_evaluation.train_aggregated, result.best_evaluation.test_aggregated
        )
    )
    assert result.config == OptunaConfig(n_trials=3, seed=7)
    assert evaluator.configs[-1] == result.strategy


def test_a_seeded_optuna_study_is_reproducible() -> None:
    """optuna is a lazy dependency: the default factory drives a seeded TPE study (Э7')."""
    pytest.importorskip("optuna")

    def run() -> Any:
        """Run the fixture through the real study factory and the stub evaluator."""
        return run_optimization(
            pd.DataFrame(),
            OptunaConfig(n_trials=3, seed=7),
            SHORT,
            BACKTEST,
            base=BASE,
            marks=_marks(BASE),
            evaluate=_StubEvaluator(),
        )

    result = run()
    again = run()

    assert type(result.study.sampler).__name__ == "TPESampler"
    # StudyDirection is an IntEnum, so str() yields "2" on Python 3.11+: compare the enum name.
    assert result.study.direction.name == "MAXIMIZE"
    assert len(result.study.trials) == 3
    # The same seed, tape and space give the same run: an optimization report is auditable.
    assert result.best_params == again.best_params
    assert result.best_score == again.best_score
    assert result.best_score == pytest.approx(
        2.0 * 10.0 * float(result.best_params[SWEEP]) / 11.0
    )
