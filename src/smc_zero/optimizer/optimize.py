"""Optuna optimization of the strategy parameters behind a walk-forward OOS gate (Э7').

This is the last stage of the logic: it turns the walk-forward of Э6' into a search.  One
run is:

1. **cache the markup once** - :func:`smc_zero.optimizer.marks.build_tape_marks` reads the
   whole tape and hands the same bias trends and level book to every trial;
2. **optimize** - a seeded TPE study maximises the score of
   :func:`smc_zero.optimizer.score.score_from_aggregates` over the ranges of
   :data:`smc_zero.optimizer.ranges.PARAM_RANGES`, each trial evaluating *all* folds of
   the tape on the out-of-sample window through :func:`run_walkforward` and the same
   parameters on the fit window as the in-sample baseline;
3. **report the winner** - the best parameters are re-applied to the base configuration
   and re-evaluated over every fold, so the returned metrics are computed again from
   scratch instead of being read back out of the study.

Nothing is simulated on a window that had not closed before it started: the fold split is
Э6' unchanged, the intents of a fold are armed on that fold's test window only (the
contract of §7.10 п.65) and the score reads the trial's own *out-of-sample* folds; the gate
that weighs them against the in-sample result is the operator's and is **off** at the
default ``penalty_power = 0`` (Э9''.1), so the layer as shipped never ranks a trial on the
past.

optuna is imported lazily by :func:`_create_study`, which is why this module - and its
tests - work without the library installed: the search space, the score and the trial
plumbing are plain objects, and only an actual study needs optuna.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias

import pandas as pd

from smc_zero.backtester.engine import DEFAULT_INSTRUMENT, run_backtest
from smc_zero.backtester.walkforward import (
    aggregate_fold_metrics,
    run_walkforward,
    split_walkforward,
)
from smc_zero.config import (
    BacktestConfig,
    InstrumentSpec,
    OptunaConfig,
    StrategyConfig,
    WalkForwardConfig,
)
from smc_zero.optimizer.marks import TapeMarks, build_tape_marks, cache_mismatches
from smc_zero.optimizer.ranges import ParamValue, TrialLike, apply_params, suggest_params
from smc_zero.optimizer.score import score_from_aggregates
from smc_zero.strategy.base import TradeIntent
from smc_zero.strategy.intents import build_intents


@dataclass(frozen=True, slots=True)
class FoldEvaluation:
    """One parameter set over all folds: both windows' tables and their aggregates.

    ``fold_metrics_train[i]`` is the Э5' metric table of the engine run on the i-th fold's
    *fit* window with this parameter set, ``fold_metrics_test[i]`` the out-of-sample table
    of the same fold from :func:`~smc_zero.backtester.walkforward.run_walkforward`; the two
    lists are aligned by fold and carry the same costs, so their difference is the effect
    of the window and not of the run.  The aggregates are the Э6' readings of each list.
    """

    fold_metrics_train: list[dict[str, float]]
    fold_metrics_test: list[dict[str, float]]
    train_aggregated: dict[str, float]
    test_aggregated: dict[str, float]


#: The Э6' builder of one parameter set: ``(train, test) -> intents`` armed on ``test``.
IntentBuilder: TypeAlias = Callable[[pd.DataFrame, pd.DataFrame], tuple[TradeIntent, ...]]

#: How a trial is scored: the fixed inputs of the run plus the trial's configuration.
Evaluator: TypeAlias = Callable[
    [pd.DataFrame, TapeMarks, StrategyConfig, WalkForwardConfig, BacktestConfig, InstrumentSpec],
    FoldEvaluation,
]

#: How a study is created; :func:`_create_study` is the optuna one.
StudyFactory: TypeAlias = Callable[[OptunaConfig], Any]


def _intents_fn(bias: pd.DataFrame, levels: pd.DataFrame, cfg: StrategyConfig) -> IntentBuilder:
    """Return the Э6' builder of one parameter set.

    The contract of :func:`run_walkforward` (§7.10 п.65) is that the *second* argument is
    the window the engine will simulate and the only window a fold's intents may be armed
    against; the fit window is what the optimizer reads parameters from.  Here the
    parameters are not fitted inside the builder at all - they arrive from the trial - so
    the fit window is used on the other side of the fold: to score the trial in sample.
    Returning only the chain's ``intents`` drops its ledger of rejected attempts, which the
    engine replaces with the accounting it needs.
    """

    def build(train: pd.DataFrame, test: pd.DataFrame) -> tuple[TradeIntent, ...]:
        """Arm the entry chain on the test window and return its accepted intents."""
        return build_intents(test, bias, levels, cfg).intents

    return build


def evaluate_params(
    df: pd.DataFrame,
    marks: TapeMarks,
    cfg_strategy: StrategyConfig,
    cfg_wf: WalkForwardConfig | None = None,
    backtest: BacktestConfig | None = None,
    instrument: InstrumentSpec | None = None,
) -> FoldEvaluation:
    """Run every fold of ``df`` with one parameter set and return both sides' tables.

    The out-of-sample side is :func:`run_walkforward` unchanged: a fold's intents are built
    on its test window and exactly that window is simulated.  The in-sample side is the
    same engine with the same costs on the same fold's *train* window, because the OOS gate
    needs a like-for-like baseline - what this parameter set did on the window it was
    allowed to see - and no other number can play that role.

    The folds are cut twice (once inside the runner, once here for the train side) rather
    than reimplementing the Э6' loop: ``split_walkforward`` is a deterministic positional
    slice of the tape, so the two lists are aligned by construction and ``walkforward.py``
    stays read-only.
    """
    config = WalkForwardConfig() if cfg_wf is None else cfg_wf
    backtest_cfg = BacktestConfig() if backtest is None else backtest
    spec = DEFAULT_INSTRUMENT if instrument is None else instrument
    build = _intents_fn(marks.bias_frame(cfg_strategy.bias.agreement), marks.levels, cfg_strategy)
    walk = run_walkforward(df, build, config, backtest_cfg, spec)
    train_metrics = [
        run_backtest(train, build(train, train), backtest_cfg, spec).metrics
        for train, _ in split_walkforward(df, config)
    ]
    return FoldEvaluation(
        fold_metrics_train=train_metrics,
        fold_metrics_test=walk.fold_metrics,
        train_aggregated=aggregate_fold_metrics(train_metrics),
        test_aggregated=walk.aggregated,
    )


def make_objective(
    df: pd.DataFrame,
    cfg_opt: OptunaConfig | None = None,
    cfg_wf: WalkForwardConfig | None = None,
    backtest: BacktestConfig | None = None,
    instrument: InstrumentSpec | None = None,
    *,
    base: StrategyConfig | None = None,
    marks: TapeMarks | None = None,
    evaluate: Evaluator = evaluate_params,
) -> Callable[[TrialLike], float]:
    """Return the objective of one optimization run, ready for ``study.optimize``.

    Each call asks the trial for a value per range of
    :data:`~smc_zero.optimizer.ranges.PARAM_RANGES` (:func:`suggest_params`), applies them
    to ``base`` (:func:`apply_params`), evaluates the whole tape with that configuration
    (:func:`evaluate_params` by default) and returns the OOS score of
    :func:`~smc_zero.optimizer.score.score_from_aggregates`; both aggregates are also stored
    on the trial as user attributes, so a finished study can be read without rerunning it.

    ``marks`` is the markup cache, built here with :func:`build_tape_marks` when not handed
    in - once for the whole study, never per trial.  ``base`` is the configuration a trial
    starts from: the constants of the run (costs, sizing, the level map) plus the starting
    values of the knobs being searched.  Its frozen parts must be the parts the cache was
    built from; a mismatch raises ``ValueError`` instead of scoring a trial against a stale
    markup.  ``evaluate`` is the seam of the tests: a stub there scores a trial without
    simulating anything.
    """
    config = OptunaConfig() if cfg_opt is None else cfg_opt
    base_cfg = StrategyConfig() if base is None else base
    walk_cfg = WalkForwardConfig() if cfg_wf is None else cfg_wf
    backtest_cfg = BacktestConfig() if backtest is None else backtest
    spec = DEFAULT_INSTRUMENT if instrument is None else instrument
    cache = build_tape_marks(df, base_cfg) if marks is None else marks
    stale = cache_mismatches(cache, base_cfg)
    if stale:
        raise ValueError(
            "the markup cache was built for another configuration: "
            f"{list(stale)} differ(s) from it. Rebuild the cache with build_tape_marks(df, cfg) "
            "or keep those fields out of the trial's search space."
        )

    def objective(trial: TrialLike) -> float:
        """Score one trial: suggest, apply, evaluate on both windows, weigh them."""
        params = suggest_params(trial)
        cfg_strategy = apply_params(base_cfg, params)
        evaluation = evaluate(df, cache, cfg_strategy, walk_cfg, backtest_cfg, spec)
        trial.set_user_attr("train", evaluation.train_aggregated)
        trial.set_user_attr("test", evaluation.test_aggregated)
        return score_from_aggregates(
            evaluation.train_aggregated, evaluation.test_aggregated, config
        )

    return objective


@dataclass(frozen=True, slots=True)
class OptunaResult:
    """The outcome of one optimization run (Э7').

    ``study`` is the finished optuna study - kept whole, because its trials, their
    parameters, the user attributes and the sampler state are the audit trail of the run.
    ``best_params`` is ``study.best_trial.params`` (the dotted paths of the search space,
    directly usable with :func:`apply_params`), ``best_score`` the value the objective
    returned for them, and ``best_trial_number`` which trial reached it.  ``strategy`` is
    the base configuration with the best parameters applied, and ``best_evaluation`` the
    re-run of exactly that configuration over all folds: the winner's metrics come from a
    fresh evaluation, not from a table read back out of the study.  ``config`` is the
    budget the study was given (trial count, seed, metric, penalty).
    """

    study: Any
    best_params: dict[str, ParamValue]
    best_score: float
    best_trial_number: int
    strategy: StrategyConfig
    best_evaluation: FoldEvaluation
    config: OptunaConfig


def _create_study(config: OptunaConfig) -> Any:
    """Create the in-memory maximization study of one run: seeded TPE, no storage.

    optuna is imported here and nowhere else in the package, so the Э7' layer can be
    imported, its search space described and its score tested without the library.  The
    sampler is TPE with ``seed=config.seed``, which is what makes a run reproducible from
    its configuration alone.  With ``n_jobs > 1`` the sampler additionally gets
    ``constant_liar=True``: parallel trials then plan against a guess of each other's
    result instead of waiting, which is the setting optuna recommends for parallel TPE.
    No storage is passed - the finished study travels in memory on
    :class:`OptunaResult`, and a run that has to survive its process is a later concern.
    """
    import optuna

    return optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=config.seed,
            constant_liar=config.n_jobs > 1,
        ),
    )


def run_optimization(
    df: pd.DataFrame,
    cfg_opt: OptunaConfig | None = None,
    cfg_wf: WalkForwardConfig | None = None,
    backtest: BacktestConfig | None = None,
    instrument: InstrumentSpec | None = None,
    *,
    base: StrategyConfig | None = None,
    marks: TapeMarks | None = None,
    evaluate: Evaluator = evaluate_params,
    study_factory: StudyFactory = _create_study,
) -> OptunaResult:
    """Maximize the OOS score of the É6' walk-forward over ``df`` and return the winner.

    The whole run: build (or accept) the markup cache once, build the objective of
    :func:`make_objective`, create the study with ``study_factory``, run
    ``config.n_trials`` trials with ``config.n_jobs`` workers in parallel, then re-apply
    the best parameters and evaluate them over every fold again.  The re-evaluation is the
    honest part of the report: ``best_score`` is what the study recorded, while
    ``best_evaluation`` is what that configuration actually produces now.

    A study in which *every* trial failed has no winner; optuna's own error is left to
    propagate rather than returning an empty result that looks like success.  Runs are
    reproducible for a fixed seed, configuration and library version - which is the
    guarantee the tests rely on, not a promise about the optimum found.
    """
    config = OptunaConfig() if cfg_opt is None else cfg_opt
    walk_cfg = WalkForwardConfig() if cfg_wf is None else cfg_wf
    backtest_cfg = BacktestConfig() if backtest is None else backtest
    spec = DEFAULT_INSTRUMENT if instrument is None else instrument
    base_cfg = StrategyConfig() if base is None else base
    cache = build_tape_marks(df, base_cfg) if marks is None else marks
    objective = make_objective(
        df,
        config,
        walk_cfg,
        backtest_cfg,
        spec,
        base=base_cfg,
        marks=cache,
        evaluate=evaluate,
    )
    study = study_factory(config)
    study.optimize(objective, n_trials=config.n_trials, n_jobs=config.n_jobs)
    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    strategy = apply_params(base_cfg, best_params)
    return OptunaResult(
        study=study,
        best_params=best_params,
        best_score=float(best_trial.value),
        best_trial_number=int(best_trial.number),
        strategy=strategy,
        best_evaluation=evaluate(df, cache, strategy, walk_cfg, backtest_cfg, spec),
        config=config,
    )
