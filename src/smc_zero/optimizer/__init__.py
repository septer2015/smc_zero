"""Optuna optimizer of the strategy parameters, behind a walk-forward OOS gate (Э7').

The layer is a chain of one-way imports - :mod:`smc_zero.optimizer.score` holds the number,
:mod:`smc_zero.optimizer.ranges` the search space, :mod:`smc_zero.optimizer.marks` the markup
cache and :mod:`smc_zero.optimizer.optimize` the trial plumbing over all three - so the faces
below are the whole public surface:

* :func:`run_optimization` runs the study and returns an :class:`OptunaResult` (the finished
  study, the best parameters and their fresh re-evaluation over every fold);
* :func:`make_objective` is the objective on its own (one trial in, one score out) for a
  caller that owns the study;
* :func:`score_from_aggregates` is the score: the out-of-sample metric, profit and drawdown of the
  folds, times the (weighted) train -> test decay of the parameter set, and a flat zero for a test
  window that made no money;
* :func:`build_tape_marks` / :func:`cache_mismatches` are the cache built once per run and the
  guard that refuses a configuration it does not cover;
* :data:`PARAM_RANGES` / :func:`suggest_params` / :func:`apply_params` / :func:`resolve_path`
  are the search space and the two bridges between a trial and a config;
* :func:`evaluate_params` is the fold evaluation of one parameter set (both windows).

optuna itself is imported lazily by the study factory, so everything above - and its tests -
works without the library installed; only an actual study needs it.
"""

from __future__ import annotations

from smc_zero.optimizer.marks import TapeMarks, build_tape_marks, cache_mismatches
from smc_zero.optimizer.optimize import (
    FoldEvaluation,
    OptunaResult,
    Scorer,
    evaluate_params,
    make_objective,
    run_optimization,
)
from smc_zero.optimizer.ranges import (
    M5_PARAM_RANGES,
    PARAM_PROFILES,
    PARAM_RANGES,
    ChoiceRange,
    FloatRange,
    IntRange,
    ParamRange,
    ParamValue,
    TrialLike,
    apply_params,
    resolve_path,
    suggest_params,
)
from smc_zero.optimizer.score import (
    degradation_factor,
    drawdown_factor,
    score_from_aggregates,
    trades_scaled_score,
)

__all__ = [
    "M5_PARAM_RANGES",
    "PARAM_PROFILES",
    "PARAM_RANGES",
    "ChoiceRange",
    "FloatRange",
    "FoldEvaluation",
    "IntRange",
    "OptunaResult",
    "ParamRange",
    "ParamValue",
    "Scorer",
    "TapeMarks",
    "TrialLike",
    "apply_params",
    "build_tape_marks",
    "cache_mismatches",
    "degradation_factor",
    "drawdown_factor",
    "evaluate_params",
    "make_objective",
    "resolve_path",
    "run_optimization",
    "score_from_aggregates",
    "suggest_params",
    "trades_scaled_score",
]
