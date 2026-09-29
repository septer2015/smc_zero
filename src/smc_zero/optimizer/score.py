"""The score of one optimization trial: out-of-sample quality, drawdown and decay (Э7').

One number has to rank two parameter sets of the same walk-forward, and it is built from
the *aggregated fold tables* of Э6' - never from a single fold and never from the train
window alone.  The score is a product of three factors (SPEC_SMC.md §7.11 п.69):

============== ==================================================================
leading        the headline metric of the out-of-sample folds the study maximises
               (``sharpe_mean`` or ``profit_mean``), i.e. what a trial actually earned;
drawdown       ``1 - max_dd_mean/100``: what the equity did on the way there;
decay gate     ``min(1, profit_mean(test) / profit_mean(train)) ** penalty_power``:
               how much of the in-sample edge survived the out-of-sample window.
============== ==================================================================

The third factor is the **OOS gate**: a parameter set that fits its train window better
than its test window is scaled down by exactly how much it fell short, so a study
maximising the score cannot buy rank by overfitting.  The edge rules of the factors (a
losing train window has no edge to decay, a losing test window scores zero, a 100 %
drawdown scores zero) are stated on the functions below and pinned by tests.

The module is a pair of pure functions over ``dict`` tables: it knows nothing about
optuna, the engine or the strategy, so the formula can be read, argued about and tested
on numbers a reader can redo on paper.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from smc_zero.config import OptunaConfig


def _metric(table: Mapping[str, float], key: str) -> float:
    """Read one finite number out of an aggregate fold table, or raise naming the miss."""
    if key not in table:
        raise ValueError(
            f"the aggregate table needs a {key!r} entry; "
            "it comes from smc_zero.backtester.walkforward.aggregate_fold_metrics(...)"
        )
    value = float(table[key])
    if not math.isfinite(value):
        raise ValueError(f"the {key!r} of the aggregate table is not finite: {value}")
    return value


def drawdown_factor(max_dd_mean: float) -> float:
    """Return ``1 - max_dd_mean/100``, floored at zero.

    The factor throttles a curve that reaches its result by way of a deep drawdown: 5 %
    of mean drawdown keeps 95 % of the score, 50 % keeps half.  At 100 % - and beyond -
    the factor is ``0`` rather than negative, so a losing account cannot be turned into a
    *better* score by losing more.
    """
    return max(0.0, 1.0 - min(float(max_dd_mean), 100.0) / 100.0)


def degradation_factor(
    train_profit_mean: float,
    test_profit_mean: float,
    power: float = 1.0,
) -> float:
    """Return ``min(1, test / train) ** power``: the penalty for an edge that did not hold.

    ``1.0`` means "the out-of-sample window did at least as well as the fit window";
    ``0.5`` that only half of the in-sample profit survived, and ``0.0`` that the test
    window lost money while the train window made some.  The ratio is clamped into
    ``[0, 1]``, so *outperforming* the train window earns no bonus (the extra is luck by
    definition, and a study that chased it would optimize noise) and *losing* out of
    sample cannot flip the sign of the score.

    A **losing train window** (``train_profit_mean <= 0``) returns ``1.0``: there is no
    in-sample edge whose decay could be measured, so the trial is judged by what it made
    out of sample - punishing it here would mean inventing a degradation that was never
    observed.  ``power`` (``OptunaConfig.penalty_power``) weighs the factor: ``1.0`` is the
    plain ratio of the sketch, a larger value is stricter about decay.
    """
    if train_profit_mean <= 0.0:
        return 1.0
    ratio = float(test_profit_mean) / float(train_profit_mean)
    return max(0.0, min(1.0, ratio)) ** power


def score_from_aggregates(
    train_aggregated: Mapping[str, float],
    test_aggregated: Mapping[str, float],
    cfg: OptunaConfig | None = None,
) -> float:
    """Return the score of one trial from the two aggregated fold tables of Э6'.

    ``train_aggregated`` / ``test_aggregated`` are what
    :func:`smc_zero.backtester.walkforward.aggregate_fold_metrics` reports over the fit
    window and over the out-of-sample window of the same folds.  The score is read out of
    them and never recomputed::

        score = <score_metric>_mean(test)
                * (1 - max_dd_mean(test) / 100)
                * min(1, profit_mean(test) / profit_mean(train)) ** penalty_power

    ``cfg`` supplies the two decisions of :class:`~smc_zero.config.OptunaConfig`:
    :attr:`~smc_zero.config.OptunaConfig.score_metric` names the leading metric
    (``"sharpe"`` or ``"profit"``) and
    :attr:`~smc_zero.config.OptunaConfig.penalty_power` the weight of the OOS gate.  The
    profit ratio of the gate is always the *profit* one, whatever the leading metric is:
    it measures how much of the in-sample result survived, and percentages are comparable
    across folds in a way a Sharpe ratio is not.

    A table without one of the needed entries - or with a non-finite number in it - raises
    ``ValueError``: a score assembled from a missing metric would rank trials by nothing.
    """
    config = OptunaConfig() if cfg is None else cfg
    leading = _metric(test_aggregated, f"{config.score_metric}_mean")
    drawdown = drawdown_factor(_metric(test_aggregated, "max_dd_mean"))
    decay = degradation_factor(
        _metric(train_aggregated, "profit_mean"),
        _metric(test_aggregated, "profit_mean"),
        config.penalty_power,
    )
    return leading * drawdown * decay
