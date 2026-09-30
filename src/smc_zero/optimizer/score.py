"""The score of one optimization trial: out-of-sample quality, drawdown and decay (Э7').

One number has to rank two parameter sets of the same walk-forward, and it is built from
the *aggregated fold tables* of Э6' - never from a single fold and never from the train
window alone.  The score is a product of four factors (SPEC_SMC.md §7.11 п.69, ред. Э9''.1):

============== ==================================================================
leading        the headline metric of the out-of-sample folds the study maximises
               (``sharpe_mean`` or ``profit_mean``), i.e. what a trial actually earned;
profit         ``profit_mean(test)``: the same window's result in percent, so a trial
               that scores by Sharpe still has to *make* money to rank high - and a
               window that made none (``<= 0``) scores a flat zero, never a magnitude;
drawdown       ``1 / (1 + max_dd_mean(test))``: what the equity paid on the way there;
decay gate     ``min(1, profit_mean(test) / profit_mean(train)) ** penalty_power``:
               how much of the in-sample edge survived the out-of-sample window.
============== ==================================================================

The product is only taken for a **profitable** out-of-sample window: ``profit_mean(test) <= 0``
returns ``0.0`` before the factors are multiplied (Э9''.2).  The guard exists because two of the
factors can be negative at once - a losing window has a negative Sharpe *and* a negative profit -
and their product is **positive**: without the guard, the worst run of a study was its winner
(measured: ``sharpe_mean(test) = -0.19`` and ``profit_mean(test) = -127.53`` scored ``+9.6154``).
A flat window already scored zero, so the guard changes no honest number, it only puts every
loser at the flat zero of the bottom - below every profitable set, which is the whole ranking
contract.

The last factor is the **OOS gate**, and it is the one factor an operator can switch off:
``penalty_power = 0`` (the default) leaves the pure out-of-sample reading - the metric,
the profit and the drawdown of the *test* window - while ``1.0`` is the plain product of
the sketch and a larger value is stricter about a parameter set that only fits its past.
The edge rules of the factors (a losing train window has no edge to decay, a losing test
window cannot flip the sign of the ratio, a curve that never drew down is not rewarded)
are stated on the functions below and pinned by tests.

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
    """Return ``1 / (1 + max_dd_mean)``: the share of the score a drawdown leaves.

    The factor throttles a trial that reaches its result by way of a deep drawdown: 5 % of
    mean drawdown keeps ``1/6`` of the score, 10 % keeps ``1/11``, 50 % keeps ``1/51``.  A
    divisor and not a subtraction, because it must never run out of range: it stays inside
    ``(0, 1]`` for every reading, so *losing* more can only ever lower a score, and a curve
    that never drew down (a negative reading cannot happen, but a zero one can) is simply
    not throttled.
    """
    return 1.0 / (1.0 + max(0.0, float(max_dd_mean)))


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
    observed.  ``power`` (``OptunaConfig.penalty_power``) weighs the factor: ``0.0``
    switches the gate off - the factor is ``1`` whatever the ratio - ``1.0`` (this
    function's own default, the plain ratio of the sketch) applies it once, and a larger
    value is stricter about decay.
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
                * profit_mean(test)
                / (1 + max_dd_mean(test))
                * min(1, profit_mean(test) / profit_mean(train)) ** penalty_power
              = 0                                if profit_mean(test) <= 0

    ``cfg`` supplies the two decisions of :class:`~smc_zero.config.OptunaConfig`:
    :attr:`~smc_zero.config.OptunaConfig.score_metric` names the leading metric
    (``"sharpe"`` or ``"profit"``) and
    :attr:`~smc_zero.config.OptunaConfig.penalty_power` the weight of the OOS gate -
    ``0.0`` (the default) switches the gate off and reads the test window alone.  Every
    other factor is read from the *test* half as well: out-of-sample is what the layer
    ranks, so a trial that only fits its past cannot buy rank here.  The profit
    ratio of the gate is always the *profit* one, whatever the leading metric is: it
    measures how much of the in-sample result survived, and percentages are comparable
    across folds in a way a Sharpe ratio is not.

    A **losing or flat test window** (``profit_mean(test) <= 0``) scores ``0.0`` whatever its
    other numbers say (п.69): the leading metric and the profit are both negative there, and
    their product is positive, so without the guard the study would rank a losing parameter set
    *above* a profitable one - the exact inversion the layer exists to prevent.  ``0.0`` is the
    same reading a window with no trades gets, so a loser never outranks a winner and neither of
    them outranks a profitable set.

    A table without one of the needed entries - or with a non-finite number in it - raises
    ``ValueError``: a score assembled from a missing metric would rank trials by nothing.
    """
    config = OptunaConfig() if cfg is None else cfg
    leading = _metric(test_aggregated, f"{config.score_metric}_mean")
    profit = _metric(test_aggregated, "profit_mean")
    train_profit = _metric(train_aggregated, "profit_mean")
    if profit <= 0.0:
        return 0.0
    drawdown = drawdown_factor(_metric(test_aggregated, "max_dd_mean"))
    decay = degradation_factor(train_profit, profit, config.penalty_power)
    return leading * profit * drawdown * decay
