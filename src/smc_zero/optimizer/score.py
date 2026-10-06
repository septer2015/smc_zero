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

The score of the *second* hierarchy is the other side of the same seam and is documented on
:func:`trades_scaled_score`: it ranks by the **pooled** profit factor of the dense out-of-sample
folds times a trade-count factor (SPEC_SMC.md §7.22, ред. Э13.1).  The first M5 study showed why a
per-fold mean cannot play that role: a fold of four trades with a perfect win rate is capped at
``5.0`` and lifts the *mean* of a losing study, while the same trades read as one pool lose money.
The pooled reading sums the folds' gross wins and gross losses and divides once, and it refuses to
read a period of too few dense folds at all.

The module is a family of pure functions over ``dict`` tables and fold sequences: it knows nothing
about optuna, the engine or the strategy, so the formula can be read, argued about and tested on
numbers a reader can redo on paper.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Protocol

from smc_zero.config import DEFAULT_SCORE_PF_CAP, OptunaConfig, TradeTargetScore


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


#: The fields a fold table must carry for the pooled reading of §7.22: the count of trades, the
#: profit of the fold and the two *uncapped* sums behind its profit factor
#: (:func:`~smc_zero.backtester.metrics.calc_metrics`).
POOL_FOLD_FIELDS: tuple[str, ...] = ("trades", "profit", "gross_win", "gross_loss")


class PoolEvaluation(Protocol):
    """The slice of a fold evaluation the pooled score reads: its out-of-sample fold tables.

    The score is a pure function of the folds, and the ``Scorer`` seam of §7.11 hands it the whole
    evaluation of a trial - :class:`~smc_zero.optimizer.optimize.FoldEvaluation` satisfies this
    protocol by carrying ``fold_metrics_test``.  The protocol stands here instead of an import
    because ``optimizer.optimize`` imports this module and never the reverse.
    """

    fold_metrics_test: Sequence[Mapping[str, float]]


def pool_profit_factor(
    folds: Sequence[Mapping[str, float]],
    pf_cap: float = DEFAULT_SCORE_PF_CAP,
) -> float:
    """Return the profit factor of the *pool* of ``folds``: their gross wins over their gross losses.

    This is the number the first M5 study should have ranked by (Э13.1).  A per-fold profit factor
    is a ratio of two small samples, and its arithmetic mean over nine folds is dominated by
    whichever fold happened to hold four trades: the winner of that study was a *losing* parameter
    set whose one fold of four winning trades was capped at ``5.0``.  Adding the folds' gross wins
    and gross losses up first and dividing once reads the same trades as *one* out-of-sample period,
    which is what the parameter set would actually have traded.

    A pool without a single loss has no finite profit factor and is reported as ``pf_cap`` - the
    same edge rule the Э5' metric applies to a single run.  Every table must carry the sums of
    :data:`POOL_FOLD_FIELDS`, or :func:`_metric` raises naming the miss instead of scoring an
    invented number.
    """
    gross_win = sum(_metric(table, "gross_win") for table in folds)
    gross_loss = sum(_metric(table, "gross_loss") for table in folds)
    if gross_loss <= 0.0:
        return pf_cap
    return min(gross_win / gross_loss, pf_cap)


def trades_pool_scorer(
    evaluation: PoolEvaluation,
    cfg: TradeTargetScore | None = None,
) -> float:
    """Adapt :func:`trades_scaled_score` to the ``Scorer`` seam: score the pool of the OOS folds.

    ``make_objective`` / ``run_optimization`` rank a trial with a ``Scorer``, which receives the
    whole fold evaluation of one parameter set (§7.11).  The pooled score of §7.22 reads the
    out-of-sample half of that evaluation and nothing else, so this one line is the whole adapter -
    :func:`trades_scaled_score` itself stays a pure function of fold tables.
    """
    return trades_scaled_score(evaluation.fold_metrics_test, cfg)


def trades_scaled_score(
    folds: Sequence[Mapping[str, float]],
    cfg: TradeTargetScore | None = None,
) -> float:
    """Return the trade-target score of the second hierarchy over the *pool* of ``folds``.

    The default score of §7.11 maximises a *ratio* (Sharpe, then profit) and is blind to how many
    trades produced it - a study of a five minute tape would then happily rank a parameter set
    whose whole out-of-sample period carried four trades.  This score ranks the same walk-forward by
    how *often* the edge showed up, which is what the second hierarchy needs (SPEC_SMC.md §7.22):

        valid = [fold for fold in folds if fold.trades >= min_fold_trades]
        score = pf_pool(valid) * min(1, trades(valid) / target_trades)
              = 0                                          if len(valid) < min_valid_folds
              = 0                                          if profit(valid) <= 0
              = 0                                          if trades(valid) < min_trades

    The factors:

    * the **density gate** - only a fold that carried at least ``min_fold_trades`` trades enters the
      pool.  A fold of one or two trades is a reading of noise, and the first M5 study paid for
      learning it: its winner's top rank was bought by a single fold of four trades whose profit
      factor the metric capped at ``5.0`` (Э13.1);
    * ``pf_pool(valid)`` - the pooled profit factor of the dense folds
      (:func:`pool_profit_factor`).  It is a ratio of *sums*, never an average of ratios: the mean
      of the folds can be lifted by one small fold, the pool cannot, because it reads the same
      trades as one out-of-sample period (Э13.1);
    * ``min(1, trades(valid) / target_trades)`` - the trade-count factor: it *caps* at one, so a
      parameter set is never rewarded for opening trades for their own sake past the target.  The
      count is the sum over the dense folds, because the target is a statement about the whole
      out-of-sample period and not about one fold;
    * the ``min_valid_folds`` gate returns a flat zero *before* anything is pooled: a "pool" of two
      or three folds is the artifact this score exists to remove, and a study must not prefer a
      parameter set measured on the fewest folds (Э13.1).  The ``min_trades`` and ``profit <= 0``
      gates are the same shape of guard as the ``profit_mean(test) <= 0`` of
      :func:`score_from_aggregates`: a period that lost money scores the flat zero of the bottom,
      never a magnitude.

    Only the out-of-sample folds are read, and only through the sums of :data:`POOL_FOLD_FIELDS`:
    this score has no OOS decay gate - it is the honest "how much of the period did you trade, and
    how well" reading, and asking it for the fit ratio would punish a parameter set for an
    in-sample window this hierarchy is not about.  A scorer that does weigh the past is
    :func:`score_from_aggregates` with its ``penalty_power``; the ``Scorer``-shaped entry point of
    this score is :func:`trades_pool_scorer`.

    ``cfg`` is :class:`~smc_zero.config.TradeTargetScore`: ``min_fold_trades`` (the count that makes
    a fold dense enough to enter the pool), ``min_valid_folds`` (how many dense folds the pool needs
    at all), ``target_trades`` (the count the factor reaches one at), ``min_trades`` (below which the
    score is a flat zero) and ``pf_cap`` (the cap of the pooled factor).  A table missing a field of
    :data:`POOL_FOLD_FIELDS` - or carrying a non-finite number - raises ``ValueError``, exactly as in
    :func:`score_from_aggregates`.
    """
    config = TradeTargetScore() if cfg is None else cfg
    valid = [table for table in folds if _metric(table, "trades") >= config.min_fold_trades]
    if len(valid) < config.min_valid_folds:
        return 0.0
    if sum(_metric(table, "profit") for table in valid) <= 0.0:
        return 0.0
    trades = sum(_metric(table, "trades") for table in valid)
    if trades < config.min_trades:
        return 0.0
    return pool_profit_factor(valid, config.pf_cap) * min(trades / config.target_trades, 1.0)
