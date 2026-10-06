"""Walk-forward splitting and fold aggregation (Э6'): out-of-sample folds of one tape.

The layer answers two questions about a tape and nothing else:

* where does an out-of-sample fold start and end - :func:`split_walkforward` cuts the frame
  into ``(train, test)`` pairs of :class:`~smc_zero.config.WalkForwardConfig` (an expanding
  "anchored" window by default, a fixed rolling one on demand);
* how did the folds go - :func:`aggregate_fold_metrics` turns the per-fold metric tables of Э5'
  into the mean *and* the population σ of the six headline numbers, so a walk-forward is read
  as a distribution and never as one lucky mean.

:func:`run_walkforward` is the thin runner that joins the two halves to the engine of Э5': it
walks the folds, asks the caller's intent builder for the intents of a fold and hands them to
:func:`smc_zero.backtester.engine.run_backtest` on that fold's *test* window with fixed
parameters.  **No optimization happens here** - fitting the parameters is Э7' (optuna), and the
version of Э6' is deliberately a mechanism check: the builder receives the fold's train window
so that a future optimizer can fit on it, while the simulation of a fold always runs on the
bars of its own test window, because intents anchored anywhere else cannot be simulated at all
(:func:`smc_zero.backtester.engine._intents_by_bar` re-derives every intent's bar from the tape
it is about to simulate and raises on a mismatch - SPEC_SMC.md §7.10 п.65).

Two invariants carry the whole layer, and both are one line away from being broken:

* **a fold never sees its own future.**  The slices are positional views of the caller's frame
  (``df.iloc[...]``), so the labels are the caller's own and ``set(train.index) &
  set(test.index)`` is empty by construction; the test window starts on the bar right after the
  train window ends.  Mutation m1 "cut the test window at ``train_end`` instead of after it"
  breaks :func:`test_a_fold_never_shares_a_bar_with_its_train_window`;
* **anchored is not rolling.**  ``anchored=True`` keeps the train window's left edge at the
  tape's first bar, ``anchored=False`` slides it.  Mutation m2 "ignore ``anchored``" breaks
  :func:`test_the_anchored_folds_always_start_at_the_first_bar`.  A third mutation, m3 "sum the
  folds instead of averaging them", breaks
  :func:`test_the_aggregate_averages_the_folds_and_reports_their_spread`.

Nothing in this module copies, closes or drops a bar: the split is positional and the unclosed
tail of a tape stays the engine's business (rule 2b), which can only touch the very last fold.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import pandas as pd

from smc_zero.backtester.engine import DEFAULT_INSTRUMENT, run_backtest
from smc_zero.config import BacktestConfig, InstrumentSpec, WalkForwardConfig
from smc_zero.data_loader import bars_per_day
from smc_zero.strategy.base import TradeIntent

#: One fold of a walk-forward: the window that may be fitted and the window that is simulated.
Fold: TypeAlias = tuple[pd.DataFrame, pd.DataFrame]
#: The builder of a fold's intents: the train window first, then the test window to arm in.
#: A builder that ignores the train window is the honest Э6' stub - it is how the mechanism is
#: checked without fitting anything (Э7' is where the first argument earns its keep).
IntentBuilder: TypeAlias = Callable[
    [pd.DataFrame, pd.DataFrame], pd.DataFrame | Iterable[TradeIntent]
]

#: The fields :func:`aggregate_fold_metrics` reports, in the order of the metric table of Э5'.
FOLD_METRIC_FIELDS: tuple[str, ...] = ("trades", "profit", "win_rate", "pf", "max_dd", "sharpe")

#: The fold windows of an entry timeframe, as ``(warm-up days, out-of-sample days)``: the M15 tape
#: of the default hierarchy keeps the 120 / 60 days of SPEC_SMC.md §7.10 п.62 (a 15 fold study on
#: the four year tape), while the five minute tape of the H4 -> M15 -> M5 hierarchy uses 60 / 30
#: days - the ruling of SPEC_SMC.md §7.20, which keeps nine folds on 1.4 years of M5 bars instead
#: of the three a strict day-for-day scaling of the M15 windows would leave.
FOLD_WINDOW_DAYS: dict[str, tuple[int, int]] = {
    "M15": (120, 60),
    "M5": (60, 30),
}


def default_walk_forward(timeframe: str) -> WalkForwardConfig:
    """Return the fold scheme of an entry timeframe: its windows in days, written in its own bars.

    ``timeframe`` is the entry timeframe of the tape the study reads, and an unknown label is
    refused instead of being served the M15 windows: a study of a five minute tape measured with
    M15-length windows would fold four times more bars than it holds.  ``anchored`` and the rolling
    length keep the :class:`~smc_zero.config.WalkForwardConfig` defaults, and the caller may still
    override any count on the command line.
    """
    try:
        warm_up_days, test_days = FOLD_WINDOW_DAYS[timeframe.upper()]
    except KeyError as exc:
        supported = ", ".join(sorted(FOLD_WINDOW_DAYS))
        raise ValueError(
            f"no walk-forward windows for {timeframe!r}; expected one of {supported}"
        ) from exc
    per_day = bars_per_day(timeframe)
    return WalkForwardConfig(
        min_train_bars=per_day * warm_up_days,
        test_period_bars=per_day * test_days,
        train_period_bars=per_day * warm_up_days,
    )


def _fold_bounds(n_bars: int, cfg: WalkForwardConfig) -> list[tuple[int, int, int, int]]:
    """Return the ``(train_start, train_end, test_start, test_end)`` of every fold, in order.

    The windows are half-open ``[start, end)`` slices of the tape.  ``train_end`` walks from
    ``min_train_bars`` up by ``test_period_bars`` while a full test window still fits, so the
    fold count of a tape does not depend on the scheme - only the left edge of the train window
    does: the first bar of the tape when anchored, ``train_end - train_period_bars`` (clipped at
    the first bar) when rolling.
    """
    bounds: list[tuple[int, int, int, int]] = []
    train_end = cfg.min_train_bars
    while train_end + cfg.test_period_bars <= n_bars:
        train_start = 0 if cfg.anchored else max(train_end - cfg.train_period_bars, 0)
        bounds.append((train_start, train_end, train_end, train_end + cfg.test_period_bars))
        train_end += cfg.test_period_bars
    return bounds


def split_walkforward(df: pd.DataFrame, cfg: WalkForwardConfig | None = None) -> list[Fold]:
    """Cut ``df`` into out-of-sample folds, each with the window that may fit it (Э6').

    ``cfg`` is the scheme: :class:`~smc_zero.config.WalkForwardConfig` - an expanding train
    window by default, ``anchored=False`` for a fixed rolling one.  The slices are *positional
    views* of the caller's frame, so their labels stay the caller's index: ``train`` and
    ``test`` of a fold can never share a label, and the first simulated bar is the one right
    after the last fitted bar.  Nothing is copied, sorted or dropped here - a tape is taken as
    handed in, and its unclosed tail remains the engine's business (rule 2b).

    A tape too short for one full fold returns an empty list rather than a partial fold: no
    out-of-sample evidence is an honest answer, a half-empty test window is not.
    """
    config = WalkForwardConfig() if cfg is None else cfg
    return [
        (df.iloc[train_start:train_end], df.iloc[test_start:test_end])
        for train_start, train_end, test_start, test_end in _fold_bounds(len(df), config)
    ]


def aggregate_fold_metrics(fold_results: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Report the headline metrics of the folds as a mean *and* a spread (Э6').

    ``fold_results`` is what :func:`run_walkforward` collects - one
    :func:`smc_zero.backtester.metrics.calc_metrics` table per fold.  Every field of
    :data:`FOLD_METRIC_FIELDS` becomes a ``<field>_mean`` and a ``<field>_std`` entry of the
    result, rounded to the two decimals of the metric table they are read from, plus the
    ``folds`` count; the fields are read through and never recomputed, so the aggregate of one
    fold is that fold's own number.

    The σ is the *population* one (``numpy``'s default, ``ddof=0``): the folds are the whole
    population that was observed, not a sample drawn from a larger one, and the number exists to
    say how uneven the folds were - a 20 day window that made +8 % in one fold and -3 % in the
    next is reported as a mean and that spread, never as the mean alone.

    Beside the pairs, the aggregate carries ``trades_total``: how many trades the folds of the
    walk-forward opened *together*.  The mean answers "how busy is a fold", the sum answers "how
    big is the sample this walk-forward is read on" - and a score that rewards the number of
    trades over the whole out-of-sample period (SPEC_SMC.md §7.22) reads the sum, because a mean
    of four trades a fold says nothing about whether the study saw 36 trades or four.

    An empty list and a table missing one of the six fields are both hard errors: a
    walk-forward without folds has no out-of-sample evidence, and averaging the zeros of an
    unfilled table would invent it.
    """
    if not fold_results:
        raise ValueError("nothing to aggregate: the walk-forward produced no fold")
    missing = sorted(
        {field for field in FOLD_METRIC_FIELDS for table in fold_results if field not in table}
    )
    if missing:
        raise ValueError(
            f"the fold metrics need the {missing} field(s); they come from calc_metrics(...)"
        )

    aggregate: dict[str, float] = {"folds": float(len(fold_results))}
    for field in FOLD_METRIC_FIELDS:
        values = np.asarray([float(table[field]) for table in fold_results], dtype="float64")
        if not bool(np.isfinite(values).all()):
            raise ValueError(f"the {field!r} of a fold is not finite")
        aggregate[f"{field}_mean"] = round(float(values.mean()), 2)
        aggregate[f"{field}_std"] = round(float(values.std()), 2)
    aggregate["trades_total"] = round(sum(float(table["trades"]) for table in fold_results), 2)
    return aggregate


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """One walk-forward run: the metric table of every fold and their aggregate (Э6').

    ``fold_metrics[i]`` is the :func:`smc_zero.backtester.metrics.calc_metrics` table of the i-th
    fold of :func:`split_walkforward` - the same ``config`` reproduces that order, so the fold
    behind a table is always reconstructible - and ``aggregated`` the
    :func:`aggregate_fold_metrics` reading of the whole list.  ``config`` / ``backtest`` are the
    two configurations the run was cut and simulated with, carried the way
    :class:`smc_zero.backtester.engine.BacktestResult` carries its own.
    """

    fold_metrics: list[dict[str, float]]
    aggregated: dict[str, float]
    config: WalkForwardConfig
    backtest: BacktestConfig


def run_walkforward(
    df: pd.DataFrame,
    intents_fn: IntentBuilder,
    cfg: WalkForwardConfig | None = None,
    backtest: BacktestConfig | None = None,
    instrument: InstrumentSpec | None = None,
) -> WalkForwardResult:
    """Run one walk-forward over ``df``: build a fold's intents, simulate its test window (Э6').

    ``intents_fn`` is called once per fold as ``intents_fn(train, test)`` and returns what
    :func:`smc_zero.backtester.engine.run_backtest` accepts (intents frame or
    :class:`~smc_zero.strategy.base.TradeIntent` sequence); the second argument is the window the
    engine will simulate and therefore the only window the intents may be armed against - the
    engine re-derives every intent's bar from the tape it is given and raises on a mismatch, so
    intents built on the train window cannot be simulated at all (§7.10 п.65).  The first
    argument is the fit window: Э7' will use it to choose parameters, and the Э6' stub ignores it,
    which is exactly what "walk-forward without optimization" means::

        # v1: the Э4' chain runs on the test window with default parameters - train is unused.
        result = run_walkforward(
            tape,
            lambda train, test: build_intents(test, marks, cfg_strategy),
            cfg_wf,
            cfg_bt,
            ALFAFOREX_SPECS["EURUSD"],
        )

    The simulation is the Э5' engine unchanged, with the caller's fixed ``backtest`` config and
    ``instrument`` price list - the same costs, gates and metrics for every fold, so the folds
    differ only in the bars they cover.  A tape that cannot hold a single fold raises instead of
    returning an empty result: a walk-forward of nothing is a wrong config, not a finding.
    """
    config = WalkForwardConfig() if cfg is None else cfg
    backtest_cfg = BacktestConfig() if backtest is None else backtest
    spec = DEFAULT_INSTRUMENT if instrument is None else instrument
    folds = split_walkforward(df, config)
    if not folds:
        raise ValueError(
            f"a tape of {len(df)} bars holds no fold: "
            f"min_train_bars={config.min_train_bars} + "
            f"test_period_bars={config.test_period_bars} need more bars"
        )

    fold_metrics = [
        run_backtest(test, intents_fn(train, test), backtest_cfg, spec).metrics
        for train, test in folds
    ]
    return WalkForwardResult(
        fold_metrics=fold_metrics,
        aggregated=aggregate_fold_metrics(fold_metrics),
        config=config,
        backtest=backtest_cfg,
    )
