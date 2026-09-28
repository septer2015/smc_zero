"""Backtest engine, metrics and reports (Э5') plus the walk-forward of Э6'.

The layer is a chain of one-way imports - :mod:`smc_zero.backtester.metrics` holds the numbers,
:mod:`smc_zero.backtester.engine` the event loop, :mod:`smc_zero.backtester.reports` the text and
:mod:`smc_zero.backtester.walkforward` the out-of-sample folds over all three - so the faces below
are the whole public surface:

* :func:`run_backtest` simulates the intents of the strategy layer on a closed M15 tape and
  returns a :class:`BacktestResult` (trade log, equity curve, ledger, metrics);
* :func:`calc_metrics` is the metric table on its own (a trade log and a curve in, numbers out);
* :func:`format_summary` / :func:`export_trades` are the reporting faces;
* :func:`split_walkforward` / :func:`aggregate_fold_metrics` / :func:`run_walkforward` are the
  walk-forward of Э6': folds of one tape, the mean and σ of their metrics, and the runner that
  joins them to :func:`run_backtest` (no optimization - that is Э7').

Nothing here may run without explicit costs: ``RiskConfig`` must carry non-zero commission,
spread and slippage before a result is reported as profitable (constitution rule 4), and
:func:`format_summary` stamps every run whose costs are not fully configured.
"""

from __future__ import annotations

from smc_zero.backtester.engine import (
    DEFAULT_INSTRUMENT,
    REASON_LIMIT_NOT_FILLED,
    REASON_NO_RESULT,
    REASON_OFF_HOURS,
    REASON_ORDER_BUDGET,
    REASON_SL_CAP,
    TRADE_COLUMNS,
    BacktestResult,
    run_backtest,
)
from smc_zero.backtester.metrics import calc_metrics
from smc_zero.backtester.reports import export_trades, format_summary
from smc_zero.backtester.walkforward import (
    FOLD_METRIC_FIELDS,
    IntentBuilder,
    WalkForwardResult,
    aggregate_fold_metrics,
    run_walkforward,
    split_walkforward,
)

__all__ = [
    "DEFAULT_INSTRUMENT",
    "FOLD_METRIC_FIELDS",
    "REASON_LIMIT_NOT_FILLED",
    "REASON_NO_RESULT",
    "REASON_OFF_HOURS",
    "REASON_ORDER_BUDGET",
    "REASON_SL_CAP",
    "TRADE_COLUMNS",
    "BacktestResult",
    "IntentBuilder",
    "WalkForwardResult",
    "aggregate_fold_metrics",
    "calc_metrics",
    "export_trades",
    "format_summary",
    "run_backtest",
    "run_walkforward",
    "split_walkforward",
]
