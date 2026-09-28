"""Backtest engine, metrics and reports (Э5'): run one set of intents on one tape.

The layer is a chain of one-way imports - :mod:`smc_zero.backtester.metrics` holds the numbers,
:mod:`smc_zero.backtester.engine` the event loop and :mod:`smc_zero.backtester.reports` the
text - so the faces below are the whole public surface:

* :func:`run_backtest` simulates the intents of the strategy layer on a closed M15 tape and
  returns a :class:`BacktestResult` (trade log, equity curve, ledger, metrics);
* :func:`calc_metrics` is the metric table on its own (a trade log and a curve in, numbers out);
* :func:`format_summary` / :func:`export_trades` are the reporting faces.

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

__all__ = [
    "DEFAULT_INSTRUMENT",
    "REASON_LIMIT_NOT_FILLED",
    "REASON_NO_RESULT",
    "REASON_OFF_HOURS",
    "REASON_ORDER_BUDGET",
    "REASON_SL_CAP",
    "TRADE_COLUMNS",
    "BacktestResult",
    "calc_metrics",
    "export_trades",
    "format_summary",
    "run_backtest",
]
