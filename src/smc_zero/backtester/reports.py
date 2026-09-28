"""Reports: the text summary of a run and the trade table it is read with (Э5').

Two faces, no arithmetic: :func:`format_summary` renders the metric table that
:func:`smc_zero.backtester.metrics.calc_metrics` already computed and :func:`export_trades`
writes the engine's trade log next to it.  Recomputing anything here would create a second
copy of a number, which is exactly what the metric layer exists to prevent.

Rule 4 of the constitution is enforced on the text: every summary prints the cost assumptions
of the run it describes - the spread, the swap legs, the slippage and the commission of the
price list and the risk profile - and a run whose ``RiskConfig`` does not carry all three of
``has_costs`` is stamped with a warning, because a positive curve without costs is not a
profit.  The median risk per trade (§7.8 п.37) and the ``pf`` cap are flagged the same way: a
reading a reader could mistake for a result is marked as a reading.

Files land in ``./reports`` (never inside ``./data``), created on demand.  The equity and
drawdown *plots* of the stub stay deferred: they are presentation, they need a headless
matplotlib backend, and §7.9 records the deferral.
"""

from __future__ import annotations

from pathlib import Path

from smc_zero.backtester.engine import BacktestResult

#: Where the export writes: a folder beside the data, never inside it.
REPORTS_DIR = "./reports"

#: The metric rows of the summary, in the order they are printed: ``key``, label, format.
#: The formats are the reading of a key, and the keys are the ones of
#: :func:`smc_zero.backtester.metrics.calc_metrics` - nothing else is ever rendered here.
_SUMMARY_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("trades", "trades", "{:d}"),
    ("win_rate", "win rate", "{:.2f} %"),
    ("profit", "profit", "{:+.2f}"),
    ("pf", "profit factor", "{:.3f}"),
    ("max_dd", "max drawdown", "{:.2f} %"),
    ("sharpe", "sharpe (per bar)", "{:.3f}"),
    ("expected", "expected per trade", "{:+.2f}"),
    ("final_balance", "final balance", "{:.2f}"),
    ("return_pct", "return", "{:+.2f} %"),
    ("median_risk_pct", "median risk per trade", "{:.2f} %"),
    ("spread_cost", "spread cost", "{:.2f}"),
    ("swap_cost", "swap result", "{:+.2f}"),
    ("avg_limit_dur", "avg limit age", "{:.1f} min"),
    ("avg_sl_pips", "avg SL", "{:.1f} pips"),
    ("avg_tp_pips", "avg TP", "{:.1f} pips"),
    ("eod_cnt", "end-of-day closes", "{:d}"),
    ("sl_sweep_cnt", "trades on swept levels", "{:d}"),
    ("sl_bos_cnt", "trades on BOS levels", "{:d}"),
)


def _span(result: BacktestResult) -> str:
    """Return the text span of a run: the first and last close stamp of its equity curve."""
    index = result.equity.index
    if len(index) == 0:
        return "empty tape"
    return f"{index[0]:%Y-%m-%d %H:%M} .. {index[-1]:%Y-%m-%d %H:%M} UTC"


def format_summary(result: BacktestResult, *, title: str | None = None) -> str:
    """Render one run as the text block a reader gets: metrics, costs, then the flags.

    ``title`` defaults to the symbol, the entry timeframe and the span of the run.  The rows are
    :data:`_SUMMARY_FIELDS` and the numbers come from ``result.metrics`` alone - the report
    layer never recomputes one.  The block closes with the cost assumptions of the run
    (``result.instrument`` and ``result.config.risk``), because rule 4 wants them on every
    report, and with three readings a reader could mistake for a result:

    * ``RiskConfig.has_costs`` is ``False`` - the costs are not all configured, so the curve must
      not be read as a profit;
    * ``pf`` sits at ``BacktestConfig.pf_cap`` - the log has no losing trade, so the factor is a
      cap rather than a measurement;
    * the ``risk_warning`` count - trades whose C7 risk is above ``warning_risk_pct``.
    """
    config = result.config
    instrument = result.instrument
    risk = config.risk
    lines = [
        title
        or f"SMC backtest: {instrument.symbol} {config.timeframes.ltf} ({_span(result)})",
        f"  {'initial capital':<24} {config.initial_capital:.2f}",
    ]
    for key, label, pattern in _SUMMARY_FIELDS:
        lines.append(f"  {label:<24} {pattern.format(result.metrics[key])}")
    lines.append(
        f"  {'costs':<24} spread {instrument.spread_pip:.1f} pip, "
        f"swap long {instrument.swap_long_pip:+.2f} / short {instrument.swap_short_pip:+.2f} "
        f"pip per night, slippage {risk.slippage / instrument.pip_size:.1f} pip, "
        f"commission {risk.commission:.2f}"
    )
    lines.append(
        f"  {'risk profile':<24} lot {risk.lot:g} on {instrument.contract_size:.0f} units, "
        f"leverage {risk.leverage:g}, margin denominator {risk.deposit:.2f} (C7), "
        f"warning above {risk.warning_risk_pct:.2f} %"
    )
    flagged = int(result.trades["risk_warning"].sum()) if not result.trades.empty else 0
    lines.append(f"  {'risk flags':<24} {flagged} trade(s) above the warning threshold")
    ledger = len(result.rejections)
    lines.append(f"  {'ledger':<24} {ledger} intent(s) that became no trade")
    if not risk.has_costs:
        lines.append(
            "  WARNING: RiskConfig.has_costs is False - the curve above is not a profit "
            "(constitution rule 4)"
        )
    if result.metrics["pf"] >= config.pf_cap:
        lines.append(
            f"  NOTE: the profit factor is capped at {config.pf_cap:g} (no losing trade in the log)"
        )
    return "\n".join(lines)


def export_trades(
    result: BacktestResult,
    directory: str | Path = REPORTS_DIR,
    stem: str = "trades",
) -> dict[str, Path]:
    """Write the trade log of a run to ``directory`` as CSV and Parquet; return the two paths.

    Both formats are written on purpose: the CSV is what a human opens, the Parquet keeps the
    dtypes (``pyarrow`` is a project dependency).  The folder is created when missing and the
    files are named ``trades.csv`` / ``trades.parquet`` under it (``./reports`` by default,
    never inside ``./data``).
    """
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    paths = {"csv": folder / f"{stem}.csv", "parquet": folder / f"{stem}.parquet"}
    result.trades.to_csv(paths["csv"], index=False)
    result.trades.to_parquet(paths["parquet"], index=False)
    return paths

