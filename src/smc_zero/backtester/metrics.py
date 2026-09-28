"""Performance metrics of a run: the trade log plus the per-bar equity curve (Э5').

The table is prod's summary block (its ``calc_metrics``, line for line) as a plain
dictionary, so :func:`smc_zero.backtester.reports.format_summary` only formats and never
recomputes.  Three decisions of this layer are pinned here:

* **``return_pct`` is measured against ``BacktestConfig.initial_capital``** - the
  backtester's own 10000 (SPEC_SMC.md §5 п.10) - and *never* against
  ``RiskConfig.deposit`` (1000).  The deposit is the C7 denominator of the reported
  ``risk_pct`` of a stop; the capital is what the equity curve starts from.  Mixing the two
  turns a 10 % return into a 1000 % one, which is the C7 trap of §7.9 and the mutation the
  tests of this module are built around.
* **``pf`` is capped at ``cfg.pf_cap``** (prod's ``PF_CAP = 5.0``) inside the metric, so
  the published number can never come from a single outlier.  A run *without* losses has
  no profit factor at all: it is reported as the cap (prod returned ``999`` and capped it
  at scoring time instead - see §7.9).
* **``sharpe`` is prod's scale** - the mean/σ of the *bar-to-bar* equity returns times
  ``sqrt(cfg.sharpe_bars_per_day)`` (96 M15 bars of a day) - not an annualisation over 252
  trading days.  The equity curve is flat between trades, so the series is prod's own.

``sl_sweep_cnt`` / ``sl_bos_cnt`` keep prod's keys: prod counts trades by level
source (SWEEP levels against BOS levels).  v1 has no BOS *levels* at all (§7.8 п.33), so
every trade of the layer is a sweep setup and the BOS slot stays empty; the pair is carried
for report parity and the reading is flagged in §7.9.  ``median_risk_pct`` is the one key
prod does not have: §7.8 п.37 promises that the report prints the median risk per trade.

Costs are read from the log per trade (``spread_cost`` / ``swap_cost``, in account
currency), never recomputed here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smc_zero.config import BacktestConfig

#: Result codes of a trade: the engine writes them, the metrics read them.  They live here
#: because this module is the leaf of the layer - the engine imports it, not the reverse.
TP_RESULT = 1
SL_RESULT = -1
EOD_RESULT = 0

#: The trade-log columns :func:`calc_metrics` reads (the engine writes them, Э5').
METRIC_COLUMNS: tuple[str, ...] = (
    "profit",
    "spread_cost",
    "swap_cost",
    "limit_dur_min",
    "sl_pips",
    "tp_pips",
    "result",
    "risk_pct",
)


def _empty_metrics(cfg: BacktestConfig) -> dict[str, float]:
    """Return the metric table of a run without a single trade (the equity stays flat)."""
    return {
        "trades": 0,
        "profit": 0.0,
        "win_rate": 0.0,
        "pf": 0.0,
        "max_dd": 0.0,
        "sharpe": 0.0,
        "expected": 0.0,
        "final_balance": round(cfg.initial_capital, 2),
        "return_pct": 0.0,
        "spread_cost": 0.0,
        "swap_cost": 0.0,
        "avg_limit_dur": 0.0,
        "avg_sl_pips": 0.0,
        "avg_tp_pips": 0.0,
        "sl_sweep_cnt": 0,
        "sl_bos_cnt": 0,
        "eod_cnt": 0,
        "median_risk_pct": 0.0,
    }


def _mean(values: list[float]) -> float:
    """Return the rounded mean of ``values``, or ``0.0`` when there is nothing to average."""
    return round(float(np.mean(values)), 1) if values else 0.0


def calc_metrics(
    trades: pd.DataFrame,
    equity: pd.Series,
    cfg: BacktestConfig | None = None,
) -> dict[str, float]:
    """Return the metric table of a run: its trade log, its equity curve and its config.

    ``trades`` is the engine's trade log (:data:`smc_zero.backtester.engine.TRADE_COLUMNS`)
    and ``equity`` the per-bar curve it built, indexed by the entry tape's stamps.  A log
    without the columns of :data:`METRIC_COLUMNS` is a hard error (the layer never invents a
    metric from absent data), an empty log returns :func:`_empty_metrics` - the capital
    untouched and every count zero.

    Drawdown is the deepest peak-to-trough drop of the curve in percent, ``expected`` the
    mean profit per trade, ``avg_limit_dur`` the mean age of the limits in minutes,
    ``avg_sl_pips`` / ``avg_tp_pips`` the mean stop and target distances of the log and
    ``median_risk_pct`` the median C7 risk of a trade (§7.8 п.37).  The costs are summed as
    they were charged per trade: the spread is a cost, the swap is signed (a negative swap
    leg credits the account).
    """
    config = BacktestConfig() if cfg is None else cfg
    missing = [name for name in METRIC_COLUMNS if name not in trades.columns]
    if missing:
        raise ValueError(f"the trade log needs the {missing} column(s); the engine writes them")

    values = np.asarray(equity.to_numpy(dtype="float64"), dtype="float64")
    if values.size and not bool(np.isfinite(values).all()):
        raise ValueError("the equity curve contains non-finite values")
    if trades.empty:
        return _empty_metrics(config)

    profits = trades["profit"].to_numpy(dtype="float64")
    wins = profits[profits > 0]
    losses = profits[profits < 0]

    gross_loss = float(abs(losses.sum()))
    # A run without losses has no finite profit factor: the cap is the honest report of it.
    pf = min(float(wins.sum()) / gross_loss, config.pf_cap) if gross_loss else config.pf_cap

    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / peak
    max_dd = float(abs(drawdown.min()) * 100.0)

    returns = pd.Series(values).pct_change().dropna()
    sharpe = (
        float(returns.mean() / returns.std() * np.sqrt(config.sharpe_bars_per_day))
        if len(returns) > 1 and returns.std() > 0
        else 0.0
    )

    durations = [float(value) for value in trades["limit_dur_min"]]
    sl_pips = [float(value) for value in trades["sl_pips"]]
    tp_pips = [float(value) for value in trades["tp_pips"]]
    risks = trades["risk_pct"].to_numpy(dtype="float64")
    results = trades["result"].to_numpy()

    return {
        "trades": len(trades),
        "profit": round(float(profits.sum()), 2),
        "win_rate": round(len(wins) / len(trades) * 100.0, 2),
        "pf": round(pf, 3),
        "max_dd": round(max_dd, 2),
        "sharpe": round(sharpe, 3),
        "expected": round(float(np.mean(profits)), 2),
        "final_balance": round(float(values[-1]), 2),
        "return_pct": round((float(values[-1]) / config.initial_capital - 1) * 100.0, 2),
        "spread_cost": round(float(trades["spread_cost"].sum()), 2),
        "swap_cost": round(float(trades["swap_cost"].sum()), 2),
        "avg_limit_dur": _mean(durations),
        "avg_sl_pips": _mean(sl_pips),
        "avg_tp_pips": _mean(tp_pips),
        # v1 trades sweep levels only (BOS levels are out of v1, §7.8 п.33): the two slots
        # are prod's, and the BOS one has nothing to count - see the module docstring.
        "sl_sweep_cnt": len(trades),
        "sl_bos_cnt": 0,
        "eod_cnt": int((results == EOD_RESULT).sum()),
        # §7.8 п.37: the report prints the median risk per trade (prod's own C7 column).
        "median_risk_pct": round(float(np.median(risks)), 2),
    }
