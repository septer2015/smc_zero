"""Metric tests: the table of a run, its denominators and its caps (Э5').

The log is hand-written, so every number below can be checked by hand: three trades of +100,
-50 and -25 are a profit of 25, a profit factor of 100 / 75 = 1.333 and an expected value of
8.33, while the curve walking 10 000 -> 10 100 -> 10 050 -> 10 025 draws down 0.74 %.

The two mutations this module is one line away from, and the test each one must break:

* m1 "``risk.deposit`` as the denominator of ``return_pct``" - the C7 trap the layer guards
  against: the same 1 000 dollar profit would read +1000 % instead of +10 %; breaks
  :func:`test_the_return_is_measured_against_the_initial_capital`;
* m2 "no profit-factor cap" - a log without a losing trade would report an unbounded factor
  instead of ``BacktestConfig.pf_cap``; breaks
  :func:`test_the_profit_factor_of_a_run_without_a_loss_is_capped`.
"""

from __future__ import annotations

import pandas as pd
import pytest

from smc_zero.backtester.metrics import SL_RESULT, TP_RESULT, calc_metrics
from smc_zero.config import BacktestConfig

START = pd.Timestamp("2026-06-09 08:00", tz="UTC")
BAR = pd.Timedelta(minutes=15)


def _curve(values: list[float]) -> pd.Series:
    """Build the per-bar equity curve the engine hands to the metric layer."""
    stamps = pd.date_range(START, periods=len(values), freq=BAR, tz="UTC")
    return pd.Series(values, index=stamps, name="equity")


def _fill(value: float | list[float], count: int) -> list[float]:
    """Return ``value`` as a column of ``count`` entries."""
    return list(value) if isinstance(value, list) else [value] * count


def _log(
    profits: list[float],
    *,
    results: list[int] | None = None,
    spread: float | list[float] = 1.4,
    swap: float | list[float] = 0.0,
    duration: float | list[float] = 15.0,
    sl_pips: float | list[float] = 30.0,
    tp_pips: float | list[float] = 60.0,
    risk_pct: float | list[float] = 3.0,
) -> pd.DataFrame:
    """Build a trade log with the columns the metric layer reads (the engine writes them)."""
    count = len(profits)
    return pd.DataFrame(
        {
            "profit": profits,
            "spread_cost": _fill(spread, count),
            "swap_cost": _fill(swap, count),
            "limit_dur_min": _fill(duration, count),
            "sl_pips": _fill(sl_pips, count),
            "tp_pips": _fill(tp_pips, count),
            "result": results if results is not None else [TP_RESULT] * count,
            "risk_pct": _fill(risk_pct, count),
        }
    )


def test_the_table_reads_the_log_and_the_curve() -> None:
    """Each key of the table is a reading of the log or of the curve, never a guess."""
    trades = _log([100.0, -50.0, -25.0])
    metrics = calc_metrics(trades, _curve([10_000.0, 10_100.0, 10_050.0, 10_025.0]), None)

    assert metrics["trades"] == 3
    assert metrics["profit"] == pytest.approx(25.0)
    assert metrics["win_rate"] == pytest.approx(33.33)
    assert metrics["pf"] == pytest.approx(100.0 / 75.0, abs=0.001)
    assert metrics["expected"] == pytest.approx(8.33)
    assert metrics["final_balance"] == pytest.approx(10_025.0)
    assert metrics["max_dd"] == pytest.approx(0.74, abs=0.01)  # 10 100 -> 10 025
    assert metrics["return_pct"] == pytest.approx(0.25)


def test_the_return_is_measured_against_the_initial_capital() -> None:
    """``return_pct`` divides by ``BacktestConfig.initial_capital``, not by C7's deposit (m1)."""
    config = BacktestConfig()
    assert config.initial_capital == 10_000.0
    assert config.risk.deposit == 1_000.0

    metrics = calc_metrics(_log([1_000.0]), _curve([10_000.0, 11_000.0]), config)
    assert metrics["final_balance"] == pytest.approx(11_000.0)
    assert metrics["return_pct"] == pytest.approx(10.0)


def test_the_profit_factor_of_a_run_without_a_loss_is_capped() -> None:
    """A log with no losing trade reports the cap of the config, not an infinite factor (m2)."""
    config = BacktestConfig()
    metrics = calc_metrics(_log([10.0, 20.0]), _curve([10_000.0, 10_030.0]), config)

    assert metrics["pf"] == pytest.approx(config.pf_cap)
    assert metrics["win_rate"] == pytest.approx(100.0)
    assert metrics["max_dd"] == 0.0


def test_the_drawdown_and_the_sharpe_are_read_from_the_curve() -> None:
    """A flat curve has no drawdown and no dispersion; a rising one has both keys positive."""
    flat = calc_metrics(_log([10.0]), _curve([10_000.0] * 10), BacktestConfig())
    assert flat["max_dd"] == 0.0
    assert flat["sharpe"] == 0.0

    rising = calc_metrics(
        _log([390.0]),
        _curve([10_000.0 + 10.0 * step for step in range(40)]),
        BacktestConfig(),
    )
    assert rising["max_dd"] == 0.0
    assert rising["sharpe"] > 0.0


def test_the_counts_and_the_averages_keep_prod_keys() -> None:
    """The two level slots of prod, the EOD count and the averages of the log."""
    trades = _log(
        [100.0, -25.0],
        results=[TP_RESULT, SL_RESULT],
        swap=-0.7,
        duration=[15.0, 30.0],
        sl_pips=[30.0, 20.0],
        tp_pips=[60.0, 40.0],
        risk_pct=[3.0, 2.0],
    )
    metrics = calc_metrics(trades, _curve([10_000.0, 10_075.0]), BacktestConfig())

    assert metrics["sl_sweep_cnt"] == 2  # v1 trades swept levels only (§7.8 п.33)
    assert metrics["sl_bos_cnt"] == 0
    assert metrics["eod_cnt"] == 0
    assert metrics["avg_limit_dur"] == pytest.approx(22.5)
    assert metrics["avg_sl_pips"] == pytest.approx(25.0)
    assert metrics["avg_tp_pips"] == pytest.approx(50.0)
    assert metrics["median_risk_pct"] == pytest.approx(2.5)
    assert metrics["spread_cost"] == pytest.approx(2.8)
    assert metrics["swap_cost"] == pytest.approx(-1.4)


def test_an_empty_log_is_the_untouched_capital() -> None:
    """No trade at all: every count is zero, the balance is the capital that started the run."""
    metrics = calc_metrics(_log([]), _curve([10_000.0, 10_000.0]), BacktestConfig())

    assert metrics["trades"] == 0
    assert metrics["profit"] == 0.0
    assert metrics["pf"] == 0.0
    assert metrics["final_balance"] == pytest.approx(10_000.0)
    assert metrics["return_pct"] == 0.0
    assert metrics["median_risk_pct"] == 0.0


def test_a_log_without_the_metric_columns_is_a_hard_error() -> None:
    """A metric is never invented from absent data: a missing column is a hard error."""
    trades = _log([10.0]).drop(columns=["risk_pct"])
    with pytest.raises(ValueError, match="risk_pct"):
        calc_metrics(trades, _curve([10_000.0]), BacktestConfig())


def test_a_curve_with_a_hole_is_a_hard_error() -> None:
    """A non-finite equity value is a broken run, not a number to report."""
    with pytest.raises(ValueError, match="non-finite"):
        calc_metrics(_log([10.0]), _curve([10_000.0, float("nan")]), BacktestConfig())