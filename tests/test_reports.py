"""Report tests: the summary reads the table, the export keeps the log (Э5').

The run is one synthetic EURUSD short that fills on the bar after its signal and reaches a
60 pip target, so the summary has a known profit to render (+57.20 = 60 pips - the 1.4 pip
spread - the $1.40 commission of the round turn on the 0.1 lot; a limit entry and a target hit
pay no slippage) and the export has a known row to write.

What is pinned here is that the report layer *spends no arithmetic* - the numbers it prints are
the ones of ``calc_metrics`` - that the cost assumptions are always on the page, that the
``has_costs`` warning of rule 4 is stamped on the run that switches the costs off (Э10': a zero
broker profile is the documented way to write one down), and that ``export_trades`` writes both
formats the way the CSV and the parquet read back as the same table (``./reports`` by default,
never inside ``./data``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from smc_zero.backtester.engine import TRADE_COLUMNS, run_backtest
from smc_zero.backtester.reports import REPORTS_DIR, export_trades, format_summary
from smc_zero.config import BacktestConfig, BrokerSpec, InstrumentSpec, RiskConfig
from smc_zero.indicators.levels import PDH
from smc_zero.strategy.base import TradeIntent

DAY = "2026-06-09"  # a Tuesday, inside Alfa's week
BAR = pd.Timedelta(minutes=15)
FLAT = (1.0950, 1.0960, 1.0940, 1.0950)
#: A profile that charges nothing at all, and the price list row that carries it (Э10').  The row
#: and the profile have to agree on the symbol-level numbers, so an uncosted run needs both.
UNCOSTED = BrokerSpec(
    spread_pip=0.0,
    commission_per_lot_usd=0.0,
    slippage_pip=0.0,
    swap_long_pip=0.0,
    swap_short_pip=0.0,
)


def _result(instrument: InstrumentSpec | None = None, **overrides: object):
    """Run one winning short: signal on bar 2, fill on 3, target on 4."""
    rows = [
        FLAT,
        FLAT,
        (1.0950, 1.1005, 1.0945, 1.0950),
        (1.0950, 1.1008, 1.0945, 1.0950),
        (1.0950, 1.0960, 1.0935, 1.0940),
        FLAT,
    ]
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    frame.insert(0, "timestamp", [pd.Timestamp(f"{DAY} 08:00", tz="UTC") + BAR * i for i in range(6)])
    frame["volume"] = 1
    frame["is_closed"] = True
    intent = TradeIntent(
        open_time=frame["timestamp"].iloc[2],
        bar=2,
        side="short",
        entry=1.1000,
        sl=1.1030,
        tp=1.0940,
        tp_source="liquidity",
        sl_source="sweep_extreme",
        sl_pips=30.0,
        rr=2.0,
        level_name=PDH,
        level_date=pd.Timestamp(DAY),
        level_price=1.1030,
        setup_type="fresh",
        sweep_bar=0,
        choch_bar=1,
        fvg_bar=2,
    )
    return run_backtest(frame, [intent], BacktestConfig(**overrides), instrument)  # type: ignore[arg-type]


def test_the_summary_prints_the_metrics_and_the_cost_assumptions() -> None:
    """Every figure of the block is a metric, and the costs of the run are printed with it."""
    result = _result()

    text = format_summary(result)
    assert "EURUSD M15" in text
    assert f"+{result.metrics['profit']:.2f}" in text
    assert f"{result.metrics['return_pct']:+.2f} %" in text
    assert "spread 1.4 pip" in text  # the EURUSD row of the price list
    assert "swap long -0.70 / short +0.00 pip per night" in text
    assert "margin denominator 1000.00 (C7)" in text

    # The shipped profile is Alfa's, so all three costs are charged and rule 4 is silent.
    assert result.config.risk.has_costs is True
    assert "slippage 0.20 pip per market leg, commission 7.00 per lot" in text
    assert "WARNING" not in text

    # The uncosted run is the one built on purpose (Э10'): a zero profile and the row that carries
    # it, on which the report has to stamp that the curve above it is not a profit (rule 4).
    uncosted = format_summary(
        _result(
            risk=RiskConfig(broker=UNCOSTED),
            instrument=InstrumentSpec(broker=UNCOSTED),
        )
    )
    assert uncosted.count("WARNING: RiskConfig.has_costs is False") == 1
    assert "spread 0.0 pip" in uncosted
    assert "commission 0.00 per lot" in uncosted

    # A run may charge its own *account-level* numbers - the commission and the slippage - while
    # still trading the row's symbol (the engine compares only the symbol-level ones).  The Э5'
    # commission keyword migrates by money: 0.35 a trade on the 0.1 lot is 1.75 per lot, which is
    # what the report prints, and a charged slippage keeps the run costed.
    configured = format_summary(_result(risk=RiskConfig(commission=0.35, slippage=0.0001)))
    assert "WARNING" not in configured
    assert "spread 1.4 pip" in configured
    assert "slippage 1.00 pip per market leg, commission 1.75 per lot" in configured


def test_the_summary_flags_the_profit_factor_cap() -> None:
    """A factor sitting at ``pf_cap`` is a cap and is labelled as one."""
    result = _result()
    assert result.metrics["pf"] == pytest.approx(result.config.pf_cap)

    text = format_summary(result, title="one winner")
    assert text.startswith("one winner")
    assert "the profit factor is capped at 5" in text

    losing = _result(max_orders_day=0)  # nothing is placed, so there is no factor at the cap
    assert "capped" not in format_summary(losing)


def test_the_export_writes_the_log_under_the_reports_folder(tmp_path: Path) -> None:
    """The CSV and the parquet are the same table, and the folder is created on demand."""
    result = _result()
    folder = tmp_path / "reports"

    paths = export_trades(result, directory=folder)
    assert paths["csv"].parent == folder
    assert paths["csv"].name == "trades.csv"
    assert paths["parquet"].name == "trades.parquet"

    csv = pd.read_csv(paths["csv"])
    parquet = pd.read_parquet(paths["parquet"])
    assert list(csv.columns) == list(TRADE_COLUMNS) == list(parquet.columns)
    assert len(csv) == len(result.trades) == 1
    assert parquet.loc[0, "profit"] == pytest.approx(result.metrics["profit"])
    assert REPORTS_DIR == "./reports"  # never inside ./data
