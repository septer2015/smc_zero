"""Smoke tests: the package, its config objects and the loader are importable."""

from __future__ import annotations

import smc_zero


def test_package_imports() -> None:
    assert smc_zero.__version__


def test_timeframe_hierarchy_defaults() -> None:
    config = smc_zero.TimeframeConfig()
    assert (config.ltf, config.mtf, config.htf) == ("M15", "H1", "D1")


def test_positions_are_immutable() -> None:
    config = smc_zero.RiskConfig(risk_pct=0.5)
    try:
        config.risk_pct = 1.0  # type: ignore[misc]
    except AttributeError:
        return
    raise AssertionError("RiskConfig must be frozen")


def test_backtest_config_defaults() -> None:
    config = smc_zero.BacktestConfig()
    assert config.initial_capital > 0
    assert config.drop_unclosed is True
    # The shipped profile is Alfa's (C6), so a run is charged its spread, its slippage and its
    # commission unless the caller builds an uncosted broker profile on purpose (Э10').
    assert config.risk.has_costs is True
    assert config.risk.broker.spread_pip > 0
    assert config.risk.broker.commission_per_lot_usd > 0


def test_data_loader_is_importable() -> None:
    from smc_zero import data_loader

    assert callable(data_loader.load_csv)
    assert callable(data_loader.align_htf_to_ltf)
    assert data_loader.period_for("m15").total_seconds() == 900
