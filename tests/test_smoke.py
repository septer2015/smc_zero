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
    assert config.risk.has_costs is False  # costs must be set explicitly


def test_data_loader_is_importable() -> None:
    from smc_zero import data_loader

    assert callable(data_loader.load_csv)
    assert callable(data_loader.align_htf_to_ltf)
    assert data_loader.period_for("m15").total_seconds() == 900
