"""Backtest engine (stub, no logic yet).

Planned contract: consume the signal frame produced by the strategy and simulate
orders bar by bar on the M15 tape with explicit costs from
:class:`smc_zero.config.RiskConfig`, position size derived from ``risk_pct``,
SL/TP handling and a trade log; the ``backtesting`` library is the intended
driver, but its API must be verified on the installed version before use.
"""

from __future__ import annotations

# TODO(phase-backtest): map signals to orders with spread/slippage/commission.
# TODO(phase-backtest): percentage-risk sizing, SL/TP, per-trade trade log.
# TODO(phase-backtest): never evaluate a signal before its bar close_time.
# TODO(tests): synthetic trade log test with negative edge after costs.
