"""Console runners of the project (Э8'): ``smc-backtest`` and ``smc-optimize``.

The package is deliberately thin: a runner is a *face*, not a layer.  It reads its arguments, calls
the layers below in one fixed order and writes what those layers returned.  No indicator, no metric
and no threshold is defined here (constitution rules 4 and 5), and nothing under this package may be
imported by :mod:`smc_zero` - the dependency points one way only.
"""
