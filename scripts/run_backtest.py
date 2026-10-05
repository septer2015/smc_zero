"""``smc-backtest``: one fixed configuration over one tape, exported to ``./reports`` (Э8').

The runner is the user's single command over the whole project:

.. code-block:: text

    smc-backtest --symbol EURUSD --start 2022-08-15 --end 2026-09-22

(without an install, from the repository root::

    PYTHONPATH=src python -m scripts.run_backtest --start 2022-08-15 --end 2022-09-15

)

It loads ``./data/EURUSD_M15.csv``, cuts the requested window, builds the markup of *that window*
once (:func:`~smc_zero.optimizer.marks.build_tape_marks`: the HTF bias frames and the level book -
the same cache the optimizer of Э7' reads), arms the M15 entry chain of Э4' on it
(:func:`~smc_zero.strategy.intents.build_intents`), simulates the intents with the Э5' engine and
writes two files into ``./reports/backtest_<symbol>_<tf>_<start>_<end>/``:

* ``trades.csv`` / ``trades.parquet`` - the trade log, written by
  :func:`~smc_zero.backtester.reports.export_trades`;
* ``summary.txt`` - the metric table of :func:`~smc_zero.backtester.reports.format_summary`,
  the same text the run prints to stdout (costs, the ``has_costs`` stamp of rule 4 and the
  ``pf`` cap note included).

Nothing is searched here and nothing is recomputed.  ``--n-trials`` and ``--seed`` are accepted
only so that one command line can be retargeted at ``smc-optimize`` without editing it, and the
metric table is the one the engine already carries: ``result.metrics`` *is* ``calc_metrics`` applied
by :func:`~smc_zero.backtester.engine.run_backtest` to this very trade log and equity curve, so a
second call would only copy numbers the report layer reads from the result (§7.9).

A wrong request is refused, never guessed: a missing tape, an empty window and a symbol without a
C6 price list all end as one line on stderr and exit code 2 (rule 4 - an uncosted run is not a
result).

A live run takes its numbers from the config of the winner instead of the project defaults (§7.19):

.. code-block:: text

    smc-backtest --config-path configs/live_eurusd_m15.yaml --start 2022-08-15 --end 2022-09-15

The four blocks of that file (``symbol`` / ``strategy`` / ``broker`` / ``backtest``) fill
:class:`~smc_zero.config.StrategyConfig` and :class:`~smc_zero.config.BacktestConfig` through
:func:`scripts._common.live_inputs`, and an argument typed on the command line still wins over the
file.  The bias frame is asked for the agreement mode of the run and not for the default one:
the config may have moved that knob.

``--data-source`` picks the base of the tape (Э11'.1): ``project`` reads ``./data`` as before, while
``mt5`` reads a raw MetaTrader 5 export under ``SMC_DATA_DIR`` (``~/_data/mt5`` by default), so a
fresh broker export feeds the same chain without a manual conversion.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from scripts import _common
from smc_zero.backtester import export_trades, format_summary, run_backtest
from smc_zero.optimizer import build_tape_marks
from smc_zero.strategy.intents import build_intents


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of ``smc-backtest``."""
    parser = argparse.ArgumentParser(
        prog="smc-backtest",
        description=(
            "Simulate one fixed SMC configuration over ./data and export its report to ./reports."
        ),
    )
    _common.add_window_arguments(parser)
    _common.add_config_argument(parser)
    _common.add_data_source_argument(parser)
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="accepted for symmetry with smc-optimize: a backtest runs one configuration",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="accepted for symmetry with smc-optimize: a backtest draws nothing",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    """Simulate the window of ``args`` with its config and write the report of the run."""
    try:
        symbol, timeframe, strategy, backtest = _common.live_inputs(args)
        tape = _common.load_windowed_tape(
            symbol, timeframe, args.start, args.end, source=args.data_source
        )
        instrument = _common.instrument_for(symbol)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    # The markup is built on the window and reused by everything below: the bias frame the chain
    # asks for a direction and the level book it tries to sweep.  The agreement mode is the one the
    # config of the run names, not the cache default: ``bias_frame`` is asked for it explicitly.
    marks = build_tape_marks(tape, strategy)
    chain = build_intents(tape, marks.bias_frame(strategy.bias.agreement), marks.levels, strategy)
    result = run_backtest(tape, chain.intents, backtest, instrument)

    summary = format_summary(result)
    folder = _common.report_folder(
        args.report_dir, f"backtest_{_common.window_label(symbol, timeframe, args.start, args.end)}"
    )
    export_trades(result, folder, stem="trades")
    (folder / "summary.txt").write_text(summary + "\n", encoding="utf-8")

    print(summary)
    print(f"report: {folder}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` (``None`` = ``sys.argv``) and run one backtest; the console entry point."""
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
