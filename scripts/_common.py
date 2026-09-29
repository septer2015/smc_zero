"""Shared plumbing of the Э8' runners: the tape path, the window slice and the report folder.

The two faces of the layer - :mod:`scripts.run_backtest` and :mod:`scripts.run_optimization` - are
command lines over the layers below, and what they have in common is plumbing only: where a tape
lives, how ``--start`` / ``--end`` are read, which price list a symbol maps to and how a report
folder is named.  Keeping it in one place means the two faces cannot drift apart on a report name,
and that neither of them owns a strategy number: every threshold stays in the config dataclasses
and every metric in the layers of :mod:`smc_zero` (constitution rules 4 and 5).

Two conventions of the window:

* a day is a whole UTC day - ``--start 2022-08-15 --end 2026-09-22`` covers the bars opened in
  ``[2022-08-15 00:00, 2026-09-23 00:00)``, so the end day is inside the window;
* the slice is taken *before* the markup is built, so a short window costs what it covers and not
  what the file holds - the level book of four years is 9040 instances, and none of them belongs to
  a five day run.

The date labels of a window come from the arguments, never from the data: a report folder says what
was asked for, and the summary it holds says what the tape actually carried.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from smc_zero.config import ALFAFOREX_SPECS, InstrumentSpec
from smc_zero.data_loader import TIMESTAMP_COLUMN, load_csv

#: Where the loader's CSVs live, relative to the working directory (rule 6: ``./`` paths).
DATA_DIR = Path("./data")
#: The default report root - the folder of the Э5' export, beside the data and never inside it.
REPORTS_DIR = Path("./reports")
#: The window of the shipped four-year tape (SPEC_SMC.md §7.10 п.62).
DEFAULT_START = "2022-08-15"
DEFAULT_END = "2026-09-22"
#: The entry timeframe: the hierarchy trades M15 entries (D1 bias -> H1 structure -> M15 entry).
DEFAULT_TIMEFRAME = "M15"
#: The symbol and the timeframe a run without arguments assumes.
DEFAULT_SYMBOL = "EURUSD"


def tape_path(symbol: str, timeframe: str) -> Path:
    """Return the tape of one symbol: ``./data/<SYMBOL>_<TIMEFRAME>.csv``."""
    return DATA_DIR / f"{symbol.upper()}_{timeframe.upper()}.csv"


def read_day(value: str) -> pd.Timestamp:
    """Read a ``YYYY-MM-DD`` argument as the UTC midnight of that day (``argparse`` ``type=``)."""
    try:
        return pd.Timestamp(value, tz="UTC")
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(f"{value!r} is not a date, use YYYY-MM-DD") from error


def slice_window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Return the bars opened in ``[start, end + 1 day)`` of ``df``, re-indexed from zero.

    The index is rebuilt on purpose: the engine and the trade log address bars by position, so the
    report of a window must speak in bars of that window and not in positions of the file the
    window was cut from.
    """
    upper = end + pd.Timedelta(days=1)
    window = df.loc[(df[TIMESTAMP_COLUMN] >= start) & (df[TIMESTAMP_COLUMN] < upper)]
    return window.reset_index(drop=True)


def load_windowed_tape(
    symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Load ``./data/<SYMBOL>_<TF>.csv`` and cut the requested window out of it.

    Raising is the contract: :class:`FileNotFoundError` when the tape is absent and
    :class:`ValueError` when no bar of it lies in the window, so a runner reports one line on
    stderr and returns a code instead of simulating an empty tape (which would report a flat curve
    as if it were a result).
    """
    path = tape_path(symbol, timeframe)
    if not path.is_file():
        raise FileNotFoundError(
            f"no tape at {path}: run from the repository root, or fetch the CSV into ./data"
        )
    window = slice_window(load_csv(path), start, end)
    if window.empty:
        raise ValueError(f"no bar of {path} is opened in {start:%Y-%m-%d} .. {end:%Y-%m-%d}")
    return window


def instrument_for(symbol: str) -> InstrumentSpec:
    """Return the C6 price list row of ``symbol``; an unpriced symbol is refused, not guessed.

    Rule 4: a run whose costs are unknown may not be presented as a result, so a symbol outside
    :data:`~smc_zero.config.ALFAFOREX_SPECS` stops the runner instead of being simulated for free.
    """
    row = ALFAFOREX_SPECS.get(symbol.upper())
    if row is None:
        known = ", ".join(sorted(ALFAFOREX_SPECS))
        raise ValueError(f"no C6 price list for {symbol.upper()!r}: v1 ships {known}")
    return row


def window_label(symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    """Return the label of a window: ``EURUSD_M15_2022-08-15_2026-09-22``."""
    return f"{symbol.upper()}_{timeframe.upper()}_{start:%Y-%m-%d}_{end:%Y-%m-%d}"


def report_folder(root: str | Path, name: str) -> Path:
    """Create and return ``root/name`` - the folder one run writes its report into."""
    folder = Path(root) / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def add_window_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments both runners share: the tape selector and the report root."""
    parser.add_argument(
        "--symbol", default=DEFAULT_SYMBOL, help="symbol of ./data/<SYMBOL>_<TIMEFRAME>.csv"
    )
    parser.add_argument(
        "--timeframe", default=DEFAULT_TIMEFRAME, help="entry timeframe of the tape (M15 for v1)"
    )
    parser.add_argument(
        "--start",
        type=read_day,
        default=read_day(DEFAULT_START),
        help="first day of the window, YYYY-MM-DD",
    )
    parser.add_argument(
        "--end",
        type=read_day,
        default=read_day(DEFAULT_END),
        help="last day of the window, YYYY-MM-DD (included)",
    )
    parser.add_argument("--report-dir", default=str(REPORTS_DIR), help="report root, ./reports")
