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

The second shared thing is the live config (§7.19): the ``--config-path`` YAML that carries the
strategy, the broker profile and the run of a real account.  :func:`live_inputs` is the one place
that reads it, so neither face can apply a different half of the file, and the priority is fixed
there once - what the caller typed wins over what the file says, and the project defaults fill what
neither of them names.  The optional ``session`` block (Э17.z) rides along: it is the one place a
*killzone window* leaves the code, and it is read by :func:`session_from_config` rather than by
:func:`flatten_block`, because a window is a pair of hours and not a threshold.

The third is the clock of a raw tape (Э17.B, §7.26).  A file holds its stamps in one clock, and the
Э17 audit found that both bases - ``./data`` and the MT5 export - hold the clock of the broker
server rather than UTC.  ``--source-clock`` and the ``source_clock`` key of a live config declare
it (:func:`resolve_source_clock`), :func:`_load_tape` carries the declaration into
:func:`~smc_zero.data_loader.load_ohlcv`, and the loader moves the stamps onto UTC there.  Nothing
above the loader sees a shifted stamp, and a caller that declares nothing keeps the UTC-naive
behaviour of every run before Э17.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from smc_zero.config import (
    ALFAFOREX_SPECS,
    DEFAULT_HIERARCHY,
    DEFAULT_SOURCE_CLOCK,
    HIERARCHY_PRESETS,
    SOURCE_CLOCKS,
    BacktestConfig,
    BrokerSpec,
    InstrumentSpec,
    RiskConfig,
    RunConfig,
    SessionConfig,
    SourceClock,
    SourceClockConfig,
    StrategyConfig,
    TimeframeConfig,
    preset_of,
)
from smc_zero.data_loader import (
    AUTO_FORMAT,
    TIMESTAMP_COLUMN,
    bars_per_day,
    load_csv,
    load_ohlcv,
)
from smc_zero.optimizer.ranges import ParamValue, apply_params

#: Where the loader's CSVs live, relative to the working directory (rule 6: ``./`` paths).
DATA_DIR = Path("./data")
#: The default report root - the folder of the Э5' export, beside the data and never inside it.
REPORTS_DIR = Path("./reports")
#: The two bases a tape may live in (Э11'.1): the project's own ``./data`` and a raw MT5 export.
PROJECT_SOURCE = "project"
MT5_SOURCE = "mt5"
DATA_SOURCES: tuple[str, ...] = (PROJECT_SOURCE, MT5_SOURCE)
#: Environment variable that overrides the MT5 export folder.
SMC_DATA_DIR_ENV = "SMC_DATA_DIR"
#: Default base of a MetaTrader 5 "Bars" export (D1 is a bare date, intraday bars full stamps).
DEFAULT_MT5_DATA_DIR = Path.home() / "_data" / "mt5"
#: The window of the shipped four-year tape (SPEC_SMC.md §7.10 п.62).
DEFAULT_START = "2022-08-15"
DEFAULT_END = "2026-09-22"
#: The entry timeframe of the default hierarchy (D1 bias -> H1 structure -> M15 entry); the second
#: hierarchy (H4 bias -> M15 structure -> M5 entry) names its own through a live config (§7.20).
DEFAULT_TIMEFRAME = "M15"
#: The symbol and the timeframe a run without arguments assumes.
DEFAULT_SYMBOL = "EURUSD"
#: The four blocks a live config has to carry: the pair, the strategy, the account and the run
#: (§7.19).  A file without one of them is refused, because the missing half would have to be
#: guessed (rule 5).
CONFIG_KEYS: tuple[str, ...] = ("symbol", "strategy", "broker", "backtest")
#: The two knobs the ``backtest`` block of a live config may set: the capital of the run and the
#: lot it trades.  Every other number of a run stays a project default (§7.19).
BACKTEST_KEYS: tuple[str, ...] = ("initial_capital", "lot")
#: The optional ``run`` block of a live config (§7.20, В3-X): the window of the run, the warm-up the
#: readable HTF tapes are loaded with and the two engine counters of §7.9.  A file without the block
#: keeps the project window of Э8', and every key it names is applied over the defaults.
RUN_BLOCK = "run"
RUN_KEYS: tuple[str, ...] = (
    "start",
    "end",
    "warmup_days",
    "limit_valid_bars",
    "max_bars_per_trade",
)
#: The optional key of a live config that names the entry hierarchy (SPEC_SMC.md §7.20).  It is not
#: in :data:`CONFIG_KEYS`: a file without it keeps the D1 -> H1 -> M15 preset of Э8'.
HIERARCHY_KEY = "hierarchy"
#: The optional key of a live config that names the clock of its *raw* tape (Э17.B, §7.26).  It is
#: not in :data:`CONFIG_KEYS` either: a file without it keeps the UTC-naive convention the pipeline
#: has always assumed, and a file that names :data:`~smc_zero.config.SOURCE_CLOCK_BROKER_SERVER`
#: declares what the Э17 audit found in both bases - the live config of Э10'.2 does.
SOURCE_CLOCK_KEY = "source_clock"
#: The optional ``session`` block of a live config (Э17.z): the killzone table the run trades.  It is
#: not in :data:`CONFIG_KEYS` either - a file without it keeps the shipped table of
#: :class:`~smc_zero.config.SessionConfig`.  A window is a *pair* of hours and not a threshold, so the
#: block is read by :func:`session_from_config` and not by :func:`flatten_block`, which refuses a list
#: leaf on purpose.
SESSION_BLOCK = "session"
#: The pair-valued fields of that block: the MSK half-open windows first, then the broker's C6 times.
SESSION_WINDOW_KEYS: tuple[str, ...] = (
    "prelondon_msk",
    "london_summer_msk",
    "london_winter_msk",
    "ny_summer_msk",
    "ny_winter_msk",
    "session_open_msk",
    "session_close_msk",
)
#: The switch-valued fields of that block: the gate, the fixed pre-London window and the research-only
#: switch of Э17.z, which a live config must never set.
SESSION_FLAG_KEYS: tuple[str, ...] = ("use_kz", "prelondon", "research_ignore_sessions")
#: Everything the ``session`` block may name.
SESSION_KEYS: tuple[str, ...] = SESSION_WINDOW_KEYS + SESSION_FLAG_KEYS


def mt5_data_dir() -> Path:
    """Return the MT5 base folder: ``SMC_DATA_DIR`` when set, else ``~/_data/mt5``.

    Read on every call and never cached, so a test or a script can retarget the source through
    ``os.environ`` without a re-import (Э11'.1).
    """
    override = os.environ.get(SMC_DATA_DIR_ENV)
    return Path(override) if override else DEFAULT_MT5_DATA_DIR


def data_base(source: str) -> Path:
    """Return the base folder of ``source``: ``./data`` or the MT5 export folder."""
    if source == PROJECT_SOURCE:
        return DATA_DIR
    if source == MT5_SOURCE:
        return mt5_data_dir()
    supported = ", ".join(DATA_SOURCES)
    raise ValueError(f"unknown data source {source!r}; expected one of {supported}")


def tape_path(symbol: str, timeframe: str, *, source: str = PROJECT_SOURCE) -> Path:
    """Return the tape of one symbol: ``<base>/<SYMBOL>_<TIMEFRAME>.csv`` (Э11'.1)."""
    base = data_base(source)
    return base / f"{symbol.upper()}_{timeframe.upper()}.csv"


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


def _load_tape(
    path: Path,
    *,
    source: str = PROJECT_SOURCE,
    source_clock: SourceClock = DEFAULT_SOURCE_CLOCK,
) -> pd.DataFrame:
    """Load one tape through the reader of ``source`` with the clock its raw stamps carry.

    ``source`` picks the reader (Э11'.1): :func:`~smc_zero.data_loader.load_csv` for the project
    layout of ``./data``, :func:`~smc_zero.data_loader.load_ohlcv` for a raw MetaTrader 5 export.
    ``source_clock`` is the Э17.B declaration of that raw file: the default reads the stamps as
    UTC-naive open times and ``"broker_server"`` has the loader move them onto UTC
    (:func:`~smc_zero.data_loader.normalise_source_clock`).

    The default clock is passed on *without* the keyword, so a run that declares nothing keeps the
    call of every runner before Э17 bit for bit - a stubbed ``load_csv`` of the test suite keeps
    its own signature.  Only a non-default clock reaches the loader as ``source_clock=``.
    """
    if source == PROJECT_SOURCE:
        reader: Any = load_csv
    else:
        reader = partial(load_ohlcv, format=AUTO_FORMAT)
    if source_clock == DEFAULT_SOURCE_CLOCK:
        return reader(path)
    return reader(path, source_clock=source_clock)


def load_windowed_tape(
    symbol: str,
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    source: str = PROJECT_SOURCE,
    source_clock: SourceClock = DEFAULT_SOURCE_CLOCK,
) -> pd.DataFrame:
    """Load ``<base>/<SYMBOL>_<TF>.csv`` and cut the requested window out of it.

    ``source`` picks the base of the tape (Э11'.1): ``"project"`` reads ``./data`` through
    :func:`~smc_zero.data_loader.load_csv`, ``"mt5"`` reads a raw MetaTrader 5 export through
    :func:`~smc_zero.data_loader.load_ohlcv` (the layout is detected from the columns).  Raising is
    the contract: :class:`FileNotFoundError` when the tape is absent and :class:`ValueError` when no
    bar of it lies in the window, so a runner reports one line on stderr and returns a code instead
    of simulating an empty tape (which would report a flat curve as if it were a result).

    ``source_clock`` travels into :func:`_load_tape` and is applied *before* the window is cut: a
    window is a stretch of real time (``[start, end + 1 day)`` of UTC), and cutting it out of
    stamps that are still in server time would drop the last two or three hours of every day.
    """
    path = tape_path(symbol, timeframe, source=source)
    if not path.is_file():
        hint = (
            f"run from the repository root, or fetch the CSV into {path.parent}"
            if source == PROJECT_SOURCE
            else f"point {SMC_DATA_DIR_ENV} at the export folder (now {path.parent})"
        )
        raise FileNotFoundError(f"no tape at {path}: {hint}")
    tape = _load_tape(path, source=source, source_clock=source_clock)
    window = slice_window(tape, start, end)
    if window.empty:
        raise ValueError(f"no bar of {path} is opened in {start:%Y-%m-%d} .. {end:%Y-%m-%d}")
    return window


def working_frame(
    symbol: str,
    timeframes: TimeframeConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    source: str = PROJECT_SOURCE,
    source_clock: SourceClock = DEFAULT_SOURCE_CLOCK,
) -> tuple[str | None, pd.DataFrame | None]:
    """Return the working timeframe of a run and its tape, or ``(None, None)``.

    The second hierarchy separates the frame the structure is *read on* from the frame that is
    traded (SPEC_SMC.md §7.20): its preset names that frame (``H4_M15_M5`` reads the structure on
    M15), the tape is loaded over the same window, and the cache joins the two grids so the entry
    chain reads a working bar only after it closed.  The v1 hierarchy has ``structure=None`` - the
    entry frame owns the structure - and this answers ``(None, None)``, which keeps every existing
    command line bit for bit.

    Raises like :func:`load_windowed_tape`: a working tape that is absent or holds no bar of the
    window stops the runner instead of running the entry alone under a structure that is not there.
    ``source_clock`` is the Э17.B clock of that raw file and travels with the call.
    """
    preset = preset_of(timeframes)
    if preset.structure is None:
        return None, None
    return preset.structure, load_windowed_tape(
        symbol,
        preset.structure,
        start,
        end,
        source=source,
        source_clock=source_clock,
    )


def markup_frames(
    symbol: str,
    timeframes: TimeframeConfig,
    window: RunWindow,
    *,
    source: str = PROJECT_SOURCE,
    source_clock: SourceClock = DEFAULT_SOURCE_CLOCK,
) -> dict[str, pd.DataFrame]:
    """Return the HTF frames of a separated bias hierarchy, loaded from their own files (В2-B).

    The second hierarchy (§7.20) reads H4 and D1 from ``<SYMBOL>_<TF>.csv`` over
    ``[start - warmup_days, end]``: resampling them out of a five minute tape would hand the bias
    bars nobody exported, and the warm-up is what lets the first bars of the window carry a defined
    trend instead of ``NaN`` until the first swing.  The v1 hierarchy keeps resampling its HTF
    frames inside :func:`~smc_zero.optimizer.marks.build_tape_marks` and this answers an empty
    mapping, so its behaviour stays bit for bit.

    В3-У: a window whose warm-up reaches past the first bar of a tape is a ``ValueError`` naming the
    earliest start that tape supports - a run may not be handed a shorter warm-up than it asked for.
    A tape that is not there raises the :class:`FileNotFoundError` of :func:`tape_path`'s loader,
    which names the file and the folder to fetch it into.

    ``source_clock`` is the Э17.B clock of those raw files: the loader moves the stamps onto UTC
    before the warm-up is measured, so a tape in server time does not look three hours older than
    it is and the window of the run is counted in real time.
    """
    preset = preset_of(timeframes)
    if preset.structure is None:
        return {}
    frames: dict[str, pd.DataFrame] = {}
    for timeframe in preset.bias:
        path = tape_path(symbol, timeframe, source=source)
        if not path.is_file():
            hint = (
                f"run from the repository root, or fetch the CSV into {path.parent}"
                if source == PROJECT_SOURCE
                else f"point {SMC_DATA_DIR_ENV} at the export folder (now {path.parent})"
            )
            raise FileNotFoundError(f"no tape at {path}: {hint}")
        tape = _load_tape(path, source=source, source_clock=source_clock)
        first = tape[TIMESTAMP_COLUMN].min()
        earliest = first + pd.Timedelta(days=window.warmup_days)
        if window.start < earliest:
            raise ValueError(
                f"нужен --start >= {earliest:%Y-%m-%d} (лента {timeframe} начинается "
                f"{first:%Y-%m-%d}, прогрев {window.warmup_days} дней)"
            )
        frames[timeframe] = slice_window(tape, window.loaded_from, window.end)
    return frames


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


def load_config(path: str | Path) -> dict[str, Any]:
    """Return the live YAML config of ``path``, refusing a file without one of the four blocks.

    The reader is ``yaml.safe_load``, because a config is data and never code.  A missing or
    unreadable file raises :class:`FileNotFoundError` (or another :class:`OSError`) as it is, a text
    that does not parse raises :class:`ValueError` naming the path, and so does a document that is
    not a mapping or that lacks one of :data:`CONFIG_KEYS` - a run that would have to guess a block
    is not a run (rule 5).
    """
    try:
        with open(path, encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
    except yaml.YAMLError as error:
        raise ValueError(f"{path}: not a readable YAML config ({error})") from error
    if not isinstance(cfg, Mapping):
        raise ValueError(f"{path}: a live config is a mapping, got {type(cfg).__name__}")
    for key in CONFIG_KEYS:
        if key not in cfg:
            raise ValueError(f"в конфиге нет обязательного ключа: {key}")
    return dict(cfg)


def block(cfg: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return one block of a live config as a mapping; a block of another shape is refused."""
    value = cfg[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"{key}: a live config block is a mapping, got {type(value).__name__}")
    return value


def flatten_block(block_: Mapping[str, Any], prefix: str = "") -> dict[str, ParamValue]:
    """Return a nested YAML block as the dotted ``{path: value}`` mapping ``apply_params`` reads.

    ``{"take_profit": {"min_tp_rr": 1.2}}`` becomes ``{"take_profit.min_tp_rr": 1.2}`` - the naming
    of the search space of Э7' (:data:`~smc_zero.optimizer.ranges.PARAM_RANGES`), so the parameter
    set of a winning trial is one dict away from the config a runner builds.  A leaf that is neither
    a number nor a word is refused here: a list or a mapping that stopped being one would be written
    into a field that expects a threshold, and a silent wrong number is the one thing a config must
    not do (rules 1 and 5).
    """
    flat: dict[str, ParamValue] = {}
    for name, value in block_.items():
        path = f"{prefix}{name}"
        if isinstance(value, Mapping):
            flat.update(flatten_block(value, f"{path}."))
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError(
                f"{path}: a threshold of the strategy is a number or a word, got {value!r}"
            )
        flat[path] = value
    return flat


def strategy_from_config(cfg: Mapping[str, Any]) -> StrategyConfig:
    """Return the strategy of the ``strategy`` block: the project defaults with those knobs set."""
    params = flatten_block(block(cfg, "strategy"))
    try:
        return apply_params(StrategyConfig(), params)
    except KeyError as error:
        raise ValueError(f"strategy: {error}") from error


def _two_hours(name: str, value: Any) -> tuple[int, int]:
    """Return the ``(hour, minute)`` pair of one window of the ``session`` block (rule 5).

    A YAML list of two whole numbers is the only shape accepted.  A bare string would slip into a
    field that unpacks two values (``"9-12"`` unpacks into four), and a silently wrong window is
    the one thing a table must not do.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{SESSION_BLOCK}.{name}: a window is a pair of hours, got {value!r}")
    parts: list[int] = []
    for part in value:
        if isinstance(part, bool) or not isinstance(part, int):
            raise ValueError(f"{SESSION_BLOCK}.{name}: a window holds whole hours, got {value!r}")
        parts.append(int(part))
    return parts[0], parts[1]


def session_from_config(cfg: Mapping[str, Any] | None) -> SessionConfig:
    """Return the killzone table of the optional ``session`` block, else the shipped one (Э17.z).

    The block is the one place where a *window* leaves the code: the killzone gate and the Э3'
    session level maps read one table (SPEC_SMC.md, C1), so moving it is a strategy decision and
    not a run knob.  A window is a pair of hours and not a threshold, which is why the block has a
    reader of its own instead of going through :func:`flatten_block` - exactly how
    :func:`run_from_config` reads the ``run`` block.

    ``None`` and a file without the block both answer the shipped
    :class:`~smc_zero.config.SessionConfig`, so a config written before Э17.z keeps behaving bit
    for bit as it did.  An unknown key is refused with the list of the settable ones (rule 5), and
    the values themselves are validated by the dataclass (half-open windows, whole hours).
    """
    if cfg is None or SESSION_BLOCK not in cfg:
        return SessionConfig()
    values = dict(block(cfg, SESSION_BLOCK))
    unknown = sorted(set(values) - set(SESSION_KEYS))
    if unknown:
        raise ValueError(
            f"{SESSION_BLOCK}: no such field(s) {', '.join(unknown)}: the table sets "
            f"{', '.join(SESSION_KEYS)}"
        )
    windows = {
        name: _two_hours(name, value)
        for name, value in values.items()
        if name in SESSION_WINDOW_KEYS
    }
    flags: dict[str, bool] = {}
    for name in SESSION_FLAG_KEYS:
        if name not in values:
            continue
        flag = values[name]
        if not isinstance(flag, bool):
            raise ValueError(f"{SESSION_BLOCK}.{name}: a switch is true or false, got {flag!r}")
        flags[name] = flag
    try:
        return SessionConfig(**windows, **flags)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{SESSION_BLOCK}: {error}") from error


def broker_from_config(cfg: Mapping[str, Any]) -> BrokerSpec:
    """Return the :class:`BrokerSpec` of the ``broker`` block: every field by its name (C6, Э10').

    The fields are the profile's own, so a config cannot invent a cost and cannot use one of the
    deprecated Э5' names either: ``commission`` is refused, ``commission_per_lot_usd`` is the field
    that exists and the money of the run is its number.  A symbol-level field of the block has to
    agree with the price-list row of the traded pair anyway - the engine refuses the pair of them
    when they disagree about the symbol (:func:`~smc_zero.backtester.engine.run_backtest`).
    """
    values = dict(block(cfg, "broker"))
    known = sorted(field.name for field in fields(BrokerSpec))
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise ValueError(
            f"broker: no such field(s) {', '.join(unknown)}: the profile has {', '.join(known)}"
        )
    try:
        return BrokerSpec(**{name: float(value) for name, value in values.items()})
    except (TypeError, ValueError) as error:
        raise ValueError(f"broker: {error}") from error


def backtest_from_config(cfg: Mapping[str, Any], run: RunConfig | None = None) -> BacktestConfig:
    """Return the run of a live config: the ``backtest`` block over the account of ``broker``.

    ``initial_capital`` and ``lot`` are the two knobs a live config sets, and the lot lands in
    :class:`~smc_zero.config.RiskConfig` - that is the profile the engine reads its lot from - while
    the capital is the backtester's own start (the two answer different questions, §7.9).  The
    broker block is written into that same risk profile, which is the single home of the money
    (Э10'), so the engine, the report and the optimizer price the account of the file.

    ``run`` is the optional ``run`` block (§7.20, В3-X): its two engine counters
    (``limit_valid_bars``, ``max_bars_per_trade``) are applied over the defaults, and every other
    key of that block is the window or the warm-up and is read by :func:`resolve_window`.
    """
    values = dict(block(cfg, "backtest"))
    unknown = sorted(set(values) - set(BACKTEST_KEYS))
    if unknown:
        raise ValueError(
            f"backtest: no such field(s) {', '.join(unknown)}: a live config sets "
            f"{', '.join(BACKTEST_KEYS)}"
        )
    settings = run_from_config(cfg) if run is None else run
    defaults = BacktestConfig()
    lot = float(values.get("lot", defaults.risk.lot))
    risk = RiskConfig(broker=broker_from_config(cfg), lot=lot)
    counters = {
        name: value
        for name in ("limit_valid_bars", "max_bars_per_trade")
        if (value := getattr(settings, name)) is not None
    }
    return replace(
        defaults,
        risk=risk,
        initial_capital=float(values.get("initial_capital", defaults.initial_capital)),
        **counters,
    )


def resolve_selector(
    argument: str | None, cfg: Mapping[str, Any] | None, key: str, fallback: str
) -> str:
    """Return the effective symbol / timeframe: the argument, else the config's, else the default.

    One order, said once: what the caller typed wins over what the file says, and a run that names
    neither keeps the project default of Э8'.  The two are never merged - a config that names
    another pair than the argument is simply outranked by it.
    """
    if argument:
        return argument.upper()
    if cfg is not None and cfg.get(key):
        return str(cfg[key]).upper()
    return fallback


def resolve_hierarchy(argument: str | None, cfg: Mapping[str, Any] | None) -> str:
    """Return the effective hierarchy preset name: the argument, the config's, else the default.

    The same order as :func:`resolve_selector`: what the caller typed wins over the file, and a run
    that names neither keeps :data:`~smc_zero.config.DEFAULT_HIERARCHY` (D1 -> H1 -> M15).  A name
    outside :data:`~smc_zero.config.HIERARCHY_PRESETS` is refused with the list of the known ones,
    because a typo would otherwise silently select the default preset.
    """
    named = argument or (None if cfg is None else cfg.get(HIERARCHY_KEY)) or DEFAULT_HIERARCHY
    name = str(named).upper()
    if name not in HIERARCHY_PRESETS:
        known = ", ".join(sorted(HIERARCHY_PRESETS))
        raise ValueError(f"hierarchy: no such preset {name!r}: expected one of {known}")
    return name


def resolve_source_clock(
    args: argparse.Namespace,
    cfg: Mapping[str, Any] | None = None,
) -> SourceClock:
    """Return the clock of the *raw* tape a run reads (Э17.B): typed, file, else ``"utc"``.

    The order is the order of the layer (:func:`resolve_selector`): ``--source-clock`` as the caller
    typed it, else the ``source_clock`` key of a live config, else
    :data:`~smc_zero.config.DEFAULT_SOURCE_CLOCK` - the UTC-naive convention every run before Э17
    assumed, which keeps a command line without a declaration working unchanged.

    ``args.source_clock`` defaults to ``None`` and not to ``"utc"`` on purpose: a default outranks
    the file without the caller ever typing it, and the live config of the winner declares
    ``"broker_server"`` (the Э17 audit found both bases in server time).

    ``cfg`` is the already read config of the caller (``None`` to read the one ``args`` names).  A
    value outside :data:`~smc_zero.config.SOURCE_CLOCKS` is refused with the list of the known ones,
    so a typo can never fall back to UTC and hide a shift of every bar (rule 5).
    """
    if cfg is None and getattr(args, "config_path", None):
        cfg = load_config(args.config_path)
    clock = getattr(args, "source_clock", None) or (
        None if cfg is None else cfg.get(SOURCE_CLOCK_KEY)
    )
    if clock is None:
        return SourceClockConfig().clock
    try:
        return SourceClockConfig(clock=str(clock)).clock
    except ValueError as error:
        known = ", ".join(SOURCE_CLOCKS)
        raise ValueError(f"{SOURCE_CLOCK_KEY}: {error}; known clocks are {known}") from error


def live_inputs(args: argparse.Namespace) -> tuple[str, str, StrategyConfig, BacktestConfig]:
    """Return the symbol, the timeframe and the two configs of a run: the file first, the arguments.

    ``args`` carries ``--config-path`` (optional), ``--symbol``, ``--timeframe`` and
    ``--hierarchy`` (``None`` when the caller typed none of them).  Without a config the result is
    the project defaults of Э8'; with one, the three blocks of §7.19 fill the two dataclasses and a
    typed argument still wins.  Both faces call this one function, so a rule about the file cannot
    hold in one of them only.

    The hierarchy and the entry timeframe are one decision, not two (§7.20).  The preset is
    resolved first (:func:`resolve_hierarchy`) and ``timeframes`` of the returned
    :class:`~smc_zero.config.BacktestConfig` is exactly it - the engine dates the tape with
    ``timeframes.ltf``, so a mismatch would silently re-price every bar.  The entry timeframe is
    then taken from the caller, the file's key or the preset itself, and a value that disagrees
    with the preset's ``ltf`` is refused instead of being run: an M5 tape under the M15 preset, or
    an M15 file under the M5 one, is a config error and not a silent reinterpretation.

    The preset also owns the *bias* frames (R1 of §7.20): the v1 set is H1 + H4 + D1 and the second
    hierarchy asks H4 + D1.  ``BiasConfig.timeframes`` of the returned strategy is therefore the
    preset's set - the other bias knobs (``agreement``, the swing settings) still come from the
    file, and a file cannot name a different set, because a YAML list is not a threshold
    (:func:`flatten_block`) and the hierarchy, not the file, decides what a direction is asked of.

    The effective entry timeframe also scales the run: ``sharpe_bars_per_day`` is set from
    :func:`~smc_zero.data_loader.bars_per_day`, so an M5 run of the second hierarchy reads its 288
    bars a day instead of the 96 of M15.  For the default M15 hierarchy the values are the
    dataclass defaults and the returned config is unchanged.

    The optional ``session`` block (Э17.z) is read by :func:`session_from_config` and lands in
    *both* configs: the entry chain and the Э3' session level maps read the strategy's copy, while
    the engine's broker-hours mask reads the run's one, and two different tables in one run would
    let a setup arm where placement is refused.
    """
    cfg = load_config(args.config_path) if args.config_path else None
    preset_name = resolve_hierarchy(getattr(args, "hierarchy", None), cfg)
    preset = HIERARCHY_PRESETS[preset_name]
    timeframes = preset.timeframes
    symbol = resolve_selector(args.symbol, cfg, "symbol", DEFAULT_SYMBOL)
    timeframe = resolve_selector(args.timeframe, cfg, "timeframe", timeframes.ltf)
    if timeframe != timeframes.ltf:
        named = getattr(args, "hierarchy", None) or (
            None if cfg is None else cfg.get(HIERARCHY_KEY)
        )
        if named:
            raise ValueError(
                f"hierarchy {preset_name!r} requires timeframe {timeframes.ltf!r}, "
                f"got {timeframe!r}"
            )
        raise ValueError(
            f"timeframe {timeframe!r} needs an explicit hierarchy: the default is "
            f"{DEFAULT_HIERARCHY!r} ({timeframes.ltf} entries), so name e.g. "
            f"'{HIERARCHY_KEY}: H4_M15_M5' (or pass --hierarchy) for an M5 run"
        )
    scale = bars_per_day(timeframe)
    session = session_from_config(cfg)
    strategy = StrategyConfig() if cfg is None else strategy_from_config(cfg)
    strategy = replace(
        strategy, bias=replace(strategy.bias, timeframes=preset.bias), session=session
    )
    if cfg is None:
        return (
            symbol,
            timeframe,
            strategy,
            BacktestConfig(timeframes=timeframes, sharpe_bars_per_day=scale),
        )
    backtest = replace(
        backtest_from_config(cfg),
        timeframes=timeframes,
        sharpe_bars_per_day=scale,
        session=session,
    )
    return symbol, timeframe, strategy, backtest


@dataclass(frozen=True, slots=True)
class RunWindow:
    """The window of a run: the two days and the warm-up of the readable tapes (§7.20).

    ``loaded_from`` is the first day the readable HTF tapes are loaded from - ``start`` minus the
    warm-up - and it is what the window validation of §7.20 (В3-У) compares with the first bar of a
    tape: a run whose warm-up reaches past the data is refused instead of being handed a shorter
    history than it asked for.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    warmup_days: int = 0

    @property
    def loaded_from(self) -> pd.Timestamp:
        """The first day the readable tapes are loaded from: ``start`` minus the warm-up."""
        return self.start - pd.Timedelta(days=self.warmup_days)


def run_from_config(cfg: Mapping[str, Any] | None) -> RunConfig:
    """Return the ``run`` block of a live config, or the neutral :class:`RunConfig` without one.

    The block is optional - the Э8' files do not carry it - and every key it names is checked against
    :data:`RUN_KEYS`, so a typo is refused instead of being ignored (rule 5).  The values themselves
    are validated by the dataclass (a negative warm-up, a zero ``max_bars_per_trade``).
    """
    if cfg is None or RUN_BLOCK not in cfg:
        return RunConfig()
    values = dict(block(cfg, RUN_BLOCK))
    unknown = sorted(set(values) - set(RUN_KEYS))
    if unknown:
        raise ValueError(
            f"{RUN_BLOCK}: no such field(s) {', '.join(unknown)}: a run sets {', '.join(RUN_KEYS)}"
        )
    try:
        return RunConfig(**values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{RUN_BLOCK}: {error}") from error


def resolve_window(args: argparse.Namespace, run: RunConfig | None = None) -> RunWindow:
    """Return the window of a run: the argument, else the file's ``run`` block, else the default.

    The order is the one of §7.19: what the caller typed wins over the file, and a run that names
    neither keeps the shipped window of Э8'.  ``--start`` / ``--end`` default to ``None`` on purpose
    (never to the project days), because such a default would outrank the file without the caller
    ever typing it - the trap the pair of §7.19 has already taught.

    ``run`` is the already read block; leaving it ``None`` reads ``--config-path`` here, so a caller
    that has the block at hand pays for one read of the file and not two.
    """
    if run is None:
        cfg = load_config(args.config_path) if args.config_path else None
        run = run_from_config(cfg)
    start = getattr(args, "start", None)
    end = getattr(args, "end", None)
    if start is None:
        start = read_day(run.start) if run.start else read_day(DEFAULT_START)
    if end is None:
        end = read_day(run.end) if run.end else read_day(DEFAULT_END)
    return RunWindow(start=start, end=end, warmup_days=run.warmup_days)


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    """Add ``--config-path``: the live YAML both faces take their numbers from (§7.19)."""
    parser.add_argument(
        "--config-path",
        default=None,
        help="live YAML config (symbol / strategy / broker / backtest); a typed argument wins",
    )


def add_window_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments both runners share: the tape selector and the report root.

    ``--symbol`` and ``--timeframe`` default to ``None`` on purpose and not to the project pair:
    with a ``--config-path`` the file names the traded pair, and a default would outrank it without
    the caller ever typing it (Э8' default, §7.19 order).  ``--start`` and ``--end`` default to
    ``None`` for the same reason: the ``run`` block of the file may name the window, and the project
    window of Э8' fills what neither names (:func:`resolve_window`).
    """
    parser.add_argument(
        "--symbol",
        default=None,
        help="symbol of ./data/<SYMBOL>_<TIMEFRAME>.csv (default: the config's, else EURUSD)",
    )
    parser.add_argument(
        "--timeframe",
        default=None,
        help="entry timeframe of the tape, M15 for v1 (default: the config's, else M15)",
    )
    parser.add_argument(
        "--hierarchy",
        default=None,
        help=(
            "entry hierarchy preset, e.g. H4_M15_M5 (default: the config's, else D1_H1_M15); "
            "it fixes the entry timeframe, so a --timeframe that disagrees is refused"
        ),
    )
    parser.add_argument(
        "--start",
        type=read_day,
        default=None,
        help="first day of the window, YYYY-MM-DD (default: the run block, else 2022-08-15)",
    )
    parser.add_argument(
        "--end",
        type=read_day,
        default=None,
        help="last day of the window, YYYY-MM-DD, included (default: the config's run block)",
    )
    parser.add_argument("--report-dir", default=str(REPORTS_DIR), help="report root, ./reports")


def add_data_source_argument(parser: argparse.ArgumentParser) -> None:
    """Add ``--data-source`` and ``--source-clock``: the base a run reads and its clock (Э17.B).

    ``project`` is the ``./data`` folder the layer shipped with; ``mt5`` is a raw MetaTrader 5
    export under ``SMC_DATA_DIR`` (``~/_data/mt5`` by default).  The default keeps every existing
    command line working unchanged.

    The clock belongs beside the base, because it is a property of the very files of that base and
    not of the pipeline: ``--source-clock`` declares what their raw stamps carry.  Its default is
    ``None`` and not ``"utc"`` on purpose (:func:`resolve_source_clock`): the ``source_clock`` key
    of a live config has to be able to outrank a default nobody typed, and ``None`` still resolves
    to the UTC-naive convention of every run before Э17.
    """
    parser.add_argument(
        "--data-source",
        choices=DATA_SOURCES,
        default=PROJECT_SOURCE,
        help="tape base: 'project' = ./data (default), 'mt5' = the export under SMC_DATA_DIR",
    )
    parser.add_argument(
        "--source-clock",
        choices=SOURCE_CLOCKS,
        default=None,
        help=(
            "clock of the raw tape file: 'utc' (default) or 'broker_server' "
            "(GMT+3/GMT+2 by the US DST calendar; what the MT5 export and ./data carry); "
            "a live config may name it through 'source_clock'"
        ),
    )
