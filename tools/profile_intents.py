"""Stage profile of the M15 entry chain: where ``build_intents`` spends its time (Э9'.1).

The tool answers one question - *which part of the chain is quadratic* - and it is a
measurement device, not a strategy: it owns no threshold, no indicator and no metric, and it
never reimplements a rule.  It loads the tape exactly like the Э8' runners do
(:func:`scripts._common.load_windowed_tape` plus the markup cache of
:func:`smc_zero.optimizer.build_tape_marks`), calls the real
:func:`smc_zero.strategy.intents.build_intents` once and reports where the seconds went.

How the numbers are obtained, and the limits of the method:

* every stage of the chain is *called through the module* (``sweep_index(...)`` is a global
  lookup inside the walk), so wrapping the name in :mod:`smc_zero.strategy.intents` counts and
  times the real calls without touching the source - the ``calls`` column is a fact and the
  ``мкс/вызов`` column is the number the vectorisation of Э9' has to beat;
* the chain has two generations of stages and the table reports the ones the tree actually has:
  the Э4' walk of ``(level, bar)`` pairs (``sweep_index``, ``_owns_price``, ``_choch_bar`` ...)
  and the Э9' event scheme (``_attempt_bars``, ``_sweep_bars``, ``_first_in_window``,
  ``_fresh_limits``, ``_ledger_frame``).  A row whose attribute is gone from the module is listed
  *under* the table instead of failing the run, so the same tool profiles the "до" tree (a
  worktree of the commit before Э9', see №7.13 п.79) and the "после" tree;
* the sweep search is the step the algorithm pays per *(level instance, bar)* pair, and each of
  its calls converts the whole frame to ``numpy``; the printed ``элементов прочитано`` count is
  that multiplication, i.e. the work the old chain does over the whole tape per pair;
* the rolling prefilter ``flatnonzero(roll_max > level + buffer)`` is inline in the walk and no
  wrapper can see it; it is **reproduced** here on the same arrays (same expression, same order)
  and reported as its own row together with the number of candidate pairs it yields;
* the Э9' event scan is inline the same way: the vectorised walk still compares every bar of the
  tape with every level instance to collect the sweep events, so ``_pierce_scan`` re-runs that
  one expression (pierce plus the close back inside, exactly as the walk does it) and reports the
  pairs, events and attempt windows it yields - the rows of the vectorised profile that no
  wrapper can attribute;
* the instrumentation itself costs time.  A no-op wrapper is calibrated once and charged to
  every wrapped call on the last line of the table, so the ``остальное`` row is not an artefact
  of the measurement.

Run it from the repository root (rule 6: data paths are relative, and the tool needs both the
package and the ``scripts`` package on the path)::

    PYTHONPATH=src:. python tools/profile_intents.py --mode stages --spec 1y
    PYTHONPATH=src:. python tools/profile_intents.py --mode windows

The "до" row of the same table is taken with this file from the working tree while the *package*
comes from a worktree of the commit before Э9' (№7.13 п.79), and there the 4-year window would cost
hours, hence ``--windows``::

    PYTHONPATH=<worktree>/src:<worktree>:. python tools/profile_intents.py \\
        --mode windows --windows 5d,1m,3m

``--mode windows`` prints the wall time of the whole chain on the measurement windows of
SPEC_SMC.md §7.13 - the "до" row of that table - while ``--mode stages`` breaks one window down
by stage.  Nothing here is a threshold of the strategy: the window labels are the spec's
measurement windows, and the two numbers of ``_prefilter`` come out of the chain's own arrays.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts import _common
from smc_zero.config import StrategyConfig
from smc_zero.optimizer import build_tape_marks
from smc_zero.strategy import intents as chain_module

#: The measurement windows of SPEC_SMC.md §7.13, all starting at the first bar of the shipped
#: tape.  The labels are the ones the spec and the reports use: the five windows Э9' is accepted
#: on, plus ``2y`` for the linearity check (1y -> 2y -> 4y).
WINDOW_SPECS: dict[str, tuple[str, str]] = {
    "5d": ("2022-08-15", "2022-08-19"),
    "1m": ("2022-08-15", "2022-09-15"),
    "3m": ("2022-08-15", "2022-11-15"),
    "1y": ("2022-08-15", "2023-08-15"),
    "2y": ("2022-08-15", "2024-08-15"),
    "4y": ("2022-08-15", "2026-09-22"),
}
#: The window ``--mode stages`` breaks down when no ``--spec`` is given.
DEFAULT_SPEC = "1y"

#: ``(label, attribute of smc_zero.strategy.intents)`` - the wrap points, in call order.  The
#: labels name the chain's own steps, so a row of the table is a paragraph of the module
#: docstring and not a new concept.
STAGE_ATTRIBUTES: tuple[tuple[str, str], ...] = (
    ("фреймы входа и книга уровней", "_entry_bars"),
    ("книга уровней (retired_at)", "_level_book"),
    ("HTF bias: join по open_time", "_bias_direction"),
    ("сессии: alfa_trading_mask", "alfa_trading_mask"),
    ("сессии: killzone_mask", "killzone_mask"),
    ("структура (структурные пробои)", "structure_breaks"),
    ("импульс CHoCH (displacement)", "displacement_gate"),
    ("FVG-разметка", "fair_value_gaps"),
    ("группы цен (дедупликация)", "_price_groups"),
    ("окно попыток уровня (объединение)", "_attempt_bars"),
    ("окно sweep (таблица максимумов)", "_sweep_bars"),
    ("первое событие окна (CHoCH/FVG)", "_first_in_window"),
    ("свежесть уровня (лимиты)", "_fresh_limits"),
    ("поиск sweep", "sweep_index"),
    ("владение уровнем (дедупликация)", "_owns_price"),
    ("приоритет уровня (внутри владения)", "_priority_key"),
    ("поиск CHoCH", "_choch_bar"),
    ("поиск FVG", "_gap_bar"),
    ("цена входа (край FVG)", "entry_level"),
    ("геометрия SL (20-60 пип)", "_sl_geometry"),
    ("take-profit", "take_profit_for"),
    ("строка леджера (отказ)", "_rejection"),
    ("сборка леджера", "_ledger"),
    ("сборка леджера (колонки)", "_ledger_frame"),
)
#: The stage label of the reproduced inline prefilter (it is not a function, so no attribute).
PREFILTER_LABEL = "префильтр rolling max/min (воспроизведён)"
#: The stage label of the reproduced event scan of Э9' - the one pass over the tape that the
#: vectorised walk runs *per level instance* to find the bars that pierced the buffered price.
#: It is inline in the walk, so it is reproduced here the way the old prefilter is.
PIERCE_LABEL = "поиск событий уровня (прокол x инстанс)"
#: The stage label of everything the wrappers do not see (gate arithmetic, walk overhead).
REMAINDER_LABEL = "остальное: гейты в цикле, арифметика, цикл"
#: The calibrated wrapper cost, charged to the wrapped calls (see the module docstring).
OVERHEAD_LABEL = "накладные расходы инструментовки"
#: The stage label whose calls the ``элементов прочитано`` line multiplies by the frame length.
SWEEP_LABEL = "поиск sweep"



class _Counter:
    """Mutable per-stage meter: how many calls a stage took and how many seconds they cost."""

    def __init__(self) -> None:
        self.calls = 0
        self.seconds = 0.0


def _instrument(attribute: str) -> _Counter | None:
    """Wrap ``smc_zero.strategy.intents.<attribute>`` with a timer and return its meter.

    The original function object is read once and the module attribute is replaced, which is
    what the walk sees: the chain calls its helpers by global lookup, so every call of the
    stage is counted - including the ones made from inside the ``(instance, bar)`` loop.

    ``None`` means the chain has no such step any more: Э9' turned several of the Э4' helpers
    into range lookups over event arrays, so the "before" and the "after" profile legitimately
    have different rows and the tool reports the absent ones instead of failing.
    """
    original = getattr(chain_module, attribute, None)
    if original is None:
        return None
    counter = _Counter()

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            counter.calls += 1
            counter.seconds += time.perf_counter() - started

    setattr(chain_module, attribute, wrapper)
    return counter


def _still_walks() -> bool:
    """Return whether the chain still runs the Э4' walk of ``(level, bar)`` pairs.

    The discriminator is :func:`smc_zero.indicators.liquidity.sweep_index`: the walk of Э4' called
    it once per pair, the vectorised chain of Э9' does not import it.  The inline rolling prefilter
    of that walk only exists while it does, so the ``--mode stages`` table reproduces it - and
    reports its absence - accordingly.
    """
    return hasattr(chain_module, "sweep_index")


def _calibrate_overhead(repeats: int = 200_000) -> float:
    """Return the seconds one wrapped call adds to a stage (the price of the instrumentation)."""

    def target() -> None:
        """The no-op call the calibration wraps."""

    def wrapper() -> None:
        started = time.perf_counter()
        try:
            target()
        finally:
            time.perf_counter() - started

    started = time.perf_counter()
    for _ in range(repeats):
        wrapper()
    return (time.perf_counter() - started) / repeats


def _prefilter(
    tape: pd.DataFrame, levels: pd.DataFrame, config: StrategyConfig
) -> tuple[float, int, int]:
    """Reproduce the inline prefilter of the walk: ``(seconds, pairs, tradable_pairs)``.

    The expression is the one ``build_intents`` runs per level instance
    (``flatnonzero(roll_max > price + sweep_buffer)``) and it is reproduced because it lives
    inside the loop, where no wrapper can see it.  ``pairs`` is the number of
    ``(instance, bar)`` pairs the walk evaluates; ``tradable_pairs`` is the half of them that
    reaches :func:`~smc_zero.indicators.liquidity.sweep_index` (the rest fails a bar gate).
    """
    bars = chain_module._entry_bars(tape)
    book = chain_module._level_book(levels)
    sweep_buffer = config.sweep_buffer_pip * config.pip_size
    lookback = config.liquidity.sweep_lookback + 1
    roll_max = (
        pd.Series(bars["high"].to_numpy(dtype="float64"))
        .rolling(lookback, min_periods=1)
        .max()
        .to_numpy()
    )
    roll_min = (
        pd.Series(bars["low"].to_numpy(dtype="float64"))
        .rolling(lookback, min_periods=1)
        .min()
        .to_numpy()
    )
    stamps = bars[chain_module.TIMESTAMP_COLUMN]
    tradable = chain_module.alfa_trading_mask(stamps, config.session).to_numpy(dtype=bool)
    inside_killzone = chain_module.killzone_mask(stamps, config.session).to_numpy(dtype=bool)
    tradable = tradable & inside_killzone
    prices = book[chain_module.LEVEL_PRICE_COLUMN].to_numpy(dtype="float64")
    uppers = book[chain_module.LEVEL_IS_UPPER_COLUMN].to_numpy(dtype=bool)
    pairs = 0
    tradable_pairs = 0
    started = time.perf_counter()
    for position in range(len(book)):
        if bool(uppers[position]):
            hits = np.flatnonzero(roll_max > float(prices[position]) + sweep_buffer)
        else:
            hits = np.flatnonzero(roll_min < float(prices[position]) - sweep_buffer)
        pairs += int(hits.size)
        tradable_pairs += int(tradable[hits].sum())
    return time.perf_counter() - started, pairs, tradable_pairs


def _pierce_scan(
    tape: pd.DataFrame, levels: pd.DataFrame, config: StrategyConfig
) -> tuple[float, int, int, int]:
    """Reproduce the event scan of the Э9' walk: ``(seconds, pairs, events, attempt_windows)``.

    The vectorised chain still compares the whole tape with the price of every level instance, so
    this step is paid per instance although the ``(level, bar)`` loop is gone.  The expression is
    the one ``build_intents`` runs - pierce of the buffered price, the close back inside when
    ``sweep_mode`` asks for it, then the union of windows - and reproducing it here keeps that
    cost in the table instead of hiding it in ``остальное``.  ``pairs`` is the ``(instance, bar)``
    multiplicity of the scan, ``events`` the pierces that survive, ``attempt_windows`` the bars
    the scan hands to the range lookups.
    """
    bars = chain_module._entry_bars(tape)
    book = chain_module._level_book(levels)
    sweep_buffer = config.sweep_buffer_pip * config.pip_size
    lookback = config.liquidity.sweep_lookback
    close_back_inside = config.liquidity.sweep_mode == "wick_close_inside"
    high = bars["high"].to_numpy(dtype="float64")
    low = bars["low"].to_numpy(dtype="float64")
    close = bars["close"].to_numpy(dtype="float64")
    prices = book[chain_module.LEVEL_PRICE_COLUMN].to_numpy(dtype="float64")
    uppers = book[chain_module.LEVEL_IS_UPPER_COLUMN].to_numpy(dtype=bool)
    length = len(bars)
    events_total = 0
    windows_total = 0
    started = time.perf_counter()
    for position in range(len(book)):
        upper = bool(uppers[position])
        price = float(prices[position])
        threshold = price + sweep_buffer if upper else price - sweep_buffer
        events = np.flatnonzero(high > threshold if upper else low < threshold)
        if events.size and close_back_inside:
            inside = close[events] < threshold if upper else close[events] > threshold
            events = events[inside]
        events_total += int(events.size)
        windows_total += int(chain_module._attempt_bars(events, lookback, length).size)
    return time.perf_counter() - started, len(book) * length, events_total, windows_total


def stage_rows(
    counters: dict[str, _Counter],
    prefilter: tuple[float, int, int] | None,
    total: float,
    pierce: tuple[float, int, int, int] | None = None,
) -> list[dict[str, Any]]:
    """Return the table rows: one per stage, then the reproductions and the unmeasured remainder.

    The remainder is ``total - sum(measured)`` and never negative by construction: the wrapped
    calls and the reproduced steps are *parts* of the same ``build_intents`` call, so what is
    left is the gate arithmetic of the walk, the intent construction and the loop itself.
    ``prefilter`` belongs to the Э4' walk and ``pierce`` to the Э9' event scan; each is ``None``
    in the tree that does not have it, so a profile is never charged for the other generation.
    """
    rows = [
        {"stage": label, "calls": counters[label].calls, "seconds": counters[label].seconds}
        for label, _ in STAGE_ATTRIBUTES
        if label in counters
    ]
    if prefilter is not None:
        rows.append({"stage": PREFILTER_LABEL, "calls": prefilter[1], "seconds": prefilter[0]})
    if pierce is not None:
        rows.append({"stage": PIERCE_LABEL, "calls": pierce[1], "seconds": pierce[0]})
    charged = sum(row["seconds"] for row in rows)
    rows.append({"stage": REMAINDER_LABEL, "calls": 0, "seconds": max(0.0, total - charged)})
    rows.sort(key=lambda row: row["seconds"], reverse=True)
    return rows


def render(rows: list[dict[str, Any]], total: float, overhead_calls: int, overhead: float) -> str:
    """Render the stage table: the shares the Э9' plan is read off, summed over the whole call.

    The share is a fraction of the measured total, so the rows add up to 100 %; the calibrated
    instrumentation cost is the last line because it is an artefact of the measurement and not
    something the chain can be optimised to remove.
    """
    lines = [f"{'участок цепочки':<44}{'вызовов':>14}{'сек':>10}{'доля':>8}{'мкс/вызов':>12}"]
    lines.append("-" * 88)
    for row in rows:
        share = 100.0 * row["seconds"] / total if total > 0 else 0.0
        per_call = 1e6 * row["seconds"] / row["calls"] if row["calls"] else float("nan")
        lines.append(
            f"{row['stage']:<44}{row['calls']:>14,}{row['seconds']:>10.2f}"
            f"{share:>7.1f}%{per_call:>12.2f}"
        )
    lines.append("-" * 88)
    lines.append(f"{'итого build_intents':<44}{'':>14}{total:>10.2f}{100.0:>7.1f}%")
    lines.append(
        f"{OVERHEAD_LABEL:<44}{overhead_calls:>14,}{overhead:>10.2f}"
        f"{100.0 * overhead / total if total else 0.0:>7.1f}%"
    )
    return "\n".join(lines)


def measure_windows(
    specs: tuple[str, ...], symbol: str, timeframe: str, config: StrategyConfig
) -> list[dict[str, Any]]:
    """Time the chain on every window of ``specs`` and return one measurement per window."""
    measured: list[dict[str, Any]] = []
    for spec in specs:
        start, end = WINDOW_SPECS[spec]
        tape = _common.load_windowed_tape(
            symbol, timeframe, _common.read_day(start), _common.read_day(end)
        )
        marks = build_tape_marks(tape, config)
        started = time.perf_counter()
        out = chain_module.build_intents(tape, marks.bias_frame(), marks.levels, config)
        measured.append(
            {
                "spec": spec,
                "seconds": time.perf_counter() - started,
                "bars": len(tape),
                "levels": len(marks.levels),
                "intents": len(out.intents),
                "rejections": len(out.rejections),
            }
        )
    return measured


def render_windows(measured: list[dict[str, Any]]) -> str:
    """Render the wall-time table of :func:`measure_windows`, with the per-bar cost of each row.

    ``сек/бар`` is the column that makes the complexity visible: a linear chain keeps it
    constant across the windows while the old quadratic walk multiplies it with the window
    (SPEC_SMC.md §7.13), and ``рост`` compares each row with the row before it.
    """
    lines = [f"{'окно':<8}{'баров':>10}{'уровней':>10}{'сек':>10}{'сек/бар':>12}{'рост':>10}"]
    lines.append("-" * 60)
    previous = 0.0
    for row in measured:
        per_bar = row["seconds"] / row["bars"] if row["bars"] else float("nan")
        growth = f"{row['seconds'] / previous:.2f}x" if previous > 0 else "-"
        lines.append(
            f"{row['spec']:<8}{row['bars']:>10,}{row['levels']:>10,}{row['seconds']:>10.2f}"
            f"{per_bar:>12.2e}{growth:>10}"
        )
        previous = row["seconds"]
    return "\n".join(lines)


def measure_stages(
    spec: str, symbol: str, timeframe: str, config: StrategyConfig
) -> dict[str, Any]:
    """Run the instrumented chain on one window and return its measurement and table."""
    start, end = WINDOW_SPECS[spec]
    tape = _common.load_windowed_tape(
        symbol, timeframe, _common.read_day(start), _common.read_day(end)
    )
    marks = build_tape_marks(tape, config)
    walks = _still_walks()
    # The reproduced steps are measured before the instrumentation is armed, so their own calls of
    # the shared helpers do not land in the stage counters of the chain.
    prefilter = _prefilter(tape, marks.levels, config) if walks else None
    pierce = None if walks else _pierce_scan(tape, marks.levels, config)
    counters: dict[str, _Counter] = {}
    absent: list[str] = []
    for label, attribute in STAGE_ATTRIBUTES:
        meter = _instrument(attribute)
        if meter is None:
            absent.append(label)
        else:
            counters[label] = meter
    overhead = _calibrate_overhead()
    started = time.perf_counter()
    out = chain_module.build_intents(tape, marks.bias_frame(), marks.levels, config)
    total = time.perf_counter() - started
    rows = stage_rows(counters, prefilter, total, pierce)
    sweep = counters.get(SWEEP_LABEL)
    wrapped = sum(meter.calls for meter in counters.values())
    # The ledger is the chain's own answer, so the reason histogram is read off it instead of
    # being counted by a second wrapper that would distort the very timings being measured.
    counts = out.rejections["reason"].value_counts()
    reasons = {str(name): int(count) for name, count in counts.items()}
    pairs = f"пар (уровень x бар)={prefilter[1]:,} из них торгуемых={prefilter[2]:,}\n" if prefilter else ""
    if pierce is not None:
        pairs = (
            f"пар (инстанс x бар)={pierce[1]:,} событий уровня={pierce[2]:,} "
            f"окон попыток={pierce[3]:,}\n"
        )
    reading = (
        f"sweep_index: вызовов={sweep.calls:,}, прочитано элементов={sweep.calls * len(tape):,} "
        f"({sweep.calls * len(tape) / 1e6:.1f} млн значений через to_numpy)\n"
        if sweep is not None
        else "sweep_index и его префильтр: в этой цепочке их нет (векторизовано, Э9')\n"
    )
    if pierce is not None:
        scanned = f"{pierce[1] / 1e6:.1f} млн" if pierce[1] >= 1_000_000 else f"{pierce[1]:,}"
        reading += (
            f"event scan: {scanned} сравнений цены с порогом (по одному проходу ленты на "
            f"инстанс), {100.0 * pierce[0] / total:.1f}% времени\n"
        )
    missing = f"стадии, которых в цепочке уже нет: {absent}\n" if absent else ""
    headline = (
        f"{spec}: баров={len(tape):,} уровней={len(marks.levels):,} {pairs}"
        f"интентов={len(out.intents):,} отказов={len(out.rejections):,}\n"
        f"{reading}{missing}отказы по причинам: {reasons}"
    )
    return {
        "report": f"{headline}\n\n{render(rows, total, wrapped, wrapped * overhead)}",
        "spec": spec,
        "seconds": total,
        "bars": len(tape),
        "levels": len(marks.levels),
        "pairs": prefilter[1] if prefilter else 0,
        "tradable_pairs": prefilter[2] if prefilter else 0,
        "scan_pairs": pierce[1] if pierce else 0,
        "events": pierce[2] if pierce else 0,
        "attempt_windows": pierce[3] if pierce else 0,
        "intents": len(out.intents),
        "rejections": len(out.rejections),
        "sweep_calls": sweep.calls if sweep is not None else 0,
        "stages": [
            {"stage": row["stage"], "calls": row["calls"], "seconds": row["seconds"]}
            for row in rows
        ],
        "reasons": dict(reasons),
    }


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and run one profile; the ``python tools/profile_intents.py`` face."""
    parser = argparse.ArgumentParser(
        prog="profile_intents",
        description="Stage profile of the M15 entry chain (SPEC_SMC.md §7.13).",
        epilog="run with PYTHONPATH=src:. from the repository root",
    )
    parser.add_argument(
        "--mode",
        choices=("stages", "windows"),
        default="stages",
        help="stages: one window by stage; windows: wall time on all measured windows",
    )
    parser.add_argument("--spec", choices=tuple(WINDOW_SPECS), default=DEFAULT_SPEC)
    parser.add_argument(
        "--windows",
        default=",".join(WINDOW_SPECS),
        help="the windows --mode windows measures, comma separated (default: all of them); "
        "the subset matters when the profile is taken on a slow tree",
    )
    parser.add_argument("--symbol", default=_common.DEFAULT_SYMBOL)
    parser.add_argument("--timeframe", default=_common.DEFAULT_TIMEFRAME)
    parser.add_argument("--json", type=Path, default=None, help="write the measurement as JSON")
    args = parser.parse_args(argv)
    config = StrategyConfig()
    if args.mode == "windows":
        specs = tuple(name.strip() for name in args.windows.split(",") if name.strip())
        unknown = [name for name in specs if name not in WINDOW_SPECS]
        if unknown:
            parser.error(f"unknown window(s) {unknown}; known: {list(WINDOW_SPECS)}")
        measured = measure_windows(specs, args.symbol, args.timeframe, config)
        print(render_windows(measured))
        payload: dict[str, Any] = {"windows": measured}
    else:
        payload = measure_stages(args.spec, args.symbol, args.timeframe, config)
        print(payload["report"])
    if args.json is not None:
        args.json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

