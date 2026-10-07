"""``smc-check-tape``: что на самом деле несёт CSV, на котором собираются прогоны (Э13.2).

Лента M5 второй иерархии (H4 -> M15 -> M5) приходит из MetaTrader 5 руками, и у такой выгрузки
три типовые беды: обрезанный хвост, стык двух выгрузок с задвоенным баром и дырка посередине.
Прогон по такой ленте не падает громко: walk-forward §7.10 п.61 просто теряет фолды, а счёт
§7.22 п.115 читает более короткий пул OOS, чем обещает окно. Этот вход отвечает на вопрос
*что в файле лежит* до того, как исследование сожжёт часы.

Чтение то же, что у раннера: файл берётся по ``--data-source`` через
:func:`scripts._common.tape_path`, поэтому ``project`` смотрит в ``./data``, а ``mt5`` - в
``SMC_DATA_DIR`` (``~/_data/mt5``).  Дубли и порядок меток проверяются по сырому файлу -
загрузчик их молча схлопывает, - а геометрия сетки по каноничной ленте
(:func:`~smc_zero.data_loader.load_ohlcv`).

Отчёт печатает строки, границы окна, дубли, порядок, метки вне сетки (не кратные периоду ТФ),
покрытие (бары против торговых дней окна, умноженных на
:func:`~smc_zero.data_loader.bars_per_day`), пропуски внутри торгового дня (больше
``--max-gap-bars`` баров сетки) и три самых тонких дня.  ``--json`` печатает то же машинно.

Вердикт ``FAIL`` ставят только дефекты, о которых молчать нельзя: дубли, нарушенный порядок,
метки вне сетки, покрытие ниже ``--min-coverage`` и пустое окно.  Длинный разрыв сам по себе не
дефект (новогодние праздники брокера дают 80 часов), поэтому о нём сообщают, а не кричат.
Незакрытый хвост - тоже информация: раннер его отбросит сам (конвенция §7.10 п.62).

Коды выхода: ``0`` - OK, ``1`` - FAIL, ``2`` - файл не читается или аргумент неверен.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import _common
from smc_zero.data_loader import (
    IS_CLOSED_COLUMN,
    MT5_FORMAT,
    MT5_TIME_COLUMN,
    SOURCE_TIME_COLUMN,
    TIMESTAMP_COLUMN,
    bars_per_day,
    detect_format,
    load_ohlcv,
    period_for,
)


@dataclass(frozen=True)
class TapeThresholds:
    """Пороги проверки: ни одного числа этого входа не лежит в другом месте (правило 5).

    ``min_coverage`` - доля баров от торговых дней окна, ниже которой лента считается короткой;
    ``max_gap_bars`` - сколько баров сетки таймфрейма должно пропасть, чтобы о разрыве сообщили
    (порог в барах, а не в часах: шаг H4 - четыре часа, и общий порог в часах считал бы нормой
    каждый шаг).  Пауза между сессиями сюда не попадает: разрыв через полночь или выходные -
    свойство рынка, а не дефект экспорта; ``thin_day_ratio`` - доля обычного дня, ниже которой
    день попадает в отчёт; ``worst_gaps`` и ``worst_days`` - сколько примеров печатать.
    """

    min_coverage: float = 0.90
    max_gap_bars: int = 12
    thin_day_ratio: float = 0.5
    worst_gaps: int = 5
    worst_days: int = 3


DEFAULT_THRESHOLDS = TapeThresholds()


@dataclass(frozen=True)
class TapeReport:
    """Что несёт файл: границы окна, геометрия сетки и найденные дефекты.

    ``gaps`` - самые длинные разрывы как ``(метка бара после разрыва, часы)``, ``thin_days`` -
    самые тонкие дни как ``(дата, число баров)``.  ``expected_bars`` считается по торговым дням
    **окна** (:func:`_window_span`), а не по дням, которые лента принесла: иначе обрезанный
    экспорт всегда выглядел бы полным.  Неполный край окна честно даёт покрытие чуть меньше
    единицы - это свойство окна, а не дефект ленты.
    """

    path: Path
    layout: str
    timeframe: str
    rows: int
    first: pd.Timestamp | None
    last: pd.Timestamp | None
    duplicates: int
    ordered: bool
    off_grid: int
    days: int
    trading_days: int
    expected_bars: int
    coverage: float
    unclosed_tail: bool
    intraday_gaps: int
    gaps: tuple[tuple[pd.Timestamp, float], ...]
    thin_days: tuple[tuple[pd.Timestamp, int], ...]

    def failures_against(self, thresholds: TapeThresholds) -> tuple[str, ...]:
        """Return the defects that make the tape unfit for a run, each named in Russian."""
        problems: list[str] = []
        if not self.rows:
            problems.append("в окне нет ни одного бара")
        if self.duplicates:
            problems.append(f"дубли меток времени: {self.duplicates}")
        if not self.ordered:
            problems.append("метки времени не отсортированы по возрастанию")
        if self.off_grid:
            problems.append(f"метки вне сетки {self.timeframe}: {self.off_grid}")
        if self.coverage < thresholds.min_coverage:
            problems.append(
                f"покрытие {self.coverage:.3f} ниже порога {thresholds.min_coverage:.2f}"
            )
        return tuple(problems)

    def lines(self, thresholds: TapeThresholds) -> tuple[str, ...]:
        """Return the human-readable report: one fact per line, then the verdict."""
        if not self.rows:
            head = (
                f"лента      : {self.path} (layout={self.layout})",
                f"окно       : пусто после нарезки ({self.timeframe})",
            )
        else:
            head = (
                f"лента      : {self.path} (layout={self.layout})",
                f"окно       : {self.first:%Y-%m-%d %H:%M} .. {self.last:%Y-%m-%d %H:%M} UTC",
                (
                    f"бары       : {self.rows} (дни {self.days}, торговые {self.trading_days},"
                    f" ожидалось {self.expected_bars}, покрытие {self.coverage:.3f})"
                ),
                (
                    f"порядок    : {'sorted' if self.ordered else 'НЕ sorted'},"
                    f" дубли {self.duplicates}, вне сетки {self.off_grid},"
                    f" незакрытый хвост {'да' if self.unclosed_tail else 'нет'}"
                ),
                (
                    f"пропуски   : > {thresholds.max_gap_bars} бар сетки {self.timeframe}"
                    f" (около {self._gap_floor(thresholds):.1f} ч)"
                    f" внутри торгового дня: {self.intraday_gaps}"
                ),
                (
                    f"тонкие дни : {len(self.thin_days)} ниже"
                    f" {self._thin_floor(thresholds):.0f} бар (примеры ниже)"
                ),
            )
        details = [
            f"  пропуск {hours:6.2f} ч перед баром {stamp:%Y-%m-%d %H:%M} UTC"
            for stamp, hours in self.gaps
        ]
        details += [
            f"  тонкий день {day:%Y-%m-%d}: {count} бар"
            for day, count in self.thin_days
        ]
        problems = self.failures_against(thresholds)
        verdict = ["вердикт    : FAIL"] + [f"  - {problem}" for problem in problems]
        if not problems:
            verdict = ["вердикт    : OK"]
        return (*head, *details, *verdict)

    def as_dict(self, thresholds: TapeThresholds) -> dict[str, object]:
        """Return the same report as data, for the ``--json`` face."""
        return {
            "path": str(self.path),
            "layout": self.layout,
            "timeframe": self.timeframe,
            "rows": self.rows,
            "first": None if self.first is None else self.first.isoformat(),
            "last": None if self.last is None else self.last.isoformat(),
            "duplicates": self.duplicates,
            "ordered": self.ordered,
            "off_grid": self.off_grid,
            "days": self.days,
            "trading_days": self.trading_days,
            "expected_bars": self.expected_bars,
            "coverage": round(self.coverage, 6),
            "unclosed_tail": self.unclosed_tail,
            "intraday_gaps": self.intraday_gaps,
            "gaps": [[stamp.isoformat(), round(hours, 4)] for stamp, hours in self.gaps],
            "thin_days": [[day.isoformat(), count] for day, count in self.thin_days],
            "failures": list(self.failures_against(thresholds)),
            "verdict": "FAIL" if self.failures_against(thresholds) else "OK",
        }

    def _thin_floor(self, thresholds: TapeThresholds) -> float:
        """Return the bar count under which a day is reported as thin."""
        return bars_per_day(self.timeframe) * thresholds.thin_day_ratio

    def _gap_floor(self, thresholds: TapeThresholds) -> float:
        """Return the hole threshold of this timeframe in hours, for the report line."""
        return (thresholds.max_gap_bars + 1) * period_for(self.timeframe).total_seconds() / 3600.0


def _raw_marks(path: Path) -> tuple[str, pd.Series]:
    """Return the layout of ``path`` and its raw time labels, as the file itself holds them.

    The labels stay strings on purpose: duplicates and ordering are properties of the export, and
    the canonical loader drops duplicates before anyone could see them (``dedup=True`` for the MT5
    layout, a ``ValueError`` for the project one).  ISO-like stamps compare correctly as text.
    """
    raw = pd.read_csv(path)
    layout = detect_format(raw.columns, source=str(path))
    column = MT5_TIME_COLUMN if layout == MT5_FORMAT else SOURCE_TIME_COLUMN
    return layout, raw[column].astype("string")


def _window_span(
    frame: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp | None
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the window of the check: what the caller asked for, else the edges of the tape."""
    stamps = frame[TIMESTAMP_COLUMN]
    return (
        stamps.min() if start is None else start,
        stamps.max() if end is None else end,
    )


def _trading_days(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Return how many Monday-to-Friday days the window holds, both edges included."""
    calendar = pd.date_range(start.normalize(), end.normalize(), freq="D")
    return int((calendar.weekday < 5).sum())


def inspect_tape(
    path: Path,
    *,
    timeframe: str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    thresholds: TapeThresholds = DEFAULT_THRESHOLDS,
) -> TapeReport:
    """Measure the bars ``path`` carries over ``[start, end]`` and name every defect found.

    The window follows the runner's convention (``_common.slice_window``): a day is a whole UTC day
    and ``end`` is inside it.  ``None`` bounds mean the edges of the file itself, so a plain call
    inspects the whole tape.  The bars are read exactly as a run reads them
    (:func:`~smc_zero.data_loader.load_ohlcv`), except that the still-forming tail is kept: whether
    the export stops on a forming bar is one of the facts being reported.
    """
    layout, labels = _raw_marks(path)
    duplicates = int(labels.duplicated().sum())
    ordered = bool(labels.is_monotonic_increasing)

    frame = load_ohlcv(path, format=layout, drop_unclosed=False)
    if frame.empty:
        return _empty_report(path, layout, timeframe, duplicates, ordered)
    window_start, window_end = _window_span(frame, start, end)
    window = _common.slice_window(frame, window_start, window_end)
    if window.empty:
        return _empty_report(path, layout, timeframe, duplicates, ordered)
    trading_days = _trading_days(window_start, window_end)

    stamps = window[TIMESTAMP_COLUMN]
    period_seconds = period_for(timeframe).total_seconds()
    offsets = (stamps - stamps.dt.normalize()).dt.total_seconds().to_numpy()
    off_grid = int((np.mod(offsets, period_seconds) != 0.0).sum())

    days_index = stamps.dt.normalize()
    unique_days = days_index.drop_duplicates()
    expected_bars = trading_days * bars_per_day(timeframe)
    coverage = len(window) / expected_bars if expected_bars else 0.0

    per_day = stamps.groupby(days_index).size()
    thin = per_day[per_day < bars_per_day(timeframe) * thresholds.thin_day_ratio]
    thin_days = tuple(
        (pd.Timestamp(day), int(count)) for day, count in thin.nsmallest(thresholds.worst_days).items()
    )

    hours = stamps.diff().dt.total_seconds() / 3600.0
    # A pause between sessions is not a defect of the export: only a hole *inside* one trading day
    # is reported, so a weekend or an overnight break never shows up here.  The threshold is counted
    # in bars of the grid, because one hour is a quarter of an H4 bar and a tenth of an M15 one.
    missing = (hours * 3600.0 / period_seconds).round() - 1.0
    same_day = (stamps.shift(1).dt.normalize() == days_index) & (days_index.dt.weekday < 5)
    intraday = hours[(missing > thresholds.max_gap_bars) & same_day]
    gaps = tuple(
        (pd.Timestamp(stamps.loc[index]), float(value))
        for index, value in intraday.nlargest(thresholds.worst_gaps).items()
    )

    return TapeReport(
        path=path,
        layout=layout,
        timeframe=timeframe.upper(),
        rows=len(window),
        first=stamps.iloc[0],
        last=stamps.iloc[-1],
        duplicates=duplicates,
        ordered=ordered,
        off_grid=off_grid,
        days=int(unique_days.size),
        trading_days=trading_days,
        expected_bars=expected_bars,
        coverage=coverage,
        unclosed_tail=not bool(window[IS_CLOSED_COLUMN].iloc[-1]),
        intraday_gaps=int(intraday.size),
        gaps=gaps,
        thin_days=thin_days,
    )


def _empty_report(
    path: Path, layout: str, timeframe: str, duplicates: int, ordered: bool
) -> TapeReport:
    """Return the report of a tape that holds no bar at all: one failure, no geometry."""
    return TapeReport(
        path=path,
        layout=layout,
        timeframe=timeframe.upper(),
        rows=0,
        first=None,
        last=None,
        duplicates=duplicates,
        ordered=ordered,
        off_grid=0,
        days=0,
        trading_days=0,
        expected_bars=0,
        coverage=0.0,
        unclosed_tail=False,
        intraday_gaps=0,
        gaps=(),
        thin_days=(),
    )


def print_report(report: TapeReport, thresholds: TapeThresholds, *, as_json: bool) -> None:
    """Print the report of ``report``: the JSON face or the readable one."""
    if as_json:
        print(json.dumps(report.as_dict(thresholds), ensure_ascii=False, indent=2))
        return
    for line in report.lines(thresholds):
        print(line)


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of ``smc-check-tape``."""
    parser = argparse.ArgumentParser(
        prog="smc-check-tape",
        description=(
            "Проверить ленту <SYMBOL>_<TIMEFRAME>.csv: дубли, порядок, сетку, покрытие, разрывы."
        ),
    )
    parser.add_argument("--symbol", default="EURUSD", help="символ ленты, по умолчанию EURUSD")
    parser.add_argument("--timeframe", default="M15", help="таймфрейм ленты, по умолчанию M15")
    parser.add_argument(
        "--start",
        type=_common.read_day,
        default=None,
        help="первый день окна, YYYY-MM-DD (по умолчанию - первый бар файла)",
    )
    parser.add_argument(
        "--end",
        type=_common.read_day,
        default=None,
        help="последний день окна, YYYY-MM-DD включительно (по умолчанию - последний бар файла)",
    )
    _common.add_data_source_argument(parser)
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=DEFAULT_THRESHOLDS.min_coverage,
        help="минимальное покрытие торговых дней "
        f"(по умолчанию {DEFAULT_THRESHOLDS.min_coverage})",
    )
    parser.add_argument(
        "--max-gap-bars",
        type=int,
        default=DEFAULT_THRESHOLDS.max_gap_bars,
        help="сколько баров сетки таймфрейма должно пропасть, чтобы сообщить о пропуске "
        f"(по умолчанию {DEFAULT_THRESHOLDS.max_gap_bars}); выходные и ночной перерыв "
        "пропуском не считаются",
    )
    parser.add_argument("--json", action="store_true", help="машинный отчёт вместо читаемого")
    return parser


def run(args: argparse.Namespace) -> int:
    """Inspect the tape ``args`` names and return the exit code of the verdict."""
    thresholds = TapeThresholds(min_coverage=args.min_coverage, max_gap_bars=args.max_gap_bars)
    path = _common.tape_path(args.symbol, args.timeframe, source=args.data_source)
    try:
        report = inspect_tape(
            path,
            timeframe=args.timeframe,
            start=args.start,
            end=args.end,
            thresholds=thresholds,
        )
    except (OSError, ValueError) as error:
        print(f"ошибка: {error}", file=sys.stderr)
        return 2
    print_report(report, thresholds, as_json=args.json)
    return 1 if report.failures_against(thresholds) else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` (``None`` = ``sys.argv``) and check one tape; the console entry point."""
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())


