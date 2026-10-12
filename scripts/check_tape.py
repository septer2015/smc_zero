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
метки вне сетки, покрытие ниже ``--min-coverage``, пустое окно и несогласованный пояс времени
(``TZ_MISMATCH``, ниже).  Длинный разрыв сам по себе не дефект (новогодние праздники брокера дают
80 часов), поэтому о нём сообщают, а не кричат.  Незакрытый хвост - тоже информация: раннер его
отбросит сам (конвенция §7.10 п.62).

Пояс времени (Э17.B, SPEC_SMC.md §7.26).  Аудит Э17 показал, что и ``./data``, и выгрузка
MetaTrader 5 несут время **брокерского сервера** (GMT+3 летом, GMT+2 зимой, переключение по
американскому календарю), а не UTC.  Поэтому вход принимает ``--source-clock``, и проверка
валидирует **сырой** файл против заявленного пояса, а не «всегда против UTC».

Сырой хвост файла сравнивается с текущим моментом, посчитанным в заявленном поясе, и - отдельно - с
текущим моментом другого пояса:

* хвост совпал с заявленным поясом (в пределах допуска) - пояс подтверждён;
* хвост опережает «сейчас» заявленного пояса больше допуска или совпал с **другим** поясом -
  ``TZ_MISMATCH``: файл не в том поясе, который заявлен.  Это дефект (код выхода 1), потому что
  сдвиг на два-три часа переставляет killzone, сессии, D1/H1-bias и незакрытый хвост;
* хвост не объясняется ни одним поясом (устаревшая выгрузка, выходные, праздник брокера) - пояс не
  подтверждён и не опровергнут: строка в отчёте, без ``FAIL``.

Допуск - до двух периодов сетки таймфрейма, но не больше часа (для D1 - сутки): незакрытый или
чуть подросший хвост свежего экспорта сам по себе не ошибка пояса.  Различить смену пояса можно
только там, где смещение сервера (2-3 ч) заметно больше допуска; на H4 (период 4 ч) это уже не так,
и ограничение записано в SPEC.  Отчёт печатает сырой хвост, ожидаемый момент и то, что прочтёт
движок после нормализации, - чтобы владелец видел разницу между «файл в broker_server» и «файл в
UTC», а не догадывался о ней.

Коды выхода: ``0`` - OK, ``1`` - FAIL (дефект ленты или ``TZ_MISMATCH``), ``2`` - файл не читается
или аргумент неверен.
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
from smc_zero.config import (
    DEFAULT_SOURCE_CLOCK,
    SOURCE_CLOCK_BROKER_SERVER,
    SOURCE_CLOCK_UTC,
)
from smc_zero.data_loader import (
    BROKER_SERVER_EPOCH,
    IS_CLOSED_COLUMN,
    MT5_FORMAT,
    MT5_TIME_COLUMN,
    SOURCE_TIME_COLUMN,
    TIMESTAMP_COLUMN,
    bars_per_day,
    detect_format,
    load_ohlcv,
    period_for,
    utc_stamps,
)
from smc_zero.utils.time import (
    BROKER_SERVER_WINTER_OFFSET_HOURS,
    broker_server_offset_hours,
)

#: Вердикт пояса времени: хвост подтверждает заявленный пояс.
TZ_OK = "OK"
#: Вердикт пояса времени: сырой файл несёт другой пояс, чем заявлено (Э17.B).
TZ_MISMATCH = "TZ_MISMATCH"
#: Вердикт пояса времени: хвост не объясняется ни одним поясом (старая выгрузка или пауза рынка).
TZ_UNCHECKED = "UNCHECKED"

#: Верхняя граница допуска проверки пояса: смещение брокерского сервера (2-3 ч) обязано быть
#: заметно больше допуска, иначе смену пояса не отличить от свежести хвоста.
TZ_TOLERANCE_CAP = pd.Timedelta(hours=1)
#: Допуск проверки пояса для D1: там «период» равен суткам, и сравниваются даты, а не часы.
TZ_DAILY_TOLERANCE = pd.Timedelta(days=1)


@dataclass(frozen=True)
class TapeThresholds:
    """Пороги проверки: ни одного числа этого входа не лежит в другом месте (правило 5).

    ``min_coverage`` - доля баров от торговых дней окна, ниже которой лента считается короткой;
    ``max_gap_bars`` - сколько баров сетки таймфрейма должно пропасть, чтобы о разрыве сообщили
    (порог в барах, а не в часах: шаг H4 - четыре часа, и общий порог в часах считал бы нормой
    каждый шаг).  Пауза между сессиями сюда не попадает: разрыв через полночь или выходные -
    свойство рынка, а не дефект экспорта; ``thin_day_ratio`` - доля обычного дня, ниже которой
    день попадает в отчёт; ``worst_gaps`` и ``worst_days`` - сколько примеров печатать.

    Пороги пояса (Э17.B): ``tz_tolerance_periods`` - допуск в периодах сетки таймфрейма, которым
    отличается незакрытый или чуть подросший хвост свежего экспорта; ``tz_tolerance_cap`` -
    верхняя граница этого допуска (:data:`TZ_TOLERANCE_CAP`); ``tz_daily_tolerance`` - допуск
    для D1, где «период» равен суткам и сравниваются даты.
    """

    min_coverage: float = 0.90
    max_gap_bars: int = 12
    thin_day_ratio: float = 0.5
    worst_gaps: int = 5
    worst_days: int = 3
    tz_tolerance_periods: int = 2
    tz_tolerance_cap: pd.Timedelta = TZ_TOLERANCE_CAP
    tz_daily_tolerance: pd.Timedelta = TZ_DAILY_TOLERANCE


DEFAULT_THRESHOLDS = TapeThresholds()


def _hours(delta: pd.Timedelta) -> float:
    """Return ``delta`` in hours - the unit the report of the clock speaks in."""
    return delta.total_seconds() / 3600.0


def _tz_tolerance(timeframe: str, thresholds: TapeThresholds) -> pd.Timedelta:
    """Return the tolerance of the clock check for ``timeframe``.

    Up to ``tz_tolerance_periods`` bars of the grid, but never more than ``tz_tolerance_cap``: the
    tolerance has to stay well below the offset of the broker server (2-3 hours), otherwise a tape
    in the wrong clock hides inside it.  A daily tape is a different question - its bars step a
    whole day - and answers ``tz_daily_tolerance``.
    """
    if timeframe.upper() == "D1":
        return thresholds.tz_daily_tolerance
    return min(thresholds.tz_tolerance_periods * period_for(timeframe), thresholds.tz_tolerance_cap)


def _server_now(now_utc: pd.Timestamp) -> pd.Timestamp:
    """Return the broker *server* instant of ``now_utc`` (UTC plus 2 or 3 hours by US DST).

    The season is read on the server date, exactly as the loader reads it on a raw stamp: the first
    guess adds the smallest offset there is and the answer corrects it once, so this check and
    :func:`~smc_zero.data_loader.utc_stamps` cannot disagree about which side of a switch a moment
    sits on.  The two switching Sundays are weekend days, so no bar sits in the hour they move.
    """
    naive = now_utc.tz_convert("UTC").tz_localize(None)
    guess = naive + pd.Timedelta(hours=BROKER_SERVER_WINTER_OFFSET_HOURS)
    return now_utc + pd.Timedelta(hours=broker_server_offset_hours(guess.date()))


@dataclass(frozen=True)
class TailClockVerdict:
    """Что сырой хвост ленты говорит о своём поясе (Э17.B).

    ``expected`` - тот момент, которым «сейчас» видит заявленный пояс, ``actual`` - последний
    **сырой** штамп файла, ``delta = actual - expected`` (положительная разница значит, что штампы
    идут впереди «сейчас», то есть в другом поясе).  ``other_delta`` - та же разница против другого
    пояса: если она укладывается в допуск, а своя нет, файл несёт не тот пояс, который заявлен.
    """

    source_clock: str
    verdict: str
    expected: pd.Timestamp
    actual: pd.Timestamp
    delta: pd.Timedelta
    other_delta: pd.Timedelta
    tolerance: pd.Timedelta
    note: str

    @property
    def other_clock(self) -> str:
        """Пояс, который не заявлен: с ним сверяется хвост для второй точки зрения."""
        if self.source_clock == SOURCE_CLOCK_UTC:
            return SOURCE_CLOCK_BROKER_SERVER
        return SOURCE_CLOCK_UTC

    def problem(self) -> str | None:
        """Return the ``TZ_MISMATCH`` line as Russian text, or ``None`` when the clock is fine."""
        if self.verdict != TZ_MISMATCH:
            return None
        return (
            f"{TZ_MISMATCH}: последний бар {self.actual:%Y-%m-%d %H:%M} не соответствует заявленному "
            f"поясу {self.source_clock}: ожидаем около {self.expected:%Y-%m-%d %H:%M}, "
            f"получили {self.actual:%Y-%m-%d %H:%M}, расхождение {_hours(self.delta):+.2f} ч"
        )

    def lines(self) -> tuple[str, ...]:
        """Return the report lines of the verdict: что заявлено и что показал сырой хвост."""
        headline = (
            f"пояс       : заявлен {self.source_clock} (допуск {_hours(self.tolerance):.2f} ч),"
            f" ожидали около {self.expected:%Y-%m-%d %H:%M},"
            f" хвост {self.actual:%Y-%m-%d %H:%M},"
            f" расхождение {_hours(self.delta):+.2f} ч -> {self.verdict}"
        )
        if self.verdict == TZ_OK:
            return (headline,)
        return (
            headline,
            f"  а в поясе {self.other_clock} расхождение было бы {_hours(self.other_delta):+.2f} ч",
            f"  {self.note}",
        )


def clock_verdict(
    last: pd.Timestamp,
    *,
    source_clock: str,
    timeframe: str,
    now: pd.Timestamp,
    thresholds: TapeThresholds = DEFAULT_THRESHOLDS,
) -> TailClockVerdict:
    """Return the verdict of the raw tail ``last`` against the clock ``source_clock`` (Э17.B).

    ``last`` is the last stamp the file itself wrote (read as if it were UTC) and ``now`` the
    current instant in UTC.  The tail is measured against «сейчас» of the declared clock and,
    separately, against «сейчас» of the other one - see the module docstring for the three
    answers that produces, and :func:`_tz_tolerance` for the tolerance of a timeframe.
    """
    tolerance = _tz_tolerance(timeframe, thresholds)
    server_now = _server_now(now)
    if source_clock == SOURCE_CLOCK_UTC:
        expected, other, other_clock = now, server_now, SOURCE_CLOCK_BROKER_SERVER
    else:
        expected, other, other_clock = server_now, now, SOURCE_CLOCK_UTC
    delta = last - expected
    other_delta = last - other
    if delta > tolerance:
        verdict = TZ_MISMATCH
        note = (
            f"сырой хвост опережает «сейчас» пояса {source_clock} на {_hours(delta):.2f} ч,"
            " а штампы файла так далеко вперёд уйти не могут"
        )
    elif abs(other_delta) <= tolerance and abs(delta) > tolerance:
        verdict = TZ_MISMATCH
        note = (
            f"сырой хвост совпадает с поясом {other_clock}, а заявлен {source_clock}:"
            " файл несёт не тот пояс"
        )
    elif abs(delta) <= tolerance:
        verdict = TZ_OK
        note = "хвост совпадает с заявленным поясом"
    else:
        verdict = TZ_UNCHECKED
        note = (
            "хвост не объясняется ни заявленным поясом, ни другим: это похоже на устаревшую"
            " выгрузку или на паузу рынка (выходные, праздник брокера), поэтому пояс не"
            " подтверждён и не опровергнут"
        )
    return TailClockVerdict(
        source_clock=source_clock,
        verdict=verdict,
        expected=expected,
        actual=last,
        delta=delta,
        other_delta=other_delta,
        tolerance=tolerance,
        note=note,
    )


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
    #: Заявленный пояс сырого файла и вердикт по нему (Э17.B): ``None`` - пояс не проверялся.
    source_clock: str = DEFAULT_SOURCE_CLOCK
    clock: TailClockVerdict | None = None
    #: Последний сырой штамп, приведённый к UTC заявленным поясом: то, что прочтёт движок.
    last_utc: pd.Timestamp | None = None

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
        if self.clock is not None and (mismatch := self.clock.problem()) is not None:
            problems.append(mismatch)
        return tuple(problems)

    def lines(self, thresholds: TapeThresholds) -> tuple[str, ...]:
        """Return the human-readable report: one fact per line, then the verdict."""
        if not self.rows:
            head = (
                f"лента      : {self.path} (layout={self.layout}, пояс={self.source_clock})",
                f"окно       : пусто после нарезки ({self.timeframe})",
            )
        else:
            head = (
                f"лента      : {self.path} (layout={self.layout}, пояс={self.source_clock})",
                f"окно       : {self.first:%Y-%m-%d %H:%M} .. {self.last:%Y-%m-%d %H:%M} (сырые штампы)",
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
        details += [f"  тонкий день {day:%Y-%m-%d}: {count} бар" for day, count in self.thin_days]
        return (*head, *self._clock_lines(), *details, *self._verdict_lines(thresholds))

    def _clock_lines(self) -> tuple[str, ...]:
        """Return the report lines of the clock (Э17.B): заявленный пояс и что прочтёт движок.

        Пустой кортеж, когда пояс не проверялся (``clock is None``) - так отчёт остаётся прежним
        для вызывающего, который проверку пояса не просил.
        """
        if self.clock is None:
            return ()
        lines = list(self.clock.lines())
        if self.last_utc is not None:
            lines.append(
                f"  после нормализации движок читает хвост {self.last_utc:%Y-%m-%d %H:%M} UTC"
            )
        if self.first is not None and self.first < BROKER_SERVER_EPOCH:
            lines.append(
                f"  хвост истории: первый бар {self.first:%Y-%m-%d} раньше"
                f" {BROKER_SERVER_EPOCH:%Y-%m-%d}, конвенция пояса до 2022 года не подтверждена"
            )
        return tuple(lines)

    def _verdict_lines(self, thresholds: TapeThresholds) -> tuple[str, ...]:
        """Return the verdict lines: ``FAIL`` with its reasons, or a plain ``OK``."""
        problems = self.failures_against(thresholds)
        if not problems:
            return ("вердикт    : OK",)
        return ("вердикт    : FAIL", *(f"  - {problem}" for problem in problems))

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
            "source_clock": self.source_clock,
            "tz_verdict": None if self.clock is None else self.clock.verdict,
            "tz_note": None if self.clock is None else self.clock.note,
            "tz_expected": None if self.clock is None else self.clock.expected.isoformat(),
            "last_utc": None if self.last_utc is None else self.last_utc.isoformat(),
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


def _now_utc() -> pd.Timestamp:
    """Return the current instant in UTC - the default «сейчас» of the clock check (Э17.B)."""
    return pd.Timestamp.now(tz="UTC")


def inspect_tape(
    path: Path,
    *,
    timeframe: str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    thresholds: TapeThresholds = DEFAULT_THRESHOLDS,
    source_clock: str = DEFAULT_SOURCE_CLOCK,
    now: pd.Timestamp | None = None,
) -> TapeReport:
    """Measure the bars ``path`` carries over ``[start, end]`` and name every defect found.

    The window follows the runner's convention (``_common.slice_window``): a day is a whole UTC day
    and ``end`` is inside it.  ``None`` bounds mean the edges of the file itself, so a plain call
    inspects the whole tape.  The bars are read exactly as a run reads them
    (:func:`~smc_zero.data_loader.load_ohlcv`), except that the still-forming tail is kept: whether
    the export stops on a forming bar is one of the facts being reported.  ``start`` / ``end`` are
    bounds of the **raw** stamps: this check reads the file as the file is, and the clock of its
    stamps is a fact it measures (``source_clock``) rather than a shift it silently applies.

    ``source_clock`` is the clock the raw file is *declared* to carry and ``now`` the instant the
    tail is judged against (``None`` = now, UTC); the verdict itself comes from
    :func:`clock_verdict` and is reported as a defect when the tail contradicts the declaration.
    """
    layout, labels = _raw_marks(path)
    duplicates = int(labels.duplicated().sum())
    ordered = bool(labels.is_monotonic_increasing)

    frame = load_ohlcv(path, format=layout, drop_unclosed=False)
    if frame.empty:
        return _empty_report(path, layout, timeframe, duplicates, ordered, source_clock)
    verdict = clock_verdict(
        frame[TIMESTAMP_COLUMN].iloc[-1],
        source_clock=source_clock,
        timeframe=timeframe,
        now=_now_utc() if now is None else now,
        thresholds=thresholds,
    )
    last_utc = utc_stamps(pd.Series([verdict.actual]), source_clock=source_clock).iloc[0]
    window_start, window_end = _window_span(frame, start, end)
    window = _common.slice_window(frame, window_start, window_end)
    if window.empty:
        return _empty_report(path, layout, timeframe, duplicates, ordered, source_clock)
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
        source_clock=source_clock,
        clock=verdict,
        last_utc=last_utc,
    )


def _empty_report(
    path: Path,
    layout: str,
    timeframe: str,
    duplicates: int,
    ordered: bool,
    source_clock: str = DEFAULT_SOURCE_CLOCK,
) -> TapeReport:
    """Return the report of a tape that holds no bar at all: one failure, no geometry.

    The clock is named but not judged: a file without a bar has no tail to compare with «сейчас»,
    and the empty window already fails on its own.
    """
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
        source_clock=source_clock,
    )


def print_report(report: TapeReport, thresholds: TapeThresholds, *, as_json: bool) -> None:
    """Print the report of ``report``: the JSON face or the readable one."""
    if as_json:
        print(json.dumps(report.as_dict(thresholds), ensure_ascii=False, indent=2))
        return
    for line in report.lines(thresholds):
        print(line)


def _read_moment(value: str) -> pd.Timestamp:
    """Read a date or a full stamp as a UTC instant (the ``argparse`` ``type=`` of ``--now``).

    ``YYYY-MM-DD`` and ``YYYY-MM-DD HH:MM:SS`` are both accepted; a stamp without a zone is read
    as UTC, because every moment this project speaks about is UTC (Э17.B).
    """
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a moment, use YYYY-MM-DD or 'YYYY-MM-DD HH:MM:SS'"
        ) from error
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of ``smc-check-tape``."""
    parser = argparse.ArgumentParser(
        prog="smc-check-tape",
        description=(
            "Проверить ленту <SYMBOL>_<TIMEFRAME>.csv: дубли, порядок, сетку, покрытие, разрывы, "
            "пояс времени."
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
    parser.add_argument(
        "--now",
        type=_read_moment,
        default=None,
        help=(
            "момент проверки пояса, YYYY-MM-DD или 'YYYY-MM-DD HH:MM:SS' в UTC "
            "(по умолчанию - текущее время); пин нужен для воспроизводимой проверки "
            "на закрытом рынке"
        ),
    )
    parser.add_argument("--json", action="store_true", help="машинный отчёт вместо читаемого")
    return parser


def run(args: argparse.Namespace) -> int:
    """Inspect the tape ``args`` names and return the exit code of the verdict."""
    thresholds = TapeThresholds(min_coverage=args.min_coverage, max_gap_bars=args.max_gap_bars)
    path = _common.tape_path(args.symbol, args.timeframe, source=args.data_source)
    try:
        clock = _common.resolve_source_clock(args)
        report = inspect_tape(
            path,
            timeframe=args.timeframe,
            start=args.start,
            end=args.end,
            thresholds=thresholds,
            source_clock=clock,
            now=args.now,
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


