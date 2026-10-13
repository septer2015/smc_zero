"""Tests of the research direction gate of Э17.z++: the switch and its invariants.

The switch exists to measure the direction gate (``bias_skip``) by handing the *unmodified* entry
chain a differently classified direction, so what has to hold is narrower than a strategy: the
shipped mode must be bit for bit the frame of the cache, the softer mode must only ever turn a
refusal into a pass, and neither may read a bar after the one it classifies.  The tests build a
seeded 1500 bar M15 walk and mark it up with the real cache of Э7
(:func:`~smc_zero.optimizer.build_tape_marks`), so the matrix under test is the one a run carries.

The mutations this file is one line away from, and the test each one must break (all three measured on
2026-10-13):

* m1 "put the shipped veto back into the softer tally (both sides)" - the softer mode stops being
  wider than the strict one; breaks :func:`test_the_softer_mode_only_adds_a_direction` and
  :func:`test_an_undefined_timeframe_abstains_and_never_counts_as_a_side`;
* m2 "let an undefined timeframe vote" (count a ``NaN``-free non-negative as a long) - a bar with no
  direction at all answers one; breaks
  :func:`test_an_undefined_timeframe_abstains_and_never_counts_as_a_side`;
* m3 "make the softer mode the default" (``DEFAULT_BIAS_GATE_MODE``) - a research run would hand the
  chain a different gate and call it the shipped one; breaks
  :func:`test_the_switch_is_the_shipped_gate_until_it_is_asked_otherwise` and, because the strict
  reading of that test is taken by default, :func:`test_the_softer_mode_only_adds_a_direction`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smc_zero.config import StrategyConfig
from smc_zero.data_loader import TIMESTAMP_COLUMN
from smc_zero.indicators.bias import classify_trends, trend_column
from smc_zero.optimizer import build_tape_marks
from smc_zero.research.bias_gate import (
    BIAS_GATE_MODES,
    DEFAULT_BIAS_GATE_MODE,
    defined_majority,
    direction_frame,
)

BARS = 1500
AGREEMENT = "majority"
DIR_COLUMNS = (TIMESTAMP_COLUMN, "bias_dir", "bias_state")


def _tape() -> pd.DataFrame:
    """Build the seeded M15 walk of the module: 15 pip steps, every bar closed."""
    rng = np.random.default_rng(7)
    close = 1.1000 + np.cumsum(rng.normal(0.0, 0.0015, BARS))
    open_ = np.concatenate(([1.1000], close[:-1]))
    frame = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + 0.0007,
            "low": np.minimum(open_, close) - 0.0007,
            "close": close,
        }
    )
    frame.insert(
        0, TIMESTAMP_COLUMN, pd.date_range("2026-06-08 00:00", periods=BARS, freq="15min", tz="UTC")
    )
    frame["volume"] = 1
    frame["is_closed"] = True
    return frame


def _trends() -> pd.DataFrame:
    """Return the cached trend matrix of the module tape, built by the real markup cache."""
    strategy = StrategyConfig()
    marks = build_tape_marks(_tape(), strategy)
    return marks.trends


def test_the_switch_is_the_shipped_gate_until_it_is_asked_otherwise() -> None:
    """По умолчанию переключатель отдаёт ровно кадр кэша разметки, и ничего больше (m3).

    Это существо гейта: исследовательский прогон сравнивает ту же стратегию, поэтому режим
    «как в проекте» обязан совпадать с ``TapeMarks.bias_frame`` строка в строку, а не «примерно».
    """
    strategy = StrategyConfig()
    marks = build_tape_marks(_tape(), strategy)
    cfg = strategy.bias

    assert DEFAULT_BIAS_GATE_MODE == "strict"
    assert set(BIAS_GATE_MODES) == {"strict", "defined_majority"}

    frame = direction_frame(marks.trends, cfg.timeframes, agreement=AGREEMENT)
    cached = marks.bias_frame(AGREEMENT)

    assert list(frame.columns) == list(DIR_COLUMNS)
    pd.testing.assert_frame_equal(
        frame.loc[:, list(DIR_COLUMNS)], cached.loc[:, list(DIR_COLUMNS)]
    )


def test_the_softer_mode_only_adds_a_direction() -> None:
    """Мягкий режим никогда не переворачивает направление, он лишь отвечает там, где был ноль (m1)."""
    cfg = StrategyConfig().bias
    trends = _trends()

    strict = direction_frame(trends, cfg.timeframes, agreement=AGREEMENT)["bias_dir"].to_numpy()
    soft = direction_frame(trends, cfg.timeframes, mode="defined_majority")["bias_dir"].to_numpy()

    defined = strict != 0
    assert (soft[defined] == strict[defined]).all()  # никаких переворотов
    assert int((soft == 0).sum()) < int((strict == 0).sum())  # на этой ленте мягкий режим шире
    # ... и мягкий режим отвечает направлением там, где строгий молчал
    assert int((soft != 0).sum()) > int(defined.sum())


def test_an_undefined_timeframe_abstains_and_never_counts_as_a_side() -> None:
    """Неизвестный ТФ воздерживается: он не голосует ни за одну сторону и не глушит строку (m2).

    Матрица ``(бары, ТФ)`` задана руками, поэтому каждый случай читается прямо: строка с одним
    известным направлением мягкий режим берёт, ничья двух известных остаётся нулём, а строка без
    единого направления остаётся ``undefined``.
    """
    values = np.array(
        [
            [np.nan, 1.0, 1.0],  # H1 неизвестен -> строго ноль, мягко +1
            [0.0, -1.0, -1.0],  # H1 без тренда -> строго ноль, мягко -1
            [np.nan, 1.0, -1.0],  # ничья двух известных -> ноль в обоих режимах
            [np.nan, 0.0, 0.0],  # ни одного направления -> ноль, состояние undefined
            [1.0, 1.0, 1.0],  # согласие всех -> +1 в обоих режимах
            [1.0, 1.0, -1.0],  # большинство известных -> +1 в обоих режимах
            [-1.0, -1.0, np.nan],  # снова один известный -> мягко -1
        ]
    )

    soft_dir, soft_state = defined_majority(values)
    strict_dir, _ = classify_trends(values, AGREEMENT)

    assert list(soft_dir) == [1, -1, 0, 0, 1, 1, -1]
    assert list(soft_state) == [
        "majority_long",
        "majority_short",
        "conflict",
        "undefined",
        "agree_long",
        "majority_long",
        "majority_short",
    ]
    # Там, где строгий режим отвечает, мягкий обязан ответить тем же.
    assert list(strict_dir[strict_dir != 0]) == list(soft_dir[strict_dir != 0])


def test_the_verdict_of_a_bar_does_not_read_the_bars_after_it() -> None:
    """Вердикт строки зависит только от её самой: правка будущих баров прошлое не меняет.

    Это проверка отсутствия утечки у самого переключателя: он читает одну строку матрицы трендов,
    а не соседние, поэтому переписанный хвост не может ничего сдвинуть в начале ленты.
    """
    cfg = StrategyConfig().bias
    trends = _trends()
    columns = [trend_column(timeframe) for timeframe in cfg.timeframes]
    before = direction_frame(trends, cfg.timeframes, mode="defined_majority")

    rewritten = trends.copy()
    rewritten.loc[rewritten.index[-10:], columns] = 1.0
    after = direction_frame(rewritten, cfg.timeframes, mode="defined_majority")

    assert (rewritten.loc[rewritten.index[-10:], columns].to_numpy() == 1.0).all()
    pd.testing.assert_frame_equal(
        before.iloc[:-10].reset_index(drop=True), after.iloc[:-10].reset_index(drop=True)
    )


def test_a_frame_that_cannot_carry_a_verdict_is_refused() -> None:
    """Неполная матрица и неизвестный режим отвергаются с именем причины (правило 5)."""
    cfg = StrategyConfig().bias
    trends = _trends()
    incomplete = trends.drop(columns=[trend_column("H4")])

    with pytest.raises(ValueError, match="trend_h4"):
        direction_frame(incomplete, cfg.timeframes, mode="defined_majority")
    with pytest.raises(ValueError, match="mode must be one of"):
        direction_frame(trends, cfg.timeframes, mode="off")
    with pytest.raises(ValueError, match="agreement must be one of"):
        direction_frame(trends, cfg.timeframes, agreement="any", mode="strict")
