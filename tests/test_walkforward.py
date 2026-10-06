"""Walk-forward split tests: anchored and rolling folds, no overlap, fold aggregation (Э6').

The tape here is synthetic and positional: bar ``i`` of a 1000 bar frame is a row labelled ``i``,
so ``train.index[-1] + 1 == test.index[0]`` and ``set(train.index) & set(test.index)`` are read by
hand.  The windows of the default configuration are the M15 units of the project (96 bars a day).

The three mutations the module is one line away from, and the test each one must break:

* m1 "cut the test window at ``train_end`` instead of after it" - the two windows share bars and
  the fold simulates its own training data; breaks
  :func:`test_a_fold_never_shares_a_bar_with_its_train_window`;
* m2 "ignore ``anchored``" - every fold slides like a rolling one, so a later fold's train window
  no longer starts at the tape's first bar; breaks
  :func:`test_the_anchored_folds_always_start_at_the_first_bar`;
* m3 "sum the folds instead of averaging them" - the aggregate reports the total, not the mean;
  breaks :func:`test_the_aggregate_averages_the_folds_and_reports_their_spread`.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from smc_zero.backtester.walkforward import (
    FOLD_METRIC_FIELDS,
    aggregate_fold_metrics,
    default_walk_forward,
    split_walkforward,
)
from smc_zero.config import WalkForwardConfig

BARS = 1000
#: The short configuration of most tests: 200 bars of warm-up, 100 bars out of sample.
SHORT = WalkForwardConfig(min_train_bars=200, test_period_bars=100)


def _frame(bars: int = BARS, start: str = "2026-06-08 00:00") -> pd.DataFrame:
    """Build a positional tape: ``bars`` M15 rows labelled ``0 .. bars - 1``."""
    return pd.DataFrame({"timestamp": pd.date_range(start, periods=bars, freq="15min", tz="UTC")})


def _bounds(folds: list[tuple[pd.DataFrame, pd.DataFrame]]) -> list[tuple[int, int, int, int]]:
    """Read a fold list as ``(train_len, train_start, test_first, test_last)``."""
    return [
        (len(train), int(train.index[0]), int(test.index[0]), int(test.index[-1]))
        for train, test in folds
    ]


def test_the_folds_of_a_short_tape_start_at_the_warm_up_the_config_asks_for() -> None:
    """1000 bars, a 200 bar warm-up and a 100 bar test window make exactly 8 folds."""
    folds = split_walkforward(_frame(), SHORT)

    assert _bounds(folds) == [
        (200, 0, 200, 299),
        (300, 0, 300, 399),
        (400, 0, 400, 499),
        (500, 0, 500, 599),
        (600, 0, 600, 699),
        (700, 0, 700, 799),
        (800, 0, 800, 899),
        (900, 0, 900, 999),
    ]

def test_the_fold_windows_of_an_entry_timeframe_are_written_in_its_own_bars() -> None:
    """A 60 / 30 day M5 window is 17 280 / 8 640 bars; the M15 pair stays the dataclass default.

    The windows are lengths *in bars of the entry tape*, so the second hierarchy cannot reuse the
    M15 numbers: its tape carries four times as many bars a day.
    """
    daily = default_walk_forward("M15")
    assert (daily.min_train_bars, daily.test_period_bars) == (96 * 120, 96 * 60)
    assert daily == WalkForwardConfig()

    fast = default_walk_forward("M5")
    assert (fast.min_train_bars, fast.test_period_bars) == (288 * 60, 288 * 30)
    assert fast.train_period_bars == 288 * 60
    # the shipped M5 tape (100 136 bars of 2025-05-06 .. 2026-09-22) holds nine folds under them
    assert (100_136 - fast.min_train_bars) // fast.test_period_bars == 9

    with pytest.raises(ValueError, match="no walk-forward windows"):
        default_walk_forward("H1")


def test_a_fold_never_shares_a_bar_with_its_train_window() -> None:
    """Train and test are adjacent windows of one tape: no shared bar and no gap (m1)."""
    for train, test in split_walkforward(_frame(), SHORT):
        assert set(train.index) & set(test.index) == set()
        assert int(test.index[0]) == int(train.index[-1]) + 1
        # The frames are views of the caller's tape, not copies of a re-indexed one.
        assert train.index[0] == test.index[0] - len(train)


def test_the_anchored_folds_always_start_at_the_first_bar() -> None:
    """``anchored=True`` keeps the left edge of the train window at the tape's first bar (m2).

    The two schemes are compared on the *same* ``train_period_bars``: with the default 120 day
    window the rolling one would be clipped at the first bar of this short tape and the scheme
    would be unobservable - and the mutation "ignore ``anchored``" would pass unseen.
    """
    cfg = WalkForwardConfig(min_train_bars=200, test_period_bars=100, train_period_bars=300)
    anchored = split_walkforward(_frame(), cfg)
    rolling = split_walkforward(_frame(), replace(cfg, anchored=False))

    assert [int(train.index[0]) for train, _ in anchored] == [0] * 8
    assert [int(train.index[0]) for train, _ in rolling] == [0, 0, 100, 200, 300, 400, 500, 600]
    assert [len(train) for train, _ in rolling] == [200, 300, 300, 300, 300, 300, 300, 300]
    # The scheme changes the train window only: the test windows are the same pair of stamps.
    assert [int(test.index[0]) for _, test in rolling] == [
        int(test.index[0]) for _, test in anchored
    ]
    for train, test in rolling:
        assert set(train.index) & set(test.index) == set()
        assert int(test.index[0]) == int(train.index[-1]) + 1


def test_a_longer_warm_up_produces_fewer_folds() -> None:
    """``min_train_bars`` moves the first test window and the fold count with it."""
    folds = split_walkforward(_frame(), WalkForwardConfig(min_train_bars=500, test_period_bars=100))

    assert _bounds(folds) == [
        (500, 0, 500, 599),
        (600, 0, 600, 699),
        (700, 0, 700, 799),
        (800, 0, 800, 899),
        (900, 0, 900, 999),
    ]


def test_a_tape_too_short_for_one_fold_has_no_folds() -> None:
    """A partial out-of-sample window is not a fold: the split returns nothing instead."""
    frame = _frame(250)
    wide = WalkForwardConfig(min_train_bars=150, test_period_bars=100)

    assert split_walkforward(frame, SHORT) == []
    assert _bounds(split_walkforward(frame, wide)) == [(150, 0, 150, 249)]


def test_the_default_windows_are_the_m15_units_of_the_project() -> None:
    """The default config is the 120 / 60 / 120 day triple in M15 bars of 96 bars a day."""
    config = WalkForwardConfig()

    assert config.anchored is True
    assert config.test_period_bars == 96 * 60
    assert config.min_train_bars == 96 * 120
    assert config.train_period_bars == 96 * 120


def test_the_default_windows_cut_four_years_into_a_grid_of_fourteen_folds() -> None:
    """On four years of M15 bars the defaults make 14 folds - the arithmetic of §7.10 п.62."""
    four_years = 96 * 250 * 4
    folds = split_walkforward(_frame(four_years), WalkForwardConfig())

    assert len(folds) == 14
    assert (len(folds[0][0]), len(folds[0][1])) == (96 * 120, 96 * 60)
    assert (len(folds[-1][0]), len(folds[-1][1])) == (96 * 120 + 13 * 96 * 60, 96 * 60)


@pytest.mark.parametrize("field", ["min_train_bars", "test_period_bars", "train_period_bars"])
def test_a_window_of_zero_bars_is_refused(field: str) -> None:
    """Every window is at least one bar long: a zero window would fake a fold."""
    with pytest.raises(ValueError, match=field):
        WalkForwardConfig(**{field: 0})

def _fold_table(
    trades: float, profit: float, win_rate: float, pf: float, max_dd: float, sharpe: float
) -> dict[str, float]:
    """One fold's metric table: the six headline fields filled in, the rest of Э5' beside them."""
    return {
        "trades": trades,
        "profit": profit,
        "win_rate": win_rate,
        "pf": pf,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "return_pct": profit / 10.0,
        "final_balance": 10_000.0 + profit,
    }


def test_the_aggregate_averages_the_folds_and_reports_their_spread() -> None:
    """Three folds with hand-chosen numbers: every field is its mean and its population σ (m3)."""
    aggregate = aggregate_fold_metrics(
        [
            _fold_table(trades=1, profit=10.0, win_rate=0.0, pf=1.0, max_dd=5.0, sharpe=0.5),
            _fold_table(trades=2, profit=20.0, win_rate=50.0, pf=2.0, max_dd=10.0, sharpe=1.0),
            _fold_table(trades=3, profit=30.0, win_rate=100.0, pf=3.0, max_dd=15.0, sharpe=1.5),
        ]
    )

    assert aggregate["folds"] == 3
    assert aggregate["trades_mean"] == pytest.approx(2.0)
    assert aggregate["trades_std"] == pytest.approx(0.82, abs=0.005)
    assert aggregate["profit_mean"] == pytest.approx(20.0)
    assert aggregate["profit_std"] == pytest.approx(8.16, abs=0.005)
    assert aggregate["win_rate_mean"] == pytest.approx(50.0)
    assert aggregate["win_rate_std"] == pytest.approx(40.82, abs=0.005)
    assert aggregate["pf_mean"] == pytest.approx(2.0)
    assert aggregate["pf_std"] == pytest.approx(0.82, abs=0.005)
    assert aggregate["max_dd_mean"] == pytest.approx(10.0)
    assert aggregate["max_dd_std"] == pytest.approx(4.08, abs=0.005)
    assert aggregate["sharpe_mean"] == pytest.approx(1.0)
    assert aggregate["sharpe_std"] == pytest.approx(0.41, abs=0.005)
    assert set(aggregate) == {"folds"} | {
        f"{field}_{suffix}" for field in FOLD_METRIC_FIELDS for suffix in ("mean", "std")
    }


def test_the_aggregate_of_one_fold_is_that_fold() -> None:
    """With a single fold the mean is the fold and the spread is zero - the reading is literal."""
    table = _fold_table(trades=4, profit=-12.5, win_rate=25.0, pf=0.4, max_dd=3.25, sharpe=-1.5)

    aggregate = aggregate_fold_metrics([table])

    assert aggregate["folds"] == 1
    for field in FOLD_METRIC_FIELDS:
        assert aggregate[f"{field}_mean"] == pytest.approx(float(table[field]))
        assert aggregate[f"{field}_std"] == 0.0


def test_the_aggregate_refuses_an_empty_fold_list() -> None:
    """No folds means no out-of-sample evidence: the aggregate raises instead of faking zeros."""
    with pytest.raises(ValueError, match="no fold"):
        aggregate_fold_metrics([])


def test_the_aggregate_refuses_a_fold_without_a_headline_field() -> None:
    """A table that did not come from ``calc_metrics`` is a hard error, never a silent NaN."""
    table = _fold_table(trades=1, profit=10.0, win_rate=0.0, pf=1.0, max_dd=5.0, sharpe=0.5)
    del table["sharpe"]

    with pytest.raises(ValueError, match="sharpe"):
        aggregate_fold_metrics([table, _fold_table(1, 1.0, 0.0, 1.0, 1.0, 0.5)])

