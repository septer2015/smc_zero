"""Walk-forward integration tests: the Э5' engine inside the Э6' fold loop (Э6').

The tape is synthetic and positional: 500 closed M15 bars from a Monday 00:00 UTC, cut by a 300
bar warm-up and a 100 bar test window into the two folds ``[300, 400)`` and ``[400, 500)``.  Every
out-of-sample window carries the same two setups - a short that reaches its target and a short that
reaches its stop - so every fold books one TP and one SL trade, while its numbers still come from
the real engine of Э5' with the EURUSD price list (spread included, rule 4).  The second window's
target sits 10 pips further out than the first one's: the folds must differ, otherwise a spread of
zero would make the aggregate's σ assertion vacuous.

The two mutations the layer is one line away from, and the test each one must break:

* m1 "hand the engine a window that starts inside its fold's train window" - the fold simulates
  bars it was allowed to fit on, the lookahead walk-forward exists to prevent; breaks
  :func:`test_no_fold_of_a_run_shares_a_bar_with_its_train_window`;
* m2 "sum the folds instead of averaging them" - the aggregate reports the total profit of the
  folds as if it were the mean; breaks
  :func:`test_the_aggregate_of_the_folds_is_their_mean_with_a_spread`.
"""

from __future__ import annotations

import pandas as pd
import pytest

from smc_zero.backtester.engine import run_backtest
from smc_zero.backtester.metrics import SL_RESULT, TP_RESULT
from smc_zero.backtester.walkforward import (
    WalkForwardResult,
    aggregate_fold_metrics,
    run_walkforward,
)
from smc_zero.config import BacktestConfig, WalkForwardConfig
from smc_zero.indicators.levels import PDH
from smc_zero.strategy.base import TradeIntent

BARS = 500
#: The two folds of the fixture: 300 bars of warm-up, then 100 bar test windows.
SHORT = WalkForwardConfig(min_train_bars=300, test_period_bars=100)
WINDOW = SHORT.test_period_bars
#: A quiet bar: the short limits of a window (1.1000) stay out of the range of this one.
FLAT = (1.0950, 1.0960, 1.0940, 1.0950)
#: The targets of the two folds: the first window takes 30 pips, the second one 40.
TARGETS = (1.0940, 1.0900)


def _window_rows(tp: float) -> list[tuple[float, float, float, float]]:
    """Return the bars of one test window: a winning short, then a short that is stopped out.

    Both setups are armed on the same MSK day (signal on bar 2 and bar 6, fills on the next bar,
    exits on bar 4 and bar 8), so no trade is carried over a date change and no swap is charged.
    """
    rows = [FLAT] * WINDOW
    rows[2] = (1.0950, 1.1010, 1.0945, 1.0950)  # the signal bar already trades the limit
    rows[3] = (1.0950, 1.1008, 1.0945, 1.0950)  # the fill at 1.1000
    rows[4] = (1.0950, 1.0960, tp - 0.0005, tp)  # the target
    rows[6] = (1.0950, 1.1010, 1.0945, 1.0950)  # the second signal
    rows[7] = (1.0950, 1.1008, 1.0945, 1.0950)  # the fill
    rows[8] = (1.0950, 1.1035, 1.0935, 1.0950)  # one bar covers the stop and the target: SL
    return rows


def _tape() -> pd.DataFrame:
    """Build the 500 bar tape of the module: a flat warm-up, then the two windows of the folds."""
    rows = [FLAT] * SHORT.min_train_bars + _window_rows(TARGETS[0]) + _window_rows(TARGETS[1])
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    frame.insert(
        0, "timestamp", pd.date_range("2026-06-08 00:00", periods=BARS, freq="15min", tz="UTC")
    )
    frame["volume"] = 1
    frame["is_closed"] = True
    return frame


def _intent(window: pd.DataFrame, bar: int, tp: float) -> TradeIntent:
    """One accepted EURUSD short of ``window``: a 30 pip stop and ``tp`` as the target."""
    return TradeIntent(
        open_time=window["timestamp"].iloc[bar],
        bar=bar,
        side="short",
        entry=1.1000,
        sl=1.1030,
        tp=tp,
        tp_source="liquidity",
        sl_source="sweep_extreme",
        sl_pips=30.0,
        rr=2.0,
        level_name=PDH,
        level_date=pd.Timestamp("2026-06-08"),
        level_price=1.1030,
        setup_type="fresh",
        sweep_bar=bar - 2,
        choch_bar=bar - 1,
        fvg_bar=bar,
    )


def _plan(window: pd.DataFrame) -> list[TradeIntent]:
    """Return the two setups of a window, armed against that window and nothing else.

    The target is picked by the window's own label so the two folds differ: the second test window
    takes 40 pips where the first takes 30.
    """
    target = TARGETS[0] if int(window.index[0]) == SHORT.min_train_bars else TARGETS[1]
    return [_intent(window, 2, target), _intent(window, 6, target)]


def _run(frame: pd.DataFrame) -> WalkForwardResult:
    """Run the walk-forward of the fixture: the Э4' stub is ``_plan`` on the test window."""
    return run_walkforward(frame, lambda train, test: _plan(test), SHORT, BacktestConfig())
def test_every_fold_is_the_engine_run_of_its_own_test_window() -> None:
    """A fold's table is the Э5' engine's own run on that fold's test window, bar for bar."""
    frame = _tape()
    walk = _run(frame)

    assert len(walk.fold_metrics) == 2
    assert (walk.config, walk.backtest) == (SHORT, BacktestConfig())
    for position, start in enumerate((SHORT.min_train_bars, SHORT.min_train_bars + WINDOW)):
        window = frame.iloc[start : start + WINDOW]
        expected = run_backtest(window, _plan(window), BacktestConfig())

        assert expected.trades["result"].tolist() == [TP_RESULT, SL_RESULT]
        assert walk.fold_metrics[position] == expected.metrics
        # Rule 4: the folds are not simulated for free - the C6 spread of EURUSD is charged.
        assert walk.fold_metrics[position]["spread_cost"] > 0


def test_the_aggregate_of_the_folds_is_their_mean_with_a_spread() -> None:
    """The aggregate is the mean of the folds and their σ, computed from the engine's tables (m2)."""
    walk = _run(_tape())
    first, second = (table["profit"] for table in walk.fold_metrics)

    assert first != second
    assert walk.aggregated == aggregate_fold_metrics(walk.fold_metrics)
    assert walk.aggregated["folds"] == 2
    assert walk.aggregated["trades_mean"] == 2.0
    assert walk.aggregated["trades_std"] == 0.0
    assert walk.aggregated["win_rate_mean"] == 50.0
    assert walk.aggregated["profit_mean"] == pytest.approx((first + second) / 2, abs=0.005)
    assert walk.aggregated["profit_std"] > 0
    # The mean is not the total: a sum would double it here (m2).
    assert walk.aggregated["profit_mean"] != pytest.approx(first + second, abs=0.005)


def test_no_fold_of_a_run_shares_a_bar_with_its_train_window() -> None:
    """The runner cuts the tape so that no train bar of a fold is simulated in it (m1)."""
    seen: list[tuple[set[int], set[int]]] = []

    def builder(train: pd.DataFrame, test: pd.DataFrame) -> list[TradeIntent]:
        seen.append((set(train.index), set(test.index)))
        return _plan(test)

    run_walkforward(_tape(), builder, SHORT, BacktestConfig())

    assert [len(test_bars) for _, test_bars in seen] == [WINDOW, WINDOW]
    for train_bars, test_bars in seen:
        assert train_bars & test_bars == set()
        assert min(test_bars) > max(train_bars)


def test_the_builder_may_not_arm_its_intents_on_the_train_window() -> None:
    """Intents built on the fit window are in no bar of the simulated one: the engine refuses them.

    This is why the builder of the runner is handed both windows (§7.10 п.65): intents anchored on
    the train window cannot be simulated on the test tape, and pretending otherwise would simulate
    bars the fold was allowed to fit on.
    """
    with pytest.raises(ValueError, match="in no bar of the tape"):
        run_walkforward(_tape(), lambda train, test: _plan(train), SHORT, BacktestConfig())


def test_a_tape_too_short_for_a_fold_is_refused() -> None:
    """A walk-forward without one fold is a wrong config: the runner raises, it does not lie."""
    too_short = WalkForwardConfig(min_train_bars=450, test_period_bars=100)

    with pytest.raises(ValueError, match="no fold"):
        run_walkforward(_tape(), lambda train, test: _plan(test), too_short, BacktestConfig())

