"""Engine tests: one order life cycle per intent, C6 costs, the C7 gates (Э5').

The tape is synthetic and small - M15 bars of a Tuesday from 08:00 UTC (11:00 MSK) - and the
numbers are EURUSD at 0.1 lot: a pip is $1, so the Alfa profile of ``BrokerSpec`` charges exactly
$1.40 of spread and $1.40 of commission (7.0 per lot over the two turns) on every trade, $0.20 of
slippage on each *market* leg and $0.70 of swap per long night; a 30 pip stop is 3 % of the C7
deposit and 0.1 lot of a 100 000 contract needs $275 of margin at the price of the intents.  Every
figure below can therefore be read by hand.

The four mutations the module is one line away from, and the test each one must break:

* m1 "fill on the signal bar" - the fill search starts at ``placed_bar`` instead of the next
  bar, which is the lookahead the layer exists to prevent; breaks
  :func:`test_a_limit_fills_on_the_next_bar_never_on_its_own`;
* m2 "check the target first" - the stop and the target of one bar swap places; breaks
  :func:`test_the_stop_wins_when_one_bar_touches_both_levels`;
* m3 "charge no spread" - the costs of the price list vanish from the log; breaks
  :func:`test_the_price_list_and_the_profile_are_charged_exactly`;
* m4 "drop the margin check" - an order is placed without the capital behind it; breaks
  :func:`test_the_margin_gate_measures_the_balance_of_the_intent_bar`.
"""

from __future__ import annotations

import pandas as pd
import pytest

from smc_zero.backtester.engine import (
    REASON_LIMIT_NOT_FILLED,
    REASON_NO_RESULT,
    REASON_OFF_HOURS,
    REASON_ORDER_BUDGET,
    REASON_SL_CAP,
    TRADE_COLUMNS,
    run_backtest,
)
from smc_zero.backtester.metrics import EOD_RESULT, SL_RESULT, TP_RESULT
from smc_zero.config import ALFAFOREX_SPECS, BacktestConfig, BrokerSpec, RiskConfig
from smc_zero.indicators.levels import PDH
from smc_zero.strategy.base import TradeIntent, intents_frame
from smc_zero.strategy.risk_gate import REASON_NO_MARGIN, RISK_REJECTION_COLUMNS

DAY = "2026-06-09"  # a Tuesday: Alfa is open all day (02:00 MSK .. 23:55 MSK)
START = pd.Timestamp(f"{DAY} 08:00", tz="UTC")
BAR = pd.Timedelta(minutes=15)
#: A quiet bar: the short limits of the tests (1.1000) and the long ones (1.0900) stay out.
FLAT = (1.0950, 1.0960, 1.0940, 1.0950)


def _tape(rows: list[tuple[float, float, float, float]], start: pd.Timestamp = START) -> pd.DataFrame:
    """Build a closed M15 tape out of ``(open, high, low, close)`` rows."""
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    frame.insert(0, "timestamp", [start + BAR * index for index in range(len(rows))])
    frame["volume"] = 1
    frame["is_closed"] = True
    return frame


def _intent(frame: pd.DataFrame, bar: int, **overrides: object) -> TradeIntent:
    """One accepted EURUSD short; only the fields a test varies are passed in."""
    fields: dict[str, object] = {
        "open_time": frame["timestamp"].iloc[bar],
        "bar": bar,
        "side": "short",
        "entry": 1.1000,
        "sl": 1.1030,
        "tp": 1.0940,
        "tp_source": "liquidity",
        "sl_source": "sweep_extreme",
        "sl_pips": 30.0,
        "rr": 2.0,
        "level_name": PDH,
        "level_date": pd.Timestamp(DAY),
        "level_price": 1.1030,
        "setup_type": "fresh",
        "sweep_bar": bar - 2,
        "choch_bar": bar - 1,
        "fvg_bar": bar,
    }
    fields.update(overrides)
    return TradeIntent(**fields)  # type: ignore[arg-type]


def _trade_rows() -> list[tuple[float, float, float, float]]:
    """The tape of one clean winning short: signal on bar 2, fill on 3, target on 4."""
    return [
        FLAT,
        FLAT,
        (1.0950, 1.1010, 1.0945, 1.0950),  # bar 2: the signal bar already trades the limit
        (1.0950, 1.1008, 1.0945, 1.0950),  # bar 3: the fill
        (1.0950, 1.0960, 1.0935, 1.0940),  # bar 4: the target at 1.0940
        FLAT,
    ]


def test_a_limit_fills_on_the_next_bar_never_on_its_own() -> None:
    """A limit is armed at the close of its signal bar and expected from the next one (m1)."""
    frame = _tape(_trade_rows())
    result = run_backtest(frame, [_intent(frame, 2)], BacktestConfig())

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert (trade["signal_bar"], trade["fill_bar"], trade["exit_bar"]) == (2, 3, 4)
    assert trade["limit_dur_min"] == 15
    assert trade["result"] == TP_RESULT
    assert trade["entry"] == pytest.approx(1.1000)
    assert trade["exit_price"] == pytest.approx(1.0940)
    assert result.rejections.empty


def test_the_stop_wins_when_one_bar_touches_both_levels() -> None:
    """The stop is looked at before the target, and the exit pays the market leg it is (m2)."""
    rows = [
        FLAT,
        FLAT,
        (1.0950, 1.1005, 1.0945, 1.0950),  # bar 2: the signal bar already trades the limit
        (1.0950, 1.1008, 1.0945, 1.0950),  # bar 3: the fill
        (1.0950, 1.1035, 1.0935, 1.0950),  # bar 4: one range covers both the stop and the target
        FLAT,
    ]
    frame = _tape(rows)
    result = run_backtest(frame, [_intent(frame, 2)], BacktestConfig())

    trade = result.trades.iloc[0]
    assert trade["result"] == SL_RESULT
    assert trade["exit_bar"] == 4
    assert trade["exit_price"] == pytest.approx(1.1030)  # the level the stop triggers at
    assert trade["slippage_cost"] == pytest.approx(0.2)  # one market leg of the Alfa profile
    # 30 pips of move, the spread of the round trip, the slippage of the exit and the commission.
    assert trade["profit"] == pytest.approx(-30.0 - 1.4 - 0.2 - 1.4)


def test_the_price_list_and_the_profile_are_charged_exactly() -> None:
    """The four charges of one trade, and the flat legacy commission migrated by money (m3)."""
    frame = _tape(_trade_rows())
    config = BacktestConfig(risk=RiskConfig(commission=0.35))  # the Э5' keyword: per trade
    result = run_backtest(frame, [_intent(frame, 2)], config)

    trade = result.trades.iloc[0]
    assert trade["spread_cost"] == pytest.approx(1.4)  # 1.4 pip at $1 a pip
    assert trade["slippage_cost"] == 0.0  # a target is a resting limit, not a market order
    assert trade["commission"] == pytest.approx(0.35)
    assert trade["swap_cost"] == 0.0
    assert trade["days_held"] == 0
    # A flat 0.35 a trade is the same money as 1.75 per lot over two turns, so the migration of
    # the old call site does not move a cent; the target is 60 pips away and the money is 60 -
    # the spread - the commission.
    assert result.config.risk.broker.commission_per_lot_usd == pytest.approx(1.75)
    assert trade["profit"] == pytest.approx(60.0 - 1.4 - 0.35)
    assert result.metrics["spread_cost"] == pytest.approx(1.4)
    assert result.metrics["profit"] == pytest.approx(58.25)
    assert result.metrics["final_balance"] == pytest.approx(10_058.25)


def test_the_run_refuses_a_row_priced_like_another_symbol() -> None:
    """The run charges one profile for one symbol, so the row and the profile have to agree (Э10')."""
    frame = _tape(_trade_rows())
    intent = _intent(frame, 2)

    # The account-level numbers are the run's own: a commission the run wants is no mismatch.
    own = BacktestConfig(risk=RiskConfig(broker=BrokerSpec(commission_per_lot_usd=3.5)))
    assert len(run_backtest(frame, [intent], own).trades) == 1

    # The symbol-level ones are not: the engine charges the profile of the run, so a GBPUSD row
    # under the EURUSD profile would label a trade one way and price it another.
    other = BacktestConfig(risk=RiskConfig(broker=ALFAFOREX_SPECS["GBPUSD"].broker))
    with pytest.raises(ValueError, match="the price list row of EURUSD"):
        run_backtest(frame, [intent], other)
    with pytest.raises(ValueError, match="spread_pip: row 1.4 != profile 2.1"):
        run_backtest(frame, [intent], other)

    # Naming the row of the run explicitly is the other way out of the mismatch.
    assert len(run_backtest(frame, [intent], other, ALFAFOREX_SPECS["GBPUSD"]).trades) == 1



def test_the_margin_gate_measures_the_balance_of_the_intent_bar() -> None:
    """C7's margin is asked with the running equity of the bar, before the order is placed (m4)."""
    frame = _tape(_trade_rows())
    intent = _intent(frame, 2)

    poor = run_backtest(frame, [intent], BacktestConfig(initial_capital=250.0))
    assert poor.trades.empty
    row = poor.rejections.iloc[0]
    assert row["reason"] == REASON_NO_MARGIN
    assert row["required_margin"] == pytest.approx(275.0)  # 0.1 lot at 1.1000, leverage 40
    assert row["equity"] == pytest.approx(250.0)
    assert row["risk_pct"] == pytest.approx(3.0)  # 30 pips against the 1000 deposit of C7
    assert bool(row["risk_warning"]) is True

    rich = run_backtest(frame, [intent], BacktestConfig(initial_capital=300.0))
    assert rich.rejections.empty
    assert len(rich.trades) == 1


def test_every_intent_ends_as_a_trade_or_as_a_ledger_row() -> None:
    """Both intents of a bar are placed, and every intent lands in the log or in the ledger."""
    rows = [
        FLAT,
        FLAT,
        (1.0950, 1.1005, 1.0945, 1.0950),  # bar 2: the signal of the winner and of the never-filled
        (1.0950, 1.1008, 1.0945, 1.0950),  # bar 3: the fill of the winner
        (1.0950, 1.0960, 1.0935, 1.0940),  # bar 4: its target
        FLAT,
        FLAT,
        FLAT,
    ]
    frame = _tape(rows)
    intents = [
        _intent(frame, 2),  # fills and wins
        _intent(frame, 2, entry=1.2000, sl=1.2030, tp=1.1940),  # placed too, never touched
        _intent(frame, 4),  # the day has spent its budget
        _intent(frame, 5),  # and stays spent
    ]
    result = run_backtest(frame, intents, BacktestConfig(max_orders_day=2))

    assert len(result.trades) == 1
    assert len(result.trades) + len(result.rejections) == len(intents)
    assert tuple(result.trades.columns) == TRADE_COLUMNS
    assert tuple(result.rejections.columns) == RISK_REJECTION_COLUMNS
    # The tail of the run has no bar of its own: an order left at the end is written there.
    assert sorted(zip(result.rejections["reason"], result.rejections["bar"], strict=True)) == [
        (REASON_LIMIT_NOT_FILLED, 7),
        (REASON_ORDER_BUDGET, 4),
        (REASON_ORDER_BUDGET, 5),
    ]


def test_off_hours_intents_are_never_placed() -> None:
    """Alfa opens Monday 02:00 MSK, so the intents of the opening hour carry no order."""
    start = pd.Timestamp("2026-06-07 21:45", tz="UTC")  # Sunday 21:45 UTC = Monday 00:45 MSK
    rows = [
        FLAT,                                    # 0: 00:45 MSK - shut
        FLAT,                                    # 1: 01:00 MSK - shut
        (1.0950, 1.1005, 1.0945, 1.0950),        # 2: 01:15 MSK - shut, its limit is touched
        (1.0950, 1.1008, 1.0945, 1.0950),        # 3: 01:30 MSK - shut, and touched again
        (1.0950, 1.1008, 1.0945, 1.0950),        # 4: 01:45 MSK - shut
        FLAT,                                    # 5: 02:00 MSK - the broker opens
        (1.0950, 1.1005, 1.0945, 1.0950),        # 6: 02:15 MSK - the live limit fills here
        (1.0950, 1.0960, 1.0935, 1.0940),        # 7: 02:30 MSK - the target
    ]
    frame = _tape(rows, start=start)
    result = run_backtest(frame, [_intent(frame, 2), _intent(frame, 5)], BacktestConfig())

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert (trade["signal_bar"], trade["fill_bar"], trade["exit_bar"]) == (5, 6, 7)
    assert list(result.rejections["reason"]) == [REASON_OFF_HOURS]
    assert list(result.rejections["bar"]) == [2]


def test_a_fill_needs_a_tradable_bar() -> None:
    """A limit touched only after the Friday close of 23:55 MSK is never filled."""
    start = pd.Timestamp("2026-06-05 20:15", tz="UTC")  # Friday 20:15 UTC = Friday 23:15 MSK
    rows = [
        FLAT,                                    # 0: 23:15 MSK - open, the signal of both limits
        (1.0950, 1.1005, 1.0945, 1.0950),        # 1: 23:30 MSK - the first limit fills here
        (1.0950, 1.1005, 1.0935, 1.0950),        # 2: 23:45 MSK - and reaches its target
        (1.0950, 1.1010, 1.0945, 1.0950),        # 3: 00:00 MSK - Saturday, the broker is shut
        (1.0950, 1.1010, 1.0945, 1.0950),        # 4: 00:15 MSK - shut, and the window closes
    ]
    frame = _tape(rows, start=start)
    result = run_backtest(
        frame,
        [_intent(frame, 0), _intent(frame, 0, entry=1.1006, sl=1.1036, tp=1.0946)],
        BacktestConfig(limit_valid_bars=3),
    )

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert (trade["signal_bar"], trade["fill_bar"], trade["exit_bar"]) == (0, 1, 2)
    assert trade["result"] == TP_RESULT
    assert list(result.rejections["reason"]) == [REASON_LIMIT_NOT_FILLED]
    assert list(result.rejections["bar"]) == [4]


def test_the_day_budget_and_the_stop_cap_gate_the_placement() -> None:
    """The budget is spent by *placed* orders and a stopped day stops (prod's two counters)."""
    rows = [
        FLAT,
        FLAT,
        (1.0950, 1.1005, 1.0945, 1.0950),  # 2: the signal of the first order
        (1.0950, 1.1008, 1.0945, 1.0950),  # 3: its fill
        (1.0950, 1.1035, 1.0945, 1.0950),  # 4: its stop
        FLAT,
        FLAT,                              # 6: the signal of the second order
        (1.0950, 1.1008, 1.0945, 1.0950),  # 7: its fill
        (1.0950, 1.0960, 1.0935, 1.0940),  # 8: its target
    ]
    frame = _tape(rows)
    intents = [_intent(frame, 2), _intent(frame, 6)]

    capped = run_backtest(frame, intents, BacktestConfig(max_sl_per_day=1))
    assert len(capped.trades) == 1
    assert capped.trades.iloc[0]["result"] == SL_RESULT
    assert list(capped.rejections["reason"]) == [REASON_SL_CAP]

    uncapped = run_backtest(frame, intents, BacktestConfig(max_sl_per_day=0))
    assert len(uncapped.trades) == 2
    assert uncapped.rejections.empty

    budget = run_backtest(frame, intents, BacktestConfig(max_orders_day=1))
    assert len(budget.trades) == 1
    assert list(budget.rejections["reason"]) == [REASON_ORDER_BUDGET]


def test_the_eod_switch_closes_the_last_bar_of_the_msk_day() -> None:
    """``force_close_eod`` books an open trade at the last close of its MSK day (prod's branch)."""
    start = pd.Timestamp(f"{DAY} 20:15", tz="UTC")  # 23:15 MSK: three bars to midnight MSK
    frame = _tape(
        [
            FLAT,                              # 0: 23:15 MSK - the signal
            (1.0950, 1.1005, 1.0945, 1.0950),  # 1: 23:30 MSK - the fill
            FLAT,                              # 2: 23:45 MSK - the last bar of the MSK day
            FLAT,                              # 3: 00:00 MSK - a new MSK day
        ],
        start=start,
    )
    # A stop and a target out of reach of the quiet bars, so only the EOD branch can close it.
    wide = _intent(frame, 0, sl=1.1100, tp=1.0900, sl_pips=100.0)
    closed = run_backtest(frame, [wide], BacktestConfig(force_close_eod=True))

    trade = closed.trades.iloc[0]
    assert trade["result"] == EOD_RESULT
    assert trade["exit_bar"] == 2
    assert trade["exit_price"] == pytest.approx(frame["close"].iloc[2])
    assert trade["days_held"] == 0
    assert trade["swap_cost"] == 0.0
    assert closed.metrics["eod_cnt"] == 1

    carried = run_backtest(frame, [wide], BacktestConfig())
    assert carried.trades.empty
    assert list(carried.rejections["reason"]) == [REASON_NO_RESULT]


def test_the_two_windows_stay_prod_bar_counts() -> None:
    """``limit_valid_bars`` expires a limit, ``max_bars_per_trade`` drops a running trade."""
    frame = _tape(_trade_rows())

    expired = run_backtest(frame, [_intent(frame, 2)], BacktestConfig(limit_valid_bars=0))
    assert expired.trades.empty
    assert list(expired.rejections["reason"]) == [REASON_LIMIT_NOT_FILLED]

    # The trade fills on bar 3 and its target sits on bar 4: a window of one bar ends first.
    dropped = run_backtest(frame, [_intent(frame, 2)], BacktestConfig(max_bars_per_trade=1))
    assert dropped.trades.empty
    assert list(dropped.rejections["bar"]) == [4]
    assert list(dropped.rejections["reason"]) == [REASON_NO_RESULT]


def test_the_unclosed_tail_of_the_tape_is_not_simulated() -> None:
    """Rule 2b: the still-forming last bar is dropped before anything is simulated."""
    rows = [FLAT, FLAT, (1.0950, 1.1005, 1.0945, 1.0950), (1.0950, 1.1008, 1.0945, 1.0950)]
    rows += [(1.0950, 1.0960, 1.0955, 1.0950), (1.0950, 1.0960, 1.0930, 1.0940)]
    frame = _tape(rows)
    frame.loc[frame.index[-1], "is_closed"] = False

    dropped = run_backtest(frame, [_intent(frame, 2)], BacktestConfig())
    assert dropped.trades.empty
    assert len(dropped.equity) == len(frame) - 1
    assert list(dropped.rejections["reason"]) == [REASON_NO_RESULT]

    kept = run_backtest(frame, [_intent(frame, 2)], BacktestConfig(drop_unclosed=False))
    assert kept.trades.iloc[0]["exit_bar"] == 5
    assert kept.trades.iloc[0]["result"] == TP_RESULT

    # A hand-built tape without the flag follows the same rule: its last bar is presumed open.
    unflagged = _tape(rows).drop(columns=["is_closed"])
    same = run_backtest(unflagged, [_intent(unflagged, 2)], BacktestConfig())
    pd.testing.assert_frame_equal(same.trades, dropped.trades)


def test_an_intents_frame_is_accepted_like_a_sequence() -> None:
    """The batch form of the strategy layer runs to the same trades as the sequence form."""
    frame = _tape(_trade_rows())
    intents = [_intent(frame, 2)]

    from_frame = run_backtest(frame, intents_frame(intents), BacktestConfig())
    from_sequence = run_backtest(frame, intents, BacktestConfig())

    pd.testing.assert_frame_equal(from_frame.trades, from_sequence.trades)
    pd.testing.assert_series_equal(from_frame.equity, from_sequence.equity)
    assert from_frame.metrics == from_sequence.metrics


def test_the_intents_must_be_aligned_with_the_tape() -> None:
    """A bar index that does not match the tape is a hard error, not a silent misalignment."""
    frame = _tape(_trade_rows())
    shifted = _tape(_trade_rows(), start=START - BAR)
    misaligned = _intent(shifted, 2)  # its stamp is bar 1 of the tape the engine gets
    foreign = _intent(frame, 2, open_time=pd.Timestamp("2030-01-01 08:00", tz="UTC"))

    with pytest.raises(ValueError, match="not aligned with the tape"):
        run_backtest(frame, [misaligned], BacktestConfig())
    with pytest.raises(ValueError, match="no bar of the tape"):
        run_backtest(frame, [foreign], BacktestConfig())


def test_the_swap_is_charged_per_msk_night_held() -> None:
    """``days_held`` counts MSK midnights - the broker's rollover, one calendar for the layer."""
    start = pd.Timestamp(f"{DAY} 20:30", tz="UTC")  # 23:30 MSK Tuesday
    frame = _tape(
        [
            (1.0970, 1.0975, 1.0955, 1.0970),  # 0: 23:30 MSK - the signal of a long limit
            (1.0970, 1.0975, 1.0940, 1.0950),  # 1: 23:45 MSK - the fill
            (1.0960, 1.0975, 1.0945, 1.0970),  # 2: 00:00 MSK Wednesday - a new MSK day
            (1.0970, 1.1005, 1.0960, 1.1000),  # 3: 00:15 MSK - the target
        ],
        start=start,
    )
    long_intent = _intent(frame, 0, side="long", entry=1.0950, sl=1.0930, tp=1.1000, sl_pips=20.0)
    trade = run_backtest(frame, [long_intent], BacktestConfig()).trades.iloc[0]

    assert trade["days_held"] == 1
    assert trade["swap_cost"] == pytest.approx(-0.7)  # the long leg of the EURUSD price list
    # 50 pips of move, the spread, the commission and one night of the long leg: the target is a
    # resting limit and pays no slippage.
    assert trade["profit"] == pytest.approx(50.0 - 1.4 - 0.7 - 1.4)

    # A night is an MSK date change, not a UTC one: a trade of one MSK day pays nothing, which
    # is where the engine deliberately leaves prod's UTC arithmetic (§7.9).
    late = pd.Timestamp(f"{DAY} 23:00", tz="UTC")  # Wednesday 23:00 UTC = Thursday 02:00 MSK
    quiet = (1.0970, 1.0975, 1.0955, 1.0970)
    late_frame = _tape(
        [
            quiet,
            quiet,                              # 02:15 MSK
            quiet,                              # 02:30 MSK
            (1.0960, 1.0965, 1.0940, 1.0950),  # 02:45 MSK - the fill
            quiet,                              # 03:00 MSK - the UTC date has changed by now
            quiet,                              # 03:15 MSK
            (1.0960, 1.1005, 1.0955, 1.1000),  # 03:30 MSK - the target, the same MSK day
        ],
        start=late,
    )
    late_intent = _intent(
        late_frame, 0, side="long", entry=1.0950, sl=1.0930, tp=1.1000, sl_pips=20.0
    )
    late_trade = run_backtest(late_frame, [late_intent], BacktestConfig()).trades.iloc[0]
    assert late_trade["days_held"] == 0
    assert late_trade["swap_cost"] == 0.0


def test_a_later_bar_cannot_change_an_earlier_trade() -> None:
    """Rule 2: a bar after the exit of a trade leaves its row and the curve up to it alone."""
    quiet = _trade_rows()
    wild = [*quiet[:5], (1.0950, 1.5000, 1.0000, 1.4000)]
    plain_frame, wild_frame = _tape(quiet), _tape(wild)
    plain = run_backtest(plain_frame, [_intent(plain_frame, 2)], BacktestConfig())
    shocked = run_backtest(wild_frame, [_intent(wild_frame, 2)], BacktestConfig())

    assert plain.trades.iloc[0]["exit_bar"] == shocked.trades.iloc[0]["exit_bar"] == 4
    assert plain.trades.iloc[0]["exit_price"] == pytest.approx(1.0940)
    assert shocked.trades.iloc[0]["exit_price"] == pytest.approx(1.0940)
    assert plain.trades.iloc[0]["result"] == shocked.trades.iloc[0]["result"] == TP_RESULT
    pd.testing.assert_series_equal(plain.equity.iloc[:5], shocked.equity.iloc[:5])
    assert plain.metrics["profit"] == shocked.metrics["profit"]


def test_the_equity_curve_is_flat_until_the_trade_closes() -> None:
    """One value per bar, indexed by the bar's close stamp, flat while nothing is closed."""
    frame = _tape(_trade_rows())
    result = run_backtest(frame, [_intent(frame, 2)], BacktestConfig())

    assert result.equity.name == "equity"
    assert list(result.equity.index) == list(frame["timestamp"] + BAR)
    assert result.equity.iloc[0] == pytest.approx(10_000.0)
    assert result.equity.iloc[:4].nunique() == 1  # the trade is still open
    assert result.equity.iloc[4] == pytest.approx(10_057.2)  # 60 pips - the spread - the commission
    assert result.equity.iloc[-1] == pytest.approx(result.metrics["final_balance"])
    assert result.config.initial_capital == 10_000.0


def test_the_slippage_is_charged_on_the_market_legs_only() -> None:
    """A target is a resting limit and pays no slippage; a stop and an EOD close pay one leg.

    Э10' books the slippage as *money* of its own column instead of a worse price, so the log
    keeps both the level the exit happened at and what that exit cost.
    """
    config = BacktestConfig(risk=RiskConfig(slippage=0.0002))  # the Э5' price unit: 2 pips
    frame = _tape(_trade_rows())
    winner = run_backtest(frame, [_intent(frame, 2)], config).trades.iloc[0]
    assert winner["result"] == TP_RESULT
    assert winner["exit_price"] == pytest.approx(1.0940)
    assert winner["slippage_cost"] == 0.0

    stopped_frame = _tape(
        [
            FLAT,
            FLAT,
            (1.0950, 1.1005, 1.0945, 1.0950),
            (1.0950, 1.1008, 1.0945, 1.0950),
            (1.0950, 1.1035, 1.0945, 1.0950),  # the stop, and beyond it the slippage
            FLAT,
        ]
    )
    stopped = run_backtest(stopped_frame, [_intent(stopped_frame, 2)], config).trades.iloc[0]
    assert stopped["result"] == SL_RESULT
    assert stopped["exit_price"] == pytest.approx(1.1030)  # the trigger, not the fill
    assert stopped["slippage_cost"] == pytest.approx(2.0)  # 2 pips on the one market leg
    assert stopped["profit"] == pytest.approx(-30.0 - 1.4 - 2.0 - 1.4)

    start = pd.Timestamp(f"{DAY} 20:30", tz="UTC")
    eod_frame = _tape([FLAT, (1.0950, 1.1005, 1.0945, 1.0950)], start=start)
    wide = _intent(eod_frame, 0, sl=1.1100, tp=1.0900, sl_pips=100.0)
    closed = run_backtest(
        eod_frame,
        [wide],
        BacktestConfig(force_close_eod=True, risk=RiskConfig(slippage=0.0002)),
    ).trades.iloc[0]
    assert closed["result"] == EOD_RESULT
    assert closed["exit_price"] == pytest.approx(eod_frame["close"].iloc[1])
    assert closed["slippage_cost"] == pytest.approx(2.0)


def test_an_empty_tape_or_an_empty_intent_list_is_a_flat_run() -> None:
    """Nothing to simulate is a valid run: an empty log, an empty ledger, a flat curve."""
    empty = run_backtest(_tape([]), [], BacktestConfig())
    assert empty.trades.empty
    assert empty.rejections.empty
    assert len(empty.equity) == 0
    assert empty.metrics["trades"] == 0
    assert empty.metrics["final_balance"] == pytest.approx(10_000.0)

    frame = _tape(_trade_rows())
    flat = run_backtest(frame, [], BacktestConfig())
    assert flat.trades.empty
    assert flat.rejections.empty
    assert len(flat.equity) == len(frame)
    assert flat.equity.nunique() == 1
    assert flat.metrics["return_pct"] == 0.0
