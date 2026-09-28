"""Event-driven backtest engine: one order life cycle per intent, prod's costs (Э5').

The engine walks the *closed* M15 tape once, bar by bar, and keeps the account state in
front of it: an intent becomes a pending limit at the close of its own bar, the limit fills
on a *later* bar, and the trade is booked on the bar that resolves it (SPEC_SMC.md §7.9).
That single chronological pass is what makes the layer honest.  prod simulated each trade to
its end inside the signal bar and booked the whole profit there, so its ``orders_today`` /
``sl_today`` counters and its balance knew the future; here the counters and the equity only
ever see events that already happened.

Four phases run on every bar, in this order:

1. **resolution** - open trades are checked against the bar with the stop looked at first
   (prod's priority: a bar that touches both levels is an SL), and a trade is *never*
   resolved on its own fill bar (prod starts its scan at ``fill_idx + 1``); the window of
   ``BacktestConfig.max_bars_per_trade`` bars closes the book of a trade that never resolved;
2. **fills** - the pending limits of *earlier* bars fill when the bar trades through the
   limit price (a short limit above the market fills on ``high >= entry``, a long one below
   it on ``low <= entry``) and expire after ``limit_valid_bars`` bars (prod's window);
3. **placement** - the intents of this bar pass four gates in a fixed order: Alfa's trading
   hours, the day's order budget, the day's stop-out cap, then the C7 margin check; every
   refusal lands in the ledger with its reason, and the day's budget is spent by *placed*
   orders (prod's counter);
4. **end of day** - with ``BacktestConfig.force_close_eod`` whatever is open is closed at the
   last close of its MSK day (prod's EOD branch), otherwise it is carried across the date
   change and charged one night of swap per MSK day.

Everything happens on tradable bars: the broker week of
:func:`smc_zero.indicators.sessions.alfa_trading_mask` gates placement, fills and resolution
alike, so no order is created and no level is hit while Alfa is shut.  The *lengths* of the
two windows stay prod's raw-bar counts.  The day is the *MSK* day (the project clock of rule
2b), which is also what ``days_held`` and the swap are counted in: one calendar, one reading.

Money is the price list's (C6): the spread of the instrument is charged once per trade, its
swap is signed and paid per night held (``days_held``), ``RiskConfig.commission`` is one fee
per trade and ``RiskConfig.slippage`` worsens the *market* exits (a stop and an EOD close) -
a limit fill keeps its limit price, so the entry is never slipped.  ``RiskConfig.spread``
stays unused on purpose: C6 moves the spread into the instrument row, and charging both
would double it (§7.9).  ``risk_pct`` is not recomputed here either: every intent is asked
:func:`smc_zero.strategy.risk_gate.check_intent` with the equity of *its own* bar, so the
margin gate and the reported percentage keep one implementation.

:func:`run_backtest` returns a :class:`BacktestResult`: the trade log (:data:`TRADE_COLUMNS`),
the per-bar equity curve indexed by the bar's close stamp, the ledger of everything that did
*not* become a trade and the metric table of :mod:`smc_zero.backtester.metrics`.  Every
intent ends either as a trade or as a ledger row - a silent refusal is a defect of this
layer, and ``len(trades) + len(rejections)`` is the number of intents handed in.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import NamedTuple

import numpy as np
import pandas as pd

from smc_zero.backtester.metrics import EOD_RESULT, SL_RESULT, TP_RESULT, calc_metrics
from smc_zero.config import ALFAFOREX_SPECS, BacktestConfig, InstrumentSpec, Side
from smc_zero.data_loader import (
    CLOSE_TIME_COLUMN,
    IS_CLOSED_COLUMN,
    PRICE_COLUMNS,
    TIMESTAMP_COLUMN,
    attach_close_time,
    drop_unclosed,
    mark_closed,
)
from smc_zero.indicators.sessions import alfa_trading_mask
from smc_zero.strategy.base import TradeIntent, intents_frame, intents_from_frame
from smc_zero.strategy.risk_gate import (
    REASON_NO_MARGIN,
    RISK_REJECTION_COLUMNS,
    check_intent,
    margin_of,
)
from smc_zero.utils.time import MSK_UTC_OFFSET_HOURS

#: The trade log of the engine - what a completed order became.  The stamps are the *open*
#: stamps of the bars involved (``open_time`` the signal, ``fill_time`` the fill, ``exit_time``
#: the exit), so ``limit_dur_min`` and ``bars_held`` are plain differences of the tape's own
#: labels; ``close_time`` is not reused as a column name here, because in the loader schema it
#: means the bar's *close* instant, not the bar a trade closed on.
TRADE_COLUMNS: tuple[str, ...] = (
    "signal_bar",
    "open_time",
    "fill_bar",
    "fill_time",
    "exit_bar",
    "exit_time",
    "bars_held",
    "days_held",
    "limit_dur_min",
    "level_name",
    "level_date",
    "level_price",
    "setup_type",
    "side",
    "entry",
    "sl",
    "tp",
    "rr",
    "sl_pips",
    "tp_pips",
    "sl_source",
    "tp_source",
    "sweep_bar",
    "choch_bar",
    "fvg_bar",
    "exit_price",
    "result",
    "profit",
    "spread_cost",
    "swap_cost",
    "commission",
    "risk_pct",
    "risk_warning",
)

#: Dtypes of :data:`TRADE_COLUMNS` that are not obvious from the data (the ints and the
#: stamps), so an empty log keeps the contract of a filled one - the style of
#: :data:`smc_zero.strategy.base._INTENT_DTYPES`.  ``level_date`` stays naive: it is the
#: trading *day* a level belongs to, exactly as the intents frame carries it.
_TRADE_DTYPES: Mapping[str, str] = {
    "signal_bar": "int64",
    "open_time": "datetime64[ns, UTC]",
    "fill_bar": "int64",
    "fill_time": "datetime64[ns, UTC]",
    "exit_bar": "int64",
    "exit_time": "datetime64[ns, UTC]",
    "bars_held": "int64",
    "days_held": "int64",
    "limit_dur_min": "int64",
    "risk_warning": "bool",
}

#: The reasons of the engine's own ledger, one per gate that can stop an order.  The margin
#: verdict is the risk gate's own (``REASON_NO_MARGIN``) and is not duplicated here.
REASON_OFF_HOURS = "off_hours"
REASON_ORDER_BUDGET = "order_budget"
REASON_SL_CAP = "sl_cap"
#: Life-cycle reasons of an order that *was* placed but never became a trade: the limit ran
#: out of ``limit_valid_bars``, or the trade never resolved inside ``max_bars_per_trade``.
REASON_LIMIT_NOT_FILLED = "limit_not_filled"
REASON_NO_RESULT = "no_result"

#: The price list row the engine charges with by default: the EURUSD fees of the Alfa table.
DEFAULT_INSTRUMENT: InstrumentSpec = ALFAFOREX_SPECS["EURUSD"]


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """One simulated run: its trade log, its equity curve, its ledger and its numbers.

    ``trades`` carries :data:`TRADE_COLUMNS` and ``rejections`` the columns of
    :data:`smc_zero.strategy.risk_gate.RISK_REJECTION_COLUMNS` (the same vocabulary as the
    batch risk gate, so a refused order reads the same wherever it was refused).  ``equity``
    is the per-bar curve indexed by the bar's ``close_time`` - the instant the bar became a
    fact - and starts at ``BacktestConfig.initial_capital``; it is flat between trades,
    because a trade changes the balance when it is closed, not while it is open.
    """

    trades: pd.DataFrame
    equity: pd.Series
    rejections: pd.DataFrame
    metrics: dict[str, float]
    config: BacktestConfig
    instrument: InstrumentSpec


class _Order(NamedTuple):
    """A limit the engine placed: the intent, its bar and the placement verdict."""

    intent: TradeIntent
    placed_bar: int
    risk_pct: float
    risk_warning: bool


class _OpenTrade(NamedTuple):
    """A filled order: the position and the bar it filled on."""

    order: _Order
    fill_bar: int


class _Run(NamedTuple):
    """The immutable context of one run: the simulated tape, its config, its price list."""

    stamps: pd.Series
    cfg: BacktestConfig
    instrument: InstrumentSpec


def _msk_days(stamps: pd.Series) -> np.ndarray:
    """Return the MSK calendar day of every stamp (rule 2b: MSK is the project clock)."""
    return np.asarray((stamps + pd.Timedelta(hours=MSK_UTC_OFFSET_HOURS)).dt.date, dtype=object)


def _nights_between(fill_day: date, exit_day: date) -> int:
    """Return the nights a position was held: the difference of its MSK calendar days.

    The broker's rollover is midnight MSK, so this is the number of swap legs the price list
    charges (prod counted the same difference on *UTC* dates of the data provider - the one
    reading Э5' corrects, see §7.9).  A same-day round trip is charged nothing.
    """
    return max((exit_day - fill_day).days, 0)


def _minutes_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Return the age of the limit in whole minutes (prod's ``limit_dur_min``)."""
    return int((end - start) / pd.Timedelta(minutes=1))


def _limit_touched(intent: TradeIntent, high: float, low: float) -> bool:
    """Return whether the bar trades through the pending limit of the intent.

    A short waits above the market, so its limit is touched when the high reaches it; a long
    waits below and is touched when the low does.  The fill price stays the limit price even
    when the bar gaps through it - prod fills there too, and the entry of a limit is the one
    price of the trade that is not slipped (see the module docstring).
    """
    if intent.side == "short":
        return bool(high >= intent.entry)
    return bool(low <= intent.entry)


def _exit_at(
    side: Side,
    high: float,
    low: float,
    sl: float,
    tp: float,
    slippage: float,
) -> tuple[int, float] | None:
    """Return ``(result, exit price)`` of the bar, or ``None`` when neither level is hit.

    The stop is looked at first - prod's own order - so a bar whose range covers both levels
    is booked as a stop, the conservative reading.  The stop is a market order once it is
    touched, so its price is worsened by the profile's ``slippage``; the target keeps its
    price, because a take profit is a resting limit like the entry.
    """
    if side == "short":
        if high >= sl:
            return SL_RESULT, sl + slippage
        if low <= tp:
            return TP_RESULT, tp
        return None
    if low <= sl:
        return SL_RESULT, sl - slippage
    if high >= tp:
        return TP_RESULT, tp
    return None


def _profit_of(intent: TradeIntent, exit_price: float, days_held: int, run: _Run) -> float:
    """Return the money a closed trade moved: prod's ``compute_profit``, priced by C6.

    The sign follows the side, so the same three lines price a target, a stop and an EOD
    close; the spread is paid once per round trip, the swap is signed (a negative leg credits
    the account) and the commission of the profile is one fee per trade.  The engine's balance
    and the ``profit`` column of the log go through here, so the two can never disagree.
    """
    instrument = run.instrument
    lot = run.cfg.risk.lot
    move = exit_price - intent.entry if intent.side == "long" else intent.entry - exit_price
    return (
        instrument.money(move, lot)
        - instrument.money(instrument.spread_abs, lot)
        + instrument.money(instrument.swap_abs(intent.side, days_held), lot)
        - run.cfg.risk.commission
    )


def _trade_row(
    trade: _OpenTrade,
    exit_bar: int,
    result: int,
    exit_price: float,
    days_held: int,
    run: _Run,
) -> dict[str, object]:
    """Build one row of the trade log: the geometry, the exit and the money it moved.

    The money is prod's ``compute_profit`` with the price list of C6 in place of its
    constants: the signed move of the trade priced by the lot, less the spread once, plus the
    signed swap of the nights held, less the profile's commission.  ``exit_price`` already
    carries the slippage of a market exit, so one formula covers the target, the stop and an
    EOD close (prod branched on the result to reconstruct what the exit prices already say).
    """
    order = trade.order
    intent = order.intent
    instrument = run.instrument
    lot = run.cfg.risk.lot
    spread_cost = instrument.money(instrument.spread_abs, lot)
    swap_cost = instrument.money(instrument.swap_abs(intent.side, days_held), lot)
    return {
        "signal_bar": intent.bar,
        "open_time": intent.open_time,
        "fill_bar": trade.fill_bar,
        "fill_time": run.stamps.iloc[trade.fill_bar],
        "exit_bar": exit_bar,
        "exit_time": run.stamps.iloc[exit_bar],
        "bars_held": exit_bar - trade.fill_bar,
        "days_held": days_held,
        "limit_dur_min": _minutes_between(intent.open_time, run.stamps.iloc[trade.fill_bar]),
        "level_name": intent.level_name,
        "level_date": intent.level_date,
        "level_price": intent.level_price,
        "setup_type": intent.setup_type,
        "side": intent.side,
        "entry": intent.entry,
        "sl": intent.sl,
        "tp": intent.tp,
        "rr": intent.rr,
        "sl_pips": intent.sl_pips,
        "tp_pips": round(abs(intent.tp - intent.entry) / instrument.pip_size, 1),
        "sl_source": intent.sl_source,
        "tp_source": intent.tp_source,
        "sweep_bar": intent.sweep_bar,
        "choch_bar": intent.choch_bar,
        "fvg_bar": intent.fvg_bar,
        "exit_price": exit_price,
        "result": result,
        "profit": _profit_of(intent, exit_price, days_held, run),
        "spread_cost": spread_cost,
        "swap_cost": swap_cost,
        "commission": run.cfg.risk.commission,
        "risk_pct": order.risk_pct,
        "risk_warning": order.risk_warning,
    }


def _ledger_row(
    intent: TradeIntent,
    bar: int,
    equity: float,
    reason: str,
    risk_pct: float,
    risk_warning: bool,
    cfg: BacktestConfig,
) -> dict[str, object]:
    """Build one ledger row: the intent, its margin arithmetic and why it is not a trade.

    The columns are the ones of the batch risk gate (:data:`RISK_REJECTION_COLUMNS`), so a
    refusal reads the same whether the batch gate or the engine produced it; ``equity`` is the
    balance of the moment, which is what the margin check measured against.
    """
    return {
        "bar": bar,
        "open_time": intent.open_time,
        "level_name": intent.level_name,
        "side": intent.side,
        "entry": intent.entry,
        "required_margin": margin_of(intent, cfg.risk),
        "equity": equity,
        "risk_pct": risk_pct,
        "risk_warning": risk_warning,
        "reason": reason,
    }


def _entry_tape(tape: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """Return the bars the engine may simulate: closed, sorted, UTC and stamped to close.

    Rule 2b: the still-forming tail of the tape is never simulated.  The flags of
    :func:`smc_zero.data_loader.mark_closed` are taken when the caller brought them and
    re-derived (last bar unclosed) when it did not, so a frame straight out of
    :func:`smc_zero.data_loader.load_csv` and a hand-built one follow one rule.  ``timestamp``
    is the bar's *open* stamp and the added ``close_time = timestamp + ltf`` is the instant the
    bar became a fact - the equity curve of the run is indexed by it.
    """
    missing = [name for name in (TIMESTAMP_COLUMN, *PRICE_COLUMNS) if name not in tape.columns]
    if missing:
        raise ValueError(f"the M15 tape needs the {missing} column(s)")
    frame = tape.copy()
    if cfg.drop_unclosed:
        if IS_CLOSED_COLUMN not in frame.columns:
            frame = mark_closed(frame)
        frame = drop_unclosed(frame)
    frame = frame.sort_values(TIMESTAMP_COLUMN, kind="stable").reset_index(drop=True)
    frame = attach_close_time(frame, cfg.timeframes.ltf)
    if not frame[TIMESTAMP_COLUMN].is_unique:
        raise ValueError("the tape repeats a stamp; one bar per stamp is required")
    return frame


def _intents_by_bar(
    intents: pd.DataFrame | Iterable[TradeIntent],
    frame: pd.DataFrame,
) -> dict[int, list[TradeIntent]]:
    """Return the intents of the run grouped by the bar they were armed on.

    A :class:`~smc_zero.strategy.base.TradeIntent` sequence (what the entry chain returns) and
    an intents frame are both accepted.  The ``bar`` of every intent is *re-derived* from its
    ``open_time`` against the tape the engine is about to simulate and compared with the index
    the intent carries, so intents built on a different frame (a shifted tape, a bar the engine
    dropped as unclosed) raise instead of being simulated on the wrong bar; the same error
    covers an ``open_time`` that is in no bar of the tape.  Grouping keeps the order the caller
    handed the intents in inside one bar.
    """
    rows = intents if isinstance(intents, pd.DataFrame) else intents_frame(intents)
    objects = intents_from_frame(rows, source="the intents handed to run_backtest")
    positions = pd.Series(np.arange(len(frame), dtype="int64"), index=frame[TIMESTAMP_COLUMN])
    grouped: dict[int, list[TradeIntent]] = {}
    for intent in objects:
        if intent.open_time not in positions.index:
            raise ValueError(
                f"the intent of bar {intent.bar} ({intent.open_time}) is in no bar of the tape"
            )
        position = int(positions.loc[intent.open_time])
        if position != intent.bar:
            raise ValueError(
                "the intents are not aligned with the tape: "
                f"bar {intent.bar} is position {position} of {intent.open_time}"
            )
        grouped.setdefault(position, []).append(intent)
    return grouped


def _result_from(
    trades: list[dict[str, object]],
    ledger: list[dict[str, object]],
    equity: pd.Series,
    run: _Run,
) -> BacktestResult:
    """Package one run: the typed trade log, the ledger, the curve and the metric table."""
    trade_frame = pd.DataFrame(trades, columns=list(TRADE_COLUMNS)).astype(_TRADE_DTYPES)
    ledger_frame = pd.DataFrame(ledger, columns=list(RISK_REJECTION_COLUMNS))
    return BacktestResult(
        trades=trade_frame,
        equity=equity,
        rejections=ledger_frame,
        metrics=calc_metrics(trade_frame, equity, run.cfg),
        config=run.cfg,
        instrument=run.instrument,
    )


def run_backtest(
    tape: pd.DataFrame,
    intents: pd.DataFrame | Iterable[TradeIntent],
    cfg: BacktestConfig | None = None,
    instrument: InstrumentSpec | None = None,
) -> BacktestResult:
    """Simulate ``intents`` on the closed M15 ``tape`` and return the result of the run (Э5').

    ``tape`` is the canonical loader frame (``timestamp`` = open stamp, OHLC, optional
    ``is_closed``) and ``intents`` the accepted setups of the strategy layer - a sequence of
    :class:`~smc_zero.strategy.base.TradeIntent` or an intents frame.  ``instrument`` is the
    price list row the costs are charged from (C6; the EURUSD row of ``ALFAFOREX_SPECS`` by
    default) and it is read *beside* ``cfg.risk``, which prices the order and the reported
    ``risk_pct``.

    The run is deterministic and look-ahead free: bar ``i`` is resolved with bar ``i`` and the
    state the earlier bars left behind, never with a stamp of a later bar.  Nothing is simulated
    on the unclosed tail of the tape (rule 2b) and no order is created or resolved while Alfa is
    shut; the four phases of the module docstring run on every bar, and the equity curve gets
    its value at the end of a bar - flat until a trade closes, because only then does the
    balance move.
    """
    config = BacktestConfig() if cfg is None else cfg
    spec = DEFAULT_INSTRUMENT if instrument is None else instrument
    frame = _entry_tape(tape, config)
    grouped = _intents_by_bar(intents, frame)
    run = _Run(stamps=frame[TIMESTAMP_COLUMN], cfg=config, instrument=spec)

    n = len(frame)
    if n == 0:
        empty = pd.Series(dtype="float64", index=pd.DatetimeIndex([], tz="UTC"), name="equity")
        return _result_from([], [], empty, run)

    high = frame["high"].to_numpy(dtype="float64")
    low = frame["low"].to_numpy(dtype="float64")
    close = frame["close"].to_numpy(dtype="float64")
    days = _msk_days(run.stamps)
    tradable = alfa_trading_mask(run.stamps, config.session).to_numpy(dtype=bool)
    # The last bar of the tape closes its MSK day as well, so the EOD phase can never leave an
    # open trade dangling at the end of the run.
    day_ends = np.ones(n, dtype=bool)
    day_ends[:-1] = days[1:] != days[:-1]

    trades: list[dict[str, object]] = []
    ledger: list[dict[str, object]] = []
    pending: list[_Order] = []
    open_trades: list[_OpenTrade] = []
    placed_today: dict[date, int] = {}
    stops_today: dict[date, int] = {}
    balance = config.initial_capital
    curve = np.empty(n, dtype="float64")

    for i in range(n):
        day: date = days[i]

        # 1. resolution: the stop first, then the target, never on the fill bar itself (the
        #    fill happens in phase 2, so every trade here is at least one bar old).  The window
        #    of ``max_bars_per_trade`` bars ends the book of a trade that never resolved.
        if open_trades:
            surviving: list[_OpenTrade] = []
            for trade in open_trades:
                order = trade.order
                intent = order.intent
                if i - trade.fill_bar >= config.max_bars_per_trade:
                    ledger.append(
                        _ledger_row(
                            intent,
                            i,
                            balance,
                            REASON_NO_RESULT,
                            order.risk_pct,
                            order.risk_warning,
                            config,
                        )
                    )
                    continue
                hit = (
                    _exit_at(
                        intent.side,
                        high[i],
                        low[i],
                        intent.sl,
                        intent.tp,
                        config.risk.slippage,
                    )
                    if tradable[i]
                    else None
                )
                if hit is None:
                    surviving.append(trade)
                    continue
                result, exit_price = hit
                held = _nights_between(days[trade.fill_bar], day)
                trades.append(_trade_row(trade, i, result, exit_price, held, run))
                balance += _profit_of(intent, exit_price, held, run)
                if result == SL_RESULT:
                    # prod counted a stop at the *signal* bar; the event-driven engine counts it
                    # where it happens, which is what the day's cap has to read (see §7.9).
                    stops_today[day] = stops_today.get(day, 0) + 1
            open_trades = surviving

        # 2. fills: only the limits of *earlier* bars (this phase runs before the placement of
        #    bar ``i``), and the window of ``limit_valid_bars`` bars expires the rest.  Both
        #    windows are counted in the bars of the tape, off-hours ones included.
        if pending:
            still_pending: list[_Order] = []
            for order in pending:
                if i - order.placed_bar > config.limit_valid_bars:
                    ledger.append(
                        _ledger_row(
                            order.intent,
                            i,
                            balance,
                            REASON_LIMIT_NOT_FILLED,
                            order.risk_pct,
                            order.risk_warning,
                            config,
                        )
                    )
                    continue
                if tradable[i] and _limit_touched(order.intent, high[i], low[i]):
                    open_trades.append(_OpenTrade(order, i))
                    continue
                still_pending.append(order)
            pending = still_pending

        # 3. placement: the intents of this bar, gated in a fixed order - Alfa's hours, the
        #    day's order budget, the day's stop-out cap, then the margin of the C7 profile
        #    against the balance of this very moment.  The budget is spent by *placed* orders
        #    (prod's counter), and every refusal is written to the ledger with its reason.
        for intent in grouped.get(i, ()):
            verdict = check_intent(intent, config.risk, balance)
            if not tradable[i]:
                reason = REASON_OFF_HOURS
            elif placed_today.get(day, 0) >= config.max_orders_day:
                reason = REASON_ORDER_BUDGET
            elif config.max_sl_per_day > 0 and stops_today.get(day, 0) >= config.max_sl_per_day:
                reason = REASON_SL_CAP
            elif not verdict.kept:
                # The risk gate is the last barrier before an order is placed (its own docstring)
                # and the engine does not duplicate the rule: whatever it refuses, it is asked
                # with the equity of *this* bar, which is why its ledger line says which number.
                reason = REASON_NO_MARGIN if verdict.reason is None else verdict.reason
            else:
                reason = None
            if reason is not None:
                ledger.append(
                    _ledger_row(
                        intent,
                        i,
                        balance,
                        reason,
                        verdict.risk_pct,
                        verdict.risk_warning,
                        config,
                    )
                )
                continue
            pending.append(_Order(intent, i, verdict.risk_pct, verdict.risk_warning))
            placed_today[day] = placed_today.get(day, 0) + 1

        # 4. end of day: prod's ``force_close_eod`` branch, on the last bar of the MSK day (the
        #    bar a date change follows).  It is a market exit, so it slips like a stop does, and
        #    prod zeroes the swap of an EOD close: the position was never held overnight.
        if open_trades and config.force_close_eod and day_ends[i]:
            for trade in open_trades:
                intent = trade.order.intent
                exit_price = (
                    close[i] - config.risk.slippage
                    if intent.side == "long"
                    else close[i] + config.risk.slippage
                )
                trades.append(_trade_row(trade, i, EOD_RESULT, exit_price, 0, run))
                balance += _profit_of(intent, exit_price, 0, run)
            open_trades = []

        curve[i] = balance

    # The tape is over: whatever is left never became a trade - a limit that ran out of bars and
    # a position that resolved in no bar of its window (or before the data ended).  prod counted
    # the two and dropped them; here they are dropped *with* their ledger row, so no intent of
    # the run disappears without a reason.
    last = n - 1
    for trade in open_trades:
        order = trade.order
        ledger.append(
            _ledger_row(
                order.intent,
                last,
                balance,
                REASON_NO_RESULT,
                order.risk_pct,
                order.risk_warning,
                config,
            )
        )
    for order in pending:
        ledger.append(
            _ledger_row(
                order.intent,
                last,
                balance,
                REASON_LIMIT_NOT_FILLED,
                order.risk_pct,
                order.risk_warning,
                config,
            )
        )

    curve_series = pd.Series(
        curve, index=pd.DatetimeIndex(frame[CLOSE_TIME_COLUMN]), name="equity"
    )
    return _result_from(trades, ledger, curve_series, run)

