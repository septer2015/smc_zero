"""Base interfaces and value objects shared by the strategy layer.

The strategy layer turns indicator markup into *intents*: one completed entry setup
with its order geometry, built before anything is placed.  :class:`TradeIntent` lives
here so the entry chain (:mod:`smc_zero.strategy.intents`), the take-profit builder
(:mod:`smc_zero.strategy.take_profit`) and the risk gate
(:mod:`smc_zero.strategy.risk_gate`) share one type instead of three dictionaries.

The tuple of frozen intents stays the *currency* of the layer (``EntryChain.intents``),
and :func:`intents_frame` / :func:`intents_from_frame` are its batch form, so the two
wrappers speak a DataFrame without the chain giving up the value objects its tests
compare.  :func:`utc_stamps` is the single implementation of the stamp rule the whole
layer shares.

Frames handed to the layer are aligned already: the entry frame carries ``is_closed``
and the HTF bias is read through :func:`smc_zero.data_loader.align_htf_to_ltf` /
``bias_frames``, and every rule reads only the bar whose close it is evaluating.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from typing import cast

import pandas as pd

from smc_zero.config import SetupType, Side, SLType, TPSource


@dataclass(frozen=True, slots=True)
class TradeIntent:
    """One accepted entry setup: the order geometry plus its provenance.

    ``bar`` / ``open_time`` are the M15 bar whose close produced the signal - the bar
    the chain evaluated, *not* the fill: ``entry`` is a pending limit, so the order
    lifecycle (validity window, fills, EOD closing) belongs to the backtester (Э5').

    ``side`` and the prices come from the level that was swept, ``sl_source`` records
    whether the stop sits behind the swept extreme or behind the gap edge (prod's
    fallback) and ``tp_source`` whether the target is a liquidity level or the RR
    fallback.  ``sl_pips`` is the stop distance in pips and ``rr`` the planned
    reward/risk of the target as built, so the report never recomputes either.

    The provenance block answers "where did this come from" without re-running the
    strategy: the level instance of
    :func:`smc_zero.indicators.levels.static_levels` (``level_name`` / ``level_date`` /
    ``level_price``) and the three positional bars of the chain - ``sweep_bar``
    (:func:`smc_zero.indicators.liquidity.sweep_index`), ``choch_bar`` (the break that
    confirmed the reversal) and ``fvg_bar`` (the middle bar of the gap hosting the
    order).
    """

    open_time: pd.Timestamp
    bar: int
    side: Side
    entry: float
    sl: float
    tp: float
    tp_source: TPSource
    sl_source: SLType
    sl_pips: float
    rr: float
    level_name: str
    level_date: pd.Timestamp
    level_price: float
    setup_type: SetupType
    sweep_bar: int
    choch_bar: int
    fvg_bar: int

#: The fields of :class:`TradeIntent` - and hence the columns of an intents frame.
INTENT_COLUMNS: tuple[str, ...] = tuple(field.name for field in fields(TradeIntent))
#: Dtypes of :func:`intents_frame`, so an empty frame keeps the batch contract as well.
_INTENT_DTYPES: Mapping[str, str] = {
    "open_time": "datetime64[ns, UTC]",
    "bar": "int64",
    "side": "object",
    "entry": "float64",
    "sl": "float64",
    "tp": "float64",
    "tp_source": "object",
    "sl_source": "object",
    "sl_pips": "float64",
    "rr": "float64",
    "level_name": "object",
    "level_date": "datetime64[ns]",
    "level_price": "float64",
    "setup_type": "object",
    "sweep_bar": "int64",
    "choch_bar": "int64",
    "fvg_bar": "int64",
}


def utc_stamps(values: pd.Series, *, source: str) -> pd.Series:
    """Return ``values`` as UTC-aware ``datetime64[ns, UTC]``.

    The loader stores open_time stamps UTC-naive, so naive input is *read as UTC* and a
    non-datetime column is a hard error: nothing in this layer may guess the unit of a
    broken pipeline.  This is the one implementation of the rule - the entry chain, the
    level book of the take-profit builder and the intents frame all call it.
    """
    if not pd.api.types.is_datetime64_any_dtype(values.dtype):
        raise ValueError(f"{source} must be a datetime column, got dtype {values.dtype}")
    if not isinstance(values.dtype, pd.DatetimeTZDtype):
        values = values.dt.tz_localize("UTC")
    return values.dt.tz_convert("UTC").astype("datetime64[ns, UTC]")


def intents_frame(intents: Iterable[TradeIntent]) -> pd.DataFrame:
    """Return the intents as a typed frame with the columns of :data:`INTENT_COLUMNS`.

    This is the *batch* form of the layer: :func:`build_intents` keeps returning the
    frozen value objects its tests compare, while the take-profit builder and the risk
    gate take and return this frame (SPEC_SMC.md §7.8 п.41).  Row order is the order the
    chain accepted the intents in, and :func:`intents_from_frame` reads it back.
    """
    rows = [
        {field.name: getattr(intent, field.name) for field in fields(TradeIntent)}
        for intent in intents
    ]
    return pd.DataFrame(rows).reindex(columns=list(INTENT_COLUMNS)).astype(_INTENT_DTYPES)


def intents_from_frame(
    frame: pd.DataFrame, *, source: str = "the intents frame"
) -> tuple[TradeIntent, ...]:
    """Return the frame as :class:`TradeIntent` objects, row order preserved.

    ``open_time`` is normalized to UTC here (naive stamps are read as UTC, like everywhere
    else in the layer) and a missing column is a hard error instead of a silent NaN: a
    wrapper may not guess what its caller meant to hand it.
    """
    missing = [name for name in INTENT_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(
            f"{source} needs the {missing} column(s); build it with intents_frame(...)"
        )
    rows = frame.loc[:, list(INTENT_COLUMNS)].reset_index(drop=True)
    stamps = utc_stamps(rows["open_time"], source=source)
    return tuple(
        TradeIntent(
            open_time=stamps.iloc[position],
            bar=int(rows["bar"].iloc[position]),
            side=cast(Side, rows["side"].iloc[position]),
            entry=float(rows["entry"].iloc[position]),
            sl=float(rows["sl"].iloc[position]),
            tp=float(rows["tp"].iloc[position]),
            tp_source=cast(TPSource, rows["tp_source"].iloc[position]),
            sl_source=cast(SLType, rows["sl_source"].iloc[position]),
            sl_pips=float(rows["sl_pips"].iloc[position]),
            rr=float(rows["rr"].iloc[position]),
            level_name=str(rows["level_name"].iloc[position]),
            level_date=pd.Timestamp(rows["level_date"].iloc[position]),
            level_price=float(rows["level_price"].iloc[position]),
            setup_type=cast(SetupType, rows["setup_type"].iloc[position]),
            sweep_bar=int(rows["sweep_bar"].iloc[position]),
            choch_bar=int(rows["choch_bar"].iloc[position]),
            fvg_bar=int(rows["fvg_bar"].iloc[position]),
        )
        for position in range(len(rows))
    )


# TODO(tests): leak test - a signal at bar i must not change when bar i+1 changes.
# TODO(phase-strategy): premium/discount and H1-zone gates of the constitution's TF
# hierarchy are not part of v1 (SPEC_SMC.md §7.8 п.33: H1 zones are expressed as levels).

