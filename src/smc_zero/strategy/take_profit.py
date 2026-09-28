"""Take-profit builder: liquidity first, RR fallback second (SPEC_SMC.md §7.8 п.36).

The rule is small enough to live in one function, and it does:

* :func:`take_profit_for` - the scalar rule.  The entry chain calls it for every accepted
  attempt, the batch wrapper calls it for every row of an intents frame, so there is one
  implementation of "which target".  It takes the nearest *counter-side* level of the book
  that is visible at the entry bar (``available_at <= t``), skips the levels whose
  reward/risk is under ``cfg.min_tp_rr`` in favour of the next farther one, and falls back
  on prod's ``entry ∓ sl_size * cfg.rr_fallback`` when no level clears the floor
  (``min_tp_rr`` is a floor, not a filter: a level *at* the floor is taken as it is).
  "Visible at ``t``" is the leak guard of C2 п.2 - a level that only becomes a fact later
  must not become a target - and the filter is a mandatory test.
* :func:`build_take_profit` - the batch face (SPEC_SMC.md §7.8 п.41): an intents frame in,
  the same frame with ``tp`` / ``tp_source`` / ``rr`` rebuilt out.

The level book is the Э3' markup (``static_levels`` after ``level_lifecycle``); this module
reads ``price``, ``is_upper`` and ``available_at`` from it and never rebuilds a level, and
it never looks at a bar after the entry bar.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from smc_zero.config import Side, TPConfig, TPMode, TPSource
from smc_zero.indicators.levels import (
    LEVEL_AVAILABLE_AT_COLUMN,
    LEVEL_IS_UPPER_COLUMN,
    LEVEL_PRICE_COLUMN,
)
from smc_zero.strategy.base import TradeIntent, intents_frame, intents_from_frame, utc_stamps

#: The columns :func:`build_take_profit` rebuilds on the intents frame.
TP_COLUMNS: tuple[str, ...] = ("tp", "tp_source", "rr")
#: The only ``TPConfig.mode`` v1 implements; the others raise (§7.8 п.33).
IMPLEMENTED_MODE: TPMode = "liquidity_with_rr_fallback"


def _check_mode(cfg: TPConfig) -> None:
    """Refuse the TP modes v1 declares but does not implement (§7.8 п.33).

    The check sits in the scalar rule and the wrapper calls it before the walk, so a
    misconfigured run fails even when the frame it was handed happens to be empty.
    """
    if cfg.mode != IMPLEMENTED_MODE:
        raise NotImplementedError(
            f"take_profit.mode={cfg.mode!r} is declared by SPEC_SMC.md §7.8 п.33 but not "
            f"implemented by v1 (v1 implements {IMPLEMENTED_MODE!r})"
        )


def take_profit_for(
    entry: float,
    sl: float,
    side: Side,
    levels: pd.DataFrame,
    t: pd.Timestamp,
    cfg: TPConfig,
) -> tuple[float, TPSource, float]:
    """Return ``(price, source, rr)`` of the target of one accepted attempt.

    ``levels`` is the level book as ``build_intents`` receives it (``price``, ``is_upper``,
    ``available_at``) and ``t`` the open_time of the decision bar.  Levels are walked
    nearest-first; the first one clearing ``cfg.min_tp_rr`` wins, and a book without such a
    level falls back on prod's multiple.  ``rr`` is returned as computed, so neither the
    intent nor the report recomputes it.
    """
    _check_mode(cfg)
    required = (LEVEL_PRICE_COLUMN, LEVEL_IS_UPPER_COLUMN, LEVEL_AVAILABLE_AT_COLUMN)
    missing = [name for name in required if name not in levels.columns]
    if missing:
        raise ValueError(
            f"a take-profit needs a level book with the {required} column(s); missing {missing}"
        )
    sl_size = abs(entry - sl)
    if sl_size <= 0:
        raise ValueError("a take-profit needs a positive stop distance")
    prices = levels[LEVEL_PRICE_COLUMN].to_numpy(dtype="float64")
    uppers = levels[LEVEL_IS_UPPER_COLUMN].to_numpy(dtype=bool)
    available = utc_stamps(levels[LEVEL_AVAILABLE_AT_COLUMN], source="the level book")
    visible = (available <= t).to_numpy(dtype=bool)
    if side == "long":
        side_sign = 1.0
        counter_side = visible & uppers & (prices > entry)
    else:
        side_sign = -1.0
        counter_side = visible & (~uppers) & (prices < entry)
    candidates = np.flatnonzero(counter_side)
    if candidates.size:
        distance = np.abs(prices[candidates] - entry)
        for position in candidates[np.lexsort((candidates, distance))]:
            rr = (prices[position] - entry) * side_sign / sl_size
            if rr >= cfg.min_tp_rr:
                return float(prices[position]), "liquidity", float(rr)
    target = entry + side_sign * sl_size * cfg.rr_fallback
    return float(target), "rr", float(cfg.rr_fallback)


def build_take_profit(
    intents_df: pd.DataFrame, levels_df: pd.DataFrame, cfg: TPConfig
) -> pd.DataFrame:
    """Return ``intents_df`` with the take-profit of every row rebuilt (SPEC_SMC.md §7.8 п.41).

    ``intents_df`` is an intents frame (``smc_zero.strategy.base.intents_frame``); the result
    keeps the columns and the row order of the input and replaces :data:`TP_COLUMNS`.
    """
    _check_mode(cfg)
    rebuilt: list[TradeIntent] = []
    for intent in intents_from_frame(intents_df):
        tp, tp_source, rr = take_profit_for(
            intent.entry, intent.sl, intent.side, levels_df, intent.open_time, cfg
        )
        rebuilt.append(replace(intent, tp=tp, tp_source=tp_source, rr=rr))
    return intents_frame(rebuilt)
