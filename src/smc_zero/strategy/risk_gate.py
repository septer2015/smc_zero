"""Risk gate: margin first, then the C7 risk percentage (SPEC_SMC.md §7.8 п.37).

The gate is the last barrier before an order is placed, and it asks exactly two questions:

* **Margin** - :func:`margin_of` is C7's own formula ``lot * contract_size * price /
  leverage`` and the order is refused (``no_margin``) when it does not fit the *current*
  equity the caller hands in.  The capital is an argument, never a constant, so the
  backtester can pass the running equity of the account (Э5').
* **Risk** - :func:`risk_pct_of` prices the stop of the intent in percent of
  ``RiskConfig.deposit`` (ruling R2 of §7.8 п.37) and a trade above
  ``RiskConfig.warning_risk_pct`` still passes, flagged ``risk_warning``: the lot is
  *given*, not derived from ``risk_pct``, and C7 says the size is not changed without a
  request.  The median risk per trade is the reporting layer's business (Э5').

:func:`check_intent` is the scalar rule and :func:`apply_risk_gate` the batch face
(§7.8 п.41): an intents frame in, the kept intents with the two reporting columns plus the
rejection ledger out.  ``deposit``, ``lot``, ``leverage`` and the FX pair all come from
:class:`~smc_zero.config.RiskConfig` - the one copy of every broker number in the layer -
so no number of C7 is hard-coded here.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from smc_zero.config import RiskConfig
from smc_zero.strategy.base import TradeIntent, intents_frame, intents_from_frame

#: The margin verdict of C7: an order that does not fit the equity is refused (п.37).
REASON_NO_MARGIN = "no_margin"
#: The two reporting columns :func:`apply_risk_gate` adds to the kept intents.
RISK_PCT_COLUMN = "risk_pct"
RISK_WARNING_COLUMN = "risk_warning"
#: Columns of the risk ledger: the intent, its margin arithmetic and the verdict.
RISK_REJECTION_COLUMNS: tuple[str, ...] = (
    "bar",
    "open_time",
    "level_name",
    "side",
    "entry",
    "required_margin",
    "equity",
    RISK_PCT_COLUMN,
    RISK_WARNING_COLUMN,
    "reason",
)


class RiskVerdict(NamedTuple):
    """Verdict of one intent: kept or refused, with the reporting numbers either way."""

    kept: bool
    reason: str | None
    risk_pct: float
    risk_warning: bool


def margin_of(intent: TradeIntent, cfg: RiskConfig) -> float:
    """Return C7's margin requirement of the intent: ``lot * contract * price / leverage``.

    ``price`` is the entry of the pending limit - what the order would have to fund at the
    moment it is placed (the fill price is the backtester's business, Э5').
    """
    return cfg.lot * cfg.contract_size * intent.entry / cfg.leverage


def risk_pct_of(intent: TradeIntent, cfg: RiskConfig) -> float:
    """Return the risk of the intent in percent of ``cfg.deposit`` (ruling R2, §7.8 п.37).

    ``sl_pips * pip_size * contract_size * lot`` is the money at stake (``sl_pips`` is the
    chain's own stop distance, ``|entry - sl| / pip_size``), and ``cfg.deposit`` - not the
    running equity - is the denominator the owner ruled for the reported column.
    """
    return intent.sl_pips * cfg.pip_size * cfg.contract_size * cfg.lot / cfg.deposit * 100.0


def check_intent(
    intent: TradeIntent, cfg: RiskConfig, current_equity: float
) -> RiskVerdict:
    """Return the verdict of one intent: the margin gate first, then the C7 risk numbers.

    A refused order still reports its risk percentage and flag, so the ledger explains a
    refusal without recomputing anything.
    """
    risk_pct = risk_pct_of(intent, cfg)
    warning = risk_pct > cfg.warning_risk_pct
    if margin_of(intent, cfg) > current_equity:
        return RiskVerdict(False, REASON_NO_MARGIN, risk_pct, warning)
    return RiskVerdict(True, None, risk_pct, warning)


def apply_risk_gate(
    intents_df: pd.DataFrame, cfg: RiskConfig, current_equity: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(kept intents, rejection ledger)`` of a batch of intents (§7.8 п.41).

    ``intents_df`` is an intents frame (``smc_zero.strategy.base.intents_frame``) and
    ``current_equity`` the equity the margin check measures against - the backtester hands
    its running equity in (Э5').  Kept rows keep their order and carry ``risk_pct`` /
    ``risk_warning``; refused rows carry the same numbers plus the margin arithmetic and
    the reason ``no_margin``, one row per refusal.
    """
    if not np.isfinite(current_equity):
        raise ValueError(f"current_equity must be a number, got {current_equity!r}")
    intents = intents_from_frame(intents_df)
    verdicts = [check_intent(intent, cfg, current_equity) for intent in intents]
    frame = intents_frame(intents)
    frame[RISK_PCT_COLUMN] = [verdict.risk_pct for verdict in verdicts]
    frame[RISK_WARNING_COLUMN] = [verdict.risk_warning for verdict in verdicts]
    ledger = pd.DataFrame(
        {
            "bar": frame["bar"],
            "open_time": frame["open_time"],
            "level_name": frame["level_name"],
            "side": frame["side"],
            "entry": frame["entry"],
            "required_margin": [margin_of(intent, cfg) for intent in intents],
            "equity": [current_equity] * len(intents),
            RISK_PCT_COLUMN: frame[RISK_PCT_COLUMN],
            RISK_WARNING_COLUMN: frame[RISK_WARNING_COLUMN],
            "reason": [verdict.reason for verdict in verdicts],
        }
    ).loc[:, list(RISK_REJECTION_COLUMNS)]
    kept = np.array([verdict.kept for verdict in verdicts], dtype=bool)
    return (
        frame.loc[kept].reset_index(drop=True),
        ledger.loc[~kept].reset_index(drop=True),
    )
