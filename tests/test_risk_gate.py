"""Risk-gate tests: the C7 margin, the risk percentage, the batch wrapper (§7.8 п.37, п.41).

The arithmetic is EURUSD and it is C7's own: 0.1 lot on a 100 000 contract at 1.0800 with a
leverage of 40 needs $270 of margin, and a pip is $1 at that lot, so a 30 pip stop risks $30 -
3 % of the ruled $1000 deposit and above the 2 % flag.

The two mutations the module has to survive are the two ways the risk number could rot:

* m1 "denominator = ``current_equity``" - the percentage becomes an account reading instead of the
  ruled deposit; it must break :func:`test_the_risk_percentage_is_measured_against_the_deposit`
  (the same trade at equity 2000 would read 1.5 %);
* m2 "hard-code 10000" - prod's old capital sneaks back in (the C7 trap); it must break the same
  test (which pins 3.0 % from ``RiskConfig.deposit = 1000``).
"""

from __future__ import annotations

import pandas as pd
import pytest

from smc_zero.config import RiskConfig
from smc_zero.indicators.levels import PDH
from smc_zero.strategy.base import INTENT_COLUMNS, TradeIntent, intents_frame
from smc_zero.strategy.risk_gate import (
    REASON_NO_MARGIN,
    RISK_PCT_COLUMN,
    RISK_REJECTION_COLUMNS,
    RISK_WARNING_COLUMN,
    RiskVerdict,
    apply_risk_gate,
    check_intent,
    margin_of,
    risk_pct_of,
)

DAY = "2026-06-10"  # Wednesday
ENTRY_TIME = pd.Timestamp(f"{DAY} 08:00", tz="UTC")


def _intent(**overrides: object) -> TradeIntent:
    """One accepted EURUSD intent; only the fields a test varies are passed in."""
    fields: dict[str, object] = {
        "open_time": ENTRY_TIME,
        "bar": 32,
        "side": "short",
        "entry": 1.0800,
        "sl": 1.0830,
        "tp": 1.0700,
        "tp_source": "liquidity",
        "sl_source": "sweep_extreme",
        "sl_pips": 30.0,
        "rr": 2.0,
        "level_name": PDH,
        "level_date": pd.Timestamp(DAY),
        "level_price": 1.0830,
        "setup_type": "fresh",
        "sweep_bar": 27,
        "choch_bar": 29,
        "fvg_bar": 31,
    }
    fields.update(overrides)
    return TradeIntent(**fields)  # type: ignore[arg-type]


def test_the_margin_of_a_tenth_lot_eurusd_is_two_hundred_seventy_dollars() -> None:
    """``0.1 * 100 000 * 1.0800 / 40`` - C7's formula, exactly."""
    cfg = RiskConfig()
    intent = _intent()
    assert margin_of(intent, cfg) == pytest.approx(270.0)
    assert check_intent(intent, cfg, 1000.0) == RiskVerdict(True, None, 3.0, True)
    refused = check_intent(intent, cfg, 100.0)  # a $100 account cannot fund the order
    assert refused.kept is False
    assert refused.reason == REASON_NO_MARGIN
    assert refused.risk_pct == pytest.approx(3.0)  # the refusal still reports the risk


def test_the_margin_boundary_is_inclusive() -> None:
    """``required <= equity`` keeps the order: the boundary itself is fundable."""
    intent = _intent()
    assert check_intent(intent, RiskConfig(), 270.0).kept is True
    assert check_intent(intent, RiskConfig(), 269.99).kept is False


def test_the_risk_percentage_is_measured_against_the_deposit() -> None:
    """30 pips at 0.1 lot is $30: 3 % of the C7 deposit, whatever the running equity is (m1/m2)."""
    cfg = RiskConfig()
    intent = _intent()
    assert risk_pct_of(intent, cfg) == pytest.approx(3.0)
    assert check_intent(intent, cfg, 2000.0).risk_pct == pytest.approx(3.0)  # not 1.5
    assert check_intent(intent, cfg, 1000.0).risk_warning is True  # above the 2 % flag
    # the denominator is the configured deposit: a $10000 deposit would read 0.3 %, not 3.0 %
    wide = RiskConfig(deposit=10_000.0)
    assert risk_pct_of(intent, wide) == pytest.approx(0.3)
    assert check_intent(intent, wide, 10_000.0).risk_warning is False


def test_the_warning_threshold_is_strict() -> None:
    """Exactly ``warning_risk_pct`` is not a warning; a hair above it is."""
    cfg = RiskConfig()
    assert check_intent(_intent(sl_pips=20.0), cfg, 1000.0).risk_warning is False  # 2.0 %
    assert check_intent(_intent(sl_pips=20.1), cfg, 1000.0).risk_warning is True  # 2.01 %


def test_a_flagged_intent_is_still_placed() -> None:
    """C7: the lot is given, so a flagged trade passes - the report is what carries the flag."""
    verdict = check_intent(_intent(sl_pips=60.0), RiskConfig(), 1000.0)  # 6 %
    assert verdict.kept is True
    assert verdict.reason is None
    assert verdict.risk_pct == pytest.approx(6.0)
    assert verdict.risk_warning is True


def test_the_batch_wrapper_splits_the_intents_and_the_ledger() -> None:
    """§7.8 п.41: kept rows keep their order, refused rows carry the C7 arithmetic."""
    intents = (_intent(), _intent(entry=2000.0, bar=33), _intent(sl_pips=50.0, bar=34))
    kept, ledger = apply_risk_gate(intents_frame(intents), RiskConfig(), 1000.0)
    assert list(kept.columns) == [*INTENT_COLUMNS, RISK_PCT_COLUMN, RISK_WARNING_COLUMN]
    assert kept["bar"].tolist() == [32, 34]  # the refused one is gone, the order is intact
    assert kept[RISK_PCT_COLUMN].tolist() == [pytest.approx(3.0), pytest.approx(5.0)]
    assert kept[RISK_WARNING_COLUMN].tolist() == [True, True]
    assert list(ledger.columns) == list(RISK_REJECTION_COLUMNS)
    assert ledger["bar"].tolist() == [33]
    assert ledger["reason"].tolist() == [REASON_NO_MARGIN]
    assert ledger["required_margin"].tolist() == [pytest.approx(500_000.0)]
    assert ledger["equity"].tolist() == [1000.0]
    assert ledger[RISK_PCT_COLUMN].tolist() == [pytest.approx(3.0)]


def test_the_scalar_and_the_batch_agree_row_by_row() -> None:
    """The wrapper is a face of the scalar rule, not a second implementation."""
    cfg = RiskConfig()
    intents = (_intent(), _intent(entry=2000.0, bar=33), _intent(sl_pips=50.0, bar=34))
    frame = intents_frame(intents)
    for equity in (10.0, 270.0, 1000.0, 1_000_000.0):
        kept, ledger = apply_risk_gate(frame, cfg, equity)
        verdicts = [check_intent(intent, cfg, equity) for intent in intents]
        kept_bars = [i.bar for i, v in zip(intents, verdicts, strict=True) if v.kept]
        refused_bars = [i.bar for i, v in zip(intents, verdicts, strict=True) if not v.kept]
        assert kept["bar"].tolist() == kept_bars
        assert ledger["bar"].tolist() == refused_bars
        assert kept[RISK_PCT_COLUMN].tolist() == pytest.approx(
            [v.risk_pct for v in verdicts if v.kept]
        )
        assert ledger[RISK_WARNING_COLUMN].tolist() == [
            v.risk_warning for v in verdicts if not v.kept
        ]


def test_the_batch_wrapper_keeps_an_empty_frame_empty() -> None:
    """No intents in, no ledger out - and both frames keep their columns."""
    kept, ledger = apply_risk_gate(intents_frame([]), RiskConfig(), 1000.0)
    assert list(kept.columns) == [*INTENT_COLUMNS, RISK_PCT_COLUMN, RISK_WARNING_COLUMN]
    assert list(ledger.columns) == list(RISK_REJECTION_COLUMNS)
    assert kept.empty
    assert ledger.empty


def test_the_batch_wrapper_validates_its_input() -> None:
    """A broken frame or a NaN equity is a wiring error, not an empty run."""
    cfg = RiskConfig()
    with pytest.raises(ValueError, match="current_equity must be a number"):
        apply_risk_gate(intents_frame([_intent()]), cfg, float("nan"))
    incomplete = intents_frame([_intent()]).drop(columns=["entry"])
    with pytest.raises(ValueError, match="needs the"):
        apply_risk_gate(incomplete, cfg, 1000.0)

