"""Config tests of Э10': one broker profile owns every money number of a run.

``BrokerSpec`` is the single source of the costs (the spread, the commission, the slippage and the
two overnight legs) and of the contract numbers (``pip_size``, ``contract_size``, ``leverage``);
:class:`~smc_zero.config.RiskConfig` and :class:`~smc_zero.config.InstrumentSpec` read it through
properties instead of keeping a second copy.  The Э5' cost keywords of both classes are still
accepted, so an old call site keeps working: they are migrated *by money* and not by name - the
price-unit ``spread`` becomes the pip ``spread_pip``, and a flat fee per trade becomes a rate per lot
over the two turns (:func:`smc_zero.config._legacy_broker`).

What is pinned here:

* the Alfa numbers of a bare profile and the four helpers that price a trade with them;
* a *zero* profile: ``BrokerSpec`` accepts zeros, because that is how an uncosted run is written
  down, and refuses a negative cost; ``RiskConfig.has_costs`` is ``True`` while *any* of the three
  costs is charged and only the all-zero profile switches it off (Э11'.2: an account that earns on
  the spread alone charges no commission and is still costed), which is the flag rule 4 stamps a
  report with;
* the money of an Э5' call site: the deprecated keywords and the deprecated properties answer the
  same amount as the profile they migrated into;
* the risk profile the Э9' oracle records - the pinned ``RiskConfig`` of ``5a9e015`` written as a
  plain dict by ``dataclasses.asdict`` - still builds on the current classes.

The mutations this module is one line away from, and the test each one must break:

* m1 "require a positive cost" - the uncosted run of Э10' cannot be expressed at all; breaks
  :func:`test_a_zero_cost_is_a_choice_and_a_negative_one_is_refused`;
* m2 "migrate the flat commission by name" - an Э5' call site starts charging 0.35 *per lot* instead
  of 0.35 *per trade*, which is five times the money; breaks
  :func:`test_the_legacy_cost_keywords_keep_the_money`;
* m3 "drop the slippage term" (``or self.broker.slippage_pip > 0``) - a profile charged by the spread
  and the slippage alone is declared uncosted; breaks
  :func:`test_has_costs_is_true_while_any_one_source_charges`;
* m4 "put the ``and`` back" - the false stamp of Э11'.2 returns: an account with a zero commission is
  reported as an uncosted run; breaks :func:`test_has_costs_is_true_while_any_one_source_charges`.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from smc_zero.config import BrokerSpec, InstrumentSpec, RiskConfig

#: ``dataclasses.asdict`` of the pinned ``RiskConfig`` of ``5a9e015``: the profile the recorder of the
#: Э9' oracle writes beside the arguments, with the six cost fields the pin kept in the risk profile.
PINNED_RISK: dict[str, float] = {
    "risk_pct": 1.0,
    "rr": 2.0,
    "commission": 0.0,
    "spread": 0.0,
    "slippage": 0.0,
    "lot": 0.1,
    "leverage": 40.0,
    "contract_size": 100_000.0,
    "pip_size": 0.0001,
    "deposit": 1000.0,
    "warning_risk_pct": 2.0,
}
#: A profile that charges nothing at all: the documented way to write down an uncosted run (Э10').
UNCOSTED = BrokerSpec(
    spread_pip=0.0,
    commission_per_lot_usd=0.0,
    slippage_pip=0.0,
    swap_long_pip=0.0,
    swap_short_pip=0.0,
)


def test_a_bare_profile_charges_the_alfa_numbers() -> None:
    """The shipped profile prices a trade: 1.4 pips of spread and $1.40 of commission at 0.1 lot."""
    broker = BrokerSpec()

    assert (broker.spread_pip, broker.commission_per_lot_usd, broker.slippage_pip) == (1.4, 7.0, 0.2)
    assert (broker.swap_long_pip, broker.swap_short_pip) == (-0.70, 0.0)
    assert (broker.contract_size, broker.pip_size, broker.leverage) == (100_000.0, 0.0001, 40.0)

    assert broker.spread_abs == pytest.approx(0.00014)  # the same spread in price units
    assert broker.slippage_abs == pytest.approx(0.00002)
    assert broker.money(1.0, 0.1) == pytest.approx(1.0)  # a pip on the 0.1 lot of C7
    assert broker.money(30.0, 0.1) == pytest.approx(30.0)  # a 30 pip stop, as the engine charges it
    assert broker.commission(0.1) == pytest.approx(1.4)  # 7.0 per lot over the two turns
    assert broker.commission(0.1, turns=1) == pytest.approx(0.7)
    assert broker.swap_pip("long") == -0.70
    assert broker.swap_pip("short") == 0.0
    assert broker.swap_abs("long", 3, 0.1) == pytest.approx(-2.1)  # three nights at -0.70 a night

    assert RiskConfig().has_costs is True


def test_a_zero_cost_is_a_choice_and_a_negative_one_is_refused() -> None:
    """Zeros express the uncosted run of Э10'; only all three together switch ``has_costs`` off (m1)."""
    uncosted = RiskConfig(broker=UNCOSTED)

    assert uncosted.has_costs is False
    assert uncosted.broker.spread_abs == 0.0
    assert uncosted.broker.slippage_abs == 0.0
    assert uncosted.broker.commission(0.1) == 0.0
    assert uncosted.broker.swap_abs("long", 2, 0.1) == 0.0

    # A run is stamped only while *nothing* is charged (Э11'.2): a single zero leaves the other two
    # sources charging, so the profile is still costed and the report keeps its silence.
    for field in ("spread_pip", "commission_per_lot_usd", "slippage_pip"):
        assert RiskConfig(broker=replace(BrokerSpec(), **{field: 0.0})).has_costs is True
        with pytest.raises(ValueError, match="must be >= 0"):
            BrokerSpec(**{field: -1e-6})

    # The rest of the profile is still a real account: a zero cost is no licence for a zero pip.
    with pytest.raises(ValueError, match="pip_size"):
        replace(UNCOSTED, pip_size=0.0)


def test_has_costs_is_true_while_any_one_source_charges() -> None:
    """The account that earns on the spread alone is a costed run: a zero commission (m3, m4).

    ``configs/live_eurusd_m15.yaml`` prices its trades with a spread of 1.4 pips and a slippage of
    0.2 pips per market leg while its commission is 0.00 - the model of a broker that earns on the
    spread.  Each of the three costs is enough on its own, and the stamp of rule 4 belongs to the
    profile that charges nothing at all.
    """
    spread_and_slippage = RiskConfig(  # the shipped Alfa account: spread + slippage, no commission
        broker=BrokerSpec(spread_pip=1.4, commission_per_lot_usd=0.0, slippage_pip=0.2)
    )
    commission_only = RiskConfig(
        broker=BrokerSpec(spread_pip=0.0, commission_per_lot_usd=7.0, slippage_pip=0.0)
    )
    slippage_only = RiskConfig(
        broker=BrokerSpec(spread_pip=0.0, commission_per_lot_usd=0.0, slippage_pip=0.2)
    )

    for profile in (spread_and_slippage, commission_only, slippage_only):
        assert profile.has_costs is True

    assert RiskConfig(broker=UNCOSTED).has_costs is False


def test_the_legacy_cost_keywords_keep_the_money() -> None:
    """An Э5' call site is migrated by money: the same spread, the same fee, the same pip (m2)."""
    with pytest.warns(DeprecationWarning):  # the six Э5' cost keywords of the risk profile
        legacy = RiskConfig(commission=0.35, spread=0.0001, slippage=0.0001, pip_size=0.0001)

    assert legacy.broker.spread_pip == pytest.approx(1.0)
    assert legacy.broker.slippage_pip == pytest.approx(1.0)
    assert legacy.broker.commission_per_lot_usd == pytest.approx(1.75)
    assert legacy.broker.pip_size == pytest.approx(0.0001)
    assert legacy.broker.contract_size == pytest.approx(100_000.0)
    assert legacy.broker.leverage == pytest.approx(40.0)

    # The money of the old call site is what had to survive: 0.35 a trade on the 0.1 lot is the 1.75
    # per lot of a round turn - five times the number and the same charge.
    assert legacy.broker.commission(0.1) == pytest.approx(0.35)
    assert legacy.broker.spread_abs == pytest.approx(0.0001)
    assert legacy.commission == pytest.approx(1.75)  # the deprecated properties answer in the new
    assert legacy.spread == pytest.approx(1.0)  # units: pips of the profile, money per lot turn
    assert legacy.slippage == pytest.approx(1.0)
    assert legacy.has_costs is True

    # Only what was given moves: the Э5' default profile charged nothing, and the two overnight legs
    # it never mentioned keep the Alfa numbers of the broker it was built on.
    with pytest.warns(DeprecationWarning):
        bare = RiskConfig(commission=0.0, spread=0.0, slippage=0.0)
    assert bare.broker.spread_pip == 0.0
    assert bare.broker.commission_per_lot_usd == 0.0
    assert bare.broker.slippage_pip == 0.0
    assert bare.broker.swap_long_pip == pytest.approx(-0.70)
    assert bare.has_costs is False


def test_the_legacy_price_list_keywords_are_a_rename() -> None:
    """The Э5' row of a symbol migrates name for name: its keywords kept both name and unit (Э10')."""
    with pytest.warns(DeprecationWarning):
        row = InstrumentSpec(spread_pip=2.1, swap_long_pip=-0.55, swap_short_pip=-0.25)

    assert row.broker == BrokerSpec(spread_pip=2.1, swap_long_pip=-0.55, swap_short_pip=-0.25)
    assert row.spread_pip == pytest.approx(2.1)
    assert row.swap_long_pip == pytest.approx(-0.55)
    assert row.swap_short_pip == pytest.approx(-0.25)
    # The row answers in *price units* (the engine multiplies by the lot), unlike the profile.
    assert row.spread_abs == pytest.approx(2.1 * 0.0001)
    assert row.swap_abs("long", 1) == pytest.approx(-0.55 * 0.0001)  # one leg, one night
    assert row.money(0.0001, 0.1) == pytest.approx(1.0)  # a pip of the 0.1 lot, in prices


def test_the_risk_profile_of_the_pinned_oracle_still_builds() -> None:
    """The recordings of the Э9' oracle rebuild on the current classes (Э10'; variant A)."""
    with pytest.warns(DeprecationWarning):
        pinned = RiskConfig(**PINNED_RISK)  # type: ignore[arg-type]

    assert (pinned.risk_pct, pinned.rr, pinned.lot, pinned.deposit) == (1.0, 2.0, 0.1, 1000.0)
    assert pinned.warning_risk_pct == pytest.approx(2.0)
    assert pinned.broker.pip_size == pytest.approx(0.0001)
    assert pinned.broker.contract_size == pytest.approx(100_000.0)
    assert pinned.broker.leverage == pytest.approx(40.0)

    # The pin charged nothing at all (its commission, spread and slippage were all 0.0), so the
    # migrated profile is the zero one - which is why the oracle can replay its recordings instead
    # of comparing two differently priced chains, while the amounts it did name survive.
    assert pinned.has_costs is False
    assert pinned.broker.spread_abs == 0.0
    assert pinned.broker.slippage_abs == 0.0
    assert pinned.broker.commission(0.1) == 0.0
