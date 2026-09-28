"""Strategy layer: turns indicator markup into entry intents and their risk verdicts.

* :mod:`smc_zero.strategy.intents` - the M15 entry chain of SPEC_SMC.md §7.8 (one
  :class:`~smc_zero.strategy.base.TradeIntent` per accepted setup, one ledger row per
  filtered attempt);
* :mod:`smc_zero.strategy.take_profit` - the target of an intent (liquidity or RR
  fallback);
* :mod:`smc_zero.strategy.risk_gate` - C7's margin check and risk percentage;
* :mod:`smc_zero.strategy.base` - the shared :class:`~smc_zero.strategy.base.TradeIntent`
  and the intents frame the batch wrappers speak.
"""
