# smc-zero

SMC backtest foundation with a strict three-timeframe hierarchy: **D1** = global bias, PDH/PDL
and major order blocks; **H1** = working structure (BOS/CHoCH), premium/discount and sessions;
**M15** = entry zone (sweep -> CHoCH -> entry into an M15 FVG/OB inside the H1 zone). M5 is
intentionally not used. **H4** takes part in the HTF bias only: `BiasConfig.timeframes` defaults
to `("H1", "H4", "D1")` (SPEC_SMC.md C5), and H4 is never an entry timeframe.

Time convention: the `datetime` column in `./data/*.csv` (M15/H1/D1) is an **open_time** stamp
(UTC-naive, localized to UTC), so `close_time = timestamp + bar period`. The last bar of every
timeframe is treated as still forming and dropped by default (`drop_unclosed=True`) to avoid thin
lookahead on the live edge.

## Install

```bash
pip install -e ".[dev]"
```

The Э7' optimizer needs optuna, which is an optional extra - the layer and its tests import cleanly
without it:

```bash
pip install -e ".[dev,optimize]"
```

## Checks

```bash
ruff check .
pytest
```

Implemented so far: the data layer (`data_loader.py`: open_time convention, closed-bar
HTF -> LTF stitching), the indicator layer (`indicators/structure.py` swings and BOS/CHoCH,
`fvg.py`, `liquidity.py` sweeps, `impulse.py` displacement gate, `sessions.py` killzones,
`bias.py` H1/H4/D1 bias, `levels.py` PDH/PDL, PWH/PWL, PMH/PML and the Asian/London/NY session
ranges with their availability gates and fresh/broken lifecycle) and the strategy layer
(`strategy/intents.py` - the M15 entry chain of SPEC_SMC.md §7.8 with `sweep -> CHoCH ->
displacement -> FVG limit`, one intent per accepted setup and a rejection ledger;
`strategy/take_profit.py` - the nearest visible liquidity level with the RR fallback;
`strategy/risk_gate.py` - the C7 margin check and the risk percentage of a batch of intents)
and the backtester layer (`backtester/engine.py` - the event-driven engine of SPEC_SMC.md §7.9:
one limit filled on the bar after its signal, the stop looked at before the target, the C6 price
list charged as spread/swap plus the profile's commission and slippage, the C7 margin gate asked
with the running equity, and a ledger row for every intent that did not become a trade;
`backtester/metrics.py` - the metric table; `backtester/reports.py` - the text summary and the
trade-log export) and the walk-forward layer (`backtester/walkforward.py` - the out-of-sample split
of one tape (Э6'): `split_walkforward` cuts anchored expanding folds by default and rolling ones
with `anchored=False`, as positional views of the caller's frame, so a fold never shares a bar with
its own train window; `aggregate_fold_metrics` reports the mean *and* the population sigma of the
six headline metrics, and `run_walkforward` joins the folds to the Э5' engine with fixed costs and
no optimization - fitting the parameters is Э7') and the optimizer layer (`optimizer/` - the search
of SPEC_SMC.md §7.11, Э7': `build_tape_marks` caches the HTF bias markup and the level book **once
per run** for every trial of a study, `score_from_aggregates` ranks one parameter set by its
out-of-sample folds - the leading metric of the walk-forward aggregate, throttled by the drawdown
and by how much of the in-sample profit survived - `run_optimization` maximizes that score with a
seeded TPE study over the ranges of `PARAM_RANGES` and re-evaluates the winner over every fold from
scratch, while `cache_mismatches` refuses a configuration the cache was not built from; optuna is
imported lazily by the study factory, so the layer - and its tests - run without it). Data files
live in `./data/`, reports land in `./reports/`.
