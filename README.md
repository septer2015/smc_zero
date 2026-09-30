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

## Quick start

Install with the checks and the optimizer:

```bash
pip install -e ".[dev,optimize]"
```

One backtest over the window you name - every threshold is the project's own, the window is the
whole UTC days of `--start` .. `--end`:

```bash
smc-backtest --symbol EURUSD --start 2022-08-15 --end 2022-09-15
```

Without the install, the same run from the repository root (`Python` needs both the package of
`src` and the runner package on its path):

```bash
PYTHONPATH=src python -m scripts.run_backtest --symbol EURUSD --start 2022-08-15 --end 2022-09-15
```

The search over the walk-forward folds, then the winner reported as one run:

```bash
smc-optimize --symbol EURUSD --n-trials 100 --jobs 4
```

Reports land in `./reports/`:

* `backtest_<symbol>_<tf>_<start>_<end>/` - `trades.csv` / `trades.parquet` and `summary.txt`;
* `optimization_<symbol>_<tf>_<start>_<end>_n<trials>/` - the same two files for the winner, plus
  `best_params.json` (parameters, score, its `score_metric` and `penalty_power`, fold aggregates) and
  `fold_metrics.csv` (train and test table of every fold, with their mean and sigma rows - the `mean`
  rows also carry the run's `train_profit_mean`, `test_profit_mean` and `degradation_ratio`).

Every run is stamped by rule 4: while `RiskConfig` carries no commission and no slippage, the
summary says so on the page and the curve must not be read as a profit (C6 / §5 п.10).

The window is a cost, not a detail: the entry chain of Э4' walks every level of the window bar by
bar, so its time used to grow roughly with the square of the window - measured on
`./data/EURUSD_M15.csv` before Э9' that was 5 days 0.3 s, 1 month 6 s, 3 months 58 s, 1 year 16 min
(`937.9 s`). The vectorised chain of Э9' (SPEC_SMC.md §7.13) answers the same rules far faster: the
same year takes 1.2-1.9 s and the full four-year tape 9.7-17.3 s, so the shipped window and the
optimization over it are affordable - what a study costs is measured by the first real run, and the
fold grid is 15 folds of 60 days by default (SPEC_SMC.md §7.10 п.62). Start with a month or a
quarter anyway - and with the costs of your profile, because rule 4 stamps a run without them.

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
displacement -> FVG limit`, one intent per accepted bar (the most significant level of a bar wins)
and a rejection ledger;
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
out-of-sample folds - the leading metric of the walk-forward aggregate times the out-of-sample
profit, divided by the drawdown, and - at the operator's `penalty_power` - throttled by how much of
the in-sample profit survived; an out-of-sample window that made no money scores a flat zero, so a
losing parameter set can never outrank a profitable one - `run_optimization` maximizes that score with
a seeded TPE study over the ranges of `PARAM_RANGES` and re-evaluates the winner over every fold from
scratch, while `cache_mismatches` refuses a configuration the cache was not built from; optuna is
imported lazily by the study factory, so the layer - and its tests - run without it) and the console
layer (`scripts/` - the Э8' runners: `smc-backtest` runs one fixed configuration over a window of
`./data` and writes its report, `smc-optimize` runs the Э7' study over a walk-forward and then
reports the winner; both are `./reports` writers with an argparse contract of their own, and neither
defines a threshold or recomputes a metric - SPEC_SMC.md §7.12). Data files live in `./data/`,
reports land in `./reports/`.
