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
`strategy/risk_gate.py` - the C7 margin check and the risk percentage of a batch of intents).
The backtester layer is still a documented stub: the engine, the costs and the equity curve
arrive with Э5'. Data files live in `./data/`.
