"""Live-config tests of Э10'.2: the loader, the two blocks and the ``--config-path`` flag.

The config of the winner is a *file*, so what is pinned here is the road it travels: the loader
that reads it, the mapping of its three blocks onto :class:`~smc_zero.config.StrategyConfig` and
:class:`~smc_zero.config.BacktestConfig`, the priority of the command line over the file, and the
two faces that consume it.  The optional ``session`` block of Э17.z is pinned here as well: it is
the one place a killzone *window* leaves the code, and the shipped configs of ``configs/`` must
never carry its research switch.  The expectation of every comparison is written down *again* here,
by hand and out of the file it checks (``_hand_built_strategy`` and its two neighbours): a test that
read its expected numbers from the same YAML would only prove that the YAML parses.

The two windows of the end-to-end tests are the shipped ``./data/EURUSD_M15.csv``, because a live
config is a statement about the real tape.  Measured on 2026-10-05 on the shipped account - the
Alfa-Forex spread account of Э11'.2: 1.4 pips of round-trip spread, 0.2 pips of slippage per market
leg and no commission:

* five days (``2022-08-15 .. 2022-08-20``, 444 bars) carry **no setup at all** - neither under the
  defaults nor under the winner's numbers - so that window pins the command line (flag accepted,
  exit code 0, both report files written) and the equivalence of the config with the hand-built
  numbers is measured on the longer window below;
* three weeks (``2022-08-15 .. 2022-09-05``, 1469 bars) carry a setup under both sides and separate
  them: measured on 2026-10-12 under the broker-server clock of Э17.B, the defaults finish at
  ``1 trade / -51.70`` and the winner at ``1 trade / +36.05``, so a config the runner forgot to
  apply cannot pass this test by accident.  The window used to be two weeks: the Э17 audit showed
  that the shipped tape carries broker server time rather than UTC, the normalisation moved the few
  bars that armed the setup, and both numbers moved with the clock (the two sides of the comparison
  are pinned to one clock - see :func:`_hand_built_profit`).

The mutations this file is one line away from, and the test each one must break (all four measured
on 2026-10-01, each red on the test named here and only there):

* m1 "ignore ``--config-path``" (``live_inputs`` returns the defaults of Э8' whatever the flag
  says) - the live run pays what the default numbers pay; breaks
  :func:`test_the_live_config_of_the_winner_pays_what_the_hand_built_numbers_pay` and
  :func:`test_an_argument_wins_over_the_config`;
* m2 "default ``--symbol`` to EURUSD again" - a default typed by nobody outranks the pair the config
  names.  Measured: the tests that use the shipped pair stay green, because the file names EURUSD
  too, so the mutation is only visible on a file of *another* pair - that is what
  :func:`test_the_pair_of_the_config_is_used_when_the_caller_types_none` writes;
* m3 "accept a config without a block" - the missing half would have to be guessed (rule 5); breaks
  :func:`test_a_config_without_a_mandatory_block_is_refused`;
* m4 "translate the deprecated Э5' cost name instead of refusing it" - two homes of one number come
  back, and ``BrokerSpec`` would be reached with a keyword it does not have; breaks
  :func:`test_a_broker_field_that_does_not_exist_is_refused`, whose assertion names the refusal
  (``no such field``) and not only the word ``commission``: a message out of a ``TypeError`` would
  carry that word as well.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import scripts._common as common
import scripts.run_backtest as backtest_face
import scripts.run_optimization as optimize_face
from smc_zero.backtester import BacktestResult, run_backtest
from smc_zero.config import (
    BacktestConfig,
    BiasConfig,
    BrokerSpec,
    DisplacementConfig,
    RiskConfig,
    RunConfig,
    SessionConfig,
    StrategyConfig,
    TimeframeConfig,
    TPConfig,
)
from smc_zero.optimizer import build_tape_marks
from smc_zero.strategy.intents import build_intents

REPO = Path(__file__).resolve().parents[1]
#: The config of the winner of Trial 42 - the file a live run reads (subtask 2.1).
LIVE_CONFIG = REPO / "configs" / "live_eurusd_m15.yaml"
#: The second hierarchy of Э12 (§7.20): M5 entry, M15 structure, H4 + D1 bias.
M5_CONFIG = REPO / "configs" / "live_eurusd_m5.yaml"
#: The first command of the guide: five days of the shipped tape.
FIVE_DAYS = ("2022-08-15", "2022-08-20")
#: The window that separates the defaults from the winner's numbers under the broker-server clock
#: of Э17.B (see the module docstring).
THREE_WEEKS = ("2022-08-15", "2022-09-05")
#: The synthetic tape of Э8': a Monday 00:00 and the 400 closed M15 bars after it.
SYNTHETIC_START = "2026-06-08"
SYNTHETIC_END = "2026-06-15"


def _hand_built_strategy() -> StrategyConfig:
    """Return the strategy of the winner typed by hand: the independent side of the comparison.

    Every number is written here a second time, out of the file: a knob the loader dropped, renamed
    or rounded would make this config and the one :func:`scripts._common.live_inputs` returns
    unequal, which is exactly what the comparison is for.
    """
    return StrategyConfig(
        sweep_buffer_pip=4,
        sl_buffer_pip=10,
        min_fvg_pip=1,
        fvg_lookback=21,
        max_fvg_age_bars=9,
        signal_max_age_bars=94,
        displacement=DisplacementConfig(
            atr_mult_min=0.7453013223207354, body_frac_min=0.44988154849338813
        ),
        bias=BiasConfig(agreement="majority"),
        take_profit=TPConfig(min_tp_rr=1.1220531901171544, rr_fallback=1.6091050846893695),
    )


def _hand_built_broker() -> BrokerSpec:
    """Return the account of the winner typed by hand: the numbers of the shipped row (C6).

    The row of ``configs/live_eurusd_m15.yaml`` is the Alfa-Forex *spread* account of Э11'.2: 1.4
    pips of round-trip spread, 0.2 pips of slippage per market leg and **no commission** - the model
    of a broker that earns on the spread and not on the fee.  The trade of the two-week window pays
    1.40 less than it did while the row carried the 7.00 of Э10'.
    """
    return BrokerSpec(
        spread_pip=1.4,
        commission_per_lot_usd=0.0,
        slippage_pip=0.2,
        swap_long_pip=-0.70,
        swap_short_pip=0.0,
        contract_size=100_000.0,
        pip_size=0.0001,
        leverage=40.0,
    )


def _hand_built_backtest() -> BacktestConfig:
    """Return the run of the winner typed by hand: 10 000 of capital and a 0.1 lot."""
    return BacktestConfig(
        initial_capital=10_000.0,
        risk=RiskConfig(broker=_hand_built_broker(), lot=0.1),
    )


@pytest.fixture
def repository_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point ``./data`` at the tape of the repository, whatever the working directory is."""
    monkeypatch.setattr(common, "DATA_DIR", REPO / "data")


def _summary_profit(text: str) -> float:
    """Return the ``profit`` row of a summary written by ``format_summary``.

    The summary of a run is the page a live trader reads, so the comparison of two runs is written
    as the comparison of two pages and not as a call to the engine a second time.
    """
    match = re.search(r"^\s+profit\s+([-+]?\d+\.\d+)$", text, flags=re.MULTILINE)
    assert match is not None, f"the summary carries no profit row:\n{text}"
    return float(match.group(1))


def _hand_built_profit(start: str, end: str) -> float:
    """Return the profit of the hand-built numbers over one window of the shipped tape, in process.

    The engine is called here and not through a runner: this is the "manual input" side of the
    comparison, built from :func:`_hand_built_strategy` and its neighbours and from nothing else.

    The clock is named explicitly, because the shipped tape of ``./data`` carries **broker server**
    time rather than UTC (Э17.B, SPEC_SMC.md §7.26) and the face of a live run declares it through
    the ``source_clock`` key of the config.  Both sides of this comparison therefore read the same
    clock; leaving this call at the UTC default would compare a normalised run with a shifted one.
    """
    tape = common.load_windowed_tape(
        "EURUSD",
        "M15",
        common.read_day(start),
        common.read_day(end),
        source_clock="broker_server",
    )
    strategy = _hand_built_strategy()
    marks = build_tape_marks(tape, strategy)
    chain = build_intents(tape, marks.bias_frame(strategy.bias.agreement), marks.levels, strategy)
    result = run_backtest(
        tape, chain.intents, _hand_built_backtest(), common.instrument_for("EURUSD")
    )
    return float(result.metrics["profit"])


def _subprocess(module: str, argv: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run one face as a process from ``cwd``: a live run *is* a command line, not a call."""
    path = os.pathsep.join([str(REPO / "src"), str(REPO), os.environ.get("PYTHONPATH", "")])
    return subprocess.run(
        [sys.executable, "-m", module, *argv],
        cwd=str(cwd),
        env={**os.environ, "PYTHONPATH": path},
        capture_output=True,
        text=True,
        check=False,
    )


def _write_synthetic_tape(data_dir: Path, bars: int = 400) -> None:
    """Write the 15 pip random walk of Э8' into ``data_dir``, in the schema of the loader."""
    rng = np.random.default_rng(7)
    close = 1.1000 + np.cumsum(rng.normal(0.0, 0.0015, bars))
    open_ = np.concatenate(([1.1000], close[:-1]))
    frame = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + 0.0007,
            "low": np.minimum(open_, close) - 0.0007,
            "close": close,
        }
    )
    frame.insert(
        0,
        "timestamp",
        pd.date_range(f"{SYNTHETIC_START} 00:00", periods=bars, freq="15min", tz="UTC"),
    )
    frame["datetime"] = frame["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    frame["volume"] = 1
    data_dir.mkdir(parents=True, exist_ok=True)
    frame[["datetime", "open", "high", "low", "close", "volume"]].to_csv(
        data_dir / "EURUSD_M15.csv", index=False
    )


@pytest.mark.parametrize("face", [backtest_face, optimize_face])
def test_the_live_config_of_the_winner_is_the_config_built_by_hand(face: object) -> None:
    """Both faces read the file into exactly the configs a human would type from it."""
    parser = face.build_parser()  # type: ignore[attr-defined]
    args = parser.parse_args(["--config-path", str(LIVE_CONFIG)])

    symbol, timeframe, strategy, backtest = common.live_inputs(args)

    assert (symbol, timeframe) == ("EURUSD", "M15")
    assert strategy == _hand_built_strategy()
    assert backtest == _hand_built_backtest()


def test_the_blocks_are_applied_over_the_defaults_and_not_instead_of_them() -> None:
    """A block sets its own knobs: everything the file does not name keeps the project default."""
    args = backtest_face.build_parser().parse_args(["--config-path", str(LIVE_CONFIG)])

    _, _, strategy, backtest = common.live_inputs(args)

    defaults = StrategyConfig()
    assert strategy.choch_wait_bars == defaults.choch_wait_bars
    assert strategy.min_sl_realistic_pip == defaults.min_sl_realistic_pip
    assert strategy.session == defaults.session
    assert strategy.entry == defaults.entry
    assert strategy.levels == defaults.levels
    assert strategy.liquidity == defaults.liquidity
    assert backtest.risk.deposit == RiskConfig().deposit
    assert backtest.risk.risk_pct == RiskConfig().risk_pct
    assert backtest.limit_valid_bars == BacktestConfig().limit_valid_bars
    assert backtest.session == BacktestConfig().session


def test_a_run_without_a_config_keeps_the_project_defaults() -> None:
    """Э8' behaviour: no flag, no file - the pair and the configs of the project."""
    args = backtest_face.build_parser().parse_args([])

    symbol, timeframe, strategy, backtest = common.live_inputs(args)

    assert (symbol, timeframe) == (common.DEFAULT_SYMBOL, common.DEFAULT_TIMEFRAME)
    assert strategy == StrategyConfig()
    assert backtest == BacktestConfig()


def test_the_entry_timeframe_scales_the_sharpe_of_the_run() -> None:
    """An M5 run reads 288 bars a day: the scale follows the entry tape, not the M15 default.

    The Sharpe ratio of a run is per bar (:attr:`BacktestConfig.sharpe_bars_per_day`), so an M5
    report scaled by 96 would understate every volatility it prints by ``sqrt(3)``.
    """
    parser = backtest_face.build_parser()

    _, timeframe, _, backtest = common.live_inputs(parser.parse_args(["--hierarchy", "H4_M15_M5"]))

    assert timeframe == "M5"
    assert backtest.sharpe_bars_per_day == 288
    assert common.live_inputs(parser.parse_args(["--timeframe", "M15"]))[3] == BacktestConfig()


def test_an_argument_wins_over_the_config() -> None:
    """The order of §7.19: what the caller typed, then the file, then the default of Э8'.

    The entry timeframe is not free any more: it belongs to the hierarchy (§7.20), so the argument
    that wins here is the pair of them - a typed ``--hierarchy`` with the ``--timeframe`` it names.
    """
    args = backtest_face.build_parser().parse_args(
        [
            "--config-path",
            str(LIVE_CONFIG),
            "--symbol",
            "GBPUSD",
            "--hierarchy",
            "H4_M15_M5",
            "--timeframe",
            "M5",
        ]
    )

    symbol, timeframe, strategy, _ = common.live_inputs(args)

    assert (symbol, timeframe) == ("GBPUSD", "M5")
    # the hierarchy owns the bias frames (§7.20), so the hand-built strategy carries the M5 set
    hand = _hand_built_strategy()
    expected = replace(hand, bias=replace(hand.bias, timeframes=("H4", "D1")))
    assert strategy == expected, "the file still fills the numbers it is asked for"
    assert common.resolve_selector(None, {"symbol": "gbpusd"}, "symbol", "EURUSD") == "GBPUSD"
    assert common.resolve_selector(None, None, "symbol", "EURUSD") == "EURUSD"
    assert common.resolve_selector("eurusd", None, "symbol", "GBPUSD") == "EURUSD"


def test_the_hierarchy_preset_sets_all_three_timeframes() -> None:
    """The M5 preset of §7.20 is one decision: entry M5, working structure M15, context H4."""
    args = backtest_face.build_parser().parse_args(["--hierarchy", "H4_M15_M5"])

    _, timeframe, _, backtest = common.live_inputs(args)

    # the preset names the entry frame itself - no ``--timeframe`` was typed or needed
    assert timeframe == "M5"
    assert backtest.timeframes == TimeframeConfig(ltf="M5", mtf="M15", htf="H4")
    assert backtest.sharpe_bars_per_day == 288
    # the v1 preset is the default and stays untouched bit for bit
    assert common.HIERARCHY_PRESETS["D1_H1_M15"].timeframes == TimeframeConfig()
    assert common.HIERARCHY_PRESETS["D1_H1_M15"].structure is None
    assert common.resolve_hierarchy(None, None) == common.DEFAULT_HIERARCHY
    assert common.resolve_hierarchy("h4_m15_m5", None) == "H4_M15_M5"


def test_the_hierarchy_preset_sets_the_bias_timeframes() -> None:
    """Р1 of §7.20: the M5 hierarchy asks H4 + D1 for a direction; the v1 set stays the default."""
    parser = backtest_face.build_parser()

    _, _, strategy, _ = common.live_inputs(parser.parse_args(["--hierarchy", "H4_M15_M5"]))

    assert strategy.bias.timeframes == ("H4", "D1")
    # the v1 run keeps exactly the configuration it had
    assert common.live_inputs(parser.parse_args([]))[2] == StrategyConfig()


def test_the_file_keeps_the_bias_knobs_it_may_set() -> None:
    """The preset owns the set of frames; ``agreement`` and the swing settings stay the file's."""
    args = backtest_face.build_parser().parse_args(["--config-path", str(LIVE_CONFIG)])

    _, _, strategy, _ = common.live_inputs(args)

    assert strategy.bias.agreement == _hand_built_strategy().bias.agreement
    assert strategy.bias.timeframes == common.HIERARCHY_PRESETS["D1_H1_M15"].bias


def test_the_run_block_sets_warmup_and_window_limits() -> None:
    """The ``run`` block of §7.20 names the window, the warm-up and the two engine counters."""
    cfg = common.load_config(M5_CONFIG)
    run = common.run_from_config(cfg)
    args = backtest_face.build_parser().parse_args([])

    window = common.resolve_window(args, run)

    assert run.warmup_days == 60
    assert (window.start, window.end) == (
        pd.Timestamp("2025-05-06", tz="UTC"),
        pd.Timestamp("2026-09-22", tz="UTC"),
    )
    assert window.loaded_from == pd.Timestamp("2025-03-07", tz="UTC")
    backtest = common.backtest_from_config(cfg, run)
    assert backtest.limit_valid_bars == 30
    assert backtest.max_bars_per_trade == 1500
    # a file without the block keeps the shipped window of Э8' and no warm-up at all
    fallback = common.resolve_window(backtest_face.build_parser().parse_args([]), RunConfig())
    assert (fallback.start, fallback.end) == (
        common.read_day(common.DEFAULT_START),
        common.read_day(common.DEFAULT_END),
    )
    assert fallback.warmup_days == 0


def test_an_argument_still_wins_over_the_run_block() -> None:
    """The order of §7.19 holds for the window too: a typed day outranks the file's."""
    args = backtest_face.build_parser().parse_args(["--start", "2025-06-01"])
    run = common.run_from_config(common.load_config(M5_CONFIG))

    window = common.resolve_window(args, run)

    assert window.start == pd.Timestamp("2025-06-01", tz="UTC")
    assert window.end == pd.Timestamp("2026-09-22", tz="UTC")


def test_the_m5_config_loads_with_correct_bias_and_windows() -> None:
    """The second hierarchy reads its file: M5 entry, M15 structure, H4 + D1 bias, 60 day warm-up."""
    args = backtest_face.build_parser().parse_args(["--config-path", str(M5_CONFIG)])

    symbol, timeframe, strategy, backtest = common.live_inputs(args)

    assert (symbol, timeframe) == ("EURUSD", "M5")
    assert strategy.bias.timeframes == ("H4", "D1")
    assert strategy.fvg_ready_bars == 6
    assert backtest.timeframes == TimeframeConfig(ltf="M5", mtf="M15", htf="H4")
    assert backtest.sharpe_bars_per_day == 288
    assert backtest.limit_valid_bars == 30
    window = common.resolve_window(args)
    assert window.warmup_days == 60
    assert (window.start, window.end) == (
        pd.Timestamp("2025-05-06", tz="UTC"),
        pd.Timestamp("2026-09-22", tz="UTC"),
    )


def test_a_mismatch_between_timeframe_and_hierarchy_is_refused(tmp_path: Path) -> None:
    """A file that names the M5 hierarchy and an M15 entry is a config error, not a silent run."""
    cfg = common.load_config(LIVE_CONFIG)
    cfg[common.HIERARCHY_KEY] = "H4_M15_M5"
    path = tmp_path / "m5_hierarchy_m15_entry.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(ValueError, match="requires timeframe 'M5'"):
        common.live_inputs(backtest_face.build_parser().parse_args(["--config-path", str(path)]))


def test_m5_without_a_hierarchy_is_refused() -> None:
    """An M5 tape under the default M15 preset would re-price every bar: it is refused."""
    args = backtest_face.build_parser().parse_args(["--timeframe", "M5"])

    with pytest.raises(ValueError, match="needs an explicit hierarchy"):
        common.live_inputs(args)


def test_a_hierarchy_preset_that_does_not_exist_is_refused() -> None:
    """A typo in the preset name may not fall back to the default hierarchy (rule 5)."""
    args = backtest_face.build_parser().parse_args(["--hierarchy", "M1_M5_M15"])

    with pytest.raises(ValueError, match="no such preset"):
        common.live_inputs(args)


def test_the_pair_of_the_config_is_used_when_the_caller_types_none(tmp_path: Path) -> None:
    """No argument default may outrank the file: ``--symbol`` / ``--timeframe`` default to ``None``.

    The check is written on a config of *another* pair on purpose.  A default of Э8' left in the
    parser would be indistinguishable from a typed argument whenever the file happens to name the
    same pair - which is exactly the case the shipped config is in, so this test names another one
    and the mutation "default the symbol again" has to go red here.  The pair has to be a whole
    hierarchy as well (§7.20), hence the preset key beside the two values it stands for.
    """
    cfg = common.load_config(LIVE_CONFIG)
    cfg["symbol"] = "GBPUSD"
    cfg["timeframe"] = "M5"
    cfg[common.HIERARCHY_KEY] = "H4_M15_M5"
    path = tmp_path / "gbpusd_m5.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    args = backtest_face.build_parser().parse_args(["--config-path", str(path)])
    symbol, timeframe, _, _ = common.live_inputs(args)

    assert args.symbol is None and args.timeframe is None
    assert (symbol, timeframe) == ("GBPUSD", "M5")


def test_a_missing_file_is_refused_as_the_missing_file_it_is() -> None:
    """A path that holds nothing is a :class:`FileNotFoundError`, not an empty config."""
    with pytest.raises(FileNotFoundError):
        common.load_config(REPO / "configs" / "no_such_config.yaml")


def test_the_loaded_config_is_the_mapping_of_the_file() -> None:
    """The loader returns the blocks of the file as plain data, one key per block."""
    cfg = common.load_config(LIVE_CONFIG)

    assert set(common.CONFIG_KEYS) <= set(cfg)
    assert cfg["symbol"] == "EURUSD"
    assert cfg["timeframe"] == "M15"
    assert cfg["strategy"]["take_profit"]["rr_fallback"] == pytest.approx(1.6091050846893695)


def test_a_config_without_a_mandatory_block_is_refused(tmp_path: Path) -> None:
    """A file missing a block is refused by name: the missing half may not be guessed (rule 5)."""
    cfg = common.load_config(LIVE_CONFIG)

    for key in common.CONFIG_KEYS:
        path = tmp_path / f"without_{key}.yaml"
        path.write_text(
            yaml.safe_dump({name: value for name, value in cfg.items() if name != key}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=key):
            common.load_config(path)


def test_a_text_that_does_not_parse_is_refused(tmp_path: Path) -> None:
    """A broken YAML is a :class:`ValueError` naming the file, not a traceback from the library."""
    path = tmp_path / "broken.yaml"
    path.write_text("symbol: [EURUSD\n", encoding="utf-8")

    with pytest.raises(ValueError, match="broken.yaml"):
        common.load_config(path)


def test_a_document_that_is_not_a_mapping_is_refused(tmp_path: Path) -> None:
    """A list of words is a readable YAML and still no config: the blocks have names."""
    path = tmp_path / "list.yaml"
    path.write_text("- EURUSD\n- M15\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        common.load_config(path)


@pytest.mark.parametrize("key", ["strategy", "broker", "backtest"])
def test_a_block_of_another_shape_is_refused(tmp_path: Path, key: str) -> None:
    """Every block is a mapping of named numbers; a list inside one is refused by its key."""
    cfg = common.load_config(LIVE_CONFIG)
    cfg[key] = [1, 2]
    path = tmp_path / f"{key}_is_a_list.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(ValueError, match=key):
        common.live_inputs(backtest_face.build_parser().parse_args(["--config-path", str(path)]))


def test_a_strategy_knob_that_does_not_exist_is_refused(tmp_path: Path) -> None:
    """A typo in a knob of the strategy may not be skipped silently (rules 1 and 5)."""
    cfg = common.load_config(LIVE_CONFIG)
    cfg["strategy"]["sweep_buffer"] = 4
    path = tmp_path / "typo.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(ValueError, match="sweep_buffer"):
        common.live_inputs(backtest_face.build_parser().parse_args(["--config-path", str(path)]))


def test_the_backtest_face_refuses_a_config_it_cannot_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A path that holds no config is one line on stderr and code 2, and no report is written."""
    report_dir = tmp_path / "reports"

    code = backtest_face.main(
        ["--config-path", str(tmp_path / "missing.yaml"), "--report-dir", str(report_dir)]
    )

    assert code == 2
    assert "error:" in capsys.readouterr().err
    assert not report_dir.exists()


def test_the_backtest_face_runs_a_live_config_on_five_days(
    tmp_path: Path, repository_data: None
) -> None:
    """The first command of the guide: five days of the shipped tape, code 0, both files.

    The pair of the report folder is the config's (``EURUSD M15``) and not an argument, because the
    caller typed none: that is the order of §7.19, pinned on the name of the folder itself.
    """
    report_dir = tmp_path / "reports"
    start, end = FIVE_DAYS

    call = _subprocess(
        "scripts.run_backtest",
        [
            "--config-path",
            str(LIVE_CONFIG),
            "--start",
            start,
            "--end",
            end,
            "--report-dir",
            str(report_dir),
        ],
        cwd=REPO,
    )

    assert call.returncode == 0, call.stderr
    folder = report_dir / f"backtest_EURUSD_M15_{start}_{end}"
    assert (folder / "trades.csv").is_file()
    assert (folder / "summary.txt").is_file()
    profit = _summary_profit((folder / "summary.txt").read_text(encoding="utf-8"))
    assert profit == pytest.approx(_hand_built_profit(start, end), abs=0.01)


def test_the_live_config_of_the_winner_pays_what_the_hand_built_numbers_pay(
    tmp_path: Path, repository_data: None
) -> None:
    """Three weeks of the shipped tape: the run of the file and the run of the numbers agree.

    The window is the one that separates the two sides (module docstring): the winner's numbers
    settle above the defaults there, so a config the runner forgot to apply cannot pass this test
    by accident - the run would report the reading of the defaults and not the one of the file.
    """
    report_dir = tmp_path / "reports"
    start, end = THREE_WEEKS

    live_code = backtest_face.main(
        [
            "--config-path",
            str(LIVE_CONFIG),
            "--start",
            start,
            "--end",
            end,
            "--report-dir",
            str(report_dir),
        ]
    )

    assert live_code == 0
    folder = report_dir / f"backtest_EURUSD_M15_{start}_{end}"
    trades = pd.read_csv(folder / "trades.csv")
    assert len(trades) >= 1, "the window carries no setup any more: pick another window"
    live = _summary_profit((folder / "summary.txt").read_text(encoding="utf-8"))
    hand = _hand_built_profit(start, end)
    assert live == pytest.approx(hand, abs=0.01)

    # The defaults are read on the very clock of the run above: the difference has to come from the
    # numbers of the file and not from a clock the caller forgot to declare (Э17.B).
    defaults_code = backtest_face.main(
        [
            "--start",
            start,
            "--end",
            end,
            "--source-clock",
            "broker_server",
            "--report-dir",
            str(report_dir / "defaults"),
        ]
    )

    assert defaults_code == 0
    defaults = _summary_profit(
        (
            report_dir / "defaults" / f"backtest_EURUSD_M15_{start}_{end}" / "summary.txt"
        ).read_text(encoding="utf-8")
    )
    assert defaults != pytest.approx(hand, abs=0.01), "the config was ignored: the defaults paid it"


def test_the_optimization_face_runs_a_live_config_on_a_synthetic_tape(tmp_path: Path) -> None:
    """Two trials of Э8' seeded by the config of the winner: four files and the config's pair."""
    _write_synthetic_tape(tmp_path / "data")
    report_dir = tmp_path / "reports"

    call = _subprocess(
        "scripts.run_optimization",
        [
            "--config-path",
            str(LIVE_CONFIG),
            "--start",
            SYNTHETIC_START,
            "--end",
            SYNTHETIC_END,
            "--n-trials",
            "2",
            "--min-train-bars",
            "100",
            "--test-period-bars",
            "100",
            "--report-dir",
            str(report_dir),
        ],
        cwd=tmp_path,
    )

    assert call.returncode == 0, call.stderr
    folder = report_dir / f"optimization_EURUSD_M15_{SYNTHETIC_START}_{SYNTHETIC_END}_n2"
    for name in ("best_params.json", "trades.csv", "summary.txt", "fold_metrics.csv"):
        assert (folder / name).is_file(), name
    payload = json.loads((folder / "best_params.json").read_text(encoding="utf-8"))
    assert payload["symbol"] == "EURUSD"
    assert payload["timeframe"] == "M15"
    assert payload["n_trials"] == 2
    assert payload["folds"] >= 1
    assert payload["best_params"]


def test_a_broker_field_that_does_not_exist_is_refused(tmp_path: Path) -> None:
    """The deprecated Э5' names are gone: ``commission`` is not a field of the profile any more."""
    cfg = common.load_config(LIVE_CONFIG)
    cfg["broker"]["commission"] = 7.0
    path = tmp_path / "legacy_broker.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(ValueError, match=r"no such field\(s\) commission"):
        common.live_inputs(backtest_face.build_parser().parse_args(["--config-path", str(path)]))


def test_a_run_field_that_does_not_exist_is_refused(tmp_path: Path) -> None:
    """A live config sets the capital and the lot of a run, and nothing else of the backtester."""
    cfg = common.load_config(LIVE_CONFIG)
    cfg["backtest"]["n_trials"] = 5
    path = tmp_path / "run_typo.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(ValueError, match="n_trials"):
        common.live_inputs(backtest_face.build_parser().parse_args(["--config-path", str(path)]))


# --- the session block of Э17.z: the killzone table from the file -------------
#: The control window of the Э17.z check: four months of the shipped tape that carry setups under
#: the numbers of the winner (measured on 2026-10-13: 17 intents, 11 trades, +66.60, about a second).
CONTROL_WINDOW = ("2026-03-01", "2026-06-30")
#: The block that names the shipped windows.  YAML has no tuple, so a window is a two-item list -
#: and the block is the only way a window leaves the code.
DEFAULT_SESSION_BLOCK: dict[str, object] = {
    "use_kz": True,
    "prelondon": False,
    "prelondon_msk": [7, 9],
    "london_summer_msk": [9, 12],
    "london_winter_msk": [10, 13],
    "ny_summer_msk": [14, 17],
    "ny_winter_msk": [15, 18],
    "session_open_msk": [2, 0],
    "session_close_msk": [23, 55],
    "research_ignore_sessions": False,
}


def _config_with_session(tmp_path: Path, name: str, session: dict[str, object]) -> Path:
    """Write the live config of the winner plus one ``session`` block and return its path."""
    cfg = common.load_config(LIVE_CONFIG)
    cfg[common.SESSION_BLOCK] = dict(session)
    path = tmp_path / name
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _shifted_session(hours: int) -> dict[str, object]:
    """Return the shipped table moved by ``hours``: the experiment Э17.z is asking about."""
    block = dict(DEFAULT_SESSION_BLOCK)
    for name in ("london_summer_msk", "london_winter_msk", "ny_summer_msk", "ny_winter_msk"):
        start, end = block[name]  # type: ignore[misc]
        block[name] = [int(start) + hours, int(end) + hours]
    return block


def _window_run(
    strategy: StrategyConfig, backtest: BacktestConfig, start: str, end: str
) -> tuple[int, BacktestResult]:
    """Return the intent count and the result of one window, built in process from two configs."""
    tape = common.load_windowed_tape(
        "EURUSD", "M15", common.read_day(start), common.read_day(end), source_clock="broker_server"
    )
    marks = build_tape_marks(tape, strategy)
    chain = build_intents(tape, marks.bias_frame(strategy.bias.agreement), marks.levels, strategy)
    result = run_backtest(tape, chain.intents, backtest, common.instrument_for("EURUSD"))
    return len(chain.intents), result



def test_the_session_block_of_the_defaults_reads_the_shipped_table(tmp_path: Path) -> None:
    """A block that names the shipped windows answers the shipped :class:`SessionConfig`.

    The block is read by :func:`scripts._common.session_from_config` and lands in *both* configs:
    the entry chain and the Э3' level maps read the strategy's copy, the engine's broker-hours mask
    reads the run's one, and two tables in one run would let a setup arm where placement is refused.
    """
    typed_path = _config_with_session(tmp_path, "session_defaults.yaml", DEFAULT_SESSION_BLOCK)

    _, _, strategy, backtest = common.live_inputs(
        backtest_face.build_parser().parse_args(["--config-path", str(typed_path)])
    )

    assert strategy.session == SessionConfig()
    assert backtest.session == SessionConfig()
    plain = common.live_inputs(
        backtest_face.build_parser().parse_args(["--config-path", str(LIVE_CONFIG)])
    )
    assert strategy == plain[2]
    assert backtest == plain[3]


def test_the_control_window_pays_the_same_with_and_without_the_session_block(
    tmp_path: Path, repository_data: None
) -> None:
    """Э17.z: naming the shipped table changes neither the intents nor a single trade.

    The comparison is measured on the tape and not only on the equality of two configs: the block
    travels through :func:`scripts._common.live_inputs`, the chain and the engine, so a conversion
    that rounded a window, swapped its ends or read it as UTC hours would move the trades here.
    """
    typed_path = _config_with_session(tmp_path, "session_defaults.yaml", DEFAULT_SESSION_BLOCK)
    plain = common.live_inputs(
        backtest_face.build_parser().parse_args(["--config-path", str(LIVE_CONFIG)])
    )
    typed = common.live_inputs(
        backtest_face.build_parser().parse_args(["--config-path", str(typed_path)])
    )

    intents_plain, result_plain = _window_run(plain[2], plain[3], *CONTROL_WINDOW)
    intents_typed, result_typed = _window_run(typed[2], typed[3], *CONTROL_WINDOW)

    assert intents_plain >= 1, "the window carries no setup any more: pick another window"
    assert intents_typed == intents_plain
    pd.testing.assert_frame_equal(result_typed.trades, result_plain.trades)


def test_a_shifted_session_block_moves_the_selection(tmp_path: Path, repository_data: None) -> None:
    """The control example of Э17.z: a table one hour later selects another day of trades.

    Measured on the shipped tape on 2026-10-13: the shipped table arms 17 intents and pays 11
    trades on the control window, the table one hour later arms 21 and pays 15.  The numbers are
    pinned so that a block the runner silently ignored cannot pass this test - the whole point of
    the block is that moving it moves the selection.
    """
    shifted_path = _config_with_session(tmp_path, "session_shift_plus1.yaml", _shifted_session(1))

    _, _, strategy, backtest = common.live_inputs(
        backtest_face.build_parser().parse_args(["--config-path", str(shifted_path)])
    )

    assert strategy.session.london_summer_msk == (10, 13)
    assert strategy.session.ny_summer_msk == (15, 18)

    intents_plain, result_plain = _window_run(
        _hand_built_strategy(), _hand_built_backtest(), *CONTROL_WINDOW
    )
    intents_shifted, result_shifted = _window_run(strategy, backtest, *CONTROL_WINDOW)

    assert (intents_plain, len(result_plain.trades)) == (17, 11)
    assert (intents_shifted, len(result_shifted.trades)) == (21, 15)


def test_the_research_switch_opens_the_gate_of_a_run(tmp_path: Path, repository_data: None) -> None:
    """``research_ignore_sessions`` is how the hour filter is measured: it carries more setups."""
    path = _config_with_session(
        tmp_path, "session_ignore.yaml", {**DEFAULT_SESSION_BLOCK, "research_ignore_sessions": True}
    )

    _, _, strategy, backtest = common.live_inputs(
        backtest_face.build_parser().parse_args(["--config-path", str(path)])
    )

    assert strategy.session.research_ignore_sessions is True

    intents_plain, _ = _window_run(_hand_built_strategy(), _hand_built_backtest(), *CONTROL_WINDOW)
    intents_open, result_open = _window_run(strategy, backtest, *CONTROL_WINDOW)

    assert intents_open > intents_plain
    assert len(result_open.trades) > 11


@pytest.mark.parametrize("path", [LIVE_CONFIG, M5_CONFIG])
def test_a_live_config_never_ignores_the_sessions(path: Path) -> None:
    """The research switch of Э17.z is not a live one: neither shipped file may carry it."""
    assert common.session_from_config(common.load_config(path)).research_ignore_sessions is False


def test_a_session_field_that_does_not_exist_is_refused(tmp_path: Path) -> None:
    """A typo in the table is refused with the list of the settable fields (rule 5)."""
    block = {**DEFAULT_SESSION_BLOCK, "london_evening_msk": [18, 20]}
    path = _config_with_session(tmp_path, "session_typo.yaml", block)

    with pytest.raises(ValueError, match="london_evening_msk"):
        common.live_inputs(backtest_face.build_parser().parse_args(["--config-path", str(path)]))


@pytest.mark.parametrize("window", ["9-12", [9], [9, 12, 15], [9.5, 12]])
def test_a_window_that_is_not_a_pair_of_hours_is_refused(tmp_path: Path, window: object) -> None:
    """A window is a pair of whole hours: a string or a fraction would reach a field unpacking two."""
    block = {**DEFAULT_SESSION_BLOCK, "london_summer_msk": window}
    path = _config_with_session(tmp_path, "session_window.yaml", block)

    with pytest.raises(ValueError, match="london_summer_msk"):
        common.live_inputs(backtest_face.build_parser().parse_args(["--config-path", str(path)]))


def test_a_window_out_of_the_day_is_refused(tmp_path: Path) -> None:
    """``SessionConfig`` owns the check: ``[12, 9]`` is not a half-open window."""
    block = {**DEFAULT_SESSION_BLOCK, "london_summer_msk": [12, 9]}
    path = _config_with_session(tmp_path, "session_backwards.yaml", block)

    with pytest.raises(ValueError, match="london_summer_msk"):
        common.live_inputs(backtest_face.build_parser().parse_args(["--config-path", str(path)]))


def test_a_switch_that_is_not_a_switch_is_refused(tmp_path: Path) -> None:
    """``use_kz: 1`` is not a switch: a number must not reach a boolean field of the table."""
    block = {**DEFAULT_SESSION_BLOCK, "use_kz": 1}
    path = _config_with_session(tmp_path, "session_switch.yaml", block)

    with pytest.raises(ValueError, match="use_kz"):
        common.live_inputs(backtest_face.build_parser().parse_args(["--config-path", str(path)]))


def test_a_session_block_without_a_field_keeps_the_shipped_value() -> None:
    """A block sets its own windows: every field it does not name keeps the shipped value."""
    session = common.session_from_config({common.SESSION_BLOCK: {"london_summer_msk": [8, 11]}})

    assert session.london_summer_msk == (8, 11)
    assert session.london_winter_msk == SessionConfig().london_winter_msk
    assert session.session_close_msk == SessionConfig().session_close_msk
