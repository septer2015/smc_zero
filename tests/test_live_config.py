"""Live-config tests of Э10'.2: the loader, the two blocks and the ``--config-path`` flag.

The config of the winner is a *file*, so what is pinned here is the road it travels: the loader
that reads it, the mapping of its three blocks onto :class:`~smc_zero.config.StrategyConfig` and
:class:`~smc_zero.config.BacktestConfig`, the priority of the command line over the file, and the
two faces that consume it.  The expectation of every comparison is written down *again* here, by
hand and out of the file it checks (``_hand_built_strategy`` and its two neighbours): a test that
read its expected numbers from the same YAML would only prove that the YAML parses.

The two windows of the end-to-end tests are the shipped ``./data/EURUSD_M15.csv``, because a live
config is a statement about the real tape.  Measured on 2026-10-05 on the shipped account - the
Alfa-Forex spread account of Э11'.2: 1.4 pips of round-trip spread, 0.2 pips of slippage per market
leg and no commission:

* five days (``2022-08-15 .. 2022-08-20``, 444 bars) carry **no setup at all** - neither under the
  defaults nor under the winner's numbers - so that window pins the command line (flag accepted,
  exit code 0, both report files written) and the equivalence of the config with the hand-built
  numbers is measured on the longer window below;
* two weeks (``2022-08-15 .. 2022-08-29``, 996 bars) carry a setup under the winner's numbers only:
  the defaults finish at ``0 trades / +0.00`` and the winner at ``1 trade / -25.25``, so the two
  runs are distinguishable and the comparison is not vacuous.  The single trade pays 1.40 less than
  it did while the row carried the 7.00 of Э10', and that is the whole of the difference: 0.1 lot
  over two market legs at 7.00 per lot is 1.40.

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
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import scripts._common as common
import scripts.run_backtest as backtest_face
import scripts.run_optimization as optimize_face
from smc_zero.backtester import run_backtest
from smc_zero.config import (
    BacktestConfig,
    BiasConfig,
    BrokerSpec,
    DisplacementConfig,
    RiskConfig,
    StrategyConfig,
    TPConfig,
)
from smc_zero.optimizer import build_tape_marks
from smc_zero.strategy.intents import build_intents

REPO = Path(__file__).resolve().parents[1]
#: The config of the winner of Trial 42 - the file a live run reads (subtask 2.1).
LIVE_CONFIG = REPO / "configs" / "live_eurusd_m15.yaml"
#: The first command of the guide: five days of the shipped tape.
FIVE_DAYS = ("2022-08-15", "2022-08-20")
#: The window that separates the defaults from the winner's numbers (see the module docstring).
TWO_WEEKS = ("2022-08-15", "2022-08-29")
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
    """
    tape = common.load_windowed_tape("EURUSD", "M15", common.read_day(start), common.read_day(end))
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


def test_an_argument_wins_over_the_config() -> None:
    """The order of §7.19: what the caller typed, then the file, then the default of Э8'."""
    args = backtest_face.build_parser().parse_args(
        ["--config-path", str(LIVE_CONFIG), "--symbol", "GBPUSD", "--timeframe", "H1"]
    )

    symbol, timeframe, strategy, _ = common.live_inputs(args)

    assert (symbol, timeframe) == ("GBPUSD", "H1")
    assert strategy == _hand_built_strategy(), "the file still fills the numbers it is asked for"
    assert common.resolve_selector(None, {"symbol": "gbpusd"}, "symbol", "EURUSD") == "GBPUSD"
    assert common.resolve_selector(None, None, "symbol", "EURUSD") == "EURUSD"
    assert common.resolve_selector("eurusd", None, "symbol", "GBPUSD") == "EURUSD"


def test_the_pair_of_the_config_is_used_when_the_caller_types_none(tmp_path: Path) -> None:
    """No argument default may outrank the file: ``--symbol`` / ``--timeframe`` default to ``None``.

    The check is written on a config of *another* pair on purpose.  A default of Э8' left in the
    parser would be indistinguishable from a typed argument whenever the file happens to name the
    same pair - which is exactly the case the shipped config is in, so this test names another one
    and the mutation "default the symbol again" has to go red here.
    """
    cfg = common.load_config(LIVE_CONFIG)
    cfg["symbol"] = "GBPUSD"
    cfg["timeframe"] = "H1"
    path = tmp_path / "gbpusd_h1.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    args = backtest_face.build_parser().parse_args(["--config-path", str(path)])
    symbol, timeframe, _, _ = common.live_inputs(args)

    assert args.symbol is None and args.timeframe is None
    assert (symbol, timeframe) == ("GBPUSD", "H1")


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
    """Two weeks of the shipped tape: the run of the file and the run of the numbers agree.

    The window is the one that separates the two sides (module docstring): the winner's numbers arm
    a setup the defaults never see, so a config the runner forgot to apply cannot pass this test by
    accident - the run would report the reading of the defaults and not the one of the file.
    """
    report_dir = tmp_path / "reports"
    start, end = TWO_WEEKS

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

    defaults_code = backtest_face.main(
        ["--start", start, "--end", end, "--report-dir", str(report_dir / "defaults")]
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
