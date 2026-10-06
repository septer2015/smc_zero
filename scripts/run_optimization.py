"""``smc-optimize``: an Optuna study over a walk-forward, then the winner's report (Э8').

The runner is the search face of the project:

.. code-block:: text

    smc-optimize --symbol EURUSD --n-trials 100 --jobs 4

(without an install, from the repository root::

    PYTHONPATH=src python -m scripts.run_optimization --symbol EURUSD --n-trials 10

)

It loads the tape, cuts the requested window, builds the markup cache **once** (the Э7' invariant:
a study pays for the bias frames and the level book one time, not once per trial), runs the study of
SPEC_SMC.md §7.11 (:func:`~smc_zero.optimizer.optimize.run_optimization`) over ``--n-trials``
parameter sets, prints the ten best finished trials and then the winner, and re-runs that winner as
a single backtest over the whole window.  Four files land in
``./reports/optimization_<symbol>_<tf>_<start>_<end>_n<t>/``:

* ``best_params.json`` - the winner's parameters, its study score, the two weights that produced it
  (``score_metric`` / ``penalty_power``), the fold counts and the aggregates of both windows: the
  audit line of the run, and the only place the in-sample numbers are written down;
* ``trades.csv`` / ``trades.parquet`` - the trade log of the winner's full-window run;
* ``summary.txt`` - the metric table of that run (the costs of rule 4 included);
* ``fold_metrics.csv`` - one long table: ``fold`` / ``window`` / the six fields of
  :data:`~smc_zero.backtester.FOLD_METRIC_FIELDS`, two rows per fold (its fit window and its
  out-of-sample window) and four summary rows labelled ``mean`` and ``std``; the ``mean`` rows
  additionally carry the three score inputs of :data:`SCORE_COLUMNS`.

The winner is *reported* on the full window on purpose: the fold tables say what the score was
computed on, the summary says what those parameters do over the whole tape with all their trades.
Both are printed as they are computed - no number is recomputed here (rules 4 and 5).

A window that cannot hold a single fold is refused before the study starts, with the arithmetic on
stderr (§7.10 п.61: ``(bars - min_train_bars) // test_period_bars`` folds); so is a missing tape and
an unpriced symbol.  optuna itself is optional: without the ``optimize`` extra the runner says so
instead of failing with an import traceback.

A ``--config-path`` file (§7.19) seeds the study instead of the project defaults: its ``strategy``
block is the base configuration the trial parameters are applied to, its ``broker`` block is the
account both windows are charged on and its ``backtest`` block is the run of the winner.  The file
does not narrow the search space - ``--ranges`` does (§7.22).

Two profiles make the runner serve both hierarchies without a second script:

* ``--ranges`` picks the space a trial moves: ``default`` is the whole Э7' space of §7.11,
  ``m5`` the narrowed five-knob space of §7.22, and ``auto`` (the default) reads the *entry
  timeframe* of the run - an M5 tape gets ``m5``, everything else keeps ``default``;
* ``--score`` picks how a trial is ranked: ``default`` is the out-of-sample score of §7.11,
  ``m5`` the pooled trade-count score ``pf_pool * min(1, trades / target)`` of §7.22 with the
  numbers of :class:`~smc_zero.config.TradeTargetScore` (the ``--target-trades`` / ``--min-trades``
  pair and the shipped density and pool gates of that dataclass), and ``auto`` follows the same
  timeframe rule.

If a ``--ranges`` profile names a knob the cache was built from, the run is refused before the study
starts (the guard of :func:`~smc_zero.optimizer.cache_mismatches`), so a profile cannot silently
score trials against a stale markup.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any

import pandas as pd

from scripts import _common
from smc_zero.backtester import (
    FOLD_METRIC_FIELDS,
    default_walk_forward,
    export_trades,
    format_summary,
    run_backtest,
    split_walkforward,
)
from smc_zero.config import (
    DEFAULT_MIN_TRADES,
    DEFAULT_TARGET_TRADES,
    OptunaConfig,
    TradeTargetScore,
    WalkForwardConfig,
)
from smc_zero.optimizer import (
    PARAM_PROFILES,
    FoldEvaluation,
    OptunaResult,
    ParamRange,
    Scorer,
    build_tape_marks,
    degradation_factor,
    run_optimization,
    trades_pool_scorer,
)
from smc_zero.strategy.intents import build_intents

#: How many finished trials the console prints, best first.
TOP_TRIALS = 10
#: The search spaces a run may name (§7.22): ``auto`` follows the entry timeframe, ``default`` and
#: ``m5`` name the two profiles of :data:`~smc_zero.optimizer.PARAM_PROFILES` explicitly.
RANGE_PROFILES: tuple[str, ...] = ("auto", "default", "m5")
#: The scorers a run may name: ``auto`` follows the entry timeframe as well, ``default`` is the
#: out-of-sample score of §7.11 and ``m5`` the trade-count score of §7.22.
SCORE_PROFILES: tuple[str, ...] = ("auto", "default", "m5")
#: The score inputs of §7.11 п.69 that ``fold_metrics.csv`` repeats on its ``mean`` rows: the
#: aggregate profit of each window and the ratio the decay gate weighs.
SCORE_COLUMNS: tuple[str, ...] = ("train_profit_mean", "test_profit_mean", "degradation_ratio")


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of ``smc-optimize``."""
    parser = argparse.ArgumentParser(
        prog="smc-optimize",
        description=(
            "Search the SMC parameters of SPEC_SMC.md §7.11 over ./data and report the winner."
        ),
    )
    _common.add_window_arguments(parser)
    _common.add_config_argument(parser)
    _common.add_data_source_argument(parser)
    parser.add_argument("--n-trials", type=int, default=100, help="parameter sets to evaluate")
    parser.add_argument("--jobs", type=int, default=1, help="parallel trials of the study")
    parser.add_argument("--seed", type=int, default=42, help="seed of the TPE sampler")
    parser.add_argument(
        "--min-train-bars",
        type=int,
        default=None,
        help="fit window of the first fold, in bars (default: WalkForwardConfig)",
    )
    parser.add_argument(
        "--test-period-bars",
        type=int,
        default=None,
        help="length of a fold's out-of-sample window, in bars (default: WalkForwardConfig)",
    )
    parser.add_argument(
        "--ranges",
        choices=RANGE_PROFILES,
        default="auto",
        help=(
            "search space: 'auto' follows the entry timeframe (M5 -> m5, else default), "
            "'default' is the Э7' space of §7.11, 'm5' the narrowed profile of §7.22"
        ),
    )
    parser.add_argument(
        "--score",
        choices=SCORE_PROFILES,
        default="auto",
        help=(
            "how a trial is ranked: 'auto' follows the entry timeframe, 'default' is the "
            "out-of-sample score of §7.11, 'm5' the trade-count score of §7.22"
        ),
    )
    parser.add_argument(
        "--target-trades",
        type=int,
        default=DEFAULT_TARGET_TRADES,
        help="trade count the m5 score saturates at (default: 25)",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=DEFAULT_MIN_TRADES,
        help="trade count below which the m5 score is a flat zero (default: 10)",
    )
    return parser


def _trade_target_scorer(args: argparse.Namespace) -> Scorer:
    """Return the pooled trade-count scorer of §7.22 with the targets of the command line.

    The counts are validated by :class:`~smc_zero.config.TradeTargetScore` before the study starts,
    so a ``--target-trades 0`` is a ``ValueError`` and not a score that always saturates.  Binding
    them here keeps the study itself free of the numbers: :data:`Scorer` is a plain function of the
    trial's whole fold evaluation, and :func:`~smc_zero.optimizer.score.trades_pool_scorer` reads
    the out-of-sample half of it - the same seam the out-of-sample score of §7.11 plugs into.  The
    density and pool gates of the dataclass (``min_fold_trades``, ``min_valid_folds``, ``pf_cap``)
    travel with it and are *not* command-line flags: they guard the reading itself, while the two
    flags below choose the search budget (Э13.1).
    """
    targets = TradeTargetScore(target_trades=args.target_trades, min_trades=args.min_trades)
    return partial(trades_pool_scorer, cfg=targets)


def _search_profile(
    args: argparse.Namespace, timeframe: str
) -> tuple[str, str, Mapping[str, ParamRange], Scorer | None]:
    """Return the profile of a run: the space it moves and how it ranks a trial (§7.22).

    The answer is ``(ranges name, score name, ranges, scorer)``.  ``auto`` - the default of both
    flags - reads the *entry timeframe*: an M5 tape is the second hierarchy of §7.20 and gets the
    narrowed space and the trade-count score of §7.22, while every other timeframe keeps the whole
    Э7' space and the out-of-sample score of §7.11.  A flag that names a profile explicitly wins
    over that pick, so the two studies differ by their command line alone and an M15 run of the
    first hierarchy behaves exactly as it did before.
    """
    auto = "m5" if timeframe.upper() == "M5" else "default"
    ranges_name = auto if args.ranges == "auto" else args.ranges
    score_name = auto if args.score == "auto" else args.score
    scorer = None if score_name == "default" else _trade_target_scorer(args)
    return ranges_name, score_name, PARAM_PROFILES[ranges_name], scorer


def _walk_forward_config(args: argparse.Namespace, timeframe: str) -> WalkForwardConfig:
    """Return the fold scheme of the run: the windows of its entry timeframe, counts overridable.

    The base is :func:`~smc_zero.backtester.walkforward.default_walk_forward` of the *effective*
    entry timeframe: an M15 tape keeps the 120 / 60 day windows of §7.10 п.62, the M5 tape of the
    second hierarchy reads the 60 / 30 days ruled in §7.20 (17 280 / 8 640 bars, nine folds).
    ``--min-train-bars`` and ``--test-period-bars`` are what makes a short window hold a fold at
    all (§7.10 п.61), which is what a smoke run or a test needs; leaving both alone keeps the
    windows of the timeframe.  The config validates the pair itself, so ``--min-train-bars 0`` is
    a ``ValueError`` and not a silent default.
    """
    defaults = default_walk_forward(timeframe)
    return WalkForwardConfig(
        anchored=defaults.anchored,
        min_train_bars=(
            defaults.min_train_bars if args.min_train_bars is None else args.min_train_bars
        ),
        test_period_bars=(
            defaults.test_period_bars if args.test_period_bars is None else args.test_period_bars
        ),
        train_period_bars=defaults.train_period_bars,
    )


def _jsonable(value: Any) -> Any:
    """Return a search space value as a JSON scalar (a ``ChoiceRange`` yields plain strings)."""
    if isinstance(value, (bool, int, float, str)):
        return value
    return float(value)


def _aggregates(table: Mapping[str, float]) -> dict[str, float]:
    """Return one aggregate as a plain float mapping, keys sorted - JSON and CSV safe."""
    return {name: float(value) for name, value in sorted(table.items())}


def _fold_frame(evaluation: FoldEvaluation) -> pd.DataFrame:
    """Return the winner's folds as one long frame: ``fold`` / ``window`` / the six metrics.

    Two rows per fold (``train`` for the window that may be fitted, ``test`` for the one the engine
    simulated) and four summary rows: ``mean`` and ``std`` of each window's column, read straight
    from the aggregates :func:`~smc_zero.optimizer.evaluate_params` already reported - so the table
    a reader opens and the score a trial was ranked by cannot disagree.

    The two ``mean`` rows also carry the three numbers the score of §7.11 п.69 reads beside its
    metrics (:data:`SCORE_COLUMNS`): the aggregate profit of each window and their ratio - the
    factor :func:`~smc_zero.optimizer.degradation_factor` hands the study (whether or not
    ``penalty_power`` applies it).  Every other row leaves them empty: a fold's own profit is
    already its ``profit`` column, and a run-level number repeated beside it would read as that
    fold's.
    """
    rows: list[dict[str, Any]] = []
    blank: dict[str, float] = dict.fromkeys(SCORE_COLUMNS, float("nan"))
    for index, (train, test) in enumerate(
        zip(evaluation.fold_metrics_train, evaluation.fold_metrics_test, strict=True)
    ):
        for window, table in (("train", train), ("test", test)):
            row: dict[str, Any] = {"fold": index, "window": window}
            row.update({field: table[field] for field in FOLD_METRIC_FIELDS})
            row.update(blank)
            rows.append(row)
    train_profit = float(evaluation.train_aggregated["profit_mean"])
    test_profit = float(evaluation.test_aggregated["profit_mean"])
    score_inputs: dict[str, float] = {
        "train_profit_mean": train_profit,
        "test_profit_mean": test_profit,
        "degradation_ratio": degradation_factor(train_profit, test_profit),
    }
    for window, aggregate in (
        ("train", evaluation.train_aggregated),
        ("test", evaluation.test_aggregated),
    ):
        for label in ("mean", "std"):
            row = {"fold": label, "window": window}
            row.update({field: aggregate[f"{field}_{label}"] for field in FOLD_METRIC_FIELDS})
            row.update(score_inputs if label == "mean" else blank)
            rows.append(row)
    return pd.DataFrame(rows)


def _test_note(trial: Any) -> str:
    """Return the out-of-sample reading of a finished trial, or ``''`` when it carries none.

    A real trial of a study records both aggregate tables as user attributes (``set_user_attr`` in
    the objective of §7.11), so the console can name what each rank was bought with: the trades the
    out-of-sample folds carried, their *mean* profit factor and their mean win rate.  The mean is
    labelled as such on purpose: under the pooled score of §7.22 it is not the number the rank was
    bought with (that one is the pool of the dense folds, Э13.1), and a note that read as the score
    would misname the study.  A stub trial carries no attributes and the note stays empty - the
    table is a *reading* of the study, never a second computation (rule 5).
    """
    attributes = getattr(trial, "user_attrs", None)
    if not isinstance(attributes, Mapping):
        return ""
    test = attributes.get("test")
    if not isinstance(test, Mapping) or "trades_total" not in test:
        return ""
    return (
        f"  test: trades {float(test['trades_total']):.0f}, "
        f"pf_fold_mean {float(test.get('pf_mean', 0.0)):.3f}, "
        f"win {float(test.get('win_rate_mean', 0.0)):.1f}%"
    )


def _print_trials(outcome: OptunaResult, limit: int = TOP_TRIALS) -> None:
    """Print the best ``limit`` finished trials of a study, best first.

    Beside the score and the parameter set of a rank, the line carries the winner's out-of-sample
    reading when the study recorded one (:func:`_test_note`), so a run of the trade-count profile
    of §7.22 can be read without opening ``fold_metrics.csv``.
    """
    finished = [trial for trial in outcome.study.trials if trial.value is not None]
    finished.sort(key=lambda trial: trial.value, reverse=True)
    print(f"trials: {len(outcome.study.trials)} run, {len(finished)} finished")
    for trial in finished[:limit]:
        print(f"  #{trial.number:<4} score {trial.value:+.4f}{_test_note(trial)}  {trial.params}")

def _report_payload(
    args: argparse.Namespace,
    symbol: str,
    timeframe: str,
    tape: pd.DataFrame,
    window: _common.RunWindow,
    walk_config: WalkForwardConfig,
    study_config: OptunaConfig,
    outcome: OptunaResult,
    *,
    ranges_name: str = "default",
    score_name: str = "default",
) -> dict[str, Any]:
    """Return the payload of ``best_params.json``: the winner, its score and its fold aggregates.

    ``symbol`` and ``timeframe`` are the *effective* pair of the run - the arguments of the caller
    if it typed them, else the config's (Э10'.2) - so the audit line names the tape the study read;
    ``window`` is the effective window (:func:`scripts._common.resolve_window`), for the same reason.
    ``ranges_name`` / ``score_name`` name the profile of §7.22 the study ran under, and the two
    trade targets are recorded beside them whenever the score reads a target at all - a report that
    named a winner without saying what ranked it would not be reproducible from its own file.
    """
    payload: dict[str, Any] = {
        "symbol": symbol.upper(),
        "timeframe": timeframe.upper(),
        "start": f"{window.start:%Y-%m-%d}",
        "end": f"{window.end:%Y-%m-%d}",
        "warmup_days": int(window.warmup_days),
        "bars": len(tape),
        "folds": len(outcome.best_evaluation.fold_metrics_test),
        "n_trials": int(study_config.n_trials),
        "n_jobs": int(study_config.n_jobs),
        "seed": int(study_config.seed),
        "score_metric": study_config.score_metric,
        "penalty_power": float(study_config.penalty_power),
        "min_train_bars": int(walk_config.min_train_bars),
        "test_period_bars": int(walk_config.test_period_bars),
        "best_trial_number": int(outcome.best_trial_number),
        "best_score": float(outcome.best_score),
        "param_ranges": ranges_name,
        "score_profile": score_name,
        "best_params": {
            name: _jsonable(value) for name, value in sorted(outcome.best_params.items())
        },
        "train": _aggregates(outcome.best_evaluation.train_aggregated),
        "test": _aggregates(outcome.best_evaluation.test_aggregated),
    }
    if score_name == "m5":
        payload["target_trades"] = int(args.target_trades)
        payload["min_trades"] = int(args.min_trades)
    return payload


def run(args: argparse.Namespace) -> int:
    """Search the parameters of ``args``, run the winner over the window and write its report."""
    try:
        symbol, timeframe, strategy, backtest = _common.live_inputs(args)
        window = _common.resolve_window(args)
        tape = _common.load_windowed_tape(
            symbol, timeframe, window.start, window.end, source=args.data_source
        )
        instrument = _common.instrument_for(symbol)
        structure_timeframe, structure_frame = _common.working_frame(
            symbol, backtest.timeframes, window.start, window.end, source=args.data_source
        )
        htf_frames = _common.markup_frames(
            symbol, backtest.timeframes, window, source=args.data_source
        )
        walk_config = _walk_forward_config(args, timeframe)
        study_config = OptunaConfig(n_trials=args.n_trials, n_jobs=args.jobs, seed=args.seed)
        ranges_name, score_name, ranges, scorer = _search_profile(args, timeframe)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if not split_walkforward(tape, walk_config):
        print(
            f"error: {len(tape)} bars of {window.start:%Y-%m-%d} .. {window.end:%Y-%m-%d} "
            f"hold no fold: min_train_bars {walk_config.min_train_bars} + "
            f"test_period_bars {walk_config.test_period_bars} need more (SPEC_SMC.md §7.10 п.61)",
            file=sys.stderr,
        )
        return 2

    # The config of the run is the base of the study: a trial moves the knobs of the chosen
    # profile (``PARAM_RANGES`` or the narrowed M5 space of §7.22) on top of it, and every fold is
    # charged the account of the same config (Э10'.2).  A separate working frame (§7.20) is marked
    # up once into the cache, beside the trends and the level book.
    print(f"profile: ranges {ranges_name}, score {score_name}, {len(tape)} bars")
    base = strategy
    marks = build_tape_marks(
        tape,
        base,
        ltf=backtest.timeframes.ltf,
        htf_frames=htf_frames or None,
        structure_frame=structure_frame,
        structure_timeframe=structure_timeframe,
    )
    try:
        outcome = run_optimization(
            tape,
            study_config,
            walk_config,
            backtest,
            instrument,
            base=base,
            marks=marks,
            ranges=ranges,
            score=scorer,
        )
    except ImportError as error:
        print(
            "error: the optimizer needs the 'optimize' extra, "
            f'pip install -e ".[optimize]" ({error})',
            file=sys.stderr,
        )
        return 2
    except ValueError as error:
        print(f"error: the study could not score this window: {error}", file=sys.stderr)
        return 2

    _print_trials(outcome)
    print(f"best: trial #{outcome.best_trial_number}, score {outcome.best_score:+.4f}")
    for name, value in sorted(outcome.best_params.items()):
        print(f"  {name:<28} {value}")

    # The winner may have moved the A/B knob of the bias agreement, and the cache answers for any
    # mode off its own trends frame - that is what ``bias_frame(agreement)`` exists for.
    chain = build_intents(
        tape,
        marks.bias_frame(outcome.strategy.bias.agreement),
        marks.levels,
        outcome.strategy,
        structure=marks.structure,
    )
    result = run_backtest(tape, chain.intents, backtest, instrument)

    folder = _common.report_folder(
        args.report_dir,
        f"optimization_{_common.window_label(symbol, timeframe, window.start, window.end)}"
        f"_n{args.n_trials}",
    )
    summary = format_summary(result)
    payload = _report_payload(
        args,
        symbol,
        timeframe,
        tape,
        window,
        walk_config,
        study_config,
        outcome,
        ranges_name=ranges_name,
        score_name=score_name,
    )
    export_trades(result, folder, stem="trades")
    (folder / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    (folder / "best_params.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    _fold_frame(outcome.best_evaluation).to_csv(folder / "fold_metrics.csv", index=False)

    print(summary)
    print(f"report: {folder}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` (``None`` = ``sys.argv``) and run one study; the console entry point."""
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
