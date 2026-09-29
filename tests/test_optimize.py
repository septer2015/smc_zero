"""Unit tests of the Э7' optimizer: the score, the search space and the trial plumbing (Э7').

Nothing here simulates anything: the score is checked against arithmetic a reader can redo on
paper, the space against the configuration it names, and the objective against a stub sketch of
a trial and a stub evaluator.  The real engine, the markup cache and the fold loop are the
business of ``tests/test_optimize_integration.py``.

The mutations the layer is one line away from, and the test each one must break:

* m1 "read the score out of the *train* half of the tables" (``score_from_aggregates`` called
  with the weight window swapped) - a study that optimizes in sample, which is the whole point
  of the gate; breaks :func:`test_the_score_reads_the_out_of_sample_side_of_the_folds`;
* m2 "drop the decay factor" (return the leading metric times the drawdown factor) - an
  overfitted parameter set scores as high as one that held up; breaks
  :func:`test_the_score_is_the_product_of_the_three_factors`;
* m3 "penalise a losing train window as if its edge had decayed" (a negative train profit
  makes the ratio negative) - a trial is punished for a reference that carries no edge; breaks
  :func:`test_a_losing_train_window_is_not_penalised`;
* m4 "accept a parameter set whose cache does not cover it" (``cache_mismatches`` returns
  nothing, or treats ``bias.agreement`` as a stale field) - trials are scored against a markup
  of another strategy; breaks :func:`test_a_frozen_field_makes_the_cache_stale` and
  :func:`test_the_ab_factor_is_allowed_to_differ_from_the_cache`.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from smc_zero.config import AGREEMENTS, OptunaConfig, StrategyConfig
from smc_zero.optimizer import (
    PARAM_RANGES,
    ChoiceRange,
    FloatRange,
    FoldEvaluation,
    IntRange,
    TapeMarks,
    apply_params,
    cache_mismatches,
    degradation_factor,
    drawdown_factor,
    make_objective,
    resolve_path,
    run_optimization,
    score_from_aggregates,
    suggest_params,
)

#: The aggregate tables of the worked example, and the score they must produce by hand:
#: ``2.0 * (1 - 0.10) * min(1, 50 / 100) = 0.9``.
TRAIN = {"profit_mean": 100.0, "max_dd_mean": 40.0, "sharpe_mean": 3.0}
TEST = {"profit_mean": 50.0, "max_dd_mean": 10.0, "sharpe_mean": 2.0}
HAND_CALC = 0.9


class _StubTrial:
    """A minimal :class:`~smc_zero.optimizer.ranges.TrialLike`: fixed values, recorded asks.

    It stands in for ``optuna.Trial`` in the unit tests: the three ``suggest_*`` methods answer
    with what the test configured (falling back to the low bound / the first choice), so a test
    can say "this trial proposes these parameters" without a sampler, and ``set_user_attr`` keeps
    the attributes a real trial would publish on the study.
    """

    def __init__(self, values: dict[str, object] | None = None) -> None:
        self.values = {} if values is None else values
        self.asked: list[str] = []
        self.attrs: dict[str, object] = {}

    def _value(self, name: str) -> object | None:
        """Record the ask and return the configured value, or ``None`` when there is none."""
        self.asked.append(name)
        return self.values.get(name)

    def suggest_int(
        self, name: str, low: int, high: int, step: int = 1, log: bool = False
    ) -> int:
        """Return the configured value of ``name``, or the low bound."""
        value = self._value(name)
        return low if value is None else int(value)

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        step: float | None = None,
        log: bool = False,
    ) -> float:
        """Return the configured value of ``name``, or the low bound."""
        value = self._value(name)
        return low if value is None else float(value)

    def suggest_categorical(self, name: str, choices: list[str]) -> str:
        """Return the configured value of ``name``, or the first choice."""
        value = self._value(name)
        return choices[0] if value is None else str(value)

    def set_user_attr(self, name: str, value: object) -> None:
        """Record an attribute the way a real trial publishes it to its study."""
        self.attrs[name] = value


def _marks(config: StrategyConfig | None = None) -> TapeMarks:
    """Return a cache record without building one: the unit tests never read its frames."""
    return TapeMarks(
        trends=pd.DataFrame(),
        levels=pd.DataFrame(),
        config=StrategyConfig() if config is None else config,
    )


def test_the_score_is_the_product_of_the_three_factors() -> None:
    """The score is ``leading * drawdown * decay`` - and dropping the gate inflates it (m2)."""
    score = score_from_aggregates(TRAIN, TEST)

    assert score == pytest.approx(2.0 * (1 - 0.10) * (50.0 / 100.0))
    assert score == pytest.approx(HAND_CALC)
    # m2: without the decay factor the overfit trial above would score 1.8 instead of 0.9.
    assert score != pytest.approx(HAND_CALC / (50.0 / 100.0))


def test_the_score_reads_the_out_of_sample_side_of_the_folds() -> None:
    """Swapping the two tables scores the fit window, i.e. exactly what the gate forbids (m1)."""
    assert score_from_aggregates(TRAIN, TEST) == pytest.approx(HAND_CALC)
    assert score_from_aggregates(TEST, TRAIN) == pytest.approx(3.0 * (1 - 0.40))


def test_the_score_metric_picks_the_leading_number_of_the_test_window() -> None:
    """``score_metric`` names the headline number; the profit gate is used either way."""
    by_profit = score_from_aggregates(TRAIN, TEST, OptunaConfig(score_metric="profit"))
    by_sharpe = score_from_aggregates(TRAIN, TEST, OptunaConfig(score_metric="sharpe"))

    assert by_profit == pytest.approx(50.0 * 0.90 * 0.50)
    assert by_sharpe == pytest.approx(2.0 * 0.90 * 0.50)


def test_the_decay_gate_always_measures_profit_not_the_leading_metric() -> None:
    """A train window with a brilliant Sharpe ratio still only gates through its profit."""
    rich_train = {**TRAIN, "sharpe_mean": 100.0}

    assert score_from_aggregates(rich_train, TEST) == pytest.approx(HAND_CALC)


def test_the_penalty_power_weighs_the_decay() -> None:
    """``penalty_power`` squares / cubes the shortfall instead of applying it once."""
    once = score_from_aggregates(TRAIN, TEST, OptunaConfig(penalty_power=1.0))
    twice = score_from_aggregates(TRAIN, TEST, OptunaConfig(penalty_power=2.0))

    assert once == pytest.approx(1.8 * 0.5)
    assert twice == pytest.approx(1.8 * 0.25)


def test_the_drawdown_factor_spans_zero_to_one_and_never_goes_negative() -> None:
    """The factor is the untouched score at a flat curve and zero at a wiped-out account."""
    assert drawdown_factor(0.0) == 1.0
    assert drawdown_factor(5.0) == pytest.approx(0.95)
    assert drawdown_factor(50.0) == pytest.approx(0.5)
    assert drawdown_factor(100.0) == 0.0
    # Beyond full drawdown the factor stays at zero: losing more cannot *raise* the score.
    assert drawdown_factor(150.0) == 0.0


def test_a_losing_test_window_scores_zero_and_a_losing_train_window_is_not_penalised() -> None:
    """The two edges of the gate: no OOS profit is no score, a losing reference has no edge (m3)."""
    losing_test = score_from_aggregates(TRAIN, {**TEST, "profit_mean": -25.0})
    losing_train = score_from_aggregates({**TRAIN, "profit_mean": -20.0}, TEST)

    assert losing_test == 0.0
    assert degradation_factor(100.0, -25.0) == 0.0
    assert degradation_factor(0.0, -25.0) == 1.0
    assert degradation_factor(-20.0, 50.0) == 1.0
    # A reference without an edge leaves the trial judged by what it made out of sample (m3).
    assert losing_train == pytest.approx(1.8)


def test_outperforming_the_train_window_earns_no_bonus() -> None:
    """The ratio is clamped: doubling the in-sample result is luck, not a better score."""
    assert degradation_factor(50.0, 100.0) == 1.0
    assert score_from_aggregates(TRAIN, {**TEST, "profit_mean": 400.0}) == pytest.approx(1.8)


def test_a_table_without_the_leading_metric_is_refused() -> None:
    """A score assembled from a missing metric would rank trials by nothing: it raises."""
    with pytest.raises(ValueError, match="sharpe_mean"):
        score_from_aggregates(TRAIN, {"profit_mean": 50.0, "max_dd_mean": 10.0})
    with pytest.raises(ValueError, match="profit_mean"):
        score_from_aggregates({"sharpe_mean": 3.0}, TEST)


def test_a_non_finite_metric_is_refused() -> None:
    """``NaN`` / ``inf`` in either window is a broken run, not a number to rank with."""
    with pytest.raises(ValueError, match="not finite"):
        score_from_aggregates(TRAIN, {**TEST, "max_dd_mean": float("nan")})
    with pytest.raises(ValueError, match="not finite"):
        score_from_aggregates({**TRAIN, "profit_mean": float("inf")}, TEST)


def test_every_range_names_a_real_field_of_the_configuration() -> None:
    """``resolve_path`` proves the space names knobs the config has - and of the right kind.

    Three pip gates (``sweep_buffer_pip``, ``sl_buffer_pip``, ``min_fvg_pip``) are *float*
    fields of the config searched on an integer grid (prod's grids are integer pips), so an
    ``IntRange`` is only asked for a whole number, not for an ``int`` field.
    """
    base = StrategyConfig()
    for path, param_range in PARAM_RANGES.items():
        value = resolve_path(base, path)
        if isinstance(param_range, ChoiceRange):
            assert isinstance(value, str), path
        elif isinstance(param_range, IntRange):
            assert isinstance(value, (int, float)) and not isinstance(value, bool), path
            assert float(value).is_integer(), path
        else:
            assert isinstance(param_range, FloatRange), path
            assert isinstance(value, float), path


#: The knobs whose shipped value is deliberately outside their range: the two thresholds of
#: the impulse gate ship as ``0.0`` (the gate is switched off by its own defaults, §7.5), so a
#: run that wants the gate to bite sets ``use_displacement`` and the thresholds explicitly.
INERT_GATE_DEFAULTS = frozenset({"displacement.atr_mult_min", "displacement.body_frac_min"})


def test_every_range_brackets_the_value_the_strategy_ships_with() -> None:
    """A study always starts inside the range of its knob - the inert impulse gate aside."""
    base = StrategyConfig()
    for path, param_range in PARAM_RANGES.items():
        default = resolve_path(base, path)
        if path in INERT_GATE_DEFAULTS:
            assert float(default) == 0.0, path
            assert not param_range.low <= float(default) <= param_range.high, path
            continue
        if isinstance(param_range, ChoiceRange):
            assert default in param_range.choices, path
        else:
            assert param_range.low <= float(default) <= param_range.high, path


def test_the_only_categorical_knob_is_the_ab_factor_of_the_bias() -> None:
    """The one non-numeric dimension of the space is ``bias.agreement``, over ``AGREEMENTS``."""
    categorical = {
        path: param_range
        for path, param_range in PARAM_RANGES.items()
        if isinstance(param_range, ChoiceRange)
    }

    assert set(categorical) == {"bias.agreement"}
    assert set(categorical["bias.agreement"].choices) == set(AGREEMENTS)
    assert len(categorical["bias.agreement"].choices) == len(AGREEMENTS)


def test_the_search_space_leaves_the_inputs_of_the_cache_alone() -> None:
    """A trial may not vary what the markup was built from, save the A/B factor (m4)."""
    frozen_prefixes = ("levels.", "session.", "bias.")
    frozen = {
        path
        for path in PARAM_RANGES
        if path.startswith(frozen_prefixes) and path != "bias.agreement"
    }

    assert frozen == set()
    # ... and the exception is real: the A/B factor *is* a dimension of the search space.
    assert "bias.agreement" in PARAM_RANGES


def test_resolve_path_refuses_a_field_the_configuration_does_not_have() -> None:
    """A typo in a path fails loudly instead of optimizing a knob that does not exist."""
    base = StrategyConfig()

    with pytest.raises(KeyError, match="sweep_buffer_slip"):
        resolve_path(base, "sweep_buffer_slip")
    with pytest.raises(KeyError, match="nope"):
        resolve_path(base, "take_profit.nope")


def test_applying_parameters_copies_the_configuration_level_by_level() -> None:
    """``apply_params`` returns a new config: the input and its untouched subtrees survive."""
    base = StrategyConfig()
    updated = apply_params(
        base,
        {"sweep_buffer_pip": 3, "take_profit.rr_fallback": 2.5, "bias.agreement": "majority"},
    )

    assert (updated.sweep_buffer_pip, updated.take_profit.rr_fallback) == (3, 2.5)
    assert updated.bias.agreement == "majority"
    assert (base.sweep_buffer_pip, base.take_profit.rr_fallback) == (1.0, 2.0)
    assert base.bias.agreement == "unanimous"
    assert updated.take_profit.min_tp_rr == base.take_profit.min_tp_rr
    assert updated.bias.structure == base.bias.structure
    assert updated.levels is base.levels
    assert updated is not base


def test_applying_parameters_refuses_an_unknown_path() -> None:
    """A parameter mapping may not invent a field: ``apply_params`` raises, it does not skip."""
    with pytest.raises(KeyError, match="nope"):
        apply_params(StrategyConfig(), {"take_profit.nope": 1.0})
    with pytest.raises(KeyError, match="nope"):
        apply_params(StrategyConfig(), {"nope": 1.0})


def test_a_suggested_mapping_round_trips_through_the_configuration() -> None:
    """``suggest_params`` speaks the names ``apply_params`` writes, one value per range."""
    base = StrategyConfig()
    values = {
        path: int(resolve_path(base, path))
        if isinstance(param_range, IntRange)
        else resolve_path(base, path)
        for path, param_range in PARAM_RANGES.items()
    }
    trial = _StubTrial(values)

    params = suggest_params(trial)

    assert set(trial.asked) == set(PARAM_RANGES)
    assert len(trial.asked) == len(PARAM_RANGES)
    assert params == values
    assert apply_params(base, params) == base


def test_a_trial_that_varies_the_whole_space_is_served_by_the_cache() -> None:
    """The space and the guard agree: no searched knob makes the markup stale (m4)."""
    base = StrategyConfig()
    varied = apply_params(base, suggest_params(_StubTrial()))

    assert cache_mismatches(_marks(base), varied) == ()


def test_a_frozen_field_makes_the_cache_stale() -> None:
    """A configuration built with another swing / killzone / level map is refused (m4)."""
    base = StrategyConfig()
    marks = _marks(base)

    assert cache_mismatches(marks, base) == ()
    # A searched knob may differ from the cached value: the trial is *expected* to vary it.
    assert cache_mismatches(marks, replace(base, sweep_buffer_pip=4.0, sl_buffer_pip=9.0)) == ()
    # A frozen one may not: the markup on the cache belongs to the config it was built from.
    assert cache_mismatches(marks, replace(base, choch_wait_bars=30)) == ("choch_wait_bars",)
    assert set(cache_mismatches(marks, replace(base, choch_wait_bars=30, min_sl_pip=9.0))) == {
        "choch_wait_bars",
        "min_sl_pip",
    }
    assert set(
        cache_mismatches(
            marks,
            replace(
                base,
                bias=replace(
                    base.bias,
                    structure=replace(base.bias.structure, swing_lookback=2),
                ),
            ),
        )
    ) == {"bias.structure.swing_lookback"}
    assert cache_mismatches(marks, replace(base, session=replace(base.session, use_kz=False))) != ()


def test_the_ab_factor_is_allowed_to_differ_from_the_cache() -> None:
    """``bias.agreement`` is recomputed per trial, so it is the one path the guard ignores (m4)."""
    base = StrategyConfig()
    marks = _marks(base)
    switched = replace(base, bias=replace(base.bias, agreement="majority"))

    assert switched.bias.agreement != base.bias.agreement
    assert cache_mismatches(marks, switched) == ()


class _RecordingEvaluator:
    """A stub evaluator: it records the configuration and returns the tables of the fixture."""

    def __init__(
        self,
        train: dict[str, float] | None = None,
        test: dict[str, float] | None = None,
    ) -> None:
        self.train = dict(TRAIN if train is None else train)
        self.test = dict(TEST if test is None else test)
        self.configs: list[StrategyConfig] = []

    def __call__(
        self,
        df: pd.DataFrame,
        marks: TapeMarks,
        cfg_strategy: StrategyConfig,
        cfg_wf: object,
        backtest: object,
        instrument: object,
    ) -> FoldEvaluation:
        """Return one fold of the fixture: the same train and test tables for every call."""
        self.configs.append(cfg_strategy)
        return FoldEvaluation(
            fold_metrics_train=[dict(self.train)],
            fold_metrics_test=[dict(self.test)],
            train_aggregated=dict(self.train),
            test_aggregated=dict(self.test),
        )


def test_the_objective_scores_the_out_of_sample_side_of_the_folds() -> None:
    """The objective returns the OOS score of the trial - in-sample would score 1.8 here (m1)."""
    evaluator = _RecordingEvaluator()
    objective = make_objective(pd.DataFrame(), marks=_marks(), evaluate=evaluator)

    score = objective(_StubTrial())

    assert score == pytest.approx(HAND_CALC)
    assert score != pytest.approx(score_from_aggregates(TEST, TRAIN))
    assert len(evaluator.configs) == 1


def test_the_objective_records_both_aggregates_on_the_trial() -> None:
    """A finished study is read without rerunning it: the trial carries both window tables."""
    objective = make_objective(pd.DataFrame(), marks=_marks(), evaluate=_RecordingEvaluator())
    trial = _StubTrial()

    objective(trial)

    assert trial.attrs["train"] == TRAIN
    assert trial.attrs["test"] == TEST


def test_the_objective_applies_the_suggested_parameters_to_the_base_configuration() -> None:
    """A trial varies the space of ``PARAM_RANGES`` on top of ``base`` and nothing else."""
    base = StrategyConfig()
    evaluator = _RecordingEvaluator()
    objective = make_objective(pd.DataFrame(), base=base, marks=_marks(), evaluate=evaluator)

    objective(_StubTrial({"sweep_buffer_pip": 4, "bias.agreement": "majority"}))
    seen = evaluator.configs[0]

    assert seen.sweep_buffer_pip == 4
    assert seen.bias.agreement == "majority"
    # The rest of the space is the trial's as well: the stub answers the low bound of a range.
    assert seen.sl_buffer_pip == PARAM_RANGES["sl_buffer_pip"].low
    assert seen.take_profit.rr_fallback == PARAM_RANGES["take_profit.rr_fallback"].low
    # Only the frozen knobs come from the base configuration, unchanged and shared.
    assert seen.choch_wait_bars == base.choch_wait_bars
    assert seen.session is base.session
    assert seen is not base
    assert base.sweep_buffer_pip == 1.0


def test_a_base_configuration_the_cache_does_not_cover_is_refused() -> None:
    """A trial scored against a markup of another strategy is the one thing the layer refuses (m4)."""
    with pytest.raises(ValueError, match="choch_wait_bars"):
        make_objective(
            pd.DataFrame(),
            base=replace(StrategyConfig(), choch_wait_bars=30),
            marks=_marks(),
        )


def test_a_run_refuses_a_stale_cache_before_creating_its_study() -> None:
    """The guard fires in the setup of a run, so a doomed search costs no trial at all (m4)."""
    created: list[OptunaConfig] = []

    def factory(config: OptunaConfig) -> object:
        """Fail if a study is created: a stale cache must stop the run before that."""
        created.append(config)
        raise AssertionError("a study must not be created for a stale cache")

    with pytest.raises(ValueError, match="choch_wait_bars"):
        run_optimization(
            pd.DataFrame(),
            OptunaConfig(n_trials=1),
            marks=_marks(replace(StrategyConfig(), choch_wait_bars=30)),
            evaluate=_RecordingEvaluator(),
            study_factory=factory,
        )

    assert created == []
