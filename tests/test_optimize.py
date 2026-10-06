"""Unit tests of the Э7' optimizer: the score, the search space and the trial plumbing (Э7').

Nothing here simulates anything: the score is checked against arithmetic a reader can redo on
paper, the space against the configuration it names, and the objective against a stub sketch of
a trial and a stub evaluator.  The real engine, the markup cache and the fold loop are the
business of ``tests/test_optimize_integration.py``.

The mutations the layer is one line away from, and the test each one must break:

* m1 "read the score out of the *train* half of the tables" (``score_from_aggregates`` called
  with the weight window swapped) - a study that optimizes in sample, which is the whole point
  of the gate; breaks :func:`test_the_score_reads_the_out_of_sample_side_of_the_folds`;
* m2 "drop a factor of the product" (the leading metric, the profit of the test window or its
  drawdown divisor) - a trial that made nothing would then rank by its headline metric alone;
  breaks :func:`test_the_score_is_the_product_of_the_test_window_factors`;
* m3 "penalise a losing train window as if its edge had decayed" (a negative train profit
  makes the ratio negative) - a trial is punished for a reference that carries no edge; breaks
  :func:`test_a_losing_train_window_is_not_penalised`;
* m4 "accept a parameter set whose cache does not cover it" (``cache_mismatches`` returns
  nothing, or treats ``bias.agreement`` as a stale field) - trials are scored against a markup
  of another strategy; breaks :func:`test_a_frozen_field_makes_the_cache_stale` and
  :func:`test_the_ab_factor_is_allowed_to_differ_from_the_cache`.
* m5 "ignore ``penalty_power``" (the decay factor is applied once whatever the run asked for) -
  a study run with the gate off would silently rank by the old ungated product; breaks
  :func:`test_the_penalty_power_weighs_the_decay_and_zero_switches_it_off`.
* m6 "rank a losing out-of-sample window by the size of its loss" (the guard at
  ``profit_mean(test) <= 0`` is dropped, or written as ``abs(...)``) - two negative factors
  multiply into a *positive* score, and the worst run of a study becomes its winner (measured:
  dropped → ``+1.8181`` instead of ``0``, ``abs()`` → ``-1.8181`` instead of ``0``; the same drop
  on the first real run scored ``+9.6154``); breaks
  :func:`test_a_losing_test_window_scores_zero_whatever_the_gate_and_its_metric`.
* m7 "suggest the whole space whatever a ``ranges`` mapping was handed in"
  (``suggest_params(trial)`` without the second argument) - the narrowed profile of §7.22 silently
  searches eleven knobs again and the study is not the study its command line names; breaks
  :func:`test_the_objective_honours_a_narrowed_search_space`.
* m8 "drop the trade-count floor of §7.22" (``min_trades`` ignored) - a walk-forward of four trades
  outranks one of forty by its profit factor alone, i.e. the study optimizes noise; breaks
  :func:`test_the_trade_count_score_is_flat_below_its_floor_and_needs_the_fold_sums`.
* m9 "hand the chain the raw cut of the structure layer" (``_fold_structure`` reduced to
  ``structure.loc[bars.index]``, or the builder forgetting to call it) - ``disp_known_at`` keeps
  the positions of the whole tape inside a fold that starts at zero, so the gate
  ``disp_known_at <= attempt`` refuses every setup of every fold but the first (measured on the
  live M5 tape: nine folds, zero trades each, against twelve trades of the whole-window run); breaks
  :func:`test_a_fold_layer_is_rebased_on_its_own_positions` and
  :func:`test_the_fold_builder_rebases_the_layer_it_hands_to_the_chain`.
* m10 "average the per-fold profit factors instead of pooling them" (``trades_scaled_score``
  divides a mean of the folds rather than their gross wins over their gross losses, i.e. the reading
  of Э13) - the first M5 study ranked a *losing* parameter set first that way, because one fold of
  four trades with a perfect win rate is capped at ``5.0`` and lifts the mean; breaks
  :func:`test_the_m5_score_rejects_a_negative_profit_oos_set` and
  :func:`test_the_m5_winner_is_now_consistent_with_pool_metrics`.
* m11 "pool the thin folds back in" (the ``min_fold_trades`` gate dropped, or the density read on
  the aggregate) - a one-trade fold with an ideal profit factor moves the pool of a fourteen-trade
  one; breaks :func:`test_the_m5_score_ignores_thin_folds_below_three_trades`.
* m12 "read a pool of two or three folds" (the ``min_valid_folds`` gate dropped) - the score is
  read on a sample the stage itself calls statistically empty (nine folds of 1.4 years, §7.22
  п.120); breaks :func:`test_the_m5_score_requires_minimum_valid_folds`.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import smc_zero.optimizer.optimize as optimize_module
from smc_zero.config import AGREEMENTS, OptunaConfig, StrategyConfig, TradeTargetScore
from smc_zero.optimizer import (
    M5_PARAM_RANGES,
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
    pool_profit_factor,
    resolve_path,
    run_optimization,
    score_from_aggregates,
    suggest_params,
    trades_pool_scorer,
    trades_scaled_score,
)
from smc_zero.optimizer.optimize import _fold_structure, _intents_fn

#: The aggregate tables of the worked example, and the score they must produce by hand:
#: ``2.0 * 50 / (1 + 10) = 9.0909...`` - the metric, the profit and the drawdown of the test
#: window, with the decay gate off (``penalty_power = 0``, the default).
TRAIN = {"profit_mean": 100.0, "max_dd_mean": 40.0, "sharpe_mean": 3.0}
TEST = {"profit_mean": 50.0, "max_dd_mean": 10.0, "sharpe_mean": 2.0}
HAND_CALC = 2.0 * 50.0 / 11.0
#: The same score with the plain decay gate of п.69 on: half of the train profit survived.
HAND_CALC_GATED = HAND_CALC * 0.5


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


def test_the_score_is_the_product_of_the_test_window_factors() -> None:
    """The score is ``metric * profit / (1 + drawdown)``: dropping a factor breaks it (m2)."""
    score = score_from_aggregates(TRAIN, TEST)

    assert score == pytest.approx(2.0 * 50.0 / (1.0 + 10.0))
    assert score == pytest.approx(HAND_CALC)
    # m2: the leading metric alone would score 2.0, and the profit without its divisor 100.0.
    assert score != pytest.approx(2.0)
    assert score != pytest.approx(2.0 * 50.0)


def test_the_score_reads_the_out_of_sample_side_of_the_folds() -> None:
    """Swapping the two tables scores the fit window, i.e. exactly what the layer forbids (m1)."""
    assert score_from_aggregates(TRAIN, TEST) == pytest.approx(HAND_CALC)
    assert score_from_aggregates(TEST, TRAIN) == pytest.approx(3.0 * 100.0 / 41.0)


def test_the_score_metric_picks_the_leading_number_of_the_test_window() -> None:
    """``score_metric`` names the headline number; the profit factor is used either way."""
    by_profit = score_from_aggregates(TRAIN, TEST, OptunaConfig(score_metric="profit"))
    by_sharpe = score_from_aggregates(TRAIN, TEST, OptunaConfig(score_metric="sharpe"))

    assert by_profit == pytest.approx(50.0 * 50.0 / 11.0)
    assert by_sharpe == pytest.approx(2.0 * 50.0 / 11.0)


def test_the_penalty_power_weighs_the_decay_and_zero_switches_it_off() -> None:
    """The gate is off at ``penalty_power = 0`` (the default) and applies the ratio when on (m5)."""
    off = score_from_aggregates(TRAIN, TEST)
    once = score_from_aggregates(TRAIN, TEST, OptunaConfig(penalty_power=1.0))
    twice = score_from_aggregates(TRAIN, TEST, OptunaConfig(penalty_power=2.0))

    assert OptunaConfig().penalty_power == 0.0
    assert off == pytest.approx(HAND_CALC)
    assert once == pytest.approx(HAND_CALC_GATED)
    assert twice == pytest.approx(HAND_CALC_GATED * 0.5)


def test_the_decay_gate_always_measures_profit_not_the_leading_metric() -> None:
    """A train window with a brilliant Sharpe ratio still only gates through its profit."""
    rich_train = {**TRAIN, "sharpe_mean": 100.0}
    gate = OptunaConfig(penalty_power=1.0)

    assert score_from_aggregates(rich_train, TEST, gate) == pytest.approx(HAND_CALC_GATED)


def test_the_drawdown_divisor_shrinks_with_the_drawdown_and_never_reaches_zero() -> None:
    """The factor is ``1 / (1 + dd)``: a flat curve keeps its score, a deep one keeps a share."""
    assert drawdown_factor(0.0) == 1.0
    assert drawdown_factor(5.0) == pytest.approx(1.0 / 6.0)
    assert drawdown_factor(50.0) == pytest.approx(1.0 / 51.0)
    assert drawdown_factor(100.0) == pytest.approx(1.0 / 101.0)
    # A curve cannot draw down *upwards*: a negative reading is a flat one, never a bonus.
    assert drawdown_factor(-5.0) == 1.0


def test_a_losing_train_window_is_not_penalised() -> None:
    """A reference without an edge is no degradation to measure: the gate stays at one (m3)."""
    gate = OptunaConfig(penalty_power=1.0)

    assert degradation_factor(0.0, -25.0) == 1.0
    assert degradation_factor(-20.0, 50.0) == 1.0
    # With the gate on the score is the test window's own - not the half the ratio would take.
    assert score_from_aggregates({**TRAIN, "profit_mean": -20.0}, TEST, gate) == pytest.approx(
        HAND_CALC
    )


def test_a_losing_test_window_scores_zero_whatever_the_gate_and_its_metric() -> None:
    """Two negative factors (Sharpe and profit) must not multiply into a positive score (m6).

    The case is the first real run: ``sharpe_mean(test) = -0.19`` and ``profit_mean(test) =
    -127.53`` scored ``+9.6154`` without the guard, i.e. a losing parameter set beat every
    profitable one.  The gate cannot save that - it is off at the default ``penalty_power`` - so
    the zero has to come from the score itself, and it is the same zero a window without trades
    scores.
    """
    losing = {"sharpe_mean": -0.2, "profit_mean": -100.0, "max_dd_mean": 10.0}
    profitable = {"sharpe_mean": 0.2, "profit_mean": 100.0, "max_dd_mean": 10.0}

    assert degradation_factor(100.0, -100.0) == 0.0
    assert score_from_aggregates(TRAIN, losing) == 0.0
    assert score_from_aggregates(TRAIN, losing, OptunaConfig(penalty_power=1.0)) == 0.0
    assert score_from_aggregates(TRAIN, {**losing, "max_dd_mean": 50.0}) == 0.0
    # The profitable twin of the very same shape is read normally: 0.2 * 100 / 11.
    assert score_from_aggregates(TRAIN, profitable) == pytest.approx(0.2 * 100.0 / 11.0)
    # ... and a window without a single trade is that same flat zero.
    assert score_from_aggregates(TRAIN, {**TEST, "profit_mean": 0.0}) == 0.0


def test_outperforming_the_train_window_earns_no_bonus() -> None:
    """The ratio is clamped: doubling the in-sample result is luck, not a better score."""
    gate = OptunaConfig(penalty_power=1.0)

    assert degradation_factor(50.0, 100.0) == 1.0
    assert score_from_aggregates(TRAIN, {**TEST, "profit_mean": 400.0}, gate) == pytest.approx(
        2.0 * 400.0 / 11.0
    )


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
    """The objective returns the OOS score of the trial - in-sample would score 7.32 here (m1)."""
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


def _pool_folds(
    count: int, trades: int, gross_win: float, gross_loss: float
) -> list[dict[str, float]]:
    """Return ``count`` identical dense fold tables for the pooled score of §7.22."""
    return [
        {
            "trades": trades,
            "profit": gross_win - gross_loss,
            "gross_win": gross_win,
            "gross_loss": gross_loss,
        }
        for _ in range(count)
    ]


def test_the_trade_count_score_scales_the_pooled_profit_factor_by_the_trade_share() -> None:
    """Below the target the score is ``pf_pool * trades / target``; at the target it is the pf."""
    cfg = TradeTargetScore()

    below = trades_scaled_score(_pool_folds(5, 4, 40.0, 20.0), cfg)
    at_target = trades_scaled_score(_pool_folds(5, 5, 40.0, 20.0), cfg)
    above = trades_scaled_score(_pool_folds(5, 8, 40.0, 20.0), cfg)

    assert below == pytest.approx(2.0 * 20.0 / 25.0)
    assert at_target == pytest.approx(2.0)
    # The factor caps at one: past the target more trades buy no rank (the honest half of the rule).
    assert above == pytest.approx(2.0)


def test_the_trade_count_score_is_flat_below_its_floor_and_needs_the_fold_sums() -> None:
    """Fewer trades than the floor scores zero whatever the pool says (m8)."""
    cfg = TradeTargetScore(min_valid_folds=2)

    assert cfg.min_trades == 10
    # Two dense folds carry six trades: below the floor of ten, whatever their profit factor is.
    assert trades_scaled_score(_pool_folds(2, 3, 9.0, 0.0), cfg) == 0.0
    # A pool that never made a win has no profit factor at all: a flat zero as well.
    assert trades_scaled_score(_pool_folds(5, 5, 0.0, 25.0), TradeTargetScore()) == 0.0
    # The aggregate table of §7.11 is not a fold table: the pooled score names what it needs.
    with pytest.raises(ValueError, match="gross_win"):
        trades_scaled_score([{"trades": 20, "profit": 1.0}] * 5, TradeTargetScore())


def test_the_trade_count_score_reads_only_the_out_of_sample_folds_of_the_evaluation() -> None:
    """The ``Scorer`` seam hands the whole evaluation over; the pooled score reads its test half."""
    cfg = TradeTargetScore()
    folds = _pool_folds(5, 5, 40.0, 20.0)
    evaluation = SimpleNamespace(
        fold_metrics_train=[{"trades": 1, "profit": 99.0, "gross_win": 99.0, "gross_loss": 0.0}],
        fold_metrics_test=folds,
    )

    assert trades_pool_scorer(evaluation, cfg) == pytest.approx(2.0)


def test_the_m5_score_rejects_a_negative_profit_oos_set() -> None:
    """A pool with a fine mean profit factor but no money in it scores a flat zero (m10)."""
    capped = _pool_folds(2, 5, 50.0, 10.0)  # a profit factor of 5.0 in both folds, +40 each
    bleeding = _pool_folds(3, 5, 1.0, 100.0)  # a profit factor of 0.01 in all three, -99 each
    pool = capped + bleeding

    # The old reading averaged the per-fold profit factors (capped at 5.0): that mean is above one...
    assert (5.0 + 5.0 + 0.01 * 3) / len(pool) > 1.0
    # ... while the pool those folds traded lost money, so the pooled score is a flat zero.
    assert sum(table["profit"] for table in pool) < 0.0
    assert trades_scaled_score(pool, TradeTargetScore()) == 0.0


def test_the_m5_score_ignores_thin_folds_below_three_trades() -> None:
    """A one-trade fold never enters the pool: the aggregate is the dense folds' alone (m11)."""
    thin = _pool_folds(2, 1, 5.0, 0.0)  # an ideal profit factor on a single trade, ignored
    dense = _pool_folds(1, 10, 12.0, 10.0)  # a profit factor of 1.2 on ten trades
    cfg = TradeTargetScore(min_valid_folds=1, min_trades=1)

    assert pool_profit_factor(dense) == pytest.approx(1.2)
    assert trades_scaled_score(dense + thin, cfg) == pytest.approx(1.2 * 10.0 / 25.0)
    assert trades_scaled_score(dense, cfg) == pytest.approx(1.2 * 10.0 / 25.0)
    # The gate is what keeps them out: pooled without it, the two one-trade folds lift the factor.
    assert pool_profit_factor(dense + thin) == pytest.approx(2.2)


def test_the_m5_score_requires_minimum_valid_folds() -> None:
    """Fewer dense folds than the minimum is not a pool: the score is a flat zero (m12)."""
    cfg = TradeTargetScore()

    assert cfg.min_valid_folds == 5
    assert trades_scaled_score(_pool_folds(4, 5, 40.0, 20.0), cfg) == 0.0
    assert trades_scaled_score(_pool_folds(5, 5, 40.0, 20.0), cfg) == pytest.approx(2.0)


#: The nine out-of-sample folds of trial #6 of the 100-trial M5 run of 2026-10-06, as the run
#: reported them (``fold_metrics.csv`` of that report): ``(trades, profit, pf)`` of every fold.  Four
#: folds are dense (``>= 3`` trades) and their profits sum to ``-11.30``, while the old
#: fold-averaged score of §7.22 ranked this trial first with ``1.0304`` (Э13.1).
E13_WINNER_FOLDS: tuple[tuple[int, float, float], ...] = (
    (1, -31.6, 0.0),
    (2, 8.2, 1.321),
    (0, 0.0, 0.0),
    (5, -113.45, 0.0),
    (1, -13.4, 0.0),
    (4, 98.9, 5.0),
    (3, -15.0, 0.639),
    (5, 18.25, 1.303),
    (2, 17.2, 1.798),
)


def _fold_from_pf(trades: int, profit: float, pf: float) -> dict[str, float]:
    """Return a fold table with the two sums behind ``pf`` (``pf <= 0`` is a fold without a win)."""
    if pf <= 0.0:
        return {"trades": trades, "profit": profit, "gross_win": 0.0, "gross_loss": abs(profit)}
    gross_loss = profit / (pf - 1.0)
    return {
        "trades": trades,
        "profit": profit,
        "gross_win": gross_loss * pf,
        "gross_loss": gross_loss,
    }


def test_the_m5_winner_is_now_consistent_with_pool_metrics() -> None:
    """The trial the old score ranked first is a flat zero under the pooled reading (m10 / m12)."""
    folds = [_fold_from_pf(trades, profit, pf) for trades, profit, pf in E13_WINNER_FOLDS]
    dense = [table for table in folds if table["trades"] >= 3]

    # Four of the nine folds are dense - below the minimum of five, so there is no pool to read.
    assert len(dense) == 4
    assert trades_scaled_score(folds, TradeTargetScore()) == 0.0
    # Even a relaxed density minimum does not save the trial: the dense folds lost money together.
    assert sum(table["profit"] for table in dense) == pytest.approx(-11.30)
    assert trades_scaled_score(folds, TradeTargetScore(min_valid_folds=1)) == 0.0


def test_the_objective_honours_a_narrowed_search_space() -> None:
    """A narrowed space searches its own paths and leaves the rest of the config at ``base`` (m7)."""
    base = StrategyConfig()
    evaluator = _RecordingEvaluator()
    objective = make_objective(
        pd.DataFrame(), base=base, marks=_marks(), evaluate=evaluator, ranges=M5_PARAM_RANGES
    )
    trial = _StubTrial()

    objective(trial)

    assert trial.asked == list(M5_PARAM_RANGES)
    seen = evaluator.configs[0]
    # The three bar windows and the two gates of the M5 profile moved (the stub answers the low end).
    assert seen.choch_wait_bars == M5_PARAM_RANGES["choch_wait_bars"].low
    assert seen.fvg_lookback == M5_PARAM_RANGES["fvg_lookback"].low
    assert seen.signal_max_age_bars == M5_PARAM_RANGES["signal_max_age_bars"].low
    assert seen.min_sl_realistic_pip == M5_PARAM_RANGES["min_sl_realistic_pip"].low
    assert seen.displacement.atr_mult_min == M5_PARAM_RANGES["displacement.atr_mult_min"].low
    # The knobs the profile does not name are untouched: they carry the live config of §7.20.
    assert seen.sl_buffer_pip == base.sl_buffer_pip
    assert seen.max_fvg_age_bars == base.max_fvg_age_bars
    assert seen.take_profit == base.take_profit
    assert seen.bias.agreement == base.bias.agreement


def test_a_custom_scorer_ranks_the_trial_instead_of_the_out_of_sample_one() -> None:
    """The ``score`` seam replaces the reading of the walk-forward and nothing else."""
    seen: list[FoldEvaluation] = []

    def scorer(evaluation: FoldEvaluation) -> float:
        """Record the evaluation and answer a fixed number."""
        seen.append(evaluation)
        return 4.25

    evaluator = _RecordingEvaluator()
    objective = make_objective(pd.DataFrame(), marks=_marks(), evaluate=evaluator, score=scorer)

    score = objective(_StubTrial())

    assert score == pytest.approx(4.25)
    assert [item.test_aggregated for item in seen] == [TEST]
    assert len(evaluator.configs) == 1


def test_the_m5_profile_names_real_fields_inside_the_gates_it_must_respect() -> None:
    """Every path of the narrowed space exists, and its bounds do not cross a shipped ceiling."""
    base = StrategyConfig()

    for path, param_range in M5_PARAM_RANGES.items():
        value = resolve_path(base, path)
        assert isinstance(value, (int, float)) and not isinstance(value, bool), path
        assert param_range.low < param_range.high, path

    # The realistic SL band of §7.8 is a *pair*: a searched floor may not cross its ceiling.
    assert M5_PARAM_RANGES["min_sl_realistic_pip"].high < base.max_sl_realistic_pip
    # The narrowed space never moves an input of the markup cache (the same rule the Э7' space
    # obeys): the level map, the sessions and the bias swings stay as the cache was built.
    assert not any(
        path.startswith(("levels.", "session.", "bias.")) for path in M5_PARAM_RANGES
    )


def test_a_fold_layer_is_rebased_on_its_own_positions() -> None:
    """The layer of a slice is re-based: ``disp_known_at`` is a position, the verdicts are not (m9).

    A fold is a positional slice of the tape, while the cached layer speaks in the positions of the
    frame it was built for; the raw rows would leave every ``disp_known_at`` ahead of every attempt
    of the fold, and the chain would refuse every setup of every fold but the first.
    """
    layer = pd.DataFrame(
        {
            "break_dir": pd.array([0, -1, -1, 1], dtype="int8"),
            "disp_ok": pd.array([False, True, True, False]),
            "disp_known_at": pd.array([3, 5, 6, 8], dtype="Int64"),
        },
        index=[7, 8, 9, 10],
    )
    bars = pd.DataFrame(index=[9, 10])

    cut = _fold_structure(layer, bars)

    assert cut.index.tolist() == [9, 10]
    # the coordinates move with the frame: the slice starts at position 2 of the layer, so the
    # ``disp_known_at`` values 6 / 8 of the layer are 4 / 6 of the fold
    assert cut["disp_known_at"].tolist() == [4, 6]
    # ... and the verdicts about a bar stay as they are
    assert cut["break_dir"].tolist() == [-1, 1]
    assert cut["disp_ok"].tolist() == [True, False]
    # the first rows of the layer are the frame the fold starts in: no shift at all
    pd.testing.assert_frame_equal(_fold_structure(layer, layer), layer)


def test_the_fold_builder_rebases_the_layer_it_hands_to_the_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The builder cuts the cached layer and re-bases it before the chain reads it (m9).

    The layer of a study covers the whole tape; a fold is a slice of it.  The builder must hand
    the chain the *re-based* rows (``_fold_structure``) rather than the raw cut, otherwise
    ``disp_known_at`` keeps the tape's positions inside a frame that starts at zero and every
    setup of the fold is refused.
    """
    layer = pd.DataFrame(
        {
            "break_dir": pd.array([0, -1, -1, 1], dtype="int8"),
            "disp_ok": pd.array([False, True, True, False]),
            "disp_known_at": pd.array([3, 5, 6, 8], dtype="Int64"),
        },
        index=[7, 8, 9, 10],
    )
    handed: list[pd.DataFrame | None] = []

    def stub_chain(
        ltf: pd.DataFrame,
        bias: pd.DataFrame,
        levels: pd.DataFrame,
        cfg: StrategyConfig | None = None,
        *,
        structure: pd.DataFrame | None = None,
    ) -> Any:
        """Record the layer the builder hands over and answer with it as the intents."""
        handed.append(structure)
        return SimpleNamespace(intents=())

    monkeypatch.setattr(optimize_module, "build_intents", stub_chain)
    build = _intents_fn(pd.DataFrame(), pd.DataFrame(), StrategyConfig(), layer)
    bars = pd.DataFrame({"is_closed": [True, True]}, index=[9, 10])

    build(pd.DataFrame(index=[9, 10]), bars)

    assert handed[0] is not None
    assert handed[0]["disp_known_at"].tolist() == [4, 6]
    # ... and a tape with no separate working frame keeps the v1 path: nothing is handed over
    build_v1 = _intents_fn(pd.DataFrame(), pd.DataFrame(), StrategyConfig(), None)

    build_v1(bars, bars)

    assert handed[1] is None
