"""The search space of the Э7' optimizer: which knob a trial may turn, and how far.

The module is deliberately free of optuna.  A range is a small frozen dataclass that
knows its bounds and asks a *trial-like* object for a value, i.e. it speaks nothing but
the three ``suggest_*`` methods of an :class:`optuna.Trial` (see :class:`TrialLike`).
That keeps the space describable, testable and reviewable without the optimizer library
installed, and it keeps the Э7' layer importable when it is not.

Two bridges connect the space to the configuration layer:

* :func:`suggest_params` turns a trial into ``{path: value}`` - the shape of
  ``trial.params`` / ``study.best_params``, which is why a path *is* the parameter name;
* :func:`apply_params` copies a :class:`~smc_zero.config.StrategyConfig` with those
  values set, one nesting level at a time, so the trial never leaks into the config.

A path is a dotted field name of :class:`~smc_zero.config.StrategyConfig`
(``"take_profit.rr_fallback"``) and :func:`resolve_path` reads it back; the tests use
that reader to prove every range names a real field, so a typo fails loudly instead of
silently optimizing nothing.

A trial may vary the paths of :data:`PARAM_RANGES` and nothing else: the markup cache of
:mod:`smc_zero.optimizer.marks` is built once per run from the parts of the config the
space does not reach, and :func:`smc_zero.optimizer.marks.cache_mismatches` refuses a
config that would have made it stale.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol, TypeAlias

from smc_zero.config import AGREEMENTS, StrategyConfig

#: The value a range may produce: the ints / floats / categories of the trial API.
ParamValue: TypeAlias = int | float | str


class TrialLike(Protocol):
    """The slice of ``optuna.Trial`` the search space asks for a value.

    ``optuna`` is imported lazily (see :mod:`smc_zero.optimizer.optimize`), so the space
    is typed against this protocol instead of against the library: it is what a real
    trial and the test stub both look like, and these three methods are the whole
    contract.
    """

    def suggest_int(self, name: str, low: int, high: int, step: int = 1, log: bool = False) -> int:
        """Ask for an integer in ``[low, high]`` (optuna's inclusive bounds)."""
        ...

    def suggest_float(
        self, name: str, low: float, high: float, step: float | None = None, log: bool = False
    ) -> float:
        """Ask for a float in ``[low, high]``, optionally on a ``step`` grid."""
        ...

    def suggest_categorical(self, name: str, choices: Sequence[str]) -> str:
        """Ask for one of ``choices``."""
        ...

    def set_user_attr(self, name: str, value: Any) -> None:
        """Attach a result to the trial: the aggregates a study is inspected through."""
        ...


@dataclass(frozen=True, slots=True)
class IntRange:
    """An integer knob of the strategy.

    ``low`` and ``high`` are *inclusive* (optuna's contract) and ``low < high`` always -
    a range that cannot vary is a constant of the config, not a search dimension.
    ``log=True`` searches the orders of magnitude, which optuna allows for integers only
    with ``step=1``; it is unused by :data:`PARAM_RANGES` and kept because it is part of
    the trial API this module mirrors.
    """

    low: int
    high: int
    step: int = 1
    log: bool = False

    def suggest(self, trial: TrialLike, name: str) -> int:
        """Ask ``trial`` for one value of this range under the name ``name``.

        optuna's integer bounds are inclusive, so ``high`` is reachable - which is what the
        ranges of :data:`PARAM_RANGES` assume (a knob searched "up to 5" includes 5).
        """
        return int(trial.suggest_int(name, self.low, self.high, step=self.step, log=self.log))

    def __post_init__(self) -> None:
        if self.low >= self.high:
            raise ValueError(f"low must be < high, got [{self.low}, {self.high}]")
        if self.step < 1:
            raise ValueError(f"step must be >= 1, got {self.step}")
        if self.log and self.step != 1:
            raise ValueError("a log-scaled integer range cannot have a step != 1")


@dataclass(frozen=True, slots=True)
class FloatRange:
    """A continuous (or gridded) float knob of the strategy.

    ``step=None`` is a continuous range; a value of ``step`` is a grid optuna is free to
    round to.  ``log=True`` needs a strictly positive ``low`` and cannot be combined with
    a ``step`` - the two rules optuna itself enforces.
    """

    low: float
    high: float
    step: float | None = None
    log: bool = False

    def __post_init__(self) -> None:
        if self.low >= self.high:
            raise ValueError(f"low must be < high, got [{self.low}, {self.high}]")
        if self.step is not None and self.step <= 0:
            raise ValueError(f"step must be > 0, got {self.step}")
        if self.log and (self.step is not None or self.low <= 0):
            raise ValueError("a log-scaled float range needs low > 0 and no step")

    def suggest(self, trial: TrialLike, name: str) -> float:
        """Ask ``trial`` for one value of this range under the name ``name``."""
        return float(trial.suggest_float(name, self.low, self.high, step=self.step, log=self.log))


@dataclass(frozen=True, slots=True)
class ChoiceRange:
    """A categorical knob: the one parameter that is not a number.

    The A/B factor of the bias (:attr:`~smc_zero.config.BiasConfig.agreement`) is the
    reason the class exists - optuna's ``suggest_categorical`` is how a study switches
    between two readings of the same rule.
    """

    choices: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.choices:
            raise ValueError("choices must not be empty")
        if len(set(self.choices)) != len(self.choices):
            raise ValueError(f"choices must be unique, got {self.choices}")

    def suggest(self, trial: TrialLike, name: str) -> str:
        """Ask ``trial`` for one value of this range under the name ``name``."""
        return str(trial.suggest_categorical(name, list(self.choices)))


#: Every kind of range a trial of Э7' may be handed.
ParamRange: TypeAlias = IntRange | FloatRange | ChoiceRange


#: The search space of Э7' (SPEC_SMC.md §7.11 п.70): dotted config path -> its range.
#: The first six knobs are the pip-sized gates of the entry chain, the next four the
#: take-profit and displacement thresholds, and the last one the A/B factor of the bias.
#: The bounds are a *trading* decision, not a code one: they bracket the prod defaults
#: (``sweep_buffer_pip=1``, ``sl_buffer_pip=5``, ``min_fvg_pip=1``, ``fvg_lookback=20``,
#: ``max_fvg_age_bars=12``, ``signal_max_age_bars=60``, ``take_profit.rr_fallback=2``,
#: ``take_profit.min_tp_rr=1``), so the study starts inside the region the spec reasoned
#: about.  A knob deliberately absent is the level map / session table: those are the
#: inputs of the markup cache (see :mod:`smc_zero.optimizer.marks`).
PARAM_RANGES: dict[str, ParamRange] = {
    "sweep_buffer_pip": IntRange(1, 5),
    "sl_buffer_pip": IntRange(3, 15),
    "min_fvg_pip": IntRange(1, 5),
    "fvg_lookback": IntRange(10, 30),
    "max_fvg_age_bars": IntRange(5, 20),
    "signal_max_age_bars": IntRange(20, 100),
    "take_profit.rr_fallback": FloatRange(1.5, 3.0),
    "take_profit.min_tp_rr": FloatRange(0.8, 1.5),
    "displacement.atr_mult_min": FloatRange(0.5, 2.0),
    "displacement.body_frac_min": FloatRange(0.3, 0.8),
    "bias.agreement": ChoiceRange(AGREEMENTS),
}


def resolve_path(cfg: StrategyConfig, path: str) -> Any:
    """Return what the dotted ``path`` of ``cfg`` points at.

    ``"sweep_buffer_pip"`` is a field of the config itself, ``"take_profit.rr_fallback"``
    a field of a nested one.  A path that does not resolve raises ``KeyError`` naming the
    part that failed: the search space may not name a knob the config does not have.
    """
    target: Any = cfg
    walked: list[str] = []
    for part in path.split("."):
        if not hasattr(target, part):
            raise KeyError(
                f"{path!r} does not name a field of "
                f"{'.'.join(walked) or type(cfg).__name__}: no {part!r}"
            )
        walked.append(part)
        target = getattr(target, part)
    return target


def suggest_params(
    trial: TrialLike,
    ranges: Mapping[str, ParamRange] = PARAM_RANGES,
) -> dict[str, ParamValue]:
    """Ask ``trial`` for one value per range and return them by their dotted names.

    The result is what :func:`apply_params` consumes and what optuna reports back as
    ``trial.params`` / ``study.best_params``, so a best-parameter mapping can be replayed
    without knowing this module.  ``ranges`` is an argument rather than only the module
    constant because a study over a *subset* of the space is a legitimate experiment.
    """
    return {name: param_range.suggest(trial, name) for name, param_range in ranges.items()}


def apply_params(cfg: StrategyConfig, params: Mapping[str, ParamValue]) -> StrategyConfig:
    """Return ``cfg`` with every ``{path: value}`` of ``params`` set.

    The config is copied level by level with :func:`dataclasses.replace`, so the input is
    never touched and only the objects on a changed path are new.  An unknown path raises
    ``KeyError`` (see :func:`resolve_path`); a value of the wrong type is left to the
    validation of the dataclass that receives it.
    """
    result = cfg
    for path, value in params.items():
        parts = path.split(".")
        owners: list[Any] = [result]
        for part in parts[:-1]:
            try:
                owners.append(getattr(owners[-1], part))
            except AttributeError as exc:
                raise KeyError(f"{path!r} does not resolve under {'.'.join(parts[:-1])!r}") from exc
        if not hasattr(owners[-1], parts[-1]):
            raise KeyError(f"{path!r} does not name a field of {type(owners[-1]).__name__}")
        updated: Any = replace(owners[-1], **{parts[-1]: value})
        for depth in range(len(parts) - 2, -1, -1):
            updated = replace(owners[depth], **{parts[depth]: updated})
        result = updated
    return result
