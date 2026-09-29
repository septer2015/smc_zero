"""The markup cache of one optimization run: the bias trends and the level book (Э7').

The entry chain of Э4' reads two whole-tape artifacts besides the bars themselves - the
HTF bias markup and the level book with its ``broken_at`` stamps - and neither depends on
the parameters a trial suggests.  Computing them per trial would mean 100 trials x 6
heavy indicator calls over the same tape; the cache computes them **once per run** and
hands every trial the same frames (SPEC_SMC.md §7.11 п.71).

What is cached, and why it stays legal (the no-lookahead argument of the constitution,
rule 2, applied to a cache):

* the trends come from :func:`smc_zero.indicators.bias.bias_frames`, i.e. closed HTF
  bars stitched on the LTF's ``open_time`` by ``close_time`` - the column of a bar only
  ever summarises bars that had *closed* before it, so a bar after the fold cannot change
  what a bar inside the fold reads;
* the level book comes from :func:`smc_zero.indicators.levels.level_lifecycle`, whose
  ``broken_at`` is a *timestamp*: a break that happens later leaves the level fresh at an
  earlier bar, exactly as a fold-local computation would - the cache adds the warm-up
  history an isolated window would have missed, not the future.

Only the row-level *verdict* of the bias is recomputed per trial, and it is cheap:
:meth:`TapeMarks.bias_frame` re-classifies the cached trend matrix, which is how the A/B
factor (``BiasConfig.agreement``) can be a search dimension without paying for the
markup again.  :func:`cache_mismatches` is the guard of the split: a configuration that
changes anything the cache was built from - a swing setting, the killzone table, the
level map - is refused instead of being served a stale markup.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import pandas as pd

from smc_zero.config import Agreement, StrategyConfig
from smc_zero.data_loader import TIMESTAMP_COLUMN, drop_unclosed, resample_to_timeframe
from smc_zero.indicators.bias import (
    BIAS_DIR_COLUMN,
    BIAS_STATE_COLUMN,
    bias_frames,
    classify_trends,
    trend_column,
)
from smc_zero.indicators.levels import level_lifecycle, static_levels
from smc_zero.optimizer.ranges import PARAM_RANGES


@dataclass(frozen=True, slots=True)
class TapeMarks:
    """The markup of one tape, computed once and read by every trial of a study.

    ``trends`` is the markup of :func:`bias_frames` for the configuration the cache was
    built with (``timestamp``, ``trend_<tf>``, ``bias_dir``, ``bias_state``), ``levels``
    the book of :func:`level_lifecycle` (one row per level instance, ``broken_at``
    included), and ``config`` the :class:`~smc_zero.config.StrategyConfig` they were built
    from - the record :func:`cache_mismatches` compares a trial's configuration against.
    """

    trends: pd.DataFrame
    levels: pd.DataFrame
    config: StrategyConfig

    def bias_frame(self, agreement: Agreement | None = None) -> pd.DataFrame:
        """Return the bias markup for one agreement mode, off the cached trends.

        The expensive half of :func:`bias_frames` - the HTF resampling, the structure
        automaton of every timeframe and the ``merge_asof`` onto the M15 bars - is already
        in :attr:`trends`; only the per-row verdict is recomputed, so a trial that
        suggests the A/B factor pays microseconds for it instead of a full markup.

        ``agreement=None`` keeps the mode the cache was built with, and the frame is then
        row for row what ``bias_frames`` returns - the invariant the tests pin.
        """
        mode = self.config.bias.agreement if agreement is None else agreement
        columns = [trend_column(timeframe) for timeframe in self.config.bias.timeframes]
        markup = self.trends.loc[:, [TIMESTAMP_COLUMN, *columns]].copy()
        bias_dir, bias_state = classify_trends(
            markup.loc[:, columns].to_numpy(dtype="float64"), mode
        )
        markup[BIAS_DIR_COLUMN] = bias_dir
        markup[BIAS_STATE_COLUMN] = bias_state
        return markup


def build_tape_marks(df: pd.DataFrame, cfg: StrategyConfig | None = None) -> TapeMarks:
    """Build the markup cache of one tape: six heavy calls, once, for the whole study.

    The six are the three :func:`~smc_zero.data_loader.resample_to_timeframe` calls (one
    per bias timeframe), :func:`~smc_zero.indicators.bias.bias_frames`,
    :func:`~smc_zero.indicators.levels.static_levels` and
    :func:`~smc_zero.indicators.levels.level_lifecycle`; the guard test of Э7' counts them
    so that a refactor cannot quietly move them back into the per-trial path (100 trials
    would be 600 calls).

    ``df`` is the whole M15 tape; its presumed still-forming tail bar is dropped here
    (rule 2b), and the HTF frames are resampled from the *closed* tape, so the live edge
    never enters the cache.  ``cfg`` supplies the inputs - the bias timeframes and swing
    settings, the killzone table of the level windows and the level map - and defaults to
    a plain :class:`~smc_zero.config.StrategyConfig`; the trials of a study must leave
    exactly those parts alone (:func:`cache_mismatches` enforces it).
    """
    config = StrategyConfig() if cfg is None else cfg
    closed = drop_unclosed(df)
    htf_frames = {
        timeframe: resample_to_timeframe(closed, timeframe)
        for timeframe in config.bias.timeframes
    }
    trends = bias_frames(closed, htf_frames, config.bias)
    levels = level_lifecycle(
        static_levels(closed, config.levels, session_cfg=config.session),
        closed,
        config.levels,
    )
    return TapeMarks(trends=trends, levels=levels, config=config)


def _leaf_paths(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested frozen dataclass into ``{dotted path: leaf value}``.

    Nested dataclasses are walked; everything else (ints, floats, strings, tuples, enums
    of a config) is a leaf.  ``pip_size`` / ``contract_size`` are properties of
    :class:`~smc_zero.config.StrategyConfig` and not fields, so they are not walked - they
    are derived from the risk profile, which is a leaf in its own right.
    """
    if is_dataclass(value) and not isinstance(value, type):
        leaves: dict[str, Any] = {}
        for info in fields(value):
            leaves.update(_leaf_paths(getattr(value, info.name), f"{prefix}{info.name}."))
        return leaves
    return {prefix[:-1]: value}


def cache_mismatches(marks: TapeMarks, cfg: StrategyConfig) -> tuple[str, ...]:
    """Return the config paths whose value makes the cache stale - empty when it is valid.

    The cache may only serve trials that vary the paths of
    :data:`smc_zero.optimizer.ranges.PARAM_RANGES`: everything else the configuration
    holds was *read* while the cache was built (the swing settings behind the trends, the
    killzone table behind the level windows, the level map itself), so a changed value
    there means the markup on the table belongs to a different strategy.  The comparison
    is field by field over the whole tree, which is why a knob added to the config later
    is covered without touching this function.

    The A/B factor is the one exception and deliberately so:
    :attr:`~smc_zero.config.BiasConfig.agreement` is recomputed per trial by
    :meth:`TapeMarks.bias_frame`, so it is allowed to differ from the cached value.
    """
    cached = _leaf_paths(marks.config)
    asked = _leaf_paths(cfg)
    varied = set(PARAM_RANGES)
    return tuple(
        path
        for path, value in cached.items()
        if path not in varied and asked.get(path, object()) != value
    )
