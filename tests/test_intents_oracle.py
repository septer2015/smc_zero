"""The oracle gate of Э9': the vectorised chain answers what the Э4' walk answered.

The rewrite of :func:`smc_zero.strategy.intents.build_intents` is accepted on *equivalence*, not
on being green: every gate, every reason and every positional bar of the old walk has to come out
of the new one unchanged (SPEC_SMC.md §7.13).  The reference is the code itself, taken from a
``git worktree`` at the commit Э9' started from - a pinned historical commit, never a copy inside
the repository, so the two implementations cannot drift into agreement.

What is compared, and how:

* the worktree runs its own *pristine* chain through a pytest plugin that records every call the
  Э4' suite makes on synthetic tapes - ``tests/test_intents.py``, the only module of that commit
  that feeds the chain synthetic tapes, so "вся синтетика" of the Э9' plan - plus one dedicated
  recording on the real 3-month window (``EURUSD_M15`` 2022-08-15 .. 2022-11-15), the same window
  the Э8' runners use;
* every recording holds the *arguments* of the call, the accepted intents and the rejection
  ledger, and the replayed call is compared face by face with
  :func:`pandas.testing.assert_frame_equal` in its exact mode (``check_exact=True``): the intents
  frame (order included, since the frame keeps the acceptance order) and the ledger, dtypes and
  all - "byte for byte", because a last-bit difference in a price would mean the two chains do not
  compute the same number even where no test can see the gap;
* the worktree is removed after the run, and nothing of the old implementation is copied into
  the tree.

The one place the two chains are *meant* to disagree is the same-bar rule Э9''.1 added to п.35:
prod - and therefore the pinned head - placed every setup of a bar, while the new chain keeps the
most significant level of it (:func:`smc_zero.strategy.intents._one_intent_per_bar`).  The record
of the old walk is therefore filtered by an independent statement of that rule
(:func:`_one_intent_per_bar` below, written out here rather than imported from the chain, so that
the two statements can disagree) and *then* compared; the ledger is compared unfiltered, because a
setup dropped by the rule is not a refused attempt and the new chain writes no row for it either.

The mutation this file is built to catch, and the test that must break:

* m3 "comment out the comparison" - the oracle would then be a recorder, so
  :func:`test_the_oracle_test_really_compares_the_two_chains` reads this file as code and fails
  when the comparison disappears from it.

The pin is a historical sha on purpose.  If it ever stops resolving, the oracle cannot answer and
the suite fails loudly instead of skipping the gate.
"""

from __future__ import annotations

import ast
import os
import pickle
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pandas.testing import assert_frame_equal

from smc_zero.indicators.levels import (
    ASIAN_HIGH,
    ASIAN_LOW,
    LONDON_HIGH,
    LONDON_LOW,
    NY_HIGH,
    NY_LOW,
    PDH,
    PDL,
    PMH,
    PML,
    PWH,
    PWL,
)
from smc_zero.strategy.base import intents_frame
from smc_zero.strategy.intents import build_intents

REPO = Path(__file__).resolve().parents[1]
#: The commit Э9' started from (the Э8' head): the pinned implementation of the Э4' walk.
PRISTINE_COMMIT = "5a9e015"
#: The test modules whose calls are recorded in the worktree.  ``tests/test_intents.py`` is the one
#: module of the pristine suite that hands *synthetic* tapes to the chain, so recording it covers
#: every synthetic input of that commit; the other callers reach the chain through the real tape
#: (the runners, the optimizer integration test) and through the module below, which does not exist
#: at the pinned commit (Э9' added it) and therefore cannot be recorded at all - its own synthetic
#: cases are pinned by ``tests/test_intents_windows.py`` instead.
RECORDED_MODULES: tuple[str, ...] = ("tests/test_intents.py",)
#: The real window of the oracle: the first three months of the shipped tape.
REAL_WINDOW: tuple[str, str] = ("2022-08-15", "2022-11-15")
#: The pytest plugin the worktree loads with ``-p``: it patches the *module attribute* of the
#: pristine chain before any test module is imported, so the ``build_intents`` the suite binds at
#: import time is the recorder (a later patch of the name would simply miss it).
RECORDER_PLUGIN = '''"""Plugin of the Э9' oracle: record every call of the pristine chain."""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import smc_zero.strategy.intents as chain

_original = chain.build_intents
_out = Path(os.environ["ORACLE_OUT"])
_count = 0


def recording(*args, **kwargs):
    """Record one call: the arguments, the accepted intents and the rejection ledger."""
    global _count
    out = _original(*args, **kwargs)
    _count += 1
    with (_out / f"synthetic_{_count:04d}.pkl").open("wb") as handle:
        pickle.dump(
            {
                "args": args,
                "kwargs": kwargs,
                "intents": out.intents,
                "rejections": out.rejections,
            },
            handle,
        )
    return out


chain.build_intents = recording
'''

#: The child script of the real window.  It refuses to run on the wrong package: the whole gate
#: rests on the child importing the *worktree* and not the working tree (an editable install is on
#: the path as well, so "it imported something named ``smc_zero``" is not enough).
REAL_SCRIPT = '''"""Record the pristine chain on the real window - run inside the Э9' worktree."""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import smc_zero

ROOT = os.environ["ORACLE_ROOT"]
if not smc_zero.__file__.startswith(ROOT):
    raise SystemExit(f"the worktree package is not the one imported: {smc_zero.__file__}")

sys.path.insert(0, ROOT)

from scripts import _common  # noqa: E402
from smc_zero.config import StrategyConfig  # noqa: E402
from smc_zero.optimizer import build_tape_marks  # noqa: E402
from smc_zero.strategy.intents import build_intents  # noqa: E402

config = StrategyConfig()
start = _common.read_day(os.environ["ORACLE_START"])
end = _common.read_day(os.environ["ORACLE_END"])
tape = _common.load_windowed_tape("EURUSD", "M15", start, end)
marks = build_tape_marks(tape, config)
chain = build_intents(tape, marks.bias_frame(), marks.levels, config)
with Path(os.environ["ORACLE_OUT"], "real_window.pkl").open("wb") as handle:
    pickle.dump(
        {
            "args": (tape, marks.bias_frame(), marks.levels, config),
            "kwargs": {},
            "intents": chain.intents,
            "rejections": chain.rejections,
        },
        handle,
    )
print(f"recorded {len(chain.intents)} intents and {len(chain.rejections)} rejections")
'''

#: Where the recorded cases land inside the temporary directory of the test.
CASES_DIRNAME = "pristine_cases"
#: The name of the helper that replays one case; the meta test finds it by that name.
REPLAY_HELPER = "_replay"
#: The name of the test the meta test guards.
ORACLE_TEST = "test_the_vectorized_chain_matches_the_pristine_head"


@contextmanager
def _worktree(path: Path) -> Iterator[Path]:
    """Hand over a worktree of ``PRISTINE_COMMIT`` and take it away again, whatever happens.

    The gate reads the *code* of the Э4' walk from git, so the old implementation stays in history
    instead of lying around in the tree as a second copy that could drift.
    """
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(path), PRISTINE_COMMIT],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        yield path
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(path)],
            cwd=REPO,
            check=False,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "worktree", "prune"], cwd=REPO, check=False, capture_output=True, text=True
        )


def _child_env(worktree: Path, cases: Path, plugins: Path) -> dict[str, str]:
    """Return the environment of a child that has to import the worktree's package.

    ``PYTHONPATH`` puts the worktree first, which is what beats the editable install of the working
    tree; the child script re-checks it, because a silent import of the wrong package would turn
    this gate into a comparison of the new chain with itself.
    """
    path = os.pathsep.join([str(worktree / "src"), str(worktree), str(plugins)])
    return {
        **os.environ,
        "PYTHONPATH": path,
        "ORACLE_ROOT": str(worktree),
        "ORACLE_OUT": str(cases),
        "ORACLE_START": REAL_WINDOW[0],
        "ORACLE_END": REAL_WINDOW[1],
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _run(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    """Run a child of the oracle and fail with its own output when it does not answer cleanly."""
    outcome = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, check=False)
    assert outcome.returncode == 0, (
        f"{' '.join(command)}\n{outcome.stdout[-4000:]}\n{outcome.stderr[-4000:]}"
    )


def _record_pristine_cases(tmp_path: Path) -> list[dict[str, Any]]:
    """Record the pristine chain on the suite's synthetic tapes and on the real window."""
    cases = tmp_path / CASES_DIRNAME
    cases.mkdir()
    plugins = tmp_path / "oracle_plugins"
    plugins.mkdir()
    (plugins / "oracle_recorder.py").write_text(RECORDER_PLUGIN, encoding="utf-8")
    script = tmp_path / "record_real_window.py"
    script.write_text(REAL_SCRIPT, encoding="utf-8")
    with _worktree(tmp_path / "pristine") as worktree:
        env = _child_env(worktree, cases, plugins)
        _run([sys.executable, str(script)], worktree, env)
        _run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "-p",
                "oracle_recorder",
                *RECORDED_MODULES,
            ],
            worktree,
            env,
        )
    return [pickle.loads(path.read_bytes()) for path in sorted(cases.glob("*.pkl"))]


#: The level significance of prod's ``LEVEL_PRIORITY`` table (``core.py`` lines 122-138), as an
#: order the oracle states for itself: PDH/PDL above PWH/PWL above PMH/PML above the session
#: ranges, Asian before London before NY.
LEVEL_SIGNIFICANCE: dict[str, int] = {
    PDH: 1,
    PDL: 1,
    PWH: 2,
    PWL: 2,
    PMH: 3,
    PML: 3,
    ASIAN_HIGH: 4,
    ASIAN_LOW: 4,
    LONDON_HIGH: 5,
    LONDON_LOW: 5,
    NY_HIGH: 6,
    NY_LOW: 6,
}


def _one_intent_per_bar(intents: tuple[Any, ...]) -> tuple[Any, ...]:
    """Return ``intents`` with the same-bar duplicates of п.35 dropped - the Э9''.1 rule, restated.

    prod left its loop over a bar at the first accepted setup, so a bar carried one order at most;
    the new chain keeps the instance standing higher in ``LEVEL_PRIORITY``, the instance the walk met
    first winning a tie, and leaves the survivors in the order they were accepted in.  The statement
    is deliberately this file's own: an oracle that imported the rule from the chain would agree with
    any bug in it.
    """
    best: dict[int, tuple[int, int]] = {}
    for index, intent in enumerate(intents):
        rank = LEVEL_SIGNIFICANCE.get(str(intent.level_name), 99)
        seen = best.get(intent.bar)
        if seen is None or rank < seen[0]:
            best[intent.bar] = (rank, index)
    keep = {index for _, index in best.values()}
    return tuple(intent for index, intent in enumerate(intents) if index in keep)


def _replay(case: dict[str, Any]) -> None:
    """Run the current chain on the recorded arguments and compare both faces of the verdict.

    The intents are compared as frames, which keeps the acceptance order (the frame is built in the
    order of the tuple) as well as every value and dtype; the recorded intents are filtered by the
    same-bar rule of Э9''.1 first (:func:`_one_intent_per_bar`), the ledger is not.  Nothing is
    normalised and nothing is allowed to be "close enough": this is the gate Э9' is accepted on
    (SPEC_SMC.md §7.13).
    """
    out = build_intents(*case["args"], **case["kwargs"])
    expected = _one_intent_per_bar(case["intents"])
    assert_frame_equal(intents_frame(out.intents), intents_frame(expected))
    assert_frame_equal(out.rejections, case["rejections"])


def test_the_vectorized_chain_matches_the_pristine_head(tmp_path: Path) -> None:
    """Э9' gate: the same intents and the same ledger as the Э4' walk, byte for byte (m1, m2).

    The cases are the synthetic tapes of the suite itself - recorded while the pristine chain runs
    *its* tests in the worktree - plus the real three-month window of the shipped tape.
    """
    cases = _record_pristine_cases(tmp_path)
    assert len(cases) > 1, "the oracle needs the synthetic cases and the real window"
    for case in cases:
        _replay(case)


def _frame_equal_calls(node: ast.AST) -> list[ast.Call]:
    """Return the ``assert_frame_equal(...)`` calls of ``node``, however they are spelled."""
    found: list[ast.Call] = []
    for candidate in ast.walk(node):
        if not isinstance(candidate, ast.Call):
            continue
        func = candidate.func
        name = func.id if isinstance(func, ast.Name) else str(getattr(func, "attr", ""))
        if name == "assert_frame_equal":
            found.append(candidate)
    return found


def test_the_oracle_test_really_compares_the_two_chains() -> None:
    """m3 of §7.13: an oracle whose comparison is commented out is a recorder, not a gate.

    The check reads this file as code and demands the three pieces the gate is made of: the test
    records the pristine cases, it replays them, and the replay compares *both* faces of the verdict
    with :func:`pandas.testing.assert_frame_equal`.  Comment any of them out and the oracle stops
    being able to fail, which is exactly what this test refuses to let pass.
    """
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert ORACLE_TEST in functions, "the oracle test is expected under its documented name"
    called = {
        node.func.id
        for node in ast.walk(functions[ORACLE_TEST])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_record_pristine_cases" in called, "the oracle has to record the pristine chain"
    assert REPLAY_HELPER in called, "the oracle has to replay what it recorded"
    assert REPLAY_HELPER in functions, f"{REPLAY_HELPER} is the comparison of the oracle"
    comparisons = _frame_equal_calls(functions[REPLAY_HELPER])
    assert len(comparisons) == 2, "both faces of the verdict have to be compared frame by frame"
    pinned = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and node.value == PRISTINE_COMMIT
    ]
    assert pinned, "the worktree has to be pinned to the commit Э9' started from"
