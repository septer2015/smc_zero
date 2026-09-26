"""Guard: ./src/ must never import or reference the read-only prod sources.

Per constitution (.clinerules p.7) and SPEC_SMC.md (S0 p.5), ./reference/* may be read
for semantics only -- import and verbatim copying are forbidden. The whole ./src/ tree
is scanned here as plain text, so nothing from ./reference/ is imported or executed.
"""

from __future__ import annotations

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
FORBIDDEN_TOKENS = (
    "reference",
    "import core",
    "from core",
    "ict_smc_backtest",
)


def _src_python_files() -> list[Path]:
    return sorted(SRC_DIR.rglob("*.py"))


def test_src_does_not_import_reference() -> None:
    files = _src_python_files()
    assert files, f"no .py files found under {SRC_DIR}"

    violations: list[str] = []
    for path in files:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for token in FORBIDDEN_TOKENS:
                if token in line:
                    rel = path.relative_to(SRC_DIR.parent)
                    violations.append(f"{rel}:{line_no}: forbidden token {token!r}")

    assert not violations, "src/ must not reference the prod sources:\n" + "\n".join(violations)
