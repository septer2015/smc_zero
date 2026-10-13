"""``python -m scripts.paper_snapshot``: one reading of the night as JSON (Э17.z++).

The daily check of Э17.z++ measures the same night twice - once with the live config of mode A and
once with the research candidate of ``configs/research_e17zp_candidate.yaml`` - and the two
readings have to be comparable without a human copying numbers out of two report folders.  This
module turns the artifacts a run already wrote into one JSON document per reading, so
``docs/PAPER_JOURNAL.md`` can be filled from the files and not from a screen:

* ``trades.csv`` of the report folder, filtered exactly like the list the shell face prints
  (``open_time`` of the first 19 characters ``>= since``), gives the signals of the night;
* ``summary.txt`` of the same folder gives the metrics of the run, e.g. how many trades the whole
  four-year window carried.

Nothing is recomputed, searched or traded here.  ``scripts/daily_signal_check.sh`` stays the only
thing that starts a backtest (the face of Э10'.4), and the runner stays the only thing that writes
a report; this helper is a reader of that artifact and of nothing else.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

#: The stable label of a row of ``summary.txt`` -> the key of the JSON document.  The labels are the
#: ones a reader of the report sees (``smc_zero.backtester.reports._SUMMARY_FIELDS``); a row that is
#: missing or unreadable is left out instead of failing the snapshot - the signals of the night are
#: the point of the file, the metrics are context.
SUMMARY_LABELS: dict[str, str] = {
    "trades": "trades",
    "win rate": "win_rate",
    "profit": "profit",
    "profit factor": "profit_factor",
    "max drawdown": "max_drawdown",
    "expected per trade": "expected_per_trade",
    "final balance": "final_balance",
}
#: The first number of a summary value: the label column and the value are separated by spaces.
NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")
#: The length of the stamp prefix the night threshold is compared against (``YYYY-MM-DD HH:MM:SS``).
STAMP_PREFIX = 19


def read_metrics(path: Path) -> dict[str, float]:
    """Return the metric rows of ``summary.txt`` keyed by :data:`SUMMARY_LABELS`, unknown rows out."""
    if not path.is_file():
        return {}
    found: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = re.split(r"\s{2,}", line.strip(), maxsplit=1)
        if len(parts) != 2 or parts[0] not in SUMMARY_LABELS:
            continue
        match = NUMBER.match(parts[1].replace("+", ""))
        if match:
            found[SUMMARY_LABELS[parts[0]]] = float(match.group(0))
    return found


def read_signals(path: Path, since: str) -> list[dict[str, str]]:
    """Return the trades opened at or after ``since`` - the ``open_time >=`` rule of the shell face.

    The comparison is the one ``awk`` makes in ``scripts/daily_signal_check.sh``: the first
    :data:`STAMP_PREFIX` characters of the stamp against the threshold, so the boundary of the night
    is included and the printed list and the JSON document cannot disagree about it.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row.get("open_time", "")[:STAMP_PREFIX] >= since]


def build_parser() -> argparse.ArgumentParser:
    """Return the parser of ``paper_snapshot``: the artifact, the reading and the output file."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.paper_snapshot",
        description="Write one reading of the night (a report folder) as JSON of the paper journal.",
    )
    parser.add_argument("--report", required=True, help="report folder of the run to read")
    parser.add_argument(
        "--role", required=True, help="name of the reading: ``base`` or a candidate"
    )
    parser.add_argument("--config", required=True, help="YAML config the run was started with")
    parser.add_argument(
        "--since", required=True, help="threshold of a new trade, 'YYYY-MM-DD HH:MM:SS' UTC"
    )
    parser.add_argument("--window-start", default=None, help="start of the window of the run")
    parser.add_argument("--window-end", default=None, help="end of the window of the run")
    parser.add_argument("--date", default=None, help="day of the reading (default: today, UTC)")
    parser.add_argument("--out", required=True, help="JSON file to write")
    return parser


def snapshot(args: argparse.Namespace) -> dict[str, object]:
    """Return the document of one reading: its role, its sources, its metrics and its signals."""
    report = Path(args.report).resolve()
    trades = report / "trades.csv"
    if not trades.is_file():
        raise FileNotFoundError(f"no trade log at {trades}")
    if len(args.since) != STAMP_PREFIX:
        raise ValueError(f"--since must be 'YYYY-MM-DD HH:MM:SS', got {args.since!r}")
    signals = read_signals(trades, args.since)
    day = args.date or datetime.now(UTC).strftime("%Y-%m-%d")
    return {
        "date": day,
        "role": args.role,
        "config": str(Path(args.config).resolve()),
        "report": str(report),
        "window": {"start": args.window_start, "end": args.window_end},
        "since_utc": args.since,
        "metrics": read_metrics(report / "summary.txt"),
        "signal_count": len(signals),
        "signals": signals,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Write the JSON document of ``argv`` (``None`` = ``sys.argv``); the console entry point."""
    args = build_parser().parse_args(argv)
    try:
        document = snapshot(args)
    except (OSError, ValueError) as error:
        print(f"error: {error}")
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"paper snapshot: {out} ({document['signal_count']} signal(s) of role {args.role})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
