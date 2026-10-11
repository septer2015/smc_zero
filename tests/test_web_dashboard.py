"""Tests of the standalone dashboard ``tools/web_dashboard`` (Э14).

The dashboard is a showcase: it computes nothing in the strategy and draws what the backtester
has already written.  Exactly that is verified here: the demo mode builds a page with Plotly from
the CDN, a real report yields the KPIs from the ``trades.csv`` and ``summary.txt`` columns, a
missing ``summary.txt`` does not break the build (the symbol and the timeframe come from the
folder name), an empty ``trades.csv`` gives a clear notice instead of a crash, the comparison of
two reports adds its own section, and a broken input fails with a clear message.

Plotly is an optional dependency, so the whole module is skipped when it is absent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("plotly")

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "web_dashboard" / "scripts" / "generate_dashboard.py"

TRADE_COLUMNS = (
    "fill_time",
    "side",
    "setup_type",
    "level_name",
    "entry",
    "sl",
    "tp",
    "rr",
    "profit",
    "result",
    "bars_held",
    "limit_dur_min",
)

SUMMARY = """SMC backtest: EURUSD M15 (2025-06-02 08:00 .. 2025-06-05 10:30 UTC)
  initial capital          10000.00
  trades                   3
  win rate                 66.67 %
  profit                   +80.00
  profit factor            5.000
  max drawdown             1.85 %
  sharpe (per bar)         0.032
"""


def _load_tool():
    """Load the tool by path: it is a standalone script, not a package module."""
    spec = importlib.util.spec_from_file_location("web_dashboard_generator", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dashboard = _load_tool()


def _trades() -> pd.DataFrame:
    """Three trades with a known sum: +40 - 20 + 60 = +80."""
    return pd.DataFrame(
        {
            "fill_time": ["2025-06-02 08:15", "2025-06-03 09:00", "2025-06-04 10:30"],
            "side": ["long", "short", "long"],
            "setup_type": ["fresh", "bos", "choch"],
            "level_name": ["LondonL", "AsianH", "NYL"],
            "entry": [1.1000, 1.1050, 1.1020],
            "sl": [1.0980, 1.1075, 1.1000],
            "tp": [1.1040, 1.1000, 1.1060],
            "rr": [2.0, 2.0, 2.0],
            "profit": [40.0, -20.0, 60.0],
            "result": [1, -1, 1],
            "bars_held": [10, 5, 20],
            "limit_dur_min": [15, 30, 45],
        }
    )


def _write_report(folder: Path, trades: pd.DataFrame, summary: str | None = SUMMARY) -> Path:
    """Assemble a report folder the way the backtester leaves it."""
    folder.mkdir(parents=True, exist_ok=True)
    trades.to_csv(folder / "trades.csv", index=False)
    if summary is not None:
        (folder / "summary.txt").write_text(summary, encoding="utf-8")
    return folder


def metric_value(html: str, label: str) -> str:
    """Pull the value of a KPI card by its label (needed to check the number format)."""
    marker = f'<span class="k">{label}</span><span class="v">'
    start = html.index(marker) + len(marker)
    return html[start : html.index("</span>", start)]


def test_demo_writes_a_standalone_page_with_plotly_from_cdn(tmp_path: Path) -> None:
    """The demo mode writes one file, pulls Plotly from the CDN and draws more than one panel."""
    output = tmp_path / "public" / "demo.html"

    assert dashboard.main(["--demo", "--output", str(output)]) == 0

    html = output.read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>")
    assert dashboard.PLOTLY_CDN in html
    assert html.rstrip().endswith("</html>")
    assert html.count("plotly-graph-div") >= 5
    assert '<h2 id="equity">' in html and '<h2 id="trades">' in html
    assert "EURUSD" in html

    # Inlining Plotly drops the CDN link and clearly increases the file.
    inline = tmp_path / "public" / "inline.html"
    assert dashboard.main(["--demo", "--inline-plotly", "--output", str(inline)]) == 0
    inline_html = inline.read_text(encoding="utf-8")
    assert dashboard.PLOTLY_CDN not in inline_html
    assert inline.stat().st_size > output.stat().st_size


def test_real_report_takes_kpis_from_the_table_and_the_summary(tmp_path: Path) -> None:
    """The KPIs come from the columns, Sharpe and the window from summary.txt, the curves sum up."""
    folder = _write_report(tmp_path / "backtest_EURUSD_M15_2025-06-02_2025-06-05", _trades())
    report = dashboard.load_report(folder)
    assert (report.symbol, report.timeframe) == ("EURUSD", "M15")
    assert report.window == "2025-06-02 08:00 .. 2025-06-05 10:30 UTC"

    metrics = dashboard.compute_metrics(report)
    assert metrics["trades"] == 3
    assert metrics["profit"] == pytest.approx(80.0)
    assert metrics["win_rate"] == pytest.approx(66.6667, rel=1e-3)
    assert metrics["profit_factor"] == pytest.approx(5.0)
    assert metrics["sharpe"] == pytest.approx(0.032)
    # The drawdown comes from the summary (1.85 %), not from the profit peak: that would give 33 %.
    assert metrics["max_drawdown"] == pytest.approx(1.85)
    assert metrics["max_drawdown_abs"] == pytest.approx(20.0)

    equity = dashboard.build_equity(report.trades)
    assert equity["equity"].tolist() == [40.0, 20.0, 80.0]
    drawdown = dashboard.build_drawdown(equity)
    assert drawdown["drawdown"].tolist() == [0.0, -20.0, 0.0]
    assert dashboard.build_monthly_pnl(report.trades)["profit"].tolist() == [80.0]

    output = tmp_path / "real.html"
    assert dashboard.main(["--report-dir", str(folder), "--output", str(output)]) == 0
    html = output.read_text(encoding="utf-8")
    for expected in ("+80.00", "66.67 %", "1.85 %", "0.032"):
        assert expected in html


def test_missing_summary_falls_back_to_the_folder_name(tmp_path: Path) -> None:
    """Without summary.txt the build goes on, and the symbol and timeframe come from the folder name."""
    folder = _write_report(
        tmp_path / "backtest_EURUSD_M5_2025-06-02_2025-06-05", _trades(), summary=None
    )
    report = dashboard.load_report(folder)
    assert report.summary == {}
    assert (report.symbol, report.timeframe) == ("EURUSD", "M5")
    assert report.window == "2025-06-02 .. 2025-06-05"

    metrics = dashboard.compute_metrics(report)
    assert metrics["sharpe"] is None
    assert metrics["max_drawdown"] is None  # no capital, so the percentage cannot be computed

    output = tmp_path / "no_summary.html"
    assert dashboard.main(["--report-dir", str(folder), "--output", str(output)]) == 0
    html = output.read_text(encoding="utf-8")
    assert "Sharpe" in html
    assert metric_value(html, "Max drawdown") == "20.00 USD"  # the absolute value, not a percentage


def test_empty_trades_gives_a_notice_instead_of_a_crash(tmp_path: Path) -> None:
    """A file with a header only means zero trades: the page is built with an honest notice."""
    empty = pd.DataFrame(columns=list(TRADE_COLUMNS))
    folder = _write_report(tmp_path / "backtest_EURUSD_M15_2025-06-02_2025-06-05", empty)

    report = dashboard.load_report(folder)
    assert report.is_empty
    assert dashboard.compute_metrics(report)["trades"] == 0

    output = tmp_path / "empty.html"
    assert dashboard.main(["--report-dir", str(folder), "--output", str(output)]) == 0
    html = output.read_text(encoding="utf-8")
    assert "Сделок в отчёте нет" in html
    assert metric_value(html, "Сделок") == "0"


def test_compare_mode_adds_the_comparison_section(tmp_path: Path) -> None:
    """The compare mode shows the KPI table, the shared curves and a radar of two reports."""
    left = _write_report(tmp_path / "backtest_EURUSD_M15_2025-06-02_2025-06-05", _trades())
    losing = _trades()
    losing["profit"] = [-5.0, -10.0, -15.0]
    right = _write_report(tmp_path / "backtest_EURUSD_M15_2026-06-02_2026-06-05", losing)

    output = tmp_path / "compare.html"
    args = ["--report-dir", str(left), "--compare", str(left), str(right), "--output", str(output)]
    assert dashboard.main(args) == 0

    html = output.read_text(encoding="utf-8")
    assert '<h2 id="compare-kpi">' in html
    assert '<h2 id="compare-equity">' in html
    assert '<h2 id="compare-radar">' in html
    assert "scatterpolar" in html
    assert left.name in html and right.name in html


def test_broken_inputs_fail_with_a_clear_message(tmp_path: Path) -> None:
    """No report - a clear error about trades.csv; an incomplete compare is rejected by argparse."""
    output = tmp_path / "broken.html"

    with pytest.raises(SystemExit) as missing:
        dashboard.main(["--report-dir", str(tmp_path / "nowhere"), "--output", str(output)])
    assert "trades.csv" in str(missing.value)

    folder = _write_report(tmp_path / "backtest_EURUSD_M15_2025-06-02_2025-06-05", _trades())
    with pytest.raises(SystemExit) as too_few:
        dashboard.main(["--report-dir", str(folder), "--compare", str(folder)])
    assert too_few.value.code != 0

    with pytest.raises(SystemExit):
        dashboard.main(["--output", str(output)])  # neither --demo, nor --report-dir, nor --compare
