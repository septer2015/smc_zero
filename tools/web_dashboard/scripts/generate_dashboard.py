"""Standalone Plotly web dashboard for ``smc_zero`` backtest reports (Э14).

The tool is a showcase, not a part of the trading pipeline.  It reads *finished* reports
(``trades.csv`` and an optional ``summary.txt``) already written by the backtester and builds
one self-contained HTML file with interactive charts.  The logic of the strategy, the engine and
the optimizer stays untouched: the dashboard recomputes nothing but the derived curves (equity,
drawdown, monthly profit) taken straight from the ``trades.csv`` columns.

Two data sources:

* ``--demo`` — plausible synthetic trades, so the dashboard can be seen without a report;
* ``--report-dir`` — a real report folder holding ``trades.csv`` and (preferably)
  ``summary.txt``.  A missing ``summary.txt`` does not break the run: the symbol, the timeframe
  and the window come from the folder name, and the Sharpe ratio is simply not shown.

One command is enough (no server, the internet is needed only to load Plotly from the CDN):

.. code-block:: text

    python tools/web_dashboard/scripts/generate_dashboard.py --demo
    python tools/web_dashboard/scripts/generate_dashboard.py --report-dir ./reports/<report>
    python tools/web_dashboard/scripts/generate_dashboard.py --report-dir ./reports/<report> \\
        --output ./tools/web_dashboard/public/dashboard.html

Plotly is an optional core dependency (extra ``dashboard``), so the import is guarded by a clear
message instead of a raw traceback.
"""

from __future__ import annotations

import argparse
import math
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.io as pio
except ImportError as error:  # pragma: no cover - runs only without the optional dependency
    raise SystemExit(
        'Plotly не установлена. Поставьте optional-зависимость: pip install -e ".[dashboard]"'
    ) from error


# --- Style and path constants -----------------------------------------------------------
#: Where the finished HTML lands by default (next to the tool itself).
DEFAULT_OUTPUT = Path("tools/web_dashboard/public/dashboard.html")
#: Plotly version the page pulls.  From the CDN, so megabytes of JS are not baked into the HTML.
PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
#: How many recent trades the table shows.
TABLE_ROWS = 50
#: The maximum number of candles the candle chart draws: more makes the HTML unmanageable.
MAX_CANDLES = 1500
#: How many recent trades get marked on the candles (entry/sl/tp).
MAX_MARKED_TRADES = 10
#: The seed of the demo generator: the demo dashboard reproduces from run to run.
DEMO_SEED = 20261011

#: The dark theme palette.
BG = "#12161f"
PANEL = "#1a2030"
GRID = "#232a38"
TEXT = "#d7dde8"
MUTED = "#8b96a8"
GREEN = "#41d19a"
RED = "#ef5350"
BLUE = "#4f9cf9"
AMBER = "#f5a623"
PURPLE = "#ab7df6"
SERIES = (BLUE, AMBER, GREEN, RED, PURPLE)

#: Minutes per bar of every timeframe - needed to turn ``bars_held`` into minutes.
TIMEFRAME_MINUTES: dict[str, int] = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
    "W1": 10080,
}

#: Report folder name: ``backtest_EURUSD_M5_2025-05-06_2026-09-22`` and its optimizer twin.
FOLDER_RE = re.compile(
    r"^(?:backtest|optimization)_(?P<symbol>[A-Za-z0-9]+)_(?P<timeframe>[A-Za-z0-9]+)"
    r"_(?P<start>\d{4}-\d{2}-\d{2})_(?P<end>\d{4}-\d{2}-\d{2})$"
)

#: The shared dark theme layout: every chart receives it.
DARK_LAYOUT: dict[str, Any] = {
    "paper_bgcolor": BG,
    "plot_bgcolor": BG,
    "font": {"color": TEXT, "family": "Segoe UI, Roboto, Arial, sans-serif", "size": 13},
    "colorway": list(SERIES),
    "margin": {"l": 62, "r": 28, "t": 54, "b": 48},
    "legend": {"bgcolor": "rgba(0,0,0,0)", "font": {"color": TEXT}},
    "xaxis": {"gridcolor": GRID, "zerolinecolor": GRID, "linecolor": GRID},
    "yaxis": {"gridcolor": GRID, "zerolinecolor": GRID, "linecolor": GRID},
    "hoverlabel": {"bgcolor": PANEL, "font": {"color": TEXT}},
}


@dataclass
class Report:
    """One loaded report: the trades, the header labels and the parsed summary metrics."""

    source: str
    symbol: str
    timeframe: str
    window: str
    trades: pd.DataFrame
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """True when the report holds no trades at all."""
        return self.trades.empty


@dataclass
class Block:
    """Page section: either a Plotly figure or a ready HTML fragment (table, note)."""

    anchor: str
    heading: str
    figure: go.Figure | None = None
    html: str | None = None


def _style(fig: go.Figure, height: int = 380) -> go.Figure:
    """Apply the dark theme and the height to the figure and return it for chaining."""
    fig.update_layout(height=height, autosize=True, **DARK_LAYOUT)
    return fig


def _to_float(text: str) -> float | None:
    """Parse a number out of a string: ``+208.00`` and ``0.45`` parse, ``—`` yields ``None``."""
    cleaned = text.strip().rstrip("%").strip().replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


#: Header of summary.txt: ``SMC backtest: EURUSD M5 (2025-05-20 17:10 .. 2026-09-22 23:55 UTC)``.
SUMMARY_HEADER_RE = re.compile(
    r"SMC backtest:\s*(?P<symbol>\S+)\s+(?P<timeframe>\S+)\s+\((?P<window>[^)]*)\)"
)
#: Metric line of summary.txt: a name, two or more spaces, a number (with an optional percent sign).
SUMMARY_LINE_RE = re.compile(
    r"^\s{2,}(?P<name>[A-Za-z][A-Za-z0-9 ()/.'-]*?)\s{2,}(?P<value>[-+]?\d+(?:\.\d+)?)"
)


# --- Command line parsing ----------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Return the parsed command line arguments of the dashboard generator."""
    parser = argparse.ArgumentParser(
        prog="generate_dashboard",
        description="Собрать standalone HTML-дашборд по отчёту бэктеста smc_zero.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--demo",
        action="store_true",
        help="демо-режим: синтетические сделки и свечи, реальный отчёт не нужен",
    )
    source.add_argument(
        "--report-dir",
        default=None,
        help="папка отчёта бэктеста с trades.csv (summary.txt желателен, но не обязателен)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"путь итогового HTML (по умолчанию {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--title", default=None, help="заголовок страницы и вкладки")
    parser.add_argument(
        "--tape",
        default=None,
        help="CSV с OHLC-лентой для свечного графика; без него свечи только в демо-режиме",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("DIR_A", "DIR_B"),
        default=None,
        help="режим сравнения двух отчётов: KPI, radar и общие кривые equity",
    )
    parser.add_argument(
        "--inline-plotly",
        action="store_true",
        help="вшить Plotly в HTML вместо загрузки с CDN (файл тяжелее, зато офлайн)",
    )
    return parser.parse_args(argv)


# --- Report loading --------------------------------------------------------------------------
def resolve_report_folder(report_dir: Path) -> Path:
    """Return the folder that really holds ``trades.csv``: the path itself or its only subfolder."""
    if (report_dir / "trades.csv").is_file():
        return report_dir
    if report_dir.is_dir():
        candidates = sorted(
            child for child in report_dir.iterdir() if (child / "trades.csv").is_file()
        )
        if len(candidates) == 1:
            return candidates[0]
    raise ValueError(f"В отчёте нет trades.csv: {report_dir / 'trades.csv'}")


def load_trades(path: Path) -> pd.DataFrame:
    """Read ``trades.csv``.  An empty file yields an empty frame, not an error (Э14).

    The mandatory minimum for the real mode is a time column (``fill_time`` or ``open_time``)
    and ``profit``.  The other columns are optional: their absence only drops a few charts.
    """
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    missing = [name for name in ("profit",) if name not in frame.columns]
    if "fill_time" not in frame.columns and "open_time" not in frame.columns:
        missing.append("fill_time/open_time")
    if missing:
        raise ValueError(f"trades.csv без обязательных колонок {', '.join(missing)}: {path}")
    for column in ("fill_time", "open_time", "exit_time"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
    for column in ("profit", "result", "rr", "sl_pips", "tp_pips", "bars_held", "limit_dur_min"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_summary(path: Path) -> dict[str, Any]:
    """Read ``summary.txt`` when it exists.  A missing file yields an empty dict, not an error."""
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    info: dict[str, Any] = {"metrics": {}}
    header = SUMMARY_HEADER_RE.search(text)
    if header:
        info["symbol"] = header.group("symbol")
        info["timeframe"] = header.group("timeframe")
        info["window"] = header.group("window")
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        match = SUMMARY_LINE_RE.match(line)
        if match:
            value = _to_float(match.group("value"))
            if value is not None:
                metrics[match.group("name").strip()] = value
    info["metrics"] = metrics
    return info


def load_tape(path: Path) -> pd.DataFrame:
    """Read an OHLC tape: a time column plus ``open``/``high``/``low``/``close``."""
    frame = pd.read_csv(path)
    time_column = next(
        (name for name in ("datetime", "timestamp", "time", "date") if name in frame.columns),
        None,
    )
    if time_column is None:
        raise ValueError(f"Лента не содержит колонки времени: {path}")
    frame = frame.rename(columns={time_column: "time"})
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["time"])
    for column in ("open", "high", "low", "close"):
        if column not in frame.columns:
            raise ValueError(f"Лента не содержит колонки {column}: {path}")
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("time").reset_index(drop=True)


# --- Demo data ----------------------------------------------------------------------------
def build_demo_data(seed: int = DEMO_SEED) -> Report:
    """Synthetic report: plausible M15 trades with a positive expectancy.

    The numbers are picked so the dashboard is interesting to look at without real data: a bit
    more than half of the trades win, and RR, durations and levels vary.  The generator is seeded,
    so the demo picture reproduces from run to run.
    """
    rng = random.Random(seed)
    start = pd.Timestamp("2025-06-02 08:00", tz="UTC")
    step = pd.Timedelta(minutes=15)
    pip = 0.0001
    level_names = ("LondonL", "LondonH", "AsianL", "AsianH", "NYH", "NYL")
    setups = ("fresh", "bos", "choch")
    rows: list[dict[str, Any]] = []
    bar = 120
    price = 1.1000
    for _ in range(260):
        bar += rng.randint(20, 60)
        fill_bar = bar + rng.randint(1, 4)
        held_bars = max(1, int(rng.uniform(4, 320)))
        fill_time = start + step * fill_bar
        side = rng.choice(("long", "short"))
        price = round(price + rng.gauss(0.0, 0.0035), 5)
        entry = round(price + rng.uniform(-0.0006, 0.0006), 5)
        sl_pips = round(rng.uniform(20.0, 45.0), 1)
        rr = round(rng.uniform(1.2, 3.0), 2)
        tp_pips = round(sl_pips * rr, 1)
        if side == "long":
            sl = round(entry - sl_pips * pip, 5)
            tp = round(entry + tp_pips * pip, 5)
        else:
            sl = round(entry + sl_pips * pip, 5)
            tp = round(entry - tp_pips * pip, 5)
        spread = 1.4
        roll = rng.random()
        if roll < 0.54:
            pips, exit_price, result = tp_pips - spread, tp, 1
        elif roll < 0.62:
            partial = rng.uniform(0.1, 0.8) * tp_pips
            sign = 1 if side == "long" else -1
            pips = partial - spread
            exit_price = round(entry + sign * partial * pip, 5)
            result = 1
        else:
            pips, exit_price, result = -sl_pips - spread, sl, -1
        rows.append(
            {
                "signal_bar": bar,
                "open_time": start + step * bar,
                "fill_bar": fill_bar,
                "fill_time": fill_time,
                "exit_bar": fill_bar + held_bars,
                "exit_time": fill_time + step * held_bars,
                "bars_held": held_bars,
                "days_held": round(held_bars * 15 / 1440, 2),
                "limit_dur_min": 15 * rng.randint(1, 4),
                "level_name": rng.choice(level_names),
                "level_date": (fill_time - pd.Timedelta(days=rng.randint(0, 3))).date().isoformat(),
                "level_price": round(entry + rng.uniform(-0.002, 0.002), 5),
                "setup_type": rng.choice(setups),
                "side": side,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "rr": rr,
                "sl_pips": sl_pips,
                "tp_pips": tp_pips,
                "sl_source": "sweep_extreme",
                "tp_source": "rr",
                "sweep_bar": bar - rng.randint(4, 12),
                "choch_bar": bar - rng.randint(1, 4),
                "fvg_bar": bar,
                "exit_price": exit_price,
                "result": result,
                "profit": round(pips * 1.0, 2),  # 0.1 EURUSD lot: one pip is roughly 1 USD
                "spread_cost": spread,
                "slippage_cost": round(rng.choice((0.0, 0.0, 0.2)), 2),
                "swap_cost": round(rng.choice((0.0, 0.0, -0.7)), 2),
                "commission": 0.0,
                "risk_pct": round(sl_pips / 10.0, 2),
                "risk_warning": True,
            }
        )
    return Report(
        source="demo",
        symbol="EURUSD",
        timeframe="M15",
        window=f"{start:%Y-%m-%d} .. {rows[-1]['exit_time']:%Y-%m-%d}",
        trades=pd.DataFrame(rows),
        summary={"metrics": {"sharpe (per bar)": 0.083}},
    )


def build_demo_tape(
    report: Report, periods: int = MAX_CANDLES, seed: int = DEMO_SEED
) -> pd.DataFrame:
    """Synthetic M15 tape ending at the last trade of the demo report.

    The tape exists only for the candle chart: the demo trades are marked on it, while
    ``trades.csv`` carries no candles and inventing them from the trades is not allowed.
    """
    rng = random.Random(seed + 1)
    step = pd.Timedelta(minutes=15)
    last = pd.to_datetime(report.trades["fill_time"].max(), utc=True)
    start = last - step * (periods - 1)
    rows: list[dict[str, Any]] = []
    price = 1.1000
    for index in range(periods):
        open_price = price
        close_price = round(open_price + rng.gauss(0.0, 0.0006), 5)
        wiggle = abs(rng.gauss(0.0, 0.0005))
        rows.append(
            {
                "time": start + step * index,
                "open": open_price,
                "high": round(max(open_price, close_price) + wiggle, 5),
                "low": round(min(open_price, close_price) - wiggle, 5),
                "close": close_price,
                "volume": rng.randint(100, 900),
            }
        )
        price = close_price
    return pd.DataFrame(rows)


# --- Derived curves and metrics -----------------------------------------------------------
def trades_in_order(trades: pd.DataFrame) -> pd.DataFrame:
    """Return trades in time order: ``fill_time``, else ``open_time``, else ``signal_bar``."""
    if trades.empty:
        return trades.reset_index(drop=True)
    for name in ("fill_time", "open_time", "exit_time"):
        if name in trades.columns:
            ordered = trades.copy()
            ordered[name] = pd.to_datetime(ordered[name], errors="coerce", utc=True)
            return ordered.sort_values(name).reset_index(drop=True)
    if "signal_bar" in trades.columns:
        return trades.sort_values("signal_bar").reset_index(drop=True)
    return trades.reset_index(drop=True)


def _timeline(trades: pd.DataFrame) -> pd.Series:
    """Time axis of the trades: the fill/open/exit date, else the bar number, else the plain index."""
    for name in ("fill_time", "open_time", "exit_time"):
        if name in trades.columns:
            return pd.to_datetime(trades[name], errors="coerce", utc=True)
    if "signal_bar" in trades.columns:
        return pd.to_numeric(trades["signal_bar"], errors="coerce")
    return pd.Series(range(len(trades)), index=trades.index)


def build_equity(trades: pd.DataFrame) -> pd.DataFrame:
    """Cumulative profit over the trades sorted in time order."""
    ordered = trades_in_order(trades)
    profit = pd.to_numeric(ordered.get("profit"), errors="coerce").fillna(0.0)
    return pd.DataFrame({"time": _timeline(ordered), "equity": profit.cumsum()})


def build_drawdown(equity: pd.DataFrame) -> pd.DataFrame:
    """Drawdown as the deviation of the cumulative profit from its peak."""
    peak = equity["equity"].cummax()
    return pd.DataFrame({"time": equity["time"], "drawdown": equity["equity"] - peak})


def build_monthly_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    """Profit by calendar month; without a readable time column the frame stays empty."""
    ordered = trades_in_order(trades)
    time_column = next(
        (name for name in ("fill_time", "open_time", "exit_time") if name in ordered.columns),
        None,
    )
    if time_column is None:
        return pd.DataFrame(columns=["month", "profit"])
    times = pd.to_datetime(ordered[time_column], errors="coerce", utc=True)
    valid = times.notna()
    if not valid.any():
        return pd.DataFrame(columns=["month", "profit"])
    profit = pd.to_numeric(ordered["profit"], errors="coerce").fillna(0.0)
    frame = pd.DataFrame(
        {"month": times[valid].dt.strftime("%Y-%m"), "profit": profit[valid].to_numpy()}
    )
    return frame.groupby("month", as_index=False, sort=True)["profit"].sum()


def compute_metrics(report: Report) -> dict[str, Any]:
    """Report KPIs: the numbers from ``trades.csv`` plus Sharpe from ``summary.txt`` when present."""
    trades = report.trades
    metrics: dict[str, Any] = {
        "trades": len(trades),
        "profit": None,
        "profit_factor": None,
        "win_rate": None,
        "max_drawdown": None,
        "max_drawdown_abs": None,
        "sharpe": None,
    }
    if not trades.empty and "profit" in trades.columns:
        profit = pd.to_numeric(trades["profit"], errors="coerce").fillna(0.0)
        metrics["profit"] = float(profit.sum())
        metrics["win_rate"] = (
            float((profit > 0).sum() / len(profit) * 100.0) if len(profit) else 0.0
        )
        gross_profit = float(profit[profit > 0].sum())
        gross_loss = float(-profit[profit < 0].sum())
        if gross_loss > 0:
            metrics["profit_factor"] = gross_profit / gross_loss
        elif gross_profit > 0:
            metrics["profit_factor"] = float("inf")
        else:
            metrics["profit_factor"] = 0.0
        equity = build_equity(trades)
        if not equity.empty:
            peak = equity["equity"].cummax()
            metrics["max_drawdown_abs"] = float((peak - equity["equity"]).max())
    summary_metrics = report.summary.get("metrics", {})
    # The percentage drawdown comes from summary.txt and is not computed against the profit peak:
    # the peak of a profit curve can sit near zero, so the ratio jumps into hundreds of percent.
    capital = None
    for key, value in summary_metrics.items():
        lowered = key.lower()
        if lowered.startswith("sharpe"):
            metrics["sharpe"] = value
        elif lowered == "max drawdown":
            metrics["max_drawdown"] = value
        elif lowered == "initial capital":
            capital = value
    if metrics["max_drawdown"] is None and metrics["max_drawdown_abs"] is not None and capital:
        metrics["max_drawdown"] = metrics["max_drawdown_abs"] / capital * 100.0
    return metrics


def _fmt_money(value: float | None) -> str:
    """Money with a sign: ``+208.00`` / ``-45.95`` / ``—``."""
    if value is None:
        return "—"
    return f"{value:+.2f}"


def _fmt_num(value: float | None, digits: int = 2) -> str:
    """A number with a fixed precision; infinity is shown as ``∞``."""
    if value is None:
        return "—"
    if value == float("inf"):
        return "∞"
    return f"{value:.{digits}f}"


def _fmt_pct(value: float | None) -> str:
    """A percentage: ``66.67 %`` / ``—``."""
    if value is None:
        return "—"
    return f"{value:.2f} %"


def _fmt_drawdown(metrics: dict[str, Any]) -> str:
    """Drawdown: the percentage when it is known, otherwise the absolute peak in USD."""
    if metrics["max_drawdown"] is not None:
        return _fmt_pct(metrics["max_drawdown"])
    if metrics["max_drawdown_abs"] is not None:
        return f"{metrics['max_drawdown_abs']:.2f} USD"
    return "—"


def build_kpi_cards(report: Report, metrics: dict[str, Any]) -> list[tuple[str, str]]:
    """List of the cards of the top KPI panel: a label and a value."""
    return [
        ("Символ", report.symbol),
        ("Таймфрейм", report.timeframe),
        ("Окно данных", report.window),
        ("Сделок", str(metrics["trades"])),
        ("Прибыль", _fmt_money(metrics["profit"])),
        ("Profit factor", _fmt_num(metrics["profit_factor"])),
        ("Win rate", _fmt_pct(metrics["win_rate"])),
        ("Max drawdown", _fmt_drawdown(metrics)),
        ("Sharpe (per bar)", _fmt_num(metrics["sharpe"], 3)),
        ("Источник", report.source),
    ]


# --- Charts --------------------------------------------------------------------------------
def fig_equity(equity: pd.DataFrame, name: str = "Кумулятивная прибыль") -> go.Figure:
    """Line of the accumulated profit with a fill down to zero."""
    fig = go.Figure(
        go.Scatter(
            x=equity["time"],
            y=equity["equity"],
            mode="lines",
            name=name,
            line={"color": BLUE, "width": 2},
            fill="tozeroy",
            fillcolor="rgba(79,156,249,0.15)",
        )
    )
    fig.update_layout(title="Equity curve", xaxis_title="Время", yaxis_title="USD")
    return _style(fig)


def fig_drawdown(drawdown: pd.DataFrame) -> go.Figure:
    """Area of the drawdown below zero."""
    fig = go.Figure(
        go.Scatter(
            x=drawdown["time"],
            y=drawdown["drawdown"],
            mode="lines",
            name="Просадка",
            line={"color": RED, "width": 2},
            fill="tozeroy",
            fillcolor="rgba(239,83,80,0.18)",
        )
    )
    fig.update_layout(title="Drawdown", xaxis_title="Время", yaxis_title="USD")
    return _style(fig)


def fig_monthly(monthly: pd.DataFrame) -> go.Figure:
    """Monthly profit bars: green for a gain, red for a loss."""
    colors = [GREEN if value >= 0 else RED for value in monthly["profit"]]
    fig = go.Figure(
        go.Bar(x=monthly["month"], y=monthly["profit"], marker_color=colors, name="P&L")
    )
    fig.update_layout(title="Monthly P&L", xaxis_title="Месяц", yaxis_title="USD")
    return _style(fig)


def fig_winloss(trades: pd.DataFrame) -> go.Figure:
    """Donut by the trade sign from the ``result`` column (1 - win, -1 - loss)."""
    result = pd.to_numeric(trades["result"], errors="coerce").dropna()
    wins = int((result > 0).sum())
    losses = int((result < 0).sum())
    fig = go.Figure(
        go.Pie(
            labels=["Плюс", "Минус"],
            values=[wins, losses],
            hole=0.55,
            marker={"colors": [GREEN, RED]},
            textinfo="label+percent",
            textfont={"color": TEXT},
        )
    )
    fig.update_layout(title="Win / Loss")
    return _style(fig)


def build_duration_series(trades: pd.DataFrame, timeframe: str) -> tuple[pd.Series | None, str]:
    """Trade durations in minutes: ``limit_dur_min``, else ``bars_held`` times the bar length."""
    if "limit_dur_min" in trades.columns:
        values = pd.to_numeric(trades["limit_dur_min"], errors="coerce")
        if values.notna().any():
            return values.dropna(), "минут (возраст лимита)"
    minutes = TIMEFRAME_MINUTES.get(str(timeframe).upper())
    if "bars_held" in trades.columns and minutes:
        values = pd.to_numeric(trades["bars_held"], errors="coerce") * minutes
        if values.notna().any():
            return values.dropna(), f"минут (bars_held × {minutes})"
    return None, ""


def fig_duration(duration: pd.Series, unit: str) -> go.Figure:
    """Histogram of the trade durations in minutes."""
    fig = go.Figure(go.Histogram(x=duration, marker_color=BLUE, nbinsx=40, name="Сделки"))
    fig.update_layout(title="Trade duration", xaxis_title=unit or "минут", yaxis_title="Сделок")
    return _style(fig)


def fig_rr(trades: pd.DataFrame) -> go.Figure:
    """Scatter of the trade RR; the colour is profit (or the result when profit is absent)."""
    rr = pd.to_numeric(trades["rr"], errors="coerce")
    valid = rr.notna()
    marker: dict[str, Any] = {"size": 8}
    if "profit" in trades.columns:
        color = pd.to_numeric(trades["profit"], errors="coerce")[valid]
        marker.update({"color": color, "colorscale": "RdYlGn", "colorbar": {"title": "USD"}})
    elif "result" in trades.columns:
        color = pd.to_numeric(trades["result"], errors="coerce")[valid]
        marker.update({"color": color, "colorscale": "RdYlGn"})
    else:
        marker.update({"color": BLUE})
    scatter = go.Scatter(
        x=_timeline(trades)[valid], y=rr[valid], mode="markers", marker=marker, name="Сделка"
    )
    fig = go.Figure(scatter)
    fig.update_layout(title="Risk / Reward", xaxis_title="Время", yaxis_title="RR")
    return _style(fig)


def fig_candles(tape: pd.DataFrame, trades: pd.DataFrame) -> go.Figure:
    """Candle chart of the tape with the last ``MAX_MARKED_TRADES`` trades marked."""
    window = tape.tail(MAX_CANDLES).reset_index(drop=True)
    fig = go.Figure(
        go.Candlestick(
            x=window["time"],
            open=window["open"],
            high=window["high"],
            low=window["low"],
            close=window["close"],
            increasing_line_color=GREEN,
            decreasing_line_color=RED,
            name="Свечи",
        )
    )
    if "entry" in trades.columns:
        fig = _mark_entries(fig, window, trades)
    fig.update_layout(title="Свечи и разметка сделок", xaxis_title="Время", yaxis_title="Цена")
    fig.update_xaxes(rangeslider_visible=False)
    return _style(fig, height=520)


def _mark_entries(fig: go.Figure, window: pd.DataFrame, trades: pd.DataFrame) -> go.Figure:
    """Mark the entry, SL and TP of the last trades that fall inside the tape window."""
    start, end = window["time"].min(), window["time"].max()
    ordered = trades_in_order(trades)
    times = pd.to_datetime(_timeline(ordered), errors="coerce", utc=True)
    inside = (times >= start) & (times <= end)
    for index, trade in ordered[inside].tail(MAX_MARKED_TRADES).iterrows():
        side = str(trade.get("side", "")).lower()
        color = GREEN if side == "long" else RED if side == "short" else BLUE
        symbol = "triangle-up" if side == "long" else "triangle-down"
        stamp = times.loc[index]
        fig.add_trace(
            go.Scatter(
                x=[stamp],
                y=[trade["entry"]],
                mode="markers",
                marker={"color": color, "size": 11, "symbol": symbol},
                name=f"Вход {side}",
                hovertemplate="Вход %{y}<extra></extra>",
                showlegend=False,
            )
        )
        last = stamp
        if "exit_time" in ordered.columns:
            exit_value = pd.to_datetime(trade.get("exit_time"), errors="coerce", utc=True)
            if pd.notna(exit_value):
                last = exit_value
        for key, dash in (("sl", "dot"), ("tp", "dash")):
            if key in ordered.columns and pd.notna(trade.get(key)):
                fig.add_trace(
                    go.Scatter(
                        x=[stamp, last],
                        y=[trade[key], trade[key]],
                        mode="lines",
                        line={"color": color, "width": 1, "dash": dash},
                        name=key.upper(),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
    return fig


# --- Page block assembly -----------------------------------------------------------------
def build_figures(
    report: Report, tape: pd.DataFrame | None = None
) -> tuple[list[Block], list[str]]:
    """Assemble the chart blocks of the main report and the notes about the skipped plots.

    Returns the pair "blocks, notes".  The mandatory charts (equity, drawdown) are always built
    when the report holds trades; the rest appear when their columns are present, and the reason
    for the absence goes into the notes shown on the page.
    """
    blocks: list[Block] = []
    notes: list[str] = []
    trades = report.trades
    if trades.empty:
        return blocks, notes
    equity = build_equity(trades)
    blocks.append(Block("equity", "Equity curve", fig_equity(equity)))
    blocks.append(Block("drawdown", "Drawdown", fig_drawdown(build_drawdown(equity))))
    monthly = build_monthly_pnl(trades)
    if monthly.empty:
        notes.append("Monthly P&L недоступен: во сделках нет читаемого времени.")
    else:
        blocks.append(Block("monthly", "Monthly P&L", fig_monthly(monthly)))
    if (
        "result" in trades.columns
        and pd.to_numeric(trades["result"], errors="coerce").notna().any()
    ):
        blocks.append(Block("winloss", "Win / Loss", fig_winloss(trades)))
    else:
        notes.append("Распределение Win / Loss недоступно: в trades.csv нет колонки result.")
    duration, unit = build_duration_series(trades, report.timeframe)
    if duration is not None:
        blocks.append(Block("duration", "Trade duration", fig_duration(duration, unit)))
    else:
        notes.append(
            "Trade duration недоступен: нет ни limit_dur_min, ни bars_held с известным таймфреймом."
        )
    if "rr" in trades.columns and pd.to_numeric(trades["rr"], errors="coerce").notna().any():
        blocks.append(Block("rr", "Risk / Reward", fig_rr(trades)))
    else:
        notes.append("Risk / Reward недоступен: в trades.csv нет колонки rr.")
    if tape is not None and not tape.empty:
        blocks.append(Block("candles", "Свечи и разметка сделок", fig_candles(tape, trades)))
    else:
        notes.append(
            "Свечной график пропущен: нет OHLC-ленты. Возьмите --demo или передайте --tape PATH."
        )
    return blocks, notes


# --- Tables --------------------------------------------------------------------------------
TABLE_COLUMNS = (
    ("date", "Дата"),
    ("side", "Сторона"),
    ("setup_type", "Setup"),
    ("level_name", "Уровень"),
    ("entry", "Вход"),
    ("sl", "SL"),
    ("tp", "TP"),
    ("rr", "RR"),
    ("profit", "Прибыль"),
    ("result", "Результат"),
)

COMPARE_ROWS = (
    ("trades", "Сделок"),
    ("profit", "Прибыль"),
    ("profit_factor", "Profit factor"),
    ("win_rate", "Win rate"),
    ("max_drawdown", "Max drawdown"),
    ("sharpe", "Sharpe"),
)


def _cell(trade: pd.Series, key: str) -> str:
    """Render one cell of the trades table, carefully handling the missing data."""
    if key == "date":
        for name in ("fill_time", "open_time", "exit_time"):
            if name in trade.index and pd.notna(trade[name]):
                return pd.to_datetime(trade[name]).strftime("%Y-%m-%d %H:%M")
        return "—"
    if key == "result":
        value = trade.get("result")
        if pd.isna(value):
            return "—"
        return "плюс" if value > 0 else "минус"
    if key in ("entry", "sl", "tp"):
        value = trade.get(key)
        return "—" if pd.isna(value) else f"{float(value):.5f}"
    if key == "rr":
        value = trade.get("rr")
        return "—" if pd.isna(value) else f"{float(value):.2f}"
    if key == "profit":
        value = trade.get("profit")
        return "—" if pd.isna(value) else f"{float(value):+.2f}"
    value = trade.get(key)
    return "—" if pd.isna(value) else str(value)


def trades_table_html(trades: pd.DataFrame, limit: int = TABLE_ROWS) -> str:
    """Compact table of the last ``limit`` trades, the freshest on top."""
    ordered = trades_in_order(trades).tail(limit)
    header = "".join(f"<th>{escape(title)}</th>" for _, title in TABLE_COLUMNS)
    rows = []
    for _, trade in ordered.iloc[::-1].iterrows():
        cells = "".join(f"<td>{escape(_cell(trade, key))}</td>" for key, _ in TABLE_COLUMNS)
        rows.append(f"<tr>{cells}</tr>")
    body = "".join(rows)
    return f'<table class="grid"><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>'


# --- Comparison of two reports -----------------------------------------------------------------
def _compare_value(key: str, value: float | None) -> str:
    """Format a metric for the comparison table."""
    if value is None:
        return "—"
    if key == "trades":
        return str(int(value))
    if key == "profit":
        return f"{value:+.2f}"
    if key in ("win_rate", "max_drawdown"):
        return f"{value:.2f} %"
    if key == "profit_factor":
        return "∞" if value == float("inf") else f"{value:.2f}"
    return f"{value:.3f}"


def _radar_scores(value_a: Any, value_b: Any, higher_is_better: bool) -> tuple[float, float]:
    """Normalise a pair of metrics into [0,1]; for ``max_drawdown`` lower is better.

    Infinity counts as the best value for the direct direction and as the worst for the inverse
    one.  A missing metric scores zero: on the radar that means "no data", not an advantage.
    """

    def as_number(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    first, second = as_number(value_a), as_number(value_b)
    finite = [value for value in (first, second) if value is not None and math.isfinite(value)]
    low, high = (min(finite), max(finite)) if finite else (None, None)
    scores: list[float] = []
    for value in (first, second):
        if value is None:
            scores.append(0.0)
        elif math.isinf(value):
            scores.append(1.0 if higher_is_better else 0.0)
        elif low is None or high is None or high == low:
            scores.append(0.5)  # nothing to tell apart - both sides get the middle
        else:
            score = (value - low) / (high - low)
            scores.append(score if higher_is_better else 1.0 - score)
    return scores[0], scores[1]


def compare_table_html(
    metrics_a: dict[str, Any], metrics_b: dict[str, Any], label_a: str, label_b: str
) -> str:
    """Three columns: the metric, the value of A, the value of B."""
    head = f"<tr><th>Метрика</th><th>{escape(label_a)}</th><th>{escape(label_b)}</th></tr>"
    rows = "".join(
        f"<tr><td>{escape(title)}</td>"
        f"<td>{escape(_compare_value(key, metrics_a.get(key)))}</td>"
        f"<td>{escape(_compare_value(key, metrics_b.get(key)))}</td></tr>"
        for key, title in COMPARE_ROWS
    )
    return f'<table class="grid"><thead>{head}</thead><tbody>{rows}</tbody></table>'


def fig_compare_equity(label_a: str, report_a: Report, label_b: str, report_b: Report) -> go.Figure:
    """Two equity curves on a single chart."""
    fig = go.Figure()
    for label, report, color in ((label_a, report_a, BLUE), (label_b, report_b, AMBER)):
        equity = build_equity(report.trades)
        fig.add_trace(
            go.Scatter(
                x=equity["time"],
                y=equity["equity"],
                mode="lines",
                name=label,
                line={"color": color, "width": 2},
            )
        )
    fig.update_layout(title="Equity A vs B", xaxis_title="Время", yaxis_title="USD")
    return _style(fig)


def fig_compare_radar(
    metrics_a: dict[str, Any], metrics_b: dict[str, Any], label_a: str, label_b: str
) -> go.Figure:
    """Radar over the normalised metrics: closer to the edge is better."""
    axes = (
        ("trades", "Сделок", True),
        ("profit", "Прибыль", True),
        ("profit_factor", "Profit factor", True),
        ("win_rate", "Win rate", True),
        ("max_drawdown", "Max drawdown", False),
        ("sharpe", "Sharpe", True),
    )
    labels = [title for _, title, _ in axes]
    scores_a: list[float] = []
    scores_b: list[float] = []
    for key, _, higher in axes:
        first, second = _radar_scores(metrics_a.get(key), metrics_b.get(key), higher)
        scores_a.append(first)
        scores_b.append(second)
    fig = go.Figure()
    for scores, label, color in ((scores_a, label_a, BLUE), (scores_b, label_b, AMBER)):
        fig.add_trace(
            go.Scatterpolar(
                r=scores + scores[:1],
                theta=labels + labels[:1],
                fill="toself",
                name=label,
                line={"color": color},
            )
        )
    fig.update_layout(
        title="Сравнение по нормализованным метрикам",
        polar={
            "bgcolor": PANEL,
            "radialaxis": {"visible": True, "range": [0, 1], "gridcolor": GRID},
            "angularaxis": {"gridcolor": GRID},
        },
    )
    return _style(fig, height=460)


def build_compare_blocks(
    label_a: str, report_a: Report, label_b: str, report_b: Report
) -> list[Block]:
    """Blocks of the comparison section: the KPI table, the shared equity curves and the radar."""
    metrics_a = compute_metrics(report_a)
    metrics_b = compute_metrics(report_b)
    return [
        Block(
            "compare-kpi",
            "Сравнение KPI",
            html=compare_table_html(metrics_a, metrics_b, label_a, label_b),
        ),
        Block(
            "compare-equity",
            "Equity A vs B",
            fig_compare_equity(label_a, report_a, label_b, report_b),
        ),
        Block(
            "compare-radar",
            "Radar A vs B",
            fig_compare_radar(metrics_a, metrics_b, label_a, label_b),
        ),
    ]


# --- Whole report loading ------------------------------------------------------------------
def _folder_meta(folder: Path) -> dict[str, str]:
    """Symbol, timeframe and window from the report folder name (a fallback label source)."""
    match = FOLDER_RE.match(folder.name)
    if not match:
        return {}
    return {
        "symbol": match.group("symbol"),
        "timeframe": match.group("timeframe"),
        "window": f"{match.group('start')} .. {match.group('end')}",
    }


def load_report(report_dir: Path) -> Report:
    """Read a report: the folder with trades.csv and the labels from summary.txt (or the name)."""
    folder = resolve_report_folder(report_dir)
    trades = load_trades(folder / "trades.csv")
    summary = load_summary(folder / "summary.txt")
    meta = _folder_meta(folder)
    return Report(
        source=str(folder),
        symbol=str(summary.get("symbol") or meta.get("symbol") or "—"),
        timeframe=str(summary.get("timeframe") or meta.get("timeframe") or "—"),
        window=str(summary.get("window") or meta.get("window") or "—"),
        trades=trades,
        summary=summary,
    )


#: Page styles: a readable dark theme, a responsive grid, scrolling for wide tables.
PAGE_CSS = """
* { box-sizing: border-box; }
body { margin: 0; background: #0d1117; color: #d7dde8;
  font-family: "Segoe UI", Roboto, Arial, sans-serif; }
header { padding: 22px 28px 6px; }
header h1 { margin: 0 0 4px; font-size: 22px; }
header .sub { margin: 0; color: #8b96a8; font-size: 13px; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px; padding: 12px 28px 4px; }
.card { background: #1a2030; border: 1px solid #232a38; border-radius: 10px;
  padding: 12px 14px; display: flex; flex-direction: column; gap: 6px; }
.card .k { color: #8b96a8; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
.card .v { font-size: 18px; font-weight: 600; word-break: break-word; }
.note { margin: 10px 28px 0; padding: 10px 14px; border-radius: 8px;
  background: #241f0f; border: 1px solid #6b5410; color: #f5d98b; font-size: 13px; }
.panel { margin: 16px 28px 0; background: #12161f; border: 1px solid #232a38;
  border-radius: 12px; padding: 8px 10px 6px; overflow-x: auto; }
.panel h2 { margin: 6px 4px 2px; font-size: 15px; }
table.grid { width: 100%; border-collapse: collapse; font-size: 13px; }
table.grid th, table.grid td { padding: 6px 8px; text-align: right;
  border-bottom: 1px solid #232a38; white-space: nowrap; }
table.grid th:first-child, table.grid td:first-child { text-align: left; }
table.grid thead th { color: #8b96a8; font-weight: 600; }
table.grid tbody tr:hover { background: #1a2030; }
footer { padding: 18px 28px 26px; color: #6b7280; font-size: 12px; }
@media (max-width: 720px) {
  .panel, .kpis, header, footer, .note { margin-left: 12px; margin-right: 12px; }
  .kpis { padding-left: 12px; padding-right: 12px; }
}
"""


# --- Page rendering ------------------------------------------------------------------------
#: Chart settings in the browser: a responsive width, no ad logo and no extra buttons.
PLOTLY_CONFIG: dict[str, Any] = {
    "responsive": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def _figure_html(fig: go.Figure, div_id: str, inline_plotly: bool, first: bool) -> str:
    """Turn a figure into an HTML fragment; in inline mode the first chart carries Plotly along."""
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=bool(inline_plotly and first),
        config=PLOTLY_CONFIG,
        div_id=div_id,
    )


def render_html(
    title: str,
    cards: list[tuple[str, str]],
    blocks: list[Block],
    notes: list[str],
    inline_plotly: bool = False,
) -> str:
    """Build the whole page: the header, the KPIs, the notes, the charts and the tables."""
    parts = [
        "<!DOCTYPE html>",
        '<html lang="ru"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{escape(title)}</title>",
    ]
    if not inline_plotly:
        parts.append(f'<script src="{PLOTLY_CDN}" charset="utf-8"></script>')
    parts.append(f"<style>{PAGE_CSS}</style>")
    parts.append("</head><body>")
    parts.append(
        f"<header><h1>{escape(title)}</h1>"
        '<p class="sub">Standalone веб-дашборд smc_zero на Plotly</p></header>'
    )
    cards_html = "".join(
        f'<div class="card"><span class="k">{escape(label)}</span>'
        f'<span class="v">{escape(value)}</span></div>'
        for label, value in cards
    )
    parts.append(f'<section class="kpis">{cards_html}</section>')
    parts.extend(f'<div class="note">{escape(note)}</div>' for note in notes)
    first_figure = True
    for block in blocks:
        anchor = escape(block.anchor)
        parts.append(f'<section class="panel"><h2 id="{anchor}">{escape(block.heading)}</h2>')
        if block.figure is not None:
            parts.append(_figure_html(block.figure, block.anchor, inline_plotly, first_figure))
            first_figure = False
        elif block.html:
            parts.append(block.html)
        parts.append("</section>")
    parts.append("<footer>Сгенерировано tools/web_dashboard/scripts/generate_dashboard.py</footer>")
    parts.append("</body></html>")
    return "\n".join(parts)


def write_dashboard(html_text: str, output: Path) -> Path:
    """Write the HTML, creating the missing directories (including the tool's ``public/``)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_text, encoding="utf-8")
    return output


# --- Entry point ----------------------------------------------------------------------------
def _resolve_sources(args: argparse.Namespace) -> tuple[Report, pd.DataFrame | None]:
    """Pick the main report and the optional OHLC tape from the command line arguments."""
    tape = load_tape(Path(args.tape)) if args.tape else None
    if args.demo:
        report = build_demo_data()
        if tape is None:
            tape = build_demo_tape(report)
        return report, tape
    if args.report_dir:
        return load_report(Path(args.report_dir)), tape
    if args.compare:
        return load_report(Path(args.compare[0])), tape
    raise ValueError("нужен --demo или --report-dir (либо --compare с двумя папками отчётов)")


def _report_with_label(path: Path) -> tuple[str, Report]:
    """Read a report for the comparison and give it a short label from the folder name."""
    report = load_report(path)
    return path.name or str(path), report


def main(argv: Sequence[str] | None = None) -> int:
    """Build the dashboard: parse the arguments, read the reports, write the HTML."""
    args = parse_args(argv)
    try:
        primary, tape = _resolve_sources(args)
        compare_blocks: list[Block] = []
        if args.compare:
            label_a, report_a = _report_with_label(Path(args.compare[0]))
            label_b, report_b = _report_with_label(Path(args.compare[1]))
            compare_blocks = build_compare_blocks(label_a, report_a, label_b, report_b)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    metrics = compute_metrics(primary)
    cards = build_kpi_cards(primary, metrics)
    blocks, notes = build_figures(primary, tape)
    if primary.is_empty:
        notes.insert(0, "Сделок в отчёте нет: показывать нечего, кроме пустых панелей.")
    else:
        blocks.append(Block("trades", "Последние сделки", html=trades_table_html(primary.trades)))
    blocks.extend(compare_blocks)
    title = args.title or f"SMC backtest dashboard — {primary.symbol} {primary.timeframe}"
    html_text = render_html(title, cards, blocks, notes, inline_plotly=args.inline_plotly)
    output = write_dashboard(html_text, Path(args.output))
    print(f"dashboard: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
