#!/usr/bin/env bash
#
# daily_signal_check.sh - one daily run of the live config for the manual demo face (Э10'.4).
#
# The script does three things and nothing else:
#
#   1. starts ``smc-backtest`` under ``nohup`` over the last four years up to today with the numbers
#      of the live config (§7.19), writing its report into ``<reports>/daily_<YYYYMMDD>`` and its
#      console output into ``<repo>/tmp/daily_backtest_<YYYYMMDD>.log`` (Э14.1);
#   2. waits for that process and stops with its exit code (0 = the run wrote a report);
#   3. prints the trade header and the trades opened since yesterday 22:00 MSK (``open_time >=``;
#      Moscow is UTC+3 all year, see :mod:`smc_zero.utils.time`), so the signals of the day are the
#      only thing a reader has to look at.
#
# The demo face is manual on purpose: the script never talks to a broker and never sends an order.
# What it prints is executed by hand and written into ``docs/TRADING_JOURNAL.md``.
#
# The tape of the run is the raw MetaTrader 5 export, never the ``./data`` folder of the repository
# (Э11'.2): the backtest is called with ``--data-source mt5`` and the export base ``SMC_DATA_DIR``,
# whose default is ``$HOME/_data/mt5`` (``/home/com/_data/mt5`` here - the folder the Python layer
# falls back to on its own, SPEC_SMC.md §7.20 п.104).  A fresh "Bars" download therefore feeds the
# check with no conversion in between.
#
# The clock of that tape is the one of the broker *server*, not UTC (Э17.B, SPEC_SMC.md §7.26): the
# live config declares ``source_clock: broker_server``, and the loader moves every stamp onto UTC
# before the run reads a bar - so the session windows, the D1/H1 bias and the signals printed below
# speak in real time.  A tape that already holds UTC needs ``SMC_SOURCE_CLOCK=utc``, and
# ``smc-check-tape --source-clock`` tells the two apart before the run (docs/LIVE_TRADING_GUIDE.md).
#
# Cron (the executable bit is part of the repository, ``core.fileMode=true``).  The base is written
# out here as well, so the line keeps its meaning if that default ever moves:
#
#   0 22 * * 1-5 SMC_DATA_DIR=/home/com/_data/mt5 /home/com/work2/python/cfd/project/smc_zero/scripts/daily_signal_check.sh >> /home/com/work2/python/cfd/project/smc_zero/tmp/cron_smc.log 2>&1
#
# Overrides - all optional, for a dry run or a test; the defaults are the live ones:
#
#   SMC_REPO           repository root (default: the parent folder of this script)
#   SMC_DATA_DIR       base of the MT5 export a run reads (default: ``$HOME/_data/mt5``)
#   SMC_WORKDIR        folder the backtest is started from (default: ``SMC_REPO``)
#   SMC_CONFIG         live YAML of the run (default: ``SMC_REPO/configs/live_eurusd_m15.yaml``)
#   SMC_SOURCE_CLOCK   clock of the raw tape, ``utc`` or ``broker_server`` (default: the
#                      ``source_clock`` key of SMC_CONFIG, i.e. ``broker_server``; Э17.B)
#   SMC_REPORT_ROOT    report root; the dated folder is made inside it (default: ``SMC_REPO/reports``)
#   SMC_START/SMC_END  window days, YYYY-MM-DD (default: four years ago .. today, UTC)
#   SMC_SINCE_UTC      "YYYY-MM-DD HH:MM:SS" UTC threshold of a new trade
#                      (default: 19:00 yesterday, which is 22:00 MSK yesterday)
#   SMC_LOG            log file of the backtest (default: ``<repo>/tmp/daily_backtest_<YYYYMMDD>.log``;
#                      the folder of the path is created on demand - Э14.1)
#   SMC_PYTHON         interpreter (default: ``SMC_REPO/.venv/bin/python``, else ``python3``)
#   SMC_WAIT_TIMEOUT   seconds to wait for the backtest (default: 10800; 0 waits forever)
#   SMC_REPORT_FOLDER  do not run a backtest, only re-print the signals of this report folder
#   SMC_PAPER          1 (default) also measures the night with the candidate of Э17.z++,
#                      0 runs the live config alone (a dry run)
#   SMC_PAPER_CONFIGS  the pairs ``role=path`` of that parallel run (default: the shipped
#                      ``configs/research_e17zp_candidate.yaml`` under the role ``candidate``)
#   SMC_PAPER_DIR      folder of the JSON snapshots (default: ``<repo>/tmp/paper``)
#
# GNU coreutils are assumed for the date arithmetic (the project runs on Linux); on another system
# set SMC_START, SMC_END and SMC_SINCE_UTC by hand.
#
# ``set -e`` is deliberately not used: the exit code of the runner is a result to forward, not a
# reason to stop before the log has been shown.

set -uo pipefail

# An absolute path, whatever the caller typed: the backtest runs from the work directory.
to_abs() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    ./*) printf '%s\n' "${PWD}/${1#./}" ;;
    *) printf '%s\n' "${PWD}/$1" ;;
  esac
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SMC_REPO=${SMC_REPO:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
SMC_WORKDIR=$(to_abs "${SMC_WORKDIR:-${SMC_REPO}}")
SMC_CONFIG=$(to_abs "${SMC_CONFIG:-${SMC_REPO}/configs/live_eurusd_m15.yaml}")
SMC_REPORT_ROOT=$(to_abs "${SMC_REPORT_ROOT:-${SMC_REPO}/reports}")
# The tape base of the check (Э11'.2): the MT5 export, never ``./data`` - a fresh "Bars" download
# feeds the run as it is.  The value of the caller wins (the cron line sets one); the default is
# ``$HOME/_data/mt5``, the same folder the Python layer falls back to.  It is exported, because the
# runner is a child process and reads the base from its environment.
SMC_DATA_DIR=${SMC_DATA_DIR:-${HOME:-/home/com}/_data/mt5}
export SMC_DATA_DIR
# The base a run reads: ``mt5`` is the export above - the flag of Э11'.1 (SPEC_SMC.md §7.20 п.104).
DATA_SOURCE=mt5
# The clock of the raw tape (Э17.B, SPEC_SMC.md §7.26).  The live config names it
# (``source_clock: broker_server``: both the export and ``./data`` carry broker server time), so
# the run of the night normalises without a flag.  ``SMC_SOURCE_CLOCK`` exists for a dry run
# against a tape that already holds UTC; when it is set it wins over the file and is passed on.
SOURCE_CLOCK=${SMC_SOURCE_CLOCK:-}
# The clock the config names, for the header only: a top-level scalar, read without a YAML parser
# so the log of the night says which clock the run used (rule 4: the sources of a run are named).
CONFIG_CLOCK=$(sed -n 's/^source_clock:[[:space:]]*\([^[:space:]#]*\).*/\1/p' "${SMC_CONFIG}" 2>/dev/null | head -1)
CLOCK_ARGS=()
if [ -n "${SOURCE_CLOCK}" ]; then
  CLOCK_ARGS=(--source-clock "${SOURCE_CLOCK}")
elif [ -z "${CONFIG_CLOCK}" ]; then
  # Without the key the runner falls back to its own default; name it here in the same words.
  CONFIG_CLOCK=utc
fi

TODAY=$(date -u '+%Y%m%d')
DAY_LABEL=$(date -u '+%Y-%m-%d')
# The log lives in ``./tmp`` of the project, never in the system ``/tmp`` (Э14.1): that folder
# survives a reboot and keeps the artifacts of one project apart from those of other processes.
# The path is built from the repository root and not from the working directory, because a cron
# run starts in the home folder.  An explicit ``SMC_LOG`` still wins over the default.
SMC_LOG=${SMC_LOG:-${SMC_REPO}/tmp/daily_backtest_${TODAY}.log}
START=${SMC_START:-$(date -u -d '4 years ago' '+%Y-%m-%d')}
END=${SMC_END:-${DAY_LABEL}}
WAIT_TIMEOUT=${SMC_WAIT_TIMEOUT:-10800}
DATED="${SMC_REPORT_ROOT}/daily_${TODAY}"
RERUN_FOLDER=${SMC_REPORT_FOLDER:+$(to_abs "${SMC_REPORT_FOLDER}")}

# The parallel paper mode of Э17.z++: the live config above stays the *base* reading of the night,
# and beside it the same night is measured with the research candidate of Э17.z++
# (``configs/research_e17zp_candidate.yaml``: the -3 h shift of the session windows plus a daily cap
# of three setups per level).  The mode is additive by construction: the base run, its report folder
# and the list printed below are exactly what they were, while every reading gains a report folder of
# its own and one JSON snapshot under ``<repo>/tmp/paper/`` - that file is what
# ``docs/PAPER_JOURNAL.md`` compares.  A reading that fails stops the check with its own code, so a
# silently missing candidate cannot pass for "no signal of the night".
SMC_PAPER=${SMC_PAPER:-1}
PAPER_CONFIGS=${SMC_PAPER_CONFIGS:-candidate=${SMC_REPO}/configs/research_e17zp_candidate.yaml}
PAPER_DIR=$(to_abs "${SMC_PAPER_DIR:-${SMC_REPO}/tmp/paper}")

if [ -n "${SMC_SINCE_UTC:-}" ]; then
  SINCE=${SMC_SINCE_UTC}
else
  SINCE=$(date -u -d 'yesterday 19:00' '+%Y-%m-%d %H:%M:%S' 2>/dev/null) || SINCE=""
fi
if [ -z "${SINCE}" ]; then
  echo "error: no GNU date here; set SMC_SINCE_UTC (and SMC_START / SMC_END) by hand" >&2
  exit 2
fi

if [ -n "${SMC_PYTHON:-}" ]; then
  PY=${SMC_PYTHON}
elif [ -x "${SMC_REPO}/.venv/bin/python" ]; then
  PY=${SMC_REPO}/.venv/bin/python
else
  PY=python3
fi

echo "=== smc daily signal check ${DAY_LABEL} ==="
echo "  workdir  ${SMC_WORKDIR}"
echo "  config   ${SMC_CONFIG}"
echo "  window   ${START} .. ${END}"
echo "  reports  ${SMC_REPORT_ROOT}"
echo "  source   ${DATA_SOURCE} under ${SMC_DATA_DIR}"
if [ -n "${SOURCE_CLOCK}" ]; then
  echo "  clock    ${SOURCE_CLOCK} (SMC_SOURCE_CLOCK, перекрывает конфиг)"
else
  echo "  clock    ${CONFIG_CLOCK} (source_clock конфига)"
fi
echo "  since    ${SINCE} UTC (yesterday 22:00 MSK)"
echo "  journal  docs/TRADING_JOURNAL.md"
if [ "${SMC_PAPER}" != "0" ] && [ -z "${RERUN_FOLDER}" ]; then
  echo "  paper    base + ${PAPER_CONFIGS} -> ${PAPER_DIR}"
fi

RC=0
if [ -n "${RERUN_FOLDER}" ]; then
  FOLDER=${RERUN_FOLDER}
  echo "  mode     re-print of ${FOLDER} (no backtest)"
else
  echo "  python   ${PY}"
  echo "  log      ${SMC_LOG}"
  # The folder of the log is made here, so a fresh clone and a cron run need nothing prepared.
  mkdir -p -- "$(dirname -- "${SMC_LOG}")" || {
    echo "error: cannot make the log folder $(dirname -- "${SMC_LOG}")" >&2
    exit 2
  }
  cd "${SMC_WORKDIR}" || {
    echo "error: no work directory ${SMC_WORKDIR}" >&2
    exit 2
  }
  # nohup plus a redirect, so the run survives this script and its terminal (rule of long tasks).
  nohup env PYTHONPATH="${SMC_REPO}/src:${SMC_REPO}" "${PY}" -m scripts.run_backtest \
    --config-path "${SMC_CONFIG}" \
    --data-source "${DATA_SOURCE}" \
    "${CLOCK_ARGS[@]}" \
    --start "${START}" \
    --end "${END}" \
    --report-dir "${DATED}" \
    >"${SMC_LOG}" 2>&1 </dev/null &
  BACKTEST_PID=$!
  echo "  pid      ${BACKTEST_PID}"
  echo "  monitor  tail -5 ${SMC_LOG}"
  echo "  status   ps -p ${BACKTEST_PID} -o pid,etime,%cpu,%mem --no-headers"
  echo "  stop     kill ${BACKTEST_PID}"

  elapsed=0
  while kill -0 "${BACKTEST_PID}" 2>/dev/null; do
    if [ "${WAIT_TIMEOUT}" -gt 0 ] && [ "${elapsed}" -ge "${WAIT_TIMEOUT}" ]; then
      echo "warning: the backtest still runs after ${WAIT_TIMEOUT}s; nohup keeps it alive" >&2
      echo "         follow it with: tail -f ${SMC_LOG}" >&2
      exit 2
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
  wait "${BACKTEST_PID}"
  RC=$?

  if [ "${RC}" -ne 0 ]; then
    echo "error: the backtest exited with ${RC}; last lines of ${SMC_LOG}:" >&2
    tail -5 "${SMC_LOG}" >&2
    exit "${RC}"
  fi

  # The runner names the folder of one run itself, so the report is read here and never guessed.
  FOLDER=""
  while IFS= read -r candidate; do
    FOLDER=${candidate%/}
    break
  done < <(ls -dt "${DATED}"/backtest_*/ 2>/dev/null)
fi

if [ -z "${FOLDER}" ] || [ ! -d "${FOLDER}" ]; then
  echo "error: no report folder of a run under ${DATED}" >&2
  exit 1
fi
TRADES="${FOLDER}/trades.csv"
if [ ! -f "${TRADES}" ]; then
  echo "error: no trade log at ${TRADES}" >&2
  exit 1
fi

echo "  report   ${FOLDER}"
if [ -f "${FOLDER}/summary.txt" ]; then
  echo "--- summary of the run ---"
  cat "${FOLDER}/summary.txt"
fi

echo "--- new signals: open_time >= ${SINCE} UTC (yesterday 22:00 MSK) ---"
awk -F, -v since="${SINCE}" '
  NR == 1 {
    print
    for (column = 1; column <= NF; column++) if ($column == "open_time") stamp = column
    next
  }
  stamp && substr($stamp, 1, 19) >= since { print; found++ }
  END { printf "signals: %d\n", found + 0 }
' "${TRADES}"

# --- Parallel paper mode: the same night read by the candidate of Э17.z++ (see the settings above). ---
if [ "${SMC_PAPER}" != "0" ] && [ -z "${RERUN_FOLDER}" ]; then
  mkdir -p -- "${PAPER_DIR}" || {
    echo "error: cannot make the paper folder ${PAPER_DIR}" >&2
    exit 2
  }
  # One reading -> one JSON snapshot.  The reader is ``scripts.paper_snapshot``: it filters the trade
  # log with the very ``open_time >=`` rule printed above, so the file and the screen agree.
  paper_snapshot() {
    local role=$1 config=$2 folder=$3 out=$4
    if ! env PYTHONPATH="${SMC_REPO}/src:${SMC_REPO}" "${PY}" -m scripts.paper_snapshot \
        --report "${folder}" \
        --role "${role}" \
        --config "${config}" \
        --since "${SINCE}" \
        --window-start "${START}" \
        --window-end "${END}" \
        --date "${DAY_LABEL}" \
        --out "${out}" >/dev/null; then
      echo "error: no paper snapshot of ${role} at ${out}" >&2
      return 1
    fi
    echo "  paper    ${role} -> ${out}"
  }
  # The base reading is the run above; every other one is a pair of ``SMC_PAPER_CONFIGS`` and is
  # started exactly like it - nohup, a log of its own, a report folder of its own under the dated
  # root, the same window and the clock its own file names.
  paper_snapshot base "${SMC_CONFIG}" "${FOLDER}" "${PAPER_DIR}/base_${TODAY}.json" || exit 2
  for pair in ${PAPER_CONFIGS}; do
    role=${pair%%=*}
    PAPER_CONFIG=$(to_abs "${pair#*=}")
    case "${role}" in
      "" | *[!A-Za-z0-9_-]*)
        echo "error: a paper role is a name of letters, digits, '_' or '-', got '${role}'" >&2
        exit 2
        ;;
    esac
    if [ ! -f "${PAPER_CONFIG}" ]; then
      echo "error: no paper config at ${PAPER_CONFIG} (pair '${pair}')" >&2
      exit 2
    fi
    PAPER_LOG="${PAPER_DIR}/${role}_${TODAY}.log"
    PAPER_ROOT="${DATED}/${role}"
    echo "--- paper role ${role}: ${PAPER_CONFIG} ---"
    cd "${SMC_WORKDIR}" || {
      echo "error: no work directory ${SMC_WORKDIR}" >&2
      exit 2
    }
    PAPER_CLOCK_ARGS=("${CLOCK_ARGS[@]}")
    PAPER_CLOCK=$(sed -n 's/^source_clock:[[:space:]]*\([^[:space:]#]*\).*/\1/p' \
      "${PAPER_CONFIG}" 2>/dev/null | head -1)
    if [ -z "${SOURCE_CLOCK}" ] && [ -n "${PAPER_CLOCK}" ]; then
      PAPER_CLOCK_ARGS=(--source-clock "${PAPER_CLOCK}")
    fi
    nohup env PYTHONPATH="${SMC_REPO}/src:${SMC_REPO}" "${PY}" -m scripts.run_backtest \
      --config-path "${PAPER_CONFIG}" \
      --data-source "${DATA_SOURCE}" \
      "${PAPER_CLOCK_ARGS[@]}" \
      --start "${START}" \
      --end "${END}" \
      --report-dir "${PAPER_ROOT}" \
      >"${PAPER_LOG}" 2>&1 </dev/null &
    PAPER_PID=$!
    echo "  pid      ${PAPER_PID}"
    echo "  monitor  tail -5 ${PAPER_LOG}"
    echo "  status   ps -p ${PAPER_PID} -o pid,etime,%cpu,%mem --no-headers"
    echo "  stop     kill ${PAPER_PID}"
    elapsed=0
    while kill -0 "${PAPER_PID}" 2>/dev/null; do
      if [ "${WAIT_TIMEOUT}" -gt 0 ] && [ "${elapsed}" -ge "${WAIT_TIMEOUT}" ]; then
        echo "warning: ${role} still runs after ${WAIT_TIMEOUT}s; nohup keeps it alive" >&2
        echo "         follow it with: tail -f ${PAPER_LOG}" >&2
        exit 2
      fi
      sleep 2
      elapsed=$((elapsed + 2))
    done
    wait "${PAPER_PID}"
    PAPER_RC=$?
    if [ "${PAPER_RC}" -ne 0 ]; then
      echo "error: ${role} exited with ${PAPER_RC}; last lines of ${PAPER_LOG}:" >&2
      tail -5 "${PAPER_LOG}" >&2
      exit "${PAPER_RC}"
    fi
    # The folder of the role is named by the runner itself, so it is read and never guessed.
    PAPER_FOLDER=""
    while IFS= read -r folder; do
      PAPER_FOLDER=${folder%/}
      break
    done < <(ls -dt "${PAPER_ROOT}"/backtest_*/ 2>/dev/null)
    if [ -z "${PAPER_FOLDER}" ] || [ ! -d "${PAPER_FOLDER}" ]; then
      echo "error: no report folder of ${role} under ${PAPER_ROOT}" >&2
      exit 1
    fi
    paper_snapshot "${role}" "${PAPER_CONFIG}" "${PAPER_FOLDER}" \
      "${PAPER_DIR}/${role}_${TODAY}.json" || exit 2
  done
  echo "--- paper journal: docs/PAPER_JOURNAL.md (snapshots in ${PAPER_DIR}) ---"
fi

echo "--- every executed signal goes into docs/TRADING_JOURNAL.md ---"
exit 0
