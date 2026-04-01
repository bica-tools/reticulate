"""Level 3 (full fidelity) session type models for real-world protocols.

Three levels of session type modeling fidelity:
  L1 — Flat branch: all methods as independent choices (over-permissive)
  L2 — Trace mining: observed sequences only (over-restrictive)
  L3 — Full protocol: methods + return-value selections + preconditions

This module provides L3 models: manually constructed session types that
faithfully represent the protocol contracts from API documentation,
including boolean/enum return selections and conditional availability.

Scientific Method (Step 97d):
  Observe   — L1 gives modular (M₃), L2 gives non-modular (N₅)
  Question  — What classification does L3 (correct) modeling give?
  Hypothesis — L3 models are predominantly distributive (like benchmarks)
  Predict   — ≥70% of L3 protocols are distributive
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice, check_distributive
from reticulate.valuation import analyze_valuations


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class L3Result:
    """Result of analyzing an L3 protocol model."""
    name: str
    level: str  # "L1", "L2", "L3"
    session_type: str
    num_states: int
    num_transitions: int
    is_lattice: bool
    is_distributive: bool
    is_modular: bool
    has_m3: bool
    has_n5: bool
    classification: str
    is_graded: bool
    rank_is_valuation: bool


# ---------------------------------------------------------------------------
# L3 Protocol Models (from Java/Python API documentation)
# ---------------------------------------------------------------------------

L3_PROTOCOLS: dict[str, dict[str, str]] = {
    # --- Java protocols ---

    "java_iterator": {
        "description": "java.util.Iterator<E> — hasNext selects TRUE/FALSE",
        "L3": "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}",
        "L1": "&{hasNext: end, next: end, remove: end}",
    },

    "jdbc_connection": {
        "description": "java.sql.Connection — createStatement then execute with result selection",
        "L3": "&{createStatement: &{executeQuery: rec X . &{next: +{TRUE: &{getString: X}, FALSE: &{close: &{commit: &{close: end}}}}}}, close: end}",
        "L1": "&{createStatement: end, executeQuery: end, executeUpdate: end, commit: end, rollback: end, close: end}",
    },

    "java_inputstream": {
        "description": "java.io.InputStream — available() selects, read returns data or EOF",
        "L3": "&{open: rec X . &{available: +{TRUE: &{read: +{data: X, EOF: &{close: end}}, skip: X}, FALSE: &{close: end}}, close: end}}",
        "L1": "&{open: end, read: end, skip: end, available: end, mark: end, reset: end, close: end}",
    },

    "java_socket": {
        "description": "java.net.Socket — connect selects success/failure",
        "L3": "&{connect: +{OK: &{getInputStream: &{getOutputStream: rec X . &{read: +{data: X, EOF: &{close: end}}, write: X, close: end}}}, FAIL: end}, close: end}",
        "L1": "&{connect: end, getInputStream: end, getOutputStream: end, close: end}",
    },

    "java_httpurlconnection": {
        "description": "HttpURLConnection — connect then response code selection",
        "L3": "&{setRequestMethod: &{connect: +{OK: &{getResponseCode: +{s200: &{getInputStream: &{read: &{disconnect: end}}}, s404: &{disconnect: end}, s500: &{disconnect: end}}}, FAIL: &{disconnect: end}}}}",
        "L1": "&{setRequestMethod: end, connect: end, getResponseCode: end, getInputStream: end, getOutputStream: end, disconnect: end}",
    },

    "java_servlet": {
        "description": "HttpServlet — init then service dispatches to doGet/doPost/doPut",
        "L3": "&{init: rec X . &{service: +{GET: &{doGet: X}, POST: &{doPost: X}, PUT: &{doPut: X}, DELETE: &{doDelete: X}}, destroy: end}}",
        "L1": "&{init: end, service: end, doGet: end, doPost: end, doPut: end, doDelete: end, destroy: end}",
    },

    "java_outputstream": {
        "description": "java.io.OutputStream — write loop, flush, close",
        "L3": "rec X . &{write: X, flush: X, close: end}",
        "L1": "&{write: end, flush: end, close: end}",
    },

    "java_lock": {
        "description": "java.util.concurrent.locks.Lock — acquire/release cycle",
        "L3": "rec X . &{lock: &{unlock: X}, tryLock: +{TRUE: &{unlock: X}, FALSE: end}}",
        "L1": "&{lock: end, unlock: end, tryLock: end, lockInterruptibly: end}",
    },

    "java_executorservice": {
        "description": "ExecutorService — submit tasks then shutdown",
        "L3": "rec X . &{submit: X, execute: X, shutdown: &{awaitTermination: +{TRUE: end, FALSE: end}}, shutdownNow: end}",
        "L1": "&{submit: end, execute: end, shutdown: end, shutdownNow: end, awaitTermination: end, invokeAll: end}",
    },

    "java_bufferedreader": {
        "description": "java.io.BufferedReader — readLine returns data or null (EOF)",
        "L3": "&{open: rec X . &{readLine: +{data: X, EOF: &{close: end}}, close: end}}",
        "L1": "&{open: end, readLine: end, read: end, ready: end, close: end}",
    },

    # --- Python protocols (L3 re-modeling) ---

    "python_file": {
        "description": "Python file object — open/read/write with EOF selection",
        "L3": "&{open: rec X . &{read: +{data: X, EOF: &{close: end}}, write: X, close: end}}",
        "L1": "&{open: end, read: end, write: end, readline: end, seek: end, flush: end, close: end}",
    },

    "python_sqlite3": {
        "description": "sqlite3.Connection — cursor/execute/fetch cycle",
        "L3": "&{connect: &{cursor: rec X . &{execute: &{fetchone: +{row: X, none: &{commit: &{close: end}}}, fetchall: &{commit: &{close: end}}}, close: &{close: end}}}}",
        "L1": "&{connect: end, cursor: end, execute: end, fetchone: end, fetchall: end, commit: end, rollback: end, close: end}",
    },

    "python_http": {
        "description": "http.client.HTTPConnection — request/response cycle",
        "L3": "&{connect: rec X . &{request: &{getresponse: +{s200: &{read: X}, s404: X, s500: X}}, close: end}}",
        "L1": "&{connect: end, request: end, getresponse: end, read: end, close: end}",
    },

    "python_socket": {
        "description": "socket.socket — server or client mode",
        "L3": "&{create: +{server: &{bind: &{listen: rec X . &{accept: &{recv: &{send: X}}, close: end}}}, client: &{connect: rec Y . &{send: &{recv: Y}, close: end}}}}",
        "L1": "&{create: end, bind: end, listen: end, accept: end, connect: end, send: end, recv: end, close: end}",
    },

    "python_iterator": {
        "description": "Python iterator protocol — __next__ raises StopIteration",
        "L3": "rec X . &{next: +{value: X, StopIteration: end}}",
        "L1": "&{iter: end, next: end}",
    },

    "python_contextmanager": {
        "description": "Context manager — __enter__/__exit__ with exception selection",
        "L3": "&{enter: rec X . &{use: X, exit: +{OK: end, EXCEPTION: end}}}",
        "L1": "&{enter: end, use: end, exit: end}",
    },
}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_l3(name: str, st_str: str, level: str = "L3") -> L3Result:
    """Analyze a single protocol at given fidelity level."""
    ast = parse(st_str)
    ss = build_statespace(ast)
    lr = check_lattice(ss)

    if not lr.is_lattice:
        return L3Result(
            name=name, level=level, session_type=st_str,
            num_states=len(ss.states), num_transitions=len(ss.transitions),
            is_lattice=False, is_distributive=False, is_modular=False,
            has_m3=False, has_n5=False, classification="non-lattice",
            is_graded=False, rank_is_valuation=False,
        )

    dr = check_distributive(ss)
    va = analyze_valuations(ss)

    return L3Result(
        name=name, level=level, session_type=st_str,
        num_states=len(ss.states), num_transitions=len(ss.transitions),
        is_lattice=True, is_distributive=dr.is_distributive,
        is_modular=dr.is_modular, has_m3=dr.has_m3, has_n5=dr.has_n5,
        classification=dr.classification,
        is_graded=va.is_graded, rank_is_valuation=va.rank_is_valuation,
    )


def analyze_all_l3() -> list[L3Result]:
    """Analyze all protocols at L3 fidelity."""
    results: list[L3Result] = []
    for name, spec in L3_PROTOCOLS.items():
        results.append(analyze_l3(name, spec["L3"], "L3"))
    return results


def analyze_all_levels() -> list[L3Result]:
    """Analyze all protocols at BOTH L1 and L3, for comparison."""
    results: list[L3Result] = []
    for name, spec in L3_PROTOCOLS.items():
        results.append(analyze_l3(name, spec["L1"], "L1"))
        results.append(analyze_l3(name, spec["L3"], "L3"))
    return results


def print_l3_report(results: list[L3Result]) -> str:
    """Format an L3 analysis report."""
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("  L3 PROTOCOL FIDELITY ANALYSIS")
    lines.append("=" * 90)

    total = len(results)
    lattice = sum(1 for r in results if r.is_lattice)
    dist = sum(1 for r in results if r.is_distributive)
    mod = sum(1 for r in results if r.is_modular)

    lines.append(f"  Protocols: {total}")
    lines.append(f"  Lattice: {lattice}/{total} ({lattice/total:.0%})")
    lines.append(f"  Distributive: {dist}/{total} ({dist/total:.0%})")
    lines.append(f"  Modular: {mod}/{total} ({mod/total:.0%})")
    lines.append("")

    lines.append(f"  {'Protocol':<26} {'Lv':<4} {'St':>3} {'Tr':>3} "
                 f"{'Class':<14} {'M3':>3} {'N5':>3} {'Graded':>7} {'RkVal':>6}")
    lines.append("  " + "-" * 78)

    for r in results:
        m3 = "Y" if r.has_m3 else "-"
        n5 = "Y" if r.has_n5 else "-"
        g = "YES" if r.is_graded else "NO"
        rv = "YES" if r.rank_is_valuation else "NO"
        lines.append(f"  {r.name:<26} {r.level:<4} {r.num_states:>3} {r.num_transitions:>3} "
                     f"{r.classification:<14} {m3:>3} {n5:>3} {g:>7} {rv:>6}")

    lines.append("=" * 90)
    return "\n".join(lines)


def print_comparison_report(results: list[L3Result]) -> str:
    """Format a side-by-side L1 vs L3 comparison report."""
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("  L1 vs L3 FIDELITY COMPARISON")
    lines.append("=" * 90)

    # Group by protocol name
    by_name: dict[str, dict[str, L3Result]] = {}
    for r in results:
        by_name.setdefault(r.name, {})[r.level] = r

    lines.append(f"  {'Protocol':<24} {'L1 Class':<14} {'L3 Class':<14} {'L1→L3 Change':<20}")
    lines.append("  " + "-" * 70)

    for name, levels in by_name.items():
        l1 = levels.get("L1")
        l3 = levels.get("L3")
        if l1 and l3:
            change = "same" if l1.classification == l3.classification else f"{l1.classification} → {l3.classification}"
            lines.append(f"  {name:<24} {l1.classification:<14} {l3.classification:<14} {change:<20}")

    # Summary
    l1_results = [r for r in results if r.level == "L1"]
    l3_results = [r for r in results if r.level == "L3"]
    l1_dist = sum(1 for r in l1_results if r.is_distributive)
    l3_dist = sum(1 for r in l3_results if r.is_distributive)
    l1_mod = sum(1 for r in l1_results if r.is_modular)
    l3_mod = sum(1 for r in l3_results if r.is_modular)

    lines.append("")
    lines.append(f"  L1: {l1_dist}/{len(l1_results)} distributive, {l1_mod}/{len(l1_results)} modular")
    lines.append(f"  L3: {l3_dist}/{len(l3_results)} distributive, {l3_mod}/{len(l3_results)} modular")
    lines.append("=" * 90)
    return "\n".join(lines)
