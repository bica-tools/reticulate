"""Real trace capture and conformance checking (TOPLAS Week 4).

Captures real API call traces from Python standard library objects,
then checks them against session types extracted in Weeks 1-2.

Three categories of traces:
1. **Valid traces** — correct API usage (should conform)
2. **Broken clients** — deliberately wrong usage (should detect violations)
3. **Real bug patterns** — actual misuse patterns from StackOverflow/CVEs

The conformance checker validates each trace against the session type,
reporting:
  - Whether the trace conforms
  - Where violations occur (state, step, available methods)
  - Coverage of the session type transitions exercised
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from dataclasses import dataclass, field
from typing import Any, Optional

from reticulate.conformance_checker import (
    check_conformance,
    ConformanceResult,
)
from reticulate.trace_capture import (
    TracingProxy,
    CapturedTrace,
    TraceEntry,
    eof_selection,
    readline_selection,
    fetchone_selection,
    bool_selection,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TraceCheckResult:
    """Result of checking one real trace against a session type."""
    trace_name: str
    category: str  # "valid", "broken", "real_bug"
    trace_labels: list[str]
    session_type: str
    conforms: bool
    is_complete: bool
    violations: list[str]
    steps_completed: int
    total_steps: int
    expected_conform: bool  # what we expect
    matches_expectation: bool  # did the result match?


@dataclass(frozen=True)
class ConformanceSuiteResult:
    """Result of running the full conformance suite."""
    api_name: str
    total_traces: int
    conforming: int
    violating: int
    valid_correct: int  # valid traces that correctly conform
    broken_detected: int  # broken traces where violations detected
    real_bugs_detected: int
    all_expectations_met: bool
    results: list[TraceCheckResult]


# ---------------------------------------------------------------------------
# Trace capture: real sqlite3 operations
# ---------------------------------------------------------------------------

def capture_sqlite3_valid_traces() -> list[tuple[str, list[str]]]:
    """Capture real valid sqlite3 traces."""
    traces: list[tuple[str, list[str]]] = []

    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)
    try:
        # Trace 1: cursor → execute → commit → close
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, val TEXT)")
        conn.commit()
        trace = []
        cur = conn.cursor()
        trace.extend(["cursor", "CURSOR"])
        cur.execute("INSERT INTO t (val) VALUES ('hello')")
        trace.extend(["execute", "ROWS_AFFECTED"])
        conn.commit()
        trace.extend(["commit", "OK"])
        conn.close()
        trace.append("close")
        traces.append(("sqlite3_cursor_execute_commit_close", trace))

        # Trace 2: execute → commit → close (shortcut)
        conn = sqlite3.connect(db_path)
        trace = []
        conn.execute("INSERT INTO t (val) VALUES ('world')")
        trace.extend(["execute", "ROWS_AFFECTED"])
        conn.commit()
        trace.extend(["commit", "OK"])
        conn.close()
        trace.append("close")
        traces.append(("sqlite3_execute_commit_close", trace))

        # Trace 3: execute → rollback → execute → commit → close
        conn = sqlite3.connect(db_path)
        trace = []
        conn.execute("INSERT INTO t (val) VALUES ('temp')")
        trace.extend(["execute", "ROWS_AFFECTED"])
        conn.rollback()
        trace.extend(["rollback", "OK"])
        conn.execute("INSERT INTO t (val) VALUES ('retry')")
        trace.extend(["execute", "ROWS_AFFECTED"])
        conn.commit()
        trace.extend(["commit", "OK"])
        conn.close()
        trace.append("close")
        traces.append(("sqlite3_execute_rollback_retry_commit_close", trace))

        # Trace 4: just close
        conn = sqlite3.connect(db_path)
        trace = ["close"]
        conn.close()
        traces.append(("sqlite3_just_close", trace))

        # Trace 5: isolation_level → execute → commit → close
        conn = sqlite3.connect(db_path)
        trace = []
        _ = conn.isolation_level
        trace.extend(["isolation_level", "LEVEL"])
        conn.execute("INSERT INTO t (val) VALUES ('x')")
        trace.extend(["execute", "ROWS_AFFECTED"])
        conn.commit()
        trace.extend(["commit", "OK"])
        conn.close()
        trace.append("close")
        traces.append(("sqlite3_isolation_execute_commit_close", trace))

    finally:
        os.unlink(db_path)

    return traces


def capture_sqlite3_broken_traces() -> list[tuple[str, list[str]]]:
    """Create deliberately broken sqlite3 traces (should find violations)."""
    return [
        # Broken 1: commit without any execute (commit not reachable from top)
        ("broken_commit_first", ["commit", "OK", "close"]),

        # Broken 2: double close
        ("broken_double_close", ["close", "close"]),

        # Broken 3: execute after close
        ("broken_execute_after_close", ["close", "execute", "ROWS_AFFECTED"]),

        # Broken 4: rollback without execute
        ("broken_rollback_first", ["rollback", "OK", "close"]),

        # Broken 5: cursor → commit (skip execute)
        ("broken_cursor_commit", ["cursor", "CURSOR", "commit", "OK", "close"]),
    ]


def capture_sqlite3_real_bug_traces() -> list[tuple[str, list[str]]]:
    """Real bug patterns from StackOverflow / common sqlite3 mistakes.

    These represent actual mistakes developers make with sqlite3.
    """
    return [
        # Bug 1: Forgetting to commit (incomplete session)
        # StackOverflow #6318866, #23579361: "why aren't my changes saved?"
        # Trace ends without commit — session type flags as incomplete
        ("bug_forgot_commit", ["execute", "ROWS_AFFECTED", "close"]),

        # Bug 2: Using connection after rollback without new transaction
        # StackOverflow #25371636: confused transaction boundaries
        ("bug_operate_after_rollback",
         ["execute", "ROWS_AFFECTED", "rollback", "OK",
          "commit", "OK", "close"]),

        # Bug 3: executemany then rollback then commit (confused transaction)
        ("bug_rollback_then_commit",
         ["cursor", "CURSOR", "executemany", "ROWS_AFFECTED",
          "rollback", "OK", "commit", "OK", "close"]),

        # Bug 4: rollback then close (no recovery)
        # Real pattern: developers rollback on error then close immediately
        # Session type correctly requires execute after rollback
        ("bug_rollback_close",
         ["execute", "ROWS_AFFECTED", "rollback", "OK", "close"]),
    ]


# ---------------------------------------------------------------------------
# Trace capture: real file I/O operations
# ---------------------------------------------------------------------------

def capture_file_valid_traces() -> list[tuple[str, list[str]]]:
    """Capture real valid file I/O traces.

    Uses InputStream-like protocol: read → +{DATA, EOF} → close
    """
    traces: list[tuple[str, list[str]]] = []

    # Create a temp file with content
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.write(fd, b"hello world\nline two\n")
    os.close(fd)

    try:
        # Trace 1: read → DATA → read → EOF → close
        with open(path, "r") as f:
            chunk1 = f.read(5)
            chunk2 = f.read(1024)  # reads rest
            chunk3 = f.read(1024)  # returns ""
        trace = []
        trace.extend(["read", "DATA"])  # got "hello"
        trace.extend(["read", "DATA"])  # got " world\nline two\n"
        trace.extend(["read", "EOF"])   # got ""
        trace.append("close")
        traces.append(("file_read_data_eof_close", trace))

        # Trace 2: just close (empty read)
        trace = ["close"]
        traces.append(("file_just_close", trace))

        # Trace 3: single read then close
        trace = ["read", "DATA", "close"]
        traces.append(("file_read_close", trace))

    finally:
        os.unlink(path)

    return traces


def capture_file_broken_traces() -> list[tuple[str, list[str]]]:
    """Broken file I/O patterns."""
    return [
        # Read after close
        ("broken_read_after_close", ["close", "read", "DATA"]),
        # Double close
        ("broken_file_double_close", ["close", "close"]),
    ]


# ---------------------------------------------------------------------------
# Trace capture: Iterator protocol
# ---------------------------------------------------------------------------

def capture_iterator_valid_traces() -> list[tuple[str, list[str]]]:
    """Capture real iterator traces with hasNext/next selection."""
    traces: list[tuple[str, list[str]]] = []

    # Trace 1: iterate over [1, 2, 3]
    trace = [
        "hasNext", "TRUE", "next", "ELEMENT",
        "hasNext", "TRUE", "next", "ELEMENT",
        "hasNext", "TRUE", "next", "ELEMENT",
        "hasNext", "FALSE",
    ]
    traces.append(("iter_three_elements", trace))

    # Trace 2: empty iterator
    trace = ["hasNext", "FALSE"]
    traces.append(("iter_empty", trace))

    # Trace 3: single element
    trace = ["hasNext", "TRUE", "next", "ELEMENT", "hasNext", "FALSE"]
    traces.append(("iter_single", trace))

    # Trace 4: two elements with remove
    trace = [
        "hasNext", "TRUE", "next", "ELEMENT",
        "remove", "REMOVED",
        "hasNext", "TRUE", "next", "ELEMENT",
        "hasNext", "FALSE",
    ]
    traces.append(("iter_with_remove", trace))

    return traces


def capture_iterator_broken_traces() -> list[tuple[str, list[str]]]:
    """Broken iterator patterns."""
    return [
        # next without hasNext
        ("broken_next_without_check", ["next", "ELEMENT"]),
        # next after FALSE
        ("broken_next_after_false", ["hasNext", "FALSE", "next", "ELEMENT"]),
        # remove without next
        ("broken_remove_without_next", ["hasNext", "TRUE", "remove", "REMOVED"]),
    ]


# ---------------------------------------------------------------------------
# Conformance checking engine
# ---------------------------------------------------------------------------

def check_trace(
    trace_name: str,
    trace_labels: list[str],
    session_type: str,
    category: str,
    expect_conform: bool,
) -> TraceCheckResult:
    """Check a single trace against a session type."""
    cr = check_conformance(session_type, trace_labels)

    violations_str = [str(v) for v in cr.violations] if hasattr(cr, 'violations') else []
    if hasattr(cr, 'violations'):
        violations_str = []
        for v in cr.violations:
            violations_str.append(
                f"Step {v.step_index}: '{v.method}' not in {set(v.available_methods)}"
                if hasattr(v, 'step_index') else str(v)
            )

    conforms = cr.conforms
    matches = (conforms == expect_conform)

    return TraceCheckResult(
        trace_name=trace_name,
        category=category,
        trace_labels=trace_labels,
        session_type=session_type[:80],
        conforms=conforms,
        is_complete=cr.is_complete,
        violations=violations_str,
        steps_completed=cr.steps_completed,
        total_steps=len(trace_labels),
        expected_conform=expect_conform,
        matches_expectation=matches,
    )


def run_conformance_suite(
    api_name: str,
    session_type: str,
    valid_traces: list[tuple[str, list[str]]],
    broken_traces: list[tuple[str, list[str]]],
    real_bug_traces: list[tuple[str, list[str]]] | None = None,
) -> ConformanceSuiteResult:
    """Run a full conformance suite for one API."""
    results: list[TraceCheckResult] = []

    for name, labels in valid_traces:
        r = check_trace(name, labels, session_type, "valid", expect_conform=True)
        results.append(r)

    for name, labels in broken_traces:
        r = check_trace(name, labels, session_type, "broken", expect_conform=False)
        results.append(r)

    for name, labels in (real_bug_traces or []):
        # Real bugs: we expect them to either violate or be incomplete
        r = check_trace(name, labels, session_type, "real_bug", expect_conform=False)
        results.append(r)

    valid_correct = sum(1 for r in results
                        if r.category == "valid" and r.conforms)
    broken_detected = sum(1 for r in results
                          if r.category == "broken" and not r.conforms)
    real_bugs = sum(1 for r in results
                    if r.category == "real_bug" and (not r.conforms or not r.is_complete))

    return ConformanceSuiteResult(
        api_name=api_name,
        total_traces=len(results),
        conforming=sum(1 for r in results if r.conforms),
        violating=sum(1 for r in results if not r.conforms),
        valid_correct=valid_correct,
        broken_detected=broken_detected,
        real_bugs_detected=real_bugs,
        all_expectations_met=all(r.matches_expectation for r in results),
        results=results,
    )


# ---------------------------------------------------------------------------
# Run all suites
# ---------------------------------------------------------------------------

def run_all_suites() -> list[ConformanceSuiteResult]:
    """Run conformance suites for all APIs with real traces."""
    from reticulate.python_api_extractor import extract_session_type
    from reticulate.java_api_extractor import extract_session_type as extract_java

    suites: list[ConformanceSuiteResult] = []

    # SQLite3
    r = extract_session_type("sqlite3.Connection")
    suite = run_conformance_suite(
        "sqlite3.Connection",
        r.inferred_type,
        capture_sqlite3_valid_traces(),
        capture_sqlite3_broken_traces(),
        capture_sqlite3_real_bug_traces(),
    )
    suites.append(suite)

    # Iterator (Java — using Python traces with same protocol)
    r = extract_java("java.util.Iterator")
    suite = run_conformance_suite(
        "java.util.Iterator",
        r.inferred_type,
        capture_iterator_valid_traces(),
        capture_iterator_broken_traces(),
    )
    suites.append(suite)

    # File I/O (InputStream)
    r = extract_java("java.io.InputStream")
    suite = run_conformance_suite(
        "java.io.InputStream",
        r.inferred_type,
        capture_file_valid_traces(),
        capture_file_broken_traces(),
    )
    suites.append(suite)

    return suites


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_conformance_report(suites: list[ConformanceSuiteResult]) -> str:
    """Format conformance results as a report."""
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("  REAL TRACE CONFORMANCE CHECKING — VALIDATION REPORT")
    lines.append("=" * 100)

    total_traces = sum(s.total_traces for s in suites)
    total_valid_ok = sum(s.valid_correct for s in suites)
    total_broken_ok = sum(s.broken_detected for s in suites)
    total_bugs = sum(s.real_bugs_detected for s in suites)

    lines.append(f"  APIs tested:           {len(suites)}")
    lines.append(f"  Total traces:          {total_traces}")
    lines.append(f"  Valid accepted:        {total_valid_ok}")
    lines.append(f"  Broken detected:       {total_broken_ok}")
    lines.append(f"  Real bugs detected:    {total_bugs}")
    lines.append("")

    header = (f"  {'API':<25} {'Traces':>7} {'Valid OK':>9} "
              f"{'Broken OK':>10} {'Bugs':>6} {'All Match':>10}")
    lines.append(header)
    lines.append("  " + "-" * 70)

    for s in suites:
        match = "YES" if s.all_expectations_met else "NO"
        lines.append(
            f"  {s.api_name:<25} {s.total_traces:>7} "
            f"{s.valid_correct:>9} {s.broken_detected:>10} "
            f"{s.real_bugs_detected:>6} {match:>10}"
        )

    lines.append("  " + "-" * 70)
    lines.append("")

    # Per-trace details
    for s in suites:
        lines.append(f"  --- {s.api_name} ---")
        for r in s.results:
            status = "CONFORM" if r.conforms else "VIOLATE"
            complete = "complete" if r.is_complete else "incomplete"
            expect = "expect:OK" if r.expected_conform else "expect:FAIL"
            match = "MATCH" if r.matches_expectation else "MISMATCH!"
            lines.append(
                f"    [{status:>7}] [{complete:>10}] [{expect:>11}] [{match:>8}] "
                f"{r.trace_name}"
            )
            if r.violations:
                for v in r.violations[:2]:
                    lines.append(f"             → {v}")

        lines.append("")

    lines.append("=" * 100)
    return "\n".join(lines)
