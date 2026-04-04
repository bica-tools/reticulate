"""Coverage analysis comparing manual vs generated test suites (TOPLAS Week 5).

Measures session-type transition coverage of:
1. Manual test suites — traces a human developer would write
2. Generated test suites — from session type path enumeration
3. Conformance traces — from Week 4 real trace checking

Shows that session-type-generated tests achieve higher coverage and
reveal gaps in manual tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from reticulate.parser import parse
from reticulate.statespace import build_statespace, StateSpace
from reticulate.testgen import enumerate_valid_paths, ValidPath, Step
from reticulate.coverage import compute_coverage, CoverageResult


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TraceCoverageResult:
    """Coverage of a set of traces against a session type."""
    suite_name: str
    num_traces: int
    transition_coverage: float
    state_coverage: float
    covered_transitions: int
    total_transitions: int
    covered_states: int
    total_states: int
    uncovered_labels: list[str]  # transition labels not exercised


@dataclass(frozen=True)
class CoverageComparison:
    """Comparison between manual and generated test coverage."""
    api_name: str
    manual: TraceCoverageResult
    generated: TraceCoverageResult
    coverage_gap: float  # generated - manual transition coverage
    uncovered_by_manual: list[str]  # labels only generated tests cover


# ---------------------------------------------------------------------------
# Trace → ValidPath conversion
# ---------------------------------------------------------------------------

def traces_to_valid_paths(
    ss: StateSpace,
    traces: list[list[str]],
) -> list[ValidPath]:
    """Convert raw label traces to ValidPath objects by walking state space.

    Each trace is a list of labels (interleaved method calls + selection
    outcomes). We walk the state space following these labels.
    """
    paths: list[ValidPath] = []
    for trace in traces:
        steps: list[Step] = []
        current = ss.top
        valid = True
        for label in trace:
            # Find matching transition from current state
            found = False
            for src, lbl, tgt in ss.transitions:
                if src == current and lbl == label:
                    kind = "selection" if label.isupper() else "method"
                    steps.append(Step(label=label, target=tgt, kind=kind))
                    current = tgt
                    found = True
                    break
            if not found:
                valid = False
                break
        if valid and steps:
            paths.append(ValidPath(steps=tuple(steps)))
    return paths


# ---------------------------------------------------------------------------
# Manual test suites (what a human developer would write)
# ---------------------------------------------------------------------------

MANUAL_SUITES: dict[str, list[list[str]]] = {
    "sqlite3.Connection": [
        # Typical developer tests for sqlite3
        ["execute", "ROWS_AFFECTED", "commit", "OK", "close"],
        ["cursor", "CURSOR", "execute", "ROWS_AFFECTED", "commit", "OK", "close"],
        ["close"],
        # Most developers test happy path only — miss error paths
    ],

    "java.util.Iterator": [
        # Standard iteration test
        ["hasNext", "TRUE", "next", "ELEMENT", "hasNext", "TRUE", "next", "ELEMENT", "hasNext", "FALSE"],
        # Empty iterator
        ["hasNext", "FALSE"],
        # Single element
        ["hasNext", "TRUE", "next", "ELEMENT", "hasNext", "FALSE"],
        # Most developers forget to test remove()
    ],

    "java.io.InputStream": [
        # Read until EOF
        ["read", "DATA", "read", "DATA", "read", "EOF", "close"],
        # Just close
        ["close"],
        # Single read
        ["read", "DATA", "close"],
        # Most developers forget mark/reset, available, skip
    ],

    "http.client.HTTPConnection": [
        # GET 200
        ["connect", "CONNECTED", "request", "SENT", "getresponse", "OK_200", "close"],
        # GET 404
        ["connect", "CONNECTED", "request", "SENT", "getresponse", "NOT_FOUND_404", "close"],
        # Most developers only test happy path + one error
    ],

    "smtplib.SMTP": [
        # Full send with TLS
        ["connect", "CONNECTED", "ehlo", "EHLO_OK", "starttls", "TLS_OK",
         "ehlo", "EHLO_OK", "login", "AUTH_OK", "sendmail", "SENT", "quit"],
        # Most developers only test one path
    ],

    "ftplib.FTP": [
        # Download
        ["connect", "CONNECTED", "login", "AUTH_OK", "cwd", "CHANGED",
         "retrbinary", "DOWNLOADED", "quit"],
        # List
        ["connect", "CONNECTED", "login", "AUTH_OK", "nlst", "LISTING", "quit"],
    ],

    "ssl.SSLSocket": [
        # Standard TLS
        ["connect", "CONNECTED", "do_handshake", "OK", "getpeercert", "CERT",
         "recv", "DATA", "send", "SENT", "close"],
        # Handshake fail
        ["connect", "CONNECTED", "do_handshake", "CERT_ERROR", "close"],
    ],
}


# ---------------------------------------------------------------------------
# Coverage computation
# ---------------------------------------------------------------------------

def compute_trace_coverage(
    session_type_str: str,
    traces: list[list[str]],
    suite_name: str,
) -> TraceCoverageResult:
    """Compute coverage of a trace suite against a session type."""
    ast = parse(session_type_str)
    ss = build_statespace(ast)

    paths = traces_to_valid_paths(ss, traces)
    cr = compute_coverage(ss, paths=paths)

    # Find uncovered transition labels
    uncovered_labels = sorted(set(
        lbl for _, lbl, _ in cr.uncovered_transitions
    ))

    return TraceCoverageResult(
        suite_name=suite_name,
        num_traces=len(traces),
        transition_coverage=cr.transition_coverage,
        state_coverage=cr.state_coverage,
        covered_transitions=len(cr.covered_transitions),
        total_transitions=len(cr.covered_transitions) + len(cr.uncovered_transitions),
        covered_states=len(cr.covered_states),
        total_states=len(cr.covered_states) + len(cr.uncovered_states),
        uncovered_labels=uncovered_labels,
    )


def compute_generated_coverage(
    session_type_str: str,
    max_paths: int = 20,
) -> TraceCoverageResult:
    """Compute coverage of session-type-generated tests."""
    ast = parse(session_type_str)
    ss = build_statespace(ast)

    paths, _ = enumerate_valid_paths(ss, max_revisits=2, max_paths=max_paths)
    cr = compute_coverage(ss, paths=paths)

    uncovered_labels = sorted(set(
        lbl for _, lbl, _ in cr.uncovered_transitions
    ))

    return TraceCoverageResult(
        suite_name="generated",
        num_traces=len(paths),
        transition_coverage=cr.transition_coverage,
        state_coverage=cr.state_coverage,
        covered_transitions=len(cr.covered_transitions),
        total_transitions=len(cr.covered_transitions) + len(cr.uncovered_transitions),
        covered_states=len(cr.covered_states),
        total_states=len(cr.covered_states) + len(cr.uncovered_states),
        uncovered_labels=uncovered_labels,
    )


def compare_coverage(
    api_name: str,
    session_type_str: str,
    manual_traces: list[list[str]],
    max_generated_paths: int = 20,
) -> CoverageComparison:
    """Compare manual vs generated test coverage for one API."""
    manual = compute_trace_coverage(session_type_str, manual_traces, "manual")
    generated = compute_generated_coverage(session_type_str, max_generated_paths)

    gap = generated.transition_coverage - manual.transition_coverage

    # Labels covered by generated but not manual
    manual_uncovered = set(manual.uncovered_labels)
    gen_uncovered = set(generated.uncovered_labels)
    only_generated = sorted(manual_uncovered - gen_uncovered)

    return CoverageComparison(
        api_name=api_name,
        manual=manual,
        generated=generated,
        coverage_gap=gap,
        uncovered_by_manual=only_generated,
    )


# ---------------------------------------------------------------------------
# Run all comparisons
# ---------------------------------------------------------------------------

def run_all_comparisons() -> list[CoverageComparison]:
    """Run coverage comparison for all APIs."""
    from reticulate.python_api_extractor import extract_session_type as extract_py
    from reticulate.java_api_extractor import extract_session_type as extract_java

    comparisons: list[CoverageComparison] = []

    apis = [
        ("sqlite3.Connection", extract_py),
        ("http.client.HTTPConnection", extract_py),
        ("smtplib.SMTP", extract_py),
        ("ftplib.FTP", extract_py),
        ("ssl.SSLSocket", extract_py),
        ("java.util.Iterator", extract_java),
        ("java.io.InputStream", extract_java),
    ]

    for api_name, extractor in apis:
        r = extractor(api_name)
        manual = MANUAL_SUITES.get(api_name, [])
        comp = compare_coverage(api_name, r.inferred_type, manual)
        comparisons.append(comp)

    return comparisons


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_coverage_report(comparisons: list[CoverageComparison]) -> str:
    """Format coverage comparison as a report."""
    lines: list[str] = []
    lines.append("=" * 110)
    lines.append("  SESSION TYPE COVERAGE ANALYSIS — MANUAL vs GENERATED TESTS")
    lines.append("=" * 110)

    avg_manual = sum(c.manual.transition_coverage for c in comparisons) / len(comparisons)
    avg_gen = sum(c.generated.transition_coverage for c in comparisons) / len(comparisons)

    lines.append(f"  APIs analyzed:          {len(comparisons)}")
    lines.append(f"  Avg manual coverage:    {avg_manual:.0%}")
    lines.append(f"  Avg generated coverage: {avg_gen:.0%}")
    lines.append(f"  Avg improvement:        +{avg_gen - avg_manual:.0%}")
    lines.append("")

    header = (f"  {'API':<30} {'Manual':>8} {'#Traces':>8} "
              f"{'Generated':>10} {'#Paths':>7} {'Gap':>6} {'Gaps Found':<20}")
    lines.append(header)
    lines.append("  " + "-" * 106)

    for c in comparisons:
        gap_str = f"+{c.coverage_gap:.0%}" if c.coverage_gap > 0 else f"{c.coverage_gap:.0%}"
        gaps = ", ".join(c.uncovered_by_manual[:3]) if c.uncovered_by_manual else "—"
        lines.append(
            f"  {c.api_name:<30} {c.manual.transition_coverage:>7.0%} "
            f"{c.manual.num_traces:>8} {c.generated.transition_coverage:>9.0%} "
            f"{c.generated.num_traces:>7} {gap_str:>6} {gaps:<20}"
        )

    lines.append("  " + "-" * 106)
    lines.append("")

    # Detail: what manual tests miss
    lines.append("  COVERAGE GAPS (labels only generated tests cover):")
    lines.append("  " + "-" * 106)
    for c in comparisons:
        if c.uncovered_by_manual:
            lines.append(f"  {c.api_name}: {', '.join(c.uncovered_by_manual)}")
        else:
            lines.append(f"  {c.api_name}: (no additional coverage from generated)")
    lines.append("")

    lines.append("=" * 110)
    return "\n".join(lines)


def format_latex_coverage_table(comparisons: list[CoverageComparison]) -> str:
    """Format coverage comparison as LaTeX table."""
    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Transition coverage: manual vs.\ session-type-generated test suites.}")
    lines.append(r"\label{tab:coverage}")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"API & Manual & $n$ & Generated & $n$ & Gap \\")
    lines.append(r"\midrule")

    for c in comparisons:
        api_short = c.api_name.split(".")[-1]
        gap = c.coverage_gap
        gap_str = f"+{gap:.0%}" if gap > 0 else f"{gap:.0%}"
        lines.append(
            f"\\texttt{{{api_short}}} & {c.manual.transition_coverage:.0%} & "
            f"{c.manual.num_traces} & {c.generated.transition_coverage:.0%} & "
            f"{c.generated.num_traces} & {gap_str} \\\\"
        )

    avg_m = sum(c.manual.transition_coverage for c in comparisons) / len(comparisons)
    avg_g = sum(c.generated.transition_coverage for c in comparisons) / len(comparisons)
    lines.append(r"\midrule")
    lines.append(f"\\textbf{{Average}} & {avg_m:.0%} & & {avg_g:.0%} & & "
                 f"+{avg_g - avg_m:.0%} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)
