"""Test coverage analysis for session type state spaces.

Computes which transitions in a StateSpace are exercised by generated
test paths (valid paths, violations, incomplete prefixes), and produces
a CoverageResult that can be passed to visualize.dot_source() for
coverage-coloured Hasse diagrams.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.lattice import LatticeResult
    from reticulate.statespace import StateSpace
    from reticulate.testgen import (
        ClientProgram,
        EnumerationResult,
        IncompletePrefix,
        Step,
        ValidPath,
        ViolationPoint,
    )


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoverageResult:
    """Coverage analysis of a state space against enumerated test paths."""
    covered_transitions: frozenset[tuple[int, str, int]]
    uncovered_transitions: frozenset[tuple[int, str, int]]
    covered_states: frozenset[int]
    uncovered_states: frozenset[int]
    transition_coverage: float  # 0.0 to 1.0
    state_coverage: float       # 0.0 to 1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_coverage(
    ss: StateSpace,
    *,
    paths: list[ValidPath] | None = None,
    violations: list[ViolationPoint] | None = None,
    incomplete_prefixes: list[IncompletePrefix] | None = None,
    result: EnumerationResult | None = None,
) -> CoverageResult:
    """Compute transition and state coverage over *ss*.

    Accepts either an ``EnumerationResult`` via *result*, or individual
    lists of paths/violations/incomplete_prefixes.  If *result* is given,
    the individual lists are ignored.
    """
    if result is not None:
        paths = result.valid_paths
        violations = result.violations
        incomplete_prefixes = result.incomplete_prefixes

    all_transitions = frozenset(ss.transitions)

    # Nothing to cover → 100%
    if not all_transitions:
        return CoverageResult(
            covered_transitions=frozenset(),
            uncovered_transitions=frozenset(),
            covered_states=frozenset(ss.states),
            uncovered_states=frozenset(),
            transition_coverage=1.0,
            state_coverage=1.0,
        )

    # Collect covered transitions from all path types
    covered: set[tuple[int, str, int]] = set()

    for steps in _iter_step_sequences(paths, violations, incomplete_prefixes):
        _collect_transitions(ss.top, steps, all_transitions, covered)

    covered_trans = frozenset(covered)
    uncovered_trans = all_transitions - covered_trans

    # States: covered if they appear as src or tgt in any covered transition
    covered_st: set[int] = set()
    for src, _, tgt in covered_trans:
        covered_st.add(src)
        covered_st.add(tgt)
    # Top is always covered if any path exists
    if covered_trans:
        covered_st.add(ss.top)

    covered_states = frozenset(covered_st)
    uncovered_states = frozenset(ss.states - covered_st)

    n_trans = len(all_transitions)
    n_states = len(ss.states)

    return CoverageResult(
        covered_transitions=covered_trans,
        uncovered_transitions=uncovered_trans,
        covered_states=covered_states,
        uncovered_states=uncovered_states,
        transition_coverage=len(covered_trans) / n_trans if n_trans else 1.0,
        state_coverage=len(covered_states) / n_states if n_states else 1.0,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _iter_step_sequences(
    paths: list[ValidPath] | None,
    violations: list[ViolationPoint] | None,
    incomplete_prefixes: list[IncompletePrefix] | None,
):
    """Yield each step sequence from the three path types."""
    if paths:
        for p in paths:
            yield p.steps
    if violations:
        for v in violations:
            yield v.prefix_path
    if incomplete_prefixes:
        for ip in incomplete_prefixes:
            yield ip.steps


def _collect_transitions(
    top: int,
    steps: tuple[Step, ...],
    all_transitions: frozenset[tuple[int, str, int]],
    covered: set[tuple[int, str, int]],
) -> None:
    """Walk a step sequence and add matching transitions to *covered*."""
    current = top
    for step in steps:
        trans = (current, step.label, step.target)
        if trans in all_transitions:
            covered.add(trans)
        current = step.target


# ---------------------------------------------------------------------------
# Coverage storyboard — one frame per generated test
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoverageFrame:
    """A single frame in the coverage storyboard."""
    test_name: str              # Java test method name (e.g. "validPath_open_read")
    test_kind: str              # "valid", "violation", or "incomplete"
    coverage: CoverageResult    # cumulative coverage after this test


def coverage_storyboard(
    ss: StateSpace,
    result: EnumerationResult,
) -> list[CoverageFrame]:
    """Build a coverage storyboard: one frame per generated test.

    Returns a list of ``CoverageFrame`` objects in the same order as
    tests appear in the generated source.  Each frame's ``coverage``
    is cumulative — it includes all transitions covered up to and
    including that test.
    """
    from reticulate.testgen import client_program_name_suffix, enumerate_client_programs

    all_transitions = frozenset(ss.transitions)
    if not all_transitions:
        return []

    covered: set[tuple[int, str, int]] = set()
    frames: list[CoverageFrame] = []

    # 1. Valid paths (client programs)
    programs, _ = enumerate_client_programs(ss)
    for program in programs:
        suffix = client_program_name_suffix(program) or "empty"
        name = f"validPath_{suffix}"
        # Walk the program's flattened steps
        _collect_from_client_program(ss.top, program, all_transitions, covered)
        frames.append(CoverageFrame(
            test_name=name,
            test_kind="valid",
            coverage=_build_coverage(ss, all_transitions, covered),
        ))

    # 2. Violations
    for v in result.violations:
        _collect_transitions(ss.top, v.prefix_path, all_transitions, covered)
        prefix_labels = v.prefix_labels
        suffix = (("initial_" + v.disabled_method) if not prefix_labels
                  else "_".join(prefix_labels) + "_" + v.disabled_method)
        frames.append(CoverageFrame(
            test_name=f"violation_{suffix}",
            test_kind="violation",
            coverage=_build_coverage(ss, all_transitions, covered),
        ))

    # 3. Incomplete prefixes
    for p in result.incomplete_prefixes:
        _collect_transitions(ss.top, p.steps, all_transitions, covered)
        suffix = "_".join(p.labels)
        frames.append(CoverageFrame(
            test_name=f"incomplete_{suffix}",
            test_kind="incomplete",
            coverage=_build_coverage(ss, all_transitions, covered),
        ))

    return frames


def render_storyboard(
    ss: StateSpace,
    frames: list[CoverageFrame],
    output_path: str,
    *,
    result: LatticeResult | None = None,
    title: str | None = None,
) -> str:
    """Render a coverage storyboard as a single HTML file.

    Each frame is an inline SVG captioned with the test method name
    and cumulative coverage percentage.  Returns *output_path*.
    """
    from reticulate.visualize import dot_source
    import subprocess

    page_title = title or "Coverage Storyboard"

    html_parts: list[str] = []
    html_parts.append("<!DOCTYPE html>")
    html_parts.append(f"<html><head><meta charset='utf-8'><title>{_html_escape(page_title)}</title>")
    html_parts.append("<style>")
    html_parts.append("body { font-family: Helvetica, Arial, sans-serif; background: #f8fafc; margin: 2em; }")
    html_parts.append("h1 { color: #1e293b; }")
    html_parts.append(".frame { display: inline-block; vertical-align: top; margin: 1em; "
                       "background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1em; }")
    html_parts.append(".frame-title { font-size: 0.85em; font-weight: bold; margin-bottom: 0.5em; }")
    html_parts.append(".frame-title .pct { color: #64748b; }")
    html_parts.append(".kind-valid .frame-title { color: #16a34a; }")
    html_parts.append(".kind-violation .frame-title { color: #dc2626; }")
    html_parts.append(".kind-incomplete .frame-title { color: #d97706; }")
    html_parts.append(".kind-initial .frame-title { color: #64748b; }")
    html_parts.append("</style></head><body>")
    html_parts.append(f"<h1>{_html_escape(page_title)}</h1>")

    # Frame 0: before any tests
    initial_cov = _build_coverage(ss, frozenset(ss.transitions), set())
    svg_0 = _dot_to_svg(dot_source(ss, result, title="Before tests", coverage=initial_cov))
    html_parts.append('<div class="frame kind-initial">')
    html_parts.append('<div class="frame-title">Before tests <span class="pct">(0%)</span></div>')
    html_parts.append(svg_0)
    html_parts.append("</div>")

    # One frame per test
    for frame in frames:
        tc = frame.coverage.transition_coverage * 100
        svg = _dot_to_svg(dot_source(ss, result, coverage=frame.coverage))
        kind_class = f"kind-{frame.test_kind}"
        html_parts.append(f'<div class="frame {kind_class}">')
        html_parts.append(
            f'<div class="frame-title">{_html_escape(frame.test_name)}() '
            f'<span class="pct">({tc:.0f}%)</span></div>'
        )
        html_parts.append(svg)
        html_parts.append("</div>")

    html_parts.append("</body></html>")

    with open(output_path, "w") as f:
        f.write("\n".join(html_parts))

    return output_path


def _html_escape(s: str) -> str:
    """Escape a string for HTML content."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _dot_to_svg(dot: str) -> str:
    """Convert DOT source to SVG using the graphviz dot command."""
    import subprocess
    proc = subprocess.run(
        ["dot", "-Tsvg"],
        input=dot,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"dot command failed: {proc.stderr}")
    # Strip the XML declaration and DOCTYPE to embed inline
    svg = proc.stdout
    for prefix in ('<?xml', '<!DOCTYPE'):
        while prefix in svg:
            start = svg.index(prefix)
            end = svg.index('>', start) + 1
            svg = svg[:start] + svg[end:]
    return svg.strip()


def _build_coverage(
    ss: StateSpace,
    all_transitions: frozenset[tuple[int, str, int]],
    covered: set[tuple[int, str, int]],
) -> CoverageResult:
    """Build a CoverageResult snapshot from the current covered set."""
    covered_trans = frozenset(covered)
    uncovered_trans = all_transitions - covered_trans

    covered_st: set[int] = set()
    for src, _, tgt in covered_trans:
        covered_st.add(src)
        covered_st.add(tgt)
    if covered_trans:
        covered_st.add(ss.top)

    covered_states = frozenset(covered_st)
    uncovered_states = frozenset(ss.states - covered_st)
    n_trans = len(all_transitions)
    n_states = len(ss.states)

    return CoverageResult(
        covered_transitions=covered_trans,
        uncovered_transitions=uncovered_trans,
        covered_states=covered_states,
        uncovered_states=uncovered_states,
        transition_coverage=len(covered_trans) / n_trans if n_trans else 1.0,
        state_coverage=len(covered_states) / n_states if n_states else 1.0,
    )


def _collect_from_client_program(
    top: int,
    program: ClientProgram,
    all_transitions: frozenset[tuple[int, str, int]],
    covered: set[tuple[int, str, int]],
) -> None:
    """Walk a ClientProgram tree and collect all transitions it covers."""
    from reticulate.testgen import MethodCallNode, SelectionSwitchNode, TerminalNode
    _walk_program(top, program, all_transitions, covered)


def _walk_program(
    current: int,
    program: ClientProgram,
    all_transitions: frozenset[tuple[int, str, int]],
    covered: set[tuple[int, str, int]],
) -> None:
    """Recursively walk a ClientProgram tree."""
    from reticulate.testgen import MethodCallNode, SelectionSwitchNode, TerminalNode

    if isinstance(program, TerminalNode):
        return
    elif isinstance(program, MethodCallNode):
        # Find matching transition
        for src, lbl, tgt in all_transitions:
            if src == current and lbl == program.label:
                covered.add((src, lbl, tgt))
                _walk_program(tgt, program.next, all_transitions, covered)
                return
    elif isinstance(program, SelectionSwitchNode):
        # Selection: all branches are covered (the object picks)
        for branch_label, branch_program in program.branches.items():
            for src, lbl, tgt in all_transitions:
                if src == current and lbl == branch_label:
                    covered.add((src, lbl, tgt))
                    _walk_program(tgt, branch_program, all_transitions, covered)
                    break
