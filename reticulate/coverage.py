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
    output_dir: str,
    *,
    fmt: str = "svg",
    result: LatticeResult | None = None,
) -> list[str]:
    """Render a coverage storyboard to numbered files in *output_dir*.

    Returns the list of output file paths.
    """
    import os
    from reticulate.visualize import render_hasse

    os.makedirs(output_dir, exist_ok=True)
    paths: list[str] = []

    # Frame 0: before any tests
    initial = _build_coverage(ss, frozenset(ss.transitions), set())
    out = render_hasse(
        ss,
        os.path.join(output_dir, "frame-000"),
        fmt=fmt,
        result=result,
        title="Before tests (0%)",
        coverage=initial,
    )
    paths.append(out)

    # One frame per test
    for i, frame in enumerate(frames):
        tc = frame.coverage.transition_coverage * 100
        out = render_hasse(
            ss,
            os.path.join(output_dir, f"frame-{i + 1:03d}"),
            fmt=fmt,
            result=result,
            title=f"{frame.test_name}() — {tc:.0f}%",
            coverage=frame.coverage,
        )
        paths.append(out)

    return paths


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
