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
    from reticulate.statespace import StateSpace
    from reticulate.testgen import (
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


@dataclass(frozen=True)
class CoverageFrame:
    """A single frame in an incremental coverage sequence."""
    test_index: int
    test_kind: str              # "valid", "violation", or "incomplete"
    description: str
    steps: tuple[Step, ...]
    coverage: CoverageResult


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


def incremental_coverage(
    ss: StateSpace,
    *,
    result: EnumerationResult | None = None,
    paths: list[ValidPath] | None = None,
    violations: list[ViolationPoint] | None = None,
    incomplete_prefixes: list[IncompletePrefix] | None = None,
) -> list[CoverageFrame]:
    """Compute coverage incrementally, one frame per test.

    Returns a list of ``CoverageFrame`` objects, one per test path/violation/
    incomplete prefix.  Each frame contains the ``CoverageResult`` after that
    test, plus a description and the step sequence that was just executed.
    """
    if result is not None:
        paths = result.valid_paths
        violations = result.violations
        incomplete_prefixes = result.incomplete_prefixes

    all_transitions = frozenset(ss.transitions)
    if not all_transitions:
        return []

    covered: set[tuple[int, str, int]] = set()
    frames: list[CoverageFrame] = []

    # Valid paths
    for i, p in enumerate(paths or []):
        _collect_transitions(ss.top, p.steps, all_transitions, covered)
        labels = " → ".join(s.label for s in p.steps) or "(empty)"
        frames.append(CoverageFrame(
            test_index=len(frames),
            test_kind="valid",
            description=f"Valid path {i + 1}: {labels}",
            steps=p.steps,
            coverage=_make_result(ss, all_transitions, covered),
        ))

    # Violations
    for i, v in enumerate(violations or []):
        _collect_transitions(ss.top, v.prefix_path, all_transitions, covered)
        prefix = " → ".join(s.label for s in v.prefix_path)
        frames.append(CoverageFrame(
            test_index=len(frames),
            test_kind="violation",
            description=f"Violation {i + 1}: [{prefix}] then {v.disabled_method}",
            steps=v.prefix_path,
            coverage=_make_result(ss, all_transitions, covered),
        ))

    # Incomplete prefixes
    for i, ip in enumerate(incomplete_prefixes or []):
        _collect_transitions(ss.top, ip.steps, all_transitions, covered)
        labels = " → ".join(s.label for s in ip.steps)
        frames.append(CoverageFrame(
            test_index=len(frames),
            test_kind="incomplete",
            description=f"Incomplete {i + 1}: {labels}",
            steps=ip.steps,
            coverage=_make_result(ss, all_transitions, covered),
        ))

    return frames


def _make_result(
    ss: StateSpace,
    all_transitions: frozenset[tuple[int, str, int]],
    covered: set[tuple[int, str, int]],
) -> CoverageResult:
    """Build a CoverageResult from the current covered set."""
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
