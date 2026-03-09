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
