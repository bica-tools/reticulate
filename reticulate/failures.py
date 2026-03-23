"""Failure semantics derived from lattice structure (Step 28).

Implements the CSP-style failures model for session type state spaces,
connecting lattice meet/join to refusal and acceptance sets.

Key theorem: in a session type lattice, the meet of two states determines
their shared refusals, and the join determines their shared acceptances.
Failure refinement corresponds to lattice embedding.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FailureResult:
    """Result of failure analysis on a state space.

    Attributes:
        failure_pairs: Set of (trace, refusal_set) pairs.
        refusal_map: Mapping from state ID to the set of refused labels.
        acceptance_map: Mapping from state ID to the set of accepted labels.
        deterministic: True iff each (state, label) has at most one successor.
        lattice_determines: True iff lattice meet/join determine failure structure.
        lattice_details: Details of lattice-determines check (per-pair results).
    """

    failure_pairs: frozenset[tuple[tuple[str, ...], frozenset[str]]]
    refusal_map: dict[int, frozenset[str]]
    acceptance_map: dict[int, frozenset[str]]
    deterministic: bool
    lattice_determines: bool
    lattice_details: str


@dataclass(frozen=True)
class RefinementResult:
    """Result of failure refinement check between two state spaces.

    Attributes:
        refines: True iff ss1 failure-refines ss2.
        witness: If not refines, a (trace, refusal) pair in ss1 not in ss2.
        ss1_failures: Number of failure pairs in ss1.
        ss2_failures: Number of failure pairs in ss2.
    """

    refines: bool
    witness: tuple[tuple[str, ...], frozenset[str]] | None
    ss1_failures: int
    ss2_failures: int


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def _all_labels(ss: StateSpace) -> frozenset[str]:
    """Collect all transition labels in the state space."""
    return frozenset(lbl for _, lbl, _ in ss.transitions)


def compute_refusals(ss: StateSpace, state: int) -> frozenset[str]:
    """Compute the refusal set at a given state.

    The refusal set is the set of labels NOT enabled at the state.
    In CSP terms, these are the events the process refuses.
    """
    all_labs = _all_labels(ss)
    enabled = frozenset(lbl for s, lbl, _ in ss.transitions if s == state)
    return all_labs - enabled


def compute_all_refusals(ss: StateSpace) -> dict[int, frozenset[str]]:
    """Compute the refusal set for every state in the state space."""
    all_labs = _all_labels(ss)
    result: dict[int, frozenset[str]] = {}
    for state in ss.states:
        enabled = frozenset(lbl for s, lbl, _ in ss.transitions if s == state)
        result[state] = all_labs - enabled
    return result


def acceptance_set(ss: StateSpace, state: int) -> frozenset[str]:
    """Compute the acceptance set at a given state.

    The acceptance set is the set of labels enabled at the state.
    This is the complement of the refusal set with respect to all labels.
    """
    return frozenset(lbl for s, lbl, _ in ss.transitions if s == state)


def compute_all_acceptances(ss: StateSpace) -> dict[int, frozenset[str]]:
    """Compute the acceptance set for every state."""
    result: dict[int, frozenset[str]] = {}
    for state in ss.states:
        result[state] = acceptance_set(ss, state)
    return result


# ---------------------------------------------------------------------------
# Trace execution
# ---------------------------------------------------------------------------

def _execute_trace(
    ss: StateSpace,
    trace: tuple[str, ...],
) -> set[int]:
    """Execute a trace from the top state, returning set of reachable states.

    For deterministic state spaces this is a singleton set.
    For nondeterministic ones (multiple targets for same label) it may be larger.
    Returns empty set if the trace is not executable.
    """
    current: set[int] = {ss.top}

    for label in trace:
        next_states: set[int] = set()
        for state in current:
            for s, lbl, t in ss.transitions:
                if s == state and lbl == label:
                    next_states.add(t)
        if not next_states:
            return set()
        current = next_states

    return current


def must_testing(ss: StateSpace, trace: tuple[str, ...]) -> frozenset[str]:
    """After executing trace, what labels MUST be accepted?

    Returns the intersection of acceptance sets over all states reachable
    after the trace. If the trace is not executable, returns empty set.
    In a deterministic system, this equals the acceptance set of the
    unique reached state.
    """
    states = _execute_trace(ss, trace)
    if not states:
        return frozenset()

    result: set[str] | None = None
    for state in states:
        acc = set(acceptance_set(ss, state))
        if result is None:
            result = acc
        else:
            result &= acc

    return frozenset(result) if result is not None else frozenset()


def may_testing(ss: StateSpace, trace: tuple[str, ...]) -> frozenset[str]:
    """After executing trace, what labels MAY be accepted?

    Returns the union of acceptance sets over all states reachable
    after the trace. If the trace is not executable, returns empty set.
    """
    states = _execute_trace(ss, trace)
    if not states:
        return frozenset()

    result: set[str] = set()
    for state in states:
        result |= set(acceptance_set(ss, state))

    return frozenset(result)


# ---------------------------------------------------------------------------
# Failure pairs enumeration
# ---------------------------------------------------------------------------

def _enumerate_traces(
    ss: StateSpace,
    max_depth: int = 20,
) -> set[tuple[str, ...]]:
    """Enumerate all traces up to max_depth from the top state.

    Uses BFS to explore all possible traces (label sequences) that
    can be executed from the initial state.
    """
    traces: set[tuple[str, ...]] = set()
    # BFS: (current_states, trace_so_far)
    queue: deque[tuple[set[int], tuple[str, ...]]] = deque()
    queue.append(({ss.top}, ()))
    traces.add(())  # empty trace is always valid

    visited_configs: set[tuple[frozenset[int], tuple[str, ...]]] = set()

    while queue:
        current_states, trace = queue.popleft()

        if len(trace) >= max_depth:
            continue

        config = (frozenset(current_states), trace)
        if config in visited_configs:
            continue
        visited_configs.add(config)

        # Find all labels enabled from any current state
        enabled_labels: set[str] = set()
        for state in current_states:
            for s, lbl, _ in ss.transitions:
                if s == state:
                    enabled_labels.add(lbl)

        for label in sorted(enabled_labels):
            next_states: set[int] = set()
            for state in current_states:
                for s, lbl, t in ss.transitions:
                    if s == state and lbl == label:
                        next_states.add(t)
            if next_states:
                new_trace = trace + (label,)
                traces.add(new_trace)
                queue.append((next_states, new_trace))

    return traces


def failure_pairs(
    ss: StateSpace,
    max_depth: int = 20,
) -> frozenset[tuple[tuple[str, ...], frozenset[str]]]:
    """Compute all (trace, refusal_set) failure pairs.

    A failure pair (s, X) means: after executing trace s, the process
    can refuse the set X of labels. For deterministic processes, there
    is exactly one refusal set per trace. For nondeterministic processes,
    each reachable state after the trace contributes its own refusals.

    The refusal sets include all subsets of refused labels (downward closed),
    but for efficiency we only record the maximal refusal (the full complement
    of the acceptance set).
    """
    all_labs = _all_labels(ss)
    traces = _enumerate_traces(ss, max_depth)
    result: set[tuple[tuple[str, ...], frozenset[str]]] = set()

    for trace in traces:
        states = _execute_trace(ss, trace)
        if not states:
            continue
        # For each reachable state, compute its refusal set
        for state in states:
            enabled = frozenset(lbl for s, lbl, _ in ss.transitions if s == state)
            refusal = all_labs - enabled
            result.add((trace, refusal))

    return frozenset(result)


# ---------------------------------------------------------------------------
# Failure refinement
# ---------------------------------------------------------------------------

def check_failure_refinement(
    ss1: StateSpace,
    ss2: StateSpace,
    max_depth: int = 20,
) -> RefinementResult:
    """Check whether ss1 failure-refines ss2.

    ss1 failure-refines ss2 iff for every (trace, refusal) failure pair
    of ss1, there exists a (trace, refusal') pair in ss2 with
    refusal (intersected with ss2's alphabet) being a subset of refusal'.

    Refusal sets are normalized to the common alphabet (intersection of
    both processes' label sets) so that labels unique to one process
    do not cause spurious refinement failures.

    This corresponds to: ss1 is a more deterministic (more specified)
    process than ss2 over their shared interface.
    """
    fp1 = failure_pairs(ss1, max_depth)
    fp2 = failure_pairs(ss2, max_depth)

    # Common alphabet for normalization
    common = _all_labels(ss1) & _all_labels(ss2)

    # Build lookup for ss2 failures by trace
    fp2_by_trace: dict[tuple[str, ...], set[frozenset[str]]] = {}
    for trace, refusal in fp2:
        fp2_by_trace.setdefault(trace, set()).add(refusal & common)

    witness: tuple[tuple[str, ...], frozenset[str]] | None = None

    for trace, refusal in fp1:
        ref_common = refusal & common
        # Check if ss2 has a failure pair with this trace where
        # ss1's refusal (restricted to common labels) is a subset
        ss2_refs = fp2_by_trace.get(trace)
        if ss2_refs is None:
            # ss2 cannot execute this trace at all; ss1 can, so
            # ss2 implicitly refuses everything after this trace
            continue
        found = any(ref_common <= r2 for r2 in ss2_refs)
        if not found:
            witness = (trace, refusal)
            break

    return RefinementResult(
        refines=witness is None,
        witness=witness,
        ss1_failures=len(fp1),
        ss2_failures=len(fp2),
    )


# ---------------------------------------------------------------------------
# Lattice-determines-failures check
# ---------------------------------------------------------------------------

def lattice_determines_failures(ss: StateSpace) -> bool:
    """Verify that the lattice ordering determines the failure structure.

    Two properties are checked, both appropriate for the heterogeneous
    alphabets of session type state spaces:

    1. **Meet-refusal property**: refusals(meet(s1, s2)), restricted to the
       labels present at *both* s1 and s2 (their common local alphabet),
       contains refusals(s1) ∩ refusals(s2) restricted likewise.  Labels
       refused by both states at the common alphabet are also refused at
       their meet.

    2. **Join-acceptance property**: acceptance(join(s1, s2)), restricted
       to the common local alphabet, contains acceptance(s1) ∩ acceptance(s2)
       restricted likewise.  Labels accepted by both states are also
       accepted at their join.

    These properties express that the lattice meet/join faithfully track
    the failure information over the shared interface of any two states.
    Session types have heterogeneous alphabets (each state enables
    different labels), so we restrict to the common alphabet before
    comparing.

    Returns True if the state space is a lattice and both properties hold
    for all pairs.
    """
    from reticulate.lattice import check_lattice, compute_meet, compute_join

    lr = check_lattice(ss)
    if not lr.is_lattice:
        return False

    all_refusals = compute_all_refusals(ss)
    all_acceptances = compute_all_acceptances(ss)

    # Precompute the local alphabet at each state (labels enabled there
    # OR appearing as labels in the full state space; we restrict to
    # labels that appear at *either* state under comparison)
    states_list = sorted(ss.states)
    for i, s1 in enumerate(states_list):
        for s2 in states_list[i + 1:]:
            # Common local alphabet: labels enabled at s1 or s2
            local_alpha = all_acceptances[s1] | all_acceptances[s2]
            if not local_alpha:
                continue  # both are terminal, nothing to check

            # Property 1: Meet-refusal over common alphabet
            m = compute_meet(ss, s1, s2)
            if m is not None:
                meet_ref = all_refusals[m] & local_alpha
                shared_ref = (all_refusals[s1] & all_refusals[s2]) & local_alpha
                if not (shared_ref <= meet_ref):
                    return False

            # Property 2: Join-acceptance over common alphabet
            j = compute_join(ss, s1, s2)
            if j is not None:
                join_acc = all_acceptances[j] & local_alpha
                shared_acc = (all_acceptances[s1] & all_acceptances[s2]) & local_alpha
                if not (shared_acc <= join_acc):
                    return False

    return True


# ---------------------------------------------------------------------------
# High-level analysis
# ---------------------------------------------------------------------------

def _is_deterministic(ss: StateSpace) -> bool:
    """Check if the state space is deterministic.

    A state space is deterministic if for each state and label,
    there is at most one target state.
    """
    seen: set[tuple[int, str]] = set()
    for s, lbl, _ in ss.transitions:
        key = (s, lbl)
        if key in seen:
            return False
        seen.add(key)
    return True


def analyze_failures(
    ss: StateSpace,
    max_depth: int = 20,
) -> FailureResult:
    """Perform complete failure analysis on a state space.

    Computes refusals, acceptances, failure pairs, and checks whether
    the lattice structure determines the failure semantics.
    """
    refusal_map = compute_all_refusals(ss)
    acceptance_map = compute_all_acceptances(ss)
    fp = failure_pairs(ss, max_depth)
    deterministic = _is_deterministic(ss)
    lat_det = lattice_determines_failures(ss)

    if lat_det:
        details = "Lattice meet/join fully determine failure structure."
    else:
        from reticulate.lattice import check_lattice
        lr = check_lattice(ss)
        if not lr.is_lattice:
            details = "State space is not a lattice; lattice-determines check not applicable."
        else:
            details = "Lattice meet/join do not fully determine failure structure."

    return FailureResult(
        failure_pairs=fp,
        refusal_map=refusal_map,
        acceptance_map=acceptance_map,
        deterministic=deterministic,
        lattice_determines=lat_det,
        lattice_details=details,
    )
