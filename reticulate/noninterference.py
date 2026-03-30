"""Non-interference as lattice morphism for session types (Step 89b).

Checks whether high-security selections can influence low-security
observations in a session type state space.  The key insight is that
non-interference corresponds to the existence of a lattice morphism
from the protocol state space to the two-element information flow
lattice {High >= Low}.

Core idea: partition transitions into high and low security levels.
A protocol satisfies non-interference if the low-observable behavior
is independent of high-level choices.  Formally, for any two execution
paths that agree on low transitions, they must reach the same
low-observable states --- high selections cannot "leak" into the low
view.

Key definitions:
- **High transitions**: transitions labeled with high-security methods
  (typically internal selections that carry secrets).
- **Low transitions**: transitions labeled with low-security methods
  (observable by an attacker or low-clearance participant).
- **Non-interference**: forall paths p1, p2 from the same state,
  if p1 and p2 differ only in high transitions, their low projections
  are identical.
- **Leakage score**: fraction of low-observable states that can
  distinguish between different high-level choices (0 = no leakage,
  1 = full leakage).

This module provides:
    ``classify_transitions(ss, high_labels, low_labels)``
        — partition transitions into high/low/mixed.
    ``information_flow_lattice(ss)``
        — build the 2-element {H >= L} information flow lattice.
    ``check_noninterference(ss, high_labels, low_labels)``
        — verify non-interference property.
    ``leakage_score(ss, high_labels, low_labels)``
        — quantify information leaked (0 = none).
    ``analyze_noninterference(ss, high_labels, low_labels)``
        — full analysis as NonInterferenceResult.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransitionClassification:
    """Classification of transitions into security levels.

    Attributes:
        high: Set of (src, label, tgt) triples classified as high.
        low: Set of (src, label, tgt) triples classified as low.
        mixed: Set of (src, label, tgt) triples with unclassified labels.
        high_labels: The high-security label set used.
        low_labels: The low-security label set used.
    """

    high: frozenset[tuple[int, str, int]]
    low: frozenset[tuple[int, str, int]]
    mixed: frozenset[tuple[int, str, int]]
    high_labels: frozenset[str]
    low_labels: frozenset[str]


@dataclass(frozen=True)
class InformationFlowLattice:
    """The two-element information flow lattice {High >= Low}.

    Attributes:
        high_state: State ID representing the High security level.
        low_state: State ID representing the Low security level.
        mapping: Mapping from protocol state IDs to {high_state, low_state}.
    """

    high_state: int
    low_state: int
    mapping: dict[int, int]


@dataclass(frozen=True)
class NonInterferenceResult:
    """Result of non-interference analysis.

    Attributes:
        is_noninterfering: True iff high choices do not affect low observations.
        classification: Transition classification used.
        flow_lattice: The information flow lattice mapping.
        leakage: Leakage score in [0, 1].
        leaking_states: States where high choices leak into low view.
        witness_paths: Example pair of paths demonstrating interference
                       (None if non-interfering).
        num_high_transitions: Count of high transitions.
        num_low_transitions: Count of low transitions.
    """

    is_noninterfering: bool
    classification: TransitionClassification
    flow_lattice: InformationFlowLattice | None
    leakage: float
    leaking_states: frozenset[int]
    witness_paths: tuple[list[tuple[int, str, int]], list[tuple[int, str, int]]] | None
    num_high_transitions: int
    num_low_transitions: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_adjacency(
    ss: StateSpace,
) -> dict[int, list[tuple[str, int]]]:
    """Build adjacency list: state -> [(label, target)]."""
    adj: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for src, label, tgt in ss.transitions:
        adj[src].append((label, tgt))
    return dict(adj)


def _low_projection(
    path: list[tuple[int, str, int]],
    low_labels: frozenset[str],
) -> list[tuple[int, str, int]]:
    """Project a path to its low-observable transitions only."""
    return [(s, l, t) for s, l, t in path if l in low_labels]


def _low_reachable_states(
    ss: StateSpace,
    state: int,
    high_labels: frozenset[str],
    low_labels: frozenset[str],
    visited: set[int] | None = None,
) -> set[int]:
    """States reachable from `state` after taking any number of high
    transitions, then observing the set of enabled low transitions.

    Returns the set of low-observable "fingerprint" states --- states
    from which the low-enabled methods can be determined.
    """
    if visited is None:
        visited = set()
    if state in visited:
        return set()
    visited.add(state)

    result: set[int] = set()
    # Include the current state as a low-observable point
    result.add(state)

    # Follow high transitions silently
    for src, label, tgt in ss.transitions:
        if src == state and label in high_labels:
            result |= _low_reachable_states(ss, tgt, high_labels, low_labels, visited)

    return result


def _enabled_labels(ss: StateSpace, state: int) -> set[str]:
    """Return the set of transition labels enabled at a state."""
    return {label for src, label, tgt in ss.transitions if src == state}


def _low_enabled(
    ss: StateSpace, state: int, low_labels: frozenset[str]
) -> frozenset[str]:
    """Return the low-security labels enabled at a state."""
    return frozenset(_enabled_labels(ss, state) & low_labels)


def _enumerate_bounded_paths(
    ss: StateSpace,
    start: int,
    *,
    max_depth: int = 20,
    max_paths: int = 500,
) -> list[list[tuple[int, str, int]]]:
    """BFS/DFS enumerate paths from start up to max_depth."""
    paths: list[list[tuple[int, str, int]]] = []
    # (current_state, path_so_far, visited_states)
    stack: list[tuple[int, list[tuple[int, str, int]], set[int]]] = [
        (start, [], {start})
    ]

    while stack and len(paths) < max_paths:
        state, path, visited = stack.pop()
        if len(path) >= max_depth:
            paths.append(path)
            continue

        successors = [(label, tgt) for src, label, tgt in ss.transitions if src == state]
        if not successors:
            paths.append(path)
            continue

        for label, tgt in successors:
            if tgt not in visited:
                stack.append((tgt, path + [(state, label, tgt)], visited | {tgt}))
            elif len(path) > 0:
                # At a cycle, record the path so far
                paths.append(path + [(state, label, tgt)])

    return paths


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def classify_transitions(
    ss: StateSpace,
    high_labels: set[str] | frozenset[str],
    low_labels: set[str] | frozenset[str],
) -> TransitionClassification:
    """Classify each transition in the state space as high, low, or mixed.

    A transition (src, label, tgt) is:
    - **high** if label is in high_labels
    - **low** if label is in low_labels
    - **mixed** if label is in neither (or both)

    Args:
        ss: The state space to classify.
        high_labels: Labels considered high-security.
        low_labels: Labels considered low-security.

    Returns:
        TransitionClassification with the partition.
    """
    h_labels = frozenset(high_labels)
    l_labels = frozenset(low_labels)

    high: set[tuple[int, str, int]] = set()
    low: set[tuple[int, str, int]] = set()
    mixed: set[tuple[int, str, int]] = set()

    for src, label, tgt in ss.transitions:
        in_high = label in h_labels
        in_low = label in l_labels
        if in_high and not in_low:
            high.add((src, label, tgt))
        elif in_low and not in_high:
            low.add((src, label, tgt))
        else:
            mixed.add((src, label, tgt))

    return TransitionClassification(
        high=frozenset(high),
        low=frozenset(low),
        mixed=frozenset(mixed),
        high_labels=h_labels,
        low_labels=l_labels,
    )


def information_flow_lattice(
    ss: StateSpace,
    high_labels: set[str] | frozenset[str] | None = None,
    low_labels: set[str] | frozenset[str] | None = None,
) -> InformationFlowLattice:
    """Build the two-element information flow lattice {High >= Low}.

    Maps each state in the protocol state space to either High or Low
    based on whether high-security transitions are enabled at that state.

    If no labels are provided, selections are treated as high and
    branches as low.

    Args:
        ss: The state space.
        high_labels: Labels considered high-security.
        low_labels: Labels considered low-security.

    Returns:
        InformationFlowLattice with mapping from states to {0, 1}
        where 1 = High and 0 = Low.
    """
    if high_labels is None:
        # Default: selection transitions are high
        h_labels: frozenset[str] = frozenset(
            label for _, label, _ in ss.selection_transitions
        )
    else:
        h_labels = frozenset(high_labels)

    mapping: dict[int, int] = {}
    for state in ss.states:
        enabled = _enabled_labels(ss, state)
        if enabled & h_labels:
            mapping[state] = 1  # High
        else:
            mapping[state] = 0  # Low

    return InformationFlowLattice(
        high_state=1,
        low_state=0,
        mapping=mapping,
    )


def check_noninterference(
    ss: StateSpace,
    high_labels: set[str] | frozenset[str],
    low_labels: set[str] | frozenset[str],
) -> bool:
    """Check whether the state space satisfies non-interference.

    Non-interference holds when high-security choices cannot affect
    low-security observations.  Formally: for every state s, if s
    has multiple high transitions leading to states s1, s2, ...,
    then the low-observable behavior from each si must be identical.

    The check verifies that for each state with high-level choices,
    the set of reachable low-enabled labels is the same regardless
    of which high transition is taken.

    Args:
        ss: The state space to check.
        high_labels: Labels considered high-security.
        low_labels: Labels considered low-security.

    Returns:
        True iff the protocol is non-interfering.
    """
    h_labels = frozenset(high_labels)
    l_labels = frozenset(low_labels)

    for state in ss.states:
        # Find high transitions from this state
        high_successors: list[int] = []
        for src, label, tgt in ss.transitions:
            if src == state and label in h_labels:
                high_successors.append(tgt)

        if len(high_successors) <= 1:
            continue

        # Check that all high-successor states have the same
        # low-observable behavior
        reference_low = _low_view(ss, high_successors[0], h_labels, l_labels)
        for succ in high_successors[1:]:
            succ_low = _low_view(ss, succ, h_labels, l_labels)
            if succ_low != reference_low:
                return False

    return True


def _low_view(
    ss: StateSpace,
    state: int,
    high_labels: frozenset[str],
    low_labels: frozenset[str],
    *,
    depth: int = 10,
) -> tuple[frozenset[str], ...]:
    """Compute the low-observable view from a state.

    Returns a tuple of frozensets representing the sequence of
    low-enabled labels reachable by following low transitions
    up to a bounded depth.
    """
    views: list[frozenset[str]] = []
    current_states = {state}

    for _ in range(depth):
        if not current_states:
            break
        # Collect low-enabled labels across all current states
        low_at_level: set[str] = set()
        next_states: set[int] = set()
        for s in current_states:
            low_at_level |= (_enabled_labels(ss, s) & low_labels)
            # Also follow high transitions silently
            for src, label, tgt in ss.transitions:
                if src == s and label in high_labels:
                    next_states.add(tgt)
            # Follow low transitions to next level
            for src, label, tgt in ss.transitions:
                if src == s and label in low_labels:
                    next_states.add(tgt)

        views.append(frozenset(low_at_level))
        current_states = next_states

    return tuple(views)


def leakage_score(
    ss: StateSpace,
    high_labels: set[str] | frozenset[str],
    low_labels: set[str] | frozenset[str],
) -> float:
    """Quantify information leakage from high to low level.

    Returns a score in [0, 1]:
    - 0.0: no leakage (perfect non-interference)
    - 1.0: full leakage (every high choice is distinguishable at low level)

    The score is computed as the fraction of states with high-level
    choices where those choices produce distinguishable low-observable
    behavior.

    Args:
        ss: The state space to analyze.
        high_labels: Labels considered high-security.
        low_labels: Labels considered low-security.

    Returns:
        Leakage score in [0.0, 1.0].
    """
    h_labels = frozenset(high_labels)
    l_labels = frozenset(low_labels)

    total_high_choice_states = 0
    leaking_states_count = 0

    for state in ss.states:
        high_successors: list[int] = []
        for src, label, tgt in ss.transitions:
            if src == state and label in h_labels:
                high_successors.append(tgt)

        if len(high_successors) <= 1:
            continue

        total_high_choice_states += 1

        # Check if the low views differ
        reference_low = _low_view(ss, high_successors[0], h_labels, l_labels)
        for succ in high_successors[1:]:
            succ_low = _low_view(ss, succ, h_labels, l_labels)
            if succ_low != reference_low:
                leaking_states_count += 1
                break

    if total_high_choice_states == 0:
        return 0.0

    return leaking_states_count / total_high_choice_states


def _find_leaking_states(
    ss: StateSpace,
    high_labels: frozenset[str],
    low_labels: frozenset[str],
) -> frozenset[int]:
    """Find all states where high choices leak into low view."""
    leaking: set[int] = set()

    for state in ss.states:
        high_successors: list[int] = []
        for src, label, tgt in ss.transitions:
            if src == state and label in high_labels:
                high_successors.append(tgt)

        if len(high_successors) <= 1:
            continue

        reference_low = _low_view(ss, high_successors[0], high_labels, low_labels)
        for succ in high_successors[1:]:
            succ_low = _low_view(ss, succ, high_labels, low_labels)
            if succ_low != reference_low:
                leaking.add(state)
                break

    return frozenset(leaking)


def _find_witness(
    ss: StateSpace,
    leaking_state: int,
    high_labels: frozenset[str],
    low_labels: frozenset[str],
) -> tuple[list[tuple[int, str, int]], list[tuple[int, str, int]]] | None:
    """Find two paths from a leaking state that demonstrate interference."""
    high_successors: list[tuple[str, int]] = []
    for src, label, tgt in ss.transitions:
        if src == leaking_state and label in high_labels:
            high_successors.append((label, tgt))

    if len(high_successors) < 2:
        return None

    label1, tgt1 = high_successors[0]
    label2, tgt2 = high_successors[1]

    path1 = [(leaking_state, label1, tgt1)]
    path2 = [(leaking_state, label2, tgt2)]

    return (path1, path2)


def analyze_noninterference(
    ss: StateSpace,
    high_labels: set[str] | frozenset[str],
    low_labels: set[str] | frozenset[str],
) -> NonInterferenceResult:
    """Full non-interference analysis.

    Classifies transitions, checks non-interference, computes leakage,
    and identifies leaking states with witness paths.

    Args:
        ss: The state space to analyze.
        high_labels: Labels considered high-security.
        low_labels: Labels considered low-security.

    Returns:
        NonInterferenceResult with all analysis data.
    """
    h_labels = frozenset(high_labels)
    l_labels = frozenset(low_labels)

    classification = classify_transitions(ss, h_labels, l_labels)
    flow = information_flow_lattice(ss, h_labels, l_labels)
    is_ni = check_noninterference(ss, h_labels, l_labels)
    leak = leakage_score(ss, h_labels, l_labels)
    leaking = _find_leaking_states(ss, h_labels, l_labels)

    witness = None
    if leaking:
        first_leaker = min(leaking)
        witness = _find_witness(ss, first_leaker, h_labels, l_labels)

    return NonInterferenceResult(
        is_noninterfering=is_ni,
        classification=classification,
        flow_lattice=flow,
        leakage=leak,
        leaking_states=leaking,
        witness_paths=witness,
        num_high_transitions=len(classification.high),
        num_low_transitions=len(classification.low),
    )
