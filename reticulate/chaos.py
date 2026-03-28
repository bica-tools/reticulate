"""Chaotic session types: Lyapunov exponents, attractors, bifurcation (Step 60d).

Applies chaos theory concepts to session type state spaces, measuring how
sensitive a protocol is to initial choices, whether it has attracting
behaviour, and how its structure changes under recursion unfolding.

The key quantities are:

- **Lyapunov exponent**: measures divergence of nearby traces.  For each
  branching state (with k > 1 outgoing transitions), pick pairs of
  transitions and measure how quickly the resulting paths diverge.  The
  exponent is the average log2(divergence_rate) over all branching states.

- **Sensitivity**: fraction of branching states where different choices
  lead to different reachable bottom-sets.  High sensitivity means small
  perturbations in choice propagate to different outcomes.

- **Attractors**: strongly connected components with no outgoing edges
  (except to the bottom state).  These are the "sinks" of the protocol
  — once entered, the protocol stays trapped.

- **Orbit analysis**: deterministic path following (lexicographically
  first label) from a given state, detecting cycles.

- **Bifurcation analysis**: unfold a recursive type to increasing depths
  and track how state/transition counts change — the "bifurcation
  diagram" of the protocol.

- **Classification**: fixed_point, periodic, quasi_periodic, or chaotic
  based on attractor structure, sensitivity, and Lyapunov exponent.

This module provides:
    ``lyapunov_exponent(ss)``        — average trace divergence rate.
    ``sensitivity(ss)``              — fraction of sensitive branching states.
    ``detect_attractors(ss)``        — sink SCCs (attractor sets).
    ``orbit_analysis(ss, state)``    — deterministic path from a state.
    ``bifurcation_analysis(s, d)``   — recursion unfolding bifurcation data.
    ``classify_dynamics(ss)``        — full ChaosResult.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChaosResult:
    """Result of chaotic dynamics analysis.

    Attributes:
        lyapunov_exponent: Average log2(divergence_rate) over branching states.
        is_chaotic: True if classified as chaotic.
        is_stable: True if classified as fixed_point or periodic.
        bifurcation_points: Recursion depths where structure changes.
        max_orbit_length: Longest deterministic orbit before cycle/end.
        sensitivity: Fraction of branching states with divergent outcomes.
        attractor_size: Total number of states in all attractors.
        classification: One of fixed_point, periodic, quasi_periodic, chaotic.
    """

    lyapunov_exponent: float
    is_chaotic: bool
    is_stable: bool
    bifurcation_points: tuple[int, ...]
    max_orbit_length: int
    sensitivity: float
    attractor_size: int
    classification: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _adjacency(ss: StateSpace) -> dict[int, list[tuple[str, int]]]:
    """Build adjacency list: state -> [(label, target)]."""
    adj: dict[int, list[tuple[str, int]]] = {s: [] for s in ss.states}
    for src, label, tgt in ss.transitions:
        adj[src].append((label, tgt))
    return adj


def _reachable_set(
    adj: dict[int, list[tuple[str, int]]], start: int, bottom: int,
) -> frozenset[int]:
    """States reachable from *start* via any path (BFS)."""
    visited: set[int] = set()
    stack = [start]
    while stack:
        s = stack.pop()
        if s in visited:
            continue
        visited.add(s)
        for _label, tgt in adj.get(s, []):
            if tgt not in visited:
                stack.append(tgt)
    return frozenset(visited)


def _reachable_bottom_paths(
    adj: dict[int, list[tuple[str, int]]],
    start: int,
    bottom: int,
    *,
    max_depth: int = 50,
) -> frozenset[int]:
    """Set of states reachable on paths from *start* to *bottom*.

    Returns the set of all states encountered on any acyclic path from
    *start* towards *bottom*, bounded by *max_depth*.
    """
    result: set[int] = set()
    stack: list[tuple[int, frozenset[int]]] = [(start, frozenset())]
    while stack:
        s, visited = stack.pop()
        result.add(s)
        if s == bottom:
            continue
        for _label, tgt in adj.get(s, []):
            if tgt not in visited and len(visited) < max_depth:
                stack.append((tgt, visited | {s}))
    return frozenset(result)


def _iterative_tarjan(ss: StateSpace) -> list[frozenset[int]]:
    """Compute SCCs using iterative Tarjan's algorithm."""
    index_counter = [0]
    stack: list[int] = []
    on_stack: set[int] = set()
    index: dict[int, int] = {}
    lowlink: dict[int, int] = {}
    result: list[frozenset[int]] = []

    adj: dict[int, list[int]] = {s: [] for s in ss.states}
    for src, _label, tgt in ss.transitions:
        adj[src].append(tgt)

    for start in ss.states:
        if start in index:
            continue
        # Iterative DFS
        work_stack: list[tuple[int, int]] = [(start, 0)]
        call_stack: list[tuple[int, list[int]]] = []

        while work_stack:
            v, ci = work_stack.pop()
            if ci == 0:
                index[v] = index_counter[0]
                lowlink[v] = index_counter[0]
                index_counter[0] += 1
                stack.append(v)
                on_stack.add(v)

            neighbors = adj.get(v, [])
            recurse = False
            for i in range(ci, len(neighbors)):
                w = neighbors[i]
                if w not in index:
                    call_stack.append((v, neighbors))
                    work_stack.append((v, i + 1))
                    work_stack.append((w, 0))
                    recurse = True
                    break
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], lowlink[w])

            if not recurse:
                if lowlink[v] == index[v]:
                    scc: set[int] = set()
                    while True:
                        w = stack.pop()
                        on_stack.discard(w)
                        scc.add(w)
                        if w == v:
                            break
                    result.append(frozenset(scc))
                if call_stack:
                    caller, _ = call_stack.pop()
                    lowlink[caller] = min(lowlink[caller], lowlink[v])

    return result


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def lyapunov_exponent(ss: StateSpace) -> float:
    """Measure divergence of nearby traces.

    For each state with k > 1 outgoing transitions, pick pairs of
    transitions and measure how many steps until the resulting paths
    diverge to different states.  The Lyapunov exponent is the average
    log2(divergence_rate) over all branching states.

    Returns 0.0 for linear (non-branching) state spaces.
    """
    adj = _adjacency(ss)
    exponents: list[float] = []

    for state in ss.states:
        out = adj.get(state, [])
        if len(out) <= 1:
            continue

        # For each pair of outgoing transitions, measure divergence
        targets = [tgt for _label, tgt in out]
        pair_count = 0
        divergence_sum = 0.0

        for i in range(len(targets)):
            for j in range(i + 1, len(targets)):
                t1, t2 = targets[i], targets[j]
                if t1 == t2:
                    # Same target — no divergence from this pair
                    continue
                # Paths already diverge at step 1
                # Measure how many distinct states reachable from each
                r1 = _reachable_set(adj, t1, ss.bottom)
                r2 = _reachable_set(adj, t2, ss.bottom)
                # Divergence rate: fraction of states not shared
                union = r1 | r2
                if not union:
                    continue
                intersection = r1 & r2
                divergence = 1.0 - len(intersection) / len(union)
                if divergence > 0:
                    divergence_sum += math.log2(1.0 + divergence)
                pair_count += 1

        if pair_count > 0:
            exponents.append(divergence_sum / pair_count)

    if not exponents:
        return 0.0
    return sum(exponents) / len(exponents)


def sensitivity(ss: StateSpace) -> float:
    """Fraction of branching states where different choices lead to different outcomes.

    For each state with > 1 outgoing transitions, check if the reachable
    sets from different transitions differ.  Return the fraction of such
    "sensitive" states among all branching states.

    Returns 0.0 for non-branching state spaces.
    """
    adj = _adjacency(ss)
    branching_count = 0
    sensitive_count = 0

    for state in ss.states:
        out = adj.get(state, [])
        if len(out) <= 1:
            continue
        branching_count += 1

        # Check if any pair of transitions leads to different reachable sets
        targets = [tgt for _label, tgt in out]
        reachable_sets: list[frozenset[int]] = []
        for tgt in targets:
            reachable_sets.append(
                _reachable_bottom_paths(adj, tgt, ss.bottom)
            )

        # If any two reachable sets differ, this state is sensitive
        if len(set(reachable_sets)) > 1:
            sensitive_count += 1

    if branching_count == 0:
        return 0.0
    return sensitive_count / branching_count


def detect_attractors(ss: StateSpace) -> list[frozenset[int]]:
    """Find attractors: SCCs with no outgoing edges to other SCCs (except bottom).

    An attractor is a strongly connected component where all outgoing
    transitions either stay within the SCC or go to the bottom state.
    These represent protocol states from which the protocol cannot escape.

    Returns a list of frozensets, each containing the state IDs in one
    attractor.  Single-state SCCs without self-loops are excluded (they
    are transient states, not attractors).
    """
    sccs = _iterative_tarjan(ss)

    # Build SCC membership map
    state_to_scc: dict[int, int] = {}
    for idx, scc in enumerate(sccs):
        for s in scc:
            state_to_scc[s] = idx

    # Check for self-loops
    has_self_loop: set[int] = set()
    for src, _label, tgt in ss.transitions:
        if src == tgt:
            has_self_loop.add(src)

    attractors: list[frozenset[int]] = []
    for idx, scc in enumerate(sccs):
        # Skip trivial SCCs (single state, no self-loop)
        if len(scc) == 1:
            state = next(iter(scc))
            if state not in has_self_loop:
                continue

        # Check: all outgoing transitions stay in SCC or go to bottom
        is_sink = True
        for src, _label, tgt in ss.transitions:
            if src in scc and tgt not in scc and tgt != ss.bottom:
                is_sink = False
                break

        if is_sink:
            attractors.append(scc)

    return attractors


def orbit_analysis(
    ss: StateSpace, state: int, max_steps: int = 100,
) -> list[int]:
    """Follow a deterministic path from *state*.

    Always takes the lexicographically first transition label at each
    step.  Returns the sequence of visited states, stopping when a
    cycle is detected or *max_steps* is reached.
    """
    adj = _adjacency(ss)
    path: list[int] = [state]
    visited: set[int] = {state}

    current = state
    for _ in range(max_steps):
        out = adj.get(current, [])
        if not out:
            break
        # Sort by label, take first
        out_sorted = sorted(out, key=lambda x: x[0])
        _label, tgt = out_sorted[0]
        path.append(tgt)
        if tgt in visited:
            # Cycle detected — include the repeated state and stop
            break
        visited.add(tgt)
        current = tgt

    return path


def bifurcation_analysis(
    type_str: str, max_depth: int = 5,
) -> list[tuple[int, int, int]]:
    """Unfold a recursive type to increasing depths, tracking state space growth.

    For each depth d in 1..max_depth, unfolds the recursive type d times,
    builds the state space, and records (depth, num_states, num_transitions).

    Returns a list of (depth, states, transitions) tuples.
    """
    from reticulate import parse, build_statespace
    from reticulate.recursion import unfold_depth

    ast = parse(type_str)
    result: list[tuple[int, int, int]] = []

    for depth in range(1, max_depth + 1):
        unfolded = unfold_depth(ast, depth)
        ss = build_statespace(unfolded)
        result.append((depth, len(ss.states), len(ss.transitions)))

    return result


def classify_dynamics(ss: StateSpace) -> ChaosResult:
    """Full chaotic dynamics analysis of a state space.

    Classification rules:
    - ``fixed_point``: single state (End type).
    - ``periodic``: has attractors with cycles (SCCs with > 1 state).
    - ``quasi_periodic``: has attractors AND sensitivity > 0.5.
    - ``chaotic``: sensitivity > 0.7 AND lyapunov > 1.0.

    Returns a ChaosResult with all computed metrics.
    """
    le = lyapunov_exponent(ss)
    sens = sensitivity(ss)
    attractors = detect_attractors(ss)
    attractor_sz = sum(len(a) for a in attractors)

    # Max orbit length: try from top
    orbit = orbit_analysis(ss, ss.top)
    max_orbit = len(orbit)

    # Also try from each state to find the longest orbit
    for state in ss.states:
        o = orbit_analysis(ss, state)
        if len(o) > max_orbit:
            max_orbit = len(o)

    # Bifurcation points: not applicable without a type string,
    # so leave empty for state-space-only analysis
    bifurcation_pts: tuple[int, ...] = ()

    # Classify
    has_cycle_attractors = any(len(a) > 1 for a in attractors)

    if len(ss.states) <= 1:
        classification = "fixed_point"
    elif sens > 0.7 and le > 1.0:
        classification = "chaotic"
    elif has_cycle_attractors and sens > 0.5:
        classification = "quasi_periodic"
    elif has_cycle_attractors:
        classification = "periodic"
    else:
        classification = "fixed_point"

    return ChaosResult(
        lyapunov_exponent=round(le, 6),
        is_chaotic=(classification == "chaotic"),
        is_stable=(classification in ("fixed_point", "periodic")),
        bifurcation_points=bifurcation_pts,
        max_orbit_length=max_orbit,
        sensitivity=round(sens, 6),
        attractor_size=attractor_sz,
        classification=classification,
    )
