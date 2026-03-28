"""Orbital session types: five interpretations of orbits (Step 60g).

Applies five distinct interpretations of "orbital" to session type state
spaces, drawing from quantum mechanics, group theory, celestial mechanics,
probability theory, and protocol evolution:

1. **Quantum Orbital Analogy**: BFS depth levels as electron shells with
   principal quantum numbers, angular momentum from branching, and Pauli
   exclusion violations from parallel composition.

2. **Group-Theoretic Orbits**: Equivalence classes of states under
   approximate automorphism (same degree profile and depth).  The quotient
   by orbits measures symmetry.

3. **Planetary Orbit Dynamics**: Cycle detection via Tarjan's SCC,
   escape velocity as BFS distance to bottom, Lagrange equilibrium
   points, and gravitational binding energy.

4. **Probability Cloud**: Steady-state probability distribution from a
   uniform random walk, Shannon entropy, expected depth, and cloud radius
   (standard deviation of depth).

5. **Type Space Orbits**: Track protocol evolution across a sequence of
   session type versions, detecting convergence, expansion, and
   periodicity in the (states, transitions, density) trajectory.

This module provides:
    ``quantum_orbitals(ss)``      -- shell structure and exclusion violations.
    ``compute_orbits(ss)``        -- group-theoretic orbit decomposition.
    ``planetary_orbits(ss)``      -- cycle dynamics and binding energy.
    ``probability_cloud(ss)``     -- random-walk probability distribution.
    ``type_space_orbit(types)``   -- protocol evolution trajectory.
    ``analyze_orbital(ss)``       -- combined four-analysis result.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantumOrbitalResult:
    """Result of quantum orbital analogy analysis.

    Attributes:
        energy_levels: Mapping from BFS depth to number of states at that depth.
        shell_labels: Mapping from BFS depth to shell name (s, p, d, f, g cycling).
        max_principal_quantum: Maximum BFS depth (principal quantum number).
        angular_momentum: Mapping from depth to average outgoing transitions.
        exclusion_violations: States reachable by two independent parallel paths.
    """

    energy_levels: dict[int, int]
    shell_labels: dict[int, str]
    max_principal_quantum: int
    angular_momentum: dict[int, int]
    exclusion_violations: int


@dataclass(frozen=True)
class GroupOrbitResult:
    """Result of group-theoretic orbit analysis.

    Attributes:
        orbits: Equivalence classes under approximate automorphism.
        orbit_count: Number of distinct orbits.
        trivial_orbits: Number of orbits of size 1 (fixed points).
        max_orbit_size: Size of the largest orbit.
        symmetry_quotient_states: Number of states after quotienting by orbits.
        symmetry_ratio: |states| / |orbits| -- higher means more symmetric.
    """

    orbits: tuple[frozenset[int], ...]
    orbit_count: int
    trivial_orbits: int
    max_orbit_size: int
    symmetry_quotient_states: int
    symmetry_ratio: float


@dataclass(frozen=True)
class PlanetaryOrbitResult:
    """Result of planetary orbit dynamics analysis.

    Attributes:
        orbital_periods: State to cycle length (0 if not in a cycle).
        escape_velocities: State to minimum transitions to reach bottom.
        bound_states: States in non-trivial SCCs (gravitationally bound).
        free_states: States not in any cycle.
        lagrange_points: States in SCCs reachable from all others in the same SCC.
        binding_energy: Fraction of states that are bound.
    """

    orbital_periods: dict[int, int]
    escape_velocities: dict[int, int]
    bound_states: frozenset[int]
    free_states: frozenset[int]
    lagrange_points: frozenset[int]
    binding_energy: float


@dataclass(frozen=True)
class ProbabilityCloudResult:
    """Result of probability cloud analysis.

    Attributes:
        state_probabilities: Steady-state probability per state (uniform random walk).
        entropy: Shannon entropy of the probability distribution.
        expected_depth: Expected BFS depth under the distribution.
        cloud_radius: Standard deviation of depth under the distribution.
        measurement_entropy: Per-state entropy of outgoing transition choice.
    """

    state_probabilities: dict[int, float]
    entropy: float
    expected_depth: float
    cloud_radius: float
    measurement_entropy: dict[int, float]


@dataclass(frozen=True)
class TypeSpaceOrbitResult:
    """Result of type space orbit (protocol evolution) analysis.

    Attributes:
        trajectory: Sequence of (states, transitions, density) per version.
        is_convergent: True if last two entries have the same state count.
        is_expanding: True if state count strictly increases across versions.
        drift_rate: Average change in state count per version step.
        orbital_period: Period of state-count cycle (0 if none).
    """

    trajectory: tuple[tuple[int, int, float], ...]
    is_convergent: bool
    is_expanding: bool
    drift_rate: float
    orbital_period: int


@dataclass(frozen=True)
class OrbitalResult:
    """Combined result of all four state-space orbital analyses.

    Attributes:
        quantum: Quantum orbital analogy result.
        group: Group-theoretic orbit result.
        planetary: Planetary orbit dynamics result.
        cloud: Probability cloud result.
        classification: "bound" if binding_energy > 0.5, "free" if < 0.2, else "mixed".
    """

    quantum: QuantumOrbitalResult
    group: GroupOrbitResult
    planetary: PlanetaryOrbitResult
    cloud: ProbabilityCloudResult
    classification: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SHELL_NAMES = ("s", "p", "d", "f", "g")


def _adjacency(ss: StateSpace) -> dict[int, list[tuple[str, int]]]:
    """Build adjacency list: state -> [(label, target)]."""
    adj: dict[int, list[tuple[str, int]]] = {s: [] for s in ss.states}
    for src, label, tgt in ss.transitions:
        adj[src].append((label, tgt))
    return adj


def _bfs_depths(ss: StateSpace) -> dict[int, int]:
    """Compute BFS depth from top for each reachable state."""
    depths: dict[int, int] = {}
    if not ss.states:
        return depths
    adj = _adjacency(ss)
    queue: deque[int] = deque([ss.top])
    depths[ss.top] = 0
    while queue:
        s = queue.popleft()
        d = depths[s]
        for _label, tgt in adj.get(s, []):
            if tgt not in depths:
                depths[tgt] = d + 1
                queue.append(tgt)
    return depths


def _reverse_adjacency(ss: StateSpace) -> dict[int, list[int]]:
    """Build reverse adjacency list: state -> [predecessor states]."""
    rev: dict[int, list[int]] = {s: [] for s in ss.states}
    for src, _label, tgt in ss.transitions:
        rev[tgt].append(src)
    return rev


def _bfs_distance_to_bottom(ss: StateSpace) -> dict[int, int]:
    """Compute minimum transition count from each state to bottom (reverse BFS)."""
    rev = _reverse_adjacency(ss)
    dist: dict[int, int] = {ss.bottom: 0}
    queue: deque[int] = deque([ss.bottom])
    while queue:
        s = queue.popleft()
        d = dist[s]
        for pred in rev.get(s, []):
            if pred not in dist:
                dist[pred] = d + 1
                queue.append(pred)
    return dist


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
# Interpretation 1: Quantum Orbital Analogy
# ---------------------------------------------------------------------------


def quantum_orbitals(ss: StateSpace) -> QuantumOrbitalResult:
    """Compute quantum orbital analogy for a session type state space.

    BFS depth levels map to electron shells (depth 0 = "1s", depth 1 = "2p",
    depth 2 = "3d", etc., cycling through s, p, d, f, g labels).  Angular
    momentum at each level is the average number of outgoing transitions for
    states at that depth.  Exclusion violations count states reachable from
    top via two independent transition sequences in product state spaces.
    """
    depths = _bfs_depths(ss)
    adj = _adjacency(ss)

    # Energy levels: depth -> count of states at that depth
    energy_levels: dict[int, int] = {}
    for _s, d in depths.items():
        energy_levels[d] = energy_levels.get(d, 0) + 1

    max_pq = max(energy_levels.keys()) if energy_levels else 0

    # Shell labels: depth -> shell name cycling through s, p, d, f, g
    shell_labels: dict[int, str] = {}
    for d in energy_levels:
        principal = d + 1
        sub = _SHELL_NAMES[d % len(_SHELL_NAMES)]
        shell_labels[d] = f"{principal}{sub}"

    # Angular momentum: depth -> average outgoing transitions
    angular_momentum: dict[int, int] = {}
    depth_states: dict[int, list[int]] = {}
    for s, d in depths.items():
        depth_states.setdefault(d, []).append(s)
    for d, states_at_d in depth_states.items():
        total_out = sum(len(adj.get(s, [])) for s in states_at_d)
        angular_momentum[d] = total_out // len(states_at_d) if states_at_d else 0

    # Exclusion violations: states reachable via two independent parallel paths
    exclusion_violations = 0
    if ss.product_coords is not None and ss.product_factors is not None:
        # For product state spaces, count states where both factors advanced
        # independently to the same reachable state via different paths
        # Approximate: count states reachable from top via two different
        # first transitions that converge to the same state
        first_transitions = adj.get(ss.top, [])
        if len(first_transitions) >= 2:
            reachable_sets: list[set[int]] = []
            for _label, tgt in first_transitions:
                reached: set[int] = set()
                q: deque[int] = deque([tgt])
                while q:
                    s = q.popleft()
                    if s in reached:
                        continue
                    reached.add(s)
                    for _l, t in adj.get(s, []):
                        if t not in reached:
                            q.append(t)
                reachable_sets.append(reached)
            if len(reachable_sets) >= 2:
                common = reachable_sets[0]
                for rs in reachable_sets[1:]:
                    common = common & rs
                exclusion_violations = len(common)

    return QuantumOrbitalResult(
        energy_levels=energy_levels,
        shell_labels=shell_labels,
        max_principal_quantum=max_pq,
        angular_momentum=angular_momentum,
        exclusion_violations=exclusion_violations,
    )


# ---------------------------------------------------------------------------
# Interpretation 2: Group-Theoretic Orbits
# ---------------------------------------------------------------------------


def compute_orbits(ss: StateSpace) -> GroupOrbitResult:
    """Find approximate automorphism orbits of states.

    Two states are orbit-equivalent if they have the same in-degree,
    out-degree, same set of outgoing label names, and same BFS depth
    from top.  This is an approximation of true automorphism (which
    requires expensive graph isomorphism).

    Returns a GroupOrbitResult with equivalence classes and symmetry metrics.
    """
    if not ss.states:
        return GroupOrbitResult(
            orbits=(),
            orbit_count=0,
            trivial_orbits=0,
            max_orbit_size=0,
            symmetry_quotient_states=0,
            symmetry_ratio=0.0,
        )

    depths = _bfs_depths(ss)
    adj = _adjacency(ss)

    # Compute in-degree
    in_degree: dict[int, int] = {s: 0 for s in ss.states}
    for _src, _label, tgt in ss.transitions:
        in_degree[tgt] = in_degree.get(tgt, 0) + 1

    # Signature: (depth, in_degree, out_degree, frozenset of outgoing labels)
    signatures: dict[int, tuple[int, int, int, frozenset[str]]] = {}
    for s in ss.states:
        d = depths.get(s, -1)
        out = adj.get(s, [])
        out_deg = len(out)
        labels = frozenset(label for label, _tgt in out)
        in_deg = in_degree.get(s, 0)
        signatures[s] = (d, in_deg, out_deg, labels)

    # Group by signature
    sig_groups: dict[tuple[int, int, int, frozenset[str]], set[int]] = {}
    for s, sig in signatures.items():
        sig_groups.setdefault(sig, set()).add(s)

    orbits = tuple(frozenset(group) for group in sig_groups.values())
    orbit_count = len(orbits)
    trivial = sum(1 for o in orbits if len(o) == 1)
    max_size = max(len(o) for o in orbits) if orbits else 0
    n_states = len(ss.states)
    ratio = n_states / orbit_count if orbit_count > 0 else 0.0

    return GroupOrbitResult(
        orbits=orbits,
        orbit_count=orbit_count,
        trivial_orbits=trivial,
        max_orbit_size=max_size,
        symmetry_quotient_states=orbit_count,
        symmetry_ratio=round(ratio, 6),
    )


# ---------------------------------------------------------------------------
# Interpretation 3: Planetary Orbit Dynamics
# ---------------------------------------------------------------------------


def planetary_orbits(ss: StateSpace) -> PlanetaryOrbitResult:
    """Analyse planetary orbit dynamics of a session type state space.

    Uses Tarjan's SCC to find cycles.  States in non-trivial SCCs (size > 1)
    are "bound" with orbital period equal to SCC size.  Escape velocity is
    the BFS distance to bottom.  Lagrange points are states in an SCC that
    have transitions to all other states in the same SCC.
    """
    sccs = _iterative_tarjan(ss)
    adj = _adjacency(ss)
    dist_to_bottom = _bfs_distance_to_bottom(ss)

    # Map state -> its SCC (only non-trivial SCCs)
    state_to_scc: dict[int, frozenset[int]] = {}
    non_trivial_sccs: list[frozenset[int]] = []
    for scc in sccs:
        if len(scc) > 1:
            non_trivial_sccs.append(scc)
            for s in scc:
                state_to_scc[s] = scc
        elif len(scc) == 1:
            # Check for self-loop
            s = next(iter(scc))
            for _label, tgt in adj.get(s, []):
                if tgt == s:
                    non_trivial_sccs.append(scc)
                    state_to_scc[s] = scc
                    break

    # Orbital periods
    orbital_periods: dict[int, int] = {}
    for s in ss.states:
        if s in state_to_scc:
            orbital_periods[s] = len(state_to_scc[s])
        else:
            orbital_periods[s] = 0

    # Escape velocities
    escape_velocities: dict[int, int] = {}
    for s in ss.states:
        escape_velocities[s] = dist_to_bottom.get(s, -1)

    # Bound and free states
    bound = frozenset(state_to_scc.keys())
    free = frozenset(ss.states - bound)

    # Lagrange points: states in SCC with transitions to all other members
    lagrange: set[int] = set()
    for scc in non_trivial_sccs:
        for s in scc:
            targets = {tgt for _label, tgt in adj.get(s, []) if tgt in scc and tgt != s}
            others = scc - {s}
            if others and targets >= others:
                lagrange.add(s)

    binding = len(bound) / len(ss.states) if ss.states else 0.0

    return PlanetaryOrbitResult(
        orbital_periods=orbital_periods,
        escape_velocities=escape_velocities,
        bound_states=bound,
        free_states=free,
        lagrange_points=frozenset(lagrange),
        binding_energy=round(binding, 6),
    )


# ---------------------------------------------------------------------------
# Interpretation 4: Probability Cloud
# ---------------------------------------------------------------------------


def probability_cloud(ss: StateSpace) -> ProbabilityCloudResult:
    """Compute probability cloud for a uniform random walk on the state space.

    For acyclic state spaces, computes exact path probabilities from top.
    At each state, transitions are chosen uniformly.  For cyclic state spaces,
    uses simulation (10000 steps) to approximate the stationary distribution.
    """
    adj = _adjacency(ss)
    depths = _bfs_depths(ss)

    if not ss.states:
        return ProbabilityCloudResult(
            state_probabilities={},
            entropy=0.0,
            expected_depth=0.0,
            cloud_radius=0.0,
            measurement_entropy={},
        )

    # Detect cycles via SCC
    sccs = _iterative_tarjan(ss)
    has_cycles = any(len(scc) > 1 for scc in sccs)
    # Also check self-loops
    if not has_cycles:
        for src, _label, tgt in ss.transitions:
            if src == tgt:
                has_cycles = True
                break

    if has_cycles:
        probs = _simulate_random_walk(ss, adj, steps=10000)
    else:
        probs = _exact_path_probabilities(ss, adj)

    # Shannon entropy of distribution
    entropy = 0.0
    for p in probs.values():
        if p > 0:
            entropy -= p * math.log2(p)

    # Expected depth
    expected_depth = 0.0
    for s, p in probs.items():
        expected_depth += p * depths.get(s, 0)

    # Cloud radius (std dev of depth)
    variance = 0.0
    for s, p in probs.items():
        d = depths.get(s, 0)
        variance += p * (d - expected_depth) ** 2
    cloud_radius = math.sqrt(variance)

    # Measurement entropy: per-state entropy of outgoing choice
    measurement_entropy: dict[int, float] = {}
    for s in ss.states:
        out = adj.get(s, [])
        n = len(out)
        if n <= 1:
            measurement_entropy[s] = 0.0
        else:
            # Uniform distribution over n choices
            measurement_entropy[s] = round(math.log2(n), 6)

    return ProbabilityCloudResult(
        state_probabilities={s: round(p, 6) for s, p in probs.items()},
        entropy=round(entropy, 6),
        expected_depth=round(expected_depth, 6),
        cloud_radius=round(cloud_radius, 6),
        measurement_entropy=measurement_entropy,
    )


def _exact_path_probabilities(
    ss: StateSpace, adj: dict[int, list[tuple[str, int]]],
) -> dict[int, float]:
    """Exact path probabilities for acyclic state spaces (BFS from top).

    Each state accumulates probability mass from all paths from top.
    At each state, mass is split uniformly among outgoing transitions.
    The result is a visit-frequency distribution (may sum > 1 for DAGs
    where paths converge), so we normalise at the end.
    """
    probs: dict[int, float] = {s: 0.0 for s in ss.states}
    probs[ss.top] = 1.0

    # Topological order via BFS
    in_count: dict[int, int] = {s: 0 for s in ss.states}
    for _src, _label, tgt in ss.transitions:
        in_count[tgt] = in_count.get(tgt, 0) + 1

    queue: deque[int] = deque()
    for s in ss.states:
        if in_count[s] == 0:
            queue.append(s)

    visited: set[int] = set()
    order: list[int] = []
    while queue:
        s = queue.popleft()
        if s in visited:
            continue
        visited.add(s)
        order.append(s)
        for _label, tgt in adj.get(s, []):
            in_count[tgt] -= 1
            if in_count[tgt] == 0:
                queue.append(tgt)

    # Propagate probabilities in topological order
    for s in order:
        out = adj.get(s, [])
        if not out:
            continue
        share = probs[s] / len(out)
        for _label, tgt in out:
            probs[tgt] += share

    # Normalise
    total = sum(probs.values())
    if total > 0:
        probs = {s: p / total for s, p in probs.items()}
    return probs


def _simulate_random_walk(
    ss: StateSpace,
    adj: dict[int, list[tuple[str, int]]],
    *,
    steps: int = 10000,
) -> dict[int, float]:
    """Simulate a uniform random walk to approximate visit frequencies.

    Uses a deterministic pseudo-random strategy (round-robin among
    outgoing transitions) for reproducibility without importing random.
    """
    counts: dict[int, int] = {s: 0 for s in ss.states}
    current = ss.top
    counter = 0

    for _ in range(steps):
        counts[current] = counts.get(current, 0) + 1
        out = adj.get(current, [])
        if not out:
            # Restart from top if stuck
            current = ss.top
            continue
        # Deterministic round-robin for reproducibility
        idx = counter % len(out)
        counter += 1
        _label, tgt = out[idx]
        current = tgt

    total = sum(counts.values())
    if total == 0:
        return {s: 0.0 for s in ss.states}
    return {s: c / total for s, c in counts.items()}


# ---------------------------------------------------------------------------
# Interpretation 5: Type Space Orbits (Protocol Evolution)
# ---------------------------------------------------------------------------


def type_space_orbit(type_strings: list[str]) -> TypeSpaceOrbitResult:
    """Track protocol evolution across a sequence of session type versions.

    Parses each type string, builds its state space, and records the
    trajectory of (states, transitions, density) across versions.  Detects
    convergence (stable state count), expansion (strictly increasing), and
    periodicity in the state-count sequence.
    """
    from reticulate import build_statespace, parse

    trajectory: list[tuple[int, int, float]] = []
    for ts in type_strings:
        st = parse(ts)
        ss = build_statespace(st)
        n_states = len(ss.states)
        n_trans = len(ss.transitions)
        density = n_trans / (n_states * (n_states - 1)) if n_states > 1 else 0.0
        trajectory.append((n_states, n_trans, round(density, 6)))

    # Convergent: last two entries have the same state count
    is_convergent = False
    if len(trajectory) >= 2:
        is_convergent = trajectory[-1][0] == trajectory[-2][0]

    # Expanding: state count strictly increases
    is_expanding = False
    if len(trajectory) >= 2:
        is_expanding = all(
            trajectory[i + 1][0] > trajectory[i][0]
            for i in range(len(trajectory) - 1)
        )

    # Drift rate: average change in state count per step
    drift_rate = 0.0
    if len(trajectory) >= 2:
        changes = [
            trajectory[i + 1][0] - trajectory[i][0]
            for i in range(len(trajectory) - 1)
        ]
        drift_rate = sum(changes) / len(changes)

    # Orbital period: detect repetition in state counts
    state_counts = [t[0] for t in trajectory]
    orbital_period = _detect_period(state_counts)

    return TypeSpaceOrbitResult(
        trajectory=tuple(trajectory),
        is_convergent=is_convergent,
        is_expanding=is_expanding,
        drift_rate=round(drift_rate, 6),
        orbital_period=orbital_period,
    )


def _detect_period(seq: list[int]) -> int:
    """Detect the shortest repeating period in a sequence.

    Returns the period length, or 0 if no repetition is found.
    """
    n = len(seq)
    if n < 2:
        return 0
    for period in range(1, n // 2 + 1):
        is_periodic = True
        for i in range(period, n):
            if seq[i] != seq[i % period]:
                is_periodic = False
                break
        if is_periodic:
            return period
    return 0


# ---------------------------------------------------------------------------
# Combined Analysis
# ---------------------------------------------------------------------------


def analyze_orbital(ss: StateSpace) -> OrbitalResult:
    """Run all four state-space orbital analyses and return combined result.

    Skips type_space_orbit (which requires multiple type strings).
    Classifies as "bound" if binding_energy > 0.5, "free" if < 0.2,
    "mixed" otherwise.
    """
    q = quantum_orbitals(ss)
    g = compute_orbits(ss)
    p = planetary_orbits(ss)
    c = probability_cloud(ss)

    if p.binding_energy > 0.5:
        classification = "bound"
    elif p.binding_energy < 0.2:
        classification = "free"
    else:
        classification = "mixed"

    return OrbitalResult(
        quantum=q,
        group=g,
        planetary=p,
        cloud=c,
        classification=classification,
    )
