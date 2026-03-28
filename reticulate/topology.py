"""Topological session types: simplicial invariants (Step 60a).

Applies topological invariants to session type state spaces by treating
the state space as a simplicial complex (1-skeleton: graph).  The key
invariants are the Euler characteristic and Betti numbers, which capture
the shape of the protocol:

    - **Euler characteristic** (V - E): distinguishes trees from cyclic protocols.
    - **Betti numbers** (b0, b1): connected components and independent cycles.
    - **Cycle rank** (= b1): counts independent loops from recursion.
    - **Edge density**: ratio of actual to maximal edges (protocol complexity).
    - **Planarity**: heuristic via Euler's inequality E <= 3V - 6.

For topology, we work with the *undirected* underlying graph: directed
edges (a, m, b) and (b, m', a) collapse to a single undirected edge {a, b}.

This module provides:
    ``euler_characteristic(ss)`` — V - E for the undirected graph.
    ``betti_numbers(ss)`` — (b0, b1) connected components and cycles.
    ``cycle_rank(ss)`` — number of independent cycles (= b1).
    ``is_tree(ss)`` — True if acyclic (b1 = 0).
    ``edge_density(ss)`` — |E| / max possible edges.
    ``classify_topology(ss)`` — full TopologicalResult.
    ``topological_distance(ss1, ss2)`` — simple metric between state spaces.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TopologicalResult:
    """Result of topological analysis.

    Attributes:
        euler_characteristic: V - E for the undirected underlying graph.
        betti_0: Number of connected components.
        betti_1: Number of independent cycles (first Betti number).
        genus: Genus of the surface embedding (= b1 for connected graphs).
        is_tree: True if the graph is acyclic (b1 = 0).
        is_planar: True if E <= 3V - 6 (Euler's planarity heuristic).
        cycle_rank: Number of independent cycles (= b1).
        edge_density: |E| / (|V| * (|V| - 1) / 2) for the undirected graph.
    """

    euler_characteristic: int
    betti_0: int
    betti_1: int
    genus: int
    is_tree: bool
    is_planar: bool
    cycle_rank: int
    edge_density: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _undirected_edges(ss: StateSpace) -> tuple[set[frozenset[int]], int]:
    """Extract undirected edges from a state space.

    Returns (proper_edges, self_loop_count) where:
    - proper_edges: set of frozenset({src, tgt}) with src != tgt
    - self_loop_count: number of distinct states with self-loops

    Both contribute to topology, but self-loops are excluded from
    edge density (which uses the simple graph formula).
    """
    proper: set[frozenset[int]] = set()
    loops: set[int] = set()
    for src, _label, tgt in ss.transitions:
        if src == tgt:
            loops.add(src)
        else:
            proper.add(frozenset({src, tgt}))
    return proper, len(loops)


def _total_undirected_edges(ss: StateSpace) -> int:
    """Total undirected edge count including self-loops."""
    proper, loop_count = _undirected_edges(ss)
    return len(proper) + loop_count


def _connected_components(ss: StateSpace) -> int:
    """Count connected components treating the graph as undirected."""
    if not ss.states:
        return 0

    # Build undirected adjacency list
    adj: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _label, tgt in ss.transitions:
        adj[src].add(tgt)
        adj[tgt].add(src)

    visited: set[int] = set()
    components = 0
    for start in ss.states:
        if start in visited:
            continue
        components += 1
        queue: deque[int] = deque([start])
        visited.add(start)
        while queue:
            s = queue.popleft()
            for neighbor in adj[s]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    return components


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def euler_characteristic(ss: StateSpace) -> int:
    """Compute the Euler characteristic V - E for the undirected graph.

    V is the number of states, E is the number of undirected edges
    (directed edges between the same pair are deduplicated).
    Self-loops count as edges for topological purposes.
    """
    v = len(ss.states)
    e = _total_undirected_edges(ss)
    return v - e


def betti_numbers(ss: StateSpace) -> tuple[int, int]:
    """Compute Betti numbers (b0, b1) for the undirected underlying graph.

    b0 = number of connected components.
    b1 = E - V + b0 = number of independent cycles.
    Self-loops contribute to b1 (each is an independent cycle).
    """
    v = len(ss.states)
    e = _total_undirected_edges(ss)
    b0 = _connected_components(ss)
    b1 = e - v + b0
    return (b0, b1)


def cycle_rank(ss: StateSpace) -> int:
    """Return the number of independent cycles (= first Betti number b1)."""
    _, b1 = betti_numbers(ss)
    return b1


def is_tree(ss: StateSpace) -> bool:
    """Check if the undirected underlying graph is a tree (or forest).

    A graph is a tree/forest if it has no independent cycles (b1 = 0).
    """
    _, b1 = betti_numbers(ss)
    return b1 == 0


def edge_density(ss: StateSpace) -> float:
    """Compute edge density: |E| / (|V| * (|V| - 1) / 2).

    Returns the ratio of non-self-loop undirected edges to the maximum
    possible edges in a simple undirected graph.  Self-loops are excluded
    since the denominator counts simple edges only.
    Returns 0.0 for graphs with fewer than 2 vertices.
    """
    v = len(ss.states)
    if v < 2:
        return 0.0
    max_edges = v * (v - 1) / 2
    proper, _ = _undirected_edges(ss)
    return len(proper) / max_edges


def classify_topology(ss: StateSpace) -> TopologicalResult:
    """Perform full topological analysis of a state space.

    Returns a TopologicalResult with all invariants computed.
    """
    v = len(ss.states)
    e = _total_undirected_edges(ss)
    b0 = _connected_components(ss)
    b1 = e - v + b0
    chi = v - e

    # Planarity heuristic: E <= 3V - 6 for V >= 3 (proper edges only)
    proper, _ = _undirected_edges(ss)
    e_proper = len(proper)
    if v < 3:
        planar = True
    else:
        planar = e_proper <= 3 * v - 6

    # Edge density (simple edges only)
    if v < 2:
        density = 0.0
    else:
        max_edges = v * (v - 1) / 2
        density = e_proper / max_edges

    # Genus: for a connected graph embedded on a surface of genus g,
    # chi = V - E + F = 2 - 2g.  For the 1-skeleton (no faces),
    # genus = b1 (number of independent cycles handles).
    genus = b1

    return TopologicalResult(
        euler_characteristic=chi,
        betti_0=b0,
        betti_1=b1,
        genus=genus,
        is_tree=(b1 == 0),
        is_planar=planar,
        cycle_rank=b1,
        edge_density=round(density, 6),
    )


def topological_distance(ss1: StateSpace, ss2: StateSpace) -> float:
    """Compute a simple topological distance between two state spaces.

    distance = |euler(ss1) - euler(ss2)| + |b1(ss1) - b1(ss2)|

    This is a metric (satisfies identity of indiscernibles for the pair
    of invariants, symmetry, and triangle inequality).
    """
    chi1 = euler_characteristic(ss1)
    chi2 = euler_characteristic(ss2)
    _, b1_1 = betti_numbers(ss1)
    _, b1_2 = betti_numbers(ss2)
    return float(abs(chi1 - chi2) + abs(b1_1 - b1_2))
