"""Discrete Morse theory for session type lattices (Step 30x).

Discrete Morse theory (Forman, 1998) constructs acyclic matchings on a
CW complex to compute homology efficiently.  For a graph (1-dimensional
CW complex), the cells are:
  - 0-cells: vertices (states)
  - 1-cells: edges (covering relations in the Hasse diagram)

A discrete Morse matching pairs 0-cells with incident 1-cells such that
each cell appears in at most one pair.  Unmatched cells are *critical*.
The number of critical k-cells bounds the k-th Betti number.

For session type lattices, discrete Morse theory provides:
  1. Efficient homology computation (fewer cells than the full complex).
  2. Gradient vector fields on the protocol state space.
  3. Morse inequalities relating critical cells to Betti numbers.

Key functions:
  - ``hasse_graph(ss)``          -- edges of the Hasse diagram
  - ``greedy_matching(ss)``      -- greedy acyclic matching on the CW complex
  - ``critical_cells(matching)`` -- cells not in any matched pair
  - ``morse_function(ss)``       -- discrete Morse function values
  - ``gradient_field(ss)``       -- gradient vector field from matching
  - ``betti_numbers(ss)``        -- Betti numbers via graph topology
  - ``morse_inequalities(ss)``   -- verify weak and strong Morse inequalities
  - ``analyze_morse(ss)``        -- full Morse theory analysis
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
class MorseMatching:
    """An acyclic matching on the CW complex of the Hasse diagram.

    In discrete Morse theory on a graph, a matching pairs vertices
    (0-cells) with incident edges (1-cells).

    Attributes:
        matched_pairs: Set of (vertex, edge_index) matched pairs.
        critical_vertices: Unmatched 0-cells (vertices).
        critical_edges: Unmatched 1-cells (edge indices).
        num_states: Total number of vertices.
        num_hasse_edges: Total number of Hasse edges (1-cells).
        is_acyclic: True iff the matching induces no directed cycles
                    in the modified Hasse diagram.
    """
    matched_pairs: frozenset[tuple[int, int]]
    critical_vertices: frozenset[int]
    critical_edges: frozenset[int]
    num_states: int
    num_hasse_edges: int
    is_acyclic: bool

    @property
    def critical_cells(self) -> frozenset[int]:
        """All critical cells (vertices only, for backward compat)."""
        return self.critical_vertices


@dataclass(frozen=True)
class MorseFunction:
    """A discrete Morse function on the state space.

    Assigns a real value to each cell (vertex and edge) such that
    for each vertex v, at most one incident edge e has f(e) <= f(v),
    and for each edge e, at most one endpoint v has f(v) >= f(e).

    Attributes:
        values: Mapping from state ID to function value.
        is_valid: True iff the function satisfies discrete Morse conditions.
    """
    values: dict[int, float]
    is_valid: bool


@dataclass(frozen=True)
class GradientField:
    """Gradient vector field from a Morse matching.

    Attributes:
        pairs: Matched gradient pairs (vertex, edge_index).
        critical: Unmatched vertices (stationary points of gradient).
        flow_graph: Adjacency for the gradient flow.
    """
    pairs: frozenset[tuple[int, int]]
    critical: frozenset[int]
    flow_graph: dict[int, list[int]]


@dataclass(frozen=True)
class MorseResult:
    """Full Morse theory analysis result.

    Attributes:
        matching: The acyclic matching.
        function: The discrete Morse function.
        gradient: The gradient vector field.
        betti_numbers: Tuple of Betti numbers (b0, b1, ...).
        critical_cells: Set of critical vertices.
        num_critical: Total number of critical cells (vertices + edges).
        euler_characteristic: V - E for the undirected graph.
        weak_morse_holds: True iff c_k >= beta_k for all k.
        strong_morse_holds: True iff sum(-1)^i c_i = chi.
    """
    matching: MorseMatching
    function: MorseFunction
    gradient: GradientField
    betti_numbers: tuple[int, ...]
    critical_cells: frozenset[int]
    num_critical: int
    euler_characteristic: int
    weak_morse_holds: bool
    strong_morse_holds: bool


# ---------------------------------------------------------------------------
# Internal: Hasse diagram construction
# ---------------------------------------------------------------------------

def _build_reachability(ss: StateSpace) -> dict[int, set[int]]:
    """Compute forward reachability (inclusive) for all states."""
    adj: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        adj[src].add(tgt)

    reach: dict[int, set[int]] = {}
    for s in ss.states:
        visited: set[int] = set()
        stack = [s]
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            for t in adj.get(v, set()):
                stack.append(t)
        reach[s] = visited
    return reach


def _compute_sccs(ss: StateSpace) -> dict[int, int]:
    """Map each state to its SCC representative (smallest member)."""
    reach = _build_reachability(ss)
    scc_map: dict[int, int] = {}
    for s in ss.states:
        # SCC of s = {t : t reachable from s AND s reachable from t}
        scc = {t for t in reach[s] if s in reach.get(t, set())}
        scc_map[s] = min(scc)
    return scc_map


def _build_covering(ss: StateSpace) -> list[tuple[int, int, str]]:
    """Extract covering relations (Hasse edges) with labels.

    Returns (source, target, label) triples.  Multiple labels between
    the same pair produce multiple edges.  Edges within the same SCC
    (from recursion cycles) are excluded.
    """
    reach = _build_reachability(ss)
    scc = _compute_sccs(ss)

    adj: dict[int, list[tuple[int, str]]] = {s: [] for s in ss.states}
    for src, label, tgt in ss.transitions:
        adj[src].append((tgt, label))

    covers: list[tuple[int, int, str]] = []
    for s in ss.states:
        for t, label in adj[s]:
            if t == s:
                continue
            # Skip edges within the same SCC (cycles from recursion)
            if scc[s] == scc[t]:
                continue
            is_cover = True
            for u, _ in adj[s]:
                if u != t and u != s and scc[u] != scc[s] and t in reach.get(u, set()):
                    is_cover = False
                    break
            if is_cover:
                covers.append((s, t, label))

    return covers


def _deduplicated_hasse(ss: StateSpace) -> list[tuple[int, int]]:
    """Deduplicated Hasse edges: one edge per covering pair (s, t).

    For the CW complex, multiple transition labels between the same
    pair of states represent the same covering relation, so we treat
    them as a single 1-cell.
    """
    covers = _build_covering(ss)
    seen: set[tuple[int, int]] = set()
    result: list[tuple[int, int]] = []
    for s, t, _ in covers:
        if (s, t) not in seen:
            seen.add((s, t))
            result.append((s, t))
    return result


def _hasse_undirected(ss: StateSpace) -> set[frozenset[int]]:
    """Undirected edges from the deduplicated Hasse diagram."""
    hasse = _deduplicated_hasse(ss)
    return {frozenset({u, v}) for u, v in hasse if u != v}


def _euler_characteristic(ss: StateSpace) -> int:
    """V - E on the Hasse CW complex (deduplicated covering relations)."""
    edges = _hasse_undirected(ss)
    return len(ss.states) - len(edges)


# ---------------------------------------------------------------------------
# Internal: connected components (undirected)
# ---------------------------------------------------------------------------

def _connected_components_count(states: set[int],
                                edges: list[tuple[int, int]]) -> int:
    """Count connected components treating edges as undirected."""
    if not states:
        return 0
    adj: dict[int, set[int]] = {s: set() for s in states}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    visited: set[int] = set()
    count = 0
    for s in states:
        if s in visited:
            continue
        count += 1
        queue: deque[int] = deque([s])
        visited.add(s)
        while queue:
            v = queue.popleft()
            for nb in adj[v]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
    return count


# ---------------------------------------------------------------------------
# Internal: acyclicity check for matching on CW complex
# ---------------------------------------------------------------------------

def _matching_is_acyclic_cw(
    states: set[int],
    hasse_edges: list[tuple[int, int]],
    vertex_to_edge: dict[int, int],
) -> bool:
    """Check acyclicity of the discrete Morse gradient V-paths.

    A gradient V-path follows matched pairs: from vertex v matched with
    edge e_i, go to the OTHER endpoint of e_i. The matching is acyclic
    iff there is no closed gradient V-path.

    We build a directed graph on vertices: v -> w if v is matched with
    edge (u, w) or (w, u) where w is the other endpoint. Then check
    for cycles in this graph.
    """
    # Build gradient successor graph on matched vertices
    gradient_succ: dict[int, int | None] = {s: None for s in states}
    for v, ei in vertex_to_edge.items():
        u, w = hasse_edges[ei]
        other = w if v == u else u
        gradient_succ[v] = other

    # Check for cycles: follow gradient paths from each matched vertex
    visited: set[int] = set()
    for start in states:
        if start not in vertex_to_edge or start in visited:
            continue
        path: set[int] = set()
        current: int | None = start
        while current is not None and current in vertex_to_edge:
            if current in path:
                return False  # cycle detected
            if current in visited:
                break
            path.add(current)
            current = gradient_succ.get(current)
        visited |= path
    return True


# ---------------------------------------------------------------------------
# Public API: Hasse graph
# ---------------------------------------------------------------------------

def hasse_graph(ss: StateSpace) -> list[tuple[int, int, str]]:
    """Return edges of the Hasse diagram as (source, target, label) triples.

    Source covers target (source > target in the poset ordering).
    Each labeled transition that is a covering relation is a separate edge.
    """
    return _build_covering(ss)


# ---------------------------------------------------------------------------
# Public API: Greedy acyclic matching
# ---------------------------------------------------------------------------

def greedy_matching(ss: StateSpace) -> MorseMatching:
    """Compute a greedy acyclic matching on the CW complex.

    Pairs vertices (0-cells) with incident edges (1-cells).  Each vertex
    is matched with at most one edge, and each edge with at most one vertex.

    Edges are deduplicated by (source, target) pair: multiple transition
    labels between the same covering pair count as one 1-cell.

    The matching greedily selects edges incident to low-degree vertices first,
    checking acyclicity at each step.

    Returns a MorseMatching with critical vertices and critical edges.
    """
    hasse_edges = _deduplicated_hasse(ss)
    num_edges = len(hasse_edges)

    # For each vertex, which edge indices are incident?
    incident: dict[int, list[int]] = {s: [] for s in ss.states}
    for i, (u, v) in enumerate(hasse_edges):
        incident[u].append(i)
        incident[v].append(i)

    matched_vertices: set[int] = set()
    matched_edge_indices: set[int] = set()
    matched_pairs: set[tuple[int, int]] = set()
    # vertex -> edge_index mapping for acyclicity check
    vertex_to_edge: dict[int, int] = {}

    degree: dict[int, int] = {s: len(incident[s]) for s in ss.states}

    # Process edges sorted by minimum endpoint degree (fewest-first)
    edge_order = sorted(
        range(num_edges),
        key=lambda i: min(degree[hasse_edges[i][0]], degree[hasse_edges[i][1]]),
    )

    for ei in edge_order:
        if ei in matched_edge_indices:
            continue
        u, v = hasse_edges[ei]

        # Try matching edge with its lower endpoint (v) first,
        # then upper endpoint (u)
        for vertex in (v, u):
            if vertex in matched_vertices:
                continue
            # Check acyclicity with this candidate
            candidate_v2e = dict(vertex_to_edge)
            candidate_v2e[vertex] = ei
            if _matching_is_acyclic_cw(ss.states, hasse_edges, candidate_v2e):
                matched_pairs.add((vertex, ei))
                matched_vertices.add(vertex)
                matched_edge_indices.add(ei)
                vertex_to_edge[vertex] = ei
                break

    crit_verts = frozenset(ss.states - matched_vertices)
    crit_edges = frozenset(set(range(num_edges)) - matched_edge_indices)

    return MorseMatching(
        matched_pairs=frozenset(matched_pairs),
        critical_vertices=crit_verts,
        critical_edges=crit_edges,
        num_states=len(ss.states),
        num_hasse_edges=num_edges,
        is_acyclic=True,  # guaranteed by construction
    )


# ---------------------------------------------------------------------------
# Public API: Critical cells
# ---------------------------------------------------------------------------

def critical_cells(matching: MorseMatching) -> frozenset[int]:
    """Return critical vertices (unmatched 0-cells)."""
    return matching.critical_vertices


# ---------------------------------------------------------------------------
# Public API: Morse function
# ---------------------------------------------------------------------------

def morse_function(ss: StateSpace) -> MorseFunction:
    """Assign discrete Morse function values to vertices.

    Uses topological-order assignment: vertices closer to top get
    higher values.  Matched pairs get adjusted values to satisfy
    the Forman condition.
    """
    matching = greedy_matching(ss)
    hasse_edges = _deduplicated_hasse(ss)

    # Topological ordering via Kahn's algorithm on Hasse DAG
    adj: dict[int, set[int]] = {s: set() for s in ss.states}
    in_degree: dict[int, int] = {s: 0 for s in ss.states}
    for src, tgt in hasse_edges:
        adj[src].add(tgt)
        in_degree[tgt] = in_degree.get(tgt, 0) + 1

    queue: deque[int] = deque()
    for s in ss.states:
        if in_degree.get(s, 0) == 0:
            queue.append(s)

    topo_order: list[int] = []
    while queue:
        s = queue.popleft()
        topo_order.append(s)
        for t in adj.get(s, set()):
            in_degree[t] -= 1
            if in_degree[t] == 0:
                queue.append(t)

    remaining = ss.states - set(topo_order)
    topo_order.extend(sorted(remaining))

    n = len(topo_order)
    values: dict[int, float] = {}
    for i, s in enumerate(topo_order):
        values[s] = float(n - i)

    # Adjust matched pairs: for a matched (vertex, edge_idx) pair,
    # modify the vertex value to create the Morse condition
    matched_edge_to_vertex: dict[int, int] = {}
    for vertex, ei in matching.matched_pairs:
        matched_edge_to_vertex[ei] = vertex

    for ei, vertex in matched_edge_to_vertex.items():
        u, v = hasse_edges[ei]
        # The matched vertex gets a value close to the edge midpoint
        values[vertex] = (values[u] + values[v]) / 2.0 + 0.1

    # Validate: discrete Morse condition on vertices
    cover_down: dict[int, list[int]] = {s: [] for s in ss.states}
    cover_up: dict[int, list[int]] = {s: [] for s in ss.states}
    for src, tgt in hasse_edges:
        cover_down[src].append(tgt)
        cover_up[tgt].append(src)

    is_valid = True
    for s in ss.states:
        exceptional_down = sum(
            1 for t in cover_down[s] if values[t] >= values[s]
        )
        exceptional_up = sum(
            1 for t in cover_up[s] if values[t] <= values[s]
        )
        if exceptional_down > 1 or exceptional_up > 1:
            is_valid = False
            break

    return MorseFunction(values=values, is_valid=is_valid)


# ---------------------------------------------------------------------------
# Public API: Gradient vector field
# ---------------------------------------------------------------------------

def gradient_field(ss: StateSpace) -> GradientField:
    """Compute gradient vector field from the acyclic matching.

    The gradient field reverses matched Hasse edges and keeps
    unmatched edges in their original direction.
    """
    matching = greedy_matching(ss)
    hasse_edges = _deduplicated_hasse(ss)

    matched_indices = {ei for _, ei in matching.matched_pairs}

    flow: dict[int, list[int]] = {s: [] for s in ss.states}
    for i, (u, v) in enumerate(hasse_edges):
        if i in matched_indices:
            flow[v].append(u)  # reverse matched edge
        else:
            flow[u].append(v)  # keep unmatched direction

    return GradientField(
        pairs=matching.matched_pairs,
        critical=matching.critical_vertices,
        flow_graph=flow,
    )


# ---------------------------------------------------------------------------
# Public API: Betti numbers
# ---------------------------------------------------------------------------

def betti_numbers(ss: StateSpace) -> tuple[int, ...]:
    """Compute Betti numbers for the Hasse CW complex.

    b0 = number of connected components.
    b1 = E - V + b0 (cycle rank).

    Uses the deduplicated Hasse diagram (covering relations).
    """
    edges = _hasse_undirected(ss)
    v = len(ss.states)
    e = len(edges)
    b0 = _connected_components_count(
        ss.states,
        [(min(edge), max(edge)) for edge in edges],
    )
    b1 = max(0, e - v + b0)
    return (b0, b1)


# ---------------------------------------------------------------------------
# Public API: Morse inequalities
# ---------------------------------------------------------------------------

def morse_inequalities(ss: StateSpace) -> tuple[bool, bool]:
    """Verify weak and strong Morse inequalities.

    Weak:   c_k >= beta_k for all k.
    Strong: sum_k (-1)^k c_k = chi (Euler characteristic).

    For the graph CW complex:
      c0 = number of critical 0-cells (unmatched vertices)
      c1 = number of critical 1-cells (unmatched edges)
      chi = V - E (undirected graph)
      strong: c0 - c1 = chi

    Returns (weak_holds, strong_holds).
    """
    matching = greedy_matching(ss)
    betti = betti_numbers(ss)

    c0 = len(matching.critical_vertices)
    c1 = len(matching.critical_edges)

    # Weak: c_k >= beta_k
    weak_holds = True
    if len(betti) > 0 and c0 < betti[0]:
        weak_holds = False
    if len(betti) > 1 and c1 < betti[1]:
        weak_holds = False

    # Strong: c0 - c1 = chi
    chi = _euler_characteristic(ss)
    strong_holds = (c0 - c1 == chi)

    return (weak_holds, strong_holds)


# ---------------------------------------------------------------------------
# Public API: Full analysis
# ---------------------------------------------------------------------------

def analyze_morse(ss: StateSpace) -> MorseResult:
    """Full Morse theory analysis of a session type state space."""
    matching = greedy_matching(ss)
    func = morse_function(ss)
    grad = gradient_field(ss)
    betti = betti_numbers(ss)
    chi = _euler_characteristic(ss)
    weak, strong = morse_inequalities(ss)

    num_crit = len(matching.critical_vertices) + len(matching.critical_edges)

    return MorseResult(
        matching=matching,
        function=func,
        gradient=grad,
        betti_numbers=betti,
        critical_cells=matching.critical_vertices,
        num_critical=num_crit,
        euler_characteristic=chi,
        weak_morse_holds=weak,
        strong_morse_holds=strong,
    )
