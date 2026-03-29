"""Protocol-aware error correction via expander codes (Step 31e).

Uses spectral expansion properties of session type lattice Hasse diagrams
to build error-correcting codes for protocol messages.  The key insight is
that good spectral expanders yield good LDPC (low-density parity-check)
codes: if the Hasse diagram has a large spectral gap, protocol messages
can be protected against bit-flips with provable distance guarantees.

Main API:
  - ``spectral_gap(ss)``            spectral gap of the Hasse adjacency
  - ``expansion_ratio(ss)``         vertex expansion ratio h(G)
  - ``is_expander(ss, threshold)``  test whether the graph is an expander
  - ``cheeger_bounds(ss)``          Cheeger inequality: relate gap to expansion
  - ``error_correction_capacity(ss)`` number of correctable errors
  - ``parity_check_matrix(ss)``     LDPC-style parity check from the lattice
  - ``code_distance(ss)``           minimum distance of the induced code
  - ``ExpanderCodeResult``          aggregate analysis result

Key property: the Cheeger inequality ties spectral gap to combinatorial
expansion:  lambda_1 / 2  <=  h(G)  <=  sqrt(2 * d_max * lambda_1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CheegerBounds:
    """Cheeger inequality bounds relating spectral gap to expansion.

    Attributes:
        lower: Lower Cheeger bound  lambda_1 / 2.
        upper: Upper Cheeger bound  sqrt(2 * d_max * lambda_1).
        spectral_gap: The Fiedler value (algebraic connectivity).
        max_degree: Maximum vertex degree in the Hasse diagram.
    """
    lower: float
    upper: float
    spectral_gap: float
    max_degree: int


@dataclass(frozen=True)
class ExpanderCodeResult:
    """Complete expander-code analysis of a session type lattice.

    Attributes:
        num_states: Number of states.
        num_edges: Number of undirected Hasse edges.
        spectral_gap: Algebraic connectivity (Fiedler value).
        expansion_ratio: Vertex expansion ratio h(G).
        is_expander: Whether the graph satisfies the expansion threshold.
        cheeger: Cheeger inequality bounds.
        code_distance: Minimum Hamming distance of the induced code.
        correctable_errors: Number of errors the code can correct.
        parity_check_rows: Number of parity check equations.
        parity_check_cols: Number of code bits.
        rate: Code rate (1 - rows/cols), clamped to [0, 1].
    """
    num_states: int
    num_edges: int
    spectral_gap: float
    expansion_ratio: float
    is_expander: bool
    cheeger: CheegerBounds
    code_distance: int
    correctable_errors: int
    parity_check_rows: int
    parity_check_cols: int
    rate: float


# ---------------------------------------------------------------------------
# Hasse graph utilities
# ---------------------------------------------------------------------------

def _hasse_edges(ss: "StateSpace") -> list[tuple[int, int]]:
    """Compute covering relation edges of the Hasse diagram."""
    adj: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        adj[src].add(tgt)

    edges: list[tuple[int, int]] = []
    for a in ss.states:
        direct = adj[a]
        for b in direct:
            is_cover = True
            for c in direct:
                if c == b:
                    continue
                visited: set[int] = set()
                stack = [c]
                while stack:
                    u = stack.pop()
                    if u == b:
                        is_cover = False
                        break
                    if u in visited:
                        continue
                    visited.add(u)
                    for v in adj[u]:
                        if v not in visited:
                            stack.append(v)
                if not is_cover:
                    break
            if is_cover:
                edges.append((a, b))
    return edges


def _undirected_adj(ss: "StateSpace") -> dict[int, set[int]]:
    """Build undirected adjacency from Hasse edges."""
    adj: dict[int, set[int]] = {s: set() for s in ss.states}
    for a, b in _hasse_edges(ss):
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _degree_sequence(ss: "StateSpace") -> dict[int, int]:
    """Vertex degrees in the undirected Hasse diagram."""
    adj = _undirected_adj(ss)
    return {s: len(adj[s]) for s in ss.states}


# ---------------------------------------------------------------------------
# Spectral gap
# ---------------------------------------------------------------------------

def spectral_gap(ss: "StateSpace") -> float:
    """Compute the spectral gap (Fiedler value / algebraic connectivity).

    This is the second-smallest eigenvalue of the Laplacian of the
    undirected Hasse diagram.  A large spectral gap means the graph
    is a good expander.

    Args:
        ss: A session type state space.

    Returns:
        The spectral gap (non-negative float).
    """
    from reticulate.matrix import fiedler_value
    return fiedler_value(ss)


# ---------------------------------------------------------------------------
# Vertex expansion ratio
# ---------------------------------------------------------------------------

def expansion_ratio(ss: "StateSpace") -> float:
    """Compute the vertex expansion ratio h(G).

    h(G) = min over all subsets S with |S| <= n/2 of
            |boundary(S)| / |S|

    where boundary(S) = {v not in S : v adjacent to some u in S}.

    For small state spaces (< 20 states) we enumerate subsets exactly.
    For larger ones we use the Cheeger lower bound from the spectral gap.

    Args:
        ss: A session type state space.

    Returns:
        The vertex expansion ratio (non-negative float).
    """
    n = len(ss.states)
    if n <= 1:
        return 0.0

    states = sorted(ss.states)
    adj = _undirected_adj(ss)

    if n > 20:
        # Use Cheeger lower bound for large graphs
        gap = spectral_gap(ss)
        return gap / 2.0

    # Exact computation for small graphs
    min_ratio = float("inf")
    # Enumerate all non-empty subsets of size <= n/2
    for mask in range(1, 1 << n):
        subset = {states[i] for i in range(n) if mask & (1 << i)}
        if len(subset) > n // 2:
            continue
        # Compute boundary
        boundary = set()
        for u in subset:
            for v in adj[u]:
                if v not in subset:
                    boundary.add(v)
        ratio = len(boundary) / len(subset)
        min_ratio = min(min_ratio, ratio)

    return min_ratio if min_ratio != float("inf") else 0.0


# ---------------------------------------------------------------------------
# Expander test
# ---------------------------------------------------------------------------

def is_expander(ss: "StateSpace", threshold: float = 0.5) -> bool:
    """Test whether the Hasse diagram is a spectral expander.

    A graph is an expander if its spectral gap exceeds a threshold.
    The default threshold of 0.5 is standard for communication graphs.

    Args:
        ss: A session type state space.
        threshold: Minimum spectral gap required (default 0.5).

    Returns:
        True if the spectral gap >= threshold.
    """
    return spectral_gap(ss) >= threshold


# ---------------------------------------------------------------------------
# Cheeger inequality
# ---------------------------------------------------------------------------

def cheeger_bounds(ss: "StateSpace") -> CheegerBounds:
    """Compute Cheeger inequality bounds.

    The discrete Cheeger inequality states:
        lambda_1 / 2  <=  h(G)  <=  sqrt(2 * d_max * lambda_1)

    where lambda_1 is the Fiedler value and d_max is the maximum degree.

    Args:
        ss: A session type state space.

    Returns:
        CheegerBounds with lower and upper bounds.
    """
    gap = spectral_gap(ss)
    degrees = _degree_sequence(ss)
    d_max = max(degrees.values()) if degrees else 0

    lower = gap / 2.0
    upper = math.sqrt(2.0 * d_max * gap) if gap > 0 and d_max > 0 else 0.0

    return CheegerBounds(
        lower=lower,
        upper=upper,
        spectral_gap=gap,
        max_degree=d_max,
    )


# ---------------------------------------------------------------------------
# Parity check matrix (LDPC-style)
# ---------------------------------------------------------------------------

def parity_check_matrix(ss: "StateSpace") -> list[list[int]]:
    """Build an LDPC-style parity check matrix from the Hasse diagram.

    Each Hasse edge becomes a column (a code bit).  Each vertex becomes
    a row (a parity check): row v has 1 in the columns corresponding to
    edges incident to v.

    This is the vertex-edge incidence matrix of the Hasse diagram,
    which is the standard construction for graph-based LDPC codes.

    Args:
        ss: A session type state space.

    Returns:
        A list-of-lists binary matrix (rows = vertices, cols = edges).
    """
    states = sorted(ss.states)
    idx = {s: i for i, s in enumerate(states)}
    edges = _hasse_edges(ss)

    n = len(states)
    m = len(edges)
    H = [[0] * m for _ in range(n)]

    for j, (a, b) in enumerate(edges):
        H[idx[a]][j] = 1
        H[idx[b]][j] = 1

    return H


# ---------------------------------------------------------------------------
# Code distance
# ---------------------------------------------------------------------------

def _hamming_weight(v: list[int]) -> int:
    """Count nonzero entries."""
    return sum(1 for x in v if x != 0)


def _gf2_add(a: list[int], b: list[int]) -> list[int]:
    """GF(2) vector addition."""
    return [(ai + bi) % 2 for ai, bi in zip(a, b)]


def code_distance(ss: "StateSpace") -> int:
    """Compute the minimum Hamming distance of the induced LDPC code.

    The code C is the null space of the parity check matrix H over GF(2).
    The minimum distance is the minimum weight of a nonzero codeword.

    For small codes we enumerate all codewords; for larger ones we
    use a lower bound from the spectral gap.

    Args:
        ss: A session type state space.

    Returns:
        The minimum Hamming distance (0 if the code is trivial).
    """
    H = parity_check_matrix(ss)
    if not H or not H[0]:
        return 0

    n_rows = len(H)
    n_cols = len(H[0])

    if n_cols == 0:
        return 0

    # Find null space of H over GF(2) by Gaussian elimination
    # Work with columns of H^T (rows of H are the checks)
    # A codeword c satisfies H * c = 0 (mod 2)

    # For small codes, enumerate candidate codewords
    if n_cols <= 16:
        min_weight = n_cols + 1
        for mask in range(1, 1 << n_cols):
            c = [0] * n_cols
            for j in range(n_cols):
                if mask & (1 << j):
                    c[j] = 1
            # Check H * c = 0 mod 2
            valid = True
            for i in range(n_rows):
                s = sum(H[i][j] * c[j] for j in range(n_cols)) % 2
                if s != 0:
                    valid = False
                    break
            if valid:
                w = _hamming_weight(c)
                min_weight = min(min_weight, w)
        return min_weight if min_weight <= n_cols else 0
    else:
        # Lower bound: for (c, d)-regular LDPC, distance >= expansion * n
        gap = spectral_gap(ss)
        degrees = _degree_sequence(ss)
        d_max = max(degrees.values()) if degrees else 1
        if d_max > 0 and gap > 0:
            # Tanner bound approximation
            bound = max(1, int(gap / d_max * n_cols))
            return bound
        return 1


# ---------------------------------------------------------------------------
# Error correction capacity
# ---------------------------------------------------------------------------

def error_correction_capacity(ss: "StateSpace") -> int:
    """Compute the number of errors the expander code can correct.

    For a code of minimum distance d, the number of correctable
    errors is floor((d - 1) / 2).

    Args:
        ss: A session type state space.

    Returns:
        Number of correctable errors (non-negative integer).
    """
    d = code_distance(ss)
    return max(0, (d - 1) // 2)


# ---------------------------------------------------------------------------
# High-level analysis
# ---------------------------------------------------------------------------

def analyze_expander_code(ss: "StateSpace", threshold: float = 0.5) -> ExpanderCodeResult:
    """Complete expander-code analysis of a session type lattice.

    Computes spectral gap, expansion ratio, Cheeger bounds, code
    distance, and error correction capacity.

    Args:
        ss: A session type state space.
        threshold: Expansion threshold for is_expander test.

    Returns:
        ExpanderCodeResult with all metrics.
    """
    gap = spectral_gap(ss)
    exp_ratio = expansion_ratio(ss)
    expander = is_expander(ss, threshold)
    cheeger = cheeger_bounds(ss)
    dist = code_distance(ss)
    correctable = error_correction_capacity(ss)
    H = parity_check_matrix(ss)
    n_rows = len(H)
    n_cols = len(H[0]) if H else 0

    rate = max(0.0, 1.0 - n_rows / n_cols) if n_cols > 0 else 0.0

    return ExpanderCodeResult(
        num_states=len(ss.states),
        num_edges=n_cols,
        spectral_gap=gap,
        expansion_ratio=exp_ratio,
        is_expander=expander,
        cheeger=cheeger,
        code_distance=dist,
        correctable_errors=correctable,
        parity_check_rows=n_rows,
        parity_check_cols=n_cols,
        rate=rate,
    )
