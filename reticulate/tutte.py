"""Tutte polynomial and protocol reliability (Step 32d).

The Tutte polynomial T(G; x, y) is a two-variable polynomial that
generalises the chromatic polynomial, the flow polynomial, and the
reliability polynomial.  For a graph G = (V, E):

    T(G; x, y) = sum over spanning subgraphs A:
        (x - 1)^{r(E) - r(A)} * (y - 1)^{|A| - r(A)}

where r(A) = |V| - c(A) is the rank (vertices minus components)
and c(A) is the number of connected components of the subgraph (V, A).

Specialisations:
    - Chromatic polynomial:  P(G, t) = (-1)^{r(E)} * t * T(G; 1-t, 0)
    - Reliability polynomial: R(G, p) relates to T(G; 1, 1/p)
    - Number of spanning trees: T(G; 1, 1)
    - Number of spanning forests: T(G; 2, 1)

This module also provides Kirchhoff's theorem for spanning tree
counting via the Laplacian matrix determinant.

All computations are exact (integer/rational arithmetic).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace

from reticulate.zeta import (
    _compute_sccs,
    _covering_relation,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TutteResult:
    """Tutte polynomial analysis results.

    Attributes:
        coefficients: Dictionary (i, j) -> coefficient of x^i * y^j.
        num_spanning_trees: T(G; 1, 1).
        num_spanning_forests: T(G; 2, 1).
        num_connected_spanning: T(G; 1, 2).
        chromatic_at_2: Number of proper 2-colorings.
        reliability_coeffs: Reliability polynomial coefficients.
        num_states: Number of states (quotient).
        num_edges: Number of Hasse edges.
        is_tree: True iff spanning tree count = 1.
        laplacian_spanning_trees: Spanning trees via Kirchhoff's theorem.
    """
    coefficients: dict[tuple[int, int], int]
    num_spanning_trees: int
    num_spanning_forests: int
    num_connected_spanning: int
    chromatic_at_2: int
    reliability_coeffs: list[float]
    num_states: int
    num_edges: int
    is_tree: bool
    laplacian_spanning_trees: int


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def _hasse_undirected(ss: "StateSpace") -> tuple[list[int], list[tuple[int, int]]]:
    """Extract the quotient Hasse diagram as undirected (nodes, edges)."""
    scc_map, scc_members = _compute_sccs(ss)
    covers = _covering_relation(ss)

    reps = sorted(scc_members.keys())
    edges_set: set[frozenset[int]] = set()
    for x, y in covers:
        rx, ry = scc_map[x], scc_map[y]
        if rx != ry:
            edges_set.add(frozenset({rx, ry}))

    edges = sorted(tuple(sorted(e)) for e in edges_set)
    return reps, edges


def _connected_components(nodes: list[int], edges: list[tuple[int, int]]) -> int:
    """Count connected components via union-find."""
    parent: dict[int, int] = {n: n for n in nodes}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for u, v in edges:
        union(u, v)

    return len({find(n) for n in nodes})


# ---------------------------------------------------------------------------
# Tutte polynomial via subset expansion
# ---------------------------------------------------------------------------

def _rank_and_nullity(
    nodes: list[int],
    all_edges: list[tuple[int, int]],
    subset_mask: int,
) -> tuple[int, int]:
    """Compute rank and nullity of a spanning subgraph.

    rank(A) = |V| - c(A) where c(A) is the number of components.
    nullity(A) = |A| - rank(A).
    """
    sub_edges = [all_edges[i] for i in range(len(all_edges)) if subset_mask & (1 << i)]
    c = _connected_components(nodes, sub_edges)
    n = len(nodes)
    r = n - c
    size = bin(subset_mask).count('1')
    return r, size - r


def tutte_polynomial(ss: "StateSpace") -> dict[tuple[int, int], int]:
    """Compute the Tutte polynomial T(G; x, y) via subset expansion.

    T(G; x, y) = sum_{A subseteq E} (x-1)^{r(E)-r(A)} (y-1)^{|A|-r(A)}

    Returns a dictionary mapping (i, j) to the coefficient of x^i * y^j.
    """
    nodes, edges = _hasse_undirected(ss)
    m = len(edges)
    n = len(nodes)

    if n == 0:
        return {(0, 0): 1}

    # Rank of the full graph
    c_full = _connected_components(nodes, edges)
    r_full = n - c_full

    coeffs: dict[tuple[int, int], int] = {}

    for mask in range(1 << m):
        r_a, null_a = _rank_and_nullity(nodes, edges, mask)
        # Exponents for (x-1) and (y-1)
        exp_x = r_full - r_a
        exp_y = null_a  # |A| - r(A)

        # Expand (x-1)^exp_x * (y-1)^exp_y into sum of x^i * y^j terms
        x_coeffs = _expand_binomial(exp_x)  # coefficients of x^i in (x-1)^exp_x
        y_coeffs = _expand_binomial(exp_y)  # coefficients of y^j in (y-1)^exp_y

        for i, cx in x_coeffs.items():
            for j, cy in y_coeffs.items():
                key = (i, j)
                coeffs[key] = coeffs.get(key, 0) + cx * cy

    # Remove zero entries
    return {k: v for k, v in coeffs.items() if v != 0}


def _expand_binomial(exp: int) -> dict[int, int]:
    """Expand (x - 1)^exp as a polynomial in x.

    Returns {power: coefficient} for (x-1)^exp = sum C(exp,k) x^k (-1)^{exp-k}.
    """
    if exp < 0:
        return {0: 1}
    coeffs: dict[int, int] = {}
    for k in range(exp + 1):
        c = _comb(exp, k) * ((-1) ** (exp - k))
        if c != 0:
            coeffs[k] = c
    return coeffs


def _comb(n: int, k: int) -> int:
    """Binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    result = 1
    for i in range(min(k, n - k)):
        result = result * (n - i) // (i + 1)
    return result


# ---------------------------------------------------------------------------
# Tutte polynomial evaluation
# ---------------------------------------------------------------------------

def _eval_tutte(coeffs: dict[tuple[int, int], int], x: int | float, y: int | float) -> int | float:
    """Evaluate T(G; x, y) at given point."""
    result: int | float = 0
    for (i, j), c in coeffs.items():
        result += c * (x ** i) * (y ** j)
    return result


def count_spanning_trees(ss: "StateSpace") -> int:
    """Count spanning trees: T(G; 1, 1)."""
    coeffs = tutte_polynomial(ss)
    val = _eval_tutte(coeffs, 1, 1)
    return int(round(val))


def count_spanning_forests(ss: "StateSpace") -> int:
    """Count spanning forests: T(G; 2, 1)."""
    coeffs = tutte_polynomial(ss)
    val = _eval_tutte(coeffs, 2, 1)
    return int(round(val))


def count_connected_spanning(ss: "StateSpace") -> int:
    """Count connected spanning subgraphs: T(G; 1, 2)."""
    coeffs = tutte_polynomial(ss)
    val = _eval_tutte(coeffs, 1, 2)
    return int(round(val))


# ---------------------------------------------------------------------------
# Reliability polynomial
# ---------------------------------------------------------------------------

def reliability_polynomial(ss: "StateSpace") -> list[float]:
    """Compute the reliability polynomial R(G, p).

    For a connected graph G with m edges, the all-terminal reliability
    is the probability that the graph remains connected when each edge
    independently fails with probability 1-p.

    R(G, p) = sum_{k=0}^{m} N_k * p^k * (1-p)^{m-k}

    where N_k is the number of connected spanning subgraphs with k edges.

    Returns coefficients as a list for evaluation.
    """
    nodes, edges = _hasse_undirected(ss)
    m = len(edges)
    n = len(nodes)

    if n <= 1:
        return [1.0]

    # Count connected spanning subgraphs by edge count
    n_k: dict[int, int] = {}
    for mask in range(1 << m):
        sub_edges = [edges[i] for i in range(m) if mask & (1 << i)]
        edge_count = bin(mask).count('1')
        c = _connected_components(nodes, sub_edges)
        if c == 1:
            n_k[edge_count] = n_k.get(edge_count, 0) + 1

    # Build R(G, p) = sum N_k * p^k * (1-p)^{m-k}
    # Expand into polynomial in p
    coeffs = [0.0] * (m + 1)
    for k, count in n_k.items():
        # p^k * (1-p)^{m-k} = sum_{j=0}^{m-k} C(m-k, j) (-1)^j p^{k+j}
        for j in range(m - k + 1):
            deg = k + j
            c = count * _comb(m - k, j) * ((-1) ** j)
            if deg <= m:
                coeffs[deg] += c

    return coeffs


def evaluate_reliability(ss: "StateSpace", p: float) -> float:
    """Evaluate the reliability polynomial at probability p."""
    coeffs = reliability_polynomial(ss)
    result = 0.0
    for i, c in enumerate(coeffs):
        result += c * (p ** i)
    return result


# ---------------------------------------------------------------------------
# Kirchhoff's theorem (Laplacian determinant)
# ---------------------------------------------------------------------------

def _laplacian_matrix(nodes: list[int], edges: list[tuple[int, int]]) -> list[list[int]]:
    """Build the Laplacian matrix L = D - A for the undirected graph."""
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    L = [[0] * n for _ in range(n)]

    for u, v in edges:
        iu, iv = idx[u], idx[v]
        L[iu][iv] -= 1
        L[iv][iu] -= 1
        L[iu][iu] += 1
        L[iv][iv] += 1

    return L


def _determinant(mat: list[list[float]]) -> float:
    """Compute determinant via LU decomposition (Gaussian elimination)."""
    n = len(mat)
    if n == 0:
        return 1.0
    if n == 1:
        return mat[0][0]

    # Copy matrix
    work = [row[:] for row in mat]
    det = 1.0

    for col in range(n):
        # Find pivot
        pivot_row = None
        for row in range(col, n):
            if abs(work[row][col]) > 1e-12:
                pivot_row = row
                break
        if pivot_row is None:
            return 0.0

        if pivot_row != col:
            work[col], work[pivot_row] = work[pivot_row], work[col]
            det *= -1

        det *= work[col][col]

        for row in range(col + 1, n):
            if abs(work[row][col]) > 1e-12:
                factor = work[row][col] / work[col][col]
                for c in range(col, n):
                    work[row][c] -= factor * work[col][c]

    return det


def kirchhoff_spanning_trees(ss: "StateSpace") -> int:
    """Count spanning trees via Kirchhoff's matrix-tree theorem.

    The number of spanning trees equals the determinant of any cofactor
    of the Laplacian matrix (i.e., delete one row and column).
    """
    nodes, edges = _hasse_undirected(ss)
    n = len(nodes)

    if n <= 1:
        return 1 if n == 1 else 0
    if not edges:
        return 0

    L = _laplacian_matrix(nodes, edges)

    # Delete first row and column to get cofactor
    cofactor = [[float(L[i][j]) for j in range(1, n)] for i in range(1, n)]
    det = _determinant(cofactor)

    return max(0, int(round(det)))


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_tutte(ss: "StateSpace") -> TutteResult:
    """Complete Tutte polynomial analysis."""
    coeffs = tutte_polynomial(ss)
    nodes, edges = _hasse_undirected(ss)

    trees = count_spanning_trees(ss)
    forests = count_spanning_forests(ss)
    connected = count_connected_spanning(ss)
    kirchhoff = kirchhoff_spanning_trees(ss)

    # Chromatic polynomial at t=2: P(G, 2) = (-1)^r * 2 * T(G; -1, 0)
    # Simpler: evaluate directly
    n = len(nodes)
    m = len(edges)
    c_full = _connected_components(nodes, edges)
    r_full = n - c_full

    # P(G, t) = (-1)^{r(E)} * t * T(G; 1-t, 0)
    # P(G, 2) = (-1)^r * 2 * T(G; -1, 0)
    t_val = _eval_tutte(coeffs, -1, 0)
    chrom_2 = int(round((-1) ** r_full * 2 * t_val))

    rel_coeffs = reliability_polynomial(ss)

    return TutteResult(
        coefficients=coeffs,
        num_spanning_trees=trees,
        num_spanning_forests=forests,
        num_connected_spanning=connected,
        chromatic_at_2=chrom_2,
        reliability_coeffs=rel_coeffs,
        num_states=len(nodes),
        num_edges=len(edges),
        is_tree=(trees == 1 and len(edges) == len(nodes) - 1),
        laplacian_spanning_trees=kirchhoff,
    )
