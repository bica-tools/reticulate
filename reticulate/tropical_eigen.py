"""Tropical eigenvalue analysis for session type lattices (Step 30l).

Deep tropical eigenvalue analysis building on the basic tropical operations
in tropical.py.  This module provides:

- **Eigenspace analysis**: basis of the tropical eigenspace via Kleene star
- **Subeigenspace**: vectors v with A ⊗ v ≤ λ ⊕ v
- **Critical components**: SCCs of the critical graph (max-cycle-mean subgraph)
- **Cyclicity**: gcd of lengths of critical cycles
- **CSR decomposition**: eventual periodicity A^{k+c} = λ^c ⊗ A^k
- **Kleene star**: A* = I ⊕ A ⊕ A² ⊕ ... when spectral radius ≤ 0
- **Definiteness**: whether A is definite (A* exists and equals A)
- **Tropical characteristic polynomial**: coefficients via permanent expansion
- **Tropical trace**: tr_k(A) = max diagonal of A^k

Mathematical background (max-plus algebra, ℝ ∪ {-∞}, max, +):
    Tropical eigenvalue λ: A ⊗ v = λ ⊕ v
    λ = max cycle mean = max_{cycles C} (weight(C) / |C|)
    Eigenspace: columns of (λ⁻¹ ⊗ A)* restricted to critical components
    Cyclicity: gcd of lengths of critical cycles
    CSR form: A^{k₀+c} = λ^c ⊗ A^{k₀} for eventual index k₀ and period c
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace

from reticulate.tropical import (
    tropical_add,
    tropical_mul,
    tropical_matrix_mul,
    tropical_matrix_power,
    tropical_eigenvalue,
    tropical_eigenvector,
    critical_graph,
    tropical_determinant,
    tropical_spectral_radius,
    _maxplus_adjacency_matrix,
)
from reticulate.zeta import _adjacency, _compute_sccs, _state_list, compute_rank

INF = float('inf')
NEG_INF = float('-inf')


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TropicalEigenResult:
    """Complete tropical eigenvalue analysis.

    Attributes:
        eigenvalue: Max cycle mean (tropical eigenvalue).
        eigenvectors: Basis of tropical eigenspace.
        eigenspace_dim: Dimension of tropical eigenspace.
        subeigenvectors: Basis of subeigenspace.
        critical_components: SCCs of the critical graph.
        cyclicity: GCD of critical cycle lengths.
        csr_index: CSR index (when periodicity starts).
        csr_period: CSR period.
        kleene_star: A* if spectral radius <= 0, else None.
        is_definite: A is definite (A* exists and A = A*).
        visualization_data: Data for plotting eigenspace.
    """
    eigenvalue: float
    eigenvectors: list[list[float]]
    eigenspace_dim: int
    subeigenvectors: list[list[float]]
    critical_components: list[list[int]]
    cyclicity: int
    csr_index: int
    csr_period: int
    kleene_star: list[list[float]] | None
    is_definite: bool
    visualization_data: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tropical_identity(n: int) -> list[list[float]]:
    """Tropical identity matrix: 0 on diagonal, -inf elsewhere."""
    I = [[NEG_INF] * n for _ in range(n)]
    for i in range(n):
        I[i][i] = 0.0
    return I


def _tropical_matrix_add(
    A: list[list[float]], B: list[list[float]]
) -> list[list[float]]:
    """Element-wise tropical addition (max) of two matrices."""
    n = len(A)
    if n == 0:
        return []
    m = len(A[0])
    C = [[NEG_INF] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            C[i][j] = tropical_add(A[i][j], B[i][j])
    return C


def _tropical_matrix_scalar(
    A: list[list[float]], s: float
) -> list[list[float]]:
    """Tropical scalar multiplication: add s to all finite entries."""
    n = len(A)
    if n == 0:
        return []
    m = len(A[0])
    C = [[NEG_INF] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            C[i][j] = tropical_mul(A[i][j], s)
    return C


def _matrices_equal(
    A: list[list[float]], B: list[list[float]], tol: float = 1e-9
) -> bool:
    """Check if two tropical matrices are equal (within tolerance)."""
    n = len(A)
    if n != len(B):
        return False
    for i in range(n):
        if len(A[i]) != len(B[i]):
            return False
        for j in range(len(A[i])):
            a, b = A[i][j], B[i][j]
            if a == NEG_INF and b == NEG_INF:
                continue
            if a == NEG_INF or b == NEG_INF:
                return False
            if abs(a - b) > tol:
                return False
    return True


def _tropical_matrix_vec(
    A: list[list[float]], v: list[float]
) -> list[float]:
    """Tropical matrix-vector multiply: result[i] = max_j(A[i][j] + v[j])."""
    n = len(A)
    result = [NEG_INF] * n
    for i in range(n):
        for j in range(len(v)):
            val = tropical_mul(A[i][j], v[j])
            result[i] = tropical_add(result[i], val)
    return result


def _cycle_lengths_in_subgraph(
    nodes: set[int], adj: dict[int, list[int]]
) -> list[int]:
    """Find lengths of all simple cycles within a subgraph defined by nodes.

    Uses iterative DFS with backtracking.
    """
    lengths: list[int] = []
    node_list = sorted(nodes)

    for start in node_list:
        # DFS from start, only through nodes in the set
        # Track path as a stack
        # (current_node, iterator_index_into_neighbors, path_so_far)
        stack: list[tuple[int, int, list[int]]] = [(start, 0, [start])]
        while stack:
            node, idx, path = stack[-1]
            neighbors = [n for n in adj.get(node, []) if n in nodes]
            found_next = False
            for i in range(idx, len(neighbors)):
                nb = neighbors[i]
                if nb == start and len(path) > 1:
                    # Found a cycle back to start
                    lengths.append(len(path))
                    continue
                if nb not in path and nb > start:
                    # Only explore neighbors > start to avoid duplicates
                    # (each cycle is found starting from its smallest node)
                    stack[-1] = (node, i + 1, path)
                    stack.append((nb, 0, path + [nb]))
                    found_next = True
                    break
            if not found_next:
                stack.pop()

    return lengths


def _gcd(a: int, b: int) -> int:
    """Greatest common divisor."""
    return math.gcd(a, b)


def _gcd_list(values: list[int]) -> int:
    """GCD of a list of positive integers."""
    if not values:
        return 0
    result = values[0]
    for v in values[1:]:
        result = _gcd(result, v)
        if result == 1:
            return 1
    return result


# ---------------------------------------------------------------------------
# Critical components
# ---------------------------------------------------------------------------

def critical_components(ss: "StateSpace") -> list[list[int]]:
    """Find SCCs of the critical graph.

    The critical graph consists of nodes and edges that lie on cycles
    achieving the maximum cycle mean (tropical eigenvalue).
    Returns a list of components, each component being a sorted list
    of state IDs.
    """
    crit_edges = critical_graph(ss)
    if not crit_edges:
        return []

    # Collect nodes on critical edges
    crit_nodes: set[int] = set()
    for u, v in crit_edges:
        crit_nodes.add(u)
        crit_nodes.add(v)

    # Build adjacency restricted to critical graph
    crit_adj: dict[int, list[int]] = {n: [] for n in crit_nodes}
    for u, v in crit_edges:
        if v not in crit_adj[u]:
            crit_adj[u].append(v)

    # Find SCCs in critical subgraph using iterative Tarjan's
    index_counter = [0]
    stack: list[int] = []
    on_stack: set[int] = set()
    index: dict[int, int] = {}
    lowlink: dict[int, int] = {}
    components: list[list[int]] = []

    for start in sorted(crit_nodes):
        if start in index:
            continue
        work: list[tuple[int, int]] = [(start, 0)]
        while work:
            v, ci = work[-1]
            if v not in index:
                index[v] = lowlink[v] = index_counter[0]
                index_counter[0] += 1
                stack.append(v)
                on_stack.add(v)
            recurse = False
            neighbors = crit_adj.get(v, [])
            for i in range(ci, len(neighbors)):
                w = neighbors[i]
                if w not in crit_nodes:
                    continue
                if w not in index:
                    work[-1] = (v, i + 1)
                    work.append((w, 0))
                    recurse = True
                    break
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], index[w])
            if recurse:
                continue
            if lowlink[v] == index[v]:
                scc: list[int] = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    scc.append(w)
                    if w == v:
                        break
                # Only include non-trivial SCCs (size > 1) or self-loops
                has_self_loop = any(
                    v2 in crit_adj.get(v2, []) for v2 in scc
                )
                if len(scc) > 1 or has_self_loop:
                    components.append(sorted(scc))
            work.pop()
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[v])

    return sorted(components, key=lambda c: c[0])


# ---------------------------------------------------------------------------
# Cyclicity
# ---------------------------------------------------------------------------

def cyclicity(ss: "StateSpace") -> int:
    """Cyclicity of the tropical matrix.

    The cyclicity is the gcd of lengths of all cycles in the critical graph.
    For acyclic state spaces (no critical graph), cyclicity is 0.
    For a single self-loop, cyclicity is 1.
    """
    crit_edges = critical_graph(ss)
    if not crit_edges:
        return 0

    # Build critical subgraph adjacency
    crit_nodes: set[int] = set()
    for u, v in crit_edges:
        crit_nodes.add(u)
        crit_nodes.add(v)

    crit_adj: dict[int, list[int]] = {n: [] for n in crit_nodes}
    for u, v in crit_edges:
        if v not in crit_adj[u]:
            crit_adj[u].append(v)

    # Find all cycle lengths in critical graph
    cycle_lens = _cycle_lengths_in_subgraph(crit_nodes, crit_adj)

    # Also check for self-loops
    for n in crit_nodes:
        if n in crit_adj.get(n, []):
            cycle_lens.append(1)

    if not cycle_lens:
        # If we have critical edges but found no cycles via simple enumeration,
        # the components themselves are cycles — use component sizes
        comps = critical_components(ss)
        for comp in comps:
            if len(comp) > 0:
                cycle_lens.append(len(comp))

    if not cycle_lens:
        return 1  # Default for cyclic graph

    return _gcd_list(cycle_lens)


# ---------------------------------------------------------------------------
# Kleene star
# ---------------------------------------------------------------------------

def kleene_star(ss: "StateSpace") -> list[list[float]] | None:
    """Compute the Kleene star A* = I ⊕ A ⊕ A^2 ⊕ ... in max-plus algebra.

    Converges if and only if the tropical spectral radius (max cycle mean)
    is <= 0.  For unit-weight session type graphs with cycles, the spectral
    radius is 1, so A* diverges.

    For acyclic session type state spaces (spectral radius = 0), A* is
    the longest-path matrix (= transitive closure with max-plus weights).

    Returns None if spectral radius > 0 (series diverges).
    """
    lam = tropical_eigenvalue(ss)
    if lam > 0.0:
        return None

    A = _maxplus_adjacency_matrix(ss)
    n = len(A)
    if n == 0:
        return []

    # A* = I ⊕ A ⊕ A^2 ⊕ ... ⊕ A^{n-1} (stabilizes for spectral radius ≤ 0)
    result = _tropical_identity(n)
    power = _tropical_identity(n)  # A^0 = I

    for _ in range(n):
        power = tropical_matrix_mul(power, A)
        result = _tropical_matrix_add(result, power)

    return result


# ---------------------------------------------------------------------------
# Eigenspace
# ---------------------------------------------------------------------------

def tropical_eigenspace(ss: "StateSpace") -> list[list[float]]:
    """Compute a basis for the tropical eigenspace.

    The tropical eigenspace of matrix A with eigenvalue λ is the set of
    vectors v satisfying A ⊗ v = λ ⊗ v.

    Method: compute B = λ^{-1} ⊗ A (subtract λ from all finite entries),
    then the eigenspace is generated by columns of B* (Kleene star of B)
    corresponding to critical components.

    For acyclic graphs (λ = 0), the eigenspace is trivially all-zeros.
    """
    A = _maxplus_adjacency_matrix(ss)
    n = len(A)
    if n == 0:
        return []

    lam = tropical_eigenvalue(ss)
    states = _state_list(ss)

    if lam == 0.0:
        # Acyclic: eigenvector is all zeros (the tropical zero vector
        # doesn't count, so return the trivial eigenvector)
        return [[0.0] * n]

    # B = λ^{-1} ⊗ A: subtract λ from all finite entries
    B = [[NEG_INF] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if A[i][j] > NEG_INF:
                B[i][j] = A[i][j] - lam

    # Compute B* = I ⊕ B ⊕ B^2 ⊕ ... ⊕ B^{n-1}
    # Since B has spectral radius 0 (we shifted by λ), this converges
    B_star = _tropical_identity(n)
    power = _tropical_identity(n)
    for _ in range(n):
        power = tropical_matrix_mul(power, B)
        B_star = _tropical_matrix_add(B_star, power)

    # Extract columns corresponding to critical nodes
    comps = critical_components(ss)
    idx = {s: i for i, s in enumerate(states)}

    basis: list[list[float]] = []
    seen_columns: set[int] = set()

    for comp in comps:
        # Pick one representative per component
        rep = comp[0]
        col_idx = idx[rep]
        if col_idx in seen_columns:
            continue
        seen_columns.add(col_idx)
        # Extract column col_idx of B*
        col = [B_star[i][col_idx] for i in range(n)]
        basis.append(col)

    # If no critical components found but graph is cyclic, return eigenvector
    if not basis:
        ev = tropical_eigenvector(ss)
        if ev:
            basis.append(ev)

    return basis if basis else [[0.0] * n]


def tropical_subeigenspace(ss: "StateSpace") -> list[list[float]]:
    """Compute the subeigenspace: {v : A ⊗ v <= λ ⊗ v}.

    The subeigenspace contains all vectors v where each component satisfies
    max_j(A[i][j] + v[j]) <= λ + v[i].

    This is generated by ALL columns of the Kleene star B* = (λ^{-1} ⊗ A)*,
    not just the critical ones.
    """
    A = _maxplus_adjacency_matrix(ss)
    n = len(A)
    if n == 0:
        return []

    lam = tropical_eigenvalue(ss)

    if lam == 0.0:
        # Acyclic: B = A, and B* columns give subeigenspace
        B = [row[:] for row in A]
    else:
        # B = λ^{-1} ⊗ A
        B = [[NEG_INF] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if A[i][j] > NEG_INF:
                    B[i][j] = A[i][j] - lam

    # B* = I ⊕ B ⊕ B^2 ⊕ ...
    B_star = _tropical_identity(n)
    power = _tropical_identity(n)
    for _ in range(n):
        power = tropical_matrix_mul(power, B)
        B_star = _tropical_matrix_add(B_star, power)

    # All columns of B* form a generating set
    basis: list[list[float]] = []
    for j in range(n):
        col = [B_star[i][j] for i in range(n)]
        # Skip all-NEG_INF columns
        if any(v > NEG_INF for v in col):
            basis.append(col)

    return basis if basis else [[0.0] * n]


def eigenspace_dimension(ss: "StateSpace") -> int:
    """Dimension of the tropical eigenspace.

    Equals the number of strongly connected components in the critical graph.
    For acyclic state spaces, the dimension is 1 (trivial eigenspace).
    """
    comps = critical_components(ss)
    if not comps:
        # Acyclic: eigenspace is 1-dimensional (trivial)
        return 1
    return len(comps)


# ---------------------------------------------------------------------------
# CSR decomposition
# ---------------------------------------------------------------------------

def csr_decomposition(ss: "StateSpace") -> tuple[int, int]:
    """Find the CSR (Cyclically Stationary Regime) index and period.

    Returns (k0, c) such that A^{k+c} = λ^c ⊗ A^k for all k >= k0.

    k0 is the CSR index (transient length).
    c is the CSR period (equals cyclicity of the critical graph).

    For acyclic graphs: k0 = 0, c = 1 (trivial periodicity).
    """
    A = _maxplus_adjacency_matrix(ss)
    n = len(A)
    if n == 0:
        return (0, 1)

    lam = tropical_eigenvalue(ss)

    if lam == 0.0:
        # Acyclic: trivial CSR
        return (0, 1)

    # Period c = cyclicity of critical graph
    c = cyclicity(ss)
    if c == 0:
        c = 1

    # Find k0: smallest k such that A^{k+c} = λ^c ⊗ A^k
    # We check this by computing successive powers
    lambda_c = lam * c  # λ^c in tropical = c * λ in ordinary arithmetic

    # Precompute powers iteratively
    max_k = n * n + 1  # Upper bound on transient
    powers: list[list[list[float]]] = []
    current = _tropical_identity(n)
    powers.append(current)  # A^0

    for i in range(1, max_k + c + 1):
        current = tropical_matrix_mul(current, A)
        powers.append(current)

    for k0 in range(max_k):
        if k0 + c >= len(powers):
            break
        A_kc = powers[k0 + c]
        A_k_shifted = _tropical_matrix_scalar(powers[k0], lambda_c)
        if _matrices_equal(A_kc, A_k_shifted):
            return (k0, c)

    # Fallback: return conservative estimate
    return (n, c)


# ---------------------------------------------------------------------------
# Definiteness
# ---------------------------------------------------------------------------

def is_definite(ss: "StateSpace") -> bool:
    """Check if the max-plus adjacency matrix A is definite.

    A matrix is definite if:
    1. The Kleene star A* exists (spectral radius <= 0)
    2. A* = A (the matrix is its own closure)

    For session type state spaces, this means the graph is acyclic
    AND the longest-path closure equals the adjacency matrix, which
    only holds for trivial (single-state or single-transition) cases.
    """
    A_star = kleene_star(ss)
    if A_star is None:
        return False

    A = _maxplus_adjacency_matrix(ss)
    return _matrices_equal(A, A_star)


# ---------------------------------------------------------------------------
# Tropical polynomial operations
# ---------------------------------------------------------------------------

def tropical_characteristic_polynomial(ss: "StateSpace") -> list[float]:
    """Coefficients of the tropical characteristic polynomial.

    The tropical characteristic polynomial of an n×n matrix A is:
        p(x) = ⊕_{k=0}^{n} c_k ⊗ x^{⊗k}

    where c_k = tropical permanent of all (n-k)×(n-k) principal submatrices
    of A, with appropriate signs handled tropically.

    In max-plus algebra: c_k = max over all (n-k)-element subsets S of
    [n] of the max-weight perfect matching in A[S,S].

    Returns coefficients [c_0, c_1, ..., c_n] where c_k is the coefficient
    of x^k (tropical power = k * x).
    """
    A = _maxplus_adjacency_matrix(ss)
    n = len(A)
    if n == 0:
        return [0.0]

    coeffs: list[float] = []

    for k in range(n + 1):
        # c_k: consider all subsets of size (n - k)
        size = n - k
        if size == 0:
            # c_n = tropical "1" = 0 (the identity element of tropical mul)
            coeffs.append(0.0)
            continue

        # Enumerate subsets of size `size` from {0, ..., n-1}
        best = NEG_INF
        _enumerate_subsets_matching(A, n, size, best_ref=[best])
        best = _enumerate_subsets_matching(A, n, size, best_ref=[NEG_INF])
        coeffs.append(best)

    return coeffs


def _enumerate_subsets_matching(
    A: list[list[float]], n: int, size: int,
    best_ref: list[float],
) -> float:
    """Find the maximum weight perfect matching across all subsets of given size.

    Uses recursive enumeration with pruning for small n.
    """
    if size == 0:
        return 0.0
    if size > 10:
        # For large matrices, use greedy approximation on the full matrix
        return _greedy_matching_weight(A, list(range(n)))

    best = NEG_INF

    def _subsets(start: int, chosen: list[int]) -> None:
        nonlocal best
        if len(chosen) == size:
            w = _max_weight_matching(A, chosen)
            best = max(best, w)
            return
        remaining = size - len(chosen)
        for i in range(start, n - remaining + 1):
            chosen.append(i)
            _subsets(i + 1, chosen)
            chosen.pop()

    _subsets(0, [])
    return best


def _max_weight_matching(
    A: list[list[float]], indices: list[int]
) -> float:
    """Maximum weight perfect matching in A restricted to rows/cols in indices.

    For small subsets, uses brute-force permutation search.
    """
    k = len(indices)
    if k == 0:
        return 0.0
    if k == 1:
        return A[indices[0]][indices[0]]

    if k <= 8:
        # Brute force
        best = NEG_INF
        _perm_search(A, indices, k, 0, 0, 0.0, [best])
        return _perm_search_result(A, indices, k)

    return _greedy_matching_weight(A, indices)


def _perm_search_result(
    A: list[list[float]], indices: list[int], k: int
) -> float:
    """Brute-force maximum weight matching via permutation search."""
    best = NEG_INF

    def _search(row: int, used: int, total: float) -> None:
        nonlocal best
        if total == NEG_INF:
            return
        if row == k:
            best = max(best, total)
            return
        for col in range(k):
            if used & (1 << col):
                continue
            val = A[indices[row]][indices[col]]
            if val == NEG_INF:
                continue
            _search(row + 1, used | (1 << col), total + val)

    _search(0, 0, 0.0)
    return best


def _perm_search(
    A: list[list[float]], indices: list[int], k: int,
    row: int, used: int, total: float, best: list[float],
) -> None:
    """Helper for permutation search (unused, kept for compatibility)."""
    pass


def _greedy_matching_weight(
    A: list[list[float]], indices: list[int]
) -> float:
    """Greedy maximum weight matching approximation."""
    k = len(indices)
    used_cols: set[int] = set()
    total = 0.0

    # Sort rows by best available entry (descending)
    for row_idx in range(k):
        i = indices[row_idx]
        best_val = NEG_INF
        best_col = -1
        for col_idx in range(k):
            if col_idx in used_cols:
                continue
            j = indices[col_idx]
            if A[i][j] > best_val:
                best_val = A[i][j]
                best_col = col_idx
        if best_col >= 0 and best_val > NEG_INF:
            used_cols.add(best_col)
            total += best_val
        else:
            return NEG_INF  # No perfect matching

    return total


def tropical_trace(ss: "StateSpace", k: int = 1) -> float:
    """Tropical trace: tr_k(A) = max diagonal of A^k.

    The tropical trace of A^k gives the maximum weight of a closed walk
    of length k.  For unit-weight graphs:
    - tr_1(A) = 1 if there is a self-loop, else -inf
    - tr_k(A) = k if there is a cycle of length <= k, else -inf

    The maximum cycle mean = max_k tr_k(A) / k.
    """
    A = _maxplus_adjacency_matrix(ss)
    n = len(A)
    if n == 0:
        return NEG_INF

    Ak = tropical_matrix_power(A, k)
    # Max diagonal
    result = NEG_INF
    for i in range(n):
        result = tropical_add(result, Ak[i][i])
    return result


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze_tropical_eigen(ss: "StateSpace") -> TropicalEigenResult:
    """Complete tropical eigenvalue analysis of a session type state space.

    Computes eigenvalue, eigenspace basis, subeigenspace, critical components,
    cyclicity, CSR decomposition, Kleene star, and definiteness.
    """
    lam = tropical_eigenvalue(ss)
    eigvecs = tropical_eigenspace(ss)
    dim = eigenspace_dimension(ss)
    subeigvecs = tropical_subeigenspace(ss)
    comps = critical_components(ss)
    cyc = cyclicity(ss)
    k0, c = csr_decomposition(ss)
    kstar = kleene_star(ss)
    definite = is_definite(ss)

    states = _state_list(ss)
    n = len(states)

    # Build visualization data
    viz: dict = {
        "num_states": n,
        "eigenvalue": lam,
        "eigenspace_dim": dim,
        "cyclicity": cyc,
        "csr_index": k0,
        "csr_period": c,
        "critical_nodes": sorted(
            {s for comp in comps for s in comp}
        ),
        "has_kleene_star": kstar is not None,
        "is_definite": definite,
    }

    return TropicalEigenResult(
        eigenvalue=lam,
        eigenvectors=eigvecs,
        eigenspace_dim=dim,
        subeigenvectors=subeigvecs,
        critical_components=comps,
        cyclicity=cyc,
        csr_index=k0,
        csr_period=c,
        kleene_star=kstar,
        is_definite=definite,
        visualization_data=viz,
    )
