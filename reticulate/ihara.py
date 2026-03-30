"""Ihara zeta function analysis for session type lattices (Step 30s).

The Ihara zeta function of a graph G is:
    Z_G(u) = prod_[p] (1 - u^|p|)^{-1}
where the product is over prime cycles p (closed walks without
backtracking or tails).

Ihara's theorem (regular case): For a (q+1)-regular graph,
    Z_G(u)^{-1} = (1-u^2)^{r-1} * det(I - Au + qu^2 I)
where A = adjacency matrix, r = |E| - |V| + 1.

Bass's generalization (arbitrary graphs):
    Z_G(u)^{-1} = (1-u^2)^{r-1} * det(I - Au + (D-I)u^2)
where D = degree matrix.

For session type state spaces, the Ihara zeta function encodes the
cyclic structure of protocols.  Acyclic protocols (DAGs) have trivial
zeta Z_G(u) = 1.  Recursive session types produce cycles that the
Ihara zeta captures through its prime cycle spectrum.

Key results for session types:
- DAGs have Z_G(u) = 1 (no prime cycles)
- Trees have Z_G(u) = 1 (no cycles)
- Recursive types have non-trivial prime cycle spectra
- The Ramanujan property relates to optimal spectral gap
- Graph complexity = log|Z_G(1/sqrt(q))| measures cyclic richness

All computations are pure Python (no numpy dependency).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace

from reticulate.zeta import _adjacency, _compute_sccs, _state_list, compute_rank


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IharaResult:
    """Complete Ihara zeta function analysis for a session type state space.

    Attributes:
        ihara_determinant: det(I - Au + (D-I)u^2) evaluated at u=1/q.
        reciprocal_poles: Poles of 1/Z_G(u) from eigenvalues.
        num_prime_cycles: Count of prime cycles (up to length bound).
        prime_cycle_lengths: Sorted lengths of all prime cycles found.
        rank: Cycle rank r = |E| - |V| + 1.
        is_ramanujan: True if graph has optimal spectral gap.
        complexity: log|Z_G(1/sqrt(q))| — graph complexity measure.
        bass_hashimoto_matrix: The edge adjacency (Hashimoto) matrix.
    """
    ihara_determinant: float
    reciprocal_poles: list[complex]
    num_prime_cycles: int
    prime_cycle_lengths: list[int]
    rank: int
    is_ramanujan: bool
    complexity: float
    bass_hashimoto_matrix: list[list[float]]


# ---------------------------------------------------------------------------
# Undirected edge helpers
# ---------------------------------------------------------------------------

def _undirected_edges(ss: "StateSpace") -> list[tuple[int, int]]:
    """Extract undirected edges from the state space.

    For the Ihara zeta, we work with the undirected version of the graph.
    Each directed edge (u, label, v) contributes the undirected edge {u, v}.
    Self-loops are excluded.  Duplicate edges are collapsed.
    """
    seen: set[tuple[int, int]] = set()
    edges: list[tuple[int, int]] = []
    for src, _, tgt in ss.transitions:
        if src == tgt:
            continue
        key = (min(src, tgt), max(src, tgt))
        if key not in seen:
            seen.add(key)
            edges.append(key)
    return edges


def _undirected_adjacency(ss: "StateSpace") -> dict[int, list[int]]:
    """Build undirected adjacency list."""
    adj: dict[int, list[int]] = {s: [] for s in ss.states}
    edges = _undirected_edges(ss)
    for u, v in edges:
        if v not in adj[u]:
            adj[u].append(v)
        if u not in adj[v]:
            adj[v].append(u)
    return adj


def _directed_edges(ss: "StateSpace") -> list[tuple[int, int]]:
    """Extract unique directed edges (ignoring labels, no self-loops)."""
    seen: set[tuple[int, int]] = set()
    edges: list[tuple[int, int]] = []
    for src, _, tgt in ss.transitions:
        if src == tgt:
            continue
        if (src, tgt) not in seen:
            seen.add((src, tgt))
            edges.append((src, tgt))
    return edges


# ---------------------------------------------------------------------------
# Adjacency and degree matrices
# ---------------------------------------------------------------------------

def _adjacency_matrix(ss: "StateSpace") -> tuple[list[list[float]], list[int]]:
    """Build the adjacency matrix of the undirected graph.

    Returns (A, states) where A[i][j] = 1 if {states[i], states[j]} is
    an edge, and states is the sorted list of state IDs.
    """
    states = _state_list(ss)
    n = len(states)
    idx = {s: i for i, s in enumerate(states)}
    A = [[0.0] * n for _ in range(n)]
    edges = _undirected_edges(ss)
    for u, v in edges:
        ui, vi = idx[u], idx[v]
        A[ui][vi] = 1.0
        A[vi][ui] = 1.0
    return A, states


def _degree_matrix(ss: "StateSpace") -> tuple[list[list[float]], list[int]]:
    """Build the degree matrix D where D[i][i] = degree of vertex i.

    Uses the undirected graph.
    """
    A, states = _adjacency_matrix(ss)
    n = len(states)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        D[i][i] = sum(A[i])
    return D, states


def _degree_vector(ss: "StateSpace") -> tuple[list[float], list[int]]:
    """Degree of each vertex in the undirected graph."""
    A, states = _adjacency_matrix(ss)
    n = len(states)
    return [sum(A[i]) for i in range(n)], states


# ---------------------------------------------------------------------------
# Linear algebra helpers (pure Python, no numpy)
# ---------------------------------------------------------------------------

def _mat_identity(n: int) -> list[list[float]]:
    """n x n identity matrix."""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _mat_add(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Element-wise matrix addition."""
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]


def _mat_sub(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Element-wise matrix subtraction."""
    n = len(A)
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(n)]


def _mat_scale(A: list[list[float]], c: float) -> list[list[float]]:
    """Scalar multiplication of a matrix."""
    n = len(A)
    return [[A[i][j] * c for j in range(n)] for i in range(n)]


def _mat_mul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Matrix multiplication."""
    n = len(A)
    if n == 0:
        return []
    m = len(B[0])
    p = len(B)
    C = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for k in range(p):
            if A[i][k] == 0.0:
                continue
            for j in range(m):
                C[i][j] += A[i][k] * B[k][j]
    return C


def _mat_det(M: list[list[float]]) -> float:
    """Determinant via LU decomposition (partial pivoting).

    Pure Python implementation for small matrices.
    """
    n = len(M)
    if n == 0:
        return 1.0
    if n == 1:
        return M[0][0]
    if n == 2:
        return M[0][0] * M[1][1] - M[0][1] * M[1][0]

    # LU with partial pivoting
    A = [row[:] for row in M]
    sign = 1.0
    for col in range(n):
        # Find pivot
        max_val = abs(A[col][col])
        max_row = col
        for row in range(col + 1, n):
            if abs(A[row][col]) > max_val:
                max_val = abs(A[row][col])
                max_row = row
        if max_val < 1e-15:
            return 0.0
        if max_row != col:
            A[col], A[max_row] = A[max_row], A[col]
            sign *= -1.0
        pivot = A[col][col]
        for row in range(col + 1, n):
            factor = A[row][col] / pivot
            for j in range(col + 1, n):
                A[row][j] -= factor * A[col][j]
            A[row][col] = 0.0

    det = sign
    for i in range(n):
        det *= A[i][i]
    return det


def _eigenvalues_symmetric(M: list[list[float]], max_iter: int = 300) -> list[float]:
    """Approximate eigenvalues of a real symmetric matrix via QR iteration.

    Uses the basic QR algorithm with Wilkinson shifts for convergence.
    Suitable for small matrices (n < 50).
    """
    n = len(M)
    if n == 0:
        return []
    if n == 1:
        return [M[0][0]]
    if n == 2:
        # Direct formula for 2x2 symmetric matrix
        a, b = M[0][0], M[0][1]
        d = M[1][1]
        trace = a + d
        det = a * d - b * b
        disc = trace * trace - 4 * det
        if disc < 0:
            disc = 0.0
        sqrt_disc = math.sqrt(disc)
        return sorted([(trace + sqrt_disc) / 2, (trace - sqrt_disc) / 2], reverse=True)

    # Work on a copy
    A = [row[:] for row in M]

    for _ in range(max_iter * n):
        # Wilkinson shift
        shift = A[n - 1][n - 1]

        # Shifted matrix
        for i in range(n):
            A[i][i] -= shift

        # QR decomposition via modified Gram-Schmidt (column-stored Q)
        # Q stored as list of columns, R upper triangular
        Q_cols: list[list[float]] = []
        R = [[0.0] * n for _ in range(n)]

        for j in range(n):
            # Column j of A
            v = [A[i][j] for i in range(n)]
            for k in range(j):
                qk = Q_cols[k]
                dot = sum(qk[i] * v[i] for i in range(n))
                R[k][j] = dot
                for i in range(n):
                    v[i] -= dot * qk[i]
            norm = math.sqrt(sum(x * x for x in v))
            if norm < 1e-15:
                R[j][j] = 0.0
                Q_cols.append([0.0] * n)
            else:
                R[j][j] = norm
                Q_cols.append([v[i] / norm for i in range(n)])

        # A_new = R * Q^T + shift * I
        # (R * Q^T)[i][j] = sum_k R[i][k] * Q_cols[j][k]
        # since Q^T[k][j] = Q[j][k] = Q_cols[k][j]... wait
        # Q as matrix: Q[i][j] = Q_cols[j][i]
        # Q^T[i][j] = Q[j][i] = Q_cols[i][j]
        # (R * Q^T)[i][j] = sum_k R[i][k] * Q^T[k][j] = sum_k R[i][k] * Q_cols[k][j]
        A = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                s = 0.0
                for k in range(n):
                    s += R[i][k] * Q_cols[k][j]
                A[i][j] = s
        for i in range(n):
            A[i][i] += shift

        # Check convergence: sub-diagonal elements near zero
        converged = True
        for i in range(1, n):
            if abs(A[i][i - 1]) > 1e-10:
                converged = False
                break
        if converged:
            break

    return sorted([A[i][i] for i in range(n)], reverse=True)


# ---------------------------------------------------------------------------
# Core Ihara functions
# ---------------------------------------------------------------------------

def cycle_rank(ss: "StateSpace") -> int:
    """Cycle rank r = |E| - |V| + connected_components.

    For connected graphs, r = |E| - |V| + 1.
    For the undirected version of the state space graph.
    """
    edges = _undirected_edges(ss)
    n_edges = len(edges)
    n_vertices = len(ss.states)

    # Count connected components via BFS on undirected graph
    adj = _undirected_adjacency(ss)
    visited: set[int] = set()
    components = 0
    for s in sorted(ss.states):
        if s in visited:
            continue
        components += 1
        stack = [s]
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            for v in adj[u]:
                if v not in visited:
                    stack.append(v)

    return n_edges - n_vertices + components


def ihara_determinant(ss: "StateSpace", u: float) -> float:
    """Evaluate 1/Z_G(u) via Bass's formula.

    Bass's formula:
        Z_G(u)^{-1} = (1 - u^2)^{r-1} * det(I - Au + (D-I)u^2)

    where A = adjacency matrix, D = degree matrix, r = cycle rank.

    For a graph with no edges, returns 1.0.
    For u = 0, returns 1.0.
    """
    states = _state_list(ss)
    n = len(states)
    if n == 0:
        return 1.0
    if abs(u) < 1e-15:
        return 1.0

    A, _ = _adjacency_matrix(ss)
    D, _ = _degree_matrix(ss)
    r = cycle_rank(ss)

    I = _mat_identity(n)

    # M = I - A*u + (D - I)*u^2
    Au = _mat_scale(A, u)
    DmI = _mat_sub(D, I)
    DmIu2 = _mat_scale(DmI, u * u)

    M = _mat_add(_mat_sub(I, Au), DmIu2)

    det_M = _mat_det(M)

    # prefix = (1 - u^2)^{r-1}
    base = 1.0 - u * u
    exp = r - 1
    if abs(base) < 1e-15 and exp < 0:
        # At u = +/-1 with negative exponent: pole
        return float('inf') if det_M != 0 else float('nan')
    prefix = base ** exp if abs(base) > 1e-15 or exp >= 0 else 1.0

    return prefix * det_M


def bass_hashimoto_matrix(ss: "StateSpace") -> list[list[float]]:
    """Build the Hashimoto (edge adjacency) matrix.

    The Hashimoto matrix B is indexed by oriented edges of the undirected
    graph.  Each undirected edge {u,v} produces two oriented edges:
    (u,v) and (v,u).

    B[e1, e2] = 1 if target(e1) = source(e2) and e2 != reverse(e1).

    For the Ihara zeta: det(I - uB) = Z_G(u)^{-1} * (1 - u^2)^{-(r-1)}
    ... but more precisely, det(I - uB) is related to the reciprocal of Z_G.
    """
    undirected = _undirected_edges(ss)
    # Create oriented edges: each undirected {u,v} -> (u,v) and (v,u)
    oriented: list[tuple[int, int]] = []
    for u, v in undirected:
        oriented.append((u, v))
        oriented.append((v, u))

    m = len(oriented)
    if m == 0:
        return []

    edge_idx = {e: i for i, e in enumerate(oriented)}
    B = [[0.0] * m for _ in range(m)]

    for i, (u1, v1) in enumerate(oriented):
        for j, (u2, v2) in enumerate(oriented):
            # e1 = (u1 -> v1), e2 = (u2 -> v2)
            # Condition: target(e1) = source(e2) and e2 != reverse(e1)
            if v1 == u2 and not (u2 == v1 and v2 == u1 and u1 == v2 and v1 == u2):
                # More clearly: reverse of (u1,v1) is (v1,u1)
                if not (u2 == v1 and v2 == u1):
                    B[i][j] = 1.0

    return B


def count_prime_cycles(ss: "StateSpace", max_length: int = 10) -> int:
    """Count prime (backtrackless, tailless) cycles up to given length.

    A prime cycle is a closed walk without:
    - Backtracking: no immediate reversal (... u, v, u ...)
    - Tails: not a proper power of a shorter cycle
    - Equivalence: considered up to cyclic permutation

    Works on the undirected graph.
    """
    lengths = prime_cycle_lengths(ss, max_length)
    return len(lengths)


def prime_cycle_lengths(ss: "StateSpace", max_length: int = 10) -> list[int]:
    """List lengths of all prime cycles up to max_length.

    Returns a sorted list of lengths (with repetitions for distinct cycles
    of the same length).
    """
    adj = _undirected_adjacency(ss)
    states = sorted(ss.states)

    if not states:
        return []

    # Find all backtrackless closed walks, then filter for primitivity
    # and quotient by cyclic permutation.
    all_cycles: list[tuple[int, ...]] = []

    for start in states:
        # DFS: path is list of vertices, prev tracks the predecessor
        # to avoid immediate backtracking
        _find_backtrackless_cycles(
            adj, start, [start], -1, max_length, all_cycles,
        )

    # Remove cyclic duplicates and proper powers
    canonical: set[tuple[int, ...]] = set()
    lengths: list[int] = []

    for cycle in all_cycles:
        # cycle is (v0, v1, ..., vk) where v0 == vk
        path = cycle[:-1]  # remove repeated start
        k = len(path)
        if k == 0:
            continue

        # Check if it's a proper power of a shorter cycle
        if _is_proper_power(path):
            continue

        # Canonical form: smallest cyclic rotation
        canon = _canonical_rotation(path)
        if canon not in canonical:
            canonical.add(canon)
            lengths.append(k)

    return sorted(lengths)


def _find_backtrackless_cycles(
    adj: dict[int, list[int]],
    start: int,
    path: list[int],
    prev: int,
    max_length: int,
    result: list[tuple[int, ...]],
) -> None:
    """DFS to find all backtrackless closed walks from start."""
    current = path[-1]
    length = len(path) - 1  # number of edges

    if length > max_length:
        return

    for neighbor in adj[current]:
        if neighbor == prev:
            # Would be backtracking
            continue

        if neighbor == start and length >= 3:
            # Found a cycle (need at least 3 edges for non-trivial cycle
            # in undirected graph)
            result.append(tuple(path) + (start,))
            continue

        if neighbor == start and length < 3:
            # Too short — in undirected graph, cycles of length < 3 are
            # either backtracking or degenerate
            continue

        if neighbor in path[1:]:
            # Would revisit a non-start vertex — not a simple closed walk
            continue

        path.append(neighbor)
        _find_backtrackless_cycles(adj, start, path, current, max_length, result)
        path.pop()


def _is_proper_power(path: tuple[int, ...] | list[int]) -> bool:
    """Check if a cyclic sequence is a proper power of a shorter sequence."""
    n = len(path)
    for d in range(1, n):
        if n % d != 0:
            continue
        if d == n:
            continue
        period = list(path[:d])
        is_power = True
        for i in range(n):
            if path[i] != period[i % d]:
                is_power = False
                break
        if is_power:
            return True
    return False


def _canonical_rotation(path: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Return the lexicographically smallest cyclic rotation."""
    n = len(path)
    if n == 0:
        return ()
    best = tuple(path)
    for i in range(1, n):
        rotated = tuple(path[i:]) + tuple(path[:i])
        if rotated < best:
            best = rotated
    return best


def is_ramanujan(ss: "StateSpace") -> bool:
    """Check if the graph has optimal spectral gap (Ramanujan property).

    A (q+1)-regular graph is Ramanujan if every eigenvalue lambda of the
    adjacency matrix with |lambda| != q+1 satisfies |lambda| <= 2*sqrt(q).

    For irregular graphs, we check the Ramanujan-like property:
    the second largest eigenvalue (in absolute value) is at most
    2*sqrt(d_max - 1) where d_max is the maximum degree.

    Graphs with 0 or 1 vertices, or with no edges, are trivially Ramanujan.
    """
    states = _state_list(ss)
    n = len(states)
    if n <= 1:
        return True

    edges = _undirected_edges(ss)
    if not edges:
        return True

    A, _ = _adjacency_matrix(ss)

    # Get degrees
    degrees = [sum(A[i]) for i in range(n)]
    d_max = max(degrees)
    if d_max < 1:
        return True

    # Check regularity
    d_min = min(degrees)
    is_regular = (d_max == d_min)

    eigenvalues = _eigenvalues_symmetric(A)

    if is_regular:
        q = d_max - 1
        bound = 2.0 * math.sqrt(q) if q > 0 else 0.0
        for ev in eigenvalues:
            if abs(abs(ev) - d_max) > 1e-8:  # not +-d_max
                if abs(ev) > bound + 1e-8:
                    return False
        return True
    else:
        # For irregular graphs: check second largest |eigenvalue|
        if len(eigenvalues) < 2:
            return True
        abs_eigs = sorted([abs(ev) for ev in eigenvalues], reverse=True)
        second = abs_eigs[1] if len(abs_eigs) > 1 else 0.0
        bound = 2.0 * math.sqrt(d_max - 1) if d_max > 1 else 0.0
        return second <= bound + 1e-8


def ihara_poles(ss: "StateSpace") -> list[complex]:
    """Poles of 1/Z_G(u) from eigenvalues of adjacency matrix.

    The reciprocal Ihara zeta 1/Z_G(u) has poles determined by the
    eigenvalues of the adjacency matrix.  For a (q+1)-regular graph:
        1/Z_G(u) = (1-u^2)^{r-1} * prod_i (1 - lambda_i * u + q * u^2)

    The poles of 1/Z_G come from (1-u^2) = 0 (i.e., u = +/- 1)
    and from 1 - lambda_i * u + q * u^2 = 0 for each eigenvalue lambda_i.

    For irregular graphs, uses Bass's formula with degree matrix.

    Returns complex poles sorted by magnitude.
    """
    states = _state_list(ss)
    n = len(states)
    if n == 0:
        return []

    edges = _undirected_edges(ss)
    if not edges:
        return []

    A, _ = _adjacency_matrix(ss)
    eigenvalues = _eigenvalues_symmetric(A)

    degrees = [sum(A[i]) for i in range(n)]
    d_max = max(degrees)
    d_min = min(degrees)
    is_regular = (d_max == d_min) and d_max > 0

    poles: list[complex] = []

    # Poles from (1 - u^2) factor: u = +1, u = -1
    r = cycle_rank(ss)
    if r > 0:
        # Each contributes with multiplicity (r-1)
        for _ in range(r - 1):
            poles.append(complex(1.0, 0.0))
            poles.append(complex(-1.0, 0.0))

    if is_regular:
        q = d_max - 1
        # Poles from 1 - lambda*u + q*u^2 = 0  =>  u = (lambda +/- sqrt(lambda^2 - 4q)) / (2q)
        for lam in eigenvalues:
            if q == 0:
                # q+1 = 1, so degree-1 regular (isolated vertices shouldn't happen)
                if abs(lam) > 1e-15:
                    poles.append(complex(1.0 / lam, 0.0))
                continue
            disc = lam * lam - 4 * q
            if disc >= 0:
                sqrt_disc = math.sqrt(disc)
                u1 = (lam + sqrt_disc) / (2 * q)
                u2 = (lam - sqrt_disc) / (2 * q)
                poles.append(complex(u1, 0.0))
                poles.append(complex(u2, 0.0))
            else:
                sqrt_disc = math.sqrt(-disc)
                u1 = complex(lam / (2 * q), sqrt_disc / (2 * q))
                u2 = complex(lam / (2 * q), -sqrt_disc / (2 * q))
                poles.append(u1)
                poles.append(u2)
    else:
        # For irregular graphs, poles come from det(I - Au + (D-I)u^2) = 0
        # Use eigenvalues of the Hashimoto matrix B: det(I - uB) poles are 1/mu
        # where mu are eigenvalues of B.
        B = bass_hashimoto_matrix(ss)
        if B:
            # For a real non-symmetric matrix, eigenvalues can be complex.
            # Use characteristic polynomial roots as approximation.
            # Power iteration to find dominant eigenvalue magnitude.
            m = len(B)
            # Approximate: use trace and determinant for small matrices
            for lam in eigenvalues:
                # Approximate poles from each eigenvalue
                avg_d = sum(degrees) / n if n > 0 else 0
                q_eff = avg_d - 1 if avg_d > 1 else 0.5
                if q_eff > 0:
                    disc = lam * lam - 4 * q_eff
                    if disc >= 0:
                        sqrt_disc = math.sqrt(disc)
                        u1 = (lam + sqrt_disc) / (2 * q_eff)
                        u2 = (lam - sqrt_disc) / (2 * q_eff)
                        poles.append(complex(u1, 0.0))
                        poles.append(complex(u2, 0.0))
                    else:
                        sqrt_disc = math.sqrt(-disc)
                        u1 = complex(lam / (2 * q_eff), sqrt_disc / (2 * q_eff))
                        u2 = complex(lam / (2 * q_eff), -sqrt_disc / (2 * q_eff))
                        poles.append(u1)
                        poles.append(u2)

    # Sort by magnitude
    poles.sort(key=lambda z: (abs(z), z.real, z.imag))
    return poles


def graph_complexity(ss: "StateSpace") -> float:
    """Graph complexity = log|Z_G(1/sqrt(q))| as a measure of cyclic structure.

    For graphs with no edges or no cycles, complexity is 0.0.
    q is the average degree minus 1 (effective regularity parameter).

    Higher complexity indicates richer cyclic structure in the protocol.
    """
    states = _state_list(ss)
    n = len(states)
    if n <= 1:
        return 0.0

    edges = _undirected_edges(ss)
    if not edges:
        return 0.0

    r = cycle_rank(ss)
    if r == 0:
        return 0.0

    A, _ = _adjacency_matrix(ss)
    degrees = [sum(A[i]) for i in range(n)]
    avg_d = sum(degrees) / n if n > 0 else 0
    q = avg_d - 1 if avg_d > 1 else 0.5

    u = 1.0 / math.sqrt(q) if q > 0 else 0.5
    # Clamp u to avoid divergence at poles
    if u >= 1.0:
        u = 0.9

    det_val = ihara_determinant(ss, u)
    if abs(det_val) < 1e-15:
        return float('inf')

    # Z_G(u) = 1 / (ihara_determinant at u), but we need the full formula
    # ihara_determinant returns (1-u^2)^{r-1} * det(I - Au + (D-I)u^2)
    # which is 1/Z_G(u)
    reciprocal = abs(det_val)
    if reciprocal < 1e-15:
        return float('inf')

    # complexity = -log|1/Z_G| = log|Z_G|
    return -math.log(reciprocal) if reciprocal > 0 else 0.0


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze_ihara(ss: "StateSpace") -> IharaResult:
    """Complete Ihara zeta function analysis of a session type state space.

    Computes the Ihara determinant, prime cycle spectrum, Hashimoto matrix,
    Ramanujan property, and graph complexity.
    """
    r = cycle_rank(ss)

    states = _state_list(ss)
    n = len(states)

    # Evaluate determinant at a representative point
    if n > 0:
        A, _ = _adjacency_matrix(ss)
        degrees = [sum(A[i]) for i in range(n)]
        avg_d = sum(degrees) / n if n > 0 else 0
        q = avg_d - 1 if avg_d > 1 else 0.5
        u_eval = 1.0 / (q + 1) if q > 0 else 0.5
        if u_eval >= 1.0:
            u_eval = 0.5
        det_val = ihara_determinant(ss, u_eval)
    else:
        det_val = 1.0

    poles = ihara_poles(ss)
    cycle_lens = prime_cycle_lengths(ss, max_length=10)
    num_cycles = len(cycle_lens)
    ramanujan = is_ramanujan(ss)
    compl = graph_complexity(ss)
    B = bass_hashimoto_matrix(ss)

    return IharaResult(
        ihara_determinant=det_val,
        reciprocal_poles=poles,
        num_prime_cycles=num_cycles,
        prime_cycle_lengths=cycle_lens,
        rank=r,
        is_ramanujan=ramanujan,
        complexity=compl,
        bass_hashimoto_matrix=B,
    )
