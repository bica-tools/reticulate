"""Smith Normal Form analysis for session type lattices (Step 30ae).

The Smith Normal Form (SNF) of an integer matrix A is a diagonal matrix
D = U @ A @ V where U, V are unimodular (det = +/-1) integer matrices
and D = diag(d1, d2, ..., dk, 0, ..., 0) with d1 | d2 | ... | dk
(each invariant factor divides the next).

For session type lattices we apply SNF to three matrices derived from
the state space (quotient DAG after SCC collapse):

1. **Adjacency matrix** — reveals rank, nullity, torsion of the
   directed graph as a Z-module.
2. **Incidence matrix** (states x edges, signed) — connects to
   simplicial / chain-complex homology of the state space.
3. **Laplacian matrix** L = D_out - A — its SNF yields the critical
   group (sandpile group / chip-firing group), whose order equals the
   number of spanning trees by the Matrix Tree Theorem.

All computations use exact integer arithmetic (no floating point).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from reticulate.zeta import (
    _adjacency,
    _compute_sccs,
    _state_list,
    compute_rank,
)

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SmithNormalFormResult:
    """Complete SNF analysis for a session type state space.

    Attributes:
        adjacency_snf: Invariant factors of the adjacency matrix (non-zero diagonal entries).
        laplacian_snf: Invariant factors of the Laplacian matrix.
        incidence_snf: Invariant factors of the incidence matrix.
        critical_group: Non-unit factors of the reduced Laplacian SNF.
        critical_group_order: Product of critical_group factors (= number of spanning trees).
        matrix_rank: Rank of the adjacency matrix.
        nullity: Nullity of the adjacency matrix (n - rank).
    """
    adjacency_snf: list[int]
    laplacian_snf: list[int]
    incidence_snf: list[int]
    critical_group: list[int]
    critical_group_order: int
    matrix_rank: int
    nullity: int


# ---------------------------------------------------------------------------
# Helpers: deep copy, extended GCD
# ---------------------------------------------------------------------------

def _deep_copy(matrix: list[list[int]]) -> list[list[int]]:
    """Return a deep copy of a 2D integer matrix."""
    return [row[:] for row in matrix]


def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Extended Euclidean algorithm.

    Returns (g, x, y) such that a*x + b*y = g = gcd(a, b).
    """
    if b == 0:
        return (a, 1, 0) if a >= 0 else (-a, -1, 0)
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    # Ensure gcd is positive
    if old_r < 0:
        old_r, old_s, old_t = -old_r, -old_s, -old_t
    return old_r, old_s, old_t


def _identity(n: int) -> list[list[int]]:
    """Return an n x n identity matrix."""
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


# ---------------------------------------------------------------------------
# Matrix construction from state space
# ---------------------------------------------------------------------------

def _quotient_data(ss: "StateSpace") -> tuple[list[int], dict[int, int], dict[int, set[int]]]:
    """Return (sorted representatives, scc_map, scc_members) for the SCC quotient."""
    scc_map, scc_members = _compute_sccs(ss)
    reps = sorted(scc_members.keys())
    return reps, scc_map, scc_members


def _quotient_edges(
    ss: "StateSpace",
    reps: list[int],
    scc_map: dict[int, int],
) -> list[tuple[int, int]]:
    """Return deduplicated directed edges of the SCC quotient DAG."""
    adj = _adjacency(ss)
    edge_set: set[tuple[int, int]] = set()
    for s in ss.states:
        sr = scc_map[s]
        for t in adj[s]:
            tr = scc_map[t]
            if sr != tr:
                edge_set.add((sr, tr))
    return sorted(edge_set)


def adjacency_matrix(ss: "StateSpace") -> list[list[int]]:
    """Build adjacency matrix on SCC quotient.

    A[i][j] = number of quotient edges from rep_i to rep_j.
    For a simple DAG quotient, entries are 0 or 1.
    """
    reps, scc_map, _ = _quotient_data(ss)
    n = len(reps)
    idx = {r: i for i, r in enumerate(reps)}
    A = [[0] * n for _ in range(n)]

    adj = _adjacency(ss)
    for s in ss.states:
        sr = scc_map[s]
        for t in adj[s]:
            tr = scc_map[t]
            if sr != tr:
                si, ti = idx[sr], idx[tr]
                A[si][ti] = 1  # Quotient: at most one edge per pair
    return A


def laplacian_matrix(ss: "StateSpace") -> list[list[int]]:
    """Build Laplacian L = D_out - A on SCC quotient.

    L[i][i] = out-degree of rep_i in quotient.
    L[i][j] = -A[i][j] for i != j.
    """
    A = adjacency_matrix(ss)
    n = len(A)
    L = [[0] * n for _ in range(n)]
    for i in range(n):
        out_deg = sum(A[i])
        for j in range(n):
            if i == j:
                L[i][j] = out_deg
            else:
                L[i][j] = -A[i][j]
    return L


def incidence_matrix(ss: "StateSpace") -> list[list[int]]:
    """Build signed incidence matrix (states x edges) on SCC quotient.

    For each directed edge (u, v) in the quotient DAG, column e has:
        M[u][e] = +1  (tail)
        M[v][e] = -1  (head)
    """
    reps, scc_map, _ = _quotient_data(ss)
    n = len(reps)
    idx = {r: i for i, r in enumerate(reps)}
    edges = _quotient_edges(ss, reps, scc_map)
    m = len(edges)

    if m == 0:
        # No edges: return n x 0 matrix (empty columns)
        return [[] for _ in range(n)]

    M = [[0] * m for _ in range(n)]
    for e_idx, (u, v) in enumerate(edges):
        M[idx[u]][e_idx] = 1
        M[idx[v]][e_idx] = -1
    return M


# ---------------------------------------------------------------------------
# Smith Normal Form computation
# ---------------------------------------------------------------------------

def smith_normal_form(
    matrix: list[list[int]],
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """Compute Smith Normal Form: returns (U, D, V) where D = U @ A @ V.

    All matrices are integer.  U is m x m, D is m x n, V is n x n.
    U and V are unimodular (det = +/-1).

    Algorithm: iterative pivot selection with row/column elimination
    using the extended GCD to ensure the divisibility chain.
    """
    if not matrix or not matrix[0]:
        m = len(matrix)
        n = len(matrix[0]) if matrix else 0
        return _identity(m), _deep_copy(matrix), _identity(n)

    m = len(matrix)
    n = len(matrix[0])

    D = _deep_copy(matrix)
    U = _identity(m)
    V = _identity(n)

    # Process each diagonal position
    size = min(m, n)
    for k in range(size):
        # Iterate until the (k,k) entry divides all entries in its row and column
        # and all entries below/right of row k, col k are zero.
        max_iterations = 100 * (m + n)  # Safety bound
        for _iter in range(max_iterations):
            # Find the smallest non-zero entry in the submatrix D[k:, k:]
            pivot_val = 0
            pivot_r, pivot_c = -1, -1
            for i in range(k, m):
                for j in range(k, n):
                    if D[i][j] != 0:
                        av = abs(D[i][j])
                        if pivot_val == 0 or av < pivot_val:
                            pivot_val = av
                            pivot_r, pivot_c = i, j

            if pivot_val == 0:
                break  # Rest of submatrix is zero

            # Move pivot to (k, k)
            if pivot_r != k:
                _swap_rows(D, k, pivot_r)
                _swap_rows(U, k, pivot_r)
            if pivot_c != k:
                _swap_cols(D, k, pivot_c)
                _swap_cols(V, k, pivot_c)

            # Alternately eliminate column and row until both are clean
            _eliminate_column(D, U, k, m)
            _eliminate_row(D, V, k, n)

            # Check if both column k and row k are clean
            col_clean = all(D[i][k] == 0 for i in range(k + 1, m))
            row_clean = all(D[k][j] == 0 for j in range(k + 1, n))
            if not col_clean or not row_clean:
                continue

            # Both row and column are clean
            # Check divisibility: D[k][k] must divide all entries in D[k+1:, k+1:]
            if not _check_divisibility(D, k, m, n):
                _fix_divisibility(D, U, V, k, m, n)
                continue

            # Make diagonal entry positive
            if D[k][k] < 0:
                for j in range(n):
                    D[k][j] = -D[k][j]
                for jj in range(m):
                    U[k][jj] = -U[k][jj]
            break

    return U, D, V


def _swap_rows(M: list[list[int]], i: int, j: int) -> None:
    """Swap rows i and j in-place."""
    M[i], M[j] = M[j], M[i]


def _swap_cols(M: list[list[int]], i: int, j: int) -> None:
    """Swap columns i and j in-place."""
    for row in M:
        row[i], row[j] = row[j], row[i]


def _eliminate_column(
    D: list[list[int]], U: list[list[int]], k: int, m: int,
) -> bool:
    """Eliminate entries in column k below row k using extended GCD.

    Returns True if any entry was changed.
    """
    changed = False
    for i in range(k + 1, m):
        if D[i][k] == 0:
            continue
        changed = True
        g, x, y = _extended_gcd(D[k][k], D[i][k])
        # Row operation: new_row_k = x * row_k + y * row_i
        #                new_row_i = -(D[i][k]/g) * row_k + (D[k][k]/g) * row_i
        a, b = D[k][k], D[i][k]
        a_over_g = a // g
        b_over_g = b // g

        n = len(D[0])
        new_k = [x * D[k][j] + y * D[i][j] for j in range(n)]
        new_i = [-b_over_g * D[k][j] + a_over_g * D[i][j] for j in range(n)]
        D[k] = new_k
        D[i] = new_i

        # Apply same to U
        cols_u = len(U[0])
        new_uk = [x * U[k][j] + y * U[i][j] for j in range(cols_u)]
        new_ui = [-b_over_g * U[k][j] + a_over_g * U[i][j] for j in range(cols_u)]
        U[k] = new_uk
        U[i] = new_ui
    return changed


def _eliminate_row(
    D: list[list[int]], V: list[list[int]], k: int, n: int,
) -> bool:
    """Eliminate entries in row k right of column k.

    If D[k][k] divides D[k][j], use simple column subtraction to avoid
    disturbing already-zeroed entries in column k.
    Otherwise, use extended GCD column operation.

    Returns True if any entry was changed.
    """
    changed = False
    for j in range(k + 1, n):
        if D[k][j] == 0:
            continue
        changed = True

        if D[k][k] != 0 and D[k][j] % D[k][k] == 0:
            # Simple case: col_j -= q * col_k (does NOT modify col_k)
            q = D[k][j] // D[k][k]
            m = len(D)
            for i in range(m):
                D[i][j] -= q * D[i][k]
            rows_v = len(V)
            for i in range(rows_v):
                V[i][j] -= q * V[i][k]
        else:
            # General case: extended GCD column operation
            g, x, y = _extended_gcd(D[k][k], D[k][j])
            a, b = D[k][k], D[k][j]
            a_over_g = a // g
            b_over_g = b // g

            m = len(D)
            for i in range(m):
                old_k, old_j = D[i][k], D[i][j]
                D[i][k] = x * old_k + y * old_j
                D[i][j] = -b_over_g * old_k + a_over_g * old_j

            rows_v = len(V)
            for i in range(rows_v):
                old_k, old_j = V[i][k], V[i][j]
                V[i][k] = x * old_k + y * old_j
                V[i][j] = -b_over_g * old_k + a_over_g * old_j
    return changed


def _check_divisibility(
    D: list[list[int]], k: int, m: int, n: int,
) -> bool:
    """Check that D[k][k] divides all entries in D[k+1:, k+1:]."""
    d = D[k][k]
    if d == 0:
        return True
    for i in range(k + 1, m):
        for j in range(k + 1, n):
            if D[i][j] % d != 0:
                return False
    return True


def _fix_divisibility(
    D: list[list[int]],
    U: list[list[int]],
    V: list[list[int]],
    k: int,
    m: int,
    n: int,
) -> None:
    """Fix divisibility by adding a row with a problematic entry to row k.

    Find an entry D[i][j] in the submatrix that D[k][k] does not divide,
    then add row i to row k so that the next elimination round will
    produce gcd(D[k][k], D[i][j]) as the new pivot.
    """
    d = D[k][k]
    for i in range(k + 1, m):
        for j in range(k + 1, n):
            if D[i][j] % d != 0:
                # Add row i to row k
                cols = len(D[0])
                for c in range(cols):
                    D[k][c] += D[i][c]
                cols_u = len(U[0])
                for c in range(cols_u):
                    U[k][c] += U[i][c]
                return


# ---------------------------------------------------------------------------
# Invariant factors extraction
# ---------------------------------------------------------------------------

def invariant_factors(matrix: list[list[int]]) -> list[int]:
    """Extract invariant factors (non-zero diagonal entries of SNF).

    Returns the list [d1, d2, ..., dk] where d1 | d2 | ... | dk > 0.
    """
    if not matrix or not matrix[0]:
        return []

    _, D, _ = smith_normal_form(matrix)
    m = len(D)
    n = len(D[0])
    factors: list[int] = []
    for i in range(min(m, n)):
        d = D[i][i]
        if d != 0:
            factors.append(abs(d))

    # Sort to ensure divisibility order
    factors.sort()
    return factors


# ---------------------------------------------------------------------------
# Critical group (sandpile group)
# ---------------------------------------------------------------------------

def critical_group(ss: "StateSpace") -> list[int]:
    """Compute the critical group (sandpile group) from reduced Laplacian SNF.

    The critical group is the torsion part of the cokernel of the Laplacian.
    We delete one row and column (the bottom/sink state) to get the reduced
    Laplacian, compute its SNF, and return the non-unit invariant factors.

    Returns list of factors > 1, e.g. [2, 6] means Z/2Z x Z/6Z.
    """
    L = laplacian_matrix(ss)
    n = len(L)
    if n <= 1:
        return []

    # Identify the bottom state's index in the quotient
    reps, scc_map, _ = _quotient_data(ss)
    idx = {r: i for i, r in enumerate(reps)}
    bottom = ss.bottom if hasattr(ss, "bottom") else min(ss.states)
    bottom_rep = scc_map.get(bottom, reps[-1] if reps else 0)
    bottom_idx = idx.get(bottom_rep, n - 1)

    # Build reduced Laplacian: delete row and column of bottom
    reduced = []
    for i in range(n):
        if i == bottom_idx:
            continue
        row = []
        for j in range(n):
            if j == bottom_idx:
                continue
            row.append(L[i][j])
        reduced.append(row)

    if not reduced or not reduced[0]:
        return []

    factors = invariant_factors(reduced)
    # Return only factors > 1 (non-unit)
    return [f for f in factors if f > 1]


def critical_group_order(ss: "StateSpace") -> int:
    """Order of the critical group = number of spanning trees (Matrix Tree Theorem).

    Equal to the product of all non-zero invariant factors of the reduced
    Laplacian, which also equals any cofactor of the Laplacian (Kirchhoff).
    """
    factors = critical_group(ss)
    if not factors:
        # Compute from reduced Laplacian determinant
        L = laplacian_matrix(ss)
        n = len(L)
        if n <= 1:
            return 1

        reps, scc_map, _ = _quotient_data(ss)
        idx = {r: i for i, r in enumerate(reps)}
        bottom = ss.bottom if hasattr(ss, "bottom") else min(ss.states)
        bottom_rep = scc_map.get(bottom, reps[-1] if reps else 0)
        bottom_idx = idx.get(bottom_rep, n - 1)

        reduced = []
        for i in range(n):
            if i == bottom_idx:
                continue
            row = []
            for j in range(n):
                if j == bottom_idx:
                    continue
                row.append(L[i][j])
            reduced.append(row)

        if not reduced:
            return 1

        det = _determinant(reduced)
        return abs(det) if det != 0 else 1

    result = 1
    for f in factors:
        result *= f
    return result


def _determinant(matrix: list[list[int]]) -> int:
    """Compute determinant of a square integer matrix using SNF.

    det(A) = det(U)^{-1} * prod(diag(D)) * det(V)^{-1}.
    Since U, V are unimodular (det = +/-1), det(A) = +/- prod(diag(D)).
    We compute it directly via row reduction to avoid sign tracking issues.
    """
    n = len(matrix)
    if n == 0:
        return 1
    if n == 1:
        return matrix[0][0]

    # Use fraction-free Gaussian elimination (Bareiss algorithm)
    M = _deep_copy(matrix)
    sign = 1

    for k in range(n):
        # Find pivot
        pivot_row = -1
        for i in range(k, n):
            if M[i][k] != 0:
                pivot_row = i
                break
        if pivot_row == -1:
            return 0  # Singular

        if pivot_row != k:
            M[k], M[pivot_row] = M[pivot_row], M[k]
            sign = -sign

        for i in range(k + 1, n):
            for j in range(k + 1, n):
                M[i][j] = M[k][k] * M[i][j] - M[i][k] * M[k][j]
                if k > 0:
                    M[i][j] //= M[k - 1][k - 1] if M[k - 1][k - 1] != 0 else 1
            M[i][k] = 0

    return sign * M[n - 1][n - 1]


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze_smith_normal_form(ss: "StateSpace") -> SmithNormalFormResult:
    """Full SNF analysis of a session type state space.

    Computes invariant factors of the adjacency, Laplacian, and incidence
    matrices on the SCC quotient DAG, derives the critical group and
    matrix rank/nullity.
    """
    A = adjacency_matrix(ss)
    L = laplacian_matrix(ss)
    Inc = incidence_matrix(ss)

    adj_factors = invariant_factors(A)
    lap_factors = invariant_factors(L)
    inc_factors = invariant_factors(Inc)

    cg = critical_group(ss)
    cg_order = critical_group_order(ss)

    n = len(A)
    rank = len(adj_factors)
    nullity = n - rank

    return SmithNormalFormResult(
        adjacency_snf=adj_factors,
        laplacian_snf=lap_factors,
        incidence_snf=inc_factors,
        critical_group=cg,
        critical_group_order=cg_order,
        matrix_rank=rank,
        nullity=nullity,
    )
