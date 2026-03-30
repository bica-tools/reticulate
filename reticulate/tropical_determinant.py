"""Tropical determinant analysis for session type lattices (Step 30m).

The tropical determinant of a matrix A over the max-plus semiring is:

    tdet(A) = max over permutations σ of Σᵢ A[i,σ(i)]

This is the optimal assignment problem — find the permutation maximizing
the sum of selected entries.  Key properties:

1. tdet(A) relates to the maximum weight perfect matching
2. Tropical permanent = tdet (max is idempotent, so sign does not matter)
3. Tropical singularity: A is tropically singular iff the maximum is
   achieved by multiple permutations
4. Connection to Hungarian algorithm for optimal assignment
5. Tropical adjugate: (adj A)[i,j] = tdet of A with row i, col j removed
6. Tropical Cramer's rule: solution to A ⊗ x = b
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import permutations, combinations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace

from reticulate.tropical import (
    tropical_add,
    tropical_mul,
    tropical_distance,
    _maxplus_adjacency_matrix,
    NEG_INF,
    INF,
)
from reticulate.zeta import _adjacency, _compute_sccs, _state_list, compute_rank


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TropicalDetResult:
    """Complete tropical determinant analysis.

    Attributes:
        determinant: The tropical determinant tdet(A).
        optimal_permutations: All permutations achieving tdet.
        is_singular: True iff multiple optimal permutations exist.
        num_optimal: Count of optimal permutations.
        adjugate: Tropical adjugate matrix.
        tropical_rank: Tropical rank of the matrix.
        assignment_weight: Same as determinant (assignment interpretation).
    """
    determinant: float
    optimal_permutations: list[tuple[int, ...]]
    is_singular: bool
    num_optimal: int
    adjugate: list[list[float]]
    tropical_rank: int
    assignment_weight: float


# ---------------------------------------------------------------------------
# Core: tropical determinant
# ---------------------------------------------------------------------------

def tropical_det(matrix: list[list[float]]) -> float:
    """Tropical determinant: max over σ of Σᵢ A[i,σ(i)].

    In the max-plus semiring, the determinant equals the permanent
    because max is idempotent (signs do not matter).

    For an empty matrix, returns 0.0 (tropical multiplicative identity).
    For a 1×1 matrix, returns the single entry.
    """
    n = len(matrix)
    if n == 0:
        return 0.0
    if n == 1:
        return matrix[0][0]

    # For small n (≤ 8) enumerate all permutations
    if n <= 8:
        return _max_perm_sum_direct(matrix, n)

    # For larger n, use Hungarian algorithm
    weight, _ = hungarian_assignment(matrix)
    return weight


def _max_perm_sum_direct(matrix: list[list[float]], n: int) -> float:
    """Find max-weight perfect matching by enumerating permutations."""
    best = NEG_INF
    for perm in permutations(range(n)):
        total = 0.0
        feasible = True
        for i, j in enumerate(perm):
            if matrix[i][j] == NEG_INF:
                feasible = False
                break
            total += matrix[i][j]
        if feasible:
            best = max(best, total)
        elif not feasible and best == NEG_INF:
            # Track even infeasible sums for comparison
            pass
    # If no feasible permutation, return NEG_INF
    if best == NEG_INF:
        # Check if ALL permutations hit -inf
        return NEG_INF
    return best


def tropical_det_from_ss(ss: "StateSpace") -> float:
    """Tropical determinant of the max-plus adjacency matrix."""
    A = _maxplus_adjacency_matrix(ss)
    return tropical_det(A)


# ---------------------------------------------------------------------------
# Optimal permutations
# ---------------------------------------------------------------------------

def optimal_permutations(matrix: list[list[float]]) -> list[tuple[int, ...]]:
    """Find all permutations achieving the tropical determinant.

    Returns a list of tuples where tuple[i] = column assigned to row i.
    """
    n = len(matrix)
    if n == 0:
        return [()]

    det_val = tropical_det(matrix)
    if det_val == NEG_INF:
        # No feasible permutation — return empty
        return []

    result: list[tuple[int, ...]] = []
    for perm in permutations(range(n)):
        total = 0.0
        feasible = True
        for i, j in enumerate(perm):
            if matrix[i][j] == NEG_INF:
                feasible = False
                break
            total += matrix[i][j]
        if feasible and math.isclose(total, det_val, abs_tol=1e-12):
            result.append(perm)

    return result


# ---------------------------------------------------------------------------
# Singularity
# ---------------------------------------------------------------------------

def is_tropically_singular(matrix: list[list[float]]) -> bool:
    """Check if multiple permutations achieve the maximum.

    A tropical matrix is singular iff the optimal assignment is not unique.
    """
    opts = optimal_permutations(matrix)
    return len(opts) != 1


# ---------------------------------------------------------------------------
# Minor operations
# ---------------------------------------------------------------------------

def _submatrix(
    matrix: list[list[float]], exclude_rows: set[int], exclude_cols: set[int]
) -> list[list[float]]:
    """Extract submatrix excluding specified rows and columns."""
    n = len(matrix)
    return [
        [matrix[i][j] for j in range(n) if j not in exclude_cols]
        for i in range(n) if i not in exclude_rows
    ]


def tropical_minor(matrix: list[list[float]], row: int, col: int) -> float:
    """Tropical determinant of the (n-1)×(n-1) submatrix with row and col removed."""
    sub = _submatrix(matrix, {row}, {col})
    return tropical_det(sub)


def all_tropical_minors(matrix: list[list[float]], k: int) -> list[float]:
    """All k×k minors of the matrix.

    A k×k minor is the tropical determinant of a k×k submatrix
    formed by choosing k rows and k columns.

    Returns a list of tropical determinants, one per (row_set, col_set) pair.
    """
    n = len(matrix)
    if k <= 0 or k > n:
        return []

    minors: list[float] = []
    for rows in combinations(range(n), k):
        for cols in combinations(range(n), k):
            sub = [[matrix[r][c] for c in cols] for r in rows]
            minors.append(tropical_det(sub))
    return minors


# ---------------------------------------------------------------------------
# Adjugate and Cramer
# ---------------------------------------------------------------------------

def tropical_adjugate(matrix: list[list[float]]) -> list[list[float]]:
    """Tropical adjugate: adj[i][j] = tdet of minor(j, i).

    Note the transpose: adj[i][j] uses minor with row j, col i removed.
    This mirrors the classical adjugate (cofactor) matrix.
    """
    n = len(matrix)
    if n == 0:
        return []
    if n == 1:
        return [[0.0]]

    adj: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            adj[i][j] = tropical_minor(matrix, j, i)
    return adj


def tropical_cramer(
    A: list[list[float]], b: list[float]
) -> list[float] | None:
    """Solve A ⊗ x = b via tropical Cramer's rule if possible.

    In the max-plus semiring, A ⊗ x = b means:
        max_j (A[i][j] + x[j]) = b[i]  for all i.

    Tropical Cramer's rule gives:
        x[j] = (adj(A) ⊗ b)[j] ⊘ tdet(A)
             = max_i (adj[j][i] + b[i]) - tdet(A)

    Returns None if tdet(A) = -inf (no feasible assignment).
    """
    n = len(A)
    if n == 0:
        return []
    if len(b) != n:
        return None

    det = tropical_det(A)
    if det == NEG_INF:
        return None

    adj = tropical_adjugate(A)

    x: list[float] = []
    for j in range(n):
        # x[j] = max_i (adj[j][i] + b[i]) - det
        best = NEG_INF
        for i in range(n):
            if adj[j][i] != NEG_INF and b[i] != NEG_INF:
                best = max(best, adj[j][i] + b[i])
        if best == NEG_INF:
            x.append(NEG_INF)
        else:
            x.append(best - det)
    return x


# ---------------------------------------------------------------------------
# Tropical rank
# ---------------------------------------------------------------------------

def tropical_rank(matrix: list[list[float]]) -> int:
    """Tropical rank: max k such that some k×k submatrix has finite tdet.

    The tropical rank is the largest k for which there exists a
    k×k submatrix with a finite (not -inf) tropical determinant.
    """
    n = len(matrix)
    if n == 0:
        return 0

    # Check from k = n down to 1
    for k in range(n, 0, -1):
        for rows in combinations(range(n), k):
            for cols in combinations(range(n), k):
                sub = [[matrix[r][c] for c in cols] for r in rows]
                det = tropical_det(sub)
                if det != NEG_INF:
                    return k
    return 0


# ---------------------------------------------------------------------------
# Hungarian algorithm (max-weight assignment)
# ---------------------------------------------------------------------------

def hungarian_assignment(
    cost_matrix: list[list[float]],
) -> tuple[float, list[int]]:
    """Hungarian algorithm for max-weight perfect assignment.

    Adapts the classic Hungarian (Kuhn-Munkres) algorithm to the
    max-weight setting.  Entries of -inf represent forbidden assignments.

    Returns:
        (total_weight, assignment) where assignment[i] = column for row i.
        Returns (NEG_INF, []) if no feasible assignment exists.
    """
    n = len(cost_matrix)
    if n == 0:
        return (0.0, [])
    if n == 1:
        return (cost_matrix[0][0], [0])

    # Use the O(n^3) Hungarian algorithm for maximisation.
    # We work with a (n+1)×(n+1) 1-indexed matrix.

    C = cost_matrix  # alias for readability

    # Replace -inf with a very large negative number for arithmetic
    LARGE_NEG = -1e15

    # u[i] = potential for worker i, v[j] = potential for job j
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    # p[j] = worker assigned to job j (0 = unassigned)
    p = [0] * (n + 1)
    # way[j] = previous job in alternating path
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        # Assign worker i
        p[0] = i
        j0 = 0  # virtual job
        min_v = [INF] * (n + 1)
        used = [False] * (n + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1

            for j in range(1, n + 1):
                if used[j]:
                    continue
                val = C[i0 - 1][j - 1] if C[i0 - 1][j - 1] != NEG_INF else LARGE_NEG
                # For maximisation: reduced cost = u[i0] + v[j] - val
                cur = u[i0] + v[j] - val
                if cur < min_v[j]:
                    min_v[j] = cur
                    way[j] = j0
                if min_v[j] < delta:
                    delta = min_v[j]
                    j1 = j

            if j1 == -1:
                break

            for j in range(n + 1):
                if used[j]:
                    u[p[j]] -= delta
                    v[j] += delta
                else:
                    min_v[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        # Augment along the path
        while j0 != 0:
            p[j0] = p[way[j0]]
            j0 = way[j0]

    # Build assignment: for each worker i, find which job j has p[j] = i
    assignment = [0] * n
    total = 0.0
    feasible = True
    for j in range(1, n + 1):
        worker = p[j] - 1
        assignment[worker] = j - 1
    for i in range(n):
        if cost_matrix[i][assignment[i]] == NEG_INF:
            feasible = False
            break
        total += cost_matrix[i][assignment[i]]

    if not feasible:
        return (NEG_INF, [])

    return (total, assignment)


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze_tropical_determinant(ss: "StateSpace") -> TropicalDetResult:
    """Complete tropical determinant analysis of a session type state space.

    Builds the max-plus adjacency matrix and computes:
    - Tropical determinant (= optimal assignment weight)
    - All optimal permutations
    - Singularity
    - Tropical adjugate
    - Tropical rank
    """
    A = _maxplus_adjacency_matrix(ss)
    n = len(A)

    det = tropical_det(A)
    opts = optimal_permutations(A) if n <= 8 else []
    if n > 8:
        # For large matrices just check if singular via Hungarian
        weight1, assign1 = hungarian_assignment(A)
        opts = [tuple(assign1)] if assign1 else []

    sing = len(opts) != 1
    adj = tropical_adjugate(A) if n <= 10 else [[NEG_INF] * n for _ in range(n)]
    rank = tropical_rank(A) if n <= 10 else n  # approximate for large

    return TropicalDetResult(
        determinant=det,
        optimal_permutations=opts,
        is_singular=sing,
        num_optimal=len(opts),
        adjugate=adj,
        tropical_rank=rank,
        assignment_weight=det,
    )
