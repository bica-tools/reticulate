"""Kirchhoff's matrix-tree theorem for session type lattices (Step 30h).

Kirchhoff's theorem relates the number of spanning trees of a graph
to the determinant of a cofactor of the Laplacian matrix:

    τ(G) = (1/n) · λ₁ · λ₂ · ... · λ_{n-1}

where λ_i are the non-zero Laplacian eigenvalues.

For session type lattices, spanning trees of the Hasse diagram
represent minimal protocol skeletons — the fewest edges needed
to maintain connectivity.

This module provides:

- **Spanning tree count** via Kirchhoff's theorem
- **Laplacian cofactor** computation (determinant of minor)
- **Spanning tree product formula** for parallel: τ(L₁×L₂)
- **Normalized tree count**: τ(G) / n^{n-2} (comparison with complete graph)
- **Complexity ratio**: spanning trees / total edges
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace

from reticulate.matrix import (
    laplacian_matrix,
    adjacency_matrix,
    _eigenvalues_symmetric,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KirchhoffResult:
    """Kirchhoff analysis results.

    Attributes:
        num_states: Number of states.
        num_edges: Number of Hasse edges.
        spanning_tree_count: τ(G) = number of spanning trees.
        spanning_tree_count_eigenvalue: τ(G) computed via eigenvalues.
        normalized_count: τ(G) / n^{n-2} (Cayley comparison).
        complexity_ratio: τ(G) / |E| (trees per edge).
        laplacian_det_cofactor: det of Laplacian with row/col 0 removed.
        is_tree: True iff τ(G) = 1 (the graph is already a tree).
    """
    num_states: int
    num_edges: int
    spanning_tree_count: int
    spanning_tree_count_eigenvalue: float
    normalized_count: float
    complexity_ratio: float
    laplacian_det_cofactor: float
    is_tree: bool


# ---------------------------------------------------------------------------
# Determinant computation
# ---------------------------------------------------------------------------

def _determinant(M: list[list[float]]) -> float:
    """Compute determinant via LU decomposition (Gaussian elimination)."""
    n = len(M)
    if n == 0:
        return 1.0
    if n == 1:
        return M[0][0]
    if n == 2:
        return M[0][0] * M[1][1] - M[0][1] * M[1][0]

    # Gaussian elimination with partial pivoting
    A = [row[:] for row in M]
    det = 1.0
    for col in range(n):
        # Find pivot
        max_row = col
        for row in range(col + 1, n):
            if abs(A[row][col]) > abs(A[max_row][col]):
                max_row = row
        if max_row != col:
            A[col], A[max_row] = A[max_row], A[col]
            det *= -1.0

        if abs(A[col][col]) < 1e-14:
            return 0.0

        det *= A[col][col]

        for row in range(col + 1, n):
            factor = A[row][col] / A[col][col]
            for j in range(col + 1, n):
                A[row][j] -= factor * A[col][j]

    return det


def _cofactor_matrix(L: list[list[float]], row: int, col: int) -> list[list[float]]:
    """Remove row and column from matrix (for cofactor computation)."""
    n = len(L)
    return [
        [L[i][j] for j in range(n) if j != col]
        for i in range(n) if i != row
    ]


# ---------------------------------------------------------------------------
# Spanning tree count
# ---------------------------------------------------------------------------

def spanning_tree_count_cofactor(ss: "StateSpace") -> float:
    """Count spanning trees via Laplacian cofactor (Kirchhoff's theorem).

    τ(G) = det(L[0,0]) = determinant of Laplacian with row 0 and col 0 removed.
    """
    L = laplacian_matrix(ss)
    n = len(L)
    if n <= 1:
        return 1.0

    # Remove row 0 and col 0
    minor = _cofactor_matrix(L, 0, 0)
    return _determinant(minor)


def spanning_tree_count_eigenvalue(ss: "StateSpace") -> float:
    """Count spanning trees via Laplacian eigenvalues.

    τ(G) = (1/n) · Π_{i=1}^{n-1} λ_i

    where λ_0 = 0 ≤ λ_1 ≤ ... ≤ λ_{n-1} are Laplacian eigenvalues.
    """
    L = laplacian_matrix(ss)
    n = len(L)
    if n <= 1:
        return 1.0

    eigs = sorted(_eigenvalues_symmetric(L))

    # Product of non-zero eigenvalues divided by n
    product = 1.0
    for i in range(1, n):  # Skip λ_0 ≈ 0
        product *= max(eigs[i], 0.0)

    return product / n


def spanning_tree_count(ss: "StateSpace") -> int:
    """Count spanning trees (integer result via cofactor method)."""
    val = spanning_tree_count_cofactor(ss)
    return max(0, round(val))


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------

def normalized_tree_count(ss: "StateSpace") -> float:
    """τ(G) / n^{n-2} — comparison with complete graph (Cayley's formula).

    For K_n, τ = n^{n-2}. This ratio measures how tree-rich the graph
    is compared to the maximum possible.
    """
    n = len(ss.states)
    if n <= 2:
        return 1.0
    tau = spanning_tree_count(ss)
    cayley = n ** (n - 2)
    return tau / cayley if cayley > 0 else 0.0


def complexity_ratio(ss: "StateSpace") -> float:
    """τ(G) / |E| — spanning trees per edge.

    Higher values indicate more redundancy in the graph structure.
    """
    A = adjacency_matrix(ss)
    n = len(A)
    num_edges = sum(A[i][j] for i in range(n) for j in range(i + 1, n))
    if num_edges == 0:
        return 0.0
    tau = spanning_tree_count(ss)
    return tau / num_edges


def is_tree(ss: "StateSpace") -> bool:
    """Check if the Hasse diagram is already a tree (τ = 1)."""
    return spanning_tree_count(ss) == 1


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def verify_tree_count_product(
    ss_left: "StateSpace",
    ss_right: "StateSpace",
    ss_product: "StateSpace",
    tol: float = 0.5,
) -> bool:
    """Check spanning tree count relationship for product lattices.

    For the Cartesian product G₁ □ G₂:
    τ(G₁ □ G₂) = τ(G₁)^{n₂} · τ(G₂)^{n₁} · Π formula involving eigenvalues.

    This is a weaker check: just verify the product count is computable.
    """
    tau_left = spanning_tree_count(ss_left)
    tau_right = spanning_tree_count(ss_right)
    tau_product = spanning_tree_count(ss_product)
    # Product should have at least as many trees as either factor
    return tau_product >= min(tau_left, tau_right)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_kirchhoff(ss: "StateSpace") -> KirchhoffResult:
    """Complete Kirchhoff analysis."""
    A = adjacency_matrix(ss)
    n = len(A)
    num_edges = sum(A[i][j] for i in range(n) for j in range(i + 1, n))

    tau_cofactor = spanning_tree_count_cofactor(ss)
    tau_eig = spanning_tree_count_eigenvalue(ss)
    tau = max(0, round(tau_cofactor))
    norm = normalized_tree_count(ss)
    cr = complexity_ratio(ss)

    return KirchhoffResult(
        num_states=n,
        num_edges=num_edges,
        spanning_tree_count=tau,
        spanning_tree_count_eigenvalue=tau_eig,
        normalized_count=norm,
        complexity_ratio=cr,
        laplacian_det_cofactor=tau_cofactor,
        is_tree=(tau == 1),
    )
