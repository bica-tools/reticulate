"""Von Neumann entropy analysis for session type lattices (Step 30r).

The von Neumann entropy of a graph measures its structural complexity
via the normalized Laplacian.  Given a graph G with n vertices:

1. **Graph Laplacian**: L = D - A, where D is the degree matrix and
   A is the adjacency matrix of the undirected version of G.

2. **Density matrix**: rho = L / tr(L)  (normalized Laplacian, trace 1).

3. **Von Neumann entropy**: S(rho) = -tr(rho log rho) = -sum_i lambda_i log(lambda_i)
   where lambda_i are eigenvalues of rho.

4. **Range**: 0 <= S(rho) <= log(n).
   - S = 0 for star graphs (maximally centralized).
   - S = log(n) for complete graphs (maximally distributed).

5. **Renyi entropy**: S_alpha(rho) = (1/(1-alpha)) log(tr(rho^alpha)).

6. **Quantum relative entropy**: S(rho||sigma) = tr(rho(log rho - log sigma)).

For session types, the entropy measures protocol complexity: simple
linear protocols have low entropy, while highly branching protocols
with symmetric structure have high entropy.

Key results for session types:
- ``end`` has entropy 0 (single state).
- Linear chains have moderate entropy.
- Parallel composition increases entropy (product lattice).
- Symmetric branching maximizes entropy for a given number of states.
- Entropy is monotone under lattice embeddings.

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
class VonNeumannResult:
    """Complete von Neumann entropy analysis for a session type state space.

    Attributes:
        entropy: Von Neumann entropy S(rho).
        max_entropy: log(n) -- maximum possible entropy.
        normalized_entropy: S(rho)/log(n) in [0, 1].
        density_eigenvalues: Eigenvalues of the density matrix.
        renyi_entropy_2: Renyi entropy at alpha=2.
        renyi_entropy_inf: Renyi entropy at alpha -> infinity (min-entropy).
        effective_dimension: exp(S(rho)) -- effective number of states.
        is_maximally_mixed: True iff S approximately equals log(n).
    """
    entropy: float
    max_entropy: float
    normalized_entropy: float
    density_eigenvalues: list[float]
    renyi_entropy_2: float
    renyi_entropy_inf: float
    effective_dimension: float
    is_maximally_mixed: bool


# ---------------------------------------------------------------------------
# Undirected graph helpers
# ---------------------------------------------------------------------------

def _undirected_edges(ss: "StateSpace") -> list[tuple[int, int]]:
    """Extract undirected edges from the state space.

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


def _mat_trace(A: list[list[float]]) -> float:
    """Trace of a square matrix."""
    return sum(A[i][i] for i in range(len(A)))


def _mat_power(A: list[list[float]], p: float) -> list[list[float]]:
    """Raise a symmetric PSD matrix to a real power via eigendecomposition.

    For integer powers, this is exact; for fractional powers, uses
    eigenvalue decomposition: A^p = Q diag(lambda_i^p) Q^T.
    """
    n = len(A)
    if n == 0:
        return []
    eigenvalues, eigenvectors = _eigen_symmetric(A)
    # A^p = sum_i lambda_i^p * v_i v_i^T
    result = [[0.0] * n for _ in range(n)]
    for k in range(n):
        lam = eigenvalues[k]
        if lam < 1e-15:
            continue  # skip zero eigenvalues
        lam_p = lam ** p
        v = eigenvectors[k]
        for i in range(n):
            for j in range(n):
                result[i][j] += lam_p * v[i] * v[j]
    return result


def _jacobi_eigen(
    M: list[list[float]], max_iter: int = 200
) -> tuple[list[float], list[list[float]]]:
    """Eigenvalues and eigenvectors of a real symmetric matrix via Jacobi iteration.

    The Jacobi eigenvalue algorithm repeatedly applies Givens rotations
    to zero out off-diagonal elements.  It is unconditionally convergent
    for real symmetric matrices and handles repeated eigenvalues correctly.

    Returns (eigenvalues, eigenvectors) where eigenvectors[k] is
    the eigenvector for eigenvalues[k], sorted by descending eigenvalue.
    """
    n = len(M)
    if n == 0:
        return [], []
    if n == 1:
        return [M[0][0]], [[1.0]]

    # Work on a copy
    A = [row[:] for row in M]
    # V accumulates eigenvectors (starts as identity)
    V = _mat_identity(n)

    for _ in range(max_iter * n * n):
        # Find largest off-diagonal element
        max_val = 0.0
        p, q = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i][j]) > max_val:
                    max_val = abs(A[i][j])
                    p, q = i, j

        if max_val < 1e-12:
            break  # Converged

        # Compute Jacobi rotation to zero out A[p][q]
        if abs(A[p][p] - A[q][q]) < 1e-15:
            theta = math.pi / 4
        else:
            theta = 0.5 * math.atan2(2.0 * A[p][q], A[p][p] - A[q][q])

        c = math.cos(theta)
        s = math.sin(theta)

        # Apply rotation: A' = G^T A G
        # Update rows/cols p and q
        new_pp = c * c * A[p][p] + 2 * s * c * A[p][q] + s * s * A[q][q]
        new_qq = s * s * A[p][p] - 2 * s * c * A[p][q] + c * c * A[q][q]
        new_pq = 0.0  # This is the point of the rotation

        # Update other entries
        for i in range(n):
            if i == p or i == q:
                continue
            aip = A[i][p]
            aiq = A[i][q]
            A[i][p] = c * aip + s * aiq
            A[p][i] = A[i][p]
            A[i][q] = -s * aip + c * aiq
            A[q][i] = A[i][q]

        A[p][p] = new_pp
        A[q][q] = new_qq
        A[p][q] = new_pq
        A[q][p] = new_pq

        # Update eigenvector matrix: V' = V * G
        for i in range(n):
            vip = V[i][p]
            viq = V[i][q]
            V[i][p] = c * vip + s * viq
            V[i][q] = -s * vip + c * viq

    eigenvalues = [A[i][i] for i in range(n)]
    # Eigenvectors are columns of V
    eigenvectors = [[V[i][j] for i in range(n)] for j in range(n)]

    # Sort by eigenvalue descending
    pairs = sorted(zip(eigenvalues, eigenvectors), key=lambda x: -x[0])
    eigenvalues = [p[0] for p in pairs]
    eigenvectors = [p[1] for p in pairs]

    return eigenvalues, eigenvectors


def _eigenvalues_symmetric(M: list[list[float]], max_iter: int = 200) -> list[float]:
    """Eigenvalues of a real symmetric matrix via Jacobi iteration.

    Handles repeated eigenvalues correctly.
    Suitable for small matrices (n < 50).
    """
    eigenvalues, _ = _jacobi_eigen(M, max_iter)
    return eigenvalues


def _eigen_symmetric(
    M: list[list[float]], max_iter: int = 200
) -> tuple[list[float], list[list[float]]]:
    """Eigenvalues and eigenvectors of a real symmetric matrix.

    Returns (eigenvalues, eigenvectors) where eigenvectors[k] is
    the eigenvector for eigenvalues[k].

    Delegates to Jacobi iteration.
    """
    return _jacobi_eigen(M, max_iter)


# ---------------------------------------------------------------------------
# Laplacian and density matrix
# ---------------------------------------------------------------------------

def laplacian_matrix(ss: "StateSpace") -> list[list[float]]:
    """Graph Laplacian L = D - A of the undirected version of the state space.

    L is symmetric and positive semi-definite.  Its smallest eigenvalue
    is always 0 (for connected graphs, with multiplicity 1).
    """
    A, states = _adjacency_matrix(ss)
    D, _ = _degree_matrix(ss)
    n = len(states)
    return [[D[i][j] - A[i][j] for j in range(n)] for i in range(n)]


def density_matrix(ss: "StateSpace") -> list[list[float]]:
    """Density matrix rho = L / tr(L) from the graph Laplacian.

    The density matrix is a valid quantum state: it is positive
    semi-definite with trace 1.

    For a single-state graph (no edges), returns [[1.0]] since tr(L)=0
    and we define rho = I/n by convention.

    Returns:
        The n x n density matrix as a list of lists.
    """
    L = laplacian_matrix(ss)
    n = len(L)
    if n == 0:
        return []
    if n == 1:
        return [[1.0]]

    tr = _mat_trace(L)
    if tr < 1e-15:
        # No edges: maximally mixed state rho = I/n
        return [[1.0 / n if i == j else 0.0 for j in range(n)]
                for i in range(n)]

    return _mat_scale(L, 1.0 / tr)


def density_eigenvalues(ss: "StateSpace") -> list[float]:
    """Eigenvalues of the density matrix rho = L / tr(L).

    Eigenvalues are non-negative and sum to 1.
    """
    rho = density_matrix(ss)
    if not rho:
        return []
    eigs = _eigenvalues_symmetric(rho)
    # Clamp small negatives to zero (numerical)
    return [max(0.0, e) for e in eigs]


# ---------------------------------------------------------------------------
# Von Neumann entropy and variants
# ---------------------------------------------------------------------------

def _xlogx(x: float) -> float:
    """Compute x * log(x) with the convention 0 * log(0) = 0."""
    if x < 1e-15:
        return 0.0
    return x * math.log(x)


def von_neumann_entropy(ss: "StateSpace") -> float:
    """Von Neumann entropy S(rho) = -sum_i lambda_i log(lambda_i).

    Uses natural logarithm (base e).

    Returns:
        The von Neumann entropy, in range [0, log(n)].
    """
    eigs = density_eigenvalues(ss)
    if not eigs:
        return 0.0
    result = -sum(_xlogx(lam) for lam in eigs)
    return max(0.0, result)  # Avoid -0.0 floating point artifact


def max_entropy(ss: "StateSpace") -> float:
    """Maximum possible entropy log(n) for a graph with n vertices."""
    n = len(ss.states)
    if n <= 1:
        return 0.0
    return math.log(n)


def normalized_entropy(ss: "StateSpace") -> float:
    """Normalized entropy S(rho) / log(n) in [0, 1].

    Returns 0.0 for single-state graphs.
    """
    me = max_entropy(ss)
    if me < 1e-15:
        return 0.0
    return von_neumann_entropy(ss) / me


def renyi_entropy(ss: "StateSpace", alpha: float) -> float:
    """Renyi entropy S_alpha = (1/(1-alpha)) log(tr(rho^alpha)).

    Special cases:
    - alpha -> 1: converges to von Neumann entropy.
    - alpha = 2: collision entropy -log(tr(rho^2)).
    - alpha -> infinity: min-entropy -log(max eigenvalue).

    Args:
        ss: The state space.
        alpha: The Renyi parameter (must be > 0, != 1).

    Returns:
        The Renyi entropy.

    Raises:
        ValueError: If alpha <= 0 or alpha == 1.
    """
    if alpha <= 0:
        raise ValueError(f"Renyi parameter alpha must be > 0, got {alpha}")
    if abs(alpha - 1.0) < 1e-10:
        return von_neumann_entropy(ss)

    eigs = density_eigenvalues(ss)
    if not eigs:
        return 0.0

    if alpha == float('inf') or alpha > 1e10:
        # Min-entropy: -log(max eigenvalue)
        max_eig = max(eigs)
        if max_eig < 1e-15:
            return 0.0
        return -math.log(max_eig)

    # tr(rho^alpha) = sum_i lambda_i^alpha
    tr_rho_alpha = sum(lam ** alpha for lam in eigs if lam > 1e-15)
    if tr_rho_alpha < 1e-15:
        return 0.0
    result = (1.0 / (1.0 - alpha)) * math.log(tr_rho_alpha)
    return max(0.0, result)


def renyi_entropy_2(ss: "StateSpace") -> float:
    """Renyi entropy at alpha=2 (collision entropy).

    S_2 = -log(tr(rho^2)) = -log(sum lambda_i^2).
    """
    return renyi_entropy(ss, 2.0)


def renyi_entropy_inf(ss: "StateSpace") -> float:
    """Renyi entropy at alpha -> infinity (min-entropy).

    S_inf = -log(max lambda_i).
    """
    eigs = density_eigenvalues(ss)
    if not eigs:
        return 0.0
    max_eig = max(eigs)
    if max_eig < 1e-15:
        return 0.0
    return max(0.0, -math.log(max_eig))


def effective_dimension(ss: "StateSpace") -> float:
    """Effective dimension exp(S(rho)).

    Measures how many states "effectively" participate in the protocol.
    Range: [1, n].
    """
    return math.exp(von_neumann_entropy(ss))


# ---------------------------------------------------------------------------
# Relative entropy and mutual information
# ---------------------------------------------------------------------------

def relative_entropy(ss1: "StateSpace", ss2: "StateSpace") -> float | None:
    """Quantum relative entropy S(rho1 || rho2) = tr(rho1 (log rho1 - log rho2)).

    Only defined when both state spaces have the same number of states
    and supp(rho1) is contained in supp(rho2).

    Returns None if the state spaces have different dimensions or if
    the relative entropy is not defined (support condition violated).
    """
    n1 = len(ss1.states)
    n2 = len(ss2.states)
    if n1 != n2:
        return None
    if n1 <= 1:
        return 0.0

    eigs1 = density_eigenvalues(ss1)
    eigs2 = density_eigenvalues(ss2)

    # Check support condition: wherever rho1 has support, rho2 must too
    for l1, l2 in zip(eigs1, eigs2):
        if l1 > 1e-15 and l2 < 1e-15:
            return None  # Support condition violated (would be +infinity)

    # S(rho1 || rho2) = sum_i lambda1_i (log lambda1_i - log lambda2_i)
    # This is an approximation using eigenvalues (exact when rho1, rho2 commute)
    result = 0.0
    for l1, l2 in zip(eigs1, eigs2):
        if l1 < 1e-15:
            continue
        result += l1 * (math.log(l1) - math.log(l2))
    return result


def mutual_information(ss: "StateSpace") -> float:
    """Mutual information of the graph structure.

    Computed as I = S(rho) - S_conditional, where S_conditional is
    the entropy of the density matrix restricted to the off-diagonal
    structure.

    For a graph, this captures the information shared between the
    "local" (degree) and "global" (connectivity) structure.

    We compute: I = log(n) - S(rho), which measures how far the
    protocol is from maximally mixed.  Higher I means more structure.
    """
    n = len(ss.states)
    if n <= 1:
        return 0.0
    return max_entropy(ss) - von_neumann_entropy(ss)


def entropy_rate(ss: "StateSpace") -> float:
    """Entropy rate: von Neumann entropy per state.

    Defined as S(rho) / n, representing the average entropy
    contribution per state in the protocol.
    """
    n = len(ss.states)
    if n <= 1:
        return 0.0
    return von_neumann_entropy(ss) / n


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze_von_neumann(ss: "StateSpace") -> VonNeumannResult:
    """Complete von Neumann entropy analysis for a session type state space.

    Computes entropy, density matrix eigenvalues, Renyi entropies,
    effective dimension, and checks for maximal mixing.

    Returns:
        VonNeumannResult with all computed quantities.
    """
    eigs = density_eigenvalues(ss)
    ent = von_neumann_entropy(ss)
    me = max_entropy(ss)
    ne = normalized_entropy(ss)
    r2 = renyi_entropy_2(ss)
    ri = renyi_entropy_inf(ss)
    ed = effective_dimension(ss)

    # Maximally mixed if normalized entropy > 0.99
    is_max = ne > 0.99 if me > 1e-15 else False

    return VonNeumannResult(
        entropy=ent,
        max_entropy=me,
        normalized_entropy=ne,
        density_eigenvalues=eigs,
        renyi_entropy_2=r2,
        renyi_entropy_inf=ri,
        effective_dimension=ed,
        is_maximally_mixed=is_max,
    )
