"""Heat kernel analysis for session type lattices (Step 30q).

The heat kernel H_t = exp(-tL) where L is the graph Laplacian describes
heat diffusion on the graph underlying a session type state space.

Key concepts:
- **Graph Laplacian** L = D - A of the undirected quotient graph
- **Heat kernel** H_t(x,y) = sum_k exp(-t*lam_k) phi_k(x) phi_k(y)
- **Heat trace** Z(t) = tr(H_t) = sum_k exp(-t*lam_k)
- **Heat kernel signature** HKS(x,t) = H_t(x,x) as a function of t
- **Diffusion distance** d_t(x,y)^2 = H_t(x,x) + H_t(y,y) - 2*H_t(x,y)
- **Heat content** Q(t) = sum of all entries of H_t
- **Heat kernel PageRank** p = exp(-tL) * (1/n * 1)

For session types, heat diffusion models information propagation through
protocol states.  Short-time behaviour reveals local connectivity;
long-time behaviour reveals global lattice structure.

All computations use pure Python (no numpy); eigendecomposition via
Jacobi iteration suitable for small matrices (<= 50 states).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace

from reticulate.zeta import _adjacency, _compute_sccs, _state_list, compute_rank


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_TIMES: list[float] = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

_EPS = 1e-12  # Numerical tolerance


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HeatKernelResult:
    """Complete heat kernel analysis for a session type state space.

    Attributes:
        laplacian: Graph Laplacian matrix L = D - A.
        eigenvalues: Sorted eigenvalues of L (non-negative).
        eigenvectors: Orthonormal eigenvectors (column-major: eigenvectors[k]
            is the k-th eigenvector).
        heat_trace: Z(t) at each sample time.
        sample_times: The t values used for sampling.
        hks_diagonal: hks_diagonal[i][j] = H_{t_j}(x_i, x_i) for state i
            at sample time j.
        total_heat_content: Q(t) at each sample time.
        diffusion_distance: Diffusion distance matrix at t = 1.0.
    """

    laplacian: list[list[float]]
    eigenvalues: list[float]
    eigenvectors: list[list[float]]
    heat_trace: list[float]
    sample_times: list[float]
    hks_diagonal: list[list[float]]
    total_heat_content: list[float]
    diffusion_distance: list[list[float]]


# ---------------------------------------------------------------------------
# Undirected adjacency from quotient
# ---------------------------------------------------------------------------

def _quotient_states_and_adj(ss: "StateSpace") -> tuple[list[int], dict[int, set[int]]]:
    """Build quotient DAG (collapsing SCCs), return rep list and adjacency.

    The quotient collapses strongly connected components into single
    representative nodes.  We then form an *undirected* adjacency by
    adding both directions for every quotient edge.
    """
    scc_map, scc_members = _compute_sccs(ss)
    adj = _adjacency(ss)
    reps = sorted(scc_members.keys())
    q_adj: dict[int, set[int]] = {r: set() for r in reps}
    for s in ss.states:
        for t in adj[s]:
            sr, tr = scc_map[s], scc_map[t]
            if sr != tr:
                q_adj[sr].add(tr)
    # Make undirected
    u_adj: dict[int, set[int]] = {r: set() for r in reps}
    for r in reps:
        for t in q_adj[r]:
            u_adj[r].add(t)
            u_adj[t].add(r)
    return reps, u_adj


def _symmetric_adjacency_matrix(
    reps: list[int], u_adj: dict[int, set[int]]
) -> list[list[float]]:
    """Build symmetric adjacency matrix from undirected adjacency."""
    n = len(reps)
    idx = {r: i for i, r in enumerate(reps)}
    A = [[0.0] * n for _ in range(n)]
    for r in reps:
        for t in u_adj[r]:
            A[idx[r]][idx[t]] = 1.0
    return A


# ---------------------------------------------------------------------------
# Graph Laplacian
# ---------------------------------------------------------------------------

def graph_laplacian(ss: "StateSpace") -> list[list[float]]:
    """Symmetric graph Laplacian L = D - A of the undirected quotient.

    L[i][j] = degree(i)  if i == j
            = -1          if i ~ j (adjacent in undirected quotient)
            =  0          otherwise

    The Laplacian is symmetric positive semi-definite.
    Row sums (and column sums) are zero.
    """
    reps, u_adj = _quotient_states_and_adj(ss)
    n = len(reps)
    if n == 0:
        return []
    A = _symmetric_adjacency_matrix(reps, u_adj)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        deg = sum(A[i])
        for j in range(n):
            if i == j:
                L[i][j] = deg
            else:
                L[i][j] = -A[i][j]
    return L


# ---------------------------------------------------------------------------
# Jacobi eigendecomposition
# ---------------------------------------------------------------------------

def _mat_copy(M: list[list[float]]) -> list[list[float]]:
    """Deep copy a square matrix."""
    return [row[:] for row in M]


def _identity(n: int) -> list[list[float]]:
    """n x n identity matrix."""
    I = [[0.0] * n for _ in range(n)]
    for i in range(n):
        I[i][i] = 1.0
    return I


def _jacobi_eigendecomposition(
    M: list[list[float]], max_iter: int = 200
) -> tuple[list[float], list[list[float]]]:
    """Eigendecomposition of a real symmetric matrix via Jacobi rotation.

    Returns (eigenvalues, eigenvectors) where eigenvectors[k] is the
    k-th eigenvector (as a list).  Eigenvalues are sorted ascending.
    Eigenvectors are orthonormal (up to numerical precision).

    Suitable for small matrices (<= 50 x 50).
    """
    n = len(M)
    if n == 0:
        return [], []
    if n == 1:
        return [M[0][0]], [[1.0]]

    A = _mat_copy(M)
    V = _identity(n)

    for _ in range(max_iter):
        # Find largest off-diagonal element
        max_val = 0.0
        p, q = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i][j]) > max_val:
                    max_val = abs(A[i][j])
                    p, q = i, j
        if max_val < _EPS:
            break

        # Compute rotation angle
        if abs(A[p][p] - A[q][q]) < _EPS:
            theta = math.pi / 4.0
        else:
            theta = 0.5 * math.atan2(2.0 * A[p][q], A[p][p] - A[q][q])

        c = math.cos(theta)
        s = math.sin(theta)

        # Apply Jacobi rotation: A' = G^T A G
        # Update rows/cols p and q
        new_A = _mat_copy(A)
        for i in range(n):
            if i != p and i != q:
                new_A[i][p] = c * A[i][p] + s * A[i][q]
                new_A[p][i] = new_A[i][p]
                new_A[i][q] = -s * A[i][p] + c * A[i][q]
                new_A[q][i] = new_A[i][q]
        new_A[p][p] = c * c * A[p][p] + 2 * s * c * A[p][q] + s * s * A[q][q]
        new_A[q][q] = s * s * A[p][p] - 2 * s * c * A[p][q] + c * c * A[q][q]
        new_A[p][q] = 0.0
        new_A[q][p] = 0.0
        A = new_A

        # Update eigenvector matrix V = V G
        for i in range(n):
            vp = V[i][p]
            vq = V[i][q]
            V[i][p] = c * vp + s * vq
            V[i][q] = -s * vp + c * vq

    # Extract eigenvalues from diagonal
    eigenvalues = [A[i][i] for i in range(n)]

    # Sort by eigenvalue ascending
    order = sorted(range(n), key=lambda k: eigenvalues[k])
    sorted_vals = [eigenvalues[k] for k in order]
    # eigenvectors: column k of V is eigenvector for eigenvalue k
    sorted_vecs: list[list[float]] = []
    for k in order:
        vec = [V[i][k] for i in range(n)]
        # Normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > _EPS:
            vec = [x / norm for x in vec]
        sorted_vecs.append(vec)

    # Clamp near-zero eigenvalues to zero (Laplacian is PSD)
    sorted_vals = [max(0.0, v) if abs(v) < _EPS else v for v in sorted_vals]

    return sorted_vals, sorted_vecs


# ---------------------------------------------------------------------------
# Eigendecomposition
# ---------------------------------------------------------------------------

def laplacian_eigendecomposition(
    ss: "StateSpace",
) -> tuple[list[float], list[list[float]]]:
    """Eigenvalues and eigenvectors of the graph Laplacian.

    Returns:
        eigenvalues: Sorted ascending, non-negative.
        eigenvectors: eigenvectors[k] is the k-th eigenvector (list of floats).
    """
    L = graph_laplacian(ss)
    if not L:
        return [], []
    return _jacobi_eigendecomposition(L)


# ---------------------------------------------------------------------------
# Heat kernel matrix
# ---------------------------------------------------------------------------

def heat_kernel_matrix(ss: "StateSpace", t: float) -> list[list[float]]:
    """Heat kernel matrix H_t = exp(-tL) via spectral decomposition.

    H_t(x,y) = sum_k exp(-t * lam_k) * phi_k(x) * phi_k(y)

    where lam_k, phi_k are eigenvalue/eigenvector pairs of the Laplacian.
    """
    eigenvalues, eigenvectors = laplacian_eigendecomposition(ss)
    n = len(eigenvalues)
    if n == 0:
        return []

    H = [[0.0] * n for _ in range(n)]
    for k in range(n):
        coeff = math.exp(-t * eigenvalues[k])
        phi = eigenvectors[k]
        for i in range(n):
            for j in range(n):
                H[i][j] += coeff * phi[i] * phi[j]
    return H


# ---------------------------------------------------------------------------
# Heat trace
# ---------------------------------------------------------------------------

def heat_trace(ss: "StateSpace", t: float) -> float:
    """Heat trace Z(t) = tr(H_t) = sum_k exp(-t * lam_k).

    At t = 0, Z(0) = n (number of quotient states).
    As t -> infinity, Z(t) -> multiplicity of eigenvalue 0.
    """
    eigenvalues, _ = laplacian_eigendecomposition(ss)
    if not eigenvalues:
        return 0.0
    return sum(math.exp(-t * lam) for lam in eigenvalues)


# ---------------------------------------------------------------------------
# Heat kernel signature
# ---------------------------------------------------------------------------

def heat_kernel_signature(
    ss: "StateSpace", state: int, times: list[float]
) -> list[float]:
    """Heat kernel signature HKS(x, t) = H_t(x, x) at given times.

    The HKS is a local descriptor: it characterizes the geometry
    around state x at multiple scales (controlled by t).

    Args:
        ss: State space.
        state: State ID (in the quotient).
        times: List of time values to evaluate.

    Returns:
        List of HKS values, one per time value.
    """
    eigenvalues, eigenvectors = laplacian_eigendecomposition(ss)
    n = len(eigenvalues)
    if n == 0:
        return [0.0] * len(times)

    # Find the index of the state in the quotient
    scc_map, scc_members = _compute_sccs(ss)
    reps = sorted(scc_members.keys())
    # Map the input state to its SCC representative
    rep = scc_map.get(state, state)
    if rep not in reps:
        return [0.0] * len(times)
    idx = reps.index(rep)

    result: list[float] = []
    for t in times:
        val = 0.0
        for k in range(n):
            val += math.exp(-t * eigenvalues[k]) * eigenvectors[k][idx] ** 2
        result.append(val)
    return result


# ---------------------------------------------------------------------------
# Diffusion distance
# ---------------------------------------------------------------------------

def diffusion_distance(ss: "StateSpace", t: float) -> list[list[float]]:
    """Diffusion distance matrix at time t.

    d_t(x, y)^2 = H_t(x,x) + H_t(y,y) - 2 * H_t(x,y)

    This is a metric: d_t(x,x) = 0, symmetric, satisfies triangle inequality.
    """
    H = heat_kernel_matrix(ss, t)
    n = len(H)
    if n == 0:
        return []

    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            sq = H[i][i] + H[j][j] - 2.0 * H[i][j]
            # Clamp small negative values from numerical noise
            D[i][j] = math.sqrt(max(0.0, sq))
    return D


# ---------------------------------------------------------------------------
# Total heat content
# ---------------------------------------------------------------------------

def total_heat_content(ss: "StateSpace", t: float) -> float:
    """Total heat content Q(t) = sum of all entries of H_t.

    Q(t) = sum_{x,y} H_t(x,y)

    At t = 0, Q(0) = n (since H_0 = I, sum = n).
    As t -> infinity, Q(t) -> n (heat spreads uniformly to 1/n per entry,
    sum over n^2 entries of 1/n = n).  For connected graphs, the long-time
    limit is n.
    """
    H = heat_kernel_matrix(ss, t)
    if not H:
        return 0.0
    return sum(H[i][j] for i in range(len(H)) for j in range(len(H)))


# ---------------------------------------------------------------------------
# Heat kernel PageRank
# ---------------------------------------------------------------------------

def heat_kernel_pagerank(ss: "StateSpace", t: float) -> list[float]:
    """Heat kernel PageRank: p = exp(-tL) * (1/n * 1).

    This is the heat diffusion from a uniform initial distribution.
    Each entry p[i] gives the heat at state i after time t, starting
    from uniform distribution 1/n at each node.

    Returns:
        Vector of PageRank values (one per quotient state).
    """
    H = heat_kernel_matrix(ss, t)
    n = len(H)
    if n == 0:
        return []

    # Multiply H by uniform vector (1/n, ..., 1/n)
    uniform = 1.0 / n
    p: list[float] = []
    for i in range(n):
        val = sum(H[i][j] for j in range(n)) * uniform
        p.append(val)
    return p


# ---------------------------------------------------------------------------
# Return probability
# ---------------------------------------------------------------------------

def return_probability(ss: "StateSpace", state: int, t: float) -> float:
    """Return probability of continuous-time random walk at time t.

    This equals H_t(x,x) / (sum_y H_t(x, y)), the probability that
    a random walker starting at x is back at x at time t.

    For the heat kernel, H_t(x,x) IS the auto-diffusion coefficient.
    """
    eigenvalues, eigenvectors = laplacian_eigendecomposition(ss)
    n = len(eigenvalues)
    if n == 0:
        return 0.0

    scc_map, scc_members = _compute_sccs(ss)
    reps = sorted(scc_members.keys())
    rep = scc_map.get(state, state)
    if rep not in reps:
        return 0.0
    idx = reps.index(rep)

    val = 0.0
    for k in range(n):
        val += math.exp(-t * eigenvalues[k]) * eigenvectors[k][idx] ** 2
    return val


# ---------------------------------------------------------------------------
# Spectral gap
# ---------------------------------------------------------------------------

def spectral_gap(ss: "StateSpace") -> float:
    """Spectral gap: smallest non-zero eigenvalue of the Laplacian.

    The spectral gap controls the rate of convergence to equilibrium.
    A larger gap means faster mixing / heat diffusion.
    Returns 0.0 for trivial graphs.
    """
    eigenvalues, _ = laplacian_eigendecomposition(ss)
    for lam in eigenvalues:
        if lam > _EPS:
            return lam
    return 0.0


# ---------------------------------------------------------------------------
# Effective resistance
# ---------------------------------------------------------------------------

def effective_resistance(ss: "StateSpace") -> list[list[float]]:
    """Effective resistance (Kirchhoff) between all pairs of quotient states.

    R(x,y) = sum_{k: lam_k > 0} (1/lam_k) * (phi_k(x) - phi_k(y))^2

    The effective resistance is a metric on the graph related to
    random walks and electrical networks.
    """
    eigenvalues, eigenvectors = laplacian_eigendecomposition(ss)
    n = len(eigenvalues)
    if n == 0:
        return []

    R = [[0.0] * n for _ in range(n)]
    for k in range(n):
        if eigenvalues[k] < _EPS:
            continue
        inv_lam = 1.0 / eigenvalues[k]
        phi = eigenvectors[k]
        for i in range(n):
            for j in range(n):
                R[i][j] += inv_lam * (phi[i] - phi[j]) ** 2
    return R


# ---------------------------------------------------------------------------
# Kemeny constant
# ---------------------------------------------------------------------------

def kemeny_constant(ss: "StateSpace") -> float:
    """Kemeny constant: sum of 1/lam_k for all non-zero eigenvalues.

    This is independent of the starting state and characterizes the
    expected number of steps for a random walk to reach stationarity.
    """
    eigenvalues, _ = laplacian_eigendecomposition(ss)
    total = 0.0
    for lam in eigenvalues:
        if lam > _EPS:
            total += 1.0 / lam
    return total


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze_heat_kernel(
    ss: "StateSpace",
    sample_times: list[float] | None = None,
) -> HeatKernelResult:
    """Complete heat kernel analysis for a session type state space.

    Computes the Laplacian, eigendecomposition, heat kernel at multiple
    time scales, HKS diagonal, diffusion distances, and total heat content.

    Args:
        ss: State space to analyze.
        sample_times: Time values for sampling (default: DEFAULT_SAMPLE_TIMES).

    Returns:
        HeatKernelResult with all computed quantities.
    """
    if sample_times is None:
        sample_times = DEFAULT_SAMPLE_TIMES[:]

    L = graph_laplacian(ss)
    n = len(L)

    if n == 0:
        return HeatKernelResult(
            laplacian=[],
            eigenvalues=[],
            eigenvectors=[],
            heat_trace=[],
            sample_times=sample_times,
            hks_diagonal=[],
            total_heat_content=[],
            diffusion_distance=[],
        )

    eigenvalues, eigenvectors = _jacobi_eigendecomposition(L)

    # Compute per-time quantities
    traces: list[float] = []
    contents: list[float] = []
    hks: list[list[float]] = [[] for _ in range(n)]  # hks[state][time_idx]

    for t in sample_times:
        # Heat kernel matrix at this time
        H = [[0.0] * n for _ in range(n)]
        for k in range(n):
            coeff = math.exp(-t * eigenvalues[k])
            phi = eigenvectors[k]
            for i in range(n):
                for j in range(n):
                    H[i][j] += coeff * phi[i] * phi[j]

        # Trace
        traces.append(sum(H[i][i] for i in range(n)))

        # HKS diagonal
        for i in range(n):
            hks[i].append(H[i][i])

        # Total heat content
        contents.append(sum(H[i][j] for i in range(n) for j in range(n)))

    # Diffusion distance at t = 1.0
    dd = diffusion_distance(ss, 1.0)

    return HeatKernelResult(
        laplacian=L,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        heat_trace=traces,
        sample_times=sample_times,
        hks_diagonal=hks,
        total_heat_content=contents,
        diffusion_distance=dd,
    )
