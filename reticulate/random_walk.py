"""Random walk mixing time analysis for session type lattices (Step 30p).

Random walks on finite state spaces model nondeterministic protocol execution.
Given a session type state space as a directed graph, we construct the
transition probability matrix P where P[i,j] = 1/deg(i) if (i,j) is an edge
(uniform random walk).

This module provides:

- **Transition matrix** construction (uniform random walk on the state space)
- **Stationary distribution** via power iteration
- **Spectral gap** analysis: 1 - |lambda_2| governs convergence speed
- **Mixing time** upper bounds from the spectral gap
- **Hitting times**: expected steps to reach target from source
- **Cover time**: expected steps to visit all states
- **Return time**: 1/pi(state) for ergodic chains
- **Commute time**: H(u,v) + H(v,u)
- **Ergodicity** detection (irreducible + aperiodic)

For session type state spaces (typically DAGs with absorbing bottom state),
the chain is generally NOT ergodic.  We handle absorbing states by adding
self-loops so that the walk is well-defined everywhere.  For ergodic
analysis we optionally consider the undirected (symmetrised) version.

All computations use pure Python (no numpy) and target small matrices
(< 50 states).
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
class RandomWalkResult:
    """Complete random walk analysis for a session type state space.

    Attributes:
        transition_matrix: Transition probability matrix P[i][j].
        stationary_distribution: Stationary distribution pi.
        spectral_gap: 1 - |lambda_2| where lambda_2 is second-largest
            eigenvalue magnitude.
        mixing_time_bound: Upper bound on mixing time.
        hitting_times: Pairwise hitting times H(i,j).
        cover_time_bound: Upper bound on expected cover time.
        is_ergodic: True if the chain is irreducible and aperiodic.
        eigenvalues: Eigenvalues of P sorted descending by magnitude.
    """

    transition_matrix: list[list[float]]
    stationary_distribution: list[float]
    spectral_gap: float
    mixing_time_bound: int
    hitting_times: dict[tuple[int, int], float]
    cover_time_bound: float
    is_ergodic: bool
    eigenvalues: list[float]


# ---------------------------------------------------------------------------
# Matrix utilities (pure Python, small matrices)
# ---------------------------------------------------------------------------

def _mat_mul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Multiply two square matrices."""
    n = len(A)
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            a_ik = A[i][k]
            if a_ik == 0.0:
                continue
            for j in range(n):
                C[i][j] += a_ik * B[k][j]
    return C


def _mat_vec(A: list[list[float]], v: list[float]) -> list[float]:
    """Multiply matrix A by column vector v."""
    n = len(A)
    result = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += A[i][j] * v[j]
        result[i] = s
    return result


def _vec_mat(v: list[float], A: list[list[float]]) -> list[float]:
    """Multiply row vector v by matrix A."""
    n = len(A)
    result = [0.0] * n
    for j in range(n):
        s = 0.0
        for i in range(n):
            s += v[i] * A[i][j]
        result[j] = s
    return result


def _identity(n: int) -> list[list[float]]:
    """n x n identity matrix."""
    M = [[0.0] * n for _ in range(n)]
    for i in range(n):
        M[i][i] = 1.0
    return M


def _transpose(A: list[list[float]]) -> list[list[float]]:
    """Transpose a square matrix."""
    n = len(A)
    return [[A[j][i] for j in range(n)] for i in range(n)]


def _vec_norm(v: list[float]) -> float:
    """Euclidean norm of a vector."""
    return math.sqrt(sum(x * x for x in v))


def _vec_sub(a: list[float], b: list[float]) -> list[float]:
    """Element-wise subtraction."""
    return [a[i] - b[i] for i in range(len(a))]


def _l1_norm(v: list[float]) -> float:
    """L1 norm."""
    return sum(abs(x) for x in v)


# ---------------------------------------------------------------------------
# Eigenvalue computation (QR iteration for small matrices)
# ---------------------------------------------------------------------------

def _householder_qr(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    """QR decomposition via Householder reflections (for small matrices)."""
    n = len(A)
    R = [row[:] for row in A]
    Q = _identity(n)

    for k in range(min(n - 1, n)):
        # Extract column below diagonal
        x = [R[i][k] if i >= k else 0.0 for i in range(n)]
        norm_x = math.sqrt(sum(x[i] ** 2 for i in range(k, n)))
        if norm_x < 1e-15:
            continue

        sign = 1.0 if x[k] >= 0 else -1.0
        x[k] += sign * norm_x
        norm_v = math.sqrt(sum(x[i] ** 2 for i in range(k, n)))
        if norm_v < 1e-15:
            continue
        for i in range(k, n):
            x[i] /= norm_v

        # Apply H = I - 2vv^T to R from left
        for j in range(k, n):
            dot = sum(x[i] * R[i][j] for i in range(k, n))
            for i in range(k, n):
                R[i][j] -= 2.0 * x[i] * dot

        # Apply H to Q from right
        for i in range(n):
            dot = sum(Q[i][j] * x[j] for j in range(k, n))
            for j in range(k, n):
                Q[i][j] -= 2.0 * dot * x[j]

    return Q, R


def _qr_eigenvalues(A: list[list[float]], max_iter: int = 200) -> list[float]:
    """Compute eigenvalues via QR iteration (real eigenvalues only).

    For the small matrices we encounter (< 50 states), this is sufficient.
    Returns eigenvalues sorted descending by magnitude.
    """
    n = len(A)
    if n == 0:
        return []
    if n == 1:
        return [A[0][0]]

    # Work on a copy
    M = [row[:] for row in A]

    for _ in range(max_iter):
        Q, R = _householder_qr(M)
        M = _mat_mul(R, Q)

        # Check convergence: sub-diagonal elements small
        off = sum(M[i][j] ** 2 for i in range(n) for j in range(i))
        if off < 1e-20:
            break

    eigs = [M[i][i] for i in range(n)]
    eigs.sort(key=lambda x: -abs(x))
    return eigs


# ---------------------------------------------------------------------------
# Core: transition matrix
# ---------------------------------------------------------------------------

def transition_matrix(ss: "StateSpace") -> list[list[float]]:
    """Build transition probability matrix P for uniform random walk.

    P[i][j] = 1/out_degree(i) if there is an edge from state i to state j.
    Absorbing states (no outgoing transitions) get a self-loop: P[i][i] = 1.

    States are indexed in sorted order (consistent with _state_list).
    """
    states = _state_list(ss)
    n = len(states)
    if n == 0:
        return []

    idx = {s: i for i, s in enumerate(states)}
    adj = _adjacency(ss)

    P = [[0.0] * n for _ in range(n)]
    for s in states:
        neighbors = adj[s]
        # Deduplicate targets
        targets: set[int] = set()
        for t in neighbors:
            targets.add(t)

        deg = len(targets)
        i = idx[s]
        if deg == 0:
            # Absorbing state: self-loop
            P[i][i] = 1.0
        else:
            prob = 1.0 / deg
            for t in targets:
                P[i][idx[t]] += prob

    return P


# ---------------------------------------------------------------------------
# Stationary distribution
# ---------------------------------------------------------------------------

def stationary_distribution(ss: "StateSpace") -> list[float]:
    """Compute stationary distribution via power iteration.

    For absorbing chains, the stationary distribution concentrates on
    absorbing states.  We iterate pi = pi * P until convergence.

    Returns distribution indexed by _state_list order.
    """
    P = transition_matrix(ss)
    n = len(P)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    # Start from uniform
    pi = [1.0 / n] * n

    for _ in range(1000):
        pi_new = _vec_mat(pi, P)
        # Normalise for numerical stability
        total = sum(pi_new)
        if total > 0:
            pi_new = [x / total for x in pi_new]

        diff = sum(abs(pi_new[i] - pi[i]) for i in range(n))
        pi = pi_new
        if diff < 1e-12:
            break

    return pi


# ---------------------------------------------------------------------------
# Eigenvalues and spectral gap
# ---------------------------------------------------------------------------

def eigenvalues_of_P(ss: "StateSpace") -> list[float]:
    """Eigenvalues of the transition matrix, sorted descending by magnitude."""
    P = transition_matrix(ss)
    if not P:
        return []
    return _qr_eigenvalues(P)


def spectral_gap(ss: "StateSpace") -> float:
    """Spectral gap = 1 - |lambda_2|.

    lambda_2 is the second-largest eigenvalue by magnitude.
    For a single-state space, the gap is 1.0.
    """
    eigs = eigenvalues_of_P(ss)
    if len(eigs) <= 1:
        return 1.0

    # Largest eigenvalue should be ~1.0 for a stochastic matrix.
    # Second largest by magnitude:
    lambda2 = abs(eigs[1]) if len(eigs) > 1 else 0.0
    gap = 1.0 - lambda2
    # Clamp to [0, 1] for numerical safety
    return max(0.0, min(1.0, gap))


# ---------------------------------------------------------------------------
# Mixing time
# ---------------------------------------------------------------------------

def mixing_time_bound(ss: "StateSpace") -> int:
    """Upper bound on mixing time: ceil(1/gap * ln(n)).

    The classical bound for total variation distance < 1/4 is:
        t_mix <= (1/gap) * ln(4n)

    For non-ergodic chains, returns n^2 as a fallback bound.
    """
    states = _state_list(ss)
    n = len(states)
    if n <= 1:
        return 0

    gap = spectral_gap(ss)
    if gap < 1e-12:
        # No spectral gap => walk does not mix; return n^2
        return n * n

    bound = (1.0 / gap) * math.log(4.0 * n)
    return max(1, math.ceil(bound))


# ---------------------------------------------------------------------------
# Hitting times
# ---------------------------------------------------------------------------

def hitting_time(ss: "StateSpace", src: int, tgt: int) -> float:
    """Expected hitting time from src to tgt.

    Solves h(s) = 0 if s == tgt, h(s) = 1 + sum_j P[s,j]*h(j) otherwise.
    Uses iterative (Gauss-Seidel) method.

    Returns float('inf') if tgt is not reachable from src.
    """
    if src == tgt:
        return 0.0

    states = _state_list(ss)
    n = len(states)
    idx = {s: i for i, s in enumerate(states)}
    P = transition_matrix(ss)

    if src not in idx or tgt not in idx:
        return float("inf")

    ti = idx[tgt]

    # h[i] = expected hitting time from state i to tgt
    h = [0.0] * n
    # tgt has h = 0; initialise others to 1
    for i in range(n):
        if i != ti:
            h[i] = 1.0

    # Check reachability first: if tgt not reachable from src in the
    # original directed graph, hitting time is infinite.
    reachable = ss.reachable_from(src)
    if tgt not in reachable:
        return float("inf")

    for _ in range(2000):
        max_delta = 0.0
        for i in range(n):
            if i == ti:
                continue
            new_h = 1.0 + sum(P[i][j] * h[j] for j in range(n))
            delta = abs(new_h - h[i])
            if delta > max_delta:
                max_delta = delta
            h[i] = new_h

        if max_delta < 1e-8:
            break

    si = idx[src]
    # If hitting time is very large, treat as infinite
    if h[si] > 1e10:
        return float("inf")
    return h[si]


def all_hitting_times(ss: "StateSpace") -> dict[tuple[int, int], float]:
    """All pairwise hitting times.

    Returns dict mapping (src_state_id, tgt_state_id) -> expected hitting time.
    """
    states = _state_list(ss)
    n = len(states)
    P = transition_matrix(ss)
    idx = {s: i for i, s in enumerate(states)}

    result: dict[tuple[int, int], float] = {}

    # Pre-compute reachability for all states
    reach: dict[int, set[int]] = {s: ss.reachable_from(s) for s in states}

    for tgt in states:
        ti = idx[tgt]
        # Solve for all sources simultaneously
        h = [0.0] * n
        for i in range(n):
            if i != ti:
                h[i] = 1.0

        for _ in range(2000):
            max_delta = 0.0
            for i in range(n):
                if i == ti:
                    continue
                new_h = 1.0 + sum(P[i][j] * h[j] for j in range(n))
                delta = abs(new_h - h[i])
                if delta > max_delta:
                    max_delta = delta
                h[i] = new_h
            if max_delta < 1e-8:
                break

        for src in states:
            si = idx[src]
            if src == tgt:
                result[(src, tgt)] = 0.0
            elif tgt not in reach[src]:
                result[(src, tgt)] = float("inf")
            elif h[si] > 1e10:
                result[(src, tgt)] = float("inf")
            else:
                result[(src, tgt)] = h[si]

    return result


# ---------------------------------------------------------------------------
# Return time
# ---------------------------------------------------------------------------

def return_time(ss: "StateSpace", state: int) -> float:
    """Expected return time to state = 1/pi(state) for ergodic chains.

    For non-ergodic chains where pi(state) ~ 0, returns float('inf').
    """
    pi = stationary_distribution(ss)
    states = _state_list(ss)
    idx = {s: i for i, s in enumerate(states)}

    if state not in idx:
        return float("inf")

    pi_s = pi[idx[state]]
    if pi_s < 1e-15:
        return float("inf")
    return 1.0 / pi_s


# ---------------------------------------------------------------------------
# Commute time
# ---------------------------------------------------------------------------

def commute_time(ss: "StateSpace", u: int, v: int) -> float:
    """Commute time C(u,v) = H(u,v) + H(v,u)."""
    h_uv = hitting_time(ss, u, v)
    h_vu = hitting_time(ss, v, u)
    if math.isinf(h_uv) or math.isinf(h_vu):
        return float("inf")
    return h_uv + h_vu


# ---------------------------------------------------------------------------
# Cover time
# ---------------------------------------------------------------------------

def cover_time_bound(ss: "StateSpace") -> float:
    """Upper bound on expected cover time.

    Matthews' bound: C <= H_max * n * H_n
    where H_max = max hitting time, n = number of states,
    H_n = n-th harmonic number.

    For single-state spaces, cover time is 0.
    """
    states = _state_list(ss)
    n = len(states)
    if n <= 1:
        return 0.0

    ht = all_hitting_times(ss)

    # Maximum finite hitting time
    h_max = 0.0
    for (s, t), h in ht.items():
        if s != t and not math.isinf(h):
            h_max = max(h_max, h)

    if h_max == 0.0:
        return 0.0

    # Harmonic number H_n = 1 + 1/2 + ... + 1/n
    harmonic = sum(1.0 / k for k in range(1, n + 1))

    return h_max * harmonic


# ---------------------------------------------------------------------------
# Ergodicity
# ---------------------------------------------------------------------------

def _is_irreducible(ss: "StateSpace") -> bool:
    """Check if the state space is strongly connected (irreducible chain)."""
    states = _state_list(ss)
    n = len(states)
    if n <= 1:
        return True

    adj = _adjacency(ss)

    # Add self-loops for absorbing states (matching transition_matrix)
    adj_with_loops: dict[int, list[int]] = {}
    for s in states:
        neighbors = adj.get(s, [])
        if not neighbors:
            adj_with_loops[s] = [s]
        else:
            adj_with_loops[s] = neighbors

    # BFS from first state
    start = states[0]
    visited: set[int] = set()
    queue = [start]
    while queue:
        u = queue.pop(0)
        if u in visited:
            continue
        visited.add(u)
        for v in adj_with_loops.get(u, []):
            if v not in visited:
                queue.append(v)

    if len(visited) < n:
        return False

    # Check reverse reachability
    rev_adj: dict[int, list[int]] = {s: [] for s in states}
    for s in states:
        for t in adj_with_loops[s]:
            rev_adj[t].append(s)

    visited2: set[int] = set()
    queue2 = [start]
    while queue2:
        u = queue2.pop(0)
        if u in visited2:
            continue
        visited2.add(u)
        for v in rev_adj.get(u, []):
            if v not in visited2:
                queue2.append(v)

    return len(visited2) >= n


def _gcd(a: int, b: int) -> int:
    """GCD of two non-negative integers."""
    while b:
        a, b = b, a % b
    return a


def _compute_period(ss: "StateSpace") -> int:
    """Compute the period of the chain.

    Period = GCD of lengths of all cycles returning to any state.
    For absorbing states with self-loops, cycle length = 1, so period = 1.
    """
    states = _state_list(ss)
    if not states:
        return 1

    adj = _adjacency(ss)
    # Add self-loops for absorbing states
    adj_loops: dict[int, list[int]] = {}
    for s in states:
        neighbors = adj.get(s, [])
        if not neighbors:
            adj_loops[s] = [s]
        else:
            adj_loops[s] = list(set(neighbors))

    # BFS from start, compute distance, find cycle lengths
    start = states[0]
    dist: dict[int, int] = {start: 0}
    queue = [start]
    period = 0

    while queue:
        u = queue.pop(0)
        for v in adj_loops.get(u, []):
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
            # Cycle length contribution
            cycle_len = dist[u] + 1 - dist[v]
            if cycle_len > 0:
                period = _gcd(period, cycle_len)

    return period if period > 0 else 1


def is_ergodic(ss: "StateSpace") -> bool:
    """Check if random walk is ergodic (irreducible and aperiodic).

    Most session type state spaces are DAGs with an absorbing bottom state,
    hence NOT ergodic (not irreducible).
    """
    if not _is_irreducible(ss):
        return False
    return _compute_period(ss) == 1


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze_random_walk(ss: "StateSpace") -> RandomWalkResult:
    """Complete random walk analysis of a session type state space.

    Computes transition matrix, stationary distribution, spectral gap,
    mixing time bound, all hitting times, cover time bound, ergodicity,
    and eigenvalues.
    """
    P = transition_matrix(ss)
    pi = stationary_distribution(ss)
    eigs = eigenvalues_of_P(ss)
    gap = spectral_gap(ss)
    mix_bound = mixing_time_bound(ss)
    ht = all_hitting_times(ss)
    cover = cover_time_bound(ss)
    erg = is_ergodic(ss)

    return RandomWalkResult(
        transition_matrix=P,
        stationary_distribution=pi,
        spectral_gap=gap,
        mixing_time_bound=mix_bound,
        hitting_times=ht,
        cover_time_bound=cover,
        is_ergodic=erg,
        eigenvalues=eigs,
    )
