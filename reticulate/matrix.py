"""Algebraic invariants of session type lattices (Step 30).

Computes matrix representations and algebraic invariants of session
type state spaces viewed as finite lattices:

- **Zeta matrix** Z: Z[i,j] = 1 if i ≥ j (reachability)
- **Möbius matrix** M = Z⁻¹: encodes inclusion-exclusion
- **Möbius value** μ(⊤,⊥): Euler characteristic of the protocol
- **Rota characteristic polynomial** χ_P(t): lattice fingerprint
- **Adjacency spectrum**: eigenvalues of the Hasse diagram
- **Fiedler value** λ₁: algebraic connectivity (bottleneck measure)
- **Tropical distance matrix**: shortest/longest paths (min-plus)
- **Tropical eigenvalue**: max cycle mean (throughput bottleneck)
- **Von Neumann entropy**: spectral complexity

Key property: all invariants compose cleanly under ∥ (parallel):
  - μ is multiplicative: μ(L₁×L₂) = μ(L₁)·μ(L₂)
  - χ factors: χ_{L₁×L₂}(t) = χ_{L₁}(t)·χ_{L₂}(t)
  - spectrum is additive: spec(L₁×L₂) = {λᵢ+μⱼ}
  - Fiedler takes min: λ₁(L₁×L₂) = min(λ₁(L₁), λ₁(L₂))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AlgebraicInvariants:
    """Complete set of algebraic invariants for a session type lattice.

    Attributes:
        num_states: Number of states.
        mobius_value: μ(⊤,⊥) — Euler characteristic.
        rota_polynomial: Coefficients of χ_P(t), highest degree first.
        eigenvalues: Sorted eigenvalues of the undirected Hasse adjacency matrix.
        spectral_radius: Largest absolute eigenvalue.
        fiedler_value: Second-smallest Laplacian eigenvalue (algebraic connectivity).
        tropical_diameter: Longest shortest path (max over all pairs).
        tropical_eigenvalue: Maximum cycle mean (0 if acyclic).
        von_neumann_entropy: Spectral entropy of the Laplacian.
        num_join_irreducibles: Number of join-irreducible elements.
        num_meet_irreducibles: Number of meet-irreducible elements.
    """
    num_states: int
    mobius_value: int
    rota_polynomial: list[int]
    eigenvalues: list[float]
    spectral_radius: float
    fiedler_value: float
    tropical_diameter: int
    tropical_eigenvalue: float
    von_neumann_entropy: float
    num_join_irreducibles: int
    num_meet_irreducibles: int


# ---------------------------------------------------------------------------
# Reachability matrix (order relation)
# ---------------------------------------------------------------------------

def _state_list(ss: "StateSpace") -> list[int]:
    """Sorted list of state IDs for consistent indexing."""
    return sorted(ss.states)


def _reachability(ss: "StateSpace") -> dict[int, set[int]]:
    """Compute transitive closure: reach[s] = set of states reachable from s."""
    adj: dict[int, list[int]] = {s: [] for s in ss.states}
    for src, _, tgt in ss.transitions:
        adj[src].append(tgt)

    reach: dict[int, set[int]] = {}
    for s in ss.states:
        visited: set[int] = set()
        stack = [s]
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            for v in adj[u]:
                stack.append(v)
        reach[s] = visited
    return reach


# ---------------------------------------------------------------------------
# Zeta and Möbius matrices
# ---------------------------------------------------------------------------

def zeta_matrix(ss: "StateSpace") -> list[list[int]]:
    """Compute the zeta matrix Z where Z[i,j] = 1 if state i ≥ state j.

    Ordering: i ≥ j iff j is reachable from i (session type convention:
    top = initial state, bottom = end).
    """
    states = _state_list(ss)
    n = len(states)
    idx = {s: i for i, s in enumerate(states)}
    reach = _reachability(ss)

    Z = [[0] * n for _ in range(n)]
    for s in states:
        for t in reach[s]:
            Z[idx[s]][idx[t]] = 1
    return Z


def mobius_matrix(ss: "StateSpace") -> list[list[int]]:
    """Compute the Möbius matrix M = Z⁻¹ (inverse of the zeta matrix).

    Uses the recursive definition: μ(x,x) = 1, and for x > y:
    μ(x,y) = -Σ_{z: x≥z>y} μ(x,z).

    Processes each row x by BFS from x, computing μ(x,y) in order
    of increasing distance.
    """
    states = _state_list(ss)
    n = len(states)
    idx = {s: i for i, s in enumerate(states)}
    reach = _reachability(ss)

    adj: dict[int, list[int]] = {s: [] for s in ss.states}
    for src, _, tgt in ss.transitions:
        adj[src].append(tgt)

    M = [[0] * n for _ in range(n)]

    for xi, x in enumerate(states):
        M[xi][xi] = 1

        # BFS from x to compute μ(x,y) in topological order
        bfs_order: list[int] = []
        visited: set[int] = {x}
        queue = [x]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    bfs_order.append(v)
                    queue.append(v)

        for y in bfs_order:
            yi = idx[y]
            # μ(x,y) = -Σ_{z: x≥z>y} μ(x,z)
            # z must be reachable from x (guaranteed by bfs_order)
            # and y must be reachable from z (z > y)
            total = 0
            for z in states:
                zi = idx[z]
                if z == y:
                    continue
                if z in reach[x] and y in reach[z]:
                    total += M[xi][zi]
            M[xi][yi] = -total

    return M


def mobius_value(ss: "StateSpace") -> int:
    """Compute μ(⊤,⊥) — the Möbius function value from top to bottom.

    This equals the reduced Euler characteristic of the order complex
    of the open interval (⊥,⊤), up to sign.
    """
    M = mobius_matrix(ss)
    states = _state_list(ss)
    idx = {s: i for i, s in enumerate(states)}
    return M[idx[ss.top]][idx[ss.bottom]]


# ---------------------------------------------------------------------------
# Rota characteristic polynomial
# ---------------------------------------------------------------------------

def rota_polynomial(ss: "StateSpace") -> list[int]:
    """Compute Rota's characteristic polynomial χ_P(t).

    χ_P(t) = Σ_{x in L} μ(⊥,x) · t^{rank(⊤) - rank(x)}

    where rank is the length of the longest chain from ⊥ to x.

    Note: We use the convention μ(⊤,x) since our ordering has ⊤ = top.
    χ_P(t) = Σ_{x} μ(⊤,x) · t^{height - rank(x)}

    Returns coefficients [a_n, a_{n-1}, ..., a_0] (highest degree first).
    """
    states = _state_list(ss)
    idx = {s: i for i, s in enumerate(states)}
    M = mobius_matrix(ss)
    top_idx = idx[ss.top]

    # Compute rank of each state (shortest path from top to state)
    # Use BFS for shortest path to avoid infinite loops on cycles
    adj: dict[int, list[int]] = {s: [] for s in ss.states}
    for src, _, tgt in ss.transitions:
        adj[src].append(tgt)

    rank: dict[int, int] = {s: -1 for s in ss.states}
    rank[ss.top] = 0
    queue = [ss.top]
    visited: set[int] = {ss.top}
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                rank[v] = rank[u] + 1
                queue.append(v)
    # Assign rank 0 to unreachable states
    for s in ss.states:
        if rank[s] < 0:
            rank[s] = 0

    max_rank = max(rank.values()) if rank else 0

    # Build polynomial: χ_P(t) = Σ_x μ(⊤,x) · t^{max_rank - rank(x)}
    coeffs = [0] * (max_rank + 1)
    for x in states:
        xi = idx[x]
        mu_val = M[top_idx][xi]
        r = rank[x]
        degree = max_rank - r
        if 0 <= degree <= max_rank:
            coeffs[max_rank - degree] += mu_val

    # Remove leading zeros
    while len(coeffs) > 1 and coeffs[0] == 0:
        coeffs.pop(0)

    return coeffs


# ---------------------------------------------------------------------------
# Adjacency matrix and spectrum
# ---------------------------------------------------------------------------

def _hasse_edges(ss: "StateSpace") -> list[tuple[int, int]]:
    """Compute the covering relation (Hasse diagram edges).

    (a, b) is a covering pair iff a > b and no c with a > c > b.
    For efficiency, we check direct transitions and filter those
    where the target is also reachable via a longer path.
    """
    adj: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        adj[src].add(tgt)

    edges: list[tuple[int, int]] = []
    for a in ss.states:
        direct_targets = adj[a]
        for b in direct_targets:
            # a covers b iff there's no other path a → ... → b of length ≥ 2
            # Check: is b reachable from any other direct successor of a?
            is_covering = True
            for c in direct_targets:
                if c == b:
                    continue
                # BFS/DFS from c to check if b is reachable
                visited: set[int] = set()
                stack = [c]
                while stack:
                    u = stack.pop()
                    if u == b:
                        is_covering = False
                        break
                    if u in visited:
                        continue
                    visited.add(u)
                    for v in adj[u]:
                        if v not in visited:
                            stack.append(v)
                if not is_covering:
                    break
            if is_covering:
                edges.append((a, b))
    return edges


def adjacency_matrix(ss: "StateSpace") -> list[list[int]]:
    """Compute the adjacency matrix of the undirected Hasse diagram."""
    states = _state_list(ss)
    n = len(states)
    idx = {s: i for i, s in enumerate(states)}

    A = [[0] * n for _ in range(n)]
    for a, b in _hasse_edges(ss):
        A[idx[a]][idx[b]] = 1
        A[idx[b]][idx[a]] = 1
    return A


def adjacency_spectrum(ss: "StateSpace") -> list[float]:
    """Compute eigenvalues of the undirected Hasse adjacency matrix.

    Uses a simple QR-like approach for small matrices, or direct
    computation for tiny ones. Returns sorted eigenvalues.
    """
    A = adjacency_matrix(ss)
    n = len(A)
    if n == 0:
        return []
    if n == 1:
        return [0.0]

    # Convert to float
    mat = [[float(A[i][j]) for j in range(n)] for i in range(n)]
    eigenvalues = _eigenvalues_symmetric(mat)
    return sorted(eigenvalues)


def _eigenvalues_symmetric(A: list[list[float]]) -> list[float]:
    """Compute eigenvalues of a real symmetric matrix via Jacobi iteration.

    Simple O(n³) implementation sufficient for our small matrices (< 100).
    """
    n = len(A)
    if n == 1:
        return [A[0][0]]
    if n == 2:
        a, b, c, d = A[0][0], A[0][1], A[1][0], A[1][1]
        trace = a + d
        det = a * d - b * c
        disc = trace * trace - 4 * det
        if disc < 0:
            disc = 0.0
        sqrt_disc = math.sqrt(disc)
        return [(trace - sqrt_disc) / 2, (trace + sqrt_disc) / 2]

    # Jacobi eigenvalue algorithm for symmetric matrices
    # Copy matrix
    S = [row[:] for row in A]
    max_iter = 100 * n * n

    for _ in range(max_iter):
        # Find largest off-diagonal element
        p, q = 0, 1
        max_val = abs(S[0][1])
        for i in range(n):
            for j in range(i + 1, n):
                if abs(S[i][j]) > max_val:
                    max_val = abs(S[i][j])
                    p, q = i, j

        if max_val < 1e-12:
            break

        # Compute rotation
        if abs(S[p][p] - S[q][q]) < 1e-15:
            theta = math.pi / 4
        else:
            theta = 0.5 * math.atan2(2 * S[p][q], S[p][p] - S[q][q])

        c = math.cos(theta)
        s = math.sin(theta)

        # Apply rotation
        new_S = [row[:] for row in S]
        for i in range(n):
            if i != p and i != q:
                new_S[i][p] = c * S[i][p] + s * S[i][q]
                new_S[p][i] = new_S[i][p]
                new_S[i][q] = -s * S[i][p] + c * S[i][q]
                new_S[q][i] = new_S[i][q]
        new_S[p][p] = c * c * S[p][p] + 2 * s * c * S[p][q] + s * s * S[q][q]
        new_S[q][q] = s * s * S[p][p] - 2 * s * c * S[p][q] + c * c * S[q][q]
        new_S[p][q] = 0.0
        new_S[q][p] = 0.0
        S = new_S

    return sorted([S[i][i] for i in range(n)])


# ---------------------------------------------------------------------------
# Laplacian and Fiedler value
# ---------------------------------------------------------------------------

def laplacian_matrix(ss: "StateSpace") -> list[list[float]]:
    """Compute the Laplacian L = D - A of the undirected Hasse diagram."""
    A = adjacency_matrix(ss)
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        deg = sum(A[i])
        for j in range(n):
            if i == j:
                L[i][j] = float(deg)
            else:
                L[i][j] = -float(A[i][j])
    return L


def fiedler_value(ss: "StateSpace") -> float:
    """Compute the Fiedler value (algebraic connectivity).

    This is the second-smallest eigenvalue of the Laplacian.
    λ₁ > 0 iff the Hasse diagram is connected.
    Small λ₁ indicates a bottleneck in the protocol.
    """
    L = laplacian_matrix(ss)
    n = len(L)
    if n <= 1:
        return 0.0

    eigenvalues = _eigenvalues_symmetric(L)
    # Sort and return second smallest (first is always ~0)
    eigenvalues.sort()
    if len(eigenvalues) < 2:
        return 0.0
    return max(0.0, eigenvalues[1])  # Clamp numerical noise


# ---------------------------------------------------------------------------
# Tropical (min-plus) distance matrix
# ---------------------------------------------------------------------------

def tropical_distance_matrix(ss: "StateSpace") -> list[list[int]]:
    """Compute the all-pairs shortest path matrix (min-plus closure).

    D[i,j] = length of shortest path from state i to state j.
    D[i,j] = -1 if j is not reachable from i.
    """
    states = _state_list(ss)
    n = len(states)
    idx = {s: i for i, s in enumerate(states)}
    INF = n + 1  # Sentinel for unreachable

    D = [[INF] * n for _ in range(n)]
    for i in range(n):
        D[i][i] = 0

    for src, _, tgt in ss.transitions:
        si, ti = idx[src], idx[tgt]
        D[si][ti] = min(D[si][ti], 1)

    # Floyd-Warshall
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if D[i][k] + D[k][j] < D[i][j]:
                    D[i][j] = D[i][k] + D[k][j]

    # Replace INF with -1
    for i in range(n):
        for j in range(n):
            if D[i][j] >= INF:
                D[i][j] = -1

    return D


def tropical_diameter(ss: "StateSpace") -> int:
    """Compute the tropical diameter: max shortest path over all reachable pairs."""
    D = tropical_distance_matrix(ss)
    n = len(D)
    max_d = 0
    for i in range(n):
        for j in range(n):
            if D[i][j] > max_d and D[i][j] >= 0:
                max_d = D[i][j]
    return max_d


def tropical_eigenvalue(ss: "StateSpace") -> float:
    """Compute the tropical (max-plus) eigenvalue: maximum cycle mean.

    For an irreducible max-plus matrix, the unique eigenvalue is:
    λ = max over all cycles C of (weight(C) / length(C))

    For session types with unit weights, this is:
    λ = max over all simple cycles C of (|C| / |C|) = 1.0 if cycles exist, 0.0 otherwise.

    For acyclic state spaces, λ = 0.
    """
    # Detect cycles via DFS
    adj: dict[int, list[int]] = {s: [] for s in ss.states}
    for src, _, tgt in ss.transitions:
        adj[src].append(tgt)

    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[int, int] = {s: WHITE for s in ss.states}
    has_cycle = False

    def dfs(u: int) -> None:
        nonlocal has_cycle
        color[u] = GRAY
        for v in adj[u]:
            if color[v] == GRAY:
                has_cycle = True
                return
            if color[v] == WHITE:
                dfs(v)
                if has_cycle:
                    return
        color[u] = BLACK

    for s in ss.states:
        if color[s] == WHITE:
            dfs(s)
            if has_cycle:
                break

    return 1.0 if has_cycle else 0.0


# ---------------------------------------------------------------------------
# Von Neumann entropy
# ---------------------------------------------------------------------------

def von_neumann_entropy(ss: "StateSpace") -> float:
    """Compute the von Neumann entropy of the Laplacian.

    S(G) = -Σ_k (λ_k / Σ_j λ_j) · log(λ_k / Σ_j λ_j)

    where λ_k are the nonzero Laplacian eigenvalues.
    Measures the spectral complexity of the protocol.
    """
    L = laplacian_matrix(ss)
    n = len(L)
    if n <= 1:
        return 0.0

    eigenvalues = _eigenvalues_symmetric(L)
    # Filter positive eigenvalues (skip the zero one)
    pos_eigs = [e for e in eigenvalues if e > 1e-10]
    if not pos_eigs:
        return 0.0

    total = sum(pos_eigs)
    if total < 1e-15:
        return 0.0

    entropy = 0.0
    for e in pos_eigs:
        p = e / total
        if p > 1e-15:
            entropy -= p * math.log(p)

    return entropy


# ---------------------------------------------------------------------------
# Join-irreducibles and meet-irreducibles
# ---------------------------------------------------------------------------

def join_irreducibles(ss: "StateSpace") -> set[int]:
    """Find join-irreducible elements of the lattice.

    An element j is join-irreducible if it has exactly one lower cover
    (exactly one element that it covers in the Hasse diagram).
    Equivalently, j cannot be written as a ∨ b for a,b < j.
    """
    # Count incoming Hasse edges (lower covers = predecessors in Hasse)
    hasse = _hasse_edges(ss)
    # a covers b means a > b, so b has a as upper cover
    # j is join-irreducible if j has exactly one lower cover
    # lower cover of j = element covered by j (j covers x means (j,x) in hasse)
    lower_cover_count: dict[int, int] = {s: 0 for s in ss.states}
    for a, b in hasse:
        # a covers b: a is upper cover of b, b is lower cover of a
        lower_cover_count[a] += 1

    # Bottom element is never join-irreducible
    return {s for s in ss.states
            if lower_cover_count[s] == 1 and s != ss.bottom}


def meet_irreducibles(ss: "StateSpace") -> set[int]:
    """Find meet-irreducible elements of the lattice.

    An element m is meet-irreducible if it has exactly one upper cover.
    """
    hasse = _hasse_edges(ss)
    upper_cover_count: dict[int, int] = {s: 0 for s in ss.states}
    for a, b in hasse:
        # a covers b: a is upper cover of b
        upper_cover_count[b] += 1

    return {s for s in ss.states
            if upper_cover_count[s] == 1 and s != ss.top}


# ---------------------------------------------------------------------------
# High-level: compute all invariants
# ---------------------------------------------------------------------------

def algebraic_invariants(ss: "StateSpace") -> AlgebraicInvariants:
    """Compute all algebraic invariants of a session type lattice.

    This is the main entry point for Step 30 / the algebraic toolkit.
    """
    mu = mobius_value(ss)
    rota = rota_polynomial(ss)
    eigs = adjacency_spectrum(ss)
    fiedler = fiedler_value(ss)
    trop_diam = tropical_diameter(ss)
    trop_eig = tropical_eigenvalue(ss)
    vn_entropy = von_neumann_entropy(ss)
    ji = join_irreducibles(ss)
    mi = meet_irreducibles(ss)

    spectral_radius = max(abs(e) for e in eigs) if eigs else 0.0

    return AlgebraicInvariants(
        num_states=len(ss.states),
        mobius_value=mu,
        rota_polynomial=rota,
        eigenvalues=eigs,
        spectral_radius=spectral_radius,
        fiedler_value=fiedler,
        tropical_diameter=trop_diam,
        tropical_eigenvalue=trop_eig,
        von_neumann_entropy=vn_entropy,
        num_join_irreducibles=len(ji),
        num_meet_irreducibles=len(mi),
    )
