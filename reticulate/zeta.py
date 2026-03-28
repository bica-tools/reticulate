"""Zeta matrix analysis for session type lattices (Step 30a).

The zeta matrix Z of a finite poset P encodes the order relation:
Z[x,y] = 1 if x ≥ y, else 0.  For session type state spaces,
x ≥ y means y is reachable from x (top = initial, bottom = end).

This module provides deep analysis of the zeta matrix beyond the
basic computation in matrix.py:

- **Zeta matrix** and its structural properties (idempotent, triangular)
- **Rank function** from the zeta matrix (longest chain to bottom)
- **Interval enumeration** [x, y] = {z : x ≥ z ≥ y}
- **Chain counting** via zeta matrix powers: Z^k[x,y] counts chains of length k
- **Width** (maximum antichain size) via Dilworth's theorem
- **Height** (length of longest chain from top to bottom)
- **Incidence algebra** structure: convolution of functions on intervals
- **Composition under ∥**: Z(L₁ × L₂) = Z(L₁) ⊗ Z(L₂) (Kronecker product)
- **Factorization detection**: when Z decomposes as a Kronecker product

All computations are exact (integer arithmetic, no floating point).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ZetaResult:
    """Complete zeta matrix analysis for a session type state space.

    Attributes:
        num_states: Number of states in the poset.
        zeta: The zeta matrix Z[i][j] = 1 if state_i ≥ state_j.
        states: Sorted list of state IDs (index order matches matrix).
        height: Length of the longest chain from top to bottom.
        width: Size of the largest antichain (Dilworth's theorem).
        rank: Mapping state → rank (longest chain from that state to bottom).
        num_comparable_pairs: Number of (x,y) pairs with x ≥ y, x ≠ y.
        num_intervals: Total number of non-trivial intervals [x,y] with x > y.
        density: Fraction of pairs that are comparable: |{(x,y): x≥y}| / n².
        chain_counts: chain_counts[k] = number of chains of length exactly k.
        is_graded: True iff all maximal chains between any pair have same length.
        is_ranked: True iff rank function is consistent with covering relation.
    """
    num_states: int
    zeta: list[list[int]]
    states: list[int]
    height: int
    width: int
    rank: dict[int, int]
    num_comparable_pairs: int
    num_intervals: int
    density: float
    chain_counts: dict[int, int]
    is_graded: bool
    is_ranked: bool


@dataclass(frozen=True)
class IntervalInfo:
    """Information about a single interval [x, y] in the poset.

    Attributes:
        top: Upper bound state ID.
        bottom: Lower bound state ID.
        elements: States in the interval.
        size: Number of elements.
        chains: Number of maximal chains in this interval.
        height: Length of longest chain from top to bottom of interval.
    """
    top: int
    bottom: int
    elements: frozenset[int]
    size: int
    chains: int
    height: int


@dataclass(frozen=True)
class ZetaCompositionResult:
    """Result of analyzing zeta matrix composition under parallel.

    Attributes:
        left_states: Number of states in left component.
        right_states: Number of states in right component.
        product_states: Number of states in product.
        kronecker_matches: True iff Z(product) = Z(left) ⊗ Z(right).
        density_left: Order density of left lattice.
        density_right: Order density of right lattice.
        density_product: Order density of product lattice.
        density_product_expected: Expected density = density_left × density_right.
    """
    left_states: int
    right_states: int
    product_states: int
    kronecker_matches: bool
    density_left: float
    density_right: float
    density_product: float
    density_product_expected: float


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------

def _state_list(ss: "StateSpace") -> list[int]:
    """Sorted list of state IDs for consistent indexing."""
    return sorted(ss.states)


def _adjacency(ss: "StateSpace") -> dict[int, list[int]]:
    """Build adjacency list from transitions."""
    adj: dict[int, list[int]] = {s: [] for s in ss.states}
    for src, _, tgt in ss.transitions:
        if tgt not in adj[src]:
            adj[src].append(tgt)
    return adj


def _reachability(ss: "StateSpace") -> dict[int, set[int]]:
    """Compute transitive closure: reach[s] = states reachable from s."""
    adj = _adjacency(ss)
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


def zeta_matrix(ss: "StateSpace") -> list[list[int]]:
    """Compute the zeta matrix Z[i][j] = 1 if state_i ≥ state_j.

    Ordering: i ≥ j iff j is reachable from i.
    The diagonal is always 1 (reflexivity).
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


def _compute_sccs(ss: "StateSpace") -> tuple[dict[int, int], dict[int, set[int]]]:
    """Compute SCCs using iterative Tarjan's algorithm.

    Returns:
        scc_map: state → representative (smallest state ID in SCC)
        scc_members: representative → set of member states
    """
    adj = _adjacency(ss)
    index_counter = [0]
    stack: list[int] = []
    on_stack: set[int] = set()
    index: dict[int, int] = {}
    lowlink: dict[int, int] = {}
    scc_map: dict[int, int] = {}
    scc_members: dict[int, set[int]] = {}

    for start in sorted(ss.states):
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
            neighbors = adj.get(v, [])
            for i in range(ci, len(neighbors)):
                w = neighbors[i]
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
                scc: set[int] = set()
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    scc.add(w)
                    if w == v:
                        break
                rep = min(scc)
                for w in scc:
                    scc_map[w] = rep
                scc_members[rep] = scc
            work.pop()
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[v])

    return scc_map, scc_members


def compute_rank(ss: "StateSpace") -> dict[int, int]:
    """Compute rank function: rank(x) = length of longest chain from x to bottom.

    Quotients by SCCs first to handle cycles from recursion.
    States in the same SCC get the same rank.
    """
    bottom = ss.bottom if hasattr(ss, 'bottom') else min(ss.states)
    scc_map, scc_members = _compute_sccs(ss)
    adj = _adjacency(ss)

    # Build quotient DAG
    reps = sorted(scc_members.keys())
    q_adj: dict[int, set[int]] = {r: set() for r in reps}
    for s in ss.states:
        for t in adj[s]:
            sr, tr = scc_map[s], scc_map[t]
            if sr != tr:
                q_adj[sr].add(tr)

    bottom_rep = scc_map[bottom]

    # Longest path in quotient DAG from each rep to bottom_rep
    q_rank: dict[int, int] = {}

    def _longest(r: int, visited: set[int]) -> int:
        if r in q_rank:
            return q_rank[r]
        if r == bottom_rep:
            q_rank[r] = 0
            return 0
        if r in visited:
            return -1
        visited.add(r)
        best = -1
        for t in q_adj[r]:
            d = _longest(t, visited)
            if d >= 0:
                best = max(best, d + 1)
        visited.discard(r)
        if best < 0:
            q_rank[r] = 0
        else:
            q_rank[r] = best
        return q_rank[r]

    for r in reps:
        _longest(r, set())

    # Map back: each state gets its SCC representative's rank
    return {s: q_rank.get(scc_map[s], 0) for s in ss.states}


def compute_height(ss: "StateSpace") -> int:
    """Height of the poset: length of longest chain from top to bottom."""
    ranks = compute_rank(ss)
    top = ss.top if hasattr(ss, 'top') else max(ss.states)
    return ranks.get(top, 0)


def compute_width(ss: "StateSpace") -> int:
    """Width of the poset: size of the largest antichain.

    Uses Dilworth's theorem: width = n - max_matching on comparability graph.
    For small state spaces, we use a direct greedy approach.
    """
    states = _state_list(ss)
    n = len(states)
    if n <= 1:
        return n

    reach = _reachability(ss)

    # Build comparability: (i,j) comparable if i reaches j or j reaches i
    comparable = [[False] * n for _ in range(n)]
    idx = {s: i for i, s in enumerate(states)}
    for s in states:
        for t in states:
            if s != t:
                si, ti = idx[s], idx[t]
                if t in reach[s] or s in reach[t]:
                    comparable[si][ti] = True

    # Find maximum antichain by finding minimum chain cover
    # Use König's theorem: max antichain = n - max matching in bipartite graph
    # Bipartite graph: left = states, right = states, edge if i < j
    # Build bipartite graph for matching
    graph: dict[int, list[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j:
                si, sj = states[i], states[j]
                if sj in reach[si] and si != sj:
                    graph[i].append(j)

    # Hopcroft-Karp-style maximum bipartite matching
    match_left: dict[int, int] = {}
    match_right: dict[int, int] = {}

    def _augment(u: int, visited: set[int]) -> bool:
        for v in graph[u]:
            if v in visited:
                continue
            visited.add(v)
            if v not in match_right or _augment(match_right[v], visited):
                match_left[u] = v
                match_right[v] = u
                return True
        return False

    for u in range(n):
        _augment(u, set())

    max_matching = len(match_left)
    return n - max_matching


def _covering_relation(ss: "StateSpace") -> list[tuple[int, int]]:
    """Compute covering relation: x covers y iff x > y and no z with x > z > y."""
    states = _state_list(ss)
    reach = _reachability(ss)
    covers: list[tuple[int, int]] = []

    for x in states:
        for y in states:
            if x == y:
                continue
            if y not in reach[x]:
                continue
            # x ≥ y — check if x covers y (no intermediate z)
            is_cover = True
            for z in states:
                if z == x or z == y:
                    continue
                if z in reach[x] and y in reach[z]:
                    is_cover = False
                    break
            if is_cover:
                covers.append((x, y))

    return covers


def check_graded(ss: "StateSpace") -> bool:
    """Check if the poset is graded (all maximal chains have the same length).

    A poset is graded iff there exists a rank function ρ such that
    x covers y implies ρ(x) = ρ(y) + 1.
    """
    covers = _covering_relation(ss)
    ranks = compute_rank(ss)

    for x, y in covers:
        if ranks[x] != ranks[y] + 1:
            return False
    return True


# ---------------------------------------------------------------------------
# Interval analysis
# ---------------------------------------------------------------------------

def compute_interval(ss: "StateSpace", x: int, y: int) -> frozenset[int]:
    """Compute the interval [x, y] = {z : x ≥ z ≥ y}.

    Returns empty set if x does not dominate y.
    """
    reach = _reachability(ss)
    if y not in reach[x]:
        return frozenset()

    return frozenset(
        z for z in ss.states
        if z in reach[x] and y in reach[z]
    )


def interval_info(ss: "StateSpace", x: int, y: int) -> IntervalInfo:
    """Compute detailed information about the interval [x, y]."""
    elements = compute_interval(ss, x, y)
    if not elements:
        return IntervalInfo(
            top=x, bottom=y, elements=frozenset(),
            size=0, chains=0, height=0,
        )

    reach = _reachability(ss)

    # Build sub-adjacency restricted to interval
    adj: dict[int, list[int]] = {s: [] for s in elements}
    for src, _, tgt in ss.transitions:
        if src in elements and tgt in elements:
            if tgt not in adj[src]:
                adj[src].append(tgt)

    # Count maximal chains from x to y via DFS
    chain_count = 0
    max_length = 0

    def _count_chains(s: int, length: int) -> None:
        nonlocal chain_count, max_length
        if s == y:
            chain_count += 1
            max_length = max(max_length, length)
            return
        for t in adj[s]:
            _count_chains(t, length + 1)

    _count_chains(x, 0)

    return IntervalInfo(
        top=x, bottom=y, elements=elements,
        size=len(elements), chains=chain_count,
        height=max_length,
    )


def enumerate_intervals(ss: "StateSpace") -> list[IntervalInfo]:
    """Enumerate all non-trivial intervals [x, y] with x > y."""
    states = _state_list(ss)
    reach = _reachability(ss)
    intervals: list[IntervalInfo] = []

    for x in states:
        for y in states:
            if x != y and y in reach[x]:
                info = interval_info(ss, x, y)
                if info.size > 0:
                    intervals.append(info)

    return intervals


# ---------------------------------------------------------------------------
# Chain counting via matrix powers
# ---------------------------------------------------------------------------

def _mat_mul(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
    """Integer matrix multiplication."""
    n = len(A)
    m = len(B[0]) if B else 0
    p = len(B)
    C = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            s = 0
            for k in range(p):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C


def zeta_power(Z: list[list[int]], k: int) -> list[list[int]]:
    """Compute Z^k. Z^k[i,j] counts chains of length ≤ k from i to j.

    Specifically, if H is the Hasse (covering) matrix, then H^k[i,j]
    counts chains of length exactly k. But Z^k uses the full order.
    """
    n = len(Z)
    # Start with identity
    result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    base = [row[:] for row in Z]
    exp = k
    while exp > 0:
        if exp % 2 == 1:
            result = _mat_mul(result, base)
        base = _mat_mul(base, base)
        exp //= 2
    return result


def chain_counts(ss: "StateSpace") -> dict[int, int]:
    """Count chains of each length from top to bottom.

    Returns dict mapping chain_length → count.
    Uses the covering (Hasse) matrix: H^k[top, bottom] = number of
    chains of length exactly k.
    """
    states = _state_list(ss)
    n = len(states)
    if n == 0:
        return {}

    idx = {s: i for i, s in enumerate(states)}
    top = ss.top if hasattr(ss, 'top') else max(ss.states)
    bottom = ss.bottom if hasattr(ss, 'bottom') else min(ss.states)
    top_i = idx[top]
    bottom_i = idx[bottom]

    # Build Hasse matrix (covering relation only)
    covers = _covering_relation(ss)
    H = [[0] * n for _ in range(n)]
    for x, y in covers:
        H[idx[x]][idx[y]] = 1

    # Compute H^k for k = 1, 2, ..., until H^k is zero
    counts: dict[int, int] = {}
    power = [row[:] for row in H]  # H^1

    for k in range(1, n + 1):
        val = power[top_i][bottom_i]
        if val > 0:
            counts[k] = val
        # Check if power matrix is all zeros
        all_zero = all(power[i][j] == 0 for i in range(n) for j in range(n))
        if all_zero:
            break
        # Next power: H^(k+1) = H^k * H
        power = _mat_mul(power, H)

    return counts


# ---------------------------------------------------------------------------
# Density and comparable pairs
# ---------------------------------------------------------------------------

def order_density(Z: list[list[int]]) -> float:
    """Compute order density: fraction of pairs (i,j) with Z[i,j] = 1.

    Density 1.0 means total order; density 1/n means just the diagonal.
    """
    n = len(Z)
    if n == 0:
        return 0.0
    total = sum(Z[i][j] for i in range(n) for j in range(n))
    return total / (n * n)


def count_comparable_pairs(Z: list[list[int]]) -> int:
    """Count ordered pairs (i,j) with i > j (strict, off-diagonal)."""
    n = len(Z)
    return sum(
        Z[i][j] for i in range(n) for j in range(n) if i != j
    )


# ---------------------------------------------------------------------------
# Composition under parallel (Kronecker product)
# ---------------------------------------------------------------------------

def kronecker_product(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
    """Compute the Kronecker (tensor) product A ⊗ B.

    For lattices L₁, L₂: Z(L₁ × L₂) = Z(L₁) ⊗ Z(L₂).
    This is the fundamental composition theorem for zeta matrices.
    """
    na = len(A)
    nb = len(B)
    n = na * nb
    C = [[0] * n for _ in range(n)]
    for i in range(na):
        for j in range(na):
            for k in range(nb):
                for l in range(nb):
                    C[i * nb + k][j * nb + l] = A[i][j] * B[k][l]
    return C


def verify_kronecker_composition(
    ss_left: "StateSpace",
    ss_right: "StateSpace",
    ss_product: "StateSpace",
) -> ZetaCompositionResult:
    """Verify that Z(product) = Z(left) ⊗ Z(right).

    This validates the fundamental theorem that the zeta matrix of a
    product lattice is the Kronecker product of the component zeta matrices.
    """
    Z_left = zeta_matrix(ss_left)
    Z_right = zeta_matrix(ss_right)
    Z_product = zeta_matrix(ss_product)
    Z_expected = kronecker_product(Z_left, Z_right)

    n_left = len(Z_left)
    n_right = len(Z_right)
    n_prod = len(Z_product)

    # Check dimensions match
    matches = (n_prod == n_left * n_right)
    if matches:
        # Check element-wise
        for i in range(n_prod):
            for j in range(n_prod):
                if Z_product[i][j] != Z_expected[i][j]:
                    matches = False
                    break
            if not matches:
                break

    d_left = order_density(Z_left)
    d_right = order_density(Z_right)
    d_product = order_density(Z_product)

    return ZetaCompositionResult(
        left_states=n_left,
        right_states=n_right,
        product_states=n_prod,
        kronecker_matches=matches,
        density_left=d_left,
        density_right=d_right,
        density_product=d_product,
        density_product_expected=d_left * d_right,
    )


# ---------------------------------------------------------------------------
# Zeta matrix properties
# ---------------------------------------------------------------------------

def is_upper_triangular(Z: list[list[int]], states: list[int], rank: dict[int, int]) -> bool:
    """Check if Z is upper triangular when rows/cols are ordered by rank.

    In a graded poset, the zeta matrix is upper triangular in rank order.
    """
    n = len(states)
    # Sort indices by descending rank (highest rank = first row)
    order = sorted(range(n), key=lambda i: -rank[states[i]])
    for row_pos, i in enumerate(order):
        for col_pos, j in enumerate(order):
            if row_pos > col_pos and Z[i][j] != 0:
                return False
    return True


def zeta_trace(Z: list[list[int]]) -> int:
    """Trace of the zeta matrix = number of states (always n for a poset)."""
    return sum(Z[i][i] for i in range(len(Z)))


def zeta_row_sums(Z: list[list[int]], states: list[int]) -> dict[int, int]:
    """Row sums of Z: row_sum(x) = |{y : x ≥ y}| = size of downset ↓x."""
    n = len(Z)
    return {states[i]: sum(Z[i]) for i in range(n)}


def zeta_col_sums(Z: list[list[int]], states: list[int]) -> dict[int, int]:
    """Column sums of Z: col_sum(y) = |{x : x ≥ y}| = size of upset ↑y."""
    n = len(Z)
    return {states[j]: sum(Z[i][j] for i in range(n)) for j in range(n)}


# ---------------------------------------------------------------------------
# Incidence algebra
# ---------------------------------------------------------------------------

def convolve(
    f: dict[tuple[int, int], int],
    g: dict[tuple[int, int], int],
    ss: "StateSpace",
) -> dict[tuple[int, int], int]:
    """Convolution in the incidence algebra: (f * g)(x, y) = Σ_{x≥z≥y} f(x,z)·g(z,y).

    f and g are functions on intervals, represented as dicts (x, y) → value.
    For cyclic state spaces, operates on the SCC quotient to ensure
    the Möbius inversion formula holds.
    """
    scc_map, scc_members = _compute_sccs(ss)
    reps = sorted(scc_members.keys())

    # Build quotient reachability
    adj = _adjacency(ss)
    q_adj: dict[int, set[int]] = {r: set() for r in reps}
    for s in ss.states:
        for t in adj[s]:
            sr, tr = scc_map[s], scc_map[t]
            if sr != tr:
                q_adj[sr].add(tr)

    q_reach: dict[int, set[int]] = {}
    for r in reps:
        visited: set[int] = set()
        stack = [r]
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            for v in q_adj[u]:
                stack.append(v)
        q_reach[r] = visited

    # Convolve on representatives, map back
    result: dict[tuple[int, int], int] = {}

    for x in ss.states:
        xr = scc_map[x]
        for y in ss.states:
            yr = scc_map[y]
            if yr not in q_reach[xr]:
                continue
            total = 0
            for zr in reps:
                if zr in q_reach[xr] and yr in q_reach[zr]:
                    # Pick representative state for z
                    z = min(scc_members[zr])
                    fval = f.get((x, z), 0)
                    gval = g.get((z, y), 0)
                    total += fval * gval
            if total != 0:
                result[(x, y)] = total

    return result


def zeta_function(ss: "StateSpace") -> dict[tuple[int, int], int]:
    """The zeta function ζ(x, y) = 1 if x ≥ y, else 0.

    For cyclic state spaces, states in the same SCC are treated as
    equivalent (ζ(x,y) = 1 for all x,y in same SCC).
    """
    scc_map, scc_members = _compute_sccs(ss)
    adj = _adjacency(ss)
    reps = sorted(scc_members.keys())

    # Quotient reachability
    q_adj: dict[int, set[int]] = {r: set() for r in reps}
    for s in ss.states:
        for t in adj[s]:
            sr, tr = scc_map[s], scc_map[t]
            if sr != tr:
                q_adj[sr].add(tr)

    q_reach: dict[int, set[int]] = {}
    for r in reps:
        visited: set[int] = set()
        stack = [r]
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            for v in q_adj[u]:
                stack.append(v)
        q_reach[r] = visited

    result: dict[tuple[int, int], int] = {}
    for x in ss.states:
        xr = scc_map[x]
        for y in ss.states:
            yr = scc_map[y]
            if yr in q_reach[xr]:
                result[(x, y)] = 1
    return result


def delta_function(ss: "StateSpace") -> dict[tuple[int, int], int]:
    """The delta (Kronecker) function δ(x, y) = 1 if x ≡ y (same SCC).

    For acyclic state spaces, this is just δ(x,x) = 1.
    For cyclic ones, states in the same SCC are identified.
    """
    scc_map, _ = _compute_sccs(ss)
    result: dict[tuple[int, int], int] = {}
    for s in ss.states:
        for t in ss.states:
            if scc_map[s] == scc_map[t]:
                result[(s, t)] = 1
    return result


def mobius_function(ss: "StateSpace") -> dict[tuple[int, int], int]:
    """The Möbius function μ(x, y) — inverse of ζ in the incidence algebra.

    For cyclic state spaces (from recursion), works on the SCC quotient
    and maps back. States in the same SCC are treated as equivalent.

    μ(x, x) = 1 for all x.
    μ(x, y) = -Σ_{x≥z>y} μ(x, z) for x > y.
    """
    scc_map, scc_members = _compute_sccs(ss)
    reps = sorted(scc_members.keys())
    adj = _adjacency(ss)

    # Build quotient adjacency and reachability
    q_adj: dict[int, set[int]] = {r: set() for r in reps}
    for s in ss.states:
        for t in adj[s]:
            sr, tr = scc_map[s], scc_map[t]
            if sr != tr:
                q_adj[sr].add(tr)

    # Quotient reachability
    q_reach: dict[int, set[int]] = {}
    for r in reps:
        visited: set[int] = set()
        stack = [r]
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            for v in q_adj[u]:
                stack.append(v)
        q_reach[r] = visited

    # Compute quotient rank for topological ordering
    q_rank: dict[int, int] = {}
    bottom_rep = scc_map.get(
        ss.bottom if hasattr(ss, 'bottom') else min(ss.states),
        reps[0] if reps else 0,
    )

    def _q_longest(r: int, vis: set[int]) -> int:
        if r in q_rank:
            return q_rank[r]
        if r == bottom_rep:
            q_rank[r] = 0
            return 0
        if r in vis:
            return -1
        vis.add(r)
        best = -1
        for t in q_adj[r]:
            d = _q_longest(t, vis)
            if d >= 0:
                best = max(best, d + 1)
        vis.discard(r)
        q_rank[r] = max(best, 0)
        return q_rank[r]

    for r in reps:
        _q_longest(r, set())

    # Compute Möbius on quotient — process in decreasing rank (top first)
    q_mu: dict[tuple[int, int], int] = {}

    for x in reps:
        q_mu[(x, x)] = 1

        # Process states reachable from x in decreasing rank order
        reachable = [r for r in reps if r in q_reach[x] and r != x]
        reachable.sort(key=lambda r: -q_rank.get(r, 0))

        for y in reachable:
            total = 0
            for z in reps:
                if z == y:
                    continue
                if z in q_reach[x] and y in q_reach[z]:
                    total += q_mu.get((x, z), 0)
            q_mu[(x, y)] = -total

    # Map back to original states
    mu: dict[tuple[int, int], int] = {}
    for s in ss.states:
        for t in ss.states:
            sr, tr = scc_map[s], scc_map[t]
            val = q_mu.get((sr, tr), 0)
            if val != 0 or s == t:
                mu[(s, t)] = val if val != 0 else (1 if s == t else 0)
            # Ensure diagonal
            if s == t:
                mu[(s, t)] = 1

    return mu


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze_zeta(ss: "StateSpace") -> ZetaResult:
    """Complete zeta matrix analysis of a session type state space.

    Computes the zeta matrix and derives all structural properties:
    rank function, height, width, intervals, chain counts, gradedness.
    """
    Z = zeta_matrix(ss)
    states = _state_list(ss)
    n = len(states)

    rank = compute_rank(ss)
    height = compute_height(ss)
    width = compute_width(ss)
    comparable = count_comparable_pairs(Z)
    density = order_density(Z)
    chains = chain_counts(ss)
    graded = check_graded(ss)

    # Count non-trivial intervals
    reach = _reachability(ss)
    num_intervals = sum(
        1 for x in states for y in states
        if x != y and y in reach[x]
    )

    # Check ranked: covering x ▷ y implies rank(x) = rank(y) + 1
    ranked = graded  # For finite posets, graded ↔ ranked

    return ZetaResult(
        num_states=n,
        zeta=Z,
        states=states,
        height=height,
        width=width,
        rank=rank,
        num_comparable_pairs=comparable,
        num_intervals=num_intervals,
        density=density,
        chain_counts=chains,
        is_graded=graded,
        is_ranked=ranked,
    )
