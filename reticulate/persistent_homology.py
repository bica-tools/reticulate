"""Persistent homology of session type lattices (Step 33c).

Persistent homology tracks how the topology of the order complex CHANGES
as we build it up through a filtration. It answers: "Which topological
features of a protocol are robust (long bars) and which are fragile
(short bars)?"

A **filtration** is a nested sequence of subcomplexes:
    ∅ = K₀ ⊆ K₁ ⊆ ... ⊆ Kₙ = K

As simplices are added, homology classes are **born** (new holes appear)
or **die** (holes get filled). The persistence diagram records (birth, death)
pairs. Long bars = robust features. Short bars = noise.

Filtrations for session types:
1. **Rank filtration**: add states by rank level (bottom → top)
2. **Reverse rank**: add from top → bottom
3. **Sublevel**: by BFS distance from a center state

Key functions:
  - ``rank_filtration(ss)``            -- build rank-based filtration
  - ``compute_persistence(filtration)`` -- persistence via column reduction
  - ``bottleneck_distance(pd1, pd2)``  -- metric between protocols
  - ``persistence_entropy(pd)``        -- Shannon entropy of bar lengths
  - ``analyze_persistence(ss)``        -- full analysis
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PersistencePair:
    """A birth-death pair in the persistence diagram.

    Attributes:
        birth: Filtration level at which the feature is born.
        death: Filtration level at which the feature dies (inf if it persists).
        dimension: Homological dimension (0 = component, 1 = loop, ...).
        persistence: death - birth (lifetime of the feature).
    """
    birth: float
    death: float
    dimension: int

    @property
    def persistence(self) -> float:
        if math.isinf(self.death):
            return float('inf')
        return self.death - self.birth

    @property
    def is_infinite(self) -> bool:
        return math.isinf(self.death)


@dataclass(frozen=True)
class PersistenceDiagram:
    """Persistence diagram: collection of birth-death pairs.

    Attributes:
        pairs: All finite persistence pairs.
        infinite_pairs: Pairs that never die (essential features).
        max_filtration: Maximum filtration level.
    """
    pairs: tuple[PersistencePair, ...]
    infinite_pairs: tuple[PersistencePair, ...]
    max_filtration: float

    @property
    def all_pairs(self) -> tuple[PersistencePair, ...]:
        return self.pairs + self.infinite_pairs

    @property
    def num_pairs(self) -> int:
        return len(self.pairs) + len(self.infinite_pairs)

    def pairs_in_dim(self, dim: int) -> list[PersistencePair]:
        return [p for p in self.all_pairs if p.dimension == dim]


@dataclass(frozen=True)
class Filtration:
    """A filtration of simplices with filtration values.

    Attributes:
        simplices: List of simplices (each a tuple of vertex IDs).
        filtration_values: Filtration level for each simplex.
        max_level: Maximum filtration level.
        num_simplices: Total number of simplices.
    """
    simplices: list[tuple[int, ...]]
    filtration_values: list[float]
    max_level: float
    num_simplices: int


@dataclass(frozen=True)
class PersistenceAnalysis:
    """Full persistent homology analysis.

    Attributes:
        diagram: The persistence diagram.
        filtration: The filtration used.
        total_persistence: Sum of all finite bar lengths.
        max_persistence: Length of the longest finite bar.
        persistence_entropy: Shannon entropy of bar length distribution.
        num_born_dim0: Number of 0-dimensional features born.
        num_born_dim1: Number of 1-dimensional features born.
        betti_0: Final Betti number β₀.
        betti_1: Final Betti number β₁.
        num_states: Number of lattice elements.
    """
    diagram: PersistenceDiagram
    filtration: Filtration
    total_persistence: float
    max_persistence: float
    persistence_entropy: float
    num_born_dim0: int
    num_born_dim1: int
    betti_0: int
    betti_1: int
    num_states: int


# ---------------------------------------------------------------------------
# Internal: rank computation
# ---------------------------------------------------------------------------

def _rank(ss: StateSpace) -> dict[int, int]:
    """BFS distance from bottom (rank)."""
    rev: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        rev.setdefault(tgt, set()).add(src)
    dist: dict[int, int] = {}
    if ss.bottom in ss.states:
        dist[ss.bottom] = 0
        queue = [ss.bottom]
        while queue:
            s = queue.pop(0)
            for pred in rev.get(s, set()):
                if pred not in dist:
                    dist[pred] = dist[s] + 1
                    queue.append(pred)
    for s in ss.states:
        if s not in dist:
            dist[s] = 0
    return dist


def _covering_relations(ss: StateSpace) -> list[tuple[int, int]]:
    """Covering relations (Hasse edges) with cycle avoidance."""
    adj: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        adj[src].add(tgt)

    # Reachability
    reach: dict[int, set[int]] = {}
    for s in ss.states:
        visited: set[int] = set()
        stack = [s]
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            for t in adj.get(v, set()):
                stack.append(t)
        reach[s] = visited

    covers: list[tuple[int, int]] = []
    for s in ss.states:
        for t in adj[s]:
            if t == s:
                continue
            is_cover = True
            for u in adj[s]:
                if u != t and u != s and t in reach.get(u, set()):
                    is_cover = False
                    break
            if is_cover:
                covers.append((s, t))
    return covers


# ---------------------------------------------------------------------------
# Public API: Filtrations
# ---------------------------------------------------------------------------

def rank_filtration(ss: StateSpace) -> Filtration:
    """Build a rank-based filtration.

    Add vertices by rank (bottom = rank 0 first), then edges between
    vertices that are both present.
    """
    r = _rank(ss)
    covers = _covering_relations(ss)
    states = sorted(ss.states, key=lambda s: r.get(s, 0))

    simplices: list[tuple[int, ...]] = []
    filt_vals: list[float] = []

    # Add vertices at their rank level
    for s in states:
        simplices.append((s,))
        filt_vals.append(float(r.get(s, 0)))

    # Add edges at max(rank of endpoints)
    for s, t in covers:
        level = max(r.get(s, 0), r.get(t, 0))
        simplices.append(tuple(sorted((s, t))))
        filt_vals.append(float(level))

    # Add 2-simplices (triangles) for chains of length 3
    state_set = set(ss.states)
    for s in ss.states:
        for t in ss.states:
            if t == s:
                continue
            for u in ss.states:
                if u == s or u == t:
                    continue
                # Check if (s, t, u) form a chain: s > t > u
                if (s, t) in set(covers) and (t, u) in set(covers):
                    tri = tuple(sorted((s, t, u)))
                    if tri not in [tuple(x) for x in simplices]:
                        level = max(r.get(s, 0), r.get(t, 0), r.get(u, 0))
                        simplices.append(tri)
                        filt_vals.append(float(level))

    max_level = max(filt_vals) if filt_vals else 0.0

    return Filtration(
        simplices=simplices,
        filtration_values=filt_vals,
        max_level=max_level,
        num_simplices=len(simplices),
    )


def reverse_rank_filtration(ss: StateSpace) -> Filtration:
    """Build a reverse-rank filtration (top → bottom)."""
    r = _rank(ss)
    max_r = max(r.values()) if r else 0
    covers = _covering_relations(ss)
    states = sorted(ss.states, key=lambda s: -r.get(s, 0))

    simplices: list[tuple[int, ...]] = []
    filt_vals: list[float] = []

    for s in states:
        simplices.append((s,))
        filt_vals.append(float(max_r - r.get(s, 0)))

    for s, t in covers:
        level = max(max_r - r.get(s, 0), max_r - r.get(t, 0))
        simplices.append(tuple(sorted((s, t))))
        filt_vals.append(float(level))

    max_level = max(filt_vals) if filt_vals else 0.0

    return Filtration(
        simplices=simplices,
        filtration_values=filt_vals,
        max_level=max_level,
        num_simplices=len(simplices),
    )


def sublevel_filtration(ss: StateSpace, center: int) -> Filtration:
    """Build a sublevel filtration by BFS distance from center."""
    adj: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        adj[src].add(tgt)
        adj[tgt].add(src)  # Undirected BFS

    dist: dict[int, int] = {center: 0}
    queue = [center]
    while queue:
        s = queue.pop(0)
        for t in adj.get(s, set()):
            if t not in dist:
                dist[t] = dist[s] + 1
                queue.append(t)

    for s in ss.states:
        if s not in dist:
            dist[s] = len(ss.states)

    covers = _covering_relations(ss)
    simplices: list[tuple[int, ...]] = []
    filt_vals: list[float] = []

    for s in sorted(ss.states, key=lambda s: dist[s]):
        simplices.append((s,))
        filt_vals.append(float(dist[s]))

    for s, t in covers:
        level = max(dist.get(s, 0), dist.get(t, 0))
        simplices.append(tuple(sorted((s, t))))
        filt_vals.append(float(level))

    max_level = max(filt_vals) if filt_vals else 0.0

    return Filtration(
        simplices=simplices,
        filtration_values=filt_vals,
        max_level=max_level,
        num_simplices=len(simplices),
    )


# ---------------------------------------------------------------------------
# Public API: Persistence computation
# ---------------------------------------------------------------------------

def compute_persistence(filtration: Filtration) -> PersistenceDiagram:
    """Compute persistent homology via the standard column reduction algorithm.

    Sorts simplices by filtration value, builds boundary matrix, reduces
    columns to extract birth-death pairs.
    """
    if filtration.num_simplices == 0:
        return PersistenceDiagram(pairs=(), infinite_pairs=(), max_filtration=0.0)

    # Sort simplices by (filtration_value, dimension, lexicographic)
    indexed = list(zip(filtration.simplices, filtration.filtration_values))
    indexed.sort(key=lambda x: (x[1], len(x[0]), x[0]))

    simplices = [s for s, _ in indexed]
    filt_vals = [v for _, v in indexed]
    n = len(simplices)

    # Simplex index lookup
    simplex_idx: dict[tuple[int, ...], int] = {}
    for i, s in enumerate(simplices):
        simplex_idx[s] = i

    # Build boundary matrix (sparse: list of column sets)
    boundary: list[set[int]] = [set() for _ in range(n)]
    for j in range(n):
        sigma = simplices[j]
        if len(sigma) <= 1:
            continue  # 0-simplices have empty boundary
        # Boundary of a k-simplex = alternating sum of (k-1)-faces
        for i in range(len(sigma)):
            face = tuple(sigma[:i] + sigma[i + 1:])
            face = tuple(sorted(face))
            if face in simplex_idx:
                boundary[j].add(simplex_idx[face])

    # Column reduction (standard persistence algorithm)
    # low[j] = max index in column j (or -1 if zero)
    low: list[int] = [-1] * n
    for j in range(n):
        low[j] = max(boundary[j]) if boundary[j] else -1

    # Reduce: for each column, if low[j] == low[j'] for j' < j, add column j' to j
    for j in range(n):
        while low[j] >= 0:
            # Find earlier column with same low
            found = False
            for j_prime in range(j):
                if low[j_prime] == low[j]:
                    # Add column j_prime to j (mod 2)
                    boundary[j] = boundary[j].symmetric_difference(boundary[j_prime])
                    low[j] = max(boundary[j]) if boundary[j] else -1
                    found = True
                    break
            if not found:
                break

    # Extract persistence pairs
    finite_pairs: list[PersistencePair] = []
    infinite_pairs: list[PersistencePair] = []
    paired: set[int] = set()

    for j in range(n):
        if low[j] >= 0:
            i = low[j]
            paired.add(i)
            paired.add(j)
            birth = filt_vals[i]
            death = filt_vals[j]
            dim = len(simplices[i]) - 1  # dimension of the born feature
            if abs(death - birth) > 1e-12:  # Skip zero-persistence pairs
                finite_pairs.append(PersistencePair(birth=birth, death=death, dimension=dim))

    # Unpaired simplices give infinite bars
    for j in range(n):
        if j not in paired:
            birth = filt_vals[j]
            dim = len(simplices[j]) - 1
            infinite_pairs.append(PersistencePair(birth=birth, death=float('inf'), dimension=dim))

    return PersistenceDiagram(
        pairs=tuple(sorted(finite_pairs, key=lambda p: (p.dimension, p.birth))),
        infinite_pairs=tuple(sorted(infinite_pairs, key=lambda p: (p.dimension, p.birth))),
        max_filtration=filtration.max_level,
    )


# ---------------------------------------------------------------------------
# Public API: Persistence invariants
# ---------------------------------------------------------------------------

def total_persistence(pd: PersistenceDiagram) -> float:
    """Sum of all finite bar lengths."""
    return sum(p.persistence for p in pd.pairs if not p.is_infinite)


def max_persistence(pd: PersistenceDiagram) -> float:
    """Length of the longest finite bar."""
    finite = [p.persistence for p in pd.pairs if not p.is_infinite]
    return max(finite) if finite else 0.0


def persistence_entropy(pd: PersistenceDiagram) -> float:
    """Shannon entropy of the bar length distribution.

    Normalized bar lengths as probabilities: p_i = l_i / Σl_j.
    """
    lengths = [p.persistence for p in pd.pairs if not p.is_infinite and p.persistence > 0]
    if not lengths:
        return 0.0
    total = sum(lengths)
    if total <= 0:
        return 0.0
    probs = [l / total for l in lengths]
    return -sum(p * math.log2(p) for p in probs if p > 0)


# ---------------------------------------------------------------------------
# Public API: Distances between persistence diagrams
# ---------------------------------------------------------------------------

def bottleneck_distance(pd1: PersistenceDiagram, pd2: PersistenceDiagram) -> float:
    """Bottleneck distance between two persistence diagrams.

    d_B = inf over matchings M of sup |p - M(p)|_∞.

    For simplicity, we compute an upper bound using a greedy matching.
    """
    # Collect all finite pairs
    pts1 = [(p.birth, p.death) for p in pd1.pairs]
    pts2 = [(p.birth, p.death) for p in pd2.pairs]

    # Add diagonal points for unmatched
    n1, n2 = len(pts1), len(pts2)

    if n1 == 0 and n2 == 0:
        return 0.0

    # Greedy: match nearest pairs
    used2: set[int] = set()
    max_cost = 0.0

    for i in range(n1):
        best_j = -1
        best_cost = float('inf')
        for j in range(n2):
            if j in used2:
                continue
            cost = max(abs(pts1[i][0] - pts2[j][0]), abs(pts1[i][1] - pts2[j][1]))
            if cost < best_cost:
                best_cost = cost
                best_j = j
        if best_j >= 0:
            used2.add(best_j)
            max_cost = max(max_cost, best_cost)
        else:
            # Unmatched: distance to diagonal
            diag_cost = (pts1[i][1] - pts1[i][0]) / 2.0
            max_cost = max(max_cost, diag_cost)

    # Unmatched points in pd2
    for j in range(n2):
        if j not in used2:
            diag_cost = (pts2[j][1] - pts2[j][0]) / 2.0
            max_cost = max(max_cost, diag_cost)

    return max_cost


def wasserstein_distance(
    pd1: PersistenceDiagram,
    pd2: PersistenceDiagram,
    p: float = 1.0,
) -> float:
    """p-Wasserstein distance between persistence diagrams (greedy approximation)."""
    pts1 = [(pt.birth, pt.death) for pt in pd1.pairs]
    pts2 = [(pt.birth, pt.death) for pt in pd2.pairs]

    if not pts1 and not pts2:
        return 0.0

    used2: set[int] = set()
    total_cost = 0.0

    for i in range(len(pts1)):
        best_j = -1
        best_cost = float('inf')
        for j in range(len(pts2)):
            if j in used2:
                continue
            cost = max(abs(pts1[i][0] - pts2[j][0]), abs(pts1[i][1] - pts2[j][1]))
            if cost < best_cost:
                best_cost = cost
                best_j = j
        if best_j >= 0:
            used2.add(best_j)
            total_cost += best_cost ** p
        else:
            diag_cost = (pts1[i][1] - pts1[i][0]) / 2.0
            total_cost += diag_cost ** p

    for j in range(len(pts2)):
        if j not in used2:
            diag_cost = (pts2[j][1] - pts2[j][0]) / 2.0
            total_cost += diag_cost ** p

    return total_cost ** (1.0 / p) if p > 0 else total_cost


# ---------------------------------------------------------------------------
# Public API: Convenience
# ---------------------------------------------------------------------------

def persistence_pairs(ss: StateSpace) -> list[PersistencePair]:
    """Compute persistence pairs using rank filtration."""
    filt = rank_filtration(ss)
    pd = compute_persistence(filt)
    return list(pd.all_pairs)


def betti_barcodes(ss: StateSpace) -> dict[int, list[tuple[float, float]]]:
    """Barcodes per dimension: {dim: [(birth, death), ...]}."""
    filt = rank_filtration(ss)
    pd = compute_persistence(filt)
    result: dict[int, list[tuple[float, float]]] = {}
    for p in pd.all_pairs:
        result.setdefault(p.dimension, []).append((p.birth, p.death))
    return result


# ---------------------------------------------------------------------------
# Public API: Full analysis
# ---------------------------------------------------------------------------

def analyze_persistence(ss: StateSpace) -> PersistenceAnalysis:
    """Full persistent homology analysis."""
    filt = rank_filtration(ss)
    pd = compute_persistence(filt)

    tp = total_persistence(pd)
    mp = max_persistence(pd)
    pe = persistence_entropy(pd)

    dim0_born = len([p for p in pd.all_pairs if p.dimension == 0])
    dim1_born = len([p for p in pd.all_pairs if p.dimension == 1])

    # Final Betti numbers from infinite bars
    betti_0 = len([p for p in pd.infinite_pairs if p.dimension == 0])
    betti_1 = len([p for p in pd.infinite_pairs if p.dimension == 1])

    return PersistenceAnalysis(
        diagram=pd,
        filtration=filt,
        total_persistence=tp,
        max_persistence=mp,
        persistence_entropy=pe,
        num_born_dim0=dim0_born,
        num_born_dim1=dim1_born,
        betti_0=betti_0,
        betti_1=betti_1,
        num_states=len(ss.states),
    )
