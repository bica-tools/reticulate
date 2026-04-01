"""Congruence lattices of session types (Step 33a).

The **congruence lattice** Con(L) of a lattice L catalogues ALL valid
quotients (abstractions) of a protocol. A congruence θ is an equivalence
relation on L compatible with meet and join:

    x ≡ y (mod θ)  ⟹  x∧z ≡ y∧z  and  x∨z ≡ y∨z

Key theorems:
- **Funayama-Nakayama (1942)**: Con(L) is always distributive when L is modular
- **Birkhoff**: For distributive L, congruences biject with downsets of J(L)
- **Products**: Con(L₁ × L₂) ≅ Con(L₁) × Con(L₂)
- **Simplicity**: Con(L) = {0,1} iff L has no non-trivial quotients

Key functions:
  - ``principal_congruence(ss, a, b)``   -- θ(a,b): smallest congruence collapsing a,b
  - ``enumerate_congruences(ss)``         -- all congruences of L
  - ``congruence_lattice(ss)``            -- Con(L) with refinement ordering
  - ``quotient_lattice(ss, cong)``        -- build L/θ as a StateSpace
  - ``is_simple(ss)``                     -- Con(L) = {0,1}?
  - ``birkhoff_congruences(ss)``          -- fast path for distributive lattices
  - ``simplification_options(ss)``        -- ranked quotient suggestions
  - ``analyze_congruences(ss)``           -- full analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Congruence:
    """A congruence on a lattice (equivalence relation compatible with ∧,∨).

    Attributes:
        classes: List of equivalence classes (each a frozenset of state IDs).
        num_classes: Number of equivalence classes.
        generators: Pairs (a,b) that generate this congruence (if known).
    """
    classes: tuple[frozenset[int], ...]
    num_classes: int
    generators: tuple[tuple[int, int], ...] = ()

    @property
    def is_trivial_bottom(self) -> bool:
        """True iff this is the identity congruence (all singletons)."""
        return all(len(c) == 1 for c in self.classes)

    @property
    def is_trivial_top(self) -> bool:
        """True iff this is the total congruence (one big class)."""
        return self.num_classes == 1

    def class_of(self, state: int) -> frozenset[int]:
        """Return the equivalence class containing state."""
        for c in self.classes:
            if state in c:
                return c
        return frozenset({state})


@dataclass(frozen=True)
class CongruenceLattice:
    """The congruence lattice Con(L).

    Attributes:
        congruences: All congruences, indexed by position.
        bottom: The identity congruence (finest partition).
        top: The total congruence (single class).
        ordering: (i,j) in ordering iff congruences[i] ≤ congruences[j] (refines).
        num_congruences: |Con(L)|.
        is_distributive: True iff Con(L) is distributive.
    """
    congruences: list[Congruence]
    bottom: int  # index of identity congruence
    top: int  # index of total congruence
    ordering: frozenset[tuple[int, int]]
    num_congruences: int
    is_distributive: bool


@dataclass(frozen=True)
class CongruenceAnalysis:
    """Full congruence analysis of a session type.

    Attributes:
        con_lattice: The congruence lattice Con(L).
        num_congruences: |Con(L)|.
        is_simple: True iff Con(L) = {⊥, ⊤} (no non-trivial quotients).
        is_modular: True iff L is modular (implies Con(L) distributive).
        is_distributive_lattice: True iff L is distributive.
        num_join_irreducibles: |J(L)| for distributive fast path.
        max_quotient_size: Largest non-trivial quotient size.
        min_quotient_size: Smallest non-trivial quotient size.
        num_states: |L|.
    """
    con_lattice: CongruenceLattice
    num_congruences: int
    is_simple: bool
    is_modular: bool
    is_distributive_lattice: bool
    num_join_irreducibles: int
    max_quotient_size: int
    min_quotient_size: int
    num_states: int


# ---------------------------------------------------------------------------
# Internal: Union-Find
# ---------------------------------------------------------------------------

class _UnionFind:
    """Efficient union-find for building congruences."""

    def __init__(self, elements: set[int]) -> None:
        self.parent: dict[int, int] = {e: e for e in elements}
        self.rank: dict[int, int] = {e: 0 for e in elements}

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

    def partition(self) -> list[frozenset[int]]:
        groups: dict[int, set[int]] = {}
        for e in self.parent:
            r = self.find(e)
            groups.setdefault(r, set()).add(e)
        return [frozenset(g) for g in groups.values()]


# ---------------------------------------------------------------------------
# Public API: Principal congruences
# ---------------------------------------------------------------------------

def principal_congruence(ss: StateSpace, a: int, b: int) -> Congruence:
    """Compute θ(a,b): the smallest congruence identifying a and b.

    Uses iterative closure: start with a ≡ b, propagate through ∧ and ∨.
    """
    from reticulate.lattice import compute_meet, compute_join

    uf = _UnionFind(ss.states)
    uf.union(a, b)

    states_list = sorted(ss.states)
    changed = True
    max_iters = len(ss.states) ** 2

    for _ in range(max_iters):
        if not changed:
            break
        changed = False
        for s1 in states_list:
            for s2 in states_list:
                if s1 >= s2 or uf.find(s1) != uf.find(s2):
                    continue
                for x in states_list:
                    m1 = compute_meet(ss, s1, x)
                    m2 = compute_meet(ss, s2, x)
                    if m1 is not None and m2 is not None:
                        if uf.union(m1, m2):
                            changed = True

                    j1 = compute_join(ss, s1, x)
                    j2 = compute_join(ss, s2, x)
                    if j1 is not None and j2 is not None:
                        if uf.union(j1, j2):
                            changed = True

    classes = tuple(sorted(uf.partition(), key=lambda c: min(c)))
    return Congruence(
        classes=classes,
        num_classes=len(classes),
        generators=((a, b),),
    )


# ---------------------------------------------------------------------------
# Public API: Enumerate congruences
# ---------------------------------------------------------------------------

def enumerate_congruences(ss: StateSpace, max_congruences: int = 1000) -> list[Congruence]:
    """Enumerate all congruences of the lattice.

    Strategy: generate all principal congruences θ(a,b), then close
    under join (finest common coarsening) to get Con(L).
    """
    states_list = sorted(ss.states)
    n = len(states_list)

    # Identity congruence (bottom of Con)
    identity = Congruence(
        classes=tuple(frozenset({s}) for s in states_list),
        num_classes=n,
    )

    # Total congruence (top of Con)
    total = Congruence(
        classes=(frozenset(ss.states),),
        num_classes=1,
    )

    if n <= 1:
        return [identity]

    # Compute all principal congruences
    principals: dict[tuple[frozenset[int], ...], Congruence] = {}
    principals[identity.classes] = identity
    principals[total.classes] = total

    for i, a in enumerate(states_list):
        for b in states_list[i + 1:]:
            if len(principals) >= max_congruences:
                break
            pc = principal_congruence(ss, a, b)
            key = pc.classes
            if key not in principals:
                principals[key] = pc

    # Close under join: for each pair of congruences, compute their join
    all_congs = dict(principals)
    changed = True
    while changed and len(all_congs) < max_congruences:
        changed = False
        keys = list(all_congs.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                if len(all_congs) >= max_congruences:
                    break
                joined = _join_congruences(ss, all_congs[keys[i]], all_congs[keys[j]])
                if joined.classes not in all_congs:
                    all_congs[joined.classes] = joined
                    changed = True

    return sorted(all_congs.values(), key=lambda c: c.num_classes)


def _join_congruences(ss: StateSpace, c1: Congruence, c2: Congruence) -> Congruence:
    """Compute the join of two congruences (finest congruence coarser than both).

    Merge all pairs that are equivalent in either c1 or c2, then close.
    """
    from reticulate.lattice import compute_meet, compute_join

    uf = _UnionFind(ss.states)

    # Merge pairs from c1
    for cls in c1.classes:
        elems = sorted(cls)
        for i in range(1, len(elems)):
            uf.union(elems[0], elems[i])

    # Merge pairs from c2
    for cls in c2.classes:
        elems = sorted(cls)
        for i in range(1, len(elems)):
            uf.union(elems[0], elems[i])

    # Close under meet/join
    states_list = sorted(ss.states)
    changed = True
    for _ in range(len(ss.states) ** 2):
        if not changed:
            break
        changed = False
        for s1 in states_list:
            for s2 in states_list:
                if s1 >= s2 or uf.find(s1) != uf.find(s2):
                    continue
                for x in states_list:
                    m1 = compute_meet(ss, s1, x)
                    m2 = compute_meet(ss, s2, x)
                    if m1 is not None and m2 is not None:
                        if uf.union(m1, m2):
                            changed = True
                    j1 = compute_join(ss, s1, x)
                    j2 = compute_join(ss, s2, x)
                    if j1 is not None and j2 is not None:
                        if uf.union(j1, j2):
                            changed = True

    classes = tuple(sorted(uf.partition(), key=lambda c: min(c)))
    return Congruence(classes=classes, num_classes=len(classes))


# ---------------------------------------------------------------------------
# Public API: Congruence lattice
# ---------------------------------------------------------------------------

def congruence_lattice(ss: StateSpace) -> CongruenceLattice:
    """Build the full congruence lattice Con(L)."""
    congs = enumerate_congruences(ss)

    # Find bottom (identity) and top (total)
    bottom_idx = 0
    top_idx = len(congs) - 1
    for i, c in enumerate(congs):
        if c.is_trivial_bottom:
            bottom_idx = i
        if c.is_trivial_top:
            top_idx = i

    # Refinement ordering: c1 ≤ c2 iff every class of c1 is a subset of some class of c2
    ordering: set[tuple[int, int]] = set()
    for i in range(len(congs)):
        for j in range(len(congs)):
            if _refines(congs[i], congs[j]):
                ordering.add((i, j))

    # Check distributivity of Con(L)
    # For small Con, check directly
    is_dist = True  # Con(L) of a finite lattice is always distributive (Funayama-Nakayama)

    return CongruenceLattice(
        congruences=congs,
        bottom=bottom_idx,
        top=top_idx,
        ordering=frozenset(ordering),
        num_congruences=len(congs),
        is_distributive=is_dist,
    )


def _refines(c1: Congruence, c2: Congruence) -> bool:
    """Check if c1 refines c2 (c1 ≤ c2): every class of c1 ⊆ some class of c2."""
    for cls1 in c1.classes:
        found = False
        for cls2 in c2.classes:
            if cls1.issubset(cls2):
                found = True
                break
        if not found:
            return False
    return True


# ---------------------------------------------------------------------------
# Public API: Quotient lattice
# ---------------------------------------------------------------------------

def quotient_lattice(ss: StateSpace, cong: Congruence) -> "StateSpace":
    """Build the quotient lattice L/θ as a new StateSpace.

    Each equivalence class becomes a state. Transitions between classes
    are induced by the original transitions.
    """
    from reticulate.statespace import StateSpace as SS

    # Map each state to its class representative (minimum element)
    class_of: dict[int, int] = {}
    for cls in cong.classes:
        rep = min(cls)
        for s in cls:
            class_of[s] = rep

    # Build quotient states and transitions
    q_states: set[int] = {class_of[s] for s in ss.states}
    q_transitions: list[tuple[int, str, int]] = []
    seen: set[tuple[int, str, int]] = set()

    for src, label, tgt in ss.transitions:
        q_src = class_of[src]
        q_tgt = class_of[tgt]
        edge = (q_src, label, q_tgt)
        if edge not in seen and q_src != q_tgt:  # Skip self-loops from merging
            seen.add(edge)
            q_transitions.append(edge)

    q_top = class_of[ss.top]
    q_bottom = class_of[ss.bottom] if ss.bottom in class_of else min(q_states)

    # Build selection transitions for the quotient
    q_selections: set[tuple[int, str, int]] = set()
    for src, label, tgt in ss.transitions:
        if ss.is_selection(src, label, tgt):
            q_src = class_of[src]
            q_tgt = class_of[tgt]
            if q_src != q_tgt:
                q_selections.add((q_src, label, q_tgt))

    return SS(
        states=q_states,
        transitions=q_transitions,
        top=q_top,
        bottom=q_bottom,
        labels={s: f"[{s}]" for s in q_states},
        selection_transitions=q_selections,
    )


# ---------------------------------------------------------------------------
# Public API: Simplicity
# ---------------------------------------------------------------------------

def is_simple(ss: StateSpace) -> bool:
    """Check if L is simple: Con(L) = {⊥, ⊤} (only trivial congruences)."""
    congs = enumerate_congruences(ss)
    # Simple iff only identity and total congruences exist
    return len(congs) <= 2


# ---------------------------------------------------------------------------
# Public API: Birkhoff fast path
# ---------------------------------------------------------------------------

def birkhoff_congruences(ss: StateSpace) -> list[Congruence] | None:
    """Fast congruence enumeration for distributive lattices.

    Birkhoff's theorem: congruences of a distributive lattice biject with
    downsets of the poset of join-irreducible elements J(L).

    Returns None if L is not distributive.
    """
    from reticulate.lattice import check_distributive, compute_meet, compute_join

    dist = check_distributive(ss)
    if not dist.is_distributive:
        return None

    # Find join-irreducible elements: x ∈ J(L) iff x covers exactly one element
    reach = _reachability(ss)
    states_list = sorted(ss.states)

    # Build covering relation
    covers: dict[int, list[int]] = {s: [] for s in states_list}
    for s in states_list:
        for t in states_list:
            if s == t:
                continue
            if t in reach.get(s, set()):
                # s covers t iff no intermediate u with s > u > t
                is_cover = True
                for u in states_list:
                    if u == s or u == t:
                        continue
                    if t in reach.get(u, set()) and u in reach.get(s, set()):
                        is_cover = False
                        break
                if is_cover:
                    covers[s].append(t)

    join_irr: list[int] = []
    for s in states_list:
        if s == ss.bottom:
            continue
        lower_covers = covers[s]
        if len(lower_covers) == 1:
            join_irr.append(s)

    # Enumerate downsets of J(L)
    # A downset D ⊆ J(L) is downward-closed in the J(L) ordering
    # For small J(L), enumerate all subsets and filter
    n_ji = len(join_irr)
    if n_ji > 20:
        return None  # Too many join-irreducibles

    # J(L) ordering: j1 ≤ j2 in J(L) iff j1 ≤ j2 in L
    ji_order: dict[int, set[int]] = {j: set() for j in join_irr}
    for j1 in join_irr:
        for j2 in join_irr:
            if j1 != j2 and j1 in reach.get(j2, set()):
                ji_order[j2].add(j1)  # j2 ≥ j1

    # Enumerate downsets
    downsets: list[frozenset[int]] = [frozenset()]  # empty downset

    for mask in range(1, 1 << n_ji):
        subset = frozenset(join_irr[i] for i in range(n_ji) if mask & (1 << i))
        # Check downward-closure
        is_downset = True
        for j in subset:
            for below in ji_order[j]:
                if below in frozenset(join_irr) and below not in subset:
                    is_downset = False
                    break
            if not is_downset:
                break
        if is_downset:
            downsets.append(subset)

    # Convert each downset to a congruence
    # Downset D corresponds to: merge j with its unique lower cover, for each j ∈ D
    congruences: list[Congruence] = []

    # Identity congruence (empty downset)
    identity = Congruence(
        classes=tuple(frozenset({s}) for s in states_list),
        num_classes=len(states_list),
    )
    congruences.append(identity)

    for ds in downsets:
        if not ds:
            continue  # Already added identity
        # Build congruence by merging each j with its lower cover
        uf = _UnionFind(ss.states)
        for j in ds:
            lc = covers[j]
            if lc:
                uf.union(j, lc[0])

        # Close under meet/join
        changed = True
        for _ in range(len(ss.states) ** 2):
            if not changed:
                break
            changed = False
            for s1 in states_list:
                for s2 in states_list:
                    if s1 >= s2 or uf.find(s1) != uf.find(s2):
                        continue
                    for x in states_list:
                        m1 = compute_meet(ss, s1, x)
                        m2 = compute_meet(ss, s2, x)
                        if m1 is not None and m2 is not None:
                            if uf.union(m1, m2):
                                changed = True
                        j1 = compute_join(ss, s1, x)
                        j2 = compute_join(ss, s2, x)
                        if j1 is not None and j2 is not None:
                            if uf.union(j1, j2):
                                changed = True

        classes = tuple(sorted(uf.partition(), key=lambda c: min(c)))
        congruences.append(Congruence(classes=classes, num_classes=len(classes)))

    # Deduplicate
    seen: set[tuple[frozenset[int], ...]] = set()
    unique: list[Congruence] = []
    for c in congruences:
        if c.classes not in seen:
            seen.add(c.classes)
            unique.append(c)

    return sorted(unique, key=lambda c: c.num_classes)


def _reachability(ss: StateSpace) -> dict[int, set[int]]:
    """Forward reachability (inclusive)."""
    adj: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        adj[src].add(tgt)
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
    return reach


# ---------------------------------------------------------------------------
# Public API: Simplification options
# ---------------------------------------------------------------------------

def simplification_options(ss: StateSpace) -> list[tuple[Congruence, int]]:
    """Return ranked list of non-trivial quotients, ordered by quotient size.

    Each entry is (congruence, quotient_size). Smaller quotients = more abstract.
    """
    congs = enumerate_congruences(ss)
    options: list[tuple[Congruence, int]] = []
    for c in congs:
        if c.is_trivial_bottom or c.is_trivial_top:
            continue
        options.append((c, c.num_classes))
    return sorted(options, key=lambda x: x[1])


# ---------------------------------------------------------------------------
# Public API: Full analysis
# ---------------------------------------------------------------------------

def analyze_congruences(ss: StateSpace) -> CongruenceAnalysis:
    """Full congruence lattice analysis."""
    from reticulate.lattice import check_distributive

    con = congruence_lattice(ss)
    dist = check_distributive(ss)
    is_mod = dist.is_modular if dist.is_lattice else False
    is_dist = dist.is_distributive if dist.is_lattice else False

    # Count join-irreducibles
    reach = _reachability(ss)
    states_list = sorted(ss.states)
    covers_count: dict[int, int] = {s: 0 for s in states_list}
    for s in states_list:
        for t in states_list:
            if s == t or t not in reach.get(s, set()):
                continue
            is_cover = True
            for u in states_list:
                if u == s or u == t:
                    continue
                if t in reach.get(u, set()) and u in reach.get(s, set()):
                    is_cover = False
                    break
            if is_cover:
                covers_count[s] += 1
    n_ji = sum(1 for s in states_list if s != ss.bottom and covers_count[s] == 1)

    # Quotient sizes
    non_trivial = [c for c in con.congruences
                   if not c.is_trivial_bottom and not c.is_trivial_top]
    max_q = max((c.num_classes for c in non_trivial), default=0)
    min_q = min((c.num_classes for c in non_trivial), default=0)

    return CongruenceAnalysis(
        con_lattice=con,
        num_congruences=con.num_congruences,
        is_simple=len(non_trivial) == 0,
        is_modular=is_mod,
        is_distributive_lattice=is_dist,
        num_join_irreducibles=n_ji,
        max_quotient_size=max_q,
        min_quotient_size=min_q,
        num_states=len(ss.states),
    )
