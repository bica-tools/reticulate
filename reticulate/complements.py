"""Complement analysis for session type lattices.

Step 363c: Analyses the complement structure of lattices derived from
session type state spaces. Provides complement finding, complemented
lattice classification, Boolean algebra detection, and relative
complementation.

Key concepts:
- Complement: a' such that a ∧ a' = ⊥ and a ∨ a' = ⊤
- Complemented lattice: every element has at least one complement
- Uniquely complemented: every element has exactly one complement
- Relatively complemented: every element in every interval has a complement
- Boolean: complemented + distributive (≅ 2^n for finite)

Operational interpretation for session types:
A complement a' of state a represents the maximally independent alternative
execution path — the path that together with a exhausts all reachable
behavior while overlapping only at terminal/initial states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from reticulate.lattice import check_lattice, check_distributive, LatticeResult, DistributivityResult
from reticulate.statespace import StateSpace


@dataclass(frozen=True)
class ComplementPair:
    """A pair (element, complement) in a lattice."""
    element: int
    complement: int


@dataclass(frozen=True)
class ComplementInfo:
    """Complement information for a single element."""
    element: int
    complements: tuple[int, ...]
    has_complement: bool
    is_unique: bool


@dataclass(frozen=True)
class RelativeComplement:
    """A relative complement within an interval [lower, upper]."""
    element: int
    complement: int
    lower: int
    upper: int


@dataclass(frozen=True)
class ComplementAnalysis:
    """Complete complement analysis of a lattice."""
    is_complemented: bool
    is_uniquely_complemented: bool
    is_relatively_complemented: bool
    is_boolean: bool
    is_distributive: bool
    complement_map: dict[int, tuple[int, ...]]
    uncomplemented_elements: tuple[int, ...]
    complement_count: int
    element_count: int
    boolean_rank: int | None  # n where L ≅ 2^n, or None


def _build_lattice_ops(ss: StateSpace, lr: LatticeResult):
    """Build meet/join lookup from lattice result using SCC quotient.

    Works on quotient representatives to handle cycles from recursion.
    reach[s] = set of states reachable from s (s can reach them).
    """
    scc_map = lr.scc_map or {}
    # Use quotient representatives as nodes
    reps = sorted(set(scc_map.values())) if scc_map else sorted(ss.states)
    if not reps:
        reps = sorted(ss.states)

    top_rep = scc_map.get(ss.top, ss.top)
    bottom_rep = scc_map.get(ss.bottom, ss.bottom)

    # Build adjacency on representatives
    fwd: dict[int, set[int]] = {r: set() for r in reps}
    for src, _lbl, tgt in ss.transitions:
        src_r = scc_map.get(src, src)
        tgt_r = scc_map.get(tgt, tgt)
        if src_r != tgt_r:
            fwd[src_r].add(tgt_r)

    # Transitive closure: reach[s] = states reachable FROM s
    reach: dict[int, set[int]] = {r: set() for r in reps}
    for s in reps:
        visited: set[int] = set()
        stack = [s]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            reach[s].add(cur)
            for t in fwd.get(cur, set()):
                if t not in visited:
                    stack.append(t)

    return reps, top_rep, bottom_rep, reach


def _compute_meet(reach: dict[int, set[int]], nodes: list[int],
                  a: int, b: int) -> int | None:
    """Compute meet (greatest lower bound) of a and b.

    n is below a iff a reaches n (n in reach[a]).
    Lower bounds of (a,b) = nodes reachable from both a and b.
    """
    if a == b:
        return a
    lower_bounds = [n for n in nodes if n in reach[a] and n in reach[b]]
    if not lower_bounds:
        return None
    # Greatest lower bound: the one that all other lower bounds can reach
    for candidate in lower_bounds:
        if all(lb in reach[candidate] or lb == candidate
               for lb in lower_bounds):
            return candidate
    return None


def _compute_join(reach: dict[int, set[int]], nodes: list[int],
                  a: int, b: int) -> int | None:
    """Compute join (least upper bound) of a and b.

    n is above a iff n reaches a (a in reach[n]).
    Upper bounds of (a,b) = nodes that can reach both a and b.
    """
    if a == b:
        return a
    upper_bounds = [n for n in nodes if a in reach[n] and b in reach[n]]
    if not upper_bounds:
        return None
    # Least upper bound: the one reachable from all other upper bounds
    for candidate in upper_bounds:
        if all(candidate in reach[ub] or candidate == ub
               for ub in upper_bounds):
            return candidate
    return None


def find_complements(ss: StateSpace,
                     element: int,
                     lr: LatticeResult | None = None) -> list[int]:
    """Find all complements of an element in the lattice.

    A complement a' of element a satisfies:
        a ∧ a' = ⊥  and  a ∨ a' = ⊤

    Returns a list of complement state IDs (quotient representatives).
    """
    if lr is None:
        lr = check_lattice(ss)
    if not lr.is_lattice:
        return []

    nodes, top, bottom, reach = _build_lattice_ops(ss, lr)
    scc_map = lr.scc_map or {}
    elem_rep = scc_map.get(element, element)
    complements = []
    for b in nodes:
        m = _compute_meet(reach, nodes, elem_rep, b)
        j = _compute_join(reach, nodes, elem_rep, b)
        if m == bottom and j == top:
            complements.append(b)
    return complements


def complement_info(ss: StateSpace,
                    element: int,
                    lr: LatticeResult | None = None) -> ComplementInfo:
    """Get detailed complement information for a single element."""
    comps = find_complements(ss, element, lr)
    return ComplementInfo(
        element=element,
        complements=tuple(comps),
        has_complement=len(comps) > 0,
        is_unique=len(comps) == 1,
    )


def find_relative_complement(ss: StateSpace,
                             element: int,
                             lower: int,
                             upper: int,
                             lr: LatticeResult | None = None) -> list[int]:
    """Find relative complements of element within interval [lower, upper].

    A relative complement c of element a in [lower, upper] satisfies:
        a ∧ c = lower  and  a ∨ c = upper
    """
    if lr is None:
        lr = check_lattice(ss)
    if not lr.is_lattice:
        return []

    nodes, _top, _bottom, reach = _build_lattice_ops(ss, lr)
    scc_map = lr.scc_map or {}
    elem_rep = scc_map.get(element, element)
    lower_rep = scc_map.get(lower, lower)
    upper_rep = scc_map.get(upper, upper)
    # Elements in the interval [lower, upper]:
    # n such that upper ≥ n ≥ lower, i.e., n in reach[upper] and lower in reach[n]
    interval = [n for n in nodes
                if n in reach[upper_rep] and lower_rep in reach[n]]
    if elem_rep not in interval:
        return []

    complements = []
    for c in interval:
        m = _compute_meet(reach, nodes, elem_rep, c)
        j = _compute_join(reach, nodes, elem_rep, c)
        if m == lower_rep and j == upper_rep:
            complements.append(c)
    return complements


def is_complemented(ss: StateSpace,
                    lr: LatticeResult | None = None) -> bool:
    """Check if the lattice is complemented (every element has a complement)."""
    if lr is None:
        lr = check_lattice(ss)
    if not lr.is_lattice:
        return False

    nodes, top, bottom, reach = _build_lattice_ops(ss, lr)
    for a in nodes:
        found = False
        for b in nodes:
            m = _compute_meet(reach, nodes, a, b)
            j = _compute_join(reach, nodes, a, b)
            if m == bottom and j == top:
                found = True
                break
        if not found:
            return False
    return True


def is_uniquely_complemented(ss: StateSpace,
                             lr: LatticeResult | None = None) -> bool:
    """Check if every element has exactly one complement."""
    if lr is None:
        lr = check_lattice(ss)
    if not lr.is_lattice:
        return False

    nodes, top, bottom, reach = _build_lattice_ops(ss, lr)
    for a in nodes:
        comp_count = 0
        for b in nodes:
            m = _compute_meet(reach, nodes, a, b)
            j = _compute_join(reach, nodes, a, b)
            if m == bottom and j == top:
                comp_count += 1
        if comp_count != 1:
            return False
    return True


def is_relatively_complemented(ss: StateSpace,
                               lr: LatticeResult | None = None) -> bool:
    """Check if the lattice is relatively complemented.

    Every element in every interval [a, b] has a complement within
    that interval.
    """
    if lr is None:
        lr = check_lattice(ss)
    if not lr.is_lattice:
        return False

    nodes, _top, _bottom, reach = _build_lattice_ops(ss, lr)
    # For each interval [lower, upper] where lower ≤ upper
    # In our ordering: upper reaches lower means upper ≥ lower
    for lower in nodes:
        for upper in nodes:
            if lower not in reach[upper]:
                continue  # upper does not reach lower, so upper < lower
            # Elements in interval [lower, upper]:
            # n such that upper ≥ n ≥ lower, i.e., n in reach[upper] and lower in reach[n]
            interval = [n for n in nodes
                        if n in reach[upper] and lower in reach[n]]
            # Each element in interval must have a complement
            for elem in interval:
                found = False
                for c in interval:
                    m = _compute_meet(reach, nodes, elem, c)
                    j = _compute_join(reach, nodes, elem, c)
                    if m == lower and j == upper:
                        found = True
                        break
                if not found:
                    return False
    return True


def is_boolean(ss: StateSpace,
               lr: LatticeResult | None = None) -> bool:
    """Check if the lattice is Boolean (complemented + distributive).

    A finite Boolean algebra is isomorphic to 2^n for some n.
    """
    if lr is None:
        lr = check_lattice(ss)
    if not lr.is_lattice:
        return False
    dr = check_distributive(ss)
    if not dr.is_distributive:
        return False
    return is_complemented(ss, lr)


def boolean_rank(ss: StateSpace,
                 lr: LatticeResult | None = None) -> int | None:
    """If the lattice is Boolean, return n where L ≅ 2^n.

    Returns None if the lattice is not Boolean.
    """
    if lr is None:
        lr = check_lattice(ss)
    if not is_boolean(ss, lr):
        return None
    # Count quotient nodes (SCC representatives)
    scc_map = lr.scc_map or {}
    n = len(set(scc_map.values())) if scc_map else len(ss.states)
    # 2^k = n  =>  k = log2(n)
    import math
    k = int(math.log2(n)) if n > 0 else 0
    if 2 ** k == n:
        return k
    return None


def analyze_complements(ss: StateSpace,
                        lr: LatticeResult | None = None) -> ComplementAnalysis:
    """Perform complete complement analysis of the lattice."""
    if lr is None:
        lr = check_lattice(ss)
    if not lr.is_lattice:
        return ComplementAnalysis(
            is_complemented=False,
            is_uniquely_complemented=False,
            is_relatively_complemented=False,
            is_boolean=False,
            is_distributive=False,
            complement_map={},
            uncomplemented_elements=tuple(range(len(ss.states))),
            complement_count=0,
            element_count=len(ss.states),
            boolean_rank=None,
        )

    nodes, top, bottom, reach = _build_lattice_ops(ss, lr)
    comp_map: dict[int, tuple[int, ...]] = {}
    uncomplemented: list[int] = []
    total_complements = 0

    for a in nodes:
        comps = []
        for b in nodes:
            m = _compute_meet(reach, nodes, a, b)
            j = _compute_join(reach, nodes, a, b)
            if m == bottom and j == top:
                comps.append(b)
        comp_map[a] = tuple(comps)
        total_complements += len(comps)
        if not comps:
            uncomplemented.append(a)

    complemented = len(uncomplemented) == 0
    uniquely_comp = complemented and all(
        len(v) == 1 for v in comp_map.values()
    )
    dr = check_distributive(ss)
    dist = dr.is_distributive
    bool_lattice = complemented and dist

    # Boolean rank
    b_rank = None
    if bool_lattice:
        import math
        n = len(nodes)
        k = int(math.log2(n)) if n > 0 else 0
        if 2 ** k == n:
            b_rank = k

    rel_comp = is_relatively_complemented(ss, lr) if complemented else False

    return ComplementAnalysis(
        is_complemented=complemented,
        is_uniquely_complemented=uniquely_comp,
        is_relatively_complemented=rel_comp,
        is_boolean=bool_lattice,
        is_distributive=dist,
        complement_map=comp_map,
        uncomplemented_elements=tuple(uncomplemented),
        complement_count=total_complements,
        element_count=len(nodes),
        boolean_rank=b_rank,
    )
