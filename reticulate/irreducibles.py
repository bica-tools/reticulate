"""Join and meet irreducibles for session type lattices.

An element j of a lattice L is **join-irreducible** if it cannot be expressed
as the join of two strictly smaller elements.  Equivalently, j has exactly one
lower cover in the Hasse diagram.  Dually, m is **meet-irreducible** if it has
exactly one upper cover.

For distributive lattices, Birkhoff's representation theorem says that L is
isomorphic to the lattice of downsets of its poset of join-irreducibles J(L).
This module computes J(L), M(L), the Birkhoff dual, and related analyses.

Step 30i of the 1000 Steps Towards Session Types as Algebraic Reticulates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace

from reticulate.lattice import (
    _build_quotient_poset,
    _join_on_quotient,
    _meet_on_quotient,
    check_distributive,
    check_lattice,
    compute_join,
    compute_meet,
)


# ---------------------------------------------------------------------------
# Hasse diagram (covering relation) — local helper
# ---------------------------------------------------------------------------

def _hasse_edges(ss: StateSpace) -> list[tuple[int, int]]:
    """Compute covering pairs (a, b) where a covers b (a > b, nothing between).

    Uses the quotient poset to handle cycles from recursion.
    """
    q = _build_quotient_poset(ss)
    edges: list[tuple[int, int]] = []
    for a in q.nodes:
        # direct successors of a in the DAG
        direct = q.fwd_adj[a]
        for b in direct:
            # a covers b iff no other direct successor c of a reaches b
            is_covering = True
            for c in direct:
                if c == b:
                    continue
                if b in q.fwd_reach[c]:
                    is_covering = False
                    break
            if is_covering:
                edges.append((q.rep[a], q.rep[b]))
    return edges


def _lower_covers(ss: StateSpace) -> dict[int, list[int]]:
    """For each state, compute its lower covers (elements it covers)."""
    hasse = _hasse_edges(ss)
    result: dict[int, list[int]] = {s: [] for s in ss.states}
    # (a, b) means a covers b => b is a lower cover of a
    for a, b in hasse:
        if a in result:
            result[a].append(b)
    return result


def _upper_covers(ss: StateSpace) -> dict[int, list[int]]:
    """For each state, compute its upper covers (elements that cover it)."""
    hasse = _hasse_edges(ss)
    result: dict[int, list[int]] = {s: [] for s in ss.states}
    # (a, b) means a covers b => a is an upper cover of b
    for a, b in hasse:
        if b in result:
            result[b].append(a)
    return result


# ---------------------------------------------------------------------------
# Single-element checks
# ---------------------------------------------------------------------------

def is_join_irreducible(ss: StateSpace, state: int) -> bool:
    """Check whether *state* is join-irreducible in the lattice of *ss*.

    An element is join-irreducible if:
    1. It is not the bottom element.
    2. It has exactly one lower cover in the Hasse diagram.
    """
    if state == ss.bottom:
        return False
    if state not in ss.states:
        raise ValueError(f"State {state} not in state space")
    lc = _lower_covers(ss)
    return len(lc.get(state, [])) == 1


def is_meet_irreducible(ss: StateSpace, state: int) -> bool:
    """Check whether *state* is meet-irreducible in the lattice of *ss*.

    An element is meet-irreducible if:
    1. It is not the top element.
    2. It has exactly one upper cover in the Hasse diagram.
    """
    if state == ss.top:
        return False
    if state not in ss.states:
        raise ValueError(f"State {state} not in state space")
    uc = _upper_covers(ss)
    return len(uc.get(state, [])) == 1


# ---------------------------------------------------------------------------
# Set-level computations
# ---------------------------------------------------------------------------

def join_irreducibles(ss: StateSpace) -> set[int]:
    """Find all join-irreducible elements of the lattice.

    An element j is join-irreducible iff it has exactly one lower cover
    and is not the bottom element.
    """
    lc = _lower_covers(ss)
    return {s for s in ss.states
            if len(lc.get(s, [])) == 1 and s != ss.bottom}


def meet_irreducibles(ss: StateSpace) -> set[int]:
    """Find all meet-irreducible elements of the lattice.

    An element m is meet-irreducible iff it has exactly one upper cover
    and is not the top element.
    """
    uc = _upper_covers(ss)
    return {s for s in ss.states
            if len(uc.get(s, [])) == 1 and s != ss.top}


# ---------------------------------------------------------------------------
# Birkhoff dual (poset of join-irreducibles)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BirkhoffDualResult:
    """Result of the Birkhoff dual computation.

    Attributes:
        join_irr: The set of join-irreducible elements.
        meet_irr: The set of meet-irreducible elements.
        dual_order: The partial order on join-irreducibles (set of (a, b)
            pairs where a <= b in the induced order).
        dual_states: The states of the dual poset (= join_irr).
        dual_hasse: Covering pairs of the dual poset.
        is_distributive: Whether the original lattice is distributive.
        downset_count: Number of downsets of J(L). For distributive lattices,
            this equals the number of quotient nodes (= |L/SCC|).
        lattice_size: Number of states in the original lattice.
        quotient_size: Number of SCC-quotient nodes (the effective lattice size).
        compression_ratio: |J(L)| / |L| — how much the irreducibles compress.
    """
    join_irr: frozenset[int]
    meet_irr: frozenset[int]
    dual_order: frozenset[tuple[int, int]]
    dual_states: frozenset[int]
    dual_hasse: frozenset[tuple[int, int]]
    is_distributive: bool
    downset_count: int
    lattice_size: int
    quotient_size: int
    compression_ratio: float


def birkhoff_dual(ss: StateSpace) -> BirkhoffDualResult:
    """Compute the Birkhoff dual: the poset of join-irreducibles J(L).

    For a distributive lattice L, Birkhoff's representation theorem says
    L is isomorphic to the lattice of downsets of J(L).  This function
    computes J(L) with its induced order, and verifies the theorem by
    counting downsets.

    The induced order on J(L) is: j1 <= j2 iff j1 <= j2 in the original
    lattice (i.e., j2 reaches j1 in the state space).
    """
    ji = join_irreducibles(ss)
    mi = meet_irreducibles(ss)

    # Build reachability for the order on J(L)
    q = _build_quotient_poset(ss)
    # Map states to quotient nodes
    order_pairs: set[tuple[int, int]] = set()
    for j1 in ji:
        for j2 in ji:
            n1 = q.state_to_node[j1]
            n2 = q.state_to_node[j2]
            # j2 >= j1 means j2 reaches j1 (j1 in fwd_reach[j2])
            if j1 in {q.rep[n] for n in q.fwd_reach[n2]}:
                order_pairs.add((j1, j2))  # j1 <= j2

    # Compute Hasse edges of the dual poset
    dual_hasse_set: set[tuple[int, int]] = set()
    for j1, j2 in order_pairs:
        if j1 == j2:
            continue
        # j2 covers j1 if no j3 with j1 < j3 < j2
        is_cover = True
        for j3 in ji:
            if j3 == j1 or j3 == j2:
                continue
            if (j1, j3) in order_pairs and (j3, j2) in order_pairs and j1 != j3 and j3 != j2:
                is_cover = False
                break
        if is_cover:
            dual_hasse_set.add((j1, j2))  # j2 covers j1

    # Check distributivity
    dist_result = check_distributive(ss)
    is_dist = dist_result.is_distributive

    # Count downsets of J(L)
    downsets = _count_downsets(ji, order_pairs)

    lattice_size = len(ss.states)
    quotient_size = len(q.nodes)
    ratio = len(ji) / lattice_size if lattice_size > 0 else 0.0

    return BirkhoffDualResult(
        join_irr=frozenset(ji),
        meet_irr=frozenset(mi),
        dual_order=frozenset(order_pairs),
        dual_states=frozenset(ji),
        dual_hasse=frozenset(dual_hasse_set),
        is_distributive=is_dist,
        downset_count=downsets,
        lattice_size=lattice_size,
        quotient_size=quotient_size,
        compression_ratio=ratio,
    )


def _count_downsets(elements: set[int], order: set[tuple[int, int]]) -> int:
    """Count the number of downsets (order ideals) of a finite poset.

    A downset D is a subset such that if x in D and y <= x then y in D.
    Uses brute force enumeration (fine for small posets).
    """
    elems = sorted(elements)
    n = len(elems)
    if n == 0:
        return 1  # empty downset

    count = 0
    # Enumerate all 2^n subsets
    for mask in range(1 << n):
        subset = {elems[i] for i in range(n) if mask & (1 << i)}
        is_downset = True
        for x in subset:
            for y_val in elems:
                if y_val not in subset:
                    continue
                # y_val is in subset; check that anything <= y_val is also in subset
            # More efficient: for each x in subset, check all y <= x are in subset
            for y_val, y_upper in order:
                if y_upper == x and y_val not in subset:
                    # y_val <= x but y_val not in subset
                    is_downset = False
                    break
            if not is_downset:
                break
        if is_downset:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Analysis helper: irreducible summary for a state space
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IrreduciblesResult:
    """Summary of irreducible analysis for a session type lattice.

    Attributes:
        is_lattice: Whether the state space forms a lattice.
        join_irr: Set of join-irreducible states.
        meet_irr: Set of meet-irreducible states.
        num_join_irr: Count of join-irreducibles.
        num_meet_irr: Count of meet-irreducibles.
        lattice_size: Total number of states.
        is_self_dual_cardinality: |J(L)| == |M(L)|.
        join_density: |J(L)| / |L|.
        meet_density: |M(L)| / |L|.
        birkhoff: Birkhoff dual result (None if not a lattice).
    """
    is_lattice: bool
    join_irr: frozenset[int]
    meet_irr: frozenset[int]
    num_join_irr: int
    num_meet_irr: int
    lattice_size: int
    is_self_dual_cardinality: bool
    join_density: float
    meet_density: float
    birkhoff: BirkhoffDualResult | None


def analyze_irreducibles(ss: StateSpace) -> IrreduciblesResult:
    """Compute a full irreducibles analysis for *ss*.

    Returns join-irreducibles, meet-irreducibles, densities, and the
    Birkhoff dual if the state space forms a lattice.
    """
    lr = check_lattice(ss)
    if not lr.is_lattice:
        return IrreduciblesResult(
            is_lattice=False,
            join_irr=frozenset(),
            meet_irr=frozenset(),
            num_join_irr=0,
            num_meet_irr=0,
            lattice_size=len(ss.states),
            is_self_dual_cardinality=True,
            join_density=0.0,
            meet_density=0.0,
            birkhoff=None,
        )

    ji = join_irreducibles(ss)
    mi = meet_irreducibles(ss)
    n = len(ss.states)

    bd = birkhoff_dual(ss)

    return IrreduciblesResult(
        is_lattice=True,
        join_irr=frozenset(ji),
        meet_irr=frozenset(mi),
        num_join_irr=len(ji),
        num_meet_irr=len(mi),
        lattice_size=n,
        is_self_dual_cardinality=len(ji) == len(mi),
        join_density=len(ji) / n if n > 0 else 0.0,
        meet_density=len(mi) / n if n > 0 else 0.0,
        birkhoff=bd,
    )
