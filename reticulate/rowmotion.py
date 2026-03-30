"""Rowmotion and toggling for session type lattices (Step 30ad).

Rowmotion is a fundamental dynamical operator on finite posets that acts
on the set of order ideals (downsets).  Given a finite poset P:

- An **order ideal** (downset) I is a subset I ⊆ P such that if x ∈ I
  and y ≤ x then y ∈ I.
- An **antichain** A is a subset where no two elements are comparable.
- There is a natural bijection between order ideals and antichains:
  an order ideal maps to its set of maximal elements (an antichain),
  and an antichain maps to its downward closure (an order ideal).

**Rowmotion** on order ideals is defined as:

    Row(I) = ↓(min(P \\ I))

That is, take the complement P \\ I, find its minimal elements, and
return their downward closure.  Equivalently, rowmotion can be decomposed
as a composition of *toggles* applied in any linear extension order
from bottom to top.

**Toggle** at element x acts on order ideal I as follows:
- If x ∉ I and I ∪ {x} is an order ideal, add x.
- If x ∈ I and I \\ {x} is an order ideal, remove x.
- Otherwise, leave I unchanged.

**Key results for session type lattices:**

1. On any finite poset, rowmotion has finite order (it permutes the
   finite set of order ideals).
2. The **Brouwer–Schrijver theorem**: on a product of chains [a] × [b],
   rowmotion has order a + b.  This applies directly to parallel
   composition of session types, since L(S₁ ∥ S₂) = L(S₁) × L(S₂).
3. **Homomesy** (Propp–Roby): a statistic f is homomesic if its average
   over every rowmotion orbit is the same constant.  The cardinality
   |I| is often homomesic on distributive lattices.

This module works on the SCC quotient of the state space to handle
cycles from recursion.  The poset ordering is s₁ ≥ s₂ iff s₂ is
reachable from s₁ (top = initial state, bottom = end state).

All computations use exact integer arithmetic where possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace

from reticulate.zeta import (
    _adjacency,
    _compute_sccs,
    _reachability,
    _state_list,
    compute_rank,
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RowmotionResult:
    """Complete rowmotion analysis for a session type state space.

    Attributes:
        order_ideals: All order ideals of the poset.
        antichains: All antichains of the poset.
        rowmotion_order: Order of rowmotion permutation (LCM of orbit sizes).
        orbit_sizes: Size of each distinct orbit.
        num_orbits: Number of distinct orbits.
        is_homomesic_cardinality: Whether |I| is homomesic across orbits.
        homomesy_average: Average of |I| over orbits (None if not homomesic).
        toggle_sequence: Linear extension used for toggle decomposition.
    """

    order_ideals: list[frozenset[int]]
    antichains: list[frozenset[int]]
    rowmotion_order: int
    orbit_sizes: list[int]
    num_orbits: int
    is_homomesic_cardinality: bool
    homomesy_average: float | None
    toggle_sequence: list[int]


# ---------------------------------------------------------------------------
# Internal helpers: build quotient poset
# ---------------------------------------------------------------------------

def _quotient_poset(ss: "StateSpace") -> tuple[
    list[int],                   # reps: sorted list of SCC representatives
    dict[int, int],              # scc_map: state -> rep
    dict[int, set[int]],         # scc_members: rep -> member states
    dict[int, set[int]],         # below: rep -> set of reps strictly below
    dict[int, set[int]],         # above: rep -> set of reps strictly above
]:
    """Build the SCC quotient poset and compute transitive closure.

    Returns representative list, SCC map/members, and for each rep the set
    of reps strictly below it (reachable) and strictly above it.
    """
    scc_map, scc_members = _compute_sccs(ss)
    reps = sorted(scc_members.keys())
    adj = _adjacency(ss)

    # Quotient adjacency
    q_adj: dict[int, set[int]] = {r: set() for r in reps}
    for s in ss.states:
        for t in adj[s]:
            sr, tr = scc_map[s], scc_map[t]
            if sr != tr:
                q_adj[sr].add(tr)

    # Transitive closure on quotient (below = reachable = lower in poset)
    below: dict[int, set[int]] = {}
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
        # strictly below: reachable minus self
        below[r] = visited - {r}

    # above: inverse of below
    above: dict[int, set[int]] = {r: set() for r in reps}
    for r in reps:
        for b in below[r]:
            above[b].add(r)

    return reps, scc_map, scc_members, below, above


def _quotient_covers(
    reps: list[int],
    below: dict[int, set[int]],
) -> dict[int, set[int]]:
    """Compute covering relation on quotient: covers[x] = set of y that x covers.

    x covers y iff x > y and there is no z with x > z > y.
    """
    covers: dict[int, set[int]] = {r: set() for r in reps}
    for x in reps:
        for y in below[x]:
            # Check if x covers y: no intermediate z
            is_cover = True
            for z in below[x]:
                if z != y and y in below[z]:
                    is_cover = False
                    break
            if is_cover:
                covers[x].add(y)
    return covers


# ---------------------------------------------------------------------------
# Core poset operations
# ---------------------------------------------------------------------------

def compute_order_relation(ss: "StateSpace") -> dict[int, set[int]]:
    """Build order relation: state -> set of states below it (downward closure).

    The ordering is s1 >= s2 iff s2 is reachable from s1.  This function
    returns, for each SCC representative, the set of reps at or below it
    in the quotient poset.

    Works on the SCC quotient to handle recursive cycles.
    """
    reps, scc_map, scc_members, below, above = _quotient_poset(ss)
    # Include self (reflexive)
    return {r: below[r] | {r} for r in reps}


def compute_covers(ss: "StateSpace") -> dict[int, set[int]]:
    """Build covering relation: state -> set of states it directly covers.

    x covers y means x > y and there is no z with x > z > y.
    Works on SCC quotient representatives.
    """
    reps, scc_map, scc_members, below, above = _quotient_poset(ss)
    return _quotient_covers(reps, below)


def order_ideals(ss: "StateSpace") -> list[frozenset[int]]:
    """Enumerate all order ideals (downsets) of the poset.

    An order ideal I satisfies: if x in I and y <= x then y in I.
    Since bottom <= everything, every order ideal contains bottom.
    Equivalently, I is a downward-closed subset.

    Works on the SCC quotient.  Enumerates by building up from the
    empty set, adding elements whose predecessors are all in the ideal.
    Uses BFS to enumerate all valid downsets without duplicates.
    """
    reps, scc_map, scc_members, below, above = _quotient_poset(ss)

    # BFS enumeration: start with empty, expand by adding feasible elements
    seen: set[frozenset[int]] = set()
    queue: list[frozenset[int]] = [frozenset()]
    seen.add(frozenset())
    ideals: list[frozenset[int]] = []

    while queue:
        current = queue.pop(0)
        ideals.append(current)
        for r in reps:
            if r in current:
                continue
            # r can be added if all elements strictly below r are in current
            if below[r].issubset(current):
                new_ideal = current | {r}
                if new_ideal not in seen:
                    seen.add(new_ideal)
                    queue.append(new_ideal)

    return ideals


def antichains(ss: "StateSpace") -> list[frozenset[int]]:
    """Enumerate all antichains of the poset.

    An antichain is a set of elements where no two are comparable.
    Uses the bijection: antichains <-> order ideals (maximal elements of ideal).
    """
    ideals = order_ideals(ss)
    result: list[frozenset[int]] = []
    for ideal in ideals:
        ac = ideal_to_antichain_from_sets(ideal, _get_below(ss))
        result.append(ac)
    return result


def _get_below(ss: "StateSpace") -> dict[int, set[int]]:
    """Helper: get the 'below' relation on quotient reps."""
    _, _, _, below, _ = _quotient_poset(ss)
    return below


def antichain_to_ideal(ss: "StateSpace", ac: frozenset[int]) -> frozenset[int]:
    """Convert antichain to order ideal (its downward closure).

    The order ideal generated by antichain A is the union of
    downsets of all elements in A: Union_{a in A} {x : x <= a}.
    """
    reps, scc_map, scc_members, below, above = _quotient_poset(ss)
    ideal: set[int] = set()
    for a in ac:
        ideal.add(a)
        ideal.update(below[a])
    return frozenset(ideal)


def ideal_to_antichain(ss: "StateSpace", ideal: frozenset[int]) -> frozenset[int]:
    """Convert order ideal to antichain (its maximal elements).

    The antichain of an order ideal I is the set of maximal elements:
    {x in I : there is no y in I with y > x}.
    """
    below = _get_below(ss)
    return ideal_to_antichain_from_sets(ideal, below)


def ideal_to_antichain_from_sets(
    ideal: frozenset[int],
    below: dict[int, set[int]],
) -> frozenset[int]:
    """Convert order ideal to antichain given the below relation.

    Maximal elements: x in I such that no y in I has x in below[y] (x < y).
    """
    if not ideal:
        return frozenset()
    maximal: set[int] = set()
    for x in ideal:
        is_maximal = True
        for y in ideal:
            if y != x and x in below[y]:
                # y > x, so x is not maximal
                is_maximal = False
                break
        if is_maximal:
            maximal.add(x)
    return frozenset(maximal)


# ---------------------------------------------------------------------------
# Toggle operations
# ---------------------------------------------------------------------------

def toggle(ss: "StateSpace", ideal: frozenset[int], x: int) -> frozenset[int]:
    """Toggle element x in/out of order ideal.

    - If x not in I and I union {x} is an order ideal: add x.
    - If x in I and I \\ {x} is an order ideal: remove x.
    - Otherwise: leave I unchanged.

    I union {x} is an order ideal iff all elements below x are in I.
    I \\ {x} is an order ideal iff no element in I has x as its only
    "support" — i.e., no y in I \\ {x} requires x to be in I.
    Equivalently, I \\ {x} is a downset iff x is maximal in I or
    no element of I is directly above x while not being above something
    else in I.  More precisely: I \\ {x} is a downset iff for every
    y in I with x in below[y], there exists z in I \\ {x} with
    z != x and x in below[z] ... but simpler: I \\ {x} is a downset
    iff no element y in I has x as a predecessor AND y not reachable
    from any other element of I \\ {x}.

    Actually the simplest check: I \\ {x} is a downset iff for all
    y in I, if x is strictly below y (x in below[y]), then the
    predecessors of y are still all in I \\ {x}. Since I is already a
    downset, the only issue is if y covers x and y is in I.

    Simplest correct implementation: I \\ {x} is a downset iff there is
    no y in I such that x is in below[y] and x not in below[z] for
    any z in (I \\ {x}) with z in below[y].

    Even simpler: I \\ {x} is a downset iff for every y in I with
    y != x, all elements below y are in I \\ {x}.  Since I is already
    a downset, the only elements that could fail are those y where
    x in below[y].  For such y, we need all of below[y] to be in
    I \\ {x}, which is true as long as x is not the only element
    in below[y] ∩ I that makes the downset property hold.

    Most correct simple implementation: just check the definition.
    """
    reps, scc_map, scc_members, below, above = _quotient_poset(ss)

    if x not in ideal:
        # Try to add: I ∪ {x} is a downset iff below[x] ⊆ I
        if below[x].issubset(ideal):
            return ideal | {x}
        return ideal
    else:
        # Try to remove: I \ {x} is a downset iff no y in I depends on x
        # I \ {x} is a downset iff for every y in (I \ {x}), below[y] ⊆ (I \ {x})
        candidate = ideal - {x}
        for y in candidate:
            if not below[y].issubset(candidate):
                # y requires x (or something not in candidate) — can't remove
                return ideal
        return candidate


def toggle_sequence(ss: "StateSpace") -> list[int]:
    """Return a linear extension from top to bottom for toggle decomposition.

    Rowmotion decomposes as the composition of toggles applied from top
    to bottom: Row = tau_{x_1} o ... o tau_{x_n} where x_1 < x_2 < ...
    in the poset.  Since composition applies right-to-left, the topmost
    toggle acts first, then we proceed downward to the bottom.

    Returns elements ordered from highest rank (top) to lowest rank (bottom).
    """
    reps, scc_map, scc_members, below, above = _quotient_poset(ss)
    rank = compute_rank(ss)

    # Map reps to their rank (using any member)
    rep_rank: dict[int, int] = {}
    for r in reps:
        rep_rank[r] = rank.get(r, 0)

    # Sort by rank descending (top first, bottom last)
    return sorted(reps, key=lambda r: -rep_rank[r])


# ---------------------------------------------------------------------------
# Rowmotion
# ---------------------------------------------------------------------------

def rowmotion_ideal(ss: "StateSpace", ideal: frozenset[int]) -> frozenset[int]:
    """Apply rowmotion to an order ideal.

    Row(I) = downset of minimal elements of P \\ I.

    The complement P \\ I has some minimal elements (elements with no
    element below them in the complement).  Take the downward closure
    of those minimal elements.
    """
    reps, scc_map, scc_members, below, above = _quotient_poset(ss)
    reps_set = frozenset(reps)

    complement = reps_set - ideal

    if not complement:
        # I = P, complement is empty, min of empty = empty, downset = empty
        return frozenset()

    # Find minimal elements of complement: x in complement such that
    # no y in complement has y < x (i.e., y in below[x] and y != x)
    minimal: set[int] = set()
    for x in complement:
        is_min = True
        for y in complement:
            if y != x and y in below[x]:
                is_min = False
                break
        if is_min:
            minimal.add(x)

    # Downward closure of minimal elements
    result: set[int] = set()
    for m in minimal:
        result.add(m)
        result.update(below[m])
    return frozenset(result)


def rowmotion_antichain(ss: "StateSpace", ac: frozenset[int]) -> frozenset[int]:
    """Apply rowmotion to an antichain.

    Maps antichain A to min(P \\ downset(A)).
    """
    ideal = antichain_to_ideal(ss, ac)
    new_ideal = rowmotion_ideal(ss, ideal)
    return ideal_to_antichain(ss, new_ideal)


def rowmotion_via_toggles(ss: "StateSpace", ideal: frozenset[int]) -> frozenset[int]:
    """Apply rowmotion as composition of toggles in linear extension order.

    Rowmotion = toggle_top ∘ ... ∘ toggle_bottom, applying toggles
    from bottom to top in a linear extension.  This is equivalent to
    the direct definition Row(I) = downset(min(P \\ I)).
    """
    seq = toggle_sequence(ss)
    current = ideal
    for x in seq:
        current = toggle(ss, current, x)
    return current


# ---------------------------------------------------------------------------
# Orbit analysis
# ---------------------------------------------------------------------------

def rowmotion_orbit(ss: "StateSpace", ideal: frozenset[int]) -> list[frozenset[int]]:
    """Compute the orbit of an order ideal under rowmotion.

    Returns the list [I, Row(I), Row^2(I), ...] until we return to I.
    """
    orbit: list[frozenset[int]] = [ideal]
    current = rowmotion_ideal(ss, ideal)
    while current != ideal:
        orbit.append(current)
        current = rowmotion_ideal(ss, current)
    return orbit


def all_orbits(ss: "StateSpace") -> list[list[frozenset[int]]]:
    """Partition all order ideals into rowmotion orbits.

    Returns a list of orbits, each orbit being a list of order ideals.
    """
    ideals = order_ideals(ss)
    seen: set[frozenset[int]] = set()
    orbits: list[list[frozenset[int]]] = []

    for ideal in ideals:
        if ideal not in seen:
            orbit = rowmotion_orbit(ss, ideal)
            for i in orbit:
                seen.add(i)
            orbits.append(orbit)

    return orbits


def rowmotion_order(ss: "StateSpace") -> int:
    """Compute the order of rowmotion (LCM of all orbit sizes).

    The order is the smallest k such that Row^k = identity on all
    order ideals.
    """
    orbits = all_orbits(ss)
    if not orbits:
        return 1
    order = 1
    for orbit in orbits:
        size = len(orbit)
        order = order * size // gcd(order, size)
    return order


# ---------------------------------------------------------------------------
# Homomesy
# ---------------------------------------------------------------------------

def check_homomesy(
    ss: "StateSpace",
    orbits: list[list[frozenset[int]]],
) -> tuple[bool, float | None]:
    """Check if cardinality statistic |I| is homomesic across orbits.

    A statistic f is homomesic under rowmotion if the average of f over
    every orbit is the same constant.

    Returns (is_homomesic, average) where average is None if not homomesic.
    """
    if not orbits:
        return True, None

    averages: list[float] = []
    for orbit in orbits:
        total = sum(len(ideal) for ideal in orbit)
        avg = total / len(orbit)
        averages.append(avg)

    # Check if all averages are the same (within floating point tolerance)
    if not averages:
        return True, None

    first = averages[0]
    # Use exact rational comparison: total_i / size_i = total_j / size_j
    # iff total_i * size_j = total_j * size_i
    is_homo = True
    for orbit, avg in zip(orbits, averages):
        # Cross multiply to avoid float issues
        t0 = sum(len(ideal) for ideal in orbits[0])
        s0 = len(orbits[0])
        ti = sum(len(ideal) for ideal in orbit)
        si = len(orbit)
        if t0 * si != ti * s0:
            is_homo = False
            break

    if is_homo:
        return True, first
    return False, None


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze_rowmotion(ss: "StateSpace") -> RowmotionResult:
    """Full rowmotion analysis: enumerate ideals, compute orbits, check homomesy.

    Builds the SCC quotient, enumerates all order ideals and antichains,
    computes rowmotion orbits and their sizes, checks cardinality homomesy,
    and returns a complete RowmotionResult.
    """
    ideals = order_ideals(ss)
    acs = antichains(ss)
    seq = toggle_sequence(ss)
    orbits = all_orbits(ss)

    orbit_sizes = sorted([len(o) for o in orbits])
    num_orbits = len(orbits)

    # Rowmotion order = LCM of orbit sizes
    order = 1
    for size in orbit_sizes:
        order = order * size // gcd(order, size)

    is_homo, avg = check_homomesy(ss, orbits)

    return RowmotionResult(
        order_ideals=ideals,
        antichains=acs,
        rowmotion_order=order,
        orbit_sizes=orbit_sizes,
        num_orbits=num_orbits,
        is_homomesic_cardinality=is_homo,
        homomesy_average=avg,
        toggle_sequence=seq,
    )
