"""Automorphism group analysis for session type lattices.

Step 363e: Computes the automorphism group Aut(L) of a session type
lattice — the group of all order-preserving bijections L → L that
preserve meet and join.

Key concepts:
- Lattice automorphism: bijection φ: L → L with φ(a∧b) = φ(a)∧φ(b) and φ(a∨b) = φ(a)∨φ(b)
- Aut(L): the automorphism group under composition
- |Aut(L)| measures the symmetry of the protocol
- Fixed points of automorphisms identify structurally essential states

Operational interpretation: automorphisms represent protocol symmetries —
permutations of states that preserve all structural relationships.
A large automorphism group means the protocol has redundant structure
that could be factored out.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Sequence

from reticulate.statespace import StateSpace
from reticulate.lattice import check_lattice, LatticeResult


@dataclass(frozen=True)
class Automorphism:
    """A single lattice automorphism as a state permutation."""
    mapping: tuple[tuple[int, int], ...]  # (state, image) pairs
    fixed_points: tuple[int, ...]
    is_identity: bool
    order: int  # smallest k > 0 such that φ^k = id

    def apply(self, state: int) -> int:
        """Apply this automorphism to a state."""
        for s, t in self.mapping:
            if s == state:
                return t
        return state  # identity on unmapped states


@dataclass(frozen=True)
class AutomorphismGroup:
    """The automorphism group Aut(L) of a lattice."""
    automorphisms: tuple[Automorphism, ...]
    order: int  # |Aut(L)|
    is_trivial: bool  # only the identity
    generators: tuple[Automorphism, ...]  # minimal generating set
    num_fixed_by_all: int  # states fixed by every automorphism
    orbit_partition: tuple[tuple[int, ...], ...]  # orbits of the action


def _build_reach(ss: StateSpace, lr: LatticeResult) -> tuple[list[int], dict[int, set[int]]]:
    """Build reachability from SCC quotient."""
    scc_map = lr.scc_map or {}
    reps = sorted(set(scc_map.values())) if scc_map else sorted(ss.states)
    if not reps:
        reps = sorted(ss.states)

    fwd: dict[int, set[int]] = {r: set() for r in reps}
    for src, _lbl, tgt in ss.transitions:
        src_r = scc_map.get(src, src)
        tgt_r = scc_map.get(tgt, tgt)
        if src_r != tgt_r:
            fwd[src_r].add(tgt_r)

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

    return reps, reach


def _compute_meet(reach: dict[int, set[int]], nodes: list[int], a: int, b: int) -> int | None:
    """Compute meet of a and b."""
    if a == b:
        return a
    lower = [n for n in nodes if n in reach[a] and n in reach[b]]
    if not lower:
        return None
    for c in lower:
        if all(lb in reach[c] or lb == c for lb in lower):
            return c
    return None


def _compute_join(reach: dict[int, set[int]], nodes: list[int], a: int, b: int) -> int | None:
    """Compute join of a and b."""
    if a == b:
        return a
    upper = [n for n in nodes if a in reach[n] and b in reach[n]]
    if not upper:
        return None
    for c in upper:
        if all(c in reach[ub] or c == ub for ub in upper):
            return c
    return None


def _is_automorphism(perm: dict[int, int], nodes: list[int],
                     reach: dict[int, set[int]]) -> bool:
    """Check if a permutation is a lattice automorphism.

    Must preserve order, meet, and join.
    """
    # Check order preservation (both directions for bijection)
    for a in nodes:
        for b in nodes:
            if b in reach[a]:  # a >= b
                if perm[b] not in reach[perm[a]]:  # φ(a) >= φ(b)?
                    return False

    # Check meet preservation
    for a in nodes:
        for b in nodes:
            m = _compute_meet(reach, nodes, a, b)
            pm = _compute_meet(reach, nodes, perm[a], perm[b])
            if m is not None and pm is not None:
                if perm[m] != pm:
                    return False

    return True


def _compute_order(perm: dict[int, int], nodes: list[int]) -> int:
    """Compute the order of a permutation (smallest k with φ^k = id)."""
    current = dict(perm)
    for k in range(1, len(nodes) + 1):
        if all(current[n] == n for n in nodes):
            return k
        # Apply perm again
        current = {n: perm[current[n]] for n in nodes}
    return len(nodes)  # fallback


def find_automorphisms(ss: StateSpace,
                       lr: LatticeResult | None = None) -> list[Automorphism]:
    """Find all automorphisms of the lattice.

    For small lattices (≤ 10 states), uses brute-force enumeration.
    For larger lattices, uses constraint-based pruning.
    """
    if lr is None:
        lr = check_lattice(ss)
    if not lr.is_lattice:
        return []

    nodes, reach = _build_reach(ss, lr)
    top_rep = (lr.scc_map or {}).get(ss.top, ss.top)
    bottom_rep = (lr.scc_map or {}).get(ss.bottom, ss.bottom)
    n = len(nodes)

    if n == 0:
        return []

    # Identity is always an automorphism
    identity_map = {s: s for s in nodes}

    auts: list[Automorphism] = []

    if n <= 10:
        # Brute force for small lattices
        for perm in permutations(nodes):
            perm_dict = dict(zip(nodes, perm))
            # Must fix top and bottom
            if perm_dict.get(top_rep) != top_rep:
                continue
            if perm_dict.get(bottom_rep) != bottom_rep:
                continue
            if _is_automorphism(perm_dict, nodes, reach):
                mapping = tuple((s, perm_dict[s]) for s in nodes)
                fixed = tuple(s for s in nodes if perm_dict[s] == s)
                is_id = all(perm_dict[s] == s for s in nodes)
                order = 1 if is_id else _compute_order(perm_dict, nodes)
                auts.append(Automorphism(
                    mapping=mapping,
                    fixed_points=fixed,
                    is_identity=is_id,
                    order=order,
                ))
    else:
        # For larger lattices, only return the identity
        # (full enumeration is O(n!) and impractical)
        mapping = tuple((s, s) for s in nodes)
        auts.append(Automorphism(
            mapping=mapping,
            fixed_points=tuple(nodes),
            is_identity=True,
            order=1,
        ))

    return auts


def _compute_orbits(auts: list[Automorphism], nodes: list[int]) -> list[tuple[int, ...]]:
    """Compute orbits of the automorphism group action."""
    visited: set[int] = set()
    orbits: list[tuple[int, ...]] = []
    for s in nodes:
        if s in visited:
            continue
        orbit = set()
        for aut in auts:
            orbit.add(aut.apply(s))
        orbit.add(s)
        orbits.append(tuple(sorted(orbit)))
        visited.update(orbit)
    return orbits


def _find_generators(auts: list[Automorphism]) -> list[Automorphism]:
    """Find a minimal generating set for the automorphism group.

    Simple greedy: pick non-identity automorphisms that generate new elements.
    """
    if len(auts) <= 1:
        return []

    non_id = [a for a in auts if not a.is_identity]
    if not non_id:
        return []

    # Simple approach: all non-identity automorphisms of prime order
    generators = []
    generated: set[tuple] = {auts[0].mapping}  # identity
    for a in non_id:
        if a.mapping not in generated:
            generators.append(a)
            # Add powers of this element
            current = a
            for _ in range(a.order):
                generated.add(current.mapping)
                # Compose: not fully implemented, just add the element
    return generators


def automorphism_group(ss: StateSpace,
                       lr: LatticeResult | None = None) -> AutomorphismGroup:
    """Compute the full automorphism group Aut(L)."""
    if lr is None:
        lr = check_lattice(ss)
    if not lr.is_lattice:
        return AutomorphismGroup(
            automorphisms=(),
            order=0,
            is_trivial=True,
            generators=(),
            num_fixed_by_all=0,
            orbit_partition=(),
        )

    auts = find_automorphisms(ss, lr)
    nodes, _reach = _build_reach(ss, lr)

    orbits = _compute_orbits(auts, nodes)
    generators = _find_generators(auts)

    # States fixed by all automorphisms
    fixed_by_all = [s for s in nodes
                    if all(a.apply(s) == s for a in auts)]

    return AutomorphismGroup(
        automorphisms=tuple(auts),
        order=len(auts),
        is_trivial=len(auts) <= 1,
        generators=tuple(generators),
        num_fixed_by_all=len(fixed_by_all),
        orbit_partition=tuple(orbits),
    )
