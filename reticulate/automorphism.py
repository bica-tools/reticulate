"""Automorphism group analysis for session type lattices.

Step 363e: Computes the automorphism group Aut(L) of a session type
lattice — the group of all order-preserving bijections L → L that
preserve meet and join.

Operational interpretation: automorphisms represent protocol symmetries —
permutations of states that preserve all structural relationships.
A large automorphism group means the protocol has redundant structure
that could be factored out.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import permutations
from typing import Sequence

from reticulate.statespace import StateSpace
from reticulate.lattice import check_lattice, LatticeResult


@dataclass(frozen=True)
class Automorphism:
    """A single lattice automorphism as a state permutation."""
    mapping: dict[int, int]
    fixed_points: tuple[int, ...]
    is_identity: bool
    order: int  # smallest k > 0 such that φ^k = id

    def apply(self, state: int) -> int:
        """Apply this automorphism to a state. O(1) via dict lookup."""
        return self.mapping.get(state, state)

    def compose(self, other: 'Automorphism') -> dict[int, int]:
        """Compute self ∘ other (apply other first, then self)."""
        return {s: self.apply(other.apply(s)) for s in self.mapping}


@dataclass(frozen=True)
class AutomorphismGroup:
    """The automorphism group Aut(L) of a lattice."""
    automorphisms: tuple[Automorphism, ...]
    order: int  # |Aut(L)|
    is_trivial: bool  # only the identity
    is_complete: bool  # True if full group was computed (False for n>10)
    generators: tuple[Automorphism, ...]
    num_fixed_by_all: int
    orbit_partition: tuple[tuple[int, ...], ...]


def _build_reach(ss: StateSpace, lr: LatticeResult) -> tuple[list[int], int, int, dict[int, set[int]]]:
    """Build reachability from SCC quotient. Returns (nodes, top, bottom, reach)."""
    scc_map = lr.scc_map or {}
    reps = sorted(set(scc_map.values())) if scc_map else sorted(ss.states)
    if not reps:
        reps = sorted(ss.states)

    top_rep = scc_map.get(ss.top, ss.top)
    bottom_rep = scc_map.get(ss.bottom, ss.bottom)

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

    return reps, top_rep, bottom_rep, reach


def _precompute_meet_table(reach: dict[int, set[int]],
                           nodes: list[int]) -> dict[tuple[int, int], int | None]:
    """Precompute full meet table for O(1) lookups."""
    table: dict[tuple[int, int], int | None] = {}
    for a in nodes:
        for b in nodes:
            if a == b:
                table[(a, b)] = a
                continue
            lower = [n for n in nodes if n in reach[a] and n in reach[b]]
            meet = None
            for c in lower:
                if all(lb in reach[c] or lb == c for lb in lower):
                    meet = c
                    break
            table[(a, b)] = meet
    return table


def _is_automorphism(perm: dict[int, int], nodes: list[int],
                     reach: dict[int, set[int]],
                     meet_table: dict[tuple[int, int], int | None]) -> bool:
    """Check if a permutation is a lattice automorphism."""
    # Order preservation
    for a in nodes:
        for b in nodes:
            if b in reach[a]:  # a >= b
                if perm[b] not in reach[perm[a]]:
                    return False

    # Meet preservation (using precomputed table)
    for a in nodes:
        for b in nodes:
            m = meet_table.get((a, b))
            pm = meet_table.get((perm[a], perm[b]))
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
        current = {n: perm[current[n]] for n in nodes}
    return len(nodes)


def find_automorphisms(ss: StateSpace,
                       lr: LatticeResult | None = None) -> tuple[list[Automorphism], bool]:
    """Find all automorphisms of the lattice.

    Returns (automorphisms, is_complete). is_complete is False when
    the lattice is too large for brute-force and only the identity is returned.
    """
    if lr is None:
        lr = check_lattice(ss)
    if not lr.is_lattice:
        return [], True

    nodes, top_rep, bottom_rep, reach = _build_reach(ss, lr)
    n = len(nodes)

    if n == 0:
        return [], True

    auts: list[Automorphism] = []

    if n <= 10:
        meet_table = _precompute_meet_table(reach, nodes)
        # Only permute non-extreme nodes (top and bottom must be fixed)
        inner = [s for s in nodes if s != top_rep and s != bottom_rep]
        fixed_prefix = {top_rep: top_rep, bottom_rep: bottom_rep}

        for perm in permutations(inner):
            perm_dict = dict(fixed_prefix)
            perm_dict.update(zip(inner, perm))
            if _is_automorphism(perm_dict, nodes, reach, meet_table):
                fixed = tuple(s for s in nodes if perm_dict[s] == s)
                is_id = all(perm_dict[s] == s for s in nodes)
                order = 1 if is_id else _compute_order(perm_dict, nodes)
                auts.append(Automorphism(
                    mapping=perm_dict,
                    fixed_points=fixed,
                    is_identity=is_id,
                    order=order,
                ))
        return auts, True
    else:
        warnings.warn(
            f"Automorphism enumeration skipped for lattice with {n} states (>10). "
            f"Only identity returned. Result is incomplete.",
            stacklevel=2,
        )
        auts.append(Automorphism(
            mapping={s: s for s in nodes},
            fixed_points=tuple(nodes),
            is_identity=True,
            order=1,
        ))
        return auts, False


def _compute_orbits(auts: list[Automorphism], nodes: list[int]) -> list[tuple[int, ...]]:
    """Compute orbits via BFS closure under all automorphisms."""
    visited: set[int] = set()
    orbits: list[tuple[int, ...]] = []
    for s in nodes:
        if s in visited:
            continue
        # BFS closure: apply all automorphisms to all discovered members
        orbit: set[int] = {s}
        queue = [s]
        while queue:
            current = queue.pop(0)
            for aut in auts:
                img = aut.apply(current)
                if img not in orbit:
                    orbit.add(img)
                    queue.append(img)
        orbits.append(tuple(sorted(orbit)))
        visited.update(orbit)
    return orbits


def _find_generators(auts: list[Automorphism]) -> list[Automorphism]:
    """Find a generating set for the automorphism group.

    Greedy: pick non-identity automorphisms that generate new elements
    via composition closure.
    """
    if len(auts) <= 1:
        return []

    non_id = [a for a in auts if not a.is_identity]
    if not non_id:
        return []

    all_mappings = {tuple(sorted(a.mapping.items())) for a in auts}
    generators: list[Automorphism] = []
    generated: set[tuple] = set()
    # Add identity
    id_aut = next(a for a in auts if a.is_identity)
    generated.add(tuple(sorted(id_aut.mapping.items())))

    for a in non_id:
        a_key = tuple(sorted(a.mapping.items()))
        if a_key not in generated:
            generators.append(a)
            # Close under composition with all known elements
            new_elements = {a_key}
            while new_elements:
                next_new: set[tuple] = set()
                for new_key in new_elements:
                    new_map = dict(new_key)
                    for existing_key in list(generated):
                        existing_map = dict(existing_key)
                        # Compose both ways
                        comp1 = {s: new_map.get(existing_map.get(s, s), s)
                                 for s in id_aut.mapping}
                        comp2 = {s: existing_map.get(new_map.get(s, s), s)
                                 for s in id_aut.mapping}
                        k1 = tuple(sorted(comp1.items()))
                        k2 = tuple(sorted(comp2.items()))
                        if k1 not in generated:
                            next_new.add(k1)
                        if k2 not in generated:
                            next_new.add(k2)
                generated.update(new_elements)
                new_elements = next_new
            # If we've generated everything, stop
            if generated == all_mappings:
                break

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
            is_complete=True,
            generators=(),
            num_fixed_by_all=0,
            orbit_partition=(),
        )

    auts, is_complete = find_automorphisms(ss, lr)
    nodes, _top, _bot, _reach = _build_reach(ss, lr)

    orbits = _compute_orbits(auts, nodes)
    generators = _find_generators(auts)

    fixed_by_all = [s for s in nodes
                    if all(a.apply(s) == s for a in auts)]

    return AutomorphismGroup(
        automorphisms=tuple(auts),
        order=len(auts),
        is_trivial=len(auts) <= 1,
        is_complete=is_complete,
        generators=tuple(generators),
        num_fixed_by_all=len(fixed_by_all),
        orbit_partition=tuple(orbits),
    )
