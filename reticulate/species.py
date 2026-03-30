"""Combinatorial species analysis for session type lattices (Step 32h).

Combinatorial species (Joyal, 1981) provide a categorical framework for
counting structures. For session type lattices, we model each lattice as
a species F where:

- F[n] = isomorphism classes of session type lattices on n elements
- The type-generating function t_F(x) = sum_n |F[n]/~| x^n / n!
- The cycle index Z_F encodes symmetry structure

Key operations:
- **Species product**: F * G corresponds to parallel composition (||)
- **Species sum**: F + G corresponds to external choice (&)
- **Isomorphism types**: equivalence classes under order-preserving bijection

This module computes species-theoretic invariants for session type lattices:
type-generating series, cycle index, isomorphism type counting, and
species operations.

All computations handle cycles via SCC quotient.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace

from reticulate.zeta import (
    _state_list,
    _reachability,
    _compute_sccs,
    compute_rank,
    zeta_matrix,
)
from reticulate.morphism import find_isomorphism


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpeciesResult:
    """Complete species analysis of a session type state space.

    Attributes:
        num_states: Number of states (quotient).
        num_automorphisms: Size of the automorphism group |Aut(P)|.
        automorphism_generators: Generating permutations for Aut(P).
        type_generating_coefficients: Coefficients of the type-generating
            series [a_0, a_1, ..., a_max_n] where a_n = isomorphism types
            of sub-posets on n elements (normalized by 1/n!).
        isomorphism_type_count: Number of distinct isomorphism types
            of sub-lattices.
        cycle_index_terms: Terms of the cycle index as list of
            (coefficient, partition) pairs.
        exponential_formula_terms: Coefficients from the exponential formula
            relating connected and general species.
        symmetry_factor: |Aut(P)| / |P|! — how symmetric the lattice is.
    """
    num_states: int
    num_automorphisms: int
    automorphism_generators: list[dict[int, int]]
    type_generating_coefficients: list[float]
    isomorphism_type_count: int
    cycle_index_terms: list[tuple[float, list[int]]]
    exponential_formula_terms: list[float]
    symmetry_factor: float


# ---------------------------------------------------------------------------
# Automorphism group
# ---------------------------------------------------------------------------

def _build_adjacency_from_reach(
    states: list[int], reach: dict[int, set[int]]
) -> dict[int, set[int]]:
    """Build strict adjacency from reachability."""
    adj: dict[int, set[int]] = {s: set() for s in states}
    for s in states:
        for t in states:
            if s != t and t in reach[s]:
                adj[s].add(t)
    return adj


def find_automorphisms(ss: "StateSpace") -> list[dict[int, int]]:
    """Find all automorphisms of the poset (order-preserving bijections to itself).

    An automorphism is a permutation sigma: P -> P such that
    x <= y iff sigma(x) <= sigma(y).

    For small posets, uses backtracking search.
    """
    states = _state_list(ss)
    n = len(states)

    if n == 0:
        return [{}]
    if n == 1:
        return [{states[0]: states[0]}]

    reach = _reachability(ss)
    rank = compute_rank(ss)

    # Group states by rank (automorphisms must preserve rank)
    rank_groups: dict[int, list[int]] = {}
    for s in states:
        r = rank.get(s, 0)
        if r not in rank_groups:
            rank_groups[r] = []
        rank_groups[r].append(s)

    # States by rank for assignment
    ranks_sorted = sorted(rank_groups.keys(), reverse=True)
    ordered_states = []
    for r in ranks_sorted:
        ordered_states.extend(sorted(rank_groups[r]))

    # Out-degree and in-degree as invariants
    out_deg: dict[int, int] = {s: 0 for s in states}
    in_deg: dict[int, int] = {s: 0 for s in states}
    for src, _, tgt in ss.transitions:
        if src in out_deg:
            out_deg[src] += 1
        if tgt in in_deg:
            in_deg[tgt] += 1

    automorphisms: list[dict[int, int]] = []

    def _backtrack(idx: int, mapping: dict[int, int], used: set[int]) -> None:
        if idx == n:
            automorphisms.append(dict(mapping))
            return

        s = ordered_states[idx]
        r = rank.get(s, 0)
        candidates = [c for c in rank_groups[r] if c not in used]

        for c in candidates:
            # Pruning: check degree invariants
            if out_deg[s] != out_deg[c] or in_deg[s] != in_deg[c]:
                continue

            # Check order consistency with already-mapped elements
            ok = True
            for mapped_s, mapped_c in mapping.items():
                # s >= mapped_s iff c >= mapped_c
                s_above = mapped_s in reach[s]
                c_above = mapped_c in reach[c]
                if s_above != c_above:
                    ok = False
                    break
                # mapped_s >= s iff mapped_c >= c
                s_below = s in reach[mapped_s]
                c_below = c in reach[mapped_c]
                if s_below != c_below:
                    ok = False
                    break

            if ok:
                mapping[s] = c
                used.add(c)
                _backtrack(idx + 1, mapping, used)
                del mapping[s]
                used.discard(c)

        # Limit search for large posets
        if len(automorphisms) > 1000:
            return

    _backtrack(0, {}, set())
    return automorphisms if automorphisms else [{s: s for s in states}]


def automorphism_group_size(ss: "StateSpace") -> int:
    """Size of the automorphism group |Aut(P)|."""
    return len(find_automorphisms(ss))


# ---------------------------------------------------------------------------
# Isomorphism types (sub-posets)
# ---------------------------------------------------------------------------

def isomorphism_types(ss: "StateSpace") -> list[frozenset[int]]:
    """Find distinct isomorphism types of non-empty sub-posets.

    Two subsets S1, S2 of P are isomorphic if there is an order-preserving
    bijection between the induced sub-posets.

    For efficiency, we classify by size and simple invariants.
    Returns representative element sets for each isomorphism class.
    """
    states = _state_list(ss)
    n = len(states)

    if n == 0:
        return []

    reach = _reachability(ss)

    # For each subset size, find isomorphism classes
    # Only do small sizes to avoid exponential blowup
    max_size = min(n, 6)
    types: list[frozenset[int]] = []
    seen_signatures: set[tuple] = set()

    def _sub_signature(subset: frozenset[int]) -> tuple:
        """Compute an isomorphism invariant for a sub-poset."""
        elems = sorted(subset)
        k = len(elems)
        # Build the order matrix within the subset
        idx = {s: i for i, s in enumerate(elems)}
        matrix_flat = []
        for i in range(k):
            for j in range(k):
                if elems[j] in reach[elems[i]]:
                    matrix_flat.append(1)
                else:
                    matrix_flat.append(0)
        # Invariant: sorted row sums + sorted column sums
        row_sums = tuple(sorted(
            sum(matrix_flat[i * k + j] for j in range(k)) for i in range(k)
        ))
        col_sums = tuple(sorted(
            sum(matrix_flat[i * k + j] for i in range(k)) for j in range(k)
        ))
        return (k, row_sums, col_sums)

    # Size 1: always one type
    types.append(frozenset([states[0]]))
    seen_signatures.add((1, (1,), (1,)))

    # Size 2 to max_size
    from itertools import combinations

    for size in range(2, max_size + 1):
        for combo in combinations(states, size):
            subset = frozenset(combo)
            sig = _sub_signature(subset)
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                types.append(subset)

    return types


# ---------------------------------------------------------------------------
# Type-generating series
# ---------------------------------------------------------------------------

def type_generating_series(ss: "StateSpace", max_n: int = 0) -> list[float]:
    """Compute the type-generating series coefficients.

    The type-generating series is:
        t_F(x) = sum_{n>=0} f_n * x^n

    where f_n counts the number of isomorphism types of sub-posets on n elements.

    For a single lattice P on N states, f_n = number of non-isomorphic
    induced sub-posets of size n.
    """
    states = _state_list(ss)
    n = len(states)

    if max_n <= 0:
        max_n = n

    max_n = min(max_n, n, 7)  # Cap for performance

    reach = _reachability(ss)
    from itertools import combinations

    coefficients = [0.0] * (max_n + 1)
    coefficients[0] = 1.0  # empty sub-poset (by convention)

    for size in range(1, max_n + 1):
        seen: set[tuple] = set()
        for combo in combinations(states, size):
            subset = sorted(combo)
            k = len(subset)
            idx = {s: i for i, s in enumerate(subset)}
            # Canonical form: sorted adjacency matrix entries
            matrix = tuple(
                1 if subset[j] in reach[subset[i]] else 0
                for i in range(k) for j in range(k)
            )
            # Simple canonical: sort row sums + adjacency pattern
            row_sums = tuple(sorted(
                sum(1 for j in range(k) if subset[j] in reach[subset[i]])
                for i in range(k)
            ))
            col_sums = tuple(sorted(
                sum(1 for i in range(k) if subset[j] in reach[subset[i]])
                for j in range(k)
            ))
            sig = (k, row_sums, col_sums)
            seen.add(sig)
        coefficients[size] = float(len(seen))

    return coefficients


# ---------------------------------------------------------------------------
# Cycle index
# ---------------------------------------------------------------------------

def _partition_from_permutation(perm: dict[int, int]) -> list[int]:
    """Extract cycle type (partition) from a permutation.

    Returns sorted list of cycle lengths in descending order.
    """
    visited: set[int] = set()
    cycles: list[int] = []

    for start in sorted(perm.keys()):
        if start in visited:
            continue
        length = 0
        current = start
        while current not in visited:
            visited.add(current)
            current = perm[current]
            length += 1
        cycles.append(length)

    return sorted(cycles, reverse=True)


def cycle_index(ss: "StateSpace") -> list[tuple[float, list[int]]]:
    """Compute the cycle index of the automorphism group.

    The cycle index is:
        Z(Aut(P)) = (1/|Aut|) * sum_{sigma in Aut} p_{lambda(sigma)}

    where lambda(sigma) is the cycle type of sigma, and
    p_lambda = p_{l1} * p_{l2} * ... (power sum symmetric functions).

    Returns list of (coefficient, partition) pairs.
    """
    auts = find_automorphisms(ss)
    n_aut = len(auts)

    if n_aut == 0:
        return [(1.0, [1])]

    # Count cycle types
    type_counts: dict[tuple[int, ...], int] = {}
    for perm in auts:
        partition = tuple(_partition_from_permutation(perm))
        type_counts[partition] = type_counts.get(partition, 0) + 1

    terms: list[tuple[float, list[int]]] = []
    for partition, count in sorted(type_counts.items()):
        coeff = count / n_aut
        terms.append((coeff, list(partition)))

    return terms


# ---------------------------------------------------------------------------
# Species operations
# ---------------------------------------------------------------------------

def species_product(ss1: "StateSpace", ss2: "StateSpace") -> dict[str, int]:
    """Compute species product invariants F * G.

    The species product corresponds to disjoint union of structures,
    which for session types is the parallel composition (||).

    Returns key invariants of the product species.
    """
    n1 = len(ss1.states)
    n2 = len(ss2.states)

    aut1 = automorphism_group_size(ss1)
    aut2 = automorphism_group_size(ss2)

    return {
        "left_states": n1,
        "right_states": n2,
        "product_states": n1 * n2,
        "left_automorphisms": aut1,
        "right_automorphisms": aut2,
        "product_automorphisms_upper": aut1 * aut2,
        "left_isomorphism_types": len(isomorphism_types(ss1)),
        "right_isomorphism_types": len(isomorphism_types(ss2)),
    }


def species_sum(ss1: "StateSpace", ss2: "StateSpace") -> dict[str, int]:
    """Compute species sum invariants F + G.

    The species sum corresponds to disjoint union (external choice &).
    """
    n1 = len(ss1.states)
    n2 = len(ss2.states)

    return {
        "left_states": n1,
        "right_states": n2,
        "sum_states": n1 + n2,
        "left_isomorphism_types": len(isomorphism_types(ss1)),
        "right_isomorphism_types": len(isomorphism_types(ss2)),
    }


# ---------------------------------------------------------------------------
# Exponential formula
# ---------------------------------------------------------------------------

def exponential_formula_terms(ss: "StateSpace", max_n: int = 0) -> list[float]:
    """Compute terms from the exponential formula.

    If T(x) = sum a_n x^n/n! is the exponential generating function for
    labeled structures, and C(x) = sum c_n x^n/n! counts connected ones,
    then T(x) = exp(C(x)).

    For a lattice (always connected), C = log(T) at the species level.
    Returns the first few terms of log of the type-generating series.
    """
    tgs = type_generating_series(ss, max_n)
    n = len(tgs)

    if n <= 1:
        return tgs

    # Compute log of the series (formal power series log)
    # If tgs = [1, a1, a2, ...], then log(tgs) = [0, c1, c2, ...]
    # c_n = a_n - (1/n) * sum_{k=1}^{n-1} k * c_k * a_{n-k}
    log_coeffs = [0.0] * n
    if n > 0 and abs(tgs[0] - 1.0) > 1e-12:
        return log_coeffs  # log only defined for tgs[0] = 1

    for i in range(1, n):
        s = tgs[i]
        for k in range(1, i):
            s -= k * log_coeffs[k] * tgs[i - k] / i
        log_coeffs[i] = s

    return log_coeffs


# ---------------------------------------------------------------------------
# Symmetry factor
# ---------------------------------------------------------------------------

def symmetry_factor(ss: "StateSpace") -> float:
    """Compute |Aut(P)| / |P|! — the symmetry factor.

    A higher symmetry factor indicates more symmetry in the lattice.
    Total order: Aut = {id}, factor = 1/n!.
    Antichain: Aut = S_n, factor = 1.
    """
    n = len(ss.states)
    if n == 0:
        return 1.0
    aut_size = automorphism_group_size(ss)
    return aut_size / math.factorial(n)


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze_species(ss: "StateSpace") -> SpeciesResult:
    """Complete combinatorial species analysis."""
    auts = find_automorphisms(ss)
    n_aut = len(auts)
    states = _state_list(ss)
    n = len(states)

    tgs = type_generating_series(ss)
    iso_types = isomorphism_types(ss)
    ci = cycle_index(ss)
    exp_terms = exponential_formula_terms(ss)
    sf = symmetry_factor(ss)

    # Extract generators (just store first few non-identity)
    identity = {s: s for s in states}
    generators = [a for a in auts if a != identity][:5]

    return SpeciesResult(
        num_states=n,
        num_automorphisms=n_aut,
        automorphism_generators=generators,
        type_generating_coefficients=tgs,
        isomorphism_type_count=len(iso_types),
        cycle_index_terms=ci,
        exponential_formula_terms=exp_terms,
        symmetry_factor=sf,
    )
