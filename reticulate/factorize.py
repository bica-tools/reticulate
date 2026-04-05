"""Lattice factorization theory for session type state spaces (Step 363f).

Detects whether a session type lattice L(S) is directly decomposable into
L1 x L2, meaning the protocol has hidden parallelism. Uses the factor
congruence characterisation: L is directly decomposable iff Con(L) contains
a complementary pair (theta, phi) with theta ^ phi = bottom and theta v phi = top.

Key concepts:
- Factor congruence pair: (theta, phi) with theta ^ phi = identity, theta v phi = total
- Directly decomposable: L ~ L/theta x L/phi for some factor pair
- Indecomposable: no factor pair exists; the protocol is atomic
- Unique factorization: finite lattices have unique factorization into indecomposables

Functions:
  - find_factor_congruences(ss) -- find all complementary pairs in Con(L)
  - is_directly_decomposable(ss) -- True if any factor pair exists
  - factorize(ss) -- recursively decompose into indecomposable factors
  - analyze_factorization(ss) -- complete analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from reticulate.lattice import check_lattice, LatticeResult
from reticulate.congruence import (
    Congruence,
    CongruenceLattice,
    congruence_lattice,
    quotient_lattice,
    enumerate_congruences,
    _refines,
    _join_congruences,
)
from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FactorizationAnalysis:
    """Complete factorization analysis of a session type lattice.

    Attributes:
        is_decomposable: True if L can be expressed as a direct product.
        is_indecomposable: Complement of is_decomposable.
        factors: Tuple of directly indecomposable factor StateSpaces.
        factor_count: Number of indecomposable factors.
        factor_sizes: Sizes of each factor (number of states).
        factor_congruence_pairs: Indices into Con(L) for each factor pair found.
    """
    is_decomposable: bool
    is_indecomposable: bool
    factors: tuple[StateSpace, ...]
    factor_count: int
    factor_sizes: tuple[int, ...]
    factor_congruence_pairs: tuple[tuple[int, int], ...]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _meet_congruences(ss: StateSpace, c1: Congruence, c2: Congruence) -> Congruence:
    """Compute the meet (intersection) of two congruences.

    The meet of two congruences is their intersection as equivalence relations:
    x equiv y (mod c1 ^ c2) iff x equiv y (mod c1) AND x equiv y (mod c2).
    """
    # Build lookup: state -> class for each congruence
    class1: dict[int, frozenset[int]] = {}
    for cls in c1.classes:
        for s in cls:
            class1[s] = cls

    class2: dict[int, frozenset[int]] = {}
    for cls in c2.classes:
        for s in cls:
            class2[s] = cls

    # Two states are in the same class of the meet iff they share classes in both
    states_list = sorted(ss.states)
    # Use equivalence class by (class1, class2) pair
    groups: dict[tuple[frozenset[int], frozenset[int]], set[int]] = {}
    for s in states_list:
        key = (class1.get(s, frozenset({s})), class2.get(s, frozenset({s})))
        groups.setdefault(key, set()).add(s)

    classes = tuple(sorted(
        [frozenset(g) for g in groups.values()],
        key=lambda c: min(c),
    ))
    return Congruence(classes=classes, num_classes=len(classes))


def _congruences_equal(c1: Congruence, c2: Congruence) -> bool:
    """Check if two congruences have the same partition."""
    return c1.classes == c2.classes


# ---------------------------------------------------------------------------
# Public API: Factor congruences
# ---------------------------------------------------------------------------

def find_factor_congruences(
    ss: StateSpace,
    lr: LatticeResult | None = None,
) -> list[tuple[Congruence, Congruence]]:
    """Find all complementary pairs in Con(L).

    A pair (theta, phi) is complementary (factor pair) if:
        theta ^ phi = identity congruence (all singletons)
        theta v phi = total congruence (one big class)

    Returns a list of (theta, phi) pairs. Empty list if none found.
    """
    if lr is None:
        lr = check_lattice(ss)
    if not lr.is_lattice:
        return []

    con_lat = congruence_lattice(ss)
    congs = con_lat.congruences
    n = len(congs)

    if n <= 1:
        return []

    # Identify the identity (bottom) and total (top) congruences
    identity = congs[con_lat.bottom]
    total = congs[con_lat.top]

    pairs: list[tuple[Congruence, Congruence]] = []

    for i in range(n):
        for j in range(i + 1, n):
            c_i = congs[i]
            c_j = congs[j]

            # Skip trivial congruences as individual factors
            if c_i.is_trivial_bottom or c_i.is_trivial_top:
                continue
            if c_j.is_trivial_bottom or c_j.is_trivial_top:
                continue

            # Check meet = identity
            meet = _meet_congruences(ss, c_i, c_j)
            if not _congruences_equal(meet, identity):
                continue

            # Check join = total
            join = _join_congruences(ss, c_i, c_j)
            if not _congruences_equal(join, total):
                continue

            # Size check: |L| must equal |L/theta| * |L/phi| for a genuine
            # direct product decomposition. Complementary congruences in Con(L)
            # that fail this are not factor congruences (the lattice is only
            # a subdirect, not direct, product of the quotients).
            n_states = len(ss.states)
            if c_i.num_classes * c_j.num_classes != n_states:
                continue

            pairs.append((c_i, c_j))

    return pairs


def find_factor_congruences_indexed(
    ss: StateSpace,
    lr: LatticeResult | None = None,
) -> list[tuple[int, int]]:
    """Find factor congruence pairs, returning indices into Con(L).

    Same as find_factor_congruences but returns index pairs.
    """
    if lr is None:
        lr = check_lattice(ss)
    if not lr.is_lattice:
        return []

    con_lat = congruence_lattice(ss)
    congs = con_lat.congruences
    n = len(congs)

    if n <= 1:
        return []

    identity = congs[con_lat.bottom]
    total = congs[con_lat.top]

    pairs: list[tuple[int, int]] = []

    for i in range(n):
        for j in range(i + 1, n):
            c_i = congs[i]
            c_j = congs[j]

            if c_i.is_trivial_bottom or c_i.is_trivial_top:
                continue
            if c_j.is_trivial_bottom or c_j.is_trivial_top:
                continue

            meet = _meet_congruences(ss, c_i, c_j)
            if not _congruences_equal(meet, identity):
                continue

            join = _join_congruences(ss, c_i, c_j)
            if not _congruences_equal(join, total):
                continue

            # Size check
            n_states = len(ss.states)
            if c_i.num_classes * c_j.num_classes != n_states:
                continue

            pairs.append((i, j))

    return pairs


# ---------------------------------------------------------------------------
# Public API: Decomposability check
# ---------------------------------------------------------------------------

def is_directly_decomposable(
    ss: StateSpace,
    lr: LatticeResult | None = None,
) -> bool:
    """Check if the lattice L(S) is directly decomposable.

    True if any complementary congruence pair exists in Con(L).
    """
    if lr is None:
        lr = check_lattice(ss)
    if not lr.is_lattice:
        return False

    # Trivial lattices (1 or 2 states) are indecomposable
    if len(ss.states) <= 2:
        return False

    pairs = find_factor_congruences(ss, lr)
    return len(pairs) > 0


# ---------------------------------------------------------------------------
# Public API: Factorize
# ---------------------------------------------------------------------------

def factorize(
    ss: StateSpace,
    lr: LatticeResult | None = None,
) -> list[StateSpace]:
    """Recursively decompose L(S) into directly indecomposable factors.

    Uses find_factor_congruences to split, then recurses on each factor
    until all factors are indecomposable.

    Returns a list of StateSpace objects, each representing an indecomposable
    factor. If L(S) is already indecomposable, returns [ss].
    """
    if lr is None:
        lr = check_lattice(ss)
    if not lr.is_lattice:
        return [ss]

    if len(ss.states) <= 2:
        return [ss]

    pairs = find_factor_congruences(ss, lr)
    if not pairs:
        return [ss]

    # Take the first factor pair
    theta, phi = pairs[0]
    factor1 = quotient_lattice(ss, theta)
    factor2 = quotient_lattice(ss, phi)

    # Recurse on each factor
    result: list[StateSpace] = []
    result.extend(factorize(factor1))
    result.extend(factorize(factor2))

    return result


# ---------------------------------------------------------------------------
# Public API: Full analysis
# ---------------------------------------------------------------------------

def analyze_factorization(
    ss: StateSpace,
    lr: LatticeResult | None = None,
) -> FactorizationAnalysis:
    """Perform complete factorization analysis of L(S).

    Returns decomposability status, indecomposable factors, sizes, and
    the congruence pairs used for decomposition.
    """
    if lr is None:
        lr = check_lattice(ss)
    if not lr.is_lattice:
        return FactorizationAnalysis(
            is_decomposable=False,
            is_indecomposable=True,
            factors=(ss,),
            factor_count=1,
            factor_sizes=(len(ss.states),),
            factor_congruence_pairs=(),
        )

    decomposable = is_directly_decomposable(ss, lr)
    factors = factorize(ss, lr)
    indexed_pairs = find_factor_congruences_indexed(ss, lr)

    return FactorizationAnalysis(
        is_decomposable=decomposable,
        is_indecomposable=not decomposable,
        factors=tuple(factors),
        factor_count=len(factors),
        factor_sizes=tuple(len(f.states) for f in factors),
        factor_congruence_pairs=tuple(indexed_pairs),
    )
