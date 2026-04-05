"""Subdirect decomposition of session type lattices.

Step 363d: Implements Birkhoff's Subdirect Representation Theorem for
session type state spaces. Decomposes a protocol lattice into its
minimal irreducible factors, revealing hidden protocol structure.

Key concepts:
- Subdirect product: sublattice of L₁ × ··· × Lₖ with surjective projections
- Subdirectly irreducible (SI): Con(L) has a unique atom
- Birkhoff's theorem: every lattice ≅ subdirect product of SI lattices
- Factors = quotients L/θ for completely meet-irreducible congruences θ

Operational interpretation: decomposing L(S) into SI factors identifies
the minimal independent protocol concerns. Each factor is a protocol
aspect that cannot be further decomposed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from reticulate.statespace import StateSpace
from reticulate.lattice import check_lattice, LatticeResult
from reticulate.congruence import (
    Congruence,
    CongruenceLattice,
    congruence_lattice,
    quotient_lattice,
    is_simple,
    enumerate_congruences,
)


@dataclass(frozen=True)
class SubdirectFactor:
    """A single factor in the subdirect decomposition."""
    congruence: Congruence
    quotient: StateSpace
    quotient_size: int
    is_simple_factor: bool


@dataclass(frozen=True)
class SubdirectAnalysis:
    """Complete subdirect decomposition analysis."""
    is_subdirectly_irreducible: bool
    is_simple: bool
    num_factors: int
    factors: tuple[SubdirectFactor, ...]
    is_direct_product: bool
    con_lattice_size: int
    num_meet_irreducibles: int
    original_size: int


def _atoms_of_con(con_lat: CongruenceLattice) -> list[int]:
    """Find atom indices (minimal non-bottom elements) of Con(L).

    Uses the ordering relation from CongruenceLattice directly.
    Returns indices into con_lat.congruences.
    """
    n = len(con_lat.congruences)
    if n <= 1:
        return []

    bot = con_lat.bottom
    # Elements above bottom
    above_bot = [i for i in range(n) if i != bot and (bot, i) in con_lat.ordering]
    # Atoms: minimal elements among those above bottom
    atoms = []
    for i in above_bot:
        is_atom = True
        for j in above_bot:
            if j != i and (bot, j) in con_lat.ordering and (j, i) in con_lat.ordering:
                is_atom = False
                break
        if is_atom:
            atoms.append(i)
    return atoms


def _meet_irreducibles_of_con(con_lat: CongruenceLattice) -> list[int]:
    """Find completely meet-irreducible congruence indices in Con(L).

    A congruence θ is completely meet-irreducible iff it has exactly one
    upper cover in Con(L).

    Returns indices into con_lat.congruences.
    """
    n = len(con_lat.congruences)
    if n <= 1:
        return []

    meet_irr = []
    for i in range(n):
        # Elements strictly above i
        above_i = [j for j in range(n)
                   if j != i and (i, j) in con_lat.ordering]
        if not above_i:
            continue  # i is top, skip
        # Upper covers: minimal elements among those above i
        covers = []
        for j in above_i:
            is_cover = True
            for k in above_i:
                if k != j and (i, k) in con_lat.ordering and (k, j) in con_lat.ordering:
                    is_cover = False
                    break
            if is_cover:
                covers.append(j)
        # Meet-irreducible: exactly one upper cover
        if len(covers) == 1:
            meet_irr.append(i)
    return meet_irr


def is_subdirectly_irreducible(ss: StateSpace,
                                lr: LatticeResult | None = None) -> bool:
    """Check if the lattice is subdirectly irreducible.

    A lattice is SI iff Con(L) has a unique atom (smallest non-trivial
    congruence). Simple lattices (|Con(L)| = 2) are always SI.
    """
    if lr is None:
        lr = check_lattice(ss)
    if not lr.is_lattice:
        return False

    # Simple lattices are SI
    if is_simple(ss):
        return True

    con_lat = congruence_lattice(ss)
    atom_indices = _atoms_of_con(con_lat)
    return len(atom_indices) == 1


def subdirect_factors(ss: StateSpace,
                      lr: LatticeResult | None = None) -> list[SubdirectFactor]:
    """Compute the subdirect decomposition factors.

    Returns the list of quotients L/θ for each completely meet-irreducible
    congruence θ in Con(L). By Birkhoff's theorem, L embeds as a subdirect
    product of these factors.

    Each factor is subdirectly irreducible.
    """
    if lr is None:
        lr = check_lattice(ss)
    if not lr.is_lattice:
        return []

    con_lat = congruence_lattice(ss)
    mi_indices = _meet_irreducibles_of_con(con_lat)

    if not mi_indices:
        # Trivial case: L itself is the only factor
        dummy_cong = con_lat.congruences[con_lat.bottom] if con_lat.congruences else Congruence((), 0)
        return [SubdirectFactor(
            congruence=dummy_cong,
            quotient=ss,
            quotient_size=len(ss.states),
            is_simple_factor=True,
        )]

    factors = []
    for idx in mi_indices:
        theta = con_lat.congruences[idx]
        # Skip the trivial bottom congruence
        if theta.is_trivial_bottom:
            continue
        q = quotient_lattice(ss, theta)
        factors.append(SubdirectFactor(
            congruence=theta,
            quotient=q,
            quotient_size=len(q.states),
            is_simple_factor=is_simple(q),
        ))

    # If no non-trivial meet-irreducibles, L is SI
    if not factors:
        dummy_cong = con_lat.congruences[con_lat.bottom] if con_lat.congruences else Congruence((), 0)
        return [SubdirectFactor(
            congruence=dummy_cong,
            quotient=ss,
            quotient_size=len(ss.states),
            is_simple_factor=is_simple(ss),
        )]

    return factors


def _check_direct_product(ss: StateSpace,
                          factors: list[SubdirectFactor]) -> bool:
    """Check if the subdirect decomposition is a direct product.

    The decomposition is a direct product iff
    |L| = |L/θ₁| × |L/θ₂| × ··· × |L/θₖ|

    This is a necessary (not sufficient) condition, but for finite lattices
    with the Birkhoff decomposition, size equality implies directness.
    """
    if len(factors) <= 1:
        return True
    product_size = 1
    for f in factors:
        product_size *= f.quotient_size
    # Use SCC quotient size for comparison
    return product_size == len(ss.states)


def analyze_subdirect(ss: StateSpace,
                      lr: LatticeResult | None = None) -> SubdirectAnalysis:
    """Perform complete subdirect decomposition analysis.

    Returns the full analysis including SI status, factors,
    and whether the decomposition is a direct product.
    """
    if lr is None:
        lr = check_lattice(ss)
    if not lr.is_lattice:
        return SubdirectAnalysis(
            is_subdirectly_irreducible=False,
            is_simple=False,
            num_factors=0,
            factors=(),
            is_direct_product=False,
            con_lattice_size=0,
            num_meet_irreducibles=0,
            original_size=len(ss.states),
        )

    simple = is_simple(ss)
    si = is_subdirectly_irreducible(ss, lr)
    con_lat = congruence_lattice(ss)
    mi_indices = _meet_irreducibles_of_con(con_lat)
    facts = subdirect_factors(ss, lr)
    is_direct = _check_direct_product(ss, facts)

    return SubdirectAnalysis(
        is_subdirectly_irreducible=si,
        is_simple=simple,
        num_factors=len(facts),
        factors=tuple(facts),
        is_direct_product=is_direct,
        con_lattice_size=len(con_lat.congruences),
        num_meet_irreducibles=len(mi_indices),
        original_size=len(ss.states),
    )
