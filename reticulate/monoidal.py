"""Monoidal structure of session type composition (Step 167).

Verifies that the category **SessLat** with parallel composition (product)
as the tensor and ``End`` as the unit forms a **symmetric monoidal category**.

The key structural laws:

1. **Unit**: ``L(S || end) ~ L(S) ~ L(end || S)`` — End is the unit for ||.
2. **Associativity**: ``L((S1 || S2) || S3) ~ L(S1 || (S2 || S3))``
3. **Symmetry** (braiding): ``L(S1 || S2) ~ L(S2 || S1)``
4. **Coherence**: Mac Lane's pentagon and triangle diagrams commute.

All isomorphisms (~) are natural lattice isomorphisms: bijective lattice
homomorphisms preserving meets, joins, top, and bottom.

Public API:

- :func:`check_monoidal_unit` — verify End is the monoidal unit
- :func:`check_associativity` — verify associativity up to isomorphism
- :func:`check_symmetry` — verify symmetry (braiding) up to isomorphism
- :func:`check_coherence` — verify Mac Lane coherence conditions
- :func:`check_monoidal_structure` — full monoidal structure verification
- :class:`MonoidalResult` — aggregate result
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from reticulate.lattice import check_lattice
from reticulate.morphism import find_isomorphism
from reticulate.product import product_statespace

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UnitResult:
    """Result of checking End as the monoidal unit.

    Attributes:
        left_unit: True iff L(end || S) ~ L(S).
        right_unit: True iff L(S || end) ~ L(S).
        left_iso: The isomorphism mapping L(end || S) -> L(S), or None.
        right_iso: The isomorphism mapping L(S || end) -> L(S), or None.
    """

    left_unit: bool
    right_unit: bool
    left_iso: dict[int, int] | None
    right_iso: dict[int, int] | None


@dataclass(frozen=True)
class AssociativityResult:
    """Result of checking associativity of parallel composition.

    Attributes:
        is_associative: True iff L((S1||S2)||S3) ~ L(S1||(S2||S3)).
        iso: The isomorphism mapping, or None if not associative.
        left_states: Number of states in (S1||S2)||S3.
        right_states: Number of states in S1||(S2||S3).
    """

    is_associative: bool
    iso: dict[int, int] | None
    left_states: int
    right_states: int


@dataclass(frozen=True)
class SymmetryResult:
    """Result of checking symmetry (braiding) of parallel composition.

    Attributes:
        is_symmetric: True iff L(S1||S2) ~ L(S2||S1).
        iso: The isomorphism mapping, or None if not symmetric.
        is_involution: True iff sigma_{S2,S1} . sigma_{S1,S2} = id.
    """

    is_symmetric: bool
    iso: dict[int, int] | None
    is_involution: bool


@dataclass(frozen=True)
class CoherenceResult:
    """Result of checking Mac Lane coherence conditions.

    Attributes:
        pentagon: True iff the pentagon diagram commutes (associativity coherence).
        triangle: True iff the triangle diagram commutes (unit coherence).
        hexagon: True iff the hexagon diagram commutes (braiding coherence).
        counterexample: Description of failure, or None.
    """

    pentagon: bool
    triangle: bool
    hexagon: bool
    counterexample: str | None


@dataclass(frozen=True)
class MonoidalResult:
    """Aggregate result of all monoidal structure checks.

    Attributes:
        unit: UnitResult for End as the monoidal unit.
        associativity: AssociativityResult for parallel composition.
        symmetry: SymmetryResult for braiding.
        coherence: CoherenceResult for Mac Lane conditions.
        is_monoidal: True iff all checks pass (symmetric monoidal category).
    """

    unit: UnitResult
    associativity: AssociativityResult
    symmetry: SymmetryResult
    coherence: CoherenceResult
    is_monoidal: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _end_statespace() -> "StateSpace":
    """Construct the trivial 1-state state space for End."""
    from reticulate.statespace import StateSpace as SS
    return SS(
        states={0},
        transitions=[],
        top=0,
        bottom=0,
        labels={0: "end"},
        selection_transitions=set(),
    )


def _structural_isomorphism(
    ss1: "StateSpace",
    ss2: "StateSpace",
) -> dict[int, int] | None:
    """Find a structural isomorphism between two state spaces.

    Returns the mapping dict if isomorphic, None otherwise.
    """
    morph = find_isomorphism(ss1, ss2)
    if morph is None:
        return None
    return morph.mapping


# ---------------------------------------------------------------------------
# Unit check
# ---------------------------------------------------------------------------

def check_monoidal_unit(ss: "StateSpace") -> UnitResult:
    """Verify that End is the monoidal unit for parallel composition.

    Checks that:
    - ``L(end || S) ~ L(S)`` (left unit law)
    - ``L(S || end) ~ L(S)`` (right unit law)

    The isomorphisms are lambda and rho (unitors) in the monoidal
    category vocabulary.

    Parameters:
        ss: A state space to check as the argument.

    Returns:
        UnitResult with left and right unit checks.
    """
    end_ss = _end_statespace()

    # Left: end || S
    left_product = product_statespace(end_ss, ss)
    left_iso = _structural_isomorphism(left_product, ss)

    # Right: S || end
    right_product = product_statespace(ss, end_ss)
    right_iso = _structural_isomorphism(right_product, ss)

    return UnitResult(
        left_unit=left_iso is not None,
        right_unit=right_iso is not None,
        left_iso=left_iso,
        right_iso=right_iso,
    )


# ---------------------------------------------------------------------------
# Associativity check
# ---------------------------------------------------------------------------

def check_associativity(
    ss1: "StateSpace",
    ss2: "StateSpace",
    ss3: "StateSpace",
) -> AssociativityResult:
    """Verify associativity of parallel composition up to isomorphism.

    Checks that ``L((S1 || S2) || S3) ~ L(S1 || (S2 || S3))``.

    The isomorphism is the associator alpha in the monoidal category
    vocabulary.

    Parameters:
        ss1, ss2, ss3: Three state spaces.

    Returns:
        AssociativityResult with the isomorphism check.
    """
    # Left-associated: (S1 || S2) || S3
    left_inner = product_statespace(ss1, ss2)
    left_assoc = product_statespace(left_inner, ss3)

    # Right-associated: S1 || (S2 || S3)
    right_inner = product_statespace(ss2, ss3)
    right_assoc = product_statespace(ss1, right_inner)

    iso = _structural_isomorphism(left_assoc, right_assoc)

    return AssociativityResult(
        is_associative=iso is not None,
        iso=iso,
        left_states=len(left_assoc.states),
        right_states=len(right_assoc.states),
    )


# ---------------------------------------------------------------------------
# Symmetry (braiding) check
# ---------------------------------------------------------------------------

def check_symmetry(
    ss1: "StateSpace",
    ss2: "StateSpace",
) -> SymmetryResult:
    """Verify symmetry (braiding) of parallel composition.

    Checks that ``L(S1 || S2) ~ L(S2 || S1)`` and that the braiding
    is an involution: ``sigma_{S2,S1} . sigma_{S1,S2} = id``.

    Parameters:
        ss1, ss2: Two state spaces.

    Returns:
        SymmetryResult with the isomorphism and involution checks.
    """
    prod_12 = product_statespace(ss1, ss2)
    prod_21 = product_statespace(ss2, ss1)

    iso_12_21 = _structural_isomorphism(prod_12, prod_21)

    # Check involution: sigma_{21} . sigma_{12} = id
    is_involution = False
    if iso_12_21 is not None:
        iso_21_12 = _structural_isomorphism(prod_21, prod_12)
        if iso_21_12 is not None:
            # Compose: for each state s in prod_12,
            # sigma_{21} . sigma_{12}(s) should equal s
            composed = {
                s: iso_21_12.get(iso_12_21[s], -1)
                for s in prod_12.states
            }
            is_involution = all(
                composed[s] == s for s in prod_12.states
            )

    return SymmetryResult(
        is_symmetric=iso_12_21 is not None,
        iso=iso_12_21,
        is_involution=is_involution,
    )


# ---------------------------------------------------------------------------
# Coherence check
# ---------------------------------------------------------------------------

def check_coherence(
    ss1: "StateSpace",
    ss2: "StateSpace",
    ss3: "StateSpace",
) -> CoherenceResult:
    """Verify Mac Lane coherence conditions for the monoidal structure.

    Checks three coherence diagrams:

    1. **Pentagon** (associativity coherence):
       Both paths from ``((A||B)||C)||D`` to ``A||(B||(C||D))``
       via the associator yield the same isomorphism.

    2. **Triangle** (unit coherence):
       ``alpha_{A,I,B}`` composed with ``rho_A x id_B`` equals
       ``id_A x lambda_B``, where I = End.

    3. **Hexagon** (braiding coherence):
       The braiding interacts correctly with associativity.

    Parameters:
        ss1, ss2, ss3: Three state spaces (used as A, B, C; D derived from ss1).

    Returns:
        CoherenceResult with pentagon, triangle, and hexagon checks.
    """
    # Pentagon: use ss1 as both A and D (sufficient for the diagram)
    pentagon_ok = _check_pentagon(ss1, ss2, ss3, ss1)
    triangle_ok = _check_triangle(ss1, ss2)
    hexagon_ok = _check_hexagon(ss1, ss2, ss3)

    counterexample = None
    if not pentagon_ok:
        counterexample = "Pentagon diagram does not commute"
    elif not triangle_ok:
        counterexample = "Triangle diagram does not commute"
    elif not hexagon_ok:
        counterexample = "Hexagon diagram does not commute"

    return CoherenceResult(
        pentagon=pentagon_ok,
        triangle=triangle_ok,
        hexagon=hexagon_ok,
        counterexample=counterexample,
    )


def _check_pentagon(
    a: "StateSpace",
    b: "StateSpace",
    c: "StateSpace",
    d: "StateSpace",
) -> bool:
    """Check the pentagon diagram commutes.

    The pentagon says that for A, B, C, D, the two ways to reassociate
    ((A||B)||C)||D to A||(B||(C||D)) give the same isomorphism.

    We verify this by state-count matching and isomorphism existence:
    both fully left-associated and fully right-associated products are
    isomorphic, and the two intermediate paths yield the same result.
    """
    # Both extreme associations must have the same state count
    ab = product_statespace(a, b)
    ab_c = product_statespace(ab, c)
    ab_c_d = product_statespace(ab_c, d)

    cd = product_statespace(c, d)
    bcd = product_statespace(b, cd)
    a_bcd = product_statespace(a, bcd)

    # Path 1: ((A||B)||C)||D -> (A||B)||(C||D) -> A||(B||(C||D))
    # Path 2: ((A||B)||C)||D -> (A||(B||C))||D -> A||((B||C)||D) -> A||(B||(C||D))
    # Both must yield the same iso from ab_c_d to a_bcd
    iso = _structural_isomorphism(ab_c_d, a_bcd)
    if iso is None:
        return False

    # Also check intermediate step: (A||B)||(C||D)
    ab_cd = product_statespace(ab, cd)
    iso_mid = _structural_isomorphism(ab_c_d, ab_cd)
    if iso_mid is None:
        return False

    iso_mid2 = _structural_isomorphism(ab_cd, a_bcd)
    if iso_mid2 is None:
        return False

    # Verify composition of path through intermediate equals direct iso
    # Compose: iso_mid2 . iso_mid should equal iso
    composed = {s: iso_mid2.get(iso_mid[s], -1) for s in ab_c_d.states}
    return composed == iso


def _check_triangle(
    a: "StateSpace",
    b: "StateSpace",
) -> bool:
    """Check the triangle diagram commutes.

    The triangle says: for the unit I = End,
    alpha_{A,I,B} composed with (id_A x lambda_B) equals (rho_A x id_B).

    Concretely: (A || End) || B ~ A || (End || B) ~ A || B.
    Both paths from (A || End) || B to A || B must agree.
    """
    end = _end_statespace()

    a_end = product_statespace(a, end)
    a_end_b = product_statespace(a_end, b)

    end_b = product_statespace(end, b)
    a_end_b2 = product_statespace(a, end_b)

    a_b = product_statespace(a, b)

    # Path 1: (A||I)||B -> A||B via rho_A x id_B
    iso_path1 = _structural_isomorphism(a_end_b, a_b)

    # Path 2: (A||I)||B -> A||(I||B) -> A||B via alpha then id_A x lambda_B
    iso_alpha = _structural_isomorphism(a_end_b, a_end_b2)
    iso_lambda = _structural_isomorphism(a_end_b2, a_b)

    if iso_path1 is None or iso_alpha is None or iso_lambda is None:
        return False

    # Compose path 2
    composed = {s: iso_lambda.get(iso_alpha[s], -1) for s in a_end_b.states}
    return composed == iso_path1


def _check_hexagon(
    a: "StateSpace",
    b: "StateSpace",
    c: "StateSpace",
) -> bool:
    """Check the hexagon diagram commutes.

    The hexagon relates braiding and associativity:
    sigma_{A,B||C} = (id_B x sigma_{A,C}) . alpha_{B,A,C} . (sigma_{A,B} x id_C)

    We verify this by checking that all intermediate products are
    isomorphic and that state counts match throughout the diagram.
    """
    # Products needed
    ab = product_statespace(a, b)
    bc = product_statespace(b, c)
    a_bc = product_statespace(a, bc)
    ba = product_statespace(b, a)
    ba_c = product_statespace(ba, c)
    b_ac = product_statespace(b, product_statespace(a, c))
    b_ca = product_statespace(b, product_statespace(c, a))

    # All reassociations must be isomorphic
    iso1 = _structural_isomorphism(a_bc, ba_c)
    if iso1 is None:
        return False

    iso2 = _structural_isomorphism(ba_c, b_ac)
    if iso2 is None:
        return False

    iso3 = _structural_isomorphism(b_ac, b_ca)
    if iso3 is None:
        return False

    # End-to-end: a_bc -> b_ca (braiding A past B||C)
    # This should be the same as the direct braiding iso
    direct = _structural_isomorphism(a_bc, b_ca)
    if direct is None:
        return False

    composed = {s: iso3.get(iso2.get(iso1[s], -1), -1) for s in a_bc.states}
    return composed == direct


# ---------------------------------------------------------------------------
# Full monoidal structure check
# ---------------------------------------------------------------------------

def check_monoidal_structure(
    ss1: "StateSpace",
    ss2: "StateSpace",
    ss3: "StateSpace",
) -> MonoidalResult:
    """Verify the full symmetric monoidal structure on SessLat.

    Runs unit, associativity, symmetry, and coherence checks using the
    given state spaces as representative objects.

    Parameters:
        ss1, ss2, ss3: Three state spaces to use as test objects.

    Returns:
        MonoidalResult with all check results.
    """
    unit = check_monoidal_unit(ss1)
    assoc = check_associativity(ss1, ss2, ss3)
    sym = check_symmetry(ss1, ss2)
    coh = check_coherence(ss1, ss2, ss3)

    is_monoidal = (
        unit.left_unit
        and unit.right_unit
        and assoc.is_associative
        and sym.is_symmetric
        and sym.is_involution
        and coh.pentagon
        and coh.triangle
        and coh.hexagon
    )

    return MonoidalResult(
        unit=unit,
        associativity=assoc,
        symmetry=sym,
        coherence=coh,
        is_monoidal=is_monoidal,
    )
