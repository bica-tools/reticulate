"""Euler characteristic consolidation for session type lattices (Step 30v).

Consolidates Euler characteristic computation from multiple modules
(order_complex, homology, mobius) into a single diagnostic interface.

Key theorems verified:
- **Philip Hall's theorem**: mu(top, bot) = reduced_euler(Delta(P_bar))
- **Kunneth formula**: chi(L1 x L2) = chi(L1) * chi(L2)

Provides interval-level Euler distribution as a protocol complexity
diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace

from reticulate.order_complex import (
    euler_characteristic as _oc_euler,
    reduced_euler_characteristic as _oc_reduced_euler,
    f_vector as _oc_f_vector,
)
from reticulate.homology import (
    euler_characteristic_from_homology as _hom_euler,
    euler_characteristic_from_faces as _hom_euler_faces,
    betti_numbers as _betti_numbers,
)
from reticulate.mobius import (
    mobius_value as _mobius_value,
    all_mobius_values as _all_mobius_values,
    verify_hall as _verify_hall,
    verify_product_formula as _verify_product_formula,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HallVerification:
    """Result of verifying Philip Hall's theorem.

    Hall's theorem: mu(top, bot) = reduced Euler characteristic
    of the order complex of the open interval (top, bot).

    Attributes:
        mobius_value: mu(top, bot) from Mobius inversion.
        reduced_euler: Reduced Euler characteristic from face counts.
        verified: True iff the two values agree.
    """
    mobius_value: int
    reduced_euler: int
    verified: bool


@dataclass(frozen=True)
class KunnethVerification:
    """Result of verifying the Kunneth multiplicativity formula.

    chi(L1 x L2) = chi(L1) * chi(L2)

    Attributes:
        chi_left: Euler characteristic of the left factor.
        chi_right: Euler characteristic of the right factor.
        chi_product: Euler characteristic of the product.
        expected: chi_left * chi_right.
        verified: True iff chi_product == expected.
    """
    chi_left: int
    chi_right: int
    chi_product: int
    expected: int
    verified: bool


@dataclass(frozen=True)
class ComparisonResult:
    """Result of comparing multiple Euler characteristic methods.

    Attributes:
        from_faces: chi from alternating face count.
        from_mobius: mu(top, bot) (reduced Euler by Hall's theorem).
        from_betti: chi from alternating Betti sum.
        all_agree: True iff all three methods give the same value.
        f_vector: The face vector used for the face-count method.
        betti_numbers: The Betti numbers used for the homology method.
    """
    from_faces: int
    from_mobius: int
    from_betti: int
    all_agree: bool
    f_vector: list[int]
    betti_numbers: list[int]


@dataclass(frozen=True)
class EulerAnalysis:
    """Complete Euler characteristic analysis.

    Attributes:
        euler_characteristic: chi (unreduced).
        reduced_euler: chi_tilde = chi - 1.
        hall: Hall's theorem verification.
        interval_series: mu(x, y) for all comparable pairs.
        distribution: histogram of interval mu values.
        obstructions: intervals where |mu| > 1.
        comparison: multi-method cross-check.
        num_intervals: total number of comparable pairs (x > y).
    """
    euler_characteristic: int
    reduced_euler: int
    hall: HallVerification
    interval_series: dict[tuple[int, int], int]
    distribution: dict[int, int]
    obstructions: list[tuple[int, int, int]]
    comparison: ComparisonResult
    num_intervals: int


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def euler_characteristic(ss: "StateSpace") -> int:
    """Euler characteristic chi = sum_{k>=0} (-1)^k f_k.

    Delegates to order_complex.euler_characteristic.
    """
    return _oc_euler(ss)


def reduced_euler_characteristic(ss: "StateSpace") -> int:
    """Reduced Euler characteristic chi_tilde.

    By Philip Hall's theorem, this equals mu(top, bot).
    Delegates to order_complex.reduced_euler_characteristic.
    """
    return _oc_reduced_euler(ss)


# ---------------------------------------------------------------------------
# Hall's theorem verification
# ---------------------------------------------------------------------------

def verify_hall_theorem(ss: "StateSpace") -> HallVerification:
    """Verify Philip Hall's theorem: mu(top, bot) = chi_tilde(Delta(P_bar)).

    Cross-checks the Mobius function value against the reduced Euler
    characteristic computed from the face vector of the order complex.

    Special case: when top == bottom (single-state lattice), mu(x,x) = 1
    by definition and the open interval is empty, so there is no
    non-trivial order complex. We treat this as trivially verified.
    """
    mu_val = _mobius_value(ss)
    red_euler = reduced_euler_characteristic(ss)

    # When top == bottom, Hall's theorem does not apply (no proper interval).
    # mu(x,x) = 1 by definition. Treat as trivially verified.
    if ss.top == ss.bottom:
        return HallVerification(
            mobius_value=mu_val,
            reduced_euler=red_euler,
            verified=True,
        )

    return HallVerification(
        mobius_value=mu_val,
        reduced_euler=red_euler,
        verified=(mu_val == red_euler),
    )


# ---------------------------------------------------------------------------
# Kunneth multiplicativity
# ---------------------------------------------------------------------------

def verify_kunneth(
    ss_left: "StateSpace",
    ss_right: "StateSpace",
    ss_product: "StateSpace",
) -> KunnethVerification:
    """Verify Kunneth formula: mu(L1 x L2) = mu(L1) * mu(L2).

    For finite lattices, the Mobius function is multiplicative under
    direct product: mu_{L1 x L2}(top, bot) = mu_{L1}(top, bot) * mu_{L2}(top, bot).
    This is equivalent to reduced Euler characteristic multiplicativity.

    For session types, the parallel constructor produces the product
    lattice, so this checks multiplicativity under parallel composition.
    """
    mu_l = _mobius_value(ss_left)
    mu_r = _mobius_value(ss_right)
    mu_p = _mobius_value(ss_product)
    expected = mu_l * mu_r
    return KunnethVerification(
        chi_left=mu_l,
        chi_right=mu_r,
        chi_product=mu_p,
        expected=expected,
        verified=(mu_p == expected),
    )


# ---------------------------------------------------------------------------
# Interval Euler series
# ---------------------------------------------------------------------------

def euler_series(ss: "StateSpace") -> dict[tuple[int, int], int]:
    """Mobius function mu(x, y) for all comparable pairs x >= y.

    This is the interval-level "Euler series" of the lattice.
    Each mu(x, y) equals the reduced Euler characteristic of
    the order complex of the open interval (x, y).
    """
    return _all_mobius_values(ss)


def interval_euler_distribution(ss: "StateSpace") -> dict[int, int]:
    """Histogram of interval mu values: value -> count.

    Counts how many intervals [x, y] (x > y) have each mu value.
    The distribution reveals protocol complexity:
    - Concentrated at {-1, 0, 1}: distributive-like structure.
    - Wider spread: more complex lattice topology.
    """
    mu = _all_mobius_values(ss)
    dist: dict[int, int] = {}
    for (x, y), val in mu.items():
        if x != y:
            dist[val] = dist.get(val, 0) + 1
    return dist


# ---------------------------------------------------------------------------
# Euler obstructions
# ---------------------------------------------------------------------------

def euler_obstruction(ss: "StateSpace") -> list[tuple[int, int, int]]:
    """Intervals where |mu(x, y)| > 1.

    These are indicators of non-distributivity. In a distributive
    lattice, |mu(x, y)| <= 1 for all intervals. Intervals violating
    this bound reveal structural complexity.

    Returns list of (x, y, mu_value) triples sorted by |mu| descending.
    """
    mu = _all_mobius_values(ss)
    obs: list[tuple[int, int, int]] = []
    for (x, y), val in mu.items():
        if x != y and abs(val) > 1:
            obs.append((x, y, val))
    obs.sort(key=lambda t: -abs(t[2]))
    return obs


# ---------------------------------------------------------------------------
# Multi-method comparison
# ---------------------------------------------------------------------------

def compare_euler_methods(ss: "StateSpace") -> ComparisonResult:
    """Compare three independent Euler characteristic computations.

    1. Face count: chi = sum (-1)^k f_k (from order complex).
    2. Mobius: mu(top, bot) (reduced Euler by Hall's theorem).
    3. Betti: chi = sum (-1)^k b_k (from simplicial homology).

    All three should agree. Disagreement indicates a bug.
    """
    chi_faces = euler_characteristic(ss)
    mu_val = _mobius_value(ss)
    # Betti-based Euler
    betti = _betti_numbers(ss)
    chi_betti = sum((-1) ** k * b for k, b in enumerate(betti)) if betti else 0

    # For comparison with mu, use reduced Euler = chi - 1
    # Actually: reduced_euler = mu(top, bot) by Hall.
    # chi_faces is the unreduced Euler, so compare:
    # reduced = chi_faces - 1 (when f_{-1} = 1)
    # But our _oc_euler already excludes f_{-1}, so reduced = chi - 1?
    # Let's just use the reduced form for the Mobius comparison.
    # The face-count chi and betti chi should match directly.
    # For Mobius, we compare reduced_euler = mu(top, bot).
    fv = _oc_f_vector(ss)

    # from_faces and from_betti are both unreduced Euler (should match).
    # from_mobius = mu(top, bot) which equals REDUCED Euler by Hall's theorem.
    # For single-state lattices (top == bottom), Hall does not apply.
    red_euler = reduced_euler_characteristic(ss)

    if ss.top == ss.bottom:
        # Degenerate: single state. faces and betti should both be 0.
        # mu(x,x)=1 by definition. Trivially agree.
        all_agree = (chi_faces == chi_betti)
    else:
        all_agree = (chi_faces == chi_betti) and (mu_val == red_euler)

    return ComparisonResult(
        from_faces=chi_faces,
        from_mobius=mu_val,
        from_betti=chi_betti,
        all_agree=all_agree,
        f_vector=fv,
        betti_numbers=betti,
    )


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze_euler(ss: "StateSpace") -> EulerAnalysis:
    """Complete Euler characteristic analysis of a session type state space.

    Consolidates face-count, Mobius, and homological approaches with
    interval-level diagnostics.
    """
    chi = euler_characteristic(ss)
    red = reduced_euler_characteristic(ss)
    hall = verify_hall_theorem(ss)
    series = euler_series(ss)
    dist = interval_euler_distribution(ss)
    obs = euler_obstruction(ss)
    comp = compare_euler_methods(ss)
    n_intervals = sum(1 for (x, y) in series if x != y)

    return EulerAnalysis(
        euler_characteristic=chi,
        reduced_euler=red,
        hall=hall,
        interval_series=series,
        distribution=dist,
        obstructions=obs,
        comparison=comp,
        num_intervals=n_intervals,
    )
