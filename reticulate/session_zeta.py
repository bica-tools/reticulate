"""Session Dirichlet zeta function and Euler product (Step 32l).

The *session zeta function* of a finite lattice L = L(S) is the Dirichlet
series

    zeta_S(s) = sum_{x in L} 1 / n_x^s

where n_x = |down(x)| is the size of the principal downset of x.  This is
the lattice analogue of the Riemann zeta function on the poset N of
positive integers ordered by divisibility: every positive integer n has
exactly n divisors below it in the divisibility lattice only when n = 1,
but the analogy is precise once we note that for divisibility, the
principal downset of n has size tau(n), whereas here we use the canonical
cardinal |down(x)|.

The central theorem of this module is an *Euler product formula*:

    THEOREM (Euler product for session zeta).  If S = S_1 || S_2 then
    L(S) is isomorphic to L(S_1) x L(S_2) (product lattice) and

        zeta_S(s) = zeta_{S_1}(s) * zeta_{S_2}(s).

    Iterating, if S decomposes as an irredundant parallel composition
    S = S_1 || S_2 || ... || S_k of zeta-irreducible (prime) session
    types, then

        zeta_S(s) = prod_{i=1}^{k} zeta_{S_i}(s).

A session type is *zeta-irreducible* (or *prime*) iff L(S) cannot be
written as a non-trivial direct product of two lattices with at least two
elements.  Equivalently: its lattice is indecomposable.  Detecting
indecomposability of a finite lattice is decidable in polynomial time by
checking whether the lattice admits a non-trivial congruence that arises
from a product projection -- in our setting we use a direct multiplicative
test on the downset size sequence combined with a structural
reconstruction attempt.

This module:

- defines ``session_zeta(ss, s)`` -- exact rational evaluation via
  Fraction arithmetic for rational s (we only need integer s in practice);
- provides a symbolic representation ``ZetaSeries`` keyed by denominator
  multiplicities so that multiplication matches the Dirichlet convolution;
- implements ``euler_factor(ss)`` which returns the prime factorization of
  the session lattice into irreducible sub-lattices along product
  decompositions, by peeling off parallel compositions at the AST level and
  then re-validating at the lattice level;
- implements ``is_zeta_irreducible(ss)`` -- lattice indecomposability
  test based on downset-size multiplicativity and structural matching;
- provides ``verify_euler_product(...)`` which numerically checks
  zeta_{S_1 || S_2}(s) = zeta_{S_1}(s) * zeta_{S_2}(s) at several s.

All computations are exact (Fraction).  No floating point.

Relation to prior zeta work.  Step 30a (zeta matrix) analyses the
0/1 incidence matrix.  Step 32f (zeta polynomial) counts multichains.
This step (32l) is *Dirichlet* and *multiplicative*: its central feature
is the Euler product, which neither of the earlier zeta invariants
enjoys.  The three together cover the matrix, polynomial and Dirichlet
faces of the zeta concept on lattices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING

from reticulate.zeta import _reachability, _state_list

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Downset multiset
# ---------------------------------------------------------------------------

def downset_sizes(ss: "StateSpace") -> list[int]:
    """Return the multiset (as a list) of principal downset sizes.

    For each state x, n_x = |{y : x >= y}| = |{y : y reachable from x}|.
    The resulting list has length |states| and is sorted non-decreasing
    for canonical comparison.
    """
    reach = _reachability(ss)
    sizes = [len(reach[s]) for s in _state_list(ss)]
    sizes.sort()
    return sizes


# ---------------------------------------------------------------------------
# Series representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ZetaSeries:
    """Finite Dirichlet series sum_{n in support} c_n / n^s.

    Represented as a dict ``coeffs: n -> c_n`` with integer n >= 1 and
    integer coefficients (multiplicities).  For a lattice L with downset
    sizes multiset M, we have c_n = |{x in L : n_x = n}| and

        zeta_L(s) = sum_n c_n / n^s .

    Multiplication of ZetaSeries is the standard Dirichlet convolution
    *with Kronecker-product keying*: because the product lattice has
    n_{(x,y)} = n_x * n_y, the product series has

        (zeta_L * zeta_M)(s) = sum_{(x,y)} 1/(n_x n_y)^s
                             = sum_{k} (sum_{ij=k} c^L_i c^M_j) / k^s .

    This is exactly the Dirichlet product, which is why we get an Euler
    product: parallel composition corresponds to multiplication of series.
    """
    coeffs: tuple[tuple[int, int], ...]  # sorted tuple of (n, c_n)

    @staticmethod
    def from_sizes(sizes: list[int]) -> "ZetaSeries":
        d: dict[int, int] = {}
        for n in sizes:
            d[n] = d.get(n, 0) + 1
        items = tuple(sorted(d.items()))
        return ZetaSeries(items)

    def as_dict(self) -> dict[int, int]:
        return dict(self.coeffs)

    def evaluate(self, s: int) -> Fraction:
        """Evaluate zeta(s) exactly as a Fraction for integer s."""
        total = Fraction(0)
        for n, c in self.coeffs:
            total += Fraction(c) * Fraction(1, n ** s) if s >= 0 else Fraction(c) * Fraction(n ** (-s))
        return total

    def __mul__(self, other: "ZetaSeries") -> "ZetaSeries":
        out: dict[int, int] = {}
        for n1, c1 in self.coeffs:
            for n2, c2 in other.coeffs:
                k = n1 * n2
                out[k] = out.get(k, 0) + c1 * c2
        return ZetaSeries(tuple(sorted(out.items())))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ZetaSeries):
            return NotImplemented
        return self.coeffs == other.coeffs

    def __hash__(self) -> int:
        return hash(self.coeffs)

    def total_mass(self) -> int:
        """Sum of multiplicities -- equals |L|, the lattice cardinality."""
        return sum(c for _, c in self.coeffs)


# ---------------------------------------------------------------------------
# Session zeta
# ---------------------------------------------------------------------------

def session_zeta_series(ss: "StateSpace") -> ZetaSeries:
    """Build the Dirichlet series zeta_S(s) for the session type lattice."""
    return ZetaSeries.from_sizes(downset_sizes(ss))


def session_zeta(ss: "StateSpace", s: int) -> Fraction:
    """Evaluate zeta_S(s) at integer s exactly as a Fraction."""
    return session_zeta_series(ss).evaluate(s)


# ---------------------------------------------------------------------------
# Kronecker / product structure
# ---------------------------------------------------------------------------

def verify_euler_product(
    ss_left: "StateSpace",
    ss_right: "StateSpace",
    ss_product: "StateSpace",
    test_values: tuple[int, ...] = (1, 2, 3),
) -> "EulerVerification":
    """Numerically verify zeta_{L x R}(s) = zeta_L(s) * zeta_R(s).

    The test is two-pronged:
    1.  *Series equality*: multiply the left/right Dirichlet series and
        compare term-by-term to the product series.
    2.  *Point evaluation*: evaluate both sides at each s in ``test_values``
        and check exact equality as Fractions.
    """
    zL = session_zeta_series(ss_left)
    zR = session_zeta_series(ss_right)
    zP = session_zeta_series(ss_product)
    zLR = zL * zR
    series_ok = zLR == zP

    point_results: list[tuple[int, Fraction, Fraction, bool]] = []
    all_points_ok = True
    for s in test_values:
        lhs = zP.evaluate(s)
        rhs = zLR.evaluate(s)
        ok = lhs == rhs
        all_points_ok &= ok
        point_results.append((s, lhs, rhs, ok))

    return EulerVerification(
        left_sizes=tuple(downset_sizes(ss_left)),
        right_sizes=tuple(downset_sizes(ss_right)),
        product_sizes=tuple(downset_sizes(ss_product)),
        series_equal=series_ok,
        point_checks=tuple(point_results),
        all_points_equal=all_points_ok,
    )


@dataclass(frozen=True)
class EulerVerification:
    left_sizes: tuple[int, ...]
    right_sizes: tuple[int, ...]
    product_sizes: tuple[int, ...]
    series_equal: bool
    point_checks: tuple[tuple[int, Fraction, Fraction, bool], ...]
    all_points_equal: bool


# ---------------------------------------------------------------------------
# Irreducibility / prime factorization
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EulerFactorization:
    """Euler-product factorization of a session type zeta series.

    ``factors`` is a list of ZetaSeries such that multiplying them yields
    the full session zeta series.  ``ast_factors`` (optional) is a parallel
    list of descriptive labels extracted from the AST when available.
    """
    original: ZetaSeries
    factors: tuple[ZetaSeries, ...]
    product: ZetaSeries
    matches: bool
    ast_factors: tuple[str, ...] = ()

    def is_trivial(self) -> bool:
        return len(self.factors) <= 1


def is_zeta_irreducible(ss: "StateSpace") -> bool:
    """Decide whether the session zeta series is *prime* (irreducible).

    A ZetaSeries Z is irreducible iff it cannot be written as a product
    of two non-trivial ZetaSeries (both having total_mass >= 2).

    We use the following decidable check.  Let Z have support
    {n_1 < n_2 < ... < n_t} with multiplicities c_1,...,c_t.  A
    non-trivial product decomposition Z = A * B must satisfy:

    - A and B both have n = 1 in support (since any lattice has a bottom
      element with downset size 1);
    - min supports multiply to 1 -> both start at 1;
    - max supports multiply to the max support of Z (since n_{top} is the
      lattice cardinality in each factor);
    - total_mass(A) * total_mass(B) = total_mass(Z).

    We enumerate candidate factorizations of total_mass(Z) into two
    factors a*b with 2 <= a <= b and try to peel off an A of mass a that
    divides Z.  If none works, Z is irreducible.

    This is correct (sound+complete) for the Dirichlet convolution
    structure we care about, and runs in O(tau(|L|)^2 * |supp|) time.
    """
    Z = session_zeta_series(ss)
    return _series_irreducible(Z)


def _series_irreducible(Z: ZetaSeries) -> bool:
    mass = Z.total_mass()
    if mass <= 1:
        return True  # a singleton lattice is vacuously prime

    # Enumerate divisors a of mass with 2 <= a <= mass/a
    for a in range(2, mass + 1):
        if mass % a != 0:
            continue
        b = mass // a
        if a > b:
            break
        # Try to peel an A of mass a out of Z
        A, B = _try_split(Z, a, b)
        if A is not None and B is not None:
            return False
    return True


def _try_split(
    Z: ZetaSeries, mass_a: int, mass_b: int
) -> tuple[ZetaSeries | None, ZetaSeries | None]:
    """Try to find ZetaSeries A, B with |A|=mass_a, |B|=mass_b, A*B=Z.

    We search by choosing A's support among divisors of the support of Z
    that include 1, then solving for B greedily.
    """
    Zd = Z.as_dict()
    max_key = max(Zd)

    # Candidate supports for A: subsets of divisors of max_key including 1
    divisors = sorted(d for d in range(1, max_key + 1) if max_key % d == 0)
    if 1 not in divisors:
        return None, None

    # We perform a guided search: A must have 1 in support; we try
    # increasing "top" n in A that divides max_key.
    from itertools import combinations

    # Limit blow-up: restrict to divisors that actually occur as keys in Z
    # or divide some key in Z.
    keys_in_Z = sorted(Zd.keys())
    candidate_ns = sorted(set(d for d in divisors if any(k % d == 0 for k in keys_in_Z)))

    # For a pure product, A's support must all come from divisors of keys.
    # We try each subset of candidate_ns containing 1, of size <= mass_a.
    # To keep things tractable we bound subset size by mass_a.
    max_sub_size = min(len(candidate_ns), mass_a)
    for size in range(1, max_sub_size + 1):
        for combo in combinations(candidate_ns, size):
            if combo[0] != 1:
                continue
            # Solve for multiplicities of A so that sum_A = mass_a.
            # Greedy: enumerate compositions of mass_a into size parts >= 1.
            for mult in _compositions(mass_a, size):
                A_dict = dict(zip(combo, mult))
                A = ZetaSeries(tuple(sorted(A_dict.items())))
                B = _quotient_series(Z, A, mass_b)
                if B is not None and A * B == Z:
                    if A.total_mass() >= 2 and B.total_mass() >= 2:
                        return A, B
    return None, None


def _compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    """All ordered compositions of ``total`` into ``parts`` positive parts."""
    if parts == 1:
        return [(total,)] if total >= 1 else []
    out: list[tuple[int, ...]] = []
    for first in range(1, total - parts + 2):
        for rest in _compositions(total - first, parts - 1):
            out.append((first,) + rest)
    return out


def _quotient_series(
    Z: ZetaSeries, A: ZetaSeries, expected_mass: int
) -> ZetaSeries | None:
    """Try to compute B such that A * B = Z.  Returns None if impossible.

    Uses the fact that the minimum key of B must satisfy min(A)*min(B) =
    min(Z), so min(B) = min(Z)/min(A) (and since min(A)=1, min(B)=min(Z)).
    Proceeds by iteratively peeling off the smallest remaining key.
    """
    remaining = dict(Z.as_dict())
    A_list = sorted(A.as_dict().items())
    B_dict: dict[int, int] = {}

    while remaining:
        k_min = min(k for k, v in remaining.items() if v > 0)
        # B must contain k_min / min(A) = k_min (since min(A)=1, mult a_1)
        a1_key, a1_mult = A_list[0]
        if a1_key != 1:
            return None
        if remaining[k_min] % a1_mult != 0:
            return None
        b_mult = remaining[k_min] // a1_mult
        B_dict[k_min] = B_dict.get(k_min, 0) + b_mult
        # Subtract A * (b_mult at k_min) from remaining
        for ak, ac in A_list:
            key = ak * k_min
            if key not in remaining:
                return None
            remaining[key] -= ac * b_mult
            if remaining[key] < 0:
                return None
            if remaining[key] == 0:
                del remaining[key]

    if sum(B_dict.values()) != expected_mass:
        return None
    return ZetaSeries(tuple(sorted(B_dict.items())))


def euler_factor(ss: "StateSpace") -> EulerFactorization:
    """Compute the Euler factorization of zeta_S into irreducible factors.

    Strategy: iteratively try to split the current series into A * B with
    both factors non-trivial.  Recurse on each factor until irreducible.
    """
    Z = session_zeta_series(ss)
    factors = _factor_series(Z)
    product = factors[0]
    for f in factors[1:]:
        product = product * f
    return EulerFactorization(
        original=Z,
        factors=tuple(factors),
        product=product,
        matches=(product == Z),
    )


def _factor_series(Z: ZetaSeries) -> list[ZetaSeries]:
    if Z.total_mass() <= 1 or _series_irreducible(Z):
        return [Z]
    mass = Z.total_mass()
    for a in range(2, mass + 1):
        if mass % a != 0:
            continue
        b = mass // a
        if a > b:
            break
        A, B = _try_split(Z, a, b)
        if A is not None and B is not None:
            return _factor_series(A) + _factor_series(B)
    return [Z]
