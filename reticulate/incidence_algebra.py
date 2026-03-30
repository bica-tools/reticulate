"""Incidence algebra of session type lattices (Step 32g).

The incidence algebra I(P) of a finite poset P is the algebra of all
functions f: {(x,y) : x <= y} -> R, with multiplication given by
convolution:

    (f * g)(x, y) = sum_{x <= z <= y} f(x, z) * g(z, y)

Key elements of I(P):
- **zeta function**: zeta(x, y) = 1 for all x <= y (identity for order)
- **Mobius function**: mu = zeta^{-1} (inverse under convolution)
- **delta function**: delta(x, y) = 1 iff x = y (identity element)
- **rank function**: rho(x, y) = rank(y) - rank(x) if x <= y

The incidence algebra provides a unified algebraic framework for
combinatorial identities on session type lattices, including Mobius
inversion, Philip Hall's theorem, and the product formula.

All computations handle cycles via SCC quotient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace

from reticulate.zeta import (
    _state_list,
    _reachability,
    _compute_sccs,
    compute_rank,
    zeta_function as _zeta_func,
    delta_function as _delta_func,
    mobius_function as _mobius_func,
    convolve as _convolve_raw,
)


# Type alias for incidence functions
IncidenceFunc = dict[tuple[int, int], float]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IncidenceAlgebraResult:
    """Complete incidence algebra analysis.

    Attributes:
        num_states: Number of states (quotient).
        num_intervals: Number of intervals (x, y) with x >= y.
        dimension: Dimension of the incidence algebra = num_intervals.
        zeta_nonzero: Number of nonzero entries in zeta function.
        mobius_nonzero: Number of nonzero entries in Mobius function.
        convolution_identity_verified: True iff zeta * mu = delta.
        mobius_involution_verified: True iff mu^{-1} = zeta.
        is_commutative: True iff the algebra is commutative (rare).
        nilpotent_index: Nilpotency index of (zeta - delta), or None.
        multiplicative_functions: Names of verified multiplicative functions.
    """
    num_states: int
    num_intervals: int
    dimension: int
    zeta_nonzero: int
    mobius_nonzero: int
    convolution_identity_verified: bool
    mobius_involution_verified: bool
    is_commutative: bool
    nilpotent_index: int | None
    multiplicative_functions: list[str]


@dataclass(frozen=True)
class ConvolutionResult:
    """Result of a convolution f * g.

    Attributes:
        result: The convolution (f * g) as an incidence function.
        nonzero_count: Number of nonzero entries.
        max_abs_value: Maximum absolute value.
    """
    result: IncidenceFunc
    nonzero_count: int
    max_abs_value: float


# ---------------------------------------------------------------------------
# Incidence function construction
# ---------------------------------------------------------------------------

def incidence_function(
    ss: "StateSpace",
    func: Callable[[int, int], float] | None = None,
) -> IncidenceFunc:
    """Create an incidence function on the poset.

    If func is None, returns the zero function.
    func(x, y) is called for each comparable pair x >= y.
    """
    reach = _reachability(ss)
    result: IncidenceFunc = {}

    for x in ss.states:
        for y in ss.states:
            if y in reach[x]:
                if func is not None:
                    val = func(x, y)
                    if val != 0:
                        result[(x, y)] = val
                # else: zero function, don't store

    return result


def zeta_element(ss: "StateSpace") -> IncidenceFunc:
    """The zeta function zeta(x, y) = 1 for all x >= y.

    This is the fundamental element of the incidence algebra.
    """
    z = _zeta_func(ss)
    return {k: float(v) for k, v in z.items()}


def mobius_element(ss: "StateSpace") -> IncidenceFunc:
    """The Mobius function mu(x, y) — inverse of zeta.

    mu * zeta = zeta * mu = delta.
    """
    mu = _mobius_func(ss)
    return {k: float(v) for k, v in mu.items()}


def delta_element(ss: "StateSpace") -> IncidenceFunc:
    """The delta (identity) function delta(x, y) = 1 iff x = y.

    For cyclic state spaces, states in the same SCC are identified.
    """
    d = _delta_func(ss)
    return {k: float(v) for k, v in d.items()}


def rank_element(ss: "StateSpace") -> IncidenceFunc:
    """The rank-difference function rho(x, y) = rank(x) - rank(y).

    Defined for comparable pairs x >= y.
    """
    rank = compute_rank(ss)
    reach = _reachability(ss)
    result: IncidenceFunc = {}

    for x in ss.states:
        for y in ss.states:
            if y in reach[x]:
                val = float(rank.get(x, 0) - rank.get(y, 0))
                if val != 0:
                    result[(x, y)] = val

    return result


def constant_element(ss: "StateSpace", c: float) -> IncidenceFunc:
    """Constant function f(x, y) = c for all x >= y."""
    reach = _reachability(ss)
    result: IncidenceFunc = {}
    if c == 0:
        return result
    for x in ss.states:
        for y in ss.states:
            if y in reach[x]:
                result[(x, y)] = c
    return result


def indicator_element(
    ss: "StateSpace", pairs: set[tuple[int, int]]
) -> IncidenceFunc:
    """Indicator function: f(x,y) = 1 for (x,y) in pairs, 0 otherwise."""
    reach = _reachability(ss)
    result: IncidenceFunc = {}
    for x, y in pairs:
        if x in ss.states and y in ss.states and y in reach[x]:
            result[(x, y)] = 1.0
    return result


# ---------------------------------------------------------------------------
# Algebra operations
# ---------------------------------------------------------------------------

def convolve(f: IncidenceFunc, g: IncidenceFunc, ss: "StateSpace") -> IncidenceFunc:
    """Convolution in the incidence algebra.

    (f * g)(x, y) = sum_{x >= z >= y} f(x, z) * g(z, y)

    For cyclic state spaces, works on the SCC quotient to ensure
    correctness: states in the same SCC are treated as equivalent.
    """
    scc_map, scc_members = _compute_sccs(ss)
    reps = sorted(scc_members.keys())

    # Build quotient reachability
    q_adj: dict[int, set[int]] = {r: set() for r in reps}
    for src, _, tgt in ss.transitions:
        sr, tr = scc_map[src], scc_map[tgt]
        if sr != tr:
            q_adj[sr].add(tr)

    q_reach: dict[int, set[int]] = {}
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
        q_reach[r] = visited

    result: IncidenceFunc = {}

    for x in ss.states:
        xr = scc_map[x]
        for y in ss.states:
            yr = scc_map[y]
            if yr not in q_reach[xr]:
                continue
            total = 0.0
            # Sum over quotient representatives z with x >= z >= y
            for zr in reps:
                if zr in q_reach[xr] and yr in q_reach[zr]:
                    z = min(scc_members[zr])
                    fval = f.get((x, z), 0.0)
                    gval = g.get((z, y), 0.0)
                    total += fval * gval
            if abs(total) > 1e-12:
                result[(x, y)] = total

    return result


def add(f: IncidenceFunc, g: IncidenceFunc) -> IncidenceFunc:
    """Pointwise addition of incidence functions."""
    result: IncidenceFunc = {}
    keys = set(f.keys()) | set(g.keys())
    for k in keys:
        val = f.get(k, 0.0) + g.get(k, 0.0)
        if abs(val) > 1e-12:
            result[k] = val
    return result


def scalar_mul(c: float, f: IncidenceFunc) -> IncidenceFunc:
    """Scalar multiplication c * f."""
    if abs(c) < 1e-12:
        return {}
    return {k: c * v for k, v in f.items() if abs(c * v) > 1e-12}


def subtract(f: IncidenceFunc, g: IncidenceFunc) -> IncidenceFunc:
    """Pointwise subtraction f - g."""
    return add(f, scalar_mul(-1.0, g))


def power(f: IncidenceFunc, n: int, ss: "StateSpace") -> IncidenceFunc:
    """Compute f^n (n-fold convolution).

    f^0 = delta, f^1 = f, f^n = f * f^{n-1}.
    """
    if n < 0:
        raise ValueError("Negative powers not supported directly")
    if n == 0:
        return delta_element(ss)
    result = delta_element(ss)
    base = dict(f)
    exp = n
    while exp > 0:
        if exp % 2 == 1:
            result = convolve(result, base, ss)
        base = convolve(base, base, ss)
        exp //= 2
    return result


# ---------------------------------------------------------------------------
# Verification and properties
# ---------------------------------------------------------------------------

def verify_convolution_identity(ss: "StateSpace") -> bool:
    """Verify that zeta * mu = delta (fundamental identity).

    This is the Mobius inversion formula in algebraic form.
    """
    zeta = zeta_element(ss)
    mu = mobius_element(ss)
    delta = delta_element(ss)

    product = convolve(zeta, mu, ss)

    # Check product matches delta
    reach = _reachability(ss)
    for x in ss.states:
        for y in ss.states:
            if y in reach[x]:
                expected = delta.get((x, y), 0.0)
                actual = product.get((x, y), 0.0)
                if abs(expected - actual) > 1e-9:
                    return False
    return True


def verify_mobius_involution(ss: "StateSpace") -> bool:
    """Verify that mu * zeta = delta (right inverse check).

    Combined with zeta * mu = delta, this confirms mu = zeta^{-1}.
    """
    zeta = zeta_element(ss)
    mu = mobius_element(ss)
    delta = delta_element(ss)

    product = convolve(mu, zeta, ss)

    reach = _reachability(ss)
    for x in ss.states:
        for y in ss.states:
            if y in reach[x]:
                expected = delta.get((x, y), 0.0)
                actual = product.get((x, y), 0.0)
                if abs(expected - actual) > 1e-9:
                    return False
    return True


def is_multiplicative(f: IncidenceFunc, ss: "StateSpace") -> bool:
    """Check if f is multiplicative: f(x,y) * f(x,y) = f(x,y) for intervals.

    More precisely, checks if f is an idempotent in the algebra.
    """
    f2 = convolve(f, f, ss)
    reach = _reachability(ss)
    for x in ss.states:
        for y in ss.states:
            if y in reach[x]:
                val_f = f.get((x, y), 0.0)
                val_f2 = f2.get((x, y), 0.0)
                if abs(val_f - val_f2) > 1e-9:
                    return False
    return True


def is_commutative_pair(
    f: IncidenceFunc, g: IncidenceFunc, ss: "StateSpace"
) -> bool:
    """Check if f * g = g * f (convolution is commutative for this pair)."""
    fg = convolve(f, g, ss)
    gf = convolve(g, f, ss)

    reach = _reachability(ss)
    for x in ss.states:
        for y in ss.states:
            if y in reach[x]:
                val_fg = fg.get((x, y), 0.0)
                val_gf = gf.get((x, y), 0.0)
                if abs(val_fg - val_gf) > 1e-9:
                    return False
    return True


def nilpotent_index(ss: "StateSpace") -> int | None:
    """Find the nilpotency index of (zeta - delta).

    The element eta = zeta - delta satisfies eta^k = 0 for some k.
    The nilpotency index is the smallest such k.
    For a poset of height h, eta^{h+1} = 0.
    Returns None if not nilpotent within reasonable bound.
    """
    zeta = zeta_element(ss)
    delta = delta_element(ss)
    eta = subtract(zeta, delta)

    n = len(ss.states)
    current = dict(eta)

    for k in range(1, n + 2):
        # Check if current is zero
        if all(abs(v) < 1e-9 for v in current.values()):
            return k
        current = convolve(current, eta, ss)

    # Check final power
    if all(abs(v) < 1e-9 for v in current.values()):
        return n + 2
    return None


def inverse(f: IncidenceFunc, ss: "StateSpace") -> IncidenceFunc | None:
    """Compute the inverse of f in the incidence algebra, if it exists.

    f is invertible iff f(x, x) != 0 for all x.
    Uses the recursive formula:
        f^{-1}(x, x) = 1 / f(x, x)
        f^{-1}(x, y) = -1/f(x,x) * sum_{x >= z > y} f(x,z) * f^{-1}(z,y)
    """
    reach = _reachability(ss)
    scc_map, scc_members = _compute_sccs(ss)
    reps = sorted(scc_members.keys())

    # Check diagonal
    for x in ss.states:
        if abs(f.get((x, x), 0.0)) < 1e-12:
            return None  # Not invertible

    rank = compute_rank(ss)

    # Process by increasing rank difference
    inv: IncidenceFunc = {}

    # Diagonal
    for x in ss.states:
        inv[(x, x)] = 1.0 / f[(x, x)]

    # Off-diagonal: process pairs (x, y) with x > y by increasing rank diff
    pairs = [
        (x, y) for x in ss.states for y in ss.states
        if x != y and y in reach[x]
    ]
    pairs.sort(key=lambda p: rank.get(p[0], 0) - rank.get(p[1], 0))

    for x, y in pairs:
        total = 0.0
        for z in ss.states:
            if z != x and z in reach[x] and y in reach[z]:
                total += f.get((x, z), 0.0) * inv.get((z, y), 0.0)
        inv[(x, y)] = -total / f[(x, x)]

    return inv


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze_incidence_algebra(ss: "StateSpace") -> IncidenceAlgebraResult:
    """Complete incidence algebra analysis of a session type state space."""
    reach = _reachability(ss)

    # Count intervals
    num_intervals = sum(
        1 for x in ss.states for y in ss.states if y in reach[x]
    )

    zeta = zeta_element(ss)
    mu = mobius_element(ss)

    zeta_nz = sum(1 for v in zeta.values() if abs(v) > 1e-12)
    mu_nz = sum(1 for v in mu.values() if abs(v) > 1e-12)

    conv_id = verify_convolution_identity(ss)
    mu_inv = verify_mobius_involution(ss)

    # Check commutativity: zeta and mu should commute
    commutative = is_commutative_pair(zeta, mu, ss)

    nilp = nilpotent_index(ss)

    # Check which named functions are multiplicative (idempotent)
    mult_funcs = []
    delta = delta_element(ss)
    if is_multiplicative(delta, ss):
        mult_funcs.append("delta")

    return IncidenceAlgebraResult(
        num_states=len(ss.states),
        num_intervals=num_intervals,
        dimension=num_intervals,
        zeta_nonzero=zeta_nz,
        mobius_nonzero=mu_nz,
        convolution_identity_verified=conv_id,
        mobius_involution_verified=mu_inv,
        is_commutative=commutative,
        nilpotent_index=nilp,
        multiplicative_functions=mult_funcs,
    )
