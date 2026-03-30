"""Zeta polynomial and chain counting for session type lattices (Step 32f).

The zeta polynomial Z(P, k) counts the number of multichains of length k
in a finite poset P:

    Z(P, k) = |{(x_0, x_1, ..., x_k) : x_0 <= x_1 <= ... <= x_k}|

For session type lattices, this polynomial encodes deep information about
the protocol state space:

- **Z(P, 1) = |P|**: the number of states
- **Z(P, 2) = |{(x,y) : x <= y}|**: the number of comparable pairs + |P|
- **Z(P, -1) = mu(0,1)**: Philip Hall's theorem connects to Mobius function
- **Z(P, k) is a polynomial in k** of degree = height of P

Key insight: multichains count ways to "thread" through the protocol
lattice with repetition allowed, quantifying information-theoretic
capacity of session type channels.

All computations handle cycles via SCC quotient (from zeta.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace

from reticulate.zeta import (
    _state_list,
    _reachability,
    _compute_sccs,
    _covering_relation,
    compute_rank,
    compute_interval,
    mobius_function as _mobius_func,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ZetaPolynomialResult:
    """Complete zeta polynomial analysis.

    Attributes:
        num_states: Number of states in the poset.
        coefficients: Polynomial coefficients [a_0, a_1, ..., a_d] so
            Z(P, k) = a_0 + a_1*k + ... + a_d*k^d.
        degree: Degree of the polynomial (= height of poset).
        values: Z(P, k) for k = 0, 1, 2, ..., degree+2.
        mobius_value: Z(P, -1) = mu(0,1) (Philip Hall's theorem).
        hall_verified: True iff Z(P, -1) matches mu(0,1).
        chain_counts: Mapping length -> number of strict chains of that length.
        multichain_counts: Mapping k -> Z(P, k) for small k.
        total_multichains_k2: Z(P, 2) = number of comparable pairs + n.
        is_palindromic: True iff Z is palindromic (self-dual lattice hint).
    """
    num_states: int
    coefficients: list[float]
    degree: int
    values: dict[int, int]
    mobius_value: int
    hall_verified: bool
    chain_counts: dict[int, int]
    multichain_counts: dict[int, int]
    total_multichains_k2: int
    is_palindromic: bool


# ---------------------------------------------------------------------------
# Core: multichain counting
# ---------------------------------------------------------------------------

def _quotient_data(ss: "StateSpace") -> tuple[
    list[int], dict[int, int], dict[int, set[int]], dict[int, set[int]]
]:
    """Get quotient poset data for a state space.

    Returns (reps, scc_map, scc_members, q_reach).
    """
    scc_map, scc_members = _compute_sccs(ss)
    reps = sorted(scc_members.keys())

    # Build quotient adjacency
    q_adj: dict[int, set[int]] = {r: set() for r in reps}
    for src, _, tgt in ss.transitions:
        sr, tr = scc_map[src], scc_map[tgt]
        if sr != tr:
            q_adj[sr].add(tr)

    # Quotient reachability
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

    return reps, scc_map, scc_members, q_reach


def multichain_count(ss: "StateSpace", k: int) -> int:
    """Count multichains of length k in the poset.

    A multichain of length k is a sequence x_0 <= x_1 <= ... <= x_k
    (with repetition allowed). Works on SCC quotient for cycle handling.

    Z(P, 0) = 1 (empty chain), Z(P, 1) = n (singletons).
    """
    if k < 0:
        # Use Mobius function for Z(P, -1), generalized for other negatives
        return _zeta_poly_at_negative(ss, k)

    if k == 0:
        return 1

    reps, scc_map, scc_members, q_reach = _quotient_data(ss)
    n_reps = len(reps)

    if n_reps == 0:
        return 0

    if k == 1:
        return n_reps

    # DP: count k-multichains ending at each rep
    # mc[r] = number of multichains of length i ending at r
    # In our poset: x >= y means y reachable from x. For a multichain
    # x_0 <= x_1 <= ... <= x_k (standard order), we need x_0 >=_our x_1 >= ... >= x_k
    # i.e., each element reaches the next.
    # Sort reps by rank descending (top first), then process chains top-down.

    rank = compute_rank(ss)
    rep_rank = {r: rank.get(r, 0) for r in reps}
    # Sort by rank ascending (bottom first) so we process lower elements first
    sorted_reps = sorted(reps, key=lambda r: rep_rank[r])

    # For multichain x_0 >= x_1 >= ... >= x_{k-1}:
    # mc_prev[r] = number of (i-1)-multichains ending at r
    mc_prev = {r: 1 for r in reps}  # length 1: each element is a 1-chain

    for step in range(1, k):
        mc_next: dict[int, int] = {r: 0 for r in reps}
        for r in reps:
            # r can extend any chain ending at s where s >= r (s reaches r)
            for s in reps:
                if r in q_reach[s]:  # s >= r
                    mc_next[r] += mc_prev[s]
        mc_prev = mc_next

    return sum(mc_prev.values())


def _zeta_poly_at_negative(ss: "StateSpace", k: int) -> int:
    """Evaluate Z(P, k) at negative integers using Mobius function.

    Z(P, -1) = mu(0, 1) by Philip Hall's theorem.
    More generally, Z(P, -k) involves the Mobius function of intervals.
    """
    if k == -1:
        mu = _mobius_func(ss)
        return mu.get((ss.top, ss.bottom), 0)

    # For k < -1, use the polynomial interpolation
    # First compute enough positive values, fit polynomial, evaluate
    coeffs = zeta_polynomial_coefficients(ss)
    return _eval_poly(coeffs, k)


def chain_count(ss: "StateSpace", length: int) -> int:
    """Count strict chains of a given length (no repeated elements).

    A strict chain of length k is x_0 < x_1 < ... < x_k (strictly increasing).
    Works on SCC quotient.
    """
    if length < 0:
        return 0
    if length == 0:
        return 1  # empty chain

    reps, scc_map, scc_members, q_reach = _quotient_data(ss)

    if length == 1:
        return len(reps)

    # Build strict order on quotient
    strict_below: dict[int, list[int]] = {r: [] for r in reps}
    for r in reps:
        for s in reps:
            if r != s and s in q_reach[r]:  # r > s (r strictly above s)
                strict_below[r].append(s)

    # Count chains of given length via DFS
    count = 0

    def _dfs(current: int, remaining: int) -> None:
        nonlocal count
        if remaining == 0:
            count += 1
            return
        for s in strict_below[current]:
            _dfs(s, remaining - 1)

    for r in reps:
        _dfs(r, length - 1)

    return count


def all_chain_counts(ss: "StateSpace") -> dict[int, int]:
    """Count strict chains of every length.

    Returns dict mapping length -> count.
    """
    reps, scc_map, scc_members, q_reach = _quotient_data(ss)
    height = _compute_height_from_reps(reps, q_reach)

    result: dict[int, int] = {}
    for k in range(0, height + 2):
        c = chain_count(ss, k)
        if c > 0:
            result[k] = c

    return result


def chains_top_to_bottom(ss: "StateSpace") -> dict[int, int]:
    """Count strict chains of each length from top to bottom.

    A chain of length k from top to bottom passes through k-2 intermediate
    elements. Philip Hall's theorem: mu(top, bottom) = sum_{k>=1} (-1)^k c_k
    where c_k is the number of such chains.

    Returns dict mapping k -> count.
    """
    reps, scc_map, scc_members, q_reach = _quotient_data(ss)
    top_rep = scc_map[ss.top]
    bot_rep = scc_map[ss.bottom]

    if top_rep == bot_rep:
        # Same SCC (single element or cycle)
        return {}

    # Build strict below relation on quotient
    strict_below: dict[int, list[int]] = {r: [] for r in reps}
    for r in reps:
        for s in reps:
            if r != s and s in q_reach[r]:
                strict_below[r].append(s)

    counts: dict[int, int] = {}

    def _dfs(current: int, depth: int) -> None:
        if current == bot_rep:
            k = depth
            counts[k] = counts.get(k, 0) + 1
            return
        for s in strict_below[current]:
            _dfs(s, depth + 1)

    _dfs(top_rep, 0)
    return counts


def _compute_height_from_reps(
    reps: list[int], q_reach: dict[int, set[int]]
) -> int:
    """Compute height of quotient poset."""
    if not reps:
        return 0

    # Height = length of longest strict chain
    # Use memoized DFS
    memo: dict[int, int] = {}

    def _longest(r: int) -> int:
        if r in memo:
            return memo[r]
        best = 0
        for s in reps:
            if s != r and s in q_reach[r]:
                best = max(best, 1 + _longest(s))
        memo[r] = best
        return best

    return max(_longest(r) for r in reps)


# ---------------------------------------------------------------------------
# Polynomial computation via interpolation
# ---------------------------------------------------------------------------

def _eval_poly(coeffs: list[float], x: int) -> int:
    """Evaluate polynomial with given coefficients at integer x."""
    val = 0.0
    for i, c in enumerate(coeffs):
        val += c * (x ** i)
    return round(val)


def _lagrange_interpolation(points: list[tuple[int, int]]) -> list[float]:
    """Compute polynomial coefficients from (x, y) points via Lagrange.

    Returns [a_0, a_1, ..., a_d] for polynomial a_0 + a_1*x + ... + a_d*x^d.
    Uses exact rational-like arithmetic via floats (sufficient for small posets).
    """
    n = len(points)
    if n == 0:
        return [0.0]

    # Build polynomial by expanding Lagrange basis
    # P(x) = sum_i y_i * prod_{j!=i} (x - x_j) / (x_i - x_j)

    # We'll compute coefficients by expanding the product polynomials
    result = [0.0] * n

    for i in range(n):
        xi, yi = points[i]
        # Compute basis polynomial L_i(x) = prod_{j!=i} (x - x_j) / (x_i - x_j)
        # Start with [1]
        basis = [1.0]
        for j in range(n):
            if j == i:
                continue
            xj = points[j][0]
            denom = xi - xj
            # Multiply basis by (x - xj) / denom
            new_basis = [0.0] * (len(basis) + 1)
            for k in range(len(basis)):
                new_basis[k + 1] += basis[k] / denom
                new_basis[k] -= basis[k] * xj / denom
            basis = new_basis

        # Add yi * basis to result
        for k in range(len(basis)):
            if k < n:
                result[k] += yi * basis[k]

    # Clean up near-zero coefficients
    for k in range(len(result)):
        if abs(result[k]) < 1e-9:
            result[k] = 0.0
        elif abs(result[k] - round(result[k])) < 1e-9:
            result[k] = float(round(result[k]))

    # Trim trailing zeros
    while len(result) > 1 and abs(result[-1]) < 1e-9:
        result.pop()

    return result


def zeta_polynomial_coefficients(ss: "StateSpace") -> list[float]:
    """Compute the coefficients of the zeta polynomial Z(P, k).

    The zeta polynomial has degree equal to the height of P.
    We compute Z(P, k) for k = 0, 1, ..., height+1 and interpolate.
    """
    reps, scc_map, scc_members, q_reach = _quotient_data(ss)
    height = _compute_height_from_reps(reps, q_reach)
    degree = height

    # We need degree+1 points to determine a polynomial of degree d
    num_points = degree + 2  # extra for safety
    points: list[tuple[int, int]] = []
    for k in range(num_points):
        val = multichain_count(ss, k)
        points.append((k, val))

    return _lagrange_interpolation(points)


def zeta_polynomial(ss: "StateSpace", k: int) -> int:
    """Evaluate the zeta polynomial Z(P, k) at integer k.

    Z(P, k) counts multichains of length k for k >= 0.
    For negative k, uses polynomial evaluation (Philip Hall's theorem).
    """
    if k >= 0:
        return multichain_count(ss, k)
    else:
        coeffs = zeta_polynomial_coefficients(ss)
        return _eval_poly(coeffs, k)


# ---------------------------------------------------------------------------
# Palindromicity check
# ---------------------------------------------------------------------------

def is_palindromic(ss: "StateSpace") -> bool:
    """Check if the zeta polynomial is palindromic.

    Z(P, k) is palindromic iff Z(P, k) = k^d * Z(P, 1/k) where d = degree.
    Equivalent to: coefficients read the same forwards and backwards.
    This hints at self-duality of the lattice.
    """
    coeffs = zeta_polynomial_coefficients(ss)
    n = len(coeffs)
    for i in range(n // 2 + 1):
        if abs(coeffs[i] - coeffs[n - 1 - i]) > 1e-9:
            return False
    return True


# ---------------------------------------------------------------------------
# Whitney number connection
# ---------------------------------------------------------------------------

def whitney_numbers_second_kind(ss: "StateSpace") -> list[int]:
    """Compute Whitney numbers of the second kind W_k(P).

    W_k = number of elements of rank k. These appear as
    the "rank sequence" and relate to the zeta polynomial:
    Z(P, 2) = sum_k W_k * (number of elements below rank k + 1).
    """
    rank = compute_rank(ss)
    scc_map, scc_members = _compute_sccs(ss)
    reps = sorted(scc_members.keys())
    rep_rank = {r: rank.get(r, 0) for r in reps}

    max_rank = max(rep_rank.values()) if rep_rank else 0
    w = [0] * (max_rank + 1)
    for r in reps:
        w[rep_rank[r]] += 1
    return w


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze_zeta_polynomial(ss: "StateSpace") -> ZetaPolynomialResult:
    """Complete zeta polynomial analysis of a session type state space."""
    coeffs = zeta_polynomial_coefficients(ss)
    degree = len(coeffs) - 1
    while degree > 0 and abs(coeffs[degree]) < 1e-9:
        degree -= 1

    # Compute values for several k
    values: dict[int, int] = {}
    for k in range(degree + 3):
        values[k] = multichain_count(ss, k)

    # Philip Hall's theorem: mu(top, bottom) = sum_{k>=1} (-1)^k c_k
    # where c_k counts strict chains of length k from top to bottom.
    mu = _mobius_func(ss)
    mu_top_bot = mu.get((ss.top, ss.bottom), 0)

    chains = all_chain_counts(ss)

    reps, _, _, _ = _quotient_data(ss)
    tb_chains = chains_top_to_bottom(ss)
    hall_alt_sum = sum((-1)**k * c for k, c in tb_chains.items() if k >= 1)
    if len(reps) == 1:
        hall_verified = (mu_top_bot == 1)
    else:
        hall_verified = (hall_alt_sum == mu_top_bot)

    mc: dict[int, int] = {}
    for k in range(degree + 3):
        mc[k] = values[k]

    z2 = values.get(2, 0)
    palin = is_palindromic(ss)

    return ZetaPolynomialResult(
        num_states=len(reps),
        coefficients=coeffs,
        degree=degree,
        values=values,
        mobius_value=mu_top_bot,
        hall_verified=hall_verified,
        chain_counts=chains,
        multichain_counts=mc,
        total_multichains_k2=z2,
        is_palindromic=palin,
    )
