"""Möbius function analysis for session type lattices (Step 30b).

Deep analysis of the Möbius function μ on session type state spaces:

- **Möbius matrix** and pointwise Möbius function μ(x, y)
- **Philip Hall's theorem**: μ(x,y) = Σ (-1)^k c_k (alternating chain count)
- **Weisner's theorem**: Σ_{x∨a=⊤} μ(⊥,x) = 0 for any a ≠ ⊥
- **Distributivity test**: |μ(x,y)| ≤ 1 for all intervals iff distributive
- **Product formula**: μ(L₁×L₂) = μ(L₁)·μ(L₂) under parallel
- **Möbius spectrum**: distribution of all μ(x,y) values across the lattice

All computations handle cycles via SCC quotient (from zeta.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace

from reticulate.zeta import (
    _state_list,
    _adjacency,
    _reachability,
    _compute_sccs,
    _covering_relation,
    compute_rank,
    mobius_function as _mobius_func,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MobiusResult:
    """Complete Möbius function analysis.

    Attributes:
        num_states: Number of states.
        mobius_top_bottom: μ(⊤,⊥) — the key invariant.
        max_abs_mobius: Maximum |μ(x,y)| across all intervals.
        is_distributive_by_mobius: True iff |μ(x,y)| ≤ 1 for all x ≥ y.
        hall_chain_counts: For the [⊤,⊥] interval: k → c_k (chains of length k).
        hall_verification: True iff μ(⊤,⊥) = Σ (-1)^k c_k.
        spectrum: Mapping value → count of intervals with that μ value.
        weisner_verified: True iff Weisner's theorem holds for all tested atoms.
        num_intervals: Total number of comparable pairs (x > y).
        product_formula_holds: True iff μ is multiplicative under parallel (when applicable).
    """
    num_states: int
    mobius_top_bottom: int
    max_abs_mobius: int
    is_distributive_by_mobius: bool
    hall_chain_counts: dict[int, int]
    hall_verification: bool
    spectrum: dict[int, int]
    weisner_verified: bool
    num_intervals: int
    product_formula_holds: bool | None


@dataclass(frozen=True)
class HallResult:
    """Philip Hall's theorem verification for an interval.

    Attributes:
        top: Upper bound state.
        bottom: Lower bound state.
        mobius_value: μ(top, bottom) computed recursively.
        chain_counts: k → number of chains of length k.
        alternating_sum: Σ (-1)^k c_k.
        verified: True iff mobius_value == alternating_sum.
    """
    top: int
    bottom: int
    mobius_value: int
    chain_counts: dict[int, int]
    alternating_sum: int
    verified: bool


# ---------------------------------------------------------------------------
# Möbius matrix (SCC-aware)
# ---------------------------------------------------------------------------

def mobius_matrix(ss: "StateSpace") -> tuple[list[list[int]], list[int]]:
    """Compute the Möbius matrix M[i][j] = μ(state_i, state_j).

    Works on SCC quotient for cycle-aware computation.
    Returns (matrix, states) where states[i] is the state ID for row/col i.
    """
    mu = _mobius_func(ss)
    states = _state_list(ss)
    n = len(states)
    idx = {s: i for i, s in enumerate(states)}

    M = [[0] * n for _ in range(n)]
    for (x, y), val in mu.items():
        if x in idx and y in idx:
            M[idx[x]][idx[y]] = val

    return M, states


def mobius_value(ss: "StateSpace") -> int:
    """Compute μ(⊤, ⊥) — the key Möbius invariant."""
    mu = _mobius_func(ss)
    return mu.get((ss.top, ss.bottom), 0)


def all_mobius_values(ss: "StateSpace") -> dict[tuple[int, int], int]:
    """Compute μ(x, y) for all comparable pairs x ≥ y.

    Returns dict (x, y) → μ(x, y).
    """
    return _mobius_func(ss)


# ---------------------------------------------------------------------------
# Philip Hall's theorem
# ---------------------------------------------------------------------------

def _count_chains_in_interval(
    ss: "StateSpace", top: int, bottom: int,
) -> dict[int, int]:
    """Count chains of each length from top to bottom (Hall's convention).

    A chain of length k is a sequence top = z_0 > z_1 > ... > z_k = bottom
    where z_i > z_{i+1} (strict order, not necessarily covering).
    Philip Hall's theorem: μ(x,y) = Σ_{k≥1} (-1)^k c_k.
    """
    reach = _reachability(ss)
    scc_map, _ = _compute_sccs(ss)

    # Check top strictly above bottom
    if bottom not in reach[top] or scc_map[top] == scc_map[bottom]:
        return {}

    # Elements strictly between top and bottom (open interval)
    between = [
        s for s in ss.states
        if s in reach[top] and bottom in reach[s]
        and scc_map[s] != scc_map[top] and scc_map[s] != scc_map[bottom]
    ]

    # DFS: enumerate chains top > z_1 > z_2 > ... > bottom
    # Each chain picks a subset of `between` that is totally ordered.
    # To avoid permutation duplicates, track used elements.
    counts: dict[int, int] = {}

    def _dfs(current: int, depth: int, used: set[int]) -> None:
        # Option 1: close the chain here → current > bottom
        counts[depth + 1] = counts.get(depth + 1, 0) + 1
        # Option 2: extend through an unused intermediate
        for s in between:
            if s in used:
                continue
            if (s in reach[current]
                    and scc_map[s] != scc_map[current]):
                used.add(s)
                _dfs(s, depth + 1, used)
                used.discard(s)

    _dfs(top, 0, set())
    return counts


def verify_hall(ss: "StateSpace", top: int | None = None, bottom: int | None = None) -> HallResult:
    """Verify Philip Hall's theorem: μ(x,y) = Σ (-1)^k c_k.

    Args:
        ss: State space.
        top: Upper bound (defaults to ss.top).
        bottom: Lower bound (defaults to ss.bottom).

    Returns:
        HallResult with verification status.
    """
    if top is None:
        top = ss.top
    if bottom is None:
        bottom = ss.bottom

    mu = _mobius_func(ss)
    mu_val = mu.get((top, bottom), 0)

    # Hall's theorem applies only for x > y (strict). For x = y, μ = 1 by definition.
    if top == bottom:
        return HallResult(
            top=top, bottom=bottom, mobius_value=1,
            chain_counts={}, alternating_sum=1, verified=True,
        )

    chain_counts = _count_chains_in_interval(ss, top, bottom)

    alt_sum = sum((-1) ** k * c for k, c in chain_counts.items())

    return HallResult(
        top=top,
        bottom=bottom,
        mobius_value=mu_val,
        chain_counts=chain_counts,
        alternating_sum=alt_sum,
        verified=(mu_val == alt_sum),
    )


# ---------------------------------------------------------------------------
# Weisner's theorem
# ---------------------------------------------------------------------------

def _find_atoms(ss: "StateSpace") -> list[int]:
    """Find atoms of the lattice: elements that cover ⊥ (bottom)."""
    covers = _covering_relation(ss)
    # Atoms cover bottom — but we need elements covered BY top
    # Actually, atoms = elements covered by some element AND covering bottom
    # In our convention, top is greatest: atoms are elements that ⊤ covers
    # Wait — atoms in lattice theory are elements that cover ⊥.
    bottom = ss.bottom
    return [y for (x, y) in covers if y == bottom and x != bottom]


def _find_coatoms(ss: "StateSpace") -> list[int]:
    """Find coatoms: elements covered by ⊤ (top)."""
    covers = _covering_relation(ss)
    top = ss.top
    return [y for (x, y) in covers if x == top]


def _compute_join(ss: "StateSpace", a: int, b: int) -> int | None:
    """Compute a ∨ b (join / least upper bound).

    Returns None if join doesn't exist.
    """
    reach = _reachability(ss)

    # Upper bounds: states that can reach both a and b
    # Wait — in our convention x ≥ y means y reachable from x.
    # So upper bound of {a, b} = state u with a ∈ reach[u] and b ∈ reach[u].
    upper_bounds = [
        u for u in ss.states
        if a in reach[u] and b in reach[u]
    ]
    if not upper_bounds:
        return None

    # Join = least upper bound = u such that u ∈ reach[v] for all upper bounds v
    # Actually, join = u that is reachable from all other upper bounds
    # No — join = least = one that all others can reach (lowest in our ordering)
    for u in upper_bounds:
        if all(u in reach[v] for v in upper_bounds):
            return u
    return None


def verify_weisner(ss: "StateSpace") -> tuple[bool, list[tuple[int, bool]]]:
    """Verify Weisner's theorem: for each coatom a, Σ_{x∨a=⊤} μ(⊥,x) = 0...

    Actually, Weisner's theorem says: for a ≠ ⊥,
    Σ_{x: x∨a = ⊤} μ(⊥, x) = 0.

    We adapt to our convention (top = initial, bottom = end).
    Weisner for top: for any a ≠ ⊤,
    Σ_{x: x∧a = ⊥} μ(x, ⊤) = 0.

    But the more standard form using our convention where ⊤ is top:
    For any a ≠ ⊥ in L, Σ_{x: x∨a = ⊤} μ(⊥, x) = 0.

    We test with coatoms (elements covered by ⊤).
    """
    mu = _mobius_func(ss)
    reach = _reachability(ss)
    top = ss.top
    bottom = ss.bottom

    # Weisner's theorem: for any a ≠ ⊥, Σ_{x: x∨a = ⊤} μ(⊥, x) = 0.
    # We need μ(⊥, x) = μ from bottom UP. Our _mobius_func gives μ(top, x).
    # Compute the "upward" Möbius function: μ_up(x) = μ(bottom, x).
    # For the upward direction, we re-derive using the reversed poset.

    scc_map, _ = _compute_sccs(ss)

    # Build reversed adjacency for upward Möbius
    # In reversed poset: x ≥_rev y iff y ≥ x iff x ∈ reach[y]
    reach = _reachability(ss)

    # Upward reachability: up_reach[s] = states that can reach s
    up_reach: dict[int, set[int]] = {s: set() for s in ss.states}
    for s in ss.states:
        for t in reach[s]:
            up_reach[t].add(s)

    # Compute μ_up(bottom, x) for all x reachable from bottom (= all x)
    # μ_up(bottom, bottom) = 1
    # μ_up(bottom, x) = -Σ_{bottom ≤ z < x} μ_up(bottom, z)
    # where bottom ≤ z means z in reach[... er, z reachable from bottom
    # In our poset: bottom ≤ z iff z ∈ up_reach[bottom]... no.
    # bottom ≤ z iff bottom ≤ z iff z is above bottom iff bottom ∈ reach[z].

    # Actually, μ(⊥, x) in the standard (upward) sense:
    # ⊥ ≤ z ≤ x means z is between ⊥ and x, i.e., bottom reachable from z
    # and x reachable from z... no. ⊥ ≤ z means z ≥ ⊥, always true in a lattice.
    # z ≤ x means z below x, i.e., z reachable from x.

    # So [⊥, x] = {z : x ≥ z ≥ ⊥} = {z : z ∈ reach[x] and ⊥ ∈ reach[z]}
    # = {z : z reachable from x and bottom reachable from z}

    # Compute μ_up via the standard recursion on the quotient
    reps = sorted(set(scc_map.values()))
    q_adj_up: dict[int, set[int]] = {r: set() for r in reps}
    adj = _adjacency(ss)
    for s in ss.states:
        for t in adj[s]:
            sr, tr = scc_map[s], scc_map[t]
            if sr != tr:
                q_adj_up[tr].add(sr)  # reversed edges for upward

    q_reach_up: dict[int, set[int]] = {}
    for r in reps:
        visited: set[int] = set()
        stack = [r]
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            for v in q_adj_up[u]:
                stack.append(v)
        q_reach_up[r] = visited

    # Rank for ordering
    rank = compute_rank(ss)
    bottom_rep = scc_map[bottom]

    # μ_up on quotient: μ_up(bottom_rep, r) for each rep r
    mu_up_q: dict[int, int] = {bottom_rep: 1}
    # Process in increasing rank order
    for r in sorted(reps, key=lambda r: rank.get(min(s for s in ss.states if scc_map[s] == r), 0)):
        if r == bottom_rep:
            continue
        if r not in q_reach_up[bottom_rep] and bottom_rep not in q_reach_up[r]:
            continue
        # μ_up(bottom_rep, r) = -Σ_{bottom_rep ≤ z < r} μ_up(bottom_rep, z)
        # z < r means r in q_reach_up[z] (r reachable upward from z) and z ≠ r
        # Actually, bottom_rep ≤ z ≤ r in upward: z ∈ [bottom_rep, r]
        # = {z : r above z and z above bottom_rep}
        # In our poset: z reachable from r (z ∈ reach_original[r])
        # and bottom reachable from z (bottom ∈ reach_original[z])
        total = 0
        for z in reps:
            if z == r:
                continue
            # z in [bottom_rep, r]: bottom_rep reachable from z upward AND z reachable from r upward
            # In original terms: z ∈ reach[r] and bottom ∈ reach[z]
            # On quotient: z reachable from r and bottom_rep reachable from z
            if z in q_reach_up[r] and bottom_rep in q_reach_up[z]:
                total += mu_up_q.get(z, 0)
        mu_up_q[r] = -total

    # Map back to states
    mu_up: dict[int, int] = {}
    for s in ss.states:
        mu_up[s] = mu_up_q.get(scc_map[s], 0)

    # Now test Weisner: for each a ≠ bottom, Σ_{x: x∨a = top} μ_up[x] = 0
    test_elements = [
        a for a in ss.states
        if scc_map[a] != scc_map[bottom]
    ]

    if not test_elements:
        return True, []

    results: list[tuple[int, bool]] = []
    all_ok = True

    for a in test_elements:
        total = 0
        for x in ss.states:
            j = _compute_join(ss, x, a)
            if j is not None and scc_map[j] == scc_map[top]:
                total += mu_up.get(x, 0)
        ok = (total == 0)
        results.append((a, ok))
        if not ok:
            all_ok = False

    return all_ok, results


# ---------------------------------------------------------------------------
# Distributivity test via |μ|
# ---------------------------------------------------------------------------

def max_abs_mobius(ss: "StateSpace") -> int:
    """Compute max |μ(x, y)| across all comparable pairs."""
    mu = _mobius_func(ss)
    if not mu:
        return 0
    return max(abs(v) for v in mu.values())


def is_distributive_by_mobius(ss: "StateSpace") -> bool:
    """Test: |μ(x,y)| ≤ 1 for all intervals iff distributive.

    Note: |μ| ≤ 1 is NECESSARY for distributivity but not sufficient.
    (N₅ has |μ| = 1 but is non-distributive.)
    This is a quick pre-check; full distributivity requires M₃/N₅ test.
    """
    return max_abs_mobius(ss) <= 1


# ---------------------------------------------------------------------------
# Möbius spectrum
# ---------------------------------------------------------------------------

def mobius_spectrum(ss: "StateSpace") -> dict[int, int]:
    """Distribution of μ values: value → number of intervals with that value.

    Only counts off-diagonal entries (x ≠ y with x ≥ y).
    """
    mu = _mobius_func(ss)
    spectrum: dict[int, int] = {}
    for (x, y), val in mu.items():
        if x != y:
            spectrum[val] = spectrum.get(val, 0) + 1
    return spectrum


# ---------------------------------------------------------------------------
# Product formula verification
# ---------------------------------------------------------------------------

def verify_product_formula(
    ss_left: "StateSpace",
    ss_right: "StateSpace",
    ss_product: "StateSpace",
) -> bool:
    """Verify μ(L₁×L₂)(⊤,⊥) = μ(L₁)(⊤,⊥) · μ(L₂)(⊤,⊥).

    This is the multiplicativity of the Möbius function under parallel.
    """
    mu_left = mobius_value(ss_left)
    mu_right = mobius_value(ss_right)
    mu_product = mobius_value(ss_product)
    return mu_product == mu_left * mu_right


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_mobius(ss: "StateSpace") -> MobiusResult:
    """Complete Möbius function analysis of a session type state space."""
    mu_top_bot = mobius_value(ss)
    max_mu = max_abs_mobius(ss)
    dist_test = is_distributive_by_mobius(ss)
    hall = verify_hall(ss)
    spectrum = mobius_spectrum(ss)

    weisner_ok, _ = verify_weisner(ss)

    mu_all = _mobius_func(ss)
    num_intervals = sum(1 for (x, y) in mu_all if x != y)

    return MobiusResult(
        num_states=len(ss.states),
        mobius_top_bottom=mu_top_bot,
        max_abs_mobius=max_mu,
        is_distributive_by_mobius=dist_test,
        hall_chain_counts=hall.chain_counts,
        hall_verification=hall.verified,
        spectrum=spectrum,
        weisner_verified=weisner_ok,
        num_intervals=num_intervals,
        product_formula_holds=None,  # Requires product state space
    )
