"""Lattice valuations and the FKG inequality for session type lattices (Step 30y).

A **valuation** on a lattice L is a function v: L → ℝ satisfying:
    v(x ∧ y) + v(x ∨ y) = v(x) + v(y)    for all x, y ∈ L.

This is the lattice-theoretic analogue of inclusion-exclusion.

Key results:
- The rank function is a valuation on modular lattices (Birkhoff 1935).
- Every valuation on a distributive lattice is determined by its values
  on join-irreducible elements.
- The FKG inequality holds on distributive lattices: monotone increasing
  functions are positively correlated under log-supermodular measures.

Key functions:
  - ``rank_function(ss)``           -- compute rank for each state
  - ``height_function(ss)``         -- compute height for each state
  - ``check_valuation(ss, v)``      -- verify valuation identity
  - ``is_rank_valuation(ss)``       -- check if rank is a valuation
  - ``valuation_defect(ss, v)``     -- measure deviation from valuation
  - ``check_fkg(ss, f, g, mu)``     -- verify FKG inequality
  - ``is_log_supermodular(ss, mu)`` -- check log-supermodularity of measure
  - ``monotone_correlation(ss)``    -- compute correlation of monotone functions
  - ``analyze_valuations(ss)``      -- full valuation analysis
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValuationResult:
    """Result of checking the valuation identity for a function.

    Attributes:
        is_valuation: True iff v(x∧y) + v(x∨y) = v(x) + v(y) for all x, y.
        max_defect: Maximum |v(x∧y) + v(x∨y) - v(x) - v(y)| over all pairs.
        num_pairs_checked: Number of incomparable pairs checked.
        failing_pair: First failing pair (x, y, defect), or None.
    """
    is_valuation: bool
    max_defect: float
    num_pairs_checked: int
    failing_pair: tuple[int, int, float] | None


@dataclass(frozen=True)
class FKGResult:
    """Result of checking the FKG inequality.

    Attributes:
        holds: True iff E[fg] >= E[f] * E[g].
        e_fg: E[fg] — expected value of f*g.
        e_f: E[f] — expected value of f.
        e_g: E[g] — expected value of g.
        correlation: E[fg] - E[f]*E[g] (non-negative if FKG holds).
        is_log_supermodular: True iff the measure is log-supermodular.
    """
    holds: bool
    e_fg: float
    e_f: float
    e_g: float
    correlation: float
    is_log_supermodular: bool


@dataclass(frozen=True)
class ValuationAnalysis:
    """Complete valuation analysis of a state space.

    Attributes:
        rank_is_valuation: True iff rank function satisfies valuation identity.
        height_is_valuation: True iff height function satisfies valuation identity.
        rank_defect: Maximum defect for rank function.
        height_defect: Maximum defect for height function.
        rank: Mapping state → rank (min distance from bottom).
        height: Mapping state → height (max distance from bottom).
        is_graded: True iff rank == height for all states.
        is_modular: True iff no N₅ sublattice (from lattice.py).
        num_states: Number of states.
        valuation_dimension_lower: Lower bound on dimension of valuation space.
    """
    rank_is_valuation: bool
    height_is_valuation: bool
    rank_defect: float
    height_defect: float
    rank: dict[int, int]
    height: dict[int, int]
    is_graded: bool
    is_modular: bool
    num_states: int
    valuation_dimension_lower: int


# ---------------------------------------------------------------------------
# Internal: reachability and ordering
# ---------------------------------------------------------------------------

def _build_adj(ss: StateSpace) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    """Build forward and reverse adjacency."""
    fwd: dict[int, set[int]] = {s: set() for s in ss.states}
    rev: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        fwd[src].add(tgt)
        rev[tgt].add(src)
    return fwd, rev


def _reachability(ss: StateSpace) -> dict[int, set[int]]:
    """Forward reachability (inclusive)."""
    fwd, _ = _build_adj(ss)
    reach: dict[int, set[int]] = {}
    for s in ss.states:
        visited: set[int] = set()
        stack = [s]
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            for t in fwd.get(v, set()):
                stack.append(t)
        reach[s] = visited
    return reach


# ---------------------------------------------------------------------------
# Public API: Rank and height functions
# ---------------------------------------------------------------------------

def rank_function(ss: StateSpace) -> dict[int, int]:
    """Compute the rank function: minimum distance from bottom for each state.

    rank(bottom) = 0, rank(x) = min path length from x to bottom.
    """
    _, rev = _build_adj(ss)
    fwd, _ = _build_adj(ss)

    # BFS from bottom
    dist: dict[int, int] = {ss.bottom: 0}
    queue = [ss.bottom]
    while queue:
        s = queue.pop(0)
        for pred in rev.get(s, set()):
            if pred not in dist:
                dist[pred] = dist[s] + 1
                queue.append(pred)

    # States unreachable from bottom get rank -1
    for s in ss.states:
        if s not in dist:
            dist[s] = -1

    return dist


def height_function(ss: StateSpace) -> dict[int, int]:
    """Compute the height function: maximum distance from bottom for each state.

    height(bottom) = 0, height(x) = longest path from x to bottom.
    """
    fwd, _ = _build_adj(ss)

    # DFS with memoization, avoiding cycles
    memo: dict[int, int] = {}

    def _height(s: int, visited: frozenset[int]) -> int:
        if s == ss.bottom:
            return 0
        if s in memo:
            return memo[s]
        best = -1
        for t in fwd.get(s, set()):
            if t not in visited:
                h = _height(t, visited | {t})
                if h >= 0:
                    best = max(best, 1 + h)
        if s not in visited:
            memo[s] = best
        return best

    result: dict[int, int] = {}
    for s in ss.states:
        result[s] = _height(s, frozenset({s}))

    return result


# ---------------------------------------------------------------------------
# Public API: Valuation checking
# ---------------------------------------------------------------------------

def check_valuation(
    ss: StateSpace,
    v: dict[int, float],
    tolerance: float = 1e-10,
) -> ValuationResult:
    """Check if a function v satisfies the valuation identity.

    v(x ∧ y) + v(x ∨ y) = v(x) + v(y) for all incomparable pairs x, y.
    """
    from reticulate.lattice import compute_meet, compute_join

    reach = _reachability(ss)
    states = sorted(ss.states)

    max_defect = 0.0
    num_checked = 0
    failing = None

    for i, a in enumerate(states):
        for b in states[i + 1:]:
            # Only check incomparable pairs (comparable pairs trivially satisfy)
            a_reaches_b = b in reach.get(a, set())
            b_reaches_a = a in reach.get(b, set())
            if a_reaches_b or b_reaches_a:
                continue

            m = compute_meet(ss, a, b)
            j = compute_join(ss, a, b)
            if m is None or j is None:
                continue  # Not a lattice for this pair

            num_checked += 1
            va = v.get(a, 0.0)
            vb = v.get(b, 0.0)
            vm = v.get(m, 0.0)
            vj = v.get(j, 0.0)

            defect = abs((vm + vj) - (va + vb))
            max_defect = max(max_defect, defect)

            if defect > tolerance and failing is None:
                failing = (a, b, defect)

    is_val = max_defect <= tolerance

    return ValuationResult(
        is_valuation=is_val,
        max_defect=max_defect,
        num_pairs_checked=num_checked,
        failing_pair=failing,
    )


def is_rank_valuation(ss: StateSpace) -> bool:
    """Check if the rank function is a valuation."""
    r = rank_function(ss)
    v = {s: float(r[s]) for s in ss.states}
    return check_valuation(ss, v).is_valuation


def valuation_defect(ss: StateSpace, v: dict[int, float]) -> float:
    """Compute the maximum defect of a function from being a valuation."""
    return check_valuation(ss, v).max_defect


# ---------------------------------------------------------------------------
# Public API: FKG inequality
# ---------------------------------------------------------------------------

def is_log_supermodular(
    ss: StateSpace,
    mu: dict[int, float],
    tolerance: float = 1e-10,
) -> bool:
    """Check if a measure mu is log-supermodular.

    mu is log-supermodular iff mu(x∧y) * mu(x∨y) >= mu(x) * mu(y)
    for all x, y.
    """
    from reticulate.lattice import compute_meet, compute_join

    reach = _reachability(ss)
    states = sorted(ss.states)

    for i, a in enumerate(states):
        for b in states[i + 1:]:
            a_reaches_b = b in reach.get(a, set())
            b_reaches_a = a in reach.get(b, set())
            if a_reaches_b or b_reaches_a:
                continue

            m = compute_meet(ss, a, b)
            j = compute_join(ss, a, b)
            if m is None or j is None:
                continue

            mu_m = mu.get(m, 0.0)
            mu_j = mu.get(j, 0.0)
            mu_a = mu.get(a, 0.0)
            mu_b = mu.get(b, 0.0)

            if mu_m * mu_j < mu_a * mu_b - tolerance:
                return False

    return True


def check_fkg(
    ss: StateSpace,
    f: dict[int, float],
    g: dict[int, float],
    mu: dict[int, float],
) -> FKGResult:
    """Check the FKG inequality: E[fg] >= E[f] * E[g].

    Args:
        ss: State space (must be a distributive lattice).
        f: Monotone increasing function on the lattice.
        g: Monotone increasing function on the lattice.
        mu: Log-supermodular probability measure.

    Returns:
        FKGResult with the inequality check.
    """
    total_mu = sum(mu.get(s, 0.0) for s in ss.states)
    if total_mu <= 0:
        return FKGResult(
            holds=True, e_fg=0, e_f=0, e_g=0,
            correlation=0, is_log_supermodular=False,
        )

    # Normalize
    norm_mu = {s: mu.get(s, 0.0) / total_mu for s in ss.states}

    e_f = sum(f.get(s, 0.0) * norm_mu[s] for s in ss.states)
    e_g = sum(g.get(s, 0.0) * norm_mu[s] for s in ss.states)
    e_fg = sum(f.get(s, 0.0) * g.get(s, 0.0) * norm_mu[s] for s in ss.states)

    correlation = e_fg - e_f * e_g
    log_sup = is_log_supermodular(ss, mu)

    return FKGResult(
        holds=correlation >= -1e-10,
        e_fg=e_fg,
        e_f=e_f,
        e_g=e_g,
        correlation=correlation,
        is_log_supermodular=log_sup,
    )


def monotone_correlation(
    ss: StateSpace,
) -> float:
    """Compute correlation between rank and height under uniform measure.

    For graded lattices, rank == height so correlation is 1.
    For non-graded, the correlation may be less than 1.
    """
    r = rank_function(ss)
    h = height_function(ss)
    n = len(ss.states)
    if n == 0:
        return 0.0

    states = list(ss.states)
    r_vals = [float(r[s]) for s in states]
    h_vals = [float(h[s]) for s in states]

    mean_r = sum(r_vals) / n
    mean_h = sum(h_vals) / n

    cov = sum((r_vals[i] - mean_r) * (h_vals[i] - mean_h) for i in range(n)) / n
    var_r = sum((rv - mean_r) ** 2 for rv in r_vals) / n
    var_h = sum((hv - mean_h) ** 2 for hv in h_vals) / n

    if var_r < 1e-14 or var_h < 1e-14:
        return 1.0  # Constant functions: perfect correlation

    return cov / math.sqrt(var_r * var_h)


# ---------------------------------------------------------------------------
# Public API: Full analysis
# ---------------------------------------------------------------------------

def analyze_valuations(ss: StateSpace) -> ValuationAnalysis:
    """Full valuation analysis of a state space."""
    from reticulate.lattice import check_distributive

    r = rank_function(ss)
    h = height_function(ss)

    v_rank = {s: float(r[s]) for s in ss.states}
    v_height = {s: float(h[s]) for s in ss.states}

    rank_result = check_valuation(ss, v_rank)
    height_result = check_valuation(ss, v_height)

    is_graded = all(r[s] == h[s] for s in ss.states if r[s] >= 0 and h[s] >= 0)

    dist = check_distributive(ss)
    is_mod = dist.is_modular if dist.is_lattice else False

    # Valuation space dimension lower bound
    # For modular lattices: at least 1 (the rank function)
    # For distributive: at least |J(L)| (join-irreducibles)
    # For general: at least 1 (constant function)
    dim_lower = 1
    if is_mod:
        dim_lower = max(dim_lower, 1)  # rank function
    if dist.is_distributive:
        # Count join-irreducibles (states with exactly one lower cover)
        fwd, _ = _build_adj(ss)
        rev, _ = {}, {}
        for src, _, tgt in ss.transitions:
            rev.setdefault(tgt, set()).add(src)
        j_irr = 0
        for s in ss.states:
            if s == ss.bottom:
                continue
            upper_covers = [u for u in rev.get(s, set()) if u != s]
            if len(upper_covers) == 1:
                j_irr += 1
        dim_lower = max(dim_lower, j_irr)

    return ValuationAnalysis(
        rank_is_valuation=rank_result.is_valuation,
        height_is_valuation=height_result.is_valuation,
        rank_defect=rank_result.max_defect,
        height_defect=height_result.max_defect,
        rank=r,
        height=h,
        is_graded=is_graded,
        is_modular=is_mod,
        num_states=len(ss.states),
        valuation_dimension_lower=dim_lower,
    )
