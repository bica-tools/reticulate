"""Correlation inequalities on session type lattices (Step 30z).

Four correlation inequalities, in increasing generality:

1. **Harris (1960)**: For a finite distributive lattice with uniform measure,
   monotone increasing functions f, g satisfy E[fg] ≥ E[f]·E[g].

2. **Holley (1974)**: If μ₁ ≤_FKG μ₂ (stochastic domination via lattice
   condition), then E₁[f] ≤ E₂[f] for monotone increasing f.

3. **FKG (1971)**: Generalizes Harris to log-supermodular measures μ on
   distributive lattices: μ(x∧y)·μ(x∨y) ≥ μ(x)·μ(y) implies positive
   correlation of monotone functions.

4. **Ahlswede-Daykin (1978)**: The four-functions theorem. For functions
   α, β, γ, δ on a distributive lattice, if
   α(x)·β(y) ≤ γ(x∨y)·δ(x∧y) for all x,y, then
   (Σα)(Σβ) ≤ (Σγ)(Σδ). All above follow as corollaries.

Application to session types: "good" protocol properties (safety, liveness,
progress) are monotone increasing on the lattice. These inequalities prove
they are positively correlated — a protocol state that is safe tends to
also be live.

Key functions:
  - ``check_harris(ss, f, g)``                  -- Harris inequality
  - ``check_holley(ss, mu1, mu2, f)``           -- Holley stochastic domination
  - ``check_fkg(ss, f, g, mu)``                 -- FKG inequality
  - ``check_ahlswede_daykin(ss, a, b, g, d)``   -- four-functions theorem
  - ``is_monotone_increasing(ss, f)``            -- monotonicity check
  - ``generate_random_monotone(ss, seed)``       -- random monotone function
  - ``fkg_gap(ss, f, g, mu)``                   -- correlation gap
  - ``correlation_profile(ss)``                  -- test all inequalities
  - ``analyze_correlation(ss)``                  -- full analysis
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CorrelationResult:
    """Result of checking a correlation inequality.

    Attributes:
        inequality: Name of the inequality ("harris", "fkg", etc.).
        holds: True iff the inequality is satisfied.
        lhs: Left-hand side value (E[fg] or Σα·Σβ).
        rhs: Right-hand side value (E[f]·E[g] or Σγ·Σδ).
        gap: lhs - rhs (non-negative when inequality holds).
        num_states: Number of lattice elements.
    """
    inequality: str
    holds: bool
    lhs: float
    rhs: float
    gap: float
    num_states: int


@dataclass(frozen=True)
class StochasticDominance:
    """Result of Holley's stochastic domination check.

    Attributes:
        dominates: True iff μ₂ stochastically dominates μ₁.
        holley_condition: True iff μ₁(x∧y)·μ₂(x∨y) ≥ μ₁(x∨y)·μ₂(x∧y).
        expectation_gap: E₂[f] - E₁[f] for the test function f.
        failing_pair: First pair (x, y) violating the condition, or None.
    """
    dominates: bool
    holley_condition: bool
    expectation_gap: float
    failing_pair: tuple[int, int] | None


@dataclass(frozen=True)
class FourFunctionsResult:
    """Result of Ahlswede-Daykin four-functions theorem check.

    Attributes:
        holds: True iff (Σα)(Σβ) ≤ (Σγ)(Σδ).
        sum_alpha: Σα over all lattice elements.
        sum_beta: Σβ.
        sum_gamma: Σγ.
        sum_delta: Σδ.
        lhs: (Σα)(Σβ).
        rhs: (Σγ)(Σδ).
        pointwise_holds: True iff α(x)β(y) ≤ γ(x∨y)δ(x∧y) for all x,y.
        failing_pair: First pair violating pointwise condition, or None.
    """
    holds: bool
    sum_alpha: float
    sum_beta: float
    sum_gamma: float
    sum_delta: float
    lhs: float
    rhs: float
    pointwise_holds: bool
    failing_pair: tuple[int, int] | None


@dataclass(frozen=True)
class CorrelationAnalysis:
    """Full correlation analysis across all four inequalities.

    Attributes:
        harris: Harris inequality result (uniform measure).
        fkg: FKG inequality result (rank-exponential measure).
        holley: Holley check between uniform and rank-exponential.
        ahlswede_daykin: AD check with standard test functions.
        is_distributive: True iff the lattice is distributive.
        num_states: Number of states.
        avg_fkg_gap: Average FKG gap over random monotone pairs.
    """
    harris: CorrelationResult
    fkg: CorrelationResult
    holley: StochasticDominance
    ahlswede_daykin: FourFunctionsResult
    is_distributive: bool
    num_states: int
    avg_fkg_gap: float


# ---------------------------------------------------------------------------
# Internal: lattice operations
# ---------------------------------------------------------------------------

def _reachability(ss: StateSpace) -> dict[int, set[int]]:
    """Forward reachability (inclusive)."""
    adj: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        adj[src].add(tgt)
    reach: dict[int, set[int]] = {}
    for s in ss.states:
        visited: set[int] = set()
        stack = [s]
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            for t in adj.get(v, set()):
                stack.append(t)
        reach[s] = visited
    return reach


def _rank(ss: StateSpace) -> dict[int, int]:
    """Minimum distance from bottom (BFS)."""
    rev: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        rev.setdefault(tgt, set()).add(src)
    dist: dict[int, int] = {}
    if ss.bottom in ss.states:
        dist[ss.bottom] = 0
        queue = [ss.bottom]
        while queue:
            s = queue.pop(0)
            for pred in rev.get(s, set()):
                if pred not in dist:
                    dist[pred] = dist[s] + 1
                    queue.append(pred)
    for s in ss.states:
        if s not in dist:
            dist[s] = 0
    return dist


def _uniform_measure(ss: StateSpace) -> dict[int, float]:
    """Uniform probability measure."""
    n = len(ss.states)
    return {s: 1.0 / n for s in ss.states} if n > 0 else {}


def _rank_exponential_measure(ss: StateSpace) -> dict[int, float]:
    """Exponential-of-rank measure: μ(s) ∝ 2^rank(s). Log-supermodular."""
    r = _rank(ss)
    raw = {s: 2.0 ** r[s] for s in ss.states}
    total = sum(raw.values())
    return {s: v / total for s, v in raw.items()} if total > 0 else raw


# ---------------------------------------------------------------------------
# Public API: Monotonicity
# ---------------------------------------------------------------------------

def is_monotone_increasing(ss: StateSpace, f: dict[int, float]) -> bool:
    """Check if f is monotone increasing: x ≥ y implies f(x) ≥ f(y).

    In our ordering, x ≥ y means y is reachable from x.
    """
    reach = _reachability(ss)
    for x in ss.states:
        for y in ss.states:
            if y in reach[x] and x != y:  # x ≥ y
                if f.get(x, 0.0) < f.get(y, 0.0) - 1e-12:
                    return False
    return True


def is_monotone_decreasing(ss: StateSpace, f: dict[int, float]) -> bool:
    """Check if f is monotone decreasing: x ≥ y implies f(x) ≤ f(y)."""
    neg_f = {s: -v for s, v in f.items()}
    return is_monotone_increasing(ss, neg_f)


def generate_random_monotone(ss: StateSpace, seed: int = 42) -> dict[int, float]:
    """Generate a random monotone increasing function on the lattice.

    Uses the rank function plus random noise that preserves monotonicity.
    """
    rng = random.Random(seed)
    r = _rank(ss)
    max_rank = max(r.values()) if r else 0
    # f(s) = rank(s) + small random perturbation
    f: dict[int, float] = {}
    for s in ss.states:
        base = r[s] / (max_rank + 1) if max_rank > 0 else 0.0
        noise = rng.uniform(0, 0.1 / (max_rank + 1))
        f[s] = base + noise
    return f


# ---------------------------------------------------------------------------
# Public API: Harris inequality
# ---------------------------------------------------------------------------

def check_harris(
    ss: StateSpace,
    f: dict[int, float],
    g: dict[int, float],
) -> CorrelationResult:
    """Check Harris inequality: E[fg] ≥ E[f]·E[g] under uniform measure.

    Harris (1960): for monotone increasing f, g on a distributive lattice
    with uniform measure, the correlation is non-negative.
    """
    mu = _uniform_measure(ss)

    e_f = sum(f.get(s, 0.0) * mu[s] for s in ss.states)
    e_g = sum(g.get(s, 0.0) * mu[s] for s in ss.states)
    e_fg = sum(f.get(s, 0.0) * g.get(s, 0.0) * mu[s] for s in ss.states)

    gap = e_fg - e_f * e_g

    return CorrelationResult(
        inequality="harris",
        holds=gap >= -1e-10,
        lhs=e_fg,
        rhs=e_f * e_g,
        gap=gap,
        num_states=len(ss.states),
    )


# ---------------------------------------------------------------------------
# Public API: FKG inequality
# ---------------------------------------------------------------------------

def is_log_supermodular(
    ss: StateSpace,
    mu: dict[int, float],
    tolerance: float = 1e-10,
) -> bool:
    """Check if μ is log-supermodular: μ(x∧y)·μ(x∨y) ≥ μ(x)·μ(y)."""
    from reticulate.lattice import compute_meet, compute_join

    reach = _reachability(ss)
    states = sorted(ss.states)

    for i, a in enumerate(states):
        for b in states[i + 1:]:
            # Only check incomparable pairs
            if b in reach.get(a, set()) or a in reach.get(b, set()):
                continue
            m = compute_meet(ss, a, b)
            j = compute_join(ss, a, b)
            if m is None or j is None:
                continue
            if mu.get(m, 0.0) * mu.get(j, 0.0) < mu.get(a, 0.0) * mu.get(b, 0.0) - tolerance:
                return False
    return True


def check_fkg(
    ss: StateSpace,
    f: dict[int, float],
    g: dict[int, float],
    mu: dict[int, float],
) -> CorrelationResult:
    """Check FKG inequality: E_μ[fg] ≥ E_μ[f]·E_μ[g].

    FKG (1971): for monotone increasing f, g on a distributive lattice
    with log-supermodular measure μ.
    """
    total = sum(mu.get(s, 0.0) for s in ss.states)
    if total <= 0:
        return CorrelationResult("fkg", True, 0, 0, 0, len(ss.states))

    norm = {s: mu.get(s, 0.0) / total for s in ss.states}

    e_f = sum(f.get(s, 0.0) * norm[s] for s in ss.states)
    e_g = sum(g.get(s, 0.0) * norm[s] for s in ss.states)
    e_fg = sum(f.get(s, 0.0) * g.get(s, 0.0) * norm[s] for s in ss.states)

    gap = e_fg - e_f * e_g

    return CorrelationResult(
        inequality="fkg",
        holds=gap >= -1e-10,
        lhs=e_fg,
        rhs=e_f * e_g,
        gap=gap,
        num_states=len(ss.states),
    )


def fkg_gap(
    ss: StateSpace,
    f: dict[int, float],
    g: dict[int, float],
    mu: dict[int, float],
) -> float:
    """Compute the FKG gap: E[fg] - E[f]·E[g]."""
    return check_fkg(ss, f, g, mu).gap


# ---------------------------------------------------------------------------
# Public API: Holley inequality
# ---------------------------------------------------------------------------

def check_holley(
    ss: StateSpace,
    mu1: dict[int, float],
    mu2: dict[int, float],
    f: dict[int, float],
) -> StochasticDominance:
    """Check Holley's stochastic domination inequality.

    Holley (1974): if μ₁(x∧y)·μ₂(x∨y) ≥ μ₁(x∨y)·μ₂(x∧y) for all x,y,
    then E₂[f] ≥ E₁[f] for all monotone increasing f.

    This is the lattice condition for stochastic domination: μ₂ ≥_FKG μ₁.
    """
    from reticulate.lattice import compute_meet, compute_join

    reach = _reachability(ss)
    states = sorted(ss.states)

    # Check Holley condition
    holley_ok = True
    failing = None
    for i, a in enumerate(states):
        for b in states[i + 1:]:
            if b in reach.get(a, set()) or a in reach.get(b, set()):
                continue
            m = compute_meet(ss, a, b)
            j = compute_join(ss, a, b)
            if m is None or j is None:
                continue
            lhs = mu1.get(m, 0.0) * mu2.get(j, 0.0)
            rhs = mu1.get(j, 0.0) * mu2.get(m, 0.0)
            if lhs < rhs - 1e-10:
                holley_ok = False
                if failing is None:
                    failing = (a, b)

    # Compute expectations
    total1 = sum(mu1.get(s, 0.0) for s in ss.states)
    total2 = sum(mu2.get(s, 0.0) for s in ss.states)

    if total1 > 0 and total2 > 0:
        e1 = sum(f.get(s, 0.0) * mu1[s] / total1 for s in ss.states)
        e2 = sum(f.get(s, 0.0) * mu2[s] / total2 for s in ss.states)
        exp_gap = e2 - e1
    else:
        exp_gap = 0.0

    dominates = holley_ok and exp_gap >= -1e-10

    return StochasticDominance(
        dominates=dominates,
        holley_condition=holley_ok,
        expectation_gap=exp_gap,
        failing_pair=failing,
    )


# ---------------------------------------------------------------------------
# Public API: Ahlswede-Daykin four-functions theorem
# ---------------------------------------------------------------------------

def check_ahlswede_daykin(
    ss: StateSpace,
    alpha: dict[int, float],
    beta: dict[int, float],
    gamma: dict[int, float],
    delta: dict[int, float],
) -> FourFunctionsResult:
    """Check the Ahlswede-Daykin four-functions theorem.

    AD (1978): if α(x)·β(y) ≤ γ(x∨y)·δ(x∧y) for all x, y in L,
    then (Σα)·(Σβ) ≤ (Σγ)·(Σδ).

    This is the master inequality from which Harris, FKG, and Holley follow.
    """
    from reticulate.lattice import compute_meet, compute_join

    reach = _reachability(ss)
    states = sorted(ss.states)

    # Check pointwise condition
    pw_ok = True
    failing = None

    for a in states:
        for b in states:
            m = compute_meet(ss, a, b)
            j = compute_join(ss, a, b)
            if m is None or j is None:
                continue
            lhs_pw = alpha.get(a, 0.0) * beta.get(b, 0.0)
            rhs_pw = gamma.get(j, 0.0) * delta.get(m, 0.0)
            if lhs_pw > rhs_pw + 1e-10:
                pw_ok = False
                if failing is None:
                    failing = (a, b)

    # Compute sums
    s_alpha = sum(alpha.get(s, 0.0) for s in ss.states)
    s_beta = sum(beta.get(s, 0.0) for s in ss.states)
    s_gamma = sum(gamma.get(s, 0.0) for s in ss.states)
    s_delta = sum(delta.get(s, 0.0) for s in ss.states)

    lhs = s_alpha * s_beta
    rhs = s_gamma * s_delta

    return FourFunctionsResult(
        holds=lhs <= rhs + 1e-10,
        sum_alpha=s_alpha,
        sum_beta=s_beta,
        sum_gamma=s_gamma,
        sum_delta=s_delta,
        lhs=lhs,
        rhs=rhs,
        pointwise_holds=pw_ok,
        failing_pair=failing,
    )


# ---------------------------------------------------------------------------
# Public API: Correlation profile and analysis
# ---------------------------------------------------------------------------

def correlation_profile(
    ss: StateSpace,
    n_trials: int = 20,
    seed: int = 42,
) -> dict[str, float]:
    """Test all four inequalities on random monotone function pairs.

    Returns average gaps for Harris and FKG across n_trials random pairs.
    """
    harris_gaps: list[float] = []
    fkg_gaps: list[float] = []

    mu_uniform = _uniform_measure(ss)
    mu_rank = _rank_exponential_measure(ss)

    for i in range(n_trials):
        f = generate_random_monotone(ss, seed=seed + i)
        g = generate_random_monotone(ss, seed=seed + i + 1000)

        h = check_harris(ss, f, g)
        harris_gaps.append(h.gap)

        fk = check_fkg(ss, f, g, mu_rank)
        fkg_gaps.append(fk.gap)

    return {
        "harris_avg_gap": sum(harris_gaps) / len(harris_gaps) if harris_gaps else 0.0,
        "harris_min_gap": min(harris_gaps) if harris_gaps else 0.0,
        "harris_all_hold": all(g >= -1e-10 for g in harris_gaps),
        "fkg_avg_gap": sum(fkg_gaps) / len(fkg_gaps) if fkg_gaps else 0.0,
        "fkg_min_gap": min(fkg_gaps) if fkg_gaps else 0.0,
        "fkg_all_hold": all(g >= -1e-10 for g in fkg_gaps),
        "n_trials": n_trials,
    }


def analyze_correlation(ss: StateSpace, seed: int = 42) -> CorrelationAnalysis:
    """Full correlation analysis across all four inequalities."""
    from reticulate.lattice import check_distributive

    f = generate_random_monotone(ss, seed=seed)
    g = generate_random_monotone(ss, seed=seed + 1)

    mu_uniform = _uniform_measure(ss)
    mu_rank = _rank_exponential_measure(ss)

    harris = check_harris(ss, f, g)
    fkg = check_fkg(ss, f, g, mu_rank)
    holley = check_holley(ss, mu_uniform, mu_rank, f)

    # For AD, use α=β=μ·f, γ=δ=μ (which recovers FKG as corollary)
    alpha = {s: mu_rank.get(s, 0.0) * f.get(s, 0.0) for s in ss.states}
    beta = {s: mu_rank.get(s, 0.0) * g.get(s, 0.0) for s in ss.states}
    gamma = {s: mu_rank.get(s, 0.0) * f.get(s, 0.0) * g.get(s, 0.0) for s in ss.states}
    delta = dict(mu_rank)
    ad = check_ahlswede_daykin(ss, alpha, beta, gamma, delta)

    dist = check_distributive(ss)

    # Average FKG gap
    profile = correlation_profile(ss, n_trials=10, seed=seed)

    return CorrelationAnalysis(
        harris=harris,
        fkg=fkg,
        holley=holley,
        ahlswede_daykin=ad,
        is_distributive=dist.is_distributive if dist.is_lattice else False,
        num_states=len(ss.states),
        avg_fkg_gap=profile["fkg_avg_gap"],
    )
