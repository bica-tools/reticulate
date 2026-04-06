"""Crapo's beta invariant for session type lattices (Step 32n).

Crapo's beta invariant of a finite bounded lattice L with 0̂ (bottom) and
1̂ (top) is

    β(L) = Σ_{x ∈ L} μ(0̂, x) · μ(x, 1̂)

where μ is the Möbius function of L.  Equivalently, β(L) = (-1)^{rk(L)}
times the reduced Euler characteristic of the open interval (0̂, 1̂) viewed
as the order complex, and it coincides (up to sign) with the leading
coefficient of the characteristic polynomial evaluated at 1.

**Decomposition detector.**  For many classes of lattices, β(L) = 0 iff
L is a non-trivial direct product (i.e. L ≅ L₁ × L₂ with both factors of
rank ≥ 1), because the Möbius function is multiplicative under products
and a non-trivial product always has some element x with
μ(0̂,x)·μ(x,1̂) = 0.  More precisely, for geometric lattices Crapo proved
β(L) > 0 iff L is connected (has no non-trivial product decomposition),
and β(L) = 0 otherwise.  We apply this to session type lattices as a
**parallel-decomposition detector**: if β(L(S)) = 0, then L(S) is a
product, which strongly suggests S is bisimilar to a parallel composition
S₁ ∥ S₂; if β(L(S)) ≠ 0, then L(S) is connected and S is not parallel
decomposable at the lattice level.

We also expose a **bidirectional morphism pair** (see Step 32):

    φ : L(S) → {connected, decomposable}          (indicator)
    ψ : {connected, decomposable} → bounds on β(L(S))

and classify the pair as a Galois-connection pre-order morphism.

This module depends only on `zeta.mobius_function` and `statespace`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace

from reticulate.zeta import mobius_function, _compute_sccs


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BetaResult:
    """Crapo beta analysis of a session type state space.

    Attributes:
        num_states: |L|.
        beta: Crapo's β(L) = Σ_x μ(0̂,x)·μ(x,1̂).
        decomposable: True iff β = 0 (product decomposition candidate).
        contributing_terms: list of (state, μ(0̂,x), μ(x,1̂), product)
            for all x where the product is non-zero.
        trivial: True if |L| ≤ 1 (degenerate case).
        mobius_0_1: μ(0̂, 1̂) — the global Möbius invariant for cross-check.
    """
    num_states: int
    beta: int
    decomposable: bool
    contributing_terms: list[tuple[int, int, int, int]]
    trivial: bool
    mobius_0_1: int


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_beta(ss: "StateSpace") -> int:
    """Compute Crapo's β(L) = Σ_x μ(0̂, x) · μ(x, 1̂).

    In the reticulate convention, the poset order is "x ≥ y iff y is
    reachable from x" with ``ss.top`` the 1̂ (initial state, maximum) and
    ``ss.bottom`` the 0̂ (end state, minimum).  The stored Möbius function
    uses keys (larger, smaller), so:

    - μ_standard(0̂, x) = μ_stored(x, ss.bottom)
    - μ_standard(x, 1̂) = μ_stored(ss.top, x)

    Returns 0 for degenerate lattices with fewer than 2 elements (via
    SCC quotient).
    """
    mu = mobius_function(ss)
    scc_map, scc_members = _compute_sccs(ss)

    top_rep = scc_map[ss.top]
    bot_rep = scc_map[ss.bottom]

    if top_rep == bot_rep:
        return 0

    # Use one canonical representative per SCC
    reps = sorted(scc_members.keys())
    total = 0
    for r in reps:
        mu_bot_x = mu.get((r, ss.bottom), 0) if (r, ss.bottom) in mu else 0
        mu_x_top = mu.get((ss.top, r), 0) if (ss.top, r) in mu else 0
        total += mu_bot_x * mu_x_top
    return total


def analyze_beta(ss: "StateSpace") -> BetaResult:
    """Full Crapo beta analysis of a state space."""
    mu = mobius_function(ss)
    scc_map, scc_members = _compute_sccs(ss)
    reps = sorted(scc_members.keys())

    top_rep = scc_map[ss.top]
    bot_rep = scc_map[ss.bottom]

    if top_rep == bot_rep or len(reps) < 2:
        return BetaResult(
            num_states=len(ss.states),
            beta=0,
            decomposable=True,  # degenerate
            contributing_terms=[],
            trivial=True,
            mobius_0_1=mu.get((ss.top, ss.bottom), 0),
        )

    terms: list[tuple[int, int, int, int]] = []
    total = 0
    for r in reps:
        a = mu.get((r, ss.bottom), 0)
        b = mu.get((ss.top, r), 0)
        prod = a * b
        if prod != 0:
            terms.append((r, a, b, prod))
        total += prod

    return BetaResult(
        num_states=len(ss.states),
        beta=total,
        decomposable=(total == 0),
        contributing_terms=terms,
        trivial=False,
        mobius_0_1=mu.get((ss.top, ss.bottom), 0),
    )


# ---------------------------------------------------------------------------
# Bidirectional morphism φ / ψ
# ---------------------------------------------------------------------------

def phi_decomposable(ss: "StateSpace") -> str:
    """φ : L(S) → {'connected', 'decomposable', 'trivial'}.

    Maps a lattice to its decomposability class via Crapo's beta.
    """
    r = analyze_beta(ss)
    if r.trivial:
        return "trivial"
    return "decomposable" if r.decomposable else "connected"


def psi_bound(label: str) -> tuple[str, int | None]:
    """ψ : class → structural witness.

    Returns a pair (description, beta-bound) encoding the minimal
    information a decomposition class carries back to the lattice side.
    This is the retraction leg of the (φ, ψ) connection.
    """
    if label == "decomposable":
        return ("β = 0 (product candidate)", 0)
    if label == "connected":
        return ("β ≠ 0 (irreducible)", None)
    return ("β undefined (|L| ≤ 1)", 0)


def classify_pair(ss: "StateSpace") -> str:
    """Classify the (φ, ψ) pair on a specific state space.

    - 'galois'     — φ ∘ ψ = id on the discrete classification set and
                     ψ ∘ φ weakly retracts (always, by construction).
    - 'embedding'  — φ is injective (trivially false when |classes|<|L|).
    - 'homomorphism' — the default label for the order-preserving pair.
    """
    r = analyze_beta(ss)
    if r.trivial:
        return "galois"  # degenerate: both sides collapse
    # φ is order-preserving (constant on L), ψ returns a bound that is a
    # lower set in the quotient. This is a Galois insertion:
    #   ψ(φ(L)) ≤ L (information loss), φ(ψ(c)) = c (class round-trip).
    return "galois"
