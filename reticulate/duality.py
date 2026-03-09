"""Session type duality (Step 8).

Implements the dual operation on session type ASTs and verifies the
relationship between L(S) and L(dual(S)).

Duality swaps the locus of control:
  - Branch (external choice by environment) ↔ Selection (internal choice by object)
  - All other constructors are structurally preserved (covariant in sub-terms)

Key properties:
  - Involution: dual(dual(S)) = S  (structurally)
  - Self-dual: dual(end) = end, dual(wait) = wait
  - State-space preservation: L(S) ≅ L(dual(S)) as posets (same reachability)
  - Selection annotation flip: selection_transitions(L(dual(S))) complements
    selection_transitions(L(S))

The duality operation is fundamental in the pi-calculus tradition (Gay–Hole),
where dual types describe complementary channel endpoints.  In our
object-oriented setting, duality swaps the perspective between the object
(which offers methods / makes selections) and the environment (which calls
methods / handles selections).
"""

from __future__ import annotations

from dataclasses import dataclass

from reticulate.parser import (
    Branch,
    Continuation,
    End,
    Parallel,
    Rec,
    Select,
    SessionType,
    Var,
    Wait,
    pretty,
)


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DualityResult:
    """Result of duality verification between S and dual(S).

    Attributes:
        type_str: Pretty-printed original type S.
        dual_str: Pretty-printed dual type dual(S).
        is_involution: True iff dual(dual(S)) == S structurally.
        is_isomorphic: True iff L(S) ≅ L(dual(S)) as posets.
        selection_flipped: True iff selection annotations are complementary.
    """
    type_str: str
    dual_str: str
    is_involution: bool
    is_isomorphic: bool
    selection_flipped: bool


# ---------------------------------------------------------------------------
# The dual operation
# ---------------------------------------------------------------------------

def dual(s: SessionType) -> SessionType:
    """Compute the dual of a session type.

    Swaps Branch ↔ Select (external ↔ internal choice).
    All other constructors are preserved structurally.

    dual(&{m₁:S₁,...,mₙ:Sₙ}) = +{m₁:dual(S₁),...,mₙ:dual(Sₙ)}
    dual(+{l₁:S₁,...,lₙ:Sₙ}) = &{l₁:dual(S₁),...,lₙ:dual(Sₙ)}
    dual(end) = end
    dual(wait) = wait
    dual(X) = X
    dual(rec X . S) = rec X . dual(S)
    dual(S₁ ∥ S₂) = dual(S₁) ∥ dual(S₂)
    dual(S₁ . S₂) = dual(S₁) . dual(S₂)
    """
    if isinstance(s, End):
        return End()
    elif isinstance(s, Wait):
        return Wait()
    elif isinstance(s, Var):
        return Var(s.name)
    elif isinstance(s, Branch):
        return Select(tuple(
            (m, dual(cont)) for m, cont in s.choices
        ))
    elif isinstance(s, Select):
        return Branch(tuple(
            (l, dual(cont)) for l, cont in s.choices
        ))
    elif isinstance(s, Parallel):
        return Parallel(dual(s.left), dual(s.right))
    elif isinstance(s, Continuation):
        return Continuation(dual(s.left), dual(s.right))
    elif isinstance(s, Rec):
        return Rec(s.var, dual(s.body))
    else:
        return s


def is_structurally_equal(s1: SessionType, s2: SessionType) -> bool:
    """Check structural equality of two session type ASTs.

    This is deeper than Python's == on frozen dataclasses, which already
    works for our ASTs.  We provide this as a named function for clarity.
    """
    return s1 == s2


# ---------------------------------------------------------------------------
# Duality verification
# ---------------------------------------------------------------------------

def check_duality(s: SessionType) -> DualityResult:
    """Verify duality properties for a session type.

    Checks:
    1. Involution: dual(dual(S)) == S
    2. State-space isomorphism: L(S) ≅ L(dual(S))
    3. Selection annotation flip: transitions that are selections in L(S)
       are non-selections in L(dual(S)) and vice versa.
    """
    from reticulate.morphism import find_isomorphism
    from reticulate.statespace import build_statespace

    d = dual(s)
    dd = dual(d)

    # 1. Involution check
    involution = is_structurally_equal(s, dd)

    # 2. State-space isomorphism
    try:
        ss_s = build_statespace(s)
        ss_d = build_statespace(d)
        iso = find_isomorphism(ss_s, ss_d)
        is_iso = iso is not None

        # 3. Selection flip check
        if is_iso and iso is not None:
            sel_flipped = _check_selection_flip(ss_s, ss_d, iso.mapping)
        else:
            sel_flipped = False
    except (ValueError, RecursionError, KeyError):
        is_iso = False
        sel_flipped = False

    return DualityResult(
        type_str=pretty(s),
        dual_str=pretty(d),
        is_involution=involution,
        is_isomorphic=is_iso,
        selection_flipped=sel_flipped,
    )


def _check_selection_flip(
    ss_s: "StateSpace",
    ss_d: "StateSpace",
    iso: dict[int, int],
) -> bool:
    """Check that selection annotations are flipped under the isomorphism.

    For each transition (src, label, tgt) in L(S):
    - If it's a selection transition in L(S), the corresponding transition
      in L(dual(S)) should NOT be a selection transition, and vice versa.

    The "corresponding" transition is found via the isomorphism mapping.
    """
    # Build transition lookup for the dual state space
    d_trans: dict[tuple[int, int, str], bool] = {}
    for src, label, tgt in ss_d.transitions:
        is_sel = (src, label, tgt) in ss_d.selection_transitions
        d_trans[(src, label, tgt)] = is_sel

    for src, label, tgt in ss_s.transitions:
        is_sel_s = (src, label, tgt) in ss_s.selection_transitions
        mapped_src = iso[src]
        mapped_tgt = iso[tgt]

        # Find the corresponding transition in the dual
        is_sel_d = d_trans.get((mapped_src, label, mapped_tgt))
        if is_sel_d is None:
            return False  # No corresponding transition

        # Selection should be flipped
        if is_sel_s == is_sel_d:
            return False

    return True
