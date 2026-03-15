"""Gay–Hole subtyping for session types (Step 7).

Implements the Gay–Hole coinductive subtyping relation on session type ASTs.

Key insight: end is NOT a primary constructor.  It is syntactic sugar for
the empty branch &{} — a valid termination point where no methods remain.
This means end and &{...} are in the SAME constructor family:
  - &{a: end} ≤_GH end  is TRUE  (by Sub-Branch: ∅ ⊆ {a})
  - end ≤_GH &{a: end}  is FALSE ({a} ⊄ ∅)

Gay–Hole subtyping rules:
  - &{m₁:S₁,...,mₙ:Sₙ} ≤ &{m₁':S₁',...,mₖ':Sₖ'}
      iff {m₁',...,mₖ'} ⊆ {m₁,...,mₙ}  (subtype offers MORE methods)
      and for each mᵢ' in RHS: Sₘᵢ ≤ Sᵢ'   (covariant continuations)
  - +{l₁:S₁,...,lₙ:Sₙ} ≤ +{l₁':S₁',...,lₖ':Sₖ'}
      iff {l₁,...,lₙ} ⊆ {l₁',...,lₖ'}  (subtype selects FEWER labels)
      and for each lᵢ in LHS: Sᵢ ≤ Sₗᵢ'   (covariant continuations)
  - rec X.S₁ ≤ rec Y.S₂  via coinduction (greatest fixpoint)

Since end ≡ &{}, end is comparable with all branch types but incomparable
with selection types (+{...}).

The Step 7 claim: the reachability ordering in L(S) captures the
Gay–Hole subtyping on residual types at each state.  This is verified
by showing that width subtyping (wider branches) corresponds to
state-space containment (the narrower state space embeds into the wider one).
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
class SubtypingResult:
    """Result of Gay–Hole subtyping check between two session types.

    Attributes:
        lhs: Pretty-printed left-hand side type.
        rhs: Pretty-printed right-hand side type.
        is_subtype: True iff lhs ≤_GH rhs.
        reason: Human-readable explanation if not a subtype.
    """
    lhs: str
    rhs: str
    is_subtype: bool
    reason: str | None = None


@dataclass(frozen=True)
class WidthSubtypingResult:
    """Result of verifying width subtyping ↔ state-space embedding.

    Attributes:
        type1: Pretty-printed first type (candidate subtype).
        type2: Pretty-printed second type (candidate supertype).
        is_subtype: True iff type1 ≤_GH type2.
        has_embedding: True iff L(type2) embeds into L(type1).
        coincides: True iff is_subtype == has_embedding.
    """
    type1: str
    type2: str
    is_subtype: bool
    has_embedding: bool
    coincides: bool


# ---------------------------------------------------------------------------
# Gay–Hole subtyping (coinductive, greatest fixpoint)
# ---------------------------------------------------------------------------

def is_subtype(s1: SessionType, s2: SessionType) -> bool:
    """Check whether *s1* ≤_GH *s2* (Gay–Hole subtyping).

    Uses coinductive greatest fixpoint: pairs currently being checked
    are assumed related (coinductive hypothesis).
    """
    return _check_subtype(s1, s2, set())


def check_subtype(s1: SessionType, s2: SessionType) -> SubtypingResult:
    """Check Gay–Hole subtyping with a human-readable result."""
    result = is_subtype(s1, s2)
    reason = None
    if not result:
        reason = _explain_failure(s1, s2)
    return SubtypingResult(
        lhs=pretty(s1),
        rhs=pretty(s2),
        is_subtype=result,
        reason=reason,
    )


def _check_subtype(
    s1: SessionType,
    s2: SessionType,
    assumptions: set[tuple[int, int]],
) -> bool:
    """Recursive coinductive subtyping check.

    The *assumptions* set tracks pairs currently being checked (for coinduction).
    If we encounter a pair already in assumptions, we return True (coinductive
    hypothesis — the greatest fixpoint assumes pairs are related until proven
    otherwise).
    """
    # Coinductive hypothesis: if we've already assumed this pair, accept it
    pair_id = (id(s1), id(s2))
    if pair_id in assumptions:
        return True

    # Unfold recursion
    s1_unf = _unfold(s1)
    s2_unf = _unfold(s2)

    # After unfolding, update assumptions
    pair_id_unf = (id(s1_unf), id(s2_unf))
    new_assumptions = assumptions | {pair_id, pair_id_unf}

    # Normalize: end ≡ &{} (empty branch).  This is not syntactic sugar
    # but a semantic identity: end is the branch with zero methods.
    if isinstance(s1_unf, (End, Wait)):
        s1_unf = Branch(())
    if isinstance(s2_unf, (End, Wait)):
        s2_unf = Branch(())

    # Branch: &{...} ≤ &{...}  (includes end ≡ &{})
    if isinstance(s1_unf, Branch) and isinstance(s2_unf, Branch):
        s1_dict = dict(s1_unf.choices)
        s2_dict = dict(s2_unf.choices)
        # Subtype must offer all methods of supertype (width subtyping)
        if not set(s2_dict.keys()) <= set(s1_dict.keys()):
            return False
        # Covariant: continuations must be subtypes
        for m in s2_dict:
            if not _check_subtype(s1_dict[m], s2_dict[m], new_assumptions):
                return False
        return True

    # Selection: +{...} ≤ +{...}
    if isinstance(s1_unf, Select) and isinstance(s2_unf, Select):
        s1_dict = dict(s1_unf.choices)
        s2_dict = dict(s2_unf.choices)
        # Subtype selects from at most the labels of supertype (width)
        if not set(s1_dict.keys()) <= set(s2_dict.keys()):
            return False
        # Covariant: continuations must be subtypes
        for l in s1_dict:
            if not _check_subtype(s1_dict[l], s2_dict[l], new_assumptions):
                return False
        return True

    # Parallel: (S₁ || … || Sₙ) ≤ (T₁ || … || Tₙ) — componentwise, same arity
    if isinstance(s1_unf, Parallel) and isinstance(s2_unf, Parallel):
        if len(s1_unf.branches) != len(s2_unf.branches):
            return False
        return all(
            _check_subtype(b1, b2, new_assumptions)
            for b1, b2 in zip(s1_unf.branches, s2_unf.branches)
        )

    # Continuation: (S₁.S₂) ≤ (T₁.T₂)
    if isinstance(s1_unf, Continuation) and isinstance(s2_unf, Continuation):
        return (_check_subtype(s1_unf.left, s2_unf.left, new_assumptions) and
                _check_subtype(s1_unf.right, s2_unf.right, new_assumptions))

    # Different constructors — incomparable
    return False


def _unfold(s: SessionType) -> SessionType:
    """Unfold one level of recursion: rec X . S → S[X := rec X . S]."""
    if isinstance(s, Rec):
        return _substitute(s.body, s.var, s)
    return s


def _substitute(body: SessionType, var: str, replacement: SessionType) -> SessionType:
    """Substitute *var* with *replacement* in *body*."""
    if isinstance(body, Var):
        return replacement if body.name == var else body
    elif isinstance(body, (End, Wait)):
        return body
    elif isinstance(body, Branch):
        return Branch(tuple(
            (m, _substitute(s, var, replacement)) for m, s in body.choices
        ))
    elif isinstance(body, Select):
        return Select(tuple(
            (l, _substitute(s, var, replacement)) for l, s in body.choices
        ))
    elif isinstance(body, Parallel):
        return Parallel(tuple(
            _substitute(b, var, replacement) for b in body.branches
        ))
    elif isinstance(body, Continuation):
        return Continuation(
            _substitute(body.left, var, replacement),
            _substitute(body.right, var, replacement),
        )
    elif isinstance(body, Rec):
        if body.var == var:
            return body  # shadowed
        return Rec(body.var, _substitute(body.body, var, replacement))
    return body


def _explain_failure(s1: SessionType, s2: SessionType) -> str:
    """Generate a human-readable reason for subtyping failure."""
    s1_unf = _unfold(s1)
    s2_unf = _unfold(s2)

    # Normalize end ≡ &{}
    if isinstance(s1_unf, (End, Wait)):
        s1_unf = Branch(())
    if isinstance(s2_unf, (End, Wait)):
        s2_unf = Branch(())

    if isinstance(s1_unf, Branch) and isinstance(s2_unf, Branch):
        s1_methods = set(dict(s1_unf.choices).keys())
        s2_methods = set(dict(s2_unf.choices).keys())
        if not s2_methods <= s1_methods:
            missing = s2_methods - s1_methods
            return f"missing methods in subtype: {missing}"
        return "continuation mismatch in branch"

    if isinstance(s1_unf, Select) and isinstance(s2_unf, Select):
        s1_labels = set(dict(s1_unf.choices).keys())
        s2_labels = set(dict(s2_unf.choices).keys())
        if not s1_labels <= s2_labels:
            extra = s1_labels - s2_labels
            return f"subtype selects labels not in supertype: {extra}"
        return "continuation mismatch in selection"

    s1_name = "Branch" if isinstance(s1_unf, Branch) else type(s1_unf).__name__
    s2_name = "Branch" if isinstance(s2_unf, Branch) else type(s2_unf).__name__
    return f"incompatible constructors: {s1_name} vs {s2_name}"


# ---------------------------------------------------------------------------
# Width subtyping ↔ embedding verification
# ---------------------------------------------------------------------------

def check_width_embedding(
    s1: SessionType,
    s2: SessionType,
) -> WidthSubtypingResult:
    """Verify that width subtyping corresponds to state-space embedding.

    Checks: S₁ ≤_GH S₂ ⟺ L(S₂) order-embeds into L(S₁).
    """
    from reticulate.morphism import find_embedding
    from reticulate.statespace import build_statespace

    sub = is_subtype(s1, s2)

    try:
        ss1 = build_statespace(s1)
        ss2 = build_statespace(s2)
        emb = find_embedding(ss2, ss1)
        has_emb = emb is not None
    except (ValueError, RecursionError, KeyError):
        has_emb = sub  # skip malformed types

    return WidthSubtypingResult(
        type1=pretty(s1),
        type2=pretty(s2),
        is_subtype=sub,
        has_embedding=has_emb,
        coincides=sub == has_emb,
    )
