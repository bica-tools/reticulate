"""Morphism hierarchy between session type state spaces.

Implements the classification of structure-preserving maps between state
spaces (labeled transition systems), ordered by strength:

    isomorphism ⊂ embedding ⊂ projection ⊂ homomorphism

Definitions (ordering: s₁ ≥ s₂ iff s₂ ∈ reachable_from(s₁)):

- **Order-preserving** (homomorphism): f: S→T preserves ≥.
  For all s₁, s₂ in S: s₂ ∈ reach(s₁) ⟹ f(s₂) ∈ reach(f(s₁)).
- **Order-reflecting**: f: S→T reflects ≥.
  For all s₁, s₂ in S: f(s₂) ∈ reach(f(s₁)) ⟹ s₂ ∈ reach(s₁).
- **Embedding**: order-preserving + injective + order-reflecting.
- **Projection**: order-preserving + surjective.
- **Isomorphism**: order-preserving + injective + surjective + order-reflecting
  (equivalently: embedding + surjective, or projection + injective + reflecting).

Also provides Galois connection checking: α ∘ γ and γ ∘ α satisfy the
adjunction condition α(x) ≤ y ⟺ x ≤ γ(y).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Morphism:
    """A classified morphism between two state spaces.

    Attributes:
        source: The domain state space.
        target: The codomain state space.
        mapping: Dict mapping source state IDs to target state IDs.
        kind: One of ``"isomorphism"``, ``"embedding"``, ``"projection"``,
              ``"homomorphism"``.
    """

    source: StateSpace
    target: StateSpace
    mapping: dict[int, int]
    kind: str


@dataclass(frozen=True)
class GaloisConnection:
    """A Galois connection between two state spaces.

    Attributes:
        source: The "concrete" domain.
        target: The "abstract" codomain.
        alpha: Abstraction map (source → target).
        gamma: Concretization map (target → source).
    """

    source: StateSpace
    target: StateSpace
    alpha: dict[int, int]
    gamma: dict[int, int]


# ---------------------------------------------------------------------------
# Internal: reachability computation
# ---------------------------------------------------------------------------

def _reachability(ss: StateSpace) -> dict[int, set[int]]:
    """Compute forward reachability for every state (inclusive)."""
    return {s: ss.reachable_from(s) for s in ss.states}


# ---------------------------------------------------------------------------
# Public API: order properties
# ---------------------------------------------------------------------------

def is_order_preserving(
    source: StateSpace,
    target: StateSpace,
    mapping: dict[int, int],
) -> bool:
    """Check whether *mapping* preserves order.

    For all s₁, s₂ in source: if s₂ ∈ reach(s₁),
    then mapping[s₂] ∈ reach(mapping[s₁]) in target.
    """
    src_reach = _reachability(source)
    tgt_reach = _reachability(target)

    for s1 in source.states:
        for s2 in src_reach[s1]:
            if mapping[s2] not in tgt_reach[mapping[s1]]:
                return False
    return True


def is_order_reflecting(
    source: StateSpace,
    target: StateSpace,
    mapping: dict[int, int],
) -> bool:
    """Check whether *mapping* reflects order.

    For all s₁, s₂ in source: if mapping[s₂] ∈ reach(mapping[s₁]),
    then s₂ ∈ reach(s₁).
    """
    src_reach = _reachability(source)
    tgt_reach = _reachability(target)

    for s1 in source.states:
        for s2 in source.states:
            if mapping[s2] in tgt_reach[mapping[s1]] and s2 not in src_reach[s1]:
                return False
    return True


# ---------------------------------------------------------------------------
# Public API: morphism classification
# ---------------------------------------------------------------------------

def classify_morphism(
    source: StateSpace,
    target: StateSpace,
    mapping: dict[int, int],
) -> Morphism:
    """Classify a mapping between state spaces.

    Raises ``ValueError`` if the mapping is not order-preserving
    (minimum requirement for a homomorphism).
    """
    if not is_order_preserving(source, target, mapping):
        raise ValueError("mapping is not order-preserving (not a homomorphism)")

    injective = len(set(mapping.values())) == len(mapping)
    surjective = set(mapping.values()) >= target.states
    reflecting = is_order_reflecting(source, target, mapping)

    if injective and surjective and reflecting:
        kind = "isomorphism"
    elif injective and reflecting:
        kind = "embedding"
    elif surjective:
        kind = "projection"
    else:
        kind = "homomorphism"

    return Morphism(source=source, target=target, mapping=mapping, kind=kind)


# ---------------------------------------------------------------------------
# Public API: find isomorphism
# ---------------------------------------------------------------------------

def find_isomorphism(ss1: StateSpace, ss2: StateSpace) -> Morphism | None:
    """Search for an isomorphism between *ss1* and *ss2*.

    Returns a ``Morphism`` with kind ``"isomorphism"`` if one exists,
    or ``None`` if the state spaces are not isomorphic.

    Uses backtracking with pruning:
    1. Quick rejection on state/transition counts.
    2. Degree-based pruning (out-degree, in-degree, reachability-set size).
    3. Fix top→top, bottom→bottom.
    4. Backtrack over remaining states.
    """
    # Quick rejection
    if len(ss1.states) != len(ss2.states):
        return None
    if len(ss1.transitions) != len(ss2.transitions):
        return None

    # Compute signatures for pruning
    sig1 = _state_signatures(ss1)
    sig2 = _state_signatures(ss2)

    # Check that signature multisets match
    sigs1_sorted = sorted(sig1[s] for s in ss1.states)
    sigs2_sorted = sorted(sig2[s] for s in ss2.states)
    if sigs1_sorted != sigs2_sorted:
        return None

    # Group target states by signature for faster matching
    sig_to_s2: dict[tuple[int, int, int], list[int]] = {}
    for s in ss2.states:
        sig = sig2[s]
        sig_to_s2.setdefault(sig, []).append(s)

    # Build adjacency for constraint checking
    reach1 = _reachability(ss1)
    reach2 = _reachability(ss2)

    # Fixed mappings: top→top, bottom→bottom
    mapping: dict[int, int] = {}
    used: set[int] = set()

    # Fix top
    mapping[ss1.top] = ss2.top
    used.add(ss2.top)

    # Fix bottom (if different from top)
    if ss1.bottom != ss1.top:
        if ss2.bottom in used:
            return None
        mapping[ss1.bottom] = ss2.bottom
        used.add(ss2.bottom)

    # Check signature compatibility for fixed states
    if sig1[ss1.top] != sig2[ss2.top]:
        return None
    if ss1.bottom != ss1.top and sig1[ss1.bottom] != sig2[ss2.bottom]:
        return None

    # Remaining states to assign
    remaining = sorted(s for s in ss1.states if s not in mapping)

    def backtrack(idx: int) -> bool:
        if idx == len(remaining):
            # Verify order-reflecting (preserving is checked incrementally)
            return _check_iso_complete(mapping, reach1, reach2, ss1)

        s1 = remaining[idx]
        sig = sig1[s1]
        candidates = [s for s in sig_to_s2.get(sig, []) if s not in used]

        for s2 in candidates:
            # Check order compatibility with already-assigned states
            if _compatible(s1, s2, mapping, reach1, reach2):
                mapping[s1] = s2
                used.add(s2)
                if backtrack(idx + 1):
                    return True
                del mapping[s1]
                used.discard(s2)

        return False

    if backtrack(0):
        return Morphism(
            source=ss1, target=ss2,
            mapping=dict(mapping), kind="isomorphism",
        )
    return None


# ---------------------------------------------------------------------------
# Public API: find embedding
# ---------------------------------------------------------------------------

def find_embedding(ss1: StateSpace, ss2: StateSpace) -> Morphism | None:
    """Search for an order-embedding of *ss1* into *ss2*.

    Returns a ``Morphism`` with kind ``"embedding"`` (or ``"isomorphism"``
    if surjective), or ``None`` if no embedding exists.

    Requires |ss1| ≤ |ss2|.
    """
    if len(ss1.states) > len(ss2.states):
        return None

    sig1 = _state_signatures(ss1)
    reach1 = _reachability(ss1)
    reach2 = _reachability(ss2)

    # Fixed: top→top, bottom→bottom
    mapping: dict[int, int] = {}
    used: set[int] = set()

    mapping[ss1.top] = ss2.top
    used.add(ss2.top)

    if ss1.bottom != ss1.top:
        if ss2.bottom in used:
            # bottom and top must be distinct in target too
            return None
        mapping[ss1.bottom] = ss2.bottom
        used.add(ss2.bottom)

    # Check top/bottom reachability compatibility
    if ss1.bottom not in reach1[ss1.top] and ss2.bottom in reach2[ss2.top]:
        pass  # OK, embedding only needs to preserve, which is vacuously true
    if ss1.bottom in reach1[ss1.top] and ss2.bottom not in reach2[ss2.top]:
        return None

    remaining = sorted(s for s in ss1.states if s not in mapping)

    # Candidate targets: must have compatible reachability-set size ≥ source
    s2_candidates = sorted(ss2.states)

    def backtrack(idx: int) -> bool:
        if idx == len(remaining):
            # Verify full order-preserving and order-reflecting
            return (
                _check_preserving(mapping, reach1, reach2, ss1)
                and _check_reflecting(mapping, reach1, reach2, ss1)
            )

        s1 = remaining[idx]
        for s2 in s2_candidates:
            if s2 in used:
                continue
            if _compatible(s1, s2, mapping, reach1, reach2):
                mapping[s1] = s2
                used.add(s2)
                if backtrack(idx + 1):
                    return True
                del mapping[s1]
                used.discard(s2)

        return False

    if backtrack(0):
        surjective = set(mapping.values()) >= ss2.states
        kind = "isomorphism" if surjective else "embedding"
        return Morphism(
            source=ss1, target=ss2,
            mapping=dict(mapping), kind=kind,
        )
    return None


# ---------------------------------------------------------------------------
# Public API: Galois connection
# ---------------------------------------------------------------------------

def is_galois_connection(
    alpha: dict[int, int],
    gamma: dict[int, int],
    source: StateSpace,
    target: StateSpace,
) -> bool:
    """Check whether (α, γ) forms a Galois connection.

    Verifies: α(x) ≤ y ⟺ x ≤ γ(y) for all x in source, y in target.

    In our ordering (≤ means "reachable from"): s₁ ≤ s₂ iff s₁ ∈ reach(s₂).
    So α(x) ≤ y means α(x) ∈ reach(y) in target.
    And x ≤ γ(y) means x ∈ reach(γ(y)) in source.
    """
    src_reach = _reachability(source)
    tgt_reach = _reachability(target)

    for x in source.states:
        for y in target.states:
            alpha_x_leq_y = alpha[x] in tgt_reach[y]
            x_leq_gamma_y = x in src_reach[gamma[y]]
            if alpha_x_leq_y != x_leq_gamma_y:
                return False
    return True


# ---------------------------------------------------------------------------
# Internal: state signatures for pruning
# ---------------------------------------------------------------------------

def _state_signatures(ss: StateSpace) -> dict[int, tuple[int, int, int]]:
    """Compute (out_degree, in_degree, reachability_set_size) for each state."""
    out_deg: dict[int, int] = {s: 0 for s in ss.states}
    in_deg: dict[int, int] = {s: 0 for s in ss.states}
    for src, _, tgt in ss.transitions:
        out_deg[src] += 1
        in_deg[tgt] += 1

    reach = _reachability(ss)
    return {
        s: (out_deg[s], in_deg[s], len(reach[s]))
        for s in ss.states
    }


# ---------------------------------------------------------------------------
# Internal: backtracking helpers
# ---------------------------------------------------------------------------

def _compatible(
    s1: int,
    s2: int,
    partial: dict[int, int],
    reach1: dict[int, set[int]],
    reach2: dict[int, set[int]],
) -> bool:
    """Check that assigning s1→s2 is compatible with the partial mapping.

    For every already-assigned pair (a, partial[a]):
    - If a ∈ reach(s1), then partial[a] ∈ reach(s2)  (preserving)
    - If s1 ∈ reach(a), then s2 ∈ reach(partial[a])  (preserving)
    - If partial[a] ∈ reach(s2), then a ∈ reach(s1)  (reflecting)
    - If s2 ∈ reach(partial[a]), then s1 ∈ reach(a)  (reflecting)
    """
    for a, fa in partial.items():
        # Preserving: s1 ≥ a ⟹ s2 ≥ fa
        if a in reach1[s1] and fa not in reach2[s2]:
            return False
        # Preserving: a ≥ s1 ⟹ fa ≥ s2
        if s1 in reach1[a] and s2 not in reach2[fa]:
            return False
        # Reflecting: s2 ≥ fa ⟹ s1 ≥ a
        if fa in reach2[s2] and a not in reach1[s1]:
            return False
        # Reflecting: fa ≥ s2 ⟹ a ≥ s1
        if s2 in reach2[fa] and s1 not in reach1[a]:
            return False
    return True


def _check_iso_complete(
    mapping: dict[int, int],
    reach1: dict[int, set[int]],
    reach2: dict[int, set[int]],
    ss1: StateSpace,
) -> bool:
    """Full verification that a complete mapping is an isomorphism."""
    return (
        _check_preserving(mapping, reach1, reach2, ss1)
        and _check_reflecting(mapping, reach1, reach2, ss1)
    )


def _check_preserving(
    mapping: dict[int, int],
    reach1: dict[int, set[int]],
    reach2: dict[int, set[int]],
    ss1: StateSpace,
) -> bool:
    """Check order-preserving for a complete mapping."""
    for s1 in ss1.states:
        for s2 in reach1[s1]:
            if mapping[s2] not in reach2[mapping[s1]]:
                return False
    return True


def _check_reflecting(
    mapping: dict[int, int],
    reach1: dict[int, set[int]],
    reach2: dict[int, set[int]],
    ss1: StateSpace,
) -> bool:
    """Check order-reflecting for a complete mapping."""
    for s1 in ss1.states:
        for s2 in ss1.states:
            if mapping[s2] in reach2[mapping[s1]] and s2 not in reach1[s1]:
                return False
    return True
