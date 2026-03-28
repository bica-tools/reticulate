"""Capability negotiation via lattice meet (Step 115a).

When two agents meet, they must compute the "greatest compatible
sub-protocol" — the most specific protocol both can execute.
This is the lattice meet of their session types.

For two session types S₁ and S₂:
    - compatible(S₁, S₂) = True iff their session types share at least
      one common method at the top level
    - negotiate(S₁, S₂) = the session type representing the maximal
      protocol both can follow
    - compatibility_score(S₁, S₂) = fraction of shared methods

The negotiation algorithm:
    1. If S₁ <: S₂, the negotiated type is S₁ (more specific)
    2. If S₂ <: S₁, the negotiated type is S₂
    3. Otherwise, compute the intersection of top-level methods,
       and recursively negotiate the continuations.

This module provides:
    ``negotiate(s1, s2)`` — compute the greatest compatible sub-protocol.
    ``compatible(s1, s2)`` — check if two types share any methods.
    ``compatibility_score(s1, s2)`` — fraction of shared methods.
    ``negotiate_group(types)`` — negotiate among N agents (iterated meet).
"""

from __future__ import annotations

from dataclasses import dataclass

from reticulate.parser import (
    Branch,
    End,
    Rec,
    Select,
    SessionType,
    Var,
    pretty,
)
from reticulate.subtyping import is_subtype


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NegotiationResult:
    """Result of a capability negotiation.

    Attributes:
        compatible: True if the agents share any common methods.
        negotiated_type: The greatest compatible sub-protocol (End if incompatible).
        shared_methods: Methods available to both agents.
        dropped_methods: Methods available to only one agent.
        score: Compatibility score (0.0 = incompatible, 1.0 = identical).
    """
    compatible: bool
    negotiated_type: SessionType
    shared_methods: tuple[str, ...]
    dropped_methods: tuple[str, ...]
    score: float


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def _get_choices(s: SessionType) -> dict[str, SessionType]:
    """Extract top-level method → continuation mapping."""
    if isinstance(s, (Branch, Select)):
        return dict(s.choices)
    if isinstance(s, Rec):
        return _get_choices(s.body)
    return {}


def _make_like(template: SessionType, choices: tuple[tuple[str, SessionType], ...]) -> SessionType:
    """Reconstruct a Branch or Select with new choices."""
    if isinstance(template, Branch):
        return Branch(choices)
    if isinstance(template, Select):
        return Select(choices)
    if isinstance(template, Rec):
        inner = _make_like(template.body, choices)
        return Rec(template.var, inner)
    return End()


def negotiate(s1: SessionType, s2: SessionType) -> SessionType:
    """Compute the greatest compatible sub-protocol of s1 and s2.

    The negotiated type contains only the methods available to both agents,
    with continuations recursively negotiated.

    Returns End if the types are completely incompatible.
    """
    # Base cases
    if isinstance(s1, End) or isinstance(s2, End):
        return End()
    if isinstance(s1, Var) or isinstance(s2, Var):
        return End()

    # Subtype shortcut
    if is_subtype(s1, s2):
        return s1
    if is_subtype(s2, s1):
        return s2

    # Extract choices
    c1 = _get_choices(s1)
    c2 = _get_choices(s2)

    if not c1 or not c2:
        return End()

    # Intersect methods
    shared_labels = set(c1.keys()) & set(c2.keys())
    if not shared_labels:
        return End()

    # Recursively negotiate continuations
    negotiated_choices: list[tuple[str, SessionType]] = []
    for label in sorted(shared_labels):
        cont = negotiate(c1[label], c2[label])
        negotiated_choices.append((label, cont))

    if not negotiated_choices:
        return End()

    # Use the polarity of s1 (caller perspective)
    return _make_like(s1, tuple(negotiated_choices))


def compatible(s1: SessionType, s2: SessionType) -> bool:
    """Check if two session types share any common methods."""
    c1 = _get_choices(s1)
    c2 = _get_choices(s2)
    if not c1 or not c2:
        return isinstance(s1, End) and isinstance(s2, End)
    return bool(set(c1.keys()) & set(c2.keys()))


def compatibility_score(s1: SessionType, s2: SessionType) -> float:
    """Compute compatibility score between two session types.

    Score = |shared methods| / |all methods| (Jaccard index at top level).
    Returns 1.0 for identical types, 0.0 for completely incompatible.
    """
    c1 = _get_choices(s1)
    c2 = _get_choices(s2)

    if not c1 and not c2:
        return 1.0  # Both End
    if not c1 or not c2:
        return 0.0

    shared = set(c1.keys()) & set(c2.keys())
    total = set(c1.keys()) | set(c2.keys())
    return round(len(shared) / len(total), 3) if total else 1.0


def check_negotiation(s1: SessionType, s2: SessionType) -> NegotiationResult:
    """Full negotiation analysis between two session types."""
    c1 = _get_choices(s1)
    c2 = _get_choices(s2)

    shared = sorted(set(c1.keys()) & set(c2.keys()))
    dropped = sorted((set(c1.keys()) | set(c2.keys())) - set(shared))

    neg_type = negotiate(s1, s2)
    is_compat = compatible(s1, s2)
    score = compatibility_score(s1, s2)

    return NegotiationResult(
        compatible=is_compat,
        negotiated_type=neg_type,
        shared_methods=tuple(shared),
        dropped_methods=tuple(dropped),
        score=score,
    )


def negotiate_group(types: list[SessionType]) -> SessionType:
    """Negotiate among N agents by iterated meet.

    negotiate_group([S₁, S₂, S₃]) = negotiate(negotiate(S₁, S₂), S₃)

    The result is the greatest protocol all agents can follow.
    """
    if not types:
        return End()
    result = types[0]
    for t in types[1:]:
        result = negotiate(result, t)
        if isinstance(result, End):
            break
    return result
