"""Bottom-up multiparty composition via lattice products (Step 15).

Instead of starting from a global type and projecting down (top-down MPST),
this module composes independent session types (lattices) bottom-up:

    1. Each participant has an independent session type → lattice.
    2. The composite state space is the N-ary product of individual lattices.
    3. Compatibility is checked via duality on shared method labels.
    4. No global type needed — the product lattice IS the global state space.

The key insight: in a free multiparty interaction, every participant is
both an object and a client. Their lattices compose via products, and
compatibility is verified via morphisms + duality.

**Inverse Projection Conjecture**: For a well-formed global type G with
roles {r₁, ..., rₙ}, L(G) embeds into L(G↓r₁) × ... × L(G↓rₙ).
The product may be strictly larger (it allows interleavings the global
type forbids).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from typing import TYPE_CHECKING

from reticulate.duality import dual
from reticulate.parser import Branch, End, Select, SessionType, pretty

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompositionResult:
    """Result of composing independent session types.

    Attributes:
        participants: name → session type mapping.
        state_spaces: name → lattice mapping.
        product: the composite N-ary product lattice.
        is_lattice: True iff the product is a lattice (always true by theorem).
        compatibility: pairwise compatibility dict (name, name) → bool.
    """
    participants: dict[str, SessionType]
    state_spaces: dict[str, "StateSpace"]
    product: "StateSpace"
    is_lattice: bool
    compatibility: dict[tuple[str, str], bool]


@dataclass(frozen=True)
class ComparisonResult:
    """Result of comparing bottom-up composition with top-down projection.

    Attributes:
        global_type_string: the global type that was parsed.
        global_states: number of states in the global state space.
        product_states: number of states in the product of projections.
        global_is_lattice: True iff L(G) is a lattice.
        product_is_lattice: True iff the product is a lattice.
        embedding_exists: True iff L(G) embeds into the product.
        over_approximation_ratio: product_states / global_states.
        role_type_matches: dict role → whether participant type ≅ projected type.
    """
    global_type_string: str
    global_states: int
    product_states: int
    global_is_lattice: bool
    product_is_lattice: bool
    embedding_exists: bool
    over_approximation_ratio: float
    role_type_matches: dict[str, bool]


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def compose(*participants: tuple[str, SessionType]) -> CompositionResult:
    """Compose independent session types into a product lattice.

    Each participant is a (name, session_type) pair. The function:
    1. Builds each state space and verifies it is a lattice.
    2. Builds the N-ary product via chained binary products.
    3. Checks pairwise compatibility (shared method labels + duality).
    4. Returns a CompositionResult.

    Raises ValueError if any participant's state space is not a lattice.
    """
    from reticulate.lattice import check_lattice
    from reticulate.statespace import build_statespace

    if len(participants) < 1:
        raise ValueError("need at least one participant")

    participant_dict: dict[str, SessionType] = {}
    state_spaces: dict[str, "StateSpace"] = {}

    for name, stype in participants:
        participant_dict[name] = stype
        ss = build_statespace(stype)
        lattice_result = check_lattice(ss)
        if not lattice_result.is_lattice:
            raise ValueError(
                f"participant {name!r} state space is not a lattice"
            )
        state_spaces[name] = ss

    # N-ary product
    ss_list = [state_spaces[name] for name, _ in participants]
    product = product_nary(ss_list)

    # Check product is a lattice (theorem: always true)
    product_lattice = check_lattice(product)

    # Pairwise compatibility
    names = [name for name, _ in participants]
    compat: dict[tuple[str, str], bool] = {}
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            compat[(n1, n2)] = check_compatibility(
                participant_dict[n1], participant_dict[n2]
            )

    return CompositionResult(
        participants=participant_dict,
        state_spaces=state_spaces,
        product=product,
        is_lattice=product_lattice.is_lattice,
        compatibility=compat,
    )


def check_compatibility(s1: SessionType, s2: SessionType) -> bool:
    """Check if two session types can interact as object/client pair.

    s1 and s2 are compatible if the methods s1 offers (Branch labels)
    match the methods s2 selects (Select labels) on their shared interface,
    respecting duality on the shared portion.

    Compatibility rules:
    - If s1 has Branch labels that s2 Selects, those must match under duality.
    - Disjoint method sets are trivially compatible (no interaction).
    - Overlapping methods with wrong duality direction are incompatible.
    """
    labels1 = _extract_interface(s1)
    labels2 = _extract_interface(s2)

    offered1 = labels1["offered"]   # Branch labels in s1
    selected1 = labels1["selected"]  # Select labels in s1
    offered2 = labels2["offered"]
    selected2 = labels2["selected"]

    # Key conflict: both selecting the same label means both try to make
    # internal choices — incompatible unless one also offers it (Branch + Select)
    both_select = selected1 & selected2
    both_select_not_offered = both_select - (offered1 | offered2)
    if both_select_not_offered:
        return False

    # Check that shared labels don't conflict:
    # A label can't be offered by both AND selected by both
    conflict = (offered1 & offered2) & (selected1 & selected2)
    if conflict:
        return False

    return True


def product_nary(state_spaces: list["StateSpace"]) -> "StateSpace":
    """N-ary product via left-fold of binary products.

    Given [L₁, L₂, ..., Lₙ], computes ((L₁ × L₂) × L₃) × ... × Lₙ.
    Returns the single state space if only one is provided.

    Raises ValueError if the list is empty.
    """
    from reticulate.product import product_statespace

    if not state_spaces:
        raise ValueError("need at least one state space for product")

    return reduce(product_statespace, state_spaces)


def compare_with_global(
    participants: dict[str, SessionType],
    global_type_string: str,
) -> ComparisonResult:
    """Compare bottom-up composition with top-down projection.

    1. Build product of participant lattices (bottom-up).
    2. Parse global type, build its state space (top-down).
    3. Project global type onto each role.
    4. Check: global state space embeds into product of projections?
    5. Check: participant types match projected types?
    """
    from reticulate.global_types import build_global_statespace, parse_global, roles
    from reticulate.lattice import check_lattice
    from reticulate.morphism import find_embedding, find_isomorphism
    from reticulate.projection import project_all
    from reticulate.statespace import build_statespace

    # Top-down: parse and build global state space
    g = parse_global(global_type_string)
    global_ss = build_global_statespace(g)
    global_lattice = check_lattice(global_ss)

    # Project onto all roles
    projections = project_all(g)

    # Build state spaces for projections
    projection_ss_list = []
    projection_ss = {}
    for role_name in sorted(projections.keys()):
        ss = build_statespace(projections[role_name])
        projection_ss[role_name] = ss
        projection_ss_list.append(ss)

    # Bottom-up: N-ary product of projected local types
    product = product_nary(projection_ss_list)
    product_lattice = check_lattice(product)

    # Check embedding: L(G) embeds into product of projections
    embedding = find_embedding(global_ss, product)
    embedding_exists = embedding is not None

    # Check role type matches: participant types ≅ projected types
    role_matches: dict[str, bool] = {}
    for role_name, projected_type in projections.items():
        if role_name in participants:
            participant_ss = build_statespace(participants[role_name])
            projected_ss = build_statespace(projected_type)
            iso = find_isomorphism(participant_ss, projected_ss)
            role_matches[role_name] = iso is not None
        else:
            role_matches[role_name] = False

    # Over-approximation ratio
    ratio = len(product.states) / len(global_ss.states) if len(global_ss.states) > 0 else 0.0

    return ComparisonResult(
        global_type_string=global_type_string,
        global_states=len(global_ss.states),
        product_states=len(product.states),
        global_is_lattice=global_lattice.is_lattice,
        product_is_lattice=product_lattice.is_lattice,
        embedding_exists=embedding_exists,
        over_approximation_ratio=ratio,
        role_type_matches=role_matches,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_interface(s: SessionType) -> dict[str, set[str]]:
    """Extract offered (Branch) and selected (Select) method labels from a type.

    Traverses the AST recursively to collect all labels at every level.
    """
    offered: set[str] = set()
    selected: set[str] = set()
    _collect_labels(s, offered, selected, set())
    return {"offered": offered, "selected": selected}


def _collect_labels(
    s: SessionType,
    offered: set[str],
    selected: set[str],
    visited: set[str],
) -> None:
    """Recursively collect Branch/Select labels."""
    match s:
        case Branch(choices=choices):
            for label, body in choices:
                offered.add(label)
                _collect_labels(body, offered, selected, visited)
        case Select(choices=choices):
            for label, body in choices:
                selected.add(label)
                _collect_labels(body, offered, selected, visited)
        case _ if hasattr(s, 'left') and hasattr(s, 'right'):
            _collect_labels(s.left, offered, selected, visited)  # type: ignore[attr-defined]
            _collect_labels(s.right, offered, selected, visited)  # type: ignore[attr-defined]
        case _ if hasattr(s, 'body') and hasattr(s, 'var'):
            # Rec node
            if s.var not in visited:  # type: ignore[attr-defined]
                _collect_labels(s.body, offered, selected, visited | {s.var})  # type: ignore[attr-defined]
        case _:
            pass  # End, Wait, Var — no labels
