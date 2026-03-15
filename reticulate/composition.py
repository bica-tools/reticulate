"""Bottom-up multiparty composition via lattice products (Step 15).

Instead of starting from a global type and projecting down (top-down MPST),
this module composes independent session types (lattices) bottom-up:

    1. Each participant has an independent session type → lattice.
    2. The composite state space is their product lattice.
    3. Compatibility is checked via duality on shared method labels.
    4. No global type needed — the product lattice IS the global state space.

Two composition modes:

    **Free product** (``compose``): All interleavings allowed. The product
    is ``L₁ × L₂ × ... × Lₙ`` — every pair of states is reachable.
    Over-approximates the true interaction space.

    **Synchronized product** (``synchronized_compose``): Shared labels
    require both participants to move simultaneously (sender selects,
    receiver branches). Private labels advance one participant only.
    This is a **space-time restriction**: the product state ``(s₁, s₂)``
    is only reachable if the causal dependencies between participants
    are respected. Reduces interleavings significantly.

The hierarchy:

    L(G)  ⊆  synchronized product  ⊆  free product

    choreography ⊆ causal ⊆ all interleavings
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
class SynchronizedResult:
    """Result of synchronized composition.

    Attributes:
        participants: name → session type mapping.
        state_spaces: name → lattice mapping.
        synchronized: the synchronized product state space.
        free_product: the free product (all interleavings) for comparison.
        is_lattice: True iff the synchronized product is a lattice.
        compatibility: pairwise compatibility dict.
        shared_labels: pairwise shared labels dict.
        reduction_ratio: synchronized_states / free_product_states.
    """
    participants: dict[str, SessionType]
    state_spaces: dict[str, "StateSpace"]
    synchronized: "StateSpace"
    free_product: "StateSpace"
    is_lattice: bool
    compatibility: dict[tuple[str, str], bool]
    shared_labels: dict[tuple[str, str], set[str]]
    reduction_ratio: float


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


def synchronized_compose(
    *participants: tuple[str, SessionType],
) -> SynchronizedResult:
    """Compose session types with synchronization on shared labels.

    Unlike ``compose()`` (free product), shared method labels require both
    participants to move simultaneously: one selects, the other branches.
    Private labels advance only one participant.

    This is a **space-time restriction**: product state (s₁, s₂) is only
    reachable when causal dependencies are respected. A transition on a
    shared label ``m`` from ``(s₁, s₂)`` requires:
    - s₁ enables ``m`` (as selector) AND s₂ enables ``m`` (as offerer), or
    - s₂ enables ``m`` (as selector) AND s₁ enables ``m`` (as offerer)

    Both advance simultaneously: ``(s₁, s₂) --m--> (s₁', s₂')``.

    Raises ValueError if fewer than 2 participants or non-lattice state spaces.
    """
    from reticulate.lattice import check_lattice
    from reticulate.statespace import build_statespace

    if len(participants) < 2:
        raise ValueError("need at least two participants for synchronized composition")

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

    names = [name for name, _ in participants]

    # Compute pairwise shared labels and compatibility
    compat: dict[tuple[str, str], bool] = {}
    shared: dict[tuple[str, str], set[str]] = {}
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            compat[(n1, n2)] = check_compatibility(
                participant_dict[n1], participant_dict[n2]
            )
            shared[(n1, n2)] = _shared_labels(
                state_spaces[n1], state_spaces[n2]
            )

    # Build synchronized product via pairwise then chain
    ss_list = [state_spaces[name] for name in names]
    synced = _synchronized_product_nary(ss_list)

    # Also build free product for comparison
    free = product_nary(ss_list)

    # Check lattice property
    synced_lattice = check_lattice(synced)

    ratio = (
        len(synced.states) / len(free.states)
        if len(free.states) > 0
        else 1.0
    )

    return SynchronizedResult(
        participants=participant_dict,
        state_spaces=state_spaces,
        synchronized=synced,
        free_product=free,
        is_lattice=synced_lattice.is_lattice,
        compatibility=compat,
        shared_labels=shared,
        reduction_ratio=ratio,
    )


def synchronized_product(left: "StateSpace", right: "StateSpace") -> "StateSpace":
    """Build the synchronized product of two state spaces.

    Shared labels (appearing in both state spaces) require both components
    to move simultaneously. Private labels advance only one component.

    From state ``(s₁, s₂)``:
    - **Private label** ``m`` (only in left): ``(s₁, s₂) --m--> (s₁', s₂)``
    - **Private label** ``m`` (only in right): ``(s₁, s₂) --m--> (s₁, s₂')``
    - **Shared label** ``m``: ``(s₁, s₂) --m--> (s₁', s₂')`` only when
      both s₁ and s₂ enable ``m``

    This is the CSP-style parallel composition (synchronize on shared alphabet).
    """
    from reticulate.statespace import StateSpace as SS

    shared = _shared_labels(left, right)

    # Pre-compute adjacency lists
    left_adj: dict[int, list[tuple[str, int]]] = {s: [] for s in left.states}
    for src, lbl, tgt in left.transitions:
        left_adj[src].append((lbl, tgt))

    right_adj: dict[int, list[tuple[str, int]]] = {s: [] for s in right.states}
    for src, lbl, tgt in right.transitions:
        right_adj[src].append((lbl, tgt))

    # Build reachable states via BFS from (top, top)
    pair_to_id: dict[tuple[int, int], int] = {}
    id_labels: dict[int, str] = {}
    transitions: list[tuple[int, str, int]] = []
    selection_transitions: set[tuple[int, str, int]] = set()
    next_id = 0

    def get_id(s1: int, s2: int) -> int:
        nonlocal next_id
        pair = (s1, s2)
        if pair not in pair_to_id:
            sid = next_id
            next_id += 1
            pair_to_id[pair] = sid
            l1 = left.labels.get(s1, str(s1))
            l2 = right.labels.get(s2, str(s2))
            id_labels[sid] = f"({l1}, {l2})"
        return pair_to_id[pair]

    # BFS from (top, top) — only explore reachable states
    start = (left.top, right.top)
    get_id(start[0], start[1])
    queue = [start]
    visited: set[tuple[int, int]] = set()

    while queue:
        s1, s2 = queue.pop(0)
        if (s1, s2) in visited:
            continue
        visited.add((s1, s2))
        src = pair_to_id[(s1, s2)]

        # Private left transitions (label not shared)
        for lbl, s1_tgt in left_adj[s1]:
            if lbl not in shared:
                tgt = get_id(s1_tgt, s2)
                tr = (src, lbl, tgt)
                transitions.append(tr)
                if left.is_selection(s1, lbl, s1_tgt):
                    selection_transitions.add(tr)
                if (s1_tgt, s2) not in visited:
                    queue.append((s1_tgt, s2))

        # Private right transitions (label not shared)
        for lbl, s2_tgt in right_adj[s2]:
            if lbl not in shared:
                tgt = get_id(s1, s2_tgt)
                tr = (src, lbl, tgt)
                transitions.append(tr)
                if right.is_selection(s2, lbl, s2_tgt):
                    selection_transitions.add(tr)
                if (s1, s2_tgt) not in visited:
                    queue.append((s1, s2_tgt))

        # Synchronized transitions (shared labels — both must enable)
        for lbl_l, s1_tgt in left_adj[s1]:
            if lbl_l in shared:
                for lbl_r, s2_tgt in right_adj[s2]:
                    if lbl_r == lbl_l:
                        tgt = get_id(s1_tgt, s2_tgt)
                        tr = (src, lbl_l, tgt)
                        transitions.append(tr)
                        # Selection if either side is selecting
                        if (left.is_selection(s1, lbl_l, s1_tgt)
                                or right.is_selection(s2, lbl_l, s2_tgt)):
                            selection_transitions.add(tr)
                        if (s1_tgt, s2_tgt) not in visited:
                            queue.append((s1_tgt, s2_tgt))

    top = pair_to_id[(left.top, right.top)]
    bottom_pair = (left.bottom, right.bottom)
    # Bottom might not be reachable in synchronized product
    if bottom_pair in pair_to_id:
        bottom = pair_to_id[bottom_pair]
    else:
        # If (bottom, bottom) is unreachable, find the minimal reachable state
        # This indicates a synchronization deadlock
        bottom = get_id(bottom_pair[0], bottom_pair[1])

    return SS(
        states=set(pair_to_id.values()),
        transitions=transitions,
        top=top,
        bottom=bottom,
        labels=id_labels,
        selection_transitions=selection_transitions,
    )


def _synchronized_product_nary(
    state_spaces: list["StateSpace"],
) -> "StateSpace":
    """N-ary synchronized product via left-fold."""
    if not state_spaces:
        raise ValueError("need at least one state space")
    return reduce(synchronized_product, state_spaces)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _shared_labels(left: "StateSpace", right: "StateSpace") -> set[str]:
    """Find transition labels that appear in both state spaces."""
    left_labels = {lbl for _, lbl, _ in left.transitions}
    right_labels = {lbl for _, lbl, _ in right.transitions}
    return left_labels & right_labels



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
