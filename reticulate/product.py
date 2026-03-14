"""Product construction for parallel session types.

Given two state spaces L₁ and L₂, constructs the product poset L₁ × L₂:

    (s₁, s₂) ≤ (s₁', s₂')   iff   s₁ ≤₁ s₁'  and  s₂ ≤₂ s₂'

Transitions from (s₁, s₂) include all transitions from s₁ (advancing the
left component) and all transitions from s₂ (advancing the right component).

See docs/specs/parallel-constructor-spec.md Section 4.2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


def product_statespace(left: StateSpace, right: StateSpace) -> StateSpace:
    """Construct the product state space L₁ × L₂.

    The result has:
    - ``|L₁| × |L₂|`` states (one per concurrent configuration)
    - Top = ``(left.top, right.top)`` — the fork point
    - Bottom = ``(left.bottom, right.bottom)`` — the join point
    - Transitions: from ``(s₁, s₂)``, every left-transition advances s₁
      and every right-transition advances s₂

    Both operands must be fully constructed ``StateSpace`` objects.
    """
    from reticulate.statespace import StateSpace as SS

    # Pre-compute adjacency lists for efficiency
    left_adj: dict[int, list[tuple[str, int]]] = {s: [] for s in left.states}
    for src, lbl, tgt in left.transitions:
        left_adj[src].append((lbl, tgt))

    right_adj: dict[int, list[tuple[str, int]]] = {s: [] for s in right.states}
    for src, lbl, tgt in right.transitions:
        right_adj[src].append((lbl, tgt))

    # Assign fresh IDs to product states
    next_id = 0
    pair_to_id: dict[tuple[int, int], int] = {}
    id_labels: dict[int, str] = {}

    for s1 in left.states:
        for s2 in right.states:
            sid = next_id
            next_id += 1
            pair_to_id[(s1, s2)] = sid
            l1 = left.labels.get(s1, str(s1))
            l2 = right.labels.get(s2, str(s2))
            id_labels[sid] = f"({l1}, {l2})"

    # Build transitions
    transitions: list[tuple[int, str, int]] = []
    selection_transitions: set[tuple[int, str, int]] = set()

    for s1 in left.states:
        for s2 in right.states:
            src = pair_to_id[(s1, s2)]
            # Left-component transitions: (s1, s2) --[l]--> (s1', s2)
            for lbl, s1_tgt in left_adj[s1]:
                tr = (src, lbl, pair_to_id[(s1_tgt, s2)])
                transitions.append(tr)
                if left.is_selection(s1, lbl, s1_tgt):
                    selection_transitions.add(tr)
            # Right-component transitions: (s1, s2) --[l]--> (s1, s2')
            for lbl, s2_tgt in right_adj[s2]:
                tr = (src, lbl, pair_to_id[(s1, s2_tgt)])
                transitions.append(tr)
                if right.is_selection(s2, lbl, s2_tgt):
                    selection_transitions.add(tr)

    top = pair_to_id[(left.top, right.top)]
    bottom = pair_to_id[(left.bottom, right.bottom)]

    # Build product coordinate map and collect factors
    left_factors = left.product_factors or [left]
    right_factors = right.product_factors or [right]
    factors = list(left_factors) + list(right_factors)

    coord_map: dict[int, tuple[int, ...]] = {}
    for (s1, s2), pid in pair_to_id.items():
        left_coord = left.product_coords[s1] if left.product_coords else (s1,)
        right_coord = right.product_coords[s2] if right.product_coords else (s2,)
        coord_map[pid] = left_coord + right_coord

    return SS(
        states=set(pair_to_id.values()),
        transitions=transitions,
        top=top,
        bottom=bottom,
        labels=id_labels,
        selection_transitions=selection_transitions,
        product_coords=coord_map,
        product_factors=factors,
    )
