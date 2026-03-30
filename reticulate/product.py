"""Product construction for parallel session types.

Given two state spaces L₁ and L₂, constructs the product poset L₁ × L₂:

    (s₁, s₂) ≤ (s₁', s₂')   iff   s₁ ≤₁ s₁'  and  s₂ ≤₂ s₂'

Transitions from (s₁, s₂) include all transitions from s₁ (advancing the
left component) and all transitions from s₂ (advancing the right component).

Also provides ``power_statespace(ss, n)`` to compute S^n = S ∥ S ∥ ... ∥ S
(n parallel copies), and ``power_type(type_str, n)`` for AST-level power.

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
        if left.product_coords and s1 in left.product_coords:
            left_coord = left.product_coords[s1]
        else:
            left_coord = (s1,)
        if right.product_coords and s2 in right.product_coords:
            right_coord = right.product_coords[s2]
        else:
            right_coord = (s2,)
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


def sync_product_statespace(left: StateSpace, right: StateSpace) -> StateSpace:
    """Construct the *synchronous* product state space L₁ ⊗ L₂.

    Unlike the interleaved product (``product_statespace``), the synchronous
    product requires both components to advance simultaneously.  A transition
    exists from ``(s₁, s₂)`` to ``(s₁', s₂')`` only when *both* components
    have an enabled transition at the same time.

    Matching is by label: if both L₁ and L₂ have a transition on label ``m``,
    they advance together.  If only one has it, it can still advance alone
    (semi-synchronous / independent progress).

    For music: voices play in lock-step on shared beats.
    For protocols: synchronous message passing (both sides act on same event).

    State space size:
        - Interleaved ∥: |L₁| × |L₂| states (all combinations reachable)
        - Synchronous ⊗: ≤ |L₁| × |L₂| states (only *reachable* pairs)

    The key difference is that sync only explores states reachable via
    paired transitions, producing much smaller state spaces for coordinated
    systems like musical scores.

    Properties:
        - L₁ ⊗ L₂ is always a sub-poset of L₁ × L₂
        - If both are lattices, L₁ ⊗ L₂ is a lattice
        - Top = (left.top, right.top), Bottom = (left.bottom, right.bottom)
    """
    from reticulate.statespace import StateSpace as SS

    # Pre-compute adjacency lists
    left_adj: dict[int, list[tuple[str, int]]] = {s: [] for s in left.states}
    for src, lbl, tgt in left.transitions:
        left_adj[src].append((lbl, tgt))

    right_adj: dict[int, list[tuple[str, int]]] = {s: [] for s in right.states}
    for src, lbl, tgt in right.transitions:
        right_adj[src].append((lbl, tgt))

    # Index right transitions by label for matching
    right_by_label: dict[int, dict[str, list[int]]] = {s: {} for s in right.states}
    for src, lbl, tgt in right.transitions:
        right_by_label[src].setdefault(lbl, []).append(tgt)

    # BFS from (top, top): only explore reachable pairs
    next_id = 0
    pair_to_id: dict[tuple[int, int], int] = {}
    id_labels: dict[int, str] = {}
    transitions: list[tuple[int, str, int]] = []
    selection_transitions: set[tuple[int, str, int]] = set()
    queue: list[tuple[int, int]] = [(left.top, right.top)]
    pair_to_id[(left.top, right.top)] = 0
    id_labels[0] = f"({left.labels.get(left.top, str(left.top))}, {right.labels.get(right.top, str(right.top))})"
    next_id = 1

    while queue:
        s1, s2 = queue.pop(0)
        src_id = pair_to_id[(s1, s2)]

        # 1. Synchronized transitions: same label on both sides
        for lbl, s1_tgt in left_adj[s1]:
            if lbl in right_by_label[s2]:
                for s2_tgt in right_by_label[s2][lbl]:
                    pair = (s1_tgt, s2_tgt)
                    if pair not in pair_to_id:
                        pair_to_id[pair] = next_id
                        l1 = left.labels.get(s1_tgt, str(s1_tgt))
                        l2 = right.labels.get(s2_tgt, str(s2_tgt))
                        id_labels[next_id] = f"({l1}, {l2})"
                        next_id += 1
                        queue.append(pair)
                    tr = (src_id, lbl, pair_to_id[pair])
                    transitions.append(tr)
                    if left.is_selection(s1, lbl, s1_tgt) or right.is_selection(s2, lbl, s2_tgt):
                        selection_transitions.add(tr)

        # 2. Independent left transitions (label not in right)
        right_labels_here = set(right_by_label[s2].keys())
        for lbl, s1_tgt in left_adj[s1]:
            if lbl not in right_labels_here:
                pair = (s1_tgt, s2)
                if pair not in pair_to_id:
                    pair_to_id[pair] = next_id
                    l1 = left.labels.get(s1_tgt, str(s1_tgt))
                    l2 = right.labels.get(s2, str(s2))
                    id_labels[next_id] = f"({l1}, {l2})"
                    next_id += 1
                    queue.append(pair)
                tr = (src_id, lbl, pair_to_id[pair])
                transitions.append(tr)
                if left.is_selection(s1, lbl, s1_tgt):
                    selection_transitions.add(tr)

        # 3. Independent right transitions (label not in left)
        left_labels_here = {lbl for lbl, _ in left_adj[s1]}
        for lbl, s2_tgt in right_adj[s2]:
            if lbl not in left_labels_here:
                pair = (s1, s2_tgt)
                if pair not in pair_to_id:
                    pair_to_id[pair] = next_id
                    l1 = left.labels.get(s1, str(s1))
                    l2 = right.labels.get(s2_tgt, str(s2_tgt))
                    id_labels[next_id] = f"({l1}, {l2})"
                    next_id += 1
                    queue.append(pair)
                tr = (src_id, lbl, pair_to_id[pair])
                transitions.append(tr)
                if right.is_selection(s2, lbl, s2_tgt):
                    selection_transitions.add(tr)

    top = pair_to_id[(left.top, right.top)]
    bottom_pair = (left.bottom, right.bottom)
    if bottom_pair not in pair_to_id:
        pair_to_id[bottom_pair] = next_id
        id_labels[next_id] = "end"
        next_id += 1
    bottom = pair_to_id[bottom_pair]

    return SS(
        states=set(pair_to_id.values()),
        transitions=transitions,
        top=top,
        bottom=bottom,
        labels=id_labels,
        selection_transitions=selection_transitions,
    )


def sync_product_n(*spaces: StateSpace) -> StateSpace:
    """N-ary synchronous product: L₁ ⊗ L₂ ⊗ ... ⊗ Lₙ.

    Reduces left-to-right via binary sync_product_statespace.
    """
    from functools import reduce
    if not spaces:
        from reticulate.statespace import StateSpace as SS
        return SS(states={0}, transitions=[], top=0, bottom=0,
                  labels={0: "end"}, selection_transitions=set())
    return reduce(sync_product_statespace, spaces)


def power_statespace(ss: StateSpace, n: int) -> StateSpace:
    """Compute S^n: the n-fold product of a state space with itself.

    ``power_statespace(ss, n)`` = ss × ss × ... × ss (n copies).

    Properties:
    - |S^n| = |S|^n states
    - S^1 = S (identity)
    - S^0 = trivial 1-state lattice (top = bottom = end)
    - S^(m+n) ≅ S^m × S^n (additive exponent)

    Parameters:
        ss: The base state space.
        n: The exponent (number of parallel copies). Must be ≥ 0.

    Returns:
        The n-fold product state space.

    Raises:
        ValueError: If n < 0.
    """
    if n < 0:
        raise ValueError(f"Exponent must be non-negative, got {n}")

    if n == 0:
        from reticulate.statespace import StateSpace as SS
        return SS(
            states={0},
            transitions=[],
            top=0,
            bottom=0,
            labels={0: "end"},
            selection_transitions=set(),
        )

    if n == 1:
        return ss

    from functools import reduce
    return reduce(product_statespace, [ss] * n)


def power_type(type_str: str, n: int) -> str:
    """Construct the session type string for S^n (n parallel copies).

    Returns the type string with label disambiguation:
    each copy's labels are suffixed with the copy number
    to satisfy the WF-Par disjointness condition.

    Parameters:
        type_str: The base session type string.
        n: The exponent (number of copies). Must be ≥ 1.

    Returns:
        A session type string for n parallel copies with unique labels.

    Raises:
        ValueError: If n < 1.
    """
    if n < 1:
        raise ValueError(f"Exponent must be ≥ 1 for type construction, got {n}")
    if n == 1:
        return type_str

    from reticulate.parser import parse, pretty
    from reticulate.parser import (
        Branch, Select, Rec, Var, End, Wait, Parallel, Continuation,
    )

    base = parse(type_str)

    def relabel(node: object, suffix: str) -> object:
        """Recursively append suffix to all method/label names."""
        if isinstance(node, End) or isinstance(node, Wait):
            return node
        if isinstance(node, Var):
            return Var(node.name + suffix)
        if isinstance(node, Branch):
            return Branch(tuple(
                (m + suffix, relabel(s, suffix))
                for m, s in node.choices
            ))
        if isinstance(node, Select):
            return Select(tuple(
                (l + suffix, relabel(s, suffix))
                for l, s in node.choices
            ))
        if isinstance(node, Rec):
            return Rec(node.var + suffix, relabel(node.body, suffix))
        if isinstance(node, Parallel):
            return Parallel(tuple(relabel(b, suffix) for b in node.branches))
        if isinstance(node, Continuation):
            return Continuation(relabel(node.left, suffix), relabel(node.right, suffix))
        return node  # pragma: no cover

    copies = [relabel(base, f"_{i+1}") for i in range(n)]
    par = Parallel(tuple(copies))
    return pretty(par)
