"""Session type origami: lattice folding, quotients, and surgery (Step 60c).

Structural transformations on session type state spaces that preserve
or intentionally modify lattice properties.  Four families of operations:

1. **Folding** — merge states that are observationally equivalent
   (same outgoing label set) to obtain a smaller state space.
2. **Extraction** — carve out sublattices rooted at a given state.
3. **Pruning** — remove dead states (unreachable from top, or not
   reaching bottom).
4. **Surgery** — cut or contract individual transitions and report
   the effect on the lattice property.

Naming analogy: origami folds a flat sheet into structure; here we fold
a state space into a smaller equivalent, or unfold by extraction.

This module provides:
    ``fold_by_label(ss)`` — merge states with identical outgoing label sets.
    ``fold_by_depth(ss)`` — conservative folding (same depth AND label set).
    ``extract_sublattice(ss, root)`` — sub-statespace reachable from root.
    ``prune_unreachable(ss)`` — remove dead states.
    ``contract_edge(ss, src, label, tgt)`` — merge two states along an edge.
    ``classify_fold(ss)`` — how foldable is this state space?
    ``surgery_cut(ss, src, label, tgt)`` — remove a transition, check lattice.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from reticulate.lattice import check_lattice
from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldResult:
    """Result of a folding operation.

    Attributes:
        original_states: Number of states before folding.
        folded_states: Number of states after folding.
        merge_map: Mapping from original state ID to folded state ID.
        is_lattice: Whether the folded state space is a lattice.
        preserved_top: Whether the top state survived folding.
        preserved_bottom: Whether the bottom state survived folding.
    """

    original_states: int
    folded_states: int
    merge_map: dict[int, int]
    is_lattice: bool
    preserved_top: bool
    preserved_bottom: bool


@dataclass(frozen=True)
class SurgeryResult:
    """Result of a surgery (transition cut) operation.

    Attributes:
        original_states: Number of states before surgery.
        result_states: Number of states after surgery (and pruning).
        cut_transitions: Number of transitions removed.
        added_transitions: Number of transitions added (always 0 for cut).
        is_lattice: Whether the result is still a lattice.
    """

    original_states: int
    result_states: int
    cut_transitions: int
    added_transitions: int
    is_lattice: bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _outgoing_labels(ss: StateSpace, state: int) -> frozenset[str]:
    """Return the set of labels on outgoing transitions from *state*."""
    return frozenset(l for s, l, t in ss.transitions if s == state)


def _bfs_depths(ss: StateSpace) -> dict[int, int]:
    """BFS from top; return shortest-path depth of each reachable state."""
    depths: dict[int, int] = {ss.top: 0}
    queue: deque[int] = deque([ss.top])
    while queue:
        s = queue.popleft()
        d = depths[s]
        for src, _label, tgt in ss.transitions:
            if src == s and tgt not in depths:
                depths[tgt] = d + 1
                queue.append(tgt)
    return depths


def _reachable_from(ss: StateSpace, state: int) -> set[int]:
    """All states reachable from *state* (inclusive)."""
    visited: set[int] = set()
    stack = [state]
    while stack:
        s = stack.pop()
        if s in visited:
            continue
        visited.add(s)
        for src, _label, tgt in ss.transitions:
            if src == s:
                visited.add(tgt) if tgt in visited else stack.append(tgt)
    return visited


def _reaches_target(ss: StateSpace, target: int) -> set[int]:
    """All states from which *target* is reachable (reverse BFS)."""
    # Build reverse adjacency
    rev: dict[int, list[int]] = {s: [] for s in ss.states}
    for src, _label, tgt in ss.transitions:
        rev.setdefault(tgt, []).append(src)
    visited: set[int] = set()
    stack = [target]
    while stack:
        s = stack.pop()
        if s in visited:
            continue
        visited.add(s)
        for pred in rev.get(s, []):
            if pred not in visited:
                stack.append(pred)
    return visited


def _build_folded_ss(
    ss: StateSpace,
    merge_map: dict[int, int],
) -> StateSpace:
    """Build a new StateSpace by applying *merge_map* to remap state IDs."""
    new_states = set(merge_map.values())
    new_top = merge_map[ss.top]
    # Bottom may not be in merge_map if it was not in ss.states
    # (e.g. recursive types where bottom ID is a placeholder).
    new_bottom = merge_map.get(ss.bottom, ss.bottom)

    seen_transitions: set[tuple[int, str, int]] = set()
    new_transitions: list[tuple[int, str, int]] = []
    new_selection: set[tuple[int, str, int]] = set()

    for src, label, tgt in ss.transitions:
        ns, nt = merge_map[src], merge_map[tgt]
        if ns == nt:
            continue  # self-loop from merging — drop
        tri = (ns, label, nt)
        if tri not in seen_transitions:
            seen_transitions.add(tri)
            new_transitions.append(tri)
        if (src, label, tgt) in ss.selection_transitions:
            new_selection.add(tri)

    # If bottom is not in new_states (e.g. recursive types with phantom
    # bottom), pick a sink or fall back to top.
    if new_bottom not in new_states:
        sinks = [
            s for s in new_states
            if not any(src == s for src, _, _ in new_transitions)
        ]
        new_bottom = sinks[0] if sinks else new_top

    # Labels: pick first encountered label for each merged state
    new_labels: dict[int, str] = {}
    for old, new in merge_map.items():
        if new not in new_labels and old in ss.labels:
            new_labels[new] = ss.labels[old]

    return StateSpace(
        states=new_states,
        transitions=new_transitions,
        top=new_top,
        bottom=new_bottom,
        labels=new_labels,
        selection_transitions=new_selection,
    )


def _fold_result(
    ss: StateSpace,
    folded: StateSpace,
    merge_map: dict[int, int],
    is_lattice: bool,
) -> FoldResult:
    """Build a FoldResult, handling missing bottom gracefully."""
    mapped_top = merge_map.get(ss.top, ss.top)
    mapped_bottom = merge_map.get(ss.bottom, ss.bottom)
    return FoldResult(
        original_states=len(ss.states),
        folded_states=len(folded.states),
        merge_map=merge_map,
        is_lattice=is_lattice,
        preserved_top=mapped_top in folded.states,
        preserved_bottom=mapped_bottom in folded.states,
    )


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def fold_by_label(ss: StateSpace) -> tuple[StateSpace, FoldResult]:
    """Merge states that have identical outgoing transition label sets.

    Two states s1, s2 are merged iff they have the exact same set of
    labels on their outgoing edges.  The merged state inherits all
    incoming and outgoing transitions (remapped).

    Returns:
        (folded_state_space, FoldResult)
    """
    # Group states by outgoing label set
    groups: dict[frozenset[str], list[int]] = {}
    for s in ss.states:
        key = _outgoing_labels(ss, s)
        groups.setdefault(key, []).append(s)

    # Build merge_map: each group gets the smallest state ID as representative
    merge_map: dict[int, int] = {}
    for members in groups.values():
        rep = min(members)
        for m in members:
            merge_map[m] = rep

    folded = _build_folded_ss(ss, merge_map)
    lattice_result = check_lattice(folded)

    return folded, _fold_result(ss, folded, merge_map, lattice_result.is_lattice)


def fold_by_depth(ss: StateSpace) -> tuple[StateSpace, FoldResult]:
    """Merge states at the same BFS depth with identical outgoing label sets.

    More conservative than ``fold_by_label``: only states at the same
    depth from top AND with the same outgoing labels are merged.

    Returns:
        (folded_state_space, FoldResult)
    """
    depths = _bfs_depths(ss)

    # Group by (depth, outgoing_label_set)
    groups: dict[tuple[int, frozenset[str]], list[int]] = {}
    for s in ss.states:
        d = depths.get(s, -1)
        key = (d, _outgoing_labels(ss, s))
        groups.setdefault(key, []).append(s)

    merge_map: dict[int, int] = {}
    for members in groups.values():
        rep = min(members)
        for m in members:
            merge_map[m] = rep

    folded = _build_folded_ss(ss, merge_map)
    lattice_result = check_lattice(folded)

    return folded, _fold_result(ss, folded, merge_map, lattice_result.is_lattice)


def extract_sublattice(ss: StateSpace, root: int) -> StateSpace:
    """Extract the sub-statespace reachable from *root*.

    The returned state space has *root* as its top.  If the original
    bottom is reachable from *root*, it becomes the new bottom;
    otherwise the bottom is set to the state with no outgoing
    transitions (or *root* itself if the subgraph is a single node).

    Raises:
        ValueError: If *root* is not in ``ss.states``.
    """
    if root not in ss.states:
        raise ValueError(f"root {root} not in state space")

    reachable = _reachable_from(ss, root)
    new_transitions = [
        (s, l, t) for s, l, t in ss.transitions
        if s in reachable and t in reachable
    ]
    new_selection = {
        (s, l, t) for s, l, t in ss.selection_transitions
        if s in reachable and t in reachable
    }

    # Determine bottom
    if ss.bottom in reachable:
        new_bottom = ss.bottom
    else:
        # Pick a sink node (no outgoing) if any
        sinks = [
            s for s in reachable
            if not any(src == s for src, _, _ in new_transitions)
        ]
        new_bottom = sinks[0] if sinks else root

    new_labels = {s: ss.labels[s] for s in reachable if s in ss.labels}

    return StateSpace(
        states=reachable,
        transitions=new_transitions,
        top=root,
        bottom=new_bottom,
        labels=new_labels,
        selection_transitions=new_selection,
    )


def prune_unreachable(ss: StateSpace) -> StateSpace:
    """Remove states not reachable from top or not reaching bottom.

    Returns a new StateSpace containing only the states that are both
    forward-reachable from top and backward-reachable from bottom.
    """
    from_top = _reachable_from(ss, ss.top)
    to_bottom = _reaches_target(ss, ss.bottom)
    live = from_top & to_bottom

    if not live:
        # Degenerate: keep at least top and bottom
        live = {ss.top, ss.bottom}

    new_transitions = [
        (s, l, t) for s, l, t in ss.transitions
        if s in live and t in live
    ]
    new_selection = {
        (s, l, t) for s, l, t in ss.selection_transitions
        if s in live and t in live
    }
    new_labels = {s: ss.labels[s] for s in live if s in ss.labels}

    return StateSpace(
        states=live,
        transitions=new_transitions,
        top=ss.top,
        bottom=ss.bottom,
        labels=new_labels,
        selection_transitions=new_selection,
    )


def contract_edge(
    ss: StateSpace,
    src: int,
    label: str,
    tgt: int,
) -> StateSpace:
    """Contract a transition by merging *src* and *tgt* into one state.

    The merged state keeps the ID of *src*.  All transitions that
    referenced *tgt* are redirected to *src*.  The contracted edge
    itself is removed.

    Raises:
        ValueError: If the transition ``(src, label, tgt)`` does not exist.
    """
    if (src, label, tgt) not in ss.transitions:
        raise ValueError(f"transition ({src}, {label!r}, {tgt}) not found")

    merged = src  # keep src's ID
    remap = {s: (merged if s == tgt else s) for s in ss.states}

    new_states = {remap[s] for s in ss.states}
    seen: set[tuple[int, str, int]] = set()
    new_transitions: list[tuple[int, str, int]] = []
    new_selection: set[tuple[int, str, int]] = set()

    for s, l, t in ss.transitions:
        ns, nt = remap[s], remap[t]
        if (s, l, t) == (src, label, tgt):
            continue  # remove contracted edge
        if ns == nt:
            continue  # self-loop from merging
        tri = (ns, l, nt)
        if tri not in seen:
            seen.add(tri)
            new_transitions.append(tri)
        if (s, l, t) in ss.selection_transitions:
            new_selection.add(tri)

    new_top = remap[ss.top]
    new_bottom = remap[ss.bottom]
    new_labels: dict[int, str] = {}
    for old, new in remap.items():
        if new not in new_labels and old in ss.labels:
            new_labels[new] = ss.labels[old]

    return StateSpace(
        states=new_states,
        transitions=new_transitions,
        top=new_top,
        bottom=new_bottom,
        labels=new_labels,
        selection_transitions=new_selection,
    )


def classify_fold(ss: StateSpace) -> str:
    """Classify how much a state space can be folded.

    Returns:
        ``"minimal"`` — no two states share an outgoing label set
            (nothing to fold).
        ``"foldable"`` — some states can be merged (1-50% reduction).
        ``"highly_foldable"`` — more than 50% of states can be merged.
    """
    _, result = fold_by_label(ss)
    if result.folded_states == result.original_states:
        return "minimal"
    reduction = 1.0 - result.folded_states / result.original_states
    if reduction > 0.5:
        return "highly_foldable"
    return "foldable"


def surgery_cut(
    ss: StateSpace,
    src: int,
    label: str,
    tgt: int,
) -> SurgeryResult:
    """Remove a transition and report whether the result is still a lattice.

    The transition ``(src, label, tgt)`` is deleted.  States that become
    unreachable are pruned.

    Raises:
        ValueError: If the transition ``(src, label, tgt)`` does not exist.
    """
    if (src, label, tgt) not in ss.transitions:
        raise ValueError(f"transition ({src}, {label!r}, {tgt}) not found")

    new_transitions = [
        (s, l, t) for s, l, t in ss.transitions
        if (s, l, t) != (src, label, tgt)
    ]
    new_selection = {
        (s, l, t) for s, l, t in ss.selection_transitions
        if (s, l, t) != (src, label, tgt)
    }

    cut_ss = StateSpace(
        states=set(ss.states),
        transitions=new_transitions,
        top=ss.top,
        bottom=ss.bottom,
        labels=dict(ss.labels),
        selection_transitions=new_selection,
    )

    # Prune states that became unreachable
    pruned = prune_unreachable(cut_ss)
    lattice_result = check_lattice(pruned)

    return SurgeryResult(
        original_states=len(ss.states),
        result_states=len(pruned.states),
        cut_transitions=1,
        added_transitions=0,
        is_lattice=lattice_result.is_lattice,
    )
