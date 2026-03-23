"""Coverability analysis for session-type Petri nets (Step 24).

Implements the Karp--Miller coverability tree algorithm for Petri nets
constructed from session type state spaces.  The coverability tree
over-approximates the reachability set by introducing the symbol omega
(representing unboundedness) whenever a marking strictly dominates an
ancestor along the same tree branch.

Key results for session types:
  - Session-type nets in state-machine encoding are always 1-safe
    (each reachable marking has exactly one token), so the coverability
    tree coincides with the reachability graph and no omega appears.
  - The interesting case arises when we *relax* the encoding to allow
    unbounded recursive unfolding depth, where tokens may accumulate.

Exported API:

  CoverabilityNode      — one node of the coverability tree
  CoverabilityResult    — full analysis summary
  build_coverability_tree   — Karp--Miller algorithm with omega-acceleration
  is_coverable              — check whether a target marking is coverable
  check_boundedness         — verify all places are bounded (no omega)
  analyze_coverability      — convenience: StateSpace -> full analysis
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from reticulate.petri import (
    Marking,
    PetriNet,
    build_petri_net,
    enabled_transitions,
    fire,
    is_enabled,
    _freeze_marking,
)

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OMEGA: int = math.inf  # type: ignore[assignment]
"""Sentinel representing unboundedness in the Karp--Miller tree."""


# ---------------------------------------------------------------------------
# Omega-marking utilities
# ---------------------------------------------------------------------------

OmegaMarking = dict[int, int | float]
"""A marking that may contain omega (math.inf) entries."""


def _freeze_omega(m: OmegaMarking) -> tuple[tuple[int, int | float], ...]:
    """Freeze an omega-marking into a hashable tuple."""
    return tuple(sorted(m.items()))


def _dominates(m1: OmegaMarking, m2: OmegaMarking) -> bool:
    """Check whether m1 strictly dominates m2 (m1 > m2 component-wise).

    m1 dominates m2 iff m1(p) >= m2(p) for all places and m1 != m2.
    """
    all_places = set(m1.keys()) | set(m2.keys())
    strictly_greater = False
    for p in all_places:
        v1 = m1.get(p, 0)
        v2 = m2.get(p, 0)
        if v1 < v2:
            return False
        if v1 > v2:
            strictly_greater = True
    return strictly_greater


def _covers(m: OmegaMarking, target: Marking) -> bool:
    """Check whether omega-marking m covers target.

    m covers target iff m(p) >= target(p) for all places p.
    """
    all_places = set(m.keys()) | set(target.keys())
    for p in all_places:
        vm = m.get(p, 0)
        vt = target.get(p, 0)
        if vm < vt:
            return False
    return True


def _omega_enabled(
    net: PetriNet, marking: OmegaMarking, transition_id: int
) -> bool:
    """Check if a transition is enabled under an omega-marking."""
    if transition_id not in net.pre:
        return False
    for place_id, weight in net.pre[transition_id]:
        v = marking.get(place_id, 0)
        if v < weight:
            return False
    return True


def _omega_fire(
    net: PetriNet, marking: OmegaMarking, transition_id: int
) -> OmegaMarking | None:
    """Fire a transition under an omega-marking.

    Returns the new omega-marking, or None if the transition is not enabled.
    Omega entries are preserved: omega - k = omega, omega + k = omega.
    """
    if not _omega_enabled(net, marking, transition_id):
        return None
    new: OmegaMarking = dict(marking)
    for place_id, weight in net.pre[transition_id]:
        v = new.get(place_id, 0)
        if v == OMEGA:
            pass  # omega - k = omega
        else:
            new[place_id] = v - weight
            if new[place_id] == 0:
                del new[place_id]
    for place_id, weight in net.post[transition_id]:
        v = new.get(place_id, 0)
        if v == OMEGA:
            pass  # omega + k = omega
        else:
            new[place_id] = v + weight
    return new


# ---------------------------------------------------------------------------
# Coverability tree node
# ---------------------------------------------------------------------------

@dataclass
class CoverabilityNode:
    """A node in the Karp--Miller coverability tree.

    Attributes:
        id: Unique node identifier.
        marking: The omega-marking at this node.
        parent: Parent node ID, or None for the root.
        transition_from_parent: Transition ID that led here, or None for root.
        children: List of child node IDs.
        is_duplicate: True if this node's marking already appeared on
                      the path from root to this node (pruned).
    """
    id: int
    marking: OmegaMarking
    parent: int | None = None
    transition_from_parent: int | None = None
    children: list[int] = field(default_factory=list)
    is_duplicate: bool = False


# ---------------------------------------------------------------------------
# Coverability result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoverabilityResult:
    """Result of coverability analysis.

    Attributes:
        tree_nodes: All nodes in the coverability tree, keyed by node ID.
        is_bounded: True if every place is bounded (no omega entries).
        unbounded_places: Set of place IDs that are unbounded (have omega).
        max_tokens_per_place: Maximum token count per place across all
                              reachable markings (omega if unbounded).
        num_nodes: Total number of nodes in the tree.
        num_distinct_markings: Number of distinct omega-markings.
        is_one_safe: True if every place has at most 1 token in every
                     reachable marking.
    """
    tree_nodes: dict[int, CoverabilityNode]
    is_bounded: bool
    unbounded_places: set[int]
    max_tokens_per_place: dict[int, int | float]
    num_nodes: int
    num_distinct_markings: int
    is_one_safe: bool


# ---------------------------------------------------------------------------
# Karp--Miller coverability tree construction
# ---------------------------------------------------------------------------

def _ancestors(
    nodes: dict[int, CoverabilityNode], node_id: int
) -> list[CoverabilityNode]:
    """Return the list of ancestor nodes from root to (excluding) node_id."""
    ancestors: list[CoverabilityNode] = []
    current = nodes[node_id].parent
    while current is not None:
        ancestors.append(nodes[current])
        current = nodes[current].parent
    ancestors.reverse()
    return ancestors


def build_coverability_tree(
    net: PetriNet,
    initial_marking: Marking | None = None,
    *,
    max_nodes: int = 10_000,
) -> CoverabilityResult:
    """Build the Karp--Miller coverability tree for a Petri net.

    The algorithm explores all possible transition firings from the
    initial marking.  When a new marking strictly dominates an ancestor
    on the same root-to-node path, the dominating components are
    accelerated to omega (infinity).

    Args:
        net: The Petri net to analyse.
        initial_marking: Starting marking; defaults to net.initial_marking.
        max_nodes: Safety bound on tree size to prevent runaway exploration.

    Returns:
        A CoverabilityResult summarising the tree.
    """
    if initial_marking is None:
        initial_marking = net.initial_marking

    root_marking: OmegaMarking = dict(initial_marking)
    nodes: dict[int, CoverabilityNode] = {}
    next_id = 0

    root = CoverabilityNode(id=next_id, marking=root_marking)
    nodes[next_id] = root
    next_id += 1

    # BFS work-list of node IDs to expand
    worklist: deque[int] = deque([root.id])
    distinct_markings: set[tuple[tuple[int, int | float], ...]] = {
        _freeze_omega(root_marking)
    }

    while worklist and next_id < max_nodes:
        current_id = worklist.popleft()
        current_node = nodes[current_id]

        if current_node.is_duplicate:
            continue

        # Check if this marking already appears on the path from root
        ancestors = _ancestors(nodes, current_id)
        frozen_current = _freeze_omega(current_node.marking)
        is_dup = any(
            _freeze_omega(a.marking) == frozen_current for a in ancestors
        )
        if is_dup:
            current_node.is_duplicate = True
            continue

        # Try firing each transition
        for tid in sorted(net.transitions.keys()):
            new_marking = _omega_fire(net, current_node.marking, tid)
            if new_marking is None:
                continue

            # Omega-acceleration: if new_marking dominates an ancestor,
            # set dominating components to omega.
            for anc in ancestors + [current_node]:
                if _dominates(new_marking, anc.marking):
                    all_places = set(new_marking.keys()) | set(
                        anc.marking.keys()
                    )
                    for p in all_places:
                        if new_marking.get(p, 0) > anc.marking.get(p, 0):
                            new_marking[p] = OMEGA
                    # Only accelerate against the first dominating ancestor
                    break

            child = CoverabilityNode(
                id=next_id,
                marking=new_marking,
                parent=current_id,
                transition_from_parent=tid,
            )
            nodes[next_id] = child
            current_node.children.append(next_id)
            distinct_markings.add(_freeze_omega(new_marking))
            next_id += 1
            worklist.append(child.id)

    # Compute summary statistics
    all_places: set[int] = set()
    for pid in net.places:
        all_places.add(pid)

    max_tokens: dict[int, int | float] = {p: 0 for p in all_places}
    unbounded: set[int] = set()

    for node in nodes.values():
        for p in all_places:
            v = node.marking.get(p, 0)
            if v == OMEGA:
                max_tokens[p] = OMEGA
                unbounded.add(p)
            elif max_tokens[p] != OMEGA:
                max_tokens[p] = max(max_tokens[p], v)

    is_bounded = len(unbounded) == 0
    is_one_safe = is_bounded and all(v <= 1 for v in max_tokens.values())

    return CoverabilityResult(
        tree_nodes=nodes,
        is_bounded=is_bounded,
        unbounded_places=unbounded,
        max_tokens_per_place=max_tokens,
        num_nodes=len(nodes),
        num_distinct_markings=len(distinct_markings),
        is_one_safe=is_one_safe,
    )


# ---------------------------------------------------------------------------
# Coverability check
# ---------------------------------------------------------------------------

def is_coverable(
    net: PetriNet,
    target_marking: Marking,
    initial_marking: Marking | None = None,
) -> bool:
    """Check whether a target marking is coverable from the initial marking.

    A marking m is coverable if there exists a reachable marking m' such
    that m'(p) >= m(p) for all places p.  This is decided by the
    coverability tree: m is coverable iff some node in the tree covers m.

    Args:
        net: The Petri net.
        target_marking: The marking to check coverability of.
        initial_marking: Starting marking; defaults to net.initial_marking.

    Returns:
        True if target_marking is coverable.
    """
    result = build_coverability_tree(net, initial_marking)
    for node in result.tree_nodes.values():
        if _covers(node.marking, target_marking):
            return True
    return False


# ---------------------------------------------------------------------------
# Boundedness check
# ---------------------------------------------------------------------------

def check_boundedness(
    net: PetriNet,
    initial_marking: Marking | None = None,
) -> CoverabilityResult:
    """Check whether a Petri net is bounded (all places have finite max tokens).

    A net is bounded iff the coverability tree contains no omega entries.
    A net is k-bounded iff every place has at most k tokens in every
    reachable marking.

    Args:
        net: The Petri net to check.
        initial_marking: Starting marking; defaults to net.initial_marking.

    Returns:
        CoverabilityResult with boundedness information.
    """
    return build_coverability_tree(net, initial_marking)


# ---------------------------------------------------------------------------
# Convenience: StateSpace → full coverability analysis
# ---------------------------------------------------------------------------

def analyze_coverability(ss: "StateSpace") -> CoverabilityResult:
    """Analyse coverability for a session type state space.

    Convenience function that builds the Petri net from a state space
    and runs the full Karp--Miller coverability analysis.

    Session-type Petri nets in the standard state-machine encoding
    are always 1-safe: each reachable marking places exactly one token
    on exactly one place.  This function confirms that property.

    Args:
        ss: A session type state space.

    Returns:
        CoverabilityResult with tree, boundedness, and 1-safety data.
    """
    net = build_petri_net(ss)
    return build_coverability_tree(net)


# ---------------------------------------------------------------------------
# Extended: multi-token nets for unbounded recursion analysis
# ---------------------------------------------------------------------------

def build_unbounded_recursion_net(
    ss: "StateSpace",
    unfold_tokens: int = 2,
) -> tuple[PetriNet, Marking]:
    """Build a Petri net variant that models unbounded recursive unfolding.

    In the standard encoding, recursion creates cycles in the Petri net
    but keeps the net 1-safe (one token, cycling through places).  This
    variant starts with multiple tokens to simulate overlapping recursive
    unfoldings, which can reveal unboundedness when the recursion body
    is non-trivial.

    Args:
        ss: A session type state space.
        unfold_tokens: Number of initial tokens (simulating concurrent
                       unfoldings).  Default 2.

    Returns:
        (net, initial_marking) where the initial marking places
        unfold_tokens tokens on the top state.
    """
    net = build_petri_net(ss)
    multi_marking: Marking = {ss.top: unfold_tokens}
    return net, multi_marking


def analyze_unbounded_recursion(
    ss: "StateSpace",
    unfold_tokens: int = 2,
) -> CoverabilityResult:
    """Analyse coverability with multiple tokens for recursion stress-testing.

    Places multiple tokens at the initial state to simulate overlapping
    recursive unfoldings and checks whether the resulting net remains
    bounded.

    Args:
        ss: A session type state space.
        unfold_tokens: Number of initial tokens.

    Returns:
        CoverabilityResult for the multi-token analysis.
    """
    net, multi_marking = build_unbounded_recursion_net(ss, unfold_tokens)
    return build_coverability_tree(net, multi_marking)
