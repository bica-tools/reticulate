"""Pendular session types: alternation classification (Step 59k).

A session type is *pendular* if every transition alternates between
Select (+) and Branch (&) polarity.  That is, for every edge (s1→s2):
    - if s1 is Select, then s2 is Branch or End
    - if s1 is Branch, then s2 is Select or End

This captures the structure of two-player alternating games and
request-response protocols.  The key property is *local*: it concerns
individual transitions, not global path lengths (game trees have
shortcuts where one player wins early, so they are NOT graded).

Classification hierarchy:
    - **Strictly pendular**: every transition alternates +/& polarity.
    - **Weakly pendular**: most transitions alternate, with bounded violations.
    - **Biased**: one polarity (+ or &) dominates — e.g., streaming, setup.
    - **Chaotic**: no regular alternation pattern.

Pendular types have special lattice properties:
    - Duality is a simple phase shift (swap who starts).
    - Along any single path, polarity alternates (players take turns).
    - The lattice has a layered flavor, though not necessarily graded
      (early termination creates shortcut edges to bottom).

This module provides:
    ``is_pendular(ss)`` — check strict alternation.
    ``pendular_depth(ss)`` — compute BFS depth and polarity at each state.
    ``classify_alternation(ss)`` — full classification with metrics.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from reticulate.reticular import classify_state
from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PendularResult:
    """Result of pendular classification.

    Attributes:
        is_pendular: True if strictly pendular (every transition alternates).
        classification: One of 'pendular', 'weakly_pendular', 'biased', 'chaotic'.
        is_graded: True if all paths top->bottom have the same length.
        max_depth: BFS depth from top to farthest state.
        levels: Mapping from BFS depth to the set of states at that depth.
        level_polarity: Mapping from BFS depth to dominant polarity at that level.
        violations: Transitions (src, label, tgt, src_pol, tgt_pol) that break alternation.
        select_ratio: Fraction of non-end states that are Select.
        branch_ratio: Fraction of non-end states that are Branch.
    """

    is_pendular: bool
    classification: str
    is_graded: bool
    max_depth: int
    levels: dict[int, set[int]]
    level_polarity: dict[int, str]
    violations: tuple[tuple[int, str, int, str, str], ...]
    select_ratio: float
    branch_ratio: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_depths(ss: StateSpace) -> dict[int, int]:
    """BFS from top to compute shortest-path depth of each state."""
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


def _state_polarity(ss: StateSpace, state: int) -> str:
    """Return the polarity of a state: 'select', 'branch', 'end', or 'product'."""
    cls = classify_state(ss, state)
    return cls.kind


def _alternation_ok(src_pol: str, tgt_pol: str) -> bool:
    """Check if a transition from src_pol to tgt_pol respects alternation.

    Rules:
        select -> branch or end: OK
        branch -> select or end: OK
        anything -> end: OK (early termination)
        end -> anything: N/A (end has no outgoing transitions)
        product -> anything: OK (parallel states are neutral)
        anything -> product: OK
    """
    if src_pol in ("end", "product") or tgt_pol in ("end", "product"):
        return True
    if src_pol == "select" and tgt_pol == "branch":
        return True
    if src_pol == "branch" and tgt_pol == "select":
        return True
    return False


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def is_pendular(ss: StateSpace) -> bool:
    """Check if a state space is strictly pendular.

    A state space is strictly pendular if for every transition (s1 -> s2),
    the polarity alternates: select->branch, branch->select, or
    either->end (early termination is always allowed).
    """
    result = classify_alternation(ss)
    return result.is_pendular


def pendular_depth(ss: StateSpace) -> dict[int, tuple[int, str]]:
    """Compute BFS depth and polarity for each state.

    Returns:
        Mapping from state ID to (depth, polarity) where polarity is
        'select', 'branch', 'end', or 'product'.
    """
    depths = _compute_depths(ss)
    return {s: (d, _state_polarity(ss, s)) for s, d in depths.items()}


def classify_alternation(ss: StateSpace) -> PendularResult:
    """Classify the alternation pattern of a state space.

    Returns a PendularResult with full analysis including violations.
    """
    depths = _compute_depths(ss)

    if not depths:
        return PendularResult(
            is_pendular=True,
            classification="pendular",
            is_graded=True,
            max_depth=0,
            levels={},
            level_polarity={},
            violations=(),
            select_ratio=0.0,
            branch_ratio=0.0,
        )

    max_depth = max(depths.values())

    # Build levels from BFS depth
    levels: dict[int, set[int]] = {}
    for s, d in depths.items():
        levels.setdefault(d, set()).add(s)

    # Check graded: every transition goes from depth d to depth d+1
    is_graded = True
    for src, _label, tgt in ss.transitions:
        if src in depths and tgt in depths:
            if depths[tgt] != depths[src] + 1:
                is_graded = False
                break

    # Compute polarity at each level
    level_polarity: dict[int, str] = {}
    for d, states in levels.items():
        polarities = {_state_polarity(ss, s) for s in states}
        if len(polarities) == 1:
            level_polarity[d] = polarities.pop()
        else:
            level_polarity[d] = "mixed"

    # Find violations: transitions where polarity doesn't alternate
    violations: list[tuple[int, str, int, str, str]] = []
    for src, label, tgt in ss.transitions:
        src_pol = _state_polarity(ss, src)
        tgt_pol = _state_polarity(ss, tgt)
        if not _alternation_ok(src_pol, tgt_pol):
            violations.append((src, label, tgt, src_pol, tgt_pol))

    is_pendular = len(violations) == 0

    # Compute ratios
    non_end = [s for s in ss.states if _state_polarity(ss, s) not in ("end",)]
    total_non_end = len(non_end) if non_end else 1
    select_count = sum(1 for s in non_end if _state_polarity(ss, s) == "select")
    branch_count = sum(1 for s in non_end if _state_polarity(ss, s) == "branch")
    select_ratio = select_count / total_non_end
    branch_ratio = branch_count / total_non_end

    # Classify
    if is_pendular:
        classification = "pendular"
    elif len(violations) <= max(1, len(ss.transitions) // 10):
        classification = "weakly_pendular"
    elif abs(select_ratio - branch_ratio) > 0.4:
        classification = "biased"
    else:
        classification = "chaotic"

    return PendularResult(
        is_pendular=is_pendular,
        classification=classification,
        is_graded=is_graded,
        max_depth=max_depth,
        levels=levels,
        level_polarity=level_polarity,
        violations=tuple(violations),
        select_ratio=round(select_ratio, 3),
        branch_ratio=round(branch_ratio, 3),
    )
