"""Protocol downgrade prevention via session type subtyping (Step 89c).

Detects and prevents protocol downgrade attacks by analyzing the subtyping
relationship between a required (strong) protocol and an offered (weak)
protocol.  A downgrade occurs when an attacker forces a protocol participant
to use a weaker protocol than intended.

Key insight: in the session type lattice, downgrade safety corresponds to
the offered protocol being a subtype (or safe replacement) of the required
protocol.  The risk is quantified by measuring what capabilities are lost
in the transition from strong to weak.

This module provides:
    ``check_downgrade(s_required, s_offered)``
        — is s_offered a safe replacement for s_required?
    ``downgrade_risk(s_strong, s_weak)``
        — quantify what capabilities are lost.
    ``safe_downgrade_set(s)``
        — enumerate safe replacement subtypes.
    ``detect_forced_downgrade(ss, attacker_choices)``
        — can attacker force a weaker protocol?
    ``analyze_downgrade(s_required, s_offered)``
        — full analysis as DowngradeResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from reticulate.parser import (
    Branch,
    End,
    Parallel,
    Rec,
    Select,
    SessionType,
    Var,
    Wait,
    pretty,
)
from reticulate.statespace import StateSpace, build_statespace
from reticulate.subtyping import is_subtype

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DowngradeRisk:
    """Quantification of downgrade risk between two protocols.

    Attributes:
        lost_methods: Methods available in strong but not in weak.
        lost_selections: Selection labels in strong but not in weak.
        retained_methods: Methods available in both.
        retained_selections: Selections available in both.
        risk_score: Fraction of capabilities lost (0 = no loss, 1 = total loss).
        depth_reduction: Difference in maximum path depth (strong - weak).
    """

    lost_methods: frozenset[str]
    lost_selections: frozenset[str]
    retained_methods: frozenset[str]
    retained_selections: frozenset[str]
    risk_score: float
    depth_reduction: int


@dataclass(frozen=True)
class ForcedDowngradeResult:
    """Result of forced downgrade detection.

    Attributes:
        is_vulnerable: True if attacker can force a weaker protocol.
        attacker_labels: Labels the attacker controls.
        forced_paths: Paths the attacker can force that bypass strong states.
        weakest_reachable: The weakest protocol state reachable by attacker.
    """

    is_vulnerable: bool
    attacker_labels: frozenset[str]
    forced_paths: list[list[tuple[int, str, int]]]
    weakest_reachable: int | None


@dataclass(frozen=True)
class DowngradeResult:
    """Full downgrade analysis result.

    Attributes:
        is_safe: True if s_offered is a safe replacement for s_required.
        is_subtype_relation: True if s_offered <= s_required (Gay-Hole).
        risk: Downgrade risk quantification.
        required_type: Pretty-printed required type.
        offered_type: Pretty-printed offered type.
        strong_state_count: Number of states in the required state space.
        weak_state_count: Number of states in the offered state space.
        safe_replacements: List of known safe replacement types (if computed).
    """

    is_safe: bool
    is_subtype_relation: bool
    risk: DowngradeRisk
    required_type: str
    offered_type: str
    strong_state_count: int
    weak_state_count: int
    safe_replacements: list[str]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_labels(ast: SessionType) -> set[str]:
    """Collect all method/label names from an AST."""
    labels: set[str] = set()
    _collect_labels_rec(ast, labels, set())
    return labels


def _collect_labels_rec(
    ast: SessionType, labels: set[str], visited: set[int]
) -> None:
    """Recursively collect labels."""
    node_id = id(ast)
    if node_id in visited:
        return
    visited.add(node_id)

    if isinstance(ast, Branch):
        for method, cont in ast.choices:
            labels.add(method)
            _collect_labels_rec(cont, labels, visited)
    elif isinstance(ast, Select):
        for label, cont in ast.choices:
            labels.add(label)
            _collect_labels_rec(cont, labels, visited)
    elif isinstance(ast, Parallel):
        _collect_labels_rec(ast.left, labels, visited)
        _collect_labels_rec(ast.right, labels, visited)
    elif isinstance(ast, Rec):
        _collect_labels_rec(ast.body, labels, visited)
    # End, Var, Wait have no labels


def _collect_branch_labels(ast: SessionType) -> set[str]:
    """Collect branch (external choice) labels only."""
    labels: set[str] = set()
    _collect_branch_rec(ast, labels, set())
    return labels


def _collect_branch_rec(
    ast: SessionType, labels: set[str], visited: set[int]
) -> None:
    node_id = id(ast)
    if node_id in visited:
        return
    visited.add(node_id)

    if isinstance(ast, Branch):
        for method, cont in ast.choices:
            labels.add(method)
            _collect_branch_rec(cont, labels, visited)
    elif isinstance(ast, Select):
        for _, cont in ast.choices:
            _collect_branch_rec(cont, labels, visited)
    elif isinstance(ast, Parallel):
        _collect_branch_rec(ast.left, labels, visited)
        _collect_branch_rec(ast.right, labels, visited)
    elif isinstance(ast, Rec):
        _collect_branch_rec(ast.body, labels, visited)


def _collect_select_labels(ast: SessionType) -> set[str]:
    """Collect selection (internal choice) labels only."""
    labels: set[str] = set()
    _collect_select_rec(ast, labels, set())
    return labels


def _collect_select_rec(
    ast: SessionType, labels: set[str], visited: set[int]
) -> None:
    node_id = id(ast)
    if node_id in visited:
        return
    visited.add(node_id)

    if isinstance(ast, Select):
        for label, cont in ast.choices:
            labels.add(label)
            _collect_select_rec(cont, labels, visited)
    elif isinstance(ast, Branch):
        for _, cont in ast.choices:
            _collect_select_rec(cont, labels, visited)
    elif isinstance(ast, Parallel):
        _collect_select_rec(ast.left, labels, visited)
        _collect_select_rec(ast.right, labels, visited)
    elif isinstance(ast, Rec):
        _collect_select_rec(ast.body, labels, visited)


def _max_path_depth(ss: StateSpace, *, max_depth: int = 50) -> int:
    """Compute the maximum path depth from top to bottom."""
    if ss.top == ss.bottom:
        return 0

    best = 0
    stack: list[tuple[int, int, set[int]]] = [(ss.top, 0, {ss.top})]
    while stack:
        state, depth, visited = stack.pop()
        if state == ss.bottom:
            best = max(best, depth)
            continue
        if depth >= max_depth:
            best = max(best, depth)
            continue
        for label, tgt in ss.enabled(state):
            if tgt not in visited:
                stack.append((tgt, depth + 1, visited | {tgt}))

    return best


def _ss_labels(ss: StateSpace) -> set[str]:
    """All transition labels in a state space."""
    return {label for _, label, _ in ss.transitions}


def _ss_selection_labels(ss: StateSpace) -> set[str]:
    """Selection labels in a state space."""
    return {label for _, label, _ in ss.selection_transitions}


def _ss_branch_labels(ss: StateSpace) -> set[str]:
    """Branch (non-selection) labels in a state space."""
    sel = _ss_selection_labels(ss)
    return _ss_labels(ss) - sel


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def check_downgrade(
    s_required: SessionType,
    s_offered: SessionType,
) -> bool:
    """Check if s_offered is a safe replacement for s_required.

    A safe replacement means the offered protocol is at least as
    capable as the required one — it is a subtype in the Gay-Hole
    sense (offers at least the required methods).

    Args:
        s_required: The required (strong) protocol type.
        s_offered: The offered (potentially weak) protocol type.

    Returns:
        True if s_offered is a safe replacement (subtype) of s_required.
    """
    return is_subtype(s_offered, s_required)


def downgrade_risk(
    s_strong: SessionType,
    s_weak: SessionType,
) -> DowngradeRisk:
    """Quantify the risk of downgrading from s_strong to s_weak.

    Measures what capabilities are lost when using s_weak instead
    of s_strong.

    Args:
        s_strong: The strong (required) protocol.
        s_weak: The weak (offered) protocol.

    Returns:
        DowngradeRisk with detailed capability comparison.
    """
    strong_methods = _collect_branch_labels(s_strong)
    weak_methods = _collect_branch_labels(s_weak)
    strong_selections = _collect_select_labels(s_strong)
    weak_selections = _collect_select_labels(s_weak)

    lost_m = strong_methods - weak_methods
    retained_m = strong_methods & weak_methods
    lost_s = strong_selections - weak_selections
    retained_s = strong_selections & weak_selections

    total_capabilities = len(strong_methods) + len(strong_selections)
    lost_capabilities = len(lost_m) + len(lost_s)

    risk_score = lost_capabilities / total_capabilities if total_capabilities > 0 else 0.0

    ss_strong = build_statespace(s_strong)
    ss_weak = build_statespace(s_weak)
    depth_strong = _max_path_depth(ss_strong)
    depth_weak = _max_path_depth(ss_weak)

    return DowngradeRisk(
        lost_methods=frozenset(lost_m),
        lost_selections=frozenset(lost_s),
        retained_methods=frozenset(retained_m),
        retained_selections=frozenset(retained_s),
        risk_score=risk_score,
        depth_reduction=depth_strong - depth_weak,
    )


def safe_downgrade_set(
    s: SessionType,
    *,
    candidates: list[SessionType] | None = None,
) -> list[SessionType]:
    """Find all safe replacement types from candidates.

    A safe replacement is any type that is a subtype of s
    (at least as capable).

    Args:
        s: The required protocol type.
        candidates: List of candidate types to check.
                    If None, generates trivial subtypes.

    Returns:
        List of session types that are safe replacements.
    """
    if candidates is None:
        # Generate the type itself and end as trivial candidates
        candidates = [s, End()]

    safe: list[SessionType] = []
    for candidate in candidates:
        if is_subtype(candidate, s):
            safe.append(candidate)

    return safe


def detect_forced_downgrade(
    ss: StateSpace,
    attacker_choices: set[str] | frozenset[str],
) -> ForcedDowngradeResult:
    """Detect if an attacker can force a weaker protocol path.

    An attacker who controls certain selection labels can force the
    protocol down specific paths.  This function checks whether any
    of those paths lead to states with fewer capabilities than the
    protocol's full potential.

    Args:
        ss: The state space of the protocol.
        attacker_choices: Labels the attacker can choose.

    Returns:
        ForcedDowngradeResult with vulnerability assessment.
    """
    atk = frozenset(attacker_choices)
    all_labels = _ss_labels(ss)
    non_attacker = all_labels - atk

    # Find states reachable when attacker controls their labels
    # The attacker can force transitions on their labels
    forced_paths: list[list[tuple[int, str, int]]] = []
    weakest: int | None = None
    min_capability = len(all_labels) + 1

    # BFS from top, attacker chooses their labels
    visited: set[int] = set()
    queue: list[tuple[int, list[tuple[int, str, int]]]] = [(ss.top, [])]

    while queue:
        state, path = queue.pop(0)
        if state in visited:
            continue
        visited.add(state)

        enabled = ss.enabled(state)
        if not enabled:
            # Terminal state — check capability
            capability = 0
            if capability < min_capability:
                min_capability = capability
                weakest = state
            if path:
                forced_paths.append(path)
            continue

        # Check capability at this state
        capability = len([l for l, _ in enabled if l not in atk])
        if capability < min_capability:
            min_capability = capability
            weakest = state

        # Attacker controls their labels, follows one path
        attacker_transitions = [(l, t) for l, t in enabled if l in atk]
        other_transitions = [(l, t) for l, t in enabled if l not in atk]

        if attacker_transitions:
            # Attacker can choose — they pick the "worst" path
            for label, tgt in attacker_transitions:
                new_path = path + [(state, label, tgt)]
                queue.append((tgt, new_path))
        elif other_transitions:
            # Non-attacker choices — all branches explored
            for label, tgt in other_transitions:
                new_path = path + [(state, label, tgt)]
                queue.append((tgt, new_path))

    # Determine if the attacker can reach a state with reduced capability
    # compared to the full protocol potential
    full_capability = len(all_labels)
    is_vuln = min_capability < full_capability and len(atk) > 0 and bool(forced_paths)

    return ForcedDowngradeResult(
        is_vulnerable=is_vuln,
        attacker_labels=atk,
        forced_paths=forced_paths,
        weakest_reachable=weakest,
    )


def analyze_downgrade(
    s_required: SessionType,
    s_offered: SessionType,
    *,
    candidates: list[SessionType] | None = None,
) -> DowngradeResult:
    """Full downgrade analysis between required and offered protocols.

    Args:
        s_required: The required (strong) protocol type.
        s_offered: The offered (potentially weak) protocol type.
        candidates: Optional list of candidate replacement types.

    Returns:
        DowngradeResult with comprehensive analysis.
    """
    is_safe = check_downgrade(s_required, s_offered)
    is_sub = is_subtype(s_offered, s_required)
    risk = downgrade_risk(s_required, s_offered)

    ss_req = build_statespace(s_required)
    ss_off = build_statespace(s_offered)

    safe_set = safe_downgrade_set(s_required, candidates=candidates)
    safe_strs = [pretty(t) for t in safe_set]

    return DowngradeResult(
        is_safe=is_safe,
        is_subtype_relation=is_sub,
        risk=risk,
        required_type=pretty(s_required),
        offered_type=pretty(s_offered),
        strong_state_count=len(ss_req.states),
        weak_state_count=len(ss_off.states),
        safe_replacements=safe_strs,
    )
