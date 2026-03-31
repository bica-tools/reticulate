"""Local modularity analysis for session type state spaces (Step 302).

Investigates LOCAL modularity in session type lattices using five definitions:
  1. Interval distributivity: [s, bottom] is distributive
  2. Transition independence: branches have disjoint reachable sets (except bottom)
  3. Meet/join locality: meet/join of successors equals the state or bottom
  4. Change containment: changes below a state do not propagate upward (WINNER)
  5. Depth symmetry: all outgoing branches have equal max depth

Key findings:
  - DEF 4 (change containment) is the right operational definition
  - A FAULT STATE is where changes below propagate upward:
    non-distributive interval but all successor intervals are distributive
  - MODULAR COVERAGE = fraction of states with distributive downward intervals
  - Branch M3 (3+ endpoints sharing terminal) has 100% modular coverage
  - N5 from nesting has specific fault states at asymmetric branch depths
  - Parallel products are always distributive -- no fault states
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from reticulate.lattice import (
    check_lattice,
    compute_meet,
    compute_join,
    _build_quotient_poset,
    _QuotientPoset,
    _meet_on_quotient,
    _join_on_quotient,
)

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Module:
    """A maximal distributive interval -- an independently modifiable part."""

    root: int                    # top state of the module
    states: frozenset[int]       # all states in the module
    labels: frozenset[str]       # transition labels within the module
    size: int


@dataclass(frozen=True)
class ModuleDecomposition:
    """Decomposition of a lattice into distributive modules + glue."""

    modules: list[Module]
    glue_states: frozenset[int]       # states not in any module
    fault_states: frozenset[int]      # glue states that are fault states
    dispatch_states: frozenset[int]   # glue states that are just choice points (harmless)
    num_modules: int
    is_fully_modular: bool            # glue has only dispatch states, no faults
    explanation: str


@dataclass(frozen=True)
class StateModularity:
    """Modularity analysis for a single state."""

    state: int
    is_modular: bool            # interval [s, bottom] is distributive
    is_fault: bool              # non-dist interval but all successors are dist
    interval_size: int          # number of states in [s, bottom]
    successor_labels: list[str]  # outgoing transition labels
    branch_depths: dict[str, int] | None  # label -> max depth (if branching)
    depth_symmetric: bool       # all branches same depth


@dataclass(frozen=True)
class LocalModularityResult:
    """Complete local modularity analysis."""

    is_globally_distributive: bool
    is_locally_modular: bool    # no fault states
    modular_coverage: float     # fraction of states that are modular (0.0-1.0)
    total_states: int
    modular_states: int
    fault_states: list[int]
    fault_labels: dict[int, list[str]]  # fault state -> branching labels
    per_state: dict[int, StateModularity]
    explanation: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _reachable_set(ss: StateSpace, state: int) -> set[int]:
    """All states reachable from *state* (inclusive)."""
    visited: set[int] = set()
    stack = [state]
    while stack:
        s = stack.pop()
        if s in visited:
            continue
        visited.add(s)
        for src, _, tgt in ss.transitions:
            if src == s:
                stack.append(tgt)
    return visited


def _interval_states(ss: StateSpace, top_state: int) -> set[int]:
    """States in the interval [top_state, bottom]."""
    return _reachable_set(ss, top_state)


def _max_depth(ss: StateSpace, state: int) -> int:
    """Max depth from state to bottom (longest path in DAG sense).

    Uses BFS/DFS with memoisation. For cyclic state spaces,
    caps at len(states) to avoid infinite loops.
    """
    memo: dict[int, int] = {}
    max_states = len(ss.states)

    def _depth(s: int, visited: frozenset[int]) -> int:
        if s == ss.bottom:
            return 0
        if s in memo:
            return memo[s]
        if s in visited:
            return 0  # cycle: don't count
        succs = [(lbl, tgt) for src, lbl, tgt in ss.transitions if src == s]
        if not succs:
            return 0
        new_visited = visited | {s}
        d = max(_depth(tgt, new_visited) + 1 for _, tgt in succs)
        if len(visited) == 0:
            memo[s] = d
        return d

    return _depth(state, frozenset())


def _outgoing(ss: StateSpace, state: int) -> list[tuple[str, int]]:
    """(label, target) pairs for transitions from state."""
    return [(lbl, tgt) for src, lbl, tgt in ss.transitions if src == state]


def _check_interval_distributive(
    ss: StateSpace,
    top_state: int,
) -> bool:
    """Check if the interval [top_state, bottom] forms a distributive lattice.

    Builds a sub-StateSpace from the interval and checks for M3/N5 sublattices.
    """
    interval = _interval_states(ss, top_state)

    if len(interval) <= 2:
        # Trivially distributive (chain of length 0 or 1)
        return True

    # Build a sub-state-space for the interval
    from reticulate.statespace import StateSpace as SS
    sub_transitions = [
        (s, l, t) for s, l, t in ss.transitions
        if s in interval and t in interval
    ]
    sub_ss = SS(
        states=interval,
        transitions=sub_transitions,
        top=top_state,
        bottom=ss.bottom,
        labels={s: ss.labels.get(s, "") for s in interval},
        selection_transitions={
            (s, l, t) for s, l, t in ss.selection_transitions
            if s in interval and t in interval
        },
    )

    # Check lattice first
    lr = check_lattice(sub_ss)
    if not lr.is_lattice:
        # Not even a lattice -- not distributive
        return False

    # Check for M3 and N5 sublattices via the quotient
    q = _build_quotient_poset(sub_ss)
    from reticulate.lattice import _find_m3, _find_n5
    return _find_m3(q) is None and _find_n5(q) is None


# ---------------------------------------------------------------------------
# Five definitions (for comparison)
# ---------------------------------------------------------------------------

def def1_interval_distributive(ss: StateSpace, state: int) -> bool:
    """DEF 1: Interval [s, bottom] is distributive."""
    return _check_interval_distributive(ss, state)


def def2_transition_independent(ss: StateSpace, state: int) -> bool:
    """DEF 2: Outgoing branches have disjoint reachable sets (except bottom)."""
    succs = _outgoing(ss, state)
    if len(succs) <= 1:
        return True
    reach_sets: list[set[int]] = []
    for _, tgt in succs:
        r = _reachable_set(ss, tgt) - {ss.bottom}
        reach_sets.append(r)
    for i in range(len(reach_sets)):
        for j in range(i + 1, len(reach_sets)):
            if reach_sets[i] & reach_sets[j]:
                return False
    return True


def def3_meet_join_local(ss: StateSpace, state: int) -> bool:
    """DEF 3: Meet/join of successor pairs equals state or bottom."""
    succs = _outgoing(ss, state)
    if len(succs) <= 1:
        return True
    targets = [tgt for _, tgt in succs]
    for i in range(len(targets)):
        for j in range(i + 1, len(targets)):
            m = compute_meet(ss, targets[i], targets[j])
            j_val = compute_join(ss, targets[i], targets[j])
            if m is not None and m != ss.bottom:
                # Meet should be bottom for independent branches
                return False
            if j_val is not None and j_val != state:
                # Join should be the state itself
                return False
    return True


def def4_change_contained(ss: StateSpace, state: int) -> bool:
    """DEF 4: Change containment -- the winning definition.

    A state is change-contained if its interval [s, bottom] is distributive.
    This is identical to DEF 1, but with the additional interpretation that
    non-modular states where all successors ARE modular are 'fault states'.
    """
    return _check_interval_distributive(ss, state)


def def5_depth_symmetric(ss: StateSpace, state: int) -> bool:
    """DEF 5: All outgoing branches have equal max depth."""
    succs = _outgoing(ss, state)
    if len(succs) <= 1:
        return True
    depths = [_max_depth(ss, tgt) for _, tgt in succs]
    return len(set(depths)) == 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_interval_distributive(ss: StateSpace, top_state: int) -> bool:
    """Check if interval [top_state, bottom] is distributive."""
    return _check_interval_distributive(ss, top_state)


def find_fault_states(ss: StateSpace) -> list[int]:
    """Find states where changes propagate upward.

    A fault state is one whose interval [s, bottom] is NOT distributive,
    but ALL successor intervals [succ, bottom] ARE distributive.
    This is the boundary where non-distributivity first appears.
    """
    faults: list[int] = []
    modular_cache: dict[int, bool] = {}

    def _is_modular(s: int) -> bool:
        if s not in modular_cache:
            modular_cache[s] = _check_interval_distributive(ss, s)
        return modular_cache[s]

    for state in sorted(ss.states):
        if state == ss.bottom:
            continue
        if _is_modular(state):
            continue
        # Non-modular state: check if all successors are modular
        succs = _outgoing(ss, state)
        if not succs:
            continue
        all_succ_modular = all(_is_modular(tgt) for _, tgt in succs)
        if all_succ_modular:
            faults.append(state)

    return faults


def modular_coverage(ss: StateSpace) -> float:
    """Fraction of states with distributive downward intervals."""
    if not ss.states:
        return 1.0
    total = len(ss.states)
    modular = sum(
        1 for s in ss.states
        if _check_interval_distributive(ss, s)
    )
    return modular / total


def change_impact(ss: StateSpace, state: int) -> dict:
    """Analyze what changing a transition at 'state' would affect.

    Returns:
        dict with keys:
          - 'state': the state being analyzed
          - 'interval_size': number of states in [state, bottom]
          - 'is_modular': whether the interval is distributive
          - 'affected_states': states in the interval
          - 'upstream_states': states that can reach this state
          - 'propagation_risk': 'none' | 'contained' | 'propagates'
    """
    interval = _interval_states(ss, state)
    is_mod = _check_interval_distributive(ss, state)

    # Compute upstream: states that can reach this state
    upstream: set[int] = set()
    for s in ss.states:
        if state in _reachable_set(ss, s):
            upstream.add(s)
    upstream.discard(state)

    if is_mod:
        risk = "none"
    else:
        fault_states = find_fault_states(ss)
        if state in fault_states:
            risk = "contained"
        else:
            risk = "propagates"

    return {
        "state": state,
        "interval_size": len(interval),
        "is_modular": is_mod,
        "affected_states": sorted(interval),
        "upstream_states": sorted(upstream),
        "propagation_risk": risk,
    }


def decompose_modules(ss: StateSpace) -> ModuleDecomposition:
    """Decompose lattice into maximal distributive intervals (modules) + glue.

    Algorithm:
    1. For each state s (bottom-up by interval size), check if [s, bottom] is distributive.
    2. A module root is a state where [s, bottom] is distributive but no parent's
       interval is distributive (maximal distributive interval boundary).
    3. Glue = states not in any module.
    4. Fault states = glue states whose interval is non-distributive but all
       successor intervals are distributive.
    5. Dispatch states = glue states where all successors are module roots or
       in some module (harmless routing states).
    """
    lr = check_lattice(ss)
    if not lr.is_lattice:
        return ModuleDecomposition(
            modules=[],
            glue_states=frozenset(ss.states),
            fault_states=frozenset(),
            dispatch_states=frozenset(),
            num_modules=0,
            is_fully_modular=False,
            explanation="State space is not a lattice; decomposition not applicable.",
        )

    # Step 1: compute distributivity of every interval [s, bottom]
    modular_cache: dict[int, bool] = {}
    interval_cache: dict[int, set[int]] = {}
    for state in ss.states:
        interval_cache[state] = _interval_states(ss, state)
        modular_cache[state] = _check_interval_distributive(ss, state)

    # Step 2: find module roots — maximal distributive intervals.
    # A state s is a module root if:
    #   (a) [s, bottom] is distributive, AND
    #   (b) for every parent p of s, [p, bottom] is NOT distributive
    #       (or s is the top and [top, bottom] is distributive)
    # Parents = states with a direct transition into s.
    parents_of: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        parents_of[tgt].add(src)

    # Detect states reachable from top (to handle cycles properly)
    reachable_from_top = _reachable_set(ss, ss.top)

    module_roots: list[int] = []
    for state in sorted(ss.states, key=lambda s: len(interval_cache[s])):
        if not modular_cache[state]:
            continue
        # The top state is always a module root if distributive
        if state == ss.top:
            module_roots.append(state)
            continue
        # Check if any parent also has a distributive interval,
        # excluding parents that are in the same SCC (cycle)
        # A parent in the same SCC means it's reachable from state too
        state_reach = _reachable_set(ss, state)
        has_dist_parent = False
        for p in parents_of[state]:
            if p == state:
                continue
            # If p is reachable from state, they're in a cycle -- skip
            if p in state_reach:
                continue
            if modular_cache.get(p, False):
                has_dist_parent = True
                break
        if not has_dist_parent:
            module_roots.append(state)

    # Step 3: build modules. Each module root claims its interval.
    # If intervals overlap, higher (larger interval) module wins.
    # Sort roots by interval size descending so larger modules are built first.
    module_roots.sort(key=lambda s: len(interval_cache[s]), reverse=True)

    claimed: set[int] = set()
    modules: list[Module] = []
    for root in module_roots:
        interval = interval_cache[root]
        # Module states = interval minus states already claimed by a larger module,
        # BUT we always include bottom (shared).
        module_states = frozenset(
            s for s in interval if s not in claimed or s == ss.bottom
        )
        if len(module_states) <= 1 and module_states == {ss.bottom}:
            # Degenerate: only bottom. Skip unless root IS bottom.
            if root != ss.bottom:
                continue
        # Compute labels within module
        module_labels = frozenset(
            lbl for src, lbl, tgt in ss.transitions
            if src in module_states and tgt in module_states
        )
        modules.append(Module(
            root=root,
            states=module_states,
            labels=module_labels,
            size=len(module_states),
        ))
        claimed.update(module_states)

    # Step 4: glue = states not in any module
    all_module_states = set()
    for m in modules:
        all_module_states.update(m.states)
    glue = frozenset(ss.states - all_module_states)

    # Step 5: classify glue states
    module_root_set = {m.root for m in modules}
    fault_set: set[int] = set()
    dispatch_set: set[int] = set()

    for g in glue:
        succs = _outgoing(ss, g)
        if not succs:
            # Isolated glue state (shouldn't normally happen)
            continue
        # Is it a fault state? Non-distributive but all successors distributive
        all_succ_dist = all(modular_cache.get(tgt, True) for _, tgt in succs)
        if all_succ_dist:
            fault_set.add(g)
        # Is it a dispatch state? All successors lead into modules
        all_succ_in_modules = all(
            tgt in all_module_states or tgt in module_root_set
            for _, tgt in succs
        )
        if all_succ_in_modules:
            dispatch_set.add(g)

    fault_states = frozenset(fault_set)
    dispatch_states = frozenset(dispatch_set)
    # Fully modular if no glue, or all glue states are dispatch (harmless routing)
    is_fully_modular = len(glue) == 0 or glue == dispatch_states

    # Explanation
    if len(glue) == 0:
        explanation = (
            f"Fully modular: {len(modules)} module(s) cover all "
            f"{len(ss.states)} states."
        )
    elif is_fully_modular:
        explanation = (
            f"{len(modules)} module(s) with {len(glue)} dispatch-only glue "
            f"state(s). Fully modular (no fault states in glue)."
        )
    else:
        explanation = (
            f"{len(modules)} module(s) with {len(glue)} glue state(s) "
            f"({len(fault_set)} fault, {len(dispatch_set)} dispatch). "
            f"Not fully modular."
        )

    return ModuleDecomposition(
        modules=modules,
        glue_states=glue,
        fault_states=fault_states,
        dispatch_states=dispatch_states,
        num_modules=len(modules),
        is_fully_modular=is_fully_modular,
        explanation=explanation,
    )


def analyze_local_modularity(ss: StateSpace) -> LocalModularityResult:
    """Complete local modularity analysis."""
    lr = check_lattice(ss)
    if not lr.is_lattice:
        return LocalModularityResult(
            is_globally_distributive=False,
            is_locally_modular=False,
            modular_coverage=0.0,
            total_states=len(ss.states),
            modular_states=0,
            fault_states=[],
            fault_labels={},
            per_state={},
            explanation="State space is not a lattice; modularity analysis not applicable.",
        )

    # Check global distributivity
    from reticulate.lattice import check_distributive
    dist = check_distributive(ss)
    is_globally_dist = dist.is_distributive

    # Per-state analysis
    modular_cache: dict[int, bool] = {}
    per_state: dict[int, StateModularity] = {}

    for state in sorted(ss.states):
        is_mod = _check_interval_distributive(ss, state)
        modular_cache[state] = is_mod

    # Now find fault states
    faults: list[int] = []
    for state in sorted(ss.states):
        if state == ss.bottom:
            continue
        if modular_cache[state]:
            continue
        succs = _outgoing(ss, state)
        if not succs:
            continue
        all_succ_modular = all(
            modular_cache.get(tgt, True) for _, tgt in succs
        )
        if all_succ_modular:
            faults.append(state)

    # Build fault labels
    fault_labels: dict[int, list[str]] = {}
    for f in faults:
        labels = [lbl for lbl, _ in _outgoing(ss, f)]
        fault_labels[f] = labels

    # Build per-state results
    for state in sorted(ss.states):
        succs = _outgoing(ss, state)
        succ_labels = [lbl for lbl, _ in succs]

        # Branch depths
        branch_depths: dict[str, int] | None = None
        depth_sym = True
        if len(succs) > 1:
            branch_depths = {}
            for lbl, tgt in succs:
                branch_depths[lbl] = _max_depth(ss, tgt)
            depth_vals = list(branch_depths.values())
            depth_sym = len(set(depth_vals)) <= 1
        elif len(succs) == 1:
            lbl, tgt = succs[0]
            branch_depths = {lbl: _max_depth(ss, tgt)}

        interval = _interval_states(ss, state)

        per_state[state] = StateModularity(
            state=state,
            is_modular=modular_cache[state],
            is_fault=state in faults,
            interval_size=len(interval),
            successor_labels=succ_labels,
            branch_depths=branch_depths,
            depth_symmetric=depth_sym,
        )

    modular_count = sum(1 for s in modular_cache.values() if s)
    total = len(ss.states)
    coverage = modular_count / total if total > 0 else 1.0

    # Generate explanation
    if is_globally_dist:
        explanation = (
            f"Globally distributive lattice with {total} states. "
            f"All intervals are distributive -- no fault states."
        )
    elif not faults:
        explanation = (
            f"Non-distributive lattice with {total} states, "
            f"but no fault states detected (non-distributivity propagates from top). "
            f"Modular coverage: {coverage:.1%}."
        )
    else:
        explanation = (
            f"Non-distributive lattice with {total} states. "
            f"{len(faults)} fault state(s) at states {faults}. "
            f"Modular coverage: {coverage:.1%}."
        )

    return LocalModularityResult(
        is_globally_distributive=is_globally_dist,
        is_locally_modular=len(faults) == 0,
        modular_coverage=coverage,
        total_states=total,
        modular_states=modular_count,
        fault_states=faults,
        fault_labels=fault_labels,
        per_state=per_state,
        explanation=explanation,
    )
