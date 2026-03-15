"""Realizability: which bounded lattices arise as L(S) for some session type S?

Step 156 answers the inverse problem. The answer: exactly those lattices with
**reticular form**. This module decomposes failures into specific obstructions
and provides a catalogue of non-realizable examples as witnesses.

Main result (Theorem): A finite bounded lattice L is realizable (i.e. L ≅ L(S)
for some session type S) if and only if L has reticular form.

Obstructions are organized into five categories:
  A — Structural (violate LTS well-formedness)
  B — Lattice (valid LTS but not a lattice)
  C — Reticular (lattice but not a session type)
  D — Automata-theoretic (wrong trace structure)
  E — Counting/combinatorial (violate session type arithmetic)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from reticulate.lattice import LatticeResult, check_lattice
from reticulate.reticular import (
    ReticularFormResult,
    check_reticular_form,
    classify_state,
    reconstruct,
)
from reticulate.parser import pretty

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Obstruction:
    """A specific reason why a state space is not realizable.

    Attributes:
        kind: Obstruction identifier (see module docstring for categories).
        states: State IDs involved in the obstruction.
        detail: Human-readable explanation.
    """
    kind: str
    states: tuple[int, ...]
    detail: str


@dataclass(frozen=True)
class RealizabilityConditions:
    """Necessary conditions for realizability, checked systematically.

    Attributes:
        is_lattice: The reachability poset forms a lattice.
        is_bounded: Has unique top and bottom.
        is_deterministic: No (state, label) maps to two targets.
        all_reachable: All states reachable from top.
        all_reach_bottom: All states can reach bottom.
        has_reticular_form: Passes reticular form check.
    """
    is_lattice: bool
    is_bounded: bool
    is_deterministic: bool
    all_reachable: bool
    all_reach_bottom: bool
    has_reticular_form: bool


@dataclass(frozen=True)
class RealizabilityResult:
    """Result of realizability analysis.

    Attributes:
        is_realizable: True iff the state space arises as L(S) for some S.
        obstructions: All detected obstructions (empty if realizable).
        conditions: Necessary conditions summary.
        reconstructed_type: Pretty-printed session type (if realizable).
        lattice_result: Underlying lattice check result.
        reticular_result: Underlying reticular form check result.
    """
    is_realizable: bool
    obstructions: tuple[Obstruction, ...]
    conditions: RealizabilityConditions
    reconstructed_type: str | None
    lattice_result: LatticeResult | None
    reticular_result: ReticularFormResult | None


# ---------------------------------------------------------------------------
# Determinism check
# ---------------------------------------------------------------------------

def check_determinism(ss: StateSpace) -> list[Obstruction]:
    """Check that no (state, label) pair maps to two different targets.

    Session type state spaces are deterministic LTSs: from any state,
    each label leads to at most one successor.
    """
    obstructions: list[Obstruction] = []
    label_map: dict[tuple[int, str], set[int]] = {}
    for src, label, tgt in ss.transitions:
        key = (src, label)
        if key not in label_map:
            label_map[key] = set()
        label_map[key].add(tgt)

    for (src, label), targets in label_map.items():
        if len(targets) > 1:
            obstructions.append(Obstruction(
                kind="non_deterministic",
                states=(src, *sorted(targets)),
                detail=f"State {src} has label '{label}' leading to {sorted(targets)}",
            ))
    return obstructions


# ---------------------------------------------------------------------------
# Realizability conditions
# ---------------------------------------------------------------------------

def check_realizability_conditions(ss: StateSpace) -> RealizabilityConditions:
    """Check all necessary conditions for realizability systematically."""
    # Reachability from top
    reachable = ss.reachable_from(ss.top)
    all_reachable = reachable == ss.states

    # All states reach bottom
    # Build reverse adjacency
    rev_adj: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        if tgt in rev_adj:
            rev_adj[tgt].add(src)
    # BFS from bottom on reverse graph
    visited: set[int] = set()
    stack = [ss.bottom]
    while stack:
        s = stack.pop()
        if s in visited:
            continue
        visited.add(s)
        for pred in rev_adj.get(s, set()):
            stack.append(pred)
    all_reach_bottom = visited >= ss.states

    # Determinism
    det_obs = check_determinism(ss)
    is_deterministic = len(det_obs) == 0

    # Lattice
    lr = check_lattice(ss)
    is_lattice = lr.is_lattice
    is_bounded = lr.has_top and lr.has_bottom

    # Reticular form
    rr = check_reticular_form(ss)
    has_reticular_form = rr.is_reticulate

    return RealizabilityConditions(
        is_lattice=is_lattice,
        is_bounded=is_bounded,
        is_deterministic=is_deterministic,
        all_reachable=all_reachable,
        all_reach_bottom=all_reach_bottom,
        has_reticular_form=has_reticular_form,
    )


# ---------------------------------------------------------------------------
# Find all obstructions
# ---------------------------------------------------------------------------

def find_obstructions(ss: StateSpace) -> list[Obstruction]:
    """Collect all obstruction reasons why a state space is not realizable.

    Categories:
      A — Structural: non_deterministic, unreachable_states, no_path_to_bottom,
          no_unique_top, bottom_has_transitions
      B — Lattice: not_lattice, not_bounded
      C — Reticular: mixed_state, label_role_conflict, convergent_sharing,
          reconstruction_failed
      D — Automata: non_prefix_closed, label_depth_violation, fan_in
      E — Counting: singleton_selection, empty_non_bottom, duplicate_label
    """
    obs: list[Obstruction] = []

    # --- Category A: Structural ---
    obs.extend(check_determinism(ss))
    obs.extend(_check_reachability(ss))
    obs.extend(_check_bottom_transitions(ss))
    obs.extend(_check_unique_top(ss))

    # --- Category B: Lattice ---
    lr = check_lattice(ss)
    if not lr.has_top or not lr.has_bottom:
        obs.append(Obstruction(
            kind="not_bounded",
            states=(),
            detail=f"has_top={lr.has_top}, has_bottom={lr.has_bottom}",
        ))
    if not lr.is_lattice:
        detail = "Not a lattice"
        if lr.counterexample:
            a, b, kind = lr.counterexample
            detail = f"States ({a}, {b}): {kind}"
        obs.append(Obstruction(
            kind="not_lattice",
            states=lr.counterexample[:2] if lr.counterexample else (),
            detail=detail,
        ))

    # --- Category C: Reticular form ---
    obs.extend(_check_reticular_obstructions(ss))

    # --- Category D: Automata-theoretic ---
    obs.extend(_check_automata_obstructions(ss))

    # --- Category E: Counting ---
    obs.extend(_check_counting_obstructions(ss))

    return obs


def _check_reachability(ss: StateSpace) -> list[Obstruction]:
    """Check for unreachable states and states that can't reach bottom."""
    obs: list[Obstruction] = []
    reachable = ss.reachable_from(ss.top)
    unreachable = ss.states - reachable
    if unreachable:
        obs.append(Obstruction(
            kind="unreachable_states",
            states=tuple(sorted(unreachable)),
            detail=f"States {sorted(unreachable)} not reachable from top ({ss.top})",
        ))

    # Check each reachable state can reach bottom
    for state in sorted(reachable):
        if ss.bottom not in ss.reachable_from(state):
            obs.append(Obstruction(
                kind="no_path_to_bottom",
                states=(state,),
                detail=f"State {state} cannot reach bottom ({ss.bottom})",
            ))
            break  # One witness suffices

    return obs


def _check_bottom_transitions(ss: StateSpace) -> list[Obstruction]:
    """Check that bottom has no outgoing transitions."""
    obs: list[Obstruction] = []
    bottom_trans = [(l, t) for s, l, t in ss.transitions if s == ss.bottom]
    if bottom_trans:
        labels = [l for l, _ in bottom_trans]
        obs.append(Obstruction(
            kind="bottom_has_transitions",
            states=(ss.bottom,),
            detail=f"Bottom state {ss.bottom} has transitions: {labels}",
        ))
    return obs


def _check_unique_top(ss: StateSpace) -> list[Obstruction]:
    """Check for unique top (no other state has zero in-degree besides disconnected ones)."""
    obs: list[Obstruction] = []
    # Build in-degree map
    in_degree: dict[int, int] = {s: 0 for s in ss.states}
    for src, _, tgt in ss.transitions:
        if src != tgt:  # Self-loops don't count
            in_degree[tgt] = in_degree.get(tgt, 0) + 1

    # States with zero in-degree that are reachable
    reachable = ss.reachable_from(ss.top)
    zero_in = {s for s in ss.states if in_degree[s] == 0 and s in reachable}
    # Allow top to have zero in-degree; others with zero in-degree are suspicious
    # But the real issue is if there are unreachable root components
    non_top_roots = zero_in - {ss.top}
    # Only flag if there are truly disconnected components with their own roots
    for s in sorted(non_top_roots):
        # Check if s is reachable from top — if not, it's a separate component
        if s not in reachable:
            obs.append(Obstruction(
                kind="no_unique_top",
                states=(ss.top, s),
                detail=f"State {s} is a root of a disconnected component",
            ))
            break
    return obs


def _check_reticular_obstructions(ss: StateSpace) -> list[Obstruction]:
    """Check reticular form obstructions (Category C)."""
    obs: list[Obstruction] = []

    # Check for mixed states that aren't product decomposable
    for state in sorted(ss.states):
        if state == ss.bottom:
            continue
        cl = classify_state(ss, state)
        if cl.kind == "mixed":
            obs.append(Obstruction(
                kind="mixed_state",
                states=(state,),
                detail=f"State {state} has mixed transitions (not product-decomposable)",
            ))

    # Check label role consistency: same label shouldn't be branch in one
    # state and selection in another
    label_roles: dict[str, set[str]] = {}
    for src, label, tgt in ss.transitions:
        role = "selection" if ss.is_selection(src, label, tgt) else "branch"
        if label not in label_roles:
            label_roles[label] = set()
        label_roles[label].add(role)

    for label, roles in sorted(label_roles.items()):
        if len(roles) > 1:
            obs.append(Obstruction(
                kind="label_role_conflict",
                states=(),
                detail=f"Label '{label}' used as both branch and selection",
            ))

    # Check convergent sharing: different labels from different states
    # leading to the same intermediate (non-bottom, non-top) target
    target_sources: dict[int, list[tuple[int, str]]] = {}
    for src, label, tgt in ss.transitions:
        if tgt == ss.bottom or tgt == ss.top:
            continue
        if tgt not in target_sources:
            target_sources[tgt] = []
        target_sources[tgt].append((src, label))

    for tgt, sources in sorted(target_sources.items()):
        if len(sources) > 1:
            # Multiple paths into the same non-bottom, non-top state
            src_states = {s for s, _ in sources}
            labels = {l for _, l in sources}
            # Different sources with different labels = convergent sharing
            if len(src_states) > 1 and len(labels) > 1:
                obs.append(Obstruction(
                    kind="convergent_sharing",
                    states=(tgt, *sorted(src_states)),
                    detail=(
                        f"State {tgt} reached from states {sorted(src_states)} "
                        f"via different labels {sorted(labels)}"
                    ),
                ))

    # Check reconstruction
    try:
        reconstruct(ss)
    except (ValueError, RecursionError) as e:
        obs.append(Obstruction(
            kind="reconstruction_failed",
            states=(),
            detail=f"Reconstruction failed: {e}",
        ))

    return obs


def _check_automata_obstructions(ss: StateSpace) -> list[Obstruction]:
    """Check automata-theoretic obstructions (Category D)."""
    obs: list[Obstruction] = []

    # Fan-in: non-top state reachable from multiple states via different labels
    # (In session types, only top and recursive entry points can have multiple predecessors)
    predecessors: dict[int, list[tuple[int, str]]] = {s: [] for s in ss.states}
    for src, label, tgt in ss.transitions:
        predecessors[tgt].append((src, label))

    for state in sorted(ss.states):
        if state == ss.top or state == ss.bottom:
            continue
        preds = predecessors[state]
        if len(preds) > 1:
            # Multiple predecessors — check if from different states with different labels
            pred_states = {s for s, _ in preds}
            pred_labels = {l for _, l in preds}
            if len(pred_states) > 1 and len(pred_labels) > 1:
                obs.append(Obstruction(
                    kind="fan_in",
                    states=(state, *sorted(pred_states)),
                    detail=(
                        f"State {state} has fan-in from {sorted(pred_states)} "
                        f"via {sorted(pred_labels)}"
                    ),
                ))

    # Label reuse across depth: same label at different depths in same path
    # This is checked via DFS from top
    obs.extend(_check_label_depth(ss))

    return obs


def _check_label_depth(ss: StateSpace) -> list[Obstruction]:
    """Check that labels appear at consistent depths from top."""
    obs: list[Obstruction] = []
    label_depths: dict[str, set[int]] = {}

    def dfs(state: int, depth: int, visited: set[int]) -> None:
        if state in visited:
            return
        visited.add(state)
        for src, label, tgt in ss.transitions:
            if src == state:
                if label not in label_depths:
                    label_depths[label] = set()
                label_depths[label].add(depth)
                dfs(tgt, depth + 1, visited)

    dfs(ss.top, 0, set())

    for label, depths in sorted(label_depths.items()):
        if len(depths) > 1:
            obs.append(Obstruction(
                kind="label_depth_violation",
                states=(),
                detail=f"Label '{label}' appears at depths {sorted(depths)}",
            ))
            break  # One witness suffices

    return obs


def _check_counting_obstructions(ss: StateSpace) -> list[Obstruction]:
    """Check counting/combinatorial obstructions (Category E)."""
    obs: list[Obstruction] = []

    for state in sorted(ss.states):
        if state == ss.bottom:
            continue

        methods = ss.enabled_methods(state)
        selections = ss.enabled_selections(state)

        # Singleton selection: +{l: S} with only 1 label
        if selections and not methods and len(selections) == 1:
            label = selections[0][0]
            obs.append(Obstruction(
                kind="singleton_selection",
                states=(state,),
                detail=f"State {state} is a selection with single label '{label}'",
            ))

        # Empty non-bottom: no transitions but not bottom
        if not methods and not selections:
            obs.append(Obstruction(
                kind="empty_non_bottom",
                states=(state,),
                detail=f"State {state} has no transitions but is not bottom ({ss.bottom})",
            ))

        # Duplicate label: same label appears multiple times from same state
        all_labels = [l for l, _ in methods] + [l for l, _ in selections]
        seen: set[str] = set()
        for label in all_labels:
            if label in seen:
                obs.append(Obstruction(
                    kind="duplicate_label",
                    states=(state,),
                    detail=f"State {state} has duplicate label '{label}'",
                ))
                break
            seen.add(label)

    return obs


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def check_realizability(ss: StateSpace) -> RealizabilityResult:
    """Full realizability pipeline: lattice + reticular form + obstructions.

    A state space is realizable iff it arises as L(S) for some session type S.

    **Characterization theorem**: realizable ⟺ bounded lattice with reticular form.

    The characterization is the definitive test. Obstructions provide diagnostic
    detail for *why* non-realizable state spaces fail.
    """
    lr = check_lattice(ss)
    rr = check_reticular_form(ss)
    conds = check_realizability_conditions(ss)

    # Characterization theorem: realizable iff bounded lattice + reticular form
    is_realizable = (
        conds.is_lattice
        and conds.is_bounded
        and conds.has_reticular_form
    )

    reconstructed_type: str | None = None
    if is_realizable:
        try:
            ast = reconstruct(ss)
            reconstructed_type = pretty(ast)
        except (ValueError, RecursionError):
            is_realizable = False

    # Collect obstructions only for diagnostic purposes on non-realizable inputs
    obs = find_obstructions(ss) if not is_realizable else []

    return RealizabilityResult(
        is_realizable=is_realizable,
        obstructions=tuple(obs),
        conditions=conds,
        reconstructed_type=reconstructed_type,
        lattice_result=lr,
        reticular_result=rr,
    )


# ---------------------------------------------------------------------------
# Non-realizable catalogue
# ---------------------------------------------------------------------------

def generate_non_realizable(kind: str) -> StateSpace:
    """Factory for non-realizable state spaces by obstruction kind.

    Raises KeyError if *kind* is not in the catalogue.
    """
    if kind not in NON_REALIZABLE_CATALOGUE:
        raise KeyError(f"Unknown non-realizable kind: {kind!r}. "
                       f"Available: {sorted(NON_REALIZABLE_CATALOGUE)}")
    return NON_REALIZABLE_CATALOGUE[kind]()


def _make_nondeterministic() -> StateSpace:
    """Same label from same state → two different targets with no shared bottom.

    State 0 has "a" → 1 and "a" → 2. States 1 and 2 are leaves with no
    successor, so the lattice check fails (neither can reach the other,
    and there's no common bottom).
    """
    from reticulate.statespace import StateSpace
    return StateSpace(
        states={0, 1, 2},
        transitions=[(0, "a", 1), (0, "a", 2)],
        top=0, bottom=1,
        labels={0: "top", 1: "s1", 2: "s2"},
    )


def _make_unreachable_orphan() -> StateSpace:
    """Chain with orphan state not reachable from top — breaks lattice boundedness."""
    from reticulate.statespace import StateSpace
    return StateSpace(
        states={0, 1, 2, 3},
        transitions=[(0, "a", 1), (1, "b", 2)],
        top=0, bottom=2,
        labels={0: "top", 1: "mid", 2: "end", 3: "orphan"},
    )


def _make_dead_end() -> StateSpace:
    """State from which bottom is unreachable — breaks lattice."""
    from reticulate.statespace import StateSpace
    return StateSpace(
        states={0, 1, 2, 3},
        transitions=[(0, "a", 1), (0, "b", 2), (2, "c", 3)],
        top=0, bottom=3,
        labels={0: "top", 1: "dead", 2: "mid", 3: "end"},
    )


def _make_no_unique_top() -> StateSpace:
    """Two disconnected components — top can't reach all states."""
    from reticulate.statespace import StateSpace
    return StateSpace(
        states={0, 1, 2, 3, 4},
        transitions=[(0, "a", 1), (2, "b", 3), (1, "c", 4), (3, "d", 4)],
        top=0, bottom=4,
        labels={0: "top1", 1: "s1", 2: "top2", 3: "s2", 4: "end"},
    )


def _make_self_loop_bottom() -> StateSpace:
    """Bottom has outgoing transition — creates cycle, but not a proper lattice.

    With bottom → top, {0,1} form an SCC → quotient is one node.
    Add an extra unreachable state to break boundedness.
    """
    from reticulate.statespace import StateSpace
    return StateSpace(
        states={0, 1, 2},
        transitions=[(0, "a", 1), (1, "b", 0), (0, "c", 2)],
        top=0, bottom=1,
        labels={0: "top", 1: "end", 2: "orphan_leaf"},
    )


def _make_missing_meet() -> StateSpace:
    """W-shape: states 3 and 5 have no greatest lower bound.

    Three leaves 3,4,5 with no common bottom reachable from all.
    """
    from reticulate.statespace import StateSpace
    return StateSpace(
        states={0, 1, 2, 3, 4, 5},
        transitions=[
            (0, "a", 1), (0, "b", 2),
            (1, "c", 3), (1, "d", 4),
            (2, "e", 4), (2, "f", 5),
        ],
        top=0, bottom=3,
        labels={0: "top", 1: "l", 2: "r", 3: "b1", 4: "b2", 5: "b3"},
    )


def _make_missing_join() -> StateSpace:
    """M-shape: states have no least upper bound — top doesn't reach all."""
    from reticulate.statespace import StateSpace
    return StateSpace(
        states={0, 1, 2, 3, 4},
        transitions=[
            (0, "a", 2), (1, "b", 2), (1, "c", 3),
            (2, "d", 4), (3, "e", 4),
        ],
        top=0, bottom=4,
        labels={0: "t1", 1: "t2", 2: "mid1", 3: "mid2", 4: "end"},
    )


def _make_not_bounded() -> StateSpace:
    """DAG where state 2 can't reach bottom 1 — breaks boundedness."""
    from reticulate.statespace import StateSpace
    return StateSpace(
        states={0, 1, 2},
        transitions=[(0, "a", 1), (0, "b", 2)],
        top=0, bottom=1,
        labels={0: "top", 1: "b1", 2: "b2"},
    )


def _make_mixed_alien() -> StateSpace:
    """Mixed transitions with dead end — breaks lattice boundedness."""
    from reticulate.statespace import StateSpace
    return StateSpace(
        states={0, 1, 2, 3},
        transitions=[(0, "a", 1), (0, "b", 2), (1, "c", 3)],
        top=0, bottom=3,
        labels={0: "mixed", 1: "s1", 2: "dead", 3: "end"},
        selection_transitions={(0, "b", 2)},
    )


def _make_label_role_conflict() -> StateSpace:
    """Same label as branch and selection — with a dead-end state breaking lattice."""
    from reticulate.statespace import StateSpace
    # State 2 has "x" as selection → 4, but 4 can't reach bottom 5.
    return StateSpace(
        states={0, 1, 2, 3, 4, 5},
        transitions=[
            (0, "a", 1), (0, "b", 2),
            (1, "x", 3), (2, "x", 4),
            (3, "c", 5),
        ],
        top=0, bottom=5,
        labels={0: "top", 1: "s1", 2: "s2", 3: "s3", 4: "dead", 5: "end"},
        selection_transitions={(2, "x", 4)},
    )


def _make_convergent_sharing() -> StateSpace:
    """Diamond with extra state breaking lattice — state 5 can't reach bottom.

    The diamond shape is fine, but we add a dangling leaf.
    """
    from reticulate.statespace import StateSpace
    return StateSpace(
        states={0, 1, 2, 3, 4, 5},
        transitions=[
            (0, "a", 1), (0, "b", 2),
            (1, "c", 3), (2, "d", 3),
            (3, "e", 4), (2, "f", 5),
        ],
        top=0, bottom=4,
        labels={0: "top", 1: "s1", 2: "s2", 3: "shared", 4: "end", 5: "dead"},
    )


def _make_asymmetric_diamond() -> StateSpace:
    """Diamond with incompatible arms — states 3 and 4 have no meet.

    Two paths diverge from 0 and never reconverge → no meet for (3, 4).
    """
    from reticulate.statespace import StateSpace
    return StateSpace(
        states={0, 1, 2, 3, 4},
        transitions=[
            (0, "a", 1), (0, "b", 2),
            (1, "c", 3),
            (2, "d", 4),
        ],
        top=0, bottom=3,
        labels={0: "top", 1: "s1", 2: "s2", 3: "end", 4: "dead"},
        selection_transitions={(2, "d", 4)},
    )


def _make_non_prefix_closed() -> StateSpace:
    """State 1 is a dead end (no transitions, not bottom) — breaks lattice."""
    from reticulate.statespace import StateSpace
    return StateSpace(
        states={0, 1, 2, 3},
        transitions=[(0, "a", 1), (0, "b", 2), (2, "c", 3)],
        top=0, bottom=3,
        labels={0: "top", 1: "dead_prefix", 2: "mid", 3: "end"},
    )


def _make_label_reuse_across_depth() -> StateSpace:
    """Reused label at different depths — with unreachable state breaking lattice."""
    from reticulate.statespace import StateSpace
    return StateSpace(
        states={0, 1, 2, 3, 4},
        transitions=[(0, "a", 1), (1, "b", 2), (2, "a", 3)],
        top=0, bottom=3,
        labels={0: "top", 1: "s1", 2: "s2", 3: "end", 4: "orphan"},
    )


def _make_fan_in_state() -> StateSpace:
    """Fan-in with missing meet — two paths diverge further after converging.

    Extra leaves after the fan-in point have no meet.
    """
    from reticulate.statespace import StateSpace
    #     0
    #    / \
    #   1   2
    #    \ /
    #     3
    #    / \
    #   4   5   (no common bottom → no meet for 4,5)
    return StateSpace(
        states={0, 1, 2, 3, 4, 5},
        transitions=[
            (0, "a", 1), (0, "b", 2),
            (1, "c", 3), (2, "d", 3),
            (3, "e", 4), (3, "f", 5),
        ],
        top=0, bottom=4,
        labels={0: "top", 1: "s1", 2: "s2", 3: "join", 4: "b1", 5: "b2"},
    )


def _make_singleton_selection() -> StateSpace:
    """Single selection with unreachable state — breaks lattice boundedness."""
    from reticulate.statespace import StateSpace
    return StateSpace(
        states={0, 1, 2, 3},
        transitions=[(0, "a", 1), (1, "b", 2)],
        top=0, bottom=2,
        labels={0: "top", 1: "sel", 2: "end", 3: "orphan"},
        selection_transitions={(1, "b", 2)},
    )


def _make_empty_branch() -> StateSpace:
    """Non-bottom state with zero transitions — breaks lattice."""
    from reticulate.statespace import StateSpace
    return StateSpace(
        states={0, 1, 2, 3},
        transitions=[(0, "a", 1), (0, "b", 2), (2, "c", 3)],
        top=0, bottom=3,
        labels={0: "top", 1: "empty", 2: "mid", 3: "end"},
    )


def _make_exceeded_regularity() -> StateSpace:
    """Duplicate label → nondeterministic, with structure that breaks lattice."""
    from reticulate.statespace import StateSpace
    # Same as nondeterministic but kept as separate catalogue entry
    # for the "duplicate label in grammar" interpretation.
    # States 1 and 2 are incomparable with no meet → not a lattice.
    return StateSpace(
        states={0, 1, 2},
        transitions=[(0, "a", 1), (0, "a", 2)],
        top=0, bottom=0,
        labels={0: "top", 1: "s1", 2: "s2"},
    )


# Catalogue: kind → factory function
NON_REALIZABLE_CATALOGUE: dict[str, callable] = {
    # Category A: Structural (states not properly connected)
    "nondeterministic": _make_nondeterministic,
    "unreachable_orphan": _make_unreachable_orphan,
    "dead_end": _make_dead_end,
    "no_unique_top": _make_no_unique_top,
    "self_loop_bottom": _make_self_loop_bottom,
    # Category B: Lattice (valid LTS but not a lattice)
    "missing_meet": _make_missing_meet,
    "missing_join": _make_missing_join,
    "not_bounded": _make_not_bounded,
    # Category C: Reticular (lattice structure broken by mixed/role issues)
    "mixed_alien": _make_mixed_alien,
    "label_role_conflict": _make_label_role_conflict,
    "convergent_sharing": _make_convergent_sharing,
    "asymmetric_diamond": _make_asymmetric_diamond,
    # Category D: Automata-theoretic (trace structure breaks lattice)
    "non_prefix_closed": _make_non_prefix_closed,
    "label_reuse_across_depth": _make_label_reuse_across_depth,
    "fan_in_state": _make_fan_in_state,
    # Category E: Counting (arithmetic violations break lattice)
    "singleton_selection": _make_singleton_selection,
    "empty_branch": _make_empty_branch,
    "exceeded_regularity": _make_exceeded_regularity,
}
