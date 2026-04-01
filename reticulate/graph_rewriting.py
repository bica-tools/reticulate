"""Graph rewriting for session type state spaces (Step 92).

Transforms session type state spaces via algebraic graph rewriting rules.
Each rule is a pair (L, R) where L is a pattern (subgraph to match) and R is
the replacement.  The module provides:

  - Pattern matching: find subgraph isomorphisms of L in a host state space.
  - Rule application: replace matched subgraph with R.
  - Fixpoint iteration: apply rules until no more matches.
  - Confluence checking: verify rule application order does not matter.
  - Termination checking: verify rules always terminate.
  - Standard simplification rules: flatten_branch, merge_ends, unfold_once,
    factor_parallel.

The theoretical foundation is the double pushout (DPO) approach from algebraic
graph transformation theory, adapted to directed labeled session graphs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from reticulate.lattice import check_lattice
from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Match:
    """A mapping from pattern states to host state-space states.

    Attributes:
        state_map: Pattern state ID -> host state ID.
        edge_map: List of (pattern_edge, host_edge) pairs where each edge
                  is (src, label, tgt).
    """
    state_map: dict[int, int]
    edge_map: list[tuple[tuple[int, str, int], tuple[int, str, int]]]


@dataclass(frozen=True)
class RewriteRule:
    """A graph rewriting rule (L, R).

    Attributes:
        name: Human-readable rule name.
        lhs: Left-hand side pattern (the subgraph to match).
        rhs: Right-hand side replacement.
        interface_states: States in L that are preserved (mapped to R).
                         Maps LHS state ID -> RHS state ID.
        description: Optional description.
    """
    name: str
    lhs: StateSpace
    rhs: StateSpace
    interface_states: dict[int, int] = field(default_factory=dict)
    description: str = ""


@dataclass(frozen=True)
class RewriteStep:
    """Record of a single rewriting step.

    Attributes:
        rule: The rule applied.
        match: The match used.
        before: State space before rewriting.
        after: State space after rewriting.
    """
    rule: RewriteRule
    match: Match
    before: StateSpace
    after: StateSpace


@dataclass(frozen=True)
class ConfluenceResult:
    """Result of confluence checking.

    Attributes:
        is_confluent: True iff the rule set is confluent.
        critical_pairs: List of (rule1, rule2, host) triples where
                       the two rules have overlapping matches.
        all_convergent: True iff all critical pairs converge.
        counterexample: Description of first non-convergent pair, or None.
    """
    is_confluent: bool
    critical_pairs: list[tuple[str, str, str]]
    all_convergent: bool
    counterexample: str | None = None


@dataclass(frozen=True)
class TerminationResult:
    """Result of termination checking.

    Attributes:
        terminates: True iff the rule set always terminates.
        measure: Description of the termination measure.
        reason: Explanation.
    """
    terminates: bool
    measure: str
    reason: str


@dataclass(frozen=True)
class RewritingAnalysis:
    """Full analysis of a rewriting session.

    Attributes:
        original: The original state space.
        final: The state space after all rewrites.
        steps: Sequence of rewrite steps applied.
        num_steps: Total number of rewrite steps.
        rules_applied: Set of rule names that were applied.
        lattice_preserved: True iff final state space is a lattice
                          (assuming original was).
        original_is_lattice: True iff original state space is a lattice.
        final_is_lattice: True iff final state space is a lattice.
    """
    original: StateSpace
    final: StateSpace
    steps: tuple[RewriteStep, ...]
    num_steps: int
    rules_applied: frozenset[str]
    lattice_preserved: bool
    original_is_lattice: bool
    final_is_lattice: bool


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------

def match_pattern(ss: StateSpace, pattern: StateSpace) -> list[Match]:
    """Find all subgraph matches of *pattern* in *ss*.

    A match is an injective mapping from pattern states to host states
    such that:
      - Every transition in the pattern has a corresponding transition
        in the host (preserving labels).
      - The pattern's top maps to a state in the host.
      - The pattern's bottom maps to a state in the host.

    Returns a list of Match objects (possibly empty).
    """
    if not pattern.states:
        return []

    pattern_adj: dict[int, list[tuple[str, int]]] = {
        s: [] for s in pattern.states
    }
    for src, lbl, tgt in pattern.transitions:
        pattern_adj[src].append((lbl, tgt))

    host_adj: dict[int, list[tuple[str, int]]] = {
        s: [] for s in ss.states
    }
    for src, lbl, tgt in ss.transitions:
        host_adj[src].append((lbl, tgt))

    # Compute out-degree constraints for pruning
    pattern_out: dict[int, int] = {
        s: len(pattern_adj[s]) for s in pattern.states
    }
    host_out: dict[int, int] = {
        s: len(host_adj[s]) for s in ss.states
    }

    # Compute label sets for each state (for pruning)
    pattern_labels: dict[int, set[str]] = {
        s: {lbl for lbl, _ in pattern_adj[s]} for s in pattern.states
    }
    host_labels: dict[int, set[str]] = {
        s: {lbl for lbl, _ in host_adj[s]} for s in ss.states
    }

    matches: list[Match] = []
    pattern_states_ordered = sorted(pattern.states)

    def _backtrack(
        idx: int,
        state_map: dict[int, int],
        used: set[int],
        edge_map: list[tuple[tuple[int, str, int], tuple[int, str, int]]],
    ) -> None:
        if idx == len(pattern_states_ordered):
            # Verify all pattern transitions are matched
            all_ok = True
            final_edge_map = list(edge_map)
            for src, lbl, tgt in pattern.transitions:
                h_src = state_map[src]
                h_tgt = state_map[tgt]
                found = False
                for hl, ht in host_adj[h_src]:
                    if hl == lbl and ht == h_tgt:
                        found = True
                        final_edge_map.append(
                            ((src, lbl, tgt), (h_src, lbl, h_tgt))
                        )
                        break
                if not found:
                    all_ok = False
                    break
            if all_ok:
                matches.append(Match(
                    state_map=dict(state_map),
                    edge_map=final_edge_map,
                ))
            return

        p_state = pattern_states_ordered[idx]
        p_labels = pattern_labels[p_state]
        p_out = pattern_out[p_state]

        for h_state in sorted(ss.states):
            if h_state in used:
                continue
            # Pruning: host state must have at least as many outgoing
            # transitions with matching labels
            if host_out[h_state] < p_out:
                continue
            if not p_labels.issubset(host_labels[h_state]):
                continue

            # Check consistency with already-mapped transitions
            ok = True
            for lbl, tgt in pattern_adj[p_state]:
                if tgt in state_map:
                    h_tgt = state_map[tgt]
                    found = any(
                        hl == lbl and ht == h_tgt
                        for hl, ht in host_adj[h_state]
                    )
                    if not found:
                        ok = False
                        break
            if not ok:
                continue

            # Also check already-mapped predecessors
            for mapped_p, mapped_h in state_map.items():
                for lbl, tgt in pattern_adj[mapped_p]:
                    if tgt == p_state:
                        found = any(
                            hl == lbl and ht == h_state
                            for hl, ht in host_adj[mapped_h]
                        )
                        if not found:
                            ok = False
                            break
                if not ok:
                    break
            if not ok:
                continue

            state_map[p_state] = h_state
            used.add(h_state)
            _backtrack(idx + 1, state_map, used, edge_map)
            del state_map[p_state]
            used.discard(h_state)

    _backtrack(0, {}, set(), [])
    return matches


# ---------------------------------------------------------------------------
# Rule application
# ---------------------------------------------------------------------------

def apply_rule(ss: StateSpace, rule: RewriteRule, match: Match) -> StateSpace:
    """Apply a rewriting rule to *ss* at the given *match*.

    The DPO-inspired approach:
    1. Remove matched edges (and non-interface matched states).
    2. Add replacement (RHS) states and edges, glued via interface.

    Returns a new StateSpace.
    """
    matched_states = set(match.state_map.values())
    matched_edges = {he for _, he in match.edge_map}

    # Interface: pattern states that map to RHS states
    interface_host: dict[int, int] = {}  # host_state -> rhs_state
    for lhs_state, rhs_state in rule.interface_states.items():
        if lhs_state in match.state_map:
            interface_host[match.state_map[lhs_state]] = rhs_state

    # States to remove: matched states NOT in interface
    states_to_remove = matched_states - set(interface_host.keys())

    # Build the context: host minus matched non-interface parts
    # Keep all host transitions that are not matched
    context_transitions = [
        (s, l, t) for s, l, t in ss.transitions
        if (s, l, t) not in matched_edges
    ]
    context_selections = {
        (s, l, t) for s, l, t in ss.selection_transitions
        if (s, l, t) not in matched_edges
    }

    # Remap: for removed states that are targets of non-matched edges,
    # we need to redirect through the interface
    # Build rhs state -> new host state mapping
    next_id = max(ss.states) + 1 if ss.states else 0
    rhs_to_host: dict[int, int] = {}

    for host_state, rhs_state in interface_host.items():
        rhs_to_host[rhs_state] = host_state

    for rhs_state in rule.rhs.states:
        if rhs_state not in rhs_to_host:
            rhs_to_host[rhs_state] = next_id
            next_id += 1

    # Add RHS transitions
    new_transitions = list(context_transitions)
    new_selections = set(context_selections)
    for src, lbl, tgt in rule.rhs.transitions:
        h_src = rhs_to_host[src]
        h_tgt = rhs_to_host[tgt]
        new_transitions.append((h_src, lbl, h_tgt))
        if (src, lbl, tgt) in rule.rhs.selection_transitions:
            new_selections.add((h_src, lbl, h_tgt))

    # Redirect any context transitions pointing to removed states
    # via interface mapping
    lhs_to_rhs = rule.interface_states
    redirect: dict[int, int] = {}
    for lhs_state, host_state in match.state_map.items():
        if host_state in states_to_remove and lhs_state in lhs_to_rhs:
            rhs_state = lhs_to_rhs[lhs_state]
            redirect[host_state] = rhs_to_host[rhs_state]

    if redirect:
        new_transitions = [
            (redirect.get(s, s), l, redirect.get(t, t))
            for s, l, t in new_transitions
        ]
        new_selections = {
            (redirect.get(s, s), l, redirect.get(t, t))
            for s, l, t in new_selections
        }

    # Compute new state set
    new_states = (ss.states - states_to_remove) | {
        rhs_to_host[s] for s in rule.rhs.states
    }

    # Update labels
    new_labels = {s: v for s, v in ss.labels.items() if s not in states_to_remove}
    for rhs_state in rule.rhs.states:
        h_state = rhs_to_host[rhs_state]
        if rhs_state in rule.rhs.labels:
            new_labels[h_state] = rule.rhs.labels[rhs_state]

    # Determine new top/bottom
    new_top = ss.top
    new_bottom = ss.bottom
    if ss.top in states_to_remove:
        new_top = rhs_to_host.get(rule.rhs.top, ss.top)
    if ss.bottom in states_to_remove:
        new_bottom = rhs_to_host.get(rule.rhs.bottom, ss.bottom)

    # Prune unreachable states
    result = StateSpace(
        states=new_states,
        transitions=new_transitions,
        top=new_top,
        bottom=new_bottom,
        labels=new_labels,
        selection_transitions=new_selections,
    )
    reachable = result.reachable_from(result.top)
    reachable.add(result.bottom)

    return StateSpace(
        states=reachable,
        transitions=[
            (s, l, t) for s, l, t in result.transitions
            if s in reachable and t in reachable
        ],
        top=result.top,
        bottom=result.bottom,
        labels={s: v for s, v in result.labels.items() if s in reachable},
        selection_transitions={
            (s, l, t) for s, l, t in result.selection_transitions
            if s in reachable and t in reachable
        },
    )


def apply_rules(
    ss: StateSpace,
    rules: Sequence[RewriteRule],
    max_iterations: int = 100,
) -> tuple[StateSpace, list[RewriteStep]]:
    """Apply all rules until fixpoint (no more matches).

    Returns (final_state_space, list_of_steps).
    """
    current = ss
    steps: list[RewriteStep] = []
    for _ in range(max_iterations):
        applied = False
        for rule in rules:
            matches = match_pattern(current, rule.lhs)
            if matches:
                match = matches[0]
                result = apply_rule(current, rule, match)
                steps.append(RewriteStep(
                    rule=rule,
                    match=match,
                    before=current,
                    after=result,
                ))
                current = result
                applied = True
                break  # restart from first rule
        if not applied:
            break
    return current, steps


# ---------------------------------------------------------------------------
# Standard rewriting rules
# ---------------------------------------------------------------------------

def _make_ss(
    states: set[int],
    transitions: list[tuple[int, str, int]],
    top: int,
    bottom: int,
    labels: dict[int, str] | None = None,
    selections: set[tuple[int, str, int]] | None = None,
) -> StateSpace:
    """Helper to construct a StateSpace."""
    return StateSpace(
        states=states,
        transitions=transitions,
        top=top,
        bottom=bottom,
        labels=labels or {},
        selection_transitions=selections or set(),
    )


def flatten_branch_rule(label: str, new_label: str) -> RewriteRule:
    """Create a rule that adds a new branch transition at a branch state.

    Pattern L: state 0 --label--> state 1
    Replacement R: state 0 --label--> state 1, state 0 --new_label--> state 1

    This corresponds to BranchExt from the paper.
    """
    lhs = _make_ss(
        states={0, 1},
        transitions=[(0, label, 1)],
        top=0, bottom=1,
        labels={0: "branch", 1: "end"},
    )
    rhs = _make_ss(
        states={0, 1},
        transitions=[(0, label, 1), (0, new_label, 1)],
        top=0, bottom=1,
        labels={0: "branch", 1: "end"},
    )
    return RewriteRule(
        name="flatten_branch",
        lhs=lhs,
        rhs=rhs,
        interface_states={0: 0, 1: 1},
        description=f"Add branch edge '{new_label}' alongside '{label}'",
    )


def merge_ends_rule() -> RewriteRule:
    """Create a rule that merges two end states into one.

    Pattern L: state 0 --a--> state 1, state 0 --b--> state 2
               where states 1 and 2 are both end states (no outgoing edges).
    Replacement R: state 0 --a--> state 1, state 0 --b--> state 1
    """
    lhs = _make_ss(
        states={0, 1, 2},
        transitions=[(0, "_a", 1), (0, "_b", 2)],
        top=0, bottom=1,
        labels={0: "branch", 1: "end1", 2: "end2"},
    )
    rhs = _make_ss(
        states={0, 1},
        transitions=[(0, "_a", 1), (0, "_b", 1)],
        top=0, bottom=1,
        labels={0: "branch", 1: "end"},
    )
    return RewriteRule(
        name="merge_ends",
        lhs=lhs,
        rhs=rhs,
        interface_states={0: 0, 1: 1},
        description="Merge two leaf end-states into one",
    )


def unfold_once_rule(label: str) -> RewriteRule:
    """Create a rule that unfolds a self-loop once.

    Pattern L: state 0 --label--> state 0 (self-loop)
    Replacement R: state 0 --label--> state 1, state 1 --label--> state 1
                   (one unfolding of the loop)

    This corresponds to the Unfold rule from the paper.
    """
    lhs = _make_ss(
        states={0},
        transitions=[(0, label, 0)],
        top=0, bottom=0,
        labels={0: "rec"},
    )
    rhs = _make_ss(
        states={0, 1},
        transitions=[(0, label, 1), (1, label, 1)],
        top=0, bottom=1,
        labels={0: "rec_unfolded", 1: "rec"},
    )
    return RewriteRule(
        name="unfold_once",
        lhs=lhs,
        rhs=rhs,
        interface_states={0: 0},
        description=f"Unfold self-loop on label '{label}' once",
    )


def add_branch_rule(existing_label: str, new_label: str) -> RewriteRule:
    """Create a BranchExt rule: add a new branch option at a branch state.

    Pattern L: a state with an outgoing edge labeled existing_label.
    Replacement R: same state with both existing_label and new_label edges.
    The new edge leads to a fresh end state.
    """
    lhs = _make_ss(
        states={0, 1},
        transitions=[(0, existing_label, 1)],
        top=0, bottom=1,
        labels={0: "branch", 1: "target"},
    )
    rhs = _make_ss(
        states={0, 1, 2},
        transitions=[
            (0, existing_label, 1),
            (0, new_label, 2),
        ],
        top=0, bottom=1,
        labels={0: "branch", 1: "target", 2: "new_end"},
    )
    return RewriteRule(
        name="add_branch",
        lhs=lhs,
        rhs=rhs,
        interface_states={0: 0, 1: 1},
        description=f"Add branch '{new_label}' at state with '{existing_label}'",
    )


def remove_selection_rule(keep_label: str, remove_label: str) -> RewriteRule:
    """Create a SelRestrict rule: remove a selection option.

    Pattern L: a selection state with two outgoing selection edges.
    Replacement R: same state with only the keep_label edge.
    """
    lhs = _make_ss(
        states={0, 1, 2},
        transitions=[(0, keep_label, 1), (0, remove_label, 2)],
        top=0, bottom=1,
        labels={0: "select", 1: "keep_target", 2: "remove_target"},
        selections={(0, keep_label, 1), (0, remove_label, 2)},
    )
    rhs = _make_ss(
        states={0, 1},
        transitions=[(0, keep_label, 1)],
        top=0, bottom=1,
        labels={0: "select", 1: "keep_target"},
        selections={(0, keep_label, 1)},
    )
    return RewriteRule(
        name="remove_selection",
        lhs=lhs,
        rhs=rhs,
        interface_states={0: 0, 1: 1},
        description=f"Remove selection '{remove_label}', keep '{keep_label}'",
    )


def replace_subgraph_rule(
    entry_label: str,
    old_continuation: StateSpace,
    new_continuation: StateSpace,
) -> RewriteRule:
    """Create a CovarDepth rule: replace continuation after a transition.

    This replaces the subgraph reachable after the entry_label transition.
    """
    # LHS: entry state --entry_label--> old_continuation
    lhs_top = max(old_continuation.states) + 1 if old_continuation.states else 0
    lhs_states = old_continuation.states | {lhs_top}
    lhs_transitions = list(old_continuation.transitions) + [
        (lhs_top, entry_label, old_continuation.top)
    ]
    lhs = _make_ss(
        states=lhs_states,
        transitions=lhs_transitions,
        top=lhs_top,
        bottom=old_continuation.bottom,
        labels={**old_continuation.labels, lhs_top: "entry"},
        selections=old_continuation.selection_transitions,
    )

    # RHS: entry state --entry_label--> new_continuation
    # Remap new_continuation to avoid ID clashes
    offset = max(lhs_states) + 1
    remap = {s: s + offset for s in new_continuation.states}
    rhs_top_entry = lhs_top  # same entry state (interface)
    rhs_states = {rhs_top_entry} | {remap[s] for s in new_continuation.states}
    rhs_transitions = [
        (rhs_top_entry, entry_label, remap[new_continuation.top])
    ] + [
        (remap[s], l, remap[t]) for s, l, t in new_continuation.transitions
    ]
    rhs_selections = {
        (remap[s], l, remap[t])
        for s, l, t in new_continuation.selection_transitions
    }
    rhs_labels = {rhs_top_entry: "entry"}
    for s, lab in new_continuation.labels.items():
        rhs_labels[remap[s]] = lab

    rhs = _make_ss(
        states=rhs_states,
        transitions=rhs_transitions,
        top=rhs_top_entry,
        bottom=remap[new_continuation.bottom],
        labels=rhs_labels,
        selections=rhs_selections,
    )

    return RewriteRule(
        name="replace_subgraph",
        lhs=lhs,
        rhs=rhs,
        interface_states={lhs_top: rhs_top_entry},
        description=f"Replace subgraph after '{entry_label}' transition",
    )


# ---------------------------------------------------------------------------
# Standard simplification
# ---------------------------------------------------------------------------

def simplify(ss: StateSpace) -> tuple[StateSpace, list[RewriteStep]]:
    """Apply standard simplification rules until fixpoint.

    Standard rules applied:
    1. Merge duplicate end states (states with no outgoing transitions
       that can be collapsed).
    2. Remove unreachable states.

    Returns (simplified_state_space, list_of_steps).
    """
    steps: list[RewriteStep] = []
    current = ss

    # Step 1: merge end-like states (states with no outgoing transitions
    # other than the designated bottom)
    end_states = {
        s for s in current.states
        if not current.enabled(s) and s != current.bottom
    }
    if end_states:
        # Redirect all transitions to end states -> bottom
        redirect = {s: current.bottom for s in end_states}
        new_transitions = [
            (s, l, redirect.get(t, t))
            for s, l, t in current.transitions
            if s not in end_states
        ]
        new_selections = {
            (s, l, redirect.get(t, t))
            for s, l, t in current.selection_transitions
            if s not in end_states
        }
        new_states = current.states - end_states
        new_labels = {
            s: v for s, v in current.labels.items()
            if s not in end_states
        }
        merged = StateSpace(
            states=new_states,
            transitions=new_transitions,
            top=current.top,
            bottom=current.bottom,
            labels=new_labels,
            selection_transitions=new_selections,
        )
        if merged.states != current.states or merged.transitions != current.transitions:
            steps.append(RewriteStep(
                rule=RewriteRule(
                    name="merge_ends",
                    lhs=_make_ss(set(), [], 0, 0),
                    rhs=_make_ss(set(), [], 0, 0),
                    description="Merge redundant end states",
                ),
                match=Match(state_map={}, edge_map=[]),
                before=current,
                after=merged,
            ))
            current = merged

    # Step 2: prune unreachable
    reachable = current.reachable_from(current.top)
    reachable.add(current.bottom)
    if reachable != current.states:
        pruned = StateSpace(
            states=reachable,
            transitions=[
                (s, l, t) for s, l, t in current.transitions
                if s in reachable and t in reachable
            ],
            top=current.top,
            bottom=current.bottom,
            labels={s: v for s, v in current.labels.items() if s in reachable},
            selection_transitions={
                (s, l, t) for s, l, t in current.selection_transitions
                if s in reachable and t in reachable
            },
        )
        steps.append(RewriteStep(
            rule=RewriteRule(
                name="prune_unreachable",
                lhs=_make_ss(set(), [], 0, 0),
                rhs=_make_ss(set(), [], 0, 0),
                description="Remove unreachable states",
            ),
            match=Match(state_map={}, edge_map=[]),
            before=current,
            after=pruned,
        ))
        current = pruned

    return current, steps


# ---------------------------------------------------------------------------
# Confluence checking
# ---------------------------------------------------------------------------

def check_confluence(
    rules: Sequence[RewriteRule],
    test_graphs: Sequence[StateSpace] | None = None,
) -> ConfluenceResult:
    """Check whether a set of rewriting rules is confluent.

    Uses empirical testing: for each test graph, try applying each pair of
    rules in both orders and check if results are isomorphic (same state/edge
    counts and structure).

    If no test_graphs are provided, returns a trivially confluent result
    for rule sets of size <= 1.
    """
    if len(rules) <= 1:
        return ConfluenceResult(
            is_confluent=True,
            critical_pairs=[],
            all_convergent=True,
        )

    if test_graphs is None:
        # With no test graphs, we can only check structural properties
        # Check that rules target disjoint categories (branch vs selection)
        critical_pairs: list[tuple[str, str, str]] = []
        for i, r1 in enumerate(rules):
            for r2 in rules[i + 1:]:
                # Check if LHS patterns could overlap
                r1_labels = {lbl for _, lbl, _ in r1.lhs.transitions}
                r2_labels = {lbl for _, lbl, _ in r2.lhs.transitions}
                if r1_labels & r2_labels:
                    critical_pairs.append((
                        r1.name, r2.name,
                        f"Shared labels: {r1_labels & r2_labels}",
                    ))
        return ConfluenceResult(
            is_confluent=len(critical_pairs) == 0,
            critical_pairs=critical_pairs,
            all_convergent=len(critical_pairs) == 0,
        )

    # Empirical confluence testing
    critical_pairs = []
    all_convergent = True
    counterexample = None

    for graph in test_graphs:
        for i, r1 in enumerate(rules):
            for r2 in rules[i + 1:]:
                m1_list = match_pattern(graph, r1.lhs)
                m2_list = match_pattern(graph, r2.lhs)
                if not m1_list or not m2_list:
                    continue

                # Apply r1 then r2
                g1 = apply_rule(graph, r1, m1_list[0])
                m2_in_g1 = match_pattern(g1, r2.lhs)
                if m2_in_g1:
                    h12 = apply_rule(g1, r2, m2_in_g1[0])
                else:
                    h12 = g1

                # Apply r2 then r1
                g2 = apply_rule(graph, r2, m2_list[0])
                m1_in_g2 = match_pattern(g2, r1.lhs)
                if m1_in_g2:
                    h21 = apply_rule(g2, r1, m1_in_g2[0])
                else:
                    h21 = g2

                # Check if results are structurally equivalent
                equiv = _structurally_equivalent(h12, h21)
                pair_desc = f"{r1.name} vs {r2.name}"
                critical_pairs.append((r1.name, r2.name, pair_desc))

                if not equiv:
                    all_convergent = False
                    counterexample = (
                        f"Non-convergent: {r1.name} then {r2.name} "
                        f"({len(h12.states)} states) vs "
                        f"{r2.name} then {r1.name} "
                        f"({len(h21.states)} states)"
                    )

    return ConfluenceResult(
        is_confluent=all_convergent,
        critical_pairs=critical_pairs,
        all_convergent=all_convergent,
        counterexample=counterexample,
    )


def _structurally_equivalent(ss1: StateSpace, ss2: StateSpace) -> bool:
    """Check if two state spaces are structurally equivalent.

    Two state spaces are structurally equivalent if they have the same
    number of states and transitions, and the same multiset of
    (out_degree, in_degree, label_set) per state.
    """
    if len(ss1.states) != len(ss2.states):
        return False
    if len(ss1.transitions) != len(ss2.transitions):
        return False

    def _signature(ss: StateSpace) -> list[tuple[int, int, tuple[str, ...]]]:
        out_deg: dict[int, int] = {s: 0 for s in ss.states}
        in_deg: dict[int, int] = {s: 0 for s in ss.states}
        out_labels: dict[int, list[str]] = {s: [] for s in ss.states}
        for s, l, t in ss.transitions:
            out_deg[s] = out_deg.get(s, 0) + 1
            in_deg[t] = in_deg.get(t, 0) + 1
            out_labels.setdefault(s, []).append(l)
        return sorted(
            (out_deg.get(s, 0), in_deg.get(s, 0), tuple(sorted(out_labels.get(s, []))))
            for s in ss.states
        )

    return _signature(ss1) == _signature(ss2)


# ---------------------------------------------------------------------------
# Termination checking
# ---------------------------------------------------------------------------

def check_termination(
    rules: Sequence[RewriteRule],
) -> TerminationResult:
    """Check whether a set of rewriting rules always terminates.

    Uses a simple structural measure: if every rule strictly reduces the
    number of states, edges, or a weighted combination, then the system
    terminates.

    Heuristic analysis:
    - Rules that reduce state count: always terminate.
    - Rules that preserve state count but reduce edges: terminate.
    - Rules that increase both: may not terminate (e.g. unbounded unfolding).
    """
    if not rules:
        return TerminationResult(
            terminates=True,
            measure="empty",
            reason="No rules to apply.",
        )

    all_reducing = True
    reasons: list[str] = []

    for rule in rules:
        lhs_states = len(rule.lhs.states)
        rhs_states = len(rule.rhs.states)
        lhs_edges = len(rule.lhs.transitions)
        rhs_edges = len(rule.rhs.transitions)

        if rhs_states < lhs_states:
            reasons.append(
                f"{rule.name}: reduces states ({lhs_states} -> {rhs_states})"
            )
        elif rhs_states == lhs_states and rhs_edges < lhs_edges:
            reasons.append(
                f"{rule.name}: reduces edges ({lhs_edges} -> {rhs_edges})"
            )
        elif rhs_states > lhs_states or rhs_edges > lhs_edges:
            all_reducing = False
            reasons.append(
                f"{rule.name}: increases size "
                f"({lhs_states} states, {lhs_edges} edges -> "
                f"{rhs_states} states, {rhs_edges} edges)"
            )
        else:
            reasons.append(f"{rule.name}: preserves size (no effect)")

    if all_reducing:
        return TerminationResult(
            terminates=True,
            measure="state+edge count",
            reason="; ".join(reasons),
        )
    else:
        return TerminationResult(
            terminates=False,
            measure="state+edge count",
            reason="; ".join(reasons),
        )


# ---------------------------------------------------------------------------
# Lattice preservation check
# ---------------------------------------------------------------------------

def check_lattice_preservation(
    ss: StateSpace,
    rule: RewriteRule,
    match: Match,
) -> bool:
    """Check whether applying a rule at a match preserves the lattice property.

    Returns True iff the result of apply_rule(ss, rule, match) is a lattice
    (given that ss is a lattice).
    """
    result = apply_rule(ss, rule, match)
    lr = check_lattice(result)
    return lr.is_lattice


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze_rewriting(
    ss: StateSpace,
    rules: Sequence[RewriteRule],
    max_iterations: int = 100,
) -> RewritingAnalysis:
    """Perform full rewriting analysis.

    Applies all rules until fixpoint and checks lattice preservation.

    Returns a RewritingAnalysis with all details.
    """
    original_lattice = check_lattice(ss)
    final, steps = apply_rules(ss, rules, max_iterations=max_iterations)
    final_lattice = check_lattice(final)

    return RewritingAnalysis(
        original=ss,
        final=final,
        steps=tuple(steps),
        num_steps=len(steps),
        rules_applied=frozenset(step.rule.name for step in steps),
        lattice_preserved=(
            not original_lattice.is_lattice or final_lattice.is_lattice
        ),
        original_is_lattice=original_lattice.is_lattice,
        final_is_lattice=final_lattice.is_lattice,
    )
