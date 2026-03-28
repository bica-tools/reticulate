"""Fundamental mechanisms of thought as session type operations (Step 102).

Formalizes 7 fundamental cognitive/philosophical mechanisms as concrete
operations on session types and their state spaces.  Analogy (the 8th
mechanism) is handled by the dedicated analogy module (Step 101).

The mechanisms:

1. **Abstraction** (Plato's Forms) — forgetting details to reveal essence.
2. **Composition** (Leibniz's Monads) — building complex from simple.
3. **Negation** (Hegel's Determinate Negation) — defining by exclusion.
4. **Recursion** (Gödel's Self-Reference) — creating infinity from finitude.
5. **Emergence** (Aristotle's "the whole is greater") — novel whole properties.
6. **Dialectic** (Hegel's Aufhebung) — thesis + antithesis → synthesis.
7. **Metaphor** (Lakoff's Conceptual Metaphor) — understanding via mapping.

Each mechanism is implemented as one or more pure functions operating on
session type ASTs or state spaces, returning concrete results.

This module provides:
    ``abstract_by_labels(ss)``       — relabel to reveal pure structure.
    ``abstract_by_depth(ss, d)``     — truncate to depth d.
    ``compose_sequential(types)``    — chain types in sequence.
    ``compose_parallel(types)``      — nest binary parallel.
    ``compose_choice(types, labels)``— offer types as branch alternatives.
    ``negate_type(s)``               — swap Branch/Select (anti-type).
    ``compute_violations(ss)``       — forbidden actions per state.
    ``make_recursive(s, var)``       — wrap type in recursion.
    ``unroll(s, depth)``             — unroll recursion to finite depth.
    ``fixed_point_depth(ss)``        — unrollings until stabilization.
    ``detect_emergence(parts, whole)``— emergent whole-vs-part properties.
    ``emergence_score(parts, whole)``— quantify emergence 0–1.
    ``dialectic(thesis, antithesis)``— Hegelian synthesis.
    ``dialectic_chain(types)``       — iterated dialectic.
    ``metaphor(source, mapping)``    — relabel via conceptual mapping.
    ``detect_metaphor(s1, s2)``      — detect isomorphic relabeling.
    ``metaphor_quality(mapping, src, tgt)`` — quality of a metaphoric map.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import reduce
from string import ascii_lowercase

from reticulate.duality import dual
from reticulate.lattice import check_lattice
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


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MechanismResult:
    """Result of applying a mechanism of thought.

    Attributes:
        mechanism: which mechanism was applied.
        input_description: textual description of the input(s).
        output_description: textual description of the output.
        preserves_lattice: whether the lattice property is preserved.
        explanation: philosophical / technical explanation of the result.
    """

    mechanism: str
    input_description: str
    output_description: str
    preserves_lattice: bool
    explanation: str


# =========================================================================
# Mechanism 1: ABSTRACTION (Plato's Forms)
# =========================================================================


def abstract_by_labels(ss: StateSpace) -> StateSpace:
    """Abstract away label information — replace all labels with generic
    letters ('a', 'b', 'c', ...) preserving distinctness within each state
    but forgetting original method names.

    Reveals pure branching structure independent of method names.
    """
    # For each state, collect outgoing labels and assign generic letters.
    # The mapping is per-state so that distinct labels at one state get
    # distinct generic names, but the *same* letter can appear at different
    # states (we only need local distinctness).
    new_transitions: list[tuple[int, str, int]] = []
    new_selection: set[tuple[int, str, int]] = set()

    # Global label mapping: first-seen order across the whole state space.
    label_map: dict[str, str] = {}
    counter = 0

    for src, lbl, tgt in ss.transitions:
        if lbl not in label_map:
            if counter < 26:
                label_map[lbl] = ascii_lowercase[counter]
            else:
                label_map[lbl] = f"x{counter}"
            counter += 1
        new_lbl = label_map[lbl]
        new_transitions.append((src, new_lbl, tgt))
        if (src, lbl, tgt) in ss.selection_transitions:
            new_selection.add((src, new_lbl, tgt))

    new_labels = dict(ss.labels)
    return StateSpace(
        states=set(ss.states),
        transitions=new_transitions,
        top=ss.top,
        bottom=ss.bottom,
        labels=new_labels,
        selection_transitions=new_selection,
    )


def abstract_by_depth(ss: StateSpace, max_depth: int) -> StateSpace:
    """Abstract by truncating: keep only states within *max_depth* of top.

    States at max_depth get their outgoing transitions redirected to bottom.
    Like Plato's cave — we see only the shadow (truncated projection) of
    the full form.
    """
    # BFS from top to compute depths.
    depth: dict[int, int] = {ss.top: 0}
    queue: deque[int] = deque([ss.top])
    while queue:
        s = queue.popleft()
        for _, _, t in ((src, l, tgt) for src, l, tgt in ss.transitions if src == s):
            if t not in depth:
                depth[t] = depth[s] + 1
                queue.append(t)

    # Keep states at depth <= max_depth + bottom (always kept).
    kept = {s for s, d in depth.items() if d <= max_depth}
    kept.add(ss.bottom)

    new_transitions: list[tuple[int, str, int]] = []
    new_selection: set[tuple[int, str, int]] = set()

    for src, lbl, tgt in ss.transitions:
        if src not in kept or src == ss.bottom:
            continue
        d_src = depth.get(src, max_depth + 1)
        if d_src >= max_depth:
            # Redirect to bottom.
            new_transitions.append((src, lbl, ss.bottom))
            if (src, lbl, tgt) in ss.selection_transitions:
                new_selection.add((src, lbl, ss.bottom))
        else:
            if tgt in kept:
                new_transitions.append((src, lbl, tgt))
                if (src, lbl, tgt) in ss.selection_transitions:
                    new_selection.add((src, lbl, tgt))
            else:
                new_transitions.append((src, lbl, ss.bottom))
                if (src, lbl, tgt) in ss.selection_transitions:
                    new_selection.add((src, lbl, ss.bottom))

    new_labels = {s: ss.labels.get(s, "") for s in kept if s in ss.labels}
    return StateSpace(
        states=kept,
        transitions=new_transitions,
        top=ss.top,
        bottom=ss.bottom,
        labels=new_labels,
        selection_transitions=new_selection,
    )


# =========================================================================
# Mechanism 2: COMPOSITION (Leibniz's Monads)
# =========================================================================


def compose_sequential(types: list[SessionType]) -> SessionType:
    """Sequential composition: S1 then S2 then S3...

    Creates a branch with each type as a phase: phase_0, phase_1, etc.
    Each phase is a separate choice — the client picks which phase to enter.

    Raises ValueError if the list is empty.
    """
    if not types:
        raise ValueError("Cannot compose empty list of types")
    if len(types) == 1:
        return types[0]
    choices = tuple((f"phase_{i}", t) for i, t in enumerate(types))
    return Branch(choices)


def compose_parallel(types: list[SessionType]) -> SessionType:
    """Parallel composition of N types using nested binary parallel.

    Raises ValueError if the list is empty.
    """
    if not types:
        raise ValueError("Cannot compose empty list of types")
    if len(types) == 1:
        return types[0]
    result = types[-1]
    for t in reversed(types[:-1]):
        result = Parallel((t, result))
    return result


def compose_choice(
    types: list[SessionType], labels: list[str] | None = None
) -> SessionType:
    """Branch composition — offer all types as labeled alternatives.

    Labels default to "option_0", "option_1", etc. if not provided.
    Raises ValueError if the list is empty or label count mismatches.
    """
    if not types:
        raise ValueError("Cannot compose empty list of types")
    if labels is None:
        labels = [f"option_{i}" for i in range(len(types))]
    if len(labels) != len(types):
        raise ValueError(
            f"Label count ({len(labels)}) must match type count ({len(types)})"
        )
    choices = tuple(zip(labels, types))
    return Branch(choices)


# =========================================================================
# Mechanism 3: NEGATION (Hegel's Determinate Negation)
# =========================================================================


def negate_type(s: SessionType) -> SessionType:
    """The anti-type: swap Branch <-> Select (like duality).

    The negation of a type describes the opposite at every choice point —
    where the original offers, the negation selects, and vice versa.
    This is the closest meaningful negation for session types.
    """
    return dual(s)


def compute_violations(ss: StateSpace) -> list[tuple[int, str]]:
    """For each state, list methods that are NOT enabled.

    These are the 'forbidden' actions — the shadow of the protocol.
    Returns a list of ``(state_id, disabled_method)`` pairs.
    """
    all_labels = {lbl for _, lbl, _ in ss.transitions}
    violations: list[tuple[int, str]] = []
    for state in sorted(ss.states):
        enabled = {lbl for _, lbl, _ in ss.transitions if _ == state}
        # Recompute enabled properly.
        enabled_set = {lbl for src, lbl, _ in ss.transitions if src == state}
        disabled = all_labels - enabled_set
        for lbl in sorted(disabled):
            violations.append((state, lbl))
    return violations


# =========================================================================
# Mechanism 4: RECURSION (Gödel's Self-Reference)
# =========================================================================


def _replace_end(s: SessionType, replacement: SessionType) -> SessionType:
    """Replace all End nodes in *s* with *replacement*."""
    if isinstance(s, End):
        return replacement
    if isinstance(s, (Var, Wait)):
        return s
    if isinstance(s, Branch):
        return Branch(tuple((m, _replace_end(c, replacement)) for m, c in s.choices))
    if isinstance(s, Select):
        return Select(tuple((m, _replace_end(c, replacement)) for m, c in s.choices))
    if isinstance(s, Parallel):
        return Parallel(tuple(_replace_end(b, replacement) for b in s.branches))
    if isinstance(s, Rec):
        return Rec(s.var, _replace_end(s.body, replacement))
    return s  # pragma: no cover


def make_recursive(s: SessionType, var_name: str = "X") -> SessionType:
    """Wrap a type in recursion: replace all End nodes with the recursive
    variable, then wrap in ``rec var_name . ...``.

    Transforms a finite protocol into an infinite loop.
    """
    body = _replace_end(s, Var(var_name))
    return Rec(var_name, body)


def unroll(s: SessionType, depth: int = 3) -> SessionType:
    """Unroll recursion to finite depth, replacing remaining variables with End.

    The inverse of make_recursive: expand recursive definitions, then
    terminate remaining open references.
    """
    from reticulate.recursion import unfold_depth as _unfold_depth, substitute as _substitute

    result = _unfold_depth(s, depth)
    # Replace any remaining Var nodes with End.
    return _terminate_vars(result)


def _terminate_vars(s: SessionType) -> SessionType:
    """Replace all Var nodes with End."""
    if isinstance(s, Var):
        return End()
    if isinstance(s, End | Wait):
        return s
    if isinstance(s, Branch):
        return Branch(tuple((m, _terminate_vars(c)) for m, c in s.choices))
    if isinstance(s, Select):
        return Select(tuple((m, _terminate_vars(c)) for m, c in s.choices))
    if isinstance(s, Parallel):
        return Parallel(tuple(_terminate_vars(b) for b in s.branches))
    if isinstance(s, Rec):
        return Rec(s.var, _terminate_vars(s.body))
    return s  # pragma: no cover


def fixed_point_depth(ss: StateSpace) -> int:
    """How many unrollings until the state space stabilizes?

    For state spaces with cycles (from recursion), the answer is 1 — the
    state space with cycles already captures the infinite unfolding.
    For acyclic state spaces (no recursion), the answer is 0.
    """
    # Detect cycles via DFS.
    adj: dict[int, list[int]] = {s: [] for s in ss.states}
    for src, _, tgt in ss.transitions:
        adj[src].append(tgt)

    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[int, int] = {s: WHITE for s in ss.states}

    def _has_cycle(v: int) -> bool:
        color[v] = GRAY
        for w in adj[v]:
            if color[w] == GRAY:
                return True
            if color[w] == WHITE and _has_cycle(w):
                return True
        color[v] = BLACK
        return False

    for s in ss.states:
        if color[s] == WHITE:
            if _has_cycle(s):
                return 1
    return 0


# =========================================================================
# Mechanism 5: EMERGENCE (Aristotle's "The whole is greater")
# =========================================================================


def _has_cycle(ss: StateSpace) -> bool:
    """Check whether the state space contains a directed cycle."""
    adj: dict[int, list[int]] = {s: [] for s in ss.states}
    for src, _, tgt in ss.transitions:
        adj[src].append(tgt)

    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[int, int] = {s: WHITE for s in ss.states}

    def _dfs(v: int) -> bool:
        color[v] = GRAY
        for w in adj[v]:
            if color[w] == GRAY:
                return True
            if color[w] == WHITE and _dfs(w):
                return True
        color[v] = BLACK
        return False

    for s in ss.states:
        if color[s] == WHITE:
            if _dfs(s):
                return True
    return False


def detect_emergence(
    parts: list[StateSpace], whole: StateSpace
) -> dict[str, tuple[object, object]]:
    """Compare properties of parts vs whole.

    Emergence = properties the whole has that no part has.
    Returns dict of ``{"property": (part_value, whole_value)}`` for
    properties that differ (emergent properties only).
    """
    if not parts:
        return {}

    results: dict[str, tuple[object, object]] = {}

    # Property 1: cycles.
    any_part_cyclic = any(_has_cycle(p) for p in parts)
    whole_cyclic = _has_cycle(whole)
    if whole_cyclic and not any_part_cyclic:
        results["cycles"] = (False, True)

    # Property 2: state count (superadditive?).
    part_states = sum(len(p.states) for p in parts)
    whole_states = len(whole.states)
    if whole_states > part_states:
        results["state_count_superadditive"] = (part_states, whole_states)

    # Property 3: transition count.
    part_trans = sum(len(p.transitions) for p in parts)
    whole_trans = len(whole.transitions)
    if whole_trans > part_trans:
        results["transition_count_superadditive"] = (part_trans, whole_trans)

    # Property 4: lattice property emergence.
    any_part_lattice = any(check_lattice(p).is_lattice for p in parts)
    whole_lattice = check_lattice(whole).is_lattice
    if whole_lattice and not any_part_lattice:
        results["lattice_property"] = (False, True)

    # Property 5: max branching factor.
    def _max_branching(ss: StateSpace) -> int:
        if not ss.states:
            return 0
        return max(
            (len([t for s2, _, t in ss.transitions if s2 == s]) for s in ss.states),
            default=0,
        )

    part_max_branch = max((_max_branching(p) for p in parts), default=0)
    whole_max_branch = _max_branching(whole)
    if whole_max_branch > part_max_branch:
        results["max_branching_factor"] = (part_max_branch, whole_max_branch)

    return results


def emergence_score(parts: list[StateSpace], whole: StateSpace) -> float:
    """Quantify emergence: fraction of checked properties that are emergent.

    0.0 = purely reductive (whole = sum of parts).
    1.0 = fully emergent (every property differs).
    """
    if not parts:
        return 0.0

    total_properties = 5  # We check 5 properties in detect_emergence.
    emergent = detect_emergence(parts, whole)
    return len(emergent) / total_properties


# =========================================================================
# Mechanism 6: DIALECTIC (Hegel's Aufhebung)
# =========================================================================


def _get_choices(s: SessionType) -> dict[str, SessionType]:
    """Extract top-level method -> continuation mapping."""
    if isinstance(s, (Branch, Select)):
        return dict(s.choices)
    if isinstance(s, Rec):
        return _get_choices(s.body)
    return {}


def dialectic(thesis: SessionType, antithesis: SessionType) -> SessionType:
    """Hegelian synthesis: UNION of all methods from thesis and antithesis.

    Unlike negotiate (which keeps only shared methods / intersection),
    dialectic PRESERVES all methods from both sides in a unified type.
    For shared methods, recursively apply dialectic on continuations.
    For unique methods, include them as-is.

    Returns a Branch with the union of all methods.
    Returns End if both are End.
    """
    if isinstance(thesis, End) and isinstance(antithesis, End):
        return End()
    if isinstance(thesis, End):
        return antithesis
    if isinstance(antithesis, End):
        return thesis

    c1 = _get_choices(thesis)
    c2 = _get_choices(antithesis)

    if not c1 and not c2:
        return End()
    if not c1:
        return antithesis
    if not c2:
        return thesis

    # Union of methods.
    all_labels = set(c1.keys()) | set(c2.keys())
    choices: list[tuple[str, SessionType]] = []
    for label in sorted(all_labels):
        if label in c1 and label in c2:
            choices.append((label, dialectic(c1[label], c2[label])))
        elif label in c1:
            choices.append((label, c1[label]))
        else:
            choices.append((label, c2[label]))

    return Branch(tuple(choices))


def dialectic_chain(types: list[SessionType]) -> SessionType:
    """Iterated dialectic: thesis + antithesis -> synthesis, then
    synthesis + next -> new synthesis...

    Left fold of dialectic over the list.
    Raises ValueError on empty list.
    """
    if not types:
        raise ValueError("Cannot dialectic an empty list of types")
    return reduce(dialectic, types)


# =========================================================================
# Mechanism 7: METAPHOR (Lakoff's Conceptual Metaphor)
# =========================================================================


def _relabel_ast(s: SessionType, mapping: dict[str, str]) -> SessionType:
    """Apply label mapping to all Branch/Select choices in the AST."""
    if isinstance(s, (End, Wait)):
        return s
    if isinstance(s, Var):
        return s
    if isinstance(s, Branch):
        return Branch(
            tuple((mapping.get(m, m), _relabel_ast(c, mapping)) for m, c in s.choices)
        )
    if isinstance(s, Select):
        return Select(
            tuple((mapping.get(m, m), _relabel_ast(c, mapping)) for m, c in s.choices)
        )
    if isinstance(s, Parallel):
        return Parallel(tuple(_relabel_ast(b, mapping) for b in s.branches))
    if isinstance(s, Rec):
        return Rec(s.var, _relabel_ast(s.body, mapping))
    return s  # pragma: no cover


def metaphor(source: SessionType, target_labels: dict[str, str]) -> SessionType:
    """Apply a conceptual metaphor: relabel source type using target domain
    vocabulary.

    E.g.::

        metaphor(argument_type, {"attack": "criticize", "defend": "justify"})

    Lakoff: 'Argument is war' — same structure, different labels.
    Unmapped labels are preserved unchanged.
    """
    return _relabel_ast(source, target_labels)


def detect_metaphor(s1: SessionType, s2: SessionType) -> dict[str, str] | None:
    """Detect if s2 is a metaphorical relabeling of s1.

    If the state spaces are isomorphic up to label renaming, return the
    label mapping from s1's labels to s2's labels.  Otherwise return None.
    """
    ss1 = build_statespace(s1)
    ss2 = build_statespace(s2)

    # Quick rejection: state and transition counts must match.
    if len(ss1.states) != len(ss2.states):
        return None
    if len(ss1.transitions) != len(ss2.transitions):
        return None

    # Collect all labels from each.
    labels1 = {lbl for _, lbl, _ in ss1.transitions}
    labels2 = {lbl for _, lbl, _ in ss2.transitions}
    if len(labels1) != len(labels2):
        return None

    # Try to find a label bijection.
    # Build transition signature per state: sorted tuple of (out_degree).
    def _out_labels(ss: StateSpace, state: int) -> list[str]:
        return sorted(lbl for src, lbl, _ in ss.transitions if src == state)

    # For simple cases, match labels by structural position.
    # BFS both state spaces in parallel, matching labels as we go.
    mapping: dict[str, str] = {}
    reverse_mapping: dict[str, str] = {}

    visited1: set[int] = set()
    visited2: set[int] = set()
    queue: deque[tuple[int, int]] = deque([(ss1.top, ss2.top)])
    visited1.add(ss1.top)
    visited2.add(ss2.top)

    while queue:
        s1_state, s2_state = queue.popleft()
        out1 = sorted(
            ((lbl, tgt) for src, lbl, tgt in ss1.transitions if src == s1_state),
            key=lambda x: x[0],
        )
        out2 = sorted(
            ((lbl, tgt) for src, lbl, tgt in ss2.transitions if src == s2_state),
            key=lambda x: x[0],
        )

        if len(out1) != len(out2):
            return None

        for (lbl1, tgt1), (lbl2, tgt2) in zip(out1, out2):
            # Check mapping consistency.
            if lbl1 in mapping:
                if mapping[lbl1] != lbl2:
                    return None
            elif lbl2 in reverse_mapping:
                if reverse_mapping[lbl2] != lbl1:
                    return None
            else:
                mapping[lbl1] = lbl2
                reverse_mapping[lbl2] = lbl1

            if tgt1 not in visited1 and tgt2 not in visited2:
                visited1.add(tgt1)
                visited2.add(tgt2)
                queue.append((tgt1, tgt2))

    if not mapping:
        # Both are trivial (End only) — identity mapping.
        return {}

    return mapping


def metaphor_quality(
    mapping: dict[str, str],
    source_ss: StateSpace,
    target_ss: StateSpace,
) -> float:
    """How good is a metaphorical mapping?

    1.0 = perfect (isomorphism under relabeling).
    0.0 = no structural correspondence.
    """
    if not source_ss.transitions and not target_ss.transitions:
        return 1.0

    # Apply mapping to source transitions.
    relabeled = {
        (src, mapping.get(lbl, lbl), tgt)
        for src, lbl, tgt in source_ss.transitions
    }
    target_set = set(target_ss.transitions)

    if not relabeled and not target_set:
        return 1.0

    # Count matching transitions (by label pattern, not state IDs).
    # Since state IDs may differ, compare label multisets.
    source_labels = sorted(mapping.get(lbl, lbl) for _, lbl, _ in source_ss.transitions)
    target_labels = sorted(lbl for _, lbl, _ in target_ss.transitions)

    if not source_labels and not target_labels:
        return 1.0

    # Label distribution match.
    from collections import Counter

    c1 = Counter(source_labels)
    c2 = Counter(target_labels)

    all_keys = set(c1.keys()) | set(c2.keys())
    if not all_keys:
        return 1.0

    matches = sum(min(c1.get(k, 0), c2.get(k, 0)) for k in all_keys)
    total = max(sum(c1.values()), sum(c2.values()))

    return matches / total if total > 0 else 1.0
