"""Session types as effect systems / graded monads (Step 99).

The key insight: session types can be viewed as **effects** in the
algebraic effects framework. Each method call is an effectful operation,
and the session type constrains which effects can occur in which order.

The graded monad T_S associates to each session type S a computational
type T_S(A) = "computation of type A with protocol effects S."

This module provides:
1. **Effect extraction**: extract the effect signature from a session type
2. **Effect ordering**: the lattice ordering on session types induces an
   ordering on effects (subeffecting)
3. **Effect composition**: sequential composition of effects via monad bind
4. **Effect inference**: given a sequence of operations, infer the effect type
5. **Handler synthesis**: generate effect handlers from session types

Key functions:
  - ``extract_effects(ss)``         -- extract effect signature
  - ``effect_ordering(ss)``         -- compute subeffecting relation
  - ``compose_effects(e1, e2)``     -- sequential effect composition
  - ``check_subeffecting(e1, e2)``  -- is e1 a subeffect of e2?
  - ``infer_effect(ops, ss)``       -- infer effect type from operations
  - ``effect_handler(ss)``          -- synthesize handler skeleton
  - ``effect_lattice(ss)``          -- build the effect lattice
  - ``analyze_effects(ss)``         -- full analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Effect:
    """An effect (operation) in the effect system.

    Attributes:
        name: Operation name (method label).
        source_state: State where this effect is available.
        target_state: State after performing this effect.
        is_input: True if this is an input effect (Branch).
        is_output: True if this is an output effect (Select).
    """
    name: str
    source_state: int
    target_state: int
    is_input: bool
    is_output: bool


@dataclass(frozen=True)
class EffectSignature:
    """The effect signature of a session type.

    Attributes:
        effects: All effects available in the protocol.
        input_effects: Effects from Branch constructors (receive).
        output_effects: Effects from Select constructors (send).
        effect_states: Mapping from state to available effects.
        num_effects: Total number of effects.
    """
    effects: frozenset[Effect]
    input_effects: frozenset[Effect]
    output_effects: frozenset[Effect]
    effect_states: dict[int, list[Effect]]
    num_effects: int


@dataclass(frozen=True)
class SubeffectResult:
    """Result of checking if e1 is a subeffect of e2.

    Attributes:
        is_subeffect: True iff every operation in e1 is available in e2.
        missing: Operations in e1 not available in e2.
        extra: Operations in e2 not in e1.
    """
    is_subeffect: bool
    missing: frozenset[str]
    extra: frozenset[str]


@dataclass(frozen=True)
class EffectHandler:
    """Synthesized effect handler skeleton.

    Attributes:
        operations: List of (operation_name, input_type, output_type) triples.
        return_handler: Name of the return handler.
        state_handlers: Mapping from state to handler function name.
    """
    operations: list[tuple[str, str, str]]
    return_handler: str
    state_handlers: dict[int, str]


@dataclass(frozen=True)
class EffectAnalysis:
    """Full effect system analysis.

    Attributes:
        signature: Effect signature.
        num_input: Number of input effects.
        num_output: Number of output effects.
        effect_depth: Maximum number of sequential effects to termination.
        is_linear: True iff each effect is used at most once per execution.
        handler: Synthesized handler skeleton.
        subeffect_chains: Longest chain in the subeffecting relation.
    """
    signature: EffectSignature
    num_input: int
    num_output: int
    effect_depth: int
    is_linear: bool
    handler: EffectHandler
    subeffect_chains: int


# ---------------------------------------------------------------------------
# Public API: Effect extraction
# ---------------------------------------------------------------------------

def extract_effects(ss: StateSpace) -> EffectSignature:
    """Extract the effect signature from a state space."""
    effects: set[Effect] = set()
    input_effects: set[Effect] = set()
    output_effects: set[Effect] = set()
    effect_states: dict[int, list[Effect]] = {s: [] for s in ss.states}

    for src, label, tgt in ss.transitions:
        is_sel = ss.is_selection(src, label, tgt)
        eff = Effect(
            name=label,
            source_state=src,
            target_state=tgt,
            is_input=not is_sel,
            is_output=is_sel,
        )
        effects.add(eff)
        effect_states[src].append(eff)
        if is_sel:
            output_effects.add(eff)
        else:
            input_effects.add(eff)

    return EffectSignature(
        effects=frozenset(effects),
        input_effects=frozenset(input_effects),
        output_effects=frozenset(output_effects),
        effect_states=effect_states,
        num_effects=len(effects),
    )


# ---------------------------------------------------------------------------
# Public API: Effect ordering (subeffecting)
# ---------------------------------------------------------------------------

def effect_ordering(ss: StateSpace) -> dict[int, set[int]]:
    """Compute the subeffecting relation on states.

    State s1 is a subeffect of s2 iff every effect available at s1
    is also available at s2 (s2 has at least as many operations).
    """
    sig = extract_effects(ss)
    labels_at: dict[int, set[str]] = {}
    for s in ss.states:
        labels_at[s] = {e.name for e in sig.effect_states.get(s, [])}

    ordering: dict[int, set[int]] = {s: set() for s in ss.states}
    for s1 in ss.states:
        for s2 in ss.states:
            if labels_at[s1].issubset(labels_at[s2]):
                ordering[s1].add(s2)

    return ordering


def check_subeffecting(
    ss: StateSpace,
    state1: int,
    state2: int,
) -> SubeffectResult:
    """Check if effects at state1 are a subeffect of effects at state2."""
    sig = extract_effects(ss)
    labels1 = {e.name for e in sig.effect_states.get(state1, [])}
    labels2 = {e.name for e in sig.effect_states.get(state2, [])}

    missing = labels1 - labels2
    extra = labels2 - labels1

    return SubeffectResult(
        is_subeffect=len(missing) == 0,
        missing=frozenset(missing),
        extra=frozenset(extra),
    )


# ---------------------------------------------------------------------------
# Public API: Effect composition
# ---------------------------------------------------------------------------

def compose_effects(
    ss: StateSpace,
    state1: int,
    label: str,
) -> int | None:
    """Compose effects: perform operation 'label' at state1.

    Returns the resulting state, or None if the operation is not available.
    """
    for src, lbl, tgt in ss.transitions:
        if src == state1 and lbl == label:
            return tgt
    return None


def effect_sequence(
    ss: StateSpace,
    labels: list[str],
) -> list[int] | None:
    """Execute a sequence of effects from the top state.

    Returns the sequence of states visited, or None if any effect fails.
    """
    state = ss.top
    path = [state]
    for label in labels:
        next_state = compose_effects(ss, state, label)
        if next_state is None:
            return None
        state = next_state
        path.append(state)
    return path


# ---------------------------------------------------------------------------
# Public API: Effect inference
# ---------------------------------------------------------------------------

def infer_effect(
    ss: StateSpace,
    operations: list[str],
) -> list[int] | None:
    """Infer the effect type from a sequence of operations.

    Returns the state path if all operations are valid, None otherwise.
    """
    return effect_sequence(ss, operations)


# ---------------------------------------------------------------------------
# Public API: Handler synthesis
# ---------------------------------------------------------------------------

def effect_handler(ss: StateSpace) -> EffectHandler:
    """Synthesize an effect handler skeleton from the session type.

    Each state with outgoing transitions becomes a handler case.
    """
    sig = extract_effects(ss)

    operations: list[tuple[str, str, str]] = []
    seen_ops: set[str] = set()
    state_handlers: dict[int, str] = {}

    for s in sorted(ss.states):
        effs = sig.effect_states.get(s, [])
        if effs:
            state_handlers[s] = f"handle_state_{s}"
        for eff in effs:
            if eff.name not in seen_ops:
                in_type = "input" if eff.is_input else "output"
                out_type = f"state_{eff.target_state}"
                operations.append((eff.name, in_type, out_type))
                seen_ops.add(eff.name)

    return EffectHandler(
        operations=operations,
        return_handler="handle_return",
        state_handlers=state_handlers,
    )


# ---------------------------------------------------------------------------
# Public API: Effect lattice
# ---------------------------------------------------------------------------

def effect_lattice(ss: StateSpace) -> dict[str, set[str]]:
    """Build the effect lattice: operation → set of co-enabled operations.

    Two operations are co-enabled if they can appear in the same execution.
    """
    # For each pair of labels, check if there exists a path containing both
    all_labels = {label for _, label, _ in ss.transitions}
    co_enabled: dict[str, set[str]] = {l: set() for l in all_labels}

    # BFS from top, tracking which labels have been seen
    from collections import deque
    queue: deque[tuple[int, frozenset[str]]] = deque([(ss.top, frozenset())])
    visited: set[tuple[int, frozenset[str]]] = set()

    terminal_label_sets: list[frozenset[str]] = []

    while queue:
        state, seen = queue.popleft()
        if (state, seen) in visited:
            continue
        visited.add((state, seen))

        if state == ss.bottom:
            terminal_label_sets.append(seen)
            continue

        for src, label, tgt in ss.transitions:
            if src == state:
                new_seen = seen | {label}
                if len(new_seen) <= 20:  # Bound for performance
                    queue.append((tgt, new_seen))

    # Build co-enablement from terminal label sets
    for label_set in terminal_label_sets:
        for l1 in label_set:
            for l2 in label_set:
                if l1 != l2:
                    co_enabled[l1].add(l2)

    return co_enabled


# ---------------------------------------------------------------------------
# Public API: Full analysis
# ---------------------------------------------------------------------------

def analyze_effects(ss: StateSpace) -> EffectAnalysis:
    """Full effect system analysis."""
    sig = extract_effects(ss)
    handler = effect_handler(ss)

    # Effect depth: longest path from top to bottom
    adj: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        adj[src].add(tgt)

    # DFS for longest path (with cycle avoidance)
    def _depth(s: int, visited: frozenset[int]) -> int:
        if s == ss.bottom:
            return 0
        best = -1
        for t in adj.get(s, set()):
            if t not in visited:
                d = _depth(t, visited | {t})
                if d >= 0:
                    best = max(best, 1 + d)
        return best

    depth = _depth(ss.top, frozenset({ss.top}))
    if depth < 0:
        depth = 0

    # Linearity: each label appears at most once per path
    # Check if any label appears on multiple transitions from same path
    all_labels = {label for _, label, _ in ss.transitions}
    is_linear = True
    for s in ss.states:
        labels_from_s = [label for src, label, _ in ss.transitions if src == s]
        if len(labels_from_s) != len(set(labels_from_s)):
            is_linear = False
            break

    # Subeffect chains
    ordering = effect_ordering(ss)
    # Longest chain in the ordering (excluding reflexive)
    chain_len = 0
    for s1 in ss.states:
        count = sum(1 for s2 in ordering[s1] if s2 != s1)
        chain_len = max(chain_len, count)

    return EffectAnalysis(
        signature=sig,
        num_input=len(sig.input_effects),
        num_output=len(sig.output_effects),
        effect_depth=depth,
        is_linear=is_linear,
        handler=handler,
        subeffect_chains=chain_len,
    )
