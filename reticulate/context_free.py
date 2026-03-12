"""Context-free session type analysis (Step 14).

Session types can be classified by the Chomsky hierarchy:

1. **Regular** (finite automaton): Types where recursion is tail-recursive —
   the state space is finite and corresponds to a regular language of traces.
   Most practical protocols fall here.

2. **Context-free** (pushdown automaton): Types where recursion is NOT
   tail-recursive — the recursive call has a continuation, requiring a stack.
   Example: rec X . &{push: X . &{pop: end}} (matched push/pop).

3. **Context-sensitive and beyond**: Not expressible in our type system.

This module characterizes session types along this spectrum:

- ``is_regular(s)`` — True if the type is regular (finite state space).
- ``classify_chomsky(s)`` — Classify as regular or context-free.
- ``check_context_free(s)`` — Detailed analysis with metrics.

Key insight: ALL session types in our grammar produce finite state spaces
(because recursion creates cycles, not infinite unfolding). The "context-free"
distinction is at the LANGUAGE level — the set of valid traces may be
context-free even though the state space is finite. However, types with
non-tail recursion (via Continuation) exhibit pushdown-like behavior in
their trace semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

from reticulate.parser import (
    Branch,
    Continuation,
    End,
    Parallel,
    Rec,
    Select,
    SessionType,
    Var,
    Wait,
    pretty,
)
from reticulate.recursion import (
    count_rec_binders,
    is_tail_recursive,
    rec_depth,
    recursive_vars,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChomskyClassification:
    """Classification of a session type in the Chomsky hierarchy.

    Attributes:
        level: "regular" or "context-free".
        is_regular: True iff the type is regular (tail-recursive or non-recursive).
        has_continuation_recursion: True if recursion occurs inside Continuation.
        trace_language_class: Human-readable description.
        num_rec_binders: Number of recursion binders.
        max_rec_depth: Maximum nesting depth of rec binders.
        stack_depth_bound: Upper bound on needed stack depth (0 for regular).
    """
    level: str
    is_regular: bool
    has_continuation_recursion: bool
    trace_language_class: str
    num_rec_binders: int
    max_rec_depth: int
    stack_depth_bound: int


# ---------------------------------------------------------------------------
# Regularity checking
# ---------------------------------------------------------------------------

def is_regular(s: SessionType) -> bool:
    """Check whether a session type is regular.

    A type is regular if:
    1. It has no recursion, or
    2. All recursion is tail-recursive (no continuation after recursive call).

    Regular types correspond to finite automata — their trace language
    is a regular language.
    """
    return classify_chomsky(s).is_regular


def has_continuation_recursion(s: SessionType) -> bool:
    """Check whether recursion occurs inside a Continuation.

    This is the key indicator of context-free behavior: after a recursive
    call completes, there is more protocol to execute, requiring a stack
    to track the return point.
    """
    return _find_continuation_recursion(s, set())


def _find_continuation_recursion(s: SessionType, bound_vars: set[str]) -> bool:
    """Check for recursive variables used inside continuation left-hand side."""
    match s:
        case End() | Wait() | Var():
            return False
        case Branch(choices=choices) | Select(choices=choices):
            return any(
                _find_continuation_recursion(body, bound_vars)
                for _, body in choices
            )
        case Parallel(branches=branches):
            return any(
                _find_continuation_recursion(b, bound_vars)
                for b in branches
            )
        case Continuation(left=left, right=right):
            # Check if the left side contains any bound recursive vars
            from reticulate.termination import _free_vars
            left_free = _free_vars(left)
            if left_free & bound_vars:
                return True
            return (
                _find_continuation_recursion(left, bound_vars)
                or _find_continuation_recursion(right, bound_vars)
            )
        case Rec(var=var, body=body):
            return _find_continuation_recursion(body, bound_vars | {var})
        case _:
            return False


# ---------------------------------------------------------------------------
# Stack depth analysis
# ---------------------------------------------------------------------------

def stack_depth_bound(s: SessionType) -> int:
    """Compute an upper bound on the stack depth needed.

    For regular types, this is 0.
    For context-free types, this is the nesting depth of continuation
    recursion — how many pending continuations can stack up.

    In practice, our types always produce finite state spaces, so the
    actual "stack" is bounded. This metric measures the conceptual
    stack depth of the trace language.
    """
    if not has_continuation_recursion(s):
        return 0
    return _compute_stack_bound(s, set(), 0)


def _compute_stack_bound(
    s: SessionType,
    bound_vars: set[str],
    current_depth: int,
) -> int:
    """Compute stack bound recursively."""
    match s:
        case End() | Wait() | Var():
            return current_depth
        case Branch(choices=choices) | Select(choices=choices):
            if not choices:
                return current_depth
            return max(
                _compute_stack_bound(body, bound_vars, current_depth)
                for _, body in choices
            )
        case Parallel(branches=branches):
            return max(
                (_compute_stack_bound(b, bound_vars, current_depth) for b in branches),
                default=current_depth,
            )
        case Continuation(left=left, right=right):
            from reticulate.termination import _free_vars
            left_free = _free_vars(left)
            depth = current_depth
            if left_free & bound_vars:
                depth = current_depth + 1
            return max(
                _compute_stack_bound(left, bound_vars, depth),
                _compute_stack_bound(right, bound_vars, current_depth),
            )
        case Rec(var=var, body=body):
            return _compute_stack_bound(body, bound_vars | {var}, current_depth)
        case _:
            return current_depth


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_chomsky(s: SessionType) -> ChomskyClassification:
    """Classify a session type in the Chomsky hierarchy."""
    n_rec = count_rec_binders(s)
    depth = rec_depth(s)
    has_cont_rec = has_continuation_recursion(s)

    if n_rec == 0:
        return ChomskyClassification(
            level="regular",
            is_regular=True,
            has_continuation_recursion=False,
            trace_language_class="finite (no recursion)",
            num_rec_binders=0,
            max_rec_depth=0,
            stack_depth_bound=0,
        )

    if not has_cont_rec:
        return ChomskyClassification(
            level="regular",
            is_regular=True,
            has_continuation_recursion=False,
            trace_language_class="regular (tail-recursive)",
            num_rec_binders=n_rec,
            max_rec_depth=depth,
            stack_depth_bound=0,
        )

    sdb = stack_depth_bound(s)
    return ChomskyClassification(
        level="context-free",
        is_regular=False,
        has_continuation_recursion=True,
        trace_language_class="context-free (continuation recursion)",
        num_rec_binders=n_rec,
        max_rec_depth=depth,
        stack_depth_bound=sdb,
    )


# ---------------------------------------------------------------------------
# Trace regularity analysis
# ---------------------------------------------------------------------------

def analyze_trace_language(s: SessionType) -> dict[str, object]:
    """Analyze the trace language properties of a session type.

    Returns a dict with:
    - chomsky_class: "regular" or "context-free"
    - is_finite: True if trace set is finite (no recursion)
    - is_regular: True if trace language is regular
    - uses_recursion: True if type has rec binders
    - uses_parallel: True if type has parallel constructor
    - uses_continuation: True if type has continuation constructor
    - state_count: Number of states in state space
    - transition_count: Number of transitions
    """
    from reticulate.statespace import build_statespace

    classification = classify_chomsky(s)
    ss = build_statespace(s)

    return {
        "chomsky_class": classification.level,
        "is_finite": count_rec_binders(s) == 0,
        "is_regular": classification.is_regular,
        "uses_recursion": count_rec_binders(s) > 0,
        "uses_parallel": _contains_parallel(s),
        "uses_continuation": _contains_continuation(s),
        "state_count": len(ss.states),
        "transition_count": len(ss.transitions),
    }


def _contains_parallel(s: SessionType) -> bool:
    """Check if type contains Parallel constructor."""
    match s:
        case Parallel():
            return True
        case Branch(choices=choices) | Select(choices=choices):
            return any(_contains_parallel(body) for _, body in choices)
        case Continuation(left=left, right=right):
            return _contains_parallel(left) or _contains_parallel(right)
        case Rec(body=body):
            return _contains_parallel(body)
        case _:
            return False


def _contains_continuation(s: SessionType) -> bool:
    """Check if type contains Continuation constructor."""
    match s:
        case Continuation():
            return True
        case Branch(choices=choices) | Select(choices=choices):
            return any(_contains_continuation(body) for _, body in choices)
        case Parallel(branches=branches):
            return any(_contains_continuation(b) for b in branches)
        case Rec(body=body):
            return _contains_continuation(body)
        case _:
            return False
