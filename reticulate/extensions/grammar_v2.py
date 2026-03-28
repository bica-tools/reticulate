"""Grammar extensions toward real-world session types (Step 500).

This module extends the session type grammar progressively with real-world
constructs.  Each extension is a NEW AST node plus state-space construction
rule.  The key question for each: DOES THE LATTICE PROPERTY SURVIVE?

Extensions
----------

1. **Timeout** -- timed session type with fallback
2. **TryCatch** -- exception handling
3. **Priority** -- prioritised external choice
4. **Probabilistic** -- probabilistic internal choice
5. **Guard** -- guarded transitions
6. **Interrupt** -- interruptible process (the key real-world extension)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reticulate.parser import (
    Branch,
    End,
    Select,
    SessionType,
)
from reticulate.statespace import StateSpace, build_statespace


# ---------------------------------------------------------------------------
# 1. New AST nodes (frozen dataclasses)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Timeout:
    """Timed session type: execute *body*; if not completed within *limit*
    abstract time units, run *fallback*.

    Syntax sketch::

        try body timeout fallback
    """

    body: SessionType
    fallback: SessionType
    limit: int = 1  # abstract time units


@dataclass(frozen=True)
class TryCatch:
    """Exception handling: execute *body*; if an exception occurs, run
    *handler*.

    Syntax sketch::

        try body catch handler
    """

    body: SessionType
    handler: SessionType


@dataclass(frozen=True)
class Priority:
    """Prioritised external choice.  Structurally like ``Branch`` but the
    ordering of *choices* encodes priority (first entry = highest).

    Syntax sketch::

        priority{high: S1, medium: S2, low: S3}
    """

    choices: tuple[tuple[str, SessionType], ...]


@dataclass(frozen=True)
class Probabilistic:
    """Probabilistic internal choice.  Structurally like ``Select`` but each
    branch carries a probability weight.

    Syntax sketch::

        prob{0.9: S1, 0.1: S2}
    """

    choices: tuple[tuple[float, SessionType], ...]


@dataclass(frozen=True)
class Guard:
    """Guarded transition: proceed only if a symbolic condition holds.

    Syntax sketch::

        guard(cond) then_branch else else_branch
    """

    condition: str
    then_branch: SessionType
    else_branch: SessionType


@dataclass(frozen=True)
class Interrupt:
    """Interruptible process.  The *body* can be interrupted at ANY state
    by *signal*, transferring control to *handler*.

    This is the key real-world extension -- it models exceptions, signals,
    timeouts, and cancellations.

    Syntax sketch::

        interrupt(signal) body handler
    """

    signal: str
    body: SessionType
    handler: SessionType


# ---------------------------------------------------------------------------
# 2. Core helper: add escape transitions
# ---------------------------------------------------------------------------

def _add_escape_transitions(
    body_ss: StateSpace,
    handler_ss: StateSpace,
    label: str,
) -> StateSpace:
    """Add an escape transition from every non-bottom body state to the
    handler's top state.

    This is the shared operation for :class:`Timeout`, :class:`TryCatch`,
    and :class:`Interrupt`.

    The handler's bottom is merged with the body's bottom so that both
    normal and exceptional paths converge to the same terminal state.
    """
    # Remap handler state IDs to avoid collisions with body IDs.
    max_id = max(body_ss.states | handler_ss.states) + 1
    remap: dict[int, int] = {s: s + max_id for s in handler_ss.states}

    new_states: set[int] = set(body_ss.states) | {remap[s] for s in handler_ss.states}
    new_transitions: list[tuple[int, str, int]] = list(body_ss.transitions)

    # Carry over handler transitions (remapped).
    for src, lbl, tgt in handler_ss.transitions:
        new_transitions.append((remap[src], lbl, remap[tgt]))

    # Escape transitions: every non-bottom body state -> handler top.
    handler_top = remap[handler_ss.top]
    for s in body_ss.states:
        if s != body_ss.bottom:
            new_transitions.append((s, label, handler_top))

    # Merge handler bottom into body bottom so both paths converge.
    handler_bottom = remap[handler_ss.bottom]
    if handler_bottom != body_ss.bottom:
        merged_transitions: list[tuple[int, str, int]] = []
        for src, lbl, tgt in new_transitions:
            new_src = body_ss.bottom if src == handler_bottom else src
            new_tgt = body_ss.bottom if tgt == handler_bottom else tgt
            merged_transitions.append((new_src, lbl, new_tgt))
        new_transitions = merged_transitions
        new_states.discard(handler_bottom)

    # Build labels dict.
    new_labels: dict[int, str] = dict(body_ss.labels)
    for s in handler_ss.states:
        remapped = remap[s]
        if remapped in new_states:
            new_labels[remapped] = handler_ss.labels.get(s, f"h_{s}")

    # Propagate selection_transitions.
    new_selection: set[tuple[int, str, int]] = set(body_ss.selection_transitions)
    for src, lbl, tgt in handler_ss.selection_transitions:
        new_src = body_ss.bottom if remap[src] == handler_bottom else remap[src]
        new_tgt = body_ss.bottom if remap[tgt] == handler_bottom else remap[tgt]
        new_selection.add((new_src, lbl, new_tgt))

    return StateSpace(
        states=new_states,
        transitions=new_transitions,
        top=body_ss.top,
        bottom=body_ss.bottom,
        labels=new_labels,
        selection_transitions=new_selection,
    )


# ---------------------------------------------------------------------------
# 3. State-space construction for each extension
# ---------------------------------------------------------------------------

def build_timeout_statespace(node: Timeout) -> StateSpace:
    """Build state space for :class:`Timeout`.

    The body's state space gains an extra ``"timeout"`` transition from every
    non-bottom state to the fallback's top state, modelling the timeout
    firing at any point.
    """
    body_ss = build_statespace(node.body)
    fallback_ss = build_statespace(node.fallback)
    return _add_escape_transitions(body_ss, fallback_ss, "timeout")


def build_trycatch_statespace(node: TryCatch) -> StateSpace:
    """Build state space for :class:`TryCatch`.

    Structurally identical to :func:`build_timeout_statespace` but the escape
    label is ``"exception"`` instead of ``"timeout"``.
    """
    body_ss = build_statespace(node.body)
    handler_ss = build_statespace(node.handler)
    return _add_escape_transitions(body_ss, handler_ss, "exception")


def build_priority_statespace(node: Priority) -> StateSpace:
    """Build state space for :class:`Priority`.

    Priority is a semantic annotation; structurally the state space is
    identical to a ``Branch`` with the same choices.
    """
    branch = Branch(node.choices)
    return build_statespace(branch)


def build_probabilistic_statespace(node: Probabilistic) -> StateSpace:
    """Build state space for :class:`Probabilistic`.

    Probabilities are semantic annotations; structurally the state space
    is identical to a ``Select`` with labels derived from the probability
    indices.
    """
    # Convert (probability, type) pairs to (label, type) pairs for Select.
    select_choices: list[tuple[str, SessionType]] = []
    for i, (prob, st) in enumerate(node.choices):
        select_choices.append((f"p{i}_{prob}", st))
    select = Select(tuple(select_choices))
    return build_statespace(select)


def build_guard_statespace(node: Guard) -> StateSpace:
    """Build state space for :class:`Guard`.

    Creates a two-branch structure equivalent to
    ``&{<cond>_true: then_branch, <cond>_false: else_branch}``.
    """
    branch = Branch((
        (f"{node.condition}_true", node.then_branch),
        (f"{node.condition}_false", node.else_branch),
    ))
    return build_statespace(branch)


def build_interrupt_statespace(node: Interrupt) -> StateSpace:
    """Build state space for :class:`Interrupt`.

    Every state in the body gets an extra transition labelled with the
    signal name to the handler's top state.  This creates a massive number
    of new transitions.

    For a body with *N* states this adds *N* - 1 new transitions (all
    except bottom).  The resulting state space may or may not be a lattice
    -- **this is the key question**.
    """
    body_ss = build_statespace(node.body)
    handler_ss = build_statespace(node.handler)
    return _add_escape_transitions(body_ss, handler_ss, node.signal)


# Dispatch table for convenience.
_BUILDERS: dict[type, Any] = {
    Timeout: build_timeout_statespace,
    TryCatch: build_trycatch_statespace,
    Priority: build_priority_statespace,
    Probabilistic: build_probabilistic_statespace,
    Guard: build_guard_statespace,
    Interrupt: build_interrupt_statespace,
}


def build_extension_statespace(node: Any) -> StateSpace:
    """Dispatch to the appropriate builder for a grammar-v2 AST node."""
    builder = _BUILDERS.get(type(node))
    if builder is None:
        raise TypeError(f"No grammar-v2 builder for {type(node).__name__}")
    return builder(node)


# ---------------------------------------------------------------------------
# 4. Lattice preservation analysis
# ---------------------------------------------------------------------------

def check_extension_lattice(
    extension_name: str,
    body_type_str: str,
    handler_type_str: str = "&{recover: end}",
) -> dict[str, Any]:
    """Check whether a grammar extension preserves the lattice property.

    Returns a dict with keys: ``extension``, ``body_states``,
    ``total_states``, ``total_transitions``, ``is_lattice``,
    ``counterexample``.
    """
    from reticulate.lattice import check_lattice
    from reticulate.parser import parse

    body = parse(body_type_str)
    handler = parse(handler_type_str)
    body_ss = build_statespace(body)
    handler_ss = build_statespace(handler)

    if extension_name == "timeout":
        ss = _add_escape_transitions(body_ss, handler_ss, "timeout")
    elif extension_name == "trycatch":
        ss = _add_escape_transitions(body_ss, handler_ss, "exception")
    elif extension_name == "interrupt":
        ss = _add_escape_transitions(body_ss, handler_ss, "interrupt")
    elif extension_name == "priority":
        ss = body_ss  # structurally identical to Branch
    elif extension_name == "probabilistic":
        ss = body_ss  # structurally identical to Select
    elif extension_name == "guard":
        ss = build_statespace(Branch((
            ("guard_true", body),
            ("guard_false", handler),
        )))
    else:
        raise ValueError(f"Unknown extension: {extension_name}")

    result = check_lattice(ss)
    return {
        "extension": extension_name,
        "body_states": len(body_ss.states),
        "total_states": len(ss.states),
        "total_transitions": len(ss.transitions),
        "is_lattice": result.is_lattice,
        "counterexample": result.counterexample,
    }


def survey_lattice_preservation() -> dict[str, dict[str, Any]]:
    """Run lattice preservation check for ALL extensions across multiple
    body types and handlers.

    Returns a dict mapping ``extension_name`` to a summary dict with keys:
    ``total_tests``, ``lattice_count``, ``non_lattice_count``,
    ``error_count``, ``details``.
    """
    bodies = [
        "end",
        "&{a: end, b: end}",
        "&{a: +{x: end, y: end}, b: end}",
        "rec X . &{a: +{ok: X, done: end}}",
        "&{a: &{b: +{c: end}}}",
    ]
    handlers = [
        "&{recover: end}",
        "+{fail: end}",
        "&{log: +{retry: end, abort: end}}",
    ]

    extensions = [
        "timeout",
        "trycatch",
        "interrupt",
        "priority",
        "probabilistic",
        "guard",
    ]

    results: dict[str, dict[str, Any]] = {}
    for ext in extensions:
        ext_results: list[dict[str, Any]] = []
        for body in bodies:
            for handler in handlers:
                try:
                    r = check_extension_lattice(ext, body, handler)
                    ext_results.append(r)
                except Exception as e:
                    ext_results.append({"extension": ext, "error": str(e)})
        results[ext] = {
            "total_tests": len(ext_results),
            "lattice_count": sum(
                1 for r in ext_results if r.get("is_lattice", False)
            ),
            "non_lattice_count": sum(
                1
                for r in ext_results
                if not r.get("is_lattice", True) and "error" not in r
            ),
            "error_count": sum(1 for r in ext_results if "error" in r),
            "details": ext_results,
        }
    return results
