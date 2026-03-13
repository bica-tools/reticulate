"""State-space construction for session types.

Given a session type AST, constructs the labeled transition system (LTS)
representing all reachable protocol states and transitions.

For sequential types, the state space is a directed graph (finite automaton).
For parallel types (∥), the state space is the product of the component
state spaces (see product.py and docs/specs/parallel-constructor-spec.md §4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
)
from reticulate.product import product_statespace


@dataclass
class StateSpace:
    """Labeled transition system for a session type.

    - ``states``: set of integer state IDs
    - ``transitions``: list of ``(source, label, target)`` triples
    - ``top``: initial protocol state (⊤)
    - ``bottom``: terminal state (⊥ = end)
    - ``labels``: human-readable description for each state
    - ``selection_transitions``: set of ``(source, label, target)`` triples
      that are selection (internal choice) transitions
    """

    states: set[int] = field(default_factory=set)
    transitions: list[tuple[int, str, int]] = field(default_factory=list)
    top: int = 0
    bottom: int = 0
    labels: dict[int, str] = field(default_factory=dict)
    selection_transitions: set[tuple[int, str, int]] = field(default_factory=set)

    def enabled(self, state: int) -> list[tuple[str, int]]:
        """Return ``(label, target)`` pairs for transitions from *state*."""
        return [(l, t) for s, l, t in self.transitions if s == state]

    def successors(self, state: int) -> set[int]:
        """Direct successor states of *state*."""
        return {t for s, l, t in self.transitions if s == state}

    def reachable_from(self, state: int) -> set[int]:
        """All states reachable from *state* (inclusive)."""
        visited: set[int] = set()
        stack = [state]
        while stack:
            s = stack.pop()
            if s in visited:
                continue
            visited.add(s)
            for t in self.successors(s):
                stack.append(t)
        return visited

    def is_selection(self, src: int, label: str, tgt: int) -> bool:
        """Return True if the transition ``(src, label, tgt)`` is a selection."""
        return (src, label, tgt) in self.selection_transitions

    def enabled_methods(self, state: int) -> list[tuple[str, int]]:
        """Return ``(label, target)`` pairs for METHOD transitions from *state*."""
        return [
            (l, t) for s, l, t in self.transitions
            if s == state and (s, l, t) not in self.selection_transitions
        ]

    def enabled_selections(self, state: int) -> list[tuple[str, int]]:
        """Return ``(label, target)`` pairs for SELECTION transitions from *state*."""
        return [
            (l, t) for s, l, t in self.transitions
            if s == state and (s, l, t) in self.selection_transitions
        ]


def build_statespace(session_type: SessionType) -> StateSpace:
    """Construct the labeled transition system for a session type.

    Handles all constructors: ``End``, ``Wait``, ``Var``, ``Branch``,
    ``Select``, ``Rec`` (with cycles), ``Continuation`` (chaining after
    parallel), and ``Parallel`` (product construction via
    :func:`product_statespace`).

    Raises ``ValueError`` on unbound type variables.
    """
    builder = _Builder()
    return builder.build(session_type)


# ---------------------------------------------------------------------------
# Internal builder
# ---------------------------------------------------------------------------

class _Builder:
    """Accumulates states and transitions during construction."""

    def __init__(self) -> None:
        self._next_id: int = 0
        self._states: dict[int, str] = {}
        self._transitions: list[tuple[int, str, int]] = []
        self._selection_transitions: set[tuple[int, str, int]] = set()

    def _fresh(self, label: str) -> int:
        sid = self._next_id
        self._next_id += 1
        self._states[sid] = label
        return sid

    # -- public entry point -------------------------------------------------

    def build(self, node: SessionType) -> StateSpace:
        end_id = self._fresh("end")
        top_id = self._build(node, {}, end_id)

        reachable = self._reachable(top_id)
        reachable_transitions = [
            (s, l, t) for s, l, t in self._transitions if s in reachable
        ]
        reachable_selections = {
            (s, l, t) for s, l, t in self._selection_transitions if s in reachable
        }
        return StateSpace(
            states=reachable,
            transitions=reachable_transitions,
            top=top_id,
            bottom=end_id,
            labels={s: self._states[s] for s in reachable if s in self._states},
            selection_transitions=reachable_selections,
        )

    # -- recursive construction ---------------------------------------------

    def _build(
        self,
        node: SessionType,
        env: dict[str, int],
        end_id: int,
    ) -> int:
        """Return the entry state ID for *node*."""
        match node:
            case End():
                return end_id

            case Wait():
                # Wait is semantically terminal (like End) at the
                # state-space level.  The distinction between wait
                # and end is enforced by well-formedness checking.
                return end_id

            case Var(name=name):
                if name not in env:
                    raise ValueError(f"unbound type variable: {name!r}")
                return env[name]

            case Branch(choices=choices):
                if len(choices) == 1:
                    label = choices[0][0]
                else:
                    label = "&{" + ", ".join(l for l, _ in choices) + "}"
                entry = self._fresh(label)
                for lbl, body in choices:
                    target = self._build(body, env, end_id)
                    self._transitions.append((entry, lbl, target))
                return entry

            case Select(choices=choices):
                label = "+{" + ", ".join(l for l, _ in choices) + "}"
                entry = self._fresh(label)
                for lbl, body in choices:
                    target = self._build(body, env, end_id)
                    tr = (entry, lbl, target)
                    self._transitions.append(tr)
                    self._selection_transitions.add(tr)
                return entry

            case Rec(var=var, body=body):
                # Pre-allocate a placeholder for the recursion variable.
                # After building the body, merge the placeholder into
                # the body's actual entry state (creating the cycle).
                placeholder = self._fresh(f"rec_{var}")
                new_env = {**env, var: placeholder}
                body_entry = self._build(body, new_env, end_id)
                if body_entry != placeholder:
                    self._merge(placeholder, body_entry)
                    return body_entry
                return placeholder

            case Continuation(left=left, right=right):
                # Build right side first; left's bottom chains to right's entry.
                right_entry = self._build(right, env, end_id)
                left_entry = self._build(left, env, right_entry)
                return left_entry

            case Parallel(branches=branches):
                return self._build_parallel(branches, end_id)

            case _:
                raise TypeError(f"unknown AST node: {type(node).__name__}")

    # -- recursion helper: merge placeholder into real entry ----------------

    def _merge(self, old_id: int, new_id: int) -> None:
        """Redirect every occurrence of *old_id* to *new_id* in transitions."""
        self._transitions = [
            (
                new_id if s == old_id else s,
                l,
                new_id if t == old_id else t,
            )
            for s, l, t in self._transitions
        ]
        self._selection_transitions = {
            (
                new_id if s == old_id else s,
                l,
                new_id if t == old_id else t,
            )
            for s, l, t in self._selection_transitions
        }
        if old_id in self._states:
            del self._states[old_id]

    # -- parallel: product construction ------------------------------------

    def _build_parallel(
        self,
        branches: tuple[SessionType, ...],
        end_id: int,
    ) -> int:
        """Build each branch independently, compute the n-ary product, embed it."""
        from functools import reduce
        spaces = [build_statespace(b) for b in branches]
        prod = reduce(product_statespace, spaces)

        # Remap product state IDs into this builder's ID space.
        remap: dict[int, int] = {}
        for sid in prod.states:
            if sid == prod.bottom:
                remap[sid] = end_id
            else:
                remap[sid] = self._fresh(prod.labels.get(sid, "?"))

        for src, lbl, tgt in prod.transitions:
            remapped = (remap[src], lbl, remap[tgt])
            self._transitions.append(remapped)
            if prod.is_selection(src, lbl, tgt):
                self._selection_transitions.add(remapped)

        return remap[prod.top]

    # -- reachability -------------------------------------------------------

    def _reachable(self, start: int) -> set[int]:
        visited: set[int] = set()
        stack = [start]
        while stack:
            s = stack.pop()
            if s in visited:
                continue
            visited.add(s)
            for src, _, tgt in self._transitions:
                if src == s:
                    visited.add(s)
                    stack.append(tgt)
        return visited
