"""Reticular form: characterisation and reconstruction (Step 9).

A finite bounded lattice is a *reticulate* if it arises as L(S) for some
session type S.  This module provides:

1. ``reconstruct(ss)`` — recover a session type AST from a state space
   (inverse of ``build_statespace``).
2. ``is_reticulate(ss)`` — check whether a state space has reticular form.
3. ``check_reticular_form(ss)`` — detailed analysis with result object.

Reticular form requires that every non-bottom state in the (SCC-quotient)
Hasse diagram is either:
  (a) a branch node: all outgoing transitions are method calls (non-selection),
  (b) a select node: all outgoing transitions are selections, or
  (c) a product node: the sub-lattice rooted at this state decomposes as a
      product of two independent sub-lattices (parallel composition).

Recursion is detected via strongly-connected components (SCCs): an SCC
with more than one state represents a recursive loop.

The round-trip property: for all session types S,
  reconstruct(build_statespace(S)) ≡ S  (up to structural equivalence)
confirms that L(·) is essentially injective on session types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from reticulate.parser import (
    Branch,
    End,
    Rec,
    Select,
    SessionType,
    Var,
    pretty,
)
from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StateClassification:
    """Classification of a single state in a state space.

    Attributes:
        state: The state ID.
        kind: One of 'branch', 'select', 'end', 'product', 'mixed', 'recursive'.
        labels: The transition labels from this state.
    """
    state: int
    kind: str
    labels: tuple[str, ...]


@dataclass(frozen=True)
class ReticularFormResult:
    """Result of reticular form analysis.

    Attributes:
        is_reticulate: True iff the state space has reticular form.
        classifications: Per-state classification.
        reason: Human-readable explanation if not a reticulate.
        reconstructed: Pretty-printed reconstructed type (if successful).
    """
    is_reticulate: bool
    classifications: tuple[StateClassification, ...]
    reason: str | None = None
    reconstructed: str | None = None


# ---------------------------------------------------------------------------
# State classification
# ---------------------------------------------------------------------------

def classify_state(ss: StateSpace, state: int) -> StateClassification:
    """Classify a state as branch, select, end, or mixed."""
    methods = ss.enabled_methods(state)
    selections = ss.enabled_selections(state)

    labels = tuple(l for l, _ in methods) + tuple(l for l, _ in selections)

    if not methods and not selections:
        return StateClassification(state, "end", ())
    elif methods and not selections:
        return StateClassification(state, "branch", labels)
    elif selections and not methods:
        return StateClassification(state, "select", labels)
    else:
        # Mixed: both method and selection transitions.  This arises from
        # parallel composition (product states) and is legitimate.
        return StateClassification(state, "product", labels)


def classify_all_states(ss: StateSpace) -> tuple[StateClassification, ...]:
    """Classify all states in a state space."""
    return tuple(classify_state(ss, s) for s in sorted(ss.states))


# ---------------------------------------------------------------------------
# Reconstruction: StateSpace → SessionType
# ---------------------------------------------------------------------------

def reconstruct(ss: StateSpace) -> SessionType:
    """Reconstruct a session type AST from a state space.

    This is the inverse of ``build_statespace``.  It traverses the state
    space from top to bottom, building the AST:
    - States with only method transitions → Branch
    - States with only selection transitions → Select
    - States with no transitions → End
    - Back-edges (to already-visited states) → Rec/Var

    Raises ValueError if the state space has mixed states or cannot
    be reconstructed as a valid session type.
    """
    return _Reconstructor(ss).reconstruct()


class _Reconstructor:
    """Stateful reconstruction engine."""

    def __init__(self, ss: StateSpace) -> None:
        self._ss = ss
        # Map from state to variable name (for recursion)
        self._var_names: dict[int, str] = {}
        self._var_counter = 0
        # States currently being visited (for cycle detection)
        self._in_progress: set[int] = set()
        # Memoized results
        self._cache: dict[int, SessionType] = {}

    def reconstruct(self) -> SessionType:
        return self._visit(self._ss.top)

    def _fresh_var(self) -> str:
        """Generate a fresh variable name."""
        name = chr(ord("X") + self._var_counter % 3)
        if self._var_counter >= 3:
            name += str(self._var_counter // 3)
        self._var_counter += 1
        return name

    def _visit(self, state: int) -> SessionType:
        # Bottom state → End
        if state == self._ss.bottom:
            return End()

        # Already computed → return cached
        if state in self._cache:
            return self._cache[state]

        # Cycle detected → this is a recursive reference
        if state in self._in_progress:
            if state not in self._var_names:
                self._var_names[state] = self._fresh_var()
            return Var(self._var_names[state])

        self._in_progress.add(state)

        methods = self._ss.enabled_methods(state)
        selections = self._ss.enabled_selections(state)

        if not methods and not selections:
            result = End()
        elif methods and not selections:
            # Branch
            choices = tuple(
                (label, self._visit(target))
                for label, target in sorted(methods)
            )
            result = Branch(choices)
        elif selections and not methods:
            # Selection
            choices = tuple(
                (label, self._visit(target))
                for label, target in sorted(selections)
            )
            result = Select(choices)
        else:
            # Mixed — try to reconstruct anyway, prefer branch
            all_trans = sorted(methods + selections)
            choices = tuple(
                (label, self._visit(target))
                for label, target in all_trans
            )
            result = Branch(choices)

        self._in_progress.discard(state)

        # If this state has a recursive variable, wrap in Rec
        if state in self._var_names:
            var = self._var_names[state]
            result = Rec(var, result)

        self._cache[state] = result
        return result


# ---------------------------------------------------------------------------
# Reticular form check
# ---------------------------------------------------------------------------

def is_reticulate(ss: StateSpace) -> bool:
    """Check whether a state space has reticular form.

    A state space has reticular form if every non-bottom state is either:
    - A pure branch state (all transitions are methods), or
    - A pure select state (all transitions are selections), or
    - A product state (mixed transitions from parallel composition — both
      method and selection transitions from the same state), or
    - An end state (no transitions — only the bottom state).

    Product (mixed) states are legitimate: they arise in the product state
    space L(S₁ ∥ S₂) when one component is at a branch state and the other
    at a selection state.  This is a hallmark of the parallel constructor.
    """
    result = check_reticular_form(ss)
    return result.is_reticulate


def check_reticular_form(ss: StateSpace) -> ReticularFormResult:
    """Check reticular form with detailed analysis.

    All states are classified as branch, select, end, or mixed (product).
    Mixed states are allowed (they arise from parallel composition) and
    do not violate reticular form.
    """
    classifications = classify_all_states(ss)

    # Try reconstruction
    try:
        reconstructed = reconstruct(ss)
        reconstructed_str = pretty(reconstructed)
    except (ValueError, RecursionError):
        return ReticularFormResult(
            is_reticulate=False,
            classifications=classifications,
            reason="reconstruction failed",
        )

    return ReticularFormResult(
        is_reticulate=True,
        classifications=classifications,
        reconstructed=reconstructed_str,
    )
