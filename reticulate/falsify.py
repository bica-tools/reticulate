"""Counterexample finder for session type conjectures.

Given a predicate over (AST, StateSpace, LatticeResult) triples, enumerate
session types up to configurable bounds and search for a counterexample that
falsifies the predicate.

Includes built-in predicates for common lattice-theoretic conjectures:
distributivity, lattice universality, duality preservation, and more.

Usage::

    >>> from reticulate.falsify import falsify, is_distributive
    >>> result = falsify(is_distributive, depth=2, labels=('a', 'b', 'c'))
    >>> result.falsified
    True
    >>> result.counterexample  # a session type string
    '&{a: &{a: end}, b: &{b: end}, c: &{c: end}}'
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Optional

from reticulate.enumerate_types import EnumerationConfig, enumerate_session_types
from reticulate.lattice import LatticeResult, check_lattice, compute_meet, compute_join
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
from reticulate.statespace import StateSpace, build_statespace
from reticulate.termination import is_terminating


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FalsifyResult:
    """Result of a counterexample search.

    Attributes:
        property_name: Name of the property being tested.
        counterexample: Pretty-printed session type that falsifies, or None.
        counterexample_ast: The AST node that falsifies, or None.
        checked: How many types were checked before stopping.
        falsified: True iff a counterexample was found.
        details: Human-readable explanation of the result.
    """

    property_name: str
    counterexample: Optional[str]
    counterexample_ast: Optional[object]
    checked: int
    falsified: bool
    details: str


# ---------------------------------------------------------------------------
# Type alias for predicates
# ---------------------------------------------------------------------------

Predicate = Callable[[SessionType, StateSpace, LatticeResult], bool]


# ---------------------------------------------------------------------------
# Main falsification engine
# ---------------------------------------------------------------------------

def falsify(
    predicate: Predicate,
    *,
    property_name: str = "custom",
    depth: int = 3,
    width: int = 3,
    labels: tuple[str, ...] = ("a", "b", "c"),
    require_terminating: bool = True,
    max_checks: int = 10000,
    include_parallel: bool = True,
    include_recursion: bool = True,
    include_selection: bool = True,
) -> FalsifyResult:
    """Search for a counterexample to *predicate* among enumerated session types.

    The predicate is called as ``predicate(ast, statespace, lattice_result)``
    and should return ``True`` if the property holds for that type.  If it
    returns ``False``, the type is a counterexample.

    Args:
        predicate: Function (ast, ss, lr) -> bool.  True means property holds.
        property_name: Human-readable name for the property.
        depth: Maximum AST depth for type enumeration.
        width: Maximum branch/select width.
        labels: Method/label names to use in enumeration.
        require_terminating: If True, skip non-terminating types.
        max_checks: Stop after this many types (0 = unlimited).
        include_parallel: Include parallel types in enumeration.
        include_recursion: Include recursive types in enumeration.
        include_selection: Include selection types in enumeration.

    Returns:
        A FalsifyResult.  If falsified is True, counterexample and
        counterexample_ast are populated with the first failing type.
    """
    config = EnumerationConfig(
        max_depth=depth,
        labels=labels,
        max_branch_width=width,
        include_parallel=include_parallel,
        include_recursion=include_recursion,
        include_selection=include_selection,
        require_terminating=require_terminating,
    )

    checked = 0

    for ast in enumerate_session_types(config):
        if require_terminating and not is_terminating(ast):
            continue

        if 0 < max_checks <= checked:
            break

        try:
            ss = build_statespace(ast)
        except (ValueError, RecursionError, KeyError):
            continue

        try:
            lr = check_lattice(ss)
        except (KeyError, ValueError):
            continue

        checked += 1

        try:
            holds = predicate(ast, ss, lr)
        except Exception:
            # Predicate crashed — treat as falsified
            type_str = pretty(ast)
            return FalsifyResult(
                property_name=property_name,
                counterexample=type_str,
                counterexample_ast=ast,
                checked=checked,
                falsified=True,
                details=(
                    f"Predicate '{property_name}' raised an exception on "
                    f"type: {type_str}"
                ),
            )

        if not holds:
            type_str = pretty(ast)
            return FalsifyResult(
                property_name=property_name,
                counterexample=type_str,
                counterexample_ast=ast,
                checked=checked,
                falsified=True,
                details=(
                    f"Counterexample found for '{property_name}' after "
                    f"checking {checked} types: {type_str}"
                ),
            )

    return FalsifyResult(
        property_name=property_name,
        counterexample=None,
        counterexample_ast=None,
        checked=checked,
        falsified=False,
        details=(
            f"Property '{property_name}' holds for all {checked} "
            f"checked types (depth={depth}, width={width}, "
            f"labels={labels})"
        ),
    )


# ---------------------------------------------------------------------------
# Built-in predicates
# ---------------------------------------------------------------------------

def is_always_lattice(ast: SessionType, ss: StateSpace, lr: LatticeResult) -> bool:
    """Check that the state space forms a lattice.

    This is the universality conjecture predicate: L(S) is always a lattice
    for terminating session types.
    """
    return lr.is_lattice


def is_distributive(ast: SessionType, ss: StateSpace, lr: LatticeResult) -> bool:
    """Check distributivity: a meet (b join c) = (a meet b) join (a meet c).

    Tests the law for ALL triples of states in the state space.
    Requires the state space to be a lattice first.
    """
    if not lr.is_lattice:
        return True  # vacuously true if not a lattice

    states = sorted(ss.states)
    if len(states) <= 2:
        return True  # trivial lattices are distributive

    for a, b, c in combinations(states, 3):
        # Check a ^ (b v c) = (a ^ b) v (a ^ c)
        j_bc = compute_join(ss, b, c)
        if j_bc is None:
            continue
        lhs = compute_meet(ss, a, j_bc)

        m_ab = compute_meet(ss, a, b)
        m_ac = compute_meet(ss, a, c)
        if m_ab is None or m_ac is None:
            continue
        rhs = compute_join(ss, m_ab, m_ac)

        if lhs != rhs:
            return False

    return True


def dual_preserves_lattice(
    ast: SessionType, ss: StateSpace, lr: LatticeResult,
) -> bool:
    """Check that dual(S) also produces a lattice.

    If L(S) is a lattice, then L(dual(S)) should also be a lattice,
    since duality preserves the state-space structure.
    """
    if not lr.is_lattice:
        return True  # vacuously true

    from reticulate.duality import dual

    d = dual(ast)
    try:
        ss_d = build_statespace(d)
        lr_d = check_lattice(ss_d)
    except (ValueError, RecursionError, KeyError):
        return False

    return lr_d.is_lattice


def binary_implies_distributive(
    ast: SessionType, ss: StateSpace, lr: LatticeResult,
) -> bool:
    """Check: if all branch/select have at most 2 options, then distributive.

    The conjecture is that binary-only types always produce distributive
    lattices (no M3 diamond possible with only 2-way branching).
    """
    if not _all_binary(ast):
        return True  # not a binary-only type, predicate is vacuously true

    if not lr.is_lattice:
        return True  # not a lattice, skip

    return is_distributive(ast, ss, lr)


def _all_binary(ast: SessionType) -> bool:
    """Check that all Branch/Select constructors have at most 2 choices."""
    if isinstance(ast, (End, Wait, Var)):
        return True
    elif isinstance(ast, (Branch, Select)):
        if len(ast.choices) > 2:
            return False
        return all(_all_binary(body) for _, body in ast.choices)
    elif isinstance(ast, Parallel):
        return all(_all_binary(b) for b in ast.branches)
    elif isinstance(ast, Rec):
        return _all_binary(ast.body)
    elif isinstance(ast, Continuation):
        return _all_binary(ast.left) and _all_binary(ast.right)
    return True


# ---------------------------------------------------------------------------
# Helpers for structural filtering
# ---------------------------------------------------------------------------

def _contains_parallel(ast: SessionType) -> bool:
    """Check whether the AST contains any Parallel node."""
    if isinstance(ast, Parallel):
        return True
    elif isinstance(ast, (End, Wait, Var)):
        return False
    elif isinstance(ast, (Branch, Select)):
        return any(_contains_parallel(body) for _, body in ast.choices)
    elif isinstance(ast, Rec):
        return _contains_parallel(ast.body)
    elif isinstance(ast, Continuation):
        return _contains_parallel(ast.left) or _contains_parallel(ast.right)
    return False


def _has_polarity_nesting(ast: SessionType) -> bool:
    """Check if a Branch contains a Select or vice versa (polarity nesting)."""
    return _check_nesting(ast, None)


def _check_nesting(ast: SessionType, outer_polarity: str | None) -> bool:
    """Recursive check for polarity nesting.

    outer_polarity is 'branch', 'select', or None.
    Returns True if nesting is detected.
    """
    if isinstance(ast, Branch):
        if outer_polarity == "select":
            return True
        return any(
            _check_nesting(body, "branch") for _, body in ast.choices
        )
    elif isinstance(ast, Select):
        if outer_polarity == "branch":
            return True
        return any(
            _check_nesting(body, "select") for _, body in ast.choices
        )
    elif isinstance(ast, (End, Wait, Var)):
        return False
    elif isinstance(ast, Parallel):
        return any(_check_nesting(b, outer_polarity) for b in ast.branches)
    elif isinstance(ast, Rec):
        return _check_nesting(ast.body, outer_polarity)
    elif isinstance(ast, Continuation):
        return (
            _check_nesting(ast.left, outer_polarity)
            or _check_nesting(ast.right, outer_polarity)
        )
    return False


# ---------------------------------------------------------------------------
# Convenience: distributivity conjecture falsifier
# ---------------------------------------------------------------------------

def falsify_distributivity_conjecture(
    *,
    depth: int = 2,
    labels: tuple[str, ...] = ("a", "b", "c"),
    width: int = 3,
) -> FalsifyResult:
    """Test the conjecture: no parallel and no polarity nesting implies distributive.

    This specifically targets the M3 counterexample: a 3-way branch where each
    arm is itself a single-method branch, e.g.
    ``&{a: &{a: end}, b: &{b: end}, c: &{c: end}}``.

    The predicate checks: if a type has no parallel and no polarity nesting
    (Branch inside Select or vice versa), then it must be distributive.
    This conjecture is FALSE -- the M3 diamond arises from purely
    branch-only types with 3+ children whose sub-branches have disjoint labels.
    """

    def no_par_no_nesting_implies_distributive(
        ast: SessionType, ss: StateSpace, lr: LatticeResult,
    ) -> bool:
        if _contains_parallel(ast):
            return True  # has parallel, skip
        if _has_polarity_nesting(ast):
            return True  # has nesting, skip
        if not lr.is_lattice:
            return True  # not a lattice, skip
        return is_distributive(ast, ss, lr)

    return falsify(
        no_par_no_nesting_implies_distributive,
        property_name="no_parallel_no_nesting_implies_distributive",
        depth=depth,
        width=width,
        labels=labels,
        require_terminating=True,
        include_parallel=False,
        include_recursion=False,
        include_selection=False,
    )
