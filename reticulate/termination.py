"""Termination checking and WF-Par well-formedness for session types.

Operates on the AST level (no state-space construction needed).

- **Termination** (spec §2.3.3): Every recursive type ``μX. S`` must have at
  least one syntactic path from the root of S to a leaf that is NOT ``Var(X)``.
  This rules out divergent definitions like ``μX. &{loop: X}`` while permitting
  ``μX. &{read: X, done: end}``.

- **WF-Par** (spec §5.1): For each ``S₁ ∥ S₂ ∥ … ∥ Sₙ`` in the AST:
  1. All branches are terminating.
  2. No cross-branch recursion variables (free vars of one branch don't clash
     with bound vars of any other).
  3. No nested ``∥`` inside a ``∥`` branch.
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
)


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TerminationResult:
    """Result of a termination check on a session type AST.

    Attributes:
        is_terminating: True iff every ``Rec`` node has an exit path.
        non_terminating_vars: Names of recursion variables whose ``Rec``
            body has no exit path (i.e., all syntactic paths lead back
            to the recursion variable).
    """

    is_terminating: bool
    non_terminating_vars: tuple[str, ...]


@dataclass(frozen=True)
class WFParallelResult:
    """Result of a WF-Par well-formedness check.

    Attributes:
        is_well_formed: True iff every ``Parallel`` node satisfies WF-Par.
        errors: Human-readable descriptions of each violation found.
    """

    is_well_formed: bool
    errors: tuple[str, ...]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_terminating(session_type: SessionType) -> bool:
    """Return True iff every ``Rec`` in *session_type* has an exit path."""
    return len(_collect_non_terminating(session_type)) == 0


def check_termination(session_type: SessionType) -> TerminationResult:
    """Full termination analysis of *session_type*."""
    bad = _collect_non_terminating(session_type)
    return TerminationResult(
        is_terminating=len(bad) == 0,
        non_terminating_vars=tuple(bad),
    )


def check_wf_parallel(session_type: SessionType) -> WFParallelResult:
    """Check WF-Par well-formedness for every ``Parallel`` in *session_type*."""
    errors = _collect_wf_par_errors(session_type)
    return WFParallelResult(
        is_well_formed=len(errors) == 0,
        errors=tuple(errors),
    )


def tau_complete(session_type: SessionType, tau_label: str = "tau") -> SessionType:
    """Return *session_type* with a ``tau : end`` exit added to every
    non-terminating recursive body.

    The tau-exit is added as a branch arm (external choice, client-
    controlled).  Recursions that already terminate are left unchanged.
    The result is always terminating (and hence, by the Reticulate
    Theorem, its state space is a bounded lattice).

    Parameters:
        session_type: The AST to tau-complete.
        tau_label: The label for the exit arm (default ``"tau"``).

    Returns:
        A new AST where every non-terminating ``Rec`` body has been
        augmented with ``tau_label : End``.
    """
    return _tau_complete(session_type, tau_label)


# ---------------------------------------------------------------------------
# Internal: tau-completion
# ---------------------------------------------------------------------------

def _tau_complete(node: SessionType, tau_label: str) -> SessionType:
    """Recursively tau-complete an AST node."""
    match node:
        case End() | Wait() | Var():
            return node
        case Branch(choices=choices):
            new_choices = tuple(
                (label, _tau_complete(body, tau_label))
                for label, body in choices
            )
            return Branch(new_choices)
        case Select(choices=choices):
            new_choices = tuple(
                (label, _tau_complete(body, tau_label))
                for label, body in choices
            )
            return Select(new_choices)
        case Continuation(left=left, right=right):
            return Continuation(
                _tau_complete(left, tau_label),
                _tau_complete(right, tau_label),
            )
        case Parallel(branches=branches):
            return Parallel(tuple(
                _tau_complete(b, tau_label) for b in branches
            ))
        case Rec(var=var, body=body):
            new_body = _tau_complete(body, tau_label)
            if _has_exit_path(new_body, var):
                # Already terminating after completing children.
                return Rec(var, new_body)
            # Body is non-terminating: add tau exit.
            new_body = _add_tau_arm(new_body, tau_label)
            return Rec(var, new_body)
        case _:
            raise TypeError(f"unknown AST node: {type(node).__name__}")


def _add_tau_arm(node: SessionType, tau_label: str) -> SessionType:
    """Add a ``tau_label : End`` arm to the outermost choice in *node*.

    If *node* is a Branch or Select, append the arm.
    Otherwise, wrap *node* in a Branch with the original as one arm
    and ``tau_label : End`` as the other.
    """
    match node:
        case Branch(choices=choices):
            # Check label collision.
            existing = {label for label, _ in choices}
            if tau_label in existing:
                return node  # Already has a tau arm.
            return Branch(choices + ((tau_label, End()),))
        case Select(choices=choices):
            existing = {label for label, _ in choices}
            if tau_label in existing:
                return node
            return Select(choices + ((tau_label, End()),))
        case _:
            # Wrap in a branch: the original continues, tau exits.
            return Branch((("_continue", node), (tau_label, End())))


# ---------------------------------------------------------------------------
# Internal: termination checking
# ---------------------------------------------------------------------------

def _has_exit_path(node: SessionType, forbidden: str) -> bool:
    """Check whether *node* has at least one syntactic path to a leaf
    that is NOT ``Var(forbidden)``.

    This is the decidable termination check from spec §2.3.3.
    """
    match node:
        case End() | Wait():
            return True
        case Var(name=name):
            # An occurrence of the forbidden var is NOT an exit;
            # an occurrence of any OTHER var IS an exit (it refers
            # to an enclosing recursion, so we're leaving this rec).
            return name != forbidden
        case Branch(choices=choices):
            return any(_has_exit_path(body, forbidden) for _, body in choices)
        case Select(choices=choices):
            return any(_has_exit_path(body, forbidden) for _, body in choices)
        case Continuation(left=left, right=right):
            return _has_exit_path(left, forbidden) and _has_exit_path(right, forbidden)
        case Parallel(branches=branches):
            return all(_has_exit_path(b, forbidden) for b in branches)
        case Rec(var=var, body=body):
            # Inner recursion — check through it. The inner var is a
            # different binding; we still track the outer forbidden var.
            return _has_exit_path(body, forbidden)
        case _:
            raise TypeError(f"unknown AST node: {type(node).__name__}")


def _collect_non_terminating(node: SessionType) -> list[str]:
    """Walk the entire AST and return var names for ``Rec`` nodes
    where ``_has_exit_path`` fails."""
    result: list[str] = []
    _walk_for_termination(node, result)
    return result


def _walk_for_termination(node: SessionType, acc: list[str]) -> None:
    """Recursive AST walker that populates *acc* with non-terminating var names."""
    match node:
        case End() | Wait() | Var():
            pass
        case Branch(choices=choices):
            for _, body in choices:
                _walk_for_termination(body, acc)
        case Select(choices=choices):
            for _, body in choices:
                _walk_for_termination(body, acc)
        case Continuation(left=left, right=right):
            _walk_for_termination(left, acc)
            _walk_for_termination(right, acc)
        case Parallel(branches=branches):
            for b in branches:
                _walk_for_termination(b, acc)
        case Rec(var=var, body=body):
            if not _has_exit_path(body, var):
                acc.append(var)
            # Also check inside the body for nested Rec nodes.
            _walk_for_termination(body, acc)
        case _:
            raise TypeError(f"unknown AST node: {type(node).__name__}")


# ---------------------------------------------------------------------------
# Internal: WF-Par checking
# ---------------------------------------------------------------------------

def _collect_wf_par_errors(node: SessionType) -> list[str]:
    """Walk the AST and collect WF-Par violations for every ``Parallel`` node."""
    errors: list[str] = []
    _walk_for_wf_par(node, errors)
    return errors


def _walk_for_wf_par(node: SessionType, errors: list[str]) -> None:
    """Recursive walker that checks WF-Par at each ``Parallel`` node."""
    match node:
        case End() | Wait() | Var():
            pass
        case Branch(choices=choices):
            for _, body in choices:
                _walk_for_wf_par(body, errors)
        case Select(choices=choices):
            for _, body in choices:
                _walk_for_wf_par(body, errors)
        case Continuation(left=left, right=right):
            _walk_for_wf_par(left, errors)
            _walk_for_wf_par(right, errors)
        case Parallel(branches=branches):
            # Check this Parallel node
            _check_wf_par_node(branches, errors)
            # Also recurse into branches (though WF-Par.3 forbids nested ∥,
            # we still want to report errors inside nested ones).
            for b in branches:
                _walk_for_wf_par(b, errors)
        case Rec(var=var, body=body):
            _walk_for_wf_par(body, errors)
        case _:
            raise TypeError(f"unknown AST node: {type(node).__name__}")


def _check_wf_par_node(
    branches: tuple[SessionType, ...],
    errors: list[str],
) -> None:
    """Check the three WF-Par conditions for a single ``Parallel(branches)``."""
    n = len(branches)

    # For binary parallel, use "left"/"right" labels for backwards compatibility.
    # For n-ary (n>2), use "branch 0", "branch 1", etc.
    def _label(i: int) -> str:
        if n == 2:
            return "left" if i == 0 else "right"
        return f"branch {i}"

    # 1. Termination: all branches must be terminating
    for i, branch in enumerate(branches):
        bad = _collect_non_terminating(branch)
        if bad:
            errors.append(
                f"{_label(i)} branch of ∥ is non-terminating "
                f"(non-terminating vars: {', '.join(bad)})"
            )

    # 2. No cross-branch variables — check all pairs (i, j) with i < j
    free_sets = [_free_vars(b) for b in branches]
    bound_sets = [_bound_vars(b) for b in branches]
    for i in range(n):
        for j in range(i + 1, n):
            cross_ij = free_sets[i] & bound_sets[j]
            if cross_ij:
                errors.append(
                    f"cross-branch variable(s) {', '.join(sorted(cross_ij))}: "
                    f"free in {_label(i)}, bound in {_label(j)}"
                )
            cross_ji = free_sets[j] & bound_sets[i]
            if cross_ji:
                errors.append(
                    f"cross-branch variable(s) {', '.join(sorted(cross_ji))}: "
                    f"free in {_label(j)}, bound in {_label(i)}"
                )

    # 3. No nested parallel
    for i, branch in enumerate(branches):
        if _contains_parallel(branch):
            errors.append(f"{_label(i)} branch of ∥ contains nested ∥")


# ---------------------------------------------------------------------------
# Internal: helper functions
# ---------------------------------------------------------------------------

def _free_vars(node: SessionType) -> set[str]:
    """Compute the set of free type variables in *node*."""
    match node:
        case End() | Wait():
            return set()
        case Var(name=name):
            return {name}
        case Branch(choices=choices):
            result: set[str] = set()
            for _, body in choices:
                result |= _free_vars(body)
            return result
        case Select(choices=choices):
            result = set()
            for _, body in choices:
                result |= _free_vars(body)
            return result
        case Continuation(left=left, right=right):
            return _free_vars(left) | _free_vars(right)
        case Parallel(branches=branches):
            result = set()
            for b in branches:
                result |= _free_vars(b)
            return result
        case Rec(var=var, body=body):
            return _free_vars(body) - {var}
        case _:
            raise TypeError(f"unknown AST node: {type(node).__name__}")


def _bound_vars(node: SessionType) -> set[str]:
    """Compute the set of bound type variables in *node*
    (variables that appear as the binding variable of some ``Rec``)."""
    match node:
        case End() | Wait() | Var():
            return set()
        case Branch(choices=choices):
            result: set[str] = set()
            for _, body in choices:
                result |= _bound_vars(body)
            return result
        case Select(choices=choices):
            result = set()
            for _, body in choices:
                result |= _bound_vars(body)
            return result
        case Continuation(left=left, right=right):
            return _bound_vars(left) | _bound_vars(right)
        case Parallel(branches=branches):
            result = set()
            for b in branches:
                result |= _bound_vars(b)
            return result
        case Rec(var=var, body=body):
            return {var} | _bound_vars(body)
        case _:
            raise TypeError(f"unknown AST node: {type(node).__name__}")


def _contains_parallel(node: SessionType) -> bool:
    """Return True iff *node* contains a ``Parallel`` node anywhere."""
    match node:
        case End() | Wait() | Var():
            return False
        case Branch(choices=choices):
            return any(_contains_parallel(body) for _, body in choices)
        case Select(choices=choices):
            return any(_contains_parallel(body) for _, body in choices)
        case Continuation(left=left, right=right):
            return _contains_parallel(left) or _contains_parallel(right)
        case Parallel():
            return True
        case Rec(var=var, body=body):
            return _contains_parallel(body)
        case _:
            raise TypeError(f"unknown AST node: {type(node).__name__}")
