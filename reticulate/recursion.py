"""Recursive type analysis (Step 13).

Deeper analysis of how recursion interacts with lattice structure:

1. **Guardedness**: Every recursive variable occurs under a constructor
   (Branch, Select, Parallel). Unguarded recursion (e.g., rec X . X) is
   degenerate and produces trivial state spaces.

2. **Contractivity**: The recursive body makes observable progress before
   recursing — not just wrapping in more recursion.

3. **Unfolding**: One-step and multi-step unfolding of recursive types,
   checking that unfolding preserves state-space isomorphism.

4. **SCC analysis**: Characterize the strongly-connected components in
   recursive state spaces — each SCC corresponds to a recursive loop.

5. **Recursive depth**: Measure the nesting depth of recursion in a type.
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


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GuardednessResult:
    """Result of guardedness analysis.

    Attributes:
        is_guarded: True iff every recursive variable is under a constructor.
        unguarded_vars: Variables that appear unguarded.
    """
    is_guarded: bool
    unguarded_vars: tuple[str, ...]


@dataclass(frozen=True)
class ContractivityResult:
    """Result of contractivity analysis.

    Attributes:
        is_contractive: True iff every recursive path makes progress.
        non_contractive_vars: Variables whose bodies don't make progress.
    """
    is_contractive: bool
    non_contractive_vars: tuple[str, ...]


@dataclass(frozen=True)
class RecursionAnalysis:
    """Complete recursion analysis of a session type.

    Attributes:
        num_rec_binders: Number of rec binders in the type.
        max_nesting_depth: Maximum nesting depth of rec binders.
        is_guarded: True iff all recursive variables are guarded.
        is_contractive: True iff all recursive bodies are contractive.
        recursive_vars: Set of all bound recursive variables.
        scc_count: Number of SCCs in the state space (if built).
        scc_sizes: Sizes of non-trivial SCCs (size > 1).
        is_tail_recursive: True iff all recursive calls are in tail position.
    """
    num_rec_binders: int
    max_nesting_depth: int
    is_guarded: bool
    is_contractive: bool
    recursive_vars: frozenset[str]
    scc_count: int
    scc_sizes: tuple[int, ...]
    is_tail_recursive: bool


# ---------------------------------------------------------------------------
# Guardedness
# ---------------------------------------------------------------------------

def is_guarded(s: SessionType) -> bool:
    """Check whether all recursive variables are guarded.

    A variable X is guarded in rec X . S if every occurrence of X in S
    is under at least one constructor (Branch, Select, Parallel, Continuation).
    """
    return check_guardedness(s).is_guarded


def check_guardedness(s: SessionType) -> GuardednessResult:
    """Check guardedness with detailed results."""
    unguarded: list[str] = []
    _check_guarded(s, set(), unguarded)
    return GuardednessResult(
        is_guarded=len(unguarded) == 0,
        unguarded_vars=tuple(sorted(set(unguarded))),
    )


def _check_guarded(
    s: SessionType,
    unguarded_vars: set[str],
    result: list[str],
) -> None:
    """Recursively check guardedness.

    unguarded_vars tracks variables that are NOT yet under a constructor.
    """
    match s:
        case End() | Wait():
            pass
        case Var(name=name):
            if name in unguarded_vars:
                result.append(name)
        case Branch(choices=choices) | Select(choices=choices):
            for _, body in choices:
                # Inside a constructor — all vars become guarded
                _check_guarded(body, set(), result)
        case Parallel(branches=branches):
            for b in branches:
                _check_guarded(b, set(), result)
        case Continuation(left=left, right=right):
            _check_guarded(left, set(), result)
            _check_guarded(right, set(), result)
        case Rec(var=var, body=body):
            # The variable is unguarded until it passes through a constructor
            _check_guarded(body, unguarded_vars | {var}, result)


# ---------------------------------------------------------------------------
# Contractivity
# ---------------------------------------------------------------------------

def is_contractive(s: SessionType) -> bool:
    """Check whether the type is contractive.

    A recursive type rec X . S is contractive if S is not itself a Rec
    or a Var — i.e., the body must begin with an observable action
    (Branch, Select, Parallel, End, Wait).
    """
    return check_contractivity(s).is_contractive


def check_contractivity(s: SessionType) -> ContractivityResult:
    """Check contractivity with detailed results."""
    non_contractive: list[str] = []
    _check_contractive(s, non_contractive)
    return ContractivityResult(
        is_contractive=len(non_contractive) == 0,
        non_contractive_vars=tuple(sorted(set(non_contractive))),
    )


def _check_contractive(s: SessionType, result: list[str]) -> None:
    """Recursively check contractivity."""
    match s:
        case End() | Wait() | Var():
            pass
        case Branch(choices=choices) | Select(choices=choices):
            for _, body in choices:
                _check_contractive(body, result)
        case Parallel(branches=branches):
            for b in branches:
                _check_contractive(b, result)
        case Continuation(left=left, right=right):
            _check_contractive(left, result)
            _check_contractive(right, result)
        case Rec(var=var, body=body):
            # Non-contractive if body is Var or another Rec
            if isinstance(body, (Var, Rec)):
                result.append(var)
            _check_contractive(body, result)


# ---------------------------------------------------------------------------
# Unfolding
# ---------------------------------------------------------------------------

def unfold(s: SessionType) -> SessionType:
    """Unfold one level of recursion: rec X . S → S[X := rec X . S]."""
    if isinstance(s, Rec):
        return substitute(s.body, s.var, s)
    return s


def unfold_depth(s: SessionType, n: int) -> SessionType:
    """Unfold recursion n times.

    Each step replaces the outermost rec with its body, substituting
    the recursive variable. After n steps, the result may still contain
    Rec nodes (for nested recursion or if n < depth).
    """
    result = s
    for _ in range(n):
        if not isinstance(result, Rec):
            break
        result = unfold(result)
    return result


def substitute(body: SessionType, var: str, replacement: SessionType) -> SessionType:
    """Substitute var with replacement in body."""
    match body:
        case Var(name=name):
            return replacement if name == var else body
        case End() | Wait():
            return body
        case Branch(choices=choices):
            return Branch(tuple(
                (m, substitute(s, var, replacement)) for m, s in choices
            ))
        case Select(choices=choices):
            return Select(tuple(
                (l, substitute(s, var, replacement)) for l, s in choices
            ))
        case Parallel(branches=branches):
            return Parallel(tuple(
                substitute(b, var, replacement) for b in branches
            ))
        case Continuation(left=left, right=right):
            return Continuation(
                substitute(left, var, replacement),
                substitute(right, var, replacement),
            )
        case Rec(var=v, body=b):
            if v == var:
                return body  # shadowed
            return Rec(v, substitute(b, var, replacement))
        case _:
            return body


# ---------------------------------------------------------------------------
# Recursive depth
# ---------------------------------------------------------------------------

def rec_depth(s: SessionType) -> int:
    """Maximum nesting depth of rec binders."""
    match s:
        case End() | Wait() | Var():
            return 0
        case Branch(choices=choices) | Select(choices=choices):
            if not choices:
                return 0
            return max(rec_depth(body) for _, body in choices)
        case Parallel(branches=branches):
            return max((rec_depth(b) for b in branches), default=0)
        case Continuation(left=left, right=right):
            return max(rec_depth(left), rec_depth(right))
        case Rec(body=body):
            return 1 + rec_depth(body)
        case _:
            return 0


def count_rec_binders(s: SessionType) -> int:
    """Count the number of rec binders in a type."""
    match s:
        case End() | Wait() | Var():
            return 0
        case Branch(choices=choices) | Select(choices=choices):
            return sum(count_rec_binders(body) for _, body in choices)
        case Parallel(branches=branches):
            return sum(count_rec_binders(b) for b in branches)
        case Continuation(left=left, right=right):
            return count_rec_binders(left) + count_rec_binders(right)
        case Rec(body=body):
            return 1 + count_rec_binders(body)
        case _:
            return 0


def recursive_vars(s: SessionType) -> frozenset[str]:
    """Collect all variables bound by rec binders."""
    match s:
        case End() | Wait() | Var():
            return frozenset()
        case Branch(choices=choices) | Select(choices=choices):
            result: frozenset[str] = frozenset()
            for _, body in choices:
                result = result | recursive_vars(body)
            return result
        case Parallel(branches=branches):
            result_set: frozenset[str] = frozenset()
            for b in branches:
                result_set = result_set | recursive_vars(b)
            return result_set
        case Continuation(left=left, right=right):
            return recursive_vars(left) | recursive_vars(right)
        case Rec(var=var, body=body):
            return frozenset({var}) | recursive_vars(body)
        case _:
            return frozenset()


# ---------------------------------------------------------------------------
# Tail recursion
# ---------------------------------------------------------------------------

def is_tail_recursive(s: SessionType) -> bool:
    """Check whether all recursive calls are in tail position.

    A recursive call X is in tail position if it appears as a direct
    continuation of a Branch or Select choice — not nested inside
    further constructors.
    """
    match s:
        case End() | Wait() | Var():
            return True
        case Branch(choices=choices) | Select(choices=choices):
            return all(is_tail_recursive(body) for _, body in choices)
        case Parallel(branches=branches):
            return all(is_tail_recursive(b) for b in branches)
        case Continuation(left=left, right=right):
            return is_tail_recursive(left) and is_tail_recursive(right)
        case Rec(var=var, body=body):
            return _all_uses_tail(body, var)
        case _:
            return True


def _all_uses_tail(s: SessionType, var: str) -> bool:
    """Check that all uses of var in s are in tail position."""
    match s:
        case End() | Wait():
            return True
        case Var(name=name):
            return True  # A Var reference is fine — it IS tail position
        case Branch(choices=choices) | Select(choices=choices):
            return all(_all_uses_tail(body, var) for _, body in choices)
        case Parallel(branches=branches):
            # Var inside parallel is NOT tail recursive
            from reticulate.termination import _free_vars
            if any(var in _free_vars(b) for b in branches):
                return False
            return True
        case Continuation(left=left, right=right):
            # Var in left of continuation is NOT tail (has continuation)
            from reticulate.termination import _free_vars
            if var in _free_vars(left):
                return False
            return _all_uses_tail(right, var)
        case Rec(var=v, body=body):
            if v == var:
                return True  # shadowed
            return _all_uses_tail(body, var)
        case _:
            return True


# ---------------------------------------------------------------------------
# SCC analysis
# ---------------------------------------------------------------------------

def analyze_sccs(s: SessionType) -> tuple[int, tuple[int, ...]]:
    """Analyze SCC structure of a session type's state space.

    Returns:
        (scc_count, non_trivial_scc_sizes)

    Non-trivial SCCs (size > 1) correspond to recursive loops.
    """
    from reticulate.statespace import build_statespace
    from reticulate.lattice import check_lattice

    ss = build_statespace(s)
    lattice_result = check_lattice(ss)

    scc_map = lattice_result.scc_map
    if scc_map is None:
        return (len(ss.states), ())

    # Count SCCs and their sizes
    scc_members: dict[int, int] = {}
    for state, rep in scc_map.items():
        scc_members[rep] = scc_members.get(rep, 0) + 1

    scc_count = len(scc_members)
    non_trivial = tuple(
        sorted((size for size in scc_members.values() if size > 1), reverse=True)
    )
    return (scc_count, non_trivial)


# ---------------------------------------------------------------------------
# Complete analysis
# ---------------------------------------------------------------------------

def analyze_recursion(s: SessionType) -> RecursionAnalysis:
    """Perform complete recursion analysis on a session type."""
    guardedness = check_guardedness(s)
    contractivity = check_contractivity(s)
    scc_count, scc_sizes = analyze_sccs(s)

    return RecursionAnalysis(
        num_rec_binders=count_rec_binders(s),
        max_nesting_depth=rec_depth(s),
        is_guarded=guardedness.is_guarded,
        is_contractive=contractivity.is_contractive,
        recursive_vars=recursive_vars(s),
        scc_count=scc_count,
        scc_sizes=scc_sizes,
        is_tail_recursive=is_tail_recursive(s),
    )
