"""Binary vs N-ary parallel composition analysis (Step 5c).

The session type grammar defines parallel composition as binary: ``S₁ || S₂``.
The AST, however, supports n-ary ``Parallel(branches: tuple[SessionType, ...])``.
The product construction in ``product.py`` builds the product of two state
spaces at a time, then flattens nested binary parallel into a single n-ary
product.  This module formalises the relationship between the two
representations and verifies empirically that they are semantically equivalent.

Key results:

1. **Flatten/nest are inverse** (up to AST equality):
   ``flatten(nest(S)) = S`` for any flat Parallel.
2. **All nestings are isomorphic**: every binary nesting of n branches
   produces an isomorphic state space.
3. **Algebraic invariants are identical** across nestings.
4. **Product coordinates coincide** after flattening.
5. The distinction is purely syntactic (AST/grammar level).

Public API:

- :func:`is_flat_parallel` — check if a Parallel has no nested Parallel branches
- :func:`flatten_parallel` — recursively flatten nested binary Parallel to n-ary
- :func:`nest_parallel` — convert n-ary to right-nested binary
- :func:`nest_parallel_left` — convert n-ary to left-nested binary
- :func:`all_nestings` — enumerate every binary nesting (Catalan number)
- :func:`check_nesting_invariance` — verify all nestings produce isomorphic state spaces
- :class:`NaryParallelResult` — summary dataclass
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NaryParallelResult:
    """Summary of n-ary parallel analysis.

    Attributes:
        is_flat: True iff the root Parallel has no nested Parallel branches.
        num_branches: Number of branches after flattening.
        all_nestings_isomorphic: True iff every binary nesting produces an
            isomorphic state space.
        catalan_count: Catalan number C(n-1) — the number of distinct binary
            nestings for *num_branches* branches.
        nestings_checked: How many nestings were actually checked.
    """

    is_flat: bool
    num_branches: int
    all_nestings_isomorphic: bool
    catalan_count: int
    nestings_checked: int


# ---------------------------------------------------------------------------
# Catalan number
# ---------------------------------------------------------------------------

def catalan(n: int) -> int:
    """Return the n-th Catalan number C(n) = (2n)! / ((n+1)! * n!).

    Used to count the number of distinct full binary trees with n+1 leaves,
    which equals the number of ways to parenthesise n+1 factors.
    """
    if n < 0:
        return 0
    return math.comb(2 * n, n) // (n + 1)


# ---------------------------------------------------------------------------
# Flat check
# ---------------------------------------------------------------------------

def is_flat_parallel(ast: SessionType) -> bool:
    """Check whether a Parallel node is already flat (no nested Parallel in branches).

    Returns True for non-Parallel nodes (vacuously flat).
    Returns True for a Parallel whose branches contain no direct Parallel children.
    """
    if not isinstance(ast, Parallel):
        return True
    return all(not isinstance(b, Parallel) for b in ast.branches)


# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------

def _collect_parallel_branches(ast: SessionType) -> list[SessionType]:
    """Recursively collect all non-Parallel leaves from nested Parallel nodes."""
    if isinstance(ast, Parallel):
        result: list[SessionType] = []
        for branch in ast.branches:
            result.extend(_collect_parallel_branches(branch))
        return result
    return [ast]


def flatten_parallel(ast: SessionType) -> SessionType:
    """Recursively flatten nested binary Parallel to n-ary.

    ``((S1 || S2) || S3)``  becomes  ``(S1 || S2 || S3)``.

    Descends into all AST nodes (Branch, Select, Rec, Continuation) so that
    deeply nested Parallel nodes are also flattened.

    Non-Parallel nodes are returned unchanged (but their children are
    recursively flattened).
    """
    if isinstance(ast, End) or isinstance(ast, Wait) or isinstance(ast, Var):
        return ast
    if isinstance(ast, Branch):
        return Branch(
            tuple((label, flatten_parallel(body)) for label, body in ast.choices)
        )
    if isinstance(ast, Select):
        return Select(
            tuple((label, flatten_parallel(body)) for label, body in ast.choices)
        )
    if isinstance(ast, Rec):
        return Rec(ast.var, flatten_parallel(ast.body))
    if isinstance(ast, Continuation):
        return Continuation(flatten_parallel(ast.left), flatten_parallel(ast.right))
    if isinstance(ast, Parallel):
        branches = _collect_parallel_branches(ast)
        flattened = tuple(flatten_parallel(b) for b in branches)
        if len(flattened) == 1:
            return flattened[0]
        return Parallel(flattened)
    return ast  # pragma: no cover


# ---------------------------------------------------------------------------
# Nest (right-associative)
# ---------------------------------------------------------------------------

def nest_parallel(ast: SessionType) -> SessionType:
    """Convert n-ary Parallel to right-nested binary.

    ``(S1 || S2 || S3)``  becomes  ``(S1 || (S2 || S3))``.

    Descends into all AST nodes.
    """
    if isinstance(ast, End) or isinstance(ast, Wait) or isinstance(ast, Var):
        return ast
    if isinstance(ast, Branch):
        return Branch(
            tuple((label, nest_parallel(body)) for label, body in ast.choices)
        )
    if isinstance(ast, Select):
        return Select(
            tuple((label, nest_parallel(body)) for label, body in ast.choices)
        )
    if isinstance(ast, Rec):
        return Rec(ast.var, nest_parallel(ast.body))
    if isinstance(ast, Continuation):
        return Continuation(nest_parallel(ast.left), nest_parallel(ast.right))
    if isinstance(ast, Parallel):
        branches = [nest_parallel(b) for b in ast.branches]
        if len(branches) <= 2:
            return Parallel(tuple(branches))
        # Right-nest: S1 || (S2 || (... || Sn))
        result: SessionType = branches[-1]
        for b in reversed(branches[:-1]):
            result = Parallel((b, result))
        return result
    return ast  # pragma: no cover


def nest_parallel_left(ast: SessionType) -> SessionType:
    """Convert n-ary Parallel to left-nested binary.

    ``(S1 || S2 || S3)``  becomes  ``((S1 || S2) || S3)``.

    Descends into all AST nodes.
    """
    if isinstance(ast, End) or isinstance(ast, Wait) or isinstance(ast, Var):
        return ast
    if isinstance(ast, Branch):
        return Branch(
            tuple((label, nest_parallel_left(body)) for label, body in ast.choices)
        )
    if isinstance(ast, Select):
        return Select(
            tuple((label, nest_parallel_left(body)) for label, body in ast.choices)
        )
    if isinstance(ast, Rec):
        return Rec(ast.var, nest_parallel_left(ast.body))
    if isinstance(ast, Continuation):
        return Continuation(nest_parallel_left(ast.left), nest_parallel_left(ast.right))
    if isinstance(ast, Parallel):
        branches = [nest_parallel_left(b) for b in ast.branches]
        if len(branches) <= 2:
            return Parallel(tuple(branches))
        # Left-nest: ((S1 || S2) || ...) || Sn
        result: SessionType = branches[0]
        for b in branches[1:]:
            result = Parallel((result, b))
        return result
    return ast  # pragma: no cover


# ---------------------------------------------------------------------------
# All nestings (enumerate all binary trees)
# ---------------------------------------------------------------------------

def _all_binary_trees(leaves: tuple[SessionType, ...]) -> list[SessionType]:
    """Enumerate all full binary trees with the given leaves (in order).

    Returns a list of Parallel ASTs, one per distinct parenthesisation.
    The number of results is the Catalan number C(n-1) where n = len(leaves).
    """
    n = len(leaves)
    if n == 1:
        return [leaves[0]]
    if n == 2:
        return [Parallel((leaves[0], leaves[1]))]
    results: list[SessionType] = []
    # Split leaves[0..k] | leaves[k..n] for every split point
    for k in range(1, n):
        left_trees = _all_binary_trees(leaves[:k])
        right_trees = _all_binary_trees(leaves[k:])
        for lt in left_trees:
            for rt in right_trees:
                results.append(Parallel((lt, rt)))
    return results


def all_nestings(branches: tuple[SessionType, ...]) -> list[SessionType]:
    """Enumerate all binary nestings of the given branches.

    The order of branches is preserved; only the parenthesisation varies.
    Returns C(n-1) distinct ASTs where n = len(branches) and C is the
    Catalan number.

    Raises ValueError if fewer than 2 branches are given.
    """
    if len(branches) < 2:
        raise ValueError("need at least 2 branches for parallel composition")
    return _all_binary_trees(branches)


# ---------------------------------------------------------------------------
# Nesting invariance check
# ---------------------------------------------------------------------------

def check_nesting_invariance(ast: SessionType) -> NaryParallelResult:
    """Verify that all binary nestings of a Parallel produce isomorphic state spaces.

    For a Parallel with n branches, this builds C(n-1) state spaces (one per
    nesting) and checks pairwise isomorphism.  For large n this is expensive;
    the function caps at n <= 6 (C(5) = 42 nestings) and raises ValueError
    for larger inputs.

    Non-Parallel nodes are reported as trivially flat with 0 or 1 branches.
    """
    from reticulate.morphism import find_isomorphism
    from reticulate.statespace import build_statespace

    if not isinstance(ast, Parallel):
        return NaryParallelResult(
            is_flat=True,
            num_branches=0,
            all_nestings_isomorphic=True,
            catalan_count=1,
            nestings_checked=0,
        )

    flat = flatten_parallel(ast)
    if not isinstance(flat, Parallel):
        # Degenerate: single branch
        return NaryParallelResult(
            is_flat=True,
            num_branches=1,
            all_nestings_isomorphic=True,
            catalan_count=1,
            nestings_checked=0,
        )

    branches = flat.branches
    n = len(branches)
    is_flat = is_flat_parallel(ast)
    cat = catalan(n - 1)

    if n > 6:
        raise ValueError(
            f"too many branches ({n}); nesting enumeration capped at 6 "
            f"(C(5)=42 nestings, you would need C({n - 1})={cat})"
        )

    nestings = all_nestings(branches)
    assert len(nestings) == cat

    # Build state space for each nesting
    spaces: list[StateSpace] = []
    for nesting_ast in nestings:
        ss = build_statespace(nesting_ast)
        spaces.append(ss)

    # Check all pairs against the first
    all_iso = True
    ref = spaces[0]
    for ss in spaces[1:]:
        iso = find_isomorphism(ref, ss)
        if iso is None:
            all_iso = False
            break

    return NaryParallelResult(
        is_flat=is_flat,
        num_branches=n,
        all_nestings_isomorphic=all_iso,
        catalan_count=cat,
        nestings_checked=len(nestings),
    )
