"""L₃ — Parallel Composition and Product Lattices.

This module analyses session types at language level L₃: types that use
parallel composition (``S₁ ∥ S₂``) but *no* recursion, continuation, or
nested parallel.  At this level, the product lattice theorem guarantees:

    L(S₁ ∥ S₂) ≅ L(S₁) × L(S₂)

Key contributions:

- **Classification**: Identify whether an L₃ state space is a chain, tree,
  product-of-chains, product-of-trees, or general product.
- **Factor decomposition**: Recover the factors of a product state space
  (inverse of the product construction).
- **Factor analysis**: Size, height, width, distributivity per factor.
- **Size prediction**: |L₁ × L₂| = |L₁| · |L₂|.
- **Distributivity preservation**: Product of distributives is distributive.
- **Enumeration**: Enumerate all L₃ types up to given depth/width.

Requires: :mod:`reticulate.parser`, :mod:`reticulate.statespace`,
:mod:`reticulate.lattice`, :mod:`reticulate.product`,
:mod:`reticulate.extensions.language_hierarchy`.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product as cartesian
from typing import Iterator

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
    parse,
    pretty,
)
from reticulate.statespace import StateSpace, build_statespace
from reticulate.lattice import (
    LatticeResult,
    check_distributive,
    check_lattice,
)
from reticulate.extensions.language_hierarchy import (
    LanguageLevel,
    classify_level,
)


# ---------------------------------------------------------------------------
# AST predicates
# ---------------------------------------------------------------------------

def _contains_rec(s: SessionType) -> bool:
    """Return True if *s* contains any ``Rec`` or ``Var`` node."""
    match s:
        case End() | Wait():
            return False
        case Var():
            return True
        case Branch(choices=choices) | Select(choices=choices):
            return any(_contains_rec(body) for _, body in choices)
        case Parallel(branches=branches):
            return any(_contains_rec(b) for b in branches)
        case Rec():
            return True
        case Continuation(left=left, right=right):
            return _contains_rec(left) or _contains_rec(right)
    return False  # pragma: no cover


def _contains_parallel(s: SessionType) -> bool:
    """Return True if *s* contains any ``Parallel`` node."""
    match s:
        case End() | Wait() | Var():
            return False
        case Branch(choices=choices) | Select(choices=choices):
            return any(_contains_parallel(body) for _, body in choices)
        case Parallel():
            return True
        case Rec(body=body):
            return _contains_parallel(body)
        case Continuation(left=left, right=right):
            return _contains_parallel(left) or _contains_parallel(right)
    return False  # pragma: no cover


def _contains_continuation(s: SessionType) -> bool:
    """Return True if *s* contains any ``Continuation`` node."""
    match s:
        case End() | Wait() | Var():
            return False
        case Branch(choices=choices) | Select(choices=choices):
            return any(_contains_continuation(body) for _, body in choices)
        case Parallel(branches=branches):
            return any(_contains_continuation(b) for b in branches)
        case Rec(body=body):
            return _contains_continuation(body)
        case Continuation():
            return True
    return False  # pragma: no cover


def _has_nested_parallel(s: SessionType) -> bool:
    """Return True if *s* has parallel nested inside parallel."""
    return _check_nesting(s, False)


def _check_nesting(s: SessionType, inside: bool) -> bool:
    match s:
        case End() | Wait() | Var():
            return False
        case Branch(choices=choices) | Select(choices=choices):
            return any(_check_nesting(body, inside) for _, body in choices)
        case Parallel(branches=branches):
            if inside:
                return True
            return any(_check_nesting(b, True) for b in branches)
        case Rec(body=body):
            return _check_nesting(body, inside)
        case Continuation(left=left, right=right):
            return _check_nesting(left, inside) or _check_nesting(right, inside)
    return False  # pragma: no cover


def is_L3_type(s: SessionType) -> bool:
    """Check if *s* uses parallel but no recursion, continuation, or nested parallel.

    An L₃ type lives at exactly level L3 in the language hierarchy:
    it uses the parallel constructor but nothing beyond it.

    Parameters
    ----------
    s : SessionType
        The AST to classify.

    Returns
    -------
    bool
        True iff *s* uses ``∥`` and does not use ``rec``/``Var`` or ``.``
        or nested ``∥``.
    """
    if not _contains_parallel(s):
        return False
    if _contains_rec(s):
        return False
    if _contains_continuation(s):
        return False
    if _has_nested_parallel(s):
        return False
    return True


# ---------------------------------------------------------------------------
# State-space structure classification
# ---------------------------------------------------------------------------

def _is_chain(ss: StateSpace) -> bool:
    """Check if *ss* is a chain (total order): every state has at most one successor."""
    for s in ss.states:
        if len(list(ss.successors(s))) > 1:
            return False
    return True


def _is_tree(ss: StateSpace) -> bool:
    """Check if *ss* is a tree (DAG where every node has at most one predecessor).

    More precisely, check if every state (except top) has exactly one
    predecessor in the transition graph.
    """
    in_count: dict[int, int] = {s: 0 for s in ss.states}
    for src, _, tgt in ss.transitions:
        if src != tgt:  # ignore self-loops
            in_count[tgt] += 1
    for s, cnt in in_count.items():
        if s != ss.top and cnt > 1:
            return False
    return True


def L3_state_space_class(s: SessionType) -> str:
    """Characterise the state-space shape of an L₃ (or lower) type.

    Parameters
    ----------
    s : SessionType
        A session type AST (should be L₃ or lower for meaningful results).

    Returns
    -------
    str
        One of ``"chain"``, ``"tree"``, ``"product_of_chains"``,
        ``"product_of_trees"``, ``"diamond"``, ``"general_product"``,
        ``"general"``.
    """
    ss = build_statespace(s)

    # If it has product structure, analyse factors
    if ss.product_factors is not None and len(ss.product_factors) >= 2:
        factor_classes = []
        for f in ss.product_factors:
            if _is_chain(f):
                factor_classes.append("chain")
            elif _is_tree(f):
                factor_classes.append("tree")
            else:
                factor_classes.append("general")

        if all(c == "chain" for c in factor_classes):
            return "product_of_chains"
        elif all(c in ("chain", "tree") for c in factor_classes):
            return "product_of_trees"
        else:
            return "general_product"

    # No product structure — classify directly
    if _is_chain(ss):
        return "chain"
    if _is_tree(ss):
        return "tree"

    # Check for diamond shape (4 states: top -> a,b -> bot)
    if len(ss.states) == 4:
        succs_top = ss.successors(ss.top)
        if len(succs_top) == 2:
            all_reach_bot = all(ss.bottom in ss.successors(s) for s in succs_top)
            if all_reach_bot:
                return "diamond"

    return "general"


# ---------------------------------------------------------------------------
# Factor decomposition (inverse of product construction)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FactorAnalysis:
    """Analysis of a single factor in a product decomposition.

    Attributes
    ----------
    index : int
        Factor index (0-based).
    size : int
        Number of states in the factor.
    height : int
        Length of the longest chain from top to bottom.
    width : int
        Maximum antichain size (maximum number of states at any depth level).
    is_distributive : bool
        Whether this factor forms a distributive lattice.
    classification : str
        Birkhoff classification: "boolean", "distributive", "modular",
        "lattice", "not_lattice".
    """

    index: int
    size: int
    height: int
    width: int
    is_distributive: bool
    classification: str


def _compute_height(ss: StateSpace) -> int:
    """Longest directed path length from top to bottom in *ss*."""
    # BFS from top, tracking distance
    dist: dict[int, int] = {ss.top: 0}
    queue = [ss.top]
    while queue:
        s = queue.pop(0)
        for _, t in ss.enabled(s):
            if t not in dist:
                dist[t] = dist[s] + 1
                queue.append(t)
            else:
                dist[t] = max(dist[t], dist[s] + 1)

    # Use longest path via topological BFS with relaxation
    # Re-do with topological ordering for correctness
    longest: dict[int, int] = {s: 0 for s in ss.states}
    longest[ss.top] = 0

    # Build adjacency
    adj: dict[int, list[int]] = {s: [] for s in ss.states}
    in_deg: dict[int, int] = {s: 0 for s in ss.states}
    for src, _, tgt in ss.transitions:
        if src != tgt:
            adj[src].append(tgt)
            in_deg[tgt] = in_deg.get(tgt, 0) + 1

    # Topological order (Kahn's algorithm)
    from collections import deque
    q: deque[int] = deque()
    for s in ss.states:
        if in_deg[s] == 0:
            q.append(s)

    order: list[int] = []
    while q:
        s = q.popleft()
        order.append(s)
        for t in adj[s]:
            in_deg[t] -= 1
            if in_deg[t] == 0:
                q.append(t)

    # Relax in topological order
    longest = {s: 0 for s in ss.states}
    for s in order:
        for t in adj[s]:
            if longest[t] < longest[s] + 1:
                longest[t] = longest[s] + 1

    return longest.get(ss.bottom, 0)


def _compute_width(ss: StateSpace) -> int:
    """Maximum number of states at any BFS depth level from top."""
    from collections import deque
    depth: dict[int, int] = {ss.top: 0}
    q: deque[int] = deque([ss.top])
    while q:
        s = q.popleft()
        for _, t in ss.enabled(s):
            if t not in depth:
                depth[t] = depth[s] + 1
                q.append(t)

    if not depth:
        return 1
    level_counts: dict[int, int] = {}
    for d in depth.values():
        level_counts[d] = level_counts.get(d, 0) + 1
    return max(level_counts.values())


def product_factors(ss: StateSpace) -> list[StateSpace] | None:
    """Decompose a product state space into its factors.

    If *ss* was built from a ``Parallel`` construct, it stores factor
    information in ``product_factors``.  This function returns those
    factors, or ``None`` if *ss* is not a product.

    Parameters
    ----------
    ss : StateSpace
        A state space (possibly a product).

    Returns
    -------
    list[StateSpace] | None
        The factor state spaces, or None if not a product.
    """
    if ss.product_factors is not None and len(ss.product_factors) >= 2:
        return list(ss.product_factors)
    return None


def factor_analysis(ss: StateSpace) -> list[FactorAnalysis] | None:
    """Analyse each factor of a product state space.

    Parameters
    ----------
    ss : StateSpace
        A state space (possibly a product).

    Returns
    -------
    list[FactorAnalysis] | None
        Per-factor analysis, or None if *ss* is not a product.
    """
    factors = product_factors(ss)
    if factors is None:
        return None

    results: list[FactorAnalysis] = []
    for i, f in enumerate(factors):
        h = _compute_height(f)
        w = _compute_width(f)
        dr = check_distributive(f)
        results.append(FactorAnalysis(
            index=i,
            size=len(f.states),
            height=h,
            width=w,
            is_distributive=dr.is_distributive,
            classification=dr.classification,
        ))
    return results


# ---------------------------------------------------------------------------
# Size prediction
# ---------------------------------------------------------------------------

def product_size_prediction(factor_sizes: list[int]) -> int:
    """Predict the product state space size from factor sizes.

    By the product lattice theorem, ``|L₁ × L₂ × … × Lₙ| = ∏ |Lᵢ|``.

    Parameters
    ----------
    factor_sizes : list[int]
        The sizes (state counts) of each factor.

    Returns
    -------
    int
        The predicted product state space size.
    """
    result = 1
    for s in factor_sizes:
        result *= s
    return result


def product_height_prediction(factor_heights: list[int]) -> int:
    """Predict the product lattice height from factor heights.

    ``height(L₁ × L₂ × … × Lₙ) = ∑ height(Lᵢ)``.

    Parameters
    ----------
    factor_heights : list[int]
        The heights of each factor lattice.

    Returns
    -------
    int
        The predicted product lattice height.
    """
    return sum(factor_heights)


def product_width_prediction(factor_widths: list[int]) -> int:
    """Predict the product lattice width from factor widths.

    The width of a product is at most ``∏ width(Lᵢ)`` (exact for
    products of chains; may be less for products of wider lattices).

    Parameters
    ----------
    factor_widths : list[int]
        The widths of each factor lattice.

    Returns
    -------
    int
        Upper bound on the product lattice width.
    """
    result = 1
    for w in factor_widths:
        result *= w
    return result


# ---------------------------------------------------------------------------
# Distributivity preservation
# ---------------------------------------------------------------------------

def distributivity_from_factors(factor_distributive: list[bool]) -> bool:
    """Product is distributive iff all factors are distributive.

    This follows from the fact that the direct product of distributive
    lattices is distributive (Grätzer, *Lattice Theory*, Theorem 104).

    Parameters
    ----------
    factor_distributive : list[bool]
        Whether each factor is distributive.

    Returns
    -------
    bool
        True iff the product is distributive.
    """
    return all(factor_distributive)


def verify_distributivity_preservation(ss: StateSpace) -> bool | None:
    """Verify the distributivity preservation theorem on *ss*.

    Checks that ``distributive(product) == all(distributive(factor))``.

    Returns None if *ss* is not a product, True if the theorem holds,
    False if it fails (should never happen).
    """
    fa = factor_analysis(ss)
    if fa is None:
        return None

    predicted = distributivity_from_factors([f.is_distributive for f in fa])
    dr = check_distributive(ss)
    return dr.is_distributive == predicted


# ---------------------------------------------------------------------------
# L₃ enumeration
# ---------------------------------------------------------------------------

def _enumerate_L3_branches(
    depth: int,
    labels: tuple[str, ...],
    max_width: int,
    for_parallel: bool = False,
) -> Iterator[SessionType]:
    """Enumerate branch/select/end types suitable for L₃.

    No recursion. Uses ``Wait`` as terminal inside parallel branches,
    ``End`` outside.
    """
    terminal = Wait() if for_parallel else End()
    yield terminal

    if depth <= 0:
        return

    # Branch: &{m1: S1, ...}
    for w in range(1, min(max_width, len(labels)) + 1):
        for label_combo in combinations(labels, w):
            subs = list(_enumerate_L3_branches(depth - 1, labels, max_width, for_parallel))
            for choices in cartesian(subs, repeat=w):
                yield Branch(tuple(zip(label_combo, choices)))

    # Selection: +{l1: S1, ...}
    for w in range(1, min(max_width, len(labels)) + 1):
        for label_combo in combinations(labels, w):
            subs = list(_enumerate_L3_branches(depth - 1, labels, max_width, for_parallel))
            for choices in cartesian(subs, repeat=w):
                yield Select(tuple(zip(label_combo, choices)))


def L3_enumerate(
    depth: int = 2,
    width: int = 2,
    labels: tuple[str, ...] = ("a", "b"),
) -> list[tuple[str, int, int, bool]]:
    """Enumerate all L₃ types up to given depth/width.

    L₃ types have exactly one top-level ``Parallel`` with two branches,
    each branch being an L₂ type (branch/select/end/wait, no recursion).

    Parameters
    ----------
    depth : int
        Maximum AST depth *within each parallel branch*.
    width : int
        Maximum branch/select width.
    labels : tuple[str, ...]
        Available method labels.  Left branches use labels as-is;
        right branches are suffixed with ``_r`` for disjointness (WF-Par).

    Returns
    -------
    list[tuple[str, int, int, bool]]
        Each entry is ``(type_string, num_states, num_transitions, is_lattice)``.
    """
    results: list[tuple[str, int, int, bool]] = []

    left_labels = labels
    right_labels = tuple(l + "_r" for l in labels)

    left_types = list(_enumerate_L3_branches(depth, left_labels, width, for_parallel=True))
    right_types = list(_enumerate_L3_branches(depth, right_labels, width, for_parallel=True))

    for left in left_types:
        for right in right_types:
            par = Parallel((left, right))
            type_str = pretty(par)
            try:
                ss = build_statespace(par)
                lr = check_lattice(ss)
                results.append((
                    type_str,
                    len(ss.states),
                    len(ss.transitions),
                    lr.is_lattice,
                ))
            except Exception:
                # Skip malformed types
                continue

    return results


# ---------------------------------------------------------------------------
# Comparison: L₂ vs L₃
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LevelComparison:
    """Comparison of structural properties between L₂ and L₃ types.

    Attributes
    ----------
    l2_count : int
        Number of L₂ types enumerated.
    l3_count : int
        Number of L₃ types enumerated.
    l2_max_states : int
        Maximum state count among L₂ types.
    l3_max_states : int
        Maximum state count among L₃ types.
    l2_max_width : int
        Maximum lattice width among L₂ types.
    l3_max_width : int
        Maximum lattice width among L₃ types.
    l2_all_lattice : bool
        Whether all L₂ types form lattices.
    l3_all_lattice : bool
        Whether all L₃ types form lattices.
    l3_has_products : bool
        Whether any L₃ types have product structure.
    """

    l2_count: int
    l3_count: int
    l2_max_states: int
    l3_max_states: int
    l2_max_width: int
    l3_max_width: int
    l2_all_lattice: bool
    l3_all_lattice: bool
    l3_has_products: bool


def compare_L2_L3(
    depth: int = 2,
    width: int = 2,
    labels: tuple[str, ...] = ("a", "b"),
) -> LevelComparison:
    """Compare L₂ and L₃ types at the given parameters.

    Parameters
    ----------
    depth : int
        AST depth bound.
    width : int
        Branch/select width bound.
    labels : tuple[str, ...]
        Method labels.

    Returns
    -------
    LevelComparison
        Summary comparison.
    """
    # Enumerate L₂ types (no parallel, no recursion)
    l2_types = list(_enumerate_L3_branches(depth, labels, width, for_parallel=False))
    l2_max_states = 0
    l2_max_width = 0
    l2_all_lattice = True
    l2_count = 0

    for t in l2_types:
        try:
            ss = build_statespace(t)
            lr = check_lattice(ss)
            l2_count += 1
            l2_max_states = max(l2_max_states, len(ss.states))
            l2_max_width = max(l2_max_width, _compute_width(ss))
            if not lr.is_lattice:
                l2_all_lattice = False
        except Exception:
            continue

    # Enumerate L₃ types
    l3_results = L3_enumerate(depth, width, labels)
    l3_count = len(l3_results)
    l3_max_states = max((r[1] for r in l3_results), default=0)
    l3_all_lattice = all(r[3] for r in l3_results)

    # Check for product structure
    l3_has_products = l3_count > 0  # All L₃ types have product structure by definition

    # Compute max width for L₃
    l3_max_width = 0
    for type_str, _, _, _ in l3_results:
        try:
            ast = parse(type_str)
            ss = build_statespace(ast)
            l3_max_width = max(l3_max_width, _compute_width(ss))
        except Exception:
            continue

    return LevelComparison(
        l2_count=l2_count,
        l3_count=l3_count,
        l2_max_states=l2_max_states,
        l3_max_states=l3_max_states,
        l2_max_width=l2_max_width,
        l3_max_width=l3_max_width,
        l2_all_lattice=l2_all_lattice,
        l3_all_lattice=l3_all_lattice,
        l3_has_products=l3_has_products,
    )


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def L3_summary(depth: int = 2, width: int = 2) -> str:
    """Generate a human-readable summary of L₃ analysis.

    Parameters
    ----------
    depth : int
        AST depth bound for enumeration.
    width : int
        Branch/select width bound.

    Returns
    -------
    str
        Multi-line summary report.
    """
    lines: list[str] = []
    lines.append("L3: Parallel Composition and Product Lattices")
    lines.append("=" * 50)
    lines.append("")

    # Enumerate
    results = L3_enumerate(depth, width)
    lines.append(f"Enumerated {len(results)} L₃ types (depth={depth}, width={width})")
    lattice_count = sum(1 for r in results if r[3])
    lines.append(f"  Lattices: {lattice_count}/{len(results)}")

    if results:
        max_states = max(r[1] for r in results)
        max_trans = max(r[2] for r in results)
        lines.append(f"  Max states: {max_states}")
        lines.append(f"  Max transitions: {max_trans}")

    lines.append("")

    # Comparison
    comp = compare_L2_L3(depth, width)
    lines.append("L₂ vs L₃ Comparison")
    lines.append("-" * 30)
    lines.append(f"  L₂ types: {comp.l2_count}")
    lines.append(f"  L₃ types: {comp.l3_count}")
    lines.append(f"  L₂ max states: {comp.l2_max_states}")
    lines.append(f"  L₃ max states: {comp.l3_max_states}")
    lines.append(f"  L₂ all lattice: {comp.l2_all_lattice}")
    lines.append(f"  L₃ all lattice: {comp.l3_all_lattice}")
    lines.append(f"  L₃ has products: {comp.l3_has_products}")

    return "\n".join(lines)
