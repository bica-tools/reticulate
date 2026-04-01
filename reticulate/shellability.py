"""Shellability and EL-labeling for session type lattices (Step 30w).

A finite lattice is **shellable** if its order complex (the simplicial complex
of all chains) can be assembled face-by-face such that each new face attaches
along a topologically nice boundary.  Shellable lattices have particularly
clean homological structure: the homology is torsion-free and concentrated
in a single degree.

The primary tool for proving shellability is **EL-labeling** (edge-lexicographic
labeling), introduced by Björner.  An EL-labeling of a bounded poset P assigns
a label λ(x, y) to each edge x ⋗ y of the Hasse diagram such that in every
interval [a, b]:
  1. There is a **unique increasing maximal chain** (labels strictly increase).
  2. This chain is **lexicographically first** among all maximal chains.

For session type lattices, the transition labels (method names) provide a natural
candidate for the EL-labeling.  We investigate when this labeling is an EL-labeling
and what it implies for the topological structure of the protocol.

Key functions:
  - ``hasse_labels(ss)``            -- extract labeled Hasse diagram
  - ``maximal_chains(ss)``          -- enumerate all maximal chains top→bottom
  - ``check_el_labeling(ss)``       -- verify EL-labeling property
  - ``check_cl_labeling(ss)``       -- verify CL-labeling (weaker condition)
  - ``check_shellability(ss)``      -- determine shellability
  - ``compute_h_vector(ss)``        -- h-vector of the order complex
  - ``compute_f_vector(ss)``        -- f-vector (face counts by dimension)
  - ``descent_set(chain, labels)``  -- descents in a labeled chain
  - ``analyze_shellability(ss)``    -- full shellability analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HasseEdge:
    """A labeled edge in the Hasse diagram.

    Attributes:
        source: Upper element (covers target).
        target: Lower element (covered by source).
        label: Transition label (method name).
    """
    source: int
    target: int
    label: str


@dataclass(frozen=True)
class LabeledChain:
    """A maximal chain with its edge labels.

    Attributes:
        states: Sequence of states from top to bottom.
        labels: Sequence of edge labels (len = len(states) - 1).
        descents: Positions where label order decreases.
        is_increasing: True iff labels are strictly increasing.
    """
    states: tuple[int, ...]
    labels: tuple[str, ...]
    descents: tuple[int, ...]
    is_increasing: bool


@dataclass(frozen=True)
class ELResult:
    """Result of EL-labeling verification.

    Attributes:
        is_el_labeling: True iff the labeling is an EL-labeling.
        num_intervals: Number of intervals checked.
        intervals_with_unique_increasing: Number of intervals with unique increasing chain.
        failing_interval: First interval where the EL property fails, or None.
        failing_reason: Reason for failure, or None.
    """
    is_el_labeling: bool
    num_intervals: int
    intervals_with_unique_increasing: int
    failing_interval: tuple[int, int] | None
    failing_reason: str | None


@dataclass(frozen=True)
class CLResult:
    """Result of CL-labeling (chain-edge labeling) verification.

    Attributes:
        is_cl_labeling: True iff the labeling is a CL-labeling.
        num_intervals: Number of intervals checked.
        failing_interval: First failing interval, or None.
    """
    is_cl_labeling: bool
    num_intervals: int
    failing_interval: tuple[int, int] | None


@dataclass(frozen=True)
class ShellabilityResult:
    """Result of shellability analysis.

    Attributes:
        is_shellable: True iff the order complex is shellable.
        is_el_shellable: True iff shellability is proved via EL-labeling.
        is_cl_shellable: True iff shellability is proved via CL-labeling.
        is_graded: True iff the lattice is graded (all max chains same length).
        el_result: EL-labeling check result.
        cl_result: CL-labeling check result.
        num_maximal_chains: Number of maximal chains from top to bottom.
        height: Lattice height (length of longest chain).
        f_vector: Face counts by dimension.
        h_vector: h-vector of the order complex.
        num_descents: Total number of descents across all maximal chains.
    """
    is_shellable: bool
    is_el_shellable: bool
    is_cl_shellable: bool
    is_graded: bool
    el_result: ELResult
    cl_result: CLResult
    num_maximal_chains: int
    height: int
    f_vector: tuple[int, ...]
    h_vector: tuple[int, ...]
    num_descents: int


# ---------------------------------------------------------------------------
# Internal: Hasse diagram extraction
# ---------------------------------------------------------------------------

def _build_reachability(ss: StateSpace) -> dict[int, set[int]]:
    """Compute forward reachability (inclusive) for all states."""
    adj: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        adj[src].add(tgt)

    reach: dict[int, set[int]] = {}
    for s in ss.states:
        visited: set[int] = set()
        stack = [s]
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            for t in adj.get(v, set()):
                stack.append(t)
        reach[s] = visited
    return reach


def _build_covering(ss: StateSpace) -> list[tuple[int, int, str]]:
    """Extract covering relations (Hasse edges) with labels.

    An edge (s, l, t) is a covering relation if there is no intermediate
    state u such that s > u > t (where > is reachability ordering).
    """
    reach = _build_reachability(ss)

    # Build adjacency with labels
    adj: dict[int, list[tuple[int, str]]] = {s: [] for s in ss.states}
    for src, label, tgt in ss.transitions:
        adj[src].append((tgt, label))

    covers: list[tuple[int, int, str]] = []
    for s in ss.states:
        for t, label in adj[s]:
            # Check if s covers t: no intermediate u with s > u > t
            is_cover = True
            for u, _ in adj[s]:
                if u != t and t in reach.get(u, set()) and u != s:
                    is_cover = False
                    break
            if is_cover:
                covers.append((s, t, label))

    return covers


# ---------------------------------------------------------------------------
# Public API: Hasse diagram
# ---------------------------------------------------------------------------

def hasse_labels(ss: StateSpace) -> list[HasseEdge]:
    """Extract the labeled Hasse diagram of the state space.

    Returns covering relations with transition labels.
    """
    covers = _build_covering(ss)
    return [HasseEdge(source=s, target=t, label=l) for s, t, l in covers]


# ---------------------------------------------------------------------------
# Public API: Chain enumeration
# ---------------------------------------------------------------------------

def maximal_chains(
    ss: StateSpace,
    max_chains: int = 10000,
) -> list[LabeledChain]:
    """Enumerate all maximal chains from top to bottom.

    A maximal chain is a sequence top = s₀ ⋗ s₁ ⋗ ... ⋗ sₖ = bottom
    using only covering relations (Hasse edges).

    Args:
        ss: State space.
        max_chains: Maximum number of chains to enumerate.

    Returns:
        List of LabeledChain objects.
    """
    covers = _build_covering(ss)

    # Build covering adjacency: state → [(target, label)]
    cov_adj: dict[int, list[tuple[int, str]]] = {s: [] for s in ss.states}
    for s, t, l in covers:
        cov_adj[s].append((t, l))

    # DFS from top to bottom (iterative to avoid recursion depth issues)
    chains: list[LabeledChain] = []

    # Stack: (state, path, labels, visited_set)
    stack: list[tuple[int, list[int], list[str], frozenset[int]]] = [
        (ss.top, [ss.top], [], frozenset({ss.top}))
    ]

    while stack and len(chains) < max_chains:
        state, path, labels, visited = stack.pop()

        if state == ss.bottom:
            lab_tuple = tuple(labels)
            desc = tuple(
                i for i in range(len(lab_tuple) - 1)
                if lab_tuple[i] > lab_tuple[i + 1]
            )
            chains.append(LabeledChain(
                states=tuple(path),
                labels=lab_tuple,
                descents=desc,
                is_increasing=len(desc) == 0,
            ))
            continue

        for t, l in sorted(cov_adj[state], key=lambda x: x[1], reverse=True):
            if t not in visited:  # Avoid cycles
                stack.append((
                    t, path + [t], labels + [l], visited | {t}
                ))

    return chains


# ---------------------------------------------------------------------------
# Public API: Descent analysis
# ---------------------------------------------------------------------------

def descent_set(chain: LabeledChain) -> frozenset[int]:
    """Return the descent set of a labeled chain.

    Position i is a descent if labels[i] > labels[i+1] (lexicographic).
    """
    return frozenset(chain.descents)


def descent_statistic(chains: list[LabeledChain]) -> dict[int, int]:
    """Count chains by number of descents.

    Returns: {num_descents: count_of_chains_with_that_many_descents}.
    """
    counts: dict[int, int] = {}
    for c in chains:
        n = len(c.descents)
        counts[n] = counts.get(n, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Public API: EL-labeling check
# ---------------------------------------------------------------------------

def check_el_labeling(ss: StateSpace) -> ELResult:
    """Check whether the transition labels form an EL-labeling.

    An EL-labeling requires that in every interval [a, b]:
    1. There is exactly one maximal chain with strictly increasing labels.
    2. This increasing chain is lexicographically first.
    """
    covers = _build_covering(ss)
    reach = _build_reachability(ss)

    # Build covering adjacency
    cov_adj: dict[int, list[tuple[int, str]]] = {s: [] for s in ss.states}
    for s, t, l in covers:
        cov_adj[s].append((t, l))

    # Check EL property for every interval [a, b] with a > b
    states = sorted(ss.states)
    num_intervals = 0
    num_ok = 0
    failing_interval = None
    failing_reason = None

    for a in states:
        for b in states:
            if a == b:
                continue
            if b not in reach.get(a, set()):
                continue
            # Interval [a, b]: a > b in the ordering
            num_intervals += 1

            # Enumerate maximal chains from a to b (bounded, iterative)
            interval_chains: list[tuple[tuple[str, ...], tuple[int, ...]]] = []

            enum_stack: list[tuple[int, list[int], list[str], frozenset[int]]] = [
                (a, [a], [], frozenset({a}))
            ]
            while enum_stack and len(interval_chains) <= 100:
                st, pth, lbs, vis = enum_stack.pop()
                if st == b:
                    interval_chains.append((tuple(lbs), tuple(pth)))
                    continue
                for t, l in sorted(cov_adj[st], key=lambda x: x[1], reverse=True):
                    if t not in vis and b in reach.get(t, set()):
                        enum_stack.append((t, pth + [t], lbs + [l], vis | {t}))

            if not interval_chains:
                # No chains — skip (a and b are in same SCC or path is non-covering)
                continue

            # Find increasing chains
            increasing = [
                (labs, path) for labs, path in interval_chains
                if all(labs[i] < labs[i + 1] for i in range(len(labs) - 1))
            ]

            # Check: exactly one increasing chain
            if len(increasing) != 1:
                if failing_interval is None:
                    failing_interval = (a, b)
                    if len(increasing) == 0:
                        failing_reason = (
                            f"No increasing chain in interval [{a}, {b}]"
                        )
                    else:
                        failing_reason = (
                            f"{len(increasing)} increasing chains in [{a}, {b}]"
                        )
                continue

            # Check: increasing chain is lexicographically first
            inc_labs = increasing[0][0]
            all_labs = sorted(interval_chains, key=lambda x: x[0])
            if all_labs[0][0] != inc_labs:
                if failing_interval is None:
                    failing_interval = (a, b)
                    failing_reason = (
                        f"Increasing chain {inc_labs} is not lex-first "
                        f"in interval [{a}, {b}]; lex-first is {all_labs[0][0]}"
                    )
                continue

            num_ok += 1

    is_el = failing_interval is None and num_intervals > 0

    return ELResult(
        is_el_labeling=is_el,
        num_intervals=num_intervals,
        intervals_with_unique_increasing=num_ok,
        failing_interval=failing_interval,
        failing_reason=failing_reason,
    )


def check_cl_labeling(ss: StateSpace) -> CLResult:
    """Check whether the labeling is a CL-labeling (chain-lexicographic).

    A CL-labeling is weaker than EL: it requires the unique increasing
    chain property but only along rooted intervals (intervals from a
    fixed root element). For bounded posets, CL implies shellability.

    We check intervals rooted at top: for every b, the interval [top, b]
    has a unique increasing maximal chain.
    """
    covers = _build_covering(ss)
    reach = _build_reachability(ss)

    cov_adj: dict[int, list[tuple[int, str]]] = {s: [] for s in ss.states}
    for s, t, l in covers:
        cov_adj[s].append((t, l))

    num_intervals = 0
    failing_interval = None

    for b in sorted(ss.states):
        if b == ss.top:
            continue
        if b not in reach.get(ss.top, set()):
            continue
        num_intervals += 1

        # Enumerate chains from top to b (iterative)
        interval_chains: list[tuple[str, ...]] = []

        cl_stack: list[tuple[int, list[str], frozenset[int]]] = [
            (ss.top, [], frozenset({ss.top}))
        ]
        while cl_stack and len(interval_chains) <= 100:
            st, lbs, vis = cl_stack.pop()
            if st == b:
                interval_chains.append(tuple(lbs))
                continue
            for t, l in sorted(cov_adj[st], key=lambda x: x[1], reverse=True):
                if t not in vis and (b in reach.get(t, set()) or t == b):
                    cl_stack.append((t, lbs + [l], vis | {t}))

        if not interval_chains:
            continue

        increasing = [
            labs for labs in interval_chains
            if all(labs[i] < labs[i + 1] for i in range(len(labs) - 1))
        ]

        if len(increasing) != 1:
            if failing_interval is None:
                failing_interval = (ss.top, b)
            continue

    is_cl = failing_interval is None and num_intervals > 0

    return CLResult(
        is_cl_labeling=is_cl,
        num_intervals=num_intervals,
        failing_interval=failing_interval,
    )


# ---------------------------------------------------------------------------
# Public API: f-vector and h-vector
# ---------------------------------------------------------------------------

def compute_f_vector(ss: StateSpace) -> tuple[int, ...]:
    """Compute the f-vector of the order complex.

    f_i = number of chains of length i (i+1 elements).
    f_0 = |states|, f_1 = |covering relations|, etc.

    The order complex has faces = chains in the poset.
    """
    reach = _build_reachability(ss)
    covers = _build_covering(ss)

    cov_adj: dict[int, list[tuple[int, str]]] = {s: [] for s in ss.states}
    for s, t, l in covers:
        cov_adj[s].append((t, l))

    # Count chains of each length via DFS from every state
    # f_0 = number of vertices (states)
    # f_1 = number of edges (covering relations)
    # f_k = number of chains of k+1 elements
    chain_counts: dict[int, int] = {0: len(ss.states)}

    # Enumerate chains from each state (with cycle avoidance)
    for start in ss.states:
        # Stack: (state, chain_length, visited)
        stack: list[tuple[int, int, frozenset[int]]] = [
            (start, 1, frozenset({start}))
        ]
        while stack:
            state, length, visited = stack.pop()
            for t, _ in cov_adj.get(state, []):
                if t not in visited:  # Avoid cycles
                    chain_counts[length] = chain_counts.get(length, 0) + 1
                    stack.append((t, length + 1, visited | {t}))

    max_dim = max(chain_counts.keys()) if chain_counts else 0
    return tuple(chain_counts.get(i, 0) for i in range(max_dim + 1))


def compute_h_vector(ss: StateSpace) -> tuple[int, ...]:
    """Compute the h-vector from the f-vector.

    The h-vector is defined by the relation:
    sum_i h_i * x^(d-i) = sum_i f_i * (x-1)^(d-i)

    where d = dimension of the order complex.

    For shellable complexes, all h_i >= 0 (a necessary condition).
    """
    f = compute_f_vector(ss)
    d = len(f) - 1
    if d < 0:
        return (1,)

    # h_k = sum_{i=0}^{k} (-1)^{k-i} * C(d-i, k-i) * f_i
    def _binomial(n: int, k: int) -> int:
        if k < 0 or k > n:
            return 0
        if k == 0 or k == n:
            return 1
        result = 1
        for i in range(min(k, n - k)):
            result = result * (n - i) // (i + 1)
        return result

    h: list[int] = []
    for k in range(d + 1):
        val = 0
        for i in range(k + 1):
            sign = (-1) ** (k - i)
            val += sign * _binomial(d - i, k - i) * (f[i] if i < len(f) else 0)
        h.append(val)

    return tuple(h)


# ---------------------------------------------------------------------------
# Public API: Shellability check
# ---------------------------------------------------------------------------

def check_shellability(ss: StateSpace) -> ShellabilityResult:
    """Full shellability analysis of a session type state space.

    Checks EL-labeling, CL-labeling, computes f-vector and h-vector,
    and determines whether the order complex is shellable.
    """
    # Get maximal chains
    chains = maximal_chains(ss)
    num_chains = len(chains)

    # Height (from maximal chains)
    height = max((len(c.states) - 1 for c in chains), default=0)

    # Check gradedness: all maximal chains have the same length
    chain_lengths = {len(c.states) for c in chains}
    is_graded = len(chain_lengths) <= 1

    # EL and CL checks
    el = check_el_labeling(ss)
    cl = check_cl_labeling(ss)

    # f-vector and h-vector
    f_vec = compute_f_vector(ss)
    h_vec = compute_h_vector(ss)

    # Shellability: EL-labeling implies shellability for bounded posets
    # CL-labeling also implies shellability
    is_el_shellable = el.is_el_labeling
    is_cl_shellable = cl.is_cl_labeling

    # Additional check: h-vector non-negativity (necessary for shellability)
    h_nonneg = all(h >= 0 for h in h_vec)

    is_shellable = is_el_shellable or is_cl_shellable

    # Total descents
    total_descents = sum(len(c.descents) for c in chains)

    return ShellabilityResult(
        is_shellable=is_shellable,
        is_el_shellable=is_el_shellable,
        is_cl_shellable=is_cl_shellable,
        is_graded=is_graded,
        el_result=el,
        cl_result=cl,
        num_maximal_chains=num_chains,
        height=height,
        f_vector=f_vec,
        h_vector=h_vec,
        num_descents=total_descents,
    )


# ---------------------------------------------------------------------------
# Public API: Convenience
# ---------------------------------------------------------------------------

def analyze_shellability(ss: StateSpace) -> ShellabilityResult:
    """Alias for check_shellability — full analysis."""
    return check_shellability(ss)


def is_shellable(ss: StateSpace) -> bool:
    """Quick check: is the order complex shellable?"""
    return check_shellability(ss).is_shellable


def is_el_labelable(ss: StateSpace) -> bool:
    """Quick check: do the transition labels form an EL-labeling?"""
    return check_el_labeling(ss).is_el_labeling
