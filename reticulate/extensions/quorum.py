"""Quorum extension: Q(k,n){S} — at least k of n copies must complete.

Grammar extension:
  S ::= ... | Q(k,n){ S }    -- at least k of n reach end

Semantics:
  L(Q(k,n){S}) = { (s1,...,sn) in L(S)^n : |{i : si = bot}| >= k }

  This is the n-fold product TRUNCATED: states where fewer than k
  components have reached bottom are REMOVED.

Key questions:
  1. Is Q(k,n){S} always a lattice? (Conjecture: NO for non-chain S)
  2. When is it a lattice? (Conjecture: when L(S) is a chain)
  3. What happens to meets/joins at the truncation boundary?
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import product as itertools_product
from typing import TYPE_CHECKING

from reticulate.parser import SessionType, parse, pretty, End
from reticulate.statespace import StateSpace, build_statespace
from reticulate.product import power_statespace
from reticulate.lattice import check_lattice, LatticeResult


# ---------------------------------------------------------------------------
# AST node
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Quorum:
    """Quorum type ``Q(k, n){ S }`` — at least *k* of *n* copies reach end.

    Attributes:
        body: The session type replicated across *n* copies.
        k: Minimum number of copies that must reach bottom (end).
        n: Total number of parallel copies.
    """
    body: SessionType
    k: int
    n: int

    def __post_init__(self) -> None:
        if self.k < 0:
            raise ValueError(f"k must be non-negative, got {self.k}")
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")
        if self.k > self.n:
            raise ValueError(f"k={self.k} cannot exceed n={self.n}")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_quorum(s: str) -> Quorum:
    """Parse a quorum type string ``Q(k,n){ S }``.

    Parameters:
        s: String of the form ``Q(k,n){ body }`` where body is a valid
           session type string.

    Returns:
        A ``Quorum`` AST node.

    Raises:
        ValueError: If the string is not a valid quorum type.
    """
    s = s.strip()
    m = re.match(r'^Q\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*\{(.*)\}\s*$', s, re.DOTALL)
    if not m:
        raise ValueError(f"Invalid quorum syntax: {s!r}. Expected Q(k,n){{ S }}")
    k = int(m.group(1))
    n = int(m.group(2))
    body_str = m.group(3).strip()
    body = parse(body_str)
    return Quorum(body=body, k=k, n=n)


# ---------------------------------------------------------------------------
# State-space construction
# ---------------------------------------------------------------------------

def build_quorum_statespace(node: Quorum) -> StateSpace:
    """Build the quorum state space: n-fold product truncated by completion count.

    The quorum state space is the n-fold product L(S)^n restricted to states
    where at least *k* components have reached bottom (end).

    The top and bottom of the full product are always retained (top has 0
    components at bottom, bottom has n components at bottom). After truncation,
    states with fewer than k components at bottom are removed UNLESS they are
    reachable from top and can reach a surviving state.

    Actually, the semantics is: we keep ALL states where the number of
    completed components is >= k, PLUS all states that can reach such a state
    (i.e., states "on the way" to the quorum). The top (all components at
    their initial state) always survives because it can reach the bottom.

    More precisely: the quorum state space keeps states where at least k
    components are at bottom. But we also need to keep the top and all states
    that form paths to the quorum-satisfying states.

    Simplest correct semantics: start with the full product, then remove
    states where fewer than k components are at bottom AND that state is
    a "terminal-ish" state (has no outgoing transitions to states with more
    completed components that satisfy the quorum).

    REVISED: The truncated product keeps:
    - All states where >= k components are at bottom (quorum met)
    - All states that can reach at least one quorum-met state (on the path)
    The new bottom is the original bottom (all n at bottom, satisfies k <= n).
    """
    base_ss = build_statespace(node.body)
    full_product = power_statespace(base_ss, node.n)

    if node.k == 0:
        # Q(0,n){S} = S^n — all states satisfy "at least 0 at bottom"
        return full_product

    # Identify bottom component for each factor
    bottom_id = base_ss.bottom

    # Build coordinate map: product state -> tuple of component states
    # power_statespace already provides product_coords
    coords = full_product.product_coords
    if coords is None:
        # Fallback: single factor (n=1)
        coords = {s: (s,) for s in full_product.states}

    # Count how many components are at bottom for each product state
    def completed_count(state_id: int) -> int:
        coord = coords[state_id]
        # Each factor's bottom is the base bottom. But after power construction,
        # IDs are remapped. We need to check via labels or coordinates.
        # product_coords maps to original factor state IDs.
        count = 0
        for c in coord:
            if c == bottom_id:
                count += 1
        return count

    # States that satisfy the quorum (>= k components at bottom)
    quorum_states = {s for s in full_product.states if completed_count(s) >= node.k}

    if not quorum_states:
        # This shouldn't happen since bottom always has n >= k components at bottom
        raise ValueError("No states satisfy quorum — impossible for valid k,n")

    # Compute reverse reachability: states that can reach a quorum state
    # Build reverse adjacency
    rev_adj: dict[int, set[int]] = {s: set() for s in full_product.states}
    fwd_adj: dict[int, set[int]] = {s: set() for s in full_product.states}
    for src, _lbl, tgt in full_product.transitions:
        rev_adj[tgt].add(src)
        fwd_adj[src].add(tgt)

    # BFS backward from quorum states
    can_reach_quorum: set[int] = set(quorum_states)
    queue = list(quorum_states)
    while queue:
        s = queue.pop()
        for pred in rev_adj[s]:
            if pred not in can_reach_quorum:
                can_reach_quorum.add(pred)
                queue.append(pred)

    # Also need forward reachability from top
    reachable_from_top = full_product.reachable_from(full_product.top)

    # Surviving states: reachable from top AND can reach quorum
    surviving = reachable_from_top & can_reach_quorum

    if full_product.top not in surviving:
        # Top can't reach any quorum state — degenerate
        surviving.add(full_product.top)

    # Build the truncated state space
    # Remap state IDs to be contiguous
    old_to_new: dict[int, int] = {}
    for i, s in enumerate(sorted(surviving)):
        old_to_new[s] = i

    new_transitions: list[tuple[int, str, int]] = []
    new_selection: set[tuple[int, str, int]] = set()
    for src, lbl, tgt in full_product.transitions:
        if src in surviving and tgt in surviving:
            tr = (old_to_new[src], lbl, old_to_new[tgt])
            new_transitions.append(tr)
            if (src, lbl, tgt) in full_product.selection_transitions:
                new_selection.add(tr)

    new_labels: dict[int, str] = {}
    for s in surviving:
        label = full_product.labels.get(s, str(s))
        cc = completed_count(s)
        new_labels[old_to_new[s]] = f"{label}[{cc}/{node.n}]"

    # New top = remapped original top
    new_top = old_to_new[full_product.top]

    # New bottom = remapped original bottom (all components at bottom)
    new_bottom = old_to_new[full_product.bottom]

    # Build new coordinate map
    new_coords: dict[int, tuple[int, ...]] = {}
    for s in surviving:
        if coords and s in coords:
            new_coords[old_to_new[s]] = coords[s]

    return StateSpace(
        states=set(old_to_new.values()),
        transitions=new_transitions,
        top=new_top,
        bottom=new_bottom,
        labels=new_labels,
        selection_transitions=new_selection,
        product_coords=new_coords,
        product_factors=full_product.product_factors,
    )


# ---------------------------------------------------------------------------
# Analysis result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuorumAnalysis:
    """Full analysis of a quorum type Q(k,n){S}.

    Attributes:
        body_type: The body session type string.
        k: Minimum completions required.
        n: Total copies.
        product_states: Number of states in the full n-fold product.
        surviving_states: Number of states in the truncated quorum space.
        removed_states: Number of states removed by truncation.
        quorum_states: Number of states where >= k components are at bottom.
        is_lattice: Whether the truncated space forms a lattice.
        lattice_result: Full lattice check result.
        counterexample: Pair of states that lost meet/join, or None.
    """
    body_type: str
    k: int
    n: int
    product_states: int
    surviving_states: int
    removed_states: int
    quorum_states: int
    is_lattice: bool
    lattice_result: LatticeResult
    counterexample: tuple[int, int, str] | None


# ---------------------------------------------------------------------------
# High-level analysis functions
# ---------------------------------------------------------------------------

def analyze_quorum(body_type: str, k: int, n: int) -> QuorumAnalysis:
    """Full analysis of Q(k,n){S}: build, truncate, check lattice.

    Parameters:
        body_type: Session type string for the body S.
        k: Minimum number of copies that must complete.
        n: Total number of copies.

    Returns:
        A ``QuorumAnalysis`` with product sizes, truncation info, and lattice check.
    """
    body = parse(body_type)
    quorum = Quorum(body=body, k=k, n=n)

    # Full product for comparison
    base_ss = build_statespace(body)
    full_product = power_statespace(base_ss, n)
    full_count = len(full_product.states)

    # Build truncated quorum space
    qss = build_quorum_statespace(quorum)
    surviving_count = len(qss.states)

    # Count quorum-satisfying states in full product
    bottom_id = base_ss.bottom
    coords = full_product.product_coords or {s: (s,) for s in full_product.states}
    quorum_count = sum(
        1 for s in full_product.states
        if sum(1 for c in coords[s] if c == bottom_id) >= k
    )

    # Check lattice
    lr = check_lattice(qss)

    return QuorumAnalysis(
        body_type=body_type,
        k=k,
        n=n,
        product_states=full_count,
        surviving_states=surviving_count,
        removed_states=full_count - surviving_count,
        quorum_states=quorum_count,
        is_lattice=lr.is_lattice,
        lattice_result=lr,
        counterexample=lr.counterexample,
    )


def is_quorum_lattice(body_type: str, k: int, n: int) -> bool:
    """Check if the truncated product Q(k,n){S} is still a lattice.

    Parameters:
        body_type: Session type string for the body S.
        k: Minimum completions required.
        n: Total copies.

    Returns:
        True iff the quorum state space forms a lattice.
    """
    return analyze_quorum(body_type, k, n).is_lattice


def quorum_counterexample(body_type: str, k: int, n: int) -> tuple[int, int, str] | None:
    """Find states that lost their meet/join due to truncation.

    Parameters:
        body_type: Session type string for the body S.
        k: Minimum completions required.
        n: Total copies.

    Returns:
        A tuple ``(state_a, state_b, "no_meet"|"no_join")`` if the quorum
        space is NOT a lattice, or None if it IS a lattice.
    """
    return analyze_quorum(body_type, k, n).counterexample


def fault_tolerance_analysis(body_type: str, max_n: int) -> list[dict]:
    """Sweep all (k,n) combinations for n=1..max_n, checking lattice structure.

    Parameters:
        body_type: Session type string for the body S.
        max_n: Maximum number of copies to test.

    Returns:
        A list of dicts, one per (k,n) pair, with keys:
        ``k``, ``n``, ``is_lattice``, ``product_states``,
        ``surviving_states``, ``removed_states``.
    """
    results: list[dict] = []
    for n in range(1, max_n + 1):
        for k in range(0, n + 1):
            analysis = analyze_quorum(body_type, k, n)
            results.append({
                "k": k,
                "n": n,
                "is_lattice": analysis.is_lattice,
                "product_states": analysis.product_states,
                "surviving_states": analysis.surviving_states,
                "removed_states": analysis.removed_states,
                "quorum_states": analysis.quorum_states,
            })
    return results


def pretty_quorum(node: Quorum) -> str:
    """Pretty-print a quorum type as ``Q(k,n){ S }``."""
    return f"Q({node.k},{node.n}){{ {pretty(node.body)} }}"
