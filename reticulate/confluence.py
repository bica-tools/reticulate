"""Confluence and Church-Rosser analysis for session-type state spaces.

This module implements Step 80m of the research programme.

The central theorem (proved in ``papers/steps/step80m-confluence-church-rosser
/proofs.tex``) is:

    For a finite, well-formed session-type state space L(S), the following
    are equivalent:

      1. L(S) is a meet-semilattice (every pair has a greatest lower bound).
      2. L(S) is globally confluent / Church-Rosser.
      3. L(S) is locally confluent.

This module provides:

  * ``find_critical_pairs`` -- enumerate every divergent diamond.
  * ``confluence_closure``  -- compute the meet (closing element) of two
    states, returning ``None`` if no common lower bound exists.
  * ``is_locally_confluent`` -- check that every critical pair closes.
  * ``is_globally_confluent`` -- check that every reachable pair has a meet.
  * ``check_confluence``    -- top-level entry point returning a frozen
    ``ConfluenceResult``.

The orientation matches ``lattice.py``: ``s1 >= s2`` iff there is a directed
path from ``s1`` to ``s2``. The meet (greatest lower bound) is therefore the
*closing* element of a divergence in the rewrite picture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConfluenceResult:
    """Outcome of a confluence analysis on a state space.

    Attributes:
        is_locally_confluent: Every critical pair (single-step divergence)
            has a common lower bound (closing element).
        is_globally_confluent: Every pair of reachable states has a meet.
        critical_pair_count: Number of distinct critical pairs.
        missing_meets: Pairs ``(x, y)`` (original state IDs) for which no
            common lower bound exists. Empty iff globally confluent.
        unclosed_critical_pairs: Critical pairs ``(origin, branch1, branch2)``
            whose two successors share no common lower bound. Empty iff
            locally confluent.
    """

    is_locally_confluent: bool
    is_globally_confluent: bool
    critical_pair_count: int
    missing_meets: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    unclosed_critical_pairs: tuple[tuple[int, int, int], ...] = field(
        default_factory=tuple
    )


# ---------------------------------------------------------------------------
# Reachability cache
# ---------------------------------------------------------------------------

def _forward_reach(ss: "StateSpace") -> dict[int, set[int]]:
    """Forward reachability under the SCC quotient.

    Cycles introduced by recursive session types are first collapsed
    (each strongly connected component becomes a single representative
    state, namely ``min(scc)``). The resulting DAG carries the
    reachability preorder used by ``lattice.py``.

    The returned mapping is keyed and valued by *original* state IDs:
    ``reach[s]`` is the set of original states ``t`` such that
    ``s reachstar t`` in the quotient -- i.e. all states in any SCC
    reachable from the SCC of ``s``.
    """
    succ: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _label, tgt in ss.transitions:
        succ[src].add(tgt)

    # Iterative reach over the raw graph; this is correct for acyclic
    # graphs and yields the *strongly connected* equivalence for cyclic
    # ones since each member of an SCC reaches every other member.
    reach: dict[int, set[int]] = {s: {s} for s in ss.states}
    changed = True
    while changed:
        changed = False
        for s in ss.states:
            before = len(reach[s])
            for t in succ[s]:
                if not reach[t].issubset(reach[s]):
                    reach[s] |= reach[t]
                    changed = True
    return reach


def _scc_rep(ss: "StateSpace") -> dict[int, int]:
    """Map each state to a canonical representative of its SCC.

    Two states ``a, b`` are in the same SCC iff each reaches the other.
    The representative is the minimum state ID in the SCC.
    """
    reach = _forward_reach(ss)
    rep: dict[int, int] = {}
    for s in ss.states:
        scc = {t for t in ss.states if t in reach[s] and s in reach[t]}
        rep[s] = min(scc)
    return rep


# ---------------------------------------------------------------------------
# Critical pairs
# ---------------------------------------------------------------------------

def find_critical_pairs(ss: "StateSpace") -> list[tuple[int, int, int]]:
    """Return every critical pair ``(origin, branch1, branch2)``.

    A critical pair is two distinct outgoing transitions from the same
    origin state. Targets are returned in sorted order so that the result
    is deterministic and each unordered pair appears exactly once.
    """
    out: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _label, tgt in ss.transitions:
        if src != tgt:
            out[src].add(tgt)

    pairs: list[tuple[int, int, int]] = []
    for origin in sorted(ss.states):
        targets = sorted(out[origin])
        n = len(targets)
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((origin, targets[i], targets[j]))
    return pairs


# ---------------------------------------------------------------------------
# Confluence closure (meet computation)
# ---------------------------------------------------------------------------

def confluence_closure(
    ss: "StateSpace", x: int, y: int
) -> int | None:
    """Return the closing element (meet) of ``x`` and ``y``, or None.

    The closing element is a state ``w`` such that ``x reachstar w`` and
    ``y reachstar w`` and every other such ``w'`` satisfies
    ``w reachstar w'``. In the lattice orientation this is the meet
    (greatest lower bound) ``x meet y``.
    """
    if x not in ss.states or y not in ss.states:
        return None
    reach = _forward_reach(ss)
    return _meet_from_reach(reach, x, y)


def _meet_from_reach(
    reach: dict[int, set[int]], x: int, y: int
) -> int | None:
    """Greatest lower bound of x and y given precomputed forward reach.

    Operates on the SCC quotient: states ``x, y`` belonging to the same
    SCC are treated as equal (their meet is themselves), and the search
    for a greatest element among lower bounds collapses each SCC to its
    representative before testing uniqueness.
    """
    if x == y:
        return x
    # SCC equivalence: same SCC iff each reaches the other.
    if y in reach[x] and x in reach[y]:
        return x
    common = reach[x] & reach[y]
    if not common:
        return None
    # Quotient *common* by SCC: keep one representative per SCC.
    scc_of: dict[int, frozenset[int]] = {}
    for c in common:
        scc_of[c] = frozenset(d for d in common if d in reach[c] and c in reach[d])
    reps = {min(s) for s in scc_of.values()}
    # The greatest lower bound is the unique representative whose
    # forward-reach contains all of *common*.
    candidates = [c for c in reps if common <= reach[c]]
    if len(candidates) == 1:
        return candidates[0]
    return None


# ---------------------------------------------------------------------------
# Local / global confluence
# ---------------------------------------------------------------------------

def is_locally_confluent(ss: "StateSpace") -> bool:
    """True iff every critical pair has a common lower bound."""
    reach = _forward_reach(ss)
    for _origin, b1, b2 in find_critical_pairs(ss):
        if not (reach[b1] & reach[b2]):
            return False
    return True


def is_globally_confluent(ss: "StateSpace") -> bool:
    """True iff every pair of states reachable from a common ancestor
    admits a common lower bound (meet).

    For finite session-type state spaces this coincides with the
    meet-semilattice property by the main theorem of Step 80m.
    """
    reach = _forward_reach(ss)
    states = sorted(ss.states)
    for i, x in enumerate(states):
        for y in states[i + 1:]:
            # Only require a meet for pairs that share a common ancestor;
            # in a session type state space all states share the top, so
            # this is universal -- but we keep the check explicit for the
            # general case (synthetic ARSs that may have multiple roots).
            common_anc_exists = any(
                x in reach[a] and y in reach[a] for a in states
            )
            if not common_anc_exists:
                continue
            if _meet_from_reach(reach, x, y) is None:
                return False
    return True


def check_confluence(ss: "StateSpace") -> ConfluenceResult:
    """Top-level confluence analysis. Returns a frozen result record."""
    reach = _forward_reach(ss)
    crit = find_critical_pairs(ss)

    unclosed: list[tuple[int, int, int]] = []
    for origin, b1, b2 in crit:
        if not (reach[b1] & reach[b2]):
            unclosed.append((origin, b1, b2))

    missing: list[tuple[int, int]] = []
    states = sorted(ss.states)
    for i, x in enumerate(states):
        for y in states[i + 1:]:
            common_anc = any(
                x in reach[a] and y in reach[a] for a in states
            )
            if not common_anc:
                continue
            if _meet_from_reach(reach, x, y) is None:
                missing.append((x, y))

    return ConfluenceResult(
        is_locally_confluent=(not unclosed),
        is_globally_confluent=(not missing),
        critical_pair_count=len(crit),
        missing_meets=tuple(missing),
        unclosed_critical_pairs=tuple(unclosed),
    )
