"""Distributive quotients of session type lattices (Step 301).

Session type: rec X . &{analyze: +{result: X}, quotient: +{result: X}, quit: end}

Given a session type whose state space forms a lattice L, compute the
smallest congruence Theta such that L/Theta is distributive.  Key results:

- M3 (diamond) lattices are simple: no non-trivial distributive quotient exists.
- N5 (pentagon) lattices from branch nesting have a unique minimal quotient.
- Product lattices from parallel composition are always distributive (no quotient needed).
- Entropy loss = log2(states_before / states_after).

Algorithm overview:
  1. Check if already distributive -> identity (no quotient needed).
  2. Check if M3-simple -> return impossible.
  3. For small lattices (<=12 states): exhaustive congruence enumeration.
  4. For larger: iterative N5 collapsing heuristic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

from reticulate.lattice import (
    check_distributive,
    check_lattice,
    compute_meet,
    compute_join,
)

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CongruenceClass:
    """An equivalence class in a lattice congruence."""
    representative: int          # min state ID in class
    members: frozenset[int]      # all state IDs in class
    labels: frozenset[str]       # transition labels within class


@dataclass(frozen=True)
class DistributiveQuotientResult:
    """Result of computing the distributive quotient."""
    original_states: int
    quotient_states: int
    is_already_distributive: bool
    is_quotient_possible: bool   # False for M3 (simple lattice)
    congruence_classes: list[CongruenceClass]
    merged_pairs: list[tuple[int, int]]  # pairs of states that were merged
    entropy_before: float        # log2(original_states)
    entropy_after: float         # log2(quotient_states)
    entropy_lost: float
    quotient_statespace: "StateSpace | None"  # the quotient, or None if impossible
    num_minimal_quotients: int   # 1=unique, >1=non-unique, 0=impossible
    explanation: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_distributive_quotient(ss: "StateSpace") -> DistributiveQuotientResult:
    """Compute the minimal distributive quotient of a lattice.

    Algorithm:
    1. Check if already distributive -> return identity
    2. Check if M3-simple -> return impossible
    3. For small lattices (<=12 states): exhaustive congruence enumeration
    4. For larger: iterative N5 collapsing heuristic
    """
    from reticulate.statespace import StateSpace

    lr = check_lattice(ss)
    if not lr.is_lattice:
        return DistributiveQuotientResult(
            original_states=len(ss.states),
            quotient_states=len(ss.states),
            is_already_distributive=False,
            is_quotient_possible=False,
            congruence_classes=[],
            merged_pairs=[],
            entropy_before=_log2_safe(len(ss.states)),
            entropy_after=_log2_safe(len(ss.states)),
            entropy_lost=0.0,
            quotient_statespace=None,
            num_minimal_quotients=0,
            explanation="Not a lattice; distributive quotient undefined.",
        )

    dr = check_distributive(ss)
    if dr.is_distributive:
        # Already distributive: identity quotient
        partition = [frozenset({s}) for s in sorted(ss.states)]
        classes = _partition_to_classes(ss, partition)
        return DistributiveQuotientResult(
            original_states=len(ss.states),
            quotient_states=len(ss.states),
            is_already_distributive=True,
            is_quotient_possible=True,
            congruence_classes=classes,
            merged_pairs=[],
            entropy_before=_log2_safe(len(ss.states)),
            entropy_after=_log2_safe(len(ss.states)),
            entropy_lost=0.0,
            quotient_statespace=ss,
            num_minimal_quotients=1,
            explanation="Already distributive; no quotient needed.",
        )

    # Non-distributive lattice: try to find a distributive quotient.
    # Check if M3-simple (has M3 but no N5 — modular non-distributive).
    # M3 is a simple lattice: its only congruences are trivial (identity and
    # the all-equal congruence). Neither yields a non-trivial distributive quotient.
    if dr.has_m3 and not dr.has_n5:
        # Modular non-distributive (contains M3 but no N5).
        # M3 is simple, so for small lattices that are essentially M3,
        # no non-trivial distributive quotient exists.
        # For larger lattices, we still attempt the exhaustive search.
        n_states = len(ss.states)
        if n_states <= 5:
            # Pure M3: definitely simple, no quotient possible
            return DistributiveQuotientResult(
                original_states=n_states,
                quotient_states=n_states,
                is_already_distributive=False,
                is_quotient_possible=False,
                congruence_classes=[],
                merged_pairs=[],
                entropy_before=_log2_safe(n_states),
                entropy_after=_log2_safe(n_states),
                entropy_lost=0.0,
                quotient_statespace=None,
                num_minimal_quotients=0,
                explanation="M3 (diamond) lattice is simple; no non-trivial distributive quotient exists.",
            )

    # Attempt to find a distributive quotient
    n_states = len(ss.states)

    if n_states <= 12:
        # Exhaustive congruence enumeration for small lattices
        result = _exhaustive_distributive_quotient(ss)
        if result is not None:
            return result
        # No distributive quotient found
        return DistributiveQuotientResult(
            original_states=n_states,
            quotient_states=n_states,
            is_already_distributive=False,
            is_quotient_possible=False,
            congruence_classes=[],
            merged_pairs=[],
            entropy_before=_log2_safe(n_states),
            entropy_after=_log2_safe(n_states),
            entropy_lost=0.0,
            quotient_statespace=None,
            num_minimal_quotients=0,
            explanation="No non-trivial distributive quotient exists (exhaustive search).",
        )

    # For larger lattices: iterative N5 collapsing heuristic
    result = _iterative_n5_collapse(ss)
    return result


def is_congruence(ss: "StateSpace", partition: list[frozenset[int]]) -> bool:
    """Check if a partition is a valid lattice congruence.

    A partition Theta is a congruence on lattice L iff for all (a, b) in Theta:
      - meet(a, x) Theta meet(b, x) for all x in L
      - join(a, x) Theta join(b, x) for all x in L

    This is the substitution property for meet and join.
    """
    # Build lookup: state -> class index
    class_of: dict[int, int] = {}
    for idx, block in enumerate(partition):
        for s in block:
            class_of[s] = idx

    # Check all states are covered
    if set(class_of.keys()) != ss.states:
        return False

    states_list = sorted(ss.states)

    for block in partition:
        if len(block) <= 1:
            continue
        members = sorted(block)
        # For each pair in the same class, check substitution property
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                for x in states_list:
                    # Check meet substitution
                    ma = compute_meet(ss, a, x)
                    mb = compute_meet(ss, b, x)
                    if ma is None or mb is None:
                        # If meets don't exist, not a lattice issue
                        continue
                    if class_of.get(ma) != class_of.get(mb):
                        return False

                    # Check join substitution
                    ja = compute_join(ss, a, x)
                    jb = compute_join(ss, b, x)
                    if ja is None or jb is None:
                        continue
                    if class_of.get(ja) != class_of.get(jb):
                        return False
    return True


def build_quotient_statespace(
    ss: "StateSpace",
    partition: list[frozenset[int]],
) -> "StateSpace":
    """Build quotient state space from a congruence partition.

    Each congruence class becomes a single state. Transitions are induced
    from the original: if (a, label, b) is a transition and a is in class C1,
    b is in class C2, then (C1_rep, label, C2_rep) is a quotient transition.
    Duplicate transitions are removed.
    """
    from reticulate.statespace import StateSpace

    # Map each state to its class representative (min element)
    state_to_rep: dict[int, int] = {}
    for block in partition:
        rep = min(block)
        for s in block:
            state_to_rep[s] = rep

    new_states: set[int] = {min(block) for block in partition}
    new_top = state_to_rep[ss.top]
    new_bottom = state_to_rep[ss.bottom]

    # Build transitions (deduplicated)
    seen_transitions: set[tuple[int, str, int]] = set()
    new_transitions: list[tuple[int, str, int]] = []
    new_selections: set[tuple[int, str, int]] = set()

    for src, lbl, tgt in ss.transitions:
        new_src = state_to_rep[src]
        new_tgt = state_to_rep[tgt]
        tr = (new_src, lbl, new_tgt)
        if tr not in seen_transitions:
            seen_transitions.add(tr)
            new_transitions.append(tr)
            if (src, lbl, tgt) in ss.selection_transitions:
                new_selections.add(tr)

    # Build labels
    new_labels: dict[int, str] = {}
    for block in partition:
        rep = min(block)
        if len(block) == 1:
            new_labels[rep] = ss.labels.get(rep, str(rep))
        else:
            new_labels[rep] = "{" + ",".join(
                ss.labels.get(s, str(s)) for s in sorted(block)
            ) + "}"

    return StateSpace(
        states=new_states,
        transitions=new_transitions,
        top=new_top,
        bottom=new_bottom,
        labels=new_labels,
        selection_transitions=new_selections,
    )


def count_minimal_quotients(ss: "StateSpace") -> int:
    """Count the number of minimal distributive quotients (for small lattices).

    Returns:
        0 if no distributive quotient exists (e.g., M3-simple)
        1 if a unique minimal distributive quotient exists
        >1 if multiple minimal distributive quotients exist
    """
    lr = check_lattice(ss)
    if not lr.is_lattice:
        return 0

    dr = check_distributive(ss)
    if dr.is_distributive:
        return 1  # Identity is the unique quotient

    n_states = len(ss.states)
    if n_states > 12:
        # Too large for exhaustive enumeration; return heuristic answer
        result = compute_distributive_quotient(ss)
        return result.num_minimal_quotients

    # Exhaustive: find all congruences that yield distributive quotients
    dist_congruences = _find_all_distributive_congruences(ss)

    if not dist_congruences:
        return 0

    # Find minimal ones (fewest merges / most classes)
    max_classes = max(len(p) for p in dist_congruences)
    minimal = [p for p in dist_congruences if len(p) == max_classes]
    return len(minimal)


def direct_distributivity_check(ss: "StateSpace") -> bool:
    """Direct check: a ^ (b v c) = (a ^ b) v (a ^ c) for all triples.

    This is the ground-truth distributivity check (slower but correct).
    Uses compute_meet and compute_join directly on all state triples.
    """
    lr = check_lattice(ss)
    if not lr.is_lattice:
        return False

    states_list = sorted(ss.states)
    for a in states_list:
        for b in states_list:
            for c in states_list:
                # Check a ^ (b v c) = (a ^ b) v (a ^ c)
                b_join_c = compute_join(ss, b, c)
                if b_join_c is None:
                    return False
                lhs = compute_meet(ss, a, b_join_c)
                if lhs is None:
                    return False

                a_meet_b = compute_meet(ss, a, b)
                a_meet_c = compute_meet(ss, a, c)
                if a_meet_b is None or a_meet_c is None:
                    return False
                rhs = compute_join(ss, a_meet_b, a_meet_c)
                if rhs is None:
                    return False

                if lhs != rhs:
                    return False
    return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log2_safe(n: int) -> float:
    """Compute log2(n), returning 0.0 for n <= 0."""
    if n <= 0:
        return 0.0
    return math.log2(n)


def _partition_to_classes(
    ss: "StateSpace",
    partition: list[frozenset[int]],
) -> list[CongruenceClass]:
    """Convert a partition to a list of CongruenceClass objects."""
    # Build transition label sets within each class
    classes: list[CongruenceClass] = []
    for block in partition:
        rep = min(block)
        labels: set[str] = set()
        for src, lbl, tgt in ss.transitions:
            if src in block and tgt in block:
                labels.add(lbl)
        classes.append(CongruenceClass(
            representative=rep,
            members=block,
            labels=frozenset(labels),
        ))
    return classes


def _merged_pairs_from_partition(
    partition: list[frozenset[int]],
) -> list[tuple[int, int]]:
    """Extract merged pairs from a partition (pairs within multi-element blocks)."""
    pairs: list[tuple[int, int]] = []
    for block in partition:
        if len(block) <= 1:
            continue
        members = sorted(block)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.append((members[i], members[j]))
    return pairs


def _generate_partitions(
    elements: list[int],
) -> list[list[frozenset[int]]]:
    """Generate all partitions of a set of elements.

    Uses a recursive algorithm. Only feasible for small sets (<=12 elements).
    Returns a list of partitions, where each partition is a list of frozensets.
    """
    if not elements:
        return [[]]
    if len(elements) == 1:
        return [[frozenset(elements)]]

    first = elements[0]
    rest = elements[1:]
    rest_partitions = _generate_partitions(rest)

    result: list[list[frozenset[int]]] = []
    for part in rest_partitions:
        # Option 1: first goes into its own singleton block
        result.append([frozenset({first})] + part)
        # Option 2: first is added to each existing block
        for i in range(len(part)):
            new_part = list(part)
            new_part[i] = part[i] | {first}
            result.append(new_part)
    return result


def _find_all_distributive_congruences(
    ss: "StateSpace",
) -> list[list[frozenset[int]]]:
    """Find all congruences on ss that yield distributive quotients.

    Only feasible for small state spaces (<=12 states).
    """
    from reticulate.statespace import StateSpace

    states_list = sorted(ss.states)
    all_partitions = _generate_partitions(states_list)

    dist_congruences: list[list[frozenset[int]]] = []
    for partition in all_partitions:
        # Skip the trivial all-in-one partition (unless it's the only option)
        if len(partition) == 1 and len(states_list) > 1:
            continue
        # Skip the identity partition (already checked)
        if all(len(block) == 1 for block in partition):
            continue
        if not is_congruence(ss, partition):
            continue
        # Build quotient and check distributivity
        quotient = build_quotient_statespace(ss, partition)
        if direct_distributivity_check(quotient):
            dist_congruences.append(partition)

    return dist_congruences


def _exhaustive_distributive_quotient(
    ss: "StateSpace",
) -> DistributiveQuotientResult | None:
    """Exhaustive search for the minimal distributive quotient.

    Tries all non-trivial congruences, finds those yielding distributive
    quotients, and returns the one with the most classes (minimal merging).

    Returns None if no distributive quotient exists.
    """
    dist_congruences = _find_all_distributive_congruences(ss)

    if not dist_congruences:
        return None

    # Find minimal: the one with the most quotient classes (fewest merges)
    max_classes = max(len(p) for p in dist_congruences)
    minimal = [p for p in dist_congruences if len(p) == max_classes]
    best = minimal[0]  # Pick first (deterministic: partitions are ordered)

    quotient = build_quotient_statespace(ss, best)
    classes = _partition_to_classes(ss, best)
    merged = _merged_pairs_from_partition(best)
    n_orig = len(ss.states)
    n_quot = len(quotient.states)

    return DistributiveQuotientResult(
        original_states=n_orig,
        quotient_states=n_quot,
        is_already_distributive=False,
        is_quotient_possible=True,
        congruence_classes=classes,
        merged_pairs=merged,
        entropy_before=_log2_safe(n_orig),
        entropy_after=_log2_safe(n_quot),
        entropy_lost=_log2_safe(n_orig) - _log2_safe(n_quot),
        quotient_statespace=quotient,
        num_minimal_quotients=len(minimal),
        explanation=f"Found {len(dist_congruences)} distributive congruence(s); "
                    f"{len(minimal)} minimal (with {max_classes} classes).",
    )


def _iterative_n5_collapse(ss: "StateSpace") -> DistributiveQuotientResult:
    """Iterative N5 collapsing heuristic for larger lattices.

    Strategy: repeatedly find N5 witnesses and merge the two incomparable
    elements that form the 'arms' of the pentagon. Repeat until the quotient
    is distributive or no further progress can be made.
    """
    from reticulate.statespace import StateSpace

    n_orig = len(ss.states)
    current_ss = ss
    all_merged: list[tuple[int, int]] = []
    # Track the cumulative partition
    partition: dict[int, set[int]] = {s: {s} for s in ss.states}

    for _ in range(n_orig):  # bounded iterations
        dr = check_distributive(current_ss)
        if dr.is_distributive:
            break
        if not dr.is_lattice:
            # Merging broke the lattice structure
            return DistributiveQuotientResult(
                original_states=n_orig,
                quotient_states=n_orig,
                is_already_distributive=False,
                is_quotient_possible=False,
                congruence_classes=[],
                merged_pairs=[],
                entropy_before=_log2_safe(n_orig),
                entropy_after=_log2_safe(n_orig),
                entropy_lost=0.0,
                quotient_statespace=None,
                num_minimal_quotients=0,
                explanation="N5 collapse broke lattice structure; no quotient possible.",
            )

        # Get N5 or M3 witness
        if dr.n5_witness is not None:
            # N5: (top, a, b, c, bot) — merge b and c (the incomparable pair
            # at the bottom of the pentagon arms)
            _, a, b, c, _ = dr.n5_witness
            merge_a, merge_b = b, c
        elif dr.m3_witness is not None:
            # M3: (top, a, b, c, bot) — try merging first two incomparables
            _, a, b, c, _ = dr.m3_witness
            merge_a, merge_b = a, b
        else:
            # No witness found but not distributive?
            break

        # Build a partition that merges these two
        states_list = sorted(current_ss.states)
        new_partition: list[frozenset[int]] = []
        merged_block: set[int] = set()
        used: set[int] = set()

        for s in states_list:
            if s in used:
                continue
            if s == merge_a or s == merge_b:
                merged_block.add(merge_a)
                merged_block.add(merge_b)
                used.add(merge_a)
                used.add(merge_b)
            else:
                new_partition.append(frozenset({s}))
                used.add(s)

        if merged_block:
            new_partition.append(frozenset(merged_block))

        # Check if this is a valid congruence
        if not is_congruence(current_ss, new_partition):
            # Try generating the congruence closure
            new_partition = _congruence_closure(current_ss, merge_a, merge_b)
            if new_partition is None or not is_congruence(current_ss, new_partition):
                return DistributiveQuotientResult(
                    original_states=n_orig,
                    quotient_states=n_orig,
                    is_already_distributive=False,
                    is_quotient_possible=False,
                    congruence_classes=[],
                    merged_pairs=[],
                    entropy_before=_log2_safe(n_orig),
                    entropy_after=_log2_safe(n_orig),
                    entropy_lost=0.0,
                    quotient_statespace=None,
                    num_minimal_quotients=0,
                    explanation="Cannot form a valid congruence by N5 collapse.",
                )

        all_merged.append((merge_a, merge_b))
        current_ss = build_quotient_statespace(current_ss, new_partition)

    # Check final result
    final_dr = check_distributive(current_ss)
    if final_dr.is_distributive:
        # Reconstruct the full partition relative to the original
        final_partition = _reconstruct_partition(ss, all_merged)
        classes = _partition_to_classes(ss, final_partition)
        n_quot = len(current_ss.states)
        return DistributiveQuotientResult(
            original_states=n_orig,
            quotient_states=n_quot,
            is_already_distributive=False,
            is_quotient_possible=True,
            congruence_classes=classes,
            merged_pairs=all_merged,
            entropy_before=_log2_safe(n_orig),
            entropy_after=_log2_safe(n_quot),
            entropy_lost=_log2_safe(n_orig) - _log2_safe(n_quot),
            quotient_statespace=current_ss,
            num_minimal_quotients=1,  # Heuristic: assume unique
            explanation=f"Iterative N5 collapse: {len(all_merged)} merge(s).",
        )

    return DistributiveQuotientResult(
        original_states=n_orig,
        quotient_states=n_orig,
        is_already_distributive=False,
        is_quotient_possible=False,
        congruence_classes=[],
        merged_pairs=[],
        entropy_before=_log2_safe(n_orig),
        entropy_after=_log2_safe(n_orig),
        entropy_lost=0.0,
        quotient_statespace=None,
        num_minimal_quotients=0,
        explanation="Iterative N5 collapse failed to produce a distributive quotient.",
    )


def _congruence_closure(
    ss: "StateSpace",
    a: int,
    b: int,
) -> list[frozenset[int]] | None:
    """Compute the smallest congruence containing the pair (a, b).

    Uses iterative closure: start with a ~ b, then propagate through
    meet and join until stable.
    """
    # Union-Find for tracking equivalence classes
    parent: dict[int, int] = {s: s for s in ss.states}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> bool:
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        parent[max(rx, ry)] = min(rx, ry)
        return True

    # Start with a ~ b
    union(a, b)
    states_list = sorted(ss.states)

    # Iterate until stable
    changed = True
    max_iters = len(ss.states) * len(ss.states)
    iters = 0
    while changed and iters < max_iters:
        changed = False
        iters += 1
        for s1 in states_list:
            for s2 in states_list:
                if find(s1) != find(s2):
                    continue
                if s1 >= s2:
                    continue
                # s1 ~ s2: propagate through meet and join with all x
                for x in states_list:
                    m1 = compute_meet(ss, s1, x)
                    m2 = compute_meet(ss, s2, x)
                    if m1 is not None and m2 is not None and find(m1) != find(m2):
                        union(m1, m2)
                        changed = True

                    j1 = compute_join(ss, s1, x)
                    j2 = compute_join(ss, s2, x)
                    if j1 is not None and j2 is not None and find(j1) != find(j2):
                        union(j1, j2)
                        changed = True

    # Check if everything collapsed to one class
    roots = {find(s) for s in ss.states}
    if len(roots) == 1:
        return None  # Trivial congruence (all states merged)

    # Build partition
    groups: dict[int, set[int]] = {}
    for s in ss.states:
        r = find(s)
        groups.setdefault(r, set()).add(s)

    return [frozenset(g) for g in groups.values()]


def _reconstruct_partition(
    ss: "StateSpace",
    merged_pairs: list[tuple[int, int]],
) -> list[frozenset[int]]:
    """Reconstruct the full partition from a list of merged pairs."""
    parent: dict[int, int] = {s: s for s in ss.states}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    for a, b in merged_pairs:
        if a in parent and b in parent:
            union(a, b)

    groups: dict[int, set[int]] = {}
    for s in ss.states:
        r = find(s)
        groups.setdefault(r, set()).add(s)

    return [frozenset(g) for g in groups.values()]
