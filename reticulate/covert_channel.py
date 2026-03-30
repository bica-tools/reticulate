"""Covert channel detection via mutual information (Step 89e).

Detects covert channels in session type state spaces by measuring
mutual information between branches that should be informationally
independent.  A covert channel exists when the behavior of one
branch can be inferred by observing another branch.

Key insight: in parallel composition (S1 || S2), the two branches
should be informationally independent --- knowing the state of S1
should give no information about the state of S2.  When the product
state space has correlations (shared labels, synchronized transitions),
mutual information becomes non-zero, indicating a potential covert
channel.

This module provides:
    ``detect_covert_channels(ss)``
        — find non-zero mutual information between branches.
    ``channel_capacity(ss, branch1_labels, branch2_labels)``
        — bits of information leakable between branches.
    ``parallel_independence(ss)``
        — verify parallel branches are informationally independent.
    ``analyze_covert_channels(ss)``
        — full analysis as CovertChannelResult.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CovertChannel:
    """A detected covert channel between two label groups.

    Attributes:
        group1_labels: Labels in the first group.
        group2_labels: Labels in the second group.
        mutual_information: Mutual information in bits.
        shared_states: States where both groups have transitions.
        description: Human-readable description.
    """

    group1_labels: frozenset[str]
    group2_labels: frozenset[str]
    mutual_information: float
    shared_states: frozenset[int]
    description: str


@dataclass(frozen=True)
class IndependenceResult:
    """Result of parallel independence check.

    Attributes:
        is_independent: True if parallel branches are independent.
        shared_labels: Labels appearing in multiple branches.
        correlation_states: States where branches are correlated.
        independence_score: 1.0 = fully independent, 0.0 = fully correlated.
    """

    is_independent: bool
    shared_labels: frozenset[str]
    correlation_states: frozenset[int]
    independence_score: float


@dataclass(frozen=True)
class CovertChannelResult:
    """Full covert channel analysis result.

    Attributes:
        has_covert_channels: True if any covert channel is detected.
        channels: List of detected covert channels.
        total_capacity: Total channel capacity in bits.
        independence: Parallel independence check result.
        num_label_groups: Number of distinct label groups analyzed.
        max_mutual_info: Maximum mutual information across all pairs.
    """

    has_covert_channels: bool
    channels: list[CovertChannel]
    total_capacity: float
    independence: IndependenceResult
    num_label_groups: int
    max_mutual_info: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_log2(x: float) -> float:
    """log2(x) with convention log2(0) = 0."""
    if x <= 0:
        return 0.0
    return math.log2(x)


def _shannon_entropy(probs: list[float]) -> float:
    """Shannon entropy H = -sum(p * log2(p))."""
    return -sum(p * _safe_log2(p) for p in probs if p > 0)


def _label_groups(ss: StateSpace) -> list[set[str]]:
    """Partition labels into groups based on which states they appear at.

    Two labels are in the same group if they appear at overlapping
    sets of states.
    """
    label_states: dict[str, set[int]] = defaultdict(set)
    for src, label, _ in ss.transitions:
        label_states[label].add(src)

    if not label_states:
        return []

    # Simple grouping: each label is its own group initially,
    # merge groups with overlapping state sets
    labels = list(label_states.keys())
    groups: list[set[str]] = [{l} for l in labels]

    changed = True
    while changed:
        changed = False
        new_groups: list[set[str]] = []
        merged: set[int] = set()
        for i in range(len(groups)):
            if i in merged:
                continue
            current = set(groups[i])
            current_states: set[int] = set()
            for l in current:
                current_states |= label_states[l]

            for j in range(i + 1, len(groups)):
                if j in merged:
                    continue
                other_states: set[int] = set()
                for l in groups[j]:
                    other_states |= label_states[l]
                if current_states & other_states:
                    current |= groups[j]
                    current_states |= other_states
                    merged.add(j)
                    changed = True
            new_groups.append(current)
        groups = new_groups

    return groups


def _transition_distribution(
    ss: StateSpace, labels: set[str] | frozenset[str]
) -> list[float]:
    """Probability distribution of transitions with given labels.

    Returns the normalized frequency of each label among
    the specified set.
    """
    counts: Counter[str] = Counter()
    for _, label, _ in ss.transitions:
        if label in labels:
            counts[label] += 1

    total = sum(counts.values())
    if total == 0:
        return []

    return [count / total for count in counts.values()]


def _joint_distribution(
    ss: StateSpace,
    labels1: set[str] | frozenset[str],
    labels2: set[str] | frozenset[str],
) -> dict[tuple[str, str], float]:
    """Joint distribution over label pairs at shared states.

    For states where both label groups have transitions, computes
    the joint probability of seeing label l1 from group1 and
    label l2 from group2.
    """
    # Find states with transitions from both groups
    states_with_g1: dict[int, set[str]] = defaultdict(set)
    states_with_g2: dict[int, set[str]] = defaultdict(set)

    for src, label, _ in ss.transitions:
        if label in labels1:
            states_with_g1[src].add(label)
        if label in labels2:
            states_with_g2[src].add(label)

    shared_states = set(states_with_g1.keys()) & set(states_with_g2.keys())
    if not shared_states:
        return {}

    # Count joint occurrences
    joint_counts: Counter[tuple[str, str]] = Counter()
    for state in shared_states:
        for l1 in states_with_g1[state]:
            for l2 in states_with_g2[state]:
                joint_counts[(l1, l2)] += 1

    total = sum(joint_counts.values())
    if total == 0:
        return {}

    return {pair: count / total for pair, count in joint_counts.items()}


def _mutual_information(
    ss: StateSpace,
    labels1: set[str] | frozenset[str],
    labels2: set[str] | frozenset[str],
) -> float:
    """Mutual information I(X;Y) between two label groups.

    I(X;Y) = H(X) + H(Y) - H(X,Y)
    where X = labels1, Y = labels2.

    Returns 0.0 if either group is empty or they share no states.
    """
    joint = _joint_distribution(ss, labels1, labels2)
    if not joint:
        return 0.0

    # Marginals
    marginal_x: Counter[str] = Counter()
    marginal_y: Counter[str] = Counter()
    for (l1, l2), p in joint.items():
        marginal_x[l1] += p
        marginal_y[l2] += p

    h_x = _shannon_entropy(list(marginal_x.values()))
    h_y = _shannon_entropy(list(marginal_y.values()))
    h_xy = _shannon_entropy(list(joint.values()))

    mi = h_x + h_y - h_xy
    return max(0.0, mi)  # Clamp to non-negative (numerical errors)


def _shared_states(
    ss: StateSpace,
    labels1: set[str] | frozenset[str],
    labels2: set[str] | frozenset[str],
) -> frozenset[int]:
    """States where both label groups have transitions."""
    states1: set[int] = set()
    states2: set[int] = set()
    for src, label, _ in ss.transitions:
        if label in labels1:
            states1.add(src)
        if label in labels2:
            states2.add(src)
    return frozenset(states1 & states2)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def detect_covert_channels(
    ss: StateSpace,
    *,
    label_groups: list[set[str]] | None = None,
    threshold: float = 1e-10,
) -> list[CovertChannel]:
    """Detect covert channels by finding non-zero mutual information.

    Examines all pairs of label groups and reports those with
    mutual information above the threshold.

    Args:
        ss: The state space to analyze.
        label_groups: Optional pre-defined label groups.
                      If None, groups are inferred from the state space.
        threshold: Minimum mutual information to report (default: 1e-10).

    Returns:
        List of detected CovertChannel instances.
    """
    if label_groups is None:
        groups = _label_groups(ss)
    else:
        groups = [set(g) for g in label_groups]

    if len(groups) < 2:
        return []

    channels: list[CovertChannel] = []

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1 = frozenset(groups[i])
            g2 = frozenset(groups[j])
            mi = _mutual_information(ss, g1, g2)
            if mi > threshold:
                shared = _shared_states(ss, g1, g2)
                channels.append(CovertChannel(
                    group1_labels=g1,
                    group2_labels=g2,
                    mutual_information=mi,
                    shared_states=shared,
                    description=(
                        f"Covert channel: {sorted(g1)} <-> {sorted(g2)}, "
                        f"MI = {mi:.4f} bits at states {sorted(shared)}"
                    ),
                ))

    return channels


def channel_capacity(
    ss: StateSpace,
    branch1_labels: set[str] | frozenset[str],
    branch2_labels: set[str] | frozenset[str],
) -> float:
    """Compute the channel capacity between two label groups.

    Channel capacity is the mutual information between the two
    groups, representing the maximum bits of information that
    can be transmitted through the covert channel.

    Args:
        ss: The state space.
        branch1_labels: Labels in the first branch.
        branch2_labels: Labels in the second branch.

    Returns:
        Channel capacity in bits (0.0 = no covert channel).
    """
    return _mutual_information(ss, branch1_labels, branch2_labels)


def parallel_independence(ss: StateSpace) -> IndependenceResult:
    """Verify that parallel branches are informationally independent.

    Checks whether the state space has product structure where
    the components are independent (no shared labels, no
    correlation between branches).

    For non-product state spaces, checks for label sharing
    that could create dependencies.

    Args:
        ss: The state space to check.

    Returns:
        IndependenceResult with independence assessment.
    """
    # Check if this is a product state space
    if ss.product_coords is not None and ss.product_factors is not None:
        return _check_product_independence(ss)

    # For non-product spaces, check for label overlap between groups
    groups = _label_groups(ss)

    if len(groups) <= 1:
        # Single group or empty -> trivially independent
        return IndependenceResult(
            is_independent=True,
            shared_labels=frozenset(),
            correlation_states=frozenset(),
            independence_score=1.0,
        )

    # Check all pairs for shared labels and mutual information
    all_shared: set[str] = set()
    all_corr_states: set[int] = set()
    total_mi = 0.0
    pair_count = 0

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            overlap = groups[i] & groups[j]
            all_shared |= overlap
            shared = _shared_states(ss, groups[i], groups[j])
            all_corr_states |= shared
            mi = _mutual_information(ss, groups[i], groups[j])
            total_mi += mi
            pair_count += 1

    avg_mi = total_mi / pair_count if pair_count > 0 else 0.0

    # Independence score: 1 - normalized MI
    max_possible_mi = max(_safe_log2(len(g)) for g in groups) if groups else 0.0
    if max_possible_mi > 0:
        score = max(0.0, 1.0 - avg_mi / max_possible_mi)
    else:
        score = 1.0

    is_indep = len(all_shared) == 0 and total_mi < 1e-10

    return IndependenceResult(
        is_independent=is_indep,
        shared_labels=frozenset(all_shared),
        correlation_states=frozenset(all_corr_states),
        independence_score=score,
    )


def _check_product_independence(ss: StateSpace) -> IndependenceResult:
    """Check independence for a product state space."""
    assert ss.product_factors is not None

    # Collect labels from each factor
    factor_labels: list[set[str]] = []
    for factor in ss.product_factors:
        labels: set[str] = set()
        for _, label, _ in factor.transitions:
            labels.add(label)
        factor_labels.append(labels)

    # Check for shared labels between factors
    shared: set[str] = set()
    for i in range(len(factor_labels)):
        for j in range(i + 1, len(factor_labels)):
            shared |= factor_labels[i] & factor_labels[j]

    # Product with disjoint labels = fully independent
    is_indep = len(shared) == 0

    return IndependenceResult(
        is_independent=is_indep,
        shared_labels=frozenset(shared),
        correlation_states=frozenset(),
        independence_score=1.0 if is_indep else 0.5,
    )


def analyze_covert_channels(
    ss: StateSpace,
    *,
    label_groups: list[set[str]] | None = None,
) -> CovertChannelResult:
    """Full covert channel analysis.

    Detects covert channels, measures their capacity, and checks
    parallel independence.

    Args:
        ss: The state space to analyze.
        label_groups: Optional label groups to analyze.

    Returns:
        CovertChannelResult with comprehensive analysis.
    """
    channels = detect_covert_channels(ss, label_groups=label_groups)
    independence = parallel_independence(ss)

    total_cap = sum(ch.mutual_information for ch in channels)
    max_mi = max((ch.mutual_information for ch in channels), default=0.0)

    if label_groups is not None:
        num_groups = len(label_groups)
    else:
        num_groups = len(_label_groups(ss))

    return CovertChannelResult(
        has_covert_channels=len(channels) > 0,
        channels=channels,
        total_capacity=total_cap,
        independence=independence,
        num_label_groups=num_groups,
        max_mutual_info=max_mi,
    )
