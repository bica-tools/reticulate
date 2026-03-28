"""Beauty as lattice symmetry in session types (Step 202).

Aesthetic judgment detects symmetry, balance, and order in session type
state spaces.  Beautiful protocols are symmetric, pendular, and have
clean topology.  The key quantities are:

- **Symmetry**: from orbital automorphism analysis — how symmetric the
  state space is under approximate graph automorphism.
- **Balance**: from pendular alternation — how evenly distributed
  select and branch states are (closeness of select_ratio to 0.5).
- **Complexity**: from information-theoretic branching entropy — how
  much choice the protocol encodes.
- **Order**: from graded structure — fraction of transitions that
  go from depth d to depth d+1.
- **Harmony**: average of symmetry, balance, and order.
- **Beauty score**: harmony weighted by complexity — beautiful things
  are harmonious AND non-trivial.

Classification hierarchy:
    - **Sublime** (beauty > 0.8): near-perfect symmetry, balance, and order.
    - **Beautiful** (beauty > 0.6): strong harmony with some complexity.
    - **Pleasant** (beauty > 0.4): moderate harmony.
    - **Neutral** (beauty > 0.2): minimal aesthetic interest.
    - **Ugly** (beauty ≤ 0.2): chaotic, unbalanced, disordered.

This module provides:
    ``analyze_aesthetics(ss)``          — full aesthetic profile.
    ``beauty_score(ss)``                — quick beauty score.
    ``compare_beauty(ss1, ss2)``        — compare two state spaces.
    ``most_beautiful(entries)``         — pick the most beautiful entry.
    ``golden_ratio_check(ss)``          — check for golden ratio in levels.
    ``classify_aesthetic(ss)``          — return classification string.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from reticulate.information import branching_entropy
from reticulate.orbital import compute_orbits
from reticulate.pendular import classify_alternation
from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AestheticProfile:
    """Result of aesthetic analysis of a session type state space.

    Attributes:
        symmetry: Symmetry score from orbital analysis (0.0 to 1.0).
        balance: Balance score from pendular select_ratio (0.0 to 1.0).
        complexity: Branching entropy (non-negative float).
        order: Order score from graded structure (0.0 to 1.0).
        harmony: Average of symmetry, balance, and order.
        beauty_score: harmony * (1 + complexity / 10).
        classification: One of 'sublime', 'beautiful', 'pleasant', 'neutral', 'ugly'.
    """

    symmetry: float
    balance: float
    complexity: float
    order: float
    harmony: float
    beauty_score: float
    classification: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_depths(ss: StateSpace) -> dict[int, int]:
    """BFS from top to compute shortest-path depth of each state."""
    depths: dict[int, int] = {ss.top: 0}
    queue: deque[int] = deque([ss.top])
    while queue:
        s = queue.popleft()
        d = depths[s]
        for src, _label, tgt in ss.transitions:
            if src == s and tgt not in depths:
                depths[tgt] = d + 1
                queue.append(tgt)
    return depths


def _compute_symmetry(ss: StateSpace) -> float:
    """Compute symmetry score from orbital analysis.

    Normalizes symmetry_ratio to [0, 1] by dividing by the number of
    states (maximum possible symmetry_ratio when all states are in one orbit).
    Returns 0.0 for empty state spaces.
    """
    if not ss.states or len(ss.states) <= 1:
        return 0.0
    orbits = compute_orbits(ss)
    # symmetry_ratio = |states| / |orbits|, max = |states| (one orbit)
    max_ratio = float(len(ss.states))
    if max_ratio <= 0:
        return 0.0
    return min(1.0, round(orbits.symmetry_ratio / max_ratio, 6))


def _compute_balance(ss: StateSpace) -> float:
    """Compute balance from pendular select_ratio closeness to 0.5.

    A perfectly balanced protocol has select_ratio = 0.5.
    Score = 1 - 2 * |select_ratio - 0.5|, clamped to [0, 1].
    """
    result = classify_alternation(ss)
    deviation = abs(result.select_ratio - 0.5)
    return round(max(0.0, 1.0 - 2.0 * deviation), 6)


def _compute_order(ss: StateSpace) -> float:
    """Compute order score from graded structure.

    A state space is fully ordered (1.0) if every transition goes from
    depth d to depth d+1.  Otherwise, the score is the fraction of
    transitions that satisfy this condition.
    """
    if not ss.transitions:
        return 1.0

    depths = _compute_depths(ss)
    total = 0
    graded = 0
    for src, _label, tgt in ss.transitions:
        if src in depths and tgt in depths:
            total += 1
            if depths[tgt] == depths[src] + 1:
                graded += 1

    if total == 0:
        return 1.0
    return round(graded / total, 6)


def _classify(beauty: float) -> str:
    """Classify a beauty score into an aesthetic category."""
    if beauty > 0.8:
        return "sublime"
    if beauty > 0.6:
        return "beautiful"
    if beauty > 0.4:
        return "pleasant"
    if beauty > 0.2:
        return "neutral"
    return "ugly"


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def analyze_aesthetics(ss: StateSpace) -> AestheticProfile:
    """Compute the full aesthetic profile of a state space.

    Combines symmetry (from orbital analysis), balance (from pendular
    classification), complexity (from information-theoretic entropy),
    and order (from graded structure) into a unified aesthetic judgment.

    Args:
        ss: The state space to analyze.

    Returns:
        AestheticProfile with all metrics and classification.
    """
    symmetry = _compute_symmetry(ss)
    balance = _compute_balance(ss)
    complexity = branching_entropy(ss)
    order = _compute_order(ss)

    harmony = round((symmetry + balance + order) / 3.0, 6)
    beauty = round(harmony * (1.0 + complexity / 10.0), 6)
    classification = _classify(beauty)

    return AestheticProfile(
        symmetry=symmetry,
        balance=balance,
        complexity=complexity,
        order=order,
        harmony=harmony,
        beauty_score=beauty,
        classification=classification,
    )


def beauty_score(ss: StateSpace) -> float:
    """Compute a quick beauty score for a state space.

    This is a convenience function that returns just the numeric score
    without the full profile.
    """
    return analyze_aesthetics(ss).beauty_score


def compare_beauty(ss1: StateSpace, ss2: StateSpace) -> int:
    """Compare the beauty of two state spaces.

    Returns:
        -1 if ss1 is less beautiful than ss2.
         0 if they are equally beautiful.
         1 if ss1 is more beautiful than ss2.
    """
    b1 = beauty_score(ss1)
    b2 = beauty_score(ss2)
    if b1 < b2:
        return -1
    if b1 > b2:
        return 1
    return 0


def most_beautiful(entries: dict[str, StateSpace]) -> str:
    """Return the name of the most beautiful state space.

    Args:
        entries: Dict mapping names to state spaces.

    Returns:
        Name of the entry with the highest beauty score.

    Raises:
        ValueError: If entries is empty.
    """
    if not entries:
        raise ValueError("entries must not be empty")
    return max(entries, key=lambda name: beauty_score(entries[name]))


def golden_ratio_check(ss: StateSpace) -> bool:
    """Check if consecutive depth levels approximate the golden ratio.

    Examines the number of states at each BFS depth level and checks
    whether any pair of consecutive levels has a ratio within 0.2 of
    the golden ratio phi = (1 + sqrt(5)) / 2 ≈ 1.618.

    Args:
        ss: The state space to check.

    Returns:
        True if any consecutive level pair approximates the golden ratio.
    """
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    depths = _compute_depths(ss)
    if not depths:
        return False

    max_depth = max(depths.values())
    level_sizes: list[int] = []
    for d in range(max_depth + 1):
        count = sum(1 for s in depths if depths[s] == d)
        level_sizes.append(count)

    for i in range(len(level_sizes) - 1):
        a = level_sizes[i]
        b = level_sizes[i + 1]
        if a == 0 or b == 0:
            continue
        ratio = max(a, b) / min(a, b)
        if abs(ratio - phi) <= 0.2:
            return True
    return False


def classify_aesthetic(ss: StateSpace) -> str:
    """Return the aesthetic classification of a state space.

    This is a convenience function that returns just the classification
    string without the full profile.

    Returns:
        One of 'sublime', 'beautiful', 'pleasant', 'neutral', 'ugly'.
    """
    return analyze_aesthetics(ss).classification
