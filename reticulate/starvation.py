"""Starvation detection via spectral analysis for session types (Step 80d).

Starvation occurs when a transition (or set of transitions) is enabled in
some state but never taken under fair scheduling.  More precisely, a
transition is *starved* if there exists a reachable state where it is
enabled but no fair execution path passes through it.

In the context of session type state spaces:
- A transition is **reachable** if it lies on at least one path from
  top to bottom.
- A transition is **starved** if it exists in the state space but no
  path from top to bottom uses it.
- A state is **starved** if it is unreachable from top, or cannot reach
  bottom, and yet belongs to the state space.

Spectral analysis provides quantitative measures:
- **Per-role spectral gap**: For transitions grouped by label, the
  spectral gap of the induced sub-transition-matrix indicates how
  evenly that label is used.  Low gap = potential starvation.
- **Fairness score**: Ratio of minimum to maximum transition coverage
  under uniform random walk from top.
- **Pendular fairness**: Alternation balance as a fairness measure.

Public API:
    detect_starvation(ss)         -- find starved transitions
    per_role_spectral_gap(ss)     -- spectral gap per label
    fairness_score(ss)            -- min/max transition coverage ratio
    pendular_fairness(ss)         -- pendular balance fairness
    is_starvation_free(ss)        -- True iff all transitions reachable
    analyze_starvation(ss)        -- comprehensive StarvationResult
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StarvedTransition:
    """A single starved transition.

    Attributes:
        src: Source state.
        label: Transition label.
        tgt: Target state.
        reason: Why this transition is starved.
    """
    src: int
    label: str
    tgt: int
    reason: str


@dataclass(frozen=True)
class RoleSpectralGap:
    """Spectral gap for a single label (role).

    Attributes:
        label: The transition label.
        count: Number of transitions with this label.
        reachable_count: Number of those on top-to-bottom paths.
        spectral_gap: Spectral gap for this label's sub-matrix.
        is_starved: True if some transitions with this label are unreachable.
    """
    label: str
    count: int
    reachable_count: int
    spectral_gap: float
    is_starved: bool


@dataclass(frozen=True)
class StarvationResult:
    """Comprehensive starvation analysis result.

    Attributes:
        is_starvation_free: True iff all transitions are on top-bottom paths.
        starved_transitions: Details of each starved transition.
        num_starved: Number of starved transitions.
        num_total_transitions: Total transitions.
        role_spectral_gaps: Spectral gap per label.
        fairness: Fairness score (0 = maximally unfair, 1 = perfectly fair).
        pendular_balance: Pendular fairness measure.
        reachable_states: States on paths from top to bottom.
        unreachable_states: States not on any top-to-bottom path.
        coverage_ratio: Fraction of transitions that are reachable.
        summary: Human-readable summary.
    """
    is_starvation_free: bool
    starved_transitions: tuple[StarvedTransition, ...]
    num_starved: int
    num_total_transitions: int
    role_spectral_gaps: tuple[RoleSpectralGap, ...]
    fairness: float
    pendular_balance: float
    reachable_states: frozenset[int]
    unreachable_states: frozenset[int]
    coverage_ratio: float
    summary: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _outgoing(ss: "StateSpace", state: int) -> list[tuple[str, int]]:
    """Return outgoing (label, target) pairs for a state."""
    return [(label, tgt) for src, label, tgt in ss.transitions if src == state]


def _reachable_from(ss: "StateSpace", start: int) -> set[int]:
    """States reachable from *start* via forward edges."""
    visited: set[int] = set()
    queue = deque([start])
    visited.add(start)
    while queue:
        s = queue.popleft()
        for label, t in _outgoing(ss, s):
            if t not in visited:
                visited.add(t)
                queue.append(t)
    return visited


def _reverse_reachable(ss: "StateSpace", target: int) -> set[int]:
    """States that can reach *target* via reverse edges."""
    rev: dict[int, list[int]] = {s: [] for s in ss.states}
    for src, label, tgt in ss.transitions:
        rev.setdefault(tgt, []).append(src)
    visited: set[int] = set()
    queue = deque([target])
    visited.add(target)
    while queue:
        s = queue.popleft()
        for pred in rev.get(s, []):
            if pred not in visited:
                visited.add(pred)
                queue.append(pred)
    return visited


def _transitions_on_paths(
    ss: "StateSpace",
    from_top: set[int],
    to_bottom: set[int],
) -> set[tuple[int, str, int]]:
    """Transitions that lie on at least one top-to-bottom path.

    A transition (s, l, t) is on a top-bottom path if:
    - s is reachable from top, AND
    - t can reach bottom.
    """
    on_path: set[tuple[int, str, int]] = set()
    for src, label, tgt in ss.transitions:
        if src in from_top and tgt in to_bottom:
            on_path.add((src, label, tgt))
    return on_path


def _random_walk_transition_coverage(
    ss: "StateSpace",
    num_steps: int = 500,
) -> dict[tuple[int, str, int], float]:
    """Estimate transition coverage under uniform random walk from top.

    Uses power method on transition matrix to compute expected visit
    frequency for each transition.
    """
    if not ss.states or not ss.transitions:
        return {}

    states = sorted(ss.states)
    n = len(states)
    idx = {s: i for i, s in enumerate(states)}

    # Build adjacency
    adj: dict[int, list[tuple[str, int]]] = {s: [] for s in states}
    for src, label, tgt in ss.transitions:
        adj[src].append((label, tgt))

    # Compute state visit probabilities via power iteration
    # Start from top
    prob = [0.0] * n
    top_idx = idx.get(ss.top, 0)
    prob[top_idx] = 1.0

    # Accumulate visit frequencies
    visit_freq = [0.0] * n
    trans_freq: dict[tuple[int, str, int], float] = {}

    for _ in range(num_steps):
        new_prob = [0.0] * n
        for s in states:
            si = idx[s]
            if prob[si] < 1e-15:
                continue
            neighbors = adj[s]
            if not neighbors:
                # Absorbing: stays here
                new_prob[si] += prob[si]
                continue
            # Deduplicate targets for probability
            targets: list[tuple[str, int]] = neighbors
            p_each = prob[si] / len(targets)
            for label, tgt in targets:
                ti = idx[tgt]
                new_prob[ti] += p_each
                key = (s, label, tgt)
                trans_freq[key] = trans_freq.get(key, 0.0) + p_each

        prob = new_prob
        for i in range(n):
            visit_freq[i] += prob[i]

    return trans_freq


def _compute_spectral_gap_for_label(
    ss: "StateSpace",
    label: str,
) -> float:
    """Compute spectral gap for transitions with a specific label.

    Builds a sub-transition matrix restricted to the given label and
    computes 1 - |lambda_2| where lambda_2 is the second-largest
    eigenvalue magnitude.

    Returns a value in [0, 1]. Higher = better connectivity for this label.
    """
    # Collect states involved in transitions with this label
    states_involved: set[int] = set()
    for src, lbl, tgt in ss.transitions:
        if lbl == label:
            states_involved.add(src)
            states_involved.add(tgt)

    if len(states_involved) <= 1:
        return 1.0  # Trivial: perfectly connected

    state_list = sorted(states_involved)
    n = len(state_list)
    idx = {s: i for i, s in enumerate(state_list)}

    # Build adjacency matrix for this label
    A = [[0.0] * n for _ in range(n)]
    for src, lbl, tgt in ss.transitions:
        if lbl == label and src in states_involved and tgt in states_involved:
            A[idx[src]][idx[tgt]] = 1.0
            A[idx[tgt]][idx[src]] = 1.0  # symmetrize

    # Degree matrix
    D = [sum(A[i]) for i in range(n)]

    # Normalized Laplacian eigenvalues via power iteration on L = I - D^{-1/2} A D^{-1/2}
    # For simplicity, compute transition matrix P = D^{-1} A and find eigenvalues
    P = [[0.0] * n for _ in range(n)]
    for i in range(n):
        if D[i] > 0:
            for j in range(n):
                P[i][j] = A[i][j] / D[i]
        else:
            P[i][i] = 1.0

    # Power iteration for second eigenvalue
    # First eigenvector is the stationary distribution
    # Use deflated power iteration for lambda_2
    eigenvalues = _small_eigenvalues(P, n)
    if len(eigenvalues) < 2:
        return 1.0

    lambda_2 = abs(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
    return max(0.0, min(1.0, 1.0 - lambda_2))


def _small_eigenvalues(
    M: list[list[float]],
    n: int,
    max_iter: int = 100,
) -> list[float]:
    """Compute eigenvalues of a small matrix via QR-like iteration.

    Returns eigenvalues sorted descending by magnitude.
    """
    if n == 0:
        return []
    if n == 1:
        return [M[0][0]]

    # Simple power iteration approach for top eigenvalues
    A = [row[:] for row in M]

    # QR iteration
    for _ in range(max_iter):
        # QR decomposition (Gram-Schmidt)
        Q = [[0.0] * n for _ in range(n)]
        R = [[0.0] * n for _ in range(n)]

        for j in range(n):
            # v = column j of A
            v = [A[i][j] for i in range(n)]

            for k in range(j):
                # R[k][j] = q_k . v
                dot = sum(Q[i][k] * v[i] for i in range(n))
                R[k][j] = dot
                for i in range(n):
                    v[i] -= dot * Q[i][k]

            norm = math.sqrt(sum(x * x for x in v))
            if norm < 1e-15:
                R[j][j] = 0.0
                for i in range(n):
                    Q[i][j] = 0.0
            else:
                R[j][j] = norm
                for i in range(n):
                    Q[i][j] = v[i] / norm

        # A = R * Q
        new_A = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    new_A[i][j] += R[i][k] * Q[k][j]
        A = new_A

        # Check convergence
        off = sum(A[i][j] ** 2 for i in range(n) for j in range(i))
        if off < 1e-20:
            break

    eigs = [A[i][i] for i in range(n)]
    eigs.sort(key=lambda x: -abs(x))
    return eigs


def _pendular_balance(ss: "StateSpace") -> float:
    """Compute pendular balance: ratio of branch to select transitions.

    A perfectly balanced protocol has equal branch and select transitions.
    Returns a value in [0, 1] where 1 = perfectly balanced.
    """
    branch_count = 0
    select_count = 0

    for src, label, tgt in ss.transitions:
        if (src, label, tgt) in ss.selection_transitions:
            select_count += 1
        else:
            branch_count += 1

    total = branch_count + select_count
    if total == 0:
        return 1.0

    ratio = min(branch_count, select_count) / max(branch_count, select_count) if max(branch_count, select_count) > 0 else 1.0
    return ratio


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_starvation(ss: "StateSpace") -> list[StarvedTransition]:
    """Find transitions that are not on any top-to-bottom path.

    A transition is starved if it exists in the state space but no
    complete execution (top -> ... -> bottom) uses it.

    Returns:
        List of StarvedTransition detailing each starved transition.
    """
    from_top = _reachable_from(ss, ss.top)
    to_bottom = _reverse_reachable(ss, ss.bottom)
    on_path = _transitions_on_paths(ss, from_top, to_bottom)

    starved: list[StarvedTransition] = []
    for src, label, tgt in ss.transitions:
        if (src, label, tgt) not in on_path:
            if src not in from_top:
                reason = f"source state {src} not reachable from top"
            elif tgt not in to_bottom:
                reason = f"target state {tgt} cannot reach bottom"
            else:
                reason = "transition not on any top-to-bottom path"
            starved.append(StarvedTransition(
                src=src,
                label=label,
                tgt=tgt,
                reason=reason,
            ))

    return starved


def per_role_spectral_gap(ss: "StateSpace") -> dict[str, RoleSpectralGap]:
    """Compute spectral gap per transition label.

    Groups transitions by label and computes the spectral gap of the
    sub-transition-matrix for each label.  Low spectral gap indicates
    poor connectivity for that label (potential starvation).

    Returns:
        Dict mapping label to RoleSpectralGap.
    """
    from_top = _reachable_from(ss, ss.top)
    to_bottom = _reverse_reachable(ss, ss.bottom)
    on_path = _transitions_on_paths(ss, from_top, to_bottom)

    # Group transitions by label
    by_label: dict[str, list[tuple[int, str, int]]] = {}
    for src, label, tgt in ss.transitions:
        by_label.setdefault(label, []).append((src, label, tgt))

    result: dict[str, RoleSpectralGap] = {}
    for label, transitions in sorted(by_label.items()):
        count = len(transitions)
        reachable_count = sum(1 for t in transitions if t in on_path)
        gap = _compute_spectral_gap_for_label(ss, label)
        is_starved = reachable_count < count

        result[label] = RoleSpectralGap(
            label=label,
            count=count,
            reachable_count=reachable_count,
            spectral_gap=gap,
            is_starved=is_starved,
        )

    return result


def fairness_score(ss: "StateSpace") -> float:
    """Compute fairness score: min/max transition coverage ratio.

    Under uniform random walk from top, measures how evenly transitions
    are used.  Returns ratio of minimum to maximum expected transition
    frequency.

    Returns:
        Float in [0, 1]. 1 = perfectly fair (all transitions equally used).
        0 = maximally unfair (some transitions never used).
    """
    if not ss.transitions:
        return 1.0

    coverage = _random_walk_transition_coverage(ss)

    if not coverage:
        return 0.0

    freqs = list(coverage.values())
    if not freqs:
        return 0.0

    max_freq = max(freqs)
    min_freq = min(freqs)

    if max_freq < 1e-15:
        return 1.0  # No transitions taken (trivial)

    # Also account for transitions not in coverage (frequency 0)
    all_transitions = set((s, l, t) for s, l, t in ss.transitions)
    if len(coverage) < len(all_transitions):
        min_freq = 0.0

    return min_freq / max_freq if max_freq > 0 else 1.0


def pendular_fairness(ss: "StateSpace") -> float:
    """Pendular balance as a fairness measure.

    Measures the ratio between branch and selection transitions.
    Perfectly pendular protocols (alternating +/&) have balance close
    to 1.0.

    Returns:
        Float in [0, 1]. 1 = perfectly balanced.
    """
    return _pendular_balance(ss)


def is_starvation_free(ss: "StateSpace") -> bool:
    """Return True iff all transitions are on some top-to-bottom path.

    A state space is starvation-free if every transition is reachable
    (its source is reachable from top and its target can reach bottom).
    """
    return len(detect_starvation(ss)) == 0


def analyze_starvation(ss: "StateSpace") -> StarvationResult:
    """Comprehensive starvation analysis.

    Combines transition reachability, spectral gap per label, fairness
    scoring, and pendular balance.

    Returns:
        StarvationResult with complete analysis.
    """
    from_top = _reachable_from(ss, ss.top)
    to_bottom = _reverse_reachable(ss, ss.bottom)

    # States on top-bottom paths
    reachable_states = frozenset(from_top & to_bottom)
    unreachable = frozenset(ss.states - reachable_states)

    # Starved transitions
    starved = detect_starvation(ss)

    # Spectral gaps
    gaps = per_role_spectral_gap(ss)

    # Fairness
    fair = fairness_score(ss)

    # Pendular balance
    pendular = pendular_fairness(ss)

    # Coverage ratio
    total = len(ss.transitions)
    coverage = (total - len(starved)) / total if total > 0 else 1.0

    is_free = len(starved) == 0

    if is_free:
        summary = "Starvation-free: all transitions are on top-to-bottom paths."
    else:
        starved_labels = {s.label for s in starved}
        summary = (
            f"STARVATION DETECTED: {len(starved)} of {total} transitions "
            f"are starved. Affected labels: {', '.join(sorted(starved_labels))}. "
            f"Fairness score: {fair:.3f}. "
            f"{len(unreachable)} unreachable states."
        )

    return StarvationResult(
        is_starvation_free=is_free,
        starved_transitions=tuple(starved),
        num_starved=len(starved),
        num_total_transitions=total,
        role_spectral_gaps=tuple(gaps.values()),
        fairness=fair,
        pendular_balance=pendular,
        reachable_states=reachable_states,
        unreachable_states=unreachable,
        coverage_ratio=coverage,
        summary=summary,
    )
