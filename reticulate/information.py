"""Information-theoretic session types: entropy, capacity, complexity (Step 60b).

Applies information theory to session type state spaces, measuring how much
*choice* a protocol encodes.  The key quantities are:

- **Branching entropy**: average Shannon entropy across states.  A state
  with *n* outgoing transitions contributes H = log2(n) bits (uniform
  distribution over choices).  Averaged over all non-terminal states.

- **Path entropy**: entropy of the *path length* distribution.  Enumerates
  top-to-bottom paths (bounded), computes the distribution of their
  lengths, and returns H of that distribution.

- **Label entropy**: entropy of the label frequency distribution.  Counts
  how often each label appears in transitions, normalises, returns H.

- **Channel capacity**: log2 of the number of distinct top-to-bottom
  paths.  This is the maximum information (in bits) that can be
  communicated by choosing a path through the protocol.

- **Determinism**: a state space is deterministic if no state has two
  outgoing transitions with the same label.

- **Information density**: channel capacity divided by the number of
  transitions — bits per transition.

This module provides:
    ``branching_entropy(ss)``    — average per-state Shannon entropy.
    ``path_entropy(ss)``         — entropy of path-length distribution.
    ``label_entropy(ss)``        — entropy of label frequencies.
    ``channel_capacity(ss)``     — log2(#paths) from top to bottom.
    ``is_deterministic(ss)``     — no duplicate labels per state.
    ``information_density(ss)``  — channel_capacity / |transitions|.
    ``analyze_information(ss)``  — full analysis as InformationResult.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InformationResult:
    """Result of information-theoretic analysis.

    Attributes:
        branching_entropy: Average Shannon entropy across non-terminal states.
        path_entropy: Entropy of the path-length distribution.
        label_entropy: Entropy of the label frequency distribution.
        channel_capacity: log2(number of distinct top-to-bottom paths).
        state_complexity: Number of states in the state space.
        transition_complexity: Number of transitions in the state space.
        information_density: channel_capacity / |transitions|.
        is_deterministic: True if no state has duplicate outgoing labels.
    """

    branching_entropy: float
    path_entropy: float
    label_entropy: float
    channel_capacity: float
    state_complexity: int
    transition_complexity: int
    information_density: float
    is_deterministic: bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_log2(x: float) -> float:
    """log2(x) with convention that log2(0) = 0."""
    if x <= 0:
        return 0.0
    return math.log2(x)


def _shannon_entropy(probabilities: list[float]) -> float:
    """Compute Shannon entropy H = -sum(p * log2(p)) for a distribution."""
    return -sum(p * _safe_log2(p) for p in probabilities if p > 0)


def _enumerate_paths(
    ss: StateSpace, *, max_paths: int = 1000
) -> list[list[int]]:
    """Bounded DFS enumeration of all top-to-bottom paths.

    Returns a list of state-ID paths from ``ss.top`` to ``ss.bottom``.
    Stops after *max_paths* complete paths to avoid explosion.
    """
    if ss.top == ss.bottom:
        return [[ss.top]]

    # Build adjacency list for efficiency
    adj: dict[int, list[int]] = {}
    for src, _label, tgt in ss.transitions:
        adj.setdefault(src, []).append(tgt)

    paths: list[list[int]] = []
    # DFS stack: (current_state, path_so_far)
    stack: list[tuple[int, list[int]]] = [(ss.top, [ss.top])]

    while stack and len(paths) < max_paths:
        state, path = stack.pop()
        if state == ss.bottom:
            paths.append(path)
            continue
        for tgt in adj.get(state, []):
            if tgt not in path:  # avoid cycles
                stack.append((tgt, path + [tgt]))

    return paths


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def branching_entropy(ss: StateSpace) -> float:
    """Average Shannon entropy across all non-terminal states.

    For each state with *n* outgoing transitions, the local entropy is
    H = log2(n) (uniform distribution: p_i = 1/n for each).  The result
    is the average over all states that have at least one outgoing
    transition.

    Returns 0.0 for single-state (``end``) types.
    """
    entropies: list[float] = []
    for state in ss.states:
        out = ss.enabled(state)
        n = len(out)
        if n == 0:
            continue
        # Uniform distribution: p_i = 1/n for each transition
        probs = [1.0 / n] * n
        entropies.append(_shannon_entropy(probs))

    if not entropies:
        return 0.0
    return sum(entropies) / len(entropies)


def path_entropy(ss: StateSpace) -> float:
    """Entropy of the path-length distribution.

    Enumerates top-to-bottom paths (bounded by 1000), computes the
    distribution of path lengths, and returns Shannon entropy of that
    distribution.

    Returns 0.0 if there is only one path or no paths.
    """
    paths = _enumerate_paths(ss, max_paths=1000)
    if len(paths) <= 1:
        return 0.0

    # Path length = number of transitions = len(path) - 1
    lengths = [len(p) - 1 for p in paths]
    counter = Counter(lengths)
    total = len(lengths)
    probs = [count / total for count in counter.values()]
    return _shannon_entropy(probs)


def label_entropy(ss: StateSpace) -> float:
    """Shannon entropy of the label frequency distribution.

    Counts how often each label appears across all transitions,
    normalises to a probability distribution, and returns H.

    Returns 0.0 if there are no transitions or only one distinct label.
    """
    if not ss.transitions:
        return 0.0

    counter = Counter(label for _src, label, _tgt in ss.transitions)
    total = sum(counter.values())
    probs = [count / total for count in counter.values()]
    return _shannon_entropy(probs)


def channel_capacity(ss: StateSpace) -> float:
    """log2 of the number of distinct top-to-bottom paths.

    This represents the maximum information (in bits) that can be
    communicated by choosing a path through the protocol.

    Returns 0.0 for types with zero or one path.
    """
    paths = _enumerate_paths(ss, max_paths=1000)
    n = len(paths)
    if n <= 0:
        return 0.0
    return _safe_log2(n)


def is_deterministic(ss: StateSpace) -> bool:
    """True if no state has two outgoing transitions with the same label.

    A deterministic state space means the label alone determines the
    next state — there is no non-deterministic branching.
    """
    for state in ss.states:
        labels = [label for _src, label, _tgt in ss.transitions if _src == state]
        if len(labels) != len(set(labels)):
            return False
    return True


def information_density(ss: StateSpace) -> float:
    """Channel capacity divided by the number of transitions.

    Measures how many bits of choice each transition contributes.
    Returns 0.0 if there are no transitions.
    """
    if not ss.transitions:
        return 0.0
    return channel_capacity(ss) / len(ss.transitions)


def analyze_information(ss: StateSpace) -> InformationResult:
    """Full information-theoretic analysis of a state space.

    Returns an InformationResult with all computed metrics.
    """
    cap = channel_capacity(ss)
    n_trans = len(ss.transitions)
    density = cap / n_trans if n_trans > 0 else 0.0

    return InformationResult(
        branching_entropy=branching_entropy(ss),
        path_entropy=path_entropy(ss),
        label_entropy=label_entropy(ss),
        channel_capacity=cap,
        state_complexity=len(ss.states),
        transition_complexity=n_trans,
        information_density=density,
        is_deterministic=is_deterministic(ss),
    )
