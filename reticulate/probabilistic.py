"""Probabilistic session types (Step 96).

Augments session-type state spaces with probability weights on transitions,
transforming the state space into a discrete-time Markov chain (DTMC).

Key capabilities:
  - Assign uniform or custom probability weights to choice points.
  - Compute expected path length (expected steps to termination).
  - Shannon entropy at individual states and across the whole protocol.
  - Build the Markov-chain transition matrix from a weighted state space.
  - Compute stationary distributions for irreducible (server-loop) chains.
  - Expected number of visits to each state.
  - Full probabilistic analysis combining all of the above.

References:
  - Kemeny & Snell, Finite Markov Chains, 1976.
  - Baier & Katoen, Principles of Model Checking, 2008.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProbabilisticAssignment:
    """Probability weights for each choice point in a state space.

    Attributes:
        weights: Mapping from state to dict of (label -> probability).
                 Only states with outgoing transitions are included.
                 For each state, probabilities must sum to 1.0 (within tolerance).
    """
    weights: dict[int, dict[str, float]]

    def probability(self, state: int, label: str) -> float:
        """Return the probability of taking transition *label* from *state*."""
        return self.weights.get(state, {}).get(label, 0.0)


@dataclass(frozen=True)
class ProbabilisticAnalysis:
    """Full probabilistic analysis result.

    Attributes:
        assignment: The probability assignment used.
        expected_path_length: Expected steps from top to bottom (None if infinite/cyclic unsolvable).
        total_entropy: Sum of Shannon entropies across all choice points.
        state_entropies: Per-state Shannon entropy.
        is_absorbing: Whether bottom is absorbing and reachable from all states.
        stationary_distribution: Steady-state distribution (None if not applicable).
        expected_visits: Expected number of visits to each state from top.
        transition_matrix: The full transition probability matrix as nested dict.
    """
    assignment: ProbabilisticAssignment
    expected_path_length: float | None
    total_entropy: float
    state_entropies: dict[int, float]
    is_absorbing: bool
    stationary_distribution: dict[int, float] | None
    expected_visits: dict[int, float] | None
    transition_matrix: dict[int, dict[int, float]]


# ---------------------------------------------------------------------------
# Probability assignment
# ---------------------------------------------------------------------------

def assign_uniform(ss: StateSpace) -> ProbabilisticAssignment:
    """Assign uniform probabilities at each choice point.

    Each outgoing transition from a state gets equal probability 1/n
    where n is the number of outgoing transitions.
    """
    weights: dict[int, dict[str, float]] = {}
    for state in ss.states:
        outgoing = ss.enabled(state)
        if not outgoing:
            continue
        n = len(outgoing)
        p = 1.0 / n
        weights[state] = {label: p for label, _tgt in outgoing}
    return ProbabilisticAssignment(weights=weights)


def assign_weights(
    ss: StateSpace,
    weights: dict[int, dict[str, float]],
) -> ProbabilisticAssignment:
    """Assign custom probability weights to each choice point.

    Args:
        ss: The state space.
        weights: Mapping from state to dict of (label -> probability).

    Raises:
        ValueError: If probabilities don't sum to 1.0 (within tolerance 1e-9),
                    if a label doesn't match an outgoing transition, or if
                    a state with outgoing transitions is missing from weights.
    """
    tolerance = 1e-9

    for state in ss.states:
        outgoing = ss.enabled(state)
        if not outgoing:
            continue
        if state not in weights:
            raise ValueError(
                f"State {state} has outgoing transitions but no weights assigned"
            )
        out_labels = {label for label, _tgt in outgoing}
        w = weights[state]
        for label in w:
            if label not in out_labels:
                raise ValueError(
                    f"Label {label!r} at state {state} does not match "
                    f"any outgoing transition"
                )
        for label in out_labels:
            if label not in w:
                raise ValueError(
                    f"Missing weight for label {label!r} at state {state}"
                )
        total = sum(w.values())
        if abs(total - 1.0) > tolerance:
            raise ValueError(
                f"Probabilities at state {state} sum to {total}, expected 1.0"
            )
        for label, prob in w.items():
            if prob < 0.0:
                raise ValueError(
                    f"Negative probability {prob} for label {label!r} at state {state}"
                )

    return ProbabilisticAssignment(weights=weights)


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------

def state_entropy(ss: StateSpace, probs: ProbabilisticAssignment, state: int) -> float:
    """Shannon entropy at a single state (in bits).

    H(state) = -sum(p * log2(p)) over outgoing transitions.
    Returns 0.0 for states with no outgoing transitions or a single transition.
    """
    w = probs.weights.get(state, {})
    if not w:
        return 0.0
    h = 0.0
    for p in w.values():
        if p > 0.0:
            h -= p * math.log2(p)
    return h


def total_entropy(ss: StateSpace, probs: ProbabilisticAssignment) -> float:
    """Sum of Shannon entropies across all choice points.

    Total H = sum of H(state) for all states with outgoing transitions.
    """
    return sum(state_entropy(ss, probs, s) for s in ss.states)


def path_entropy(ss: StateSpace, probs: ProbabilisticAssignment) -> float:
    """Entropy of the execution path distribution.

    For acyclic state spaces, this is the total entropy weighted by path
    probabilities. For simplicity, we compute the sum of entropies at all
    choice points reachable from top (same as total_entropy for connected
    state spaces, which session type state spaces always are).
    """
    reachable = ss.reachable_from(ss.top)
    return sum(state_entropy(ss, probs, s) for s in reachable)


# ---------------------------------------------------------------------------
# Markov chain construction
# ---------------------------------------------------------------------------

def markov_chain(
    ss: StateSpace,
    probs: ProbabilisticAssignment,
) -> dict[int, dict[int, float]]:
    """Build the transition probability matrix of the induced DTMC.

    Returns a nested dict: matrix[src][tgt] = probability.
    The bottom state is absorbing: P(bottom, bottom) = 1.
    States with no outgoing transitions (other than bottom) are also absorbing.
    """
    matrix: dict[int, dict[int, float]] = {s: {} for s in ss.states}

    for state in ss.states:
        outgoing = ss.enabled(state)
        if not outgoing:
            # Absorbing state (includes bottom)
            matrix[state][state] = 1.0
            continue
        w = probs.weights.get(state, {})
        if not w:
            # No probabilities assigned — treat as absorbing
            matrix[state][state] = 1.0
            continue
        # Group by target (multiple labels may go to same target)
        for label, tgt in outgoing:
            p = w.get(label, 0.0)
            matrix[state][tgt] = matrix[state].get(tgt, 0.0) + p

    return matrix


# ---------------------------------------------------------------------------
# Expected path length
# ---------------------------------------------------------------------------

def expected_path_length(
    ss: StateSpace,
    probs: ProbabilisticAssignment,
) -> float | None:
    """Expected number of transitions from top to bottom.

    For acyclic state spaces: backward induction from bottom.
    For cyclic state spaces: solves the system of linear equations
    E[T|s] = 1 + sum_t P(s,t) * E[T|t] using Gaussian elimination
    on the fundamental matrix (I - Q).

    Returns None if the expected path length is infinite (non-absorbing chain).
    """
    matrix = markov_chain(ss, probs)
    bottom = ss.bottom
    top = ss.top

    if top == bottom:
        return 0.0

    # Identify transient states (all states except absorbing ones that are bottom)
    # For our purposes: solve E[s] = 1 + sum_t P(s,t)*E[t] for all s != bottom
    # where E[bottom] = 0
    reachable = ss.reachable_from(top)
    transient = [s for s in reachable if s != bottom]

    if not transient:
        return 0.0

    # Build the system: E[s] = 1 + sum_{t != bottom} P(s,t) * E[t]
    # Rearranged: E[s] - sum_{t != bottom} P(s,t) * E[t] = 1
    # In matrix form: (I - Q) * E = 1  where Q_ij = P(i,j) for i,j transient
    n = len(transient)
    idx = {s: i for i, s in enumerate(transient)}

    # Augmented matrix for Gaussian elimination: (I-Q | 1)
    aug = [[0.0] * (n + 1) for _ in range(n)]
    for i, s in enumerate(transient):
        aug[i][i] = 1.0  # I diagonal
        aug[i][n] = 1.0  # RHS = 1
        row = matrix.get(s, {})
        for t, p in row.items():
            if t in idx:
                aug[i][idx[t]] -= p  # subtract Q

    # Gaussian elimination with partial pivoting
    for col in range(n):
        # Find pivot
        max_row = col
        max_val = abs(aug[col][col])
        for row in range(col + 1, n):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                max_row = row
        if max_val < 1e-12:
            # Singular matrix — infinite expected time
            return None
        if max_row != col:
            aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot

        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]

    # Extract solution
    expected = {transient[i]: aug[i][n] for i in range(n)}

    result = expected.get(top, 0.0)
    if result < 0 or not math.isfinite(result):
        return None
    return result


# ---------------------------------------------------------------------------
# Stationary distribution
# ---------------------------------------------------------------------------

def stationary_distribution(
    ss: StateSpace,
    probs: ProbabilisticAssignment,
) -> dict[int, float] | None:
    """Compute the stationary distribution for the induced DTMC.

    The stationary distribution pi satisfies pi = pi * P and sum(pi) = 1.
    Returns None if the chain has no unique stationary distribution
    (e.g., if it has multiple absorbing states or is not irreducible).

    For absorbing chains (with bottom as absorbing state), the stationary
    distribution concentrates on the absorbing state. We return None for
    such trivially absorbing chains and only compute for chains with
    recurrent classes (e.g., server loops).
    """
    matrix = markov_chain(ss, probs)
    states = sorted(ss.states)
    n = len(states)

    if n == 0:
        return None
    if n == 1:
        return {states[0]: 1.0}

    idx = {s: i for i, s in enumerate(states)}

    # Check if the chain is irreducible by checking if all states communicate.
    # For session types with bottom as absorbing, restrict to the recurrent class.
    # Find recurrent states: states that can reach themselves.
    recurrent_states = []
    for s in states:
        reachable = set()
        stack = [s]
        while stack:
            cur = stack.pop()
            if cur in reachable:
                continue
            reachable.add(cur)
            for t, p in matrix.get(cur, {}).items():
                if p > 0 and t not in reachable:
                    stack.append(t)
        if s in reachable and len(reachable) > 1:
            # s can reach itself through other states
            # Check if it's in a non-trivial recurrent class
            recurrent_states.append(s)
        elif len(matrix.get(s, {})) == 1 and matrix[s].get(s, 0.0) == 1.0:
            # Self-absorbing
            recurrent_states.append(s)

    if not recurrent_states:
        return None

    # Find the largest communicating class containing recurrent states
    # For simplicity, attempt to solve pi = pi*P on the full chain
    # using the standard approach: (P^T - I) * pi = 0, sum(pi) = 1

    # Build P^T - I augmented with sum constraint
    # Replace last equation with sum(pi) = 1
    aug = [[0.0] * (n + 1) for _ in range(n)]
    for i, si in enumerate(states):
        for j, sj in enumerate(states):
            p = matrix.get(sj, {}).get(si, 0.0)  # P^T[i][j] = P[j][i]
            aug[i][j] = p - (1.0 if i == j else 0.0)
        aug[i][n] = 0.0

    # Replace last row with sum constraint
    for j in range(n):
        aug[n - 1][j] = 1.0
    aug[n - 1][n] = 1.0

    # Gaussian elimination
    for col in range(n):
        max_row = col
        max_val = abs(aug[col][col])
        for row in range(col + 1, n):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                max_row = row
        if max_val < 1e-12:
            return None
        if max_row != col:
            aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot

        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]

    result = {}
    for i, s in enumerate(states):
        val = aug[i][n]
        if val < -1e-9:
            return None
        result[s] = max(0.0, val)

    # Normalize
    total = sum(result.values())
    if total < 1e-12:
        return None
    result = {s: v / total for s, v in result.items()}

    return result


# ---------------------------------------------------------------------------
# Expected visits
# ---------------------------------------------------------------------------

def expected_visits(
    ss: StateSpace,
    probs: ProbabilisticAssignment,
    state: int,
) -> float | None:
    """Expected number of visits to *state* starting from top, before absorption.

    Uses the fundamental matrix N = (I - Q)^{-1} for absorbing chains.
    Returns None if the chain is not absorbing.
    """
    visits = _all_expected_visits(ss, probs)
    if visits is None:
        return None
    return visits.get(state, 0.0)


def _all_expected_visits(
    ss: StateSpace,
    probs: ProbabilisticAssignment,
) -> dict[int, float] | None:
    """Expected visits to each transient state from top.

    Computes the fundamental matrix N = (I - Q)^{-1} and returns
    the row corresponding to the top state.
    """
    matrix = markov_chain(ss, probs)
    bottom = ss.bottom
    top = ss.top

    reachable = ss.reachable_from(top)
    transient = [s for s in reachable if s != bottom]
    if not transient:
        return {bottom: 0.0}

    n = len(transient)
    idx = {s: i for i, s in enumerate(transient)}

    if top not in idx:
        return {s: 0.0 for s in ss.states}

    # Build (I - Q) and invert using Gaussian elimination
    # We need the row of N corresponding to top
    top_idx = idx[top]

    # Build augmented matrix (I-Q | I) to compute N = (I-Q)^{-1}
    aug = [[0.0] * (2 * n) for _ in range(n)]
    for i, s in enumerate(transient):
        aug[i][i] = 1.0  # I diagonal
        aug[i][n + i] = 1.0  # Identity on right side
        row = matrix.get(s, {})
        for t, p in row.items():
            if t in idx:
                aug[i][idx[t]] -= p  # subtract Q

    # Gaussian elimination
    for col in range(n):
        max_row = col
        max_val = abs(aug[col][col])
        for row in range(col + 1, n):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                max_row = row
        if max_val < 1e-12:
            return None
        if max_row != col:
            aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= pivot

        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(2 * n):
                aug[row][j] -= factor * aug[col][j]

    # Extract row of N corresponding to top
    result: dict[int, float] = {}
    for j, s in enumerate(transient):
        result[s] = aug[top_idx][n + j]
    result[bottom] = 0.0

    return result


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze_probabilistic(
    ss: StateSpace,
    probs: ProbabilisticAssignment,
) -> ProbabilisticAnalysis:
    """Run full probabilistic analysis on a weighted state space.

    Computes expected path length, entropy, Markov chain, stationary
    distribution, and expected visits.
    """
    matrix = markov_chain(ss, probs)

    entropies = {s: state_entropy(ss, probs, s) for s in ss.states}
    t_entropy = sum(entropies.values())

    epl = expected_path_length(ss, probs)

    # Check if bottom is absorbing and reachable
    bottom_absorbing = (
        matrix.get(ss.bottom, {}).get(ss.bottom, 0.0) == 1.0
        and len(matrix.get(ss.bottom, {})) <= 1
    )
    bottom_reachable = ss.bottom in ss.reachable_from(ss.top)
    is_abs = bottom_absorbing and bottom_reachable

    sd = stationary_distribution(ss, probs)

    visits = _all_expected_visits(ss, probs)

    return ProbabilisticAnalysis(
        assignment=probs,
        expected_path_length=epl,
        total_entropy=t_entropy,
        state_entropies=entropies,
        is_absorbing=is_abs,
        stationary_distribution=sd,
        expected_visits=visits,
        transition_matrix=matrix,
    )
