"""Quantum walks on session type lattices (Step 31c).

Discrete-time quantum walk on the Hasse diagram of a session type
lattice. The walk evolves a probability amplitude vector using the
adjacency matrix as the coin operator:

- **Quantum walk evolution**: U = e^{iAt} for continuous-time
- **Transition probability**: |<j|U|i>|² at time t
- **Mixing rate**: how fast the quantum walk spreads
- **Localization**: probability of return to start
- **Quantum hitting time**: expected time to reach bottom from top
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace

from reticulate.matrix import adjacency_matrix, _eigenvalues_symmetric


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuantumWalkResult:
    """Quantum walk analysis results.

    Attributes:
        num_states: Number of states.
        evolution_matrix: |U(t=1)|² = transition probability matrix at t=1.
        return_probability: Probability of being at start after time t=1.
        spread: Standard deviation of probability distribution at t=1.
        top_to_bottom_prob: Probability of reaching bottom from top at t=1.
        mixing_time_estimate: Estimated quantum mixing time.
    """
    num_states: int
    evolution_matrix: list[list[float]]
    return_probability: float
    spread: float
    top_to_bottom_prob: float
    mixing_time_estimate: float


# ---------------------------------------------------------------------------
# Matrix exponential (simple Padé approximation for small matrices)
# ---------------------------------------------------------------------------

def _mat_scale(A: list[list[float]], c: float) -> list[list[float]]:
    return [[c * A[i][j] for j in range(len(A))] for i in range(len(A))]


def _mat_add(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]


def _mat_mul_f(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    n = len(A)
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C


def _mat_identity(n: int) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def matrix_exp_taylor(A: list[list[float]], terms: int = 20) -> list[list[float]]:
    """Matrix exponential via Taylor series: e^A = Σ A^k / k!."""
    n = len(A)
    result = _mat_identity(n)
    power = _mat_identity(n)  # A^0 = I
    factorial = 1.0

    for k in range(1, terms + 1):
        power = _mat_mul_f(power, A)
        factorial *= k
        result = _mat_add(result, _mat_scale(power, 1.0 / factorial))

    return result


# ---------------------------------------------------------------------------
# Quantum walk evolution
# ---------------------------------------------------------------------------

def quantum_evolution(ss: "StateSpace", t: float = 1.0) -> list[list[float]]:
    """Compute quantum walk transition probabilities at time t.

    U(t) = e^{iAt} where A is the adjacency matrix.
    Returns P[i][j] = |<j|U(t)|i>|² = transition probability.

    Since A is real symmetric, U(t) is unitary.
    We compute via eigendecomposition: U = V diag(e^{iλt}) V^T.
    """
    A = adjacency_matrix(ss)
    n = len(A)
    if n == 0:
        return []
    if n == 1:
        return [[1.0]]

    # For small matrices, use the Taylor series of e^{iAt}
    # Split into real and imaginary parts:
    # e^{iAt} = cos(At) + i·sin(At)
    # |e^{iAt}_{jk}|² = cos²(At)_{jk} + sin²(At)_{jk}

    # Compute cos(At) and sin(At) via Taylor series
    At = _mat_scale([[float(A[i][j]) for j in range(n)] for i in range(n)], t)

    # cos(At) = I - (At)²/2! + (At)⁴/4! - ...
    # sin(At) = At - (At)³/3! + (At)⁵/5! - ...
    cos_mat = _mat_identity(n)
    sin_mat = [[0.0] * n for _ in range(n)]
    power = _mat_identity(n)
    factorial = 1.0

    for k in range(1, 20):
        power = _mat_mul_f(power, At)
        factorial *= k
        if k % 4 == 0:
            cos_mat = _mat_add(cos_mat, _mat_scale(power, 1.0 / factorial))
        elif k % 4 == 1:
            sin_mat = _mat_add(sin_mat, _mat_scale(power, 1.0 / factorial))
        elif k % 4 == 2:
            cos_mat = _mat_add(cos_mat, _mat_scale(power, -1.0 / factorial))
        elif k % 4 == 3:
            sin_mat = _mat_add(sin_mat, _mat_scale(power, -1.0 / factorial))

    # Transition probability: P[i][j] = cos²[i][j] + sin²[i][j]
    P = [[cos_mat[i][j] ** 2 + sin_mat[i][j] ** 2 for j in range(n)] for i in range(n)]
    return P


def return_probability(ss: "StateSpace", t: float = 1.0) -> float:
    """Probability of returning to the initial state (top) at time t."""
    P = quantum_evolution(ss, t)
    states = sorted(ss.states)
    idx = {s: i for i, s in enumerate(states)}
    top_i = idx[ss.top]
    return P[top_i][top_i]


def top_to_bottom_probability(ss: "StateSpace", t: float = 1.0) -> float:
    """Probability of reaching bottom from top at time t."""
    P = quantum_evolution(ss, t)
    states = sorted(ss.states)
    idx = {s: i for i, s in enumerate(states)}
    top_i = idx[ss.top]
    bot_i = idx[ss.bottom]
    return P[top_i][bot_i]


def quantum_spread(ss: "StateSpace", t: float = 1.0) -> float:
    """Standard deviation of probability distribution at time t.

    Measures how spread out the quantum walk is from starting position.
    """
    P = quantum_evolution(ss, t)
    states = sorted(ss.states)
    idx = {s: i for i, s in enumerate(states)}
    top_i = idx[ss.top]
    n = len(states)

    # Probability distribution starting from top
    probs = [P[top_i][j] for j in range(n)]
    mean_pos = sum(j * probs[j] for j in range(n))
    variance = sum((j - mean_pos) ** 2 * probs[j] for j in range(n))
    return math.sqrt(max(0.0, variance))


def quantum_mixing_estimate(ss: "StateSpace") -> float:
    """Estimate quantum mixing time from spectral gap.

    Quantum walks mix in O(1/Δ) time where Δ is the spectral gap.
    Classical walks mix in O(1/Δ² · log n).
    """
    eigs = sorted(_eigenvalues_symmetric(
        [[float(adjacency_matrix(ss)[i][j]) for j in range(len(ss.states))]
         for i in range(len(ss.states))]
    ))
    n = len(eigs)
    if n < 2:
        return 0.0
    gap = eigs[-1] - eigs[-2] if eigs[-1] != eigs[-2] else 1.0
    return 1.0 / max(gap, 1e-10)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_quantum_walk(ss: "StateSpace") -> QuantumWalkResult:
    """Complete quantum walk analysis."""
    P = quantum_evolution(ss, 1.0)
    ret = return_probability(ss, 1.0)
    spread = quantum_spread(ss, 1.0)
    ttb = top_to_bottom_probability(ss, 1.0)
    mixing = quantum_mixing_estimate(ss)

    return QuantumWalkResult(
        num_states=len(ss.states),
        evolution_matrix=P,
        return_probability=ret,
        spread=spread,
        top_to_bottom_prob=ttb,
        mixing_time_estimate=mixing,
    )
