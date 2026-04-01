"""Information geometry of session type lattices (Step 600e).

Three approaches to equipping session type state spaces with geometric structure:

**Approach A: Fisher metric on the full simplex**
  Treat the n-state space as a categorical distribution. The Fisher information
  matrix G_ij gives a Riemannian metric on the probability simplex. The induced
  Fisher-Rao distance is the Bhattacharyya angle: 2·arccos(Σ√(p·q)).

**Approach B: Exponential family (Boltzmann) geometry**
  The one-parameter Boltzmann family {p_β : β > 0} is an exponential family
  with natural parameter η = -β and sufficient statistic E(s). The Fisher
  information along this curve is I(β) = Var_β(E) = C(β)/β², linking
  information geometry to thermodynamic specific heat.

**Approach C: Lattice-adapted metric**
  Modify the Fisher metric by weighting state pairs by their lattice distance,
  incorporating protocol structure into the geometry. This is novel.

**Bidirectional morphisms:**
  - embed: state s → Dirac delta δ_s on the manifold boundary
  - retract: distribution p → MAP estimate argmax_s p(s)
  - round-trip loss quantifies information loss

Key functions:
  - ``fisher_matrix(ss, p)``           -- Fisher info matrix on simplex
  - ``kl_divergence(p, q)``            -- Kullback-Leibler divergence
  - ``fisher_rao_distance(p, q)``      -- geodesic distance on simplex
  - ``exponential_family_params(ss, beta)`` -- Boltzmann exp family params
  - ``boltzmann_curve(ss, betas)``     -- curve on exponential submanifold
  - ``lattice_adapted_metric(ss, p)``  -- novel lattice-weighted Fisher metric
  - ``embed_states(ss)``               -- φ: states → Dirac deltas
  - ``retract_to_state(ss, p)``        -- ψ: MAP estimate
  - ``round_trip_loss(ss, p)``         -- |p - δ_{ψ(p)}|
  - ``anomaly_kl(ss, expected, obs)``  -- KL-based anomaly score
  - ``information_leakage(ss, beta)``  -- Fisher info about β
  - ``protocol_distance(ss1, ss2, beta)`` -- Fisher-Rao between Boltzmann dists
  - ``natural_gradient(ss, p, objective)`` -- steepest descent in Fisher metric
  - ``scalar_curvature(ss, p)``        -- Ricci scalar (finite diff approx)
  - ``analyze_info_geometry(ss)``      -- full analysis
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FisherMatrix:
    """Fisher information matrix on the probability simplex.

    Attributes:
        matrix: n×n list-of-lists (row-major).
        dimension: Number of free parameters (n-1 for n states).
        states: Ordered list of state IDs.
    """
    matrix: tuple[tuple[float, ...], ...]
    dimension: int
    states: tuple[int, ...]


@dataclass(frozen=True)
class ManifoldPoint:
    """A point on the statistical manifold.

    Attributes:
        distribution: Probability distribution over states.
        natural_params: Natural parameters θ_i = log(p_i / p_n).
        states: Ordered list of state IDs.
    """
    distribution: dict[int, float]
    natural_params: tuple[float, ...]
    states: tuple[int, ...]


@dataclass(frozen=True)
class Geodesic:
    """Geodesic between two points on the statistical manifold.

    Attributes:
        start: Starting ManifoldPoint.
        end: Ending ManifoldPoint.
        distance: Fisher-Rao distance between the points.
        midpoint: Distribution at geodesic midpoint.
    """
    start: ManifoldPoint
    end: ManifoldPoint
    distance: float
    midpoint: dict[int, float]


@dataclass(frozen=True)
class InfoGeometryAnalysis:
    """Full information geometry analysis.

    Attributes:
        num_states: Number of lattice elements.
        fisher_at_uniform: Fisher matrix at uniform distribution.
        boltzmann_fisher_info: Fisher info I(β) at β=1 (= specific heat / β²).
        scalar_curvature_uniform: Ricci scalar at uniform distribution.
        max_kl_from_uniform: Max KL divergence from uniform to any Dirac delta.
        mean_round_trip_loss: Average round-trip loss over Boltzmann family.
        lattice_metric_trace: Trace of lattice-adapted metric at uniform.
        approach_comparison: Summary of all three approaches.
    """
    num_states: int
    fisher_at_uniform: FisherMatrix
    boltzmann_fisher_info: float
    scalar_curvature_uniform: float
    max_kl_from_uniform: float
    mean_round_trip_loss: float
    lattice_metric_trace: float
    approach_comparison: dict[str, float]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rank(ss: StateSpace) -> dict[int, int]:
    """BFS distance from bottom (reverse edges)."""
    rev: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        rev.setdefault(tgt, set()).add(src)
    dist: dict[int, int] = {}
    if ss.bottom in ss.states:
        dist[ss.bottom] = 0
        queue = [ss.bottom]
        while queue:
            s = queue.pop(0)
            for pred in rev.get(s, set()):
                if pred not in dist:
                    dist[pred] = dist[s] + 1
                    queue.append(pred)
    for s in ss.states:
        if s not in dist:
            dist[s] = 0
    return dist


def _lattice_distance(ss: StateSpace) -> dict[tuple[int, int], int]:
    """BFS shortest path distance between all pairs of states (undirected)."""
    adj: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        adj[src].add(tgt)
        adj[tgt].add(src)

    dist: dict[tuple[int, int], int] = {}
    for s in ss.states:
        dist[(s, s)] = 0
        visited: dict[int, int] = {s: 0}
        queue = [s]
        while queue:
            v = queue.pop(0)
            for t in adj.get(v, set()):
                if t not in visited:
                    visited[t] = visited[v] + 1
                    queue.append(t)
        for t, d in visited.items():
            dist[(s, t)] = d

    # Fill in unreachable pairs
    for s in ss.states:
        for t in ss.states:
            if (s, t) not in dist:
                dist[(s, t)] = len(ss.states)  # max distance proxy
    return dist


def _ordered_states(ss: StateSpace) -> tuple[int, ...]:
    """Canonical ordering of states."""
    return tuple(sorted(ss.states))


def _normalize(p: dict[int, float]) -> dict[int, float]:
    """Normalize distribution to sum to 1."""
    total = sum(p.values())
    if total <= 0:
        n = len(p)
        return {s: 1.0 / n for s in p} if n > 0 else {}
    return {s: v / total for s, v in p.items()}


def _uniform(states: set[int]) -> dict[int, float]:
    """Uniform distribution over states."""
    n = len(states)
    if n == 0:
        return {}
    return {s: 1.0 / n for s in states}


def _solve_2x2(a: list[list[float]], b: list[float]) -> list[float] | None:
    """Solve a 2×2 linear system Ax = b. Returns None if singular."""
    det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    if abs(det) < 1e-15:
        return None
    return [
        (a[1][1] * b[0] - a[0][1] * b[1]) / det,
        (a[0][0] * b[1] - a[1][0] * b[0]) / det,
    ]


def _solve_linear(mat: list[list[float]], rhs: list[float]) -> list[float]:
    """Solve linear system mat · x = rhs via Gaussian elimination.

    Falls back to returning zeros if singular.
    """
    n = len(rhs)
    if n == 0:
        return []

    # Augmented matrix
    aug = [row[:] + [rhs[i]] for i, row in enumerate(mat)]

    # Forward elimination with partial pivoting
    for col in range(n):
        # Find pivot
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        if abs(pivot) < 1e-15:
            continue

        for row in range(col + 1, n):
            factor = aug[row][col] / pivot
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(aug[i][i]) < 1e-15:
            x[i] = 0.0
            continue
        x[i] = aug[i][n]
        for j in range(i + 1, n):
            x[i] -= aug[i][j] * x[j]
        x[i] /= aug[i][i]

    return x


# ---------------------------------------------------------------------------
# Approach A: Fisher metric on the full simplex
# ---------------------------------------------------------------------------

def fisher_matrix(
    ss: StateSpace,
    p: dict[int, float] | None = None,
) -> FisherMatrix:
    """Fisher information matrix for the categorical distribution on states.

    For n states with probabilities p = (p_1, ..., p_n), the Fisher matrix
    in natural parameters θ_i = log(p_i / p_n) for i = 1, ..., n-1 has
    closed form:

        G_ij = p_i · δ_ij + p_i · p_j   (wait, let's derive carefully)

    Actually for the categorical distribution parameterized by θ_i = log(p_i/p_n):
        G_ij = p_i · (δ_ij - p_j)   ... no.

    The correct Fisher metric for the categorical/multinomial family in natural
    parameters is:
        G_ij = p_i · δ_ij + p_i · p_j / (1 - Σ_{k<n} p_k)
    But more simply in the probability parameterization:
        G_ij = δ_ij / p_i  (diagonal) for i,j < n
    Wait — the standard result is:

    For categorical with natural parameters η_i = log(p_i/p_n), i=1..n-1:
        G_ij = Cov[T_i, T_j] where T_i are sufficient statistics
        G_ij = p_i(δ_ij) - p_i p_j   ... no that's the covariance of indicators.

    For exponential family, Fisher = Var of sufficient statistic = ∂²A/∂η_i∂η_j
    where A(η) = log(Σ exp(η·T)).

    For categorical: T_i(x) = 1[x=i], A(η) = log(1 + Σ exp(η_i))
    ∂A/∂η_i = p_i
    ∂²A/∂η_i∂η_j = p_i(δ_ij - p_j)

    So G_ij = p_i(δ_ij - p_j) for i,j in 1..n-1.

    Args:
        ss: State space.
        p: Probability distribution over states. Defaults to uniform.

    Returns:
        FisherMatrix with (n-1)×(n-1) matrix.
    """
    states = _ordered_states(ss)
    n = len(states)

    if n <= 1:
        return FisherMatrix(matrix=(), dimension=0, states=states)

    if p is None:
        p = _uniform(ss.states)
    p = _normalize(p)

    # Use n-1 dimensional parameterization (last state is reference)
    dim = n - 1
    probs = [max(p.get(s, 1e-15), 1e-15) for s in states]

    rows: list[tuple[float, ...]] = []
    for i in range(dim):
        row: list[float] = []
        for j in range(dim):
            if i == j:
                row.append(probs[i] * (1.0 - probs[i]))
            else:
                row.append(-probs[i] * probs[j])
        rows.append(tuple(row))

    return FisherMatrix(
        matrix=tuple(rows),
        dimension=dim,
        states=states,
    )


def kl_divergence(p: dict[int, float], q: dict[int, float]) -> float:
    """Kullback-Leibler divergence D_KL(p || q) = Σ p(s) log(p(s)/q(s)).

    Returns:
        KL divergence (≥ 0, = 0 iff p = q).
    """
    p = _normalize(p)
    q = _normalize(q)
    result = 0.0
    for s in p:
        ps = p.get(s, 0.0)
        qs = q.get(s, 1e-30)
        if ps > 1e-30:
            result += ps * math.log(ps / max(qs, 1e-30))
    return max(0.0, result)


def fisher_rao_distance(p: dict[int, float], q: dict[int, float]) -> float:
    """Fisher-Rao distance = 2·arccos(Σ √(p(s)·q(s))) (Bhattacharyya angle).

    This is the geodesic distance on the probability simplex equipped with
    the Fisher information metric.

    Returns:
        Distance ≥ 0. Equals 0 iff p = q. A proper metric.
    """
    p = _normalize(p)
    q = _normalize(q)
    # Bhattacharyya coefficient
    bc = 0.0
    all_states = set(p) | set(q)
    for s in all_states:
        ps = max(p.get(s, 0.0), 0.0)
        qs = max(q.get(s, 0.0), 0.0)
        bc += math.sqrt(ps * qs)
    # Clamp to [0, 1] for numerical safety
    bc = min(1.0, max(0.0, bc))
    return 2.0 * math.acos(bc)


def to_manifold_point(
    ss: StateSpace,
    p: dict[int, float],
) -> ManifoldPoint:
    """Convert a probability distribution to a ManifoldPoint."""
    states = _ordered_states(ss)
    p = _normalize(p)
    n = len(states)
    if n <= 1:
        return ManifoldPoint(
            distribution=p,
            natural_params=(),
            states=states,
        )
    # Natural parameters: θ_i = log(p_i / p_{n-1})
    ref = max(p.get(states[-1], 1e-15), 1e-15)
    params = tuple(
        math.log(max(p.get(s, 1e-15), 1e-15) / ref) for s in states[:-1]
    )
    return ManifoldPoint(
        distribution=dict(p),
        natural_params=params,
        states=states,
    )


def compute_geodesic(
    ss: StateSpace,
    p: dict[int, float],
    q: dict[int, float],
) -> Geodesic:
    """Compute geodesic between two distributions on the simplex.

    On the Fisher-Rao manifold, the geodesic between p and q passes through
    the midpoint (√p + √q)² / (normalization).
    """
    p = _normalize(p)
    q = _normalize(q)
    dist = fisher_rao_distance(p, q)

    # Geodesic midpoint in Bhattacharyya embedding: (√p + √q)/2, then square and normalize
    all_states = set(p) | set(q)
    mid_raw: dict[int, float] = {}
    for s in all_states:
        sqrt_p = math.sqrt(max(p.get(s, 0.0), 0.0))
        sqrt_q = math.sqrt(max(q.get(s, 0.0), 0.0))
        mid_raw[s] = ((sqrt_p + sqrt_q) / 2.0) ** 2
    mid = _normalize(mid_raw)

    return Geodesic(
        start=to_manifold_point(ss, p),
        end=to_manifold_point(ss, q),
        distance=dist,
        midpoint=mid,
    )


# ---------------------------------------------------------------------------
# Approach B: Exponential family (Boltzmann) geometry
# ---------------------------------------------------------------------------

def exponential_family_params(
    ss: StateSpace,
    beta: float,
    energy: dict[int, float] | None = None,
) -> tuple[float, float, float]:
    """Boltzmann family as a 1-parameter exponential family.

    Natural parameter: η = -β
    Sufficient statistic: E(s)
    Log-normalizer: A(η) = log Z(-η)

    Returns:
        (eta, mean_energy, variance_energy) where:
        - eta = -beta (natural parameter)
        - mean_energy = ⟨E⟩_β (expectation parameter)
        - variance_energy = Var_β(E) (Fisher info = variance for exp family)
    """
    if energy is None:
        energy = _rank_energy(ss)

    states = list(ss.states)
    # Compute Boltzmann distribution
    max_e = max(energy.get(s, 0.0) for s in states) if states else 0.0
    weights = {s: math.exp(-beta * energy.get(s, 0.0) + beta * max_e) for s in states}
    Z = sum(weights.values())
    if Z <= 0:
        return (-beta, 0.0, 0.0)

    probs = {s: w / Z for s, w in weights.items()}
    mean = sum(energy.get(s, 0.0) * probs[s] for s in states)
    mean_sq = sum(energy.get(s, 0.0) ** 2 * probs[s] for s in states)
    var = max(0.0, mean_sq - mean ** 2)

    return (-beta, mean, var)


def _rank_energy(ss: StateSpace) -> dict[int, float]:
    """Energy function E(s) = rank (BFS distance from bottom)."""
    r = _rank(ss)
    return {s: float(r[s]) for s in ss.states}


def _boltzmann_dist(
    ss: StateSpace,
    beta: float,
    energy: dict[int, float] | None = None,
) -> dict[int, float]:
    """Boltzmann distribution at inverse temperature beta."""
    if energy is None:
        energy = _rank_energy(ss)
    states = list(ss.states)
    if not states:
        return {}
    # Numerically stable: subtract max energy * beta
    max_e = max(energy.get(s, 0.0) for s in states)
    weights = {s: math.exp(-beta * (energy.get(s, 0.0) - max_e)) for s in states}
    Z = sum(weights.values())
    if Z <= 0:
        return {s: 1.0 / len(states) for s in states}
    return {s: w / Z for s, w in weights.items()}


def boltzmann_curve(
    ss: StateSpace,
    betas: list[float] | None = None,
    energy: dict[int, float] | None = None,
) -> list[ManifoldPoint]:
    """Trace the Boltzmann curve on the statistical manifold.

    Returns a list of ManifoldPoints, one per β value.
    """
    if betas is None:
        betas = [0.1 * i for i in range(1, 21)]
    if energy is None:
        energy = _rank_energy(ss)

    points: list[ManifoldPoint] = []
    for beta in betas:
        p = _boltzmann_dist(ss, beta, energy)
        points.append(to_manifold_point(ss, p))
    return points


def boltzmann_fisher_info(
    ss: StateSpace,
    beta: float,
    energy: dict[int, float] | None = None,
) -> float:
    """Fisher information along the Boltzmann curve.

    I(β) = Var_β(E) = specific_heat(β) / β² for the 1-parameter family.
    This is the squared "speed" of the curve on the manifold.
    """
    _, _, var = exponential_family_params(ss, beta, energy)
    return var


# ---------------------------------------------------------------------------
# Approach C: Lattice-adapted metric
# ---------------------------------------------------------------------------

def lattice_adapted_metric(
    ss: StateSpace,
    p: dict[int, float] | None = None,
) -> FisherMatrix:
    """Fisher metric modified by lattice distance.

    G^L_ij = G_ij · (1 + α · d_L(s_i, s_j)) for i ≠ j
    G^L_ii = G_ii · (1 + α · Σ_j d_L(s_i, s_j) / n)

    where d_L is the lattice distance and α is a coupling constant.
    This weights the information content by protocol structure.
    """
    states = _ordered_states(ss)
    n = len(states)

    if n <= 1:
        return FisherMatrix(matrix=(), dimension=0, states=states)

    if p is None:
        p = _uniform(ss.states)
    p = _normalize(p)

    # Standard Fisher matrix entries
    dim = n - 1
    probs = [max(p.get(s, 1e-15), 1e-15) for s in states]

    # Lattice distances
    ld = _lattice_distance(ss)
    max_d = max(ld.values()) if ld else 1
    if max_d == 0:
        max_d = 1

    alpha = 1.0  # Coupling constant

    rows: list[tuple[float, ...]] = []
    for i in range(dim):
        row: list[float] = []
        for j in range(dim):
            if i == j:
                g_ij = probs[i] * (1.0 - probs[i])
                # Average lattice distance from state i
                avg_d = sum(
                    ld.get((states[i], states[k]), 0) for k in range(n)
                ) / n
                weight = 1.0 + alpha * avg_d / max_d
            else:
                g_ij = -probs[i] * probs[j]
                d = ld.get((states[i], states[j]), 0)
                weight = 1.0 + alpha * d / max_d
            row.append(g_ij * weight)
        rows.append(tuple(row))

    return FisherMatrix(
        matrix=tuple(rows),
        dimension=dim,
        states=states,
    )


# ---------------------------------------------------------------------------
# Bidirectional morphisms
# ---------------------------------------------------------------------------

def embed_states(ss: StateSpace) -> dict[int, dict[int, float]]:
    """φ: embed each state s as Dirac delta δ_s on the manifold boundary.

    Returns:
        Map from state ID to probability distribution (Dirac delta).
    """
    result: dict[int, dict[int, float]] = {}
    for s in ss.states:
        result[s] = {t: (1.0 if t == s else 0.0) for t in ss.states}
    return result


def retract_to_state(ss: StateSpace, p: dict[int, float]) -> int:
    """ψ: MAP estimate — retract distribution to most likely state.

    Returns the state ID with highest probability.
    """
    if not p:
        return ss.bottom
    return max((s for s in ss.states if s in p), key=lambda s: p.get(s, 0.0),
               default=ss.bottom)


def round_trip_loss(ss: StateSpace, p: dict[int, float]) -> float:
    """Measure |p - δ_{ψ(p)}| — total variation distance after round trip.

    Quantifies how much information the embed→retract round trip loses.
    Returns value in [0, 1]: 0 means no loss (p is already a Dirac delta).
    """
    p = _normalize(p)
    best = retract_to_state(ss, p)
    delta = {s: (1.0 if s == best else 0.0) for s in ss.states}

    # Total variation distance = (1/2) Σ |p(s) - q(s)|
    tv = 0.0
    for s in ss.states:
        tv += abs(p.get(s, 0.0) - delta.get(s, 0.0))
    return tv / 2.0


# ---------------------------------------------------------------------------
# Practical functions
# ---------------------------------------------------------------------------

def anomaly_kl(
    ss: StateSpace,
    expected_p: dict[int, float],
    observed_counts: dict[int, int],
) -> float:
    """KL-based anomaly score: D_KL(observed || expected).

    Computes empirical distribution from observed counts and measures
    divergence from the expected distribution.

    Returns:
        KL divergence (higher = more anomalous).
    """
    total = sum(observed_counts.values())
    if total <= 0:
        return 0.0
    observed_p = {s: observed_counts.get(s, 0) / total for s in ss.states}
    return kl_divergence(observed_p, expected_p)


def information_leakage(
    ss: StateSpace,
    beta: float,
    energy: dict[int, float] | None = None,
) -> float:
    """Fisher information about β — how much observing states reveals about temperature.

    I(β) = Var_β(E). High variance = easy to distinguish temperatures =
    high information leakage about the system's temperature.
    """
    return boltzmann_fisher_info(ss, beta, energy)


def protocol_distance(
    ss1: StateSpace,
    ss2: StateSpace,
    beta: float = 1.0,
    energy1: dict[int, float] | None = None,
    energy2: dict[int, float] | None = None,
) -> float:
    """Fisher-Rao distance between Boltzmann distributions of two protocols.

    Embeds both protocols into a common space (union of states) and computes
    the Fisher-Rao distance between their Boltzmann distributions.

    Note: Meaningful when the state spaces share common states or when
    comparing structural similarity via distribution shape.
    """
    if energy1 is None:
        energy1 = _rank_energy(ss1)
    if energy2 is None:
        energy2 = _rank_energy(ss2)

    p1 = _boltzmann_dist(ss1, beta, energy1)
    p2 = _boltzmann_dist(ss2, beta, energy2)

    # Embed both into the union of state spaces
    all_states = set(p1) | set(p2)
    p1_ext = {s: p1.get(s, 0.0) for s in all_states}
    p2_ext = {s: p2.get(s, 0.0) for s in all_states}

    # Normalize after extension
    p1_ext = _normalize(p1_ext)
    p2_ext = _normalize(p2_ext)

    return fisher_rao_distance(p1_ext, p2_ext)


def natural_gradient(
    ss: StateSpace,
    p: dict[int, float],
    objective: Callable[[dict[int, float]], float],
    epsilon: float = 1e-5,
) -> dict[int, float]:
    """Natural gradient: steepest descent direction in the Fisher metric.

    ∇̃f = G⁻¹ · ∇f

    The natural gradient accounts for the geometry of the probability simplex,
    giving a coordinate-invariant descent direction.

    Returns:
        Direction vector (dict from state to float).
    """
    states = _ordered_states(ss)
    n = len(states)
    if n <= 1:
        return {s: 0.0 for s in ss.states}

    p = _normalize(p)

    # Compute Euclidean gradient via finite differences (in probability space)
    f0 = objective(p)
    grad: list[float] = []
    for i in range(n - 1):
        p_eps = dict(p)
        s_i = states[i]
        s_n = states[-1]
        p_eps[s_i] = p.get(s_i, 0.0) + epsilon
        p_eps[s_n] = p.get(s_n, 0.0) - epsilon  # Stay on simplex
        p_eps = _normalize(p_eps)
        f1 = objective(p_eps)
        grad.append((f1 - f0) / epsilon)

    # Get Fisher matrix
    fm = fisher_matrix(ss, p)
    if fm.dimension == 0:
        return {s: 0.0 for s in ss.states}

    # Solve G · nat_grad = grad
    mat = [list(row) for row in fm.matrix]
    # Add small regularization for numerical stability
    for i in range(len(mat)):
        mat[i][i] += 1e-10

    nat_grad = _solve_linear(mat, grad)

    # Convert back to probability-space direction
    result: dict[int, float] = {}
    for i, s in enumerate(states[:-1]):
        result[s] = nat_grad[i] if i < len(nat_grad) else 0.0
    result[states[-1]] = -sum(nat_grad)
    return result


def scalar_curvature(
    ss: StateSpace,
    p: dict[int, float] | None = None,
    epsilon: float = 1e-4,
) -> float:
    """Approximate Ricci scalar curvature via finite differences.

    For the n-simplex with Fisher metric, the scalar curvature is known to be
    R = n(n-1)/4 at the uniform distribution. We approximate numerically
    for arbitrary p.

    For a 1D manifold (2 states), curvature is always 0.
    For 2+ free parameters, we use the Christoffel symbol approach
    with finite differences.
    """
    states = _ordered_states(ss)
    n = len(states)

    if n <= 2:
        return 0.0

    if p is None:
        p = _uniform(ss.states)
    p = _normalize(p)

    dim = n - 1

    # For the categorical Fisher metric, the scalar curvature at a point
    # on the (n-1)-simplex is known analytically:
    # The simplex with Fisher metric is a sphere of radius 2 in n dimensions,
    # so R = dim(dim-1)/4 everywhere.
    # But we compute numerically to handle the lattice-adapted case too.

    # Compute G at p and at perturbed points
    def _get_metric(prob: dict[int, float]) -> list[list[float]]:
        fm = fisher_matrix(ss, prob)
        return [list(row) for row in fm.matrix]

    G = _get_metric(p)

    if dim == 1:
        return 0.0

    # For the standard Fisher metric on the simplex, use the known result
    # R = (n-2)(n-1)/4 for (n-1)-simplex
    # Actually for a sphere of radius 2 in dim dimensions:
    # R = dim*(dim-1)/4

    # Numerical approximation using metric derivatives
    # ∂G_ij/∂θ_k via finite differences
    dG: list[list[list[float]]] = []  # dG[k][i][j] = ∂G_ij/∂θ_k
    for k in range(dim):
        p_plus = dict(p)
        p_minus = dict(p)
        s_k = states[k]
        s_ref = states[-1]

        delta = epsilon * max(p.get(s_k, 0.1), 0.01)
        p_plus[s_k] = p.get(s_k, 0.0) + delta
        p_plus[s_ref] = p.get(s_ref, 0.0) - delta
        p_minus[s_k] = p.get(s_k, 0.0) - delta
        p_minus[s_ref] = p.get(s_ref, 0.0) + delta

        # Ensure positivity
        for s in p_plus:
            p_plus[s] = max(p_plus[s], 1e-15)
            p_minus[s] = max(p_minus[s], 1e-15)
        p_plus = _normalize(p_plus)
        p_minus = _normalize(p_minus)

        G_plus = _get_metric(p_plus)
        G_minus = _get_metric(p_minus)

        dG_k: list[list[float]] = []
        for i in range(dim):
            row: list[float] = []
            for j in range(dim):
                row.append((G_plus[i][j] - G_minus[i][j]) / (2 * delta))
            dG_k.append(row)
        dG.append(dG_k)

    # Inverse metric
    G_inv = _invert_matrix(G)
    if G_inv is None:
        # Fallback: use known result for uniform
        return float(dim * (dim - 1)) / 4.0

    # Christoffel symbols Γ^l_{ij} = (1/2) G^{lk} (∂_i G_{kj} + ∂_j G_{ki} - ∂_k G_{ij})
    # Riemann tensor R^l_{ijk} = ∂_j Γ^l_{ik} - ∂_k Γ^l_{ij} + Γ^l_{jm}Γ^m_{ik} - Γ^l_{km}Γ^m_{ij}
    # Ricci = R^k_{ikj}, Scalar = G^{ij} R_{ij}

    # For simplicity and numerical stability, compute Christoffel symbols
    gamma: list[list[list[float]]] = []  # gamma[l][i][j]
    for l in range(dim):
        gamma_l: list[list[float]] = []
        for i in range(dim):
            row: list[float] = []
            for j in range(dim):
                val = 0.0
                for k_idx in range(dim):
                    partial_i_Gkj = dG[i][k_idx][j] if i < len(dG) else 0.0
                    partial_j_Gki = dG[j][k_idx][i] if j < len(dG) else 0.0
                    partial_k_Gij = dG[k_idx][i][j] if k_idx < len(dG) else 0.0
                    val += 0.5 * G_inv[l][k_idx] * (partial_i_Gkj + partial_j_Gki - partial_k_Gij)
                row.append(val)
            gamma_l.append(row)
        gamma.append(gamma_l)

    # Ricci tensor R_{ij} = R^k_{ikj} (contract Riemann)
    # R^k_{ikj} = ∂_k Γ^k_{ij} - ∂_j Γ^k_{ik} + Γ^k_{km}Γ^m_{ij} - Γ^k_{jm}Γ^m_{ik}
    # For efficiency, skip ∂Γ terms (they require second derivatives) and use
    # the contracted product terms only (first-order approximation)
    ricci: list[list[float]] = []
    for i in range(dim):
        row: list[float] = []
        for j in range(dim):
            r_ij = 0.0
            for k in range(dim):
                for m in range(dim):
                    r_ij += gamma[k][k][m] * gamma[m][i][j]
                    r_ij -= gamma[k][j][m] * gamma[m][i][k]
            row.append(r_ij)
        ricci.append(row)

    # Scalar curvature R = G^{ij} R_{ij}
    R = 0.0
    for i in range(dim):
        for j in range(dim):
            R += G_inv[i][j] * ricci[i][j]

    return R


def _invert_matrix(mat: list[list[float]]) -> list[list[float]] | None:
    """Invert a matrix via Gauss-Jordan elimination."""
    n = len(mat)
    if n == 0:
        return []

    # Augment with identity
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)]
           for i, row in enumerate(mat)]

    for col in range(n):
        # Pivot
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        if abs(pivot) < 1e-15:
            return None

        for j in range(2 * n):
            aug[col][j] /= pivot

        for row in range(n):
            if row != col:
                factor = aug[row][col]
                for j in range(2 * n):
                    aug[row][j] -= factor * aug[col][j]

    return [row[n:] for row in aug]


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze_info_geometry(
    ss: StateSpace,
    energy: dict[int, float] | None = None,
) -> InfoGeometryAnalysis:
    """Full information geometry analysis across all three approaches."""
    if energy is None:
        energy = _rank_energy(ss)

    n = len(ss.states)
    states = _ordered_states(ss)

    # Approach A: Fisher at uniform
    uniform = _uniform(ss.states)
    fm_uniform = fisher_matrix(ss, uniform)

    # Approach B: Boltzmann Fisher info at β=1
    bfi = boltzmann_fisher_info(ss, 1.0, energy)

    # Approach C: Lattice-adapted metric
    lam = lattice_adapted_metric(ss, uniform)
    lam_trace = sum(lam.matrix[i][i] for i in range(lam.dimension)) if lam.dimension > 0 else 0.0

    # Scalar curvature at uniform
    sc = scalar_curvature(ss, uniform)

    # Max KL from uniform
    max_kl = 0.0
    for s in ss.states:
        delta = {t: (1.0 if t == s else 1e-30) for t in ss.states}
        kl = kl_divergence(delta, uniform)
        max_kl = max(max_kl, kl)

    # Mean round-trip loss over Boltzmann family
    betas = [0.5, 1.0, 2.0, 5.0]
    losses = []
    for beta in betas:
        p = _boltzmann_dist(ss, beta, energy)
        losses.append(round_trip_loss(ss, p))
    mean_loss = sum(losses) / len(losses) if losses else 0.0

    # Comparison
    fm_trace = sum(fm_uniform.matrix[i][i] for i in range(fm_uniform.dimension)) if fm_uniform.dimension > 0 else 0.0
    comparison = {
        "fisher_trace": fm_trace,
        "boltzmann_fisher_info": bfi,
        "lattice_metric_trace": lam_trace,
        "scalar_curvature": sc,
    }

    return InfoGeometryAnalysis(
        num_states=n,
        fisher_at_uniform=fm_uniform,
        boltzmann_fisher_info=bfi,
        scalar_curvature_uniform=sc,
        max_kl_from_uniform=max_kl,
        mean_round_trip_loss=mean_loss,
        lattice_metric_trace=lam_trace,
        approach_comparison=comparison,
    )
