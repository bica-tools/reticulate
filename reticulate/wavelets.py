"""Wavelet analysis of session type lattices (Step 31j).

Wavelets provide **localized multi-scale decomposition** of signals on
the protocol lattice.  Unlike eigenvalues (global, non-local), a wavelet
coefficient W(s, t) captures structure near state s at scale t.

A "signal" on L(S) is any function f: states → ℝ (rank, entropy, degree,
Boltzmann probability).  Wavelet decomposition separates f into:
- **Coarse components**: the macro flow (pipeline skeleton)
- **Fine components**: the micro decisions (branch logic)

Three approaches (all implemented):
A. **Spectral graph wavelets** (Hammond et al. 2011) — Laplacian-based
B. **Diffusion wavelets** (Coifman & Maggioni 2006) — random-walk-based
C. **Lattice-native wavelets** (novel) — use lattice distance balls

Engineering payoff:
- Coarse = architecture (risky to change)
- Fine = logic (safe to change)
- Anomaly at (state, scale) = localized problem at specific resolution

Key functions:
  - ``spectral_wavelets(ss, f, scales)``     -- approach A
  - ``diffusion_wavelets(ss, f, scales)``     -- approach B
  - ``lattice_wavelets(ss, f, scales)``       -- approach C
  - ``decompose(ss, f, n_scales)``            -- multi-scale decomposition
  - ``reconstruct(decomposition)``            -- inverse transform
  - ``anomaly_localize(ss, f_exp, f_obs)``    -- find (state, scale) anomalies
  - ``analyze_wavelets(ss)``                  -- full analysis
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WaveletCoefficient:
    """A single wavelet coefficient: value at (state, scale).

    Attributes:
        state: Lattice state ID.
        scale: Scale level (0 = finest, higher = coarser).
        value: Coefficient magnitude.
    """
    state: int
    scale: int
    value: float


@dataclass(frozen=True)
class ScaleDecomposition:
    """Multi-scale decomposition of a signal on the lattice.

    Attributes:
        original: The original signal f.
        coarse: The coarsest approximation (scale = max_scale).
        details: List of detail signals, one per scale (finest to coarsest).
        coefficients: All wavelet coefficients.
        n_scales: Number of scale levels.
        approach: Which approach was used ("spectral", "diffusion", "lattice").
        energy_per_scale: Fraction of total energy in each scale.
    """
    original: dict[int, float]
    coarse: dict[int, float]
    details: list[dict[int, float]]
    coefficients: list[WaveletCoefficient]
    n_scales: int
    approach: str
    energy_per_scale: list[float]


@dataclass(frozen=True)
class WaveletAnalysis:
    """Full wavelet analysis comparing all three approaches.

    Attributes:
        spectral: Decomposition via spectral graph wavelets.
        diffusion: Decomposition via diffusion wavelets.
        lattice: Decomposition via lattice-native wavelets.
        reconstruction_errors: ||f - reconstruct(decompose(f))|| per approach.
        dominant_scale: Scale with most energy per approach.
        num_states: Number of lattice elements.
    """
    spectral: ScaleDecomposition
    diffusion: ScaleDecomposition
    lattice: ScaleDecomposition
    reconstruction_errors: dict[str, float]
    dominant_scale: dict[str, int]
    num_states: int


# ---------------------------------------------------------------------------
# Internal: graph infrastructure
# ---------------------------------------------------------------------------

def _undirected_adj(ss: StateSpace) -> dict[int, set[int]]:
    """Undirected adjacency (symmetric)."""
    adj: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        adj[src].add(tgt)
        adj[tgt].add(src)
    return adj


def _bfs_distance(ss: StateSpace, center: int) -> dict[int, int]:
    """BFS distance from center (undirected)."""
    adj = _undirected_adj(ss)
    dist: dict[int, int] = {center: 0}
    queue = [center]
    while queue:
        s = queue.pop(0)
        for t in adj.get(s, set()):
            if t not in dist:
                dist[t] = dist[s] + 1
                queue.append(t)
    for s in ss.states:
        if s not in dist:
            dist[s] = len(ss.states)  # unreachable
    return dist


def _laplacian(ss: StateSpace) -> tuple[list[int], list[list[float]]]:
    """Graph Laplacian of the undirected Hasse diagram."""
    states = sorted(ss.states)
    n = len(states)
    idx = {s: i for i, s in enumerate(states)}
    adj = _undirected_adj(ss)

    L = [[0.0] * n for _ in range(n)]
    for s in states:
        i = idx[s]
        deg = len(adj[s])
        L[i][i] = float(deg)
        for t in adj[s]:
            j = idx[t]
            L[i][j] -= 1.0

    return states, L


def _eigen_decomposition(
    A: list[list[float]], max_iter: int = 200,
) -> tuple[list[float], list[list[float]]]:
    """Eigenvalues and eigenvectors of a real symmetric matrix.

    Returns (eigenvalues, eigenvectors) where eigenvectors[k] is the k-th
    eigenvector as a list of floats.  Uses QR iteration.
    """
    n = len(A)
    if n == 0:
        return [], []
    if n == 1:
        return [A[0][0]], [[1.0]]

    # Work on a copy
    M = [row[:] for row in A]
    # Accumulate eigenvectors
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for _ in range(max_iter):
        # QR decomposition (Gram-Schmidt)
        Q = [[0.0] * n for _ in range(n)]
        R = [[0.0] * n for _ in range(n)]

        for j in range(n):
            v = [M[i][j] for i in range(n)]
            for i in range(j):
                dot = sum(Q[k][i] * v[k] for k in range(n))
                R[i][j] = dot
                for k in range(n):
                    v[k] -= dot * Q[k][i]
            norm = math.sqrt(sum(x * x for x in v))
            R[j][j] = norm
            if norm > 1e-14:
                for k in range(n):
                    Q[k][j] = v[k] / norm
            else:
                for k in range(n):
                    Q[k][j] = 1.0 if k == j else 0.0

        # M = R * Q
        M_new = [[sum(R[i][k] * Q[k][j] for k in range(n))
                   for j in range(n)] for i in range(n)]

        # V = V * Q
        V_new = [[sum(V[i][k] * Q[k][j] for k in range(n))
                   for j in range(n)] for i in range(n)]

        M = M_new
        V = V_new

        off = sum(abs(M[i][j]) for i in range(n) for j in range(n) if i != j)
        if off < 1e-10 * n:
            break

    eigenvalues = [M[i][i] for i in range(n)]
    eigenvectors = [[V[i][k] for i in range(n)] for k in range(n)]

    # Sort by eigenvalue
    pairs = sorted(zip(eigenvalues, eigenvectors))
    eigenvalues = [p[0] for p in pairs]
    eigenvectors = [p[1] for p in pairs]

    return eigenvalues, eigenvectors


# ---------------------------------------------------------------------------
# Approach A: Spectral graph wavelets
# ---------------------------------------------------------------------------

def _wavelet_kernel(t: float, lam: float) -> float:
    """Mexican hat wavelet kernel g(t·λ) = t·λ · exp(-t·λ)."""
    x = t * lam
    if x > 50:
        return 0.0  # numerical safety
    return x * math.exp(-x)


def _scaling_kernel(t: float, lam: float) -> float:
    """Low-pass scaling kernel h(t·λ) = exp(-t·λ)."""
    x = t * lam
    if x > 50:
        return 0.0
    return math.exp(-x)


def spectral_wavelets(
    ss: StateSpace,
    f: dict[int, float],
    scales: list[float] | None = None,
) -> ScaleDecomposition:
    """Approach A: Spectral graph wavelets (Hammond et al. 2011).

    Uses Laplacian eigenvectors as the frequency basis.
    Wavelet at (state s, scale t):
        ψ_{s,t}(x) = Σ_k g(t·λ_k) · φ_k(s) · φ_k(x)
    Coefficient:
        W_f(s,t) = ⟨f, ψ_{s,t}⟩ = Σ_k g(t·λ_k) · φ_k(s) · f̂_k
    """
    states, L = _laplacian(ss)
    n = len(states)
    idx = {s: i for i, s in enumerate(states)}

    if n <= 1:
        return _trivial_decomposition(f, "spectral")

    eigenvalues, eigenvectors = _eigen_decomposition(L)

    if scales is None:
        # Default: logarithmically spaced scales
        lam_max = max(max(eigenvalues), 1e-10)
        scales = [2.0 ** k / lam_max for k in range(max(1, int(math.log2(n))))]

    # Forward transform: f̂_k = Σ_i φ_k(i) · f(states[i])
    f_hat = []
    for k in range(n):
        f_hat.append(sum(eigenvectors[k][i] * f.get(states[i], 0.0) for i in range(n)))

    # Wavelet coefficients
    coefficients: list[WaveletCoefficient] = []
    details: list[dict[int, float]] = []
    energy_per_scale: list[float] = []

    for scale_idx, t in enumerate(scales):
        detail: dict[int, float] = {}
        scale_energy = 0.0
        for s in states:
            i = idx[s]
            # W_f(s,t) = Σ_k g(t·λ_k) · φ_k(s) · f̂_k
            val = sum(
                _wavelet_kernel(t, eigenvalues[k]) * eigenvectors[k][i] * f_hat[k]
                for k in range(n)
            )
            detail[s] = val
            coefficients.append(WaveletCoefficient(state=s, scale=scale_idx, value=val))
            scale_energy += val * val
        details.append(detail)
        energy_per_scale.append(scale_energy)

    # Coarse approximation via scaling function
    t_max = scales[-1] if scales else 1.0
    coarse: dict[int, float] = {}
    for s in states:
        i = idx[s]
        val = sum(
            _scaling_kernel(t_max, eigenvalues[k]) * eigenvectors[k][i] * f_hat[k]
            for k in range(n)
        )
        coarse[s] = val

    # Normalize energy fractions
    total_energy = sum(energy_per_scale) or 1.0
    energy_per_scale = [e / total_energy for e in energy_per_scale]

    return ScaleDecomposition(
        original=dict(f),
        coarse=coarse,
        details=details,
        coefficients=coefficients,
        n_scales=len(scales),
        approach="spectral",
        energy_per_scale=energy_per_scale,
    )


# ---------------------------------------------------------------------------
# Approach B: Diffusion wavelets
# ---------------------------------------------------------------------------

def diffusion_wavelets(
    ss: StateSpace,
    f: dict[int, float],
    scales: list[int] | None = None,
) -> ScaleDecomposition:
    """Approach B: Diffusion wavelets.

    Uses powers of the diffusion operator T = D⁻¹A.
    Scale t = t-step random walk averaging.
    Detail at scale t = T^t f - T^{t+1} f.
    """
    adj = _undirected_adj(ss)
    states = sorted(ss.states)
    n = len(states)
    idx = {s: i for i, s in enumerate(states)}

    if n <= 1:
        return _trivial_decomposition(f, "diffusion")

    if scales is None:
        scales = list(range(max(1, int(math.log2(n)) + 1)))

    # Build diffusion matrix T = D⁻¹A
    T = [[0.0] * n for _ in range(n)]
    for s in states:
        i = idx[s]
        deg = len(adj[s])
        if deg > 0:
            for t in adj[s]:
                j = idx[t]
                T[i][j] = 1.0 / deg

    # Compute T^k f for each scale k
    # Start with f as a vector
    current = [f.get(states[i], 0.0) for i in range(n)]
    smoothed = [list(current)]  # T^0 f = f

    for k in range(max(scales) + 1):
        next_vec = [0.0] * n
        for i in range(n):
            next_vec[i] = sum(T[i][j] * current[j] for j in range(n))
        current = next_vec
        smoothed.append(list(current))

    # Details = T^k f - T^{k+1} f (difference between consecutive smoothings)
    details: list[dict[int, float]] = []
    coefficients: list[WaveletCoefficient] = []
    energy_per_scale: list[float] = []

    for scale_idx, k in enumerate(scales):
        detail: dict[int, float] = {}
        scale_energy = 0.0
        for i, s in enumerate(states):
            if k + 1 < len(smoothed):
                val = smoothed[k][i] - smoothed[k + 1][i]
            else:
                val = smoothed[k][i]
            detail[s] = val
            coefficients.append(WaveletCoefficient(state=s, scale=scale_idx, value=val))
            scale_energy += val * val
        details.append(detail)
        energy_per_scale.append(scale_energy)

    # Coarse = most smoothed version
    max_k = max(scales) + 1 if scales else 1
    coarse_idx = min(max_k, len(smoothed) - 1)
    coarse = {states[i]: smoothed[coarse_idx][i] for i in range(n)}

    total_energy = sum(energy_per_scale) or 1.0
    energy_per_scale = [e / total_energy for e in energy_per_scale]

    return ScaleDecomposition(
        original=dict(f),
        coarse=coarse,
        details=details,
        coefficients=coefficients,
        n_scales=len(scales),
        approach="diffusion",
        energy_per_scale=energy_per_scale,
    )


# ---------------------------------------------------------------------------
# Approach C: Lattice-native wavelets (novel)
# ---------------------------------------------------------------------------

def lattice_wavelets(
    ss: StateSpace,
    f: dict[int, float],
    scales: list[int] | None = None,
) -> ScaleDecomposition:
    """Approach C: Lattice-native wavelets (novel construction).

    Uses lattice distance balls: B(s, t) = {x : d(s, x) ≤ t}.
    Average at scale t: A_t f(s) = mean(f(x) for x in B(s, t)).
    Wavelet coefficient: W(s, t) = A_t f(s) - A_{t+1} f(s).

    This is a Haar-like construction native to the lattice ordering.
    """
    states = sorted(ss.states)
    n = len(states)

    if n <= 1:
        return _trivial_decomposition(f, "lattice")

    # Precompute all pairwise distances
    dist_matrix: dict[int, dict[int, int]] = {}
    for s in states:
        dist_matrix[s] = _bfs_distance(ss, s)

    max_dist = max(
        dist_matrix[s][t] for s in states for t in states
        if dist_matrix[s][t] < len(states)
    )

    if scales is None:
        scales = list(range(max_dist + 1))

    # Compute ball averages at each scale
    averages: list[dict[int, float]] = []
    for t in range(max(scales) + 2):
        avg: dict[int, float] = {}
        for s in states:
            ball = [x for x in states if dist_matrix[s].get(x, n) <= t]
            if ball:
                avg[s] = sum(f.get(x, 0.0) for x in ball) / len(ball)
            else:
                avg[s] = f.get(s, 0.0)
        averages.append(avg)

    # Details = difference between consecutive ball averages
    details: list[dict[int, float]] = []
    coefficients: list[WaveletCoefficient] = []
    energy_per_scale: list[float] = []

    for scale_idx, t in enumerate(scales):
        detail: dict[int, float] = {}
        scale_energy = 0.0
        for s in states:
            if t < len(averages) and t + 1 < len(averages):
                val = averages[t][s] - averages[t + 1][s]
            elif t < len(averages):
                val = averages[t][s]
            else:
                val = 0.0
            detail[s] = val
            coefficients.append(WaveletCoefficient(state=s, scale=scale_idx, value=val))
            scale_energy += val * val
        details.append(detail)
        energy_per_scale.append(scale_energy)

    # Coarse = largest ball average
    max_t = min(max(scales) + 1, len(averages) - 1) if scales else 0
    coarse = dict(averages[max_t])

    total_energy = sum(energy_per_scale) or 1.0
    energy_per_scale = [e / total_energy for e in energy_per_scale]

    return ScaleDecomposition(
        original=dict(f),
        coarse=coarse,
        details=details,
        coefficients=coefficients,
        n_scales=len(scales),
        approach="lattice",
        energy_per_scale=energy_per_scale,
    )


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def reconstruct(decomp: ScaleDecomposition) -> dict[int, float]:
    """Reconstruct signal from wavelet decomposition.

    For diffusion and lattice wavelets: f ≈ coarse + Σ details.
    (Not exact for spectral wavelets due to non-tight frame.)
    """
    result: dict[int, float] = dict(decomp.coarse)
    for detail in decomp.details:
        for s, v in detail.items():
            result[s] = result.get(s, 0.0) + v
    return result


def reconstruction_error(decomp: ScaleDecomposition) -> float:
    """||f - reconstruct(decompose(f))|| in L2 norm."""
    recon = reconstruct(decomp)
    return math.sqrt(sum(
        (decomp.original.get(s, 0.0) - recon.get(s, 0.0)) ** 2
        for s in decomp.original
    ))


# ---------------------------------------------------------------------------
# Thresholding / compression
# ---------------------------------------------------------------------------

def threshold_coefficients(
    decomp: ScaleDecomposition,
    threshold: float,
) -> ScaleDecomposition:
    """Zero out wavelet coefficients below threshold (compression)."""
    new_details = []
    new_coeffs = []
    new_energy = []

    for scale_idx, detail in enumerate(decomp.details):
        new_detail: dict[int, float] = {}
        scale_e = 0.0
        for s, v in detail.items():
            if abs(v) >= threshold:
                new_detail[s] = v
                scale_e += v * v
            else:
                new_detail[s] = 0.0
            new_coeffs.append(WaveletCoefficient(state=s, scale=scale_idx, value=new_detail[s]))
        new_details.append(new_detail)
        new_energy.append(scale_e)

    total = sum(new_energy) or 1.0
    new_energy = [e / total for e in new_energy]

    return ScaleDecomposition(
        original=decomp.original,
        coarse=decomp.coarse,
        details=new_details,
        coefficients=new_coeffs,
        n_scales=decomp.n_scales,
        approach=decomp.approach,
        energy_per_scale=new_energy,
    )


# ---------------------------------------------------------------------------
# Multi-scale embedding (φ)
# ---------------------------------------------------------------------------

def wavelet_embedding(
    ss: StateSpace,
    f: dict[int, float],
    n_scales: int = 3,
) -> dict[int, tuple[float, ...]]:
    """φ: L(S) → wavelet space.

    Each state maps to its wavelet coefficient vector across all scales.
    """
    decomp = lattice_wavelets(ss, f)
    embedding: dict[int, tuple[float, ...]] = {}
    for s in ss.states:
        vec = tuple(
            detail.get(s, 0.0)
            for detail in decomp.details[:n_scales]
        )
        embedding[s] = vec
    return embedding


# ---------------------------------------------------------------------------
# Anomaly localization
# ---------------------------------------------------------------------------

def anomaly_localize(
    ss: StateSpace,
    f_expected: dict[int, float],
    f_observed: dict[int, float],
    n_scales: int | None = None,
) -> list[WaveletCoefficient]:
    """Find (state, scale) pairs where observed deviates from expected.

    Returns coefficients of the residual f_observed - f_expected,
    sorted by absolute value (largest deviation first).
    """
    residual = {s: f_observed.get(s, 0.0) - f_expected.get(s, 0.0) for s in ss.states}
    decomp = lattice_wavelets(ss, residual)
    sorted_coeffs = sorted(decomp.coefficients, key=lambda c: abs(c.value), reverse=True)
    return sorted_coeffs


# ---------------------------------------------------------------------------
# Change impact analysis
# ---------------------------------------------------------------------------

def change_impact(
    ss: StateSpace,
    f_before: dict[int, float],
    f_after: dict[int, float],
) -> dict[int, float]:
    """Analyze which scales are affected by a change.

    Returns energy of the change signal per scale.
    """
    delta = {s: f_after.get(s, 0.0) - f_before.get(s, 0.0) for s in ss.states}
    decomp = lattice_wavelets(ss, delta)
    return {i: e for i, e in enumerate(decomp.energy_per_scale)}


# ---------------------------------------------------------------------------
# Convenience: default signal
# ---------------------------------------------------------------------------

def _default_signal(ss: StateSpace) -> dict[int, float]:
    """Default signal: rank (distance from bottom)."""
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
    return {s: float(dist[s]) for s in ss.states}


def _trivial_decomposition(f: dict[int, float], approach: str) -> ScaleDecomposition:
    """Trivial decomposition for single-state lattices."""
    return ScaleDecomposition(
        original=dict(f),
        coarse=dict(f),
        details=[],
        coefficients=[],
        n_scales=0,
        approach=approach,
        energy_per_scale=[],
    )


# ---------------------------------------------------------------------------
# Unified decomposition
# ---------------------------------------------------------------------------

def decompose(
    ss: StateSpace,
    f: dict[int, float] | None = None,
    n_scales: int | None = None,
    approach: str = "lattice",
) -> ScaleDecomposition:
    """Decompose a signal using the specified approach."""
    if f is None:
        f = _default_signal(ss)

    if approach == "spectral":
        return spectral_wavelets(ss, f)
    elif approach == "diffusion":
        return diffusion_wavelets(ss, f)
    elif approach == "lattice":
        return lattice_wavelets(ss, f)
    else:
        raise ValueError(f"Unknown approach: {approach}. Use 'spectral', 'diffusion', or 'lattice'.")


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze_wavelets(ss: StateSpace) -> WaveletAnalysis:
    """Full wavelet analysis comparing all three approaches."""
    f = _default_signal(ss)

    spec = spectral_wavelets(ss, f)
    diff = diffusion_wavelets(ss, f)
    latt = lattice_wavelets(ss, f)

    errors = {
        "spectral": reconstruction_error(spec),
        "diffusion": reconstruction_error(diff),
        "lattice": reconstruction_error(latt),
    }

    def dominant(decomp: ScaleDecomposition) -> int:
        if not decomp.energy_per_scale:
            return 0
        return max(range(len(decomp.energy_per_scale)),
                   key=lambda i: decomp.energy_per_scale[i])

    dom = {
        "spectral": dominant(spec),
        "diffusion": dominant(diff),
        "lattice": dominant(latt),
    }

    return WaveletAnalysis(
        spectral=spec,
        diffusion=diff,
        lattice=latt,
        reconstruction_errors=errors,
        dominant_scale=dom,
        num_states=len(ss.states),
    )
