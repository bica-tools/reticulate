"""Persistent spectral protocol fingerprints (Step 31k).

Combines spectral graph theory with persistent filtrations. For each
filtration level ``t``, we form the subgraph ``G_t`` containing all
vertices and Hasse edges whose filtration value is at most ``t``. We
then compute the spectrum of its (graph) Laplacian ``L(G_t)``. Tracking
how each eigenvalue evolves as ``t`` grows produces a *persistent
spectral fingerprint* of the protocol.

Motivation
----------
Static spectral invariants (algebraic connectivity, spectral radius,
graph energy) provide a single snapshot of protocol structure. Static
persistent homology tracks the birth and death of topological features
(H_0 components, H_1 loops) but ignores the geometric *weight* of
those features. Persistent spectral analysis interpolates between the
two: eigenvalue-zero multiplicity tracks components (recovering H_0
persistence), while non-zero eigenvalues measure bottlenecks,
expansion, and subgraph mixing at every filtration level.

Public API
----------
- ``persistent_laplacian_spectra(ss)`` -- list of (level, eigenvalues)
- ``persistent_fiedler_trace(ss)``     -- algebraic connectivity per level
- ``spectral_persistence_diagram(ss)`` -- birth/death pairs for λ_k
- ``persistent_spectral_fingerprint(ss)`` -- compact summary vector
- ``fingerprint_distance(f1, f2)``     -- L^∞ distance in fingerprint space
- ``analyze_persistent_spectrum(ss)``  -- full analysis

All functions are pure, deterministic, and side-effect free. They
depend only on ``reticulate.statespace`` and ``reticulate.matrix``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace

from reticulate.matrix import _eigenvalues_symmetric
from reticulate.persistent_homology import (
    _rank,
    _covering_relations,
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpectralSnapshot:
    """Spectrum of G_t at one filtration level.

    Attributes:
        level: Filtration level ``t``.
        num_vertices: Vertices present at level ``t``.
        num_edges: Hasse edges present at level ``t``.
        eigenvalues: Sorted eigenvalues of the Laplacian of G_t.
    """
    level: float
    num_vertices: int
    num_edges: int
    eigenvalues: tuple[float, ...]

    @property
    def fiedler(self) -> float:
        """Second-smallest eigenvalue (algebraic connectivity of G_t)."""
        if len(self.eigenvalues) < 2:
            return 0.0
        return self.eigenvalues[1]

    @property
    def spectral_radius(self) -> float:
        """Largest Laplacian eigenvalue."""
        if not self.eigenvalues:
            return 0.0
        return max(self.eigenvalues)

    @property
    def zero_multiplicity(self) -> int:
        """Number of (near-)zero eigenvalues = number of components."""
        return sum(1 for e in self.eigenvalues if abs(e) < 1e-9)


@dataclass(frozen=True)
class SpectralPersistencePair:
    """Birth/death of a tracked eigenvalue across the filtration.

    Attributes:
        index: Eigenvalue index (0 = smallest, sorted ascending at birth).
        birth: Level at which the eigenvalue first appears (vertex appears).
        death: Level at which the eigenvalue merges into another component
               or stabilises; ``inf`` for essential eigenvalues.
        birth_value: Eigenvalue at birth.
        death_value: Eigenvalue at death (NaN if infinite).
    """
    index: int
    birth: float
    death: float
    birth_value: float
    death_value: float

    @property
    def persistence(self) -> float:
        if math.isinf(self.death):
            return float('inf')
        return self.death - self.birth


@dataclass(frozen=True)
class PersistentSpectralFingerprint:
    """Compact fingerprint used for protocol comparison.

    Attributes:
        num_levels: Number of filtration snapshots.
        fiedler_trace: Fiedler value at every level.
        radius_trace: Spectral radius at every level.
        zero_mult_trace: Component count at every level.
        energy_trace: Graph energy (Σ|λ|) at every level.
        total_spectral_persistence: Sum of persistences of non-zero eigenvalues.
        max_fiedler: Largest Fiedler value across the filtration.
        final_spectrum: Eigenvalues at the final level.
    """
    num_levels: int
    fiedler_trace: tuple[float, ...]
    radius_trace: tuple[float, ...]
    zero_mult_trace: tuple[int, ...]
    energy_trace: tuple[float, ...]
    total_spectral_persistence: float
    max_fiedler: float
    final_spectrum: tuple[float, ...]

    def as_vector(self) -> list[float]:
        """Project the fingerprint to a flat numeric vector."""
        return (
            list(self.fiedler_trace)
            + list(self.radius_trace)
            + [float(m) for m in self.zero_mult_trace]
            + list(self.energy_trace)
            + [self.total_spectral_persistence, self.max_fiedler]
        )


@dataclass(frozen=True)
class PersistentSpectralAnalysis:
    """Full persistent spectral analysis of a state space."""
    snapshots: tuple[SpectralSnapshot, ...]
    pairs: tuple[SpectralPersistencePair, ...]
    fingerprint: PersistentSpectralFingerprint
    num_states: int


# ---------------------------------------------------------------------------
# Filtration construction
# ---------------------------------------------------------------------------

def _vertex_filtration_rank(ss: "StateSpace") -> dict[int, float]:
    """Vertex filtration value = rank (BFS distance from bottom)."""
    r = _rank(ss)
    return {s: float(r.get(s, 0)) for s in ss.states}


def _vertex_filtration_reverse_rank(ss: "StateSpace") -> dict[int, float]:
    r = _rank(ss)
    max_r = max(r.values()) if r else 0
    return {s: float(max_r - r.get(s, 0)) for s in ss.states}


def _levels(vertex_filt: dict[int, float], covers: list[tuple[int, int]]) -> list[float]:
    """Distinct filtration levels (vertex ranks + edge ranks), sorted."""
    levels: set[float] = set()
    for v, lv in vertex_filt.items():
        levels.add(lv)
    for s, t in covers:
        levels.add(max(vertex_filt.get(s, 0.0), vertex_filt.get(t, 0.0)))
    return sorted(levels)


def _laplacian_of_subgraph(
    vertices: list[int],
    edges: list[tuple[int, int]],
) -> list[list[float]]:
    """Build the (dense) Laplacian of the undirected subgraph."""
    n = len(vertices)
    idx = {v: i for i, v in enumerate(vertices)}
    L = [[0.0] * n for _ in range(n)]
    for a, b in edges:
        if a not in idx or b not in idx or a == b:
            continue
        i, j = idx[a], idx[b]
        L[i][j] -= 1.0
        L[j][i] -= 1.0
        L[i][i] += 1.0
        L[j][j] += 1.0
    return L


def _subgraph_eigenvalues(
    vertices: list[int],
    edges: list[tuple[int, int]],
) -> list[float]:
    """Sorted ascending eigenvalues of the Laplacian of the given subgraph."""
    n = len(vertices)
    if n == 0:
        return []
    if n == 1:
        return [0.0]
    L = _laplacian_of_subgraph(vertices, edges)
    return sorted(_eigenvalues_symmetric(L))


# ---------------------------------------------------------------------------
# Public: per-level spectra
# ---------------------------------------------------------------------------

def persistent_laplacian_spectra(
    ss: "StateSpace",
    filtration: str = "rank",
) -> list[SpectralSnapshot]:
    """Compute the Laplacian spectrum of G_t for each filtration level t.

    Parameters
    ----------
    ss:
        State space.
    filtration:
        Either ``"rank"`` (bottom→top) or ``"reverse_rank"`` (top→bottom).
    """
    if filtration == "rank":
        vfilt = _vertex_filtration_rank(ss)
    elif filtration == "reverse_rank":
        vfilt = _vertex_filtration_reverse_rank(ss)
    else:
        raise ValueError(f"Unknown filtration: {filtration!r}")

    covers = _covering_relations(ss)
    levels = _levels(vfilt, covers)

    snapshots: list[SpectralSnapshot] = []
    for t in levels:
        verts = [v for v in ss.states if vfilt.get(v, 0.0) <= t + 1e-12]
        edges = [
            (s, u) for (s, u) in covers
            if max(vfilt.get(s, 0.0), vfilt.get(u, 0.0)) <= t + 1e-12
            and s in set(verts) and u in set(verts)
        ]
        eigs = _subgraph_eigenvalues(verts, edges)
        snapshots.append(
            SpectralSnapshot(
                level=t,
                num_vertices=len(verts),
                num_edges=len(edges),
                eigenvalues=tuple(eigs),
            )
        )
    return snapshots


def persistent_fiedler_trace(
    ss: "StateSpace",
    filtration: str = "rank",
) -> list[tuple[float, float]]:
    """Trace of the Fiedler value across the filtration.

    Returns list of (level, fiedler) pairs.
    """
    snaps = persistent_laplacian_spectra(ss, filtration=filtration)
    return [(snap.level, snap.fiedler) for snap in snaps]


# ---------------------------------------------------------------------------
# Spectral persistence diagram
# ---------------------------------------------------------------------------

def spectral_persistence_diagram(
    ss: "StateSpace",
    filtration: str = "rank",
) -> list[SpectralPersistencePair]:
    """Extract birth/death pairs for each indexed eigenvalue.

    We index eigenvalues by their position (smallest-to-largest) at the
    final filtration level, then walk backwards and look for the last
    level at which index ``k`` was present. The *birth* of eigenvalue k
    is the first level at which the snapshot had at least ``k+1``
    eigenvalues; the *death* of eigenvalue k's zero-status (for k>0)
    is the first level where it becomes non-zero (components merging).

    The zero-index (``k=0``) is always essential: every non-empty graph
    has at least one zero Laplacian eigenvalue.
    """
    snaps = persistent_laplacian_spectra(ss, filtration=filtration)
    if not snaps:
        return []

    final = snaps[-1]
    pairs: list[SpectralPersistencePair] = []

    # Determine, for each index k, the first level at which
    #   (a) the snapshot has ≥ k+1 eigenvalues (birth),
    #   (b) the eigenvalue at index k becomes strictly positive (death of
    #       zero-status) -- only relevant for tracking component merges.
    m = len(final.eigenvalues)
    for k in range(m):
        birth_level = math.inf
        birth_val = 0.0
        death_level = math.inf
        death_val = float('nan')
        for snap in snaps:
            if len(snap.eigenvalues) > k:
                if math.isinf(birth_level):
                    birth_level = snap.level
                    birth_val = snap.eigenvalues[k]
                if k == 0:
                    # component count == 1 always eventually
                    continue
                # Track when eigenvalue k leaves the zero cluster
                if snap.eigenvalues[k] > 1e-9 and math.isinf(death_level):
                    death_level = snap.level
                    death_val = snap.eigenvalues[k]
        if math.isinf(birth_level):
            continue
        pairs.append(
            SpectralPersistencePair(
                index=k,
                birth=birth_level,
                death=death_level,
                birth_value=birth_val,
                death_value=death_val,
            )
        )
    return pairs


# ---------------------------------------------------------------------------
# Fingerprint and distance
# ---------------------------------------------------------------------------

def persistent_spectral_fingerprint(
    ss: "StateSpace",
    filtration: str = "rank",
) -> PersistentSpectralFingerprint:
    """Compact fingerprint summarising the persistent spectrum."""
    snaps = persistent_laplacian_spectra(ss, filtration=filtration)
    if not snaps:
        return PersistentSpectralFingerprint(
            num_levels=0,
            fiedler_trace=(),
            radius_trace=(),
            zero_mult_trace=(),
            energy_trace=(),
            total_spectral_persistence=0.0,
            max_fiedler=0.0,
            final_spectrum=(),
        )

    fiedler = tuple(snap.fiedler for snap in snaps)
    radius = tuple(snap.spectral_radius for snap in snaps)
    zeros = tuple(snap.zero_multiplicity for snap in snaps)
    energy = tuple(sum(abs(e) for e in snap.eigenvalues) for snap in snaps)

    pairs = spectral_persistence_diagram(ss, filtration=filtration)
    total_pers = 0.0
    for p in pairs:
        if p.index == 0:
            continue
        if math.isinf(p.death):
            continue
        total_pers += max(0.0, p.death - p.birth)

    return PersistentSpectralFingerprint(
        num_levels=len(snaps),
        fiedler_trace=fiedler,
        radius_trace=radius,
        zero_mult_trace=zeros,
        energy_trace=energy,
        total_spectral_persistence=total_pers,
        max_fiedler=max(fiedler) if fiedler else 0.0,
        final_spectrum=snaps[-1].eigenvalues,
    )


def fingerprint_distance(
    f1: PersistentSpectralFingerprint,
    f2: PersistentSpectralFingerprint,
) -> float:
    """L^∞ distance between two fingerprint vectors (shorter padded w/ 0)."""
    v1 = f1.as_vector()
    v2 = f2.as_vector()
    n = max(len(v1), len(v2))
    v1 = v1 + [0.0] * (n - len(v1))
    v2 = v2 + [0.0] * (n - len(v2))
    return max(abs(a - b) for a, b in zip(v1, v2)) if n > 0 else 0.0


# ---------------------------------------------------------------------------
# Bidirectional morphisms (Step 31k contribution)
# ---------------------------------------------------------------------------

def phi_lattice_to_spectrum(ss: "StateSpace") -> PersistentSpectralFingerprint:
    """Forward morphism φ: L(S) → Fingerprint space.

    Sends a state-space lattice to its persistent spectral fingerprint.
    Deterministic, order-sensitive, compositional in a controlled sense
    (see main paper, Theorem on product composition).
    """
    return persistent_spectral_fingerprint(ss)


def psi_spectrum_to_action(
    fingerprint: PersistentSpectralFingerprint,
    bottleneck_threshold: float = 0.5,
    complexity_threshold: float = 2.0,
) -> dict[str, object]:
    """Backward morphism ψ: Fingerprint → engineering actions.

    Maps a persistent spectral fingerprint to a dictionary of concrete
    safety/monitoring/design/testing recommendations:

    - ``bottleneck``: True if any level has Fiedler < threshold
                      (recommendation: add redundant paths).
    - ``complexity_alert``: True if total spectral persistence > threshold
                            (recommendation: refactor / split).
    - ``fragile_levels``: list of (level, fiedler) pairs where fiedler
                          dropped below the threshold.
    - ``monitoring_level``: suggested filtration level at which to place
                            a runtime monitor (the level of minimum
                            Fiedler value).
    - ``stability_score``: ratio of non-fragile levels (higher = safer).
    """
    trace = list(fingerprint.fiedler_trace)
    fragile = [
        (i, f) for i, f in enumerate(trace) if f < bottleneck_threshold and f > 0
    ]
    bottleneck = any(f < bottleneck_threshold for f in trace if f > 0)
    complexity_alert = fingerprint.total_spectral_persistence > complexity_threshold

    if trace:
        positive = [f for f in trace if f > 0]
        if positive:
            min_f = min(positive)
            monitor_level = trace.index(min_f)
        else:
            monitor_level = 0
    else:
        monitor_level = 0

    stability = (
        1.0 - (len(fragile) / len(trace)) if trace else 1.0
    )

    return {
        "bottleneck": bottleneck,
        "complexity_alert": complexity_alert,
        "fragile_levels": fragile,
        "monitoring_level": monitor_level,
        "stability_score": stability,
    }


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze_persistent_spectrum(
    ss: "StateSpace",
    filtration: str = "rank",
) -> PersistentSpectralAnalysis:
    """Run the full persistent spectral analysis on ``ss``."""
    snaps = persistent_laplacian_spectra(ss, filtration=filtration)
    pairs = spectral_persistence_diagram(ss, filtration=filtration)
    fp = persistent_spectral_fingerprint(ss, filtration=filtration)
    return PersistentSpectralAnalysis(
        snapshots=tuple(snaps),
        pairs=tuple(pairs),
        fingerprint=fp,
        num_states=len(ss.states),
    )
