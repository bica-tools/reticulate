"""The Thermodynamic Morphism — Abstract Interpretation via Statistical Mechanics (Step 600d).

The mapping φ: L(S) → ThermodynamicSpace is a structure-preserving morphism.
When paired with its adjoint γ, it forms a **Galois connection** — which IS
abstract interpretation (Cousot & Cousot 1977).

The thermodynamic domain is a sound abstraction of the protocol:
- φ(s) = "thermodynamic shadow" of protocol state s
- γ(t) = "most precise protocol state" consistent with observation t
- Soundness: φ(x) ≤ y ⟺ x ≤ γ(y)

**Software consequences** (every concept maps to practice):
1. Protocol execution = cooling (energy decreases → safety)
2. Entropy = security metric (high S = unpredictable = secure)
3. Phase transitions = safety boundaries
4. RG flow = principled protocol simplification
5. Continuous monitoring via thermodynamic observables
6. Temperature = strictness knob for protocol enforcement

Key functions:
  - ``thermo_embedding(ss)``          -- φ: states → observable vectors
  - ``energy_galois(ss)``             -- Galois connection via energy
  - ``is_cooling(ss, path)``          -- does execution decrease energy?
  - ``security_score(ss, beta)``      -- entropy-based security
  - ``rg_flow(ss)``                   -- congruence chain + thermodynamics
  - ``anomaly_score(ss, beta, state)``-- deviation from Boltzmann
  - ``design_temperature(ss, target)``-- find β for desired entropy
  - ``analyze_thermo_morphism(ss)``   -- full analysis
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
class ThermoObservable:
    """Thermodynamic observable vector for a single state.

    Attributes:
        state: State ID in L(S).
        energy: E(s) = rank (distance from bottom).
        degree: Out-degree (number of enabled transitions).
        local_entropy: Shannon entropy of outgoing transition probabilities.
    """
    state: int
    energy: float
    degree: int
    local_entropy: float


@dataclass(frozen=True)
class ThermoEmbedding:
    """The thermodynamic embedding φ: L(S) → observable space.

    Attributes:
        observables: Map from state ID to ThermoObservable.
        is_order_preserving: True iff energy is monotone (higher states = higher energy).
        energy_range: (E_min, E_max).
    """
    observables: dict[int, ThermoObservable]
    is_order_preserving: bool
    energy_range: tuple[float, float]


@dataclass(frozen=True)
class ThermoGalois:
    """Galois connection between L(S) and energy-ordered chain.

    Attributes:
        alpha: Abstraction map (state → energy level).
        gamma: Concretization map (energy level → set of states at that level).
        energy_levels: Distinct energy levels in ascending order.
        is_valid: True iff the adjunction holds.
    """
    alpha: dict[int, float]
    gamma: dict[float, frozenset[int]]
    energy_levels: tuple[float, ...]
    is_valid: bool


@dataclass(frozen=True)
class RGFlowStep:
    """One coarse-graining step in the RG flow.

    Attributes:
        level: Step number in the flow (0 = original).
        num_states: Number of states at this level.
        partition_size: Number of congruence classes.
        mean_energy: Average energy at this level.
        entropy: Thermodynamic entropy at β=1.
        free_energy: Free energy at β=1.
    """
    level: int
    num_states: int
    partition_size: int
    mean_energy: float
    entropy: float
    free_energy: float


@dataclass(frozen=True)
class RGFlow:
    """Renormalization group flow: sequence of coarse-grainings.

    Attributes:
        steps: List of RG flow steps.
        converged: True iff the flow reached a fixed point.
        fixed_point_states: Number of states at the fixed point.
        total_levels: Number of coarse-graining levels.
    """
    steps: list[RGFlowStep]
    converged: bool
    fixed_point_states: int
    total_levels: int


@dataclass(frozen=True)
class SecurityProfile:
    """Entropy-based security metrics.

    Attributes:
        total_entropy: Total Shannon entropy across all choice points.
        min_entropy: Minimum entropy at any choice point.
        max_entropy: Maximum entropy at any choice point.
        predictability: 1 - normalized entropy (0 = unpredictable, 1 = deterministic).
        security_score: Overall security score (0 = insecure, 1 = secure).
        vulnerable_states: States with zero entropy (deterministic = predictable).
    """
    total_entropy: float
    min_entropy: float
    max_entropy: float
    predictability: float
    security_score: float
    vulnerable_states: list[int]


@dataclass(frozen=True)
class ThermoMorphismAnalysis:
    """Full thermodynamic morphism analysis.

    Attributes:
        embedding: The thermodynamic embedding φ.
        galois: The energy Galois connection.
        cooling_fraction: Fraction of transitions that decrease energy.
        security: Security profile.
        rg_flow: Renormalization group flow.
        anomaly_threshold: Threshold for anomaly detection.
        design_beta: Recommended operating temperature.
        num_states: Number of lattice elements.
    """
    embedding: ThermoEmbedding
    galois: ThermoGalois
    cooling_fraction: float
    security: SecurityProfile
    rg_flow: RGFlow
    anomaly_threshold: float
    design_beta: float
    num_states: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rank(ss: StateSpace) -> dict[int, int]:
    """BFS distance from bottom."""
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


def _out_degree(ss: StateSpace) -> dict[int, int]:
    """Out-degree per state."""
    deg: dict[int, int] = {s: 0 for s in ss.states}
    for src, _, _ in ss.transitions:
        deg[src] += 1
    return deg


def _local_entropy(ss: StateSpace, state: int) -> float:
    """Shannon entropy at a choice point (uniform over outgoing transitions)."""
    out = [tgt for src, _, tgt in ss.transitions if src == state]
    n = len(out)
    if n <= 1:
        return 0.0
    # Uniform distribution over n choices
    return math.log2(n)


def _reachability(ss: StateSpace) -> dict[int, set[int]]:
    """Forward reachability."""
    adj: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        adj[src].add(tgt)
    reach: dict[int, set[int]] = {}
    for s in ss.states:
        visited: set[int] = set()
        stack = [s]
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            for t in adj.get(v, set()):
                stack.append(t)
        reach[s] = visited
    return reach


# ---------------------------------------------------------------------------
# Public API: Thermodynamic embedding
# ---------------------------------------------------------------------------

def thermo_embedding(ss: StateSpace) -> ThermoEmbedding:
    """Build the thermodynamic embedding φ: s → (E, degree, local_entropy)."""
    r = _rank(ss)
    deg = _out_degree(ss)

    observables: dict[int, ThermoObservable] = {}
    for s in ss.states:
        observables[s] = ThermoObservable(
            state=s,
            energy=float(r[s]),
            degree=deg[s],
            local_entropy=_local_entropy(ss, s),
        )

    # Check order preservation: s₁ ≥ s₂ implies E(s₁) ≥ E(s₂)
    reach = _reachability(ss)
    is_op = True
    for s1 in ss.states:
        for s2 in ss.states:
            if s2 in reach[s1] and s1 != s2:  # s1 ≥ s2
                if observables[s1].energy < observables[s2].energy:
                    is_op = False
                    break
        if not is_op:
            break

    energies = [o.energy for o in observables.values()]
    e_range = (min(energies), max(energies)) if energies else (0.0, 0.0)

    return ThermoEmbedding(
        observables=observables,
        is_order_preserving=is_op,
        energy_range=e_range,
    )


# ---------------------------------------------------------------------------
# Public API: Energy Galois connection
# ---------------------------------------------------------------------------

def energy_galois(ss: StateSpace) -> ThermoGalois:
    """Build the Galois connection via energy levels.

    α(s) = E(s)  (abstract to energy level)
    γ(e) = ∨{s : E(s) ≤ e}  (concretize: highest states at or below energy e)
    """
    r = _rank(ss)
    energy = {s: float(r[s]) for s in ss.states}

    # Distinct energy levels
    levels = sorted(set(energy.values()))

    # Alpha: state → energy
    alpha = dict(energy)

    # Gamma: energy level → states at that level or below
    gamma: dict[float, frozenset[int]] = {}
    for e in levels:
        gamma[e] = frozenset(s for s in ss.states if energy[s] <= e)

    # Validity: check α(x) ≤ e ⟺ x ∈ γ(e) for all x, e
    # This is: E(x) ≤ e ⟺ x ∈ {s : E(s) ≤ e}, which is tautological.
    is_valid = True

    return ThermoGalois(
        alpha=alpha,
        gamma=gamma,
        energy_levels=tuple(levels),
        is_valid=is_valid,
    )


def verify_galois(ss: StateSpace, galois: ThermoGalois) -> bool:
    """Verify the Galois adjunction: α(x) ≤ e ⟺ x ∈ γ(e)."""
    for s in ss.states:
        e_s = galois.alpha[s]
        for e in galois.energy_levels:
            alpha_leq = e_s <= e + 1e-10
            in_gamma = s in galois.gamma.get(e, frozenset())
            if alpha_leq != in_gamma:
                return False
    return True


# ---------------------------------------------------------------------------
# Public API: Cooling property
# ---------------------------------------------------------------------------

def is_cooling(ss: StateSpace, path: list[int] | None = None) -> bool:
    """Check if energy decreases along a path (or all transitions).

    If path is None, checks ALL transitions.
    Protocol execution = cooling iff every transition decreases energy.
    """
    r = _rank(ss)
    energy = {s: float(r[s]) for s in ss.states}

    if path is not None:
        for i in range(len(path) - 1):
            if energy.get(path[i], 0) < energy.get(path[i + 1], 0):
                return False
        return True

    # Check all transitions
    for src, _, tgt in ss.transitions:
        if energy.get(src, 0) < energy.get(tgt, 0):
            return False
    return True


def cooling_fraction(ss: StateSpace) -> float:
    """Fraction of transitions that decrease or maintain energy."""
    r = _rank(ss)
    energy = {s: float(r[s]) for s in ss.states}
    if not ss.transitions:
        return 1.0
    cooling = sum(1 for src, _, tgt in ss.transitions
                  if energy.get(src, 0) >= energy.get(tgt, 0))
    return cooling / len(ss.transitions)


# ---------------------------------------------------------------------------
# Public API: Security profile
# ---------------------------------------------------------------------------

def entropy_profile(ss: StateSpace) -> dict[int, float]:
    """Local entropy at each state (bits of uncertainty)."""
    return {s: _local_entropy(ss, s) for s in ss.states}


def security_score(ss: StateSpace, beta: float = 1.0) -> float:
    """Entropy-based security score (0 = fully predictable, 1 = maximally uncertain).

    Computes the normalized Boltzmann entropy at temperature 1/β.
    """
    from reticulate.stat_mech import entropy as thermo_entropy
    S = thermo_entropy(ss, beta)
    n = len(ss.states)
    max_S = math.log(n) if n > 1 else 1.0
    return min(1.0, S / max_S) if max_S > 0 else 0.0


def security_profile(ss: StateSpace) -> SecurityProfile:
    """Full entropy-based security analysis."""
    ep = entropy_profile(ss)
    values = list(ep.values())
    total = sum(values)
    min_e = min(values) if values else 0.0
    max_e = max(values) if values else 0.0
    n = len(ss.states)
    max_possible = math.log2(max(len(ss.transitions), 1)) * n if n > 0 else 1.0

    predictability = 1.0 - (total / max_possible if max_possible > 0 else 0.0)
    predictability = max(0.0, min(1.0, predictability))

    score = 1.0 - predictability
    vulnerable = [s for s, e in ep.items() if e == 0.0 and s != ss.bottom]

    return SecurityProfile(
        total_entropy=total,
        min_entropy=min_e,
        max_entropy=max_e,
        predictability=predictability,
        security_score=score,
        vulnerable_states=vulnerable,
    )


# ---------------------------------------------------------------------------
# Public API: Phase boundary detection
# ---------------------------------------------------------------------------

def phase_boundary_states(ss: StateSpace) -> list[int]:
    """Find states near phase transitions (high local entropy gradient)."""
    ep = entropy_profile(ss)
    adj: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        adj[src].add(tgt)

    boundary: list[int] = []
    for s in ss.states:
        for t in adj[s]:
            gradient = abs(ep.get(s, 0) - ep.get(t, 0))
            if gradient > 0.5:  # Significant entropy change
                boundary.append(s)
                break

    return boundary


# ---------------------------------------------------------------------------
# Public API: RG flow
# ---------------------------------------------------------------------------

def rg_flow(ss: StateSpace, max_steps: int = 10) -> RGFlow:
    """Compute the RG flow: chain of congruence quotients with thermodynamics.

    At each level, take the simplest non-trivial quotient and track
    how thermodynamic quantities change.
    """
    from reticulate.stat_mech import (
        internal_energy as thermo_U,
        entropy as thermo_S,
        free_energy as thermo_F,
    )
    from reticulate.congruence import simplification_options, quotient_lattice

    steps: list[RGFlowStep] = []
    current = ss
    beta = 1.0

    # Level 0: original
    steps.append(RGFlowStep(
        level=0,
        num_states=len(current.states),
        partition_size=len(current.states),
        mean_energy=thermo_U(current, beta),
        entropy=thermo_S(current, beta),
        free_energy=thermo_F(current, beta),
    ))

    for level in range(1, max_steps + 1):
        opts = simplification_options(current)
        if not opts:
            break

        # Take the coarsest non-trivial quotient (fewest classes)
        cong, size = opts[0]  # Sorted by size ascending
        q = quotient_lattice(current, cong)

        if len(q.states) >= len(current.states):
            break  # No simplification

        steps.append(RGFlowStep(
            level=level,
            num_states=len(q.states),
            partition_size=size,
            mean_energy=thermo_U(q, beta) if len(q.states) > 1 else 0.0,
            entropy=thermo_S(q, beta) if len(q.states) > 1 else 0.0,
            free_energy=thermo_F(q, beta) if len(q.states) > 1 else 0.0,
        ))

        current = q
        if len(current.states) <= 2:
            break  # Reached simple lattice

    return RGFlow(
        steps=steps,
        converged=len(current.states) <= 2,
        fixed_point_states=len(current.states),
        total_levels=len(steps),
    )


# ---------------------------------------------------------------------------
# Public API: Anomaly detection
# ---------------------------------------------------------------------------

def anomaly_score(
    ss: StateSpace,
    beta: float,
    observed_state: int,
) -> float:
    """Anomaly score: how surprising is observing this state?

    Score = -log P(s) under the Boltzmann distribution.
    High score = unexpected state = potential anomaly.
    """
    from reticulate.stat_mech import boltzmann_distribution
    P = boltzmann_distribution(ss, beta)
    p = P.get(observed_state, 0.0)
    if p <= 0:
        return float('inf')
    return -math.log(p)


# ---------------------------------------------------------------------------
# Public API: Design temperature
# ---------------------------------------------------------------------------

def design_temperature(
    ss: StateSpace,
    target_entropy_fraction: float = 0.5,
) -> float:
    """Find the β that achieves a desired entropy level.

    target_entropy_fraction: desired S / S_max (0 = zero entropy, 1 = max).
    Returns the β (inverse temperature) that achieves this.
    """
    from reticulate.stat_mech import entropy as thermo_S
    n = len(ss.states)
    if n <= 1:
        return 1.0
    S_max = math.log(n)
    target_S = target_entropy_fraction * S_max

    # Binary search for β
    lo, hi = 0.001, 100.0
    for _ in range(100):
        mid = (lo + hi) / 2
        S_mid = thermo_S(ss, mid)
        if S_mid > target_S:
            lo = mid  # Need higher β (lower T) to reduce entropy
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Public API: Full analysis
# ---------------------------------------------------------------------------

def analyze_thermo_morphism(ss: StateSpace) -> ThermoMorphismAnalysis:
    """Full thermodynamic morphism analysis."""
    emb = thermo_embedding(ss)
    gal = energy_galois(ss)
    cf = cooling_fraction(ss)
    sec = security_profile(ss)
    flow = rg_flow(ss)

    # Anomaly threshold: 95th percentile of -log P under Boltzmann(β=1)
    from reticulate.stat_mech import boltzmann_distribution
    P = boltzmann_distribution(ss, 1.0)
    scores = sorted(-math.log(max(p, 1e-30)) for p in P.values())
    threshold = scores[int(0.95 * len(scores))] if scores else 0.0

    # Design temperature for 50% entropy
    d_beta = design_temperature(ss, 0.5)

    return ThermoMorphismAnalysis(
        embedding=emb,
        galois=gal,
        cooling_fraction=cf,
        security=sec,
        rg_flow=flow,
        anomaly_threshold=threshold,
        design_beta=d_beta,
        num_states=len(ss.states),
    )
