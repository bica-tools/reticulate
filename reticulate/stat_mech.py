"""Statistical mechanics of session type lattices (Step 600c).

Treat the protocol state space as a physical system: assign energy to
each state, define a partition function, and compute thermodynamic
quantities (free energy, entropy, specific heat).

**Key insight**: The parallel constructor is a thermodynamic tensor product:
    Z(S₁ ∥ S₂) = Z(S₁) · Z(S₂)
because E(s₁, s₂) = E(s₁) + E(s₂) on product lattices.

**Zero-temperature limit**: Z(β) → exp(-β·E_min) recovers the tropical
(min-plus) semiring value, connecting to Steps 30l/30m.

**High-temperature limit**: Z(β) → |L|, uniform over all states.

Key functions:
  - ``partition_function(ss, beta)``       -- Z(β) = Σ exp(-β·E(s))
  - ``free_energy(ss, beta)``              -- F = -1/β · ln Z
  - ``internal_energy(ss, beta)``          -- U = ⟨E⟩ under Boltzmann
  - ``entropy(ss, beta)``                  -- S = β(U - F)
  - ``specific_heat(ss, beta)``            -- C = β² · Var(E)
  - ``boltzmann_distribution(ss, beta)``   -- probability per state
  - ``detect_phase_transitions(ss)``       -- find specific heat peaks
  - ``check_product_factorization(ss)``    -- verify Z = Z₁·Z₂
  - ``analyze_stat_mech(ss)``              -- full analysis
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
class ThermodynamicState:
    """Thermodynamic quantities at a given inverse temperature β.

    Attributes:
        beta: Inverse temperature (β = 1/kT).
        Z: Partition function.
        F: Free energy (Helmholtz).
        U: Internal energy (mean energy).
        S: Entropy.
        C: Specific heat.
    """
    beta: float
    Z: float
    F: float
    U: float
    S: float
    C: float


@dataclass(frozen=True)
class PhaseTransition:
    """A detected phase transition (specific heat peak).

    Attributes:
        beta_c: Critical inverse temperature.
        specific_heat_peak: Value of C at the peak.
        sharpness: Second derivative of C (higher = sharper transition).
    """
    beta_c: float
    specific_heat_peak: float
    sharpness: float


@dataclass(frozen=True)
class StatMechAnalysis:
    """Full statistical mechanics analysis.

    Attributes:
        num_states: Number of lattice elements.
        energy_range: (E_min, E_max).
        high_T_entropy: ln|L| (maximum entropy).
        low_T_ground_degeneracy: Number of ground states.
        curve: Thermodynamic curve at sampled temperatures.
        phase_transitions: Detected phase transitions.
        product_factorizes: True iff Z factors for product lattices.
        tropical_limit: Ground state energy (E_min).
    """
    num_states: int
    energy_range: tuple[float, float]
    high_T_entropy: float
    low_T_ground_degeneracy: int
    curve: list[ThermodynamicState]
    phase_transitions: list[PhaseTransition]
    product_factorizes: bool
    tropical_limit: float


# ---------------------------------------------------------------------------
# Energy functions
# ---------------------------------------------------------------------------

def rank_energy(ss: StateSpace) -> dict[int, float]:
    """Default energy: E(s) = rank(s) = BFS distance from bottom.

    Ground state (E=0) is bottom. Top has highest energy.
    Additive on products: E(s₁,s₂) = E(s₁) + E(s₂).
    """
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


def degree_energy(ss: StateSpace) -> dict[int, float]:
    """Alternative energy: E(s) = out-degree (number of enabled transitions).

    Higher branching = higher energy (more complex states are excited).
    """
    deg: dict[int, int] = {s: 0 for s in ss.states}
    for src, _, _ in ss.transitions:
        deg[src] += 1
    return {s: float(deg[s]) for s in ss.states}


# ---------------------------------------------------------------------------
# Partition function and thermodynamic quantities
# ---------------------------------------------------------------------------

def partition_function(
    ss: StateSpace,
    beta: float,
    energy: dict[int, float] | None = None,
) -> float:
    """Compute Z(β) = Σ_s exp(-β·E(s))."""
    if energy is None:
        energy = rank_energy(ss)
    return sum(math.exp(-beta * energy.get(s, 0.0)) for s in ss.states)


def free_energy(
    ss: StateSpace,
    beta: float,
    energy: dict[int, float] | None = None,
) -> float:
    """Helmholtz free energy F(β) = -1/β · ln Z(β)."""
    Z = partition_function(ss, beta, energy)
    if Z <= 0 or beta == 0:
        return 0.0
    return -(1.0 / beta) * math.log(Z)


def internal_energy(
    ss: StateSpace,
    beta: float,
    energy: dict[int, float] | None = None,
) -> float:
    """Internal energy U(β) = ⟨E⟩ = Σ_s E(s)·P(s) under Boltzmann."""
    if energy is None:
        energy = rank_energy(ss)
    Z = partition_function(ss, beta, energy)
    if Z <= 0:
        return 0.0
    return sum(energy.get(s, 0.0) * math.exp(-beta * energy.get(s, 0.0)) / Z
               for s in ss.states)


def entropy(
    ss: StateSpace,
    beta: float,
    energy: dict[int, float] | None = None,
) -> float:
    """Thermodynamic entropy S(β) = β(U - F) = ln Z + β⟨E⟩."""
    if energy is None:
        energy = rank_energy(ss)
    Z = partition_function(ss, beta, energy)
    U = internal_energy(ss, beta, energy)
    if Z <= 0:
        return 0.0
    return math.log(Z) + beta * U


def specific_heat(
    ss: StateSpace,
    beta: float,
    energy: dict[int, float] | None = None,
) -> float:
    """Specific heat C(β) = β² · Var(E) = β²(⟨E²⟩ - ⟨E⟩²)."""
    if energy is None:
        energy = rank_energy(ss)
    Z = partition_function(ss, beta, energy)
    if Z <= 0:
        return 0.0

    probs = {s: math.exp(-beta * energy.get(s, 0.0)) / Z for s in ss.states}
    mean_E = sum(energy.get(s, 0.0) * probs[s] for s in ss.states)
    mean_E2 = sum(energy.get(s, 0.0) ** 2 * probs[s] for s in ss.states)
    var_E = mean_E2 - mean_E ** 2

    return beta ** 2 * max(0.0, var_E)


# ---------------------------------------------------------------------------
# Boltzmann distribution
# ---------------------------------------------------------------------------

def boltzmann_distribution(
    ss: StateSpace,
    beta: float,
    energy: dict[int, float] | None = None,
) -> dict[int, float]:
    """Boltzmann probability distribution P(s) = exp(-β·E(s)) / Z."""
    if energy is None:
        energy = rank_energy(ss)
    Z = partition_function(ss, beta, energy)
    if Z <= 0:
        return {s: 0.0 for s in ss.states}
    return {s: math.exp(-beta * energy.get(s, 0.0)) / Z for s in ss.states}


# ---------------------------------------------------------------------------
# Thermodynamic curve
# ---------------------------------------------------------------------------

def thermodynamic_curve(
    ss: StateSpace,
    betas: list[float] | None = None,
    energy: dict[int, float] | None = None,
) -> list[ThermodynamicState]:
    """Compute thermodynamic quantities across a range of temperatures."""
    if betas is None:
        betas = [0.01 * i for i in range(1, 101)]  # β from 0.01 to 1.0
    if energy is None:
        energy = rank_energy(ss)

    curve: list[ThermodynamicState] = []
    for b in betas:
        Z = partition_function(ss, b, energy)
        F = free_energy(ss, b, energy)
        U = internal_energy(ss, b, energy)
        S = entropy(ss, b, energy)
        C = specific_heat(ss, b, energy)
        curve.append(ThermodynamicState(beta=b, Z=Z, F=F, U=U, S=S, C=C))

    return curve


# ---------------------------------------------------------------------------
# Phase transitions
# ---------------------------------------------------------------------------

def detect_phase_transitions(
    ss: StateSpace,
    energy: dict[int, float] | None = None,
    n_samples: int = 200,
) -> list[PhaseTransition]:
    """Detect phase transitions as peaks in specific heat C(β).

    Scans β from 0.1 to 10 and finds local maxima of C(β).
    """
    if energy is None:
        energy = rank_energy(ss)

    betas = [0.05 + 0.05 * i for i in range(n_samples)]
    heats = [specific_heat(ss, b, energy) for b in betas]

    transitions: list[PhaseTransition] = []
    for i in range(1, len(heats) - 1):
        if heats[i] > heats[i - 1] and heats[i] > heats[i + 1]:
            # Local maximum
            sharpness = abs(heats[i] - heats[i - 1]) + abs(heats[i] - heats[i + 1])
            if heats[i] > 0.01:  # Skip trivial peaks
                transitions.append(PhaseTransition(
                    beta_c=betas[i],
                    specific_heat_peak=heats[i],
                    sharpness=sharpness,
                ))

    return transitions


# ---------------------------------------------------------------------------
# Product factorization
# ---------------------------------------------------------------------------

def check_product_factorization(
    ss: StateSpace,
    beta: float = 1.0,
    energy: dict[int, float] | None = None,
) -> bool:
    """Check Z(L₁×L₂) = Z(L₁)·Z(L₂) for product state spaces.

    Only meaningful when ss is built from parallel composition.
    """
    if ss.product_factors is None or len(ss.product_factors) < 2:
        return True  # Vacuously true for non-products

    if energy is None:
        energy = rank_energy(ss)

    Z_total = partition_function(ss, beta, energy)

    Z_product = 1.0
    for factor in ss.product_factors:
        factor_energy = rank_energy(factor)
        Z_product *= partition_function(factor, beta, factor_energy)

    return abs(Z_total - Z_product) / max(Z_total, Z_product, 1e-15) < 0.01


# ---------------------------------------------------------------------------
# Tropical limit
# ---------------------------------------------------------------------------

def tropical_limit(
    ss: StateSpace,
    energy: dict[int, float] | None = None,
) -> float:
    """Zero-temperature limit: E_min (ground state energy).

    As β → ∞, F(β) → E_min and Z(β) → g·exp(-β·E_min)
    where g is the ground state degeneracy.
    """
    if energy is None:
        energy = rank_energy(ss)
    return min(energy.values()) if energy else 0.0


def ground_degeneracy(
    ss: StateSpace,
    energy: dict[int, float] | None = None,
) -> int:
    """Number of ground states (states with minimum energy)."""
    if energy is None:
        energy = rank_energy(ss)
    e_min = min(energy.values()) if energy else 0.0
    return sum(1 for e in energy.values() if abs(e - e_min) < 1e-10)


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze_stat_mech(
    ss: StateSpace,
    energy: dict[int, float] | None = None,
) -> StatMechAnalysis:
    """Full statistical mechanics analysis."""
    if energy is None:
        energy = rank_energy(ss)

    e_min = min(energy.values()) if energy else 0.0
    e_max = max(energy.values()) if energy else 0.0

    n = len(ss.states)
    high_T_S = math.log(n) if n > 0 else 0.0
    g = ground_degeneracy(ss, energy)

    betas = [0.1 * i for i in range(1, 51)]  # β from 0.1 to 5.0
    curve = thermodynamic_curve(ss, betas, energy)
    transitions = detect_phase_transitions(ss, energy)

    product_ok = check_product_factorization(ss, beta=1.0, energy=energy)
    trop = tropical_limit(ss, energy)

    return StatMechAnalysis(
        num_states=n,
        energy_range=(e_min, e_max),
        high_T_entropy=high_T_S,
        low_T_ground_degeneracy=g,
        curve=curve,
        phase_transitions=transitions,
        product_factorizes=product_ok,
        tropical_limit=trop,
    )
