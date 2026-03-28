"""Fundamental Physics as Session Types (Step 207).

The four fundamental forces, thermodynamic laws, quantum mechanics, and
cosmological evolution — all encoded as session types.  The universe IS a
session type executing itself.

Every physical process follows a protocol: particles exchange bosons
according to interaction rules, thermodynamic systems evolve through
constrained state sequences, and the cosmos unfolds from singularity
to heat death along a determined session.

This module provides:
    ``get_physics(name)``              -- look up a physics entry by name.
    ``physics_by_domain(domain)``      -- find entries in a domain.
    ``physics_by_scale(scale)``        -- find entries at a given scale.
    ``unification_score(n1, n2)``      -- compatibility between two physics.
    ``all_physics_form_lattices()``    -- verify every entry forms a lattice.
    ``grand_unified_type()``           -- negotiate all force types together.
"""

from __future__ import annotations

from dataclasses import dataclass

from reticulate.parser import parse, pretty
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice
from reticulate.negotiation import compatibility_score, negotiate, negotiate_group


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhysicsEntry:
    """A single entry in the physics session type library.

    Attributes:
        name: Unique identifier for this physics concept.
        domain: Physics domain (quantum, thermodynamics, relativity,
                cosmology, forces, particles).
        session_type_str: Session type in textual syntax.
        description: Short human-readable description.
        scale: Physical scale (planck, atomic, human, cosmic).
    """

    name: str
    domain: str
    session_type_str: str
    description: str
    scale: str


# ---------------------------------------------------------------------------
# The Physics Library
# ---------------------------------------------------------------------------

PHYSICS_LIBRARY: dict[str, PhysicsEntry] = {
    # -- Fundamental Forces --
    "gravity": PhysicsEntry(
        "gravity", "forces",
        "rec X . &{mass: +{attract: &{orbit: X, collapse: end, escape: end}}}",
        "Gravitational interaction: mass attracts, orbits or collapses",
        "cosmic",
    ),
    "electromagnetism": PhysicsEntry(
        "electromagnetism", "forces",
        "rec X . &{charge: +{attract: X, repel: X, radiate: end}}",
        "Electromagnetic force: charges attract, repel, or radiate",
        "atomic",
    ),
    "strong_nuclear": PhysicsEntry(
        "strong_nuclear", "forces",
        "&{quarks: +{bind: &{hadron: end}}}",
        "Strong nuclear force: quarks bind into hadrons",
        "planck",
    ),
    "weak_nuclear": PhysicsEntry(
        "weak_nuclear", "forces",
        "&{decay: +{transmute: &{neutrino: end}}}",
        "Weak nuclear force: particle decay and transmutation",
        "planck",
    ),

    # -- Quantum Mechanics --
    "superposition": PhysicsEntry(
        "superposition", "quantum",
        "&{prepare: +{measure: &{collapse_up: end, collapse_down: end}}}",
        "Quantum superposition: prepare, measure, collapse",
        "planck",
    ),
    "entanglement": PhysicsEntry(
        "entanglement", "quantum",
        "&{entangle: +{measure: &{correlate: end}}}",
        "Quantum entanglement: non-local correlations upon measurement",
        "planck",
    ),
    "tunneling": PhysicsEntry(
        "tunneling", "quantum",
        "&{barrier: +{classical_reflect: end, quantum_tunnel: end}}",
        "Quantum tunneling: classically forbidden barrier penetration",
        "planck",
    ),
    "uncertainty": PhysicsEntry(
        "uncertainty", "quantum",
        "+{measure_position: &{lose_momentum: end}, measure_momentum: &{lose_position: end}}",
        "Heisenberg uncertainty: measuring one conjugate disturbs the other",
        "planck",
    ),
    "decoherence": PhysicsEntry(
        "decoherence", "quantum",
        "&{interact_environment: +{classical: end}}",
        "Decoherence: environment interaction destroys quantum coherence",
        "atomic",
    ),
    "wave_function": PhysicsEntry(
        "wave_function", "quantum",
        "rec X . &{evolve: +{interfere: X, measure: end}}",
        "Wave function: unitary evolution until measurement",
        "planck",
    ),

    # -- Thermodynamics --
    "first_law": PhysicsEntry(
        "first_law", "thermodynamics",
        "&{energy_in: +{work: &{heat: +{conserve: end}}}}",
        "First law: energy is conserved across work and heat",
        "human",
    ),
    "second_law": PhysicsEntry(
        "second_law", "thermodynamics",
        "rec X . &{process: +{entropy_increase: X, equilibrium: end}}",
        "Second law: entropy increases until equilibrium",
        "human",
    ),
    "third_law": PhysicsEntry(
        "third_law", "thermodynamics",
        "&{cool: +{approach_zero: &{never_reach: end}}}",
        "Third law: absolute zero is asymptotically unreachable",
        "human",
    ),
    "carnot_cycle": PhysicsEntry(
        "carnot_cycle", "thermodynamics",
        "rec X . &{isothermal_expand: +{adiabatic_expand: &{isothermal_compress: +{adiabatic_compress: X, stop: end}}}}",
        "Carnot cycle: ideal thermodynamic engine cycle with stop",
        "human",
    ),

    # -- Cosmology --
    "big_bang": PhysicsEntry(
        "big_bang", "cosmology",
        "&{singularity: +{inflation: &{cooling: +{matter: &{stars: +{planets: &{life: end}}}}}}}",
        "Big Bang: from singularity through inflation to structure formation",
        "cosmic",
    ),
    "stellar_evolution": PhysicsEntry(
        "stellar_evolution", "cosmology",
        "&{nebula: +{collapse: &{fusion: +{giant: &{supernova: +{remnant: end}}}}}}",
        "Stellar evolution: nebula to remnant through fusion stages",
        "cosmic",
    ),
    "black_hole": PhysicsEntry(
        "black_hole", "cosmology",
        "rec X . &{accrete: +{absorb: X, hawking: end}}",
        "Black hole: accretion with eventual Hawking evaporation",
        "cosmic",
    ),
    "heat_death": PhysicsEntry(
        "heat_death", "cosmology",
        "&{expand: +{cool: &{dilute: +{silence: end}}}}",
        "Heat death: cosmic expansion to maximum entropy silence",
        "cosmic",
    ),
    "cosmic_evolution": PhysicsEntry(
        "cosmic_evolution", "cosmology",
        "&{radiation_era: +{matter_era: &{dark_energy_era: +{heat_death: end}}}}",
        "Cosmic evolution through radiation, matter, and dark energy eras",
        "cosmic",
    ),

    # -- Particles --
    "electron": PhysicsEntry(
        "electron", "particles",
        "rec X . &{absorb_photon: +{excite: &{emit_photon: X}}, scatter: end}",
        "Electron: photon absorption/emission cycle with scattering exit",
        "atomic",
    ),
    "photon": PhysicsEntry(
        "photon", "particles",
        "&{emit: +{travel: &{absorb: end, scatter: end}}}",
        "Photon lifecycle: emission, travel, absorption or scattering",
        "atomic",
    ),
    "hydrogen_atom": PhysicsEntry(
        "hydrogen_atom", "particles",
        "&{ground_state: +{excite: &{emit: +{ground_state: end}}}}",
        "Hydrogen atom: excitation and relaxation cycle",
        "atomic",
    ),

    "neutrino": PhysicsEntry(
        "neutrino", "particles",
        "&{produce: +{oscillate: &{detect: end, pass_through: end}}}",
        "Neutrino: produced in reactions, oscillates flavour, rarely detected",
        "planck",
    ),

    # -- Relativity --
    "time_dilation": PhysicsEntry(
        "time_dilation", "relativity",
        "&{accelerate: +{time_slows: &{observe: end}}}",
        "Time dilation: acceleration causes observable time slowdown",
        "cosmic",
    ),
    "spacetime_curvature": PhysicsEntry(
        "spacetime_curvature", "relativity",
        "&{mass_present: +{curve_spacetime: &{geodesic: end}}}",
        "Spacetime curvature: mass curves geometry, objects follow geodesics",
        "cosmic",
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_physics(name: str) -> PhysicsEntry:
    """Look up a physics entry by name.

    Raises:
        KeyError: If *name* is not in the library.
    """
    if name not in PHYSICS_LIBRARY:
        raise KeyError(f"Unknown physics entry: {name!r}")
    return PHYSICS_LIBRARY[name]


def physics_by_domain(domain: str) -> list[PhysicsEntry]:
    """Return all entries belonging to *domain*."""
    return [e for e in PHYSICS_LIBRARY.values() if e.domain == domain]


def physics_by_scale(scale: str) -> list[PhysicsEntry]:
    """Return all entries at the given physical *scale*."""
    return [e for e in PHYSICS_LIBRARY.values() if e.scale == scale]


def unification_score(name1: str, name2: str) -> float:
    """Compute how 'unifiable' two physics concepts are.

    Uses the negotiation compatibility_score between the parsed session
    types of the two entries.  A score of 1.0 means structurally identical;
    0.0 means completely incompatible.
    """
    e1 = get_physics(name1)
    e2 = get_physics(name2)
    s1 = parse(e1.session_type_str)
    s2 = parse(e2.session_type_str)
    return compatibility_score(s1, s2)


def all_physics_form_lattices() -> bool:
    """Verify that every entry in the physics library forms a lattice.

    Returns True if and only if every entry parses and its state space
    is a lattice.
    """
    for entry in PHYSICS_LIBRARY.values():
        ast = parse(entry.session_type_str)
        ss = build_statespace(ast)
        result = check_lattice(ss)
        if not result.is_lattice:
            return False
    return True


def grand_unified_type() -> str:
    """Negotiate all four fundamental force types into a 'GUT session type'.

    Computes the greatest common sub-protocol among gravity,
    electromagnetism, strong nuclear, and weak nuclear forces.
    Returns the pretty-printed negotiated type string.
    """
    force_names = ["gravity", "electromagnetism", "strong_nuclear", "weak_nuclear"]
    force_asts = [parse(get_physics(n).session_type_str) for n in force_names]
    result = negotiate_group(force_asts)
    return pretty(result)
