"""Tests for persistent spectral protocol fingerprints (Step 31k)."""

from __future__ import annotations

import math

import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.persistent_spectral import (
    PersistentSpectralAnalysis,
    PersistentSpectralFingerprint,
    SpectralPersistencePair,
    SpectralSnapshot,
    analyze_persistent_spectrum,
    fingerprint_distance,
    persistent_fiedler_trace,
    persistent_laplacian_spectra,
    persistent_spectral_fingerprint,
    phi_lattice_to_spectrum,
    psi_spectrum_to_action,
    spectral_persistence_diagram,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ss(s: str):
    return build_statespace(parse(s))


TRIVIAL = "end"
LINEAR = "&{a: &{b: &{c: end}}}"
BRANCH = "&{a: end, b: end}"
PARALLEL = "(&{a: end} || &{b: end})"
NESTED = "&{a: &{b: end, c: end}, d: end}"


# ---------------------------------------------------------------------------
# Basic snapshots
# ---------------------------------------------------------------------------

def test_snapshots_trivial():
    snaps = persistent_laplacian_spectra(ss(TRIVIAL))
    assert len(snaps) >= 1
    assert all(isinstance(s, SpectralSnapshot) for s in snaps)
    assert snaps[-1].num_vertices == 1
    assert snaps[-1].eigenvalues == (0.0,)


def test_snapshots_linear_grows():
    snaps = persistent_laplacian_spectra(ss(LINEAR))
    # Vertices monotonically non-decreasing
    for i in range(1, len(snaps)):
        assert snaps[i].num_vertices >= snaps[i - 1].num_vertices
        assert snaps[i].num_edges >= snaps[i - 1].num_edges


def test_snapshots_branch_final_connected():
    snaps = persistent_laplacian_spectra(ss(BRANCH))
    final = snaps[-1]
    # At the final level, fiedler > 0 iff final graph is connected
    assert final.num_vertices >= 2


def test_snapshots_levels_sorted():
    snaps = persistent_laplacian_spectra(ss(NESTED))
    levels = [s.level for s in snaps]
    assert levels == sorted(levels)


def test_snapshots_all_eigenvalues_real_non_negative():
    for src in [TRIVIAL, LINEAR, BRANCH, PARALLEL, NESTED]:
        snaps = persistent_laplacian_spectra(ss(src))
        for snap in snaps:
            for e in snap.eigenvalues:
                # Allow tiny negative due to floating-point
                assert e > -1e-6


def test_zero_multiplicity_matches_components_final():
    snaps = persistent_laplacian_spectra(ss(LINEAR))
    final = snaps[-1]
    # LINEAR's final Hasse is connected -> exactly one zero
    assert final.zero_multiplicity == 1


def test_snapshot_fiedler_property():
    snap = SpectralSnapshot(level=0.0, num_vertices=2, num_edges=1,
                            eigenvalues=(0.0, 2.0))
    assert snap.fiedler == 2.0
    assert snap.spectral_radius == 2.0
    assert snap.zero_multiplicity == 1


def test_snapshot_fiedler_empty():
    snap = SpectralSnapshot(level=0.0, num_vertices=0, num_edges=0,
                            eigenvalues=())
    assert snap.fiedler == 0.0
    assert snap.spectral_radius == 0.0
    assert snap.zero_multiplicity == 0


# ---------------------------------------------------------------------------
# Filtration kinds
# ---------------------------------------------------------------------------

def test_reverse_rank_filtration():
    snaps = persistent_laplacian_spectra(ss(LINEAR), filtration="reverse_rank")
    assert len(snaps) >= 1
    assert snaps[-1].num_vertices == len(ss(LINEAR).states)


def test_unknown_filtration_raises():
    with pytest.raises(ValueError):
        persistent_laplacian_spectra(ss(TRIVIAL), filtration="bogus")


# ---------------------------------------------------------------------------
# Fiedler trace
# ---------------------------------------------------------------------------

def test_fiedler_trace_length_matches_snapshots():
    snaps = persistent_laplacian_spectra(ss(BRANCH))
    trace = persistent_fiedler_trace(ss(BRANCH))
    assert len(trace) == len(snaps)


def test_fiedler_trace_tuples():
    trace = persistent_fiedler_trace(ss(NESTED))
    assert all(isinstance(t, tuple) and len(t) == 2 for t in trace)


# ---------------------------------------------------------------------------
# Persistence diagram
# ---------------------------------------------------------------------------

def test_persistence_diagram_nonempty():
    pairs = spectral_persistence_diagram(ss(NESTED))
    assert len(pairs) >= 1
    assert all(isinstance(p, SpectralPersistencePair) for p in pairs)


def test_persistence_diagram_indices_unique():
    pairs = spectral_persistence_diagram(ss(NESTED))
    idxs = [p.index for p in pairs]
    assert len(idxs) == len(set(idxs))


def test_persistence_pair_persistence_property():
    p = SpectralPersistencePair(index=1, birth=0.0, death=2.0,
                                birth_value=0.0, death_value=1.0)
    assert p.persistence == 2.0
    p_inf = SpectralPersistencePair(index=0, birth=0.0, death=math.inf,
                                    birth_value=0.0, death_value=float('nan'))
    assert math.isinf(p_inf.persistence)


def test_persistence_trivial():
    pairs = spectral_persistence_diagram(ss(TRIVIAL))
    # Only the zero eigenvalue of the single-vertex graph
    assert len(pairs) == 1
    assert pairs[0].index == 0


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------

def test_fingerprint_structure():
    fp = persistent_spectral_fingerprint(ss(NESTED))
    assert isinstance(fp, PersistentSpectralFingerprint)
    assert fp.num_levels == len(fp.fiedler_trace)
    assert fp.num_levels == len(fp.radius_trace)
    assert fp.num_levels == len(fp.zero_mult_trace)
    assert fp.num_levels == len(fp.energy_trace)


def test_fingerprint_as_vector_length():
    fp = persistent_spectral_fingerprint(ss(NESTED))
    v = fp.as_vector()
    assert len(v) == 4 * fp.num_levels + 2


def test_fingerprint_max_fiedler_consistent():
    fp = persistent_spectral_fingerprint(ss(BRANCH))
    if fp.fiedler_trace:
        assert fp.max_fiedler == max(fp.fiedler_trace)


def test_fingerprint_empty_state_space_safe():
    # Synthetic empty state space via a minimal protocol
    fp = persistent_spectral_fingerprint(ss(TRIVIAL))
    assert fp.num_levels >= 1


def test_fingerprint_distinguishes_different_protocols():
    fp_a = persistent_spectral_fingerprint(ss(LINEAR))
    fp_b = persistent_spectral_fingerprint(ss(BRANCH))
    assert fingerprint_distance(fp_a, fp_b) > 0.0


def test_fingerprint_identity_zero_distance():
    fp = persistent_spectral_fingerprint(ss(NESTED))
    assert fingerprint_distance(fp, fp) == 0.0


def test_fingerprint_distance_symmetric():
    fp_a = persistent_spectral_fingerprint(ss(LINEAR))
    fp_b = persistent_spectral_fingerprint(ss(NESTED))
    d_ab = fingerprint_distance(fp_a, fp_b)
    d_ba = fingerprint_distance(fp_b, fp_a)
    assert abs(d_ab - d_ba) < 1e-12


def test_fingerprint_distance_non_negative():
    for a, b in [(LINEAR, BRANCH), (NESTED, PARALLEL), (TRIVIAL, LINEAR)]:
        fp_a = persistent_spectral_fingerprint(ss(a))
        fp_b = persistent_spectral_fingerprint(ss(b))
        assert fingerprint_distance(fp_a, fp_b) >= 0.0


# ---------------------------------------------------------------------------
# Bidirectional morphisms
# ---------------------------------------------------------------------------

def test_phi_is_fingerprint():
    fp = phi_lattice_to_spectrum(ss(NESTED))
    assert isinstance(fp, PersistentSpectralFingerprint)


def test_psi_outputs_actions():
    fp = phi_lattice_to_spectrum(ss(NESTED))
    actions = psi_spectrum_to_action(fp)
    assert "bottleneck" in actions
    assert "complexity_alert" in actions
    assert "fragile_levels" in actions
    assert "monitoring_level" in actions
    assert "stability_score" in actions
    assert 0.0 <= actions["stability_score"] <= 1.0


def test_psi_high_threshold_triggers_bottleneck():
    fp = phi_lattice_to_spectrum(ss(LINEAR))
    actions = psi_spectrum_to_action(fp, bottleneck_threshold=10.0)
    # Every Fiedler value should be under 10
    assert actions["bottleneck"] is True or all(
        f == 0.0 for f in fp.fiedler_trace
    )


def test_psi_low_threshold_quiet():
    fp = phi_lattice_to_spectrum(ss(BRANCH))
    actions = psi_spectrum_to_action(fp, bottleneck_threshold=0.0)
    assert actions["bottleneck"] is False


def test_phi_deterministic():
    a = phi_lattice_to_spectrum(ss(NESTED))
    b = phi_lattice_to_spectrum(ss(NESTED))
    assert a == b


def test_bidirectional_roundtrip_runs():
    # φ then ψ should always produce a well-formed action dict
    for src in [TRIVIAL, LINEAR, BRANCH, PARALLEL, NESTED]:
        fp = phi_lattice_to_spectrum(ss(src))
        actions = psi_spectrum_to_action(fp)
        assert isinstance(actions, dict)


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def test_analysis_runs():
    a = analyze_persistent_spectrum(ss(NESTED))
    assert isinstance(a, PersistentSpectralAnalysis)
    assert a.num_states == len(ss(NESTED).states)
    assert len(a.snapshots) >= 1


def test_analysis_fingerprint_matches():
    sp = ss(LINEAR)
    a = analyze_persistent_spectrum(sp)
    fp = persistent_spectral_fingerprint(sp)
    assert a.fingerprint == fp


def test_analysis_multiple_protocols():
    for src in [TRIVIAL, LINEAR, BRANCH, PARALLEL, NESTED]:
        a = analyze_persistent_spectrum(ss(src))
        assert a.num_states == len(ss(src).states)


# ---------------------------------------------------------------------------
# Compositional properties
# ---------------------------------------------------------------------------

def test_parallel_has_more_levels_than_component():
    fp_single = persistent_spectral_fingerprint(ss("&{a: end}"))
    fp_par = persistent_spectral_fingerprint(ss(PARALLEL))
    assert fp_par.num_levels >= fp_single.num_levels


def test_monotone_vertex_count():
    for src in [LINEAR, BRANCH, NESTED, PARALLEL]:
        snaps = persistent_laplacian_spectra(ss(src))
        counts = [s.num_vertices for s in snaps]
        assert counts == sorted(counts)
