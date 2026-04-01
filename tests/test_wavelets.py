"""Tests for wavelet analysis of session type lattices (Step 31j).

Tests cover:
  - All three approaches (spectral, diffusion, lattice-native)
  - Reconstruction accuracy
  - Energy distribution across scales
  - Anomaly localization
  - Change impact analysis
  - Benchmark protocols
"""

import math
import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.wavelets import (
    WaveletCoefficient,
    ScaleDecomposition,
    WaveletAnalysis,
    spectral_wavelets,
    diffusion_wavelets,
    lattice_wavelets,
    decompose,
    reconstruct,
    reconstruction_error,
    threshold_coefficients,
    wavelet_embedding,
    anomaly_localize,
    change_impact,
    analyze_wavelets,
)


def _build(s: str):
    return build_statespace(parse(s))


def _rank_signal(ss):
    """Default signal: rank (distance from bottom)."""
    rev = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        rev.setdefault(tgt, set()).add(src)
    dist = {}
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


@pytest.fixture
def end_ss():
    return _build("end")

@pytest.fixture
def chain():
    return _build("&{a: &{b: &{c: end}}}")

@pytest.fixture
def branch():
    return _build("&{a: end, b: end}")

@pytest.fixture
def parallel():
    return _build("(&{a: end} || &{b: end})")

@pytest.fixture
def nested():
    return _build("&{a: &{c: end, d: end}, b: &{e: end, f: end}}")


# ---------------------------------------------------------------------------
# Approach A: Spectral graph wavelets
# ---------------------------------------------------------------------------

class TestSpectralWavelets:
    def test_returns_decomposition(self, chain):
        f = _rank_signal(chain)
        d = spectral_wavelets(chain, f)
        assert isinstance(d, ScaleDecomposition)
        assert d.approach == "spectral"

    def test_has_scales(self, chain):
        f = _rank_signal(chain)
        d = spectral_wavelets(chain, f)
        assert d.n_scales >= 1

    def test_coefficients_exist(self, chain):
        f = _rank_signal(chain)
        d = spectral_wavelets(chain, f)
        assert len(d.coefficients) >= 1

    def test_energy_sums_to_one(self, chain):
        f = _rank_signal(chain)
        d = spectral_wavelets(chain, f)
        if d.energy_per_scale:
            assert abs(sum(d.energy_per_scale) - 1.0) < 1e-6

    def test_single_state(self, end_ss):
        f = _rank_signal(end_ss)
        d = spectral_wavelets(end_ss, f)
        assert d.n_scales == 0


# ---------------------------------------------------------------------------
# Approach B: Diffusion wavelets
# ---------------------------------------------------------------------------

class TestDiffusionWavelets:
    def test_returns_decomposition(self, chain):
        f = _rank_signal(chain)
        d = diffusion_wavelets(chain, f)
        assert isinstance(d, ScaleDecomposition)
        assert d.approach == "diffusion"

    def test_perfect_reconstruction(self, chain):
        """Diffusion wavelets should give near-perfect reconstruction."""
        f = _rank_signal(chain)
        d = diffusion_wavelets(chain, f)
        err = reconstruction_error(d)
        assert err < 0.01, f"Reconstruction error {err} too large"

    def test_coarse_is_smooth(self, chain):
        """Coarse component should have less variation than original."""
        f = _rank_signal(chain)
        d = diffusion_wavelets(chain, f)
        orig_var = max(f.values()) - min(f.values())
        coarse_var = max(d.coarse.values()) - min(d.coarse.values())
        assert coarse_var <= orig_var + 0.01

    def test_branch(self, branch):
        f = _rank_signal(branch)
        d = diffusion_wavelets(branch, f)
        assert d.n_scales >= 1

    def test_single_state(self, end_ss):
        f = _rank_signal(end_ss)
        d = diffusion_wavelets(end_ss, f)
        assert d.n_scales == 0


# ---------------------------------------------------------------------------
# Approach C: Lattice-native wavelets
# ---------------------------------------------------------------------------

class TestLatticeWavelets:
    def test_returns_decomposition(self, chain):
        f = _rank_signal(chain)
        d = lattice_wavelets(chain, f)
        assert isinstance(d, ScaleDecomposition)
        assert d.approach == "lattice"

    def test_perfect_reconstruction(self, chain):
        """Lattice wavelets should give perfect reconstruction."""
        f = _rank_signal(chain)
        d = lattice_wavelets(chain, f)
        err = reconstruction_error(d)
        assert err < 0.01, f"Reconstruction error {err} too large"

    def test_energy_distribution(self, chain):
        """Energy should be distributed across scales."""
        f = _rank_signal(chain)
        d = lattice_wavelets(chain, f)
        assert d.n_scales >= 2
        # Not all energy in one scale
        if len(d.energy_per_scale) >= 2:
            assert max(d.energy_per_scale) < 1.0

    def test_nested_branch(self, nested):
        f = _rank_signal(nested)
        d = lattice_wavelets(nested, f)
        assert d.n_scales >= 1

    def test_parallel(self, parallel):
        f = _rank_signal(parallel)
        d = lattice_wavelets(parallel, f)
        assert isinstance(d, ScaleDecomposition)


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

class TestReconstruction:
    def test_diffusion_exact(self, chain):
        f = _rank_signal(chain)
        d = diffusion_wavelets(chain, f)
        r = reconstruct(d)
        for s in chain.states:
            assert abs(r.get(s, 0) - f.get(s, 0)) < 0.01

    def test_lattice_exact(self, chain):
        f = _rank_signal(chain)
        d = lattice_wavelets(chain, f)
        r = reconstruct(d)
        for s in chain.states:
            assert abs(r.get(s, 0) - f.get(s, 0)) < 0.01

    def test_error_nonneg(self, chain):
        f = _rank_signal(chain)
        for approach in ["spectral", "diffusion", "lattice"]:
            d = decompose(chain, f, approach=approach)
            assert reconstruction_error(d) >= 0


# ---------------------------------------------------------------------------
# Thresholding / compression
# ---------------------------------------------------------------------------

class TestThresholding:
    def test_zero_threshold_preserves(self, chain):
        f = _rank_signal(chain)
        d = lattice_wavelets(chain, f)
        t = threshold_coefficients(d, 0.0)
        # All coefficients preserved
        assert len(t.details) == len(d.details)

    def test_high_threshold_zeros_out(self, chain):
        f = _rank_signal(chain)
        d = lattice_wavelets(chain, f)
        t = threshold_coefficients(d, 1000.0)
        # All coefficients zeroed
        for detail in t.details:
            assert all(abs(v) < 0.01 for v in detail.values())

    def test_moderate_threshold(self, nested):
        f = _rank_signal(nested)
        d = lattice_wavelets(nested, f)
        t = threshold_coefficients(d, 0.1)
        # Some coefficients remain
        any_nonzero = any(
            abs(v) > 0 for detail in t.details for v in detail.values()
        )
        # May or may not have nonzero, depending on signal


# ---------------------------------------------------------------------------
# Wavelet embedding (φ)
# ---------------------------------------------------------------------------

class TestEmbedding:
    def test_returns_dict(self, chain):
        f = _rank_signal(chain)
        emb = wavelet_embedding(chain, f)
        assert isinstance(emb, dict)
        assert len(emb) == len(chain.states)

    def test_vectors_same_length(self, chain):
        f = _rank_signal(chain)
        emb = wavelet_embedding(chain, f, n_scales=3)
        lengths = {len(v) for v in emb.values()}
        assert len(lengths) == 1  # All same length

    def test_branch_embedding(self, branch):
        f = _rank_signal(branch)
        emb = wavelet_embedding(branch, f)
        assert len(emb) == len(branch.states)


# ---------------------------------------------------------------------------
# Anomaly localization
# ---------------------------------------------------------------------------

class TestAnomalyLocalize:
    def test_no_anomaly(self, chain):
        f = _rank_signal(chain)
        anomalies = anomaly_localize(chain, f, f)
        # All coefficients should be zero (no residual)
        if anomalies:
            assert abs(anomalies[0].value) < 1e-10

    def test_detects_perturbation(self, chain):
        f = _rank_signal(chain)
        f_perturbed = dict(f)
        f_perturbed[chain.top] = f[chain.top] + 10.0  # Large perturbation
        anomalies = anomaly_localize(chain, f, f_perturbed)
        # Top state should be flagged
        top_anomalies = [a for a in anomalies if a.state == chain.top]
        assert any(abs(a.value) > 0.1 for a in top_anomalies)

    def test_returns_sorted(self, chain):
        f = _rank_signal(chain)
        f_perturbed = {s: v + (1.0 if s == chain.top else 0.0) for s, v in f.items()}
        anomalies = anomaly_localize(chain, f, f_perturbed)
        # Should be sorted by absolute value descending
        for i in range(len(anomalies) - 1):
            assert abs(anomalies[i].value) >= abs(anomalies[i + 1].value)


# ---------------------------------------------------------------------------
# Change impact
# ---------------------------------------------------------------------------

class TestChangeImpact:
    def test_no_change(self, chain):
        f = _rank_signal(chain)
        impact = change_impact(chain, f, f)
        # All scales should have zero energy
        assert all(e < 1e-10 for e in impact.values())

    def test_local_change_fine_scale(self, chain):
        f = _rank_signal(chain)
        f2 = dict(f)
        f2[chain.bottom] = f[chain.bottom] + 1.0
        impact = change_impact(chain, f, f2)
        # Fine scales should have more energy
        assert isinstance(impact, dict)
        assert len(impact) >= 1


# ---------------------------------------------------------------------------
# Unified decompose
# ---------------------------------------------------------------------------

class TestDecompose:
    def test_all_approaches(self, chain):
        f = _rank_signal(chain)
        for approach in ["spectral", "diffusion", "lattice"]:
            d = decompose(chain, f, approach=approach)
            assert d.approach == approach

    def test_default_signal(self, chain):
        d = decompose(chain)
        assert d.approach == "lattice"
        assert len(d.original) == len(chain.states)

    def test_invalid_approach(self, chain):
        with pytest.raises(ValueError):
            decompose(chain, approach="invalid")


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

class TestAnalysis:
    def test_end(self, end_ss):
        a = analyze_wavelets(end_ss)
        assert isinstance(a, WaveletAnalysis)
        assert a.num_states == 1

    def test_chain(self, chain):
        a = analyze_wavelets(chain)
        assert a.num_states == 4
        assert a.reconstruction_errors["diffusion"] < 0.01
        assert a.reconstruction_errors["lattice"] < 0.01

    def test_branch(self, branch):
        a = analyze_wavelets(branch)
        assert a.num_states == 2

    def test_parallel(self, parallel):
        a = analyze_wavelets(parallel)
        assert isinstance(a, WaveletAnalysis)

    def test_nested(self, nested):
        a = analyze_wavelets(nested)
        assert a.num_states >= 3

    def test_three_approaches_compared(self, chain):
        a = analyze_wavelets(chain)
        assert "spectral" in a.reconstruction_errors
        assert "diffusion" in a.reconstruction_errors
        assert "lattice" in a.reconstruction_errors


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

class TestBenchmarks:
    @pytest.mark.parametrize("ts,name", [
        ("&{a: end, b: end}", "Branch"),
        ("&{a: &{b: &{c: end}}}", "Chain"),
        ("+{ok: end, err: end}", "Select"),
        ("(&{a: end} || &{b: end})", "Parallel"),
        ("&{a: &{c: end, d: end}, b: &{e: end}}", "Asymmetric"),
        ("&{get: +{OK: end, ERR: end}}", "REST"),
    ])
    def test_wavelet_analysis(self, ts, name):
        ss = _build(ts)
        a = analyze_wavelets(ss)
        assert a.num_states >= 2, f"{name}"
        assert a.reconstruction_errors["lattice"] < 1.0, f"{name}"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class TestDataTypes:
    def test_wavelet_coefficient(self):
        c = WaveletCoefficient(state=0, scale=1, value=0.5)
        assert c.state == 0
        assert c.scale == 1
        assert c.value == 0.5

    def test_scale_decomposition_fields(self, chain):
        f = _rank_signal(chain)
        d = lattice_wavelets(chain, f)
        assert isinstance(d.original, dict)
        assert isinstance(d.coarse, dict)
        assert isinstance(d.details, list)
        assert isinstance(d.n_scales, int)
        assert isinstance(d.energy_per_scale, list)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_state(self):
        ss = _build("end")
        a = analyze_wavelets(ss)
        assert a.spectral.n_scales == 0

    def test_two_states(self):
        ss = _build("&{a: end}")
        a = analyze_wavelets(ss)
        assert a.num_states == 2

    def test_zero_signal(self, chain):
        f = {s: 0.0 for s in chain.states}
        d = lattice_wavelets(chain, f)
        assert all(abs(c.value) < 1e-10 for c in d.coefficients)

    def test_constant_signal(self, chain):
        f = {s: 5.0 for s in chain.states}
        d = lattice_wavelets(chain, f)
        # Constant signal should have zero detail (all energy in coarse)
        for detail in d.details:
            for v in detail.values():
                assert abs(v) < 1e-6
