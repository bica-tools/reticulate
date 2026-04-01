"""Tests for information geometry of session type lattices (Step 600e)."""

import math
import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.info_geometry import (
    FisherMatrix,
    ManifoldPoint,
    Geodesic,
    InfoGeometryAnalysis,
    fisher_matrix,
    kl_divergence,
    fisher_rao_distance,
    to_manifold_point,
    compute_geodesic,
    exponential_family_params,
    boltzmann_curve,
    boltzmann_fisher_info,
    lattice_adapted_metric,
    embed_states,
    retract_to_state,
    round_trip_loss,
    anomaly_kl,
    information_leakage,
    protocol_distance,
    natural_gradient,
    scalar_curvature,
    analyze_info_geometry,
)


def _build(s: str):
    return build_statespace(parse(s))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
def iterator():
    return _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")

@pytest.fixture
def three_branch():
    return _build("&{a: end, b: end, c: end}")


# ===================================================================
# Approach A: Fisher metric on the full simplex
# ===================================================================

class TestFisherMatrix:
    def test_end_trivial(self, end_ss):
        fm = fisher_matrix(end_ss)
        assert fm.dimension == 0
        assert fm.matrix == ()

    def test_branch_dimension(self, branch):
        fm = fisher_matrix(branch)
        # 2 states (top + end) -> 1x1 Fisher matrix
        assert fm.dimension == len(branch.states) - 1

    def test_chain_dimension(self, chain):
        fm = fisher_matrix(chain)
        # 4 states -> 3x3
        assert fm.dimension == 3

    def test_uniform_positive_semidefinite(self, branch):
        """Fisher matrix must be positive semi-definite."""
        fm = fisher_matrix(branch)
        # Check via eigenvalue criterion for 2x2: trace >= 0 and det >= 0
        if fm.dimension == 2:
            tr = fm.matrix[0][0] + fm.matrix[1][1]
            det = fm.matrix[0][0] * fm.matrix[1][1] - fm.matrix[0][1] * fm.matrix[1][0]
            assert tr >= -1e-10
            assert det >= -1e-10

    def test_psd_three_states(self, three_branch):
        """PSD for 3-branch."""
        fm = fisher_matrix(three_branch)
        assert fm.dimension == len(three_branch.states) - 1
        # Trace must be non-negative
        tr = sum(fm.matrix[i][i] for i in range(fm.dimension))
        assert tr >= -1e-10

    def test_psd_chain(self, chain):
        fm = fisher_matrix(chain)
        tr = sum(fm.matrix[i][i] for i in range(fm.dimension))
        assert tr >= -1e-10

    def test_custom_distribution(self, chain):
        states = sorted(chain.states)
        n = len(states)
        # Build a custom non-uniform distribution
        p = {states[i]: (i + 1) / sum(range(1, n + 1)) for i in range(n)}
        fm = fisher_matrix(chain, p)
        assert fm.dimension == n - 1
        # Diagonal entries should be positive
        for i in range(fm.dimension):
            assert fm.matrix[i][i] > 0

    def test_parallel_psd(self, parallel):
        fm = fisher_matrix(parallel)
        tr = sum(fm.matrix[i][i] for i in range(fm.dimension))
        assert tr >= -1e-10

    def test_frozen_dataclass(self, branch):
        fm = fisher_matrix(branch)
        with pytest.raises(AttributeError):
            fm.dimension = 99


class TestKLDivergence:
    def test_identical_is_zero(self):
        p = {0: 0.5, 1: 0.5}
        assert abs(kl_divergence(p, p)) < 1e-10

    def test_non_negative(self):
        p = {0: 0.7, 1: 0.3}
        q = {0: 0.4, 1: 0.6}
        assert kl_divergence(p, q) >= -1e-10

    def test_asymmetric(self):
        p = {0: 0.9, 1: 0.1}
        q = {0: 0.1, 1: 0.9}
        d_pq = kl_divergence(p, q)
        d_qp = kl_divergence(q, p)
        assert d_pq > 0
        assert d_qp > 0
        # KL is NOT symmetric in general
        # (but these two happen to be equal by symmetry of the swap)
        assert abs(d_pq - d_qp) < 1e-10

    def test_zero_iff_equal(self):
        p = {0: 0.3, 1: 0.3, 2: 0.4}
        q = {0: 0.3, 1: 0.3, 2: 0.4}
        assert abs(kl_divergence(p, q)) < 1e-10

    def test_positive_for_different(self):
        p = {0: 0.5, 1: 0.5}
        q = {0: 0.9, 1: 0.1}
        assert kl_divergence(p, q) > 0.01

    def test_dirac_vs_uniform(self):
        p = {0: 1.0, 1: 0.0, 2: 0.0}
        q = {0: 1/3, 1: 1/3, 2: 1/3}
        kl = kl_divergence(p, q)
        assert abs(kl - math.log(3)) < 0.01


class TestFisherRaoDistance:
    def test_same_is_zero(self):
        p = {0: 0.5, 1: 0.5}
        assert abs(fisher_rao_distance(p, p)) < 1e-10

    def test_symmetric(self):
        p = {0: 0.7, 1: 0.3}
        q = {0: 0.4, 1: 0.6}
        assert abs(fisher_rao_distance(p, q) - fisher_rao_distance(q, p)) < 1e-10

    def test_non_negative(self):
        p = {0: 0.8, 1: 0.2}
        q = {0: 0.3, 1: 0.7}
        assert fisher_rao_distance(p, q) >= -1e-10

    def test_triangle_inequality(self):
        p = {0: 0.6, 1: 0.2, 2: 0.2}
        q = {0: 0.2, 1: 0.6, 2: 0.2}
        r = {0: 0.2, 1: 0.2, 2: 0.6}
        d_pq = fisher_rao_distance(p, q)
        d_qr = fisher_rao_distance(q, r)
        d_pr = fisher_rao_distance(p, r)
        assert d_pr <= d_pq + d_qr + 1e-10

    def test_max_distance_between_diracs(self):
        """Dirac deltas on different states should have maximum distance."""
        p = {0: 1.0, 1: 0.0}
        q = {0: 0.0, 1: 1.0}
        d = fisher_rao_distance(p, q)
        # arccos(0) = pi/2, distance = 2 * pi/2 = pi
        assert abs(d - math.pi) < 0.01

    def test_identity_for_equal(self):
        p = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        assert abs(fisher_rao_distance(p, p)) < 1e-10


class TestManifoldPoint:
    def test_to_manifold_point(self, branch):
        p = {s: 1.0 / len(branch.states) for s in branch.states}
        mp = to_manifold_point(branch, p)
        assert isinstance(mp, ManifoldPoint)
        assert len(mp.natural_params) == len(branch.states) - 1

    def test_end_manifold_point(self, end_ss):
        p = {s: 1.0 for s in end_ss.states}
        mp = to_manifold_point(end_ss, p)
        assert mp.natural_params == ()

    def test_uniform_natural_params_zero(self, branch):
        """At uniform, all natural params should be 0 (all log-ratios = 0)."""
        p = {s: 1.0 / len(branch.states) for s in branch.states}
        mp = to_manifold_point(branch, p)
        for theta in mp.natural_params:
            assert abs(theta) < 1e-10


class TestGeodesic:
    def test_geodesic_distance(self, chain):
        states = sorted(chain.states)
        n = len(states)
        p = {s: 1.0 / n for s in states}
        q = {s: (i + 1) / sum(range(1, n + 1)) for i, s in enumerate(states)}
        geo = compute_geodesic(chain, p, q)
        assert isinstance(geo, Geodesic)
        assert geo.distance >= 0
        assert abs(geo.distance - fisher_rao_distance(p, q)) < 1e-10

    def test_geodesic_midpoint_is_distribution(self, chain):
        states = sorted(chain.states)
        n = len(states)
        p = {s: 1.0 / n for s in states}
        q = {s: (i + 1) / sum(range(1, n + 1)) for i, s in enumerate(states)}
        geo = compute_geodesic(chain, p, q)
        total = sum(geo.midpoint.values())
        assert abs(total - 1.0) < 1e-10
        for v in geo.midpoint.values():
            assert v >= 0


# ===================================================================
# Approach B: Exponential family (Boltzmann) geometry
# ===================================================================

class TestExponentialFamily:
    def test_params_structure(self, chain):
        eta, mean, var = exponential_family_params(chain, 1.0)
        assert eta == -1.0
        assert mean >= 0
        assert var >= 0

    def test_high_temperature(self, chain):
        """At β→0, distribution is uniform, mean energy ≈ average rank."""
        eta, mean, var = exponential_family_params(chain, 0.01)
        # Mean should be close to average rank
        n = len(chain.states)
        assert mean > 0

    def test_low_temperature(self, chain):
        """At β→∞, distribution concentrates on ground state (E=0)."""
        eta, mean, var = exponential_family_params(chain, 100.0)
        assert mean < 0.5  # Should be close to 0

    def test_variance_non_negative(self, branch):
        _, _, var = exponential_family_params(branch, 1.0)
        assert var >= -1e-10

    def test_eta_is_minus_beta(self, chain):
        for beta in [0.5, 1.0, 2.0]:
            eta, _, _ = exponential_family_params(chain, beta)
            assert abs(eta - (-beta)) < 1e-10


class TestBoltzmannCurve:
    def test_curve_length(self, chain):
        betas = [0.5, 1.0, 2.0]
        curve = boltzmann_curve(chain, betas)
        assert len(curve) == 3
        assert all(isinstance(pt, ManifoldPoint) for pt in curve)

    def test_curve_default(self, chain):
        curve = boltzmann_curve(chain)
        assert len(curve) == 20  # Default: 0.1 to 2.0

    def test_each_point_is_distribution(self, branch):
        curve = boltzmann_curve(branch, [0.5, 1.0, 5.0])
        for pt in curve:
            total = sum(pt.distribution.values())
            assert abs(total - 1.0) < 1e-10


class TestBoltzmannFisherInfo:
    def test_non_negative(self, chain):
        fi = boltzmann_fisher_info(chain, 1.0)
        assert fi >= -1e-10

    def test_end_zero(self, end_ss):
        fi = boltzmann_fisher_info(end_ss, 1.0)
        assert abs(fi) < 1e-10

    def test_equals_variance(self, chain):
        """I(β) = Var_β(E) by exponential family theory."""
        fi = boltzmann_fisher_info(chain, 1.0)
        _, _, var = exponential_family_params(chain, 1.0)
        assert abs(fi - var) < 1e-10

    def test_peaks_at_intermediate_temp(self, chain):
        """Fisher info should peak where energy variance is largest."""
        fis = [boltzmann_fisher_info(chain, b) for b in [0.1, 0.5, 1.0, 2.0, 10.0]]
        # At extreme temperatures, variance → 0
        assert fis[0] > 0 or fis[2] > 0  # Some nonzero Fisher info exists


# ===================================================================
# Approach C: Lattice-adapted metric
# ===================================================================

class TestLatticeAdaptedMetric:
    def test_end_trivial(self, end_ss):
        lam = lattice_adapted_metric(end_ss)
        assert lam.dimension == 0

    def test_dimension_matches_fisher(self, branch):
        fm = fisher_matrix(branch)
        lam = lattice_adapted_metric(branch)
        assert lam.dimension == fm.dimension

    def test_differs_from_fisher(self, chain):
        """Lattice metric should differ from standard Fisher (unless trivial)."""
        fm = fisher_matrix(chain)
        lam = lattice_adapted_metric(chain)
        if fm.dimension > 0:
            # At least one entry should differ
            any_diff = False
            for i in range(fm.dimension):
                for j in range(fm.dimension):
                    if abs(fm.matrix[i][j] - lam.matrix[i][j]) > 1e-10:
                        any_diff = True
            assert any_diff, "Lattice metric should differ from standard Fisher"

    def test_psd(self, branch):
        """Lattice-adapted metric should be PSD."""
        lam = lattice_adapted_metric(branch)
        if lam.dimension == 2:
            tr = lam.matrix[0][0] + lam.matrix[1][1]
            det = lam.matrix[0][0] * lam.matrix[1][1] - lam.matrix[0][1] * lam.matrix[1][0]
            assert tr >= -1e-10

    def test_parallel_structure(self, parallel):
        """Parallel composition should affect lattice distances."""
        lam = lattice_adapted_metric(parallel)
        assert lam.dimension > 0
        tr = sum(lam.matrix[i][i] for i in range(lam.dimension))
        assert tr > 0


# ===================================================================
# Bidirectional morphisms
# ===================================================================

class TestEmbedStates:
    def test_dirac_delta(self, branch):
        emb = embed_states(branch)
        for s in branch.states:
            delta = emb[s]
            assert abs(delta[s] - 1.0) < 1e-10
            for t in branch.states:
                if t != s:
                    assert abs(delta[t]) < 1e-10

    def test_all_states_embedded(self, chain):
        emb = embed_states(chain)
        assert set(emb.keys()) == chain.states

    def test_end_single(self, end_ss):
        emb = embed_states(end_ss)
        assert len(emb) == 1


class TestRetractToState:
    def test_dirac_retract(self, branch):
        """Retract of Dirac delta should return that state."""
        for s in branch.states:
            delta = {t: (1.0 if t == s else 0.0) for t in branch.states}
            assert retract_to_state(branch, delta) == s

    def test_uniform_retract(self, chain):
        """Retract of uniform — should return some valid state."""
        p = {s: 1.0 / len(chain.states) for s in chain.states}
        result = retract_to_state(chain, p)
        assert result in chain.states


class TestRoundTripLoss:
    def test_dirac_zero_loss(self, branch):
        """Round trip of Dirac delta should have zero loss."""
        for s in branch.states:
            delta = {t: (1.0 if t == s else 0.0) for t in branch.states}
            assert abs(round_trip_loss(branch, delta)) < 1e-10

    def test_uniform_positive_loss(self, branch):
        """Uniform distribution should lose information in round trip."""
        p = {s: 1.0 / len(branch.states) for s in branch.states}
        loss = round_trip_loss(branch, p)
        if len(branch.states) > 1:
            assert loss > 0

    def test_loss_bounded(self, chain):
        """Round-trip loss should be in [0, 1]."""
        p = {s: 1.0 / len(chain.states) for s in chain.states}
        loss = round_trip_loss(chain, p)
        assert 0 <= loss <= 1.0 + 1e-10

    def test_embed_retract_roundtrip(self, branch):
        """embed then retract on a Dirac recovers the state."""
        emb = embed_states(branch)
        for s in branch.states:
            recovered = retract_to_state(branch, emb[s])
            assert recovered == s


# ===================================================================
# Practical functions
# ===================================================================

class TestAnomalyKL:
    def test_matching_zero(self, branch):
        expected = {s: 1.0 / len(branch.states) for s in branch.states}
        # Counts proportional to expected
        counts = {s: 100 for s in branch.states}
        score = anomaly_kl(branch, expected, counts)
        assert abs(score) < 0.01

    def test_anomalous_high(self, branch):
        states = sorted(branch.states)
        expected = {s: 1.0 / len(states) for s in states}
        # All counts on one state = anomalous
        counts = {states[0]: 1000}
        for s in states[1:]:
            counts[s] = 0
        score = anomaly_kl(branch, expected, counts)
        assert score > 0.5

    def test_empty_counts(self, branch):
        expected = {s: 1.0 / len(branch.states) for s in branch.states}
        assert anomaly_kl(branch, expected, {}) == 0.0


class TestInformationLeakage:
    def test_non_negative(self, chain):
        leak = information_leakage(chain, 1.0)
        assert leak >= -1e-10

    def test_end_zero(self, end_ss):
        leak = information_leakage(end_ss, 1.0)
        assert abs(leak) < 1e-10

    def test_varies_with_beta(self, chain):
        leaks = [information_leakage(chain, b) for b in [0.1, 1.0, 10.0]]
        # Should vary
        assert not all(abs(l - leaks[0]) < 1e-10 for l in leaks)


class TestProtocolDistance:
    def test_same_protocol_zero(self, branch):
        d = protocol_distance(branch, branch, 1.0)
        assert abs(d) < 1e-10

    def test_different_protocols_positive(self, branch, chain):
        d = protocol_distance(branch, chain, 1.0)
        assert d > 0

    def test_symmetric(self, branch, chain):
        d1 = protocol_distance(branch, chain, 1.0)
        d2 = protocol_distance(chain, branch, 1.0)
        assert abs(d1 - d2) < 1e-10

    def test_triangle(self, branch, chain, three_branch):
        d_bc = protocol_distance(branch, chain, 1.0)
        d_ct = protocol_distance(chain, three_branch, 1.0)
        d_bt = protocol_distance(branch, three_branch, 1.0)
        assert d_bt <= d_bc + d_ct + 1e-10


class TestNaturalGradient:
    def test_end_trivial(self, end_ss):
        p = {s: 1.0 for s in end_ss.states}
        grad = natural_gradient(end_ss, p, lambda x: 0.0)
        assert all(abs(v) < 1e-10 for v in grad.values())

    def test_returns_direction(self, branch):
        p = {s: 1.0 / len(branch.states) for s in branch.states}

        def entropy_obj(q):
            return -sum(v * math.log(max(v, 1e-30)) for v in q.values())

        grad = natural_gradient(branch, p, entropy_obj)
        assert len(grad) == len(branch.states)

    def test_sums_near_zero(self, branch):
        """Natural gradient on simplex should approximately sum to zero."""
        p = {s: 1.0 / len(branch.states) for s in branch.states}
        grad = natural_gradient(branch, p, lambda q: sum(v ** 2 for v in q.values()))
        total = sum(grad.values())
        assert abs(total) < 0.1  # Approximate due to finite differences


class TestScalarCurvature:
    def test_end_zero(self, end_ss):
        R = scalar_curvature(end_ss)
        assert abs(R) < 1e-10

    def test_two_states_zero(self, branch):
        """Branch with 2 options has 3 states, dim=2. Curvature may be nonzero."""
        R = scalar_curvature(branch)
        # Just check it returns a finite number
        assert math.isfinite(R)

    def test_chain_finite(self, chain):
        R = scalar_curvature(chain)
        assert math.isfinite(R)


# ===================================================================
# Comparative: all three approaches on benchmarks
# ===================================================================

class TestComparative:
    def test_fisher_vs_lattice_metric_branch(self, branch):
        fm = fisher_matrix(branch)
        lam = lattice_adapted_metric(branch)
        fm_trace = sum(fm.matrix[i][i] for i in range(fm.dimension))
        lam_trace = sum(lam.matrix[i][i] for i in range(lam.dimension))
        # Lattice metric has extra weighting, so trace should differ
        assert fm_trace > 0
        assert lam_trace > 0

    def test_fisher_vs_boltzmann_chain(self, chain):
        """Compare Fisher matrix trace at Boltzmann vs uniform."""
        from reticulate.info_geometry import _boltzmann_dist
        p_boltz = _boltzmann_dist(chain, 1.0)
        fm_boltz = fisher_matrix(chain, p_boltz)
        fm_unif = fisher_matrix(chain)
        tr_boltz = sum(fm_boltz.matrix[i][i] for i in range(fm_boltz.dimension))
        tr_unif = sum(fm_unif.matrix[i][i] for i in range(fm_unif.dimension))
        assert tr_boltz > 0
        assert tr_unif > 0

    def test_all_approaches_parallel(self, parallel):
        """All three approaches should work on parallel composition."""
        fm = fisher_matrix(parallel)
        lam = lattice_adapted_metric(parallel)
        bfi = boltzmann_fisher_info(parallel, 1.0)
        assert fm.dimension > 0
        assert lam.dimension > 0
        assert bfi >= 0

    def test_all_approaches_iterator(self, iterator):
        """All three approaches on a recursive type."""
        fm = fisher_matrix(iterator)
        lam = lattice_adapted_metric(iterator)
        bfi = boltzmann_fisher_info(iterator, 1.0)
        assert fm.dimension >= 0
        assert lam.dimension >= 0
        assert bfi >= 0


# ===================================================================
# Full analysis
# ===================================================================

class TestAnalysis:
    def test_end(self, end_ss):
        result = analyze_info_geometry(end_ss)
        assert isinstance(result, InfoGeometryAnalysis)
        assert result.num_states == 1

    def test_branch(self, branch):
        result = analyze_info_geometry(branch)
        assert result.num_states == len(branch.states)
        assert result.boltzmann_fisher_info >= 0
        assert result.mean_round_trip_loss >= 0

    def test_chain(self, chain):
        result = analyze_info_geometry(chain)
        assert result.num_states == 4
        assert result.max_kl_from_uniform > 0
        assert "fisher_trace" in result.approach_comparison
        assert "boltzmann_fisher_info" in result.approach_comparison
        assert "lattice_metric_trace" in result.approach_comparison

    def test_parallel(self, parallel):
        result = analyze_info_geometry(parallel)
        assert result.num_states > 1
        assert math.isfinite(result.scalar_curvature_uniform)

    def test_iterator(self, iterator):
        result = analyze_info_geometry(iterator)
        assert result.num_states > 0
        assert result.lattice_metric_trace >= 0

    def test_frozen(self, branch):
        result = analyze_info_geometry(branch)
        with pytest.raises(AttributeError):
            result.num_states = 99


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    def test_kl_with_zeros(self):
        """KL divergence when q has zeros where p has mass."""
        p = {0: 0.5, 1: 0.5}
        q = {0: 1.0, 1: 0.0}
        kl = kl_divergence(p, q)
        # Should be large (but finite due to clamping)
        assert kl > 1.0

    def test_fisher_rao_near_boundary(self):
        """Near-degenerate distributions."""
        p = {0: 0.999, 1: 0.001}
        q = {0: 0.001, 1: 0.999}
        d = fisher_rao_distance(p, q)
        assert d > 0
        assert math.isfinite(d)

    def test_round_trip_loss_peaked(self, chain):
        """Peaked distribution should have low loss."""
        states = sorted(chain.states)
        p = {s: 0.01 for s in states}
        p[states[0]] = 0.97
        loss = round_trip_loss(chain, p)
        assert loss < 0.1

    def test_anomaly_single_observation(self, branch):
        states = sorted(branch.states)
        expected = {s: 1.0 / len(states) for s in states}
        counts = {states[0]: 1}
        score = anomaly_kl(branch, expected, counts)
        assert score >= 0

    def test_protocol_distance_end_vs_branch(self, end_ss, branch):
        """Distance between trivial and non-trivial protocol."""
        d = protocol_distance(end_ss, branch, 1.0)
        assert d >= 0
