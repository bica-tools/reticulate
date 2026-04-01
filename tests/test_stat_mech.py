"""Tests for statistical mechanics of session type lattices (Step 600c)."""

import math
import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.stat_mech import (
    ThermodynamicState,
    PhaseTransition,
    StatMechAnalysis,
    rank_energy,
    degree_energy,
    partition_function,
    free_energy,
    internal_energy,
    entropy,
    specific_heat,
    boltzmann_distribution,
    thermodynamic_curve,
    detect_phase_transitions,
    check_product_factorization,
    tropical_limit,
    ground_degeneracy,
    analyze_stat_mech,
)


def _build(s: str):
    return build_statespace(parse(s))


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


# ---------------------------------------------------------------------------
# Energy functions
# ---------------------------------------------------------------------------

class TestEnergy:
    def test_rank_bottom_is_zero(self, chain):
        e = rank_energy(chain)
        assert e[chain.bottom] == 0.0

    def test_rank_monotone(self, chain):
        e = rank_energy(chain)
        assert e[chain.top] >= e[chain.bottom]

    def test_rank_chain_values(self, chain):
        e = rank_energy(chain)
        assert e[chain.top] == 3.0  # 4-state chain: ranks 0,1,2,3

    def test_degree_end_is_zero(self, end_ss):
        e = degree_energy(end_ss)
        assert e[end_ss.bottom] == 0.0

    def test_degree_branch_top(self, branch):
        e = degree_energy(branch)
        assert e[branch.top] == 2.0  # Two outgoing transitions


# ---------------------------------------------------------------------------
# Partition function
# ---------------------------------------------------------------------------

class TestPartitionFunction:
    def test_end_Z_is_one(self, end_ss):
        """Z(end) = exp(0) = 1 for any β."""
        Z = partition_function(end_ss, beta=1.0)
        assert abs(Z - 1.0) < 1e-10

    def test_high_T_limit(self, chain):
        """β → 0: Z → |L| (all states equally weighted)."""
        Z = partition_function(chain, beta=0.001)
        assert abs(Z - len(chain.states)) < 0.1

    def test_low_T_limit(self, chain):
        """β → ∞: Z → g·exp(-β·E_min) where g = ground degeneracy."""
        e = rank_energy(chain)
        beta = 100.0
        Z = partition_function(chain, beta, e)
        e_min = min(e.values())
        g = sum(1 for v in e.values() if abs(v - e_min) < 1e-10)
        expected = g * math.exp(-beta * e_min)
        assert abs(Z - expected) / max(Z, expected) < 0.01

    def test_Z_positive(self, branch):
        Z = partition_function(branch, beta=1.0)
        assert Z > 0

    def test_chain_exact(self, chain):
        """Z for 4-state chain with rank energy: Σ exp(-β·k) for k=0..3."""
        beta = 1.0
        e = rank_energy(chain)
        Z = partition_function(chain, beta, e)
        expected = sum(math.exp(-beta * k) for k in range(4))
        assert abs(Z - expected) < 1e-10


# ---------------------------------------------------------------------------
# Thermodynamic quantities
# ---------------------------------------------------------------------------

class TestThermodynamics:
    def test_free_energy_finite(self, chain):
        F = free_energy(chain, beta=1.0)
        assert math.isfinite(F)

    def test_internal_energy_bounded(self, chain):
        e = rank_energy(chain)
        U = internal_energy(chain, beta=1.0, energy=e)
        e_min = min(e.values())
        e_max = max(e.values())
        assert e_min <= U <= e_max

    def test_entropy_nonneg(self, chain):
        S = entropy(chain, beta=1.0)
        assert S >= -1e-10  # Entropy should be non-negative

    def test_entropy_high_T(self, chain):
        """At high T, entropy → ln|L|."""
        S = entropy(chain, beta=0.001)
        expected = math.log(len(chain.states))
        assert abs(S - expected) < 0.1

    def test_specific_heat_nonneg(self, chain):
        C = specific_heat(chain, beta=1.0)
        assert C >= -1e-10

    def test_thermodynamic_identity(self, chain):
        """F = U - T·S, i.e., F = U - S/β."""
        beta = 1.0
        F = free_energy(chain, beta)
        U = internal_energy(chain, beta)
        S = entropy(chain, beta)
        # S = β(U - F) → F = U - S/β
        assert abs(F - (U - S / beta)) < 1e-6


# ---------------------------------------------------------------------------
# Boltzmann distribution
# ---------------------------------------------------------------------------

class TestBoltzmann:
    def test_sums_to_one(self, chain):
        P = boltzmann_distribution(chain, beta=1.0)
        assert abs(sum(P.values()) - 1.0) < 1e-10

    def test_all_nonneg(self, chain):
        P = boltzmann_distribution(chain, beta=1.0)
        assert all(p >= 0 for p in P.values())

    def test_high_T_uniform(self, branch):
        """At high T, all states equally likely."""
        P = boltzmann_distribution(branch, beta=0.001)
        n = len(branch.states)
        for p in P.values():
            assert abs(p - 1.0 / n) < 0.01

    def test_low_T_ground_state(self, chain):
        """At low T, ground state dominates."""
        P = boltzmann_distribution(chain, beta=100.0)
        assert P[chain.bottom] > 0.99


# ---------------------------------------------------------------------------
# Product factorization
# ---------------------------------------------------------------------------

class TestProductFactorization:
    def test_parallel_factorizes(self, parallel):
        assert check_product_factorization(parallel, beta=1.0)

    def test_non_product_trivially_true(self, chain):
        assert check_product_factorization(chain, beta=1.0)

    def test_parallel_exact(self, parallel):
        """Z(L₁×L₂) = Z(L₁)·Z(L₂) exactly."""
        e = rank_energy(parallel)
        Z_total = partition_function(parallel, 1.0, e)
        assert Z_total > 0
        if parallel.product_factors:
            Z_prod = 1.0
            for f in parallel.product_factors:
                Z_prod *= partition_function(f, 1.0, rank_energy(f))
            assert abs(Z_total - Z_prod) / Z_total < 0.01


# ---------------------------------------------------------------------------
# Tropical limit
# ---------------------------------------------------------------------------

class TestTropicalLimit:
    def test_chain_ground_is_zero(self, chain):
        tl = tropical_limit(chain)
        assert tl == 0.0  # Bottom has rank 0

    def test_ground_degeneracy_chain(self, chain):
        g = ground_degeneracy(chain)
        assert g == 1  # Only bottom has rank 0

    def test_ground_degeneracy_branch(self, branch):
        """&{a: end, b: end}: bottom is the only rank-0 state."""
        g = ground_degeneracy(branch)
        assert g == 1


# ---------------------------------------------------------------------------
# Thermodynamic curve
# ---------------------------------------------------------------------------

class TestCurve:
    def test_curve_length(self, chain):
        betas = [0.1, 0.5, 1.0, 2.0, 5.0]
        curve = thermodynamic_curve(chain, betas)
        assert len(curve) == 5

    def test_curve_elements(self, chain):
        curve = thermodynamic_curve(chain, [1.0])
        assert isinstance(curve[0], ThermodynamicState)
        assert curve[0].beta == 1.0
        assert curve[0].Z > 0

    def test_entropy_decreases_with_beta(self, chain):
        """Entropy should decrease as β increases (cooling)."""
        betas = [0.1, 1.0, 10.0]
        curve = thermodynamic_curve(chain, betas)
        for i in range(len(curve) - 1):
            assert curve[i].S >= curve[i + 1].S - 0.01


# ---------------------------------------------------------------------------
# Phase transitions
# ---------------------------------------------------------------------------

class TestPhaseTransitions:
    def test_returns_list(self, chain):
        pts = detect_phase_transitions(chain)
        assert isinstance(pts, list)

    def test_transition_structure(self, chain):
        pts = detect_phase_transitions(chain)
        for pt in pts:
            assert isinstance(pt, PhaseTransition)
            assert pt.beta_c > 0
            assert pt.specific_heat_peak > 0


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

class TestAnalysis:
    def test_end(self, end_ss):
        a = analyze_stat_mech(end_ss)
        assert isinstance(a, StatMechAnalysis)
        assert a.num_states == 1
        assert a.tropical_limit == 0.0

    def test_chain(self, chain):
        a = analyze_stat_mech(chain)
        assert a.num_states == 4
        assert a.energy_range == (0.0, 3.0)
        assert a.low_T_ground_degeneracy == 1

    def test_branch(self, branch):
        a = analyze_stat_mech(branch)
        assert a.num_states >= 2

    def test_parallel(self, parallel):
        a = analyze_stat_mech(parallel)
        assert a.product_factorizes

    def test_high_T_entropy(self, chain):
        a = analyze_stat_mech(chain)
        assert abs(a.high_T_entropy - math.log(4)) < 1e-10


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

class TestBenchmarks:
    @pytest.mark.parametrize("ts,name", [
        ("&{a: end, b: end}", "Branch"),
        ("&{a: &{b: &{c: end}}}", "Chain"),
        ("+{ok: end, err: end}", "Select"),
        ("(&{a: end} || &{b: end})", "Parallel"),
        ("&{get: +{OK: end, NOT_FOUND: end}}", "REST"),
        ("rec X . &{req: +{OK: X, ERR: end}}", "Retry"),
    ])
    def test_analysis_on_benchmarks(self, ts, name):
        ss = _build(ts)
        a = analyze_stat_mech(ss)
        assert a.num_states >= 1, f"{name}"
        assert a.tropical_limit >= 0, f"{name}"
        assert len(a.curve) > 0, f"{name}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_state(self):
        ss = _build("end")
        assert partition_function(ss, 1.0) == 1.0
        assert ground_degeneracy(ss) == 1

    def test_recursive_type(self):
        ss = _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        a = analyze_stat_mech(ss)
        assert isinstance(a, StatMechAnalysis)

    def test_very_high_beta(self, chain):
        """Very low temperature: should not overflow."""
        Z = partition_function(chain, beta=50.0)
        assert Z > 0 and math.isfinite(Z)

    def test_very_low_beta(self, chain):
        """Very high temperature: Z ≈ |L|."""
        Z = partition_function(chain, beta=1e-6)
        assert abs(Z - len(chain.states)) < 0.01
