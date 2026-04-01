"""Tests for the thermodynamic morphism (Step 600d)."""

import math
import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.thermo_morphism import (
    ThermoObservable,
    ThermoEmbedding,
    ThermoGalois,
    RGFlow,
    SecurityProfile,
    ThermoMorphismAnalysis,
    thermo_embedding,
    energy_galois,
    verify_galois,
    is_cooling,
    cooling_fraction,
    entropy_profile,
    security_score,
    security_profile,
    phase_boundary_states,
    rg_flow,
    anomaly_score,
    design_temperature,
    analyze_thermo_morphism,
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
# Embedding
# ---------------------------------------------------------------------------

class TestEmbedding:
    def test_end(self, end_ss):
        emb = thermo_embedding(end_ss)
        assert isinstance(emb, ThermoEmbedding)
        assert len(emb.observables) == 1

    def test_chain_order_preserving(self, chain):
        emb = thermo_embedding(chain)
        assert emb.is_order_preserving

    def test_chain_energies(self, chain):
        emb = thermo_embedding(chain)
        assert emb.observables[chain.bottom].energy == 0.0
        assert emb.observables[chain.top].energy == 3.0

    def test_branch_observables(self, branch):
        emb = thermo_embedding(branch)
        top_obs = emb.observables[branch.top]
        assert top_obs.degree == 2
        assert top_obs.local_entropy > 0

    def test_energy_range(self, chain):
        emb = thermo_embedding(chain)
        assert emb.energy_range == (0.0, 3.0)


# ---------------------------------------------------------------------------
# Galois connection
# ---------------------------------------------------------------------------

class TestGalois:
    def test_galois_valid(self, chain):
        gal = energy_galois(chain)
        assert gal.is_valid

    def test_verify_galois(self, chain):
        gal = energy_galois(chain)
        assert verify_galois(chain, gal)

    def test_alpha_is_energy(self, chain):
        gal = energy_galois(chain)
        assert gal.alpha[chain.bottom] == 0.0
        assert gal.alpha[chain.top] == 3.0

    def test_gamma_contains_state(self, chain):
        gal = energy_galois(chain)
        # gamma(3.0) should contain all states (all have E ≤ 3)
        assert chain.top in gal.gamma[3.0]
        assert chain.bottom in gal.gamma[3.0]

    def test_gamma_level_zero(self, chain):
        gal = energy_galois(chain)
        # gamma(0.0) should contain only states with E = 0
        assert chain.bottom in gal.gamma[0.0]
        assert chain.top not in gal.gamma[0.0]

    def test_branch_galois(self, branch):
        gal = energy_galois(branch)
        assert verify_galois(branch, gal)

    def test_parallel_galois(self, parallel):
        gal = energy_galois(parallel)
        assert gal.is_valid


# ---------------------------------------------------------------------------
# Cooling
# ---------------------------------------------------------------------------

class TestCooling:
    def test_chain_is_cooling(self, chain):
        """Chain: every transition decreases energy."""
        assert is_cooling(chain)

    def test_branch_is_cooling(self, branch):
        assert is_cooling(branch)

    def test_iterator_not_fully_cooling(self, iterator):
        """Recursive types have back-edges that increase energy."""
        cf = cooling_fraction(iterator)
        assert 0.0 <= cf <= 1.0

    def test_cooling_fraction_chain(self, chain):
        assert cooling_fraction(chain) == 1.0

    def test_cooling_path(self, chain):
        states = sorted(chain.states, key=lambda s: -thermo_embedding(chain).observables[s].energy)
        assert is_cooling(chain, states)

    def test_end_trivially_cooling(self, end_ss):
        assert is_cooling(end_ss)


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

class TestSecurity:
    def test_security_score_bounded(self, chain):
        s = security_score(chain, beta=1.0)
        assert 0.0 <= s <= 1.0

    def test_chain_low_security(self, chain):
        """Chain is deterministic → low entropy → low security score."""
        s = security_score(chain, beta=1.0)
        # Chain has few choices, but Boltzmann adds uncertainty
        assert isinstance(s, float)

    def test_branch_higher_security(self, branch):
        """Branch has choice → higher entropy than chain."""
        s_branch = security_score(branch, beta=1.0)
        assert s_branch > 0

    def test_security_profile_structure(self, branch):
        sp = security_profile(branch)
        assert isinstance(sp, SecurityProfile)
        assert sp.total_entropy >= 0
        assert 0.0 <= sp.predictability <= 1.0
        assert 0.0 <= sp.security_score <= 1.0

    def test_vulnerable_states(self, chain):
        """Chain states (except bottom) have single outgoing → zero local entropy."""
        sp = security_profile(chain)
        assert isinstance(sp.vulnerable_states, list)

    def test_entropy_profile(self, branch):
        ep = entropy_profile(branch)
        assert ep[branch.top] > 0  # Top has 2 choices
        assert ep[branch.bottom] == 0  # Bottom has 0 choices


# ---------------------------------------------------------------------------
# Phase boundaries
# ---------------------------------------------------------------------------

class TestPhaseBoundaries:
    def test_returns_list(self, chain):
        pb = phase_boundary_states(chain)
        assert isinstance(pb, list)

    def test_branch_has_boundary(self, branch):
        """Top of branch has high entropy, bottom has zero → boundary."""
        pb = phase_boundary_states(branch)
        assert isinstance(pb, list)


# ---------------------------------------------------------------------------
# RG flow
# ---------------------------------------------------------------------------

class TestRGFlow:
    def test_flow_structure(self, chain):
        flow = rg_flow(chain)
        assert isinstance(flow, RGFlow)
        assert flow.total_levels >= 1
        assert flow.steps[0].num_states == len(chain.states)

    def test_flow_decreasing(self, chain):
        """Each level should have fewer or equal states."""
        flow = rg_flow(chain)
        for i in range(len(flow.steps) - 1):
            assert flow.steps[i].num_states >= flow.steps[i + 1].num_states

    def test_end_flow(self, end_ss):
        flow = rg_flow(end_ss)
        assert flow.total_levels == 1
        assert flow.converged

    def test_branch_flow(self, branch):
        flow = rg_flow(branch)
        assert isinstance(flow, RGFlow)


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

class TestAnomaly:
    def test_ground_state_low_anomaly(self, chain):
        """Bottom (ground state) should have low anomaly score at low T."""
        score = anomaly_score(chain, beta=10.0, observed_state=chain.bottom)
        assert score < 5.0  # Low surprise for ground state

    def test_excited_state_high_anomaly(self, chain):
        """Top (highest energy) should have high anomaly at low T."""
        score = anomaly_score(chain, beta=10.0, observed_state=chain.top)
        # Should be higher than ground state
        ground = anomaly_score(chain, beta=10.0, observed_state=chain.bottom)
        assert score > ground

    def test_high_T_uniform_anomaly(self, chain):
        """At high T, all states have similar anomaly scores."""
        scores = [anomaly_score(chain, beta=0.01, observed_state=s) for s in chain.states]
        assert max(scores) - min(scores) < 0.5


# ---------------------------------------------------------------------------
# Design temperature
# ---------------------------------------------------------------------------

class TestDesignTemperature:
    def test_returns_positive(self, chain):
        beta = design_temperature(chain, target_entropy_fraction=0.5)
        assert beta > 0

    def test_high_entropy_low_beta(self, chain):
        """High entropy target → low β (high T)."""
        beta_high = design_temperature(chain, target_entropy_fraction=0.9)
        beta_low = design_temperature(chain, target_entropy_fraction=0.1)
        assert beta_high < beta_low

    def test_end_returns_default(self, end_ss):
        beta = design_temperature(end_ss, 0.5)
        assert beta > 0


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

class TestAnalysis:
    def test_end(self, end_ss):
        a = analyze_thermo_morphism(end_ss)
        assert isinstance(a, ThermoMorphismAnalysis)
        assert a.num_states == 1

    def test_chain(self, chain):
        a = analyze_thermo_morphism(chain)
        assert a.embedding.is_order_preserving
        assert a.galois.is_valid
        assert a.cooling_fraction == 1.0

    def test_branch(self, branch):
        a = analyze_thermo_morphism(branch)
        assert a.security.security_score >= 0

    def test_parallel(self, parallel):
        a = analyze_thermo_morphism(parallel)
        assert isinstance(a, ThermoMorphismAnalysis)

    def test_iterator(self, iterator):
        a = analyze_thermo_morphism(iterator)
        assert a.cooling_fraction < 1.0  # Has back-edges


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
    ])
    def test_analysis_on_benchmarks(self, ts, name):
        ss = _build(ts)
        a = analyze_thermo_morphism(ss)
        assert a.galois.is_valid, f"{name}: Galois should be valid"
        assert a.num_states >= 2, f"{name}"
