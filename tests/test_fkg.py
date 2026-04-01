"""Tests for correlation inequalities on session type lattices (Step 30z)."""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.fkg import (
    CorrelationResult,
    StochasticDominance,
    FourFunctionsResult,
    CorrelationAnalysis,
    check_harris,
    check_holley,
    check_fkg,
    check_ahlswede_daykin,
    is_monotone_increasing,
    is_monotone_decreasing,
    is_log_supermodular,
    generate_random_monotone,
    fkg_gap,
    correlation_profile,
    analyze_correlation,
)


def _build(type_string: str):
    return build_statespace(parse(type_string))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def end_ss():
    return _build("end")

@pytest.fixture
def simple_branch():
    return _build("&{a: end, b: end}")

@pytest.fixture
def deep_chain():
    return _build("&{a: &{b: &{c: end}}}")

@pytest.fixture
def iterator_ss():
    return _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")

@pytest.fixture
def parallel_ss():
    return _build("(&{a: end} || &{b: end})")

@pytest.fixture
def rest_ss():
    return _build("&{get: +{OK: end, NOT_FOUND: end}, put: +{OK: end, ERR: end}}")


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------

class TestMonotonicity:
    def test_constant_is_monotone(self, simple_branch):
        f = {s: 1.0 for s in simple_branch.states}
        assert is_monotone_increasing(simple_branch, f)

    def test_rank_is_monotone(self, deep_chain):
        from reticulate.fkg import _rank
        r = _rank(deep_chain)
        f = {s: float(r[s]) for s in deep_chain.states}
        assert is_monotone_increasing(deep_chain, f)

    def test_anti_rank_is_decreasing(self, deep_chain):
        from reticulate.fkg import _rank
        r = _rank(deep_chain)
        f = {s: -float(r[s]) for s in deep_chain.states}
        assert is_monotone_decreasing(deep_chain, f)

    def test_random_monotone_is_monotone(self, simple_branch):
        f = generate_random_monotone(simple_branch, seed=42)
        assert is_monotone_increasing(simple_branch, f)

    def test_random_monotone_different_seeds(self, simple_branch):
        f1 = generate_random_monotone(simple_branch, seed=1)
        f2 = generate_random_monotone(simple_branch, seed=2)
        assert f1 != f2

    def test_non_monotone_detected(self, deep_chain):
        """A function that assigns higher values to lower states is not monotone."""
        from reticulate.fkg import _rank
        r = _rank(deep_chain)
        f = {s: -float(r[s]) for s in deep_chain.states}
        assert not is_monotone_increasing(deep_chain, f)


# ---------------------------------------------------------------------------
# Harris inequality
# ---------------------------------------------------------------------------

class TestHarris:
    def test_end(self, end_ss):
        f = {s: 1.0 for s in end_ss.states}
        r = check_harris(end_ss, f, f)
        assert r.holds
        assert r.inequality == "harris"

    def test_constant_functions(self, simple_branch):
        f = {s: 1.0 for s in simple_branch.states}
        g = {s: 2.0 for s in simple_branch.states}
        r = check_harris(simple_branch, f, g)
        assert r.holds
        assert abs(r.gap) < 1e-10  # Equality for constants

    def test_monotone_pair(self, deep_chain):
        f = generate_random_monotone(deep_chain, seed=42)
        g = generate_random_monotone(deep_chain, seed=43)
        r = check_harris(deep_chain, f, g)
        assert r.holds

    def test_harris_on_parallel(self, parallel_ss):
        f = generate_random_monotone(parallel_ss, seed=1)
        g = generate_random_monotone(parallel_ss, seed=2)
        r = check_harris(parallel_ss, f, g)
        assert r.holds

    def test_harris_result_structure(self, simple_branch):
        f = generate_random_monotone(simple_branch, seed=1)
        g = generate_random_monotone(simple_branch, seed=2)
        r = check_harris(simple_branch, f, g)
        assert isinstance(r, CorrelationResult)
        assert isinstance(r.gap, float)
        assert isinstance(r.holds, bool)

    def test_harris_on_rest(self, rest_ss):
        f = generate_random_monotone(rest_ss, seed=10)
        g = generate_random_monotone(rest_ss, seed=20)
        r = check_harris(rest_ss, f, g)
        assert r.holds


# ---------------------------------------------------------------------------
# FKG inequality
# ---------------------------------------------------------------------------

class TestFKG:
    def test_fkg_uniform(self, simple_branch):
        """FKG with uniform measure (reduces to Harris)."""
        mu = {s: 1.0 for s in simple_branch.states}
        f = generate_random_monotone(simple_branch, seed=1)
        g = generate_random_monotone(simple_branch, seed=2)
        r = check_fkg(simple_branch, f, g, mu)
        assert r.holds
        assert r.inequality == "fkg"

    def test_fkg_rank_measure(self, deep_chain):
        from reticulate.fkg import _rank_exponential_measure
        mu = _rank_exponential_measure(deep_chain)
        f = generate_random_monotone(deep_chain, seed=1)
        g = generate_random_monotone(deep_chain, seed=2)
        r = check_fkg(deep_chain, f, g, mu)
        assert r.holds

    def test_fkg_on_parallel(self, parallel_ss):
        from reticulate.fkg import _rank_exponential_measure
        mu = _rank_exponential_measure(parallel_ss)
        f = generate_random_monotone(parallel_ss, seed=5)
        g = generate_random_monotone(parallel_ss, seed=6)
        r = check_fkg(parallel_ss, f, g, mu)
        assert r.holds

    def test_fkg_gap_nonnegative(self, simple_branch):
        mu = {s: 1.0 for s in simple_branch.states}
        f = generate_random_monotone(simple_branch, seed=1)
        g = generate_random_monotone(simple_branch, seed=2)
        gap = fkg_gap(simple_branch, f, g, mu)
        assert gap >= -1e-10

    def test_fkg_zero_measure(self, simple_branch):
        mu = {s: 0.0 for s in simple_branch.states}
        f = {s: 1.0 for s in simple_branch.states}
        r = check_fkg(simple_branch, f, f, mu)
        assert r.holds  # Trivially true


# ---------------------------------------------------------------------------
# Log-supermodularity
# ---------------------------------------------------------------------------

class TestLogSupermodular:
    def test_uniform_is_log_super(self, simple_branch):
        mu = {s: 1.0 for s in simple_branch.states}
        assert is_log_supermodular(simple_branch, mu)

    def test_rank_exponential_is_log_super(self, parallel_ss):
        from reticulate.fkg import _rank_exponential_measure
        mu = _rank_exponential_measure(parallel_ss)
        assert is_log_supermodular(parallel_ss, mu)

    def test_constant_is_log_super(self, deep_chain):
        mu = {s: 5.0 for s in deep_chain.states}
        assert is_log_supermodular(deep_chain, mu)


# ---------------------------------------------------------------------------
# Holley inequality
# ---------------------------------------------------------------------------

class TestHolley:
    def test_holley_uniform_vs_rank(self, simple_branch):
        from reticulate.fkg import _uniform_measure, _rank_exponential_measure
        mu1 = _uniform_measure(simple_branch)
        mu2 = _rank_exponential_measure(simple_branch)
        f = generate_random_monotone(simple_branch, seed=1)
        r = check_holley(simple_branch, mu1, mu2, f)
        assert isinstance(r, StochasticDominance)

    def test_holley_same_measure(self, deep_chain):
        mu = {s: 1.0 for s in deep_chain.states}
        f = generate_random_monotone(deep_chain, seed=1)
        r = check_holley(deep_chain, mu, mu, f)
        assert r.holley_condition  # Same measure always satisfies Holley condition
        assert abs(r.expectation_gap) < 1e-10  # Same expectations

    def test_holley_result_structure(self, simple_branch):
        mu1 = {s: 1.0 for s in simple_branch.states}
        mu2 = {s: 2.0 for s in simple_branch.states}
        f = {s: 1.0 for s in simple_branch.states}
        r = check_holley(simple_branch, mu1, mu2, f)
        assert isinstance(r.dominates, bool)
        assert isinstance(r.holley_condition, bool)
        assert isinstance(r.expectation_gap, float)


# ---------------------------------------------------------------------------
# Ahlswede-Daykin
# ---------------------------------------------------------------------------

class TestAhlswedeDaykin:
    def test_ad_constant_functions(self, simple_branch):
        """Constant functions: α=β=γ=δ=1 → (n)(n) ≤ (n)(n). Equality."""
        c = {s: 1.0 for s in simple_branch.states}
        r = check_ahlswede_daykin(simple_branch, c, c, c, c)
        assert r.holds
        assert r.pointwise_holds
        assert abs(r.lhs - r.rhs) < 1e-10

    def test_ad_result_structure(self, simple_branch):
        c = {s: 1.0 for s in simple_branch.states}
        r = check_ahlswede_daykin(simple_branch, c, c, c, c)
        assert isinstance(r, FourFunctionsResult)
        assert isinstance(r.sum_alpha, float)
        assert isinstance(r.pointwise_holds, bool)

    def test_ad_on_chain(self, deep_chain):
        c = {s: 1.0 for s in deep_chain.states}
        r = check_ahlswede_daykin(deep_chain, c, c, c, c)
        assert r.holds

    def test_ad_measure_derived(self, simple_branch):
        """Use μ-derived functions: should recover FKG."""
        from reticulate.fkg import _rank_exponential_measure
        mu = _rank_exponential_measure(simple_branch)
        f = generate_random_monotone(simple_branch, seed=1)
        g = generate_random_monotone(simple_branch, seed=2)
        alpha = {s: mu[s] * f.get(s, 0) for s in simple_branch.states}
        beta = {s: mu[s] * g.get(s, 0) for s in simple_branch.states}
        gamma = {s: mu[s] * f.get(s, 0) * g.get(s, 0) for s in simple_branch.states}
        delta = dict(mu)
        r = check_ahlswede_daykin(simple_branch, alpha, beta, gamma, delta)
        assert isinstance(r, FourFunctionsResult)


# ---------------------------------------------------------------------------
# Correlation profile
# ---------------------------------------------------------------------------

class TestCorrelationProfile:
    def test_profile_structure(self, simple_branch):
        p = correlation_profile(simple_branch, n_trials=5)
        assert "harris_avg_gap" in p
        assert "fkg_avg_gap" in p
        assert "harris_all_hold" in p
        assert "fkg_all_hold" in p
        assert p["n_trials"] == 5

    def test_profile_on_chain(self, deep_chain):
        p = correlation_profile(deep_chain, n_trials=10)
        assert p["harris_all_hold"]
        assert p["fkg_all_hold"]

    def test_profile_on_parallel(self, parallel_ss):
        p = correlation_profile(parallel_ss, n_trials=5)
        assert isinstance(p["harris_avg_gap"], float)


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

class TestAnalysis:
    def test_analysis_end(self, end_ss):
        a = analyze_correlation(end_ss)
        assert isinstance(a, CorrelationAnalysis)
        assert a.num_states == 1

    def test_analysis_branch(self, simple_branch):
        a = analyze_correlation(simple_branch)
        assert a.harris.holds
        assert a.fkg.holds
        assert isinstance(a.is_distributive, bool)

    def test_analysis_chain(self, deep_chain):
        a = analyze_correlation(deep_chain)
        assert a.harris.holds
        assert a.fkg.holds

    def test_analysis_parallel(self, parallel_ss):
        a = analyze_correlation(parallel_ss)
        assert a.harris.holds
        assert isinstance(a.avg_fkg_gap, float)

    def test_analysis_iterator(self, iterator_ss):
        a = analyze_correlation(iterator_ss)
        assert isinstance(a, CorrelationAnalysis)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

class TestBenchmarks:
    @pytest.mark.parametrize("type_string,name", [
        ("&{a: end, b: end}", "SimpleBranch"),
        ("&{a: &{b: &{c: end}}}", "DeepChain"),
        ("+{ok: end, err: end}", "SimpleSelect"),
        ("(&{a: end} || &{b: end})", "SimpleParallel"),
        ("&{get: +{OK: end, NOT_FOUND: end}, put: +{OK: end, ERR: end}}", "REST"),
        ("rec X . &{request: +{OK: X, ERR: end}}", "RetryLoop"),
        ("&{a: end, b: end, c: end, d: end}", "WideBranch"),
        ("&{a: &{c: end, d: end}, b: &{e: end}}", "Asymmetric"),
    ])
    def test_harris_on_benchmarks(self, type_string, name):
        ss = _build(type_string)
        f = generate_random_monotone(ss, seed=100)
        g = generate_random_monotone(ss, seed=200)
        r = check_harris(ss, f, g)
        assert r.holds, f"Harris should hold on {name}"

    @pytest.mark.parametrize("type_string,name", [
        ("&{a: end, b: end}", "SimpleBranch"),
        ("&{a: &{b: &{c: end}}}", "DeepChain"),
        ("(&{a: end} || &{b: end})", "SimpleParallel"),
        ("&{get: +{OK: end, NOT_FOUND: end}, put: +{OK: end, ERR: end}}", "REST"),
    ])
    def test_fkg_on_benchmarks(self, type_string, name):
        ss = _build(type_string)
        from reticulate.fkg import _rank_exponential_measure
        mu = _rank_exponential_measure(ss)
        f = generate_random_monotone(ss, seed=100)
        g = generate_random_monotone(ss, seed=200)
        r = check_fkg(ss, f, g, mu)
        assert r.holds, f"FKG should hold on {name}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_state(self):
        ss = _build("end")
        a = analyze_correlation(ss)
        assert a.harris.holds
        assert a.fkg.holds

    def test_two_states(self):
        ss = _build("&{a: end}")
        f = generate_random_monotone(ss, seed=1)
        assert is_monotone_increasing(ss, f)
        r = check_harris(ss, f, f)
        assert r.holds

    def test_identical_functions(self, simple_branch):
        f = generate_random_monotone(simple_branch, seed=1)
        r = check_harris(simple_branch, f, f)
        assert r.holds
        assert r.gap >= -1e-10  # E[f²] ≥ E[f]²

    def test_zero_function(self, simple_branch):
        f = {s: 0.0 for s in simple_branch.states}
        g = generate_random_monotone(simple_branch, seed=1)
        r = check_harris(simple_branch, f, g)
        assert r.holds  # 0 ≥ 0

    def test_recursive_type(self):
        """Pure cycle rec X . &{a: X} — degenerate: bottom not in states."""
        ss = _build("rec X . &{a: X}")
        # This type has bottom=0 not in states={2} — a known degenerate case.
        # Harris still works (no meet/join needed for uniform measure).
        f = generate_random_monotone(ss, seed=1)
        r = check_harris(ss, f, f)
        assert r.holds
