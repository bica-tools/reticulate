"""Tests for lattice valuations and FKG inequality."""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.valuation import (
    ValuationResult,
    FKGResult,
    ValuationAnalysis,
    rank_function,
    height_function,
    check_valuation,
    is_rank_valuation,
    valuation_defect,
    is_log_supermodular,
    check_fkg,
    monotone_correlation,
    analyze_valuations,
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
def nested():
    return _build("&{a: &{c: end, d: end}, b: &{e: end, f: end}}")


# ---------------------------------------------------------------------------
# Rank function
# ---------------------------------------------------------------------------

class TestRankFunction:
    def test_end(self, end_ss):
        r = rank_function(end_ss)
        assert r[end_ss.bottom] == 0

    def test_simple_branch(self, simple_branch):
        r = rank_function(simple_branch)
        assert r[simple_branch.bottom] == 0
        assert r[simple_branch.top] >= 1

    def test_deep_chain(self, deep_chain):
        r = rank_function(deep_chain)
        assert r[deep_chain.bottom] == 0
        assert r[deep_chain.top] == 3

    def test_parallel(self, parallel_ss):
        r = rank_function(parallel_ss)
        assert r[parallel_ss.bottom] == 0
        assert r[parallel_ss.top] >= 2

    def test_monotone(self, simple_branch):
        """Rank should be monotone: x >= y implies rank(x) >= rank(y)."""
        r = rank_function(simple_branch)
        assert r[simple_branch.top] >= r[simple_branch.bottom]


# ---------------------------------------------------------------------------
# Height function
# ---------------------------------------------------------------------------

class TestHeightFunction:
    def test_end(self, end_ss):
        h = height_function(end_ss)
        assert h[end_ss.bottom] == 0

    def test_deep_chain(self, deep_chain):
        h = height_function(deep_chain)
        assert h[deep_chain.bottom] == 0
        assert h[deep_chain.top] == 3

    def test_simple_branch(self, simple_branch):
        h = height_function(simple_branch)
        assert h[simple_branch.bottom] == 0
        assert h[simple_branch.top] >= 1

    def test_iterator(self, iterator_ss):
        h = height_function(iterator_ss)
        assert h[iterator_ss.bottom] == 0


# ---------------------------------------------------------------------------
# Valuation checking
# ---------------------------------------------------------------------------

class TestCheckValuation:
    def test_constant_is_valuation(self, simple_branch):
        """Constant function is always a valuation."""
        v = {s: 1.0 for s in simple_branch.states}
        r = check_valuation(simple_branch, v)
        assert r.is_valuation

    def test_rank_on_chain(self, deep_chain):
        """Rank is a valuation on a chain (trivially — no incomparable pairs)."""
        rk = rank_function(deep_chain)
        v = {s: float(rk[s]) for s in deep_chain.states}
        r = check_valuation(deep_chain, v)
        assert r.is_valuation

    def test_result_structure(self, simple_branch):
        v = {s: 0.0 for s in simple_branch.states}
        r = check_valuation(simple_branch, v)
        assert isinstance(r, ValuationResult)
        assert isinstance(r.is_valuation, bool)
        assert isinstance(r.max_defect, float)
        assert r.max_defect >= 0

    def test_rank_on_parallel(self, parallel_ss):
        """Rank on parallel (product lattice) — should be modular hence valuation."""
        rk = rank_function(parallel_ss)
        v = {s: float(rk[s]) for s in parallel_ss.states}
        r = check_valuation(parallel_ss, v)
        # Product lattice is distributive (hence modular) — rank IS a valuation
        assert r.is_valuation


# ---------------------------------------------------------------------------
# is_rank_valuation
# ---------------------------------------------------------------------------

class TestIsRankValuation:
    def test_end(self, end_ss):
        assert is_rank_valuation(end_ss)

    def test_chain(self, deep_chain):
        assert is_rank_valuation(deep_chain)

    def test_simple_branch(self, simple_branch):
        # Simple branch is distributive → modular → rank is valuation
        assert is_rank_valuation(simple_branch)

    def test_parallel(self, parallel_ss):
        assert is_rank_valuation(parallel_ss)


# ---------------------------------------------------------------------------
# FKG inequality
# ---------------------------------------------------------------------------

class TestFKG:
    def test_uniform_measure(self, simple_branch):
        """FKG should hold with uniform measure and constant functions."""
        mu = {s: 1.0 for s in simple_branch.states}
        f = {s: 1.0 for s in simple_branch.states}
        g = {s: 1.0 for s in simple_branch.states}
        r = check_fkg(simple_branch, f, g, mu)
        assert r.holds
        assert isinstance(r, FKGResult)

    def test_rank_and_height(self, deep_chain):
        """FKG with rank and height functions — both monotone."""
        rk = rank_function(deep_chain)
        ht = height_function(deep_chain)
        mu = {s: 1.0 for s in deep_chain.states}
        f = {s: float(rk[s]) for s in deep_chain.states}
        g = {s: float(ht[s]) for s in deep_chain.states}
        r = check_fkg(deep_chain, f, g, mu)
        assert r.holds

    def test_fkg_on_parallel(self, parallel_ss):
        """FKG on product lattice with rank functions."""
        rk = rank_function(parallel_ss)
        mu = {s: 1.0 for s in parallel_ss.states}
        f = {s: float(rk[s]) for s in parallel_ss.states}
        g = {s: float(rk[s]) for s in parallel_ss.states}
        r = check_fkg(parallel_ss, f, g, mu)
        assert r.holds
        assert r.correlation >= -1e-10

    def test_result_structure(self, simple_branch):
        mu = {s: 1.0 for s in simple_branch.states}
        f = {s: 1.0 for s in simple_branch.states}
        g = {s: 1.0 for s in simple_branch.states}
        r = check_fkg(simple_branch, f, g, mu)
        assert isinstance(r.e_f, float)
        assert isinstance(r.e_g, float)
        assert isinstance(r.e_fg, float)
        assert isinstance(r.correlation, float)


# ---------------------------------------------------------------------------
# Log-supermodularity
# ---------------------------------------------------------------------------

class TestLogSupermodular:
    def test_uniform_is_log_super(self, simple_branch):
        mu = {s: 1.0 for s in simple_branch.states}
        assert is_log_supermodular(simple_branch, mu)

    def test_rank_based_measure(self, parallel_ss):
        """Exponential of rank is log-supermodular on distributive lattices."""
        rk = rank_function(parallel_ss)
        mu = {s: 2.0 ** rk[s] for s in parallel_ss.states}
        assert is_log_supermodular(parallel_ss, mu)


# ---------------------------------------------------------------------------
# Monotone correlation
# ---------------------------------------------------------------------------

class TestMonotoneCorrelation:
    def test_end(self, end_ss):
        c = monotone_correlation(end_ss)
        assert isinstance(c, float)

    def test_chain_perfect(self, deep_chain):
        """Chain is graded: rank == height, correlation = 1."""
        c = monotone_correlation(deep_chain)
        assert abs(c - 1.0) < 1e-10

    def test_simple_branch(self, simple_branch):
        c = monotone_correlation(simple_branch)
        assert isinstance(c, float)
        assert c >= -1.0 - 1e-10
        assert c <= 1.0 + 1e-10


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

class TestAnalyzeValuations:
    def test_end(self, end_ss):
        a = analyze_valuations(end_ss)
        assert isinstance(a, ValuationAnalysis)
        assert a.num_states == 1

    def test_simple_branch(self, simple_branch):
        a = analyze_valuations(simple_branch)
        assert a.rank_is_valuation
        assert isinstance(a.is_graded, bool)
        assert a.num_states >= 2

    def test_deep_chain(self, deep_chain):
        a = analyze_valuations(deep_chain)
        assert a.rank_is_valuation
        assert a.is_graded  # Chain is always graded
        assert a.rank_defect < 1e-10

    def test_parallel(self, parallel_ss):
        a = analyze_valuations(parallel_ss)
        assert a.rank_is_valuation
        assert a.is_modular  # Product lattice is distributive

    def test_iterator(self, iterator_ss):
        a = analyze_valuations(iterator_ss)
        assert isinstance(a, ValuationAnalysis)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

class TestBenchmarks:
    @pytest.mark.parametrize("type_string,name", [
        ("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}", "Iterator"),
        ("&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}", "File"),
        ("&{a: end, b: end}", "SimpleBranch"),
        ("(&{a: end} || &{b: end})", "SimpleParallel"),
        ("&{a: &{b: &{c: end}}}", "DeepChain"),
        ("+{ok: end, err: end}", "SimpleSelect"),
        ("&{get: +{OK: end, NOT_FOUND: end}, put: +{OK: end, ERR: end}}", "REST"),
        ("rec X . &{request: +{OK: X, ERR: end}}", "RetryLoop"),
    ])
    def test_analysis_on_benchmarks(self, type_string, name):
        ss = _build(type_string)
        a = analyze_valuations(ss)
        assert isinstance(a, ValuationAnalysis), f"{name}"
        assert a.num_states >= 1

    @pytest.mark.parametrize("type_string,name", [
        ("&{a: end, b: end}", "SimpleBranch"),
        ("(&{a: end} || &{b: end})", "SimpleParallel"),
        ("&{a: &{b: &{c: end}}}", "DeepChain"),
        ("+{ok: end, err: end}", "SimpleSelect"),
    ])
    def test_fkg_on_benchmarks(self, type_string, name):
        ss = _build(type_string)
        rk = rank_function(ss)
        mu = {s: 1.0 for s in ss.states}
        f = {s: float(rk[s]) for s in ss.states}
        g = {s: float(rk[s]) for s in ss.states}
        r = check_fkg(ss, f, g, mu)
        assert r.holds, f"{name} FKG should hold"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_state(self):
        ss = _build("end")
        assert is_rank_valuation(ss)
        a = analyze_valuations(ss)
        assert a.rank_defect < 1e-10

    def test_pure_recursion(self):
        ss = _build("rec X . &{a: X}")
        r = rank_function(ss)
        assert isinstance(r, dict)

    def test_empty_measure(self, simple_branch):
        mu = {s: 0.0 for s in simple_branch.states}
        r = check_fkg(simple_branch, mu, mu, mu)
        assert r.holds  # Trivially true with zero measure

    def test_zero_valuation(self, simple_branch):
        v = {s: 0.0 for s in simple_branch.states}
        r = check_valuation(simple_branch, v)
        assert r.is_valuation  # Zero is always a valuation
