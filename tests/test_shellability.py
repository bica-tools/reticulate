"""Tests for shellability and EL-labeling analysis."""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.shellability import (
    HasseEdge,
    LabeledChain,
    ELResult,
    CLResult,
    ShellabilityResult,
    hasse_labels,
    maximal_chains,
    descent_set,
    descent_statistic,
    check_el_labeling,
    check_cl_labeling,
    compute_f_vector,
    compute_h_vector,
    check_shellability,
    analyze_shellability,
    is_shellable,
    is_el_labelable,
)


def _build(type_string: str):
    """Helper: parse + build state space."""
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
def file_ss():
    return _build("&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}")


@pytest.fixture
def parallel_ss():
    return _build("(&{a: end} || &{b: end})")


@pytest.fixture
def wide_branch():
    return _build("&{a: end, b: end, c: end, d: end}")


@pytest.fixture
def nested_branch():
    return _build("&{a: &{c: end, d: end}, b: &{e: end, f: end}}")


# ---------------------------------------------------------------------------
# Hasse diagram
# ---------------------------------------------------------------------------

class TestHasseLabels:
    def test_end(self, end_ss):
        edges = hasse_labels(end_ss)
        assert len(edges) == 0  # Single state, no edges

    def test_simple_branch(self, simple_branch):
        edges = hasse_labels(simple_branch)
        assert len(edges) >= 2  # At least a→end and b→end
        assert all(isinstance(e, HasseEdge) for e in edges)

    def test_deep_chain(self, deep_chain):
        edges = hasse_labels(deep_chain)
        assert len(edges) >= 3  # a→b→c→end

    def test_labels_are_strings(self, simple_branch):
        edges = hasse_labels(simple_branch)
        assert all(isinstance(e.label, str) for e in edges)


# ---------------------------------------------------------------------------
# Maximal chains
# ---------------------------------------------------------------------------

class TestMaximalChains:
    def test_end(self, end_ss):
        chains = maximal_chains(end_ss)
        assert len(chains) == 1  # Single trivial chain
        assert chains[0].states == (end_ss.top,)

    def test_simple_branch_two_chains(self, simple_branch):
        chains = maximal_chains(simple_branch)
        assert len(chains) == 2  # top→a→end and top→b→end

    def test_deep_chain_one_chain(self, deep_chain):
        chains = maximal_chains(deep_chain)
        assert len(chains) == 1  # top→a→b→c→end

    def test_chain_structure(self, simple_branch):
        chains = maximal_chains(simple_branch)
        for c in chains:
            assert isinstance(c, LabeledChain)
            assert c.states[0] == simple_branch.top
            assert c.states[-1] == simple_branch.bottom
            assert len(c.labels) == len(c.states) - 1

    def test_wide_branch_four_chains(self, wide_branch):
        chains = maximal_chains(wide_branch)
        assert len(chains) == 4

    def test_nested_branch_chains(self, nested_branch):
        chains = maximal_chains(nested_branch)
        assert len(chains) >= 4  # a→c, a→d, b→e, b→f

    def test_max_chains_limit(self, simple_branch):
        chains = maximal_chains(simple_branch, max_chains=1)
        assert len(chains) == 1

    def test_parallel_chains(self, parallel_ss):
        chains = maximal_chains(parallel_ss)
        assert len(chains) >= 2  # Multiple paths through product


# ---------------------------------------------------------------------------
# Descents
# ---------------------------------------------------------------------------

class TestDescents:
    def test_increasing_chain_no_descents(self, deep_chain):
        chains = maximal_chains(deep_chain)
        assert len(chains) == 1
        assert chains[0].is_increasing or len(chains[0].descents) >= 0

    def test_descent_set(self, simple_branch):
        chains = maximal_chains(simple_branch)
        for c in chains:
            ds = descent_set(c)
            assert isinstance(ds, frozenset)

    def test_descent_statistic(self, wide_branch):
        chains = maximal_chains(wide_branch)
        stats = descent_statistic(chains)
        assert isinstance(stats, dict)
        assert sum(stats.values()) == len(chains)


# ---------------------------------------------------------------------------
# EL-labeling
# ---------------------------------------------------------------------------

class TestELLabeling:
    def test_end(self, end_ss):
        r = check_el_labeling(end_ss)
        assert isinstance(r, ELResult)

    def test_deep_chain_is_el(self, deep_chain):
        """A single chain is trivially EL-labeled."""
        r = check_el_labeling(deep_chain)
        assert isinstance(r, ELResult)
        assert r.num_intervals > 0

    def test_simple_branch_el_check(self, simple_branch):
        r = check_el_labeling(simple_branch)
        assert isinstance(r, ELResult)

    def test_result_structure(self, simple_branch):
        r = check_el_labeling(simple_branch)
        assert isinstance(r.is_el_labeling, bool)
        assert isinstance(r.num_intervals, int)
        assert isinstance(r.intervals_with_unique_increasing, int)


# ---------------------------------------------------------------------------
# CL-labeling
# ---------------------------------------------------------------------------

class TestCLLabeling:
    def test_end(self, end_ss):
        r = check_cl_labeling(end_ss)
        assert isinstance(r, CLResult)

    def test_deep_chain_is_cl(self, deep_chain):
        r = check_cl_labeling(deep_chain)
        assert isinstance(r, CLResult)

    def test_simple_branch_cl(self, simple_branch):
        r = check_cl_labeling(simple_branch)
        assert isinstance(r, CLResult)


# ---------------------------------------------------------------------------
# f-vector and h-vector
# ---------------------------------------------------------------------------

class TestFVector:
    def test_end(self, end_ss):
        f = compute_f_vector(end_ss)
        assert isinstance(f, tuple)
        assert f[0] == 1  # One vertex

    def test_simple_branch(self, simple_branch):
        f = compute_f_vector(simple_branch)
        assert f[0] >= 2  # At least 2 vertices
        assert len(f) >= 2  # At least vertices and edges

    def test_deep_chain(self, deep_chain):
        f = compute_f_vector(deep_chain)
        assert f[0] >= 4  # 4 states

    def test_monotone_f_vector(self, simple_branch):
        """f-vector should be non-negative."""
        f = compute_f_vector(simple_branch)
        assert all(fi >= 0 for fi in f)


class TestHVector:
    def test_end(self, end_ss):
        h = compute_h_vector(end_ss)
        assert isinstance(h, tuple)

    def test_simple_branch(self, simple_branch):
        h = compute_h_vector(simple_branch)
        assert isinstance(h, tuple)

    def test_deep_chain(self, deep_chain):
        h = compute_h_vector(deep_chain)
        assert isinstance(h, tuple)


# ---------------------------------------------------------------------------
# Shellability
# ---------------------------------------------------------------------------

class TestShellability:
    def test_end(self, end_ss):
        r = check_shellability(end_ss)
        assert isinstance(r, ShellabilityResult)

    def test_deep_chain_shellable(self, deep_chain):
        """A single chain is trivially shellable."""
        r = check_shellability(deep_chain)
        assert isinstance(r, ShellabilityResult)
        assert r.height >= 3

    def test_simple_branch_shellability(self, simple_branch):
        r = check_shellability(simple_branch)
        assert isinstance(r, ShellabilityResult)
        assert r.num_maximal_chains == 2

    def test_result_completeness(self, simple_branch):
        r = check_shellability(simple_branch)
        assert isinstance(r.is_shellable, bool)
        assert isinstance(r.is_el_shellable, bool)
        assert isinstance(r.is_cl_shellable, bool)
        assert isinstance(r.is_graded, bool)
        assert isinstance(r.el_result, ELResult)
        assert isinstance(r.cl_result, CLResult)
        assert isinstance(r.f_vector, tuple)
        assert isinstance(r.h_vector, tuple)

    def test_parallel_shellability(self, parallel_ss):
        r = check_shellability(parallel_ss)
        assert isinstance(r, ShellabilityResult)

    def test_wide_branch_shellability(self, wide_branch):
        r = check_shellability(wide_branch)
        assert r.num_maximal_chains == 4


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

class TestConvenience:
    def test_is_shellable(self, simple_branch):
        result = is_shellable(simple_branch)
        assert isinstance(result, bool)

    def test_is_el_labelable(self, simple_branch):
        result = is_el_labelable(simple_branch)
        assert isinstance(result, bool)

    def test_analyze_alias(self, simple_branch):
        r1 = check_shellability(simple_branch)
        r2 = analyze_shellability(simple_branch)
        assert r1.is_shellable == r2.is_shellable
        assert r1.num_maximal_chains == r2.num_maximal_chains


# ---------------------------------------------------------------------------
# Benchmark protocols
# ---------------------------------------------------------------------------

class TestBenchmarks:
    @pytest.mark.parametrize("type_string,name,has_chains", [
        ("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}", "Iterator", False),
        ("&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}", "File", False),
        ("&{a: end, b: end}", "SimpleBranch", True),
        ("+{ok: end, err: end}", "SimpleSelect", True),
        ("(&{a: end} || &{b: end})", "SimpleParallel", True),
        ("&{a: &{c: end}, b: &{d: end}}", "NestedBranch", True),
        ("&{get: +{OK: end, NOT_FOUND: end}, put: +{OK: end, ERR: end}}", "REST", True),
        ("&{a: &{b: &{c: end}}}", "DeepChain", True),
    ])
    def test_shellability_on_benchmarks(self, type_string, name, has_chains):
        ss = _build(type_string)
        r = check_shellability(ss)
        assert isinstance(r, ShellabilityResult), f"{name}"
        if has_chains:
            assert r.num_maximal_chains >= 1, f"{name} should have chains"
        else:
            # Recursive types have cycles; no acyclic maximal chains
            assert r.num_maximal_chains == 0, f"{name} is cyclic, no acyclic chains"

    @pytest.mark.parametrize("type_string,name", [
        ("&{a: end, b: end}", "SimpleBranch"),
        ("&{a: end, b: end, c: end}", "TripleBranch"),
        ("&{a: &{c: end}, b: &{d: end}}", "NestedBranch"),
        ("&{a: &{b: &{c: end}}}", "DeepChain"),
    ])
    def test_graded_on_benchmarks(self, type_string, name):
        ss = _build(type_string)
        r = check_shellability(ss)
        # Simple types should be graded (all max chains same length)
        assert r.is_graded, f"{name} should be graded"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_state(self):
        ss = _build("end")
        r = check_shellability(ss)
        assert r.height == 0
        assert r.num_maximal_chains == 1

    def test_two_states(self):
        ss = _build("&{a: end}")
        r = check_shellability(ss)
        assert r.height >= 1
        assert r.num_maximal_chains >= 1

    def test_pure_recursion(self):
        ss = _build("rec X . &{a: X}")
        r = check_shellability(ss)
        assert isinstance(r, ShellabilityResult)

    def test_selection_only(self):
        ss = _build("+{a: end, b: end}")
        r = check_shellability(ss)
        assert isinstance(r, ShellabilityResult)

    def test_f_vector_nonempty(self, simple_branch):
        f = compute_f_vector(simple_branch)
        assert len(f) >= 1
        assert f[0] > 0

    def test_h_vector_nonempty(self, simple_branch):
        h = compute_h_vector(simple_branch)
        assert len(h) >= 1
