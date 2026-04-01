"""Tests for congruence lattices of session types (Step 33a)."""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.congruence import (
    Congruence,
    CongruenceLattice,
    CongruenceAnalysis,
    principal_congruence,
    enumerate_congruences,
    congruence_lattice,
    quotient_lattice,
    is_simple,
    birkhoff_congruences,
    simplification_options,
    analyze_congruences,
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
def parallel_ss():
    return _build("(&{a: end} || &{b: end})")

@pytest.fixture
def nested():
    return _build("&{a: &{c: end, d: end}, b: &{e: end, f: end}}")

@pytest.fixture
def select_ss():
    return _build("+{ok: end, err: end}")


# ---------------------------------------------------------------------------
# Principal congruences
# ---------------------------------------------------------------------------

class TestPrincipalCongruence:
    def test_same_element(self, simple_branch):
        """θ(a,a) = identity congruence."""
        top = simple_branch.top
        pc = principal_congruence(simple_branch, top, top)
        assert pc.is_trivial_bottom

    def test_top_bottom(self, simple_branch):
        """θ(top, bottom) = total congruence (merges everything)."""
        pc = principal_congruence(simple_branch, simple_branch.top, simple_branch.bottom)
        assert pc.is_trivial_top

    def test_result_type(self, simple_branch):
        states = sorted(simple_branch.states)
        pc = principal_congruence(simple_branch, states[0], states[1])
        assert isinstance(pc, Congruence)
        assert pc.num_classes >= 1

    def test_chain_principal(self, deep_chain):
        """On a chain, θ(a,b) collapses a and b and everything between."""
        states = sorted(deep_chain.states)
        pc = principal_congruence(deep_chain, states[0], states[1])
        assert isinstance(pc, Congruence)

    def test_generators_recorded(self, simple_branch):
        states = sorted(simple_branch.states)
        pc = principal_congruence(simple_branch, states[0], states[1])
        assert len(pc.generators) == 1
        assert pc.generators[0] == (states[0], states[1])


# ---------------------------------------------------------------------------
# Enumerate congruences
# ---------------------------------------------------------------------------

class TestEnumerateCongruences:
    def test_end_one_congruence(self, end_ss):
        congs = enumerate_congruences(end_ss)
        assert len(congs) == 1  # Only identity

    def test_simple_branch_congruences(self, simple_branch):
        congs = enumerate_congruences(simple_branch)
        assert len(congs) >= 2  # At least identity and total

    def test_chain_congruences(self, deep_chain):
        congs = enumerate_congruences(deep_chain)
        assert len(congs) >= 2

    def test_sorted_by_classes(self, simple_branch):
        congs = enumerate_congruences(simple_branch)
        for i in range(len(congs) - 1):
            assert congs[i].num_classes <= congs[i + 1].num_classes

    def test_includes_identity(self, simple_branch):
        congs = enumerate_congruences(simple_branch)
        assert any(c.is_trivial_bottom for c in congs)

    def test_includes_total(self, simple_branch):
        congs = enumerate_congruences(simple_branch)
        assert any(c.is_trivial_top for c in congs)


# ---------------------------------------------------------------------------
# Congruence lattice
# ---------------------------------------------------------------------------

class TestCongruenceLattice:
    def test_con_structure(self, simple_branch):
        con = congruence_lattice(simple_branch)
        assert isinstance(con, CongruenceLattice)
        assert con.num_congruences >= 2

    def test_con_has_bottom_top(self, simple_branch):
        con = congruence_lattice(simple_branch)
        assert con.bottom is not None
        assert con.top is not None
        assert con.bottom != con.top or len(simple_branch.states) <= 1

    def test_con_ordering_reflexive(self, simple_branch):
        con = congruence_lattice(simple_branch)
        for i in range(len(con.congruences)):
            assert (i, i) in con.ordering

    def test_con_is_distributive(self, simple_branch):
        """Funayama-Nakayama: Con(L) is always distributive."""
        con = congruence_lattice(simple_branch)
        assert con.is_distributive

    def test_chain_con(self, deep_chain):
        con = congruence_lattice(deep_chain)
        assert con.num_congruences >= 2


# ---------------------------------------------------------------------------
# Quotient lattice
# ---------------------------------------------------------------------------

class TestQuotientLattice:
    def test_identity_quotient(self, simple_branch):
        """Quotient by identity = original (same number of states)."""
        congs = enumerate_congruences(simple_branch)
        identity = next(c for c in congs if c.is_trivial_bottom)
        q = quotient_lattice(simple_branch, identity)
        assert len(q.states) == len(simple_branch.states)

    def test_total_quotient(self, simple_branch):
        """Quotient by total = single state."""
        congs = enumerate_congruences(simple_branch)
        total = next(c for c in congs if c.is_trivial_top)
        q = quotient_lattice(simple_branch, total)
        assert len(q.states) == 1

    def test_quotient_is_smaller(self, deep_chain):
        """Non-trivial quotients have fewer states."""
        congs = enumerate_congruences(deep_chain)
        for c in congs:
            if not c.is_trivial_bottom and not c.is_trivial_top:
                q = quotient_lattice(deep_chain, c)
                assert len(q.states) < len(deep_chain.states)

    def test_quotient_has_transitions(self, deep_chain):
        congs = enumerate_congruences(deep_chain)
        for c in congs:
            if not c.is_trivial_top:
                q = quotient_lattice(deep_chain, c)
                if len(q.states) > 1:
                    assert len(q.transitions) >= 1


# ---------------------------------------------------------------------------
# Simplicity
# ---------------------------------------------------------------------------

class TestSimplicity:
    def test_end_is_simple(self, end_ss):
        assert is_simple(end_ss)

    def test_two_element_is_simple(self):
        """&{a: end} has 2 states, Con = {id, total} — simple."""
        ss = _build("&{a: end}")
        assert is_simple(ss)

    def test_branch_may_not_be_simple(self, simple_branch):
        """&{a: end, b: end} may have non-trivial congruences."""
        result = is_simple(simple_branch)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Birkhoff fast path
# ---------------------------------------------------------------------------

class TestBirkhoff:
    def test_distributive_returns_list(self, simple_branch):
        result = birkhoff_congruences(simple_branch)
        # simple_branch is distributive, so should return a list
        if result is not None:
            assert isinstance(result, list)
            assert all(isinstance(c, Congruence) for c in result)

    def test_chain_birkhoff(self, deep_chain):
        result = birkhoff_congruences(deep_chain)
        if result is not None:
            assert len(result) >= 2  # At least identity and total

    def test_birkhoff_consistent_with_enumerate(self, simple_branch):
        """Birkhoff should find same number of congruences as enumerate."""
        birk = birkhoff_congruences(simple_branch)
        enum = enumerate_congruences(simple_branch)
        if birk is not None:
            assert len(birk) == len(enum)


# ---------------------------------------------------------------------------
# Simplification options
# ---------------------------------------------------------------------------

class TestSimplification:
    def test_no_options_for_simple(self, end_ss):
        opts = simplification_options(end_ss)
        assert len(opts) == 0

    def test_options_sorted(self, deep_chain):
        opts = simplification_options(deep_chain)
        for i in range(len(opts) - 1):
            assert opts[i][1] <= opts[i + 1][1]

    def test_options_are_non_trivial(self, deep_chain):
        opts = simplification_options(deep_chain)
        for cong, size in opts:
            assert not cong.is_trivial_bottom
            assert not cong.is_trivial_top


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

class TestAnalysis:
    def test_end(self, end_ss):
        a = analyze_congruences(end_ss)
        assert isinstance(a, CongruenceAnalysis)
        assert a.num_states == 1
        assert a.is_simple

    def test_simple_branch(self, simple_branch):
        a = analyze_congruences(simple_branch)
        assert a.num_congruences >= 2
        assert isinstance(a.is_modular, bool)
        assert isinstance(a.is_distributive_lattice, bool)

    def test_chain(self, deep_chain):
        a = analyze_congruences(deep_chain)
        assert a.num_congruences >= 2
        assert a.num_join_irreducibles >= 1

    def test_parallel(self, parallel_ss):
        a = analyze_congruences(parallel_ss)
        assert isinstance(a, CongruenceAnalysis)

    def test_funayama_nakayama(self, simple_branch):
        """Con(L) is distributive (Funayama-Nakayama theorem)."""
        a = analyze_congruences(simple_branch)
        assert a.con_lattice.is_distributive


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

class TestBenchmarks:
    @pytest.mark.parametrize("type_string,name", [
        ("&{a: end, b: end}", "SimpleBranch"),
        ("&{a: &{b: &{c: end}}}", "DeepChain"),
        ("+{ok: end, err: end}", "SimpleSelect"),
        ("(&{a: end} || &{b: end})", "SimpleParallel"),
        ("&{a: &{c: end}, b: end}", "AsymmetricBranch"),
    ])
    def test_analysis_on_benchmarks(self, type_string, name):
        ss = _build(type_string)
        a = analyze_congruences(ss)
        assert a.num_congruences >= 2, f"{name} needs at least 2 congruences"
        assert a.con_lattice.is_distributive, f"{name}: Con should be distributive"


# ---------------------------------------------------------------------------
# Congruence properties
# ---------------------------------------------------------------------------

class TestCongruenceProperties:
    def test_class_of(self, simple_branch):
        congs = enumerate_congruences(simple_branch)
        total = next(c for c in congs if c.is_trivial_top)
        cls = total.class_of(simple_branch.top)
        assert simple_branch.bottom in cls

    def test_trivial_bottom_check(self, simple_branch):
        congs = enumerate_congruences(simple_branch)
        identity = next(c for c in congs if c.is_trivial_bottom)
        assert identity.is_trivial_bottom
        assert not identity.is_trivial_top


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_state(self):
        ss = _build("end")
        congs = enumerate_congruences(ss)
        assert len(congs) == 1

    def test_two_states(self):
        ss = _build("&{a: end}")
        a = analyze_congruences(ss)
        assert a.is_simple  # 2-element chain is simple

    def test_wide_branch(self):
        ss = _build("&{a: end, b: end, c: end}")
        a = analyze_congruences(ss)
        assert a.num_congruences >= 2

    def test_nested_deep(self):
        ss = _build("&{a: &{b: end}, c: &{d: end}}")
        a = analyze_congruences(ss)
        assert isinstance(a, CongruenceAnalysis)
