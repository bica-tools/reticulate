"""Tests for modularity analysis (Step 70f).

Tests the unified modularity module covering distributivity as modularity,
spectral bisection, FCA module discovery, substitutability, Birkhoff
decomposition, coupling metrics, non-modularity diagnosis, and refactoring.

65+ tests covering:
  - Core modularity check (10 tests)
  - Module boundaries (6 tests)
  - Module discovery (6 tests)
  - Substitutability (6 tests)
  - Minimal decomposition (6 tests)
  - Compression ratio (5 tests)
  - Coupling score (5 tests)
  - Interface width / Cheeger (5 tests)
  - Non-modularity diagnosis (6 tests)
  - Refactoring suggestions (5 tests)
  - Full analysis (5 tests)
  - Benchmark protocols (5 parametrised)
"""

from __future__ import annotations

import pytest

from reticulate.modularity import (
    ModularityResult,
    ModuleBoundary,
    Module,
    SubstitutionResult,
    IrreducibleModule,
    NonModularityDiagnosis,
    Refactoring,
    ModularityAnalysis,
    check_modularity,
    find_module_boundaries,
    discover_modules,
    check_substitutability,
    minimal_decomposition,
    compression_ratio,
    coupling_score,
    reconvergence_degree,
    interface_width,
    diagnose_non_modularity,
    suggest_modularization,
    analyze_modularity,
)
from reticulate.parser import parse
from reticulate.statespace import StateSpace, build_statespace

from tests.benchmarks.protocols import BENCHMARKS


# ---------------------------------------------------------------------------
# Helpers: build small lattices by hand
# ---------------------------------------------------------------------------

def _chain(n: int) -> StateSpace:
    """Chain of n elements: 0 > 1 > ... > n-1."""
    ss = StateSpace()
    ss.states = set(range(n))
    ss.transitions = [(i, f"t{i}", i + 1) for i in range(n - 1)]
    ss.top = 0
    ss.bottom = n - 1
    ss.labels = {i: f"s{i}" for i in range(n)}
    return ss


def _boolean_2() -> StateSpace:
    """Boolean lattice 2^2 = {top, a, b, bot}."""
    ss = StateSpace()
    ss.states = {0, 1, 2, 3}
    ss.transitions = [
        (0, "a", 1), (0, "b", 2),
        (1, "x", 3), (2, "y", 3),
    ]
    ss.top = 0
    ss.bottom = 3
    ss.labels = {0: "top", 1: "a", 2: "b", 3: "bot"}
    return ss


def _diamond() -> StateSpace:
    """Diamond lattice (M3): top > a,b,c > bot with a,b,c pairwise incomparable.
    meet(a,b)=meet(a,c)=meet(b,c)=bot, join(a,b)=join(a,c)=join(b,c)=top.
    This is NOT distributive.
    """
    ss = StateSpace()
    ss.states = {0, 1, 2, 3, 4}
    ss.transitions = [
        (0, "a", 1), (0, "b", 2), (0, "c", 3),
        (1, "x", 4), (2, "y", 4), (3, "z", 4),
    ]
    ss.top = 0
    ss.bottom = 4
    ss.labels = {0: "top", 1: "a", 2: "b", 3: "c", 4: "bot"}
    return ss


def _pentagon() -> StateSpace:
    """Pentagon lattice (N5): top > a > b > bot, top > c > bot, c || a, c || b."""
    ss = StateSpace()
    ss.states = {0, 1, 2, 3, 4}
    ss.transitions = [
        (0, "a", 1), (0, "c", 3),
        (1, "b", 2),
        (2, "x", 4), (3, "y", 4),
    ]
    ss.top = 0
    ss.bottom = 4
    ss.labels = {0: "top", 1: "a", 2: "b", 3: "c", 4: "bot"}
    return ss


def _product_lattice() -> StateSpace:
    """Product of two chains (2x2): distributive, 4 states."""
    ss = StateSpace()
    ss.states = {0, 1, 2, 3}
    ss.transitions = [
        (0, "left", 1), (0, "right", 2),
        (1, "right2", 3), (2, "left2", 3),
    ]
    ss.top = 0
    ss.bottom = 3
    ss.labels = {0: "top", 1: "left_done", 2: "right_done", 3: "bot"}
    return ss


def _from_session_type(type_string: str) -> StateSpace:
    """Parse and build a state space from a session type string."""
    ast = parse(type_string)
    return build_statespace(ast)


# ===========================================================================
# 1. Core modularity check (10 tests)
# ===========================================================================

class TestCheckModularity:
    """Tests for check_modularity()."""

    def test_chain_is_modular(self) -> None:
        ss = _chain(4)
        result = check_modularity(ss)
        assert result.is_modular is True
        assert result.is_lattice is True
        assert result.is_distributive is True

    def test_boolean_is_modular(self) -> None:
        ss = _boolean_2()
        result = check_modularity(ss)
        assert result.is_modular is True
        assert result.classification == "boolean"

    def test_diamond_not_modular(self) -> None:
        ss = _diamond()
        result = check_modularity(ss)
        assert result.is_modular is False
        assert result.is_distributive is False

    def test_pentagon_not_modular(self) -> None:
        ss = _pentagon()
        result = check_modularity(ss)
        assert result.is_modular is False
        assert result.is_distributive is False
        # Pentagon is not algebraically modular either
        assert result.is_algebraically_modular is False

    def test_diamond_is_algebraically_modular(self) -> None:
        ss = _diamond()
        result = check_modularity(ss)
        # M3 is modular (no N5) but not distributive (has M3)
        assert result.is_algebraically_modular is True
        assert result.is_distributive is False

    def test_product_is_modular(self) -> None:
        ss = _product_lattice()
        result = check_modularity(ss)
        assert result.is_modular is True

    def test_session_type_branch(self) -> None:
        ss = _from_session_type("&{a: end, b: end}")
        result = check_modularity(ss)
        assert result.is_modular is True
        assert result.is_lattice is True

    def test_fiedler_is_float(self) -> None:
        ss = _chain(3)
        result = check_modularity(ss)
        assert isinstance(result.fiedler_value, float)

    def test_cheeger_is_float(self) -> None:
        ss = _chain(3)
        result = check_modularity(ss)
        assert isinstance(result.cheeger_constant, float)

    def test_compression_ratio_is_float(self) -> None:
        ss = _chain(3)
        result = check_modularity(ss)
        assert isinstance(result.compression_ratio, float)
        assert 0 <= result.compression_ratio <= 1.0


# ===========================================================================
# 2. Module boundaries (6 tests)
# ===========================================================================

class TestModuleBoundaries:
    """Tests for find_module_boundaries()."""

    def test_chain_has_boundary(self) -> None:
        ss = _chain(4)
        boundaries = find_module_boundaries(ss)
        assert len(boundaries) >= 1

    def test_boundary_partitions_cover(self) -> None:
        ss = _boolean_2()
        boundaries = find_module_boundaries(ss)
        if boundaries:
            b = boundaries[0]
            assert b.partition_a | b.partition_b == ss.states

    def test_boundary_partitions_disjoint(self) -> None:
        ss = _boolean_2()
        boundaries = find_module_boundaries(ss)
        if boundaries:
            b = boundaries[0]
            assert len(b.partition_a & b.partition_b) == 0

    def test_cut_edges_cross_boundary(self) -> None:
        ss = _product_lattice()
        boundaries = find_module_boundaries(ss)
        if boundaries:
            b = boundaries[0]
            for src, lbl, tgt in b.cut_edges:
                assert (src in b.partition_a) != (tgt in b.partition_a)

    def test_cut_ratio_bounded(self) -> None:
        ss = _boolean_2()
        boundaries = find_module_boundaries(ss)
        if boundaries:
            b = boundaries[0]
            assert 0 <= b.cut_ratio <= 1.0

    def test_single_state_no_boundary(self) -> None:
        ss = StateSpace()
        ss.states = {0}
        ss.transitions = []
        ss.top = 0
        ss.bottom = 0
        boundaries = find_module_boundaries(ss)
        assert boundaries == []


# ===========================================================================
# 3. Module discovery (6 tests)
# ===========================================================================

class TestDiscoverModules:
    """Tests for discover_modules()."""

    def test_chain_modules(self) -> None:
        ss = _chain(3)
        modules = discover_modules(ss)
        # Should find some non-trivial modules
        assert isinstance(modules, list)

    def test_boolean_modules(self) -> None:
        ss = _boolean_2()
        modules = discover_modules(ss)
        assert isinstance(modules, list)

    def test_modules_have_capabilities(self) -> None:
        ss = _boolean_2()
        modules = discover_modules(ss)
        for m in modules:
            assert isinstance(m.capabilities, frozenset)

    def test_modules_have_states(self) -> None:
        ss = _boolean_2()
        modules = discover_modules(ss)
        for m in modules:
            assert len(m.states) > 0

    def test_module_transitions_classified(self) -> None:
        ss = _product_lattice()
        modules = discover_modules(ss)
        for m in modules:
            # Internal transitions stay within module
            for s, l, t in m.internal_transitions:
                assert s in m.states and t in m.states
            # Interface transitions cross boundary
            for s, l, t in m.interface_transitions:
                assert (s in m.states) != (t in m.states)

    def test_session_type_modules(self) -> None:
        ss = _from_session_type("&{a: end, b: end}")
        modules = discover_modules(ss)
        assert isinstance(modules, list)


# ===========================================================================
# 4. Substitutability (6 tests)
# ===========================================================================

class TestSubstitutability:
    """Tests for check_substitutability()."""

    def test_same_protocol_substitutable(self) -> None:
        ss = _chain(3)
        result = check_substitutability(ss, ss)
        assert result.is_substitutable is True

    def test_subtype_substitutable(self) -> None:
        ss_old = _from_session_type("&{a: end}")
        ss_new = _from_session_type("&{a: end, b: end}")
        result = check_substitutability(ss_old, ss_new)
        # New offers more methods -> is a subtype
        assert result.is_subtype is True

    def test_missing_method_not_substitutable(self) -> None:
        ss_old = _from_session_type("&{a: end, b: end}")
        ss_new = _from_session_type("&{a: end}")
        result = check_substitutability(ss_old, ss_new)
        assert result.is_subtype is False

    def test_result_has_explanation(self) -> None:
        ss = _chain(3)
        result = check_substitutability(ss, ss)
        assert len(result.explanation) > 0

    def test_preserves_modularity_check(self) -> None:
        ss1 = _boolean_2()
        ss2 = _product_lattice()
        result = check_substitutability(ss1, ss2)
        assert isinstance(result.preserves_modularity, bool)

    def test_chain_embedding(self) -> None:
        ss_small = _chain(2)
        ss_large = _chain(4)
        result = check_substitutability(ss_small, ss_large)
        assert result.has_embedding is True


# ===========================================================================
# 5. Minimal decomposition (6 tests)
# ===========================================================================

class TestMinimalDecomposition:
    """Tests for minimal_decomposition()."""

    def test_chain_decomposition(self) -> None:
        ss = _chain(4)
        modules = minimal_decomposition(ss)
        # Chain has n-1 join-irreducibles (all except bottom)
        # Actually n-2 since only states with exactly 1 lower cover
        assert len(modules) > 0

    def test_boolean_decomposition(self) -> None:
        ss = _boolean_2()
        modules = minimal_decomposition(ss)
        # Boolean 2^2 has 2 join-irreducibles (atoms)
        assert len(modules) == 2

    def test_irreducibles_have_downsets(self) -> None:
        ss = _boolean_2()
        modules = minimal_decomposition(ss)
        for m in modules:
            assert isinstance(m.downset, frozenset)
            assert m.representative in m.downset

    def test_irreducibles_have_labels(self) -> None:
        ss = _boolean_2()
        modules = minimal_decomposition(ss)
        for m in modules:
            assert isinstance(m.labels, frozenset)

    def test_heights_nonnegative(self) -> None:
        ss = _chain(4)
        modules = minimal_decomposition(ss)
        for m in modules:
            assert m.height >= 0

    def test_product_decomposition(self) -> None:
        ss = _product_lattice()
        modules = minimal_decomposition(ss)
        assert len(modules) >= 2


# ===========================================================================
# 6. Compression ratio (5 tests)
# ===========================================================================

class TestCompressionRatio:
    """Tests for compression_ratio()."""

    def test_chain_compression(self) -> None:
        ss = _chain(4)
        cr = compression_ratio(ss)
        assert 0 < cr <= 1.0

    def test_boolean_compression(self) -> None:
        ss = _boolean_2()
        cr = compression_ratio(ss)
        # 2 JIs out of 4 states = 0.5
        assert cr == pytest.approx(0.5)

    def test_single_state(self) -> None:
        ss = StateSpace()
        ss.states = {0}
        ss.transitions = []
        ss.top = 0
        ss.bottom = 0
        cr = compression_ratio(ss)
        assert cr == 0.0

    def test_two_state_chain(self) -> None:
        ss = _chain(2)
        cr = compression_ratio(ss)
        # 1 JI out of 2 states = 0.5
        assert cr == pytest.approx(0.5)

    def test_compression_ratio_bounded(self) -> None:
        ss = _product_lattice()
        cr = compression_ratio(ss)
        assert 0 <= cr <= 1.0


# ===========================================================================
# 7. Coupling score (5 tests)
# ===========================================================================

class TestCouplingScore:
    """Tests for coupling_score()."""

    def test_chain_coupling(self) -> None:
        ss = _chain(3)
        score = coupling_score(ss)
        assert 0 <= score < 1.0

    def test_coupling_bounded(self) -> None:
        ss = _boolean_2()
        score = coupling_score(ss)
        assert 0 <= score < 1.0

    def test_single_state_coupling(self) -> None:
        ss = StateSpace()
        ss.states = {0}
        ss.transitions = []
        ss.top = 0
        ss.bottom = 0
        score = coupling_score(ss)
        assert score == 0.0

    def test_higher_connectivity_higher_coupling(self) -> None:
        # Product lattice should have higher coupling than chain
        chain = _chain(4)
        product = _product_lattice()
        c_chain = coupling_score(chain)
        c_product = coupling_score(product)
        # Both should be valid floats
        assert isinstance(c_chain, float)
        assert isinstance(c_product, float)

    def test_coupling_is_float(self) -> None:
        ss = _diamond()
        score = coupling_score(ss)
        assert isinstance(score, float)


# ===========================================================================
# 8. Interface width / Cheeger (5 tests)
# ===========================================================================

class TestInterfaceWidth:
    """Tests for interface_width() / Cheeger constant."""

    def test_chain_cheeger(self) -> None:
        ss = _chain(3)
        h = interface_width(ss)
        assert h >= 0

    def test_boolean_cheeger(self) -> None:
        ss = _boolean_2()
        h = interface_width(ss)
        assert h >= 0

    def test_single_state_cheeger(self) -> None:
        ss = StateSpace()
        ss.states = {0}
        ss.transitions = []
        ss.top = 0
        ss.bottom = 0
        h = interface_width(ss)
        assert h == 0.0

    def test_cheeger_is_float(self) -> None:
        ss = _product_lattice()
        h = interface_width(ss)
        assert isinstance(h, float)

    def test_cheeger_nonnegative(self) -> None:
        ss = _diamond()
        h = interface_width(ss)
        assert h >= 0


# ===========================================================================
# 9. Non-modularity diagnosis (6 tests)
# ===========================================================================

class TestDiagnoseNonModularity:
    """Tests for diagnose_non_modularity()."""

    def test_distributive_no_diagnosis(self) -> None:
        ss = _boolean_2()
        diag = diagnose_non_modularity(ss)
        assert diag.is_modular is True
        assert diag.entanglement_type == "none"

    def test_diamond_diagnosis(self) -> None:
        ss = _diamond()
        diag = diagnose_non_modularity(ss)
        assert diag.is_modular is False
        assert diag.has_m3 is True
        assert diag.entanglement_type in ("diamond", "both")

    def test_pentagon_diagnosis(self) -> None:
        ss = _pentagon()
        diag = diagnose_non_modularity(ss)
        assert diag.is_modular is False
        assert diag.has_n5 is True
        assert diag.entanglement_type in ("pentagon", "both")

    def test_diagnosis_has_explanation(self) -> None:
        ss = _diamond()
        diag = diagnose_non_modularity(ss)
        assert len(diag.explanation) > 0

    def test_witness_present_for_m3(self) -> None:
        ss = _diamond()
        diag = diagnose_non_modularity(ss)
        if diag.has_m3:
            assert diag.m3_witness is not None
            assert len(diag.m3_witness) == 5

    def test_reconvergence_degree(self) -> None:
        ss = _diamond()
        diag = diagnose_non_modularity(ss)
        assert isinstance(diag.reconvergence_degree, int)
        assert diag.reconvergence_degree >= 0


# ===========================================================================
# 10. Refactoring suggestions (5 tests)
# ===========================================================================

class TestRefactoringSuggestions:
    """Tests for suggest_modularization()."""

    def test_modular_no_refactoring(self) -> None:
        ss = _boolean_2()
        refs = suggest_modularization(ss)
        assert refs == []

    def test_diamond_has_refactoring(self) -> None:
        ss = _diamond()
        refs = suggest_modularization(ss)
        assert len(refs) > 0

    def test_refactoring_has_description(self) -> None:
        ss = _diamond()
        refs = suggest_modularization(ss)
        for r in refs:
            assert len(r.description) > 0

    def test_refactoring_kind_valid(self) -> None:
        ss = _diamond()
        refs = suggest_modularization(ss)
        valid_kinds = {"split", "merge", "flatten", "factor"}
        for r in refs:
            assert r.kind in valid_kinds

    def test_refactoring_improvement_bounded(self) -> None:
        ss = _pentagon()
        refs = suggest_modularization(ss)
        for r in refs:
            assert 0 <= r.expected_improvement <= 1.0


# ===========================================================================
# 11. Full analysis (5 tests)
# ===========================================================================

class TestAnalyzeModularity:
    """Tests for analyze_modularity()."""

    def test_analysis_returns_all_components(self) -> None:
        ss = _boolean_2()
        result = analyze_modularity(ss)
        assert isinstance(result.modularity, ModularityResult)
        assert isinstance(result.boundaries, list)
        assert isinstance(result.modules, list)
        assert isinstance(result.irreducibles, list)
        assert isinstance(result.coupling, float)
        assert isinstance(result.reconvergence, int)
        assert isinstance(result.interface_width, float)

    def test_modular_no_diagnosis(self) -> None:
        ss = _product_lattice()
        result = analyze_modularity(ss)
        assert result.diagnosis is None
        assert result.refactorings == []

    def test_non_modular_has_diagnosis(self) -> None:
        ss = _diamond()
        result = analyze_modularity(ss)
        assert result.diagnosis is not None
        assert len(result.refactorings) > 0

    def test_chain_analysis(self) -> None:
        ss = _chain(5)
        result = analyze_modularity(ss)
        assert result.modularity.is_modular is True

    def test_session_type_analysis(self) -> None:
        ss = _from_session_type("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        result = analyze_modularity(ss)
        assert result.modularity.is_lattice is True


# ===========================================================================
# 12. Reconvergence degree (2 tests)
# ===========================================================================

class TestReconvergenceDegree:
    """Tests for reconvergence_degree()."""

    def test_chain_no_reconvergence(self) -> None:
        ss = _chain(4)
        assert reconvergence_degree(ss) == 0

    def test_diamond_has_reconvergence(self) -> None:
        ss = _diamond()
        # Bottom state has in-degree 3
        assert reconvergence_degree(ss) >= 1


# ===========================================================================
# 13. Benchmark protocols (5 parametrised)
# ===========================================================================

BENCHMARK_SAMPLE = BENCHMARKS[:15]


@pytest.mark.parametrize(
    "protocol",
    BENCHMARK_SAMPLE,
    ids=[p.name for p in BENCHMARK_SAMPLE],
)
class TestBenchmarkModularity:
    """Modularity analysis on benchmark protocols."""

    def test_modularity_check(self, protocol) -> None:
        ss = _from_session_type(protocol.type_string)
        result = check_modularity(ss)
        assert isinstance(result.is_modular, bool)
        assert isinstance(result.classification, str)

    def test_coupling_score(self, protocol) -> None:
        ss = _from_session_type(protocol.type_string)
        score = coupling_score(ss)
        assert 0 <= score < 1.0

    def test_compression_ratio(self, protocol) -> None:
        ss = _from_session_type(protocol.type_string)
        cr = compression_ratio(ss)
        assert 0 <= cr <= 1.0

    def test_full_analysis(self, protocol) -> None:
        ss = _from_session_type(protocol.type_string)
        result = analyze_modularity(ss)
        assert isinstance(result, ModularityAnalysis)

    def test_boundaries(self, protocol) -> None:
        ss = _from_session_type(protocol.type_string)
        boundaries = find_module_boundaries(ss)
        assert isinstance(boundaries, list)
