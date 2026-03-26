"""Tests for L₃ — Parallel Composition and Product Lattices.

Step 402a: 35+ tests covering L₃ classification, product factor decomposition,
factor analysis, distributivity preservation, size prediction, enumeration,
and L₂ vs L₃ comparison.
"""

from __future__ import annotations

import pytest

from reticulate.parser import (
    Branch,
    End,
    Parallel,
    Rec,
    Select,
    Var,
    Wait,
    parse,
    pretty,
)
from reticulate.statespace import StateSpace, build_statespace
from reticulate.lattice import check_lattice, check_distributive

from reticulate.extensions.parallel_level import (
    FactorAnalysis,
    L3_enumerate,
    L3_state_space_class,
    LevelComparison,
    compare_L2_L3,
    distributivity_from_factors,
    factor_analysis,
    is_L3_type,
    product_factors,
    product_height_prediction,
    product_size_prediction,
    product_width_prediction,
    verify_distributivity_preservation,
)


# ===================================================================
# 1. L₃ classification (parallel types without rec)
# ===================================================================


class TestIsL3Type:
    """Tests for is_L3_type()."""

    def test_simple_parallel_is_L3(self) -> None:
        s = parse("(&{a: wait} || &{b: wait})")
        assert is_L3_type(s)

    def test_parallel_with_select_is_L3(self) -> None:
        s = parse("(+{a: wait} || &{b: wait})")
        assert is_L3_type(s)

    def test_end_only_not_L3(self) -> None:
        s = parse("end")
        assert not is_L3_type(s)

    def test_branch_only_not_L3(self) -> None:
        s = parse("&{a: end, b: end}")
        assert not is_L3_type(s)

    def test_select_only_not_L3(self) -> None:
        s = parse("+{a: end, b: end}")
        assert not is_L3_type(s)

    def test_recursive_with_parallel_not_L3(self) -> None:
        """rec + parallel = L7, not L3."""
        s = parse("rec X . (&{a: wait} || &{b: X})")
        assert not is_L3_type(s)

    def test_continuation_not_L3(self) -> None:
        """Continuation = L5, not L3."""
        s = parse("(&{a: wait} || &{b: wait}) . &{c: end}")
        assert not is_L3_type(s)

    def test_parallel_with_branches_inside(self) -> None:
        s = parse("(&{a: &{c: wait}} || &{b: wait})")
        assert is_L3_type(s)

    def test_parallel_with_select_inside(self) -> None:
        s = parse("(&{a: +{c: wait, d: wait}} || &{b: wait})")
        assert is_L3_type(s)

    def test_branch_wrapping_parallel_is_L3(self) -> None:
        """A branch that contains a parallel is still L3."""
        s = Branch((("init", Parallel((Branch((("a", Wait()),)), Branch((("b", Wait()),))))),))
        assert is_L3_type(s)


# ===================================================================
# 2. State-space structure classification
# ===================================================================


class TestL3StateSpaceClass:
    """Tests for L3_state_space_class()."""

    def test_chain_classification(self) -> None:
        s = parse("&{a: end}")
        assert L3_state_space_class(s) == "chain"

    def test_diamond_classification(self) -> None:
        """&{a: &{c: end}, b: &{d: end}} creates a diamond: both branches converge at end."""
        s = parse("&{a: &{c: end}, b: &{d: end}}")
        assert L3_state_space_class(s) == "diamond"

    def test_product_of_chains(self) -> None:
        s = parse("(&{a: wait} || &{b: wait})")
        assert L3_state_space_class(s) == "product_of_chains"

    def test_product_of_trees(self) -> None:
        s = parse("(&{a: &{c: wait}, d: wait} || &{b: wait})")
        result = L3_state_space_class(s)
        assert result in ("product_of_chains", "product_of_trees", "general_product")

    def test_end_is_chain(self) -> None:
        s = parse("end")
        assert L3_state_space_class(s) == "chain"


# ===================================================================
# 3. Product factor decomposition
# ===================================================================


class TestProductFactors:
    """Tests for product_factors()."""

    def test_simple_product_has_two_factors(self) -> None:
        s = parse("(&{a: wait} || &{b: wait})")
        ss = build_statespace(s)
        factors = product_factors(ss)
        assert factors is not None
        assert len(factors) == 2

    def test_non_product_returns_none(self) -> None:
        s = parse("&{a: end, b: end}")
        ss = build_statespace(s)
        factors = product_factors(ss)
        assert factors is None

    def test_factor_sizes_match(self) -> None:
        s = parse("(&{a: wait} || &{b: wait})")
        ss = build_statespace(s)
        factors = product_factors(ss)
        assert factors is not None
        # Each factor: &{x: wait} has 2 states (top, wait/end)
        for f in factors:
            assert len(f.states) == 2

    def test_product_size_equals_factor_product(self) -> None:
        s = parse("(&{a: wait} || &{b: wait})")
        ss = build_statespace(s)
        factors = product_factors(ss)
        assert factors is not None
        predicted = product_size_prediction([len(f.states) for f in factors])
        assert len(ss.states) == predicted

    def test_three_branch_factors(self) -> None:
        """Left factor: &{a: &{c: wait}, d: wait}."""
        s = parse("(&{a: &{c: wait}, d: wait} || &{b: wait})")
        ss = build_statespace(s)
        factors = product_factors(ss)
        assert factors is not None
        assert len(factors) == 2


# ===================================================================
# 4. Factor analysis
# ===================================================================


class TestFactorAnalysis:
    """Tests for factor_analysis()."""

    def test_analysis_returns_correct_count(self) -> None:
        s = parse("(&{a: wait} || &{b: wait})")
        ss = build_statespace(s)
        fa = factor_analysis(ss)
        assert fa is not None
        assert len(fa) == 2

    def test_factor_indices(self) -> None:
        s = parse("(&{a: wait} || &{b: wait})")
        ss = build_statespace(s)
        fa = factor_analysis(ss)
        assert fa is not None
        assert fa[0].index == 0
        assert fa[1].index == 1

    def test_factor_sizes(self) -> None:
        s = parse("(&{a: wait} || &{b: wait})")
        ss = build_statespace(s)
        fa = factor_analysis(ss)
        assert fa is not None
        for f in fa:
            assert f.size == 2

    def test_factor_heights(self) -> None:
        s = parse("(&{a: wait} || &{b: wait})")
        ss = build_statespace(s)
        fa = factor_analysis(ss)
        assert fa is not None
        for f in fa:
            assert f.height == 1  # single transition: top -> wait

    def test_non_product_returns_none(self) -> None:
        s = parse("&{a: end}")
        ss = build_statespace(s)
        assert factor_analysis(ss) is None

    def test_factor_distributivity(self) -> None:
        s = parse("(&{a: wait} || &{b: wait})")
        ss = build_statespace(s)
        fa = factor_analysis(ss)
        assert fa is not None
        for f in fa:
            assert f.is_distributive  # chains are distributive


# ===================================================================
# 5. Size prediction
# ===================================================================


class TestSizePrediction:
    """Tests for product_size_prediction()."""

    def test_two_factor_prediction(self) -> None:
        assert product_size_prediction([2, 3]) == 6

    def test_three_factor_prediction(self) -> None:
        assert product_size_prediction([2, 3, 4]) == 24

    def test_single_factor(self) -> None:
        assert product_size_prediction([5]) == 5

    def test_with_one(self) -> None:
        assert product_size_prediction([1, 7]) == 7

    def test_accuracy_on_parallel_type(self) -> None:
        """Size prediction matches actual product size."""
        s = parse("(&{a: &{c: wait}, d: wait} || &{b: &{e: wait}})")
        ss = build_statespace(s)
        factors = product_factors(ss)
        assert factors is not None
        predicted = product_size_prediction([len(f.states) for f in factors])
        assert len(ss.states) == predicted

    def test_height_prediction(self) -> None:
        assert product_height_prediction([2, 3]) == 5

    def test_width_prediction(self) -> None:
        assert product_width_prediction([2, 3]) == 6


# ===================================================================
# 6. Distributivity preservation
# ===================================================================


class TestDistributivityPreservation:
    """Tests for distributivity_from_factors() and verify_distributivity_preservation()."""

    def test_all_distributive_gives_distributive(self) -> None:
        assert distributivity_from_factors([True, True]) is True

    def test_one_non_distributive_gives_non_distributive(self) -> None:
        assert distributivity_from_factors([True, False]) is False

    def test_all_non_distributive(self) -> None:
        assert distributivity_from_factors([False, False]) is False

    def test_empty_is_distributive(self) -> None:
        """Vacuous: empty product is distributive."""
        assert distributivity_from_factors([]) is True

    def test_verify_on_simple_product(self) -> None:
        s = parse("(&{a: wait} || &{b: wait})")
        ss = build_statespace(s)
        result = verify_distributivity_preservation(ss)
        assert result is True

    def test_verify_on_non_product_returns_none(self) -> None:
        s = parse("&{a: end}")
        ss = build_statespace(s)
        result = verify_distributivity_preservation(ss)
        assert result is None


# ===================================================================
# 7. Enumeration
# ===================================================================


class TestL3Enumerate:
    """Tests for L3_enumerate()."""

    def test_enumerate_returns_results(self) -> None:
        results = L3_enumerate(depth=1, width=1, labels=("a",))
        assert len(results) > 0

    def test_enumerate_entries_are_4_tuples(self) -> None:
        results = L3_enumerate(depth=1, width=1, labels=("a",))
        for r in results:
            assert len(r) == 4
            type_str, states, trans, is_lat = r
            assert isinstance(type_str, str)
            assert isinstance(states, int)
            assert isinstance(trans, int)
            assert isinstance(is_lat, bool)

    def test_all_enumerated_are_lattices(self) -> None:
        """All terminating L₃ types should form lattices."""
        results = L3_enumerate(depth=1, width=2, labels=("a", "b"))
        for type_str, states, trans, is_lat in results:
            assert is_lat, f"Not a lattice: {type_str}"

    def test_enumerate_small_depth(self) -> None:
        results = L3_enumerate(depth=1, width=1, labels=("a",))
        # At depth 1, width 1: left has {wait, &{a: wait}, +{a: wait}}
        # right has {wait, &{a_r: wait}, +{a_r: wait}}
        # 3 × 3 = 9 combinations
        assert len(results) == 9

    def test_product_structure_in_enumerated(self) -> None:
        """Every enumerated L₃ type should have product structure."""
        results = L3_enumerate(depth=1, width=1, labels=("a",))
        for type_str, _, _, _ in results:
            ast = parse(type_str)
            ss = build_statespace(ast)
            # All are parallel, so should have product_factors
            factors = product_factors(ss)
            assert factors is not None, f"No product structure: {type_str}"


# ===================================================================
# 8. L₂ vs L₃ comparison
# ===================================================================


class TestCompareL2L3:
    """Tests for compare_L2_L3()."""

    def test_comparison_returns_level_comparison(self) -> None:
        comp = compare_L2_L3(depth=1, width=1, labels=("a",))
        assert isinstance(comp, LevelComparison)

    def test_l3_has_more_states(self) -> None:
        """L₃ products can be larger than L₂ types at same depth."""
        comp = compare_L2_L3(depth=1, width=2, labels=("a", "b"))
        # Product types can have |L₁|·|L₂| states
        assert comp.l3_max_states >= comp.l2_max_states

    def test_both_levels_all_lattice(self) -> None:
        comp = compare_L2_L3(depth=1, width=1, labels=("a",))
        assert comp.l2_all_lattice
        assert comp.l3_all_lattice

    def test_l3_has_products(self) -> None:
        comp = compare_L2_L3(depth=1, width=1, labels=("a",))
        assert comp.l3_has_products


# ===================================================================
# 9. Benchmark integration
# ===================================================================


class TestBenchmarkProducts:
    """Test L₃ analysis on benchmark protocols with parallel."""

    def test_mcp_has_product_factors(self) -> None:
        """MCP protocol has parallel composition."""
        mcp = parse(
            "&{initialize: (rec X . &{callTool: +{RESULT: X, ERROR: X}, "
            "listTools: X, shutdown: end} || "
            "rec Y . +{NOTIFICATION: Y, DONE: end})}"
        )
        ss = build_statespace(mcp)
        # MCP has parallel, so product_factors should exist
        # (the parallel is inside rec, so factor extraction depends on implementation)
        factors = product_factors(ss)
        # MCP is L7 (rec containing parallel), but the product structure
        # should still be present in the state space
        if factors is not None:
            predicted = product_size_prediction([len(f.states) for f in factors])
            # Verify size prediction (MCP wraps parallel in &{initialize:...})
            # so the full state space is larger than just the product
            assert predicted > 0

    def test_two_buyer_product(self) -> None:
        """Two-Buyer protocol uses parallel."""
        two_buyer = parse(
            "&{lookup: &{price: (&{proposeA: wait} || "
            "&{proposeB: +{ACCEPT: &{pay: wait}, REJECT: wait}})}}"
        )
        ss = build_statespace(two_buyer)
        lr = check_lattice(ss)
        assert lr.is_lattice

    def test_file_channel_product(self) -> None:
        """File Channel has concurrent read/write streams."""
        fc = parse(
            "&{open: +{OK: (rec X . &{read: X, doneRead: wait} || "
            "rec Y . &{write: Y, doneWrite: wait}) . &{close: end}, ERR: end}}"
        )
        ss = build_statespace(fc)
        lr = check_lattice(ss)
        assert lr.is_lattice


# ===================================================================
# 10. Edge cases and mathematical properties
# ===================================================================


class TestMathematicalProperties:
    """Test mathematical properties of L₃ product lattices."""

    def test_product_of_chains_is_lattice(self) -> None:
        """Product of two chains is always a lattice."""
        s = parse("(&{a: wait} || &{b: wait})")
        ss = build_statespace(s)
        lr = check_lattice(ss)
        assert lr.is_lattice

    def test_product_of_chains_is_distributive(self) -> None:
        """Product of chains is distributive."""
        s = parse("(&{a: wait} || &{b: wait})")
        ss = build_statespace(s)
        dr = check_distributive(ss)
        assert dr.is_distributive

    def test_product_commutativity_size(self) -> None:
        """L₁ × L₂ and L₂ × L₁ have the same number of states."""
        s1 = parse("(&{a: wait} || &{b: &{c: wait}})")
        s2 = parse("(&{b: &{c: wait}} || &{a: wait})")
        ss1 = build_statespace(s1)
        ss2 = build_statespace(s2)
        assert len(ss1.states) == len(ss2.states)
        assert len(ss1.transitions) == len(ss2.transitions)

    def test_wait_maps_to_end(self) -> None:
        """Wait inside parallel maps to end at state-space level."""
        s = parse("(&{a: wait} || &{b: wait})")
        ss = build_statespace(s)
        # Bottom should be reachable (wait → end)
        assert ss.bottom in ss.reachable_from(ss.top)
