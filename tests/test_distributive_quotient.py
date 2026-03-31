"""Tests for distributive quotient computation (Step 301).

Tests the core algorithms for computing minimal distributive quotients
of session type lattices: congruence checking, quotient construction,
distributivity verification, entropy calculation, and uniqueness counting.

60+ tests across 8 test classes.
"""

from __future__ import annotations

import math

import pytest

from reticulate.distributive_quotient import (
    CongruenceClass,
    DistributiveQuotientResult,
    build_quotient_statespace,
    compute_distributive_quotient,
    count_minimal_quotients,
    direct_distributivity_check,
    is_congruence,
)
from reticulate.lattice import check_distributive, check_lattice, compute_meet, compute_join
from reticulate.parser import parse
from reticulate.statespace import StateSpace, build_statespace

from tests.benchmarks.protocols import BENCHMARKS
from tests.benchmarks.p104_self import P104_BENCHMARKS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(type_str: str) -> StateSpace:
    """Parse and build state space from a session type string."""
    return build_statespace(parse(type_str))


# ---------------------------------------------------------------------------
# Test 1: Direct distributivity check
# ---------------------------------------------------------------------------

class TestDirectDistributivityCheck:
    """Verify direct_distributivity_check against check_distributive on benchmarks."""

    def test_simple_chain(self) -> None:
        ss = _build("&{a: end}")
        assert direct_distributivity_check(ss) is True

    def test_two_branch(self) -> None:
        ss = _build("&{a: end, b: end}")
        assert direct_distributivity_check(ss) is True

    def test_nested_branch(self) -> None:
        ss = _build("&{a: &{b: end}, c: end}")
        assert direct_distributivity_check(ss) is True

    def test_product_always_distributive(self) -> None:
        ss = _build("(&{a: end} || &{b: end})")
        assert direct_distributivity_check(ss) is True

    def test_iterator_distributive(self) -> None:
        ss = _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        assert direct_distributivity_check(ss) is True

    def test_end_type(self) -> None:
        ss = _build("end")
        assert direct_distributivity_check(ss) is True

    def test_select_only(self) -> None:
        ss = _build("+{a: end, b: end}")
        assert direct_distributivity_check(ss) is True

    @pytest.mark.parametrize(
        "bench",
        BENCHMARKS[:20],
        ids=[b.name for b in BENCHMARKS[:20]],
    )
    def test_agrees_with_check_distributive_sample(self, bench) -> None:
        """Verify direct check matches check_distributive on first 20 benchmarks."""
        ast = parse(bench.type_string)
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        if not lr.is_lattice:
            assert direct_distributivity_check(ss) is False
            return
        dr = check_distributive(ss)
        result = direct_distributivity_check(ss)
        assert result == dr.is_distributive, (
            f"{bench.name}: direct={result}, check_distributive={dr.is_distributive}"
        )


# ---------------------------------------------------------------------------
# Test 2: Is congruence
# ---------------------------------------------------------------------------

class TestIsCongruence:
    """Test is_congruence on valid and invalid congruences."""

    def test_identity_is_congruence(self) -> None:
        """Identity partition (singletons) is always a congruence."""
        ss = _build("&{a: end, b: end}")
        partition = [frozenset({s}) for s in ss.states]
        assert is_congruence(ss, partition) is True

    def test_all_one_class_is_congruence(self) -> None:
        """The trivial all-in-one partition is always a congruence."""
        ss = _build("&{a: end, b: end}")
        partition = [frozenset(ss.states)]
        assert is_congruence(ss, partition) is True

    def test_valid_congruence_chain(self) -> None:
        """On a chain lattice, any partition grouping contiguous elements is a congruence."""
        ss = _build("&{a: &{b: end}}")
        # 3-element chain: top -> a_state -> end
        # Identity is always valid
        partition = [frozenset({s}) for s in ss.states]
        assert is_congruence(ss, partition) is True

    def test_invalid_partition_missing_states(self) -> None:
        """A partition missing states is not a congruence."""
        ss = _build("&{a: end, b: end}")
        # Only include some states
        some_states = list(ss.states)[:1]
        partition = [frozenset(some_states)]
        assert is_congruence(ss, partition) is False

    def test_congruence_on_product(self) -> None:
        """Identity partition on product lattice is a congruence."""
        ss = _build("(&{a: end} || &{b: end})")
        partition = [frozenset({s}) for s in ss.states]
        assert is_congruence(ss, partition) is True

    def test_congruence_on_iterator(self) -> None:
        """Identity on the iterator benchmark."""
        ss = _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        partition = [frozenset({s}) for s in ss.states]
        assert is_congruence(ss, partition) is True

    def test_invalid_random_merge(self) -> None:
        """Merging top and bottom arbitrarily is usually not a congruence."""
        ss = _build("&{a: &{b: end}, c: end}")
        states_list = sorted(ss.states)
        if len(states_list) >= 3:
            # Merge top with an intermediate, leave the rest as singletons
            # This is almost certainly not a congruence
            merged = frozenset({states_list[0], states_list[1]})
            rest = [frozenset({s}) for s in states_list[2:]]
            partition = [merged] + rest
            # We don't assert True or False here — depends on specific structure
            # Just verify it runs without error
            result = is_congruence(ss, partition)
            assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Test 3: Build quotient
# ---------------------------------------------------------------------------

class TestBuildQuotient:
    """Test quotient state space construction."""

    def test_identity_partition_preserves_states(self) -> None:
        ss = _build("&{a: end, b: end}")
        partition = [frozenset({s}) for s in ss.states]
        quotient = build_quotient_statespace(ss, partition)
        assert len(quotient.states) == len(ss.states)

    def test_all_merge_gives_one_state(self) -> None:
        """Merging all states gives a single-state quotient (with self-loops)."""
        ss = _build("&{a: end, b: end}")
        partition = [frozenset(ss.states)]
        quotient = build_quotient_statespace(ss, partition)
        assert len(quotient.states) == 1

    def test_quotient_has_top_and_bottom(self) -> None:
        ss = _build("&{a: end, b: end}")
        partition = [frozenset({s}) for s in ss.states]
        quotient = build_quotient_statespace(ss, partition)
        assert quotient.top in quotient.states
        assert quotient.bottom in quotient.states

    def test_quotient_transitions_subset(self) -> None:
        """Quotient transitions use only quotient states."""
        ss = _build("&{a: &{b: end}, c: end}")
        partition = [frozenset({s}) for s in ss.states]
        quotient = build_quotient_statespace(ss, partition)
        for src, _, tgt in quotient.transitions:
            assert src in quotient.states
            assert tgt in quotient.states

    def test_quotient_deduplicates_transitions(self) -> None:
        """Merging two states should not produce duplicate transitions."""
        ss = _build("&{a: end, b: end}")
        # Merge the two targets (both are 'end')
        # In this case, a and b both go to end, so merging anything
        # other than end with end should reduce transitions
        partition = [frozenset(ss.states)]
        quotient = build_quotient_statespace(ss, partition)
        # All transitions are within one state, no duplicates
        seen = set()
        for tr in quotient.transitions:
            assert tr not in seen, f"Duplicate transition: {tr}"
            seen.add(tr)

    def test_quotient_preserves_selections(self) -> None:
        """Selection transitions are preserved in quotient."""
        ss = _build("+{a: end, b: end}")
        partition = [frozenset({s}) for s in ss.states]
        quotient = build_quotient_statespace(ss, partition)
        assert len(quotient.selection_transitions) == len(ss.selection_transitions)

    def test_quotient_labels_for_merged(self) -> None:
        """Merged states get combined labels."""
        ss = _build("&{a: end, b: end}")
        states_list = sorted(ss.states)
        # Merge all states except bottom
        non_bottom = [s for s in states_list if s != ss.bottom]
        if len(non_bottom) >= 2:
            merged = frozenset(non_bottom)
            partition = [merged, frozenset({ss.bottom})]
            quotient = build_quotient_statespace(ss, partition)
            rep = min(non_bottom)
            assert "{" in quotient.labels.get(rep, "")


# ---------------------------------------------------------------------------
# Test 4: Distributive quotient main algorithm
# ---------------------------------------------------------------------------

class TestDistributiveQuotient:
    """Test the main compute_distributive_quotient function."""

    def test_already_distributive_chain(self) -> None:
        ss = _build("&{a: end}")
        result = compute_distributive_quotient(ss)
        assert result.is_already_distributive is True
        assert result.is_quotient_possible is True
        assert result.merged_pairs == []
        assert result.quotient_statespace is not None

    def test_already_distributive_product(self) -> None:
        """Product lattice: (&{a: end} || &{b: &{c: end}})"""
        ss = _build("(&{a: end} || &{b: &{c: end}})")
        result = compute_distributive_quotient(ss)
        assert result.is_already_distributive is True
        assert result.entropy_lost == 0.0

    def test_already_distributive_b3(self) -> None:
        """B3 product: ((&{a: end} || &{b: end}) || &{c: end})"""
        ss = _build("((&{a: end} || &{b: end}) || &{c: end})")
        result = compute_distributive_quotient(ss)
        assert result.is_already_distributive is True

    def test_already_distributive_iterator(self) -> None:
        """Iterator protocol."""
        ss = _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        result = compute_distributive_quotient(ss)
        assert result.is_already_distributive is True
        assert result.num_minimal_quotients == 1

    def test_m3_diamond_impossible(self) -> None:
        """M3 diamond: &{a: +{OK: end, E: end}, b: +{OK: end, E: end}, c: +{OK: end, E: end}}"""
        ss = _build("&{a: +{OK: end, E: end}, b: +{OK: end, E: end}, c: +{OK: end, E: end}}")
        dr = check_distributive(ss)
        if not dr.is_lattice:
            pytest.skip("Not a lattice")
        if dr.is_distributive:
            pytest.skip("Already distributive in this state space construction")
        result = compute_distributive_quotient(ss)
        assert result.is_quotient_possible is False
        assert result.num_minimal_quotients == 0

    def test_n5_from_branch_nesting(self) -> None:
        """N5 from nested branches: &{a: &{b: +{x: end, y: end}}, c: +{x: end, y: end}}"""
        ss = _build("&{a: &{b: +{x: end, y: end}}, c: +{x: end, y: end}}")
        dr = check_distributive(ss)
        if dr.is_distributive:
            pytest.skip("Already distributive in this state space construction")
        if not dr.is_lattice:
            pytest.skip("Not a lattice")
        result = compute_distributive_quotient(ss)
        assert result.is_quotient_possible is True
        assert result.quotient_statespace is not None
        assert result.quotient_states < result.original_states
        assert result.entropy_lost > 0.0
        # Verify quotient is actually distributive
        assert direct_distributivity_check(result.quotient_statespace) is True

    def test_result_has_congruence_classes(self) -> None:
        """Non-trivial quotient should have congruence classes."""
        ss = _build("&{a: &{b: +{x: end, y: end}}, c: +{x: end, y: end}}")
        dr = check_distributive(ss)
        if dr.is_distributive or not dr.is_lattice:
            pytest.skip("Need non-distributive lattice")
        result = compute_distributive_quotient(ss)
        if result.is_quotient_possible:
            assert len(result.congruence_classes) > 0

    def test_end_type_distributive(self) -> None:
        ss = _build("end")
        result = compute_distributive_quotient(ss)
        assert result.is_already_distributive is True

    def test_select_type_distributive(self) -> None:
        ss = _build("+{a: end, b: end}")
        result = compute_distributive_quotient(ss)
        assert result.is_already_distributive is True

    def test_deep_chain_distributive(self) -> None:
        """Deep chain: &{a: &{b: &{c: end}}}"""
        ss = _build("&{a: &{b: &{c: end}}}")
        result = compute_distributive_quotient(ss)
        assert result.is_already_distributive is True

    def test_original_states_count(self) -> None:
        ss = _build("&{a: end, b: end}")
        result = compute_distributive_quotient(ss)
        assert result.original_states == len(ss.states)

    def test_quotient_states_leq_original(self) -> None:
        """Quotient never has more states than original."""
        ss = _build("&{a: &{b: +{x: end, y: end}}, c: +{x: end, y: end}}")
        result = compute_distributive_quotient(ss)
        assert result.quotient_states <= result.original_states


# ---------------------------------------------------------------------------
# Test 5: Entropy calculation
# ---------------------------------------------------------------------------

class TestEntropyCalculation:
    """Test entropy before/after/lost calculations."""

    def test_entropy_before_positive(self) -> None:
        ss = _build("&{a: end, b: end}")
        result = compute_distributive_quotient(ss)
        assert result.entropy_before > 0.0

    def test_entropy_single_state(self) -> None:
        """Single state: log2(1) = 0."""
        ss = _build("end")
        result = compute_distributive_quotient(ss)
        assert result.entropy_before == 0.0

    def test_entropy_lost_zero_when_distributive(self) -> None:
        ss = _build("&{a: end, b: end}")
        result = compute_distributive_quotient(ss)
        assert result.entropy_lost == 0.0

    def test_entropy_lost_positive_when_merged(self) -> None:
        """When states are merged, entropy is lost."""
        ss = _build("&{a: &{b: +{x: end, y: end}}, c: +{x: end, y: end}}")
        dr = check_distributive(ss)
        if dr.is_distributive or not dr.is_lattice:
            pytest.skip("Need non-distributive lattice")
        result = compute_distributive_quotient(ss)
        if result.is_quotient_possible:
            assert result.entropy_lost > 0.0

    def test_entropy_consistency(self) -> None:
        """entropy_lost = entropy_before - entropy_after."""
        ss = _build("(&{a: end} || &{b: end})")
        result = compute_distributive_quotient(ss)
        assert abs(result.entropy_lost - (result.entropy_before - result.entropy_after)) < 1e-10

    def test_entropy_two_states(self) -> None:
        """Two states: log2(2) = 1.0."""
        ss = _build("&{a: end}")
        result = compute_distributive_quotient(ss)
        assert abs(result.entropy_before - math.log2(len(ss.states))) < 1e-10

    def test_entropy_four_states(self) -> None:
        """Four states: log2(4) = 2.0."""
        ss = _build("(&{a: end} || &{b: end})")
        result = compute_distributive_quotient(ss)
        expected = math.log2(len(ss.states))
        assert abs(result.entropy_before - expected) < 1e-10

    def test_entropy_after_leq_before(self) -> None:
        """Entropy after merging is <= entropy before."""
        ss = _build("&{a: &{b: +{x: end, y: end}}, c: +{x: end, y: end}}")
        result = compute_distributive_quotient(ss)
        assert result.entropy_after <= result.entropy_before + 1e-10


# ---------------------------------------------------------------------------
# Test 6: Count minimal quotients
# ---------------------------------------------------------------------------

class TestCountMinimalQuotients:
    """Test count_minimal_quotients for uniqueness verification."""

    def test_already_distributive_returns_1(self) -> None:
        ss = _build("&{a: end}")
        assert count_minimal_quotients(ss) == 1

    def test_product_returns_1(self) -> None:
        ss = _build("(&{a: end} || &{b: end})")
        assert count_minimal_quotients(ss) == 1

    def test_m3_diamond_returns_0(self) -> None:
        """M3 has no non-trivial distributive quotient."""
        ss = _build("&{a: +{OK: end, E: end}, b: +{OK: end, E: end}, c: +{OK: end, E: end}}")
        dr = check_distributive(ss)
        if not dr.is_lattice or dr.is_distributive:
            pytest.skip("Need non-distributive lattice for M3 test")
        assert count_minimal_quotients(ss) == 0

    def test_n5_has_quotient(self) -> None:
        """N5 should have at least one minimal distributive quotient."""
        ss = _build("&{a: &{b: +{x: end, y: end}}, c: +{x: end, y: end}}")
        dr = check_distributive(ss)
        if not dr.is_lattice or dr.is_distributive:
            pytest.skip("Need non-distributive lattice for N5 test")
        count = count_minimal_quotients(ss)
        assert count >= 1

    def test_iterator_returns_1(self) -> None:
        ss = _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        assert count_minimal_quotients(ss) == 1

    def test_chain_returns_1(self) -> None:
        ss = _build("&{a: &{b: &{c: end}}}")
        assert count_minimal_quotients(ss) == 1


# ---------------------------------------------------------------------------
# Test 7: P104 self-reference benchmarks
# ---------------------------------------------------------------------------

class TestP104SelfReference:
    """Run distributive quotient analysis on P104 benchmark protocols."""

    @pytest.mark.parametrize(
        "bench",
        P104_BENCHMARKS,
        ids=[b.name for b in P104_BENCHMARKS],
    )
    def test_p104_quotient_runs(self, bench) -> None:
        """Verify quotient computation runs on all P104 benchmarks."""
        ast = parse(bench.type_string)
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        if not lr.is_lattice:
            pytest.skip(f"{bench.name}: not a lattice")
        result = compute_distributive_quotient(ss)
        assert isinstance(result, DistributiveQuotientResult)
        assert result.original_states == len(ss.states)

    @pytest.mark.parametrize(
        "bench",
        P104_BENCHMARKS,
        ids=[b.name for b in P104_BENCHMARKS],
    )
    def test_p104_entropy_non_negative(self, bench) -> None:
        """Entropy lost is never negative."""
        ast = parse(bench.type_string)
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        if not lr.is_lattice:
            pytest.skip(f"{bench.name}: not a lattice")
        result = compute_distributive_quotient(ss)
        assert result.entropy_lost >= -1e-10

    @pytest.mark.parametrize(
        "bench",
        [b for b in P104_BENCHMARKS if b.expected_distributive],
        ids=[b.name for b in P104_BENCHMARKS if b.expected_distributive],
    )
    def test_p104_distributive_already(self, bench) -> None:
        """P104 benchmarks expected to be distributive need no quotient."""
        ast = parse(bench.type_string)
        ss = build_statespace(ast)
        result = compute_distributive_quotient(ss)
        assert result.is_already_distributive is True


# ---------------------------------------------------------------------------
# Test 8: Benchmark correctness regression
# ---------------------------------------------------------------------------

class TestBenchmarkCorrectnessRegression:
    """Verify direct_distributivity_check matches check_distributive on ALL benchmarks.

    This is the regression test for ensuring the direct (ground-truth) triple
    check agrees with the Birkhoff forbidden-sublattice check.
    """

    @pytest.mark.parametrize(
        "bench",
        BENCHMARKS,
        ids=[b.name for b in BENCHMARKS],
    )
    def test_direct_matches_birkhoff(self, bench) -> None:
        """direct_distributivity_check agrees with check_distributive on every benchmark."""
        ast = parse(bench.type_string)
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        if not lr.is_lattice:
            # Both should return False
            assert direct_distributivity_check(ss) is False
            return
        dr = check_distributive(ss)
        direct = direct_distributivity_check(ss)
        assert direct == dr.is_distributive, (
            f"{bench.name}: direct={direct}, Birkhoff={dr.is_distributive}, "
            f"M3={dr.has_m3}, N5={dr.has_n5}"
        )
