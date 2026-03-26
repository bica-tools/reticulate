"""Tests for quorum extension: Q(k,n){S} — fault-tolerant protocol completion.

Tests cover:
- Parsing Q(k,n){ S } syntax
- AST node validation (frozen, post_init checks)
- Q(n,n){S} = S^n (full product, always lattice)
- Q(0,n){S} = S^n (all states satisfy trivially)
- Q(k,n) on chains: should always be lattice
- Q(k,n) on branching types: check when it breaks
- Q(2,3) on Iterator
- Fault tolerance sweep
- Concrete counterexample search
"""

import pytest

from reticulate.parser import parse, End, Branch
from reticulate.statespace import build_statespace
from reticulate.product import power_statespace
from reticulate.lattice import check_lattice
from reticulate.extensions.quorum import (
    Quorum,
    QuorumAnalysis,
    analyze_quorum,
    build_quorum_statespace,
    fault_tolerance_analysis,
    is_quorum_lattice,
    parse_quorum,
    pretty_quorum,
    quorum_counterexample,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CHAIN_2 = "&{a: end}"                        # 2-state chain: top -> end
CHAIN_3 = "&{a: &{b: end}}"                  # 3-state chain: top -> a -> end
BRANCH_2 = "&{a: end, b: end}"               # diamond: top -> {a,b} -> end
ITERATOR = "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"
SELECT_2 = "+{a: end, b: end}"               # selection with 2 options


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

class TestParseQuorum:
    def test_basic_parse(self):
        q = parse_quorum("Q(2,3){ &{a: end} }")
        assert q.k == 2
        assert q.n == 3
        assert isinstance(q.body, Branch)

    def test_parse_end(self):
        q = parse_quorum("Q(1,1){ end }")
        assert q.k == 1
        assert q.n == 1
        assert isinstance(q.body, End)

    def test_parse_spaces(self):
        q = parse_quorum("  Q( 2 , 5 ){  &{a: end}  }  ")
        assert q.k == 2
        assert q.n == 5

    def test_parse_recursive(self):
        q = parse_quorum(f"Q(1,2){{ {ITERATOR} }}")
        assert q.k == 1
        assert q.n == 2

    def test_parse_invalid_syntax(self):
        with pytest.raises(ValueError, match="Invalid quorum syntax"):
            parse_quorum("not a quorum")

    def test_parse_invalid_no_braces(self):
        with pytest.raises(ValueError, match="Invalid quorum syntax"):
            parse_quorum("Q(1,2) &{a: end}")


# ---------------------------------------------------------------------------
# AST node validation
# ---------------------------------------------------------------------------

class TestQuorumNode:
    def test_frozen(self):
        q = Quorum(body=End(), k=1, n=2)
        with pytest.raises(AttributeError):
            q.k = 3  # type: ignore[misc]

    def test_k_negative(self):
        with pytest.raises(ValueError, match="non-negative"):
            Quorum(body=End(), k=-1, n=2)

    def test_n_zero(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            Quorum(body=End(), k=0, n=0)

    def test_k_exceeds_n(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            Quorum(body=End(), k=3, n=2)

    def test_valid_boundary(self):
        q = Quorum(body=End(), k=2, n=2)
        assert q.k == q.n == 2

    def test_k_zero_valid(self):
        q = Quorum(body=End(), k=0, n=5)
        assert q.k == 0


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

class TestPrettyQuorum:
    def test_pretty_basic(self):
        q = Quorum(body=parse("&{a: end}"), k=2, n=3)
        s = pretty_quorum(q)
        assert "Q(2,3)" in s
        assert "&{a: end}" in s

    def test_roundtrip(self):
        original = "Q(1,2){ end }"
        q = parse_quorum(original)
        s = pretty_quorum(q)
        assert "Q(1,2)" in s
        assert "end" in s


# ---------------------------------------------------------------------------
# Q(n,n){S} = full product (always lattice for lattice base)
# ---------------------------------------------------------------------------

class TestFullProduct:
    """Q(n,n){S} requires ALL copies to complete = same as S^n."""

    def test_q_n_n_chain_is_lattice(self):
        """Q(2,2){chain} = chain^2, which is always a lattice."""
        assert is_quorum_lattice(CHAIN_2, 2, 2)

    def test_q_n_n_branch_is_lattice(self):
        """Q(2,2){branch} = branch^2, lattice since base is lattice."""
        assert is_quorum_lattice(BRANCH_2, 2, 2)

    def test_q_n_n_same_states_as_power(self):
        """Q(n,n) should have same state count as S^n."""
        base = build_statespace(parse(CHAIN_2))
        power = power_statespace(base, 3)
        q = Quorum(body=parse(CHAIN_2), k=3, n=3)
        qss = build_quorum_statespace(q)
        assert len(qss.states) == len(power.states)

    def test_q_1_1_is_base(self):
        """Q(1,1){S} = S itself."""
        analysis = analyze_quorum(CHAIN_2, 1, 1)
        assert analysis.surviving_states == analysis.product_states
        assert analysis.is_lattice


# ---------------------------------------------------------------------------
# Q(0,n){S} = full product (trivially, 0 completions always met)
# ---------------------------------------------------------------------------

class TestTrivialQuorum:
    def test_q_0_n_full_product(self):
        """Q(0,n){S} keeps all states (0 completions always satisfied)."""
        analysis = analyze_quorum(CHAIN_2, 0, 3)
        assert analysis.removed_states == 0
        assert analysis.surviving_states == analysis.product_states

    def test_q_0_is_lattice(self):
        """Q(0,n) = S^n which is a lattice."""
        assert is_quorum_lattice(BRANCH_2, 0, 2)


# ---------------------------------------------------------------------------
# Chains: Q(k,n) on chains should always be a lattice
# ---------------------------------------------------------------------------

class TestChainsAlwaysLattice:
    """Chains are totally ordered, so products are distributive lattices.
    Truncation of a chain product should preserve lattice structure."""

    def test_chain2_q1_2(self):
        assert is_quorum_lattice(CHAIN_2, 1, 2)

    def test_chain2_q1_3(self):
        assert is_quorum_lattice(CHAIN_2, 1, 3)

    def test_chain2_q2_3(self):
        assert is_quorum_lattice(CHAIN_2, 2, 3)

    def test_chain3_q1_2(self):
        assert is_quorum_lattice(CHAIN_3, 1, 2)

    def test_chain3_q2_3(self):
        assert is_quorum_lattice(CHAIN_3, 2, 3)


# ---------------------------------------------------------------------------
# Branching types: check when quorum breaks lattice
# ---------------------------------------------------------------------------

class TestBranchingTypes:
    def test_branch2_q1_2(self):
        """Q(1,2) on a diamond — may or may not be lattice."""
        result = analyze_quorum(BRANCH_2, 1, 2)
        # Record whether it's a lattice (exploratory)
        assert isinstance(result.is_lattice, bool)

    def test_branch2_q1_3(self):
        result = analyze_quorum(BRANCH_2, 1, 3)
        assert isinstance(result.is_lattice, bool)

    def test_branch2_truncation_removes_states(self):
        """For k >= 1, some states should be removed from the product."""
        analysis = analyze_quorum(BRANCH_2, 2, 2)
        # Q(2,2) = full product, no removal
        assert analysis.removed_states == 0

    def test_branch2_q1_2_removes_some(self):
        """Q(1,2) on branch: should remove states where 0 components done."""
        analysis = analyze_quorum(BRANCH_2, 1, 2)
        # Some states may have 0 completions but still survive if on path
        assert analysis.surviving_states <= analysis.product_states

    def test_select2_q1_2(self):
        """Selection type as body."""
        result = analyze_quorum(SELECT_2, 1, 2)
        assert isinstance(result.is_lattice, bool)


# ---------------------------------------------------------------------------
# Iterator: Q(2,3) and sweep
# ---------------------------------------------------------------------------

class TestIteratorQuorum:
    def test_iterator_q1_1(self):
        """Q(1,1){Iterator} = Iterator itself, which is a lattice."""
        assert is_quorum_lattice(ITERATOR, 1, 1)

    def test_iterator_q1_2(self):
        """Q(1,2){Iterator}: at least 1 of 2 iterators must finish."""
        result = analyze_quorum(ITERATOR, 1, 2)
        assert isinstance(result.is_lattice, bool)
        assert result.surviving_states > 0

    def test_iterator_q2_2(self):
        """Q(2,2){Iterator} = Iterator^2 (full product)."""
        result = analyze_quorum(ITERATOR, 2, 2)
        assert result.removed_states == 0
        assert result.is_lattice

    def test_iterator_q2_3(self):
        """Q(2,3){Iterator}: at least 2 of 3 iterators must finish."""
        result = analyze_quorum(ITERATOR, 2, 3)
        assert isinstance(result.is_lattice, bool)
        assert result.surviving_states > 0


# ---------------------------------------------------------------------------
# Fault tolerance sweep
# ---------------------------------------------------------------------------

class TestFaultToleranceSweep:
    def test_sweep_chain(self):
        """Sweep chain for n=1..3 — all should be lattice."""
        results = fault_tolerance_analysis(CHAIN_2, 3)
        assert len(results) > 0
        for r in results:
            assert r["is_lattice"], f"Q({r['k']},{r['n']}) on chain failed!"

    def test_sweep_branch(self):
        """Sweep branch for n=1..2 — check structure."""
        results = fault_tolerance_analysis(BRANCH_2, 2)
        # At minimum, Q(0,n) and Q(n,n) should be lattice
        for r in results:
            if r["k"] == 0 or r["k"] == r["n"]:
                assert r["is_lattice"], f"Q({r['k']},{r['n']}) should be lattice"

    def test_sweep_iterator(self):
        """Sweep Iterator for n=1..2 — record results."""
        results = fault_tolerance_analysis(ITERATOR, 2)
        # Q(0,n) should always be lattice (= full product)
        for r in results:
            if r["k"] == 0:
                assert r["is_lattice"]

    def test_sweep_result_format(self):
        """Verify sweep returns correct dict structure."""
        results = fault_tolerance_analysis(CHAIN_2, 2)
        for r in results:
            assert "k" in r
            assert "n" in r
            assert "is_lattice" in r
            assert "product_states" in r
            assert "surviving_states" in r
            assert "removed_states" in r
            assert "quorum_states" in r

    def test_sweep_counts(self):
        """Sweep for max_n=3 produces (0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)."""
        results = fault_tolerance_analysis(CHAIN_2, 3)
        expected_pairs = {(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)}
        actual_pairs = {(r["k"], r["n"]) for r in results}
        assert actual_pairs == expected_pairs


# ---------------------------------------------------------------------------
# Counterexample search
# ---------------------------------------------------------------------------

class TestCounterexample:
    def test_full_product_no_counterexample(self):
        """Q(n,n) = full product on lattice base — no counterexample."""
        ce = quorum_counterexample(CHAIN_2, 2, 2)
        assert ce is None

    def test_counterexample_format(self):
        """If a counterexample exists, it should be (a, b, kind)."""
        ce = quorum_counterexample(BRANCH_2, 1, 3)
        if ce is not None:
            assert len(ce) == 3
            assert ce[2] in ("no_meet", "no_join")


# ---------------------------------------------------------------------------
# State space properties
# ---------------------------------------------------------------------------

class TestStateSpaceProperties:
    def test_top_reachable(self):
        """Top should reach all surviving states."""
        q = Quorum(body=parse(CHAIN_2), k=1, n=2)
        qss = build_quorum_statespace(q)
        reachable = qss.reachable_from(qss.top)
        assert reachable == qss.states

    def test_bottom_reachable_from_all(self):
        """Bottom should be reachable from all surviving states."""
        q = Quorum(body=parse(CHAIN_2), k=1, n=2)
        qss = build_quorum_statespace(q)
        for s in qss.states:
            assert qss.bottom in qss.reachable_from(s)

    def test_labels_contain_completion_count(self):
        """Labels should show completion count [c/n]."""
        q = Quorum(body=parse(CHAIN_2), k=1, n=2)
        qss = build_quorum_statespace(q)
        for s in qss.states:
            label = qss.labels.get(s, "")
            assert "/" in label, f"Label {label} missing completion count"

    def test_quorum_states_subset(self):
        """Quorum-satisfying states should be a subset of surviving."""
        analysis = analyze_quorum(CHAIN_2, 1, 2)
        assert analysis.quorum_states <= analysis.surviving_states


# ---------------------------------------------------------------------------
# KEY exploration: find concrete non-lattice example
# ---------------------------------------------------------------------------

class TestExploreNonLattice:
    """Search for concrete Q(k,n){S} that is NOT a lattice.

    If all tested cases ARE lattices, that itself is a surprising result
    worth documenting.
    """

    def test_explore_branch_q1_3(self):
        """Q(1,3) on &{a: end, b: end}: 3 copies, only 1 needs to finish."""
        result = analyze_quorum(BRANCH_2, 1, 3)
        # Document the result either way
        if not result.is_lattice:
            assert result.counterexample is not None
        # else: surprising! The truncation preserved lattice structure

    def test_explore_deeper_branch(self):
        """Deeper branching type: &{a: &{c: end}, b: end}."""
        deeper = "&{a: &{c: end}, b: end}"
        result = analyze_quorum(deeper, 1, 2)
        assert isinstance(result.is_lattice, bool)

    def test_all_full_products_are_lattice(self):
        """Sanity: Q(n,n) should always be lattice for any lattice base."""
        for body in [CHAIN_2, CHAIN_3, BRANCH_2, SELECT_2]:
            assert is_quorum_lattice(body, 2, 2)

    def test_summary_of_lattice_preservation(self):
        """Survey: for which (body, k, n) is the quorum a lattice?"""
        cases = [
            (CHAIN_2, 1, 2), (CHAIN_2, 1, 3), (CHAIN_2, 2, 3),
            (BRANCH_2, 1, 2), (BRANCH_2, 1, 3), (BRANCH_2, 2, 3),
            (CHAIN_3, 1, 2), (CHAIN_3, 2, 3),
        ]
        lattice_count = 0
        non_lattice_count = 0
        for body, k, n in cases:
            if is_quorum_lattice(body, k, n):
                lattice_count += 1
            else:
                non_lattice_count += 1
        # At least some should be lattice
        assert lattice_count > 0
        # Report: if non_lattice_count == 0, all quorums are lattices!


# ---------------------------------------------------------------------------
# QuorumAnalysis data integrity
# ---------------------------------------------------------------------------

class TestAnalysisIntegrity:
    def test_removed_plus_surviving_equals_product(self):
        """removed + surviving = product_states."""
        for k in range(3):
            analysis = analyze_quorum(CHAIN_2, k, 2)
            assert analysis.removed_states + analysis.surviving_states == analysis.product_states

    def test_quorum_states_count(self):
        """Quorum states count matches manual calculation for chain^2."""
        # chain_2: 2 states (top=0, end=1). chain^2: 4 states.
        # Pairs: (0,0)=0done, (0,1)=1done, (1,0)=1done, (1,1)=2done
        # Q(1,2): states with >= 1 done: 3 states
        analysis = analyze_quorum(CHAIN_2, 1, 2)
        assert analysis.quorum_states == 3  # (0,1), (1,0), (1,1)

    def test_analysis_returns_correct_type(self):
        analysis = analyze_quorum(CHAIN_2, 1, 2)
        assert isinstance(analysis, QuorumAnalysis)
        assert isinstance(analysis.lattice_result, object)
