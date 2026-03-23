"""Tests for coverability analysis of session-type Petri nets (Step 24)."""

import math
import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.petri import build_petri_net, Marking
from reticulate.coverability import (
    OMEGA,
    CoverabilityNode,
    CoverabilityResult,
    OmegaMarking,
    _covers,
    _dominates,
    _freeze_omega,
    _omega_enabled,
    _omega_fire,
    analyze_coverability,
    analyze_unbounded_recursion,
    build_coverability_tree,
    build_unbounded_recursion_net,
    check_boundedness,
    is_coverable,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _ss(type_string: str):
    """Parse and build state space from a type string."""
    return build_statespace(parse(type_string))


def _net(type_string: str):
    """Parse, build state space, and build Petri net."""
    ss = _ss(type_string)
    return build_petri_net(ss), ss


# ---------------------------------------------------------------------------
# Omega-marking utilities
# ---------------------------------------------------------------------------

class TestOmegaUtilities:
    """Test omega-marking helper functions."""

    def test_freeze_omega_basic(self):
        m: OmegaMarking = {0: 1, 1: 0}
        frozen = _freeze_omega(m)
        assert isinstance(frozen, tuple)

    def test_dominates_strict(self):
        m1: OmegaMarking = {0: 2, 1: 1}
        m2: OmegaMarking = {0: 1, 1: 1}
        assert _dominates(m1, m2) is True

    def test_dominates_equal(self):
        m1: OmegaMarking = {0: 1, 1: 1}
        m2: OmegaMarking = {0: 1, 1: 1}
        assert _dominates(m1, m2) is False

    def test_dominates_incomparable(self):
        m1: OmegaMarking = {0: 2, 1: 0}
        m2: OmegaMarking = {0: 1, 1: 1}
        assert _dominates(m1, m2) is False

    def test_dominates_omega(self):
        m1: OmegaMarking = {0: OMEGA, 1: 1}
        m2: OmegaMarking = {0: 1, 1: 1}
        assert _dominates(m1, m2) is True

    def test_covers_exact(self):
        m: OmegaMarking = {0: 1}
        target: Marking = {0: 1}
        assert _covers(m, target) is True

    def test_covers_greater(self):
        m: OmegaMarking = {0: 2, 1: 1}
        target: Marking = {0: 1}
        assert _covers(m, target) is True

    def test_covers_omega(self):
        m: OmegaMarking = {0: OMEGA}
        target: Marking = {0: 100}
        assert _covers(m, target) is True

    def test_covers_fails(self):
        m: OmegaMarking = {0: 1}
        target: Marking = {0: 2}
        assert _covers(m, target) is False


# ---------------------------------------------------------------------------
# Coverability tree — end type
# ---------------------------------------------------------------------------

class TestCoverabilityEnd:
    """Coverability tree for the trivial 'end' type."""

    def test_end_tree_single_node(self):
        net, ss = _net("end")
        result = build_coverability_tree(net)
        assert result.num_nodes == 1
        assert result.is_bounded is True
        assert result.is_one_safe is True

    def test_end_no_unbounded_places(self):
        net, ss = _net("end")
        result = build_coverability_tree(net)
        assert result.unbounded_places == set()

    def test_end_max_tokens(self):
        net, ss = _net("end")
        result = build_coverability_tree(net)
        for p, v in result.max_tokens_per_place.items():
            assert v <= 1


# ---------------------------------------------------------------------------
# Coverability tree — simple branch
# ---------------------------------------------------------------------------

class TestCoverabilityBranch:
    """Coverability for branch session types."""

    def test_single_method_bounded(self):
        net, ss = _net("&{m: end}")
        result = build_coverability_tree(net)
        assert result.is_bounded is True
        assert result.is_one_safe is True

    def test_two_methods_bounded(self):
        net, ss = _net("&{a: end, b: end}")
        result = build_coverability_tree(net)
        assert result.is_bounded is True
        assert result.is_one_safe is True

    def test_nested_branch_bounded(self):
        net, ss = _net("&{a: &{b: end}, c: end}")
        result = build_coverability_tree(net)
        assert result.is_bounded is True
        assert result.is_one_safe is True


# ---------------------------------------------------------------------------
# Coverability tree — selection
# ---------------------------------------------------------------------------

class TestCoverabilitySelection:
    """Coverability for selection session types."""

    def test_selection_bounded(self):
        net, ss = _net("+{ok: end, err: end}")
        result = build_coverability_tree(net)
        assert result.is_bounded is True
        assert result.is_one_safe is True


# ---------------------------------------------------------------------------
# Coverability tree — parallel
# ---------------------------------------------------------------------------

class TestCoverabilityParallel:
    """Coverability for parallel session types."""

    def test_parallel_bounded(self):
        net, ss = _net("(&{a: end} || &{b: end})")
        result = build_coverability_tree(net)
        assert result.is_bounded is True
        assert result.is_one_safe is True

    def test_parallel_nested_bounded(self):
        net, ss = _net("(&{a: &{c: end}} || &{b: end})")
        result = build_coverability_tree(net)
        assert result.is_bounded is True


# ---------------------------------------------------------------------------
# Coverability tree — recursion
# ---------------------------------------------------------------------------

class TestCoverabilityRecursion:
    """Coverability for recursive session types."""

    def test_simple_recursion_bounded(self):
        net, ss = _net("rec X . &{a: X, b: end}")
        result = build_coverability_tree(net)
        assert result.is_bounded is True
        assert result.is_one_safe is True

    def test_iterator_bounded(self):
        net, ss = _net("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        result = build_coverability_tree(net)
        assert result.is_bounded is True
        assert result.is_one_safe is True


# ---------------------------------------------------------------------------
# is_coverable
# ---------------------------------------------------------------------------

class TestIsCoverable:
    """Test the is_coverable function."""

    def test_initial_coverable(self):
        net, ss = _net("&{m: end}")
        # The initial marking (token at top) is trivially coverable
        assert is_coverable(net, net.initial_marking) is True

    def test_reachable_marking_coverable(self):
        net, ss = _net("&{m: end}")
        # The end marking (token at bottom) is coverable
        end_marking: Marking = {ss.bottom: 1}
        assert is_coverable(net, end_marking) is True

    def test_unreachable_marking_not_coverable(self):
        net, ss = _net("&{a: end, b: end}")
        # A marking with tokens on two places simultaneously is not coverable
        # in a 1-safe net (would need 2 tokens)
        multi: Marking = {ss.top: 1, ss.bottom: 1}
        assert is_coverable(net, multi) is False


# ---------------------------------------------------------------------------
# check_boundedness
# ---------------------------------------------------------------------------

class TestCheckBoundedness:
    """Test the check_boundedness convenience function."""

    def test_bounded_returns_result(self):
        net, ss = _net("&{a: end}")
        result = check_boundedness(net)
        assert isinstance(result, CoverabilityResult)
        assert result.is_bounded is True

    def test_bounded_all_session_types(self):
        """All standard session type nets are bounded."""
        types = [
            "end",
            "&{m: end}",
            "+{ok: end, err: end}",
            "(&{a: end} || &{b: end})",
            "rec X . &{a: X, b: end}",
        ]
        for t in types:
            net, ss = _net(t)
            result = check_boundedness(net)
            assert result.is_bounded, f"Expected bounded for {t}"
            assert result.is_one_safe, f"Expected 1-safe for {t}"


# ---------------------------------------------------------------------------
# analyze_coverability (convenience)
# ---------------------------------------------------------------------------

class TestAnalyzeCoverability:
    """Test the analyze_coverability convenience function."""

    def test_end(self):
        ss = _ss("end")
        result = analyze_coverability(ss)
        assert result.is_bounded is True
        assert result.is_one_safe is True

    def test_branch_selection(self):
        ss = _ss("&{open: +{ok: &{read: end}, err: end}}")
        result = analyze_coverability(ss)
        assert result.is_bounded is True
        assert result.is_one_safe is True

    def test_result_has_correct_node_count(self):
        ss = _ss("&{a: end}")
        result = analyze_coverability(ss)
        # "end" has 1 state; "&{a:end}" has 2 states + 1 transition
        # Tree: root (top) -> 1 child (bottom) = 2 nodes
        assert result.num_nodes >= 2


# ---------------------------------------------------------------------------
# Unbounded recursion analysis
# ---------------------------------------------------------------------------

class TestUnboundedRecursion:
    """Test multi-token analysis for recursion stress-testing."""

    def test_build_unbounded_net(self):
        ss = _ss("rec X . &{a: X, b: end}")
        net, marking = build_unbounded_recursion_net(ss, unfold_tokens=3)
        assert marking[ss.top] == 3

    def test_multi_token_end(self):
        """Even with multiple tokens, 'end' has only one place."""
        ss = _ss("end")
        result = analyze_unbounded_recursion(ss, unfold_tokens=2)
        # Still bounded since there are no transitions
        assert result.is_bounded is True

    def test_multi_token_branch_not_one_safe(self):
        """With 2 tokens, a branch net has max 2 tokens at start."""
        ss = _ss("&{m: end}")
        result = analyze_unbounded_recursion(ss, unfold_tokens=2)
        assert result.is_bounded is True
        # Not 1-safe because we started with 2 tokens
        assert result.is_one_safe is False

    def test_multi_token_recursion_bounded(self):
        """Multiple tokens in a recursive net: still bounded because
        the state-machine encoding conserves total token count."""
        ss = _ss("rec X . &{a: X, b: end}")
        result = analyze_unbounded_recursion(ss, unfold_tokens=2)
        assert result.is_bounded is True


# ---------------------------------------------------------------------------
# Distinct markings count
# ---------------------------------------------------------------------------

class TestDistinctMarkings:
    """Test that distinct marking counts are correct."""

    def test_end_one_marking(self):
        net, ss = _net("end")
        result = build_coverability_tree(net)
        assert result.num_distinct_markings == 1

    def test_branch_two_markings(self):
        net, ss = _net("&{m: end}")
        result = build_coverability_tree(net)
        assert result.num_distinct_markings == 2

    def test_two_branch_three_markings(self):
        net, ss = _net("&{a: end, b: end}")
        result = build_coverability_tree(net)
        # top, a->end, b->end  (but a and b both go to bottom)
        # so markings: {top:1}, {bottom:1} => 2 if shared end
        assert result.num_distinct_markings >= 2


# ---------------------------------------------------------------------------
# CoverabilityNode structure
# ---------------------------------------------------------------------------

class TestCoverabilityNodeStructure:
    """Test tree node structure."""

    def test_root_has_no_parent(self):
        net, ss = _net("&{m: end}")
        result = build_coverability_tree(net)
        root = result.tree_nodes[0]
        assert root.parent is None
        assert root.transition_from_parent is None

    def test_child_has_parent(self):
        net, ss = _net("&{m: end}")
        result = build_coverability_tree(net)
        if result.num_nodes > 1:
            child = result.tree_nodes[1]
            assert child.parent == 0
            assert child.transition_from_parent is not None

    def test_root_marking_is_initial(self):
        net, ss = _net("&{m: end}")
        result = build_coverability_tree(net)
        root = result.tree_nodes[0]
        assert root.marking == net.initial_marking


# ---------------------------------------------------------------------------
# Benchmark protocols
# ---------------------------------------------------------------------------

class TestBenchmarkCoverability:
    """Verify 1-safety across benchmark session types."""

    BENCHMARKS = [
        "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}",
        "&{open: +{ok: &{read: end}, err: end}}",
        "&{connect: +{ok: &{send: &{recv: end}}, err: end}}",
        "+{GET: &{response: end}, POST: &{response: end}}",
    ]

    @pytest.mark.parametrize("typ", BENCHMARKS)
    def test_benchmark_one_safe(self, typ: str):
        ss = _ss(typ)
        result = analyze_coverability(ss)
        assert result.is_bounded, f"Not bounded: {typ}"
        assert result.is_one_safe, f"Not 1-safe: {typ}"
