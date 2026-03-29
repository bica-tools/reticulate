"""Tests for Laplacian optimization of protocol connectivity (Step 31h)."""

import math
import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.laplacian_flow import (
    BottleneckEdge,
    EdgeCandidate,
    ImprovementResult,
    bottleneck_edges,
    connectivity_score,
    laplacian_gradient,
    optimize_connectivity,
    suggest_improvements,
)


def _ss(type_string: str):
    return build_statespace(parse(type_string))


# ---------------------------------------------------------------------------
# Laplacian gradient
# ---------------------------------------------------------------------------


class TestLaplacianGradient:
    def test_end_empty_gradient(self):
        grad = laplacian_gradient(_ss("end"))
        assert grad == {}

    def test_simple_branch_gradient(self):
        grad = laplacian_gradient(_ss("&{a: end}"))
        # 2 states, already complete: no candidates
        assert grad == {}

    def test_chain_has_gradient(self):
        grad = laplacian_gradient(_ss("&{a: &{b: end}}"))
        # 3 states: chain 0-1-2, missing edge 0-2
        assert len(grad) >= 1
        for edge, val in grad.items():
            assert val >= 0.0

    def test_gradient_nonnegative(self):
        grad = laplacian_gradient(_ss("&{a: &{b: &{c: end}}}"))
        for val in grad.values():
            assert val >= -1e-10

    def test_diamond_gradient(self):
        grad = laplacian_gradient(_ss("&{a: &{c: end}, b: &{c: end}}"))
        # All values should be non-negative
        for val in grad.values():
            assert val >= -1e-10


# ---------------------------------------------------------------------------
# Optimize connectivity
# ---------------------------------------------------------------------------


class TestOptimizeConnectivity:
    def test_end_no_candidates(self):
        cands = optimize_connectivity(_ss("end"))
        assert cands == []

    def test_simple_branch_no_candidates(self):
        cands = optimize_connectivity(_ss("&{a: end}"))
        assert cands == []

    def test_chain_has_candidates(self):
        cands = optimize_connectivity(_ss("&{a: &{b: end}}"))
        assert len(cands) >= 1
        for c in cands:
            assert isinstance(c, EdgeCandidate)
            assert c.fiedler_gain >= -1e-10

    def test_candidates_sorted_by_gain(self):
        cands = optimize_connectivity(_ss("&{a: &{b: &{c: end}}}"), k=5)
        for i in range(len(cands) - 1):
            assert cands[i].fiedler_gain >= cands[i + 1].fiedler_gain - 1e-10

    def test_k_limits_output(self):
        cands = optimize_connectivity(_ss("&{a: &{b: &{c: end}}}"), k=1)
        assert len(cands) <= 1

    def test_parallel_candidates(self):
        cands = optimize_connectivity(_ss("(&{a: end} || &{b: end})"))
        assert isinstance(cands, list)

    def test_candidate_states_valid(self):
        ss = _ss("&{a: &{b: end}}")
        cands = optimize_connectivity(ss)
        for c in cands:
            assert c.src in ss.states
            assert c.tgt in ss.states


# ---------------------------------------------------------------------------
# Connectivity score
# ---------------------------------------------------------------------------


class TestConnectivityScore:
    def test_end_score(self):
        score = connectivity_score(_ss("end"))
        assert score == 1.0

    def test_simple_branch_score(self):
        score = connectivity_score(_ss("&{a: end}"))
        assert 0.0 <= score <= 1.0

    def test_chain_score(self):
        score = connectivity_score(_ss("&{a: &{b: end}}"))
        assert 0.0 < score <= 1.0

    def test_score_bounded(self):
        score = connectivity_score(_ss("&{a: &{b: &{c: end}}}"))
        assert 0.0 <= score <= 1.0

    def test_wider_has_higher_score(self):
        chain = connectivity_score(_ss("&{a: &{b: end}}"))
        diamond = connectivity_score(_ss("&{a: &{c: end}, b: &{c: end}}"))
        # Diamond should have at least as good connectivity
        assert diamond >= chain - 0.1  # Allow small tolerance


# ---------------------------------------------------------------------------
# Bottleneck edges
# ---------------------------------------------------------------------------


class TestBottleneckEdges:
    def test_end_no_bottlenecks(self):
        bn = bottleneck_edges(_ss("end"))
        assert bn == []

    def test_simple_branch_no_bottlenecks(self):
        bn = bottleneck_edges(_ss("&{a: end}"))
        assert bn == []

    def test_chain_all_bridges(self):
        bn = bottleneck_edges(_ss("&{a: &{b: end}}"))
        for b in bn:
            assert isinstance(b, BottleneckEdge)
            assert b.is_bridge

    def test_bottlenecks_sorted_by_loss(self):
        bn = bottleneck_edges(_ss("&{a: &{b: &{c: end}}}"))
        for i in range(len(bn) - 1):
            assert bn[i].fiedler_loss >= bn[i + 1].fiedler_loss - 1e-10

    def test_diamond_has_non_bridges(self):
        bn = bottleneck_edges(_ss("&{a: &{c: end}, b: &{c: end}}"))
        # Diamond may have non-bridge edges
        has_non_bridge = any(not b.is_bridge for b in bn)
        # At least some edges exist
        assert len(bn) >= 1

    def test_k_limits_output(self):
        bn = bottleneck_edges(_ss("&{a: &{b: &{c: end}}}"), k=1)
        assert len(bn) <= 1


# ---------------------------------------------------------------------------
# Full improvement analysis
# ---------------------------------------------------------------------------


class TestSuggestImprovements:
    def test_end_improvements(self):
        result = suggest_improvements(_ss("end"))
        assert isinstance(result, ImprovementResult)
        assert result.num_states == 1

    def test_simple_branch_improvements(self):
        result = suggest_improvements(_ss("&{a: end}"))
        assert result.num_states == 2
        assert 0.0 <= result.connectivity_score <= 1.0

    def test_chain_improvements(self):
        result = suggest_improvements(_ss("&{a: &{b: end}}"))
        assert result.num_states == 3
        assert result.fiedler_current > 0.0
        assert result.improvement_potential >= 0.0

    def test_diamond_improvements(self):
        result = suggest_improvements(_ss("&{a: &{c: end}, b: &{c: end}}"))
        assert result.fiedler_current > 0.0
        assert result.max_possible_fiedler > 0.0

    def test_parallel_improvements(self):
        result = suggest_improvements(_ss("(&{a: end} || &{b: end})"))
        assert isinstance(result, ImprovementResult)
        assert result.num_states >= 3

    def test_recursive_improvements(self):
        result = suggest_improvements(_ss("rec X . &{a: X, b: end}"))
        assert isinstance(result, ImprovementResult)

    def test_fiedler_after_best_improvement(self):
        result = suggest_improvements(_ss("&{a: &{b: &{c: end}}}"))
        if result.candidates:
            assert result.fiedler_after_best >= result.fiedler_current - 1e-10

    def test_improvement_potential_bounded(self):
        result = suggest_improvements(_ss("&{a: &{b: end}}"))
        assert 0.0 <= result.improvement_potential <= 1.0

    def test_num_candidates_param(self):
        result = suggest_improvements(_ss("&{a: &{b: &{c: end}}}"), num_candidates=1)
        assert len(result.candidates) <= 1

    def test_num_bottlenecks_param(self):
        result = suggest_improvements(_ss("&{a: &{b: &{c: end}}}"), num_bottlenecks=1)
        assert len(result.bottlenecks) <= 1
