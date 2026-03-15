"""Tests for reticulate.coverage — transition/state coverage analysis."""

from __future__ import annotations

import pytest

from reticulate.coverage import CoverageFrame, CoverageResult, compute_coverage, coverage_storyboard
from reticulate.parser import parse
from reticulate.statespace import StateSpace, build_statespace
from reticulate.testgen import (
    EnumerationResult,
    IncompletePrefix,
    Step,
    ValidPath,
    ViolationPoint,
    TestGenConfig,
    enumerate,
)
from reticulate.visualize import dot_source


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def ss(type_str: str) -> StateSpace:
    return build_statespace(parse(type_str))


# =========================================================================
# Coverage computation — trivial cases
# =========================================================================

class TestComputeCoverageEnd:
    def test_end_no_transitions_full_coverage(self):
        s = ss("end")
        cov = compute_coverage(s)
        assert cov.transition_coverage == 1.0
        assert cov.state_coverage == 1.0
        assert cov.uncovered_transitions == frozenset()

    def test_end_with_empty_paths(self):
        s = ss("end")
        cov = compute_coverage(s, paths=[ValidPath(steps=())])
        assert cov.transition_coverage == 1.0


# =========================================================================
# Coverage computation — chain (sequential)
# =========================================================================

class TestComputeCoverageChain:
    def test_chain_single_path_full_coverage(self):
        s = ss("&{a: &{b: end}}")
        # Path: top --a--> mid --b--> bottom
        top = s.top
        mid = [tgt for (src, lbl, tgt) in s.transitions if src == top and lbl == "a"][0]
        bot = s.bottom
        path = ValidPath(steps=(Step("a", mid), Step("b", bot)))
        cov = compute_coverage(s, paths=[path])
        assert cov.transition_coverage == 1.0
        assert cov.uncovered_transitions == frozenset()

    def test_chain_no_paths_zero_coverage(self):
        s = ss("&{a: &{b: end}}")
        cov = compute_coverage(s)
        assert cov.transition_coverage == 0.0
        assert len(cov.uncovered_transitions) == 2

    def test_chain_covered_transitions_correct(self):
        s = ss("&{a: &{b: end}}")
        top = s.top
        mid = [tgt for (src, lbl, tgt) in s.transitions if src == top and lbl == "a"][0]
        bot = s.bottom
        path = ValidPath(steps=(Step("a", mid), Step("b", bot)))
        cov = compute_coverage(s, paths=[path])
        assert (top, "a", mid) in cov.covered_transitions
        assert (mid, "b", bot) in cov.covered_transitions


# =========================================================================
# Coverage computation — branch (partial coverage)
# =========================================================================

class TestComputeCoverageBranch:
    def test_branch_one_path_partial(self):
        s = ss("&{m: end, n: end}")
        bot = s.bottom
        # Find the target of m
        m_tgt = [tgt for (src, lbl, tgt) in s.transitions if lbl == "m"][0]
        path = ValidPath(steps=(Step("m", m_tgt),))
        cov = compute_coverage(s, paths=[path])
        # m is covered, n is not → 50%
        assert cov.transition_coverage == pytest.approx(0.5)
        assert len(cov.uncovered_transitions) == 1

    def test_branch_both_paths_full(self):
        s = ss("&{m: end, n: end}")
        m_tgt = [tgt for (src, lbl, tgt) in s.transitions if lbl == "m"][0]
        n_tgt = [tgt for (src, lbl, tgt) in s.transitions if lbl == "n"][0]
        paths = [
            ValidPath(steps=(Step("m", m_tgt),)),
            ValidPath(steps=(Step("n", n_tgt),)),
        ]
        cov = compute_coverage(s, paths=paths)
        assert cov.transition_coverage == 1.0

    def test_uncovered_transitions_match_untaken(self):
        s = ss("&{m: end, n: end}")
        m_tgt = [tgt for (src, lbl, tgt) in s.transitions if lbl == "m"][0]
        path = ValidPath(steps=(Step("m", m_tgt),))
        cov = compute_coverage(s, paths=[path])
        uncov_labels = {lbl for (_, lbl, _) in cov.uncovered_transitions}
        assert "n" in uncov_labels
        assert "m" not in uncov_labels


# =========================================================================
# Coverage computation — violations contribute
# =========================================================================

class TestComputeCoverageWithViolations:
    def test_violations_contribute(self):
        s = ss("&{a: &{b: end}}")
        top = s.top
        mid = [tgt for (src, lbl, tgt) in s.transitions if src == top and lbl == "a"][0]
        # Violation at mid: prefix goes top→mid
        viol = ViolationPoint(
            state=mid,
            disabled_method="x",
            enabled_methods=frozenset({"b"}),
            prefix_path=(Step("a", mid),),
        )
        cov = compute_coverage(s, violations=[viol])
        assert (top, "a", mid) in cov.covered_transitions
        assert cov.transition_coverage == pytest.approx(0.5)

    def test_violations_plus_valid_paths(self):
        s = ss("&{a: &{b: end}}")
        top = s.top
        mid = [tgt for (src, lbl, tgt) in s.transitions if src == top and lbl == "a"][0]
        bot = s.bottom
        path = ValidPath(steps=(Step("a", mid), Step("b", bot)))
        viol = ViolationPoint(
            state=mid,
            disabled_method="x",
            enabled_methods=frozenset({"b"}),
            prefix_path=(Step("a", mid),),
        )
        cov = compute_coverage(s, paths=[path], violations=[viol])
        assert cov.transition_coverage == 1.0


# =========================================================================
# Coverage computation — incomplete prefixes
# =========================================================================

class TestComputeCoverageWithIncompletePrefixes:
    def test_incomplete_prefixes_contribute(self):
        s = ss("&{a: &{b: end}}")
        top = s.top
        mid = [tgt for (src, lbl, tgt) in s.transitions if src == top and lbl == "a"][0]
        ip = IncompletePrefix(
            steps=(Step("a", mid),),
            remaining_methods=frozenset({"b"}),
        )
        cov = compute_coverage(s, incomplete_prefixes=[ip])
        assert (top, "a", mid) in cov.covered_transitions


# =========================================================================
# Coverage computation — EnumerationResult
# =========================================================================

class TestComputeCoverageWithEnumerationResult:
    def test_from_enumeration_result(self):
        s = ss("&{m: end, n: end}")
        result = enumerate(s, TestGenConfig(class_name="T"))
        cov = compute_coverage(s, result=result)
        # enumerate should produce paths covering both branches
        assert cov.transition_coverage == 1.0

    def test_enumeration_result_overrides_individual_lists(self):
        s = ss("&{m: end, n: end}")
        result = enumerate(s, TestGenConfig(class_name="T"))
        # Pass empty paths but a full result — result should win
        cov = compute_coverage(s, paths=[], result=result)
        assert cov.transition_coverage == 1.0


# =========================================================================
# Coverage computation — recursive types
# =========================================================================

class TestComputeCoverageRecursive:
    def test_recursive_coverage(self):
        s = ss("rec X . &{next: X, done: end}")
        result = enumerate(s, TestGenConfig(class_name="T"))
        cov = compute_coverage(s, result=result)
        # All transitions should be covered (next loop + done exit)
        assert cov.transition_coverage == 1.0

    def test_recursive_partial_coverage(self):
        s = ss("rec X . &{next: X, done: end}")
        # Only take the done branch (skip next)
        done_tgt = [tgt for (src, lbl, tgt) in s.transitions if lbl == "done"][0]
        path = ValidPath(steps=(Step("done", done_tgt),))
        cov = compute_coverage(s, paths=[path])
        assert cov.transition_coverage < 1.0


# =========================================================================
# Coverage computation — parallel types
# =========================================================================

class TestComputeCoverageParallel:
    def test_parallel_full_coverage(self):
        s = ss("(&{a: end} || &{b: end})")
        result = enumerate(s, TestGenConfig(class_name="T"))
        cov = compute_coverage(s, result=result)
        assert cov.transition_coverage == 1.0

    def test_parallel_product_state_count(self):
        s = ss("(&{a: end} || &{b: end})")
        # 2x2 = 4 states, 4 transitions
        assert len(s.states) == 4
        result = enumerate(s, TestGenConfig(class_name="T"))
        cov = compute_coverage(s, result=result)
        assert len(cov.covered_transitions) == len(s.transitions)


# =========================================================================
# Coverage computation — state coverage
# =========================================================================

class TestComputeCoverageStates:
    def test_covered_states_include_top(self):
        s = ss("&{a: &{b: end}}")
        top = s.top
        mid = [tgt for (src, lbl, tgt) in s.transitions if src == top and lbl == "a"][0]
        path = ValidPath(steps=(Step("a", mid),))
        cov = compute_coverage(s, paths=[path])
        assert top in cov.covered_states

    def test_uncovered_states(self):
        s = ss("&{m: &{a: end}, n: &{b: end}}")
        # Only take the m branch
        top = s.top
        m_tgt = [tgt for (src, lbl, tgt) in s.transitions if src == top and lbl == "m"][0]
        a_tgt = [tgt for (src, lbl, tgt) in s.transitions if src == m_tgt and lbl == "a"][0]
        path = ValidPath(steps=(Step("m", m_tgt), Step("a", a_tgt)))
        cov = compute_coverage(s, paths=[path])
        # n branch state should be uncovered
        n_tgt = [tgt for (src, lbl, tgt) in s.transitions if src == top and lbl == "n"][0]
        assert n_tgt in cov.uncovered_states


# =========================================================================
# Coverage result properties
# =========================================================================

class TestCoverageResultProperties:
    def test_frozen(self):
        cov = CoverageResult(
            covered_transitions=frozenset(),
            uncovered_transitions=frozenset(),
            covered_states=frozenset(),
            uncovered_states=frozenset(),
            transition_coverage=1.0,
            state_coverage=1.0,
        )
        with pytest.raises(AttributeError):
            cov.transition_coverage = 0.5  # type: ignore[misc]

    def test_covered_plus_uncovered_equals_all(self):
        s = ss("&{m: end, n: end}")
        m_tgt = [tgt for (src, lbl, tgt) in s.transitions if lbl == "m"][0]
        path = ValidPath(steps=(Step("m", m_tgt),))
        cov = compute_coverage(s, paths=[path])
        assert cov.covered_transitions | cov.uncovered_transitions == frozenset(s.transitions)


# =========================================================================
# DOT source with coverage coloring
# =========================================================================

class TestDotSourceCoverage:
    def test_covered_edge_green(self):
        s = ss("&{m: end, n: end}")
        m_tgt = [tgt for (src, lbl, tgt) in s.transitions if lbl == "m"][0]
        path = ValidPath(steps=(Step("m", m_tgt),))
        cov = compute_coverage(s, paths=[path])
        dot = dot_source(s, coverage=cov)
        assert "#22c55e" in dot  # green for covered

    def test_uncovered_edge_red_dashed(self):
        s = ss("&{m: end, n: end}")
        m_tgt = [tgt for (src, lbl, tgt) in s.transitions if lbl == "m"][0]
        path = ValidPath(steps=(Step("m", m_tgt),))
        cov = compute_coverage(s, paths=[path])
        dot = dot_source(s, coverage=cov)
        assert "#ef4444" in dot  # red for uncovered
        assert "dashed" in dot

    def test_full_coverage_no_red(self):
        s = ss("&{m: end, n: end}")
        result = enumerate(s, TestGenConfig(class_name="T"))
        cov = compute_coverage(s, result=result)
        dot = dot_source(s, coverage=cov)
        assert "#ef4444" not in dot
        assert "#22c55e" in dot

    def test_coverage_with_no_edge_labels(self):
        s = ss("&{m: end, n: end}")
        m_tgt = [tgt for (src, lbl, tgt) in s.transitions if lbl == "m"][0]
        path = ValidPath(steps=(Step("m", m_tgt),))
        cov = compute_coverage(s, paths=[path])
        dot = dot_source(s, edge_labels=False, coverage=cov)
        assert "#22c55e" in dot  # coverage colors still applied

    def test_no_coverage_no_change(self):
        s = ss("&{m: end, n: end}")
        dot_without = dot_source(s)
        dot_with_none = dot_source(s, coverage=None)
        assert dot_without == dot_with_none

    def test_uncovered_state_fill(self):
        s = ss("&{m: &{a: end}, n: &{b: end}}")
        top = s.top
        m_tgt = [tgt for (src, lbl, tgt) in s.transitions if src == top and lbl == "m"][0]
        a_tgt = [tgt for (src, lbl, tgt) in s.transitions if src == m_tgt and lbl == "a"][0]
        path = ValidPath(steps=(Step("m", m_tgt), Step("a", a_tgt)))
        cov = compute_coverage(s, paths=[path])
        dot = dot_source(s, coverage=cov)
        assert "#fee2e2" in dot  # light red fill for uncovered states


# =========================================================================
# Coverage storyboard
# =========================================================================

class TestCoverageStoryboard:
    def test_end_type_no_frames(self):
        s = ss("end")
        result = enumerate(s, TestGenConfig(class_name="T"))
        frames = coverage_storyboard(s, result)
        assert frames == []

    def test_branch_has_frames(self):
        s = ss("&{m: end, n: end}")
        result = enumerate(s, TestGenConfig(class_name="T"))
        frames = coverage_storyboard(s, result)
        assert len(frames) > 0

    def test_frame_names_match_test_methods(self):
        s = ss("&{m: &{a: end}, n: &{b: end}}")
        result = enumerate(s, TestGenConfig(class_name="T"))
        frames = coverage_storyboard(s, result)
        names = [f.test_name for f in frames]
        # Valid paths come first
        valid_names = [n for n in names if n.startswith("validPath_")]
        assert len(valid_names) >= 2  # at least m→a and n→b paths
        # Violations present
        violation_names = [n for n in names if n.startswith("violation_")]
        assert len(violation_names) > 0

    def test_each_frame_is_independent(self):
        s = ss("&{m: &{a: end}, n: &{b: end}}")
        result = enumerate(s, TestGenConfig(class_name="T"))
        frames = coverage_storyboard(s, result)
        # Valid paths should each cover only their own transitions
        valid_frames = [f for f in frames if f.test_kind == "valid"]
        assert len(valid_frames) >= 2
        # Each valid path covers only part of the transitions
        for f in valid_frames:
            assert f.coverage.transition_coverage < 1.0

    def test_valid_path_covers_own_transitions(self):
        s = ss("&{m: &{a: end}, n: &{b: end}}")
        result = enumerate(s, TestGenConfig(class_name="T"))
        frames = coverage_storyboard(s, result)
        # Find the m_a path frame
        ma_frame = next(f for f in frames if "m_a" in f.test_name)
        covered_labels = {lbl for (_, lbl, _) in ma_frame.coverage.covered_transitions}
        assert "m" in covered_labels
        assert "a" in covered_labels
        assert "n" not in covered_labels
        assert "b" not in covered_labels

    def test_valid_before_violations_before_incomplete(self):
        s = ss("&{m: &{a: end}, n: &{b: end}}")
        result = enumerate(s, TestGenConfig(class_name="T"))
        frames = coverage_storyboard(s, result)
        # Check ordering: all valids, then all violations, then all incompletes
        valid_idx = [i for i in range(len(frames)) if frames[i].test_kind == "valid"]
        viol_idx = [i for i in range(len(frames)) if frames[i].test_kind == "violation"]
        inc_idx = [i for i in range(len(frames)) if frames[i].test_kind == "incomplete"]
        if valid_idx and viol_idx:
            assert max(valid_idx) < min(viol_idx)
        if viol_idx and inc_idx:
            assert max(viol_idx) < min(inc_idx)

    def test_frame_is_frozen(self):
        s = ss("&{m: end}")
        result = enumerate(s, TestGenConfig(class_name="T"))
        frames = coverage_storyboard(s, result)
        if frames:
            with pytest.raises(AttributeError):
                frames[0].test_name = "x"  # type: ignore[misc]

    def test_recursive_storyboard(self):
        s = ss("rec X . &{next: X, done: end}")
        result = enumerate(s, TestGenConfig(class_name="T"))
        frames = coverage_storyboard(s, result)
        assert len(frames) > 0
        # Each frame shows its own coverage, not cumulative
        for f in frames:
            assert 0.0 <= f.coverage.transition_coverage <= 1.0

    def test_parallel_storyboard(self):
        s = ss("(&{a: end} || &{b: end})")
        result = enumerate(s, TestGenConfig(class_name="T"))
        frames = coverage_storyboard(s, result)
        assert len(frames) > 0
        for f in frames:
            assert 0.0 <= f.coverage.transition_coverage <= 1.0


# =========================================================================
# Integration: enumerate → coverage → dot_source
# =========================================================================

class TestCoverageIntegration:
    @pytest.mark.parametrize("type_str", [
        "end",
        "&{a: end}",
        "&{m: end, n: end}",
        "+{ok: end, err: end}",
        "&{a: &{b: end}}",
        "rec X . &{next: X, done: end}",
        "(&{a: end} || &{b: end})",
    ])
    def test_enumerate_then_coverage_then_dot(self, type_str: str):
        s = ss(type_str)
        result = enumerate(s, TestGenConfig(class_name="T"))
        cov = compute_coverage(s, result=result)
        dot = dot_source(s, coverage=cov)
        assert "digraph" in dot
        # Full enumeration should give full coverage
        assert cov.transition_coverage == pytest.approx(1.0)
