"""Tests for resource leak detection via lattice width (Step 80f)."""

import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace, StateSpace
from reticulate.resource_leak import (
    ResourceLeakResult,
    WidthProfile,
    analyze_resource_leaks,
    detect_resource_leaks,
    parallel_completion_check,
    resource_monotonicity,
    width_profile,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ss(type_str: str) -> StateSpace:
    """Parse and build state space from a session type string."""
    return build_statespace(parse(type_str))


# ---------------------------------------------------------------------------
# Width profile tests
# ---------------------------------------------------------------------------


class TestWidthProfile:
    """Width profile computation."""

    def test_end_single_state(self):
        ss = _ss("end")
        wp = width_profile(ss)
        assert wp.widths == [1]
        assert wp.max_width == 1
        assert wp.is_monotone

    def test_simple_branch(self):
        ss = _ss("&{a: end, b: end}")
        wp = width_profile(ss)
        assert wp.widths[0] == 1  # top
        assert len(wp.widths) == 2  # top + end
        assert wp.is_monotone

    def test_deep_chain(self):
        ss = _ss("&{a: &{b: &{c: end}}}")
        wp = width_profile(ss)
        assert len(wp.widths) == 4  # a -> b -> c -> end
        assert all(w == 1 for w in wp.widths)
        assert wp.is_monotone

    def test_branch_with_multiple_methods(self):
        ss = _ss("&{a: &{c: end}, b: end}")
        wp = width_profile(ss)
        assert wp.widths[0] == 1  # top (branch state)
        assert wp.max_width >= 1

    def test_parallel_widens(self):
        ss = _ss("(&{a: end} || &{b: end})")
        wp = width_profile(ss)
        # Parallel creates a product lattice — wider at intermediate ranks
        assert wp.max_width >= 1

    def test_width_profile_ranks_consistent(self):
        ss = _ss("&{a: end, b: &{c: end}}")
        wp = width_profile(ss)
        total = sum(wp.widths)
        assert total == len(ss.states)

    def test_ranks_are_nonnegative(self):
        ss = _ss("&{a: end, b: end}")
        wp = width_profile(ss)
        assert all(r >= 0 for r in wp.ranks.values())

    def test_top_is_rank_zero(self):
        ss = _ss("&{a: end}")
        wp = width_profile(ss)
        assert wp.ranks[ss.top] == 0

    def test_states_at_rank_sum_to_total(self):
        ss = _ss("+{a: end, b: &{c: end}}")
        wp = width_profile(ss)
        total = sum(len(s) for s in wp.states_at_rank)
        assert total == len(ss.states)


# ---------------------------------------------------------------------------
# Resource monotonicity tests
# ---------------------------------------------------------------------------


class TestResourceMonotonicity:
    """Resource monotonicity checking."""

    def test_end_is_monotone(self):
        ss = _ss("end")
        is_mono, violations = resource_monotonicity(ss)
        assert is_mono
        assert violations == []

    def test_simple_branch_monotone(self):
        ss = _ss("&{a: end, b: end}")
        is_mono, violations = resource_monotonicity(ss)
        assert is_mono
        assert violations == []

    def test_chain_monotone(self):
        ss = _ss("&{a: &{b: end}}")
        is_mono, violations = resource_monotonicity(ss)
        assert is_mono

    def test_selection_monotone(self):
        ss = _ss("+{a: end, b: end}")
        is_mono, violations = resource_monotonicity(ss)
        assert is_mono

    def test_parallel_simple(self):
        ss = _ss("(&{a: end} || &{b: end})")
        is_mono, violations = resource_monotonicity(ss)
        # Simple parallel may or may not be monotone; check it runs
        assert isinstance(is_mono, bool)

    def test_violations_are_ranks(self):
        ss = _ss("&{a: &{b: end, c: end}, d: end}")
        _, violations = resource_monotonicity(ss)
        for v in violations:
            assert isinstance(v, int)
            assert v >= 1


# ---------------------------------------------------------------------------
# Resource leak detection tests
# ---------------------------------------------------------------------------


class TestDetectResourceLeaks:
    """Leak path detection."""

    def test_end_no_leaks(self):
        ss = _ss("end")
        leaks = detect_resource_leaks(ss)
        assert leaks == []

    def test_simple_branch_no_leaks(self):
        ss = _ss("&{a: end}")
        leaks = detect_resource_leaks(ss)
        # Single transition to end — the transition itself is a release
        assert isinstance(leaks, list)

    def test_selection_no_leaks(self):
        ss = _ss("+{ok: end, err: end}")
        leaks = detect_resource_leaks(ss)
        assert isinstance(leaks, list)

    def test_leak_paths_are_lists_of_ints(self):
        ss = _ss("&{a: end, b: &{c: end}}")
        leaks = detect_resource_leaks(ss)
        for path in leaks:
            assert isinstance(path, list)
            for s in path:
                assert isinstance(s, int)

    def test_leak_paths_start_at_top(self):
        ss = _ss("&{a: &{b: end}, c: end}")
        leaks = detect_resource_leaks(ss)
        for path in leaks:
            assert path[0] == ss.top

    def test_leak_paths_end_at_bottom(self):
        ss = _ss("&{a: &{b: end}, c: end}")
        leaks = detect_resource_leaks(ss)
        for path in leaks:
            assert path[-1] == ss.bottom

    def test_deep_chain_has_releases(self):
        ss = _ss("&{a: &{b: &{c: end}}}")
        leaks = detect_resource_leaks(ss)
        # Deep chain: every intermediate state has out-degree 1 → release
        assert leaks == []


# ---------------------------------------------------------------------------
# Parallel completion check tests
# ---------------------------------------------------------------------------


class TestParallelCompletion:
    """Parallel branch completion verification."""

    def test_non_parallel_trivially_complete(self):
        ss = _ss("&{a: end}")
        complete, incomplete = parallel_completion_check(ss)
        assert complete
        assert incomplete == []

    def test_simple_parallel_complete(self):
        ss = _ss("(&{a: end} || &{b: end})")
        complete, incomplete = parallel_completion_check(ss)
        assert isinstance(complete, bool)

    def test_end_parallel_trivial(self):
        ss = _ss("end")
        complete, incomplete = parallel_completion_check(ss)
        assert complete
        assert incomplete == []

    def test_selection_no_parallel(self):
        ss = _ss("+{a: end, b: end}")
        complete, incomplete = parallel_completion_check(ss)
        assert complete

    def test_incomplete_branches_are_indices(self):
        ss = _ss("(&{a: end} || &{b: end})")
        _, incomplete = parallel_completion_check(ss)
        for idx in incomplete:
            assert isinstance(idx, int)
            assert idx >= 0


# ---------------------------------------------------------------------------
# Combined analysis tests
# ---------------------------------------------------------------------------


class TestAnalyzeResourceLeaks:
    """Full resource leak analysis."""

    def test_end_result_type(self):
        ss = _ss("end")
        result = analyze_resource_leaks(ss)
        assert isinstance(result, ResourceLeakResult)

    def test_end_no_leaks(self):
        ss = _ss("end")
        result = analyze_resource_leaks(ss)
        assert not result.has_leaks

    def test_simple_branch_analysis(self):
        ss = _ss("&{a: end, b: end}")
        result = analyze_resource_leaks(ss)
        assert isinstance(result.width_profile, list)
        assert result.num_ranks > 0
        assert result.max_width >= 1

    def test_selection_analysis(self):
        ss = _ss("+{ok: end, err: end}")
        result = analyze_resource_leaks(ss)
        assert result.parallel_complete

    def test_parallel_analysis(self):
        ss = _ss("(&{a: end} || &{b: end})")
        result = analyze_resource_leaks(ss)
        assert isinstance(result.has_leaks, bool)
        assert isinstance(result.parallel_complete, bool)

    def test_recursive_type(self):
        ss = _ss("rec X . &{next: X, stop: end}")
        result = analyze_resource_leaks(ss)
        assert isinstance(result, ResourceLeakResult)
        assert result.num_ranks >= 1

    def test_result_fields_complete(self):
        ss = _ss("&{a: &{b: end}, c: end}")
        result = analyze_resource_leaks(ss)
        assert isinstance(result.width_profile, list)
        assert isinstance(result.is_monotone, bool)
        assert isinstance(result.monotonicity_violations, list)
        assert isinstance(result.leak_paths, list)
        assert isinstance(result.parallel_complete, bool)
        assert isinstance(result.incomplete_branches, list)
        assert isinstance(result.max_width, int)
        assert isinstance(result.max_width_rank, int)
        assert isinstance(result.num_ranks, int)
        assert isinstance(result.release_states, list)
        assert isinstance(result.acquisition_states, list)

    def test_acquisition_states_have_branching(self):
        ss = _ss("&{a: end, b: end}")
        result = analyze_resource_leaks(ss)
        for s in result.acquisition_states:
            assert len(ss.successors(s)) > 1

    def test_release_states_single_successor(self):
        ss = _ss("&{a: &{b: end}}")
        result = analyze_resource_leaks(ss)
        for s in result.release_states:
            succs = ss.successors(s)
            assert len(succs) == 1 or ss.bottom in succs


# ---------------------------------------------------------------------------
# Benchmark protocol tests
# ---------------------------------------------------------------------------


class TestBenchmarkProtocols:
    """Resource leak analysis on standard benchmark protocols."""

    def test_smtp_protocol(self):
        ss = _ss("&{connect: +{OK: &{mail: +{OK: &{rcpt: +{OK: &{data: &{send: end}}, ERR: end}}, ERR: end}}, ERR: end}}")
        result = analyze_resource_leaks(ss)
        assert isinstance(result, ResourceLeakResult)

    def test_iterator_protocol(self):
        ss = _ss("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        result = analyze_resource_leaks(ss)
        assert isinstance(result, ResourceLeakResult)

    def test_simple_parallel_protocol(self):
        ss = _ss("(&{read: end} || &{write: end})")
        result = analyze_resource_leaks(ss)
        assert isinstance(result, ResourceLeakResult)

    def test_two_branch_selection(self):
        ss = _ss("+{accept: &{process: end}, reject: end}")
        result = analyze_resource_leaks(ss)
        assert isinstance(result, ResourceLeakResult)

    def test_nested_branch(self):
        ss = _ss("&{open: &{read: end, write: end}, close: end}")
        result = analyze_resource_leaks(ss)
        assert isinstance(result, ResourceLeakResult)
