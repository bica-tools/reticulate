"""Tests for failure semantics derived from lattice structure (Step 28)."""

import pytest

from reticulate.parser import parse
from reticulate.statespace import StateSpace, build_statespace
from reticulate.failures import (
    FailureResult,
    RefinementResult,
    acceptance_set,
    analyze_failures,
    check_failure_refinement,
    compute_all_acceptances,
    compute_all_refusals,
    compute_refusals,
    failure_pairs,
    lattice_determines_failures,
    may_testing,
    must_testing,
)


# ---------------------------------------------------------------------------
# Refusal computation
# ---------------------------------------------------------------------------


class TestComputeRefusals:
    """Tests for compute_refusals and compute_all_refusals."""

    def test_end_state_refuses_everything(self):
        """The bottom (end) state has no enabled transitions, so refuses all."""
        ss = build_statespace(parse("&{a: end}"))
        refusals = compute_refusals(ss, ss.bottom)
        assert "a" in refusals

    def test_top_state_no_refusals_single_branch(self):
        """A single-branch state accepts its only label."""
        ss = build_statespace(parse("&{a: end}"))
        refusals = compute_refusals(ss, ss.top)
        assert "a" not in refusals

    def test_two_branch_top_accepts_both(self):
        """A two-branch state accepts both labels."""
        ss = build_statespace(parse("&{a: end, b: end}"))
        refusals = compute_refusals(ss, ss.top)
        assert "a" not in refusals
        assert "b" not in refusals

    def test_refusals_all_states(self):
        """compute_all_refusals returns a dict covering all states."""
        ss = build_statespace(parse("&{a: &{b: end}}"))
        all_ref = compute_all_refusals(ss)
        assert set(all_ref.keys()) == ss.states

    def test_refusals_intermediate_state(self):
        """An intermediate state refuses labels not enabled there."""
        ss = build_statespace(parse("&{a: &{b: end}}"))
        # After 'a', we should be at a state that accepts 'b' but refuses 'a'
        for s, lbl, t in ss.transitions:
            if s == ss.top and lbl == "a":
                mid = t
                break
        refusals = compute_refusals(ss, mid)
        assert "a" in refusals
        assert "b" not in refusals

    def test_selection_refusals(self):
        """Selection states also have refusal sets."""
        ss = build_statespace(parse("+{a: end, b: end}"))
        refusals = compute_refusals(ss, ss.top)
        # Selection enables both labels
        assert "a" not in refusals
        assert "b" not in refusals


# ---------------------------------------------------------------------------
# Acceptance sets
# ---------------------------------------------------------------------------


class TestAcceptanceSet:
    """Tests for acceptance_set and compute_all_acceptances."""

    def test_end_accepts_nothing(self):
        """The end state has no enabled transitions."""
        ss = build_statespace(parse("&{a: end}"))
        acc = acceptance_set(ss, ss.bottom)
        assert len(acc) == 0

    def test_branch_accepts_all_labels(self):
        """A branch state accepts all its labels."""
        ss = build_statespace(parse("&{a: end, b: end, c: end}"))
        acc = acceptance_set(ss, ss.top)
        assert acc == frozenset({"a", "b", "c"})

    def test_all_acceptances_covers_states(self):
        """compute_all_acceptances returns entries for all states."""
        ss = build_statespace(parse("&{a: &{b: end}}"))
        all_acc = compute_all_acceptances(ss)
        assert set(all_acc.keys()) == ss.states

    def test_acceptance_complement_of_refusal(self):
        """Acceptance set is the complement of refusal set."""
        ss = build_statespace(parse("&{a: end, b: end}"))
        for state in ss.states:
            acc = acceptance_set(ss, state)
            ref = compute_refusals(ss, state)
            all_labels = frozenset(lbl for _, lbl, _ in ss.transitions)
            assert acc | ref == all_labels
            assert acc & ref == frozenset()


# ---------------------------------------------------------------------------
# Trace execution and testing
# ---------------------------------------------------------------------------


class TestMustMayTesting:
    """Tests for must_testing and may_testing."""

    def test_empty_trace_must(self):
        """Empty trace leads to top state; must accepts what top accepts."""
        ss = build_statespace(parse("&{a: end, b: end}"))
        result = must_testing(ss, ())
        assert "a" in result
        assert "b" in result

    def test_empty_trace_may(self):
        """Empty trace may-accepts what top accepts."""
        ss = build_statespace(parse("&{a: end, b: end}"))
        result = may_testing(ss, ())
        assert "a" in result
        assert "b" in result

    def test_after_trace_must(self):
        """After executing a single-step trace, check must-acceptance."""
        ss = build_statespace(parse("&{a: &{b: end}}"))
        result = must_testing(ss, ("a",))
        assert "b" in result
        assert "a" not in result

    def test_after_trace_may(self):
        """After a single-step trace, may-acceptance equals must for deterministic."""
        ss = build_statespace(parse("&{a: &{b: end}}"))
        must = must_testing(ss, ("a",))
        may = may_testing(ss, ("a",))
        assert must == may  # deterministic: must == may

    def test_invalid_trace_returns_empty(self):
        """A trace that cannot be executed returns empty set."""
        ss = build_statespace(parse("&{a: end}"))
        result = must_testing(ss, ("b",))
        assert result == frozenset()

    def test_multi_step_trace(self):
        """Execute a multi-step trace."""
        ss = build_statespace(parse("&{a: &{b: &{c: end}}}"))
        result = must_testing(ss, ("a", "b"))
        assert "c" in result

    def test_must_subset_of_may(self):
        """must-acceptance is always a subset of may-acceptance."""
        ss = build_statespace(parse("&{a: end, b: &{c: end}}"))
        for trace in [(), ("a",), ("b",)]:
            must = must_testing(ss, trace)
            may = may_testing(ss, trace)
            assert must <= may


# ---------------------------------------------------------------------------
# Failure pairs
# ---------------------------------------------------------------------------


class TestFailurePairs:
    """Tests for failure_pairs enumeration."""

    def test_simple_branch_failures(self):
        """Simple branch: empty trace + end-state refusals."""
        ss = build_statespace(parse("&{a: end}"))
        fp = failure_pairs(ss)
        assert len(fp) > 0
        # Must include empty trace with no refusal of 'a'
        traces = {t for t, _ in fp}
        assert () in traces

    def test_failure_pairs_contain_end_refusals(self):
        """After reaching end, everything is refused."""
        ss = build_statespace(parse("&{a: end}"))
        fp = failure_pairs(ss)
        # Find the failure pair for trace ("a",)
        end_failures = {(t, r) for t, r in fp if t == ("a",)}
        assert len(end_failures) >= 1
        # The end state refuses 'a'
        for _, refusal in end_failures:
            assert "a" in refusal

    def test_returns_frozenset(self):
        """failure_pairs returns a frozenset."""
        ss = build_statespace(parse("&{a: end}"))
        fp = failure_pairs(ss)
        assert isinstance(fp, frozenset)

    def test_deeper_type_more_failures(self):
        """Deeper types generate more failure pairs."""
        ss1 = build_statespace(parse("&{a: end}"))
        ss2 = build_statespace(parse("&{a: &{b: end}}"))
        fp1 = failure_pairs(ss1)
        fp2 = failure_pairs(ss2)
        assert len(fp2) >= len(fp1)


# ---------------------------------------------------------------------------
# Failure refinement
# ---------------------------------------------------------------------------


class TestFailureRefinement:
    """Tests for check_failure_refinement."""

    def test_self_refinement(self):
        """Every process failure-refines itself."""
        ss = build_statespace(parse("&{a: end, b: end}"))
        result = check_failure_refinement(ss, ss)
        assert result.refines

    def test_refinement_result_type(self):
        """check_failure_refinement returns RefinementResult."""
        ss = build_statespace(parse("&{a: end}"))
        result = check_failure_refinement(ss, ss)
        assert isinstance(result, RefinementResult)

    def test_more_branches_refines_fewer(self):
        """A type with more branches (fewer refusals) refines one with fewer branches."""
        ss1 = build_statespace(parse("&{a: end, b: end}"))
        ss2 = build_statespace(parse("&{a: end}"))
        # ss1 has {a,b} enabled at top, ss2 has {a}
        # ss1 refuses less -> ss1's failures should be contained in ss2's
        result = check_failure_refinement(ss1, ss2)
        assert result.refines


# ---------------------------------------------------------------------------
# Lattice determines failures
# ---------------------------------------------------------------------------


class TestLatticeDeterminesFailures:
    """Tests for lattice_determines_failures."""

    def test_simple_branch_lattice_determines(self):
        """Simple branch type: lattice determines failures."""
        ss = build_statespace(parse("&{a: end}"))
        assert lattice_determines_failures(ss)

    def test_two_branch_lattice_determines(self):
        """Two-branch type: lattice determines failures."""
        ss = build_statespace(parse("&{a: end, b: end}"))
        assert lattice_determines_failures(ss)

    def test_nested_branch_lattice_determines(self):
        """Nested branches: lattice determines failures."""
        ss = build_statespace(parse("&{a: &{b: end}, c: end}"))
        assert lattice_determines_failures(ss)

    def test_selection_lattice_determines(self):
        """Selection type: lattice determines failures."""
        ss = build_statespace(parse("+{a: end, b: end}"))
        assert lattice_determines_failures(ss)

    def test_parallel_lattice_determines(self):
        """Parallel type: lattice determines failures."""
        ss = build_statespace(parse("(&{a: end} || &{b: end})"))
        assert lattice_determines_failures(ss)

    def test_recursive_lattice_determines(self):
        """Recursive type: lattice determines failures."""
        ss = build_statespace(parse("rec X . &{a: X, b: end}"))
        assert lattice_determines_failures(ss)


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------


class TestAnalyzeFailures:
    """Tests for the high-level analyze_failures function."""

    def test_returns_failure_result(self):
        """analyze_failures returns a FailureResult."""
        ss = build_statespace(parse("&{a: end}"))
        result = analyze_failures(ss)
        assert isinstance(result, FailureResult)

    def test_deterministic_flag(self):
        """Simple branch types are deterministic."""
        ss = build_statespace(parse("&{a: end, b: end}"))
        result = analyze_failures(ss)
        assert result.deterministic

    def test_refusal_map_populated(self):
        """Refusal map covers all states."""
        ss = build_statespace(parse("&{a: end, b: end}"))
        result = analyze_failures(ss)
        assert set(result.refusal_map.keys()) == ss.states

    def test_acceptance_map_populated(self):
        """Acceptance map covers all states."""
        ss = build_statespace(parse("&{a: end, b: end}"))
        result = analyze_failures(ss)
        assert set(result.acceptance_map.keys()) == ss.states

    def test_lattice_details_string(self):
        """Lattice details is a non-empty string."""
        ss = build_statespace(parse("&{a: end}"))
        result = analyze_failures(ss)
        assert len(result.lattice_details) > 0


# ---------------------------------------------------------------------------
# Benchmark protocols
# ---------------------------------------------------------------------------


class TestBenchmarkFailures:
    """Failure analysis on benchmark protocols."""

    def test_iterator_lattice_determines(self):
        """Java Iterator: lattice determines failures."""
        ss = build_statespace(parse(
            "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"
        ))
        assert lattice_determines_failures(ss)

    def test_file_object_lattice_determines(self):
        """File Object: lattice determines failures."""
        ss = build_statespace(parse(
            "&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}"
        ))
        assert lattice_determines_failures(ss)

    def test_iterator_deterministic(self):
        """Java Iterator is deterministic."""
        ss = build_statespace(parse(
            "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"
        ))
        result = analyze_failures(ss)
        assert result.deterministic

    def test_parallel_benchmark(self):
        """Parallel benchmark: lattice determines failures."""
        ss = build_statespace(parse("(&{read: end} || &{write: end})"))
        result = analyze_failures(ss)
        assert result.lattice_determines
        assert result.deterministic
