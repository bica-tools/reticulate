"""Tests for Petri net benchmark analysis (Step 25).

Verifies that all benchmark protocols produce valid, 1-safe Petri nets
with reachability graphs isomorphic to their original state spaces.
"""

from __future__ import annotations

import pytest

from reticulate.parser import parse
from reticulate.petri import session_type_to_petri_net
from reticulate.petri_benchmarks import (
    PetriBenchmarkResult,
    PetriBenchmarkSummary,
    analyze_all_benchmarks,
    analyze_all_multiparty_benchmarks,
    analyze_benchmark,
    format_petri_benchmark_table,
    format_summary,
    petri_benchmark_summary,
)
from reticulate.statespace import build_statespace
from tests.benchmarks.protocols import BENCHMARKS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def binary_results() -> list[PetriBenchmarkResult]:
    """Compute Petri net results for all binary benchmarks (cached)."""
    return analyze_all_benchmarks()


@pytest.fixture(scope="module")
def multiparty_results() -> list[PetriBenchmarkResult]:
    """Compute Petri net results for all multiparty benchmarks (cached)."""
    return analyze_all_multiparty_benchmarks()


@pytest.fixture(scope="module")
def all_results(
    binary_results: list[PetriBenchmarkResult],
    multiparty_results: list[PetriBenchmarkResult],
) -> list[PetriBenchmarkResult]:
    """Combined binary + multiparty results."""
    return binary_results + multiparty_results


@pytest.fixture(scope="module")
def summary(all_results: list[PetriBenchmarkResult]) -> PetriBenchmarkSummary:
    """Aggregate summary across all benchmarks."""
    return petri_benchmark_summary(all_results)


# ---------------------------------------------------------------------------
# Test: all benchmarks produce valid Petri nets
# ---------------------------------------------------------------------------

class TestAllBenchmarksValid:
    """Every benchmark must produce a valid Petri net."""

    def test_binary_benchmarks_count(
        self, binary_results: list[PetriBenchmarkResult]
    ) -> None:
        """We have results for all binary benchmarks."""
        assert len(binary_results) == len(BENCHMARKS)

    def test_multiparty_benchmarks_count(
        self, multiparty_results: list[PetriBenchmarkResult]
    ) -> None:
        """We have results for all multiparty benchmarks."""
        from tests.benchmarks.multiparty_protocols import MULTIPARTY_BENCHMARKS
        assert len(multiparty_results) == len(MULTIPARTY_BENCHMARKS)

    def test_all_have_places(
        self, all_results: list[PetriBenchmarkResult]
    ) -> None:
        """Every Petri net must have at least 2 places (top + bottom)."""
        for r in all_results:
            assert r.places >= 2, f"{r.name}: expected >= 2 places, got {r.places}"

    def test_all_have_transitions(
        self, all_results: list[PetriBenchmarkResult]
    ) -> None:
        """Every Petri net must have at least 1 transition."""
        for r in all_results:
            assert r.transitions >= 1, (
                f"{r.name}: expected >= 1 transition, got {r.transitions}"
            )


# ---------------------------------------------------------------------------
# Test: 1-safety guarantee
# ---------------------------------------------------------------------------

class TestOneSafety:
    """All benchmark Petri nets must be 1-safe."""

    def test_all_binary_one_safe(
        self, binary_results: list[PetriBenchmarkResult]
    ) -> None:
        """Every binary benchmark net is 1-safe."""
        for r in binary_results:
            assert r.is_one_safe, f"{r.name} is not 1-safe"

    def test_all_multiparty_one_safe(
        self, multiparty_results: list[PetriBenchmarkResult]
    ) -> None:
        """Every multiparty benchmark net is 1-safe."""
        for r in multiparty_results:
            assert r.is_one_safe, f"{r.name} is not 1-safe"


# ---------------------------------------------------------------------------
# Test: reachability isomorphism
# ---------------------------------------------------------------------------

class TestReachabilityIsomorphism:
    """Reachability graph must be isomorphic to original state space."""

    def test_all_binary_isomorphic(
        self, binary_results: list[PetriBenchmarkResult]
    ) -> None:
        """Every binary benchmark has reachability isomorphism."""
        for r in binary_results:
            assert r.reachability_isomorphic, (
                f"{r.name}: reachability graph not isomorphic"
            )

    def test_all_multiparty_isomorphic(
        self, multiparty_results: list[PetriBenchmarkResult]
    ) -> None:
        """Every multiparty benchmark has reachability isomorphism."""
        for r in multiparty_results:
            assert r.reachability_isomorphic, (
                f"{r.name}: reachability graph not isomorphic"
            )

    def test_reachable_markings_equal_states(
        self, all_results: list[PetriBenchmarkResult]
    ) -> None:
        """Number of reachable markings equals number of states."""
        for r in all_results:
            assert r.reachable_markings == r.ss_states, (
                f"{r.name}: {r.reachable_markings} markings != {r.ss_states} states"
            )


# ---------------------------------------------------------------------------
# Test: statistics consistency
# ---------------------------------------------------------------------------

class TestStatisticsConsistency:
    """Petri net metrics must be consistent with state space metrics."""

    def test_places_equal_states(
        self, all_results: list[PetriBenchmarkResult]
    ) -> None:
        """In state-machine encoding, #places == #states."""
        for r in all_results:
            assert r.places == r.ss_states, (
                f"{r.name}: {r.places} places != {r.ss_states} states"
            )

    def test_transitions_equal_edges(
        self, all_results: list[PetriBenchmarkResult]
    ) -> None:
        """In state-machine encoding, #transitions == #edges."""
        for r in all_results:
            assert r.transitions == r.ss_transitions, (
                f"{r.name}: {r.transitions} transitions != "
                f"{r.ss_transitions} edges"
            )

    def test_summary_totals(
        self,
        all_results: list[PetriBenchmarkResult],
        summary: PetriBenchmarkSummary,
    ) -> None:
        """Summary totals match sum of individual results."""
        assert summary.total_places == sum(r.places for r in all_results)
        assert summary.total_transitions == sum(
            r.transitions for r in all_results
        )

    def test_summary_counts(
        self,
        all_results: list[PetriBenchmarkResult],
        summary: PetriBenchmarkSummary,
    ) -> None:
        """Summary counts match individual results."""
        assert summary.total_benchmarks == len(all_results)
        assert summary.num_occurrence_nets == sum(
            1 for r in all_results if r.is_occurrence_net
        )
        assert summary.num_with_recursion == sum(
            1 for r in all_results if r.has_recursion
        )


# ---------------------------------------------------------------------------
# Test: free-choice property
# ---------------------------------------------------------------------------

class TestFreeChoice:
    """Free-choice property in state-machine encoding."""

    def test_all_free_choice(
        self, all_results: list[PetriBenchmarkResult]
    ) -> None:
        """State-machine encoding always yields free-choice nets."""
        for r in all_results:
            assert r.is_free_choice, f"{r.name} is not free-choice"


# ---------------------------------------------------------------------------
# Test: occurrence net vs recursion
# ---------------------------------------------------------------------------

class TestOccurrenceNet:
    """Occurrence net property corresponds to absence of recursion."""

    def test_occurrence_implies_no_recursion(
        self, all_results: list[PetriBenchmarkResult]
    ) -> None:
        """Occurrence net iff no recursion (acyclic)."""
        for r in all_results:
            assert r.is_occurrence_net != r.has_recursion, (
                f"{r.name}: occurrence_net={r.is_occurrence_net} "
                f"but has_recursion={r.has_recursion}"
            )

    def test_some_are_occurrence_nets(
        self, all_results: list[PetriBenchmarkResult]
    ) -> None:
        """At least some benchmarks are occurrence nets (acyclic)."""
        count = sum(1 for r in all_results if r.is_occurrence_net)
        assert count > 0, "No occurrence nets found"

    def test_some_have_recursion(
        self, all_results: list[PetriBenchmarkResult]
    ) -> None:
        """At least some benchmarks have recursion (cyclic)."""
        count = sum(1 for r in all_results if r.has_recursion)
        assert count > 0, "No recursive benchmarks found"


# ---------------------------------------------------------------------------
# Test: conflict places
# ---------------------------------------------------------------------------

class TestConflictPlaces:
    """Conflict places correspond to branching/selection points."""

    def test_some_have_conflicts(
        self, all_results: list[PetriBenchmarkResult]
    ) -> None:
        """At least some benchmarks have conflict places."""
        count = sum(1 for r in all_results if r.num_conflict_places > 0)
        assert count > 0, "No benchmarks with conflict places"

    def test_linear_protocols_no_conflicts(
        self, binary_results: list[PetriBenchmarkResult]
    ) -> None:
        """Protocols without branching should have no conflict places."""
        # mRNA Lifecycle is purely linear (no branching)
        for r in binary_results:
            if r.name == "mRNA Lifecycle":
                assert r.num_conflict_places == 0, (
                    f"mRNA Lifecycle should have no conflicts, got {r.num_conflict_places}"
                )
                break


# ---------------------------------------------------------------------------
# Test: individual benchmarks (spot checks)
# ---------------------------------------------------------------------------

class TestIndividualBenchmarks:
    """Spot-check specific well-known benchmarks."""

    def test_java_iterator(self) -> None:
        """Java Iterator: 4 states, 4 transitions, recursive."""
        r = analyze_benchmark(
            "Java Iterator",
            "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}",
        )
        assert r.places == 4
        assert r.transitions == 4
        assert r.is_one_safe
        assert r.reachability_isomorphic
        assert r.has_recursion  # recursive type

    def test_two_buyer(self) -> None:
        """Two-Buyer: uses parallel, occurrence net."""
        r = analyze_benchmark(
            "Two-Buyer",
            "&{lookup: &{getPrice: (&{proposeA: end} || "
            "&{proposeB: +{ACCEPT: &{pay: end}, REJECT: end}})}}",
            uses_parallel=True,
        )
        assert r.is_one_safe
        assert r.reachability_isomorphic
        assert r.is_occurrence_net  # no recursion

    def test_end_only(self) -> None:
        """Minimal session type: just 'end'."""
        r = analyze_benchmark("End", "end")
        assert r.places == 1
        assert r.transitions == 0
        assert r.is_one_safe
        assert r.reachable_markings == 1

    def test_single_branch(self) -> None:
        """Single branch: &{m: end}."""
        r = analyze_benchmark("Single", "&{m: end}")
        assert r.places == 2
        assert r.transitions == 1
        assert r.is_one_safe
        assert r.reachability_isomorphic


# ---------------------------------------------------------------------------
# Test: formatting
# ---------------------------------------------------------------------------

class TestFormatting:
    """Table and summary formatting produce valid output."""

    def test_table_has_header(
        self, binary_results: list[PetriBenchmarkResult]
    ) -> None:
        """Table output contains a header line."""
        table = format_petri_benchmark_table(binary_results)
        assert "Name" in table
        assert "FC" in table  # Free-choice column

    def test_table_has_all_benchmarks(
        self, binary_results: list[PetriBenchmarkResult]
    ) -> None:
        """Table contains a line for each benchmark."""
        table = format_petri_benchmark_table(binary_results)
        for r in binary_results:
            assert r.name[:20] in table, f"Missing {r.name} in table"

    def test_summary_format(
        self, summary: PetriBenchmarkSummary
    ) -> None:
        """Summary formatting produces valid output."""
        text = format_summary(summary)
        assert "Total benchmarks:" in text
        assert "All 1-safe:" in text

    def test_empty_results(self) -> None:
        """Empty results produce a valid summary."""
        s = petri_benchmark_summary([])
        assert s.total_benchmarks == 0
        assert s.all_valid is True


# ---------------------------------------------------------------------------
# Test: summary invariants
# ---------------------------------------------------------------------------

class TestSummaryInvariants:
    """Summary statistics satisfy expected invariants."""

    def test_all_one_safe_true(
        self, summary: PetriBenchmarkSummary
    ) -> None:
        """All benchmarks are 1-safe (fundamental guarantee)."""
        assert summary.all_one_safe is True

    def test_all_reachability_iso_true(
        self, summary: PetriBenchmarkSummary
    ) -> None:
        """All benchmarks have reachability isomorphism."""
        assert summary.all_reachability_iso is True

    def test_all_free_choice_true(
        self, summary: PetriBenchmarkSummary
    ) -> None:
        """All benchmarks are free-choice (state-machine encoding)."""
        assert summary.all_free_choice is True

    def test_place_to_state_ratio_is_one(
        self, summary: PetriBenchmarkSummary
    ) -> None:
        """In state-machine encoding, place/state ratio is always 1.0."""
        for name, ratio in summary.place_to_state_ratios:
            assert ratio == 1.0, f"{name}: place/state ratio = {ratio}"

    def test_transition_to_edge_ratio_is_one(
        self, summary: PetriBenchmarkSummary
    ) -> None:
        """In state-machine encoding, transition/edge ratio is always 1.0."""
        for name, ratio in summary.transition_to_edge_ratios:
            assert ratio == 1.0, f"{name}: transition/edge ratio = {ratio}"
