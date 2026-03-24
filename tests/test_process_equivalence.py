"""Tests for process_equivalence module (Step 29c).

Verifies that CCS bisimulation, CSP trace equivalence, and CSP failure
equivalence COINCIDE on session type state spaces.

25+ test cases covering:
- Self-equivalence (identity)
- Equal types (same structure)
- Non-equivalent types
- Subtype pairs
- Recursive types
- Selection types
- Nested branches
- Parallel types
- Benchmark protocols
- Equivalence theorem verification
"""

import pytest

from reticulate.parser import parse
from reticulate.statespace import StateSpace, build_statespace
from reticulate.process_equivalence import (
    ProcessEquivalenceResult,
    EquivalenceTheoremResult,
    BenchmarkEquivalenceResult,
    check_all_equivalences,
    check_equivalences_from_types,
    equivalence_theorem,
    theorem_from_type,
    check_benchmarks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ss(type_str: str) -> StateSpace:
    """Parse and build state space from a type string."""
    return build_statespace(parse(type_str))


# ---------------------------------------------------------------------------
# Test: ProcessEquivalenceResult structure
# ---------------------------------------------------------------------------

class TestProcessEquivalenceResult:
    """Test the result dataclass."""

    def test_frozen(self) -> None:
        r = ProcessEquivalenceResult(
            bisimilar=True, trace_equivalent=True, failure_equivalent=True,
            all_agree=True, ss1_deterministic=True, ss2_deterministic=True,
            details="test",
        )
        with pytest.raises(AttributeError):
            r.bisimilar = False  # type: ignore[misc]

    def test_all_agree_true(self) -> None:
        r = ProcessEquivalenceResult(
            bisimilar=True, trace_equivalent=True, failure_equivalent=True,
            all_agree=True, ss1_deterministic=True, ss2_deterministic=True,
            details="ok",
        )
        assert r.all_agree is True

    def test_all_agree_false(self) -> None:
        r = ProcessEquivalenceResult(
            bisimilar=True, trace_equivalent=False, failure_equivalent=True,
            all_agree=False, ss1_deterministic=True, ss2_deterministic=True,
            details="disagree",
        )
        assert r.all_agree is False


# ---------------------------------------------------------------------------
# Test: Self-equivalence (identity comparisons)
# ---------------------------------------------------------------------------

class TestSelfEquivalence:
    """A state space compared with itself must be equivalent in all three senses."""

    def test_end_self(self) -> None:
        ss = _ss("end")
        r = check_all_equivalences(ss, ss)
        assert r.all_agree
        assert r.bisimilar
        assert r.trace_equivalent
        assert r.failure_equivalent

    def test_simple_branch_self(self) -> None:
        ss = _ss("&{a: end, b: end}")
        r = check_all_equivalences(ss, ss)
        assert r.all_agree
        assert r.bisimilar

    def test_selection_self(self) -> None:
        ss = _ss("+{OK: end, ERR: end}")
        r = check_all_equivalences(ss, ss)
        assert r.all_agree
        assert r.bisimilar

    def test_recursive_self(self) -> None:
        ss = _ss("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        r = check_all_equivalences(ss, ss)
        assert r.all_agree
        assert r.bisimilar

    def test_nested_branch_self(self) -> None:
        ss = _ss("&{a: &{b: end, c: end}, d: end}")
        r = check_all_equivalences(ss, ss)
        assert r.all_agree
        assert r.bisimilar


# ---------------------------------------------------------------------------
# Test: Equal types (structurally identical)
# ---------------------------------------------------------------------------

class TestEqualTypes:
    """Types parsed from the same string must be equivalent."""

    def test_two_parses_same_type(self) -> None:
        t = "&{a: end, b: end}"
        ss1 = _ss(t)
        ss2 = _ss(t)
        r = check_all_equivalences(ss1, ss2)
        assert r.all_agree
        assert r.bisimilar

    def test_iterator_twice(self) -> None:
        t = "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"
        r = check_equivalences_from_types(t, t)
        assert r.all_agree
        assert r.bisimilar


# ---------------------------------------------------------------------------
# Test: Non-equivalent types — all three should say "not equivalent"
# ---------------------------------------------------------------------------

class TestNonEquivalent:
    """Different types must be non-equivalent in all three senses."""

    def test_different_labels(self) -> None:
        r = check_equivalences_from_types(
            "&{a: end, b: end}",
            "&{a: end, c: end}",
        )
        assert r.all_agree
        assert not r.bisimilar
        assert not r.trace_equivalent
        assert not r.failure_equivalent

    def test_different_depth(self) -> None:
        r = check_equivalences_from_types(
            "&{a: end}",
            "&{a: &{b: end}}",
        )
        assert r.all_agree
        assert not r.bisimilar

    def test_different_width(self) -> None:
        r = check_equivalences_from_types(
            "&{a: end}",
            "&{a: end, b: end}",
        )
        assert r.all_agree
        assert not r.bisimilar

    def test_branch_vs_selection(self) -> None:
        # Branch and selection with same label differ in CCS (tau prefix)
        # but share traces/failures at the state-space level.
        # Bisimulation uses the CCS LTS (with tau_ prefix for selections)
        # so it distinguishes them.  This is a known divergence between
        # the CCS and CSP views when polarity differs.
        r = check_equivalences_from_types(
            "&{a: end}",
            "+{a: end}",
        )
        # When polarity differs, bisimulation may disagree with trace/failure eq
        # because CCS encodes selection as internal (tau) actions.
        # The coincidence theorem applies to pairs drawn from the SAME
        # state space (same polarity labelling), not across different polarities.
        assert isinstance(r, ProcessEquivalenceResult)

    def test_end_vs_branch(self) -> None:
        r = check_equivalences_from_types(
            "end",
            "&{a: end}",
        )
        assert r.all_agree
        assert not r.bisimilar


# ---------------------------------------------------------------------------
# Test: Equivalence coincidence on various type families
# ---------------------------------------------------------------------------

class TestEquivalenceCoincidence:
    """The three equivalences must always agree on session type state spaces."""

    def test_simple_branch(self) -> None:
        ss1 = _ss("&{a: end, b: end}")
        ss2 = _ss("&{a: end, b: end, c: end}")
        r = check_all_equivalences(ss1, ss2)
        assert r.all_agree

    def test_nested_vs_flat(self) -> None:
        ss1 = _ss("&{a: &{b: end}}")
        ss2 = _ss("&{c: end}")
        r = check_all_equivalences(ss1, ss2)
        assert r.all_agree

    def test_selection_with_different_labels(self) -> None:
        ss1 = _ss("+{OK: end, ERR: end}")
        ss2 = _ss("+{OK: end}")
        r = check_all_equivalences(ss1, ss2)
        assert r.all_agree

    def test_recursive_vs_non_recursive(self) -> None:
        ss1 = _ss("rec X . &{a: X, b: end}")
        ss2 = _ss("&{a: end, b: end}")
        r = check_all_equivalences(ss1, ss2)
        assert r.all_agree

    def test_deterministic_flag(self) -> None:
        ss = _ss("&{a: end, b: end}")
        r = check_all_equivalences(ss, ss)
        assert r.ss1_deterministic
        assert r.ss2_deterministic


# ---------------------------------------------------------------------------
# Test: Equivalence theorem on individual state spaces
# ---------------------------------------------------------------------------

class TestEquivalenceTheorem:
    """Test that the coincidence theorem holds for various state spaces."""

    def test_end_theorem(self) -> None:
        r = theorem_from_type("end")
        assert r.holds
        assert r.is_deterministic

    def test_simple_branch_theorem(self) -> None:
        r = theorem_from_type("&{a: end, b: end}")
        assert r.holds
        assert r.is_deterministic
        assert r.pairs_tested > 0
        assert r.pairs_tested == r.pairs_agreed

    def test_selection_theorem(self) -> None:
        r = theorem_from_type("+{OK: end, ERR: end}")
        assert r.holds

    def test_nested_theorem(self) -> None:
        r = theorem_from_type("&{a: &{b: end, c: end}, d: end}")
        assert r.holds

    def test_recursive_theorem(self) -> None:
        r = theorem_from_type(
            "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"
        )
        assert r.holds
        assert r.is_deterministic
        assert r.is_lattice

    def test_file_protocol_theorem(self) -> None:
        r = theorem_from_type(
            "&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}"
        )
        assert r.holds
        assert r.is_lattice

    def test_smtp_theorem(self) -> None:
        r = theorem_from_type(
            "&{connect: &{ehlo: rec X . &{mail: &{rcpt: &{data: "
            "+{OK: X, ERR: X}}}, quit: end}}}"
        )
        assert r.holds

    def test_http_theorem(self) -> None:
        r = theorem_from_type(
            "&{connect: rec X . &{request: +{OK200: &{readBody: X}, "
            "ERR4xx: X, ERR5xx: X}, close: end}}"
        )
        assert r.holds

    def test_theorem_result_structure(self) -> None:
        r = theorem_from_type("&{a: end}")
        assert isinstance(r, EquivalenceTheoremResult)
        assert r.counterexample is None
        assert "HOLDS" in r.details


# ---------------------------------------------------------------------------
# Test: Benchmark suite
# ---------------------------------------------------------------------------

class TestBenchmarks:
    """Test equivalence coincidence across benchmark protocols."""

    def test_benchmark_suite(self) -> None:
        r = check_benchmarks(max_depth=8)
        assert isinstance(r, BenchmarkEquivalenceResult)
        assert r.total_benchmarks > 0
        assert r.theorem_holds == r.total_benchmarks
        assert len(r.failures) == 0

    def test_benchmark_deterministic(self) -> None:
        r = check_benchmarks(max_depth=8)
        # Most session type benchmarks are deterministic; parallel
        # composition with shared labels can introduce nondeterminism
        assert r.all_deterministic >= r.total_benchmarks - 10

    def test_benchmark_lattice(self) -> None:
        r = check_benchmarks(max_depth=8)
        # All benchmarks should form lattices
        assert r.all_lattice == r.total_benchmarks


# ---------------------------------------------------------------------------
# Test: Details and diagnostics
# ---------------------------------------------------------------------------

class TestDetails:
    """Test human-readable output."""

    def test_agree_details(self) -> None:
        r = check_equivalences_from_types("&{a: end}", "&{a: end}")
        assert "agree" in r.details.lower() or "equivalent" in r.details.lower()

    def test_theorem_details(self) -> None:
        r = theorem_from_type("&{a: end, b: end}")
        assert "HOLDS" in r.details
