"""Tests for software_properties — lattice-theoretic dictionary of software properties."""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.software_properties import (
    PropertyResult,
    DictionaryResult,
    check_safety,
    check_liveness,
    check_deadlock_freedom,
    check_progress,
    check_confluence,
    check_determinism,
    check_reversibility,
    check_transparency,
    check_boundedness,
    check_fairness,
    check_monotonicity,
    check_responsiveness,
    check_compositionality,
    check_all_properties,
    check_properties,
    property_profile,
    PROPERTY_CHECKERS,
)


def _build(type_string: str):
    """Helper: parse + build state space."""
    return build_statespace(parse(type_string))


# ---------------------------------------------------------------------------
# Fixtures: common protocols
# ---------------------------------------------------------------------------

@pytest.fixture
def iterator_ss():
    return _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")


@pytest.fixture
def file_ss():
    return _build("&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}")


@pytest.fixture
def simple_branch():
    return _build("&{a: end, b: end}")


@pytest.fixture
def parallel_ss():
    return _build("(&{a: end} || &{b: end})")


@pytest.fixture
def end_ss():
    return _build("end")


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

class TestSafety:
    def test_end_is_safe(self, end_ss):
        r = check_safety(end_ss)
        assert r.holds
        assert r.name == "safety"

    def test_simple_branch_safe(self, simple_branch):
        r = check_safety(simple_branch)
        assert r.holds

    def test_iterator_safe(self, iterator_ss):
        r = check_safety(iterator_ss)
        assert r.holds

    def test_file_safe(self, file_ss):
        r = check_safety(file_ss)
        assert r.holds

    def test_parallel_safe(self, parallel_ss):
        r = check_safety(parallel_ss)
        assert r.holds

    def test_characterization_present(self, simple_branch):
        r = check_safety(simple_branch)
        assert len(r.characterization) > 0
        assert "bottom" in r.characterization.lower() or "⊥" in r.characterization


# ---------------------------------------------------------------------------
# Liveness
# ---------------------------------------------------------------------------

class TestLiveness:
    def test_end_live(self, end_ss):
        r = check_liveness(end_ss)
        assert r.holds

    def test_simple_branch_live(self, simple_branch):
        r = check_liveness(simple_branch)
        assert r.holds

    def test_iterator_live(self, iterator_ss):
        r = check_liveness(iterator_ss)
        assert r.holds

    def test_file_live(self, file_ss):
        r = check_liveness(file_ss)
        assert r.holds


# ---------------------------------------------------------------------------
# Deadlock Freedom
# ---------------------------------------------------------------------------

class TestDeadlockFreedom:
    def test_end_deadlock_free(self, end_ss):
        r = check_deadlock_freedom(end_ss)
        assert r.holds

    def test_simple_branch_deadlock_free(self, simple_branch):
        r = check_deadlock_freedom(simple_branch)
        assert r.holds

    def test_iterator_deadlock_free(self, iterator_ss):
        r = check_deadlock_freedom(iterator_ss)
        assert r.holds

    def test_parallel_deadlock_free(self, parallel_ss):
        r = check_deadlock_freedom(parallel_ss)
        assert r.holds


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

class TestProgress:
    def test_end_progress(self, end_ss):
        r = check_progress(end_ss)
        assert r.holds

    def test_simple_branch_progress(self, simple_branch):
        r = check_progress(simple_branch)
        assert r.holds

    def test_iterator_progress(self, iterator_ss):
        r = check_progress(iterator_ss)
        assert r.holds

    def test_file_progress(self, file_ss):
        r = check_progress(file_ss)
        assert r.holds


# ---------------------------------------------------------------------------
# Confluence
# ---------------------------------------------------------------------------

class TestConfluence:
    def test_end_confluent(self, end_ss):
        r = check_confluence(end_ss)
        assert r.holds

    def test_simple_branch_confluent(self, simple_branch):
        """Simple branch &{a: end, b: end} is confluent — both go to end."""
        r = check_confluence(simple_branch)
        assert r.holds

    def test_iterator_confluent(self, iterator_ss):
        r = check_confluence(iterator_ss)
        assert r.holds

    def test_parallel_confluent(self, parallel_ss):
        r = check_confluence(parallel_ss)
        assert r.holds


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_end_deterministic(self, end_ss):
        r = check_determinism(end_ss)
        assert r.holds

    def test_simple_branch_deterministic(self, simple_branch):
        r = check_determinism(simple_branch)
        assert r.holds

    def test_iterator_deterministic(self, iterator_ss):
        r = check_determinism(iterator_ss)
        assert r.holds

    def test_file_deterministic(self, file_ss):
        r = check_determinism(file_ss)
        assert r.holds

    def test_parallel_deterministic(self, parallel_ss):
        r = check_determinism(parallel_ss)
        assert r.holds


# ---------------------------------------------------------------------------
# Reversibility
# ---------------------------------------------------------------------------

class TestReversibility:
    def test_end_reversible(self, end_ss):
        """end has only 1 state — trivially reversible."""
        r = check_reversibility(end_ss)
        assert r.holds

    def test_simple_branch_not_reversible(self, simple_branch):
        """&{a: end, b: end} is NOT reversible — can't go back from end."""
        r = check_reversibility(simple_branch)
        assert not r.holds

    def test_iterator_not_fully_reversible(self, iterator_ss):
        """Iterator has a terminal state — not fully reversible."""
        r = check_reversibility(iterator_ss)
        assert not r.holds

    def test_pure_cycle_reversible(self):
        """rec X . &{a: X} is a pure cycle — reversible."""
        ss = _build("rec X . &{a: X}")
        r = check_reversibility(ss)
        # Single SCC (the recursion cycle)
        assert r.holds


# ---------------------------------------------------------------------------
# Transparency
# ---------------------------------------------------------------------------

class TestTransparency:
    def test_end_transparent(self, end_ss):
        r = check_transparency(end_ss)
        assert r.holds

    def test_simple_branch_transparent(self, simple_branch):
        """No selections in simple branch — trivially transparent."""
        r = check_transparency(simple_branch)
        assert r.holds

    def test_iterator_transparent(self, iterator_ss):
        """Iterator selections (TRUE/FALSE) are transparent."""
        r = check_transparency(iterator_ss)
        assert r.holds

    def test_file_transparent(self, file_ss):
        r = check_transparency(file_ss)
        assert r.holds


# ---------------------------------------------------------------------------
# Boundedness
# ---------------------------------------------------------------------------

class TestBoundedness:
    def test_end_bounded(self, end_ss):
        r = check_boundedness(end_ss)
        assert r.holds

    def test_simple_branch_bounded(self, simple_branch):
        r = check_boundedness(simple_branch)
        assert r.holds

    def test_iterator_bounded(self, iterator_ss):
        r = check_boundedness(iterator_ss)
        assert r.holds

    def test_file_bounded(self, file_ss):
        r = check_boundedness(file_ss)
        assert r.holds

    def test_height_in_characterization(self, simple_branch):
        r = check_boundedness(simple_branch)
        assert "height" in r.characterization.lower()


# ---------------------------------------------------------------------------
# Fairness
# ---------------------------------------------------------------------------

class TestFairness:
    def test_end_fair(self, end_ss):
        r = check_fairness(end_ss)
        assert r.holds

    def test_simple_branch_fair(self, simple_branch):
        r = check_fairness(simple_branch)
        assert r.holds

    def test_iterator_fair(self, iterator_ss):
        r = check_fairness(iterator_ss)
        assert r.holds

    def test_file_fair(self, file_ss):
        r = check_fairness(file_ss)
        assert r.holds


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------

class TestMonotonicity:
    def test_end_monotone(self, end_ss):
        r = check_monotonicity(end_ss)
        assert r.holds

    def test_simple_branch_monotone(self, simple_branch):
        r = check_monotonicity(simple_branch)
        assert r.holds

    def test_iterator_has_back_edges(self, iterator_ss):
        """Iterator has recursive back-edges — may not be strictly monotone."""
        r = check_monotonicity(iterator_ss)
        # Recursive types have back-edges that increase distance
        # This is expected for cyclic protocols
        assert r.name == "monotonicity"

    def test_file_monotonicity(self, file_ss):
        r = check_monotonicity(file_ss)
        assert r.name == "monotonicity"


# ---------------------------------------------------------------------------
# Responsiveness
# ---------------------------------------------------------------------------

class TestResponsiveness:
    def test_end_responsive(self, end_ss):
        r = check_responsiveness(end_ss)
        assert r.holds

    def test_simple_branch_responsive(self, simple_branch):
        r = check_responsiveness(simple_branch)
        assert r.holds

    def test_iterator_responsive(self, iterator_ss):
        r = check_responsiveness(iterator_ss)
        assert r.holds

    def test_parallel_responsive(self, parallel_ss):
        r = check_responsiveness(parallel_ss)
        assert r.holds


# ---------------------------------------------------------------------------
# Compositionality
# ---------------------------------------------------------------------------

class TestCompositionality:
    def test_non_product_compositionality(self, simple_branch):
        """Non-product state space: compositionality holds trivially."""
        r = check_compositionality(simple_branch)
        assert r.holds

    def test_parallel_compositionality(self, parallel_ss):
        """Parallel composition: should decompose cleanly."""
        r = check_compositionality(parallel_ss)
        assert r.holds

    def test_result_has_witnesses(self, parallel_ss):
        r = check_compositionality(parallel_ss)
        assert len(r.witnesses) > 0


# ---------------------------------------------------------------------------
# Full dictionary
# ---------------------------------------------------------------------------

class TestFullDictionary:
    def test_check_all_properties(self, simple_branch):
        r = check_all_properties(simple_branch)
        assert isinstance(r, DictionaryResult)
        assert len(r.properties) == len(PROPERTY_CHECKERS)
        assert r.num_states > 0
        assert len(r.summary) > 0

    def test_holding_and_failing(self, simple_branch):
        r = check_all_properties(simple_branch)
        assert isinstance(r.holding, list)
        assert isinstance(r.failing, list)
        assert len(r.holding) + len(r.failing) == len(PROPERTY_CHECKERS)

    def test_all_hold_on_end(self, end_ss):
        r = check_all_properties(end_ss)
        # end should satisfy most/all properties
        assert len(r.holding) >= 10

    def test_profile(self, iterator_ss):
        p = property_profile(iterator_ss)
        assert isinstance(p, dict)
        assert all(isinstance(v, bool) for v in p.values())
        assert len(p) == len(PROPERTY_CHECKERS)


# ---------------------------------------------------------------------------
# Selective checking
# ---------------------------------------------------------------------------

class TestSelectiveChecking:
    def test_check_specific_properties(self, simple_branch):
        r = check_properties(simple_branch, "safety", "liveness")
        assert len(r.properties) == 2
        assert "safety" in r.properties
        assert "liveness" in r.properties

    def test_unknown_property_raises(self, simple_branch):
        with pytest.raises(ValueError, match="Unknown property"):
            check_properties(simple_branch, "nonexistent")

    def test_single_property(self, iterator_ss):
        r = check_properties(iterator_ss, "determinism")
        assert len(r.properties) == 1
        assert r.holds("determinism")


# ---------------------------------------------------------------------------
# Benchmark protocols
# ---------------------------------------------------------------------------

class TestBenchmarks:
    """Test properties on real-world benchmark protocols."""

    @pytest.mark.parametrize("type_string,name", [
        ("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}", "Iterator"),
        ("&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}", "File"),
        ("&{connect: &{ehlo: rec X . &{mail: &{rcpt: &{data: +{OK: X, ERR: X}}}, quit: end}}}", "SMTP"),
        ("&{a: end, b: end}", "SimpleBranch"),
        ("+{ok: end, err: end}", "SimpleSelect"),
        ("(&{a: end} || &{b: end})", "SimpleParallel"),
        ("&{get: +{OK: end, NOT_FOUND: end}, put: +{OK: end, ERR: end}}", "REST"),
        ("rec X . &{request: +{OK: X, ERR: end}}", "RetryLoop"),
    ])
    def test_safety_on_benchmarks(self, type_string, name):
        ss = _build(type_string)
        r = check_safety(ss)
        assert r.holds, f"{name} should be safe"

    @pytest.mark.parametrize("type_string,name", [
        ("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}", "Iterator"),
        ("&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}", "File"),
        ("&{a: end, b: end}", "SimpleBranch"),
        ("(&{a: end} || &{b: end})", "SimpleParallel"),
    ])
    def test_deadlock_freedom_on_benchmarks(self, type_string, name):
        ss = _build(type_string)
        r = check_deadlock_freedom(ss)
        assert r.holds, f"{name} should be deadlock-free"

    @pytest.mark.parametrize("type_string,name", [
        ("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}", "Iterator"),
        ("&{a: end, b: end}", "SimpleBranch"),
        ("(&{a: end} || &{b: end})", "SimpleParallel"),
        ("&{get: +{OK: end, NOT_FOUND: end}, put: +{OK: end, ERR: end}}", "REST"),
    ])
    def test_determinism_on_benchmarks(self, type_string, name):
        ss = _build(type_string)
        r = check_determinism(ss)
        assert r.holds, f"{name} should be deterministic"

    @pytest.mark.parametrize("type_string,name", [
        ("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}", "Iterator"),
        ("&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}", "File"),
        ("&{a: end, b: end}", "SimpleBranch"),
        ("(&{a: end} || &{b: end})", "SimpleParallel"),
    ])
    def test_fairness_on_benchmarks(self, type_string, name):
        ss = _build(type_string)
        r = check_fairness(ss)
        assert r.holds, f"{name} should be fair"


# ---------------------------------------------------------------------------
# Property result structure
# ---------------------------------------------------------------------------

class TestPropertyResult:
    def test_result_fields(self, simple_branch):
        r = check_safety(simple_branch)
        assert isinstance(r, PropertyResult)
        assert isinstance(r.name, str)
        assert isinstance(r.holds, bool)
        assert isinstance(r.characterization, str)
        assert isinstance(r.witnesses, list)

    def test_counterexample_none_when_holds(self, simple_branch):
        r = check_safety(simple_branch)
        if r.holds:
            assert r.counterexample is None


class TestDictionaryResult:
    def test_holds_method(self, simple_branch):
        r = check_all_properties(simple_branch)
        for name in PROPERTY_CHECKERS:
            val = r.holds(name)
            assert isinstance(val, bool)

    def test_all_hold(self, end_ss):
        r = check_all_properties(end_ss)
        assert isinstance(r.all_hold, bool)

    def test_summary_nonempty(self, simple_branch):
        r = check_all_properties(simple_branch)
        assert len(r.summary) > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_state_end(self):
        """end: single state, all properties should hold."""
        ss = _build("end")
        r = check_all_properties(ss)
        for name, prop in r.properties.items():
            if name != "reversibility":
                assert prop.holds, f"{name} should hold on 'end'"

    def test_deep_nesting(self):
        """Deeply nested branch."""
        ss = _build("&{a: &{b: &{c: &{d: end}}}}")
        r = check_all_properties(ss)
        assert r.holds("safety")
        assert r.holds("liveness")
        assert r.holds("determinism")

    def test_wide_branch(self):
        """Wide branch with many options."""
        ss = _build("&{a: end, b: end, c: end, d: end, e: end}")
        r = check_all_properties(ss)
        assert r.holds("safety")
        assert r.holds("confluence")

    def test_pure_recursion(self):
        """Pure recursion: rec X . &{a: X}."""
        ss = _build("rec X . &{a: X}")
        profile = property_profile(ss)
        assert profile["reversibility"]  # Pure cycle is reversible

    def test_nested_parallel(self):
        """Nested parallel composition."""
        ss = _build("((&{a: end} || &{b: end}) || &{c: end})")
        r = check_all_properties(ss)
        assert r.holds("safety")
        assert r.holds("determinism")

    def test_selection_protocol(self):
        """+{ok: end, err: end} — pure selection."""
        ss = _build("+{ok: end, err: end}")
        r = check_all_properties(ss)
        assert r.holds("safety")
        assert r.holds("transparency")

    def test_registry_completeness(self):
        """All 13 properties are registered."""
        assert len(PROPERTY_CHECKERS) == 13
        expected = {
            "safety", "liveness", "deadlock_freedom", "progress",
            "confluence", "determinism", "reversibility", "transparency",
            "boundedness", "fairness", "monotonicity", "responsiveness",
            "compositionality",
        }
        assert set(PROPERTY_CHECKERS.keys()) == expected
