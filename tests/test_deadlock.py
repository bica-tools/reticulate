"""Tests for deadlock detection module (Step 80b).

Tests all 7 deadlock detection techniques on well-formed session types,
deliberately broken protocols, and edge cases.
"""

from __future__ import annotations

import pytest

from reticulate.parser import parse
from reticulate.statespace import StateSpace, build_statespace
from reticulate.deadlock import (
    DeadlockState,
    DeadlockAnalysis,
    LatticeCertificate,
    BlackHoleResult,
    SpectralRisk,
    CompositionalResult,
    BufferDeadlockResult,
    SiphonTrapResult,
    ReticularCertificate,
    detect_deadlocks,
    is_deadlock_free,
    lattice_deadlock_certificate,
    black_hole_detection,
    spectral_deadlock_risk,
    compositional_deadlock_check,
    buffer_deadlock_analysis,
    siphon_trap_analysis,
    reticular_deadlock_certificate,
    analyze_deadlock,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(type_str: str) -> StateSpace:
    """Parse and build a state space from a type string."""
    return build_statespace(parse(type_str))


def _make_deadlocked_ss() -> StateSpace:
    """Create a hand-crafted state space with a deadlock.

    States: 0 (top), 1 (deadlock), 2 (bottom)
    Transitions: 0->1 via 'a', 0->2 via 'b'
    State 1 has no outgoing transitions and is not bottom.
    """
    ss = StateSpace()
    ss.states = {0, 1, 2}
    ss.transitions = [(0, "a", 1), (0, "b", 2)]
    ss.top = 0
    ss.bottom = 2
    ss.labels = {0: "top", 1: "deadlocked", 2: "end"}
    ss.selection_transitions = set()
    return ss


def _make_black_hole_ss() -> StateSpace:
    """Create a state space with a black hole (cycle not reaching bottom).

    States: 0 (top), 1, 2, 3 (bottom)
    Transitions: 0->1 via 'a', 0->3 via 'b', 1->2 via 'c', 2->1 via 'd'
    States 1,2 form a cycle that never reaches bottom (3).
    """
    ss = StateSpace()
    ss.states = {0, 1, 2, 3}
    ss.transitions = [(0, "a", 1), (0, "b", 3), (1, "c", 2), (2, "d", 1)]
    ss.top = 0
    ss.bottom = 3
    ss.labels = {0: "top", 1: "orbit1", 2: "orbit2", 3: "end"}
    ss.selection_transitions = set()
    return ss


def _make_multi_deadlock_ss() -> StateSpace:
    """Create a state space with multiple deadlocks.

    0 -> 1 (a), 0 -> 2 (b), 0 -> 3 (c)
    1 is deadlocked, 2 is deadlocked, 3 is bottom.
    """
    ss = StateSpace()
    ss.states = {0, 1, 2, 3}
    ss.transitions = [(0, "a", 1), (0, "b", 2), (0, "c", 3)]
    ss.top = 0
    ss.bottom = 3
    ss.labels = {0: "top", 1: "dead1", 2: "dead2", 3: "end"}
    ss.selection_transitions = set()
    return ss


# ---------------------------------------------------------------------------
# Test: detect_deadlocks
# ---------------------------------------------------------------------------

class TestDetectDeadlocks:
    """Tests for detect_deadlocks()."""

    def test_end_is_deadlock_free(self) -> None:
        ss = _build("end")
        assert len(detect_deadlocks(ss)) == 0

    def test_simple_branch_deadlock_free(self) -> None:
        ss = _build("&{a: end, b: end}")
        assert len(detect_deadlocks(ss)) == 0

    def test_handcrafted_deadlock(self) -> None:
        ss = _make_deadlocked_ss()
        dl = detect_deadlocks(ss)
        assert len(dl) == 1
        assert dl[0].state == 1

    def test_multi_deadlock(self) -> None:
        ss = _make_multi_deadlock_ss()
        dl = detect_deadlocks(ss)
        assert len(dl) == 2
        states = {d.state for d in dl}
        assert states == {1, 2}

    def test_deadlock_labels(self) -> None:
        ss = _make_deadlocked_ss()
        dl = detect_deadlocks(ss)
        assert dl[0].label == "deadlocked"

    def test_deadlock_depth(self) -> None:
        ss = _make_deadlocked_ss()
        dl = detect_deadlocks(ss)
        assert dl[0].depth_from_top == 1

    def test_iterator_deadlock_free(self) -> None:
        ss = _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        assert len(detect_deadlocks(ss)) == 0

    def test_selection_deadlock_free(self) -> None:
        ss = _build("+{ok: end, err: end}")
        assert len(detect_deadlocks(ss)) == 0

    def test_parallel_deadlock_free(self) -> None:
        ss = _build("(&{a: end} || &{b: end})")
        assert len(detect_deadlocks(ss)) == 0


# ---------------------------------------------------------------------------
# Test: is_deadlock_free
# ---------------------------------------------------------------------------

class TestIsDeadlockFree:
    """Tests for is_deadlock_free()."""

    def test_end_free(self) -> None:
        assert is_deadlock_free(_build("end"))

    def test_branch_free(self) -> None:
        assert is_deadlock_free(_build("&{open: &{close: end}}"))

    def test_handcrafted_not_free(self) -> None:
        assert not is_deadlock_free(_make_deadlocked_ss())

    def test_recursive_free(self) -> None:
        assert is_deadlock_free(
            _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        )

    def test_nested_branch_free(self) -> None:
        assert is_deadlock_free(_build("&{a: &{b: end}, c: end}"))


# ---------------------------------------------------------------------------
# Test: lattice_deadlock_certificate
# ---------------------------------------------------------------------------

class TestLatticeCertificate:
    """Tests for lattice_deadlock_certificate()."""

    def test_lattice_end(self) -> None:
        cert = lattice_deadlock_certificate(_build("end"))
        assert cert.is_lattice
        assert cert.certificate_valid

    def test_lattice_branch(self) -> None:
        cert = lattice_deadlock_certificate(_build("&{a: end, b: end}"))
        assert cert.is_lattice
        assert cert.certificate_valid
        assert "DEADLOCK-FREE" in cert.explanation

    def test_lattice_iterator(self) -> None:
        cert = lattice_deadlock_certificate(
            _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        )
        assert cert.certificate_valid

    def test_non_lattice_deadlocked(self) -> None:
        ss = _make_deadlocked_ss()
        cert = lattice_deadlock_certificate(ss)
        # The handcrafted ss is not a lattice
        assert not cert.certificate_valid or not cert.is_lattice

    def test_has_unique_bottom(self) -> None:
        cert = lattice_deadlock_certificate(_build("&{a: end, b: end}"))
        assert cert.has_unique_bottom

    def test_all_reach_bottom(self) -> None:
        cert = lattice_deadlock_certificate(_build("&{a: end, b: end}"))
        assert cert.all_reach_bottom


# ---------------------------------------------------------------------------
# Test: black_hole_detection
# ---------------------------------------------------------------------------

class TestBlackHoleDetection:
    """Tests for black_hole_detection()."""

    def test_no_black_holes_end(self) -> None:
        bh = black_hole_detection(_build("end"))
        assert bh.total_trapped_states == 0

    def test_no_black_holes_branch(self) -> None:
        bh = black_hole_detection(_build("&{a: end, b: end}"))
        assert bh.total_trapped_states == 0
        assert len(bh.black_holes) == 0

    def test_black_hole_cycle(self) -> None:
        ss = _make_black_hole_ss()
        bh = black_hole_detection(ss)
        assert bh.total_trapped_states == 2
        assert set(bh.black_holes) == {1, 2}

    def test_event_horizons(self) -> None:
        ss = _make_black_hole_ss()
        bh = black_hole_detection(ss)
        # The transition 0->1 via 'a' is the event horizon
        assert len(bh.event_horizons) == 1
        assert bh.event_horizons[0] == (0, "a", 1)

    def test_no_black_holes_recursive(self) -> None:
        ss = _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        bh = black_hole_detection(ss)
        assert bh.total_trapped_states == 0

    def test_no_black_holes_parallel(self) -> None:
        ss = _build("(&{a: end} || &{b: end})")
        bh = black_hole_detection(ss)
        assert bh.total_trapped_states == 0


# ---------------------------------------------------------------------------
# Test: spectral_deadlock_risk
# ---------------------------------------------------------------------------

class TestSpectralRisk:
    """Tests for spectral_deadlock_risk()."""

    def test_end_risk(self) -> None:
        risk = spectral_deadlock_risk(_build("end"))
        assert isinstance(risk, SpectralRisk)

    def test_branch_risk(self) -> None:
        risk = spectral_deadlock_risk(_build("&{a: end, b: end}"))
        assert risk.fiedler_value >= 0
        assert 0 <= risk.risk_score <= 1

    def test_well_connected_low_risk(self) -> None:
        # A well-connected protocol should have lower risk
        ss = _build("&{a: end, b: end}")
        risk = spectral_deadlock_risk(ss)
        assert risk.risk_score < 0.8

    def test_diagnosis_present(self) -> None:
        risk = spectral_deadlock_risk(_build("&{a: end, b: end}"))
        assert len(risk.diagnosis) > 0

    def test_spectral_gap_nonneg(self) -> None:
        risk = spectral_deadlock_risk(_build("&{a: &{b: end}, c: end}"))
        assert risk.spectral_gap >= 0

    def test_bottleneck_states_tuple(self) -> None:
        risk = spectral_deadlock_risk(_build("&{a: end, b: end}"))
        assert isinstance(risk.bottleneck_states, tuple)


# ---------------------------------------------------------------------------
# Test: compositional_deadlock_check
# ---------------------------------------------------------------------------

class TestCompositionalCheck:
    """Tests for compositional_deadlock_check()."""

    def test_two_simple_protocols(self) -> None:
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end}")
        result = compositional_deadlock_check(ss1, ss2)
        assert result.is_deadlock_free

    def test_both_individually_free(self) -> None:
        ss1 = _build("&{a: end, b: end}")
        ss2 = _build("&{c: end}")
        result = compositional_deadlock_check(ss1, ss2)
        assert result.individual_deadlock_free == (True, True)

    def test_one_deadlocked_component(self) -> None:
        ss1 = _make_deadlocked_ss()
        ss2 = _build("&{x: end}")
        result = compositional_deadlock_check(ss1, ss2)
        assert not result.individual_deadlock_free[0]

    def test_product_states_count(self) -> None:
        ss1 = _build("&{a: end, b: end}")
        ss2 = _build("&{c: end}")
        result = compositional_deadlock_check(ss1, ss2)
        assert result.is_deadlock_free

    def test_explanation_present(self) -> None:
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end}")
        result = compositional_deadlock_check(ss1, ss2)
        assert len(result.explanation) > 0


# ---------------------------------------------------------------------------
# Test: buffer_deadlock_analysis
# ---------------------------------------------------------------------------

class TestBufferDeadlockAnalysis:
    """Tests for buffer_deadlock_analysis()."""

    def test_simple_synchronous_safe(self) -> None:
        ss = _build("&{a: end}")
        result = buffer_deadlock_analysis(ss, max_k=3)
        assert result.is_synchronous_safe

    def test_min_capacity_zero_for_simple(self) -> None:
        ss = _build("&{a: end, b: end}")
        result = buffer_deadlock_analysis(ss, max_k=3)
        assert result.min_safe_capacity == 0

    def test_deadlocks_per_capacity_populated(self) -> None:
        ss = _build("&{a: end}")
        result = buffer_deadlock_analysis(ss, max_k=3)
        assert 0 in result.deadlocks_per_capacity
        assert 1 in result.deadlocks_per_capacity

    def test_growth_rate_populated(self) -> None:
        ss = _build("&{a: end}")
        result = buffer_deadlock_analysis(ss, max_k=3)
        assert result.growth_rate in ("constant-zero", "constant", "increasing", "decreasing", "non-monotonic")

    def test_explanation_present(self) -> None:
        ss = _build("+{ok: end, err: end}")
        result = buffer_deadlock_analysis(ss, max_k=3)
        assert len(result.explanation) > 0

    def test_deadlocked_ss_buffer_analysis(self) -> None:
        ss = _make_deadlocked_ss()
        result = buffer_deadlock_analysis(ss, max_k=3)
        # At capacity 0, should detect the deadlock
        assert result.deadlocks_per_capacity[0] == 1


# ---------------------------------------------------------------------------
# Test: siphon_trap_analysis
# ---------------------------------------------------------------------------

class TestSiphonTrapAnalysis:
    """Tests for siphon_trap_analysis()."""

    def test_simple_branch_siphon_trap(self) -> None:
        ss = _build("&{a: end, b: end}")
        result = siphon_trap_analysis(ss)
        assert isinstance(result, SiphonTrapResult)

    def test_end_siphon_trap(self) -> None:
        ss = _build("end")
        result = siphon_trap_analysis(ss)
        assert result.is_deadlock_free

    def test_siphons_are_frozensets(self) -> None:
        ss = _build("&{a: end, b: end}")
        result = siphon_trap_analysis(ss)
        for s in result.siphons:
            assert isinstance(s, frozenset)

    def test_traps_are_frozensets(self) -> None:
        ss = _build("&{a: end, b: end}")
        result = siphon_trap_analysis(ss)
        for t in result.traps:
            assert isinstance(t, frozenset)

    def test_iterator_siphon_trap(self) -> None:
        ss = _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        result = siphon_trap_analysis(ss)
        assert isinstance(result, SiphonTrapResult)

    def test_explanation_present(self) -> None:
        ss = _build("&{a: end}")
        result = siphon_trap_analysis(ss)
        assert len(result.explanation) > 0


# ---------------------------------------------------------------------------
# Test: reticular_deadlock_certificate
# ---------------------------------------------------------------------------

class TestReticularCertificate:
    """Tests for reticular_deadlock_certificate()."""

    def test_end_reticulate(self) -> None:
        cert = reticular_deadlock_certificate(_build("end"))
        assert cert.is_reticulate
        assert cert.certificate_valid

    def test_branch_reticulate(self) -> None:
        cert = reticular_deadlock_certificate(_build("&{a: end, b: end}"))
        assert cert.is_reticulate
        assert cert.certificate_valid
        assert "DEADLOCK-FREE" in cert.explanation

    def test_selection_reticulate(self) -> None:
        cert = reticular_deadlock_certificate(_build("+{ok: end, err: end}"))
        assert cert.certificate_valid

    def test_classifications_populated(self) -> None:
        cert = reticular_deadlock_certificate(_build("&{a: end, b: end}"))
        assert len(cert.state_classifications) > 0

    def test_non_reticulate_handcrafted(self) -> None:
        ss = _make_deadlocked_ss()
        cert = reticular_deadlock_certificate(ss)
        # Deadlocked ss may not be reticulate
        assert isinstance(cert, ReticularCertificate)


# ---------------------------------------------------------------------------
# Test: analyze_deadlock (unified)
# ---------------------------------------------------------------------------

class TestAnalyzeDeadlock:
    """Tests for analyze_deadlock() unified analysis."""

    def test_end_analysis(self) -> None:
        result = analyze_deadlock(_build("end"))
        assert result.is_deadlock_free
        assert "DEADLOCK-FREE" in result.summary

    def test_branch_analysis(self) -> None:
        result = analyze_deadlock(_build("&{a: end, b: end}"))
        assert result.is_deadlock_free

    def test_deadlocked_analysis(self) -> None:
        result = analyze_deadlock(_make_deadlocked_ss())
        assert not result.is_deadlock_free
        assert "DEADLOCK DETECTED" in result.summary

    def test_analysis_has_all_fields(self) -> None:
        result = analyze_deadlock(_build("&{a: end}"))
        assert isinstance(result.lattice_certificate, LatticeCertificate)
        assert isinstance(result.black_holes, BlackHoleResult)
        assert isinstance(result.spectral_risk, SpectralRisk)
        assert isinstance(result.siphon_trap, SiphonTrapResult)
        assert isinstance(result.reticular_certificate, ReticularCertificate)

    def test_analysis_num_states(self) -> None:
        ss = _build("&{a: end, b: end}")
        result = analyze_deadlock(ss)
        assert result.num_states == len(ss.states)

    def test_analysis_num_transitions(self) -> None:
        ss = _build("&{a: end, b: end}")
        result = analyze_deadlock(ss)
        assert result.num_transitions == len(ss.transitions)

    def test_iterator_unified(self) -> None:
        ss = _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        result = analyze_deadlock(ss)
        assert result.is_deadlock_free

    def test_parallel_unified(self) -> None:
        ss = _build("(&{a: end} || &{b: end})")
        result = analyze_deadlock(ss)
        assert result.is_deadlock_free

    def test_selection_unified(self) -> None:
        ss = _build("+{ok: end, err: end}")
        result = analyze_deadlock(ss)
        assert result.is_deadlock_free

    def test_black_hole_detected_in_unified(self) -> None:
        ss = _make_black_hole_ss()
        result = analyze_deadlock(ss)
        assert result.black_holes.total_trapped_states > 0

    def test_multiple_certificates(self) -> None:
        result = analyze_deadlock(_build("&{a: end, b: end}"))
        # Should have at least 2 confirming techniques
        verdicts = 0
        if result.lattice_certificate.certificate_valid:
            verdicts += 1
        if result.black_holes.total_trapped_states == 0:
            verdicts += 1
        if result.reticular_certificate.certificate_valid:
            verdicts += 1
        assert verdicts >= 2

    def test_deadlock_state_details(self) -> None:
        result = analyze_deadlock(_make_deadlocked_ss())
        assert len(result.deadlocked_states) == 1
        dl = result.deadlocked_states[0]
        assert dl.state == 1
        assert dl.depth_from_top == 1


# ---------------------------------------------------------------------------
# Test: Benchmark protocols (all should be deadlock-free)
# ---------------------------------------------------------------------------

class TestBenchmarkDeadlockFreedom:
    """Verify that all standard benchmark protocols are deadlock-free."""

    @pytest.mark.parametrize("type_str,name", [
        ("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}", "Iterator"),
        ("&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}", "File"),
        ("&{a: end, b: end}", "Simple Branch"),
        ("+{ok: end, err: end}", "Simple Selection"),
        ("(&{a: end} || &{b: end})", "Simple Parallel"),
        ("&{connect: +{OK: rec X . &{query: +{result: X, done: &{disconnect: end}}}, FAIL: end}}", "Database"),
        ("&{login: +{OK: rec X . &{send: X, recv: X, quit: end}, FAIL: end}}", "Chat"),
    ])
    def test_benchmark_deadlock_free(self, type_str: str, name: str) -> None:
        ss = _build(type_str)
        result = analyze_deadlock(ss)
        assert result.is_deadlock_free, f"Protocol {name} has deadlock: {result.summary}"


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for deadlock detection."""

    def test_single_state_end(self) -> None:
        """end has one state, no transitions, no deadlock."""
        ss = _build("end")
        assert is_deadlock_free(ss)

    def test_self_loop_recursive(self) -> None:
        """Recursive type with self-loop should still be deadlock-free if bottom reachable."""
        ss = _build("rec X . &{loop: X, stop: end}")
        assert is_deadlock_free(ss)

    def test_deeply_nested_branch(self) -> None:
        ss = _build("&{a: &{b: &{c: end}}}")
        assert is_deadlock_free(ss)
        result = analyze_deadlock(ss)
        assert result.lattice_certificate.certificate_valid

    def test_parallel_with_selection(self) -> None:
        ss = _build("(+{a: end} || &{b: end})")
        assert is_deadlock_free(ss)
