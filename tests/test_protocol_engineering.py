"""Tests for protocol engineering toolkit (Step 60r)."""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.protocol_engineering import (
    describe_protocol,
    evaluate_design,
    verify_protocol,
    RuntimeMonitor,
)


def _build(s: str):
    return build_statespace(parse(s))


# ---------------------------------------------------------------------------
# 1. DESCRIBE tests
# ---------------------------------------------------------------------------

class TestDescribe:
    def test_end(self):
        profiles = describe_protocol(_build("end"))
        assert len(profiles) == 1
        assert profiles[0].potential == 0.0
        assert profiles[0].kinetic == 0.0

    def test_chain(self):
        profiles = describe_protocol(_build("&{a: &{b: end}}"))
        assert len(profiles) == 3
        # Top has highest potential
        top_p = [p for p in profiles if p.state_id == _build("&{a: &{b: end}}").top][0]
        assert top_p.potential > 0

    def test_branch_hotspot(self):
        """Branch point should be flagged as hotspot."""
        ss = _build("&{a: end, b: end, c: end}")
        profiles = describe_protocol(ss)
        top_p = [p for p in profiles if p.state_id == ss.top][0]
        assert top_p.is_hotspot  # 3 transitions = high T

    def test_no_deadlocks_in_valid(self):
        for typ in ["&{a: end}", "&{a: &{b: end}}", "(&{a: end} || &{b: end})"]:
            profiles = describe_protocol(_build(typ))
            assert not any(p.is_deadlock for p in profiles)


# ---------------------------------------------------------------------------
# 2. BUILD tests
# ---------------------------------------------------------------------------

class TestBuild:
    def test_chain_score(self):
        score = evaluate_design(_build("&{a: &{b: end}}"))
        assert score.termination is True
        assert score.overall > 0
        assert score.efficiency > 0

    def test_parallel_more_robust(self):
        """Parallel should have higher robustness than chain."""
        s_chain = evaluate_design(_build("&{a: &{b: end}}"))
        s_par = evaluate_design(_build("(&{a: end} || &{b: end})"))
        assert s_par.robustness >= s_chain.robustness

    def test_suggestions_for_simple(self):
        score = evaluate_design(_build("&{a: end}"))
        assert isinstance(score.suggestions, list)

    def test_all_scores_bounded(self):
        for typ in ["end", "&{a: end}", "&{a: end, b: end}", "(&{a: end} || &{b: end})"]:
            score = evaluate_design(_build(typ))
            assert 0 <= score.efficiency <= 1.01
            assert 0 <= score.smoothness <= 1.01
            assert 0 <= score.robustness <= 1.01
            assert 0 <= score.overall <= 1.01


# ---------------------------------------------------------------------------
# 3. VERIFY tests
# ---------------------------------------------------------------------------

class TestVerify:
    def test_valid_chain(self):
        r = verify_protocol(_build("&{a: &{b: end}}"))
        assert r.all_passed is True

    def test_valid_parallel(self):
        r = verify_protocol(_build("(&{a: end} || &{b: end})"))
        assert r.all_passed is True

    def test_check_names(self):
        r = verify_protocol(_build("&{a: end}"))
        check_names = [name for name, _, _ in r.checks]
        assert "no_deadlock" in check_names
        assert "termination" in check_names
        assert "no_isolated" in check_names

    def test_recursive_verified(self):
        r = verify_protocol(_build("rec X . &{a: X, b: end}"))
        # Should pass: has escape via b
        term_check = [passed for name, passed, _ in r.checks if name == "termination"]
        assert all(term_check)


# ---------------------------------------------------------------------------
# 4. MONITOR tests
# ---------------------------------------------------------------------------

class TestMonitor:
    def test_simple_execution(self):
        ss = _build("&{a: &{b: end}}")
        mon = RuntimeMonitor(ss)
        assert mon.current_state == ss.top
        assert not mon.is_terminated

        anomalies = mon.step("a")
        assert len(anomalies) == 0

        anomalies = mon.step("b")
        assert len(anomalies) == 0
        assert mon.is_terminated

    def test_invalid_transition(self):
        ss = _build("&{a: end}")
        mon = RuntimeMonitor(ss)
        anomalies = mon.step("nonexistent")
        assert len(anomalies) == 1
        assert "INVALID_TRANSITION" in anomalies[0]

    def test_energy_decreases(self):
        ss = _build("&{a: &{b: end}}")
        mon = RuntimeMonitor(ss)
        e0 = mon.current_potential
        mon.step("a")
        e1 = mon.current_potential
        assert e1 < e0  # Potential decreases

    def test_branch_execution(self):
        ss = _build("&{a: end, b: end}")
        mon = RuntimeMonitor(ss)
        mon.step("a")
        assert mon.is_terminated

    def test_parallel_execution(self):
        ss = _build("(&{a: end} || &{b: end})")
        mon = RuntimeMonitor(ss)
        # Execute one transition
        anomalies = mon.step("a")
        assert not mon.is_terminated
        anomalies = mon.step("b")
        assert mon.is_terminated

    def test_history_tracking(self):
        ss = _build("&{a: &{b: end}}")
        mon = RuntimeMonitor(ss)
        mon.step("a")
        mon.step("b")
        assert len(mon.history) == 2

    def test_energy_trace(self):
        ss = _build("&{a: &{b: end}}")
        mon = RuntimeMonitor(ss)
        mon.step("a")
        mon.step("b")
        trace = mon.energy_trace()
        assert len(trace) >= 2

    def test_all_anomalies(self):
        ss = _build("&{a: end}")
        mon = RuntimeMonitor(ss)
        mon.step("wrong")
        assert len(mon.all_anomalies) >= 1


# ---------------------------------------------------------------------------
# 5. ANALYZE (integration) tests
# ---------------------------------------------------------------------------

class TestAnalyze:
    """Integration tests: describe → build → verify → monitor pipeline."""

    def test_full_pipeline_chain(self):
        ss = _build("&{a: &{b: end}}")

        # Describe
        profiles = describe_protocol(ss)
        assert len(profiles) == 3

        # Build evaluation
        score = evaluate_design(ss)
        assert score.termination is True

        # Verify
        vr = verify_protocol(ss)
        assert vr.all_passed is True

        # Monitor execution
        mon = RuntimeMonitor(ss)
        mon.step("a")
        mon.step("b")
        assert mon.is_terminated
        assert len(mon.all_anomalies) == 0

    def test_full_pipeline_parallel(self):
        ss = _build("(&{a: end} || &{b: end})")
        profiles = describe_protocol(ss)
        score = evaluate_design(ss)
        vr = verify_protocol(ss)
        assert vr.all_passed
        mon = RuntimeMonitor(ss)
        mon.step("a")
        mon.step("b")
        assert mon.is_terminated

    def test_full_pipeline_branch(self):
        ss = _build("&{a: end, b: end}")
        profiles = describe_protocol(ss)
        score = evaluate_design(ss)
        vr = verify_protocol(ss)
        mon = RuntimeMonitor(ss)
        mon.step("b")  # Take branch b
        assert mon.is_terminated


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------

class TestBenchmarks:
    BENCHMARKS = [
        ("Java Iterator", "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"),
        ("File Object", "&{open: &{read: end, write: end}}"),
        ("SMTP", "&{EHLO: &{MAIL: &{RCPT: &{DATA: end}}}}"),
        ("Two-choice", "&{a: end, b: end}"),
        ("Parallel", "(&{a: end} || &{b: end})"),
    ]

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_describe(self, name, typ):
        profiles = describe_protocol(_build(typ))
        assert len(profiles) >= 1

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_design_score(self, name, typ):
        score = evaluate_design(_build(typ))
        assert 0 <= score.overall <= 1.01

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_verify(self, name, typ):
        vr = verify_protocol(_build(typ))
        assert len(vr.checks) >= 3
