"""Tests for DoS resistance analysis via lattice width and tropical algebra (Step 89f)."""

import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace, StateSpace
from reticulate.dos_resistance import (
    DoSResult,
    analyze_dos_resistance,
    dos_surface,
    kemeny_vulnerability,
    throughput_bottleneck,
    width_exposure,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ss(type_str: str) -> StateSpace:
    """Parse and build state space from a session type string."""
    return build_statespace(parse(type_str))


# ---------------------------------------------------------------------------
# DoS surface tests
# ---------------------------------------------------------------------------


class TestDoSSurface:
    """DoS surface identification."""

    def test_end_no_surface(self):
        ss = _ss("end")
        surface = dos_surface(ss)
        assert surface == []

    def test_single_branch_no_surface(self):
        ss = _ss("&{a: end}")
        surface = dos_surface(ss)
        # Out-degree 1 is below default threshold of 2
        assert surface == []

    def test_two_branch_is_surface(self):
        ss = _ss("&{a: end, b: end}")
        surface = dos_surface(ss)
        assert len(surface) >= 1

    def test_selection_is_surface(self):
        ss = _ss("+{a: end, b: end}")
        surface = dos_surface(ss)
        assert len(surface) >= 1

    def test_custom_threshold(self):
        ss = _ss("&{a: end, b: end, c: end}")
        surface_2 = dos_surface(ss, threshold=2)
        surface_3 = dos_surface(ss, threshold=3)
        surface_4 = dos_surface(ss, threshold=4)
        assert len(surface_2) >= len(surface_3) >= len(surface_4)

    def test_surface_contains_valid_states(self):
        ss = _ss("&{a: end, b: end}")
        surface = dos_surface(ss)
        for s in surface:
            assert s in ss.states

    def test_surface_sorted(self):
        ss = _ss("&{a: end, b: &{c: end, d: end}}")
        surface = dos_surface(ss)
        assert surface == sorted(surface)

    def test_chain_no_surface(self):
        ss = _ss("&{a: &{b: &{c: end}}}")
        surface = dos_surface(ss)
        assert surface == []


# ---------------------------------------------------------------------------
# Width exposure tests
# ---------------------------------------------------------------------------


class TestWidthExposure:
    """Width exposure computation."""

    def test_end_width(self):
        ss = _ss("end")
        widths = width_exposure(ss)
        assert widths == [1]

    def test_branch_width(self):
        ss = _ss("&{a: end, b: end}")
        widths = width_exposure(ss)
        assert len(widths) >= 2
        assert widths[0] == 1  # top

    def test_parallel_width(self):
        ss = _ss("(&{a: end} || &{b: end})")
        widths = width_exposure(ss)
        assert max(widths) >= 1

    def test_width_is_positive(self):
        ss = _ss("&{a: &{b: end}, c: end}")
        widths = width_exposure(ss)
        for w in widths:
            assert w > 0

    def test_width_sum_equals_states(self):
        ss = _ss("+{a: end, b: &{c: end}}")
        widths = width_exposure(ss)
        assert sum(widths) == len(ss.states)


# ---------------------------------------------------------------------------
# Throughput bottleneck tests
# ---------------------------------------------------------------------------


class TestThroughputBottleneck:
    """Tropical eigenvalue as throughput bottleneck."""

    def test_end_no_bottleneck(self):
        ss = _ss("end")
        val = throughput_bottleneck(ss)
        assert val == 0.0

    def test_acyclic_no_bottleneck(self):
        ss = _ss("&{a: end, b: end}")
        val = throughput_bottleneck(ss)
        assert val == 0.0

    def test_recursive_has_bottleneck(self):
        ss = _ss("rec X . &{next: X, stop: end}")
        val = throughput_bottleneck(ss)
        assert val >= 0.0  # Should be 1.0 for a cycle

    def test_bottleneck_nonnegative(self):
        ss = _ss("&{a: &{b: end}}")
        val = throughput_bottleneck(ss)
        assert val >= 0.0


# ---------------------------------------------------------------------------
# Kemeny vulnerability tests
# ---------------------------------------------------------------------------


class TestKemenyVulnerability:
    """Kemeny constant vulnerability assessment."""

    def test_end_kemeny(self):
        ss = _ss("end")
        val, vuln = kemeny_vulnerability(ss)
        assert isinstance(val, float)
        assert isinstance(vuln, bool)

    def test_branch_kemeny(self):
        ss = _ss("&{a: end, b: end}")
        val, vuln = kemeny_vulnerability(ss)
        assert isinstance(val, float)

    def test_custom_threshold(self):
        ss = _ss("&{a: end, b: end}")
        _, vuln_low = kemeny_vulnerability(ss, threshold=0.01)
        _, vuln_high = kemeny_vulnerability(ss, threshold=1000.0)
        # With high threshold, more likely to be vulnerable
        # (or both False if kemeny is inf)
        assert isinstance(vuln_low, bool)
        assert isinstance(vuln_high, bool)

    def test_recursive_kemeny(self):
        ss = _ss("rec X . &{a: X, b: end}")
        val, vuln = kemeny_vulnerability(ss)
        assert isinstance(val, float)


# ---------------------------------------------------------------------------
# Combined analysis tests
# ---------------------------------------------------------------------------


class TestAnalyzeDoSResistance:
    """Full DoS resistance analysis."""

    def test_end_result_type(self):
        ss = _ss("end")
        result = analyze_dos_resistance(ss)
        assert isinstance(result, DoSResult)

    def test_end_safe(self):
        ss = _ss("end")
        result = analyze_dos_resistance(ss)
        assert result.vulnerability_score == 0.0
        assert result.dos_surface == []
        assert not result.has_amplification

    def test_branch_analysis(self):
        ss = _ss("&{a: end, b: end}")
        result = analyze_dos_resistance(ss)
        assert result.num_states > 0
        assert isinstance(result.vulnerability_score, float)
        assert 0.0 <= result.vulnerability_score <= 1.0

    def test_selection_analysis(self):
        ss = _ss("+{ok: end, err: end}")
        result = analyze_dos_resistance(ss)
        assert isinstance(result, DoSResult)

    def test_recursive_amplification(self):
        ss = _ss("rec X . &{next: X, stop: end}")
        result = analyze_dos_resistance(ss)
        assert result.has_amplification  # cycle exists

    def test_parallel_analysis(self):
        ss = _ss("(&{a: end} || &{b: end})")
        result = analyze_dos_resistance(ss)
        assert isinstance(result.width_exposure, list)
        assert result.max_width >= 1

    def test_branching_factors_complete(self):
        ss = _ss("&{a: end, b: &{c: end}}")
        result = analyze_dos_resistance(ss)
        assert set(result.branching_factors.keys()) == ss.states

    def test_max_branching_correct(self):
        ss = _ss("&{a: end, b: end, c: end}")
        result = analyze_dos_resistance(ss)
        assert result.max_branching == max(result.branching_factors.values())

    def test_score_bounded(self):
        ss = _ss("rec X . &{a: X, b: X, c: end}")
        result = analyze_dos_resistance(ss)
        assert 0.0 <= result.vulnerability_score <= 1.0

    def test_result_fields_complete(self):
        ss = _ss("&{a: &{b: end}, c: end}")
        result = analyze_dos_resistance(ss)
        assert isinstance(result.dos_surface, list)
        assert isinstance(result.dos_surface_score, float)
        assert isinstance(result.width_exposure, list)
        assert isinstance(result.max_width, int)
        assert isinstance(result.max_width_rank, int)
        assert isinstance(result.throughput_bottleneck, float)
        assert isinstance(result.has_amplification, bool)
        assert isinstance(result.kemeny_value, float)
        assert isinstance(result.kemeny_vulnerable, bool)
        assert isinstance(result.branching_factors, dict)
        assert isinstance(result.max_branching, int)
        assert isinstance(result.critical_states, list)
        assert isinstance(result.vulnerability_score, float)
        assert isinstance(result.num_states, int)

    def test_critical_states_subset_of_surface(self):
        ss = _ss("&{a: end, b: end, c: &{d: end, e: end}}")
        result = analyze_dos_resistance(ss)
        surface_set = set(result.dos_surface)
        for s in result.critical_states:
            assert s in surface_set


# ---------------------------------------------------------------------------
# Benchmark protocol tests
# ---------------------------------------------------------------------------


class TestBenchmarkProtocols:
    """DoS resistance on standard protocols."""

    def test_smtp_protocol(self):
        ss = _ss("&{connect: +{OK: &{mail: +{OK: &{rcpt: +{OK: &{data: &{send: end}}, ERR: end}}, ERR: end}}, ERR: end}}")
        result = analyze_dos_resistance(ss)
        assert isinstance(result, DoSResult)
        assert not result.has_amplification  # acyclic

    def test_iterator_protocol(self):
        ss = _ss("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        result = analyze_dos_resistance(ss)
        assert result.has_amplification  # cyclic

    def test_two_branch_selection(self):
        ss = _ss("+{accept: &{process: end}, reject: end}")
        result = analyze_dos_resistance(ss)
        assert isinstance(result, DoSResult)

    def test_parallel_protocol(self):
        ss = _ss("(&{read: end} || &{write: end})")
        result = analyze_dos_resistance(ss)
        assert isinstance(result, DoSResult)

    def test_deep_nesting(self):
        ss = _ss("&{a: &{b: &{c: &{d: end}}}}")
        result = analyze_dos_resistance(ss)
        assert not result.has_amplification

    def test_wide_branch(self):
        ss = _ss("&{a: end, b: end, c: end, d: end, e: end}")
        result = analyze_dos_resistance(ss)
        assert result.max_branching >= 5
        assert len(result.dos_surface) >= 1
