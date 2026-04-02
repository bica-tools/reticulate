"""Tests for precondition → selection extractor (Step 97e)."""

import os
import pytest
from reticulate.precondition_extractor import (
    extract_preconditions,
    group_into_phases,
    propose_selection,
    analyze_java_file,
    Precondition,
    PhaseGroup,
    SelectionProposal,
)

# ---------------------------------------------------------------------------
# The real StatsAccumulator source (or path to it)
# ---------------------------------------------------------------------------

STATS_PATH = "/tmp/guava/guava/src/com/google/common/math/StatsAccumulator.java"

SIMPLE_JAVA = '''
public class Counter {
    private int count = 0;

    public void increment() { count++; }

    public int getCount() { return count; }

    public double average() {
        checkState(count != 0);
        return total / count;
    }

    public double variance() {
        checkState(count > 1);
        return sumSq / (count - 1);
    }

    public void reset() { count = 0; }
}
'''


class TestPreconditionExtraction:
    def test_extract_checkstate(self):
        pcs = extract_preconditions(SIMPLE_JAVA)
        assert len(pcs) >= 2
        methods = [p.method_name for p in pcs]
        assert "average" in methods
        assert "variance" in methods

    def test_condition_parsing(self):
        pcs = extract_preconditions(SIMPLE_JAVA)
        avg_pc = [p for p in pcs if p.method_name == "average"][0]
        assert avg_pc.field_name == "count"
        assert avg_pc.operator == "!="
        assert avg_pc.threshold == "0"

    def test_variance_needs_gt_1(self):
        pcs = extract_preconditions(SIMPLE_JAVA)
        var_pc = [p for p in pcs if p.method_name == "variance"][0]
        assert var_pc.operator == ">"
        assert var_pc.threshold == "1"


class TestPhaseGrouping:
    def test_groups_by_threshold(self):
        pcs = extract_preconditions(SIMPLE_JAVA)
        public_methods = ["increment", "getCount", "average", "variance", "reset"]
        phases, always = group_into_phases(pcs, public_methods)
        assert len(phases) >= 2  # count!=0 and count>1
        assert "increment" in always or "getCount" in always

    def test_always_available(self):
        pcs = extract_preconditions(SIMPLE_JAVA)
        public_methods = ["increment", "getCount", "average", "variance", "reset"]
        _, always = group_into_phases(pcs, public_methods)
        # Methods without preconditions
        assert "increment" in always
        assert "getCount" in always

    def test_phase_names(self):
        pcs = extract_preconditions(SIMPLE_JAVA)
        phases, _ = group_into_phases(pcs, ["increment", "getCount", "average", "variance"])
        names = [p.phase_name for p in phases]
        assert "NON_EMPTY" in names or any("count" in n.lower() for n in names)


class TestSelectionProposal:
    def test_proposes_enum(self):
        result = propose_selection("Counter", SIMPLE_JAVA,
                                   ["increment", "getCount", "average", "variance", "reset"])
        assert result is not None
        assert result.enum_name == "CounterPhase"
        assert len(result.enum_values) >= 3  # INITIAL + 2 phases
        assert "INITIAL" in result.enum_values

    def test_session_type_generated(self):
        result = propose_selection("Counter", SIMPLE_JAVA,
                                   ["increment", "getCount", "average", "variance", "reset"])
        assert result is not None
        assert "rec X" in result.session_type
        assert "phase" in result.session_type
        assert "NON_EMPTY" in result.session_type or "NONZERO" in result.session_type

    def test_refactored_code(self):
        result = propose_selection("Counter", SIMPLE_JAVA,
                                   ["increment", "getCount", "average", "variance", "reset"])
        assert result is not None
        assert "enum" in result.refactored_code
        assert "phase()" in result.refactored_code

    def test_session_type_parseable(self):
        """Generated session type should be parseable."""
        from reticulate.parser import parse
        result = propose_selection("Counter", SIMPLE_JAVA,
                                   ["increment", "getCount", "average", "variance", "reset"])
        assert result is not None
        ast = parse(result.session_type)
        assert ast is not None

    def test_session_type_is_lattice(self):
        from reticulate.parser import parse
        from reticulate.statespace import build_statespace
        from reticulate.lattice import check_lattice, check_distributive
        result = propose_selection("Counter", SIMPLE_JAVA,
                                   ["increment", "getCount", "average", "variance", "reset"])
        ast = parse(result.session_type)
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        assert lr.is_lattice
        dr = check_distributive(ss)
        assert dr.is_distributive  # phase-based should be distributive


class TestRealStatsAccumulator:
    @pytest.mark.skipif(
        not os.path.exists(STATS_PATH),
        reason="Guava source not available"
    )
    def test_extract_from_real_source(self):
        result = analyze_java_file(STATS_PATH)
        assert result is not None
        assert result.class_name == "StatsAccumulator"

    @pytest.mark.skipif(
        not os.path.exists(STATS_PATH),
        reason="Guava source not available"
    )
    def test_finds_count_preconditions(self):
        result = analyze_java_file(STATS_PATH)
        assert result is not None
        assert result.field_name == "count"
        assert len(result.phases) >= 2

    @pytest.mark.skipif(
        not os.path.exists(STATS_PATH),
        reason="Guava source not available"
    )
    def test_mean_in_nonempty_phase(self):
        result = analyze_java_file(STATS_PATH)
        assert result is not None
        nonempty = [p for p in result.phases if "NON_EMPTY" in p.phase_name
                    or p.threshold == "0"]
        assert len(nonempty) >= 1
        assert "mean" in nonempty[0].methods

    @pytest.mark.skipif(
        not os.path.exists(STATS_PATH),
        reason="Guava source not available"
    )
    def test_sampleVariance_in_multi_phase(self):
        result = analyze_java_file(STATS_PATH)
        assert result is not None
        multi = [p for p in result.phases if p.threshold == "1"]
        assert len(multi) >= 1
        assert "sampleVariance" in multi[0].methods

    @pytest.mark.skipif(
        not os.path.exists(STATS_PATH),
        reason="Guava source not available"
    )
    def test_proposes_enum_and_selector(self):
        result = analyze_java_file(STATS_PATH)
        assert result is not None
        assert "enum" in result.refactored_code
        assert "phase()" in result.refactored_code
        print("\n" + result.refactored_code)

    @pytest.mark.skipif(
        not os.path.exists(STATS_PATH),
        reason="Guava source not available"
    )
    def test_generated_session_type(self):
        from reticulate.parser import parse
        from reticulate.statespace import build_statespace
        from reticulate.lattice import check_lattice, check_distributive
        result = analyze_java_file(STATS_PATH)
        assert result is not None
        ast = parse(result.session_type)
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        assert lr.is_lattice
        dr = check_distributive(ss)
        print(f"\nStatsAccumulator session type: {result.session_type[:100]}...")
        print(f"States: {len(ss.states)}, Classification: {dr.classification}")
        print(f"Phases: {result.enum_values}")
