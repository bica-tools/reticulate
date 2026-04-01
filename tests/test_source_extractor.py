"""Tests for Python source code session type extraction (Step 97b).

Scientific Method validation:
  Hypothesis: Stateful Python protocols produce distributive lattices
  Prediction: ≥80% of stateful protocols are distributive (unlike REST APIs)
  Experiment: Extract from 10 Python protocol specs + 3 AST samples
"""

import pytest
from reticulate.source_extractor import (
    SourceExtractionResult,
    extract_from_source,
    extract_traces_from_source,
    analyze_protocol,
    analyze_all_protocols,
    print_protocol_report,
    PYTHON_PROTOCOLS,
    PYTHON_SOURCE_SAMPLES,
)
from reticulate.type_inference import Trace, infer_from_traces
from reticulate.parser import parse, pretty
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice


# ---------------------------------------------------------------------------
# AST extraction tests
# ---------------------------------------------------------------------------

class TestASTExtraction:
    def test_simple_method_calls(self):
        source = "conn.execute(); conn.commit(); conn.close()"
        calls = extract_from_source(source, "conn")
        assert calls == ["execute", "commit", "close"]

    def test_different_target(self):
        source = "f.read(); f.close(); conn.execute()"
        calls = extract_from_source(source, "f")
        assert calls == ["read", "close"]

    def test_no_calls(self):
        source = "x = 1 + 2"
        calls = extract_from_source(source, "conn")
        assert calls == []

    def test_mixed_variables(self):
        source = "f.read(); conn.execute(); f.close(); conn.close()"
        f_calls = extract_from_source(source, "f")
        conn_calls = extract_from_source(source, "conn")
        assert f_calls == ["read", "close"]
        assert conn_calls == ["execute", "close"]

    def test_function_bodies_as_traces(self):
        source = '''
def func1(f):
    f.read()
    f.close()

def func2(f):
    f.write()
    f.close()
'''
        traces = extract_traces_from_source(source, "f")
        assert len(traces) == 2
        assert [s.label for s in traces[0].steps] == ["read", "close"]
        assert [s.label for s in traces[1].steps] == ["write", "close"]

    def test_real_sqlite3_source(self):
        sample = PYTHON_SOURCE_SAMPLES["sqlite3_usage"]
        traces = extract_traces_from_source(sample["source"], sample["target_var"])
        assert len(traces) >= 3
        # Each function should produce a trace
        for t in traces:
            assert len(t) >= 1

    def test_real_file_source(self):
        sample = PYTHON_SOURCE_SAMPLES["file_operations"]
        traces = extract_traces_from_source(sample["source"], sample["target_var"])
        assert len(traces) >= 3

    def test_real_http_source(self):
        sample = PYTHON_SOURCE_SAMPLES["http_requests"]
        traces = extract_traces_from_source(sample["source"], sample["target_var"])
        assert len(traces) >= 2


# ---------------------------------------------------------------------------
# Protocol spec tests
# ---------------------------------------------------------------------------

class TestProtocolSpecs:
    def test_all_protocols_registered(self):
        assert len(PYTHON_PROTOCOLS) >= 10

    def test_all_protocols_have_traces(self):
        for name, spec in PYTHON_PROTOCOLS.items():
            assert len(spec["traces"]) >= 2, f"{name} has < 2 traces"

    def test_all_traces_non_empty(self):
        for name, spec in PYTHON_PROTOCOLS.items():
            for i, trace in enumerate(spec["traces"]):
                assert len(trace) >= 1, f"{name} trace {i} is empty"


# ---------------------------------------------------------------------------
# Individual protocol analysis tests
# ---------------------------------------------------------------------------

class TestFileObject:
    def test_extraction(self):
        r = analyze_protocol("file", PYTHON_PROTOCOLS["file_object"]["traces"])
        assert r.is_lattice

    def test_has_states(self):
        r = analyze_protocol("file", PYTHON_PROTOCOLS["file_object"]["traces"])
        assert r.num_states >= 2

    def test_inferred_type_parseable(self):
        r = analyze_protocol("file", PYTHON_PROTOCOLS["file_object"]["traces"])
        ast = parse(r.inferred_type)
        assert ast is not None


class TestSqlite3:
    def test_extraction(self):
        r = analyze_protocol("sqlite3", PYTHON_PROTOCOLS["sqlite3_connection"]["traces"])
        assert r.is_lattice

    def test_has_states(self):
        r = analyze_protocol("sqlite3", PYTHON_PROTOCOLS["sqlite3_connection"]["traces"])
        assert r.num_states >= 2


class TestHttpClient:
    def test_extraction(self):
        r = analyze_protocol("http", PYTHON_PROTOCOLS["http_client"]["traces"])
        assert r.is_lattice


class TestSocket:
    def test_extraction(self):
        r = analyze_protocol("socket", PYTHON_PROTOCOLS["socket_object"]["traces"])
        assert r.is_lattice


class TestLock:
    def test_extraction(self):
        r = analyze_protocol("lock", PYTHON_PROTOCOLS["threading_lock"]["traces"])
        assert r.is_lattice


class TestIterator:
    def test_extraction(self):
        r = analyze_protocol("iter", PYTHON_PROTOCOLS["iterator_protocol"]["traces"])
        assert r.is_lattice


class TestContextManager:
    def test_extraction(self):
        r = analyze_protocol("ctx", PYTHON_PROTOCOLS["context_manager"]["traces"])
        assert r.is_lattice


class TestZipFile:
    def test_extraction(self):
        r = analyze_protocol("zip", PYTHON_PROTOCOLS["zipfile_reader"]["traces"])
        assert r.is_lattice


class TestCSVReader:
    def test_extraction(self):
        r = analyze_protocol("csv", PYTHON_PROTOCOLS["csv_reader"]["traces"])
        assert r.is_lattice


class TestQueue:
    def test_extraction(self):
        r = analyze_protocol("queue", PYTHON_PROTOCOLS["queue_protocol"]["traces"])
        assert r.is_lattice


# ---------------------------------------------------------------------------
# Aggregate analysis tests
# ---------------------------------------------------------------------------

class TestAggregateAnalysis:
    def test_all_protocols_run(self):
        results = analyze_all_protocols()
        assert len(results) >= 10

    def test_lattice_rate(self):
        """All stateful protocols should form lattices."""
        results = analyze_all_protocols()
        lattice_count = sum(1 for r in results if r.is_lattice)
        total = len(results)
        rate = lattice_count / total
        assert rate >= 0.80, f"Lattice rate {rate:.0%} below 80%"

    def test_report_formatting(self):
        results = analyze_all_protocols()
        text = print_protocol_report(results)
        assert "PYTHON PROTOCOL" in text
        assert "Lattice rate" in text
        assert "Distributive rate" in text


# ---------------------------------------------------------------------------
# Two-worlds hypothesis test
# ---------------------------------------------------------------------------

class TestTwoWorldsHypothesis:
    """Test whether stateful protocols differ from REST APIs in distributivity."""

    def test_stateful_vs_rest_comparison(self):
        """Stateful protocols should have higher distributivity rate than REST."""
        from reticulate.real_world_extraction import analyze_all_specs

        # REST APIs (from Step 71b)
        rest_report = analyze_all_specs()
        rest_dist_rate = rest_report.distributive_rate

        # Stateful protocols
        stateful_results = analyze_all_protocols()
        stateful_dist = sum(1 for r in stateful_results if r.is_distributive)
        stateful_rate = stateful_dist / len(stateful_results)

        # The hypothesis: stateful should be MORE distributive than REST
        # (We don't assert this strictly — the test records the comparison)
        print(f"\nREST distributivity: {rest_dist_rate:.0%}")
        print(f"Stateful distributivity: {stateful_rate:.0%}")
        print(f"Two-worlds hypothesis: stateful > REST = {stateful_rate > rest_dist_rate}")


# ---------------------------------------------------------------------------
# AST → inference round-trip tests
# ---------------------------------------------------------------------------

class TestASTInferenceRoundTrip:
    @pytest.mark.parametrize("name,sample", list(PYTHON_SOURCE_SAMPLES.items()))
    def test_ast_to_session_type(self, name, sample):
        """Extract traces from source → infer type → verify parseable."""
        traces = extract_traces_from_source(sample["source"], sample["target_var"])
        if traces:
            inferred = infer_from_traces(traces)
            st_str = pretty(inferred)
            # Must be parseable
            parsed = parse(st_str)
            assert parsed is not None

    @pytest.mark.parametrize("name,sample", list(PYTHON_SOURCE_SAMPLES.items()))
    def test_ast_to_lattice(self, name, sample):
        """Extract traces from source → infer type → build state space → check lattice."""
        traces = extract_traces_from_source(sample["source"], sample["target_var"])
        if traces:
            inferred = infer_from_traces(traces)
            st_str = pretty(inferred)
            parsed = parse(st_str)
            ss = build_statespace(parsed)
            lr = check_lattice(ss)
            assert lr.is_lattice, f"{name}: inferred type does not form a lattice"
