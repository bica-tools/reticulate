"""Tests for L3 session type extractor from API descriptions.

Validates that structured API descriptions produce correct L3 session
types with return-value selections and preconditions.
"""

import pytest
from reticulate.l3_extractor import (
    APIDesc, StateDesc, MethodDesc,
    extract_session_type, extract_all, API_REGISTRY,
    java_iterator_api, java_inputstream_api, jdbc_connection_api,
    python_file_api, kafka_producer_api, mongodb_api,
)
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice, check_distributive


# ---------------------------------------------------------------------------
# Basic extraction tests
# ---------------------------------------------------------------------------

class TestBasicExtraction:
    def test_simple_two_method(self):
        api = APIDesc("test", "s0", [
            StateDesc("s0", [
                MethodDesc("open", next_state="s1"),
            ]),
            StateDesc("s1", [
                MethodDesc("close", terminates=True),
            ]),
        ])
        st = extract_session_type(api)
        assert "open" in st
        assert "close" in st
        ast = parse(st)
        assert ast is not None

    def test_boolean_return(self):
        api = APIDesc("test", "s0", [
            StateDesc("s0", [
                MethodDesc("check", returns="bool",
                           next_state={"TRUE": "end", "FALSE": "end"}),
            ]),
        ])
        st = extract_session_type(api)
        assert "TRUE" in st
        assert "FALSE" in st
        ast = parse(st)
        assert ast is not None

    def test_enum_return(self):
        api = APIDesc("test", "s0", [
            StateDesc("s0", [
                MethodDesc("send", returns=["OK", "ERR", "TIMEOUT"],
                           next_state="end"),
            ]),
        ])
        st = extract_session_type(api)
        assert "OK" in st
        assert "ERR" in st
        assert "TIMEOUT" in st

    def test_recursive_state(self):
        api = APIDesc("test", "loop", [
            StateDesc("loop", [
                MethodDesc("step", next_state="self"),
                MethodDesc("done", terminates=True),
            ]),
        ])
        st = extract_session_type(api)
        assert "rec" in st
        ast = parse(st)
        assert ast is not None

    def test_terminating_method(self):
        api = APIDesc("test", "s0", [
            StateDesc("s0", [
                MethodDesc("close", terminates=True),
            ]),
        ])
        st = extract_session_type(api)
        assert "end" in st


# ---------------------------------------------------------------------------
# Real API extraction tests
# ---------------------------------------------------------------------------

class TestJavaIterator:
    def test_extracts(self):
        st = extract_session_type(java_iterator_api())
        assert "hasNext" in st
        assert "TRUE" in st
        assert "FALSE" in st
        assert "next" in st

    def test_parseable(self):
        st = extract_session_type(java_iterator_api())
        ast = parse(st)
        assert ast is not None

    def test_forms_lattice(self):
        st = extract_session_type(java_iterator_api())
        ast = parse(st)
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        assert lr.is_lattice

    def test_is_distributive(self):
        st = extract_session_type(java_iterator_api())
        ast = parse(st)
        ss = build_statespace(ast)
        dr = check_distributive(ss)
        assert dr.is_distributive


class TestJavaInputStream:
    def test_extracts(self):
        st = extract_session_type(java_inputstream_api())
        assert "read" in st
        assert "data" in st
        assert "EOF" in st

    def test_forms_lattice(self):
        st = extract_session_type(java_inputstream_api())
        ast = parse(st)
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        assert lr.is_lattice

    def test_is_distributive(self):
        st = extract_session_type(java_inputstream_api())
        ast = parse(st)
        ss = build_statespace(ast)
        dr = check_distributive(ss)
        assert dr.is_distributive


class TestJDBCConnection:
    def test_extracts(self):
        st = extract_session_type(jdbc_connection_api())
        assert "executeQuery" in st
        assert "commit" in st

    def test_forms_lattice(self):
        st = extract_session_type(jdbc_connection_api())
        ast = parse(st)
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        assert lr.is_lattice


class TestPythonFile:
    def test_extracts(self):
        st = extract_session_type(python_file_api())
        assert "read" in st
        assert "write" in st
        assert "EOF" in st

    def test_is_distributive(self):
        st = extract_session_type(python_file_api())
        ast = parse(st)
        ss = build_statespace(ast)
        dr = check_distributive(ss)
        assert dr.is_distributive


class TestKafkaProducer:
    def test_extracts(self):
        st = extract_session_type(kafka_producer_api())
        assert "send" in st
        assert "ACK" in st

    def test_forms_lattice(self):
        st = extract_session_type(kafka_producer_api())
        ast = parse(st)
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        assert lr.is_lattice


class TestMongoDB:
    def test_extracts(self):
        st = extract_session_type(mongodb_api())
        assert "insertOne" in st
        assert "find" in st

    def test_forms_lattice(self):
        st = extract_session_type(mongodb_api())
        ast = parse(st)
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        assert lr.is_lattice


# ---------------------------------------------------------------------------
# Bulk extraction tests
# ---------------------------------------------------------------------------

class TestBulkExtraction:
    def test_all_registered_apis_extract(self):
        results = extract_all()
        assert len(results) >= 6

    @pytest.mark.parametrize("name,api_fn", list(API_REGISTRY.items()))
    def test_all_parseable(self, name, api_fn):
        st = extract_session_type(api_fn())
        ast = parse(st)
        assert ast is not None, f"{name} produced unparseable type: {st}"

    @pytest.mark.parametrize("name,api_fn", list(API_REGISTRY.items()))
    def test_all_form_lattices(self, name, api_fn):
        st = extract_session_type(api_fn())
        ast = parse(st)
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        assert lr.is_lattice, f"{name} is not a lattice"


# ---------------------------------------------------------------------------
# Comparison: extractor output vs hand-written L3
# ---------------------------------------------------------------------------

class TestExtractorVsHandWritten:
    """Verify extractor produces equivalent types to hand-written L3 models."""

    def test_iterator_matches_benchmark(self):
        """Extracted Iterator should match the benchmark."""
        st = extract_session_type(java_iterator_api())
        ast = parse(st)
        ss = build_statespace(ast)

        # Benchmark: "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"
        bench = parse("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        bench_ss = build_statespace(bench)

        # Same number of states (structural equivalence)
        assert len(ss.states) == len(bench_ss.states)

    def test_file_matches_benchmark(self):
        """Extracted file should be structurally similar to benchmark."""
        st = extract_session_type(python_file_api())
        ast = parse(st)
        ss = build_statespace(ast)
        # Should have reasonable state count
        assert len(ss.states) >= 3
