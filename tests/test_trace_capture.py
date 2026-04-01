"""Tests for real execution trace capture + conformance (Step 80c).

These tests execute REAL Python operations (file I/O, iteration, sqlite3)
and validate that the captured traces conform to L3 session types.

This is the first time we validate theory on ACTUAL execution data.
"""

import os
import tempfile
import sqlite3
import pytest
from reticulate.trace_capture import (
    TracingProxy, CapturedTrace, TraceEntry,
    trace_file, trace_cursor,
    capture_file_read, capture_file_write,
    capture_iterator, capture_sqlite_query,
    eof_selection, readline_selection, bool_selection,
    fetchone_selection,
)
from reticulate.conformance_checker import check_conformance


# ---------------------------------------------------------------------------
# Selection mapping tests
# ---------------------------------------------------------------------------

class TestSelectionMappings:
    def test_eof_empty_string(self):
        assert eof_selection("") == "EOF"

    def test_eof_empty_bytes(self):
        assert eof_selection(b"") == "EOF"

    def test_eof_none(self):
        assert eof_selection(None) == "EOF"

    def test_eof_data(self):
        assert eof_selection("hello") == "data"

    def test_readline_eof(self):
        assert readline_selection("") == "EOF"

    def test_readline_data(self):
        assert readline_selection("line\n") == "data"

    def test_bool_true(self):
        assert bool_selection(True) == "TRUE"

    def test_bool_false(self):
        assert bool_selection(False) == "FALSE"

    def test_fetchone_none(self):
        assert fetchone_selection(None) == "NONE"

    def test_fetchone_row(self):
        assert fetchone_selection((1, "test")) == "ROW"


# ---------------------------------------------------------------------------
# TracingProxy tests
# ---------------------------------------------------------------------------

class TestTracingProxy:
    def test_wraps_method_calls(self):
        proxy = TracingProxy([1, 2, 3], name="list")
        proxy.append(4)
        trace = proxy.get_trace()
        assert len(trace.entries) == 1
        assert trace.entries[0].method == "append"

    def test_returns_real_values(self):
        proxy = TracingProxy([1, 2, 3], name="list")
        result = proxy.pop()
        assert result == 3

    def test_selection_mapping(self):
        proxy = TracingProxy({"a": 1}, name="dict",
                             selection_map={"get": lambda v: "FOUND" if v else "MISS"})
        proxy.get("a")
        proxy.get("z")
        trace = proxy.get_trace()
        assert trace.entries[0].selection_label == "FOUND"
        assert trace.entries[1].selection_label == "MISS"

    def test_as_labels(self):
        proxy = TracingProxy({"a": 1}, name="dict",
                             selection_map={"get": lambda v: "FOUND" if v else "MISS"})
        proxy.get("a")
        proxy.get("z")
        labels = proxy.get_trace().as_labels()
        assert labels == ["get", "FOUND", "get", "MISS"]

    def test_close_marks_complete(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
            tmp.write("test")
            tmp_path = tmp.name

        try:
            f = trace_file(open(tmp_path, "r"), name="file")
            f.read()
            f.close()
            trace = f.get_trace()
            assert trace.complete
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# REAL file I/O traces
# ---------------------------------------------------------------------------

class TestRealFileTraces:
    """Execute REAL file operations and check conformance."""

    def _make_file(self, content: str) -> str:
        tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt")
        tmp.write(content)
        tmp.close()
        return tmp.name

    def test_read_small_file(self):
        path = self._make_file("hello world")
        try:
            trace = capture_file_read(path)
            labels = trace.as_labels()
            assert trace.complete
            # Should have: read→data, ..., read→EOF, close
            assert "data" in labels
            assert "EOF" in labels
            assert labels[-1] == "close"
        finally:
            os.unlink(path)

    def test_read_empty_file(self):
        path = self._make_file("")
        try:
            trace = capture_file_read(path)
            labels = trace.as_labels()
            assert trace.complete
            assert "EOF" in labels
        finally:
            os.unlink(path)

    def test_write_file(self):
        path = tempfile.mktemp(suffix=".txt")
        try:
            trace = capture_file_write(path, "test content")
            assert trace.complete
            labels = trace.as_labels()
            assert "write" in labels
            assert "close" in labels
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_file_read_conforms_to_l3(self):
        """Real file read trace should conform to L3 file protocol."""
        FILE_L3 = "rec X . &{read: +{data: X, EOF: &{close: end}}, close: end}"
        path = self._make_file("hello")
        try:
            trace = capture_file_read(path)
            labels = trace.as_labels()
            result = check_conformance(FILE_L3, labels)
            assert result.conforms, f"Real file trace violates L3: {result.violations}"
            assert result.is_complete
        finally:
            os.unlink(path)

    def test_file_read_empty_conforms(self):
        """Empty file read should also conform."""
        FILE_L3 = "rec X . &{read: +{data: X, EOF: &{close: end}}, close: end}"
        path = self._make_file("")
        try:
            trace = capture_file_read(path)
            labels = trace.as_labels()
            result = check_conformance(FILE_L3, labels)
            assert result.conforms
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# REAL iterator traces
# ---------------------------------------------------------------------------

class TestRealIteratorTraces:
    """Execute REAL iteration and check conformance."""

    def test_iterate_list(self):
        trace = capture_iterator([10, 20, 30])
        labels = trace.as_labels()
        assert trace.complete
        assert labels.count("value") == 3
        assert "StopIteration" in labels

    def test_iterate_empty(self):
        trace = capture_iterator([])
        labels = trace.as_labels()
        assert trace.complete
        assert "StopIteration" in labels

    def test_iterate_string(self):
        trace = capture_iterator("abc")
        assert trace.complete
        assert len([e for e in trace.entries if e.selection_label == "value"]) == 3

    def test_iterator_conforms_to_l3(self):
        """Real iteration should conform to L3 iterator protocol."""
        ITER_L3 = "rec X . &{next: +{value: X, StopIteration: end}}"
        trace = capture_iterator([1, 2, 3])
        labels = trace.as_labels()
        result = check_conformance(ITER_L3, labels)
        assert result.conforms, f"Real iterator trace violates L3: {result.violations}"
        assert result.is_complete

    def test_empty_iterator_conforms(self):
        ITER_L3 = "rec X . &{next: +{value: X, StopIteration: end}}"
        trace = capture_iterator([])
        labels = trace.as_labels()
        result = check_conformance(ITER_L3, labels)
        assert result.conforms
        assert result.is_complete


# ---------------------------------------------------------------------------
# REAL SQLite traces
# ---------------------------------------------------------------------------

class TestRealSQLiteTraces:
    """Execute REAL SQLite operations and capture traces."""

    def _make_db(self) -> str:
        path = tempfile.mktemp(suffix=".db")
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO test VALUES (1, 'alice')")
        conn.execute("INSERT INTO test VALUES (2, 'bob')")
        conn.commit()
        conn.close()
        return path

    def test_sqlite_query_trace(self):
        path = self._make_db()
        try:
            trace = capture_sqlite_query(path, "SELECT * FROM test")
            assert trace.complete
            methods = trace.as_method_only()
            assert "cursor" in methods
            assert "close" in methods
        finally:
            os.unlink(path)

    def test_sqlite_empty_query(self):
        path = self._make_db()
        try:
            trace = capture_sqlite_query(path, "SELECT * FROM test WHERE id > 100")
            assert trace.complete
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Cross-validation: real traces vs L2 traces
# ---------------------------------------------------------------------------

class TestRealVsL2:
    """Verify that real traces include selection labels that L2 traces miss."""

    def test_real_trace_has_selections(self):
        """Real file trace should have data/EOF selections."""
        path = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt")
        path.write("test")
        path.close()
        try:
            trace = capture_file_read(path.name)
            labels = trace.as_labels()
            methods_only = trace.as_method_only()
            # L3 trace has selections interleaved
            assert len(labels) > len(methods_only)
            # Selection labels present
            assert any(l in ("data", "EOF") for l in labels)
        finally:
            os.unlink(path.name)

    def test_l2_trace_would_violate(self):
        """L2 trace (methods only, no selections) should violate L3 protocol."""
        FILE_L3 = "rec X . &{read: +{data: X, EOF: &{close: end}}, close: end}"
        # L2 trace: no selections
        l2_trace = ["read", "read", "close"]
        result = check_conformance(FILE_L3, l2_trace)
        assert not result.conforms  # L2 trace violates L3 protocol!
