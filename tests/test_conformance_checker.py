"""Tests for client conformance checker (Step 80b).

Tests real client traces against L3 session types.
Validates violation detection, completeness checking, and batch analysis.
"""

import pytest
from reticulate.conformance_checker import (
    check_conformance, check_batch, check_against_api,
    format_report, ConformanceResult, BatchResult, Violation,
)


# ---------------------------------------------------------------------------
# The canonical session types for testing
# ---------------------------------------------------------------------------

ITERATOR = "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"
FILE = "&{open: rec X . &{read: +{data: X, EOF: &{close: end}}, close: end}}"
FILE_RW = "rec X . &{read: +{data: X, EOF: &{close: end}}, write: X, close: end}"


# ---------------------------------------------------------------------------
# Valid traces
# ---------------------------------------------------------------------------

class TestValidTraces:
    def test_iterator_normal(self):
        r = check_conformance(ITERATOR, ["hasNext", "TRUE", "next", "hasNext", "FALSE"])
        assert r.conforms
        assert r.is_complete

    def test_iterator_immediate_false(self):
        r = check_conformance(ITERATOR, ["hasNext", "FALSE"])
        assert r.conforms
        assert r.is_complete

    def test_iterator_multiple_elements(self):
        r = check_conformance(ITERATOR, [
            "hasNext", "TRUE", "next",
            "hasNext", "TRUE", "next",
            "hasNext", "TRUE", "next",
            "hasNext", "FALSE"
        ])
        assert r.conforms
        assert r.is_complete
        assert r.trace_length == 11

    def test_file_read_close(self):
        r = check_conformance(FILE, ["open", "read", "data", "read", "EOF", "close"])
        assert r.conforms
        assert r.is_complete

    def test_file_immediate_close(self):
        r = check_conformance(FILE, ["open", "close"])
        assert r.conforms
        assert r.is_complete

    def test_file_single_read(self):
        r = check_conformance(FILE, ["open", "read", "EOF", "close"])
        assert r.conforms
        assert r.is_complete

    def test_file_rw_write_then_read(self):
        r = check_conformance(FILE_RW, ["write", "read", "EOF", "close"])
        assert r.conforms
        assert r.is_complete


# ---------------------------------------------------------------------------
# Violations
# ---------------------------------------------------------------------------

class TestViolations:
    def test_iterator_next_without_hasNext(self):
        """Client calls next() without checking hasNext() first."""
        r = check_conformance(ITERATOR, ["next"])
        assert not r.conforms
        assert len(r.violations) == 1
        assert r.violations[0].method == "next"
        assert "hasNext" in r.violations[0].available_methods

    def test_iterator_next_after_false(self):
        """Client calls next() after hasNext() returned FALSE."""
        r = check_conformance(ITERATOR, ["hasNext", "FALSE", "next"])
        assert not r.conforms
        v = r.violations[0]
        assert v.step_index == 2
        assert v.method == "next"
        assert "Session has ended" in v.explanation

    def test_file_read_read_without_selection(self):
        """The L2 trace [read, read, close] — missing selection outcomes."""
        r = check_conformance(FILE, ["open", "read", "read", "close"])
        assert not r.conforms
        v = r.violations[0]
        assert v.step_index == 2
        assert v.method == "read"
        assert "selection outcome" in v.explanation or "data" in str(v.available_methods)

    def test_file_write_on_readonly(self):
        """Write is not available on the FILE protocol (read-only)."""
        r = check_conformance(FILE, ["open", "write"])
        assert not r.conforms
        assert r.violations[0].method == "write"

    def test_double_close(self):
        """Closing a file twice is a violation."""
        r = check_conformance(FILE, ["open", "close", "close"])
        assert not r.conforms
        v = r.violations[0]
        assert v.step_index == 2
        assert "Session has ended" in v.explanation

    def test_method_after_end(self):
        """Any method after end is a violation."""
        r = check_conformance(ITERATOR, ["hasNext", "FALSE", "hasNext"])
        assert not r.conforms

    def test_wrong_selection_value(self):
        """Client provides a selection value that doesn't exist."""
        r = check_conformance(ITERATOR, ["hasNext", "MAYBE"])
        assert not r.conforms
        assert r.violations[0].method == "MAYBE"


# ---------------------------------------------------------------------------
# Incomplete traces
# ---------------------------------------------------------------------------

class TestIncompleteTraces:
    def test_iterator_stopped_early(self):
        r = check_conformance(ITERATOR, ["hasNext", "TRUE", "next"])
        assert r.conforms  # no violation
        assert not r.is_complete  # but session not finished

    def test_file_opened_not_closed(self):
        r = check_conformance(FILE, ["open", "read", "data"])
        assert r.conforms
        assert not r.is_complete

    def test_empty_trace(self):
        r = check_conformance(ITERATOR, [])
        assert r.conforms
        assert not r.is_complete
        assert r.steps_completed == 0


# ---------------------------------------------------------------------------
# Batch checking
# ---------------------------------------------------------------------------

class TestBatchChecking:
    def test_mixed_batch(self):
        traces = [
            ["hasNext", "TRUE", "next", "hasNext", "FALSE"],  # valid, complete
            ["hasNext", "TRUE", "next"],                        # valid, incomplete
            ["next"],                                            # violation
            ["hasNext", "FALSE"],                               # valid, complete
        ]
        batch = check_batch(ITERATOR, traces)
        assert batch.total_traces == 4
        assert batch.conforming == 2  # complete conforming
        assert batch.violating == 1
        assert batch.incomplete == 1
        assert batch.conformance_rate == 0.5

    def test_all_valid_batch(self):
        traces = [
            ["hasNext", "TRUE", "next", "hasNext", "FALSE"],
            ["hasNext", "FALSE"],
        ]
        batch = check_batch(ITERATOR, traces)
        assert batch.conformance_rate == 1.0
        assert batch.violating == 0

    def test_report_formatting(self):
        traces = [
            ["hasNext", "TRUE", "next", "hasNext", "FALSE"],
            ["next"],  # violation
        ]
        batch = check_batch(ITERATOR, traces)
        report = format_report(batch)
        assert "CONFORMANCE CHECK" in report
        assert "PASS" in report
        assert "FAIL" in report


# ---------------------------------------------------------------------------
# L3 extractor API integration
# ---------------------------------------------------------------------------

class TestAPIIntegration:
    def test_check_against_iterator_api(self):
        """L3 extractor Iterator is non-recursive: hasNext→TRUE→next→end."""
        batch = check_against_api("java_iterator", [
            ["hasNext", "TRUE", "next"],  # single iteration (extractor is non-recursive)
            ["hasNext", "FALSE"],
        ])
        assert batch.conforming >= 1

    def test_check_violation_against_api(self):
        batch = check_against_api("java_iterator", [
            ["next"],  # violation: no hasNext first
        ])
        assert batch.violating == 1

    def test_unknown_api_raises(self):
        with pytest.raises(ValueError, match="Unknown API"):
            check_against_api("nonexistent", [["foo"]])


# ---------------------------------------------------------------------------
# Real-world client scenarios
# ---------------------------------------------------------------------------

class TestRealWorldScenarios:
    """Simulate real client bugs caught by conformance checking."""

    def test_forgot_to_check_hasNext(self):
        """Common Java bug: calling next() without hasNext()."""
        r = check_conformance(ITERATOR, ["next"])
        assert not r.conforms
        assert "hasNext" in r.violations[0].available_methods

    def test_ignored_eof(self):
        """Common file bug: reading past EOF."""
        r = check_conformance(FILE, [
            "open", "read", "EOF", "read"  # reading after EOF
        ])
        assert not r.conforms
        assert r.violations[0].step_index == 3

    def test_missing_return_check(self):
        """Bug: calling methods without checking return value selection."""
        r = check_conformance(FILE, [
            "open", "read", "close"  # forgot to handle data/EOF
        ])
        assert not r.conforms
        # After read, must get data or EOF before anything else
        v = r.violations[0]
        assert v.step_index == 2
        assert v.method == "close"

    def test_resource_leak(self):
        """Bug: opened file, did work, but never closed."""
        r = check_conformance(FILE, [
            "open", "read", "data", "read", "data"
        ])
        assert r.conforms  # no violation
        assert not r.is_complete  # but resource leak — never closed

    def test_correct_usage(self):
        """Correct: check, use, handle all outcomes, close."""
        r = check_conformance(FILE, [
            "open", "read", "data", "read", "data", "read", "EOF", "close"
        ])
        assert r.conforms
        assert r.is_complete
        assert r.coverage > 0


# ---------------------------------------------------------------------------
# Violation explanation quality
# ---------------------------------------------------------------------------

class TestExplanations:
    def test_selection_missing_explanation(self):
        """Explanation should mention 'selection outcome' when at selection point."""
        r = check_conformance(ITERATOR, ["hasNext", "next"])
        assert not r.conforms
        # After hasNext, we're at a selection point: TRUE or FALSE
        v = r.violations[0]
        assert "TRUE" in v.available_methods or "FALSE" in v.available_methods

    def test_session_ended_explanation(self):
        r = check_conformance(ITERATOR, ["hasNext", "FALSE", "next"])
        assert "ended" in r.violations[0].explanation.lower()

    def test_violation_str(self):
        r = check_conformance(ITERATOR, ["next"])
        v_str = str(r.violations[0])
        assert "next" in v_str
        assert "not valid" in v_str

    def test_summary_pass(self):
        r = check_conformance(ITERATOR, ["hasNext", "FALSE"])
        assert "PASS" in r.summary()

    def test_summary_fail(self):
        r = check_conformance(ITERATOR, ["next"])
        assert "FAIL" in r.summary()

    def test_summary_incomplete(self):
        r = check_conformance(ITERATOR, ["hasNext", "TRUE"])
        assert "INCOMPLETE" in r.summary()
