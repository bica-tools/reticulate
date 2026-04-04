"""Tests for real trace conformance checking (TOPLAS validation Week 4).

Validates that:
1. Real valid traces conform to extracted session types
2. Deliberately broken traces are detected as violations
3. Real bug patterns (StackOverflow-sourced) are caught
4. The conformance checker pinpoints violation location
"""

import pytest
from reticulate.real_trace_conformance import (
    capture_sqlite3_valid_traces,
    capture_sqlite3_broken_traces,
    capture_sqlite3_real_bug_traces,
    capture_file_valid_traces,
    capture_file_broken_traces,
    capture_iterator_valid_traces,
    capture_iterator_broken_traces,
    check_trace,
    run_conformance_suite,
    run_all_suites,
    format_conformance_report,
    ConformanceSuiteResult,
    TraceCheckResult,
)
from reticulate.python_api_extractor import extract_session_type as extract_python
from reticulate.java_api_extractor import extract_session_type as extract_java


# ---------------------------------------------------------------------------
# SQLite3 conformance
# ---------------------------------------------------------------------------

class TestSqlite3Conformance:
    @pytest.fixture
    def session_type(self):
        return extract_python("sqlite3.Connection").inferred_type

    def test_valid_traces_all_conform(self, session_type):
        traces = capture_sqlite3_valid_traces()
        for name, labels in traces:
            r = check_trace(name, labels, session_type, "valid", True)
            assert r.conforms, f"{name} should conform: {r.violations}"

    def test_valid_traces_are_complete(self, session_type):
        traces = capture_sqlite3_valid_traces()
        for name, labels in traces:
            r = check_trace(name, labels, session_type, "valid", True)
            assert r.is_complete, f"{name} should be complete"

    def test_broken_commit_first_detected(self, session_type):
        r = check_trace("broken", ["commit", "OK", "close"],
                        session_type, "broken", False)
        assert not r.conforms
        assert r.steps_completed == 0  # fails at very first step

    def test_broken_double_close_detected(self, session_type):
        r = check_trace("broken", ["close", "close"],
                        session_type, "broken", False)
        assert not r.conforms

    def test_broken_execute_after_close_detected(self, session_type):
        r = check_trace("broken", ["close", "execute", "ROWS_AFFECTED"],
                        session_type, "broken", False)
        assert not r.conforms

    def test_broken_rollback_first_detected(self, session_type):
        r = check_trace("broken", ["rollback", "OK", "close"],
                        session_type, "broken", False)
        assert not r.conforms

    def test_broken_cursor_skip_execute(self, session_type):
        r = check_trace("broken", ["cursor", "CURSOR", "commit", "OK", "close"],
                        session_type, "broken", False)
        assert not r.conforms

    def test_all_broken_detected(self, session_type):
        traces = capture_sqlite3_broken_traces()
        for name, labels in traces:
            r = check_trace(name, labels, session_type, "broken", False)
            assert not r.conforms, f"{name} should be detected as violation"

    # Real bug patterns
    def test_bug_forgot_commit(self, session_type):
        """SO #6318866: why aren't my changes saved?"""
        r = check_trace("bug", ["execute", "ROWS_AFFECTED", "close"],
                        session_type, "real_bug", False)
        assert not r.conforms
        assert "close" in r.violations[0]  # close not available after execute

    def test_bug_operate_after_rollback(self, session_type):
        r = check_trace("bug",
                        ["execute", "ROWS_AFFECTED", "rollback", "OK",
                         "commit", "OK", "close"],
                        session_type, "real_bug", False)
        assert not r.conforms

    def test_bug_rollback_then_commit(self, session_type):
        r = check_trace("bug",
                        ["cursor", "CURSOR", "executemany", "ROWS_AFFECTED",
                         "rollback", "OK", "commit", "OK", "close"],
                        session_type, "real_bug", False)
        assert not r.conforms

    def test_bug_rollback_close(self, session_type):
        """Session type requires execute after rollback, not close."""
        r = check_trace("bug",
                        ["execute", "ROWS_AFFECTED", "rollback", "OK", "close"],
                        session_type, "real_bug", False)
        assert not r.conforms

    def test_all_real_bugs_detected(self, session_type):
        traces = capture_sqlite3_real_bug_traces()
        detected = 0
        for name, labels in traces:
            r = check_trace(name, labels, session_type, "real_bug", False)
            if not r.conforms:
                detected += 1
        assert detected == len(traces), f"Only detected {detected}/{len(traces)} bugs"


# ---------------------------------------------------------------------------
# Iterator conformance
# ---------------------------------------------------------------------------

class TestIteratorConformance:
    @pytest.fixture
    def session_type(self):
        return extract_java("java.util.Iterator").inferred_type

    def test_valid_traces_all_conform(self, session_type):
        for name, labels in capture_iterator_valid_traces():
            r = check_trace(name, labels, session_type, "valid", True)
            assert r.conforms, f"{name} should conform: {r.violations}"

    def test_next_without_hasNext_detected(self, session_type):
        r = check_trace("broken", ["next", "ELEMENT"],
                        session_type, "broken", False)
        assert not r.conforms
        assert r.steps_completed == 0  # next not allowed at top

    def test_next_after_false_detected(self, session_type):
        r = check_trace("broken",
                        ["hasNext", "FALSE", "next", "ELEMENT"],
                        session_type, "broken", False)
        assert not r.conforms

    def test_remove_without_next_detected(self, session_type):
        r = check_trace("broken",
                        ["hasNext", "TRUE", "remove", "REMOVED"],
                        session_type, "broken", False)
        assert not r.conforms

    def test_all_broken_detected(self, session_type):
        for name, labels in capture_iterator_broken_traces():
            r = check_trace(name, labels, session_type, "broken", False)
            assert not r.conforms, f"{name} should be violation"


# ---------------------------------------------------------------------------
# File I/O (InputStream) conformance
# ---------------------------------------------------------------------------

class TestInputStreamConformance:
    @pytest.fixture
    def session_type(self):
        return extract_java("java.io.InputStream").inferred_type

    def test_valid_traces_all_conform(self, session_type):
        for name, labels in capture_file_valid_traces():
            r = check_trace(name, labels, session_type, "valid", True)
            assert r.conforms, f"{name} should conform: {r.violations}"

    def test_read_after_close_detected(self, session_type):
        r = check_trace("broken", ["close", "read", "DATA"],
                        session_type, "broken", False)
        assert not r.conforms

    def test_double_close_detected(self, session_type):
        r = check_trace("broken", ["close", "close"],
                        session_type, "broken", False)
        assert not r.conforms

    def test_all_broken_detected(self, session_type):
        for name, labels in capture_file_broken_traces():
            r = check_trace(name, labels, session_type, "broken", False)
            assert not r.conforms


# ---------------------------------------------------------------------------
# Full suite
# ---------------------------------------------------------------------------

class TestFullSuite:
    def test_run_all_suites(self):
        suites = run_all_suites()
        assert len(suites) == 3

    def test_all_expectations_met(self):
        suites = run_all_suites()
        for s in suites:
            assert s.all_expectations_met, (
                f"{s.api_name}: expectations not met"
            )

    def test_total_traces(self):
        suites = run_all_suites()
        total = sum(s.total_traces for s in suites)
        assert total >= 25

    def test_real_bugs_detected(self):
        """At least 1 real bug pattern detected (validation plan success criterion)."""
        suites = run_all_suites()
        total_bugs = sum(s.real_bugs_detected for s in suites)
        assert total_bugs >= 1, "No real bugs detected!"

    def test_report_formatting(self):
        suites = run_all_suites()
        report = format_conformance_report(suites)
        assert "CONFORMANCE" in report
        assert "sqlite3" in report
        assert "Iterator" in report
        assert "InputStream" in report

    def test_sqlite3_five_valid(self):
        suites = run_all_suites()
        sqlite_suite = [s for s in suites if "sqlite3" in s.api_name][0]
        assert sqlite_suite.valid_correct == 5

    def test_sqlite3_five_broken(self):
        suites = run_all_suites()
        sqlite_suite = [s for s in suites if "sqlite3" in s.api_name][0]
        assert sqlite_suite.broken_detected == 5

    def test_sqlite3_four_real_bugs(self):
        suites = run_all_suites()
        sqlite_suite = [s for s in suites if "sqlite3" in s.api_name][0]
        assert sqlite_suite.real_bugs_detected == 4
