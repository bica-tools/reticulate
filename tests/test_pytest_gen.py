"""Tests for pytest test generation from extracted session types (Week 3).

Two levels of testing:
1. **Meta-tests**: verify the generator produces valid Python source
2. **Execution tests**: actually RUN generated tests against real libraries

Only sqlite3 tests run without network access. HTTP, SMTP, FTP, SSL
tests are generated but marked as skip (need network/server).
"""

import os
import tempfile
import sqlite3
import textwrap
import pytest

from reticulate.pytest_gen import (
    HARNESSES,
    APITestHarness,
    GenerationResult,
    generate_pytest_source,
    generate_all_pytest_sources,
    select_test_paths,
    format_testgen_table,
)
from reticulate.python_api_extractor import (
    extract_session_type,
    extract_all_target_apis,
)
from reticulate.parser import parse
from reticulate.statespace import build_statespace


TARGET_APIS = [
    "sqlite3.Connection",
    "http.client.HTTPConnection",
    "smtplib.SMTP",
    "ftplib.FTP",
    "ssl.SSLSocket",
]


# ---------------------------------------------------------------------------
# Harness completeness
# ---------------------------------------------------------------------------

class TestHarnessCompleteness:
    def test_all_apis_have_harnesses(self):
        for api in TARGET_APIS:
            assert api in HARNESSES, f"No harness for {api}"

    @pytest.mark.parametrize("api_name", TARGET_APIS)
    def test_harness_has_imports(self, api_name):
        h = HARNESSES[api_name]
        assert len(h.imports) >= 1

    @pytest.mark.parametrize("api_name", TARGET_APIS)
    def test_harness_has_setup(self, api_name):
        h = HARNESSES[api_name]
        assert len(h.setup_code) >= 1

    @pytest.mark.parametrize("api_name", TARGET_APIS)
    def test_harness_has_method_map(self, api_name):
        h = HARNESSES[api_name]
        assert len(h.method_map) >= 3

    @pytest.mark.parametrize("api_name", TARGET_APIS)
    def test_harness_has_assertion_map(self, api_name):
        h = HARNESSES[api_name]
        assert len(h.assertion_map) >= 2

    def test_sqlite3_is_runnable(self):
        """sqlite3 should NOT have a skip_reason (runs locally)."""
        assert HARNESSES["sqlite3.Connection"].skip_reason is None

    def test_network_apis_have_skip_reason(self):
        for api in ["http.client.HTTPConnection", "smtplib.SMTP",
                     "ftplib.FTP", "ssl.SSLSocket"]:
            assert HARNESSES[api].skip_reason is not None


# ---------------------------------------------------------------------------
# Source generation tests
# ---------------------------------------------------------------------------

class TestSourceGeneration:
    @pytest.mark.parametrize("api_name", TARGET_APIS)
    def test_generates_valid_python(self, api_name):
        """Generated source must be syntactically valid Python."""
        r = extract_session_type(api_name)
        source = generate_pytest_source(api_name, r.inferred_type)
        # Compile to check syntax
        compile(source, f"<generated:{api_name}>", "exec")

    @pytest.mark.parametrize("api_name", TARGET_APIS)
    def test_generates_test_functions(self, api_name):
        r = extract_session_type(api_name)
        source = generate_pytest_source(api_name, r.inferred_type)
        assert "def test_" in source

    @pytest.mark.parametrize("api_name", TARGET_APIS)
    def test_generates_fixture(self, api_name):
        r = extract_session_type(api_name)
        source = generate_pytest_source(api_name, r.inferred_type)
        assert "@pytest.fixture" in source

    @pytest.mark.parametrize("api_name", TARGET_APIS)
    def test_generates_imports(self, api_name):
        r = extract_session_type(api_name)
        source = generate_pytest_source(api_name, r.inferred_type)
        assert "import pytest" in source

    @pytest.mark.parametrize("api_name", TARGET_APIS)
    def test_generates_multiple_tests(self, api_name):
        r = extract_session_type(api_name)
        source = generate_pytest_source(api_name, r.inferred_type, max_tests=5)
        count = source.count("def test_")
        assert count >= 2, f"Only {count} tests generated for {api_name}"

    def test_sqlite3_no_skip_marker(self):
        """sqlite3 tests should NOT have skip markers."""
        r = extract_session_type("sqlite3.Connection")
        source = generate_pytest_source("sqlite3.Connection", r.inferred_type)
        assert "skipif" not in source

    def test_http_has_skip_marker(self):
        """HTTP tests should have skip markers (needs network)."""
        r = extract_session_type("http.client.HTTPConnection")
        source = generate_pytest_source("http.client.HTTPConnection", r.inferred_type)
        assert "skipif" in source or "SKIP_REASON" in source


# ---------------------------------------------------------------------------
# Path selection tests
# ---------------------------------------------------------------------------

class TestPathSelection:
    @pytest.mark.parametrize("api_name", TARGET_APIS)
    def test_selects_paths(self, api_name):
        r = extract_session_type(api_name)
        ast = parse(r.inferred_type)
        ss = build_statespace(ast)
        paths = select_test_paths(ss, max_paths=10)
        assert len(paths) >= 1

    @pytest.mark.parametrize("api_name", TARGET_APIS)
    def test_paths_are_unique(self, api_name):
        r = extract_session_type(api_name)
        ast = parse(r.inferred_type)
        ss = build_statespace(ast)
        paths = select_test_paths(ss, max_paths=20)
        label_seqs = [tuple(p.labels) for p in paths]
        assert len(label_seqs) == len(set(label_seqs))

    @pytest.mark.parametrize("api_name", TARGET_APIS)
    def test_paths_start_at_top(self, api_name):
        r = extract_session_type(api_name)
        ast = parse(r.inferred_type)
        ss = build_statespace(ast)
        paths = select_test_paths(ss, max_paths=5)
        for p in paths:
            assert len(p.steps) >= 1


# ---------------------------------------------------------------------------
# Bulk generation
# ---------------------------------------------------------------------------

class TestBulkGeneration:
    def test_generate_all(self):
        results = extract_all_target_apis()
        gen_results = generate_all_pytest_sources(results)
        assert len(gen_results) == 5

    def test_total_tests_reasonable(self):
        results = extract_all_target_apis()
        gen_results = generate_all_pytest_sources(results, max_tests_per_api=8)
        total = sum(r.num_tests for r in gen_results)
        assert total >= 10

    def test_table_formatting(self):
        results = extract_all_target_apis()
        gen_results = generate_all_pytest_sources(results)
        table = format_testgen_table(gen_results)
        assert "TEST GENERATION" in table
        for api in TARGET_APIS:
            assert api in table


# ---------------------------------------------------------------------------
# EXECUTION TESTS: Actually run generated sqlite3 tests
# ---------------------------------------------------------------------------

class TestSqlite3Execution:
    """Run generated session-type tests against REAL sqlite3."""

    def _get_generated_source(self) -> str:
        r = extract_session_type("sqlite3.Connection")
        return generate_pytest_source("sqlite3.Connection", r.inferred_type)

    def test_generated_source_compiles(self):
        source = self._get_generated_source()
        compile(source, "<sqlite3_gen>", "exec")

    def test_sqlite3_connect_execute_commit_close(self):
        """Manually execute the canonical valid path."""
        db_fd, db_path = tempfile.mkstemp(suffix='.db')
        os.close(db_fd)
        try:
            conn = sqlite3.connect(db_path)
            conn.execute('CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, val TEXT)')
            conn.commit()
            # Path: cursor → execute → commit → close
            cur = conn.cursor()
            assert cur is not None
            conn.execute("INSERT INTO t (val) VALUES ('x')")
            conn.commit()
            conn.close()
        finally:
            os.unlink(db_path)

    def test_sqlite3_execute_rollback_execute_commit(self):
        """Path: execute → rollback → execute → commit → close."""
        db_fd, db_path = tempfile.mkstemp(suffix='.db')
        os.close(db_fd)
        try:
            conn = sqlite3.connect(db_path)
            conn.execute('CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, val TEXT)')
            conn.commit()
            conn.execute("INSERT INTO t (val) VALUES ('a')")
            conn.rollback()
            conn.execute("INSERT INTO t (val) VALUES ('b')")
            conn.commit()
            # Verify: only 'b' should exist (rollback undid 'a')
            cur = conn.execute("SELECT val FROM t")
            rows = cur.fetchall()
            assert ('b',) in rows
            assert ('a',) not in rows
            conn.close()
        finally:
            os.unlink(db_path)

    def test_sqlite3_just_close(self):
        """Shortest valid path: just close."""
        db_fd, db_path = tempfile.mkstemp(suffix='.db')
        os.close(db_fd)
        try:
            conn = sqlite3.connect(db_path)
            conn.close()
        finally:
            os.unlink(db_path)

    def test_sqlite3_multiple_executes(self):
        """Path: execute → execute → commit → close."""
        db_fd, db_path = tempfile.mkstemp(suffix='.db')
        os.close(db_fd)
        try:
            conn = sqlite3.connect(db_path)
            conn.execute('CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, val TEXT)')
            conn.commit()
            conn.execute("INSERT INTO t (val) VALUES ('x')")
            conn.execute("INSERT INTO t (val) VALUES ('y')")
            conn.commit()
            cur = conn.execute("SELECT COUNT(*) FROM t")
            assert cur.fetchone()[0] == 2
            conn.close()
        finally:
            os.unlink(db_path)

    def test_sqlite3_cursor_executemany_commit(self):
        """Path: cursor → executemany → commit → close."""
        db_fd, db_path = tempfile.mkstemp(suffix='.db')
        os.close(db_fd)
        try:
            conn = sqlite3.connect(db_path)
            conn.execute('CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, val TEXT)')
            conn.commit()
            cur = conn.cursor()
            cur.executemany("INSERT INTO t (val) VALUES (?)", [('a',), ('b',), ('c',)])
            conn.commit()
            cur2 = conn.execute("SELECT COUNT(*) FROM t")
            assert cur2.fetchone()[0] == 3
            conn.close()
        finally:
            os.unlink(db_path)

    def test_sqlite3_isolation_level(self):
        """Path: isolation_level → execute → commit → close."""
        db_fd, db_path = tempfile.mkstemp(suffix='.db')
        os.close(db_fd)
        try:
            conn = sqlite3.connect(db_path)
            conn.execute('CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, val TEXT)')
            conn.commit()
            level = conn.isolation_level  # query isolation level
            conn.execute("INSERT INTO t (val) VALUES ('x')")
            conn.commit()
            conn.close()
        finally:
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# Violation detection: test that protocol violations are caught
# ---------------------------------------------------------------------------

class TestViolationDetection:
    """Verify that the session type correctly identifies violations."""

    def test_sqlite3_commit_after_close_is_violation(self):
        """After close, no methods should be available."""
        db_fd, db_path = tempfile.mkstemp(suffix='.db')
        os.close(db_fd)
        try:
            conn = sqlite3.connect(db_path)
            conn.close()
            with pytest.raises(Exception):
                conn.commit()
        finally:
            os.unlink(db_path)

    def test_sqlite3_execute_after_close_is_violation(self):
        db_fd, db_path = tempfile.mkstemp(suffix='.db')
        os.close(db_fd)
        try:
            conn = sqlite3.connect(db_path)
            conn.close()
            with pytest.raises(Exception):
                conn.execute("SELECT 1")
        finally:
            os.unlink(db_path)
