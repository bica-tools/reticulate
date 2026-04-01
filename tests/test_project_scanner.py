"""Tests for project scanner — static conformance checking (Step 80d).

Scans the sample_project/ directory containing both correct and buggy
Python code, and verifies that the scanner detects violations.
"""

import os
import pytest
from reticulate.project_scanner import (
    extract_clients_from_file,
    extract_clients_from_dir,
    scan_project,
    format_project_report,
    ClientTrace,
    ProjectReport,
)

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_project")

# Simple file protocol (L2 — no selections, just method order)
FILE_L2 = "rec X . &{read: X, readline: X, write: X, flush: X, close: end}"

# Simple DB protocol (L2)
DB_L2 = "rec X . &{execute: X, commit: X, rollback: X, close: end}"


# ---------------------------------------------------------------------------
# Client extraction tests
# ---------------------------------------------------------------------------

class TestClientExtraction:
    def test_extract_good_file(self):
        path = os.path.join(SAMPLE_DIR, "good_file_usage.py")
        clients = extract_clients_from_file(path, ["f"])
        assert len(clients) >= 4
        names = [c.function_name for c in clients]
        assert "read_config" in names
        assert "write_and_close" in names

    def test_extract_bad_file(self):
        path = os.path.join(SAMPLE_DIR, "bad_file_usage.py")
        clients = extract_clients_from_file(path, ["f"])
        assert len(clients) >= 3

    def test_extract_db(self):
        path = os.path.join(SAMPLE_DIR, "db_operations.py")
        clients = extract_clients_from_file(path, ["conn"])
        assert len(clients) >= 3

    def test_extract_from_dir(self):
        clients, files = extract_clients_from_dir(SAMPLE_DIR, ["f", "conn"])
        assert files >= 3
        assert len(clients) >= 8

    def test_client_has_methods(self):
        path = os.path.join(SAMPLE_DIR, "good_file_usage.py")
        clients = extract_clients_from_file(path, ["f"])
        for c in clients:
            assert len(c.methods) >= 1

    def test_client_has_location(self):
        path = os.path.join(SAMPLE_DIR, "good_file_usage.py")
        clients = extract_clients_from_file(path, ["f"])
        for c in clients:
            assert c.line_number > 0
            assert c.function_name


# ---------------------------------------------------------------------------
# Project scanning tests
# ---------------------------------------------------------------------------

class TestProjectScan:
    def test_scan_file_clients(self):
        report = scan_project(SAMPLE_DIR, ["f"], FILE_L2)
        assert report.files_scanned >= 2
        assert report.clients_found >= 5

    def test_finds_violations(self):
        """Bad file usage should produce violations."""
        report = scan_project(SAMPLE_DIR, ["f"], FILE_L2)
        assert report.violating >= 1, "Should find at least one violation"

    def test_finds_conforming(self):
        """Good file usage should conform."""
        report = scan_project(SAMPLE_DIR, ["f"], FILE_L2)
        assert report.conforming >= 1, "Should find at least one conforming function"

    def test_finds_incomplete(self):
        """forgot_to_close should be INCOMPLETE."""
        report = scan_project(SAMPLE_DIR, ["f"], FILE_L2)
        assert report.incomplete >= 1, "Should find at least one incomplete trace"

    def test_db_scan(self):
        report = scan_project(SAMPLE_DIR, ["conn"], DB_L2)
        assert report.clients_found >= 3
        assert report.conforming >= 1

    def test_report_formatting(self):
        report = scan_project(SAMPLE_DIR, ["f"], FILE_L2)
        text = format_project_report(report)
        assert "PROJECT CONFORMANCE" in text
        assert "VIOLATIONS" in text or "Conforming" in text

    def test_violation_has_location(self):
        """Violations should include file path and line number."""
        report = scan_project(SAMPLE_DIR, ["f"], FILE_L2)
        for detail in report.violation_details:
            assert ":" in detail  # file:line format
            assert "()" in detail  # function name


# ---------------------------------------------------------------------------
# Real project demo
# ---------------------------------------------------------------------------

class TestScanDemo:
    def test_full_scan_report(self, capsys):
        """Print a demo report for the sample project."""
        report = scan_project(SAMPLE_DIR, ["f", "conn"], FILE_L2)
        text = format_project_report(report)
        print("\n" + text)
        assert report.clients_found >= 5
