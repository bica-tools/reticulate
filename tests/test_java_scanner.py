"""Tests for Java project scanner (Step 80e).

Scans the actual BICA Reborn Java project (242 .java files)
for protocol conformance.
"""

import os
import pytest
from reticulate.java_scanner import (
    extract_clients_from_java_file,
    extract_clients_from_java_dir,
    scan_java_project,
    format_java_report,
    HAS_JAVALANG,
)

BICA_SRC = os.path.join(
    os.path.dirname(__file__), "..", "..", "bica", "src", "main", "java"
)

# StateSpace query protocol
SS_QUERY = ("rec X . &{states: X, transitions: X, top: X, bottom: X, "
            "enabled: X, enabledLabels: X, transitionsFrom: X, labels: X}")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(BICA_SRC),
    reason="BICA source not found"
)


class TestJavaFileParsing:
    @pytest.mark.skipif(not HAS_JAVALANG, reason="javalang not installed")
    def test_parse_java_file(self):
        """Should parse at least one Java file from BICA."""
        java_files = []
        for root, _, files in os.walk(BICA_SRC):
            for f in files:
                if f.endswith(".java"):
                    java_files.append(os.path.join(root, f))
                    if len(java_files) >= 3:
                        break
            if java_files:
                break
        assert len(java_files) >= 1
        clients = extract_clients_from_java_file(java_files[0], ["ss", "stateSpace"])
        # May or may not find clients depending on the file
        assert isinstance(clients, list)


class TestBICAProjectScan:
    @pytest.mark.skipif(not HAS_JAVALANG, reason="javalang not installed")
    def test_scan_finds_files(self):
        report = scan_java_project(BICA_SRC, ["ss", "stateSpace"], SS_QUERY)
        assert report.java_files_scanned >= 100

    @pytest.mark.skipif(not HAS_JAVALANG, reason="javalang not installed")
    def test_scan_finds_classes(self):
        report = scan_java_project(BICA_SRC, ["ss", "stateSpace"], SS_QUERY)
        assert report.classes_found >= 20

    @pytest.mark.skipif(not HAS_JAVALANG, reason="javalang not installed")
    def test_scan_finds_clients(self):
        report = scan_java_project(BICA_SRC, ["ss", "stateSpace"], SS_QUERY)
        assert report.clients_found >= 30

    @pytest.mark.skipif(not HAS_JAVALANG, reason="javalang not installed")
    def test_low_violation_rate(self):
        """BICA clients mostly use StateSpace correctly — few violations from incomplete protocol model."""
        report = scan_java_project(BICA_SRC, ["ss", "stateSpace"], SS_QUERY)
        violation_rate = report.violating / report.clients_found if report.clients_found else 0
        assert violation_rate < 0.10, f"Violation rate {violation_rate:.0%} too high"

    @pytest.mark.skipif(not HAS_JAVALANG, reason="javalang not installed")
    def test_report_formatting(self):
        report = scan_java_project(BICA_SRC, ["ss", "stateSpace"], SS_QUERY)
        text = format_java_report(report)
        assert "JAVA PROJECT" in text
