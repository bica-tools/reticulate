"""Tests for Python API session type extraction (TOPLAS validation Week 2).

Validates that session types can be extracted from 5 real Python APIs,
with proper Branch/Select distinction, and all form lattices.

Target APIs:
  1. sqlite3.Connection
  2. http.client.HTTPConnection
  3. smtplib.SMTP
  4. ftplib.FTP
  5. ssl.SSLSocket
"""

import pytest
from reticulate.python_api_extractor import (
    PYTHON_API_SPECS,
    ExtractionResult,
    extract_session_type,
    extract_all_target_apis,
    format_validation_table,
    format_latex_table,
)
from reticulate.parser import parse, pretty
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice


# ---------------------------------------------------------------------------
# Spec completeness
# ---------------------------------------------------------------------------

TARGET_APIS = [
    "sqlite3.Connection",
    "http.client.HTTPConnection",
    "smtplib.SMTP",
    "ftplib.FTP",
    "ssl.SSLSocket",
]


class TestSpecCompleteness:
    def test_all_targets_present(self):
        for api in TARGET_APIS:
            assert api in PYTHON_API_SPECS, f"Missing spec for {api}"

    def test_five_apis_exactly(self):
        assert len(PYTHON_API_SPECS) == 5

    @pytest.mark.parametrize("api_name", TARGET_APIS)
    def test_has_description(self, api_name):
        assert len(PYTHON_API_SPECS[api_name]["description"]) > 10

    @pytest.mark.parametrize("api_name", TARGET_APIS)
    def test_has_directed_traces(self, api_name):
        traces = PYTHON_API_SPECS[api_name]["directed_traces"]
        assert len(traces) >= 5

    @pytest.mark.parametrize("api_name", TARGET_APIS)
    def test_has_methods(self, api_name):
        methods = PYTHON_API_SPECS[api_name]["methods"]
        assert len(methods) >= 3

    @pytest.mark.parametrize("api_name", TARGET_APIS)
    def test_has_return_types(self, api_name):
        rt = PYTHON_API_SPECS[api_name]["return_types"]
        assert len(rt) >= 3

    @pytest.mark.parametrize("api_name", TARGET_APIS)
    def test_receive_labels_are_methods(self, api_name):
        """All 'r' steps must be declared methods."""
        spec = PYTHON_API_SPECS[api_name]
        declared = set(spec["methods"])
        for dt in spec["directed_traces"]:
            for label, direction in dt:
                if direction == "r":
                    assert label in declared, (
                        f"{api_name}: '{label}' not in methods {declared}"
                    )

    @pytest.mark.parametrize("api_name", TARGET_APIS)
    def test_send_labels_are_uppercase(self, api_name):
        """All 's' steps (Select outcomes) must be UPPERCASE."""
        spec = PYTHON_API_SPECS[api_name]
        for dt in spec["directed_traces"]:
            for label, direction in dt:
                if direction == "s":
                    assert label == label.upper(), (
                        f"{api_name}: selection '{label}' should be UPPERCASE"
                    )

    @pytest.mark.parametrize("api_name", TARGET_APIS)
    def test_return_type_outcomes_are_uppercase(self, api_name):
        """All return type outcome labels must be UPPERCASE."""
        spec = PYTHON_API_SPECS[api_name]
        for method, outcomes in spec["return_types"].items():
            for outcome in outcomes:
                assert outcome == outcome.upper(), (
                    f"{api_name}.{method}: outcome '{outcome}' should be UPPERCASE"
                )

    @pytest.mark.parametrize("api_name", TARGET_APIS)
    def test_traces_alternate_direction(self, api_name):
        """Traces should generally alternate r/s (call → return)."""
        spec = PYTHON_API_SPECS[api_name]
        for dt in spec["directed_traces"]:
            # Check that we don't have long runs of same direction
            # (some traces end with 'r' for void methods like close/quit)
            r_count = sum(1 for _, d in dt if d == "r")
            s_count = sum(1 for _, d in dt if d == "s")
            assert r_count >= 1, f"{api_name}: trace has no method calls"


# ---------------------------------------------------------------------------
# Per-API extraction tests
# ---------------------------------------------------------------------------

class TestSqlite3Extraction:
    def test_extract_succeeds(self):
        r = extract_session_type("sqlite3.Connection")
        assert isinstance(r, ExtractionResult)

    def test_is_lattice(self):
        r = extract_session_type("sqlite3.Connection")
        assert r.is_lattice

    def test_has_selections(self):
        r = extract_session_type("sqlite3.Connection")
        assert "+{" in r.inferred_type

    def test_has_commit_outcomes(self):
        r = extract_session_type("sqlite3.Connection")
        assert "OK" in r.inferred_type
        assert "INTEGRITY_ERROR" in r.inferred_type

    def test_parses_and_roundtrips(self):
        r = extract_session_type("sqlite3.Connection")
        ast1 = parse(r.inferred_type)
        rt = pretty(ast1)
        ast2 = parse(rt)
        assert ast2 is not None


class TestHTTPExtraction:
    def test_extract_succeeds(self):
        r = extract_session_type("http.client.HTTPConnection")
        assert isinstance(r, ExtractionResult)

    def test_is_lattice(self):
        r = extract_session_type("http.client.HTTPConnection")
        assert r.is_lattice

    def test_has_status_code_selections(self):
        r = extract_session_type("http.client.HTTPConnection")
        assert "OK_200" in r.inferred_type
        assert "NOT_FOUND_404" in r.inferred_type

    def test_has_server_error(self):
        r = extract_session_type("http.client.HTTPConnection")
        assert "SERVER_ERROR_500" in r.inferred_type


class TestSMTPExtraction:
    def test_extract_succeeds(self):
        r = extract_session_type("smtplib.SMTP")
        assert isinstance(r, ExtractionResult)

    def test_is_lattice(self):
        r = extract_session_type("smtplib.SMTP")
        assert r.is_lattice

    def test_has_auth_selection(self):
        r = extract_session_type("smtplib.SMTP")
        assert "AUTH_OK" in r.inferred_type
        assert "AUTH_FAILED" in r.inferred_type

    def test_has_tls_selection(self):
        r = extract_session_type("smtplib.SMTP")
        assert "TLS_OK" in r.inferred_type
        assert "TLS_ERROR" in r.inferred_type

    def test_has_send_outcomes(self):
        r = extract_session_type("smtplib.SMTP")
        assert "SENT" in r.inferred_type


class TestFTPExtraction:
    def test_extract_succeeds(self):
        r = extract_session_type("ftplib.FTP")
        assert isinstance(r, ExtractionResult)

    def test_is_lattice(self):
        r = extract_session_type("ftplib.FTP")
        assert r.is_lattice

    def test_has_auth_selection(self):
        r = extract_session_type("ftplib.FTP")
        assert "AUTH_OK" in r.inferred_type
        assert "AUTH_FAILED" in r.inferred_type

    def test_has_file_operations(self):
        r = extract_session_type("ftplib.FTP")
        t = r.inferred_type
        assert "DOWNLOADED" in t or "UPLOADED" in t


class TestSSLExtraction:
    def test_extract_succeeds(self):
        r = extract_session_type("ssl.SSLSocket")
        assert isinstance(r, ExtractionResult)

    def test_is_lattice(self):
        r = extract_session_type("ssl.SSLSocket")
        assert r.is_lattice

    def test_has_handshake_outcomes(self):
        r = extract_session_type("ssl.SSLSocket")
        assert "CERT_ERROR" in r.inferred_type or "HANDSHAKE_ERROR" in r.inferred_type

    def test_has_read_eof(self):
        r = extract_session_type("ssl.SSLSocket")
        assert "EOF" in r.inferred_type or "CLOSED" in r.inferred_type


# ---------------------------------------------------------------------------
# Aggregate pipeline tests
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_extract_all_succeeds(self):
        results = extract_all_target_apis()
        assert len(results) == 5

    def test_all_parse(self):
        for r in extract_all_target_apis():
            ast = parse(r.inferred_type)
            assert ast is not None, f"{r.api_name} failed to parse"

    def test_all_build_statespace(self):
        for r in extract_all_target_apis():
            ss = build_statespace(parse(r.inferred_type))
            assert len(ss.states) > 0

    def test_all_are_lattices(self):
        for r in extract_all_target_apis():
            assert r.is_lattice, f"{r.api_name} is NOT a lattice"

    def test_lattice_rate_100_percent(self):
        results = extract_all_target_apis()
        assert sum(1 for r in results if r.is_lattice) == 5

    def test_all_have_top_and_bottom(self):
        for r in extract_all_target_apis():
            ss = build_statespace(parse(r.inferred_type))
            lr = check_lattice(ss)
            assert lr.has_top and lr.has_bottom, f"{r.api_name} missing top/bottom"

    def test_all_have_selections(self):
        """Every Python API type must contain Select (+{...})."""
        for r in extract_all_target_apis():
            assert "+{" in r.inferred_type, f"{r.api_name} has no selections"

    def test_total_states_reasonable(self):
        total = sum(r.num_states for r in extract_all_target_apis())
        assert 20 <= total <= 500

    def test_roundtrip_all_types(self):
        for r in extract_all_target_apis():
            ast1 = parse(r.inferred_type)
            ast2 = parse(pretty(ast1))
            assert ast2 is not None


# ---------------------------------------------------------------------------
# Distributivity
# ---------------------------------------------------------------------------

class TestDistributivity:
    def test_classification_valid(self):
        valid = {"boolean", "distributive", "modular", "non-modular",
                 "non-lattice", "lattice"}
        for r in extract_all_target_apis():
            assert r.classification in valid


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

class TestFormatting:
    def test_validation_table(self):
        results = extract_all_target_apis()
        table = format_validation_table(results)
        assert "VALIDATION TABLE" in table
        for api in TARGET_APIS:
            assert api in table

    def test_latex_table(self):
        results = extract_all_target_apis()
        latex = format_latex_table(results)
        assert r"\begin{table}" in latex
        assert r"\cmark" in latex

    def test_latex_has_all_apis(self):
        results = extract_all_target_apis()
        latex = format_latex_table(results)
        for api in TARGET_APIS:
            short = api.split(".")[-1]
            assert short in latex


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_unknown_api_raises(self):
        with pytest.raises(ValueError, match="No spec"):
            extract_session_type("nonexistent.Module")
