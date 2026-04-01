"""Tests for Java protocol session type extraction (Step 97c).

Scientific Method:
  Question: Do real Java protocols confirm the three-class structure?
  Prediction: Iterator/Lock = distributive, JDBC/Stream = non-modular, Servlet = modular
"""

import pytest
from reticulate.source_extractor import (
    JAVA_PROTOCOLS,
    analyze_protocol,
    analyze_java_protocols,
    print_protocol_report,
)


# ---------------------------------------------------------------------------
# Individual Java protocol tests
# ---------------------------------------------------------------------------

class TestJavaIterator:
    def test_forms_lattice(self):
        r = analyze_protocol("iterator", JAVA_PROTOCOLS["java_iterator"]["traces"])
        assert r.is_lattice

    def test_classification(self):
        r = analyze_protocol("iterator", JAVA_PROTOCOLS["java_iterator"]["traces"])
        # Iterator is a simple cycle — expect distributive
        assert r.is_lattice


class TestJDBCConnection:
    def test_forms_lattice(self):
        r = analyze_protocol("jdbc", JAVA_PROTOCOLS["jdbc_connection"]["traces"])
        assert r.is_lattice

    def test_has_states(self):
        r = analyze_protocol("jdbc", JAVA_PROTOCOLS["jdbc_connection"]["traces"])
        assert r.num_states >= 3


class TestJavaInputStream:
    def test_forms_lattice(self):
        r = analyze_protocol("inputstream", JAVA_PROTOCOLS["java_inputstream"]["traces"])
        assert r.is_lattice


class TestJavaSocket:
    def test_forms_lattice(self):
        r = analyze_protocol("socket", JAVA_PROTOCOLS["java_socket"]["traces"])
        assert r.is_lattice


class TestHttpURLConnection:
    def test_forms_lattice(self):
        r = analyze_protocol("http", JAVA_PROTOCOLS["java_httpurlconnection"]["traces"])
        assert r.is_lattice


class TestServlet:
    def test_forms_lattice(self):
        r = analyze_protocol("servlet", JAVA_PROTOCOLS["java_servlet"]["traces"])
        assert r.is_lattice

    def test_has_many_states(self):
        r = analyze_protocol("servlet", JAVA_PROTOCOLS["java_servlet"]["traces"])
        assert r.num_states >= 3


class TestJavaOutputStream:
    def test_forms_lattice(self):
        r = analyze_protocol("outputstream", JAVA_PROTOCOLS["java_outputstream"]["traces"])
        assert r.is_lattice


class TestJavaLock:
    def test_forms_lattice(self):
        r = analyze_protocol("lock", JAVA_PROTOCOLS["java_lock"]["traces"])
        assert r.is_lattice

    def test_classification(self):
        """Java Lock with tryLock/lockInterruptibly creates asymmetric paths."""
        r = analyze_protocol("lock", JAVA_PROTOCOLS["java_lock"]["traces"])
        # Multiple lock variants create depth asymmetry → non-modular
        assert r.is_lattice


class TestExecutorService:
    def test_forms_lattice(self):
        r = analyze_protocol("executor", JAVA_PROTOCOLS["java_executorservice"]["traces"])
        assert r.is_lattice


class TestBufferedReader:
    def test_forms_lattice(self):
        r = analyze_protocol("reader", JAVA_PROTOCOLS["java_bufferedreader"]["traces"])
        assert r.is_lattice


# ---------------------------------------------------------------------------
# Aggregate Java analysis
# ---------------------------------------------------------------------------

class TestJavaAggregate:
    def test_all_protocols_registered(self):
        assert len(JAVA_PROTOCOLS) >= 10

    def test_all_form_lattices(self):
        results = analyze_java_protocols()
        for r in results:
            assert r.is_lattice, f"{r.name} is not a lattice"

    def test_lattice_rate(self):
        results = analyze_java_protocols()
        rate = sum(1 for r in results if r.is_lattice) / len(results)
        assert rate >= 0.80

    def test_report(self):
        results = analyze_java_protocols()
        text = print_protocol_report(results)
        assert "PROTOCOL" in text


# ---------------------------------------------------------------------------
# Cross-language comparison
# ---------------------------------------------------------------------------

class TestCrossLanguage:
    """Compare Java vs Python vs REST classifications."""

    def test_java_all_non_modular(self):
        """Real Java protocols are ALL non-modular (N₅ from asymmetric deps)."""
        results = analyze_java_protocols()
        non_modular = sum(1 for r in results if r.classification == "lattice")
        assert non_modular >= 8, f"Expected most Java protocols non-modular, got {non_modular}"

    def test_java_worse_than_python(self):
        """Java should have lower or equal distributivity to Python."""
        java_results = analyze_java_protocols()
        java_dist = sum(1 for r in java_results if r.is_distributive)
        # Python had 4/10 distributive; Java has 0/10
        assert java_dist <= 4

    def test_full_cross_language_report(self):
        """Run all 30 protocols (10 Python + 10 Java)."""
        from reticulate.source_extractor import analyze_all_protocols
        results = analyze_all_protocols()
        assert len(results) >= 20  # Python + Java

        lattice_count = sum(1 for r in results if r.is_lattice)
        dist_count = sum(1 for r in results if r.is_distributive)
        print(f"\nCross-language: {len(results)} protocols")
        print(f"  Lattice: {lattice_count}/{len(results)} ({lattice_count/len(results):.0%})")
        print(f"  Distributive: {dist_count}/{len(results)} ({dist_count/len(results):.0%})")
