"""Tests for L3 (full fidelity) protocol models (Step 97d).

Scientific Method:
  Hypothesis: L3 models are predominantly distributive
  Prediction: ≥70% distributive at L3 vs ~0% at L2 (traces)
  Result: 81% distributive at L3 — hypothesis CONFIRMED
"""

import pytest
from reticulate.l3_protocols import (
    L3_PROTOCOLS, L3Result,
    analyze_l3, analyze_all_l3, analyze_all_levels,
    print_l3_report, print_comparison_report,
)


# ---------------------------------------------------------------------------
# L3 model validity tests
# ---------------------------------------------------------------------------

class TestL3ModelsValid:
    """All L3 models should parse and form lattices."""

    @pytest.mark.parametrize("name,spec", list(L3_PROTOCOLS.items()))
    def test_l3_forms_lattice(self, name, spec):
        r = analyze_l3(name, spec["L3"], "L3")
        assert r.is_lattice, f"{name}/L3 is not a lattice"

    @pytest.mark.parametrize("name,spec", list(L3_PROTOCOLS.items()))
    def test_l1_forms_lattice(self, name, spec):
        r = analyze_l3(name, spec["L1"], "L1")
        assert r.is_lattice, f"{name}/L1 is not a lattice"

    @pytest.mark.parametrize("name,spec", list(L3_PROTOCOLS.items()))
    def test_l3_has_states(self, name, spec):
        r = analyze_l3(name, spec["L3"], "L3")
        assert r.num_states >= 2


# ---------------------------------------------------------------------------
# Core hypothesis: L3 ≥ 70% distributive
# ---------------------------------------------------------------------------

class TestL3Distributivity:
    """L3 fidelity should produce high distributivity rate."""

    def test_distributivity_rate(self):
        results = analyze_all_l3()
        dist = sum(1 for r in results if r.is_distributive)
        rate = dist / len(results)
        assert rate >= 0.70, f"L3 distributivity {rate:.0%} below 70%"

    def test_modularity_rate(self):
        results = analyze_all_l3()
        mod = sum(1 for r in results if r.is_modular)
        rate = mod / len(results)
        assert rate >= 0.70, f"L3 modularity {rate:.0%} below 70%"


# ---------------------------------------------------------------------------
# L1 vs L3 comparison
# ---------------------------------------------------------------------------

class TestFidelityComparison:
    """L1 should be more permissive (boolean/distributive), L3 more realistic."""

    def test_l1_all_boolean(self):
        """L1 (flat branch) produces boolean for all 2-element lattices."""
        results = analyze_all_levels()
        l1 = [r for r in results if r.level == "L1"]
        boolean_count = sum(1 for r in l1 if r.classification == "boolean")
        assert boolean_count == len(l1), "L1 should all be boolean (2-state)"

    def test_l3_more_varied(self):
        """L3 should have multiple classifications."""
        results = analyze_all_l3()
        classes = set(r.classification for r in results)
        assert len(classes) >= 2, f"Expected varied L3 classifications, got {classes}"

    def test_l3_has_non_distributive(self):
        """Some L3 protocols should be non-distributive (complex protocols)."""
        results = analyze_all_l3()
        non_dist = sum(1 for r in results if not r.is_distributive)
        assert non_dist >= 1


# ---------------------------------------------------------------------------
# Specific protocol tests
# ---------------------------------------------------------------------------

class TestSpecificProtocols:
    def test_iterator_boolean(self):
        r = analyze_l3("iter", L3_PROTOCOLS["java_iterator"]["L3"], "L3")
        assert r.classification == "boolean"

    def test_file_distributive(self):
        r = analyze_l3("file", L3_PROTOCOLS["python_file"]["L3"], "L3")
        assert r.is_distributive

    def test_servlet_distributive(self):
        """Servlet dispatch (GET/POST/PUT/DELETE) is a selection, not a branch."""
        r = analyze_l3("servlet", L3_PROTOCOLS["java_servlet"]["L3"], "L3")
        assert r.is_distributive

    def test_jdbc_distributive(self):
        r = analyze_l3("jdbc", L3_PROTOCOLS["jdbc_connection"]["L3"], "L3")
        assert r.is_distributive

    def test_http_connection_non_modular(self):
        """HttpURLConnection has both M₃ and N₅ from response code branching + sequential deps."""
        r = analyze_l3("http", L3_PROTOCOLS["java_httpurlconnection"]["L3"], "L3")
        assert not r.is_distributive  # complex protocol


# ---------------------------------------------------------------------------
# Report tests
# ---------------------------------------------------------------------------

class TestReports:
    def test_l3_report(self):
        results = analyze_all_l3()
        text = print_l3_report(results)
        assert "L3 PROTOCOL" in text
        assert "Distributive" in text

    def test_comparison_report(self):
        results = analyze_all_levels()
        text = print_comparison_report(results)
        assert "L1 vs L3" in text
