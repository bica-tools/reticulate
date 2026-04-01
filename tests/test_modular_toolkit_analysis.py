"""Tests for modular toolkit analysis on real-world protocols (Step 30ag).

Scientific Method validation:
  Hypothesis: Valuations and grading provide useful protocol invariants
  Prediction: Modular (REST) → rank is valuation; non-modular → rank is not
"""

import pytest
from reticulate.modular_toolkit_analysis import (
    ToolkitResult,
    ToolkitReport,
    analyze_single,
    analyze_all,
    print_toolkit_report,
)
from reticulate.real_world_extraction import REAL_WORLD_SPECS
from reticulate.source_extractor import PYTHON_PROTOCOLS
from reticulate.importers import from_openapi
from reticulate.parser import pretty
from reticulate.type_inference import Trace, infer_from_traces


# ---------------------------------------------------------------------------
# Individual protocol toolkit tests
# ---------------------------------------------------------------------------

class TestRESTValuations:
    """All REST APIs should have rank as valuation (modular)."""

    @pytest.mark.parametrize("name,spec_fn", list(REAL_WORLD_SPECS.items()))
    def test_rank_is_valuation(self, name, spec_fn):
        spec = spec_fn()
        types = from_openapi(spec)
        for tag, st_str in types.items():
            r = analyze_single(name, "REST", st_str)
            assert r.rank_is_valuation, f"{name}: rank should be valuation (modular)"

    @pytest.mark.parametrize("name,spec_fn", list(REAL_WORLD_SPECS.items()))
    def test_is_graded(self, name, spec_fn):
        spec = spec_fn()
        types = from_openapi(spec)
        for tag, st_str in types.items():
            r = analyze_single(name, "REST", st_str)
            assert r.is_graded, f"{name}: REST API should be graded"

    @pytest.mark.parametrize("name,spec_fn", list(REAL_WORLD_SPECS.items()))
    def test_classification_modular(self, name, spec_fn):
        spec = spec_fn()
        types = from_openapi(spec)
        for tag, st_str in types.items():
            r = analyze_single(name, "REST", st_str)
            assert r.classification == "modular", f"{name}: expected modular"

    @pytest.mark.parametrize("name,spec_fn", list(REAL_WORLD_SPECS.items()))
    def test_width_reflects_endpoints(self, name, spec_fn):
        """Width should be ≥3 (number of endpoints at branch state)."""
        spec = spec_fn()
        types = from_openapi(spec)
        for tag, st_str in types.items():
            r = analyze_single(name, "REST", st_str)
            assert r.width >= 3, f"{name}: width should be ≥3 for REST API"

    @pytest.mark.parametrize("name,spec_fn", list(REAL_WORLD_SPECS.items()))
    def test_depth_is_two(self, name, spec_fn):
        """REST APIs have depth 2: branch → select → end."""
        spec = spec_fn()
        types = from_openapi(spec)
        for tag, st_str in types.items():
            r = analyze_single(name, "REST", st_str)
            assert r.depth == 2, f"{name}: depth should be 2"


class TestDistributiveProtocols:
    """Distributive protocols (lock, iterator) should have richer structure."""

    def _get_st(self, name: str) -> str:
        traces = [Trace.from_labels(t) for t in PYTHON_PROTOCOLS[name]["traces"]]
        return pretty(infer_from_traces(traces))

    def test_lock_is_distributive(self):
        r = analyze_single("lock", "Python", self._get_st("threading_lock"))
        assert r.classification == "distributive"

    def test_lock_rank_is_valuation(self):
        r = analyze_single("lock", "Python", self._get_st("threading_lock"))
        assert r.rank_is_valuation

    def test_lock_higher_dimension(self):
        """Distributive lattices should have higher valuation dimension."""
        r = analyze_single("lock", "Python", self._get_st("threading_lock"))
        assert r.valuation_dimension >= 2

    def test_iterator_is_distributive(self):
        r = analyze_single("iter", "Python", self._get_st("iterator_protocol"))
        assert r.classification == "distributive"

    def test_iterator_rank_is_valuation(self):
        r = analyze_single("iter", "Python", self._get_st("iterator_protocol"))
        assert r.rank_is_valuation


class TestNonModularProtocols:
    """Non-modular protocols should have rank NOT as valuation."""

    def _get_st(self, name: str) -> str:
        traces = [Trace.from_labels(t) for t in PYTHON_PROTOCOLS[name]["traces"]]
        return pretty(infer_from_traces(traces))

    def test_file_not_modular(self):
        r = analyze_single("file", "Python", self._get_st("file_object"))
        assert r.classification == "lattice"  # non-modular, just "lattice"

    def test_file_rank_not_valuation(self):
        r = analyze_single("file", "Python", self._get_st("file_object"))
        assert not r.rank_is_valuation

    def test_file_not_graded(self):
        r = analyze_single("file", "Python", self._get_st("file_object"))
        assert not r.is_graded

    def test_sqlite3_rank_not_valuation(self):
        r = analyze_single("sqlite3", "Python", self._get_st("sqlite3_connection"))
        assert not r.rank_is_valuation


# ---------------------------------------------------------------------------
# Aggregate tests
# ---------------------------------------------------------------------------

class TestAggregateAnalysis:
    def test_analyze_all_runs(self):
        report = analyze_all()
        assert report.total == 20

    def test_rank_valuation_rate(self):
        """Rank should be valuation for ≥60% of protocols (modular + distributive)."""
        report = analyze_all()
        assert report.rank_valuation_rate >= 0.60

    def test_report_formatting(self):
        report = analyze_all()
        text = print_toolkit_report(report)
        assert "MODULAR TOOLKIT" in text
        assert "BY CLASSIFICATION" in text


# ---------------------------------------------------------------------------
# Core hypothesis: classification predicts toolkit availability
# ---------------------------------------------------------------------------

class TestToolkitHierarchy:
    """The toolkit hierarchy theorem (Step 29d) should hold empirically."""

    def test_distributive_implies_rank_valuation(self):
        """Every distributive protocol should have rank as valuation."""
        report = analyze_all()
        for r in report.results:
            if r.classification == "distributive":
                assert r.rank_is_valuation, f"{r.name}: distributive but rank not valuation"

    def test_modular_implies_rank_valuation(self):
        """Every modular protocol should have rank as valuation."""
        report = analyze_all()
        for r in report.results:
            if r.classification == "modular":
                assert r.rank_is_valuation, f"{r.name}: modular but rank not valuation"

    def test_non_modular_implies_rank_not_valuation(self):
        """Non-modular protocols should NOT have rank as valuation."""
        report = analyze_all()
        for r in report.results:
            if r.classification == "lattice":  # non-modular
                assert not r.rank_is_valuation, f"{r.name}: non-modular but rank is valuation"

    def test_classification_predicts_grading(self):
        """Modular REST APIs should be graded; non-modular should not."""
        report = analyze_all()
        for r in report.results:
            if r.source == "REST":
                assert r.is_graded, f"{r.name}: REST should be graded"
