"""Tests for spectral clustering of session type lattices (Step 31a)."""

import math
import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.spectral import (
    ClusteringResult,
    SimilarityResult,
    SpectralCluster,
    SpectralFeatures,
    cluster_benchmarks,
    find_similar,
    spectral_distance,
    spectral_features,
)


def _ss(type_string: str):
    return build_statespace(parse(type_string))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


class TestSpectralFeatures:
    def test_end_features(self):
        ss = _ss("end")
        f = spectral_features(ss)
        assert f.num_states == 1
        assert f.num_transitions == 0
        assert f.spectral_radius == 0.0
        assert f.fiedler_value == 0.0
        assert f.von_neumann_entropy == 0.0
        assert f.width == 1
        assert f.height == 0

    def test_simple_branch(self):
        ss = _ss("&{a: end}")
        f = spectral_features(ss)
        assert f.num_states == 2
        assert f.num_transitions == 1
        assert f.width == 1
        assert f.height == 1
        assert f.spectral_radius > 0.0

    def test_two_branch(self):
        ss = _ss("&{a: end, b: end}")
        f = spectral_features(ss)
        assert f.num_states == 2
        assert f.num_transitions == 2
        assert f.width == 1

    def test_chain(self):
        ss = _ss("&{a: &{b: end}}")
        f = spectral_features(ss)
        assert f.num_states == 3
        assert f.num_transitions == 2
        assert f.height == 2
        assert f.width == 1

    def test_diamond(self):
        ss = _ss("&{a: &{c: end}, b: &{c: end}}")
        f = spectral_features(ss)
        assert f.num_states >= 3
        assert f.width >= 1

    def test_parallel_features(self):
        ss = _ss("(&{a: end} || &{b: end})")
        f = spectral_features(ss)
        assert f.num_states >= 3
        assert f.num_transitions >= 2
        # Parallel should produce a wider lattice
        assert f.width >= 1

    def test_recursive_features(self):
        ss = _ss("rec X . &{a: X, b: end}")
        f = spectral_features(ss)
        assert f.num_states >= 2
        assert f.num_transitions >= 2

    def test_as_vector_length(self):
        ss = _ss("&{a: end}")
        f = spectral_features(ss)
        v = f.as_vector()
        assert len(v) == 7
        assert all(isinstance(x, float) for x in v)

    def test_as_vector_values(self):
        ss = _ss("&{a: end}")
        f = spectral_features(ss)
        v = f.as_vector()
        assert v[0] == f.spectral_radius
        assert v[1] == f.fiedler_value
        assert v[2] == f.von_neumann_entropy
        assert v[3] == float(f.width)
        assert v[4] == float(f.height)
        assert v[5] == float(f.num_states)
        assert v[6] == float(f.num_transitions)

    def test_select_features(self):
        ss = _ss("+{a: end, b: end}")
        f = spectral_features(ss)
        assert f.num_states == 2
        assert f.num_transitions == 2

    def test_iterator_features(self):
        ss = _ss("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        f = spectral_features(ss)
        assert f.num_states == 4
        assert f.num_transitions == 4


# ---------------------------------------------------------------------------
# Distance computation
# ---------------------------------------------------------------------------


class TestSpectralDistance:
    def test_self_distance_zero(self):
        ss = _ss("&{a: end}")
        d = spectral_distance(ss, ss)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_symmetric(self):
        ss1 = _ss("&{a: end}")
        ss2 = _ss("&{a: end, b: end}")
        d1 = spectral_distance(ss1, ss2)
        d2 = spectral_distance(ss2, ss1)
        assert d1 == pytest.approx(d2, abs=1e-10)

    def test_triangle_inequality(self):
        ss1 = _ss("&{a: end}")
        ss2 = _ss("&{a: &{b: end}}")
        ss3 = _ss("&{a: end, b: end}")
        d12 = spectral_distance(ss1, ss2)
        d23 = spectral_distance(ss2, ss3)
        d13 = spectral_distance(ss1, ss3)
        assert d13 <= d12 + d23 + 1e-10

    def test_identical_types_zero(self):
        ss1 = _ss("&{a: &{b: end}}")
        ss2 = _ss("&{a: &{b: end}}")
        d = spectral_distance(ss1, ss2)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_different_types_positive(self):
        ss1 = _ss("end")
        ss2 = _ss("&{a: &{b: &{c: end}}}")
        d = spectral_distance(ss1, ss2)
        assert d > 0.0


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


class TestClusterBenchmarks:
    @pytest.fixture
    def small_benchmarks(self):
        return [
            ("end", _ss("end")),
            ("branch1", _ss("&{a: end}")),
            ("branch2", _ss("&{a: end, b: end}")),
            ("chain2", _ss("&{a: &{b: end}}")),
            ("chain3", _ss("&{a: &{b: &{c: end}}}")),
            ("select", _ss("+{a: end, b: end}")),
        ]

    def test_basic_clustering(self, small_benchmarks):
        result = cluster_benchmarks(small_benchmarks, k=2)
        assert isinstance(result, ClusteringResult)
        assert result.k == 2
        assert len(result.clusters) == 2

    def test_all_assigned(self, small_benchmarks):
        result = cluster_benchmarks(small_benchmarks, k=2)
        assert len(result.assignments) == len(small_benchmarks)
        for i in range(len(small_benchmarks)):
            assert i in result.assignments
            assert 0 <= result.assignments[i] < 2

    def test_cluster_members(self, small_benchmarks):
        result = cluster_benchmarks(small_benchmarks, k=2)
        total_members = sum(len(c.members) for c in result.clusters)
        assert total_members == len(small_benchmarks)

    def test_k_equals_n(self, small_benchmarks):
        result = cluster_benchmarks(small_benchmarks, k=len(small_benchmarks))
        assert result.k == len(small_benchmarks)

    def test_k_exceeds_n_clamped(self, small_benchmarks):
        result = cluster_benchmarks(small_benchmarks, k=100)
        assert result.k == len(small_benchmarks)

    def test_k_equals_one(self, small_benchmarks):
        result = cluster_benchmarks(small_benchmarks, k=1)
        assert result.k == 1
        assert len(result.clusters) == 1
        assert len(result.clusters[0].members) == len(small_benchmarks)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            cluster_benchmarks([], k=2)

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            cluster_benchmarks([("x", _ss("end"))], k=0)

    def test_features_stored(self, small_benchmarks):
        result = cluster_benchmarks(small_benchmarks, k=2)
        assert len(result.features) == len(small_benchmarks)
        for name, feat in result.features:
            assert isinstance(feat, SpectralFeatures)

    def test_inertia_nonnegative(self, small_benchmarks):
        result = cluster_benchmarks(small_benchmarks, k=2)
        assert result.total_inertia >= 0.0
        for c in result.clusters:
            assert c.inertia >= 0.0

    def test_silhouette_range(self, small_benchmarks):
        result = cluster_benchmarks(small_benchmarks, k=2)
        assert -1.0 <= result.silhouette_avg <= 1.0

    def test_single_benchmark(self):
        result = cluster_benchmarks([("end", _ss("end"))], k=1)
        assert result.k == 1
        assert len(result.clusters[0].members) == 1


# ---------------------------------------------------------------------------
# Similarity search
# ---------------------------------------------------------------------------


class TestFindSimilar:
    @pytest.fixture
    def benchmarks(self):
        return [
            ("end", _ss("end")),
            ("branch1", _ss("&{a: end}")),
            ("chain2", _ss("&{a: &{b: end}}")),
            ("chain3", _ss("&{a: &{b: &{c: end}}}")),
            ("wide", _ss("&{a: end, b: end, c: end}")),
        ]

    def test_basic_similarity(self, benchmarks):
        query = _ss("&{a: end}")
        result = find_similar(query, benchmarks, n=3)
        assert isinstance(result, SimilarityResult)
        assert len(result.neighbors) == 3

    def test_self_is_nearest(self, benchmarks):
        query = _ss("&{a: end}")
        result = find_similar(query, benchmarks, n=5)
        # "branch1" should be the nearest (distance ~0)
        assert result.neighbors[0][2] == "branch1"
        assert result.neighbors[0][0] == pytest.approx(0.0, abs=1e-6)

    def test_sorted_ascending(self, benchmarks):
        query = _ss("&{a: &{b: end}}")
        result = find_similar(query, benchmarks, n=5)
        for i in range(len(result.neighbors) - 1):
            assert result.neighbors[i][0] <= result.neighbors[i + 1][0] + 1e-10

    def test_n_exceeds_benchmarks(self, benchmarks):
        query = _ss("end")
        result = find_similar(query, benchmarks, n=100)
        assert len(result.neighbors) == len(benchmarks)

    def test_n_equals_one(self, benchmarks):
        query = _ss("end")
        result = find_similar(query, benchmarks, n=1)
        assert len(result.neighbors) == 1

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            find_similar(_ss("end"), [], n=1)

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            find_similar(_ss("end"), [("x", _ss("end"))], n=0)

    def test_query_features_present(self, benchmarks):
        query = _ss("&{a: end}")
        result = find_similar(query, benchmarks, n=3)
        assert isinstance(result.query_features, SpectralFeatures)


# ---------------------------------------------------------------------------
# Benchmark integration
# ---------------------------------------------------------------------------


class TestBenchmarkIntegration:
    """Test spectral features on real benchmark protocols."""

    def test_iterator_vs_file(self):
        it_ss = _ss("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        file_ss = _ss("&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}")
        d = spectral_distance(it_ss, file_ss)
        # They're similar but not identical
        assert d > 0.0

    def test_cluster_protocols(self):
        benchmarks = [
            ("iterator", _ss("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")),
            ("file", _ss("&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}")),
            ("simple", _ss("&{a: end}")),
            ("chain", _ss("&{a: &{b: &{c: end}}}")),
        ]
        result = cluster_benchmarks(benchmarks, k=2)
        assert result.k == 2
        assert len(result.assignments) == 4

    def test_find_similar_iterator(self):
        benchmarks = [
            ("file", _ss("&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}")),
            ("simple", _ss("&{a: end}")),
            ("chain", _ss("&{a: &{b: &{c: end}}}")),
        ]
        query = _ss("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        result = find_similar(query, benchmarks, n=3)
        # Iterator (4 states, 4 trans) is closest to chain (4 states, 3 trans)
        # by feature distance; file (5 states, 5 trans) is further
        nearest_name = result.neighbors[0][2]
        assert nearest_name in ("chain", "file")
        # "simple" with only 2 states should be the farthest
        assert result.neighbors[-1][2] == "simple"
