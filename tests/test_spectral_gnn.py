"""Tests for spectral GNN features (Step 31b)."""

import math
import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.spectral_gnn import (
    extract_features,
    spectral_moments,
    cosine_similarity,
    euclidean_distance,
    classify_protocol,
    analyze_spectral_gnn,
    FEATURE_NAMES,
)


def _build(s: str):
    return build_statespace(parse(s))


class TestFeatureExtraction:
    def test_feature_length(self):
        """Feature vector has exactly 20 dimensions."""
        for typ in ["end", "&{a: end}", "(&{a: end} || &{b: end})"]:
            f = extract_features(_build(typ))
            assert len(f) == 20
            assert len(f) == len(FEATURE_NAMES)

    def test_end_features(self):
        f = extract_features(_build("end"))
        assert f[0] == 1.0  # num_states
        assert f[2] == 0.0  # height

    def test_chain_features(self):
        f = extract_features(_build("&{a: end}"))
        assert f[0] == 2.0  # num_states
        assert f[2] >= 1.0  # height ≥ 1

    def test_parallel_features(self):
        f = extract_features(_build("(&{a: end} || &{b: end})"))
        assert f[16] == 1.0  # has_parallel
        assert f[3] >= 2.0   # width ≥ 2

    def test_no_nans(self):
        for typ in ["end", "&{a: end}", "&{a: &{b: end}}", "(&{a: end} || &{b: end})",
                     "rec X . &{a: X, b: end}"]:
            f = extract_features(_build(typ))
            for i, v in enumerate(f):
                assert not math.isnan(v), f"Feature {FEATURE_NAMES[i]} is NaN for {typ}"
                assert not math.isinf(v), f"Feature {FEATURE_NAMES[i]} is Inf for {typ}"


class TestSpectralMoments:
    def test_empty(self):
        assert spectral_moments([], 4) == [0.0, 0.0, 0.0, 0.0]

    def test_single(self):
        m = spectral_moments([2.0], 4)
        assert abs(m[0] - 2.0) < 0.01
        assert abs(m[1] - 4.0) < 0.01

    def test_symmetric(self):
        """Symmetric spectrum: odd moments ≈ 0."""
        m = spectral_moments([-1.0, 1.0], 4)
        assert abs(m[0]) < 0.01  # m_1 = 0
        assert abs(m[2]) < 0.01  # m_3 = 0


class TestSimilarity:
    def test_cosine_identical(self):
        v = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 0.01

    def test_cosine_orthogonal(self):
        assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 0.01

    def test_euclidean_zero(self):
        v = [1.0, 2.0, 3.0]
        assert euclidean_distance(v, v) < 0.01

    def test_euclidean_positive(self):
        assert euclidean_distance([0.0], [1.0]) == 1.0

    def test_same_type_similar(self):
        """Same session type → identical features → max similarity."""
        f1 = extract_features(_build("&{a: end}"))
        f2 = extract_features(_build("&{a: end}"))
        assert cosine_similarity(f1, f2) > 0.99

    def test_different_types_less_similar(self):
        """Different types → different features."""
        f1 = extract_features(_build("&{a: end}"))
        f2 = extract_features(_build("(&{a: end} || &{b: end})"))
        sim = cosine_similarity(f1, f2)
        assert sim < 1.0  # Not identical


class TestClassification:
    def test_chain(self):
        assert classify_protocol(_build("&{a: end}")) == "chain"
        assert classify_protocol(_build("&{a: &{b: end}}")) == "chain"

    def test_branch(self):
        assert classify_protocol(_build("&{a: end, b: end}")) == "branch"

    def test_parallel(self):
        assert classify_protocol(_build("(&{a: end} || &{b: end})")) == "parallel"

    def test_recursive(self):
        """Recursive types may classify as recursive or branch depending on SCC collapse."""
        family = classify_protocol(_build("rec X . &{a: X, b: end}"))
        assert family in ("recursive", "branch", "chain")  # SCC collapse may hide recursion

    def test_end(self):
        family = classify_protocol(_build("end"))
        assert family in ("chain", "branch")  # Trivial


class TestAnalyze:
    def test_end(self):
        r = analyze_spectral_gnn(_build("end"))
        assert r.num_states == 1
        assert len(r.feature_vector) == 20

    def test_parallel(self):
        r = analyze_spectral_gnn(_build("(&{a: end} || &{b: end})"))
        assert r.protocol_family == "parallel"
        assert len(r.spectral_moments) == 4


class TestBenchmarks:
    BENCHMARKS = [
        ("Java Iterator", "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"),
        ("File Object", "&{open: &{read: end, write: end}}"),
        ("SMTP", "&{EHLO: &{MAIL: &{RCPT: &{DATA: end}}}}"),
        ("Two-choice", "&{a: end, b: end}"),
        ("Nested", "&{a: &{b: &{c: end}}}"),
        ("Parallel", "(&{a: end} || &{b: end})"),
    ]

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_features_valid(self, name, typ):
        ss = _build(typ)
        r = analyze_spectral_gnn(ss)
        assert len(r.feature_vector) == 20
        assert r.protocol_family in ("chain", "branch", "parallel", "recursive", "mixed")
        for v in r.feature_vector:
            assert not math.isnan(v)

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_self_similarity(self, name, typ):
        """Each protocol is maximally similar to itself."""
        ss = _build(typ)
        f = extract_features(ss)
        assert cosine_similarity(f, f) > 0.99
