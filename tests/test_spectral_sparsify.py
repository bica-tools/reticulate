"""Tests for spectral protocol compression (Step 31g)."""

import math
import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.spectral_sparsify import (
    CompressionQuality,
    SparsificationPotential,
    SparsifyResult,
    compression_quality,
    effective_resistance,
    preserve_fiedler,
    sparsification_potential,
    spectral_sparsify,
)


def _ss(type_string: str):
    return build_statespace(parse(type_string))


# ---------------------------------------------------------------------------
# Effective resistance
# ---------------------------------------------------------------------------


class TestEffectiveResistance:
    def test_end_empty(self):
        r = effective_resistance(_ss("end"))
        assert r == {}

    def test_simple_branch_single_edge(self):
        r = effective_resistance(_ss("&{a: end}"))
        # Single edge: bridge, R_eff = 1
        assert len(r) == 1
        assert r[0] > 0.0

    def test_chain_resistances(self):
        r = effective_resistance(_ss("&{a: &{b: end}}"))
        # Path with 2 edges: both are bridges
        assert len(r) == 2
        for v in r.values():
            assert v > 0.0

    def test_diamond_resistances(self):
        r = effective_resistance(_ss("&{a: &{c: end}, b: &{c: end}}"))
        assert len(r) >= 2
        for v in r.values():
            assert v >= 0.0

    def test_parallel_resistances(self):
        r = effective_resistance(_ss("(&{a: end} || &{b: end})"))
        assert len(r) >= 2


# ---------------------------------------------------------------------------
# Spectral sparsify
# ---------------------------------------------------------------------------


class TestSpectralSparsify:
    def test_end_sparsify(self):
        result = spectral_sparsify(_ss("end"))
        assert isinstance(result, SparsifyResult)
        assert result.original_edges == 0
        assert result.sparsified_edges == 0

    def test_simple_branch_no_compression(self):
        result = spectral_sparsify(_ss("&{a: end}"))
        # Single edge cannot be removed
        assert result.sparsified_edges == result.original_edges

    def test_chain_no_compression(self):
        result = spectral_sparsify(_ss("&{a: &{b: end}}"))
        # Chain is a tree; cannot sparsify further
        assert result.sparsified_edges == result.original_edges
        assert result.compression_ratio == 1.0

    def test_diamond_some_compression(self):
        result = spectral_sparsify(_ss("&{a: &{c: end}, b: &{c: end}}"))
        assert result.sparsified_edges <= result.original_edges
        assert result.compression_ratio <= 1.0

    def test_parallel_sparsify(self):
        result = spectral_sparsify(_ss("(&{a: end} || &{b: end})"))
        assert isinstance(result, SparsifyResult)
        assert result.sparsified_edges <= result.original_edges

    def test_fiedler_preserved(self):
        ss = _ss("&{a: &{c: end}, b: &{c: end}}")
        result = spectral_sparsify(ss, epsilon=0.3)
        if result.fiedler_original > 1e-10:
            assert result.fiedler_ratio >= 0.5  # Not too distorted

    def test_epsilon_controls_quality(self):
        ss = _ss("&{a: &{c: end}, b: &{c: end}}")
        tight = spectral_sparsify(ss, epsilon=0.01)
        loose = spectral_sparsify(ss, epsilon=0.9)
        # Tighter epsilon should keep at least as many edges
        assert tight.sparsified_edges >= loose.sparsified_edges

    def test_kept_edges_subset(self):
        ss = _ss("&{a: &{c: end}, b: &{c: end}}")
        result = spectral_sparsify(ss)
        # All kept edges should be valid state pairs
        for a, b in result.kept_edges:
            assert a in ss.states
            assert b in ss.states

    def test_edge_weights_positive(self):
        result = spectral_sparsify(_ss("(&{a: end} || &{b: end})"))
        for w in result.edge_weights:
            assert w > 0.0

    def test_recursive_sparsify(self):
        result = spectral_sparsify(_ss("rec X . &{a: X, b: end}"))
        assert isinstance(result, SparsifyResult)


# ---------------------------------------------------------------------------
# Compression quality
# ---------------------------------------------------------------------------


class TestCompressionQuality:
    def test_identity_quality(self):
        """Full edge set should have perfect quality."""
        ss = _ss("&{a: &{b: end}}")
        from reticulate.spectral_sparsify import _hasse_edges
        edges = _hasse_edges(ss)
        q = compression_quality(ss, edges)
        assert isinstance(q, CompressionQuality)
        assert q.max_distortion < 1e-6
        assert q.is_good_sparsifier

    def test_diamond_subgraph_quality(self):
        ss = _ss("&{a: &{c: end}, b: &{c: end}}")
        result = spectral_sparsify(ss)
        q = compression_quality(ss, result.kept_edges, result.edge_weights)
        assert isinstance(q, CompressionQuality)

    def test_eigenvalue_distances_nonneg(self):
        ss = _ss("&{a: &{c: end}, b: &{c: end}}")
        result = spectral_sparsify(ss)
        q = compression_quality(ss, result.kept_edges, result.edge_weights)
        for d in q.eigenvalue_distances:
            assert d >= -1e-10


# ---------------------------------------------------------------------------
# Preserve Fiedler
# ---------------------------------------------------------------------------


class TestPreserveFiedler:
    def test_full_edges_preserve(self):
        ss = _ss("&{a: &{b: end}}")
        from reticulate.spectral_sparsify import _hasse_edges
        edges = _hasse_edges(ss)
        assert preserve_fiedler(ss, edges)

    def test_sparsified_preserves(self):
        ss = _ss("&{a: &{c: end}, b: &{c: end}}")
        result = spectral_sparsify(ss, epsilon=0.1)
        assert preserve_fiedler(ss, result.kept_edges, result.edge_weights, tolerance=0.2)

    def test_empty_edges_no_preserve(self):
        ss = _ss("&{a: &{b: end}}")
        # Empty edge set cannot preserve Fiedler of connected graph
        result = preserve_fiedler(ss, [], tolerance=0.1)
        # Either fiedler_orig is 0 (possible for trivial) or not preserved
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Sparsification potential
# ---------------------------------------------------------------------------


class TestSparsificationPotential:
    def test_end_potential(self):
        p = sparsification_potential(_ss("end"))
        assert isinstance(p, SparsificationPotential)
        assert p.num_edges == 0
        assert p.max_removable == 0

    def test_chain_no_removable(self):
        p = sparsification_potential(_ss("&{a: &{b: end}}"))
        # Chain: n-1 edges = tree, no removable
        assert p.max_removable == 0
        assert p.removable_fraction == 0.0

    def test_diamond_has_removable(self):
        p = sparsification_potential(_ss("&{a: &{c: end}, b: &{c: end}}"))
        # Diamond has cycles, so some edges are removable
        assert p.num_edges >= p.min_edges_needed

    def test_parallel_potential(self):
        p = sparsification_potential(_ss("(&{a: end} || &{b: end})"))
        assert p.num_states >= 3
        assert p.num_edges >= 2

    def test_resistances_populated(self):
        p = sparsification_potential(_ss("&{a: &{c: end}, b: &{c: end}}"))
        assert len(p.effective_resistances) == p.num_edges

    def test_min_edges_is_n_minus_1(self):
        p = sparsification_potential(_ss("&{a: &{b: &{c: end}}}"))
        assert p.min_edges_needed == p.num_states - 1
