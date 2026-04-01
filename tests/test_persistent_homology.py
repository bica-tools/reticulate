"""Tests for persistent homology of session type lattices (Step 33c).

Tests cover:
  - Filtration construction (rank, reverse rank, sublevel)
  - Column reduction / persistence computation
  - Persistence invariants (total, max, entropy)
  - Bottleneck and Wasserstein distances
  - Full analysis pipeline
  - Benchmark protocol analysis
"""

from __future__ import annotations

import math

import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.persistent_homology import (
    PersistencePair,
    PersistenceDiagram,
    Filtration,
    PersistenceAnalysis,
    rank_filtration,
    reverse_rank_filtration,
    sublevel_filtration,
    compute_persistence,
    total_persistence,
    max_persistence,
    persistence_entropy,
    bottleneck_distance,
    wasserstein_distance,
    persistence_pairs,
    betti_barcodes,
    analyze_persistence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ss(type_str: str):
    """Parse a session type and build state space."""
    return build_statespace(parse(type_str))


# ---------------------------------------------------------------------------
# Test: PersistencePair
# ---------------------------------------------------------------------------

class TestPersistencePair:
    def test_finite_pair(self):
        p = PersistencePair(birth=1.0, death=3.0, dimension=0)
        assert p.persistence == 2.0
        assert not p.is_infinite

    def test_infinite_pair(self):
        p = PersistencePair(birth=0.0, death=float('inf'), dimension=0)
        assert p.persistence == float('inf')
        assert p.is_infinite

    def test_zero_persistence(self):
        p = PersistencePair(birth=2.0, death=2.0, dimension=1)
        assert p.persistence == 0.0
        assert not p.is_infinite

    def test_dimension(self):
        p0 = PersistencePair(birth=0, death=1, dimension=0)
        p1 = PersistencePair(birth=0, death=1, dimension=1)
        assert p0.dimension == 0
        assert p1.dimension == 1


# ---------------------------------------------------------------------------
# Test: PersistenceDiagram
# ---------------------------------------------------------------------------

class TestPersistenceDiagram:
    def test_empty_diagram(self):
        pd = PersistenceDiagram(pairs=(), infinite_pairs=(), max_filtration=0.0)
        assert pd.num_pairs == 0
        assert pd.all_pairs == ()

    def test_mixed_diagram(self):
        fp = PersistencePair(birth=0, death=2, dimension=0)
        ip = PersistencePair(birth=0, death=float('inf'), dimension=0)
        pd = PersistenceDiagram(pairs=(fp,), infinite_pairs=(ip,), max_filtration=3.0)
        assert pd.num_pairs == 2
        assert len(pd.all_pairs) == 2

    def test_pairs_in_dim(self):
        p0 = PersistencePair(birth=0, death=1, dimension=0)
        p1 = PersistencePair(birth=1, death=2, dimension=1)
        pd = PersistenceDiagram(pairs=(p0, p1), infinite_pairs=(), max_filtration=2.0)
        assert len(pd.pairs_in_dim(0)) == 1
        assert len(pd.pairs_in_dim(1)) == 1
        assert len(pd.pairs_in_dim(2)) == 0


# ---------------------------------------------------------------------------
# Test: Filtration construction
# ---------------------------------------------------------------------------

class TestRankFiltration:
    def test_end_type(self):
        ss = _ss("end")
        filt = rank_filtration(ss)
        assert filt.num_simplices >= 1
        assert filt.max_level >= 0

    def test_simple_branch(self):
        ss = _ss("&{a: end}")
        filt = rank_filtration(ss)
        assert filt.num_simplices >= 2  # At least 2 vertices
        assert filt.max_level >= 0

    def test_chain_ordering(self):
        ss = _ss("&{a: &{b: end}}")
        filt = rank_filtration(ss)
        # Should have vertices and edges
        vertices = [s for s in filt.simplices if len(s) == 1]
        edges = [s for s in filt.simplices if len(s) == 2]
        assert len(vertices) >= 2
        assert len(edges) >= 1

    def test_flat_branch(self):
        ss = _ss("&{a: end, b: end}")
        filt = rank_filtration(ss)
        assert filt.num_simplices >= 3  # top, bottom, + edges

    def test_parallel(self):
        ss = _ss("(&{a: end} || &{b: end})")
        filt = rank_filtration(ss)
        assert filt.num_simplices >= 4  # Product has 4 states


class TestReverseRankFiltration:
    def test_basic(self):
        ss = _ss("&{a: &{b: end}}")
        filt = reverse_rank_filtration(ss)
        assert filt.num_simplices >= 2
        # Top should have filtration value 0
        for s, v in zip(filt.simplices, filt.filtration_values):
            if len(s) == 1 and s[0] == ss.top:
                assert v == 0.0

    def test_matches_rank_simplex_count(self):
        ss = _ss("&{a: end, b: end}")
        r_filt = rank_filtration(ss)
        rr_filt = reverse_rank_filtration(ss)
        assert r_filt.num_simplices == rr_filt.num_simplices


class TestSublevelFiltration:
    def test_from_top(self):
        ss = _ss("&{a: &{b: end}}")
        filt = sublevel_filtration(ss, ss.top)
        # Top has distance 0
        for s, v in zip(filt.simplices, filt.filtration_values):
            if len(s) == 1 and s[0] == ss.top:
                assert v == 0.0

    def test_from_bottom(self):
        ss = _ss("&{a: &{b: end}}")
        filt = sublevel_filtration(ss, ss.bottom)
        # Bottom has distance 0
        for s, v in zip(filt.simplices, filt.filtration_values):
            if len(s) == 1 and s[0] == ss.bottom:
                assert v == 0.0


# ---------------------------------------------------------------------------
# Test: Persistence computation
# ---------------------------------------------------------------------------

class TestComputePersistence:
    def test_empty_filtration(self):
        filt = Filtration(simplices=[], filtration_values=[], max_level=0.0, num_simplices=0)
        pd = compute_persistence(filt)
        assert pd.num_pairs == 0

    def test_single_vertex(self):
        filt = Filtration(
            simplices=[(0,)],
            filtration_values=[0.0],
            max_level=0.0,
            num_simplices=1,
        )
        pd = compute_persistence(filt)
        assert len(pd.infinite_pairs) == 1
        assert pd.infinite_pairs[0].dimension == 0

    def test_two_vertices_one_edge(self):
        filt = Filtration(
            simplices=[(0,), (1,), (0, 1)],
            filtration_values=[0.0, 1.0, 1.0],
            max_level=1.0,
            num_simplices=3,
        )
        pd = compute_persistence(filt)
        # At least one infinite H0 bar
        assert pd.num_pairs >= 1

    def test_chain_type(self):
        ss = _ss("&{a: &{b: end}}")
        filt = rank_filtration(ss)
        pd = compute_persistence(filt)
        inf_dim0 = [p for p in pd.infinite_pairs if p.dimension == 0]
        assert len(inf_dim0) >= 1

    def test_branch_type(self):
        ss = _ss("&{a: end, b: end}")
        filt = rank_filtration(ss)
        pd = compute_persistence(filt)
        assert pd.num_pairs >= 1

    def test_parallel_type(self):
        ss = _ss("(&{a: end} || &{b: end})")
        filt = rank_filtration(ss)
        pd = compute_persistence(filt)
        assert pd.num_pairs >= 1


# ---------------------------------------------------------------------------
# Test: Persistence invariants
# ---------------------------------------------------------------------------

class TestPersistenceInvariants:
    def test_total_persistence_empty(self):
        pd = PersistenceDiagram(pairs=(), infinite_pairs=(), max_filtration=0.0)
        assert total_persistence(pd) == 0.0

    def test_total_persistence_finite(self):
        p1 = PersistencePair(birth=0, death=2, dimension=0)
        p2 = PersistencePair(birth=1, death=4, dimension=0)
        pd = PersistenceDiagram(pairs=(p1, p2), infinite_pairs=(), max_filtration=4.0)
        assert total_persistence(pd) == 5.0  # 2 + 3

    def test_max_persistence_empty(self):
        pd = PersistenceDiagram(pairs=(), infinite_pairs=(), max_filtration=0.0)
        assert max_persistence(pd) == 0.0

    def test_max_persistence_finite(self):
        p1 = PersistencePair(birth=0, death=2, dimension=0)
        p2 = PersistencePair(birth=1, death=5, dimension=0)
        pd = PersistenceDiagram(pairs=(p1, p2), infinite_pairs=(), max_filtration=5.0)
        assert max_persistence(pd) == 4.0

    def test_entropy_empty(self):
        pd = PersistenceDiagram(pairs=(), infinite_pairs=(), max_filtration=0.0)
        assert persistence_entropy(pd) == 0.0

    def test_entropy_single_bar(self):
        p = PersistencePair(birth=0, death=3, dimension=0)
        pd = PersistenceDiagram(pairs=(p,), infinite_pairs=(), max_filtration=3.0)
        assert persistence_entropy(pd) == 0.0

    def test_entropy_uniform(self):
        p1 = PersistencePair(birth=0, death=2, dimension=0)
        p2 = PersistencePair(birth=1, death=3, dimension=0)
        pd = PersistenceDiagram(pairs=(p1, p2), infinite_pairs=(), max_filtration=3.0)
        assert abs(persistence_entropy(pd) - 1.0) < 1e-10

    def test_entropy_nonuniform(self):
        p1 = PersistencePair(birth=0, death=1, dimension=0)
        p2 = PersistencePair(birth=0, death=3, dimension=0)
        pd = PersistenceDiagram(pairs=(p1, p2), infinite_pairs=(), max_filtration=3.0)
        ent = persistence_entropy(pd)
        assert 0 < ent < 1.0


# ---------------------------------------------------------------------------
# Test: Distances
# ---------------------------------------------------------------------------

class TestBottleneckDistance:
    def test_identical_diagrams(self):
        p = PersistencePair(birth=0, death=2, dimension=0)
        pd = PersistenceDiagram(pairs=(p,), infinite_pairs=(), max_filtration=2.0)
        assert bottleneck_distance(pd, pd) == 0.0

    def test_empty_diagrams(self):
        pd1 = PersistenceDiagram(pairs=(), infinite_pairs=(), max_filtration=0.0)
        pd2 = PersistenceDiagram(pairs=(), infinite_pairs=(), max_filtration=0.0)
        assert bottleneck_distance(pd1, pd2) == 0.0

    def test_one_empty(self):
        p = PersistencePair(birth=0, death=4, dimension=0)
        pd1 = PersistenceDiagram(pairs=(p,), infinite_pairs=(), max_filtration=4.0)
        pd2 = PersistenceDiagram(pairs=(), infinite_pairs=(), max_filtration=0.0)
        dist = bottleneck_distance(pd1, pd2)
        assert dist > 0

    def test_symmetry(self):
        p1 = PersistencePair(birth=0, death=2, dimension=0)
        p2 = PersistencePair(birth=0, death=3, dimension=0)
        pd1 = PersistenceDiagram(pairs=(p1,), infinite_pairs=(), max_filtration=2.0)
        pd2 = PersistenceDiagram(pairs=(p2,), infinite_pairs=(), max_filtration=3.0)
        assert bottleneck_distance(pd1, pd2) == bottleneck_distance(pd2, pd1)

    def test_nonnegative(self):
        p1 = PersistencePair(birth=0, death=2, dimension=0)
        p2 = PersistencePair(birth=1, death=5, dimension=0)
        pd1 = PersistenceDiagram(pairs=(p1,), infinite_pairs=(), max_filtration=2.0)
        pd2 = PersistenceDiagram(pairs=(p2,), infinite_pairs=(), max_filtration=5.0)
        assert bottleneck_distance(pd1, pd2) >= 0


class TestWassersteinDistance:
    def test_identical(self):
        p = PersistencePair(birth=0, death=2, dimension=0)
        pd = PersistenceDiagram(pairs=(p,), infinite_pairs=(), max_filtration=2.0)
        assert wasserstein_distance(pd, pd) == 0.0

    def test_empty(self):
        pd = PersistenceDiagram(pairs=(), infinite_pairs=(), max_filtration=0.0)
        assert wasserstein_distance(pd, pd) == 0.0

    def test_symmetry(self):
        p1 = PersistencePair(birth=0, death=2, dimension=0)
        p2 = PersistencePair(birth=0, death=4, dimension=0)
        pd1 = PersistenceDiagram(pairs=(p1,), infinite_pairs=(), max_filtration=2.0)
        pd2 = PersistenceDiagram(pairs=(p2,), infinite_pairs=(), max_filtration=4.0)
        assert wasserstein_distance(pd1, pd2) == wasserstein_distance(pd2, pd1)

    def test_nonnegative(self):
        p1 = PersistencePair(birth=0, death=1, dimension=0)
        p2 = PersistencePair(birth=2, death=5, dimension=0)
        pd1 = PersistenceDiagram(pairs=(p1,), infinite_pairs=(), max_filtration=1.0)
        pd2 = PersistenceDiagram(pairs=(p2,), infinite_pairs=(), max_filtration=5.0)
        assert wasserstein_distance(pd1, pd2) >= 0


# ---------------------------------------------------------------------------
# Test: Convenience functions
# ---------------------------------------------------------------------------

class TestConvenienceFunctions:
    def test_persistence_pairs_chain(self):
        ss = _ss("&{a: &{b: end}}")
        pairs = persistence_pairs(ss)
        assert len(pairs) >= 1

    def test_persistence_pairs_branch(self):
        ss = _ss("&{a: end, b: end}")
        pairs = persistence_pairs(ss)
        assert len(pairs) >= 1

    def test_betti_barcodes(self):
        ss = _ss("&{a: &{b: end}}")
        bc = betti_barcodes(ss)
        assert 0 in bc
        assert len(bc[0]) >= 1


# ---------------------------------------------------------------------------
# Test: Full analysis
# ---------------------------------------------------------------------------

class TestAnalyzePersistence:
    def test_end(self):
        ss = _ss("end")
        result = analyze_persistence(ss)
        assert isinstance(result, PersistenceAnalysis)
        assert result.num_states == len(ss.states)

    def test_chain(self):
        ss = _ss("&{a: &{b: end}}")
        result = analyze_persistence(ss)
        assert result.betti_0 >= 1
        assert result.total_persistence >= 0
        assert result.max_persistence >= 0
        assert result.persistence_entropy >= 0

    def test_branch(self):
        ss = _ss("&{a: end, b: end}")
        result = analyze_persistence(ss)
        assert result.betti_0 >= 1

    def test_parallel(self):
        ss = _ss("(&{a: end} || &{b: end})")
        result = analyze_persistence(ss)
        assert result.betti_0 >= 1
        assert result.num_states >= 4

    def test_recursive(self):
        ss = _ss("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        result = analyze_persistence(ss)
        assert result.betti_0 >= 1

    def test_selection(self):
        ss = _ss("+{ok: end, err: end}")
        result = analyze_persistence(ss)
        assert result.betti_0 >= 1


# ---------------------------------------------------------------------------
# Test: Benchmark protocols
# ---------------------------------------------------------------------------

class TestBenchmarkProtocols:
    """Test persistent homology on real-world benchmark protocols."""

    def test_java_iterator(self):
        ss = _ss("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        result = analyze_persistence(ss)
        assert result.betti_0 >= 1
        assert result.num_states == 4

    def test_file_object(self):
        ss = _ss("&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}")
        result = analyze_persistence(ss)
        assert result.betti_0 >= 1
        assert result.num_states == 5

    def test_smtp(self):
        ss = _ss(
            "&{connect: &{ehlo: rec X . &{mail: &{rcpt: &{data: "
            "+{OK: X, ERR: X}}}, quit: end}}}"
        )
        result = analyze_persistence(ss)
        assert result.betti_0 >= 1
        assert result.num_states == 7

    def test_http(self):
        ss = _ss(
            "&{connect: rec X . &{request: +{OK200: &{readBody: X}, "
            "ERR4xx: X, ERR5xx: X}, close: end}}"
        )
        result = analyze_persistence(ss)
        assert result.betti_0 >= 1

    def test_simple_parallel(self):
        ss = _ss("(&{a: end} || &{b: end})")
        result = analyze_persistence(ss)
        assert result.betti_0 >= 1

    def test_deep_chain(self):
        ss = _ss("&{a: &{b: &{c: &{d: end}}}}")
        result = analyze_persistence(ss)
        assert result.betti_0 >= 1
        assert result.num_states == 5

    def test_wide_branch(self):
        ss = _ss("&{a: end, b: end, c: end, d: end}")
        result = analyze_persistence(ss)
        assert result.betti_0 >= 1

    def test_nested_selection(self):
        ss = _ss("&{init: +{ok: &{use: end}, err: end}}")
        result = analyze_persistence(ss)
        assert result.betti_0 >= 1


# ---------------------------------------------------------------------------
# Test: Distance between benchmarks
# ---------------------------------------------------------------------------

class TestProtocolDistances:
    def test_same_protocol_distance_zero(self):
        ss = _ss("&{a: end, b: end}")
        filt = rank_filtration(ss)
        pd = compute_persistence(filt)
        assert bottleneck_distance(pd, pd) == 0.0

    def test_different_protocols_nonnegative(self):
        ss1 = _ss("&{a: end}")
        ss2 = _ss("&{a: end, b: end}")
        pd1 = compute_persistence(rank_filtration(ss1))
        pd2 = compute_persistence(rank_filtration(ss2))
        assert bottleneck_distance(pd1, pd2) >= 0.0

    def test_wasserstein_between_protocols(self):
        ss1 = _ss("&{a: &{b: end}}")
        ss2 = _ss("&{a: end, b: end}")
        pd1 = compute_persistence(rank_filtration(ss1))
        pd2 = compute_persistence(rank_filtration(ss2))
        assert wasserstein_distance(pd1, pd2) >= 0.0


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_state(self):
        ss = _ss("end")
        result = analyze_persistence(ss)
        assert result.num_states >= 1

    def test_two_state_chain(self):
        ss = _ss("&{a: end}")
        result = analyze_persistence(ss)
        assert result.num_states == 2

    def test_entropy_no_finite_bars(self):
        """If no finite bars, entropy should be 0."""
        pd = PersistenceDiagram(
            pairs=(),
            infinite_pairs=(PersistencePair(0, float('inf'), 0),),
            max_filtration=5.0,
        )
        assert persistence_entropy(pd) == 0.0
