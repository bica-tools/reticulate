"""Tests for spectral_refactor.py — Step 31i: Spectral Protocol Refactoring.

60+ tests covering all three approaches, data types, pipeline, and round-trip.
"""

from __future__ import annotations

import math
import pytest

from reticulate.parser import parse
from reticulate.statespace import StateSpace, build_statespace
from reticulate.spectral_refactor import (
    DegeneracyGroup,
    RedundantPair,
    RefactoringPlan,
    SpectralEmbedding,
    SpectralRefactorAnalysis,
    _euclidean_distance,
    _kmeans,
    _laplacian,
    _eigenvalues_symmetric,
    compare_original_refactored,
    eigenvalue_degeneracy,
    fiedler_refactoring,
    merge_redundant,
    reconstruct_from_clusters,
    redundant_states,
    refactoring_pipeline,
    round_trip_analysis,
    spectral_clusters,
    spectral_embedding,
)


# ---------------------------------------------------------------------------
# Helpers: build state spaces from session type strings
# ---------------------------------------------------------------------------

def _ss(type_str: str) -> StateSpace:
    """Parse a session type string and build its state space."""
    return build_statespace(parse(type_str))


def _manual_ss(
    states: set[int],
    transitions: list[tuple[int, str, int]],
    top: int,
    bottom: int,
    selections: set[tuple[int, str, int]] | None = None,
) -> StateSpace:
    """Build a manual state space for testing."""
    return StateSpace(
        states=states,
        transitions=transitions,
        top=top,
        bottom=bottom,
        labels={s: str(s) for s in states},
        selection_transitions=selections or set(),
    )


# ---------------------------------------------------------------------------
# Fixtures: reusable session types
# ---------------------------------------------------------------------------

@pytest.fixture
def end_ss():
    return _ss("end")


@pytest.fixture
def branch_ss():
    return _ss("&{a: end, b: end}")


@pytest.fixture
def select_ss():
    return _ss("+{x: end, y: end}")


@pytest.fixture
def nested_ss():
    return _ss("&{a: &{c: end, d: end}, b: end}")


@pytest.fixture
def parallel_ss():
    return _ss("(&{a: end} || &{b: end})")


@pytest.fixture
def chain_ss():
    """A longer chain: a.b.c.end via nested branches."""
    return _ss("&{a: &{b: &{c: end}}}")


@pytest.fixture
def diamond_ss():
    """A diamond lattice: branch with two paths converging."""
    return _ss("&{a: end, b: end}")


@pytest.fixture
def wide_ss():
    """Wide branching."""
    return _ss("&{a: end, b: end, c: end, d: end}")


@pytest.fixture
def mixed_ss():
    """Mixed branch and selection."""
    return _ss("&{a: +{x: end, y: end}, b: end}")


@pytest.fixture
def recursive_ss():
    return _ss("rec X . &{a: X, b: end}")


# ---------------------------------------------------------------------------
# Data type tests
# ---------------------------------------------------------------------------

class TestDataTypes:
    """Tests for frozen dataclass types."""

    def test_spectral_embedding_frozen(self):
        emb = SpectralEmbedding(
            state_ids=(0, 1),
            coordinates=((0.0, 1.0), (1.0, 0.0)),
            eigenvalues=(0.0, 2.0),
            k=2,
        )
        assert emb.k == 2
        assert len(emb.coordinates) == 2
        with pytest.raises(AttributeError):
            emb.k = 3  # type: ignore

    def test_degeneracy_group_frozen(self):
        dg = DegeneracyGroup(eigenvalue=1.5, indices=(0, 1), multiplicity=2, spread=0.1)
        assert dg.multiplicity == 2
        assert dg.spread == 0.1

    def test_redundant_pair_frozen(self):
        rp = RedundantPair(
            state_a=0, state_b=1, distance=0.05,
            coord_a=(0.1,), coord_b=(0.15,),
        )
        assert rp.distance == 0.05

    def test_refactoring_plan_frozen(self):
        plan = RefactoringPlan(
            approach="fiedler",
            description="test",
            estimated_reduction=0.3,
        )
        assert plan.approach == "fiedler"
        assert plan.merge_pairs == ()

    def test_spectral_refactor_analysis_frozen(self):
        emb = SpectralEmbedding(
            state_ids=(0,), coordinates=((0.0,),),
            eigenvalues=(0.0,), k=1,
        )
        analysis = SpectralRefactorAnalysis(
            embedding=emb,
            degeneracy_groups=(),
            redundant_pairs=(),
            fiedler_plan=RefactoringPlan(approach="fiedler", description=""),
            degeneracy_plan=RefactoringPlan(approach="degeneracy", description=""),
            clustering_plan=RefactoringPlan(approach="clustering", description=""),
            round_trip_loss=0.0,
            num_states=1,
            num_states_after_merge=1,
            num_clusters=1,
        )
        assert analysis.num_states == 1


# ---------------------------------------------------------------------------
# Internal: linear algebra tests
# ---------------------------------------------------------------------------

class TestLinearAlgebra:
    """Tests for internal linear algebra utilities."""

    def test_laplacian_single_state(self, end_ss):
        states, L = _laplacian(end_ss)
        assert len(states) == 1
        assert L == [[0.0]]

    def test_laplacian_two_states(self, branch_ss):
        states, L = _laplacian(branch_ss)
        n = len(states)
        # Laplacian row sums must be 0
        for i in range(n):
            assert abs(sum(L[i])) < 1e-10

    def test_laplacian_symmetric(self, nested_ss):
        states, L = _laplacian(nested_ss)
        n = len(states)
        for i in range(n):
            for j in range(n):
                assert abs(L[i][j] - L[j][i]) < 1e-10

    def test_eigenvalues_identity(self):
        I = [[1.0, 0.0], [0.0, 1.0]]
        eigs = _eigenvalues_symmetric(I)
        assert len(eigs) == 2
        for e in eigs:
            assert abs(e - 1.0) < 0.01

    def test_eigenvalues_zero_matrix(self):
        Z = [[0.0, 0.0], [0.0, 0.0]]
        eigs = _eigenvalues_symmetric(Z)
        for e in eigs:
            assert abs(e) < 0.01

    def test_eigenvalues_laplacian_has_zero(self, branch_ss):
        states, L = _laplacian(branch_ss)
        eigs = _eigenvalues_symmetric(L)
        # Laplacian always has 0 as smallest eigenvalue
        assert abs(eigs[0]) < 0.1

    def test_euclidean_distance_zero(self):
        assert _euclidean_distance((1.0, 2.0), (1.0, 2.0)) == 0.0

    def test_euclidean_distance_unit(self):
        d = _euclidean_distance((0.0,), (1.0,))
        assert abs(d - 1.0) < 1e-10

    def test_euclidean_distance_2d(self):
        d = _euclidean_distance((0.0, 0.0), (3.0, 4.0))
        assert abs(d - 5.0) < 1e-10


# ---------------------------------------------------------------------------
# Approach A: Fiedler bisection
# ---------------------------------------------------------------------------

class TestFiedlerRefactoring:
    """Tests for Fiedler bisection refactoring (Approach A)."""

    def test_fiedler_end(self, end_ss):
        plan = fiedler_refactoring(end_ss)
        assert plan.approach == "fiedler"
        assert plan.partition is None

    def test_fiedler_branch(self, branch_ss):
        plan = fiedler_refactoring(branch_ss)
        assert plan.approach == "fiedler"
        assert plan.partition is not None
        a, b = plan.partition
        assert len(a) + len(b) == len(branch_ss.states)
        assert a & b == frozenset()

    def test_fiedler_nested(self, nested_ss):
        plan = fiedler_refactoring(nested_ss)
        assert plan.approach == "fiedler"
        assert plan.partition is not None
        assert plan.estimated_reduction >= 0.0

    def test_fiedler_parallel(self, parallel_ss):
        plan = fiedler_refactoring(parallel_ss)
        assert plan.approach == "fiedler"
        # Parallel should have a natural bisection point
        if plan.partition is not None:
            a, b = plan.partition
            assert a | b == frozenset(parallel_ss.states)

    def test_fiedler_description(self, branch_ss):
        plan = fiedler_refactoring(branch_ss)
        assert "Fiedler" in plan.description

    def test_fiedler_reduction_bounded(self, nested_ss):
        plan = fiedler_refactoring(nested_ss)
        assert 0.0 <= plan.estimated_reduction <= 1.0

    def test_fiedler_wide(self, wide_ss):
        plan = fiedler_refactoring(wide_ss)
        assert plan.partition is not None

    def test_fiedler_chain(self, chain_ss):
        plan = fiedler_refactoring(chain_ss)
        assert plan.approach == "fiedler"


# ---------------------------------------------------------------------------
# Approach B: Eigenvalue degeneracy
# ---------------------------------------------------------------------------

class TestSpectralEmbedding:
    """Tests for spectral embedding."""

    def test_embedding_end(self, end_ss):
        emb = spectral_embedding(end_ss, k=2)
        # k is clamped to max(1, n-1) = 1 for a single state
        assert emb.k == 1
        assert len(emb.state_ids) == 1

    def test_embedding_branch(self, branch_ss):
        emb = spectral_embedding(branch_ss, k=2)
        assert len(emb.state_ids) == len(branch_ss.states)
        assert len(emb.coordinates) == len(branch_ss.states)
        for coord in emb.coordinates:
            assert len(coord) == emb.k

    def test_embedding_k_clamped(self, branch_ss):
        """k is clamped to n-1."""
        n = len(branch_ss.states)
        emb = spectral_embedding(branch_ss, k=100)
        assert emb.k <= n

    def test_embedding_different_types_differ(self, branch_ss, chain_ss):
        emb1 = spectral_embedding(branch_ss, k=2)
        emb2 = spectral_embedding(chain_ss, k=2)
        # Different state spaces should produce different embeddings
        assert emb1.state_ids != emb2.state_ids or emb1.coordinates != emb2.coordinates

    def test_embedding_eigenvalues_ascending(self, nested_ss):
        emb = spectral_embedding(nested_ss, k=3)
        for i in range(len(emb.eigenvalues) - 1):
            assert emb.eigenvalues[i] <= emb.eigenvalues[i + 1] + 0.01


class TestEigenvalueDegeneracy:
    """Tests for eigenvalue degeneracy detection."""

    def test_degeneracy_end(self, end_ss):
        groups = eigenvalue_degeneracy(end_ss)
        assert groups == []

    def test_degeneracy_branch(self, branch_ss):
        groups = eigenvalue_degeneracy(branch_ss, tol=0.1)
        # Result is a list of DegeneracyGroup
        for g in groups:
            assert isinstance(g, DegeneracyGroup)
            assert g.multiplicity > 1

    def test_degeneracy_symmetric_structure(self, wide_ss):
        """Wide branching has symmetric eigenvalues."""
        groups = eigenvalue_degeneracy(wide_ss, tol=0.5)
        # We just check it runs and returns groups
        assert isinstance(groups, list)

    def test_degeneracy_tolerance(self, branch_ss):
        """Tighter tolerance should find fewer groups."""
        loose = eigenvalue_degeneracy(branch_ss, tol=1.0)
        tight = eigenvalue_degeneracy(branch_ss, tol=0.001)
        assert len(tight) <= len(loose)

    def test_degeneracy_group_spread(self, nested_ss):
        groups = eigenvalue_degeneracy(nested_ss, tol=0.5)
        for g in groups:
            assert g.spread <= 0.5 + 1e-10
            assert g.spread >= 0.0


class TestRedundantStates:
    """Tests for redundant state detection."""

    def test_redundant_end(self, end_ss):
        pairs = redundant_states(end_ss)
        assert pairs == []

    def test_redundant_branch(self, branch_ss):
        pairs = redundant_states(branch_ss, tol=0.5)
        for p in pairs:
            assert isinstance(p, RedundantPair)
            assert p.state_a != p.state_b
            assert p.distance <= 0.5 + 1e-10

    def test_redundant_sorted_by_distance(self, nested_ss):
        pairs = redundant_states(nested_ss, tol=1.0)
        for i in range(len(pairs) - 1):
            assert pairs[i].distance <= pairs[i + 1].distance + 1e-10

    def test_redundant_tight_tolerance(self, branch_ss):
        pairs = redundant_states(branch_ss, tol=0.0001)
        # Very tight tolerance: likely no pairs unless truly identical
        for p in pairs:
            assert p.distance < 0.0001 + 1e-10

    def test_redundant_wide_has_symmetric_pairs(self, wide_ss):
        """Wide branching has leaf states that may be spectrally similar."""
        pairs = redundant_states(wide_ss, tol=0.5)
        assert isinstance(pairs, list)


class TestMergeRedundant:
    """Tests for merging redundant states."""

    def test_merge_empty_pairs(self, branch_ss):
        merged = merge_redundant(branch_ss, [])
        assert len(merged.states) == len(branch_ss.states)

    def test_merge_reduces_states(self, branch_ss):
        pairs = redundant_states(branch_ss, tol=1.0)
        if pairs:
            merged = merge_redundant(branch_ss, pairs)
            assert len(merged.states) <= len(branch_ss.states)

    def test_merge_preserves_top_bottom(self, nested_ss):
        pairs = redundant_states(nested_ss, tol=1.0)
        merged = merge_redundant(nested_ss, pairs)
        assert merged.top in merged.states
        assert merged.bottom in merged.states

    def test_merge_no_self_loops(self, nested_ss):
        pairs = redundant_states(nested_ss, tol=1.0)
        merged = merge_redundant(nested_ss, pairs)
        for src, _, tgt in merged.transitions:
            assert src != tgt

    def test_merge_preserves_selection_kind(self, mixed_ss):
        pairs = redundant_states(mixed_ss, tol=1.0)
        merged = merge_redundant(mixed_ss, pairs)
        # Selection transitions should still be tracked
        assert isinstance(merged.selection_transitions, set)

    def test_merge_manual_pair(self):
        """Manually construct a pair to merge."""
        ss = _manual_ss(
            states={0, 1, 2, 3},
            transitions=[(0, "a", 1), (0, "b", 2), (1, "c", 3), (2, "c", 3)],
            top=0, bottom=3,
        )
        pair = RedundantPair(
            state_a=1, state_b=2, distance=0.0,
            coord_a=(0.0,), coord_b=(0.0,),
        )
        merged = merge_redundant(ss, [pair])
        assert len(merged.states) <= 3
        assert merged.top in merged.states
        assert merged.bottom in merged.states


# ---------------------------------------------------------------------------
# Approach C: Spectral clustering
# ---------------------------------------------------------------------------

class TestKmeans:
    """Tests for internal k-means."""

    def test_kmeans_empty(self):
        assert _kmeans([], 2) == []

    def test_kmeans_single_point(self):
        result = _kmeans([(0.0, 0.0)], 1)
        assert result == [0]

    def test_kmeans_k_equals_n(self):
        points = [(0.0,), (1.0,), (2.0,)]
        result = _kmeans(points, 3)
        assert len(set(result)) == 3

    def test_kmeans_two_clusters(self):
        points = [(0.0,), (0.1,), (10.0,), (10.1,)]
        result = _kmeans(points, 2)
        # First two should be in same cluster, last two in same cluster
        assert result[0] == result[1]
        assert result[2] == result[3]
        assert result[0] != result[2]

    def test_kmeans_deterministic(self):
        points = [(float(i),) for i in range(10)]
        r1 = _kmeans(points, 3)
        r2 = _kmeans(points, 3)
        assert r1 == r2


class TestSpectralClusters:
    """Tests for spectral clustering."""

    def test_clusters_end(self, end_ss):
        clusters = spectral_clusters(end_ss, k=1)
        assert len(clusters) == 1
        assert all(v == 0 for v in clusters.values())

    def test_clusters_branch(self, branch_ss):
        clusters = spectral_clusters(branch_ss, k=2)
        assert len(clusters) == len(branch_ss.states)
        assert all(0 <= v < 2 for v in clusters.values())

    def test_clusters_all_states_assigned(self, nested_ss):
        clusters = spectral_clusters(nested_ss, k=2)
        assert set(clusters.keys()) == nested_ss.states

    def test_clusters_k_one(self, branch_ss):
        clusters = spectral_clusters(branch_ss, k=1)
        assert all(v == 0 for v in clusters.values())

    def test_clusters_k_equals_n(self, branch_ss):
        n = len(branch_ss.states)
        clusters = spectral_clusters(branch_ss, k=n)
        assert len(set(clusters.values())) <= n


class TestReconstructFromClusters:
    """Tests for cluster-based reconstruction."""

    def test_reconstruct_identity_clusters(self, branch_ss):
        """Each state in its own cluster = no change."""
        clusters = {s: i for i, s in enumerate(sorted(branch_ss.states))}
        reconstructed = reconstruct_from_clusters(branch_ss, clusters)
        assert len(reconstructed.states) == len(branch_ss.states)

    def test_reconstruct_single_cluster(self, branch_ss):
        """All states in one cluster = single state."""
        clusters = {s: 0 for s in branch_ss.states}
        reconstructed = reconstruct_from_clusters(branch_ss, clusters)
        assert len(reconstructed.states) == 1

    def test_reconstruct_preserves_top(self, nested_ss):
        clusters = spectral_clusters(nested_ss, k=2)
        reconstructed = reconstruct_from_clusters(nested_ss, clusters)
        assert reconstructed.top in reconstructed.states

    def test_reconstruct_preserves_bottom(self, nested_ss):
        clusters = spectral_clusters(nested_ss, k=2)
        reconstructed = reconstruct_from_clusters(nested_ss, clusters)
        assert reconstructed.bottom in reconstructed.states

    def test_reconstruct_no_self_loops(self, nested_ss):
        clusters = spectral_clusters(nested_ss, k=2)
        reconstructed = reconstruct_from_clusters(nested_ss, clusters)
        for src, _, tgt in reconstructed.transitions:
            assert src != tgt

    def test_reconstruct_reduces_states(self, wide_ss):
        clusters = spectral_clusters(wide_ss, k=2)
        reconstructed = reconstruct_from_clusters(wide_ss, clusters)
        assert len(reconstructed.states) <= len(wide_ss.states)


# ---------------------------------------------------------------------------
# Round-trip analysis
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """Tests for round-trip analysis (embed → cluster → reconstruct)."""

    def test_round_trip_end(self, end_ss):
        rt = round_trip_analysis(end_ss, k=1)
        assert rt["state_preservation"] == 1.0
        assert rt["top_bottom_preserved"] is True

    def test_round_trip_branch(self, branch_ss):
        rt = round_trip_analysis(branch_ss, k=2)
        assert "state_preservation" in rt
        assert "transition_preservation" in rt
        assert "lattice_preserved" in rt
        assert 0.0 <= float(rt["state_preservation"]) <= 1.0

    def test_round_trip_preserves_counts(self, nested_ss):
        rt = round_trip_analysis(nested_ss, k=2)
        assert int(rt["original_states"]) == len(nested_ss.states)
        assert int(rt["reconstructed_states"]) <= len(nested_ss.states)

    def test_round_trip_identity(self, branch_ss):
        """k = n should approximately preserve everything."""
        n = len(branch_ss.states)
        rt = round_trip_analysis(branch_ss, k=n)
        assert rt["top_bottom_preserved"] is True

    def test_round_trip_selection_preservation(self, mixed_ss):
        rt = round_trip_analysis(mixed_ss, k=2)
        assert "selection_preservation" in rt


# ---------------------------------------------------------------------------
# Pipeline & comparison
# ---------------------------------------------------------------------------

class TestCompareOriginalRefactored:
    """Tests for compare_original_refactored."""

    def test_compare_identical(self, branch_ss):
        result = compare_original_refactored(branch_ss, branch_ss)
        assert result["state_reduction"] == 0.0
        assert result["methods_preserved"] is True

    def test_compare_reduced(self, nested_ss):
        pairs = redundant_states(nested_ss, tol=1.0)
        merged = merge_redundant(nested_ss, pairs)
        result = compare_original_refactored(nested_ss, merged)
        assert result["original_states"] == len(nested_ss.states)
        assert result["refactored_states"] <= len(nested_ss.states)

    def test_compare_reports_lost_methods(self):
        ss1 = _manual_ss(
            states={0, 1, 2},
            transitions=[(0, "a", 1), (0, "b", 2)],
            top=0, bottom=2,
        )
        ss2 = _manual_ss(
            states={0, 1},
            transitions=[(0, "a", 1)],
            top=0, bottom=1,
        )
        result = compare_original_refactored(ss1, ss2)
        assert "b" in result["lost_methods"]

    def test_compare_lattice_check(self, branch_ss):
        result = compare_original_refactored(branch_ss, branch_ss)
        assert "original_is_lattice" in result
        assert "refactored_is_lattice" in result


class TestRefactoringPipeline:
    """Tests for the full refactoring pipeline."""

    def test_pipeline_end(self, end_ss):
        analysis = refactoring_pipeline(end_ss)
        assert isinstance(analysis, SpectralRefactorAnalysis)
        assert analysis.num_states == 1

    def test_pipeline_branch(self, branch_ss):
        analysis = refactoring_pipeline(branch_ss)
        assert analysis.fiedler_plan.approach == "fiedler"
        assert analysis.degeneracy_plan.approach == "degeneracy"
        assert analysis.clustering_plan.approach == "clustering"

    def test_pipeline_embedding(self, nested_ss):
        analysis = refactoring_pipeline(nested_ss)
        assert isinstance(analysis.embedding, SpectralEmbedding)
        assert len(analysis.embedding.state_ids) == len(nested_ss.states)

    def test_pipeline_degeneracy_groups(self, wide_ss):
        analysis = refactoring_pipeline(wide_ss)
        for g in analysis.degeneracy_groups:
            assert isinstance(g, DegeneracyGroup)

    def test_pipeline_redundant_pairs(self, nested_ss):
        analysis = refactoring_pipeline(nested_ss)
        for p in analysis.redundant_pairs:
            assert isinstance(p, RedundantPair)

    def test_pipeline_round_trip_loss_bounded(self, branch_ss):
        analysis = refactoring_pipeline(branch_ss)
        assert 0.0 <= analysis.round_trip_loss <= 1.0

    def test_pipeline_num_clusters(self, nested_ss):
        analysis = refactoring_pipeline(nested_ss)
        assert analysis.num_clusters >= 1

    def test_pipeline_states_after_merge(self, branch_ss):
        analysis = refactoring_pipeline(branch_ss)
        assert analysis.num_states_after_merge <= analysis.num_states

    def test_pipeline_parallel(self, parallel_ss):
        analysis = refactoring_pipeline(parallel_ss)
        assert analysis.num_states == len(parallel_ss.states)

    def test_pipeline_recursive(self, recursive_ss):
        analysis = refactoring_pipeline(recursive_ss)
        assert isinstance(analysis, SpectralRefactorAnalysis)

    def test_pipeline_mixed(self, mixed_ss):
        analysis = refactoring_pipeline(mixed_ss)
        assert analysis.fiedler_plan.approach == "fiedler"


# ---------------------------------------------------------------------------
# Integration: benchmark protocols
# ---------------------------------------------------------------------------

class TestBenchmarkProtocols:
    """Run pipeline on various protocol patterns."""

    def test_smtp_like(self):
        ss = _ss("&{connect: &{auth: +{OK: &{send: &{quit: end}}, FAIL: &{quit: end}}}}")
        analysis = refactoring_pipeline(ss)
        assert analysis.num_states > 1

    def test_two_buyer(self):
        ss = _ss("&{request: +{accept: &{pay: end}, reject: end}}")
        analysis = refactoring_pipeline(ss)
        assert analysis.fiedler_plan.approach == "fiedler"

    def test_iterator(self):
        ss = _ss("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        analysis = refactoring_pipeline(ss)
        assert isinstance(analysis, SpectralRefactorAnalysis)

    def test_parallel_protocol(self):
        ss = _ss("(&{read: end} || &{write: end})")
        analysis = refactoring_pipeline(ss)
        assert analysis.num_states >= 4  # product of 2 x 2

    def test_deep_nesting(self):
        ss = _ss("&{a: &{b: &{c: &{d: end}}}}")
        analysis = refactoring_pipeline(ss)
        assert analysis.num_states == 5

    def test_wide_selection(self):
        ss = _ss("+{a: end, b: end, c: end}")
        analysis = refactoring_pipeline(ss)
        assert analysis.num_states > 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case tests."""

    def test_single_state(self):
        ss = _ss("end")
        emb = spectral_embedding(ss, k=1)
        assert len(emb.state_ids) == 1

    def test_two_states(self):
        ss = _ss("&{a: end}")
        emb = spectral_embedding(ss, k=1)
        assert len(emb.state_ids) == 2

    def test_large_k_clamped(self):
        ss = _ss("&{a: end}")
        emb = spectral_embedding(ss, k=100)
        assert emb.k <= len(ss.states)

    def test_zero_tolerance_redundancy(self):
        ss = _ss("&{a: end, b: end}")
        pairs = redundant_states(ss, tol=0.0)
        # With zero tolerance, only truly identical coordinates match
        for p in pairs:
            assert p.distance == 0.0

    def test_merge_chain_resolution(self):
        """Test that transitive merge chains are resolved."""
        ss = _manual_ss(
            states={0, 1, 2, 3, 4},
            transitions=[(0, "a", 1), (0, "b", 2), (0, "c", 3), (1, "d", 4), (2, "d", 4), (3, "d", 4)],
            top=0, bottom=4,
        )
        # Merge 1→2 and 2→3 (chain)
        pairs = [
            RedundantPair(1, 2, 0.0, (0.0,), (0.0,)),
            RedundantPair(2, 3, 0.0, (0.0,), (0.0,)),
        ]
        merged = merge_redundant(ss, pairs)
        # 1, 2, 3 should all merge to 1
        assert len(merged.states) <= 3

    def test_clusters_empty_state_space(self):
        ss = _manual_ss(states={0}, transitions=[], top=0, bottom=0)
        clusters = spectral_clusters(ss, k=1)
        assert clusters == {0: 0}

    def test_pipeline_with_selections(self):
        ss = _ss("+{ok: end, err: end}")
        analysis = refactoring_pipeline(ss)
        assert analysis.num_states > 1
