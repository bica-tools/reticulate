"""Tests for Galois connections from polarity (Birkhoff IV.9-10)."""

import pytest

from reticulate.parser import parse
from reticulate.statespace import StateSpace, build_statespace
from reticulate.polarity import (
    PolarityResult,
    all_labels,
    build_concept_lattice,
    build_polarity,
    check_polarity,
    compute_concepts,
    is_galois_pair,
    label_closure,
    state_closure,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _ss(type_string: str) -> StateSpace:
    return build_statespace(parse(type_string))


# ---------------------------------------------------------------------------
# Sprint 1: Core polarity functions
# ---------------------------------------------------------------------------


class TestBuildPolarity:
    """Build binary relation R ⊆ States × Labels."""

    def test_end_type(self):
        ss = _ss("end")
        rel = build_polarity(ss)
        # end has one state, no transitions
        assert len(rel) == 1
        for labels in rel.values():
            assert labels == frozenset()

    def test_single_branch(self):
        ss = _ss("&{a: end}")
        rel = build_polarity(ss)
        # top state has label "a", bottom (end) has none
        assert rel[ss.top] == frozenset({"a"})
        assert rel[ss.bottom] == frozenset()

    def test_two_branches(self):
        ss = _ss("&{a: end, b: end}")
        rel = build_polarity(ss)
        assert rel[ss.top] == frozenset({"a", "b"})

    def test_selection(self):
        ss = _ss("+{a: end, b: end}")
        rel = build_polarity(ss)
        assert rel[ss.top] == frozenset({"a", "b"})

    def test_nested(self):
        ss = _ss("&{a: &{b: end}}")
        rel = build_polarity(ss)
        assert "a" in rel[ss.top]
        assert "b" not in rel[ss.top]

    def test_recursive(self):
        ss = _ss("rec X . &{a: X, b: end}")
        rel = build_polarity(ss)
        assert rel[ss.top] == frozenset({"a", "b"})

    def test_all_states_present(self):
        ss = _ss("&{a: &{b: end}, c: end}")
        rel = build_polarity(ss)
        assert set(rel.keys()) == ss.states


class TestAllLabels:
    """Extract all transition labels."""

    def test_end_no_labels(self):
        ss = _ss("end")
        assert all_labels(ss) == frozenset()

    def test_simple(self):
        ss = _ss("&{a: end, b: end}")
        assert all_labels(ss) == frozenset({"a", "b"})

    def test_nested(self):
        ss = _ss("&{a: &{b: end}}")
        assert all_labels(ss) == frozenset({"a", "b"})


class TestStateClosure:
    """state_closure(B) = states enabling ALL labels in B."""

    def test_empty_labels_returns_all(self):
        ss = _ss("&{a: end, b: end}")
        rel = build_polarity(ss)
        result = state_closure(frozenset(), rel)
        assert result == frozenset(ss.states)

    def test_single_label(self):
        ss = _ss("&{a: &{b: end}}")
        rel = build_polarity(ss)
        # "a" enabled only at top
        result = state_closure(frozenset({"a"}), rel)
        assert ss.top in result

    def test_intersection_semantics(self):
        ss = _ss("&{a: end, b: end}")
        rel = build_polarity(ss)
        # Both "a" and "b" are enabled at top
        result = state_closure(frozenset({"a", "b"}), rel)
        assert ss.top in result
        # Bottom has neither
        assert ss.bottom not in result

    def test_nonexistent_label(self):
        ss = _ss("&{a: end}")
        rel = build_polarity(ss)
        result = state_closure(frozenset({"z"}), rel)
        assert result == frozenset()


class TestLabelClosure:
    """label_closure(A) = labels enabled at ALL states in A."""

    def test_empty_states_returns_all(self):
        ss = _ss("&{a: end, b: end}")
        rel = build_polarity(ss)
        result = label_closure(frozenset(), rel)
        assert result == frozenset({"a", "b"})

    def test_single_state_top(self):
        ss = _ss("&{a: end, b: end}")
        rel = build_polarity(ss)
        result = label_closure(frozenset({ss.top}), rel)
        assert result == frozenset({"a", "b"})

    def test_single_state_bottom(self):
        ss = _ss("&{a: end}")
        rel = build_polarity(ss)
        result = label_closure(frozenset({ss.bottom}), rel)
        assert result == frozenset()

    def test_intersection_semantics(self):
        """Labels common to multiple states."""
        ss = _ss("&{a: &{a: end, b: end}}")
        rel = build_polarity(ss)
        # top has "a", middle state has "a" and "b"
        # intersection: {a}
        all_states = frozenset(ss.states - {ss.bottom})
        result = label_closure(all_states, rel)
        assert "a" in result


class TestClosureProperties:
    """Closure operators are extensive, monotone, idempotent."""

    def test_extensive_state_closure(self):
        """A ⊆ state_closure(label_closure(A))."""
        ss = _ss("&{a: &{b: end}, c: end}")
        rel = build_polarity(ss)
        for s in ss.states:
            a = frozenset({s})
            lc = label_closure(a, rel)
            sc = state_closure(lc, rel)
            assert a <= sc, f"state {s}: {a} not ⊆ {sc}"

    def test_extensive_label_closure(self):
        """B ⊆ label_closure(state_closure(B))."""
        ss = _ss("&{a: &{b: end}, c: end}")
        rel = build_polarity(ss)
        labs = all_labels(ss)
        for m in labs:
            b = frozenset({m})
            sc = state_closure(b, rel)
            lc = label_closure(sc, rel)
            assert b <= lc, f"label {m}: {b} not ⊆ {lc}"

    def test_idempotent_state(self):
        """state_closure(label_closure(state_closure(label_closure(A)))) = state_closure(label_closure(A))."""
        ss = _ss("&{a: &{b: end}, c: end}")
        rel = build_polarity(ss)
        for s in ss.states:
            a = frozenset({s})
            sc1 = state_closure(label_closure(a, rel), rel)
            sc2 = state_closure(label_closure(sc1, rel), rel)
            assert sc1 == sc2

    def test_idempotent_label(self):
        ss = _ss("&{a: &{b: end}, c: end}")
        rel = build_polarity(ss)
        labs = all_labels(ss)
        for m in labs:
            b = frozenset({m})
            lc1 = label_closure(state_closure(b, rel), rel)
            lc2 = label_closure(state_closure(lc1, rel), rel)
            assert lc1 == lc2

    def test_monotone_state(self):
        """B₁ ⊆ B₂ ⟹ state_closure(B₁) ⊇ state_closure(B₂)."""
        ss = _ss("&{a: end, b: end}")
        rel = build_polarity(ss)
        b1 = frozenset({"a"})
        b2 = frozenset({"a", "b"})
        assert b1 <= b2
        sc1 = state_closure(b1, rel)
        sc2 = state_closure(b2, rel)
        # Antitone: more labels → fewer states
        assert sc2 <= sc1


# ---------------------------------------------------------------------------
# Sprint 2: Concept computation
# ---------------------------------------------------------------------------


class TestComputeConcepts:
    """Compute formal concepts (fixed points)."""

    def test_end_type_one_concept(self):
        ss = _ss("end")
        concepts = compute_concepts(ss)
        # One state, no labels → one concept: ({bottom}, {})
        assert len(concepts) >= 1

    def test_single_branch(self):
        ss = _ss("&{a: end}")
        concepts = compute_concepts(ss)
        # Should have at least 2 concepts
        assert len(concepts) >= 2
        # Check top concept (all states, common labels)
        extents = [c[0] for c in concepts]
        assert any(ss.top in ext for ext in extents)

    def test_concepts_are_fixed_points(self):
        """Each concept (A, B) satisfies A = state_closure(B) and B = label_closure(A)."""
        ss = _ss("&{a: &{b: end}, c: end}")
        rel = build_polarity(ss)
        concepts = compute_concepts(ss)
        for extent, intent in concepts:
            assert state_closure(intent, rel) == extent, \
                f"Extent {extent} != state_closure({intent})"
            assert label_closure(extent, rel) == intent, \
                f"Intent {intent} != label_closure({extent})"

    def test_two_branch_concepts(self):
        ss = _ss("&{a: end, b: end}")
        concepts = compute_concepts(ss)
        # At minimum: ({top, bottom}, {}), ({top}, {a, b}), ({bottom}, {})
        # Actually: top has {a,b}, bottom has {}
        # Concepts: ({top, bottom}, ∅), ({top}, {a,b})
        assert len(concepts) >= 2

    def test_recursive_concepts(self):
        ss = _ss("rec X . &{a: X, b: end}")
        concepts = compute_concepts(ss)
        assert len(concepts) >= 1
        # Verify all are fixed points
        rel = build_polarity(ss)
        for extent, intent in concepts:
            assert state_closure(intent, rel) == extent
            assert label_closure(extent, rel) == intent

    def test_parallel_concepts(self):
        ss = _ss("(&{a: end} || &{b: end})")
        concepts = compute_concepts(ss)
        assert len(concepts) >= 1

    def test_selection_concepts(self):
        ss = _ss("+{ok: end, err: end}")
        concepts = compute_concepts(ss)
        rel = build_polarity(ss)
        for extent, intent in concepts:
            assert state_closure(intent, rel) == extent


class TestBuildConceptLattice:
    """Build covering relation on concepts."""

    def test_single_concept_no_edges(self):
        concepts = [(frozenset({0}), frozenset())]
        edges = build_concept_lattice(concepts)
        assert edges == []

    def test_two_concepts_one_edge(self):
        concepts = [
            (frozenset({0, 1}), frozenset()),
            (frozenset({0}), frozenset({"a"})),
        ]
        edges = build_concept_lattice(concepts)
        assert len(edges) == 1
        # Concept 0 (larger extent) covers concept 1
        assert (0, 1) in edges

    def test_chain(self):
        concepts = [
            (frozenset({0, 1, 2}), frozenset()),
            (frozenset({0, 1}), frozenset({"a"})),
            (frozenset({0}), frozenset({"a", "b"})),
        ]
        edges = build_concept_lattice(concepts)
        # Should be a chain: 0 → 1 → 2
        assert (0, 1) in edges
        assert (1, 2) in edges
        # No skip: 0 should NOT directly cover 2
        assert (0, 2) not in edges

    def test_diamond(self):
        """Two incomparable concepts between top and bottom."""
        concepts = [
            (frozenset({0, 1, 2}), frozenset()),        # top
            (frozenset({0, 1}), frozenset({"a"})),       # left
            (frozenset({0, 2}), frozenset({"b"})),       # right
            (frozenset({0}), frozenset({"a", "b"})),     # bottom
        ]
        edges = build_concept_lattice(concepts)
        assert (0, 1) in edges
        assert (0, 2) in edges
        assert (1, 3) in edges
        assert (2, 3) in edges
        assert len(edges) == 4


# ---------------------------------------------------------------------------
# Sprint 3: Galois verification + full analysis
# ---------------------------------------------------------------------------


class TestIsGaloisPair:
    """Verify adjunction property."""

    def test_simple_branch(self):
        ss = _ss("&{a: end}")
        rel = build_polarity(ss)
        assert is_galois_pair(rel) is True

    def test_two_branches(self):
        ss = _ss("&{a: end, b: end}")
        rel = build_polarity(ss)
        assert is_galois_pair(rel) is True

    def test_nested(self):
        ss = _ss("&{a: &{b: end}, c: end}")
        rel = build_polarity(ss)
        assert is_galois_pair(rel) is True

    def test_recursive(self):
        ss = _ss("rec X . &{a: X, b: end}")
        rel = build_polarity(ss)
        assert is_galois_pair(rel) is True

    def test_parallel(self):
        ss = _ss("(&{a: end} || &{b: end})")
        rel = build_polarity(ss)
        assert is_galois_pair(rel) is True


class TestCheckPolarity:
    """Full polarity analysis."""

    def test_end_type(self):
        ss = _ss("end")
        result = check_polarity(ss)
        assert isinstance(result, PolarityResult)
        assert result.is_galois is True
        assert result.num_concepts >= 1

    def test_single_branch(self):
        ss = _ss("&{a: end}")
        result = check_polarity(ss)
        assert result.is_galois is True
        assert result.num_concepts >= 2

    def test_result_fields(self):
        ss = _ss("&{a: end, b: end}")
        result = check_polarity(ss)
        assert isinstance(result.relation, dict)
        assert isinstance(result.concepts, list)
        assert isinstance(result.concept_lattice_edges, list)
        assert isinstance(result.num_concepts, int)

    def test_concepts_are_fixed_points(self):
        ss = _ss("rec X . &{a: X, b: end}")
        result = check_polarity(ss)
        rel = result.relation
        for extent, intent in result.concepts:
            assert state_closure(intent, rel) == extent
            assert label_closure(extent, rel) == intent


# ---------------------------------------------------------------------------
# Benchmark sweep
# ---------------------------------------------------------------------------


class TestBenchmarkSweep:
    """Run polarity analysis on all 79 benchmarks."""

    @pytest.fixture(scope="class")
    def benchmark_protocols(self):
        from tests.benchmarks.protocols import BENCHMARKS
        return BENCHMARKS

    def test_all_benchmarks_galois(self, benchmark_protocols):
        """Every benchmark's polarity forms a Galois connection."""
        for bp in benchmark_protocols:
            ss = build_statespace(parse(bp.type_string))
            result = check_polarity(ss)
            assert result.is_galois, f"{bp.name}: not Galois"

    def test_all_benchmarks_concepts_are_fixed_points(self, benchmark_protocols):
        """Every concept in every benchmark is a genuine fixed point."""
        for bp in benchmark_protocols:
            ss = build_statespace(parse(bp.type_string))
            rel = build_polarity(ss)
            concepts = compute_concepts(ss)
            for extent, intent in concepts:
                assert state_closure(intent, rel) == extent, \
                    f"{bp.name}: extent not fixed"
                assert label_closure(extent, rel) == intent, \
                    f"{bp.name}: intent not fixed"

    def test_all_benchmarks_closure_extensive(self, benchmark_protocols):
        """Closure operators are extensive on all benchmarks."""
        for bp in benchmark_protocols:
            ss = build_statespace(parse(bp.type_string))
            rel = build_polarity(ss)
            for s in ss.states:
                a = frozenset({s})
                sc = state_closure(label_closure(a, rel), rel)
                assert a <= sc, f"{bp.name}: not extensive at state {s}"

    def test_all_benchmarks_closure_idempotent(self, benchmark_protocols):
        """Closure operators are idempotent on all benchmarks."""
        for bp in benchmark_protocols:
            ss = build_statespace(parse(bp.type_string))
            rel = build_polarity(ss)
            for s in ss.states:
                a = frozenset({s})
                sc1 = state_closure(label_closure(a, rel), rel)
                sc2 = state_closure(label_closure(sc1, rel), rel)
                assert sc1 == sc2, f"{bp.name}: not idempotent at state {s}"

    def test_all_benchmarks_have_concepts(self, benchmark_protocols):
        """Every benchmark produces at least one concept."""
        for bp in benchmark_protocols:
            ss = build_statespace(parse(bp.type_string))
            concepts = compute_concepts(ss)
            assert len(concepts) >= 1, f"{bp.name}: no concepts"

    def test_concept_count_summary(self, benchmark_protocols):
        """Print concept count summary (informational)."""
        results = []
        for bp in benchmark_protocols:
            ss = build_statespace(parse(bp.type_string))
            result = check_polarity(ss)
            results.append((bp.name, result.num_concepts, len(ss.states)))
        # Just verify we got results for all
        assert len(results) == len(benchmark_protocols)
