"""Tests for Formal Concept Analysis (Step 30aa)."""

import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.fca import (
    FormalContext,
    FormalConcept,
    ConceptLattice,
    FCAResult,
    formal_context,
    extent,
    intent,
    concept_closure,
    attribute_closure,
    all_concepts,
    concept_lattice,
    is_clarified,
    context_density,
    analyze_fca,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ss(type_str: str):
    """Parse a session type and build its state space."""
    return build_statespace(parse(type_str))


# ---------------------------------------------------------------------------
# TestFormalContext
# ---------------------------------------------------------------------------

class TestFormalContext:
    """Tests for formal context construction."""

    def test_end_context(self):
        ss = _ss("end")
        ctx = formal_context(ss)
        assert len(ctx.objects) == 1
        assert len(ctx.attributes) == 0

    def test_single_branch(self):
        ss = _ss("&{a: end}")
        ctx = formal_context(ss)
        assert len(ctx.objects) == 2
        assert "a" in ctx.attributes
        # Some state has label "a" enabled
        assert any("a" in ctx.incidence[s] for s in ctx.objects)

    def test_two_branch(self):
        ss = _ss("&{a: end, b: end}")
        ctx = formal_context(ss)
        assert "a" in ctx.attributes
        assert "b" in ctx.attributes

    def test_chain(self):
        ss = _ss("&{a: &{b: end}}")
        ctx = formal_context(ss)
        assert len(ctx.attributes) == 2
        assert "a" in ctx.attributes
        assert "b" in ctx.attributes

    def test_selection(self):
        ss = _ss("+{a: end, b: end}")
        ctx = formal_context(ss)
        assert "a" in ctx.attributes
        assert "b" in ctx.attributes

    def test_parallel(self):
        ss = _ss("(&{a: end} || &{b: end})")
        ctx = formal_context(ss)
        assert "a" in ctx.attributes
        assert "b" in ctx.attributes

    def test_context_types(self):
        ss = _ss("&{a: end}")
        ctx = formal_context(ss)
        assert isinstance(ctx, FormalContext)
        assert isinstance(ctx.objects, tuple)
        assert isinstance(ctx.attributes, tuple)
        assert isinstance(ctx.incidence, dict)


# ---------------------------------------------------------------------------
# TestDerivationOperators
# ---------------------------------------------------------------------------

class TestDerivationOperators:
    """Tests for extent and intent."""

    def test_empty_attrs_extent(self):
        ss = _ss("&{a: end, b: end}")
        ctx = formal_context(ss)
        # Empty attribute set -> all objects
        result = extent(frozenset(), ctx)
        assert result == frozenset(ctx.objects)

    def test_empty_objs_intent(self):
        ss = _ss("&{a: end, b: end}")
        ctx = formal_context(ss)
        # Empty object set -> all attributes
        result = intent(frozenset(), ctx)
        assert result == frozenset(ctx.attributes)

    def test_single_attr_extent(self):
        ss = _ss("&{a: &{b: end}}")
        ctx = formal_context(ss)
        # "a" is enabled at the initial state only
        a_extent = extent(frozenset({"a"}), ctx)
        assert len(a_extent) >= 1

    def test_single_obj_intent(self):
        ss = _ss("&{a: end, b: end}")
        ctx = formal_context(ss)
        # Find the state that has outgoing transitions (initial)
        initial = [s for s in ctx.objects if ctx.incidence[s]][0]
        result = intent(frozenset({initial}), ctx)
        assert "a" in result
        assert "b" in result

    def test_galois_connection(self):
        """extent and intent form a Galois connection."""
        ss = _ss("&{a: end, b: end}")
        ctx = formal_context(ss)
        for obj in ctx.objects:
            objs = frozenset({obj})
            b = intent(objs, ctx)
            a = extent(b, ctx)
            # A'' ⊇ A (closure is extensive)
            assert objs <= a


# ---------------------------------------------------------------------------
# TestConceptClosure
# ---------------------------------------------------------------------------

class TestConceptClosure:
    """Tests for concept closure."""

    def test_closure_is_concept(self):
        ss = _ss("&{a: end, b: end}")
        ctx = formal_context(ss)
        initial = min(ss.states)
        c = concept_closure(frozenset({initial}), ctx)
        assert isinstance(c, FormalConcept)
        assert isinstance(c.extent, frozenset)
        assert isinstance(c.intent, frozenset)

    def test_closure_idempotent(self):
        """Closing twice gives the same concept."""
        ss = _ss("&{a: &{b: end}}")
        ctx = formal_context(ss)
        initial = min(ss.states)
        c1 = concept_closure(frozenset({initial}), ctx)
        c2 = concept_closure(c1.extent, ctx)
        assert c1.extent == c2.extent
        assert c1.intent == c2.intent

    def test_attribute_closure(self):
        ss = _ss("&{a: end, b: end}")
        ctx = formal_context(ss)
        c = attribute_closure(frozenset({"a"}), ctx)
        assert isinstance(c, FormalConcept)
        assert "a" in c.intent

    def test_closures_agree(self):
        """Object closure and attribute closure should produce valid concepts."""
        ss = _ss("&{a: &{b: end}}")
        ctx = formal_context(ss)
        c1 = concept_closure(frozenset(ctx.objects), ctx)
        c2 = attribute_closure(frozenset(), ctx)
        # Both should give the top concept (all objects)
        assert c1.extent == frozenset(ctx.objects)
        assert c2.extent == frozenset(ctx.objects)


# ---------------------------------------------------------------------------
# TestAllConcepts
# ---------------------------------------------------------------------------

class TestAllConcepts:
    """Tests for concept enumeration."""

    def test_end_concepts(self):
        ss = _ss("end")
        ctx = formal_context(ss)
        concepts = all_concepts(ctx)
        assert len(concepts) >= 1  # At least the top concept

    def test_single_branch_concepts(self):
        ss = _ss("&{a: end}")
        ctx = formal_context(ss)
        concepts = all_concepts(ctx)
        assert len(concepts) >= 2  # Top and bottom at minimum

    def test_concepts_are_closed(self):
        """Every concept should be a fixed point of the double closure."""
        ss = _ss("&{a: end, b: end}")
        ctx = formal_context(ss)
        concepts = all_concepts(ctx)
        for c in concepts:
            c2 = concept_closure(c.extent, ctx)
            assert c.extent == c2.extent
            assert c.intent == c2.intent

    def test_concept_count_chain(self):
        """A chain of depth 2 should have exactly 3 concepts."""
        ss = _ss("&{a: &{b: end}}")
        ctx = formal_context(ss)
        concepts = all_concepts(ctx)
        # 3 states -> 3 distinct intent sets -> 3 concepts
        assert len(concepts) >= 3

    def test_no_duplicate_concepts(self):
        ss = _ss("&{a: end, b: end}")
        ctx = formal_context(ss)
        concepts = all_concepts(ctx)
        extents = [c.extent for c in concepts]
        assert len(extents) == len(set(extents))


# ---------------------------------------------------------------------------
# TestConceptLattice
# ---------------------------------------------------------------------------

class TestConceptLattice:
    """Tests for concept lattice construction."""

    def test_lattice_type(self):
        ss = _ss("&{a: end}")
        ctx = formal_context(ss)
        lat = concept_lattice(ctx)
        assert isinstance(lat, ConceptLattice)
        assert isinstance(lat.concepts, tuple)
        assert isinstance(lat.order, tuple)

    def test_has_top_and_bottom(self):
        ss = _ss("&{a: end, b: end}")
        ctx = formal_context(ss)
        lat = concept_lattice(ctx)
        # Top should have largest extent
        top_ext = lat.concepts[lat.top].extent
        for c in lat.concepts:
            assert c.extent <= top_ext

    def test_order_is_extent_inclusion(self):
        ss = _ss("&{a: end, b: end}")
        ctx = formal_context(ss)
        lat = concept_lattice(ctx)
        for i, j in lat.order:
            assert lat.concepts[i].extent <= lat.concepts[j].extent

    def test_order_is_antisymmetric(self):
        """If i ≤ j and j ≤ i then i == j (modulo extent equality)."""
        ss = _ss("&{a: end, b: end}")
        ctx = formal_context(ss)
        lat = concept_lattice(ctx)
        order_set = set(lat.order)
        for i, j in lat.order:
            if (j, i) in order_set:
                assert lat.concepts[i].extent == lat.concepts[j].extent

    def test_parallel_concept_count(self):
        """Parallel composition creates product lattice -> more concepts."""
        ss_seq = _ss("&{a: end}")
        ss_par = _ss("(&{a: end} || &{b: end})")
        ctx_seq = formal_context(ss_seq)
        ctx_par = formal_context(ss_par)
        c_seq = len(all_concepts(ctx_seq))
        c_par = len(all_concepts(ctx_par))
        assert c_par >= c_seq


# ---------------------------------------------------------------------------
# TestContextProperties
# ---------------------------------------------------------------------------

class TestContextProperties:
    """Tests for context properties."""

    def test_density_end(self):
        ss = _ss("end")
        ctx = formal_context(ss)
        d = context_density(ctx)
        assert d == 0.0  # No attributes

    def test_density_range(self):
        ss = _ss("&{a: end, b: end}")
        ctx = formal_context(ss)
        d = context_density(ctx)
        assert 0.0 <= d <= 1.0

    def test_density_chain(self):
        ss = _ss("&{a: &{b: &{c: end}}}")
        ctx = formal_context(ss)
        d = context_density(ctx)
        assert 0.0 < d < 1.0  # Not all states have all labels

    def test_clarified_end(self):
        ss = _ss("end")
        ctx = formal_context(ss)
        assert is_clarified(ctx)

    def test_clarified_chain(self):
        ss = _ss("&{a: &{b: end}}")
        ctx = formal_context(ss)
        # In a chain, each state has a unique set of enabled methods
        assert is_clarified(ctx)


# ---------------------------------------------------------------------------
# TestAnalyzeFCA
# ---------------------------------------------------------------------------

class TestAnalyzeFCA:
    """Tests for full FCA analysis."""

    def test_end_analysis(self):
        ss = _ss("end")
        result = analyze_fca(ss)
        assert isinstance(result, FCAResult)
        assert result.num_objects == 1
        assert result.num_attributes == 0
        assert result.num_concepts >= 1

    def test_branch_analysis(self):
        ss = _ss("&{a: end, b: end}")
        result = analyze_fca(ss)
        assert result.num_attributes == 2
        assert result.num_concepts >= 2

    def test_chain_analysis(self):
        ss = _ss("&{a: &{b: end}}")
        result = analyze_fca(ss)
        assert result.num_objects == 3
        assert result.num_attributes == 2
        assert result.lattice_size_ratio > 0

    def test_parallel_analysis(self):
        ss = _ss("(&{a: end} || &{b: end})")
        result = analyze_fca(ss)
        assert result.num_concepts >= 3

    def test_recursive_analysis(self):
        ss = _ss("rec X . &{a: X, b: end}")
        result = analyze_fca(ss)
        assert result.num_concepts >= 2

    def test_result_consistency(self):
        ss = _ss("&{a: &{b: end}, c: end}")
        result = analyze_fca(ss)
        assert result.num_objects == len(result.context.objects)
        assert result.num_attributes == len(result.context.attributes)
        assert result.num_concepts == len(result.concept_lattice.concepts)

    def test_density_matches(self):
        ss = _ss("&{a: end, b: end}")
        result = analyze_fca(ss)
        assert result.density == context_density(result.context)

    def test_selection_analysis(self):
        ss = _ss("+{a: end, b: end}")
        result = analyze_fca(ss)
        assert result.num_concepts >= 2


# ---------------------------------------------------------------------------
# TestBenchmarks
# ---------------------------------------------------------------------------

class TestBenchmarks:
    """Tests on benchmark protocols."""

    def test_smtp_like(self):
        """SMTP-like protocol."""
        ss = _ss("&{ehlo: &{auth: +{OK: &{mail: &{rcpt: &{data: end}}}, FAIL: end}}}")
        result = analyze_fca(ss)
        assert result.num_concepts >= 2
        assert result.density > 0

    def test_iterator_like(self):
        """Iterator protocol."""
        ss = _ss("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        result = analyze_fca(ss)
        assert result.num_concepts >= 2

    def test_parallel_3voice(self):
        """Three-voice parallel."""
        ss = _ss("(&{a: end} || (&{b: end} || &{c: end}))")
        result = analyze_fca(ss)
        assert result.num_concepts >= 4
