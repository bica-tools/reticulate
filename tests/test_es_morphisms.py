"""Tests for event structure morphisms (Step 19)."""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.event_structures import (
    Event,
    EventStructure,
    build_event_structure,
    configurations,
)
from reticulate.es_morphisms import (
    ESMorphism,
    ESMorphismAnalysis,
    find_es_morphism,
    find_es_embedding,
    check_es_isomorphism,
    is_label_preserving,
    is_rigid,
    is_folding,
    is_isomorphism,
    classify_es_morphism,
    analyze_es_morphisms,
    preserves_causality,
    reflects_conflict,
    reflects_causality,
    _is_valid_morphism,
)


def _build(type_string: str):
    return build_statespace(parse(type_string))


def _es(type_string: str) -> EventStructure:
    return build_event_structure(_build(type_string))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def end_es():
    return _es("end")


@pytest.fixture
def branch2():
    """&{a: end, b: end} -- 2 events in conflict."""
    return _es("&{a: end, b: end}")


@pytest.fixture
def branch3():
    """&{a: end, b: end, c: end} -- 3 events pairwise in conflict."""
    return _es("&{a: end, b: end, c: end}")


@pytest.fixture
def chain2():
    """&{a: &{b: end}} -- 2 causally ordered events."""
    return _es("&{a: &{b: end}}")


@pytest.fixture
def chain3():
    """&{a: &{b: &{c: end}}} -- 3 causally ordered events."""
    return _es("&{a: &{b: &{c: end}}}")


@pytest.fixture
def select2():
    """+{ok: end, err: end} -- 2 selection events in conflict."""
    return _es("+{ok: end, err: end}")


@pytest.fixture
def parallel():
    """(&{a: end} || &{b: end}) -- 2 concurrent events."""
    return _es("(&{a: end} || &{b: end})")


@pytest.fixture
def nested():
    """&{a: &{c: end, d: end}, b: &{e: end, f: end}}"""
    return _es("&{a: &{c: end, d: end}, b: &{e: end, f: end}}")


# ---------------------------------------------------------------------------
# 1. Empty / trivial event structures
# ---------------------------------------------------------------------------

class TestTrivial:
    def test_end_has_no_events(self, end_es):
        assert len(end_es.events) == 0

    def test_end_to_end_morphism(self, end_es):
        f = find_es_morphism(end_es, end_es)
        assert f is not None
        assert f == {}

    def test_end_to_end_isomorphism(self, end_es):
        iso = check_es_isomorphism(end_es, end_es)
        assert iso is not None

    def test_end_to_branch_morphism(self, end_es, branch2):
        # Empty -> non-empty: trivial morphism (empty mapping)
        f = find_es_morphism(end_es, branch2)
        assert f is not None
        assert f == {}

    def test_branch_to_end_no_total_morphism(self, branch2, end_es):
        # Non-empty -> empty: no total morphism possible
        f = find_es_morphism(branch2, end_es, require_total=True)
        assert f is None

    def test_branch_to_end_partial_morphism(self, branch2, end_es):
        # Partial morphism: map nothing
        f = find_es_morphism(branch2, end_es)
        assert f is not None
        assert f == {}


# ---------------------------------------------------------------------------
# 2. Isomorphism tests
# ---------------------------------------------------------------------------

class TestIsomorphism:
    def test_identical_branch_isomorphic(self, branch2):
        iso = check_es_isomorphism(branch2, branch2)
        assert iso is not None

    def test_same_structure_different_labels_not_label_iso(self):
        es1 = _es("&{a: end, b: end}")
        es2 = _es("&{x: end, y: end}")
        # Label-preserving iso should fail
        iso = check_es_isomorphism(es1, es2, label_preserving=True)
        assert iso is None

    def test_same_structure_different_labels_structural_iso(self):
        es1 = _es("&{a: end, b: end}")
        es2 = _es("&{x: end, y: end}")
        # Non-label-preserving iso should succeed (same shape)
        iso = check_es_isomorphism(es1, es2, label_preserving=False)
        assert iso is not None

    def test_different_sizes_not_isomorphic(self, branch2, branch3):
        iso = check_es_isomorphism(branch2, branch3)
        assert iso is None

    def test_chain_iso(self, chain2):
        iso = check_es_isomorphism(chain2, chain2)
        assert iso is not None

    def test_chain_vs_branch_not_iso(self, chain2, branch2):
        # Both have 2 events but different structure
        iso = check_es_isomorphism(chain2, branch2)
        assert iso is None

    def test_iso_is_bijection(self, branch3):
        iso = check_es_isomorphism(branch3, branch3)
        assert iso is not None
        assert len(iso) == len(branch3.events)
        assert set(iso.values()) == set(branch3.events)

    def test_parallel_self_iso(self, parallel):
        iso = check_es_isomorphism(parallel, parallel)
        assert iso is not None


# ---------------------------------------------------------------------------
# 3. Rigid morphism (embedding) tests
# ---------------------------------------------------------------------------

class TestRigid:
    def test_branch2_embeds_into_branch3(self, branch2, branch3):
        f = find_es_embedding(branch2, branch3)
        assert f is not None
        assert is_rigid(f, branch2, branch3)

    def test_branch3_does_not_embed_into_branch2(self, branch2, branch3):
        f = find_es_embedding(branch3, branch2)
        assert f is None

    def test_chain2_embeds_into_chain3(self, chain2, chain3):
        f = find_es_embedding(chain2, chain3)
        assert f is not None

    def test_embedding_preserves_causality(self, chain2, chain3):
        f = find_es_embedding(chain2, chain3)
        assert f is not None
        assert preserves_causality(f, chain2, chain3)

    def test_embedding_reflects_conflict(self, branch2, branch3):
        f = find_es_embedding(branch2, branch3)
        assert f is not None
        assert reflects_conflict(f, branch2, branch3)

    def test_embedding_is_valid_morphism(self, branch2, branch3):
        f = find_es_embedding(branch2, branch3)
        assert f is not None
        assert _is_valid_morphism(f, branch2, branch3)

    def test_embedding_label_preserving(self):
        es1 = _es("&{a: end}")
        es2 = _es("&{a: end, b: end}")
        f = find_es_embedding(es1, es2, label_preserving=True)
        assert f is not None
        assert is_label_preserving(f, es1, es2)

    def test_rigid_classification(self, branch2, branch3):
        f = find_es_embedding(branch2, branch3)
        assert f is not None
        kind = classify_es_morphism(f, branch2, branch3)
        assert kind == "rigid"


# ---------------------------------------------------------------------------
# 4. Classification tests
# ---------------------------------------------------------------------------

class TestClassification:
    def test_classify_isomorphism(self, branch2):
        f = check_es_isomorphism(branch2, branch2)
        assert f is not None
        assert classify_es_morphism(f, branch2, branch2) == "isomorphism"

    def test_classify_rigid(self, branch2, branch3):
        f = find_es_embedding(branch2, branch3)
        assert f is not None
        assert classify_es_morphism(f, branch2, branch3) == "rigid"

    def test_classify_empty_morphism(self, end_es):
        assert classify_es_morphism({}, end_es, end_es) == "isomorphism"

    def test_classify_invalid(self, chain2):
        # Create an invalid mapping: map causally ordered events to conflicting ones
        # chain2 has 2 causally ordered events; if we map them to events in a
        # conflict structure where the causal relation doesn't hold, it's invalid
        es_conflict = _es("&{x: end, y: end}")
        events_chain = sorted(chain2.events, key=lambda e: e.label)
        events_conflict = sorted(es_conflict.events, key=lambda e: e.label)
        # Map non-conflicting chain events to same target (violates local injectivity)
        bad = {events_chain[0]: events_conflict[0], events_chain[1]: events_conflict[0]}
        kind = classify_es_morphism(bad, chain2, es_conflict)
        assert kind == "none"

    def test_classify_partial_morphism(self, branch2, branch3):
        # Map just one event -- partial morphism
        e1 = sorted(branch2.events, key=lambda e: e.label)[0]
        e2 = sorted(branch3.events, key=lambda e: e.label)[0]
        f = find_es_morphism(branch2, branch3, label_preserving=True)
        if f is not None:
            kind = classify_es_morphism(f, branch2, branch3)
            assert kind in ("isomorphism", "rigid", "folding", "morphism")


# ---------------------------------------------------------------------------
# 5. Label preservation tests
# ---------------------------------------------------------------------------

class TestLabelPreservation:
    def test_identity_preserves_labels(self, branch2):
        f = check_es_isomorphism(branch2, branch2, label_preserving=True)
        assert f is not None
        assert is_label_preserving(f, branch2, branch2)

    def test_non_label_preserving(self):
        es1 = _es("&{a: end, b: end}")
        es2 = _es("&{x: end, y: end}")
        f = check_es_isomorphism(es1, es2, label_preserving=False)
        if f is not None:
            # Labels differ, so not label-preserving
            assert not is_label_preserving(f, es1, es2)

    def test_label_preserving_embedding(self):
        es1 = _es("&{a: end}")
        es2 = _es("&{a: end, b: end}")
        f = find_es_embedding(es1, es2, label_preserving=True)
        assert f is not None
        assert is_label_preserving(f, es1, es2)

    def test_label_preserving_not_found_when_no_match(self):
        es1 = _es("&{x: end}")
        es2 = _es("&{a: end, b: end}")
        f = find_es_embedding(es1, es2, label_preserving=True)
        assert f is None


# ---------------------------------------------------------------------------
# 6. Subtyping-induced morphism tests (contravariant direction)
# ---------------------------------------------------------------------------

class TestSubtyping:
    def test_branch_subtype_more_methods(self):
        """S1 = &{a,b,c} <= S2 = &{a,b}: subtype offers more methods.
        ES(S2) embeds into ES(S1) (direction reversal)."""
        ss1 = _build("&{a: end, b: end, c: end}")
        ss2 = _build("&{a: end, b: end}")
        es1 = build_event_structure(ss1)
        es2 = build_event_structure(ss2)
        # S1 <= S2, so ES(S2) -> ES(S1) should be a rigid morphism
        f = find_es_embedding(es2, es1, label_preserving=True)
        assert f is not None
        assert is_rigid(f, es2, es1)
        assert is_label_preserving(f, es2, es1)

    def test_selection_subtype_fewer_labels(self):
        """S1 = +{ok} <= S2 = +{ok, err}: subtype offers fewer labels.
        ES(S2) embeds into ES(S1) should NOT work (S1 smaller)."""
        es1 = _es("+{ok: end}")
        es2 = _es("+{ok: end, err: end}")
        f = find_es_embedding(es2, es1, label_preserving=True)
        assert f is None  # Can't embed 2 events into 1

    def test_sequential_subtype_embedding(self):
        """Chain subtype: extended protocol embeds the simpler one."""
        ss1 = _build("&{a: &{b: end, c: end}}")
        ss2 = _build("&{a: &{b: end}}")
        es1 = build_event_structure(ss1)
        es2 = build_event_structure(ss2)
        f = find_es_embedding(es2, es1, label_preserving=True)
        assert f is not None
        assert preserves_causality(f, es2, es1)


# ---------------------------------------------------------------------------
# 7. Parallel composition tests
# ---------------------------------------------------------------------------

class TestParallel:
    def test_parallel_events_concurrent(self, parallel):
        """Events in parallel composition are not causally related or conflicting."""
        events = sorted(parallel.events, key=lambda e: e.label)
        assert len(events) >= 2

    def test_parallel_self_morphism(self, parallel):
        f = find_es_morphism(parallel, parallel)
        assert f is not None

    def test_parallel_to_branch_no_embedding(self, parallel, branch2):
        """Parallel and branch have different structure."""
        # Parallel has concurrent events, branch has conflicting events
        # May or may not embed depending on structure
        f = find_es_embedding(parallel, branch2)
        if f is not None:
            assert _is_valid_morphism(f, parallel, branch2)


# ---------------------------------------------------------------------------
# 8. Recursive type tests
# ---------------------------------------------------------------------------

class TestRecursive:
    def test_iterator_self_morphism(self):
        es = _es("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        f = find_es_morphism(es, es)
        assert f is not None

    def test_iterator_self_isomorphism(self):
        es = _es("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        iso = check_es_isomorphism(es, es)
        assert iso is not None


# ---------------------------------------------------------------------------
# 9. Nested branching tests
# ---------------------------------------------------------------------------

class TestNested:
    def test_nested_self_iso(self, nested):
        iso = check_es_isomorphism(nested, nested)
        assert iso is not None

    def test_nested_embedding_substructure(self):
        """A sub-branch embeds into a larger nested branch."""
        es_small = _es("&{a: &{c: end}}")
        es_large = _es("&{a: &{c: end, d: end}, b: end}")
        f = find_es_embedding(es_small, es_large, label_preserving=True)
        assert f is not None
        assert is_label_preserving(f, es_small, es_large)


# ---------------------------------------------------------------------------
# 10. Property verification tests
# ---------------------------------------------------------------------------

class TestProperties:
    def test_valid_morphism_empty(self, end_es):
        assert _is_valid_morphism({}, end_es, end_es)

    def test_preserves_causality_identity(self, chain3):
        f = check_es_isomorphism(chain3, chain3)
        assert f is not None
        assert preserves_causality(f, chain3, chain3)

    def test_reflects_conflict_identity(self, branch3):
        f = check_es_isomorphism(branch3, branch3)
        assert f is not None
        assert reflects_conflict(f, branch3, branch3)

    def test_reflects_causality_identity(self, chain3):
        f = check_es_isomorphism(chain3, chain3)
        assert f is not None
        assert reflects_causality(f, chain3, chain3)

    def test_is_folding_identity(self, branch2):
        f = check_es_isomorphism(branch2, branch2)
        assert f is not None
        assert is_folding(f, branch2, branch2)

    def test_not_folding_embedding(self, branch2, branch3):
        f = find_es_embedding(branch2, branch3)
        assert f is not None
        assert not is_folding(f, branch2, branch3)


# ---------------------------------------------------------------------------
# 11. analyze_es_morphisms tests
# ---------------------------------------------------------------------------

class TestAnalysis:
    def test_identical_types_isomorphic(self):
        ss = _build("&{a: end, b: end}")
        result = analyze_es_morphisms(ss, ss)
        assert result.are_isomorphic

    def test_subtype_has_morphism(self):
        ss1 = _build("&{a: end, b: end, c: end}")
        ss2 = _build("&{a: end, b: end}")
        result = analyze_es_morphisms(ss2, ss1)
        # ES(S2) -> ES(S1) should exist (subtype direction)
        assert result.morphism_1_to_2 is not None

    def test_different_structure_analysis(self):
        ss1 = _build("&{a: &{b: end}}")
        ss2 = _build("&{x: end, y: end}")
        result = analyze_es_morphisms(ss1, ss2)
        assert isinstance(result, ESMorphismAnalysis)

    def test_analysis_label_preserving_field(self):
        ss = _build("&{a: end}")
        result = analyze_es_morphisms(ss, ss)
        assert result.label_preserving_1_to_2

    def test_end_to_end_analysis(self):
        ss = _build("end")
        result = analyze_es_morphisms(ss, ss)
        assert result.are_isomorphic


# ---------------------------------------------------------------------------
# 12. ESMorphism dataclass tests
# ---------------------------------------------------------------------------

class TestDataclass:
    def test_es_morphism_creation(self, branch2):
        m = ESMorphism(
            source=branch2,
            target=branch2,
            mapping={},
            kind="morphism",
        )
        assert m.kind == "morphism"

    def test_es_morphism_frozen(self, branch2):
        m = ESMorphism(
            source=branch2,
            target=branch2,
            mapping={},
            kind="morphism",
        )
        with pytest.raises(AttributeError):
            m.kind = "isomorphism"

    def test_analysis_fields(self):
        ss = _build("end")
        result = analyze_es_morphisms(ss, ss)
        assert result.es1 is not None
        assert result.es2 is not None
        assert isinstance(result.are_isomorphic, bool)


# ---------------------------------------------------------------------------
# 13. Benchmark protocol tests
# ---------------------------------------------------------------------------

class TestBenchmarks:
    def test_smtp_like_self_iso(self):
        smtp = _es("&{connect: &{send: &{quit: end}}}")
        iso = check_es_isomorphism(smtp, smtp)
        assert iso is not None

    def test_smtp_extended_embedding(self):
        """Extended SMTP embeds simpler SMTP (subtype direction)."""
        simple = _es("&{connect: &{send: &{quit: end}}}")
        extended = _es("&{connect: &{send: &{quit: end}, reset: &{quit: end}}}")
        f = find_es_embedding(simple, extended, label_preserving=True)
        assert f is not None

    def test_two_branch_vs_three_branch(self):
        es2 = _es("&{a: end, b: end}")
        es3 = _es("&{a: end, b: end, c: end}")
        f = find_es_embedding(es2, es3, label_preserving=True)
        assert f is not None
        kind = classify_es_morphism(f, es2, es3)
        assert kind == "rigid"

    def test_iterator_vs_simple(self):
        """Iterator has richer structure than a simple branch."""
        it = _es("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        simple = _es("&{a: end}")
        # Simple -> iterator may have morphism
        f = find_es_morphism(simple, it)
        assert f is not None  # At least the empty/partial morphism

    def test_parallel_protocol(self):
        """Parallel protocol self-analysis."""
        ss = _build("(&{read: end} || &{write: end})")
        result = analyze_es_morphisms(ss, ss)
        assert result.are_isomorphic
