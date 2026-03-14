"""Tests for coequalizer analysis in SessLat (Step 166).

Verifies that SessLat does NOT have general coequalizers (negative result).
The quotient R/θ is always a lattice, but may not be realizable as a
session type — paralleling the coproduct failure (Step 164).

Combined with Step 165: SessLat is finitely complete but NOT finitely
cocomplete — a fundamental asymmetry.
"""

from __future__ import annotations

import pytest

from reticulate.parser import parse
from reticulate.statespace import StateSpace, build_statespace
from reticulate.lattice import check_lattice
from reticulate.category import is_lattice_homomorphism
from reticulate.coequalizer import (
    Congruence,
    CoequalizerResult,
    FiniteCocompletenessResult,
    _UnionFind,
    _find_all_homomorphisms,
    coequalizer_seed_pairs,
    check_coequalizer,
    check_coequalizer_universal_property,
    check_finite_cocompleteness,
    congruence_closure,
    is_coequalizer,
    quotient_lattice,
)


# ---------------------------------------------------------------------------
# Helpers: build small test lattices
# ---------------------------------------------------------------------------

def _end() -> StateSpace:
    """1-element lattice (end): top == bottom."""
    return build_statespace(parse("end"))


def _chain2() -> StateSpace:
    """2-element chain: &{a: end}."""
    return build_statespace(parse("&{a: end}"))


def _chain3() -> StateSpace:
    """3-element chain: &{a: &{b: end}}."""
    return build_statespace(parse("&{a: &{b: end}}"))


def _diamond() -> StateSpace:
    """4-element diamond (product lattice): (&{a: end} || &{b: end})."""
    return build_statespace(parse("(&{a: end} || &{b: end})"))


def _branch2() -> StateSpace:
    """3-element: &{a: end, b: end}."""
    return build_statespace(parse("&{a: end, b: end}"))


def _selection2() -> StateSpace:
    """3-element selection: +{a: end, b: end}."""
    return build_statespace(parse("+{a: end, b: end}"))


# ---------------------------------------------------------------------------
# Seed pair tests
# ---------------------------------------------------------------------------

class TestSeedPairs:
    """Tests for coequalizer_seed_pairs."""

    def test_identity_maps_diagonal(self):
        """Identity maps produce diagonal: {(s, s) : s ∈ L}."""
        ss = _chain2()
        f = {s: s for s in ss.states}
        g = {s: s for s in ss.states}
        seeds = coequalizer_seed_pairs(ss, ss, f, g)
        for s in ss.states:
            assert (s, s) in seeds

    def test_constant_maps_single_pair(self):
        """Two constant maps to same target produce single pair."""
        src = _end()
        tgt = _chain2()
        f = {s: tgt.top for s in src.states}
        g = {s: tgt.top for s in src.states}
        seeds = coequalizer_seed_pairs(src, tgt, f, g)
        assert all(a == b for a, b in seeds)

    def test_distinct_maps_nontrivial(self):
        """Different maps produce nontrivial seed pairs."""
        src = _chain2()
        tgt = _diamond()
        # Find two distinct homomorphisms
        homos = _find_all_homomorphisms(src, tgt)
        if len(homos) >= 2:
            f, g = homos[0], homos[1]
            seeds = coequalizer_seed_pairs(src, tgt, f, g)
            assert len(seeds) >= 1

    def test_seed_pairs_from_same_source(self):
        """Seeds always have |source| pairs (with possible duplicates)."""
        ss = _chain3()
        f = {s: s for s in ss.states}
        g = {s: s for s in ss.states}
        seeds = coequalizer_seed_pairs(ss, ss, f, g)
        # Each seed is (f(s), g(s)) for s in source; unique pairs ≤ |source|
        assert len(seeds) <= len(ss.states)

    def test_equal_maps_all_diagonal(self):
        """Equal maps f = g produce only diagonal pairs."""
        ss = _branch2()
        f = {s: s for s in ss.states}
        seeds = coequalizer_seed_pairs(ss, ss, f, f)
        assert all(a == b for a, b in seeds)


# ---------------------------------------------------------------------------
# Congruence closure tests
# ---------------------------------------------------------------------------

class TestCongruenceClosure:
    """Tests for congruence_closure."""

    def test_empty_seeds_discrete(self):
        """No seed pairs → discrete congruence (identity relation)."""
        ss = _chain2()
        # Diagonal pairs only
        seeds = {(s, s) for s in ss.states}
        cong = congruence_closure(ss, seeds)
        assert cong.is_discrete
        assert cong.num_classes == len(ss.states)

    def test_all_identified_trivial(self):
        """Identifying all states → trivial congruence (one class)."""
        ss = _chain2()
        seeds = {(ss.top, ss.bottom)}
        cong = congruence_closure(ss, seeds)
        assert cong.is_trivial
        assert cong.num_classes == 1

    def test_reflexive(self):
        """Congruence is reflexive: (s, s) for all s."""
        ss = _chain3()
        seeds = {(ss.top, ss.top)}
        cong = congruence_closure(ss, seeds)
        for s in ss.states:
            assert (s, s) in cong.pairs

    def test_symmetric(self):
        """Congruence is symmetric: (a, b) implies (b, a)."""
        ss = _chain3()
        seeds = {(ss.top, ss.bottom)}
        cong = congruence_closure(ss, seeds)
        for a, b in cong.pairs:
            assert (b, a) in cong.pairs

    def test_transitive(self):
        """Congruence is transitive."""
        ss = _chain3()
        seeds = {(ss.top, ss.bottom)}
        cong = congruence_closure(ss, seeds)
        for a, b in cong.pairs:
            for c in ss.states:
                if (b, c) in cong.pairs:
                    assert (a, c) in cong.pairs

    def test_classes_partition(self):
        """Congruence classes partition the state set."""
        ss = _diamond()
        seeds = {(ss.top, ss.top)}
        cong = congruence_closure(ss, seeds)
        all_members: set[int] = set()
        for members in cong.classes.values():
            assert not all_members & members  # disjoint
            all_members |= members
        assert all_members == ss.states

    def test_meet_compatible(self):
        """Congruence is compatible with meets."""
        ss = _diamond()
        # Find two elements to merge
        non_extremal = sorted(ss.states - {ss.top, ss.bottom})
        if len(non_extremal) >= 2:
            seeds = {(non_extremal[0], non_extremal[1])}
            cong = congruence_closure(ss, seeds)
            # Verify meet compatibility
            from reticulate.lattice import compute_meet
            for a1, a2 in list(cong.pairs):
                for b1, b2 in list(cong.pairs):
                    m1 = compute_meet(ss, a1, b1)
                    m2 = compute_meet(ss, a2, b2)
                    if m1 is not None and m2 is not None:
                        assert (m1, m2) in cong.pairs

    def test_join_compatible(self):
        """Congruence is compatible with joins."""
        ss = _diamond()
        non_extremal = sorted(ss.states - {ss.top, ss.bottom})
        if len(non_extremal) >= 2:
            seeds = {(non_extremal[0], non_extremal[1])}
            cong = congruence_closure(ss, seeds)
            from reticulate.lattice import compute_join
            for a1, a2 in list(cong.pairs):
                for b1, b2 in list(cong.pairs):
                    j1 = compute_join(ss, a1, b1)
                    j2 = compute_join(ss, a2, b2)
                    if j1 is not None and j2 is not None:
                        assert (j1, j2) in cong.pairs


# ---------------------------------------------------------------------------
# Quotient lattice tests
# ---------------------------------------------------------------------------

class TestQuotientLattice:
    """Tests for quotient_lattice."""

    def test_discrete_congruence_isomorphic(self):
        """Discrete congruence: quotient isomorphic to original."""
        ss = _chain2()
        seeds = {(s, s) for s in ss.states}
        cong = congruence_closure(ss, seeds)
        q = quotient_lattice(ss, cong)
        assert len(q.states) == len(ss.states)

    def test_trivial_congruence_one_state(self):
        """Trivial congruence: quotient is 1-element lattice."""
        ss = _chain2()
        seeds = {(ss.top, ss.bottom)}
        cong = congruence_closure(ss, seeds)
        q = quotient_lattice(ss, cong)
        assert len(q.states) == 1
        assert q.top == q.bottom

    def test_quotient_has_top_bottom(self):
        """Quotient always has top and bottom."""
        ss = _chain3()
        seeds = {(s, s) for s in ss.states}
        cong = congruence_closure(ss, seeds)
        q = quotient_lattice(ss, cong)
        assert q.top in q.states
        assert q.bottom in q.states

    def test_quotient_state_count(self):
        """Quotient has as many states as congruence classes."""
        ss = _diamond()
        non_extremal = sorted(ss.states - {ss.top, ss.bottom})
        if len(non_extremal) >= 2:
            seeds = {(non_extremal[0], non_extremal[1])}
            cong = congruence_closure(ss, seeds)
            q = quotient_lattice(ss, cong)
            assert len(q.states) == cong.num_classes

    def test_quotient_is_valid_statespace(self):
        """Quotient is a valid StateSpace."""
        ss = _chain3()
        seeds = {(ss.top, ss.bottom)}
        cong = congruence_closure(ss, seeds)
        q = quotient_lattice(ss, cong)
        assert isinstance(q, StateSpace)
        assert q.top in q.states
        assert q.bottom in q.states

    def test_quotient_of_diamond_merge(self):
        """Merging two middle elements of diamond collapses everything.

        In M₂ (diamond), identifying the two atoms a,b forces:
        - meet(a,b)=⊥ and meet(a,a)=a must be in same class → ⊥ ≡ a
        - join(a,b)=⊤ and join(a,a)=a must be in same class → ⊤ ≡ a
        So the entire lattice collapses to one element.
        """
        ss = _diamond()
        non_extremal = sorted(ss.states - {ss.top, ss.bottom})
        if len(non_extremal) >= 2:
            seeds = {(non_extremal[0], non_extremal[1])}
            cong = congruence_closure(ss, seeds)
            q = quotient_lattice(ss, cong)
            # Diamond is simple → only trivial congruences
            assert len(q.states) == 1

    def test_quotient_is_lattice(self):
        """Quotient of a lattice by a congruence is always a lattice."""
        ss = _diamond()
        non_extremal = sorted(ss.states - {ss.top, ss.bottom})
        if len(non_extremal) >= 2:
            seeds = {(non_extremal[0], non_extremal[1])}
            cong = congruence_closure(ss, seeds)
            q = quotient_lattice(ss, cong)
            assert check_lattice(q).is_lattice

    def test_quotient_end_lattice(self):
        """Quotient of 1-element lattice is still 1-element."""
        ss = _end()
        seeds = {(s, s) for s in ss.states}
        cong = congruence_closure(ss, seeds)
        q = quotient_lattice(ss, cong)
        assert len(q.states) == 1


# ---------------------------------------------------------------------------
# Check coequalizer tests
# ---------------------------------------------------------------------------

class TestCheckCoequalizer:
    """Tests for check_coequalizer."""

    def test_identity_pair_trivial(self):
        """f = g = id: coequalizer is R itself (discrete congruence)."""
        ss = _chain2()
        f = {s: s for s in ss.states}
        result = check_coequalizer(ss, ss, f, f)
        assert isinstance(result, CoequalizerResult)
        assert result.is_lattice
        assert result.congruence is not None
        assert result.congruence.is_discrete

    def test_coequalizer_result_fields(self):
        """Result has all required fields."""
        ss = _chain2()
        f = {s: s for s in ss.states}
        result = check_coequalizer(ss, ss, f, f)
        assert result.quotient_space is not None
        assert result.quotient_map is not None
        assert result.congruence is not None
        assert result.f_mapping == f
        assert result.g_mapping == f

    def test_coequalizer_coequal(self):
        """q ∘ f = q ∘ g for the quotient map."""
        src = _chain2()
        tgt = _chain3()
        homos = _find_all_homomorphisms(src, tgt)
        if len(homos) >= 2:
            f, g = homos[0], homos[1]
            result = check_coequalizer(src, tgt, f, g)
            if result.quotient_map is not None:
                for s in src.states:
                    assert result.quotient_map[f[s]] == result.quotient_map[g[s]]

    def test_source_not_lattice(self):
        """Non-lattice source → immediate failure."""
        # Build a non-lattice manually
        bad = StateSpace(
            states={0, 1, 2, 3},
            transitions=[(0, "a", 1), (0, "b", 2), (1, "c", 3), (2, "d", 3)],
            top=0, bottom=3, labels={},
            selection_transitions=set(),
        )
        # Even with diamond target, source check should fail
        # But first check if bad is actually not a lattice
        if not check_lattice(bad).is_lattice:
            tgt = _chain2()
            f = {s: tgt.top for s in bad.states}
            g = {s: tgt.top for s in bad.states}
            result = check_coequalizer(bad, tgt, f, g)
            assert not result.has_coequalizer
            assert "source" in (result.counterexample or "")

    def test_end_to_end_chain(self):
        """End-to-end with chain lattices."""
        src = _end()
        tgt = _chain2()
        f = {s: tgt.top for s in src.states}
        g = {s: tgt.bottom for s in src.states}
        result = check_coequalizer(src, tgt, f, g)
        assert result.is_lattice
        assert result.quotient_space is not None
        # top and bottom merged → 1-element quotient
        assert len(result.quotient_space.states) == 1

    def test_equal_maps_identity_coequalizer(self):
        """f = g: coequalizer quotient ≅ target."""
        ss = _branch2()
        f = {s: s for s in ss.states}
        result = check_coequalizer(ss, ss, f, f)
        assert result.is_lattice
        assert result.congruence is not None
        assert result.congruence.is_discrete
        assert result.quotient_space is not None
        assert len(result.quotient_space.states) == len(ss.states)

    def test_real_homomorphism_pair(self):
        """Coequalizer of real homomorphism pair from benchmarks."""
        src = _chain2()
        tgt = _diamond()
        homos = _find_all_homomorphisms(src, tgt)
        if len(homos) >= 2:
            result = check_coequalizer(src, tgt, homos[0], homos[1])
            assert isinstance(result, CoequalizerResult)
            assert result.quotient_space is not None

    def test_frozen_result(self):
        """CoequalizerResult is frozen (immutable)."""
        ss = _chain2()
        f = {s: s for s in ss.states}
        result = check_coequalizer(ss, ss, f, f)
        with pytest.raises(AttributeError):
            result.has_coequalizer = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Universal property tests
# ---------------------------------------------------------------------------

class TestUniversalProperty:
    """Tests for coequalizer universal property."""

    def test_identity_coequalizer_up(self):
        """f = g = id: universal property holds (quotient = target)."""
        ss = _chain2()
        f = {s: s for s in ss.states}
        result = check_coequalizer(ss, ss, f, f)
        assert result.universal_property_holds

    def test_trivial_quotient_up(self):
        """Trivial quotient (1 element): UP holds trivially."""
        src = _end()
        tgt = _chain2()
        f = {s: tgt.top for s in src.states}
        g = {s: tgt.bottom for s in src.states}
        result = check_coequalizer(src, tgt, f, g)
        # 1-element quotient; UP should hold
        assert result.universal_property_holds

    def test_up_with_end_lattice(self):
        """Coequalizer from/to 1-element lattice."""
        src = _end()
        tgt = _end()
        f = {s: s for s in src.states}
        g = {s: s for s in src.states}
        result = check_coequalizer(src, tgt, f, g)
        assert result.has_coequalizer
        assert result.universal_property_holds

    def test_is_coequalizer_boolean(self):
        """is_coequalizer returns bool matching check_coequalizer."""
        ss = _chain2()
        f = {s: s for s in ss.states}
        result = check_coequalizer(ss, ss, f, f)
        if result.quotient_space is not None and result.quotient_map is not None:
            assert is_coequalizer(
                result.quotient_space, ss, ss, f, f, result.quotient_map,
            ) == result.has_coequalizer

    def test_coequalizer_map_surjective(self):
        """Quotient map q is surjective (every quotient state is hit)."""
        src = _chain2()
        tgt = _diamond()
        homos = _find_all_homomorphisms(src, tgt)
        if len(homos) >= 2:
            result = check_coequalizer(src, tgt, homos[0], homos[1])
            if result.quotient_map is not None and result.quotient_space is not None:
                image = set(result.quotient_map.values())
                assert image == result.quotient_space.states

    def test_up_chain3_homomorphisms(self):
        """UP check with chain3 as source and target."""
        ss = _chain3()
        f = {s: s for s in ss.states}
        result = check_coequalizer(ss, ss, f, f)
        assert result.universal_property_holds

    def test_up_diamond_pair(self):
        """UP check with diamond lattice and actual hom pair."""
        src = _chain2()
        tgt = _diamond()
        homos = _find_all_homomorphisms(src, tgt)
        if len(homos) >= 2:
            result = check_coequalizer(src, tgt, homos[0], homos[1])
            assert isinstance(result.universal_property_holds, bool)

    def test_up_selection_lattice(self):
        """UP check with selection lattice."""
        src = _end()
        tgt = _selection2()
        f = {s: tgt.top for s in src.states}
        g = {s: tgt.bottom for s in src.states}
        result = check_coequalizer(src, tgt, f, g)
        assert result.quotient_space is not None


# ---------------------------------------------------------------------------
# Finite cocompleteness tests
# ---------------------------------------------------------------------------

class TestFiniteCocompleteness:
    """Tests for check_finite_cocompleteness."""

    def test_not_cocomplete(self):
        """SessLat is NOT finitely cocomplete."""
        spaces = [_end(), _chain2(), _branch2()]
        result = check_finite_cocompleteness(spaces)
        assert not result.is_finitely_cocomplete

    def test_result_fields(self):
        """Result has all fields."""
        spaces = [_end(), _chain2()]
        result = check_finite_cocompleteness(spaces)
        assert isinstance(result, FiniteCocompletenessResult)
        assert isinstance(result.has_coproducts, bool)
        assert isinstance(result.has_coequalizers, bool)
        assert isinstance(result.is_finitely_cocomplete, bool)
        assert result.num_spaces_tested >= 1

    def test_coproducts_fail(self):
        """Coproducts already fail (known from Step 164)."""
        spaces = [_chain2(), _branch2(), _diamond()]
        result = check_finite_cocompleteness(spaces)
        # At least one of coproducts or coequalizers must fail
        assert not result.is_finitely_cocomplete

    def test_frozen_result(self):
        """FiniteCocompletenessResult is frozen."""
        spaces = [_end()]
        result = check_finite_cocompleteness(spaces)
        with pytest.raises(AttributeError):
            result.is_finitely_cocomplete = True  # type: ignore[misc]

    def test_counterexample_present_on_failure(self):
        """Counterexample string present when cocomplete fails."""
        spaces = [_chain2(), _branch2()]
        result = check_finite_cocompleteness(spaces)
        if not result.is_finitely_cocomplete:
            assert result.counterexample is not None


# ---------------------------------------------------------------------------
# Benchmark sweep tests
# ---------------------------------------------------------------------------

class TestBenchmarkSweep:
    """Parametrized tests over session type benchmarks."""

    @pytest.mark.parametrize("type_str", [
        "end",
        "&{a: end}",
        "&{a: end, b: end}",
        "+{a: end, b: end}",
        "(&{a: end} || &{b: end})",
    ])
    def test_identity_coequalizer(self, type_str: str):
        """f = g = id always gives valid coequalizer (quotient = original)."""
        ss = build_statespace(parse(type_str))
        f = {s: s for s in ss.states}
        result = check_coequalizer(ss, ss, f, f)
        assert result.is_lattice
        assert result.congruence is not None
        assert result.congruence.is_discrete

    @pytest.mark.parametrize("type_str", [
        "&{a: end}",
        "&{a: end, b: end}",
        "+{a: end, b: end}",
        "(&{a: end} || &{b: end})",
        "&{a: &{b: end}}",
    ])
    def test_quotient_always_lattice(self, type_str: str):
        """Quotient R/θ is always a lattice (standard lattice theory)."""
        ss = build_statespace(parse(type_str))
        src = _end()
        # Map end to top and bottom — forces merge
        f = {s: ss.top for s in src.states}
        g = {s: ss.bottom for s in src.states}
        result = check_coequalizer(src, ss, f, g)
        assert result.is_lattice


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and structural properties."""

    def test_maps_from_one_element(self):
        """Coequalizer with 1-element source."""
        src = _end()
        tgt = _chain2()
        f = {s: tgt.top for s in src.states}
        g = {s: tgt.top for s in src.states}
        result = check_coequalizer(src, tgt, f, g)
        assert result.is_lattice
        assert result.quotient_space is not None
        # Same map → discrete congruence → quotient ≅ target
        assert len(result.quotient_space.states) == len(tgt.states)

    def test_congruence_frozen(self):
        """Congruence dataclass is frozen."""
        ss = _chain2()
        seeds = {(s, s) for s in ss.states}
        cong = congruence_closure(ss, seeds)
        with pytest.raises(AttributeError):
            cong.num_classes = 42  # type: ignore[misc]

    def test_union_find_basic(self):
        """Union-find correctly merges and finds."""
        uf = _UnionFind({0, 1, 2, 3})
        assert uf.find(0) != uf.find(1)
        uf.union(0, 1)
        assert uf.find(0) == uf.find(1)
        uf.union(2, 3)
        assert uf.find(2) == uf.find(3)
        assert uf.find(0) != uf.find(2)
        uf.union(1, 3)
        assert uf.find(0) == uf.find(3)
        cls = uf.classes()
        assert len(cls) == 1
