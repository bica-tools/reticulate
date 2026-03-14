"""Tests for category.py — SessLat category, products, projections (Step 163)."""

from __future__ import annotations

import pytest

from reticulate.parser import parse
from reticulate.statespace import StateSpace, build_statespace
from reticulate.lattice import check_lattice
from reticulate.category import (
    CategoryResult,
    DecompositionResult,
    LatticeHomomorphism,
    ProductResult,
    check_product_universal_property,
    check_sesslat_category,
    compose,
    find_product_decomposition,
    find_projections,
    identity_morphism,
    is_lattice_homomorphism,
    is_product,
    is_subdirectly_irreducible,
    make_homomorphism,
    universal_morphism,
    universal_morphism_nary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def end_ss() -> StateSpace:
    return build_statespace(parse("end"))


@pytest.fixture
def chain2() -> StateSpace:
    """&{a: end} — 2-state chain."""
    return build_statespace(parse("&{a: end}"))


@pytest.fixture
def chain3() -> StateSpace:
    """&{a: &{b: end}} — 3-state chain."""
    return build_statespace(parse("&{a: &{b: end}}"))


@pytest.fixture
def diamond() -> StateSpace:
    """&{a: &{c: end}, b: end} — 4-state lattice with incomparable middle states."""
    return build_statespace(parse("&{a: &{c: end}, b: end}"))


@pytest.fixture
def par2() -> StateSpace:
    """(&{a: end} || &{b: end}) — 2×2 product."""
    return build_statespace(parse("(&{a: end} || &{b: end})"))


@pytest.fixture
def par3() -> StateSpace:
    """(&{a: end} || &{b: end} || &{c: end}) — 2×2×2 product."""
    return build_statespace(parse("(&{a: end} || &{b: end} || &{c: end})"))


@pytest.fixture
def par2_deep() -> StateSpace:
    """(&{a: &{c: end}} || &{b: end}) — 3×2 product."""
    return build_statespace(parse("(&{a: &{c: end}} || &{b: end})"))


# ---------------------------------------------------------------------------
# TestLatticeHomomorphism
# ---------------------------------------------------------------------------

class TestLatticeHomomorphism:

    def test_identity_is_homomorphism(self, chain2: StateSpace) -> None:
        m = {s: s for s in chain2.states}
        assert is_lattice_homomorphism(chain2, chain2, m)

    def test_identity_on_diamond(self, diamond: StateSpace) -> None:
        m = {s: s for s in diamond.states}
        assert is_lattice_homomorphism(diamond, diamond, m)

    def test_projection_is_homomorphism(self, par2: StateSpace) -> None:
        projs = find_projections(par2)
        for pi in projs:
            assert is_lattice_homomorphism(pi.source, pi.target, pi.mapping)

    def test_constant_top_is_homomorphism(self, diamond: StateSpace) -> None:
        """Constant map to top is a valid lattice homomorphism (1-element sublattice)."""
        m = {s: diamond.top for s in diamond.states}
        assert is_lattice_homomorphism(diamond, diamond, m)

    def test_constant_bottom_is_homomorphism(self, diamond: StateSpace) -> None:
        """Constant map to bottom is a valid lattice homomorphism."""
        m = {s: diamond.bottom for s in diamond.states}
        assert is_lattice_homomorphism(diamond, diamond, m)

    def test_swapped_non_trivial_not_homomorphism(self, diamond: StateSpace) -> None:
        """Swapping top and bottom should NOT be a homomorphism (reverses order)."""
        m = {s: s for s in diamond.states}
        m[diamond.top] = diamond.bottom
        m[diamond.bottom] = diamond.top
        assert not is_lattice_homomorphism(diamond, diamond, m)

    def test_missing_state_rejected(self, chain2: StateSpace) -> None:
        assert not is_lattice_homomorphism(chain2, chain2, {})

    def test_end_identity(self, end_ss: StateSpace) -> None:
        m = {s: s for s in end_ss.states}
        assert is_lattice_homomorphism(end_ss, end_ss, m)

    def test_make_homomorphism_valid(self, chain2: StateSpace) -> None:
        h = make_homomorphism(chain2, chain2, {s: s for s in chain2.states})
        assert h.preserves_meets
        assert h.preserves_joins

    def test_make_homomorphism_raises_on_invalid(self, diamond: StateSpace) -> None:
        """Swapping top↔bottom reverses order, should fail."""
        m = {s: s for s in diamond.states}
        m[diamond.top] = diamond.bottom
        m[diamond.bottom] = diamond.top
        with pytest.raises(ValueError, match="preserve"):
            make_homomorphism(diamond, diamond, m)

    def test_make_homomorphism_raises_missing_state(self, chain2: StateSpace) -> None:
        with pytest.raises(ValueError, match="missing"):
            make_homomorphism(chain2, chain2, {})


# ---------------------------------------------------------------------------
# TestIdentityMorphism
# ---------------------------------------------------------------------------

class TestIdentityMorphism:

    def test_mapping_correct(self, chain2: StateSpace) -> None:
        id_m = identity_morphism(chain2)
        assert id_m.mapping == {s: s for s in chain2.states}

    def test_preserves_meets(self, chain2: StateSpace) -> None:
        assert identity_morphism(chain2).preserves_meets

    def test_preserves_joins(self, chain2: StateSpace) -> None:
        assert identity_morphism(chain2).preserves_joins

    def test_source_target_same(self, diamond: StateSpace) -> None:
        id_m = identity_morphism(diamond)
        assert id_m.source is id_m.target

    def test_identity_on_product(self, par2: StateSpace) -> None:
        id_m = identity_morphism(par2)
        assert len(id_m.mapping) == len(par2.states)
        assert all(id_m.mapping[s] == s for s in par2.states)


# ---------------------------------------------------------------------------
# TestCompose
# ---------------------------------------------------------------------------

class TestCompose:

    def test_compose_identity_left(self, par2: StateSpace) -> None:
        pi = find_projections(par2)[0]
        id_m = identity_morphism(par2)
        comp = compose(id_m, pi)
        assert comp.mapping == pi.mapping

    def test_compose_identity_right(self, par2: StateSpace) -> None:
        pi = find_projections(par2)[0]
        id_tgt = identity_morphism(pi.target)
        comp = compose(pi, id_tgt)
        assert comp.mapping == pi.mapping

    def test_compose_preserves_meets(self, par2: StateSpace) -> None:
        id_m = identity_morphism(par2)
        comp = compose(id_m, id_m)
        assert comp.preserves_meets

    def test_compose_preserves_joins(self, par2: StateSpace) -> None:
        id_m = identity_morphism(par2)
        comp = compose(id_m, id_m)
        assert comp.preserves_joins

    def test_compose_domain_mismatch_raises(self, par2: StateSpace) -> None:
        """Projecting then composing with a mapping that lacks the factor states."""
        pi = find_projections(par2)[0]
        # Create a morphism whose mapping doesn't cover pi's target states
        fake_mapping = {999: 0}
        fake = LatticeHomomorphism(
            source=pi.target, target=pi.target,
            mapping=fake_mapping,
            preserves_meets=True, preserves_joins=True,
        )
        with pytest.raises((ValueError, KeyError)):
            compose(pi, fake)

    def test_compose_associative(self, par2: StateSpace) -> None:
        id_m = identity_morphism(par2)
        fg = compose(id_m, id_m)
        fgh1 = compose(fg, id_m)
        gh = compose(id_m, id_m)
        fgh2 = compose(id_m, gh)
        assert fgh1.mapping == fgh2.mapping

    def test_compose_projection_chain(self, par2: StateSpace) -> None:
        """Compose projection with identity on factor."""
        pi = find_projections(par2)[0]
        id_factor = identity_morphism(pi.target)
        comp = compose(pi, id_factor)
        assert comp.mapping == pi.mapping

    def test_compose_two_identities(self, diamond: StateSpace) -> None:
        id_m = identity_morphism(diamond)
        comp = compose(id_m, id_m)
        assert comp.mapping == id_m.mapping


# ---------------------------------------------------------------------------
# TestFindProjections
# ---------------------------------------------------------------------------

class TestFindProjections:

    def test_binary_product_two_projections(self, par2: StateSpace) -> None:
        projs = find_projections(par2)
        assert len(projs) == 2

    def test_ternary_product_three_projections(self, par3: StateSpace) -> None:
        projs = find_projections(par3)
        assert len(projs) == 3

    def test_projection_target_is_factor(self, par2: StateSpace) -> None:
        projs = find_projections(par2)
        assert par2.product_factors is not None
        for i, pi in enumerate(projs):
            assert pi.target is par2.product_factors[i]

    def test_projection_is_surjective(self, par2: StateSpace) -> None:
        projs = find_projections(par2)
        for pi in projs:
            image = set(pi.mapping.values())
            assert image == pi.target.states

    def test_projection_preserves_top(self, par2: StateSpace) -> None:
        projs = find_projections(par2)
        assert par2.product_factors is not None
        for i, pi in enumerate(projs):
            assert pi.mapping[par2.top] == par2.product_factors[i].top

    def test_projection_preserves_bottom(self, par2: StateSpace) -> None:
        projs = find_projections(par2)
        assert par2.product_factors is not None
        for i, pi in enumerate(projs):
            assert pi.mapping[par2.bottom] == par2.product_factors[i].bottom

    def test_no_product_coords_raises(self, diamond: StateSpace) -> None:
        with pytest.raises(ValueError, match="no product metadata"):
            find_projections(diamond)

    def test_deep_product_projections(self, par2_deep: StateSpace) -> None:
        projs = find_projections(par2_deep)
        assert len(projs) == 2
        # Left factor has 3 states, right has 2
        assert par2_deep.product_factors is not None
        assert len(par2_deep.product_factors[0].states) == 3
        assert len(par2_deep.product_factors[1].states) == 2


# ---------------------------------------------------------------------------
# TestUniversalMorphism
# ---------------------------------------------------------------------------

class TestUniversalMorphism:

    def test_mediating_morphism_exists(self, par2: StateSpace) -> None:
        projs = find_projections(par2)
        med = universal_morphism(projs[0], projs[1], par2)
        assert len(med.mapping) == len(par2.states)

    def test_mediating_commutes_left(self, par2: StateSpace) -> None:
        projs = find_projections(par2)
        med = universal_morphism(projs[0], projs[1], par2)
        comp = compose(med, projs[0])
        assert comp.mapping == projs[0].mapping

    def test_mediating_commutes_right(self, par2: StateSpace) -> None:
        projs = find_projections(par2)
        med = universal_morphism(projs[0], projs[1], par2)
        comp = compose(med, projs[1])
        assert comp.mapping == projs[1].mapping

    def test_mediating_is_identity(self, par2: StateSpace) -> None:
        """⟨π₁, π₂⟩ = id on the product."""
        projs = find_projections(par2)
        med = universal_morphism(projs[0], projs[1], par2)
        assert med.mapping == {s: s for s in par2.states}

    def test_nary_mediating(self, par3: StateSpace) -> None:
        projs = find_projections(par3)
        med = universal_morphism_nary(projs, par3)
        assert med.mapping == {s: s for s in par3.states}

    def test_nary_commutes(self, par3: StateSpace) -> None:
        projs = find_projections(par3)
        med = universal_morphism_nary(projs, par3)
        for i, pi in enumerate(projs):
            comp = compose(med, pi)
            assert comp.mapping == pi.mapping

    def test_from_factor_identity(self, par2: StateSpace) -> None:
        """Use identity on each factor to build mediating into product."""
        assert par2.product_factors is not None
        f1 = identity_morphism(par2.product_factors[0])
        f2 = identity_morphism(par2.product_factors[1])
        # Build product from two factor identities (source = factor, not product)
        # This should map factor1.top → product state with (top, top) for dim 1
        # Not directly applicable since f1, f2 have different sources.
        # Instead, test with projections (which share the same source).
        projs = find_projections(par2)
        med = universal_morphism(projs[0], projs[1], par2)
        assert med.mapping == {s: s for s in par2.states}

    def test_invalid_product_raises(self, diamond: StateSpace) -> None:
        id_m = identity_morphism(diamond)
        with pytest.raises(ValueError, match="no product_coords"):
            universal_morphism(id_m, id_m, diamond)


# ---------------------------------------------------------------------------
# TestCheckProductUniversalProperty
# ---------------------------------------------------------------------------

class TestCheckProductUniversalProperty:

    def test_parallel_is_product(self, par2: StateSpace) -> None:
        result = check_product_universal_property(par2)
        assert result.is_product

    def test_result_fields(self, par2: StateSpace) -> None:
        result = check_product_universal_property(par2)
        assert result.universal_property_holds
        assert result.counterexample is None
        assert len(result.projections) == 2

    def test_non_product_fails(self, diamond: StateSpace) -> None:
        result = check_product_universal_property(diamond)
        assert not result.is_product

    def test_three_way_product(self, par3: StateSpace) -> None:
        result = check_product_universal_property(par3)
        assert result.is_product
        assert len(result.projections) == 3

    def test_deep_product(self, par2_deep: StateSpace) -> None:
        result = check_product_universal_property(par2_deep)
        assert result.is_product

    def test_end_is_not_product(self, end_ss: StateSpace) -> None:
        result = check_product_universal_property(end_ss)
        assert not result.is_product


# ---------------------------------------------------------------------------
# TestIsProduct
# ---------------------------------------------------------------------------

class TestIsProduct:

    def test_parallel_is_product(self, par2: StateSpace) -> None:
        assert is_product(par2)

    def test_non_parallel_is_not_product(self, diamond: StateSpace) -> None:
        assert not is_product(diamond)

    def test_end_not_product(self, end_ss: StateSpace) -> None:
        assert not is_product(end_ss)

    def test_chain_not_product(self, chain3: StateSpace) -> None:
        assert not is_product(chain3)


# ---------------------------------------------------------------------------
# TestCheckSesslatCategory
# ---------------------------------------------------------------------------

class TestCheckSesslatCategory:

    def test_single_object(self, chain2: StateSpace) -> None:
        result = check_sesslat_category([chain2])
        assert result.identity_ok
        assert result.composition_ok
        assert result.associativity_ok
        assert result.num_objects == 1

    def test_multiple_objects(self, chain2: StateSpace, diamond: StateSpace, par2: StateSpace) -> None:
        result = check_sesslat_category([chain2, diamond, par2])
        assert result.identity_ok
        assert result.composition_ok
        assert result.associativity_ok
        assert result.num_objects == 3

    def test_identity_axiom(self, diamond: StateSpace) -> None:
        result = check_sesslat_category([diamond])
        assert result.identity_ok

    def test_result_type(self, chain2: StateSpace) -> None:
        result = check_sesslat_category([chain2])
        assert isinstance(result, CategoryResult)
        assert result.counterexample is None

    def test_product_composition(self, par2: StateSpace) -> None:
        result = check_sesslat_category([par2])
        assert result.composition_ok


# ---------------------------------------------------------------------------
# TestSubdirectlyIrreducible
# ---------------------------------------------------------------------------

class TestSubdirectlyIrreducible:

    def test_chain_is_irreducible(self, chain3: StateSpace) -> None:
        assert is_subdirectly_irreducible(chain3)

    def test_product_not_irreducible(self, par2: StateSpace) -> None:
        assert not is_subdirectly_irreducible(par2)

    def test_end_is_irreducible(self, end_ss: StateSpace) -> None:
        assert is_subdirectly_irreducible(end_ss)

    def test_diamond_is_irreducible(self, diamond: StateSpace) -> None:
        # Diamond (not from parallel) is irreducible
        assert is_subdirectly_irreducible(diamond)

    def test_two_element_chain(self, chain2: StateSpace) -> None:
        assert is_subdirectly_irreducible(chain2)

    def test_three_way_product_not_irreducible(self, par3: StateSpace) -> None:
        assert not is_subdirectly_irreducible(par3)


# ---------------------------------------------------------------------------
# TestFindProductDecomposition
# ---------------------------------------------------------------------------

class TestFindProductDecomposition:

    def test_parallel_has_decomposition(self, par2: StateSpace) -> None:
        factors = find_product_decomposition(par2)
        assert factors is not None
        assert len(factors) == 2

    def test_non_product_returns_none(self, diamond: StateSpace) -> None:
        assert find_product_decomposition(diamond) is None

    def test_decomposition_factors_correct(self, par2: StateSpace) -> None:
        factors = find_product_decomposition(par2)
        assert factors is not None
        assert par2.product_factors is not None
        for i, f in enumerate(factors):
            assert f is par2.product_factors[i]

    def test_end_returns_none(self, end_ss: StateSpace) -> None:
        assert find_product_decomposition(end_ss) is None

    def test_three_way_decomposition(self, par3: StateSpace) -> None:
        factors = find_product_decomposition(par3)
        assert factors is not None
        assert len(factors) == 3


# ---------------------------------------------------------------------------
# Benchmark sweep
# ---------------------------------------------------------------------------

class TestBenchmarkCategory:

    @pytest.fixture
    def all_benchmarks(self) -> list[tuple[str, StateSpace]]:
        from tests.benchmarks.protocols import BENCHMARKS
        results = []
        for bp in BENCHMARKS:
            ss = build_statespace(parse(bp.type_string))
            if check_lattice(ss).is_lattice:
                results.append((bp.name, ss))
        return results

    def test_all_benchmarks_identity(self, all_benchmarks: list[tuple[str, StateSpace]]) -> None:
        for name, ss in all_benchmarks:
            id_m = identity_morphism(ss)
            assert is_lattice_homomorphism(ss, ss, id_m.mapping), f"identity failed on {name}"

    def test_parallel_benchmarks_product_property(self, all_benchmarks: list[tuple[str, StateSpace]]) -> None:
        for name, ss in all_benchmarks:
            if ss.product_coords is not None:
                # Skip types with continuation states beyond the product
                if not all(sid in ss.product_coords for sid in ss.states):
                    continue
                result = check_product_universal_property(ss)
                assert result.is_product, f"product property failed on {name}: {result.counterexample}"

    def test_category_axioms_hold(self, all_benchmarks: list[tuple[str, StateSpace]]) -> None:
        spaces = [ss for _, ss in all_benchmarks[:10]]  # first 10 for speed
        result = check_sesslat_category(spaces)
        assert result.identity_ok
        assert result.composition_ok
        assert result.associativity_ok
