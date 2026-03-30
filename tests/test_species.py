"""Tests for combinatorial species analysis (Step 32h).

Tests cover:
- Automorphism group computation
- Isomorphism type counting
- Type-generating series
- Cycle index computation
- Species product and sum
- Exponential formula terms
- Symmetry factor
- Benchmark protocol analysis
"""

import pytest
import math
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.species import (
    find_automorphisms,
    automorphism_group_size,
    isomorphism_types,
    type_generating_series,
    cycle_index,
    species_product,
    species_sum,
    exponential_formula_terms,
    symmetry_factor,
    analyze_species,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(s: str):
    """Parse and build state space."""
    return build_statespace(parse(s))


# ---------------------------------------------------------------------------
# Automorphism group
# ---------------------------------------------------------------------------

class TestAutomorphisms:
    """Tests for automorphism group computation."""

    def test_end_automorphism(self):
        """end has exactly 1 automorphism (identity)."""
        ss = _build("end")
        auts = find_automorphisms(ss)
        assert len(auts) == 1

    def test_single_branch_automorphism(self):
        """&{a: end} has identity as only automorphism (top != bottom)."""
        ss = _build("&{a: end}")
        auts = find_automorphisms(ss)
        # Top and bottom are distinguishable, so only identity
        assert len(auts) == 1

    def test_automorphism_is_valid(self):
        """Each automorphism must be a bijection on states."""
        ss = _build("&{a: end, b: end}")
        auts = find_automorphisms(ss)
        for aut in auts:
            # Bijection check
            assert set(aut.keys()) == ss.states
            assert set(aut.values()) == ss.states

    def test_identity_always_present(self):
        """Identity is always an automorphism."""
        for t in ["end", "&{a: end}", "&{a: end, b: end}", "+{a: end}"]:
            ss = _build(t)
            auts = find_automorphisms(ss)
            identity = {s: s for s in ss.states}
            assert identity in auts

    def test_symmetric_branch_has_swap(self):
        """&{a: end, b: end}: middle states may be swappable."""
        ss = _build("&{a: end, b: end}")
        auts = find_automorphisms(ss)
        # There should be at least 1 (identity), possibly more for symmetric branches
        assert len(auts) >= 1

    def test_chain_no_nontrivial(self):
        """Pure chain &{a: &{b: end}} has only identity automorphism."""
        ss = _build("&{a: &{b: end}}")
        auts = find_automorphisms(ss)
        assert len(auts) == 1

    def test_automorphism_group_size(self):
        """Convenience function matches list length."""
        ss = _build("&{a: end, b: end}")
        assert automorphism_group_size(ss) == len(find_automorphisms(ss))


# ---------------------------------------------------------------------------
# Isomorphism types
# ---------------------------------------------------------------------------

class TestIsomorphismTypes:
    """Tests for isomorphism type counting."""

    def test_end_types(self):
        """end has at least 1 isomorphism type (the single point)."""
        ss = _build("end")
        types = isomorphism_types(ss)
        assert len(types) >= 1

    def test_single_branch_types(self):
        """&{a: end} has types for size 1 and size 2."""
        ss = _build("&{a: end}")
        types = isomorphism_types(ss)
        assert len(types) >= 2  # point + chain

    def test_types_are_frozensets(self):
        """Each type is represented as a frozenset of states."""
        ss = _build("&{a: end, b: end}")
        types = isomorphism_types(ss)
        for t in types:
            assert isinstance(t, frozenset)

    def test_types_nonempty(self):
        """All isomorphism types are non-empty."""
        ss = _build("&{a: end, b: end}")
        types = isomorphism_types(ss)
        for t in types:
            assert len(t) > 0

    def test_more_states_more_types(self):
        """Larger lattice should have at least as many types as smaller."""
        ss_small = _build("&{a: end}")
        ss_large = _build("&{a: end, b: end}")
        t_small = len(isomorphism_types(ss_small))
        t_large = len(isomorphism_types(ss_large))
        assert t_large >= t_small


# ---------------------------------------------------------------------------
# Type-generating series
# ---------------------------------------------------------------------------

class TestTypeGeneratingSeries:
    """Tests for type-generating series computation."""

    def test_end_series(self):
        """end: t_F(x) = 1 + x (empty + single point)."""
        ss = _build("end")
        tgs = type_generating_series(ss)
        assert tgs[0] == 1.0  # empty sub-poset

    def test_series_a0_is_1(self):
        """First coefficient is always 1 (empty sub-poset)."""
        for t in ["end", "&{a: end}", "&{a: end, b: end}"]:
            ss = _build(t)
            tgs = type_generating_series(ss)
            assert tgs[0] == 1.0

    def test_series_a1_is_1(self):
        """Second coefficient is 1 (single point, one type)."""
        for t in ["end", "&{a: end}", "&{a: end, b: end}"]:
            ss = _build(t)
            tgs = type_generating_series(ss)
            if len(tgs) > 1:
                assert tgs[1] == 1.0

    def test_series_nonnegative(self):
        """All coefficients are non-negative."""
        ss = _build("&{a: end, b: end}")
        tgs = type_generating_series(ss)
        for c in tgs:
            assert c >= 0

    def test_series_length(self):
        """Series has at most n+1 terms for n states."""
        ss = _build("&{a: end, b: end}")
        n = len(ss.states)
        tgs = type_generating_series(ss)
        assert len(tgs) <= n + 1


# ---------------------------------------------------------------------------
# Cycle index
# ---------------------------------------------------------------------------

class TestCycleIndex:
    """Tests for cycle index computation."""

    def test_end_cycle_index(self):
        """end: cycle index = 1 * p_1 (identity only)."""
        ss = _build("end")
        ci = cycle_index(ss)
        assert len(ci) >= 1
        # Sum of coefficients should be 1
        total = sum(c for c, _ in ci)
        assert abs(total - 1.0) < 1e-9

    def test_coefficients_sum_to_1(self):
        """Cycle index coefficients always sum to 1."""
        for t in ["end", "&{a: end}", "&{a: end, b: end}", "+{a: end}"]:
            ss = _build(t)
            ci = cycle_index(ss)
            total = sum(c for c, _ in ci)
            assert abs(total - 1.0) < 1e-9

    def test_partitions_valid(self):
        """Each partition sums to |P|."""
        ss = _build("&{a: end, b: end}")
        n = len(ss.states)
        ci = cycle_index(ss)
        for _, partition in ci:
            assert sum(partition) == n

    def test_coefficients_positive(self):
        """All cycle index coefficients are positive."""
        ss = _build("&{a: end}")
        ci = cycle_index(ss)
        for c, _ in ci:
            assert c > 0


# ---------------------------------------------------------------------------
# Species product
# ---------------------------------------------------------------------------

class TestSpeciesProduct:
    """Tests for species product (parallel composition)."""

    def test_product_states(self):
        """Product states = left * right."""
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end}")
        result = species_product(ss1, ss2)
        assert result["product_states"] == result["left_states"] * result["right_states"]

    def test_product_automorphisms_upper_bound(self):
        """Product automorphisms <= |Aut(L1)| * |Aut(L2)|."""
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end}")
        result = species_product(ss1, ss2)
        assert result["product_automorphisms_upper"] == (
            result["left_automorphisms"] * result["right_automorphisms"]
        )

    def test_product_symmetric(self):
        """species_product(A, B) and species_product(B, A) have same product_states."""
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end, c: end}")
        r1 = species_product(ss1, ss2)
        r2 = species_product(ss2, ss1)
        assert r1["product_states"] == r2["product_states"]


# ---------------------------------------------------------------------------
# Species sum
# ---------------------------------------------------------------------------

class TestSpeciesSum:
    """Tests for species sum (external choice)."""

    def test_sum_states(self):
        """Sum states = left + right."""
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end}")
        result = species_sum(ss1, ss2)
        assert result["sum_states"] == result["left_states"] + result["right_states"]


# ---------------------------------------------------------------------------
# Exponential formula
# ---------------------------------------------------------------------------

class TestExponentialFormula:
    """Tests for exponential formula terms."""

    def test_end_exp_terms(self):
        """end: exponential formula terms exist."""
        ss = _build("end")
        terms = exponential_formula_terms(ss)
        assert len(terms) >= 1

    def test_terms_length(self):
        """Terms length matches type-generating series length."""
        ss = _build("&{a: end}")
        tgs = type_generating_series(ss)
        terms = exponential_formula_terms(ss)
        assert len(terms) == len(tgs)


# ---------------------------------------------------------------------------
# Symmetry factor
# ---------------------------------------------------------------------------

class TestSymmetryFactor:
    """Tests for symmetry factor computation."""

    def test_end_symmetry(self):
        """end: symmetry factor = 1/1! = 1."""
        ss = _build("end")
        sf = symmetry_factor(ss)
        assert abs(sf - 1.0) < 1e-9

    def test_chain_symmetry(self):
        """Pure chain: only identity automorphism, factor = 1/n!."""
        ss = _build("&{a: &{b: end}}")
        sf = symmetry_factor(ss)
        n = len(ss.states)
        expected = 1.0 / math.factorial(n)
        assert abs(sf - expected) < 1e-9

    def test_symmetry_bounded(self):
        """Symmetry factor is between 0 and 1."""
        for t in ["end", "&{a: end}", "&{a: end, b: end}"]:
            ss = _build(t)
            sf = symmetry_factor(ss)
            assert 0 < sf <= 1.0


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

class TestAnalyzeSpecies:
    """Tests for the complete analysis function."""

    def test_end_analysis(self):
        """Complete analysis of end type."""
        ss = _build("end")
        result = analyze_species(ss)
        assert result.num_states == 1
        assert result.num_automorphisms == 1
        assert result.symmetry_factor == 1.0

    def test_single_branch_analysis(self):
        """Complete analysis of &{a: end}."""
        ss = _build("&{a: end}")
        result = analyze_species(ss)
        assert result.num_states == 2
        assert result.num_automorphisms >= 1
        assert len(result.type_generating_coefficients) >= 1

    def test_two_branch_analysis(self):
        """Complete analysis of &{a: end, b: end} (2 states, branches merge)."""
        ss = _build("&{a: end, b: end}")
        result = analyze_species(ss)
        assert result.num_states == 2
        assert result.isomorphism_type_count >= 1

    def test_selection_analysis(self):
        """Complete analysis of +{a: end, b: end}."""
        ss = _build("+{a: end, b: end}")
        result = analyze_species(ss)
        assert result.num_automorphisms >= 1

    def test_nested_analysis(self):
        """Complete analysis of nested type."""
        ss = _build("&{a: &{c: end}, b: end}")
        result = analyze_species(ss)
        assert result.num_states >= 3

    def test_recursive_analysis(self):
        """Analysis handles recursive types."""
        ss = _build("rec X . &{a: X, b: end}")
        result = analyze_species(ss)
        assert result.num_states >= 1


# ---------------------------------------------------------------------------
# Benchmark protocols
# ---------------------------------------------------------------------------

class TestBenchmarks:
    """Tests on standard benchmark protocols."""

    def test_iterator(self):
        """Java Iterator protocol."""
        ss = _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        result = analyze_species(ss)
        assert result.num_states >= 2
        assert result.num_automorphisms >= 1

    def test_simple_resource(self):
        """Simple resource: open, use, close."""
        ss = _build("&{open: &{use: &{close: end}}}")
        result = analyze_species(ss)
        assert result.num_states == 4
        assert result.num_automorphisms == 1  # chain, no symmetry

    def test_parallel_simple(self):
        """Parallel composition."""
        ss = _build("(&{a: end} || &{b: end})")
        result = analyze_species(ss)
        assert result.num_states >= 4

    def test_binary_choice(self):
        """Binary selection."""
        ss = _build("+{left: end, right: end}")
        result = analyze_species(ss)
        assert result.num_states >= 2

    def test_deep_chain(self):
        """Deep chain: no non-trivial automorphisms."""
        ss = _build("&{a: &{b: &{c: &{d: end}}}}")
        result = analyze_species(ss)
        assert result.num_automorphisms == 1
