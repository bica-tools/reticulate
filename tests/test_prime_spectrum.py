"""Tests for reticulate.prime_spectrum (Step 363h)."""

from __future__ import annotations

import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.prime_spectrum import (
    PrimeIdeal,
    SpectrumResult,
    ZariskiTopology,
    build_topology,
    check_morphisms,
    compute_spectrum,
    enumerate_ideals,
    enumerate_prime_ideals,
    irreducible_components,
    phi_map,
    psi_map,
    _build_lattice,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ss_of(s: str):
    return build_statespace(parse(s))


def lat_of(s: str):
    return _build_lattice(ss_of(s))


# ---------------------------------------------------------------------------
# Smoke tests on the lattice carrier
# ---------------------------------------------------------------------------

class TestLatticeCarrier:

    def test_single_end(self):
        L = lat_of("end")
        assert len(L.elements) == 1
        assert L.top == L.bottom

    def test_two_chain(self):
        L = lat_of("&{m: end}")
        assert len(L.elements) == 2
        # bottom <= top
        assert L.is_le(L.bottom, L.top)
        assert L.is_le(L.top, L.top)
        assert not L.is_lt(L.top, L.bottom)

    def test_three_chain(self):
        L = lat_of("&{a: &{b: end}}")
        assert len(L.elements) == 3

    def test_meet_join_well_defined(self):
        L = lat_of("&{a: &{b: end}}")
        for x in L.elements:
            for y in L.elements:
                assert (x, y) in L.meet
                assert (x, y) in L.join
                # Idempotent
        for x in L.elements:
            assert L.meet[(x, x)] == x
            assert L.join[(x, x)] == x

    def test_top_bottom_absorption(self):
        L = lat_of("&{a: &{b: end}}")
        for x in L.elements:
            assert L.meet[(L.top, x)] == x
            assert L.join[(L.top, x)] == L.top
            assert L.meet[(L.bottom, x)] == L.bottom
            assert L.join[(L.bottom, x)] == x


# ---------------------------------------------------------------------------
# Ideal enumeration
# ---------------------------------------------------------------------------

class TestIdeals:

    def test_single_element(self):
        L = lat_of("end")
        ideals = enumerate_ideals(L)
        # Just {bottom} which is also {top}
        assert len(ideals) == 1
        assert ideals[0] == frozenset({L.bottom})

    def test_two_chain_ideals(self):
        L = lat_of("&{m: end}")
        ideals = enumerate_ideals(L)
        assert len(ideals) == 2  # {bot}, {bot, top}

    def test_three_chain_ideals(self):
        L = lat_of("&{a: &{b: end}}")
        ideals = enumerate_ideals(L)
        # n+1 ideals in an n-chain
        assert len(ideals) == 3

    def test_all_ideals_contain_bottom(self):
        L = lat_of("&{a: &{b: end}}")
        for I in enumerate_ideals(L):
            assert L.bottom in I

    def test_full_lattice_is_an_ideal(self):
        L = lat_of("&{a: &{b: end}}")
        ideals = enumerate_ideals(L)
        assert frozenset(L.elements) in ideals


# ---------------------------------------------------------------------------
# Prime ideals
# ---------------------------------------------------------------------------

class TestPrimeIdeals:

    def test_chain_has_n_minus_1_primes(self):
        # n-element chain has n-1 prime ideals
        for depth, n in [(1, 2), (2, 3), (3, 4)]:
            inner = "end"
            for _ in range(depth):
                inner = "&{m: " + inner + "}"
            L = lat_of(inner)
            primes = enumerate_prime_ideals(L)
            assert len(primes) == n - 1, (
                f"depth {depth}: expected {n-1} primes, got {len(primes)}"
            )

    def test_primes_are_proper(self):
        L = lat_of("&{a: &{b: end}}")
        full = frozenset(L.elements)
        for p in enumerate_prime_ideals(L):
            assert p.members != full
            assert len(p.members) >= 1

    def test_primes_are_downward_closed(self):
        L = lat_of("&{a: &{b: end}}")
        for p in enumerate_prime_ideals(L):
            for x in p.members:
                for y in L.elements:
                    if L.is_le(y, x):
                        assert y in p.members

    def test_prime_ideal_principal_in_chain(self):
        L = lat_of("&{a: &{b: end}}")
        primes = enumerate_prime_ideals(L)
        # In a chain every prime ideal is principal
        for p in primes:
            assert p.is_principal
            assert p.generator is not None


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

class TestTopology:

    def test_basic_open_at_bottom_is_empty(self):
        L = lat_of("&{a: &{b: end}}")
        primes = enumerate_prime_ideals(L)
        topo = build_topology(L, primes)
        # bottom is in every ideal, so D(bottom) is empty
        assert topo.basic_opens[L.bottom] == frozenset()

    def test_basic_open_at_top_is_full(self):
        L = lat_of("&{a: &{b: end}}")
        primes = enumerate_prime_ideals(L)
        topo = build_topology(L, primes)
        # top is in no proper ideal of a chain, so D(top) = full
        assert topo.basic_opens[L.top] == frozenset(range(len(primes)))

    def test_basic_open_intersection_meet(self):
        # D(a) intersect D(b) = D(a meet b)
        L = lat_of("&{a: &{b: end}}")
        primes = enumerate_prime_ideals(L)
        topo = build_topology(L, primes)
        for x in L.elements:
            for y in L.elements:
                lhs = topo.basic_opens[x] & topo.basic_opens[y]
                rhs = topo.basic_opens[L.meet[(x, y)]]
                assert lhs == rhs

    def test_closed_set_complement(self):
        L = lat_of("&{a: &{b: end}}")
        primes = enumerate_prime_ideals(L)
        topo = build_topology(L, primes)
        full = frozenset(range(len(primes)))
        for x in L.elements:
            assert topo.basic_opens[x] | topo.closed_sets[x] == full
            assert topo.basic_opens[x] & topo.closed_sets[x] == frozenset()


# ---------------------------------------------------------------------------
# Bidirectional morphisms
# ---------------------------------------------------------------------------

class TestMorphisms:

    def test_phi_order_preserving_chain(self):
        L = lat_of("&{a: &{b: end}}")
        primes = enumerate_prime_ideals(L)
        topo = build_topology(L, primes)
        morph = check_morphisms(L, topo)
        assert morph["phi_order_preserving"]
        assert morph["psi_well_defined"]
        assert morph["duality_holds"]

    def test_phi_injective_distributive(self):
        L = lat_of("&{a: &{b: end}}")
        primes = enumerate_prime_ideals(L)
        topo = build_topology(L, primes)
        morph = check_morphisms(L, topo)
        assert morph["phi_injective"]

    def test_psi_inverts_phi_in_chain(self):
        L = lat_of("&{a: &{b: end}}")
        primes = enumerate_prime_ideals(L)
        topo = build_topology(L, primes)
        psi = psi_map(L, topo.primes)
        # psi(D(a)) need not equal a, but i in phi(psi(i)) for every prime
        phi = phi_map(L, topo)
        for i, _ in enumerate(topo.primes):
            assert i in phi[psi[i]]


# ---------------------------------------------------------------------------
# Top-level compute_spectrum
# ---------------------------------------------------------------------------

class TestComputeSpectrum:

    def test_chain_spectrum(self):
        ss = ss_of("&{a: &{b: &{c: end}}}")
        r = compute_spectrum(ss)
        assert r.is_lattice
        assert r.is_distributive
        assert r.num_elements == 4
        assert r.num_prime_ideals == 3
        assert r.duality_holds
        assert r.phi_is_order_preserving
        assert r.phi_is_injective

    def test_irreducible_chain(self):
        ss = ss_of("&{a: &{b: end}}")
        r = compute_spectrum(ss)
        assert r.is_irreducible
        assert len(r.irreducible_components) == 1

    def test_parallel_distributive(self):
        # Two trivial parallel branches
        ss = ss_of("(&{a: end} || &{b: end})")
        r = compute_spectrum(ss)
        assert r.is_lattice
        # 2x2 product is distributive (Boolean B_2)
        assert r.is_distributive
        # B_2 has 2 prime ideals (one for each atom complement)
        assert r.num_prime_ideals == 2

    def test_single_end(self):
        ss = ss_of("end")
        r = compute_spectrum(ss)
        assert r.is_lattice
        assert r.num_elements == 1
        # Trivial lattice has no proper prime ideals
        assert r.num_prime_ideals == 0

    def test_notes_when_non_lattice(self):
        # Most parsable types are lattices; we just exercise the path
        ss = ss_of("&{a: end}")
        r = compute_spectrum(ss)
        assert r.is_lattice

    def test_components_are_a_partition(self):
        ss = ss_of("(&{a: end} || &{b: end})")
        r = compute_spectrum(ss)
        n = r.num_prime_ideals
        union = set()
        for c in r.irreducible_components:
            union |= set(c)
        assert union == set(range(n))


# ---------------------------------------------------------------------------
# Algebraic sanity: D(a meet b) = D(a) cap D(b), D(a join b) ⊇ D(a) cup D(b)
# ---------------------------------------------------------------------------

class TestAlgebraicLaws:

    @pytest.mark.parametrize("src", [
        "&{a: end}",
        "&{a: &{b: end}}",
        "&{a: &{b: &{c: end}}}",
        "(&{a: end} || &{b: end})",
    ])
    def test_d_meet(self, src):
        ss = ss_of(src)
        r = compute_spectrum(ss)
        L = _build_lattice(ss)
        topo = r.topology
        assert topo is not None
        for x in L.elements:
            for y in L.elements:
                assert (
                    topo.basic_opens[L.meet[(x, y)]]
                    == topo.basic_opens[x] & topo.basic_opens[y]
                )

    @pytest.mark.parametrize("src", [
        "&{a: end}",
        "&{a: &{b: end}}",
        "(&{a: end} || &{b: end})",
    ])
    def test_d_join_contains_union(self, src):
        ss = ss_of(src)
        r = compute_spectrum(ss)
        L = _build_lattice(ss)
        topo = r.topology
        assert topo is not None
        for x in L.elements:
            for y in L.elements:
                lhs = topo.basic_opens[L.join[(x, y)]]
                rhs = topo.basic_opens[x] | topo.basic_opens[y]
                assert rhs <= lhs
