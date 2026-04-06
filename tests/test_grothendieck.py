"""Tests for the Grothendieck group module (Step 32k)."""

from collections import Counter

import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.grothendieck import (
    GrothendieckGroup,
    GrothendieckResult,
    VirtualLattice,
    build_k0,
    lattice_signature,
)


def ss(src: str):
    return build_statespace(parse(src))


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------

def test_signature_deterministic():
    s1 = lattice_signature(ss("&{a: end, b: end}"))
    s2 = lattice_signature(ss("&{a: end, b: end}"))
    assert s1 == s2


def test_signature_distinguishes_different_shapes():
    a = lattice_signature(ss("end"))
    b = lattice_signature(ss("&{a: end}"))
    c = lattice_signature(ss("&{a: end, b: end}"))
    assert a != b != c and a != c


def test_signature_renaming_invariant():
    s1 = lattice_signature(ss("&{foo: end, bar: end}"))
    s2 = lattice_signature(ss("&{x: end, y: end}"))
    assert s1 == s2  # rank profile + degree multiset don't see labels


def test_signature_end_is_singleton():
    sig = lattice_signature(ss("end"))
    assert sig[0] == 1  # single state
    assert sig[1] == 0  # height 0


# ---------------------------------------------------------------------------
# VirtualLattice algebra
# ---------------------------------------------------------------------------

def test_zero_virtual_lattice():
    z = VirtualLattice.zero()
    assert z.is_zero()
    assert z + z == z


def test_generator_round_trip():
    sig = lattice_signature(ss("&{a: end}"))
    g = VirtualLattice.generator(sig)
    assert not g.is_zero()
    assert g.pos_counter() == Counter({sig: 1})


def test_addition_commutative():
    s1 = lattice_signature(ss("end"))
    s2 = lattice_signature(ss("&{a: end}"))
    g1 = VirtualLattice.generator(s1)
    g2 = VirtualLattice.generator(s2)
    assert g1 + g2 == g2 + g1


def test_addition_associative():
    s1 = lattice_signature(ss("end"))
    s2 = lattice_signature(ss("&{a: end}"))
    s3 = lattice_signature(ss("&{a: end, b: end}"))
    g1 = VirtualLattice.generator(s1)
    g2 = VirtualLattice.generator(s2)
    g3 = VirtualLattice.generator(s3)
    assert (g1 + g2) + g3 == g1 + (g2 + g3)


def test_inverse_law():
    sig = lattice_signature(ss("&{a: end, b: end}"))
    g = VirtualLattice.generator(sig)
    assert (g + (-g)).is_zero()
    assert (g - g).is_zero()


def test_cancellation():
    s1 = lattice_signature(ss("end"))
    s2 = lattice_signature(ss("&{a: end}"))
    g1 = VirtualLattice.generator(s1)
    g2 = VirtualLattice.generator(s2)
    v = (g1 + g2) - g2
    assert v == g1


def test_total_size_signed():
    s1 = lattice_signature(ss("&{a: end, b: end}"))  # 3 states
    s2 = lattice_signature(ss("end"))                # 1 state
    g1 = VirtualLattice.generator(s1)
    g2 = VirtualLattice.generator(s2)
    assert (g1 - g2).total_size() == s1[0] - s2[0]


def test_double_negation():
    sig = lattice_signature(ss("&{a: end}"))
    g = VirtualLattice.generator(sig)
    assert -(-g) == g


# ---------------------------------------------------------------------------
# Grothendieck group
# ---------------------------------------------------------------------------

def test_group_register_and_phi():
    G = GrothendieckGroup("sum")
    s = ss("&{a: end, b: end}")
    sig = G.register(s)
    assert sig in G.generators()
    g = G.phi(s)
    assert g == VirtualLattice.generator(sig)


def test_psi_of_phi_is_identity_on_generators():
    G = GrothendieckGroup()
    s = ss("&{a: end}")
    v = G.phi(s)
    pos, neg = G.psi(v)
    assert sum(pos.values()) == 1
    assert sum(neg.values()) == 0


def test_psi_of_difference():
    G = GrothendieckGroup()
    a = ss("&{a: end}")
    b = ss("&{a: end, b: end}")
    v = G.virtual_difference(a, b)
    pos, neg = G.psi(v)
    assert sum(pos.values()) == 1
    assert sum(neg.values()) == 1


def test_same_space_difference_is_zero():
    G = GrothendieckGroup()
    a = ss("&{a: end, b: end}")
    b = ss("&{x: end, y: end}")  # same signature via rename invariance
    assert G.virtual_difference(a, b).is_zero()


def test_combine_signatures_sum():
    G = GrothendieckGroup("sum")
    s1 = lattice_signature(ss("&{a: end}"))
    s2 = lattice_signature(ss("&{a: end}"))
    comb = G.combine_signatures(s1, s2)
    # Rank profiles sum pointwise.
    assert comb[2] == tuple(2 * x for x in s1[2])


def test_combine_signatures_product_convolves():
    G = GrothendieckGroup("product")
    s1 = lattice_signature(ss("&{a: end}"))  # rank profile (1,1)
    s2 = lattice_signature(ss("&{a: end}"))
    comb = G.combine_signatures(s1, s2)
    # Convolution of (1,1) with (1,1) = (1,2,1).
    assert comb[2] == (1, 2, 1)


def test_combine_signatures_unknown_op():
    G = GrothendieckGroup("banana")
    s1 = lattice_signature(ss("end"))
    with pytest.raises(ValueError):
        G.combine_signatures(s1, s1)


# ---------------------------------------------------------------------------
# build_k0
# ---------------------------------------------------------------------------

def test_build_k0_basic():
    spaces = [ss("end"), ss("&{a: end}"), ss("&{a: end, b: end}")]
    r = build_k0(spaces)
    assert isinstance(r, GrothendieckResult)
    assert r.num_generators == 3
    assert r.is_free_abelian
    assert r.relations_verified == 2 * len(spaces)


def test_build_k0_deduplicates_isomorphic():
    spaces = [ss("&{a: end, b: end}"), ss("&{x: end, y: end}")]
    r = build_k0(spaces)
    assert r.num_generators == 1


def test_build_k0_product_operation():
    spaces = [ss("&{a: end}"), ss("&{a: end, b: end}")]
    r = build_k0(spaces, operation="product")
    assert r.operation == "product"
    assert r.num_generators == 2


def test_build_k0_empty():
    r = build_k0([])
    assert r.num_generators == 0
    assert r.is_free_abelian


# ---------------------------------------------------------------------------
# Morphism laws
# ---------------------------------------------------------------------------

def test_phi_isomorphism_invariant():
    G = GrothendieckGroup()
    a = G.phi(ss("&{a: end, b: end}"))
    b = G.phi(ss("&{u: end, v: end}"))
    assert a == b


def test_virtual_difference_antisymmetric():
    G = GrothendieckGroup()
    a = ss("&{a: end}")
    b = ss("&{a: end, b: end}")
    assert G.virtual_difference(a, b) == -(G.virtual_difference(b, a))


def test_reduced_form_unique():
    """Two equivalent virtual differences collapse to the same reduced pair."""
    s1 = lattice_signature(ss("&{a: end}"))
    s2 = lattice_signature(ss("&{a: end, b: end}"))
    g1 = VirtualLattice.generator(s1)
    g2 = VirtualLattice.generator(s2)
    # (g1 + g2) − g2  ==  g1
    v1 = (g1 + g2) - g2
    v2 = g1
    assert v1 == v2


def test_phi_monoid_morphism_sum_at_signature_level():
    """φ(L₁⊕L₂) corresponds to combine_signatures at the sig level."""
    G = GrothendieckGroup("sum")
    s1 = ss("&{a: end}")
    s2 = ss("&{a: end, b: end}")
    sig1 = lattice_signature(s1)
    sig2 = lattice_signature(s2)
    combined = G.combine_signatures(sig1, sig2)
    # Combined has pointwise-summed rank profile.
    assert combined[0] == sig1[0] + sig2[0]


def test_k0_has_abelian_addition_on_benchmarks():
    """Multiple independent lattices behave as a free abelian group."""
    G = GrothendieckGroup()
    sources = [
        "end",
        "&{a: end}",
        "+{x: end, y: end}",
        "&{a: end, b: +{x: end}}",
    ]
    virtuals = [G.phi(ss(s)) for s in sources]
    total = VirtualLattice.zero()
    for v in virtuals:
        total = total + v
    # Reverse order
    total2 = VirtualLattice.zero()
    for v in reversed(virtuals):
        total2 = total2 + v
    assert total == total2
    # Subtract all: zero.
    for v in virtuals:
        total = total - v
    assert total.is_zero()


def test_signature_degree_multiset_present():
    sig = lattice_signature(ss("&{a: end, b: end}"))
    # Degree multiset is a tuple of (in,out) pairs.
    assert isinstance(sig[3], tuple)
    assert all(isinstance(p, tuple) and len(p) == 2 for p in sig[3])
