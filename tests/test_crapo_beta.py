"""Tests for Crapo's beta invariant (Step 32n)."""

from __future__ import annotations

import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.crapo_beta import (
    BetaResult,
    analyze_beta,
    compute_beta,
    phi_decomposable,
    psi_bound,
    classify_pair,
)
from reticulate.zeta import mobius_function


def ss_of(t: str):
    return build_statespace(parse(t))


# ---------------------------------------------------------------------------
# Basic invariants
# ---------------------------------------------------------------------------

class TestCoreBeta:

    def test_trivial_single_state(self):
        ss = ss_of("end")
        r = analyze_beta(ss)
        assert r.trivial is True
        assert r.beta == 0
        assert r.decomposable is True

    def test_two_chain(self):
        # &{a: end} → two states: top=initial, bottom=end
        ss = ss_of("&{a: end}")
        r = analyze_beta(ss)
        assert r.num_states == 2
        # β = μ(0,0)μ(0,1) + μ(0,1)μ(1,1) = 1·(-1)+(-1)·1 = -2
        assert r.beta == -2
        assert r.decomposable is False

    def test_three_chain(self):
        # 3-chain has β = 1 in our sign convention (contribution only from
        # the interior element).
        ss = ss_of("&{a: &{b: end}}")
        assert compute_beta(ss) == 1

    def test_four_chain_is_zero(self):
        # Longer chains vanish: μ(0̂,1̂)=0 and interior cancellations give β=0.
        ss = ss_of("&{a: &{b: &{c: end}}}")
        assert compute_beta(ss) == 0

    def test_beta_is_integer(self):
        for t in ["end", "&{a: end}", "&{a: end, b: end}",
                  "(&{a: end} || &{b: end})", "&{a: &{b: end}}"]:
            assert isinstance(compute_beta(ss_of(t)), int)


class TestMultiplicativity:
    """β is multiplicative under parallel: β(L₁×L₂) = β(L₁)·β(L₂)."""

    def test_product_of_two_chains(self):
        l1 = ss_of("&{a: end}")
        l2 = ss_of("&{b: end}")
        prod = ss_of("(&{a: end} || &{b: end})")
        assert compute_beta(prod) == compute_beta(l1) * compute_beta(l2)

    def test_product_square(self):
        prod = ss_of("(&{a: end} || &{b: end})")
        single = ss_of("&{a: end}")
        # β(single) = -2, β(prod) = 4
        assert compute_beta(prod) == compute_beta(single) ** 2

    def test_product_with_chain(self):
        prod = ss_of("(&{a: &{b: end}} || &{c: end})")
        c3 = ss_of("&{a: &{b: end}}")
        c2 = ss_of("&{c: end}")
        assert compute_beta(prod) == compute_beta(c3) * compute_beta(c2)


class TestDecomposabilityDetector:

    def test_trivial_decomposable(self):
        assert analyze_beta(ss_of("end")).decomposable is True

    def test_single_method_not_decomposable(self):
        assert analyze_beta(ss_of("&{a: end}")).decomposable is False

    def test_long_chain_decomposable(self):
        # β(C_n)=0 for n ≥ 4 in our convention — detector flags it.
        assert analyze_beta(ss_of("&{a: &{b: &{c: end}}}")).decomposable is True

    def test_fan_branch(self):
        r = analyze_beta(ss_of("&{a: end, b: end, c: end}"))
        assert isinstance(r.beta, int)

    def test_selection_lattice(self):
        r = analyze_beta(ss_of("+{x: end, y: end}"))
        assert r.num_states == 2
        assert r.beta == -2


class TestContributingTerms:

    def test_terms_nonzero(self):
        ss = ss_of("&{a: end}")
        r = analyze_beta(ss)
        # All contributing terms should be non-zero
        for state, a, b, prod in r.contributing_terms:
            assert prod != 0
            assert a * b == prod

    def test_terms_sum_to_beta(self):
        ss = ss_of("(&{a: end} || &{b: end})")
        r = analyze_beta(ss)
        assert sum(t[3] for t in r.contributing_terms) == r.beta


class TestMobiusCrossCheck:

    def test_mu_0_1_recorded(self):
        ss = ss_of("&{a: end}")
        r = analyze_beta(ss)
        mu = mobius_function(ss)
        assert r.mobius_0_1 == mu.get((ss.top, ss.bottom), 0)

    def test_mu_endpoints_contribute(self):
        # The terms x = 0̂ and x = 1̂ contribute μ(0̂,1̂) each
        ss = ss_of("&{a: end}")
        mu = mobius_function(ss)
        mu01 = mu.get((ss.top, ss.bottom), 0)
        # term(bottom) = μ(0,0)·μ(0,1) = 1·mu01
        # term(top)    = μ(0,1)·μ(1,1) = mu01·1
        assert 2 * mu01 == compute_beta(ss)


class TestBidirectionalMorphism:

    def test_phi_trivial(self):
        assert phi_decomposable(ss_of("end")) == "trivial"

    def test_phi_connected(self):
        assert phi_decomposable(ss_of("&{a: end}")) == "connected"

    def test_phi_decomposable_chain(self):
        assert phi_decomposable(ss_of("&{a: &{b: &{c: end}}}")) == "decomposable"

    def test_psi_roundtrip_connected(self):
        desc, bound = psi_bound("connected")
        assert bound is None
        assert "irreducible" in desc

    def test_psi_roundtrip_decomposable(self):
        desc, bound = psi_bound("decomposable")
        assert bound == 0

    def test_psi_trivial(self):
        desc, bound = psi_bound("trivial")
        assert bound == 0

    def test_classify_pair_returns_galois(self):
        for t in ["end", "&{a: end}", "&{a: &{b: end}}"]:
            assert classify_pair(ss_of(t)) == "galois"

    def test_phi_psi_roundtrip_on_classes(self):
        # φ(L) yields a class; ψ of that class yields a witness; the witness
        # 'bound' must be consistent with β.
        for t in ["end", "&{a: end}", "&{a: &{b: &{c: end}}}",
                  "(&{a: end} || &{b: end})"]:
            ss = ss_of(t)
            c = phi_decomposable(ss)
            desc, bound = psi_bound(c)
            if bound == 0 and c == "decomposable":
                assert compute_beta(ss) == 0


class TestBenchmarkBatch:
    """Smoke test across a handful of benchmark-style protocols."""

    @pytest.mark.parametrize("type_string", [
        "end",
        "&{open: &{close: end}}",
        "&{login: +{ok: &{logout: end}, err: end}}",
        "(&{a: end} || &{b: end})",
        "rec X . &{next: X, done: end}",
        "&{a: end, b: &{c: end}}",
    ])
    def test_beta_computes(self, type_string):
        ss = ss_of(type_string)
        r = analyze_beta(ss)
        assert isinstance(r, BetaResult)
        assert isinstance(r.beta, int)
        # Decomposable flag must match β == 0 for non-trivial lattices
        if not r.trivial:
            assert r.decomposable == (r.beta == 0)
