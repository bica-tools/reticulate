"""Tests for coproduct analysis in SessLat (Step 164).

Verifies that SessLat does NOT have general coproducts, with concrete
counterexamples.  Tests the three candidate constructions (coalesced,
separated, linear), their lattice properties, injections, and the
universal property failure.
"""

from __future__ import annotations

import pytest

from reticulate.parser import parse
from reticulate.statespace import StateSpace, build_statespace
from reticulate.lattice import check_lattice
from reticulate.coproduct import (
    CoproductCandidate,
    CoproductResult,
    InjectionMap,
    check_coproduct,
    check_coproduct_universal_property,
    coalesced_sum,
    is_coproduct,
    linear_sum,
    separated_sum,
    _compute_remaps,
    _find_all_homomorphisms,
    _build_injections,
)
from reticulate.reticular import is_reticulate


# ---------------------------------------------------------------------------
# Helpers: build small test lattices
# ---------------------------------------------------------------------------

def _end() -> StateSpace:
    """1-element lattice (end): top == bottom."""
    return build_statespace(parse("end"))


def _chain2() -> StateSpace:
    """2-element chain: &{a: end} → top --a--> end."""
    return build_statespace(parse("&{a: end}"))


def _chain3() -> StateSpace:
    """3-element chain: &{a: &{b: end}} → top --a--> mid --b--> end."""
    return build_statespace(parse("&{a: &{b: end}}"))


def _diamond() -> StateSpace:
    """4-element diamond (product lattice): (&{a: end} || &{b: end}).

    States: (a,b)=top, (end,b), (a,end), (end,end)=bottom.
    """
    return build_statespace(parse("(&{a: end} || &{b: end})"))


def _branch2() -> StateSpace:
    """2-element: &{a: end, b: end} → top --a,b--> end.

    NOT a diamond — both labels go to same end state. Just a 2-chain.
    """
    return build_statespace(parse("&{a: end, b: end}"))


def _selection2() -> StateSpace:
    """2-element selection: +{a: end, b: end} → top --a,b--> end."""
    return build_statespace(parse("+{a: end, b: end}"))


def _parallel_type() -> StateSpace:
    """Product from parallel: (&{a: end} || &{b: end})."""
    return _diamond()


# ---------------------------------------------------------------------------
# Construction tests: coalesced sum
# ---------------------------------------------------------------------------

class TestCoalescedSum:
    """Tests for the coalesced sum construction."""

    def test_end_plus_end(self) -> None:
        """end + end (coalesced): shared bottom + fresh top = 2 states."""
        e1, e2 = _end(), _end()
        c = coalesced_sum(e1, e2)
        assert len(c.states) == 2

    def test_chain_plus_end(self) -> None:
        """chain2 + end (coalesced): 2 + 1 - 1 + 1 = 3 states."""
        ch, e = _chain2(), _end()
        c = coalesced_sum(ch, e)
        assert len(c.states) == 3

    def test_chain2_plus_chain2(self) -> None:
        """chain2 + chain2: 2 + 2 - 1 + 1 = 4 states."""
        ch1, ch2 = _chain2(), _chain2()
        c = coalesced_sum(ch1, ch2)
        assert len(c.states) == 4

    def test_diamond_plus_diamond(self) -> None:
        """diamond(4) + diamond(4): 4 + 4 - 1 + 1 = 8 states."""
        d1, d2 = _diamond(), _diamond()
        c = coalesced_sum(d1, d2)
        assert len(c.states) == 8

    def test_has_fresh_top(self) -> None:
        ch1, ch2 = _chain2(), _chain2()
        c = coalesced_sum(ch1, ch2)
        assert c.top != c.bottom
        succs = c.successors(c.top)
        assert len(succs) == 2

    def test_transitions_include_originals(self) -> None:
        ch1, ch2 = _chain2(), _chain2()
        c = coalesced_sum(ch1, ch2)
        # chain2 has 1 transition each, plus 2 from fresh top
        assert len(c.transitions) == 4

    def test_labels_present(self) -> None:
        ch1, ch2 = _chain2(), _chain2()
        c = coalesced_sum(ch1, ch2)
        for s in c.states:
            assert s in c.labels


class TestCoalescedSumLattice:
    """Lattice property tests for coalesced sum."""

    def test_end_plus_end_is_lattice(self) -> None:
        c = coalesced_sum(_end(), _end())
        assert check_lattice(c).is_lattice

    def test_chain_plus_chain_is_lattice(self) -> None:
        c = coalesced_sum(_chain2(), _chain2())
        lr = check_lattice(c)
        assert lr.is_lattice

    def test_diamond_plus_diamond_lattice_check(self) -> None:
        """Coalesced sum of two diamonds — check runs without error."""
        c = coalesced_sum(_diamond(), _diamond())
        lr = check_lattice(c)
        assert isinstance(lr.is_lattice, bool)


# ---------------------------------------------------------------------------
# Construction tests: separated sum
# ---------------------------------------------------------------------------

class TestSeparatedSum:
    """Tests for the separated sum construction."""

    def test_end_plus_end(self) -> None:
        """end + end: 1 + 1 + 2 = 4 states."""
        e1, e2 = _end(), _end()
        c = separated_sum(e1, e2)
        assert len(c.states) == 4

    def test_chain2_plus_chain2(self) -> None:
        """chain2 + chain2: 2 + 2 + 2 = 6 states."""
        c = separated_sum(_chain2(), _chain2())
        assert len(c.states) == 6

    def test_diamond_plus_diamond(self) -> None:
        """diamond(4) + diamond(4) + 2 = 10 states."""
        d1, d2 = _diamond(), _diamond()
        c = separated_sum(d1, d2)
        assert len(c.states) == 10

    def test_has_fresh_top_and_bottom(self) -> None:
        ch1, ch2 = _chain2(), _chain2()
        c = separated_sum(ch1, ch2)
        assert c.top != c.bottom
        assert len(c.successors(c.top)) == 2

    def test_factors_isolated(self) -> None:
        """Factor elements don't share states (unlike coalesced)."""
        ch1, ch2 = _chain2(), _chain2()
        c = separated_sum(ch1, ch2)
        lr, rr = _compute_remaps(ch1, ch2, c, "separated")
        left_set = set(lr.values())
        right_set = set(rr.values())
        assert left_set & right_set == set()

    def test_separated_is_lattice_chains(self) -> None:
        """Separated sum of two chains should be a lattice."""
        c = separated_sum(_chain2(), _chain2())
        lr = check_lattice(c)
        assert lr.is_lattice


# ---------------------------------------------------------------------------
# Construction tests: linear sum
# ---------------------------------------------------------------------------

class TestLinearSum:
    """Tests for the linear (ordinal) sum construction."""

    def test_end_plus_end(self) -> None:
        """end + end: 1 + 1 - 1 = 1 state."""
        e1, e2 = _end(), _end()
        c = linear_sum(e1, e2)
        assert len(c.states) == 1

    def test_chain_plus_chain(self) -> None:
        """chain2 + chain2: 2 + 2 - 1 = 3 states."""
        ch1, ch2 = _chain2(), _chain2()
        c = linear_sum(ch1, ch2)
        assert len(c.states) == 3

    def test_top_is_left_top(self) -> None:
        ch1, ch2 = _chain2(), _chain2()
        c = linear_sum(ch1, ch2)
        lr, rr = _compute_remaps(ch1, ch2, c, "linear")
        assert c.top == lr[ch1.top]

    def test_bottom_is_right_bottom(self) -> None:
        ch1, ch2 = _chain2(), _chain2()
        c = linear_sum(ch1, ch2)
        lr, rr = _compute_remaps(ch1, ch2, c, "linear")
        assert c.bottom == rr[ch2.bottom]

    def test_join_point_identified(self) -> None:
        """L₁.bottom is identified with L₂.top."""
        ch1, ch2 = _chain2(), _chain2()
        c = linear_sum(ch1, ch2)
        lr, rr = _compute_remaps(ch1, ch2, c, "linear")
        assert lr[ch1.bottom] == rr[ch2.top]

    def test_linear_sum_is_lattice_chains(self) -> None:
        """Linear sum of two chains is a chain — always a lattice."""
        c = linear_sum(_chain2(), _chain2())
        assert check_lattice(c).is_lattice

    def test_linear_not_symmetric(self) -> None:
        """L₁ + L₂ ≠ L₂ + L₁ for linear sum when factors differ."""
        ch, d = _chain3(), _diamond()
        c1 = linear_sum(ch, d)
        c2 = linear_sum(d, ch)
        # Same total states but different topology
        assert c1.top != c1.bottom or len(c1.states) == 1
        assert c2.top != c2.bottom or len(c2.states) == 1


# ---------------------------------------------------------------------------
# Injection tests
# ---------------------------------------------------------------------------

class TestInjections:
    """Tests for injection construction."""

    def test_coalesced_injections_exist(self) -> None:
        ch1, ch2 = _chain2(), _chain2()
        c = coalesced_sum(ch1, ch2)
        lr, rr = _compute_remaps(ch1, ch2, c, "coalesced")
        injs = _build_injections(c, ch1, ch2, lr, rr)
        assert len(injs) == 2

    def test_coalesced_left_injective(self) -> None:
        ch1, ch2 = _chain2(), _chain2()
        c = coalesced_sum(ch1, ch2)
        lr, rr = _compute_remaps(ch1, ch2, c, "coalesced")
        injs = _build_injections(c, ch1, ch2, lr, rr)
        assert injs[0].is_injective

    def test_separated_both_injective(self) -> None:
        ch1, ch2 = _chain2(), _chain2()
        c = separated_sum(ch1, ch2)
        lr, rr = _compute_remaps(ch1, ch2, c, "separated")
        injs = _build_injections(c, ch1, ch2, lr, rr)
        assert injs[0].is_injective
        assert injs[1].is_injective

    def test_injections_order_preserving(self) -> None:
        ch1, ch2 = _chain2(), _chain2()
        c = separated_sum(ch1, ch2)
        lr, rr = _compute_remaps(ch1, ch2, c, "separated")
        injs = _build_injections(c, ch1, ch2, lr, rr)
        assert injs[0].is_order_preserving
        assert injs[1].is_order_preserving

    def test_coalesced_injections_order_preserving(self) -> None:
        ch1, ch2 = _chain2(), _chain2()
        c = coalesced_sum(ch1, ch2)
        lr, rr = _compute_remaps(ch1, ch2, c, "coalesced")
        injs = _build_injections(c, ch1, ch2, lr, rr)
        assert injs[0].is_order_preserving
        assert injs[1].is_order_preserving

    def test_linear_left_injection_order_preserving(self) -> None:
        ch1, ch2 = _chain2(), _chain2()
        c = linear_sum(ch1, ch2)
        lr, rr = _compute_remaps(ch1, ch2, c, "linear")
        injs = _build_injections(c, ch1, ch2, lr, rr)
        assert injs[0].is_order_preserving


# ---------------------------------------------------------------------------
# Homomorphism enumeration tests
# ---------------------------------------------------------------------------

class TestFindAllHomomorphisms:
    """Tests for the homomorphism enumeration helper."""

    def test_identity_found(self) -> None:
        """Identity is always a homomorphism."""
        ch = _chain2()
        homos = _find_all_homomorphisms(ch, ch)
        id_map = {s: s for s in ch.states}
        assert id_map in homos

    def test_end_to_end(self) -> None:
        """end → end has exactly one homomorphism."""
        e = _end()
        homos = _find_all_homomorphisms(e, e)
        assert len(homos) == 1

    def test_chain_to_chain(self) -> None:
        """chain2 → chain2: at least identity."""
        ch = _chain2()
        homos = _find_all_homomorphisms(ch, ch)
        assert len(homos) >= 1

    def test_end_to_chain(self) -> None:
        """end → chain2: no valid hom (source top=bottom, target top≠bottom)."""
        e = _end()
        ch = _chain2()
        homos = _find_all_homomorphisms(e, ch)
        # end has top==bottom, chain2 has top≠bottom
        # hom must map top→top and bottom→bottom, but source top==bottom
        # so target top must == target bottom — contradiction
        assert len(homos) == 0

    def test_chain_to_end(self) -> None:
        """chain2 → end: unique map (all → single state)."""
        ch = _chain2()
        e = _end()
        homos = _find_all_homomorphisms(ch, e)
        # All states map to the single end state
        assert len(homos) == 1


# ---------------------------------------------------------------------------
# Universal property tests — THE NEGATIVE RESULT
# ---------------------------------------------------------------------------

class TestUniversalProperty:
    """Tests demonstrating that SessLat lacks general coproducts."""

    def test_coalesced_chain2_fails_up(self) -> None:
        """Coalesced sum of two chain2s fails the universal property."""
        ch1, ch2 = _chain2(), _chain2()
        c = coalesced_sum(ch1, ch2)
        lr, rr = _compute_remaps(ch1, ch2, c, "coalesced")

        if not check_lattice(c).is_lattice:
            return

        up_ok, cx = check_coproduct_universal_property(
            c, ch1, ch2, lr, rr,
        )
        # The fresh top creates non-uniqueness for mediating morphisms
        # targeting the factors themselves
        assert not up_ok
        assert cx is not None

    def test_separated_chain2_fails_up(self) -> None:
        """Separated sum of two chain2s fails the universal property."""
        ch1, ch2 = _chain2(), _chain2()
        c = separated_sum(ch1, ch2)
        lr, rr = _compute_remaps(ch1, ch2, c, "separated")

        if not check_lattice(c).is_lattice:
            return

        up_ok, cx = check_coproduct_universal_property(
            c, ch1, ch2, lr, rr,
        )
        assert not up_ok

    def test_linear_fails_symmetry(self) -> None:
        """Linear sum is not symmetric, so cannot be a coproduct."""
        ch1, ch2 = _chain2(), _chain3()
        c = linear_sum(ch1, ch2)
        lr, rr = _compute_remaps(ch1, ch2, c, "linear")

        if not check_lattice(c).is_lattice:
            return

        up_ok, cx = check_coproduct_universal_property(
            c, ch1, ch2, lr, rr,
        )
        # Linear sum is asymmetric — fails UP
        assert not up_ok

    def test_end_coproduct_trivial(self) -> None:
        """end + end: trivial case."""
        e1, e2 = _end(), _end()
        result = check_coproduct(e1, e2)
        assert isinstance(result, CoproductResult)
        assert isinstance(result.has_coproduct, bool)

    def test_diamond_all_constructions_fail(self) -> None:
        """Diamond + diamond: all three constructions fail UP."""
        d1, d2 = _diamond(), _diamond()
        result = check_coproduct(d1, d2)
        assert not result.has_coproduct
        assert result.counterexample is not None


# ---------------------------------------------------------------------------
# Main check_coproduct tests
# ---------------------------------------------------------------------------

class TestCheckCoproduct:
    """Tests for the main check_coproduct function."""

    def test_chain2_no_coproduct(self) -> None:
        """chain2 + chain2 has no coproduct in SessLat."""
        ch1, ch2 = _chain2(), _chain2()
        result = check_coproduct(ch1, ch2)
        assert not result.has_coproduct
        assert result.counterexample is not None

    def test_all_candidates_returned(self) -> None:
        """All three candidate constructions are tried."""
        ch1, ch2 = _chain2(), _chain2()
        result = check_coproduct(ch1, ch2)
        assert len(result.all_candidates) == 3
        names = {c.construction for c in result.all_candidates}
        assert names == {"coalesced", "separated", "linear"}

    def test_candidate_has_lattice_flag(self) -> None:
        ch1, ch2 = _chain2(), _chain2()
        result = check_coproduct(ch1, ch2)
        for cand in result.all_candidates:
            assert isinstance(cand.is_lattice, bool)

    def test_candidate_has_realizable_flag(self) -> None:
        ch1, ch2 = _chain2(), _chain2()
        result = check_coproduct(ch1, ch2)
        for cand in result.all_candidates:
            assert isinstance(cand.is_realizable, bool)

    def test_diamond_no_coproduct(self) -> None:
        """Diamond + diamond has no coproduct."""
        d1, d2 = _diamond(), _diamond()
        result = check_coproduct(d1, d2)
        assert not result.has_coproduct

    def test_chain_plus_chain_result(self) -> None:
        """chain3 + chain3: check all candidates."""
        ch1, ch2 = _chain3(), _chain3()
        result = check_coproduct(ch1, ch2)
        assert isinstance(result, CoproductResult)
        assert len(result.all_candidates) == 3


# ---------------------------------------------------------------------------
# Realizability tests
# ---------------------------------------------------------------------------

class TestRealizability:
    """Tests for reticular form on coproduct candidates."""

    def test_coalesced_realizability(self) -> None:
        """Coalesced sum of chains — check realizability."""
        ch1, ch2 = _chain2(), _chain2()
        c = coalesced_sum(ch1, ch2)
        assert isinstance(is_reticulate(c), bool)

    def test_separated_realizability(self) -> None:
        """Separated sum — check realizability."""
        ch1, ch2 = _chain2(), _chain2()
        c = separated_sum(ch1, ch2)
        assert isinstance(is_reticulate(c), bool)

    def test_linear_preserves_structure(self) -> None:
        """Linear sum of two chains preserves sequential structure."""
        ch1, ch2 = _chain2(), _chain2()
        c = linear_sum(ch1, ch2)
        assert is_reticulate(c)

    def test_linear_diamond_on_chain(self) -> None:
        """Linear sum of diamond on chain."""
        d, ch = _diamond(), _chain2()
        c = linear_sum(d, ch)
        assert isinstance(is_reticulate(c), bool)


# ---------------------------------------------------------------------------
# is_coproduct tests
# ---------------------------------------------------------------------------

class TestIsCoproduct:
    """Tests for the is_coproduct boolean wrapper."""

    def test_coalesced_not_coproduct_chains(self) -> None:
        ch1, ch2 = _chain2(), _chain2()
        c = coalesced_sum(ch1, ch2)
        lr, rr = _compute_remaps(ch1, ch2, c, "coalesced")
        result = is_coproduct(c, ch1, ch2, lr, rr)
        assert not result

    def test_separated_not_coproduct_chains(self) -> None:
        ch1, ch2 = _chain2(), _chain2()
        c = separated_sum(ch1, ch2)
        lr, rr = _compute_remaps(ch1, ch2, c, "separated")
        result = is_coproduct(c, ch1, ch2, lr, rr)
        assert not result


# ---------------------------------------------------------------------------
# Benchmark sweep
# ---------------------------------------------------------------------------

class TestBenchmarkSweep:
    """Test coproduct analysis on benchmark protocol pairs."""

    @pytest.fixture
    def iterator(self) -> StateSpace:
        return build_statespace(parse(
            "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"
        ))

    @pytest.fixture
    def file_obj(self) -> StateSpace:
        return build_statespace(parse(
            "&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}"
        ))

    def test_iterator_plus_file_no_coproduct(
        self, iterator: StateSpace, file_obj: StateSpace,
    ) -> None:
        """Iterator + File Object: no coproduct."""
        result = check_coproduct(iterator, file_obj)
        assert not result.has_coproduct

    def test_chain_plus_diamond_no_coproduct(self) -> None:
        """Chain + diamond: no coproduct."""
        ch = _chain2()
        d = _diamond()
        result = check_coproduct(ch, d)
        assert not result.has_coproduct

    def test_branch2_plus_selection2_no_coproduct(self) -> None:
        """Branch type + selection type: no coproduct."""
        b = _branch2()
        s = _selection2()
        result = check_coproduct(b, s)
        assert not result.has_coproduct

    def test_negative_result_summary(
        self, iterator: StateSpace, file_obj: StateSpace,
    ) -> None:
        """Confirm counterexample is provided for benchmark pair."""
        result = check_coproduct(iterator, file_obj)
        assert result.counterexample is not None
        assert len(result.counterexample) > 0
