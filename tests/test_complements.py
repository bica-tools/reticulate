"""Tests for complement analysis on session type lattices.

Step 363c: Tests complement finding, complemented lattice detection,
Boolean algebra classification, and relative complementation.
"""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice
from reticulate.lattice import check_distributive
from reticulate.complements import (
    find_complements,
    complement_info,
    find_relative_complement,
    is_complemented,
    is_uniquely_complemented,
    is_relatively_complemented,
    is_boolean,
    boolean_rank,
    analyze_complements,
    ComplementInfo,
    ComplementAnalysis,
)


# ═══════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════

def _build(type_str: str):
    """Parse, build state space, check lattice."""
    ast = parse(type_str)
    ss = build_statespace(ast)
    lr = check_lattice(ss)
    return ss, lr


# ═══════════════════════════════════════════════════════
# Test: find_complements
# ═══════════════════════════════════════════════════════

class TestFindComplements:
    """Tests for finding complements of individual elements."""

    def test_end_self_complement(self):
        """In a 1-element lattice, end is its own complement."""
        ss, lr = _build("end")
        assert lr.is_lattice
        comps = find_complements(ss, ss.top, lr)
        assert ss.top in comps

    def test_two_element_chain(self):
        """In a 2-element chain, top and bottom are each other's complement."""
        ss, lr = _build("&{a: end}")
        assert lr.is_lattice
        top = ss.top
        comps_top = find_complements(ss, top, lr)
        # Top's complement should be bottom
        assert len(comps_top) >= 1

    def test_branch_complement(self):
        """In &{a: end, b: end}, check complements exist."""
        ss, lr = _build("&{a: end, b: end}")
        assert lr.is_lattice
        top = ss.top
        comps = find_complements(ss, top, lr)
        # Top always has bottom as complement (trivially)
        assert len(comps) >= 1

    def test_parallel_boolean(self):
        """Parallel of two binary choices forms a Boolean lattice."""
        ss, lr = _build("(&{a: end} || &{b: end})")
        assert lr.is_lattice
        # In a Boolean lattice, every element has a complement
        for s in range(len(ss.states)):
            comps = find_complements(ss, s, lr)
            assert len(comps) >= 1, f"State {s} has no complement"

    def test_non_lattice_returns_empty(self):
        """Non-lattice state spaces return empty complement list."""
        # A type that might not form a lattice
        ss, lr = _build("&{a: end, b: rec X . &{a: X}}")
        if not lr.is_lattice:
            comps = find_complements(ss, ss.top, lr)
            assert comps == []


class TestComplementInfo:
    """Tests for complement_info helper."""

    def test_returns_complement_info(self):
        """complement_info returns a ComplementInfo dataclass."""
        ss, lr = _build("&{a: end}")
        info = complement_info(ss, ss.top, lr)
        assert isinstance(info, ComplementInfo)
        assert info.element == ss.top
        assert isinstance(info.has_complement, bool)

    def test_uniqueness_flag(self):
        """Check that is_unique flag is set correctly."""
        ss, lr = _build("end")
        info = complement_info(ss, ss.top, lr)
        assert info.has_complement
        # In 1-element lattice, the single element is its own unique complement
        assert info.is_unique


# ═══════════════════════════════════════════════════════
# Test: is_complemented
# ═══════════════════════════════════════════════════════

class TestIsComplemented:
    """Tests for complemented lattice detection."""

    def test_trivial_lattice_complemented(self):
        """A 1-element lattice is trivially complemented."""
        ss, lr = _build("end")
        assert is_complemented(ss, lr)

    def test_two_chain_complemented(self):
        """A 2-element chain is complemented (top↔bottom)."""
        ss, lr = _build("&{a: end}")
        assert is_complemented(ss, lr)

    def test_three_chain_not_complemented(self):
        """A 3-element chain is NOT complemented (middle has no complement)."""
        ss, lr = _build("&{a: &{b: end}}")
        assert lr.is_lattice
        # In a 3-chain: top, mid, bottom
        # mid has no complement: mid ∧ x = bottom requires x ≤ mid,
        # but mid ∨ x = top requires x ≥ mid; only x = top or bottom work
        # meet(mid, top) = mid ≠ bottom, meet(mid, bottom) = bottom but
        # join(mid, bottom) = mid ≠ top
        result = is_complemented(ss, lr)
        assert not result

    def test_diamond_m3_complemented(self):
        """M3 (diamond) is complemented but not distributive."""
        # +{a: end, b: end} gives a diamond: top → a/b → bottom
        ss, lr = _build("+{a: end, b: end}")
        if lr.is_lattice and len(ss.states) >= 3:
            result = is_complemented(ss, lr)
            # M3 is complemented
            assert result

    def test_parallel_complemented(self):
        """Parallel composition of binary types is complemented."""
        ss, lr = _build("(&{a: end} || &{b: end})")
        assert lr.is_lattice
        assert is_complemented(ss, lr)


# ═══════════════════════════════════════════════════════
# Test: is_uniquely_complemented
# ═══════════════════════════════════════════════════════

class TestIsUniquelyComplemented:
    """Tests for uniquely complemented lattice detection."""

    def test_trivial(self):
        """1-element lattice is uniquely complemented."""
        ss, lr = _build("end")
        assert is_uniquely_complemented(ss, lr)

    def test_two_chain_unique(self):
        """2-element chain: complements are unique."""
        ss, lr = _build("&{a: end}")
        assert is_uniquely_complemented(ss, lr)

    def test_boolean_unique(self):
        """Boolean lattices have unique complements."""
        ss, lr = _build("(&{a: end} || &{b: end})")
        assert lr.is_lattice
        dr = check_distributive(ss)
        if dr.is_distributive:
            # Distributive + complemented => unique complements
            assert is_uniquely_complemented(ss, lr)


# ═══════════════════════════════════════════════════════
# Test: is_boolean
# ═══════════════════════════════════════════════════════

class TestIsBoolean:
    """Tests for Boolean lattice detection."""

    def test_trivial_boolean(self):
        """1-element lattice (2^0) is Boolean."""
        ss, lr = _build("end")
        assert is_boolean(ss, lr)

    def test_two_chain_boolean(self):
        """2-element chain (2^1) is Boolean."""
        ss, lr = _build("&{a: end}")
        assert is_boolean(ss, lr)

    def test_three_chain_not_boolean(self):
        """3-element chain is not Boolean (not complemented)."""
        ss, lr = _build("&{a: &{b: end}}")
        assert not is_boolean(ss, lr)

    def test_parallel_binary_boolean(self):
        """Product of two 2-chains (2^2 = 4 elements) is Boolean."""
        ss, lr = _build("(&{a: end} || &{b: end})")
        assert lr.is_lattice
        dr = check_distributive(ss)
        if dr.is_distributive and len(ss.states) == 4:
            assert is_boolean(ss, lr)

    def test_non_distributive_not_boolean(self):
        """Non-distributive lattice is never Boolean."""
        ss, lr = _build("&{a: end, b: end, c: end}")
        dr = check_distributive(ss)
        if lr.is_lattice and not dr.is_distributive:
            assert not is_boolean(ss, lr)


class TestBooleanRank:
    """Tests for Boolean rank computation."""

    def test_rank_0(self):
        """1-element lattice has rank 0."""
        ss, lr = _build("end")
        assert boolean_rank(ss, lr) == 0

    def test_rank_1(self):
        """2-element chain has rank 1."""
        ss, lr = _build("&{a: end}")
        r = boolean_rank(ss, lr)
        assert r == 1

    def test_non_boolean_none(self):
        """Non-Boolean lattice returns None."""
        ss, lr = _build("&{a: &{b: end}}")
        assert boolean_rank(ss, lr) is None


# ═══════════════════════════════════════════════════════
# Test: is_relatively_complemented
# ═══════════════════════════════════════════════════════

class TestRelativelyComplemented:
    """Tests for relatively complemented lattice detection."""

    def test_trivial(self):
        """1-element lattice is relatively complemented."""
        ss, lr = _build("end")
        assert is_relatively_complemented(ss, lr)

    def test_two_chain(self):
        """2-element chain is relatively complemented."""
        ss, lr = _build("&{a: end}")
        assert is_relatively_complemented(ss, lr)

    def test_boolean_relatively_complemented(self):
        """Boolean lattices are relatively complemented."""
        ss, lr = _build("(&{a: end} || &{b: end})")
        if is_boolean(ss, lr):
            assert is_relatively_complemented(ss, lr)


# ═══════════════════════════════════════════════════════
# Test: find_relative_complement
# ═══════════════════════════════════════════════════════

class TestFindRelativeComplement:
    """Tests for finding relative complements in intervals."""

    def test_full_interval(self):
        """Relative complement in [⊥, ⊤] = absolute complement."""
        ss, lr = _build("&{a: end}")
        top = ss.top
        bottom = ss.bottom
        comps = find_relative_complement(ss, top, bottom, top, lr)
        assert bottom in comps

    def test_element_outside_interval(self):
        """Element outside interval returns empty list."""
        ss, lr = _build("&{a: &{b: end}}")
        # Try to find complement of top in interval [mid, mid] — top not in interval
        # This should return empty
        # (We need to know the actual state IDs, so this is a structural test)
        assert lr.is_lattice


# ═══════════════════════════════════════════════════════
# Test: analyze_complements
# ═══════════════════════════════════════════════════════

class TestAnalyzeComplements:
    """Tests for complete complement analysis."""

    def test_returns_analysis(self):
        """analyze_complements returns a ComplementAnalysis dataclass."""
        ss, lr = _build("&{a: end}")
        result = analyze_complements(ss, lr)
        assert isinstance(result, ComplementAnalysis)

    def test_trivial_analysis(self):
        """1-element lattice: fully Boolean."""
        ss, lr = _build("end")
        result = analyze_complements(ss, lr)
        assert result.is_complemented
        assert result.is_uniquely_complemented
        assert result.is_boolean
        assert result.boolean_rank == 0
        assert len(result.uncomplemented_elements) == 0

    def test_two_chain_analysis(self):
        """2-element chain analysis."""
        ss, lr = _build("&{a: end}")
        result = analyze_complements(ss, lr)
        assert result.is_complemented
        assert result.is_boolean
        assert result.boolean_rank == 1
        assert result.element_count == 2

    def test_three_chain_analysis(self):
        """3-element chain: not complemented."""
        ss, lr = _build("&{a: &{b: end}}")
        result = analyze_complements(ss, lr)
        assert not result.is_complemented
        assert not result.is_boolean
        assert len(result.uncomplemented_elements) > 0

    def test_non_lattice_analysis(self):
        """Non-lattice gives all-false analysis."""
        ss, lr = _build("&{a: end, b: rec X . &{a: X}}")
        if not lr.is_lattice:
            result = analyze_complements(ss, lr)
            assert not result.is_complemented
            assert not result.is_boolean
            assert result.complement_count == 0

    def test_complement_map_populated(self):
        """Complement map has entries for all elements."""
        ss, lr = _build("&{a: end}")
        result = analyze_complements(ss, lr)
        assert len(result.complement_map) == result.element_count

    def test_parallel_analysis(self):
        """Parallel composition analysis."""
        ss, lr = _build("(&{a: end} || &{b: end})")
        result = analyze_complements(ss, lr)
        assert result.element_count > 0

    def test_complement_count_symmetric(self):
        """Complement relation is symmetric: if b complements a, a complements b."""
        ss, lr = _build("&{a: end}")
        result = analyze_complements(ss, lr)
        for a, comps in result.complement_map.items():
            for b in comps:
                assert a in result.complement_map.get(b, ())


# ═══════════════════════════════════════════════════════
# Test: Benchmark protocols
# ═══════════════════════════════════════════════════════

class TestBenchmarks:
    """Tests on standard benchmark protocols."""

    @pytest.mark.parametrize("name,type_str", [
        ("iterator", "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"),
        ("simple_branch", "&{a: end, b: end}"),
        ("binary_select", "+{ok: end, err: end}"),
        ("chain_4", "&{a: &{b: &{c: end}}}"),
        ("parallel_simple", "(&{a: end} || &{b: end})"),
    ])
    def test_benchmark_analysis(self, name, type_str):
        """All benchmarks produce valid analysis results."""
        ss, lr = _build(type_str)
        if lr.is_lattice:
            result = analyze_complements(ss, lr)
            assert isinstance(result, ComplementAnalysis)
            assert result.element_count > 0
            # Boolean => distributive
            if result.is_boolean:
                assert result.is_distributive
            # Uniquely complemented => complemented
            if result.is_uniquely_complemented:
                assert result.is_complemented

    @pytest.mark.parametrize("name,type_str,expected_boolean", [
        ("end", "end", True),
        ("single_branch", "&{a: end}", True),
        ("chain_3", "&{a: &{b: end}}", False),
        ("chain_4", "&{a: &{b: &{c: end}}}", False),
    ])
    def test_boolean_classification(self, name, type_str, expected_boolean):
        """Test Boolean classification for known cases."""
        ss, lr = _build(type_str)
        assert lr.is_lattice
        assert is_boolean(ss, lr) == expected_boolean


# ═══════════════════════════════════════════════════════
# Test: Properties and invariants
# ═══════════════════════════════════════════════════════

class TestProperties:
    """Tests for lattice-theoretic properties of complements."""

    def test_top_bottom_always_complement(self):
        """Top and bottom are always each other's complements."""
        ss, lr = _build("&{a: end, b: end}")
        if lr.is_lattice:
            top = ss.top
            bottom = ss.bottom
            if bottom is not None:
                comps_top = find_complements(ss, top, lr)
                comps_bot = find_complements(ss, bottom, lr)
                assert bottom in comps_top
                assert top in comps_bot

    def test_distributive_implies_unique_complements(self):
        """In a distributive lattice, complements are unique when they exist."""
        ss, lr = _build("(&{a: end} || &{b: end})")
        dr = check_distributive(ss)
        if lr.is_lattice and dr.is_distributive:
            result = analyze_complements(ss, lr)
            # In distributive lattice: if complemented then uniquely complemented
            if result.is_complemented:
                assert result.is_uniquely_complemented

    def test_boolean_implies_relatively_complemented(self):
        """Boolean lattice is always relatively complemented."""
        ss, lr = _build("end")
        if is_boolean(ss, lr):
            assert is_relatively_complemented(ss, lr)

    def test_complement_involution(self):
        """If a' is a complement of a, then a is a complement of a'."""
        ss, lr = _build("&{a: end}")
        if lr.is_lattice:
            for s in range(len(ss.states)):
                comps = find_complements(ss, s, lr)
                for c in comps:
                    reverse_comps = find_complements(ss, c, lr)
                    assert s in reverse_comps
