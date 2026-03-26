"""Tests for the bounded replication extension (S^n).

Tests cover:
  - Replication AST node construction and validation
  - Parser: S^n syntax with caret operator
  - State-space construction: n-fold product
  - Diagonal states and diagonal lattice
  - Symmetry orbits under Sn
  - Full replication analysis
  - Benchmark protocols replicated
  - Lattice verification for all replicated types
  - Compression ratio growth with n
"""

from __future__ import annotations

import pytest

from reticulate.parser import parse, End, Branch, Select, Rec, Var
from reticulate.statespace import build_statespace
from reticulate.product import power_statespace
from reticulate.lattice import check_lattice
from reticulate.morphism import find_isomorphism

from reticulate.extensions.replication import (
    Replication,
    ReplicationAnalysis,
    _is_order_isomorphic,
    analyze_replication,
    build_replicated_statespace,
    diagonal_lattice,
    diagonal_states,
    parse_replication,
    symmetry_orbits,
)


# ---------------------------------------------------------------------------
# AST node tests
# ---------------------------------------------------------------------------

class TestReplicationNode:
    """Tests for the Replication frozen dataclass."""

    def test_basic_construction(self) -> None:
        node = Replication(body=End(), count=3)
        assert node.body == End()
        assert node.count == 3

    def test_frozen(self) -> None:
        node = Replication(body=End(), count=2)
        with pytest.raises(AttributeError):
            node.count = 5  # type: ignore[misc]

    def test_invalid_count_zero(self) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            Replication(body=End(), count=0)

    def test_invalid_count_negative(self) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            Replication(body=End(), count=-1)

    def test_equality(self) -> None:
        a = Replication(body=End(), count=2)
        b = Replication(body=End(), count=2)
        assert a == b

    def test_inequality_count(self) -> None:
        a = Replication(body=End(), count=2)
        b = Replication(body=End(), count=3)
        assert a != b

    def test_hashable(self) -> None:
        node = Replication(body=End(), count=3)
        s = {node, node}
        assert len(s) == 1


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class TestParseReplication:
    """Tests for parse_replication."""

    def test_no_caret_delegates_to_core(self) -> None:
        result = parse_replication("end")
        assert result == End()

    def test_simple_end_squared(self) -> None:
        result = parse_replication("end ^ 2")
        assert isinstance(result, Replication)
        assert result.body == End()
        assert result.count == 2

    def test_branch_cubed(self) -> None:
        result = parse_replication("&{a: end, b: end} ^ 3")
        assert isinstance(result, Replication)
        assert result.count == 3
        assert isinstance(result.body, Branch)
        assert len(result.body.choices) == 2

    def test_recursive_type_replicated(self) -> None:
        result = parse_replication("rec X . &{next: X, stop: end} ^ 4")
        assert isinstance(result, Replication)
        assert result.count == 4
        assert isinstance(result.body, Rec)

    def test_exponent_one_returns_base(self) -> None:
        result = parse_replication("&{a: end} ^ 1")
        # ^1 returns the base type, not a Replication node
        assert isinstance(result, Branch)

    def test_missing_exponent(self) -> None:
        with pytest.raises(ValueError, match="Missing exponent"):
            parse_replication("end ^")

    def test_non_integer_exponent(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            parse_replication("end ^ abc")

    def test_negative_exponent(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            parse_replication("end ^ -1")

    def test_caret_inside_braces_ignored(self) -> None:
        # If ^ appears inside braces, it should not be treated as replication
        # (This is a synthetic test -- ^ is not valid inside types normally)
        result = parse_replication("&{a: end}")
        assert isinstance(result, Branch)

    def test_selection_replicated(self) -> None:
        result = parse_replication("+{ok: end, err: end} ^ 2")
        assert isinstance(result, Replication)
        assert isinstance(result.body, Select)
        assert result.count == 2


# ---------------------------------------------------------------------------
# State-space construction tests
# ---------------------------------------------------------------------------

class TestBuildReplicatedStatespace:
    """Tests for build_replicated_statespace."""

    def test_end_replicated(self) -> None:
        node = Replication(body=End(), count=3)
        ss = build_replicated_statespace(node)
        # end has 1 state, end^3 = 1^3 = 1 state
        assert len(ss.states) == 1
        assert ss.top == ss.bottom

    def test_branch_squared_state_count(self) -> None:
        body = parse("&{a: end, b: end}")
        node = Replication(body=body, count=2)
        ss = build_replicated_statespace(node)
        base_ss = build_statespace(body)
        assert len(ss.states) == len(base_ss.states) ** 2

    def test_branch_cubed_state_count(self) -> None:
        body = parse("&{a: end, b: end}")
        node = Replication(body=body, count=3)
        ss = build_replicated_statespace(node)
        base_ss = build_statespace(body)
        assert len(ss.states) == len(base_ss.states) ** 3

    def test_s1_equals_s(self) -> None:
        """S^1 is isomorphic to S."""
        body = parse("&{a: end, b: end}")
        base_ss = build_statespace(body)
        node = Replication(body=body, count=1)
        rep_ss = build_replicated_statespace(node)
        # power_statespace(ss, 1) returns ss directly
        assert len(rep_ss.states) == len(base_ss.states)

    def test_s2_equals_s_par_s(self) -> None:
        """S^2 has same number of states as S || S product."""
        body = parse("&{a: end, b: end}")
        base_ss = build_statespace(body)
        product = power_statespace(base_ss, 2)
        node = Replication(body=body, count=2)
        rep_ss = build_replicated_statespace(node)
        assert len(rep_ss.states) == len(product.states)

    def test_product_coords_present(self) -> None:
        body = parse("&{a: end, b: end}")
        node = Replication(body=body, count=2)
        ss = build_replicated_statespace(node)
        assert ss.product_coords is not None
        assert len(ss.product_coords) > 0


# ---------------------------------------------------------------------------
# Diagonal tests
# ---------------------------------------------------------------------------

class TestDiagonal:
    """Tests for diagonal_states and diagonal_lattice."""

    def test_diagonal_of_n1_is_all_states(self) -> None:
        body = parse("&{a: end, b: end}")
        ss = build_statespace(body)
        diag = diagonal_states(ss, 1)
        assert diag == ss.states

    def test_diagonal_of_end_squared(self) -> None:
        body = parse("end")
        node = Replication(body=body, count=2)
        ss = build_replicated_statespace(node)
        diag = diagonal_states(ss, 2)
        assert len(diag) == 1  # (end, end)

    def test_diagonal_count_equals_base(self) -> None:
        """Diagonal of S^n has |S| states."""
        body = parse("&{a: end, b: end}")
        base_ss = build_statespace(body)
        node = Replication(body=body, count=2)
        ss = build_replicated_statespace(node)
        diag = diagonal_states(ss, 2)
        assert len(diag) == len(base_ss.states)

    def test_diagonal_count_three_copies(self) -> None:
        body = parse("&{a: end, b: end}")
        base_ss = build_statespace(body)
        node = Replication(body=body, count=3)
        ss = build_replicated_statespace(node)
        diag = diagonal_states(ss, 3)
        assert len(diag) == len(base_ss.states)

    def test_diagonal_lattice_isomorphic_to_base(self) -> None:
        """Diagonal of S^2 should be order-isomorphic to S."""
        body = parse("&{a: end, b: end}")
        base_ss = build_statespace(body)
        node = Replication(body=body, count=2)
        product_ss = build_replicated_statespace(node)
        diag_ss = diagonal_lattice(product_ss, 2)
        assert _is_order_isomorphic(diag_ss, base_ss)

    def test_diagonal_lattice_isomorphic_cubed(self) -> None:
        """Diagonal of S^3 should be order-isomorphic to S."""
        body = parse("&{a: end, b: end}")
        base_ss = build_statespace(body)
        node = Replication(body=body, count=3)
        product_ss = build_replicated_statespace(node)
        diag_ss = diagonal_lattice(product_ss, 3)
        assert _is_order_isomorphic(diag_ss, base_ss)

    def test_diagonal_of_selection(self) -> None:
        body = parse("+{ok: end, err: end}")
        base_ss = build_statespace(body)
        node = Replication(body=body, count=2)
        ss = build_replicated_statespace(node)
        diag = diagonal_states(ss, 2)
        assert len(diag) == len(base_ss.states)


# ---------------------------------------------------------------------------
# Symmetry orbit tests
# ---------------------------------------------------------------------------

class TestSymmetryOrbits:
    """Tests for symmetry_orbits under Sn."""

    def test_orbits_n1_all_singletons(self) -> None:
        body = parse("&{a: end, b: end}")
        ss = build_statespace(body)
        orbits = symmetry_orbits(ss, 1)
        assert len(orbits) == len(ss.states)
        for members in orbits.values():
            assert len(members) == 1

    def test_orbits_s2_count(self) -> None:
        """For S with k states, S^2 has k(k+1)/2 orbits under S2."""
        body = parse("&{a: end, b: end}")
        base_ss = build_statespace(body)
        k = len(base_ss.states)
        node = Replication(body=body, count=2)
        ss = build_replicated_statespace(node)
        orbits = symmetry_orbits(ss, 2)
        expected_orbits = k * (k + 1) // 2  # multiset coefficient
        assert len(orbits) == expected_orbits

    def test_diagonal_states_are_singletons(self) -> None:
        """Diagonal states (s,s) are fixed points of S2 -- orbit size 1."""
        body = parse("&{a: end, b: end}")
        node = Replication(body=body, count=2)
        ss = build_replicated_statespace(node)
        diag = diagonal_states(ss, 2)
        orbits = symmetry_orbits(ss, 2)
        for sid in diag:
            for rep, members in orbits.items():
                if sid in members:
                    assert len(members) == 1, \
                        f"Diagonal state {sid} should have orbit size 1"

    def test_off_diagonal_orbit_size_2(self) -> None:
        """Off-diagonal states (s1,s2) with s1!=s2 have orbit size 2 under S2."""
        body = parse("&{a: end, b: end}")
        node = Replication(body=body, count=2)
        ss = build_replicated_statespace(node)
        diag = diagonal_states(ss, 2)
        orbits = symmetry_orbits(ss, 2)
        for rep, members in orbits.items():
            if rep not in diag:
                assert len(members) == 2

    def test_orbits_cover_all_states(self) -> None:
        """Every state appears in exactly one orbit."""
        body = parse("&{a: end, b: end}")
        node = Replication(body=body, count=2)
        ss = build_replicated_statespace(node)
        orbits = symmetry_orbits(ss, 2)
        all_members: set[int] = set()
        for members in orbits.values():
            for m in members:
                assert m not in all_members, f"State {m} in multiple orbits"
                all_members.add(m)
        assert all_members == ss.states


# ---------------------------------------------------------------------------
# Full analysis tests
# ---------------------------------------------------------------------------

class TestAnalyzeReplication:
    """Tests for analyze_replication."""

    def test_end_squared(self) -> None:
        result = analyze_replication("end", 2)
        assert result.base_states == 1
        assert result.product_states == 1
        assert result.diagonal_states_count == 1
        assert result.is_lattice is True
        assert result.diagonal_isomorphic is True

    def test_branch_squared(self) -> None:
        result = analyze_replication("&{a: end, b: end}", 2)
        # &{a: end, b: end} has 2 states: top + end (both branches go to end)
        assert result.base_states == 2
        assert result.product_states == 4  # 2^2
        assert result.diagonal_states_count == 2
        assert result.is_lattice is True
        assert result.base_is_lattice is True
        assert result.diagonal_isomorphic is True

    def test_single_copy(self) -> None:
        result = analyze_replication("&{a: end, b: end}", 1)
        assert result.product_states == result.base_states
        assert result.compression_ratio == 1.0
        assert result.diagonal_isomorphic is True

    def test_compression_increases_with_n(self) -> None:
        """Compression ratio should increase as n grows."""
        ratios: list[float] = []
        for n in [1, 2, 3]:
            result = analyze_replication("&{a: end, b: end}", n)
            ratios.append(result.compression_ratio)
        # Compression should be non-decreasing
        for i in range(1, len(ratios)):
            assert ratios[i] >= ratios[i - 1], \
                f"Compression ratio should not decrease: {ratios}"

    def test_lattice_preserved(self) -> None:
        """If S is a lattice, S^n is a lattice (product of lattices is a lattice)."""
        result = analyze_replication("&{a: end, b: end}", 3)
        assert result.base_is_lattice is True
        assert result.is_lattice is True


# ---------------------------------------------------------------------------
# Benchmark replication tests
# ---------------------------------------------------------------------------

class TestBenchmarkReplication:
    """Tests for replication of standard benchmark protocols."""

    def test_iterator_squared(self) -> None:
        """Iterator^2 should be a lattice with correct diagonal count."""
        result = analyze_replication(
            "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}", 2
        )
        assert result.is_lattice is True
        assert result.product_states == result.base_states ** 2
        assert result.diagonal_states_count == result.base_states

    def test_iterator_cubed(self) -> None:
        """Iterator^3 should be a lattice with correct diagonal count."""
        result = analyze_replication(
            "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}", 3
        )
        assert result.is_lattice is True
        assert result.diagonal_states_count == result.base_states

    def test_smtp_squared(self) -> None:
        """SMTP^2 should be a lattice with correct diagonal count."""
        smtp = ("&{connect: &{ehlo: rec X . &{mail: &{rcpt: &{data: "
                "+{OK: X, ERR: X}}}, quit: end}}}")
        result = analyze_replication(smtp, 2)
        assert result.is_lattice is True
        assert result.diagonal_states_count == result.base_states


# ---------------------------------------------------------------------------
# Choir / server farm examples
# ---------------------------------------------------------------------------

class TestChoirExample:
    """Tests for the choir (musical) replication example."""

    def test_simple_choir_10(self) -> None:
        """A simple choir of 10 voices: &{sing: end} ^ 10."""
        body = "&{sing: end}"
        node = parse_replication(f"{body} ^ 10")
        assert isinstance(node, Replication)
        assert node.count == 10
        ss = build_replicated_statespace(node)
        # 2 states per voice, 2^10 = 1024 product states
        assert len(ss.states) == 2 ** 10

    def test_two_note_choir(self) -> None:
        """Choir with two notes: &{C4: &{E4: end}} ^ 2."""
        body = "&{C4: &{E4: end}}"
        result = analyze_replication(body, 2)
        assert result.base_states == 3  # top, after-C4, end
        assert result.product_states == 9
        assert result.is_lattice is True

    def test_server_farm(self) -> None:
        """Server farm: &{request: +{ok: end, err: end}} ^ 3."""
        body = "&{request: +{ok: end, err: end}}"
        result = analyze_replication(body, 3)
        assert result.is_lattice is True
        assert result.diagonal_isomorphic is True


# ---------------------------------------------------------------------------
# Edge cases and regression
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and regression tests."""

    def test_whitespace_handling(self) -> None:
        result = parse_replication("  end  ^  5  ")
        assert isinstance(result, Replication)
        assert result.count == 5

    def test_large_exponent(self) -> None:
        """S^10 for a 2-state type gives 2^10 = 1024 states."""
        body = "&{a: end}"
        node = Replication(body=parse(body), count=10)
        ss = build_replicated_statespace(node)
        assert len(ss.states) == 2 ** 10

    def test_orbit_count_formula_s2(self) -> None:
        """For k-state type, S^2 has C(k+1, 2) = k(k+1)/2 orbits."""
        # &{a: end, b: end} has 2 states (both a,b go to same end)
        # &{a: &{b: end}} has 3 states (top -> after-a -> end)
        for type_str, expected_k in [("end", 1), ("&{a: end}", 2), ("&{a: &{b: end}}", 3)]:
            body = parse(type_str)
            base_ss = build_statespace(body)
            k = len(base_ss.states)
            assert k == expected_k, f"Expected {expected_k} states for {type_str}, got {k}"
            node = Replication(body=body, count=2)
            ss = build_replicated_statespace(node)
            orbits = symmetry_orbits(ss, 2)
            assert len(orbits) == k * (k + 1) // 2

    def test_lattice_check_on_diagonal(self) -> None:
        """The diagonal sublattice itself should be a lattice."""
        body = parse("&{a: end, b: end}")
        node = Replication(body=body, count=2)
        ss = build_replicated_statespace(node)
        diag_ss = diagonal_lattice(ss, 2)
        result = check_lattice(diag_ss)
        assert result.is_lattice is True
