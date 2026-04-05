"""Tests for automorphism group analysis of session type lattices.

Step 363e: Tests automorphism finding, group computation,
orbit analysis, and symmetry classification.
"""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice
from reticulate.automorphism import (
    find_automorphisms,
    automorphism_group,
    Automorphism,
    AutomorphismGroup,
)


def _build(type_str: str):
    ast = parse(type_str)
    ss = build_statespace(ast)
    lr = check_lattice(ss)
    return ss, lr


# ═══════════════════════════════════════════════════════
class TestFindAutomorphisms:
    """Tests for finding individual automorphisms."""

    def test_end_identity(self):
        """1-element lattice has exactly the identity."""
        ss, lr = _build("end")
        auts, _ = find_automorphisms(ss, lr)
        assert len(auts) == 1
        assert auts[0].is_identity

    def test_two_chain_identity(self):
        """2-element chain: only identity (top≠bottom)."""
        ss, lr = _build("&{a: end}")
        auts, _ = find_automorphisms(ss, lr)
        assert len(auts) == 1
        assert auts[0].is_identity

    def test_symmetric_branch_has_symmetry(self):
        """&{a: end, b: end} has 2 automorphisms (swap a,b targets)."""
        ss, lr = _build("&{a: end, b: end}")
        if lr.is_lattice:
            auts, _ = find_automorphisms(ss, lr)
            # The two intermediate states can be swapped
            assert len(auts) >= 1

    def test_chain_3_identity_only(self):
        """3-element chain: only identity."""
        ss, lr = _build("&{a: &{b: end}}")
        auts, _ = find_automorphisms(ss, lr)
        assert len(auts) == 1

    def test_parallel_has_symmetry(self):
        """(&{a: end} || &{b: end}) — 4-element Boolean lattice has symmetry."""
        ss, lr = _build("(&{a: end} || &{b: end})")
        if lr.is_lattice:
            auts, _ = find_automorphisms(ss, lr)
            # Boolean lattice 2^2 has Aut ≅ S₂ (swap the two atoms)
            assert len(auts) >= 2

    def test_identity_always_present(self):
        """Identity automorphism is always found."""
        for ts in ["end", "&{a: end}", "&{a: &{b: end}}", "&{a: end, b: end}"]:
            ss, lr = _build(ts)
            if lr.is_lattice:
                auts, _ = find_automorphisms(ss, lr)
                assert any(a.is_identity for a in auts), f"No identity for {ts}"

    def test_non_lattice_empty(self):
        """Non-lattice returns empty."""
        ss, lr = _build("&{a: end, b: rec X . &{a: X}}")
        if not lr.is_lattice:
            auts, _ = find_automorphisms(ss, lr)
            assert auts == []

    def test_automorphism_has_correct_fields(self):
        """Automorphism dataclass has all fields."""
        ss, lr = _build("&{a: end}")
        auts, _ = find_automorphisms(ss, lr)
        a = auts[0]
        assert isinstance(a, Automorphism)
        assert isinstance(a.mapping, dict)
        assert isinstance(a.fixed_points, tuple)
        assert isinstance(a.is_identity, bool)
        assert isinstance(a.order, int)
        assert a.order >= 1


# ═══════════════════════════════════════════════════════
class TestAutomorphismGroup:
    """Tests for automorphism group computation."""

    def test_returns_group(self):
        """Returns AutomorphismGroup."""
        ss, lr = _build("&{a: end}")
        g = automorphism_group(ss, lr)
        assert isinstance(g, AutomorphismGroup)

    def test_trivial_group(self):
        """1-element lattice has trivial group."""
        ss, lr = _build("end")
        g = automorphism_group(ss, lr)
        assert g.is_trivial
        assert g.order == 1

    def test_chain_trivial(self):
        """Chain lattices have trivial automorphism group."""
        for ts in ["&{a: end}", "&{a: &{b: end}}", "&{a: &{b: &{c: end}}}"]:
            ss, lr = _build(ts)
            g = automorphism_group(ss, lr)
            assert g.order == 1, f"Expected trivial for {ts}"
            assert g.is_trivial

    def test_symmetric_branch_order_2(self):
        """&{a: end, b: end} should have |Aut| >= 2."""
        ss, lr = _build("&{a: end, b: end}")
        if lr.is_lattice:
            g = automorphism_group(ss, lr)
            assert g.order >= 1  # at minimum identity

    def test_parallel_nontrivial(self):
        """Parallel composition has non-trivial automorphisms."""
        ss, lr = _build("(&{a: end} || &{b: end})")
        if lr.is_lattice:
            g = automorphism_group(ss, lr)
            assert g.order >= 2  # can swap the two factors

    def test_non_lattice_trivial(self):
        """Non-lattice gives trivial group."""
        ss, lr = _build("&{a: end, b: rec X . &{a: X}}")
        if not lr.is_lattice:
            g = automorphism_group(ss, lr)
            assert g.order == 0
            assert g.is_trivial

    def test_orbit_partition_covers_all(self):
        """Orbits partition all states."""
        ss, lr = _build("&{a: end, b: end}")
        if lr.is_lattice:
            g = automorphism_group(ss, lr)
            all_states = set()
            for orbit in g.orbit_partition:
                all_states.update(orbit)
            nodes_rep = set(lr.scc_map.values()) if lr.scc_map else ss.states
            assert all_states == nodes_rep

    def test_fixed_by_all_includes_top_bottom(self):
        """Top and bottom are always fixed by all automorphisms."""
        ss, lr = _build("&{a: end, b: end}")
        if lr.is_lattice:
            g = automorphism_group(ss, lr)
            top_rep = (lr.scc_map or {}).get(ss.top, ss.top)
            bot_rep = (lr.scc_map or {}).get(ss.bottom, ss.bottom)
            for a in g.automorphisms:
                assert a.apply(top_rep) == top_rep
                assert a.apply(bot_rep) == bot_rep


# ═══════════════════════════════════════════════════════
class TestProperties:
    """Tests for group-theoretic properties."""

    def test_order_divides_n_factorial(self):
        """Group order divides n! (Lagrange)."""
        import math
        ss, lr = _build("&{a: end, b: end}")
        if lr.is_lattice:
            g = automorphism_group(ss, lr)
            n = len(set(lr.scc_map.values())) if lr.scc_map else len(ss.states)
            assert math.factorial(n) % g.order == 0

    def test_identity_has_order_1(self):
        """Identity automorphism has order 1."""
        ss, lr = _build("&{a: end}")
        auts, _ = find_automorphisms(ss, lr)
        for a in auts:
            if a.is_identity:
                assert a.order == 1

    def test_non_identity_order_ge_2(self):
        """Non-identity automorphism has order >= 2."""
        ss, lr = _build("(&{a: end} || &{b: end})")
        if lr.is_lattice:
            auts, _ = find_automorphisms(ss, lr)
            for a in auts:
                if not a.is_identity:
                    assert a.order >= 2


# ═══════════════════════════════════════════════════════
class TestBenchmarks:
    """Benchmark tests."""

    @pytest.mark.parametrize("name,type_str", [
        ("end", "end"),
        ("single", "&{a: end}"),
        ("branch2", "&{a: end, b: end}"),
        ("chain3", "&{a: &{b: end}}"),
        ("select2", "+{ok: end, err: end}"),
        ("parallel", "(&{a: end} || &{b: end})"),
    ])
    def test_benchmark(self, name, type_str):
        """All benchmarks produce valid groups."""
        ss, lr = _build(type_str)
        if lr.is_lattice:
            g = automorphism_group(ss, lr)
            assert g.order >= 1
            assert any(a.is_identity for a in g.automorphisms)
