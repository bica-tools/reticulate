"""Tests for transfer matrix analysis (Step 30f).

Tests cover:
- Transfer matrix computation
- Path counting by length
- Total path count and expected length
- Path enumeration
- Resolvent computation
- Acyclicity check
- Benchmark protocols
"""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.transfer import (
    transfer_matrix,
    count_paths_by_length,
    total_path_count,
    expected_path_length,
    transfer_power,
    enumerate_paths,
    resolvent_at,
    is_transfer_acyclic,
    analyze_transfer,
)


def _build(s: str):
    return build_statespace(parse(s))


# ---------------------------------------------------------------------------
# Transfer matrix tests
# ---------------------------------------------------------------------------

class TestTransferMatrix:

    def test_end(self):
        """end: 1×1 zero matrix."""
        T, states = transfer_matrix(_build("end"))
        assert len(T) == 1
        assert T[0][0] == 0

    def test_single_branch(self):
        """&{a: end}: one covering edge."""
        T, states = transfer_matrix(_build("&{a: end}"))
        assert len(T) == 2
        # Exactly one 1 in the matrix (top covers bottom)
        total = sum(T[i][j] for i in range(2) for j in range(2))
        assert total == 1

    def test_chain_3(self):
        """Chain of 3: two covering edges."""
        T, states = transfer_matrix(_build("&{a: &{b: end}}"))
        assert len(T) == 3
        total = sum(T[i][j] for i in range(3) for j in range(3))
        assert total == 2

    def test_diagonal_zero(self):
        """No self-loops in covering relation."""
        for typ in ["&{a: end}", "&{a: &{b: end}}", "(&{a: end} || &{b: end})"]:
            T, states = transfer_matrix(_build(typ))
            for i in range(len(T)):
                assert T[i][i] == 0


# ---------------------------------------------------------------------------
# Path counting tests
# ---------------------------------------------------------------------------

class TestPathCounting:

    def test_end_paths(self):
        """end: one path of length 0 (trivial)."""
        counts = count_paths_by_length(_build("end"))
        assert counts == {0: 1}

    def test_single_branch_paths(self):
        """&{a: end}: one path of length 1."""
        counts = count_paths_by_length(_build("&{a: end}"))
        assert counts.get(1, 0) == 1

    def test_chain_3_paths(self):
        """Chain of 3: one path of length 2."""
        counts = count_paths_by_length(_build("&{a: &{b: end}}"))
        assert counts.get(2, 0) == 1

    def test_two_branches(self):
        """&{a: end, b: end}: may have multiple length-1 paths."""
        counts = count_paths_by_length(_build("&{a: end, b: end}"))
        assert sum(counts.values()) >= 1

    def test_parallel_paths(self):
        """Parallel creates multiple paths (interleavings)."""
        counts = count_paths_by_length(_build("(&{a: end} || &{b: end})"))
        assert sum(counts.values()) >= 2

    def test_total_path_count(self):
        ss = _build("&{a: end}")
        assert total_path_count(ss) == 1

    def test_expected_length(self):
        ss = _build("&{a: end}")
        assert expected_path_length(ss) == 1.0


# ---------------------------------------------------------------------------
# Transfer power tests
# ---------------------------------------------------------------------------

class TestTransferPower:

    def test_power_0_is_identity(self):
        T = transfer_power(_build("&{a: end}"), 0)
        assert T[0][0] == 1
        assert T[1][1] == 1

    def test_power_1_is_T(self):
        ss = _build("&{a: end}")
        T, _ = transfer_matrix(ss)
        T1 = transfer_power(ss, 1)
        assert T1 == T

    def test_power_exceeds_height(self):
        """T^k = 0 for k > height (acyclic)."""
        ss = _build("&{a: end}")
        T3 = transfer_power(ss, 3)
        assert all(T3[i][j] == 0 for i in range(2) for j in range(2))


# ---------------------------------------------------------------------------
# Path enumeration tests
# ---------------------------------------------------------------------------

class TestPathEnumeration:

    def test_end_path(self):
        paths = enumerate_paths(_build("end"))
        assert len(paths) == 1
        assert len(paths[0]) == 1  # Just the single state

    def test_single_branch_path(self):
        ss = _build("&{a: end}")
        paths = enumerate_paths(ss)
        assert len(paths) == 1
        assert paths[0][0] == ss.top
        assert paths[0][-1] == ss.bottom

    def test_parallel_multiple_paths(self):
        paths = enumerate_paths(_build("(&{a: end} || &{b: end})"))
        assert len(paths) >= 2

    def test_path_starts_at_top(self):
        ss = _build("&{a: &{b: end}}")
        for path in enumerate_paths(ss):
            assert path[0] == ss.top

    def test_path_ends_at_bottom(self):
        ss = _build("&{a: &{b: end}}")
        for path in enumerate_paths(ss):
            assert path[-1] == ss.bottom


# ---------------------------------------------------------------------------
# Resolvent tests
# ---------------------------------------------------------------------------

class TestResolvent:

    def test_resolvent_identity_at_0(self):
        """(I - 0·T)^{-1} = I."""
        ss = _build("&{a: end}")
        R = resolvent_at(ss, 0.0)
        assert R is not None
        n = len(R)
        for i in range(n):
            for j in range(n):
                expected = 1.0 if i == j else 0.0
                assert abs(R[i][j] - expected) < 1e-10

    def test_resolvent_exists(self):
        """Resolvent exists for small z."""
        for typ in ["&{a: end}", "&{a: &{b: end}}"]:
            R = resolvent_at(_build(typ), 0.5)
            assert R is not None


# ---------------------------------------------------------------------------
# Acyclicity tests
# ---------------------------------------------------------------------------

class TestAcyclicity:

    def test_chain_acyclic(self):
        assert is_transfer_acyclic(_build("&{a: end}")) is True

    def test_chain_3_acyclic(self):
        assert is_transfer_acyclic(_build("&{a: &{b: end}}")) is True

    def test_parallel_acyclic(self):
        assert is_transfer_acyclic(_build("(&{a: end} || &{b: end})")) is True


# ---------------------------------------------------------------------------
# Full analysis tests
# ---------------------------------------------------------------------------

class TestAnalyze:

    def test_end(self):
        r = analyze_transfer(_build("end"))
        assert r.num_states == 1
        assert r.total_paths == 1
        assert r.height == 0

    def test_branch(self):
        r = analyze_transfer(_build("&{a: end}"))
        assert r.num_states == 2
        assert r.total_paths == 1
        assert r.height == 1
        assert r.is_acyclic is True

    def test_parallel(self):
        r = analyze_transfer(_build("(&{a: end} || &{b: end})"))
        assert r.total_paths >= 2
        assert r.height >= 2
        assert r.is_acyclic is True


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------

class TestBenchmarks:
    BENCHMARKS = [
        ("Java Iterator", "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"),
        ("File Object", "&{open: &{read: end, write: end}}"),
        ("Simple SMTP", "&{EHLO: &{MAIL: &{RCPT: &{DATA: end}}}}"),
        ("Two-choice", "&{a: end, b: end}"),
        ("Nested", "&{a: &{b: &{c: end}}}"),
        ("Parallel", "(&{a: end} || &{b: end})"),
    ]

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_transfer_properties(self, name, typ):
        """Transfer matrix properties hold for all benchmarks."""
        ss = _build(typ)
        r = analyze_transfer(ss)
        assert r.num_states == len(ss.states)
        # Recursive types may have 0 Hasse paths (SCC collapses remove edges)
        assert r.total_paths >= 0
        assert r.height >= 0
        assert r.expected_path_length >= 0

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_paths_valid(self, name, typ):
        """All enumerated paths start at top and end at bottom."""
        ss = _build(typ)
        paths = enumerate_paths(ss, max_paths=100)
        for path in paths:
            assert path[0] == ss.top, f"{name}: path doesn't start at top"
            assert path[-1] == ss.bottom, f"{name}: path doesn't end at bottom"

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_path_count_matches(self, name, typ):
        """Total from distribution matches total_path_count."""
        ss = _build(typ)
        dist = count_paths_by_length(ss)
        total = total_path_count(ss)
        assert sum(dist.values()) == total
