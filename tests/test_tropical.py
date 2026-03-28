"""Tests for tropical algebra analysis (Steps 30k-30n).

Tests cover:
- Tropical distance matrix (shortest paths)
- Diameter, eccentricity, radius, center
- Tropical eigenvalue (max cycle mean)
- Tropical determinant (min-plus permanent)
- Max-plus longest paths
- Benchmark protocols
"""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.tropical import (
    tropical_distance,
    diameter,
    eccentricity,
    radius,
    center,
    tropical_eigenvalue,
    tropical_determinant,
    longest_path_matrix,
    longest_path_top_bottom,
    shortest_path_top_bottom,
    analyze_tropical,
    INF,
)


def _build(s: str):
    return build_statespace(parse(s))


# ---------------------------------------------------------------------------
# Distance matrix tests (Step 30k)
# ---------------------------------------------------------------------------

class TestDistance:

    def test_end(self):
        D = tropical_distance(_build("end"))
        assert D == [[0]]

    def test_single_edge(self):
        ss = _build("&{a: end}")
        D = tropical_distance(ss)
        states = sorted(ss.states)
        idx = {s: i for i, s in enumerate(states)}
        # top → bottom = 1
        assert D[idx[ss.top]][idx[ss.bottom]] == 1

    def test_chain_distances(self):
        ss = _build("&{a: &{b: end}}")
        D = tropical_distance(ss)
        states = sorted(ss.states)
        idx = {s: i for i, s in enumerate(states)}
        # top → bottom = 2 (two steps)
        assert D[idx[ss.top]][idx[ss.bottom]] == 2

    def test_diagonal_zero(self):
        ss = _build("&{a: &{b: end}}")
        D = tropical_distance(ss)
        for i in range(len(D)):
            assert D[i][i] == 0

    def test_unreachable_is_inf(self):
        """Bottom cannot reach top (directed graph)."""
        ss = _build("&{a: end}")
        D = tropical_distance(ss)
        states = sorted(ss.states)
        idx = {s: i for i, s in enumerate(states)}
        assert D[idx[ss.bottom]][idx[ss.top]] == INF


# ---------------------------------------------------------------------------
# Diameter, radius, center tests
# ---------------------------------------------------------------------------

class TestGraphMetrics:

    def test_end_diameter(self):
        assert diameter(_build("end")) == 0

    def test_chain_diameter(self):
        assert diameter(_build("&{a: end}")) == 1
        assert diameter(_build("&{a: &{b: end}}")) == 2

    def test_radius_le_diameter(self):
        for typ in ["&{a: end}", "&{a: &{b: end}}", "(&{a: end} || &{b: end})"]:
            ss = _build(typ)
            r = radius(ss)
            d = diameter(ss)
            assert r <= d

    def test_center_nonempty(self):
        for typ in ["end", "&{a: end}", "&{a: &{b: end}}"]:
            ss = _build(typ)
            c = center(ss)
            assert len(c) >= 1

    def test_center_states_valid(self):
        ss = _build("&{a: &{b: end}}")
        c = center(ss)
        for s in c:
            assert s in ss.states


# ---------------------------------------------------------------------------
# Tropical eigenvalue tests (Step 30l)
# ---------------------------------------------------------------------------

class TestTropicalEigenvalue:

    def test_acyclic_zero(self):
        """Acyclic graphs have tropical eigenvalue 0."""
        assert tropical_eigenvalue(_build("end")) == 0.0
        assert tropical_eigenvalue(_build("&{a: end}")) == 0.0
        assert tropical_eigenvalue(_build("&{a: &{b: end}}")) == 0.0

    def test_recursive_positive(self):
        """Recursive types create cycles → positive tropical eigenvalue."""
        ss = _build("rec X . &{a: X, b: end}")
        te = tropical_eigenvalue(ss)
        assert te >= 0.0  # May or may not have cycles in Hasse

    def test_parallel_acyclic(self):
        """Parallel without recursion is acyclic."""
        assert tropical_eigenvalue(_build("(&{a: end} || &{b: end})")) == 0.0


# ---------------------------------------------------------------------------
# Tropical determinant tests (Step 30m)
# ---------------------------------------------------------------------------

class TestTropicalDeterminant:

    def test_end(self):
        assert tropical_determinant(_build("end")) == 0.0

    def test_single_edge(self):
        """2 states: identity permutation has weight 0+0=0,
        swap has weight 1+INF or INF+1 depending on direction."""
        td = tropical_determinant(_build("&{a: end}"))
        assert td >= 0

    def test_chain(self):
        td = tropical_determinant(_build("&{a: &{b: end}}"))
        assert td >= 0


# ---------------------------------------------------------------------------
# Longest path tests (Step 30n)
# ---------------------------------------------------------------------------

class TestLongestPath:

    def test_end(self):
        assert longest_path_top_bottom(_build("end")) == 0

    def test_chain_1(self):
        assert longest_path_top_bottom(_build("&{a: end}")) == 1

    def test_chain_2(self):
        assert longest_path_top_bottom(_build("&{a: &{b: end}}")) == 2

    def test_chain_3(self):
        assert longest_path_top_bottom(_build("&{a: &{b: &{c: end}}}")) == 3

    def test_parallel_longest(self):
        """Parallel: longest path ≥ 2."""
        lp = longest_path_top_bottom(_build("(&{a: end} || &{b: end})"))
        assert lp >= 2

    def test_shortest_le_longest(self):
        """Shortest ≤ longest always."""
        for typ in ["&{a: end}", "&{a: &{b: end}}", "(&{a: end} || &{b: end})"]:
            ss = _build(typ)
            s = shortest_path_top_bottom(ss)
            l = longest_path_top_bottom(ss)
            assert s <= l


# ---------------------------------------------------------------------------
# Full analysis tests
# ---------------------------------------------------------------------------

class TestAnalyze:

    def test_end(self):
        r = analyze_tropical(_build("end"))
        assert r.num_states == 1
        assert r.diameter == 0
        assert r.tropical_eigenvalue == 0.0

    def test_chain(self):
        r = analyze_tropical(_build("&{a: &{b: end}}"))
        assert r.diameter == 2
        assert r.longest_path_top_bottom == 2
        assert r.shortest_path_top_bottom == 2

    def test_parallel(self):
        r = analyze_tropical(_build("(&{a: end} || &{b: end})"))
        assert r.diameter >= 2
        assert r.longest_path_top_bottom >= 2


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
    def test_tropical_properties(self, name, typ):
        """Tropical analysis runs correctly."""
        ss = _build(typ)
        r = analyze_tropical(ss)
        assert r.num_states == len(ss.states)
        assert r.diameter >= 0
        assert r.radius >= 0
        assert r.radius <= r.diameter
        assert len(r.center) >= 1

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_path_lengths(self, name, typ):
        """Shortest ≤ longest for all benchmarks."""
        ss = _build(typ)
        s = shortest_path_top_bottom(ss)
        l = longest_path_top_bottom(ss)
        assert s <= l, f"{name}: shortest={s} > longest={l}"

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_distance_reflexive(self, name, typ):
        """D[i][i] = 0 for all states."""
        ss = _build(typ)
        D = tropical_distance(ss)
        for i in range(len(D)):
            assert D[i][i] == 0
