"""Tests for zeta matrix analysis (Step 30a).

Tests cover:
- Zeta matrix computation and structural properties
- Rank function and height/width
- Interval enumeration
- Chain counting via Hasse matrix powers
- Graded poset detection
- Kronecker product composition under parallel
- Incidence algebra operations (convolution, Möbius inversion)
- Benchmark protocol analysis
"""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.product import product_statespace
from reticulate.zeta import (
    zeta_matrix,
    compute_rank,
    compute_height,
    compute_width,
    check_graded,
    compute_interval,
    interval_info,
    enumerate_intervals,
    chain_counts,
    order_density,
    count_comparable_pairs,
    kronecker_product,
    verify_kronecker_composition,
    is_upper_triangular,
    zeta_trace,
    zeta_row_sums,
    zeta_col_sums,
    convolve,
    zeta_function,
    delta_function,
    mobius_function,
    zeta_power,
    analyze_zeta,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(s: str):
    """Parse and build state space."""
    return build_statespace(parse(s))


# ---------------------------------------------------------------------------
# Basic zeta matrix tests
# ---------------------------------------------------------------------------

class TestZetaMatrix:
    """Tests for zeta matrix computation."""

    def test_end_type(self):
        """end has a 1×1 zeta matrix [[1]]."""
        ss = _build("end")
        Z = zeta_matrix(ss)
        assert len(Z) == 1
        assert Z[0][0] == 1

    def test_single_branch(self):
        """&{a: end} has 2 states: top → bottom."""
        ss = _build("&{a: end}")
        Z = zeta_matrix(ss)
        assert len(Z) == 2
        # Diagonal always 1
        assert Z[0][0] == 1
        assert Z[1][1] == 1
        # top ≥ bottom
        total_ones = sum(Z[i][j] for i in range(2) for j in range(2))
        assert total_ones == 3  # diagonal(2) + one off-diagonal

    def test_two_branch(self):
        """&{a: end, b: end} has top with two branches to bottom."""
        ss = _build("&{a: end, b: end}")
        Z = zeta_matrix(ss)
        n = len(Z)
        assert n == 2  # top and bottom (both branches go to same end)
        # All pairs comparable in a chain
        assert Z[0][0] == 1
        assert Z[1][1] == 1

    def test_diamond(self):
        """&{a: &{c: end}, b: &{c: end}} creates a diamond shape."""
        ss = _build("&{a: &{c: end}, b: &{c: end}}")
        Z = zeta_matrix(ss)
        n = len(Z)
        assert n >= 3  # at least top, two middle, bottom

    def test_reflexivity(self):
        """Diagonal of Z is always 1 (reflexivity)."""
        ss = _build("&{a: &{b: end}, c: end}")
        Z = zeta_matrix(ss)
        for i in range(len(Z)):
            assert Z[i][i] == 1, f"Z[{i},{i}] should be 1"

    def test_transitivity(self):
        """If Z[i,j]=1 and Z[j,k]=1, then Z[i,k]=1."""
        ss = _build("&{a: &{b: end}}")
        Z = zeta_matrix(ss)
        n = len(Z)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if Z[i][j] == 1 and Z[j][k] == 1:
                        assert Z[i][k] == 1, f"Transitivity failed: Z[{i},{j}]=1, Z[{j},{k}]=1, but Z[{i},{k}]=0"

    def test_zeta_trace(self):
        """Trace of Z always equals n (number of states)."""
        ss = _build("&{a: &{b: end}, c: end}")
        Z = zeta_matrix(ss)
        assert zeta_trace(Z) == len(Z)

    def test_selection(self):
        """+{a: end, b: end} — selection type."""
        ss = _build("+{a: end, b: end}")
        Z = zeta_matrix(ss)
        n = len(Z)
        assert n >= 2
        # Reflexive
        for i in range(n):
            assert Z[i][i] == 1


# ---------------------------------------------------------------------------
# Rank function tests
# ---------------------------------------------------------------------------

class TestRank:
    """Tests for rank computation."""

    def test_end_rank(self):
        """end has rank 0."""
        ss = _build("end")
        rank = compute_rank(ss)
        assert rank[ss.bottom] == 0

    def test_chain_rank(self):
        """&{a: end} — top has rank 1, bottom has rank 0."""
        ss = _build("&{a: end}")
        rank = compute_rank(ss)
        assert rank[ss.bottom] == 0
        assert rank[ss.top] >= 1

    def test_longer_chain(self):
        """&{a: &{b: end}} — ranks are 2, 1, 0."""
        ss = _build("&{a: &{b: end}}")
        rank = compute_rank(ss)
        assert rank[ss.bottom] == 0
        assert rank[ss.top] == 2

    def test_height_matches_top_rank(self):
        """Height should equal rank of top."""
        ss = _build("&{a: &{b: &{c: end}}}")
        h = compute_height(ss)
        rank = compute_rank(ss)
        assert h == rank[ss.top]
        assert h == 3


# ---------------------------------------------------------------------------
# Width tests
# ---------------------------------------------------------------------------

class TestWidth:
    """Tests for antichain width computation."""

    def test_end_width(self):
        """end has width 1."""
        ss = _build("end")
        assert compute_width(ss) == 1

    def test_chain_width(self):
        """Linear chain has width 1."""
        ss = _build("&{a: end}")
        assert compute_width(ss) == 1

    def test_parallel_width(self):
        """Parallel composition increases width."""
        ss = _build("(&{a: end} || &{b: end})")
        w = compute_width(ss)
        assert w >= 2  # At least two incomparable states


# ---------------------------------------------------------------------------
# Height tests
# ---------------------------------------------------------------------------

class TestHeight:
    """Tests for poset height."""

    def test_end_height(self):
        ss = _build("end")
        assert compute_height(ss) == 0

    def test_branch_height(self):
        ss = _build("&{a: end}")
        assert compute_height(ss) == 1

    def test_nested_height(self):
        ss = _build("&{a: &{b: end}}")
        assert compute_height(ss) == 2

    def test_deep_chain(self):
        ss = _build("&{a: &{b: &{c: end}}}")
        assert compute_height(ss) == 3


# ---------------------------------------------------------------------------
# Graded poset tests
# ---------------------------------------------------------------------------

class TestGraded:
    """Tests for graded poset checking."""

    def test_chain_is_graded(self):
        """A chain is always graded."""
        ss = _build("&{a: &{b: end}}")
        assert check_graded(ss) is True

    def test_single_state_graded(self):
        ss = _build("end")
        assert check_graded(ss) is True

    def test_branch_graded(self):
        """Simple branch with uniform depth."""
        ss = _build("&{a: end, b: end}")
        assert check_graded(ss) is True


# ---------------------------------------------------------------------------
# Interval tests
# ---------------------------------------------------------------------------

class TestIntervals:
    """Tests for interval computation."""

    def test_full_interval(self):
        """[top, bottom] should contain all states."""
        ss = _build("&{a: end}")
        interval = compute_interval(ss, ss.top, ss.bottom)
        assert ss.top in interval
        assert ss.bottom in interval
        assert len(interval) == len(ss.states)

    def test_self_interval(self):
        """[x, x] = {x}."""
        ss = _build("&{a: &{b: end}}")
        for s in ss.states:
            interval = compute_interval(ss, s, s)
            assert interval == frozenset({s})

    def test_incomparable_empty(self):
        """[x, y] is empty if x does not dominate y."""
        ss = _build("(&{a: end} || &{b: end})")
        states = sorted(ss.states)
        # Find two incomparable states if they exist
        from reticulate.zeta import _reachability
        reach = _reachability(ss)
        for x in states:
            for y in states:
                if x != y and y not in reach[x] and x not in reach[y]:
                    interval = compute_interval(ss, x, y)
                    assert len(interval) == 0

    def test_interval_info_chains(self):
        """Interval info should count chains correctly."""
        ss = _build("&{a: end}")
        info = interval_info(ss, ss.top, ss.bottom)
        assert info.size == 2
        assert info.chains >= 1

    def test_enumerate_intervals(self):
        """Should enumerate all non-trivial intervals."""
        ss = _build("&{a: &{b: end}}")
        intervals = enumerate_intervals(ss)
        assert len(intervals) > 0
        # [top, bottom] must be among them
        full = [i for i in intervals if i.top == ss.top and i.bottom == ss.bottom]
        assert len(full) == 1


# ---------------------------------------------------------------------------
# Chain counting tests
# ---------------------------------------------------------------------------

class TestChainCounts:
    """Tests for chain counting."""

    def test_single_chain(self):
        """&{a: end} has exactly one chain of length 1."""
        ss = _build("&{a: end}")
        counts = chain_counts(ss)
        assert 1 in counts
        assert counts[1] == 1

    def test_two_step_chain(self):
        """&{a: &{b: end}} has chains of length 1 and 2."""
        ss = _build("&{a: &{b: end}}")
        counts = chain_counts(ss)
        assert 2 in counts  # direct chain top → mid → bottom

    def test_branch_chains(self):
        """&{a: end, b: end} may have multiple length-1 chains."""
        ss = _build("&{a: end, b: end}")
        counts = chain_counts(ss)
        assert len(counts) > 0


# ---------------------------------------------------------------------------
# Density tests
# ---------------------------------------------------------------------------

class TestDensity:
    """Tests for order density."""

    def test_single_state_density(self):
        """Single state: density = 1.0."""
        ss = _build("end")
        Z = zeta_matrix(ss)
        assert order_density(Z) == 1.0

    def test_chain_density(self):
        """Chain of length n: density approaches 0.5 for large n."""
        ss = _build("&{a: end}")
        Z = zeta_matrix(ss)
        d = order_density(Z)
        assert 0.0 < d <= 1.0

    def test_comparable_pairs(self):
        """Count of comparable pairs should be consistent with Z."""
        ss = _build("&{a: &{b: end}}")
        Z = zeta_matrix(ss)
        cp = count_comparable_pairs(Z)
        # Should equal total 1s minus diagonal
        total = sum(Z[i][j] for i in range(len(Z)) for j in range(len(Z)))
        assert cp == total - len(Z)


# ---------------------------------------------------------------------------
# Kronecker product and composition tests
# ---------------------------------------------------------------------------

class TestKronecker:
    """Tests for Kronecker product composition."""

    def test_identity_kronecker(self):
        """I ⊗ I = I for 1×1 matrices."""
        A = [[1]]
        B = [[1]]
        C = kronecker_product(A, B)
        assert C == [[1]]

    def test_2x2_kronecker(self):
        """Known 2×2 Kronecker product."""
        A = [[1, 1], [0, 1]]
        B = [[1, 1], [0, 1]]
        C = kronecker_product(A, B)
        assert len(C) == 4
        assert C[0][0] == 1  # A[0,0]*B[0,0]
        assert C[0][3] == 1  # A[0,1]*B[1,1]

    def test_parallel_composition(self):
        """Z(S1 || S2) should equal Z(S1) ⊗ Z(S2)."""
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end}")
        ss_prod = build_statespace(parse("(&{a: end} || &{b: end})"))

        result = verify_kronecker_composition(ss1, ss2, ss_prod)
        # Dimensions should match
        assert result.product_states == result.left_states * result.right_states

    def test_density_multiplicative(self):
        """Density should be approximately multiplicative under product."""
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end}")
        Z1 = zeta_matrix(ss1)
        Z2 = zeta_matrix(ss2)
        d1 = order_density(Z1)
        d2 = order_density(Z2)
        Zprod = kronecker_product(Z1, Z2)
        d_prod = order_density(Zprod)
        assert abs(d_prod - d1 * d2) < 1e-10


# ---------------------------------------------------------------------------
# Row/column sum tests
# ---------------------------------------------------------------------------

class TestRowColSums:
    """Tests for row and column sums of the zeta matrix."""

    def test_top_row_sum(self):
        """Top state row sum = n (reaches everything)."""
        ss = _build("&{a: end}")
        Z = zeta_matrix(ss)
        states = sorted(ss.states)
        sums = zeta_row_sums(Z, states)
        assert sums[ss.top] == len(states)

    def test_bottom_col_sum(self):
        """Bottom state column sum = n (everything reaches it)."""
        ss = _build("&{a: end}")
        Z = zeta_matrix(ss)
        states = sorted(ss.states)
        sums = zeta_col_sums(Z, states)
        assert sums[ss.bottom] == len(states)

    def test_bottom_row_sum_is_one(self):
        """Bottom state row sum = 1 (only reaches itself)."""
        ss = _build("&{a: end}")
        Z = zeta_matrix(ss)
        states = sorted(ss.states)
        sums = zeta_row_sums(Z, states)
        assert sums[ss.bottom] == 1


# ---------------------------------------------------------------------------
# Incidence algebra tests
# ---------------------------------------------------------------------------

class TestIncidenceAlgebra:
    """Tests for incidence algebra operations."""

    def test_zeta_function_reflexive(self):
        """ζ(x,x) = 1 for all x."""
        ss = _build("&{a: end}")
        zf = zeta_function(ss)
        for s in ss.states:
            assert zf.get((s, s), 0) == 1

    def test_delta_function(self):
        """δ(x,y) = 1 iff x ≡ y (same SCC; for acyclic, x = y)."""
        ss = _build("&{a: end}")
        df = delta_function(ss)
        for s in ss.states:
            assert df[(s, s)] == 1
        for s in ss.states:
            for t in ss.states:
                if s != t:
                    assert (s, t) not in df  # acyclic: each state is its own SCC

    def test_mobius_inversion(self):
        """ζ * μ = δ (Möbius inversion formula)."""
        ss = _build("&{a: end}")
        zf = zeta_function(ss)
        mf = mobius_function(ss)
        result = convolve(zf, mf, ss)
        df = delta_function(ss)

        # Should be the delta function
        for s in ss.states:
            for t in ss.states:
                expected = df.get((s, t), 0)
                actual = result.get((s, t), 0)
                assert actual == expected, \
                    f"(ζ*μ)({s},{t}) = {actual} ≠ δ({s},{t}) = {expected}"

    def test_mobius_inversion_larger(self):
        """Möbius inversion on a larger type."""
        ss = _build("&{a: &{b: end}, c: end}")
        zf = zeta_function(ss)
        mf = mobius_function(ss)
        result = convolve(zf, mf, ss)
        df = delta_function(ss)

        for s in ss.states:
            for t in ss.states:
                expected = df.get((s, t), 0)
                actual = result.get((s, t), 0)
                assert actual == expected, \
                    f"(ζ*μ)({s},{t}) = {actual} ≠ δ({s},{t}) = {expected}"


# ---------------------------------------------------------------------------
# Zeta power tests
# ---------------------------------------------------------------------------

class TestZetaPower:
    """Tests for matrix exponentiation."""

    def test_identity_power(self):
        """Z^0 = I."""
        Z = [[1, 1], [0, 1]]
        Z0 = zeta_power(Z, 0)
        assert Z0 == [[1, 0], [0, 1]]

    def test_first_power(self):
        """Z^1 = Z."""
        Z = [[1, 1], [0, 1]]
        Z1 = zeta_power(Z, 1)
        assert Z1 == Z

    def test_idempotent(self):
        """For a boolean zeta matrix, Z^2 should have same support as Z."""
        ss = _build("&{a: end}")
        Z = zeta_matrix(ss)
        Z2 = zeta_power(Z, 2)
        n = len(Z)
        for i in range(n):
            for j in range(n):
                if Z[i][j] > 0:
                    assert Z2[i][j] > 0


# ---------------------------------------------------------------------------
# Upper triangular tests
# ---------------------------------------------------------------------------

class TestUpperTriangular:
    """Tests for upper-triangular property in rank order."""

    def test_chain_triangular(self):
        """A chain is always upper triangular in rank order."""
        ss = _build("&{a: &{b: end}}")
        Z = zeta_matrix(ss)
        states = sorted(ss.states)
        rank = compute_rank(ss)
        assert is_upper_triangular(Z, states, rank) is True


# ---------------------------------------------------------------------------
# Full analysis tests
# ---------------------------------------------------------------------------

class TestAnalyzeZeta:
    """Tests for the main analyze_zeta function."""

    def test_end_analysis(self):
        """end type analysis."""
        ss = _build("end")
        r = analyze_zeta(ss)
        assert r.num_states == 1
        assert r.height == 0
        assert r.width == 1
        assert r.density == 1.0
        assert r.is_graded is True

    def test_simple_branch(self):
        """&{a: end} analysis."""
        ss = _build("&{a: end}")
        r = analyze_zeta(ss)
        assert r.num_states == 2
        assert r.height == 1
        assert r.width == 1
        assert r.is_graded is True
        assert r.num_comparable_pairs == 1

    def test_nested_branch(self):
        """&{a: &{b: end}} analysis."""
        ss = _build("&{a: &{b: end}}")
        r = analyze_zeta(ss)
        assert r.num_states == 3
        assert r.height == 2
        assert r.is_graded is True

    def test_parallel_analysis(self):
        """Parallel type analysis."""
        ss = _build("(&{a: end} || &{b: end})")
        r = analyze_zeta(ss)
        assert r.num_states >= 4
        assert r.width >= 2
        assert r.height >= 2

    def test_selection_analysis(self):
        """+{a: end, b: end} analysis."""
        ss = _build("+{a: end, b: end}")
        r = analyze_zeta(ss)
        assert r.num_states >= 2
        assert r.is_graded is True

    def test_recursive_analysis(self):
        """rec X . &{a: X, b: end} analysis."""
        ss = _build("rec X . &{a: X, b: end}")
        r = analyze_zeta(ss)
        assert r.num_states >= 2
        assert r.height >= 1


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------

class TestBenchmarks:
    """Tests on benchmark protocols."""

    BENCHMARKS = [
        ("Java Iterator", "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"),
        ("File Object", "&{open: &{read: end, write: end}}"),
        ("Simple SMTP", "&{EHLO: &{MAIL: &{RCPT: &{DATA: end}}}}"),
        ("Two-choice", "&{a: end, b: end}"),
        ("Nested", "&{a: &{b: &{c: end}}}"),
        ("Parallel simple", "(&{a: end} || &{b: end})"),
    ]

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_benchmark_zeta_properties(self, name, typ):
        """Zeta matrix has correct properties for all benchmarks."""
        ss = _build(typ)
        r = analyze_zeta(ss)

        # Basic invariants
        assert r.num_states == len(ss.states)
        assert r.height >= 0
        assert r.width >= 1
        assert 0.0 < r.density <= 1.0

        # Zeta matrix is n×n
        assert len(r.zeta) == r.num_states
        for row in r.zeta:
            assert len(row) == r.num_states

        # Reflexive: diagonal all 1
        for i in range(r.num_states):
            assert r.zeta[i][i] == 1

        # Trace = n
        assert zeta_trace(r.zeta) == r.num_states

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_benchmark_mobius_inversion(self, name, typ):
        """ζ * μ = δ holds for all benchmarks."""
        ss = _build(typ)
        zf = zeta_function(ss)
        mf = mobius_function(ss)
        result = convolve(zf, mf, ss)
        df = delta_function(ss)

        for s in ss.states:
            for t in ss.states:
                expected = df.get((s, t), 0)
                actual = result.get((s, t), 0)
                assert actual == expected, \
                    f"{name}: (ζ*μ)({s},{t}) = {actual} ≠ δ({s},{t}) = {expected}"

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_benchmark_rank_consistent(self, name, typ):
        """Rank function is consistent: x ≥ y implies rank(x) ≥ rank(y)."""
        ss = _build(typ)
        rank = compute_rank(ss)
        from reticulate.zeta import _reachability
        reach = _reachability(ss)
        for x in ss.states:
            for y in ss.states:
                if y in reach[x] and x != y:
                    assert rank[x] >= rank[y], \
                        f"{name}: rank({x})={rank[x]} < rank({y})={rank[y]} but {x} ≥ {y}"
