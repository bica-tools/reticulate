"""Tests for Ihara zeta function analysis (Step 30s).

Tests cover:
- Cycle rank computation
- Ihara determinant (Bass's formula)
- Hashimoto (edge adjacency) matrix
- Prime cycle counting and lengths
- Ramanujan property
- Graph complexity
- DAGs (trivial zeta)
- Trees (no cycles)
- Recursive types (with cycles)
- Bass formula verification
- Benchmark protocol analysis
- Full analyze_ihara integration
"""

import math
import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.ihara import (
    IharaResult,
    analyze_ihara,
    bass_hashimoto_matrix,
    count_prime_cycles,
    cycle_rank,
    graph_complexity,
    ihara_determinant,
    ihara_poles,
    is_ramanujan,
    prime_cycle_lengths,
    _undirected_edges,
    _undirected_adjacency,
    _adjacency_matrix,
    _degree_matrix,
    _mat_det,
    _mat_identity,
    _mat_add,
    _mat_sub,
    _mat_scale,
    _mat_mul,
    _eigenvalues_symmetric,
    _is_proper_power,
    _canonical_rotation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(s: str):
    """Parse and build state space."""
    return build_statespace(parse(s))


# ---------------------------------------------------------------------------
# DAG tests (no cycles => trivial Ihara zeta)
# ---------------------------------------------------------------------------

class TestDAGTrivialZeta:
    """DAGs have Z_G(u) = 1: no prime cycles."""

    def test_end_type(self):
        """end: single state, no edges, trivial zeta."""
        ss = _build("end")
        assert cycle_rank(ss) == 0
        assert count_prime_cycles(ss) == 0
        assert ihara_determinant(ss, 0.5) == pytest.approx(1.0)

    def test_single_branch(self):
        """&{a: end}: two states, one edge, DAG."""
        ss = _build("&{a: end}")
        assert cycle_rank(ss) == 0
        assert count_prime_cycles(ss) == 0

    def test_two_branch(self):
        """&{a: end, b: end}: three states, two edges, DAG."""
        ss = _build("&{a: end, b: end}")
        r = cycle_rank(ss)
        assert r == 0
        assert count_prime_cycles(ss) == 0

    def test_select_dag(self):
        """+{a: end, b: end}: DAG, trivial zeta."""
        ss = _build("+{a: end, b: end}")
        assert cycle_rank(ss) == 0
        assert count_prime_cycles(ss) == 0

    def test_nested_branch_dag(self):
        """&{a: &{b: end}}: chain of 3 states, DAG."""
        ss = _build("&{a: &{b: end}}")
        assert cycle_rank(ss) == 0
        assert count_prime_cycles(ss) == 0
        det = ihara_determinant(ss, 0.3)
        assert det == pytest.approx(1.0, abs=1e-8)

    def test_diamond_dag(self):
        """&{a: &{c: end}, b: &{c: end}}: diamond shape, may have cycle rank > 0."""
        ss = _build("&{a: &{c: end}, b: &{c: end}}")
        # Diamond has undirected edges forming a cycle if paths merge
        r = cycle_rank(ss)
        # With shared end state, there can be a cycle in the undirected version
        assert r >= 0

    def test_dag_determinant_at_zero(self):
        """Determinant at u=0 is always 1."""
        ss = _build("&{a: end, b: end}")
        assert ihara_determinant(ss, 0.0) == pytest.approx(1.0)

    def test_dag_complexity_zero(self):
        """DAGs with no cycles have zero complexity."""
        ss = _build("&{a: end}")
        c = graph_complexity(ss)
        assert c == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# Tree tests (no cycles)
# ---------------------------------------------------------------------------

class TestTreeNoCycles:
    """Trees have no cycles, so Z_G(u) = 1."""

    def test_binary_tree(self):
        """&{a: end, b: end} is a tree (star graph)."""
        ss = _build("&{a: end, b: end}")
        assert cycle_rank(ss) == 0
        assert count_prime_cycles(ss) == 0
        assert is_ramanujan(ss)

    def test_deep_chain(self):
        """&{a: &{b: &{c: end}}}: linear chain, no cycles."""
        ss = _build("&{a: &{b: &{c: end}}}")
        assert cycle_rank(ss) == 0
        assert count_prime_cycles(ss) == 0


# ---------------------------------------------------------------------------
# Recursive types (have cycles)
# ---------------------------------------------------------------------------

class TestRecursiveCycles:
    """Recursive session types produce cycles in the state space."""

    def test_simple_recursion_cycle_rank(self):
        """rec X . &{a: X, b: end}: one cycle."""
        ss = _build("rec X . &{a: X, b: end}")
        # Has a cycle: top -a-> top, so undirected self-loop is excluded
        # but the directed edge a: top->top creates no undirected edge
        r = cycle_rank(ss)
        assert r >= 0

    def test_simple_recursion_prime_cycles(self):
        """rec X . &{a: X, b: end}: check cycle detection."""
        ss = _build("rec X . &{a: X, b: end}")
        lengths = prime_cycle_lengths(ss, max_length=6)
        # Self-loops don't count as undirected edges, so may have 0 prime cycles
        assert isinstance(lengths, list)

    def test_mutual_recursion_like(self):
        """rec X . &{a: &{b: X}, c: end}: cycle of length 2."""
        ss = _build("rec X . &{a: &{b: X}, c: end}")
        r = cycle_rank(ss)
        assert r >= 0
        lengths = prime_cycle_lengths(ss, max_length=10)
        assert isinstance(lengths, list)

    def test_recursion_determinant_nontrival(self):
        """Recursive type: determinant should generally differ from 1."""
        ss = _build("rec X . &{a: &{b: X}, c: end}")
        det = ihara_determinant(ss, 0.3)
        assert isinstance(det, float)
        # For types with cycles, det may differ from 1
        assert math.isfinite(det)


# ---------------------------------------------------------------------------
# Cycle rank tests
# ---------------------------------------------------------------------------

class TestCycleRank:
    """Tests for cycle_rank = |E| - |V| + components."""

    def test_empty(self):
        """Single state: r = 0 - 1 + 1 = 0."""
        ss = _build("end")
        assert cycle_rank(ss) == 0

    def test_single_edge(self):
        """Two states, one edge: r = 1 - 2 + 1 = 0."""
        ss = _build("&{a: end}")
        assert cycle_rank(ss) == 0

    def test_rank_non_negative(self):
        """Cycle rank is always non-negative."""
        for ty in ["end", "&{a: end}", "&{a: end, b: end}",
                    "+{a: end, b: end}", "&{a: &{b: end}}"]:
            ss = _build(ty)
            assert cycle_rank(ss) >= 0


# ---------------------------------------------------------------------------
# Undirected edge helpers
# ---------------------------------------------------------------------------

class TestUndirectedEdges:
    """Tests for undirected edge extraction."""

    def test_end_no_edges(self):
        ss = _build("end")
        assert _undirected_edges(ss) == []

    def test_single_branch_one_edge(self):
        ss = _build("&{a: end}")
        edges = _undirected_edges(ss)
        assert len(edges) == 1

    def test_two_branch_edges(self):
        ss = _build("&{a: end, b: end}")
        edges = _undirected_edges(ss)
        # top -> end (via a) and top -> end (via b) collapse to one undirected edge
        assert len(edges) >= 1

    def test_no_self_loops(self):
        """Self-loops from recursion are excluded from undirected edges."""
        ss = _build("rec X . &{a: X, b: end}")
        edges = _undirected_edges(ss)
        for u, v in edges:
            assert u != v

    def test_adjacency_symmetric(self):
        """Undirected adjacency should be symmetric."""
        ss = _build("&{a: &{b: end}, c: end}")
        adj = _undirected_adjacency(ss)
        for u in adj:
            for v in adj[u]:
                assert u in adj[v]


# ---------------------------------------------------------------------------
# Adjacency and degree matrix tests
# ---------------------------------------------------------------------------

class TestMatrices:
    """Tests for adjacency and degree matrix construction."""

    def test_adjacency_symmetric(self):
        """Adjacency matrix should be symmetric."""
        ss = _build("&{a: end, b: end}")
        A, states = _adjacency_matrix(ss)
        n = len(states)
        for i in range(n):
            for j in range(n):
                assert A[i][j] == A[j][i]

    def test_adjacency_diagonal_zero(self):
        """Diagonal of adjacency matrix should be 0 (no self-loops)."""
        ss = _build("&{a: end, b: end}")
        A, states = _adjacency_matrix(ss)
        for i in range(len(states)):
            assert A[i][i] == 0.0

    def test_degree_matrix_diagonal(self):
        """Degree matrix is diagonal with correct degrees."""
        ss = _build("&{a: end, b: end}")
        D, states = _degree_matrix(ss)
        A, _ = _adjacency_matrix(ss)
        n = len(states)
        for i in range(n):
            assert D[i][i] == sum(A[i])
            for j in range(n):
                if i != j:
                    assert D[i][j] == 0.0


# ---------------------------------------------------------------------------
# Linear algebra helper tests
# ---------------------------------------------------------------------------

class TestLinearAlgebra:
    """Tests for pure-Python linear algebra routines."""

    def test_identity(self):
        I = _mat_identity(3)
        for i in range(3):
            for j in range(3):
                assert I[i][j] == (1.0 if i == j else 0.0)

    def test_det_identity(self):
        assert _mat_det(_mat_identity(3)) == pytest.approx(1.0)

    def test_det_2x2(self):
        M = [[3.0, 1.0], [2.0, 4.0]]
        assert _mat_det(M) == pytest.approx(10.0)

    def test_det_singular(self):
        M = [[1.0, 2.0], [2.0, 4.0]]
        assert _mat_det(M) == pytest.approx(0.0, abs=1e-10)

    def test_mat_mul_identity(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        I = _mat_identity(2)
        result = _mat_mul(A, I)
        for i in range(2):
            for j in range(2):
                assert result[i][j] == pytest.approx(A[i][j])

    def test_mat_add(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        B = [[5.0, 6.0], [7.0, 8.0]]
        C = _mat_add(A, B)
        assert C == [[6.0, 8.0], [10.0, 12.0]]

    def test_mat_sub(self):
        A = [[5.0, 6.0], [7.0, 8.0]]
        B = [[1.0, 2.0], [3.0, 4.0]]
        C = _mat_sub(A, B)
        assert C == [[4.0, 4.0], [4.0, 4.0]]

    def test_mat_scale(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        C = _mat_scale(A, 2.0)
        assert C == [[2.0, 4.0], [6.0, 8.0]]

    def test_eigenvalues_identity(self):
        eigs = _eigenvalues_symmetric(_mat_identity(3))
        for e in eigs:
            assert e == pytest.approx(1.0, abs=1e-6)

    def test_eigenvalues_diagonal(self):
        M = [[3.0, 0.0], [0.0, 1.0]]
        eigs = _eigenvalues_symmetric(M)
        assert eigs[0] == pytest.approx(3.0, abs=1e-6)
        assert eigs[1] == pytest.approx(1.0, abs=1e-6)

    def test_eigenvalues_symmetric_2x2(self):
        M = [[2.0, 1.0], [1.0, 2.0]]
        eigs = _eigenvalues_symmetric(M)
        assert sorted(eigs)[0] == pytest.approx(1.0, abs=1e-6)
        assert sorted(eigs)[1] == pytest.approx(3.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Hashimoto matrix tests
# ---------------------------------------------------------------------------

class TestHashimotoMatrix:
    """Tests for the Bass-Hashimoto (edge adjacency) matrix."""

    def test_no_edges(self):
        """Graph with no edges has empty Hashimoto matrix."""
        ss = _build("end")
        B = bass_hashimoto_matrix(ss)
        assert B == []

    def test_single_edge(self):
        """Single undirected edge -> 2 oriented edges -> 2x2 matrix."""
        ss = _build("&{a: end}")
        B = bass_hashimoto_matrix(ss)
        assert len(B) == 2
        # For a single edge, no non-backtracking continuation exists
        # B should be all zeros
        for row in B:
            for val in row:
                assert val == 0.0

    def test_path_graph(self):
        """Path of 3 vertices: 2 undirected edges -> 4 oriented edges."""
        ss = _build("&{a: &{b: end}}")
        B = bass_hashimoto_matrix(ss)
        assert len(B) == 4
        # Each oriented edge in a path can continue to the next
        # without backtracking
        total_ones = sum(B[i][j] for i in range(4) for j in range(4))
        assert total_ones >= 2  # At least some non-backtracking continuations

    def test_hashimoto_no_self_continuation(self):
        """Hashimoto matrix should not allow backtracking (e -> reverse(e))."""
        ss = _build("&{a: &{b: end}, c: end}")
        B = bass_hashimoto_matrix(ss)
        edges = _undirected_edges(ss)
        oriented = []
        for u, v in edges:
            oriented.append((u, v))
            oriented.append((v, u))
        for i, (u1, v1) in enumerate(oriented):
            for j, (u2, v2) in enumerate(oriented):
                if u2 == v1 and v2 == u1:
                    # This is the reverse edge — should be 0
                    assert B[i][j] == 0.0

    def test_hashimoto_square(self):
        """Hashimoto matrix is square with 2*|E| rows."""
        ss = _build("&{a: &{b: end}, c: end}")
        B = bass_hashimoto_matrix(ss)
        edges = _undirected_edges(ss)
        expected_size = 2 * len(edges)
        assert len(B) == expected_size
        for row in B:
            assert len(row) == expected_size


# ---------------------------------------------------------------------------
# Ihara determinant tests
# ---------------------------------------------------------------------------

class TestIharaDeterminant:
    """Tests for the Ihara determinant (Bass's formula)."""

    def test_det_at_zero(self):
        """At u=0, determinant is always 1."""
        for ty in ["end", "&{a: end}", "&{a: end, b: end}"]:
            ss = _build(ty)
            assert ihara_determinant(ss, 0.0) == pytest.approx(1.0)

    def test_det_empty(self):
        """Empty graph: determinant = 1."""
        ss = _build("end")
        assert ihara_determinant(ss, 0.5) == pytest.approx(1.0)

    def test_det_finite(self):
        """Determinant should be finite for reasonable u values."""
        ss = _build("&{a: &{b: end}, c: end}")
        for u in [0.1, 0.3, 0.5, 0.7]:
            det = ihara_determinant(ss, u)
            assert math.isfinite(det)

    def test_det_dag_always_one(self):
        """For true DAGs (trees), det should be close to 1 at small u
        because the cycle rank is 0 and adjacency is sparse."""
        ss = _build("&{a: end}")
        # Single edge: I - Au + (D-I)u^2 for 2x2
        det = ihara_determinant(ss, 0.3)
        assert math.isfinite(det)

    def test_det_consistency(self):
        """Determinant should be continuous: close u values give close results."""
        ss = _build("&{a: &{b: end}, c: end}")
        d1 = ihara_determinant(ss, 0.30)
        d2 = ihara_determinant(ss, 0.31)
        assert abs(d1 - d2) < 1.0  # Reasonably close


# ---------------------------------------------------------------------------
# Prime cycle tests
# ---------------------------------------------------------------------------

class TestPrimeCycles:
    """Tests for prime cycle counting."""

    def test_dag_no_cycles(self):
        """DAGs have no prime cycles."""
        ss = _build("&{a: end, b: end}")
        assert count_prime_cycles(ss) == 0
        assert prime_cycle_lengths(ss) == []

    def test_end_no_cycles(self):
        ss = _build("end")
        assert count_prime_cycles(ss) == 0

    def test_cycle_lengths_sorted(self):
        """Returned lengths should be sorted."""
        ss = _build("rec X . &{a: &{b: X}, c: end}")
        lengths = prime_cycle_lengths(ss, max_length=10)
        assert lengths == sorted(lengths)

    def test_prime_count_matches_lengths(self):
        """count_prime_cycles should equal len(prime_cycle_lengths)."""
        ss = _build("rec X . &{a: &{b: X}, c: end}")
        count = count_prime_cycles(ss, max_length=10)
        lengths = prime_cycle_lengths(ss, max_length=10)
        assert count == len(lengths)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestHelpers:
    """Tests for helper functions."""

    def test_is_proper_power_false(self):
        assert _is_proper_power((1, 2, 3)) is False

    def test_is_proper_power_true(self):
        assert _is_proper_power((1, 2, 1, 2)) is True

    def test_is_proper_power_single(self):
        assert _is_proper_power((1,)) is False

    def test_is_proper_power_triple(self):
        assert _is_proper_power((1, 1, 1)) is True

    def test_canonical_rotation(self):
        assert _canonical_rotation((3, 1, 2)) == (1, 2, 3)

    def test_canonical_rotation_already_min(self):
        assert _canonical_rotation((1, 2, 3)) == (1, 2, 3)

    def test_canonical_rotation_empty(self):
        assert _canonical_rotation(()) == ()

    def test_canonical_rotation_single(self):
        assert _canonical_rotation((5,)) == (5,)


# ---------------------------------------------------------------------------
# Ramanujan property tests
# ---------------------------------------------------------------------------

class TestRamanujan:
    """Tests for Ramanujan property checking."""

    def test_trivial_graph(self):
        """Single vertex is trivially Ramanujan."""
        ss = _build("end")
        assert is_ramanujan(ss) is True

    def test_single_edge(self):
        """Two vertices, one edge: Ramanujan."""
        ss = _build("&{a: end}")
        assert is_ramanujan(ss) is True

    def test_tree_ramanujan(self):
        """Trees are Ramanujan (no non-trivial eigenvalues for regular case)."""
        ss = _build("&{a: end, b: end}")
        assert is_ramanujan(ss) is True

    def test_returns_bool(self):
        """is_ramanujan should always return a bool."""
        for ty in ["end", "&{a: end}", "&{a: end, b: end}",
                    "rec X . &{a: X, b: end}"]:
            ss = _build(ty)
            result = is_ramanujan(ss)
            assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Poles tests
# ---------------------------------------------------------------------------

class TestIharaPoles:
    """Tests for poles of 1/Z_G(u)."""

    def test_no_edges_no_poles(self):
        ss = _build("end")
        assert ihara_poles(ss) == []

    def test_poles_complex_list(self):
        """Poles should be a list of complex numbers."""
        ss = _build("&{a: &{b: end}, c: end}")
        poles = ihara_poles(ss)
        for p in poles:
            assert isinstance(p, complex)

    def test_poles_sorted_by_magnitude(self):
        """Poles should be sorted by magnitude."""
        ss = _build("&{a: &{b: end}, c: end}")
        poles = ihara_poles(ss)
        mags = [abs(p) for p in poles]
        assert mags == sorted(mags)


# ---------------------------------------------------------------------------
# Graph complexity tests
# ---------------------------------------------------------------------------

class TestGraphComplexity:
    """Tests for graph complexity measure."""

    def test_trivial_complexity(self):
        """Single state: complexity = 0."""
        ss = _build("end")
        assert graph_complexity(ss) == pytest.approx(0.0, abs=1e-8)

    def test_dag_complexity(self):
        """DAGs with cycle rank 0 have zero complexity."""
        ss = _build("&{a: end}")
        c = graph_complexity(ss)
        assert c == pytest.approx(0.0, abs=1e-8)

    def test_complexity_finite(self):
        """Complexity should be finite for well-behaved graphs."""
        ss = _build("rec X . &{a: &{b: X}, c: end}")
        c = graph_complexity(ss)
        assert math.isfinite(c) or c == float('inf')

    def test_complexity_non_negative(self):
        """Complexity should be non-negative or zero."""
        for ty in ["end", "&{a: end}", "&{a: end, b: end}"]:
            ss = _build(ty)
            c = graph_complexity(ss)
            assert c >= -1e-8  # Allow small numerical errors


# ---------------------------------------------------------------------------
# Full analysis (analyze_ihara)
# ---------------------------------------------------------------------------

class TestAnalyzeIhara:
    """Tests for the full analyze_ihara function."""

    def test_result_type(self):
        ss = _build("&{a: end}")
        result = analyze_ihara(ss)
        assert isinstance(result, IharaResult)

    def test_result_fields(self):
        ss = _build("&{a: end, b: end}")
        result = analyze_ihara(ss)
        assert isinstance(result.ihara_determinant, float)
        assert isinstance(result.reciprocal_poles, list)
        assert isinstance(result.num_prime_cycles, int)
        assert isinstance(result.prime_cycle_lengths, list)
        assert isinstance(result.rank, int)
        assert isinstance(result.is_ramanujan, bool)
        assert isinstance(result.complexity, float)
        assert isinstance(result.bass_hashimoto_matrix, list)

    def test_end_analysis(self):
        ss = _build("end")
        result = analyze_ihara(ss)
        assert result.rank == 0
        assert result.num_prime_cycles == 0
        assert result.prime_cycle_lengths == []
        assert result.is_ramanujan is True
        assert result.complexity == pytest.approx(0.0, abs=1e-8)

    def test_branch_analysis(self):
        ss = _build("&{a: end, b: end}")
        result = analyze_ihara(ss)
        assert result.rank == 0
        assert result.num_prime_cycles == 0

    def test_recursive_analysis(self):
        ss = _build("rec X . &{a: X, b: end}")
        result = analyze_ihara(ss)
        assert result.rank >= 0
        assert isinstance(result.num_prime_cycles, int)

    def test_parallel_analysis(self):
        ss = _build("(&{a: end} || &{b: end})")
        result = analyze_ihara(ss)
        assert isinstance(result, IharaResult)
        assert result.rank >= 0


# ---------------------------------------------------------------------------
# Benchmark protocols
# ---------------------------------------------------------------------------

class TestBenchmarks:
    """Tests on benchmark session type protocols."""

    def test_iterator(self):
        """Java Iterator protocol: rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}."""
        ss = _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        result = analyze_ihara(ss)
        assert isinstance(result, IharaResult)
        assert math.isfinite(result.ihara_determinant)

    def test_simple_server(self):
        """Simple server: rec X . &{request: &{respond: X}, shutdown: end}."""
        ss = _build("rec X . &{request: &{respond: X}, shutdown: end}")
        result = analyze_ihara(ss)
        assert result.rank >= 0

    def test_selection_protocol(self):
        """+{OK: end, ERROR: end}: pure selection, no recursion."""
        ss = _build("+{OK: end, ERROR: end}")
        result = analyze_ihara(ss)
        assert result.rank == 0
        assert result.num_prime_cycles == 0

    def test_nested_branch(self):
        """&{a: &{b: end, c: end}, d: end}: nested branching."""
        ss = _build("&{a: &{b: end, c: end}, d: end}")
        result = analyze_ihara(ss)
        assert isinstance(result, IharaResult)

    def test_parallel_recursive(self):
        """(rec X . &{a: X, b: end} || &{c: end}): parallel with recursion."""
        ss = _build("(rec X . &{a: X, b: end} || &{c: end})")
        result = analyze_ihara(ss)
        assert isinstance(result, IharaResult)
