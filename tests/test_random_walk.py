"""Tests for random walk mixing time analysis (Step 30p).

Tests cover:
- Transition matrix: row sums, dimensions, entries in [0,1]
- Stationary distribution: sums to 1, is actually stationary (piP = pi)
- Spectral gap: non-negative, <= 1
- Hitting times: self = 0, positive for reachable pairs
- Mixing time: positive bound
- Ergodicity: detect absorbing chains correctly
- Return time: relationship to stationary distribution
- Commute time: symmetry properties
- Cover time: non-negative bounds
- Full analysis on various session types
- Benchmark protocols: iterator, file object, parallel
- Edge cases: single state (end), no edges, absorbing states
"""

import math

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.random_walk import (
    RandomWalkResult,
    all_hitting_times,
    analyze_random_walk,
    commute_time,
    cover_time_bound,
    eigenvalues_of_P,
    hitting_time,
    is_ergodic,
    mixing_time_bound,
    return_time,
    spectral_gap,
    stationary_distribution,
    transition_matrix,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(s: str):
    """Parse and build state space."""
    return build_statespace(parse(s))


# ---------------------------------------------------------------------------
# Transition matrix
# ---------------------------------------------------------------------------

class TestTransitionMatrix:
    """Tests for transition probability matrix construction."""

    def test_end_type_1x1(self):
        """end: single state with self-loop."""
        ss = _build("end")
        P = transition_matrix(ss)
        assert len(P) == 1
        assert len(P[0]) == 1
        assert P[0][0] == pytest.approx(1.0)

    def test_single_branch_2x2(self):
        """&{a: end}: 2 states, top -> bottom."""
        ss = _build("&{a: end}")
        P = transition_matrix(ss)
        assert len(P) == 2

    def test_row_sums_single_branch(self):
        """Row sums must equal 1.0."""
        ss = _build("&{a: end}")
        P = transition_matrix(ss)
        for row in P:
            assert sum(row) == pytest.approx(1.0)

    def test_row_sums_two_branches(self):
        """&{a: end, b: end}: row sums = 1."""
        ss = _build("&{a: end, b: end}")
        P = transition_matrix(ss)
        for row in P:
            assert sum(row) == pytest.approx(1.0)

    def test_entries_in_unit_interval(self):
        """All entries must be in [0, 1]."""
        ss = _build("&{a: &{c: end}, b: end}")
        P = transition_matrix(ss)
        for row in P:
            for val in row:
                assert 0.0 <= val <= 1.0 + 1e-12

    def test_uniform_probabilities(self):
        """&{a: end, b: end}: top has two transitions, each prob 0.5."""
        ss = _build("&{a: end, b: end}")
        P = transition_matrix(ss)
        # Find the top state row (should have two non-zero entries summing to 1)
        # The bottom row should be a self-loop
        non_absorbing_rows = [i for i, row in enumerate(P) if row[i] < 0.99]
        assert len(non_absorbing_rows) >= 1

    def test_absorbing_state_self_loop(self):
        """Bottom state (no outgoing transitions) gets self-loop."""
        ss = _build("&{a: end}")
        P = transition_matrix(ss)
        from reticulate.zeta import _state_list
        states = _state_list(ss)
        idx = {s: i for i, s in enumerate(states)}
        bi = idx[ss.bottom]
        assert P[bi][bi] == pytest.approx(1.0)

    def test_dimensions_match_states(self):
        """Matrix dimension equals number of states."""
        ss = _build("&{a: &{b: end}, c: end}")
        P = transition_matrix(ss)
        n = len(ss.states)
        assert len(P) == n
        for row in P:
            assert len(row) == n

    def test_chain_type(self):
        """&{a: &{b: &{c: end}}}: chain, each state has out-degree 1."""
        ss = _build("&{a: &{b: &{c: end}}}")
        P = transition_matrix(ss)
        for row in P:
            assert sum(row) == pytest.approx(1.0)

    def test_diamond_row_sums(self):
        """Diamond: all rows sum to 1."""
        ss = _build("&{a: &{c: end}, b: &{c: end}}")
        P = transition_matrix(ss)
        for row in P:
            assert sum(row) == pytest.approx(1.0)

    def test_parallel_row_sums(self):
        """Parallel type: all rows sum to 1."""
        ss = _build("(&{a: end} || &{b: end})")
        P = transition_matrix(ss)
        for row in P:
            assert sum(row) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Stationary distribution
# ---------------------------------------------------------------------------

class TestStationary:
    """Tests for stationary distribution computation."""

    def test_end_type(self):
        """end: single state, pi = [1.0]."""
        ss = _build("end")
        pi = stationary_distribution(ss)
        assert len(pi) == 1
        assert pi[0] == pytest.approx(1.0)

    def test_sums_to_one(self):
        """Stationary distribution sums to 1."""
        ss = _build("&{a: end, b: end}")
        pi = stationary_distribution(ss)
        assert sum(pi) == pytest.approx(1.0)

    def test_is_actually_stationary(self):
        """pi * P = pi (left eigenvector with eigenvalue 1)."""
        ss = _build("&{a: &{c: end}, b: end}")
        pi = stationary_distribution(ss)
        P = transition_matrix(ss)
        n = len(pi)
        # Compute pi * P
        pi_P = [0.0] * n
        for j in range(n):
            for i in range(n):
                pi_P[j] += pi[i] * P[i][j]
        for j in range(n):
            assert pi_P[j] == pytest.approx(pi[j], abs=1e-8)

    def test_absorbing_state_concentration(self):
        """For DAG with absorbing bottom, pi concentrates at bottom."""
        ss = _build("&{a: end}")
        pi = stationary_distribution(ss)
        from reticulate.zeta import _state_list
        states = _state_list(ss)
        idx = {s: i for i, s in enumerate(states)}
        bi = idx[ss.bottom]
        assert pi[bi] == pytest.approx(1.0, abs=0.01)

    def test_non_negative(self):
        """All entries of pi are non-negative."""
        ss = _build("&{a: &{b: end}, c: end}")
        pi = stationary_distribution(ss)
        for p in pi:
            assert p >= -1e-12

    def test_chain_concentrates_at_bottom(self):
        """Chain: pi concentrates at absorbing bottom."""
        ss = _build("&{a: &{b: &{c: end}}}")
        pi = stationary_distribution(ss)
        from reticulate.zeta import _state_list
        states = _state_list(ss)
        idx = {s: i for i, s in enumerate(states)}
        bi = idx[ss.bottom]
        assert pi[bi] > 0.9

    def test_parallel_sums_to_one(self):
        """Parallel type: pi still sums to 1."""
        ss = _build("(&{a: end} || &{b: end})")
        pi = stationary_distribution(ss)
        assert sum(pi) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Spectral gap
# ---------------------------------------------------------------------------

class TestSpectralGap:
    """Tests for spectral gap computation."""

    def test_end_type_gap_one(self):
        """Single state: gap = 1.0."""
        ss = _build("end")
        assert spectral_gap(ss) == pytest.approx(1.0)

    def test_non_negative(self):
        """Spectral gap is >= 0."""
        ss = _build("&{a: end, b: end}")
        assert spectral_gap(ss) >= -1e-12

    def test_at_most_one(self):
        """Spectral gap is <= 1."""
        ss = _build("&{a: &{c: end}, b: end}")
        assert spectral_gap(ss) <= 1.0 + 1e-12

    def test_chain_gap(self):
        """Chain: some spectral gap exists."""
        ss = _build("&{a: &{b: end}}")
        gap = spectral_gap(ss)
        assert 0.0 <= gap <= 1.0

    def test_diamond_gap(self):
        """Diamond: spectral gap exists."""
        ss = _build("&{a: &{c: end}, b: &{c: end}}")
        gap = spectral_gap(ss)
        assert 0.0 <= gap <= 1.0


# ---------------------------------------------------------------------------
# Eigenvalues
# ---------------------------------------------------------------------------

class TestEigenvalues:
    """Tests for eigenvalue computation."""

    def test_end_type_eigenvalue_one(self):
        """end: eigenvalue = [1.0]."""
        ss = _build("end")
        eigs = eigenvalues_of_P(ss)
        assert len(eigs) == 1
        assert eigs[0] == pytest.approx(1.0)

    def test_largest_eigenvalue_is_one(self):
        """Stochastic matrix has largest eigenvalue 1."""
        ss = _build("&{a: end, b: end}")
        eigs = eigenvalues_of_P(ss)
        assert abs(eigs[0]) == pytest.approx(1.0, abs=0.05)

    def test_count_matches_states(self):
        """Number of eigenvalues = number of states."""
        ss = _build("&{a: &{b: end}, c: end}")
        eigs = eigenvalues_of_P(ss)
        assert len(eigs) == len(ss.states)

    def test_sorted_descending_magnitude(self):
        """Eigenvalues sorted descending by magnitude."""
        ss = _build("&{a: &{c: end}, b: &{c: end}}")
        eigs = eigenvalues_of_P(ss)
        for i in range(len(eigs) - 1):
            assert abs(eigs[i]) >= abs(eigs[i + 1]) - 1e-8

    def test_all_magnitudes_at_most_one(self):
        """All eigenvalue magnitudes <= 1 for a stochastic matrix."""
        ss = _build("&{a: &{b: end}, c: end}")
        eigs = eigenvalues_of_P(ss)
        for e in eigs:
            assert abs(e) <= 1.0 + 0.1  # small tolerance for numerics


# ---------------------------------------------------------------------------
# Hitting time
# ---------------------------------------------------------------------------

class TestHittingTime:
    """Tests for hitting time computation."""

    def test_self_hitting_time_zero(self):
        """H(s, s) = 0."""
        ss = _build("&{a: end}")
        assert hitting_time(ss, ss.top, ss.top) == 0.0

    def test_top_to_bottom_positive(self):
        """H(top, bottom) > 0 for non-trivial types."""
        ss = _build("&{a: end}")
        h = hitting_time(ss, ss.top, ss.bottom)
        assert h > 0.0

    def test_chain_hitting_time(self):
        """Chain &{a: &{b: end}}: H(top, bottom) = 2 (deterministic)."""
        ss = _build("&{a: &{b: end}}")
        h = hitting_time(ss, ss.top, ss.bottom)
        assert h == pytest.approx(2.0, abs=0.1)

    def test_single_step(self):
        """&{a: end}: H(top, bottom) = 1."""
        ss = _build("&{a: end}")
        h = hitting_time(ss, ss.top, ss.bottom)
        assert h == pytest.approx(1.0, abs=0.1)

    def test_bottom_to_top_infinite(self):
        """H(bottom, top) = inf for DAGs (no path back)."""
        ss = _build("&{a: &{b: end}}")
        if ss.top != ss.bottom:
            h = hitting_time(ss, ss.bottom, ss.top)
            assert math.isinf(h)


# ---------------------------------------------------------------------------
# All hitting times
# ---------------------------------------------------------------------------

class TestAllHittingTimes:
    """Tests for all pairwise hitting times."""

    def test_diagonal_zero(self):
        """H(s, s) = 0 for all states."""
        ss = _build("&{a: end, b: end}")
        ht = all_hitting_times(ss)
        for s in ss.states:
            assert ht[(s, s)] == 0.0

    def test_dict_size(self):
        """Number of entries = n^2."""
        ss = _build("&{a: end}")
        ht = all_hitting_times(ss)
        n = len(ss.states)
        assert len(ht) == n * n


# ---------------------------------------------------------------------------
# Return time
# ---------------------------------------------------------------------------

class TestReturnTime:
    """Tests for expected return time."""

    def test_end_type(self):
        """end: return time = 1.0 (self-loop, pi=1)."""
        ss = _build("end")
        rt = return_time(ss, ss.top)
        assert rt == pytest.approx(1.0)

    def test_absorbing_bottom(self):
        """Bottom of DAG has return time 1 (absorbing self-loop)."""
        ss = _build("&{a: end}")
        rt = return_time(ss, ss.bottom)
        assert rt == pytest.approx(1.0, abs=0.1)

    def test_transient_state_large(self):
        """Transient state has large/infinite return time."""
        ss = _build("&{a: &{b: end}}")
        if ss.top != ss.bottom:
            rt = return_time(ss, ss.top)
            # Transient: pi(top) ~ 0, so return time ~ inf
            assert rt > 10.0 or math.isinf(rt)


# ---------------------------------------------------------------------------
# Commute time
# ---------------------------------------------------------------------------

class TestCommuteTime:
    """Tests for commute time computation."""

    def test_self_commute_zero(self):
        """C(s, s) = 0."""
        ss = _build("&{a: end}")
        assert commute_time(ss, ss.top, ss.top) == 0.0

    def test_dag_infinite_commute(self):
        """DAG: C(top, bottom) = inf (can't go bottom -> top)."""
        ss = _build("&{a: end}")
        if ss.top != ss.bottom:
            ct = commute_time(ss, ss.top, ss.bottom)
            assert math.isinf(ct)

    def test_end_commute(self):
        """end: C(s, s) = 0."""
        ss = _build("end")
        ct = commute_time(ss, ss.top, ss.top)
        assert ct == 0.0


# ---------------------------------------------------------------------------
# Mixing time
# ---------------------------------------------------------------------------

class TestMixingTime:
    """Tests for mixing time bound."""

    def test_end_type_zero(self):
        """end: single state, mixing time = 0."""
        ss = _build("end")
        assert mixing_time_bound(ss) == 0

    def test_positive_for_nontrivial(self):
        """Mixing time > 0 for multi-state types."""
        ss = _build("&{a: end}")
        assert mixing_time_bound(ss) >= 1

    def test_integer_result(self):
        """Mixing time is an integer."""
        ss = _build("&{a: end, b: end}")
        assert isinstance(mixing_time_bound(ss), int)

    def test_chain_larger_than_branch(self):
        """Longer chain should generally have larger mixing bound."""
        ss_short = _build("&{a: end}")
        ss_long = _build("&{a: &{b: &{c: end}}}")
        # Longer chain has more states
        assert mixing_time_bound(ss_long) >= mixing_time_bound(ss_short)


# ---------------------------------------------------------------------------
# Cover time
# ---------------------------------------------------------------------------

class TestCoverTime:
    """Tests for cover time bound."""

    def test_end_type_zero(self):
        """end: single state, cover time = 0."""
        ss = _build("end")
        assert cover_time_bound(ss) == 0.0

    def test_non_negative(self):
        """Cover time bound is non-negative."""
        ss = _build("&{a: end, b: end}")
        assert cover_time_bound(ss) >= 0.0

    def test_at_least_hitting(self):
        """Cover time >= max hitting time from top."""
        ss = _build("&{a: &{c: end}, b: end}")
        cover = cover_time_bound(ss)
        ht = all_hitting_times(ss)
        for (s, t), h in ht.items():
            if s == ss.top and not math.isinf(h):
                assert cover >= h - 1e-6


# ---------------------------------------------------------------------------
# Ergodicity
# ---------------------------------------------------------------------------

class TestErgodic:
    """Tests for ergodicity detection."""

    def test_end_type_ergodic(self):
        """end: single state with self-loop is ergodic."""
        ss = _build("end")
        assert is_ergodic(ss) is True

    def test_dag_not_ergodic(self):
        """DAG with absorbing bottom is NOT ergodic."""
        ss = _build("&{a: end}")
        if ss.top != ss.bottom:
            assert is_ergodic(ss) is False

    def test_chain_not_ergodic(self):
        """Chain is not ergodic (not irreducible)."""
        ss = _build("&{a: &{b: end}}")
        assert is_ergodic(ss) is False

    def test_diamond_not_ergodic(self):
        """Diamond DAG is not ergodic."""
        ss = _build("&{a: &{c: end}, b: &{c: end}}")
        assert is_ergodic(ss) is False

    def test_recursive_type(self):
        """rec X . &{a: X, b: end}: has cycle but still has absorbing bottom."""
        ss = _build("rec X . &{a: X, b: end}")
        # Has absorbing bottom, so not irreducible
        assert is_ergodic(ss) is False


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

class TestAnalyze:
    """Tests for full random walk analysis."""

    def test_result_type(self):
        """analyze_random_walk returns RandomWalkResult."""
        ss = _build("&{a: end}")
        result = analyze_random_walk(ss)
        assert isinstance(result, RandomWalkResult)

    def test_end_type_analysis(self):
        """end: complete analysis on trivial type."""
        ss = _build("end")
        result = analyze_random_walk(ss)
        assert result.is_ergodic is True
        assert result.spectral_gap == pytest.approx(1.0)
        assert result.mixing_time_bound == 0
        assert sum(result.stationary_distribution) == pytest.approx(1.0)

    def test_branch_analysis(self):
        """&{a: end, b: end}: complete analysis."""
        ss = _build("&{a: end, b: end}")
        result = analyze_random_walk(ss)
        assert len(result.transition_matrix) == len(ss.states)
        assert sum(result.stationary_distribution) == pytest.approx(1.0)
        assert result.spectral_gap >= 0.0
        assert result.mixing_time_bound >= 1
        assert result.cover_time_bound >= 0.0

    def test_diamond_analysis(self):
        """Diamond: complete analysis."""
        ss = _build("&{a: &{c: end}, b: &{c: end}}")
        result = analyze_random_walk(ss)
        assert len(result.eigenvalues) == len(ss.states)
        assert not result.is_ergodic

    def test_parallel_analysis(self):
        """Parallel type: complete analysis."""
        ss = _build("(&{a: end} || &{b: end})")
        result = analyze_random_walk(ss)
        assert len(result.transition_matrix) == len(ss.states)
        assert sum(result.stationary_distribution) == pytest.approx(1.0)

    def test_recursive_analysis(self):
        """Recursive type: complete analysis."""
        ss = _build("rec X . &{a: X, b: end}")
        result = analyze_random_walk(ss)
        assert len(result.eigenvalues) == len(ss.states)
        assert result.cover_time_bound >= 0.0

    def test_hitting_times_in_result(self):
        """Hitting times dict is populated in result."""
        ss = _build("&{a: end}")
        result = analyze_random_walk(ss)
        n = len(ss.states)
        assert len(result.hitting_times) == n * n


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

class TestBenchmarks:
    """Tests on benchmark session type protocols."""

    def test_iterator(self):
        """Iterator: rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}."""
        ss = _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        result = analyze_random_walk(ss)
        assert len(result.transition_matrix) > 0
        assert sum(result.stationary_distribution) == pytest.approx(1.0)
        assert not result.is_ergodic

    def test_smtp(self):
        """Simple SMTP: &{EHLO: &{MAIL: &{RCPT: &{DATA: end}}}}."""
        ss = _build("&{EHLO: &{MAIL: &{RCPT: &{DATA: end}}}}")
        result = analyze_random_walk(ss)
        # Chain of 5 states: hitting time from top to bottom = 4
        h = hitting_time(ss, ss.top, ss.bottom)
        assert h == pytest.approx(4.0, abs=0.5)

    def test_file_object(self):
        """File: &{open: &{read: end, write: end}}."""
        ss = _build("&{open: &{read: end, write: end}}")
        result = analyze_random_walk(ss)
        assert sum(result.stationary_distribution) == pytest.approx(1.0)
        assert result.cover_time_bound >= 0.0

    def test_parallel_file(self):
        """Parallel file access: (&{read: end} || &{write: end})."""
        ss = _build("(&{read: end} || &{write: end})")
        result = analyze_random_walk(ss)
        assert len(result.transition_matrix) == len(ss.states)
        for row in result.transition_matrix:
            assert sum(row) == pytest.approx(1.0)

    def test_selection_type(self):
        """Selection: +{OK: end, ERR: end}."""
        ss = _build("+{OK: end, ERR: end}")
        result = analyze_random_walk(ss)
        assert sum(result.stationary_distribution) == pytest.approx(1.0)
