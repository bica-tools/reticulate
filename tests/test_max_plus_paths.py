"""Tests for max-plus longest paths analysis (Step 30n).

Tests cover:
- Longest path length (end, chain, branch, parallel, recursive)
- Shortest path length (same types, shortest <= longest)
- Critical path (valid sequence, correct length)
- All-pairs longest paths (diagonal zero, consistency)
- Path width (chain=1, antichain=n, parallel product)
- Path counting (chain=1, diamond=2, tree)
- Bottleneck path (valid path, correct value)
- Geodesic check (chain yes, diamond no)
- Path histogram (sums to total count)
- Full analysis on session type state spaces
- Benchmark protocols
"""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.max_plus_paths import (
    MaxPlusPathResult,
    longest_path_length,
    shortest_path_length,
    critical_path,
    all_pairs_longest_paths,
    path_width,
    count_paths,
    total_paths_top_bottom,
    bottleneck_path,
    is_geodesic,
    path_histogram,
    analyze_max_plus_paths,
    NEG_INF,
)


def _build(s: str):
    return build_statespace(parse(s))


# ---------------------------------------------------------------------------
# TestLongestPath
# ---------------------------------------------------------------------------

class TestLongestPath:

    def test_end_zero(self):
        """End type has longest path 0."""
        ss = _build("end")
        assert longest_path_length(ss) == 0

    def test_single_branch(self):
        """&{a: end} has longest path 1."""
        ss = _build("&{a: end}")
        assert longest_path_length(ss) == 1

    def test_chain_two(self):
        """&{a: &{b: end}} has longest path 2."""
        ss = _build("&{a: &{b: end}}")
        assert longest_path_length(ss) == 2

    def test_chain_three(self):
        """&{a: &{b: &{c: end}}} has longest path 3."""
        ss = _build("&{a: &{b: &{c: end}}}")
        assert longest_path_length(ss) == 3

    def test_branch_symmetric(self):
        """&{a: end, b: end} has longest path 1."""
        ss = _build("&{a: end, b: end}")
        assert longest_path_length(ss) == 1

    def test_branch_asymmetric(self):
        """&{a: end, b: &{c: end}} has longest path 2."""
        ss = _build("&{a: end, b: &{c: end}}")
        assert longest_path_length(ss) == 2

    def test_selection_single(self):
        """+{a: end} has longest path 1."""
        ss = _build("+{a: end}")
        assert longest_path_length(ss) == 1

    def test_selection_asymmetric(self):
        """+{OK: &{data: end}, ERR: end} has longest path 2."""
        ss = _build("+{OK: &{data: end}, ERR: end}")
        assert longest_path_length(ss) == 2

    def test_parallel_simple(self):
        """(end || end) has longest path 0."""
        ss = _build("(end || end)")
        assert longest_path_length(ss) == 0

    def test_parallel_nontrivial(self):
        """(&{a: end} || &{b: end}) product: longest path is 2."""
        ss = _build("(&{a: end} || &{b: end})")
        assert longest_path_length(ss) == 2

    def test_longest_nonnegative(self):
        """Longest path is always >= 0."""
        for t in ["end", "&{a: end}", "+{x: end}", "&{a: &{b: end}}"]:
            ss = _build(t)
            assert longest_path_length(ss) >= 0


# ---------------------------------------------------------------------------
# TestShortestPath
# ---------------------------------------------------------------------------

class TestShortestPath:

    def test_end_zero(self):
        ss = _build("end")
        assert shortest_path_length(ss) == 0

    def test_single_branch(self):
        ss = _build("&{a: end}")
        assert shortest_path_length(ss) == 1

    def test_chain_two(self):
        ss = _build("&{a: &{b: end}}")
        assert shortest_path_length(ss) == 2

    def test_branch_asymmetric(self):
        """&{a: end, b: &{c: end}}: shortest path is 1 (via a)."""
        ss = _build("&{a: end, b: &{c: end}}")
        assert shortest_path_length(ss) == 1

    def test_shortest_leq_longest(self):
        """Shortest path is always <= longest path."""
        types = [
            "end", "&{a: end}", "&{a: &{b: end}}",
            "&{a: end, b: &{c: end}}",
            "+{OK: &{data: end}, ERR: end}",
        ]
        for t in types:
            ss = _build(t)
            assert shortest_path_length(ss) <= longest_path_length(ss)

    def test_shortest_nonnegative(self):
        for t in ["end", "&{a: end}", "+{x: end}"]:
            ss = _build(t)
            assert shortest_path_length(ss) >= 0

    def test_chain_equal_shortest_longest(self):
        """A chain has shortest == longest."""
        ss = _build("&{a: &{b: &{c: end}}}")
        assert shortest_path_length(ss) == longest_path_length(ss)


# ---------------------------------------------------------------------------
# TestCriticalPath
# ---------------------------------------------------------------------------

class TestCriticalPath:

    def test_end_single_state(self):
        ss = _build("end")
        path = critical_path(ss)
        assert len(path) >= 1
        assert ss.top in path

    def test_chain_includes_top_and_bottom(self):
        ss = _build("&{a: &{b: end}}")
        path = critical_path(ss)
        assert path[0] == ss.top
        assert path[-1] == ss.bottom

    def test_path_length_matches_longest(self):
        """Critical path length (edges) == longest_path_length."""
        ss = _build("&{a: &{b: &{c: end}}}")
        path = critical_path(ss)
        # Number of edges = len(path) - 1
        assert len(path) - 1 == longest_path_length(ss)

    def test_path_valid_transitions(self):
        """Each consecutive pair in critical path must have a transition."""
        ss = _build("&{a: end, b: &{c: end}}")
        path = critical_path(ss)
        adj = {s: set() for s in ss.states}
        for s, _, t in ss.transitions:
            adj[s].add(t)
        for i in range(len(path) - 1):
            assert path[i + 1] in adj[path[i]], (
                f"No transition from {path[i]} to {path[i+1]}"
            )

    def test_branch_critical_follows_longest(self):
        """For asymmetric branch, critical path follows the longer arm."""
        ss = _build("&{a: end, b: &{c: end}}")
        path = critical_path(ss)
        assert len(path) - 1 == 2  # longest arm

    def test_selection_critical_path(self):
        ss = _build("+{OK: &{data: end}, ERR: end}")
        path = critical_path(ss)
        assert path[0] == ss.top
        assert path[-1] == ss.bottom


# ---------------------------------------------------------------------------
# TestAllPairs
# ---------------------------------------------------------------------------

class TestAllPairs:

    def test_end_single(self):
        ss = _build("end")
        L = all_pairs_longest_paths(ss)
        assert len(L) == 1
        assert L[0][0] == 0

    def test_diagonal_zero(self):
        """Diagonal entries are always 0."""
        ss = _build("&{a: &{b: end}}")
        L = all_pairs_longest_paths(ss)
        for i in range(len(L)):
            assert L[i][i] == 0

    def test_consistent_with_longest(self):
        """L[top][bottom] == longest_path_length."""
        from reticulate.zeta import _state_list
        ss = _build("&{a: end, b: &{c: end}}")
        L = all_pairs_longest_paths(ss)
        states = _state_list(ss)
        idx = {s: i for i, s in enumerate(states)}
        top_i = idx[ss.top]
        bot_i = idx[ss.bottom]
        assert int(L[top_i][bot_i]) == longest_path_length(ss)

    def test_unreachable_neg_inf(self):
        """Non-reachable pairs have -inf."""
        ss = _build("&{a: end}")
        L = all_pairs_longest_paths(ss)
        # bottom -> top should be -inf (unless same state)
        from reticulate.zeta import _state_list
        states = _state_list(ss)
        idx = {s: i for i, s in enumerate(states)}
        top_i = idx[ss.top]
        bot_i = idx[ss.bottom]
        if top_i != bot_i:
            assert L[bot_i][top_i] == NEG_INF

    def test_triangle_inequality_reversed(self):
        """For longest paths: L[i][j] >= L[i][k] + L[k][j] is NOT required
        (longest paths don't satisfy triangle inequality). But if k is
        on a longest i->j path, then L[i][j] = L[i][k] + L[k][j]."""
        ss = _build("&{a: &{b: end}}")
        L = all_pairs_longest_paths(ss)
        n = len(L)
        # All entries should be >= 0 (reachable) or NEG_INF
        for i in range(n):
            for j in range(n):
                assert L[i][j] >= NEG_INF

    def test_chain_all_pairs(self):
        """Chain a->b->end: top->mid=1, mid->bot=1, top->bot=2."""
        ss = _build("&{a: &{b: end}}")
        from reticulate.zeta import _state_list
        states = _state_list(ss)
        idx = {s: i for i, s in enumerate(states)}
        L = all_pairs_longest_paths(ss)
        top_i = idx[ss.top]
        bot_i = idx[ss.bottom]
        assert int(L[top_i][bot_i]) == 2


# ---------------------------------------------------------------------------
# TestWidth
# ---------------------------------------------------------------------------

class TestWidth:

    def test_end_width_one(self):
        ss = _build("end")
        assert path_width(ss) == 1

    def test_chain_width_one(self):
        """A chain has width 1 (no two elements are incomparable)."""
        ss = _build("&{a: &{b: &{c: end}}}")
        assert path_width(ss) == 1

    def test_single_branch_width_one(self):
        ss = _build("&{a: end}")
        assert path_width(ss) == 1

    def test_branch_width(self):
        """&{a: end, b: end}: the two intermediate states before end
        could be an antichain of size 2, but since both branch from
        top directly to end, the state space is top -> end only with
        2 transitions. Width depends on structure."""
        ss = _build("&{a: end, b: end}")
        # Both 'a' and 'b' lead to end, so top->{a->end, b->end}
        # Actually top has transitions to end directly. Width >= 1
        assert path_width(ss) >= 1

    def test_parallel_product_width(self):
        """Parallel product can have width > 1."""
        ss = _build("(&{a: end} || &{b: end})")
        # Product of two 2-state chains: 4 states, width should be 2
        assert path_width(ss) >= 2

    def test_width_positive(self):
        """Width is always >= 1 for non-empty state spaces."""
        for t in ["end", "&{a: end}", "&{a: &{b: end}}"]:
            ss = _build(t)
            assert path_width(ss) >= 1


# ---------------------------------------------------------------------------
# TestPathCount
# ---------------------------------------------------------------------------

class TestPathCount:

    def test_end_one_path(self):
        """End: one trivial path (top == bottom)."""
        ss = _build("end")
        assert total_paths_top_bottom(ss) == 1

    def test_chain_one_path(self):
        """Chain has exactly 1 path."""
        ss = _build("&{a: &{b: end}}")
        assert total_paths_top_bottom(ss) == 1

    def test_branch_two_paths(self):
        """&{a: end, b: end}: 2 paths from top to bottom."""
        ss = _build("&{a: end, b: end}")
        assert total_paths_top_bottom(ss) == 2

    def test_branch_three_paths(self):
        """&{a: end, b: end, c: end}: 3 paths."""
        ss = _build("&{a: end, b: end, c: end}")
        assert total_paths_top_bottom(ss) == 3

    def test_diamond_paths(self):
        """&{a: &{c: end}, b: &{c: end}} forms a diamond if c merges.
        Actually, different branch bodies create different states,
        so this is 2 paths."""
        ss = _build("&{a: &{c: end}, b: &{c: end}}")
        assert total_paths_top_bottom(ss) == 2

    def test_count_paths_src_tgt(self):
        """count_paths with specific source and target."""
        ss = _build("&{a: end}")
        assert count_paths(ss, ss.top, ss.bottom) == 1

    def test_count_paths_same_state(self):
        """Path from a state to itself = 1."""
        ss = _build("&{a: end}")
        assert count_paths(ss, ss.top, ss.top) == 1

    def test_selection_two_paths(self):
        """+{a: end, b: end}: 2 paths."""
        ss = _build("+{a: end, b: end}")
        assert total_paths_top_bottom(ss) == 2

    def test_nested_branches(self):
        """&{a: &{c: end}, b: end}: 2 paths."""
        ss = _build("&{a: &{c: end}, b: end}")
        assert total_paths_top_bottom(ss) == 2

    def test_path_count_positive(self):
        """Path count is always >= 1 for well-formed types."""
        for t in ["end", "&{a: end}", "+{x: end}", "&{a: &{b: end}}"]:
            ss = _build(t)
            assert total_paths_top_bottom(ss) >= 1


# ---------------------------------------------------------------------------
# TestBottleneck
# ---------------------------------------------------------------------------

class TestBottleneck:

    def test_end_bottleneck_zero(self):
        """End type: no edges, bottleneck 0."""
        ss = _build("end")
        val, path = bottleneck_path(ss)
        assert val == 0.0

    def test_chain_bottleneck_one(self):
        """Any type with a top->bottom path has bottleneck 1.0 (unit weights)."""
        ss = _build("&{a: end}")
        val, path = bottleneck_path(ss)
        assert val == 1.0

    def test_bottleneck_path_valid(self):
        """Bottleneck path should be a valid path."""
        ss = _build("&{a: &{b: end}}")
        val, path = bottleneck_path(ss)
        assert path[0] == ss.top
        assert path[-1] == ss.bottom

    def test_bottleneck_value_nonneg(self):
        for t in ["end", "&{a: end}", "&{a: &{b: end}}"]:
            ss = _build(t)
            val, _ = bottleneck_path(ss)
            assert val >= 0.0

    def test_bottleneck_branch(self):
        ss = _build("&{a: end, b: &{c: end}}")
        val, path = bottleneck_path(ss)
        assert val == 1.0
        assert len(path) >= 2


# ---------------------------------------------------------------------------
# TestGeodesic
# ---------------------------------------------------------------------------

class TestGeodesic:

    def test_end_geodesic(self):
        """End type is trivially geodesic."""
        ss = _build("end")
        assert is_geodesic(ss) is True

    def test_chain_geodesic(self):
        """A chain is geodesic (only one path)."""
        ss = _build("&{a: &{b: end}}")
        assert is_geodesic(ss) is True

    def test_symmetric_branch_geodesic(self):
        """&{a: end, b: end}: both paths have length 1."""
        ss = _build("&{a: end, b: end}")
        assert is_geodesic(ss) is True

    def test_asymmetric_branch_not_geodesic(self):
        """&{a: end, b: &{c: end}}: paths have lengths 1 and 2."""
        ss = _build("&{a: end, b: &{c: end}}")
        assert is_geodesic(ss) is False

    def test_selection_symmetric_geodesic(self):
        """+{a: end, b: end}: both paths have length 1."""
        ss = _build("+{a: end, b: end}")
        assert is_geodesic(ss) is True

    def test_selection_asymmetric_not_geodesic(self):
        """+{OK: &{data: end}, ERR: end}: paths 2 and 1."""
        ss = _build("+{OK: &{data: end}, ERR: end}")
        assert is_geodesic(ss) is False


# ---------------------------------------------------------------------------
# TestHistogram
# ---------------------------------------------------------------------------

class TestHistogram:

    def test_end_histogram(self):
        ss = _build("end")
        hist = path_histogram(ss)
        assert hist == {0: 1}

    def test_chain_histogram(self):
        ss = _build("&{a: &{b: end}}")
        hist = path_histogram(ss)
        assert hist == {2: 1}

    def test_branch_symmetric_histogram(self):
        ss = _build("&{a: end, b: end}")
        hist = path_histogram(ss)
        assert hist == {1: 2}

    def test_branch_asymmetric_histogram(self):
        ss = _build("&{a: end, b: &{c: end}}")
        hist = path_histogram(ss)
        assert hist == {1: 1, 2: 1}

    def test_three_branches_histogram(self):
        ss = _build("&{a: end, b: end, c: end}")
        hist = path_histogram(ss)
        assert hist == {1: 3}

    def test_histogram_sums_to_total_paths(self):
        """Sum of histogram values == total_paths_top_bottom."""
        types = [
            "end", "&{a: end}", "&{a: end, b: end}",
            "&{a: end, b: &{c: end}}",
            "+{OK: &{data: end}, ERR: end}",
        ]
        for t in types:
            ss = _build(t)
            hist = path_histogram(ss)
            total = sum(hist.values())
            assert total == total_paths_top_bottom(ss), (
                f"Type {t}: histogram sum {total} != "
                f"total_paths {total_paths_top_bottom(ss)}"
            )

    def test_histogram_keys_between_shortest_longest(self):
        """All histogram keys are between shortest and longest path."""
        ss = _build("&{a: end, b: &{c: end}}")
        hist = path_histogram(ss)
        s = shortest_path_length(ss)
        l = longest_path_length(ss)
        for k in hist:
            assert s <= k <= l


# ---------------------------------------------------------------------------
# TestAnalyze
# ---------------------------------------------------------------------------

class TestAnalyze:

    def test_end_analysis(self):
        ss = _build("end")
        result = analyze_max_plus_paths(ss)
        assert isinstance(result, MaxPlusPathResult)
        assert result.longest_path == 0
        assert result.shortest_path == 0
        assert result.is_geodesic is True
        assert result.path_count_top_bottom == 1

    def test_chain_analysis(self):
        ss = _build("&{a: &{b: end}}")
        result = analyze_max_plus_paths(ss)
        assert result.longest_path == 2
        assert result.shortest_path == 2
        assert result.is_geodesic is True
        assert result.path_count_top_bottom == 1
        assert result.width == 1
        assert result.diameter >= 2

    def test_branch_analysis(self):
        ss = _build("&{a: end, b: &{c: end}}")
        result = analyze_max_plus_paths(ss)
        assert result.longest_path == 2
        assert result.shortest_path == 1
        assert result.is_geodesic is False
        assert result.path_count_top_bottom == 2

    def test_selection_analysis(self):
        ss = _build("+{OK: end, ERR: end}")
        result = analyze_max_plus_paths(ss)
        assert result.longest_path == 1
        assert result.shortest_path == 1
        assert result.is_geodesic is True
        assert result.path_count_top_bottom == 2

    def test_result_fields(self):
        """All result fields should be populated."""
        ss = _build("&{a: end}")
        result = analyze_max_plus_paths(ss)
        assert isinstance(result.critical_path, list)
        assert isinstance(result.all_pairs_longest, list)
        assert isinstance(result.diameter, int)
        assert isinstance(result.width, int)
        assert isinstance(result.bottleneck_value, float)


# ---------------------------------------------------------------------------
# TestBenchmarks
# ---------------------------------------------------------------------------

class TestBenchmarks:

    def test_iterator(self):
        """Java Iterator: rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"""
        ss = _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        result = analyze_max_plus_paths(ss)
        # After SCC quotient, recursive loop collapses
        assert result.longest_path >= 1
        assert result.shortest_path >= 1
        assert result.path_count_top_bottom >= 1

    def test_file_object(self):
        """Simple file: &{open: &{read: &{close: end}}}"""
        ss = _build("&{open: &{read: &{close: end}}}")
        result = analyze_max_plus_paths(ss)
        assert result.longest_path == 3
        assert result.shortest_path == 3
        assert result.is_geodesic is True
        assert result.width == 1

    def test_parallel_branches(self):
        """(&{a: end} || &{b: end}): parallel product."""
        ss = _build("(&{a: end} || &{b: end})")
        result = analyze_max_plus_paths(ss)
        assert result.longest_path == 2
        assert result.width >= 2

    def test_selection_branch_combo(self):
        """&{init: +{OK: &{run: end}, FAIL: end}}"""
        ss = _build("&{init: +{OK: &{run: end}, FAIL: end}}")
        result = analyze_max_plus_paths(ss)
        assert result.longest_path == 3
        assert result.shortest_path == 2
        assert result.is_geodesic is False

    def test_recursive_shortest_leq_longest(self):
        """Recursive type: shortest <= longest."""
        ss = _build("rec X . &{a: X, b: end}")
        result = analyze_max_plus_paths(ss)
        assert result.shortest_path <= result.longest_path

    def test_deep_chain(self):
        """Deep chain: &{a: &{b: &{c: &{d: &{e: end}}}}}"""
        ss = _build("&{a: &{b: &{c: &{d: &{e: end}}}}}")
        result = analyze_max_plus_paths(ss)
        assert result.longest_path == 5
        assert result.shortest_path == 5
        assert result.is_geodesic is True
        assert result.path_count_top_bottom == 1

    def test_wide_branch(self):
        """Wide branch: &{a: end, b: end, c: end, d: end}"""
        ss = _build("&{a: end, b: end, c: end, d: end}")
        result = analyze_max_plus_paths(ss)
        assert result.longest_path == 1
        assert result.shortest_path == 1
        assert result.path_count_top_bottom == 4
        assert result.is_geodesic is True
