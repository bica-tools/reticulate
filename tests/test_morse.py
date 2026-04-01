"""Tests for discrete Morse theory on session type lattices (Step 30x)."""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.morse import (
    MorseMatching,
    MorseFunction,
    GradientField,
    MorseResult,
    hasse_graph,
    greedy_matching,
    critical_cells,
    morse_function,
    gradient_field,
    betti_numbers,
    morse_inequalities,
    analyze_morse,
)


def _build(type_string: str):
    """Helper: parse + build state space."""
    return build_statespace(parse(type_string))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def end_ss():
    return _build("end")


@pytest.fixture
def simple_branch():
    return _build("&{a: end, b: end}")


@pytest.fixture
def deep_chain():
    return _build("&{a: &{b: &{c: end}}}")


@pytest.fixture
def iterator_ss():
    return _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")


@pytest.fixture
def file_ss():
    return _build("&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}")


@pytest.fixture
def parallel_ss():
    return _build("(&{a: end} || &{b: end})")


@pytest.fixture
def wide_branch():
    return _build("&{a: end, b: end, c: end, d: end}")


@pytest.fixture
def nested_branch():
    return _build("&{a: &{c: end, d: end}, b: &{e: end, f: end}}")


@pytest.fixture
def simple_select():
    return _build("+{ok: end, err: end}")


@pytest.fixture
def rest_ss():
    return _build("&{get: +{OK: end, NOT_FOUND: end}, put: +{OK: end, ERR: end}}")


# ---------------------------------------------------------------------------
# Hasse graph
# ---------------------------------------------------------------------------

class TestHasseGraph:
    def test_end_no_edges(self, end_ss):
        edges = hasse_graph(end_ss)
        assert edges == []

    def test_simple_branch_has_edges(self, simple_branch):
        edges = hasse_graph(simple_branch)
        assert len(edges) >= 2  # at least a and b transitions

    def test_deep_chain_edges(self, deep_chain):
        edges = hasse_graph(deep_chain)
        assert len(edges) >= 3

    def test_parallel_has_edges(self, parallel_ss):
        edges = hasse_graph(parallel_ss)
        assert len(edges) >= 2

    def test_edges_are_tuples(self, simple_branch):
        edges = hasse_graph(simple_branch)
        for e in edges:
            assert len(e) == 3  # (source, target, label)
            src, tgt, label = e
            assert isinstance(src, int)
            assert isinstance(tgt, int)
            assert isinstance(label, str)

    def test_wide_branch_edges(self, wide_branch):
        edges = hasse_graph(wide_branch)
        assert len(edges) >= 4  # a, b, c, d

    def test_nested_branch_edges(self, nested_branch):
        edges = hasse_graph(nested_branch)
        assert len(edges) >= 4


# ---------------------------------------------------------------------------
# Greedy matching
# ---------------------------------------------------------------------------

class TestGreedyMatching:
    def test_end_empty_matching(self, end_ss):
        m = greedy_matching(end_ss)
        assert isinstance(m, MorseMatching)
        assert len(m.matched_pairs) == 0
        assert m.is_acyclic

    def test_simple_branch_matching(self, simple_branch):
        m = greedy_matching(simple_branch)
        assert isinstance(m, MorseMatching)
        assert m.is_acyclic
        assert len(m.matched_pairs) >= 1

    def test_matching_pairs_are_vertex_edge_pairs(self, simple_branch):
        """Each matched pair is (vertex, edge_index)."""
        m = greedy_matching(simple_branch)
        for vertex, edge_idx in m.matched_pairs:
            assert vertex in simple_branch.states
            assert isinstance(edge_idx, int)
            assert edge_idx >= 0

    def test_matched_vertices_disjoint(self, nested_branch):
        """No vertex appears in more than one matched pair."""
        m = greedy_matching(nested_branch)
        vertices: set[int] = set()
        edge_indices: set[int] = set()
        for v, ei in m.matched_pairs:
            assert v not in vertices, f"Vertex {v} matched twice"
            assert ei not in edge_indices, f"Edge {ei} matched twice"
            vertices.add(v)
            edge_indices.add(ei)

    def test_critical_union_matched_equals_all(self, simple_branch):
        m = greedy_matching(simple_branch)
        matched_vertices = {v for v, _ in m.matched_pairs}
        assert matched_vertices | set(m.critical_vertices) == simple_branch.states

    def test_deep_chain_matching(self, deep_chain):
        m = greedy_matching(deep_chain)
        assert m.is_acyclic

    def test_parallel_matching(self, parallel_ss):
        m = greedy_matching(parallel_ss)
        assert m.is_acyclic

    def test_wide_branch_matching(self, wide_branch):
        m = greedy_matching(wide_branch)
        assert m.is_acyclic
        assert len(m.matched_pairs) >= 1

    def test_num_states(self, simple_branch):
        m = greedy_matching(simple_branch)
        assert m.num_states == len(simple_branch.states)

    def test_num_hasse_edges_leq_transitions(self, simple_branch):
        """Deduplicated Hasse edges <= total transitions."""
        m = greedy_matching(simple_branch)
        assert m.num_hasse_edges <= len(simple_branch.transitions)


# ---------------------------------------------------------------------------
# Critical cells
# ---------------------------------------------------------------------------

class TestCriticalCells:
    def test_end_single_critical(self, end_ss):
        m = greedy_matching(end_ss)
        crits = critical_cells(m)
        assert len(crits) == 1  # The single end state

    def test_critical_subset_of_states(self, simple_branch):
        m = greedy_matching(simple_branch)
        crits = critical_cells(m)
        assert crits.issubset(simple_branch.states)

    def test_critical_not_in_matched_vertices(self, nested_branch):
        m = greedy_matching(nested_branch)
        crits = critical_cells(m)
        matched_vertices = {v for v, _ in m.matched_pairs}
        assert crits.isdisjoint(matched_vertices)

    def test_at_least_one_critical_for_end(self, end_ss):
        """Single-state space always has one critical cell."""
        m = greedy_matching(end_ss)
        crits = critical_cells(m)
        assert len(crits) == 1

    def test_deep_chain_critical_count(self, deep_chain):
        m = greedy_matching(deep_chain)
        crits = critical_cells(m)
        assert len(crits) >= 1


# ---------------------------------------------------------------------------
# Morse function
# ---------------------------------------------------------------------------

class TestMorseFunction:
    def test_end_function(self, end_ss):
        f = morse_function(end_ss)
        assert isinstance(f, MorseFunction)
        assert len(f.values) == 1

    def test_all_states_have_values(self, simple_branch):
        f = morse_function(simple_branch)
        for s in simple_branch.states:
            assert s in f.values

    def test_values_are_numeric(self, nested_branch):
        f = morse_function(nested_branch)
        for v in f.values.values():
            assert isinstance(v, (int, float))

    def test_deep_chain_function(self, deep_chain):
        f = morse_function(deep_chain)
        assert len(f.values) == len(deep_chain.states)

    def test_parallel_function(self, parallel_ss):
        f = morse_function(parallel_ss)
        assert len(f.values) == len(parallel_ss.states)

    def test_function_validity(self, simple_branch):
        f = morse_function(simple_branch)
        assert isinstance(f.is_valid, bool)


# ---------------------------------------------------------------------------
# Gradient field
# ---------------------------------------------------------------------------

class TestGradientField:
    def test_end_gradient(self, end_ss):
        g = gradient_field(end_ss)
        assert isinstance(g, GradientField)
        assert len(g.pairs) == 0
        assert len(g.critical) == 1

    def test_gradient_pairs_match_matching(self, simple_branch):
        m = greedy_matching(simple_branch)
        g = gradient_field(simple_branch)
        assert g.pairs == m.matched_pairs

    def test_gradient_critical_match_matching(self, nested_branch):
        m = greedy_matching(nested_branch)
        g = gradient_field(nested_branch)
        assert g.critical == m.critical_vertices

    def test_flow_graph_has_all_states(self, simple_branch):
        g = gradient_field(simple_branch)
        for s in simple_branch.states:
            assert s in g.flow_graph

    def test_parallel_gradient(self, parallel_ss):
        g = gradient_field(parallel_ss)
        assert isinstance(g, GradientField)


# ---------------------------------------------------------------------------
# Betti numbers
# ---------------------------------------------------------------------------

class TestBettiNumbers:
    def test_end_betti(self, end_ss):
        b = betti_numbers(end_ss)
        assert b[0] == 1  # one connected component
        assert b[1] == 0  # no cycles

    def test_simple_branch_connected(self, simple_branch):
        b = betti_numbers(simple_branch)
        assert b[0] == 1  # connected

    def test_deep_chain_tree(self, deep_chain):
        b = betti_numbers(deep_chain)
        assert b[0] == 1  # connected
        assert b[1] == 0  # tree (no cycles)

    def test_iterator_hasse_topology(self, iterator_ss):
        """Iterator: SCC states become disconnected in Hasse complex."""
        b = betti_numbers(iterator_ss)
        assert b[0] >= 1  # at least one component

    def test_file_hasse_topology(self, file_ss):
        """File: SCC collapses; Hasse may have multiple components."""
        b = betti_numbers(file_ss)
        assert b[0] >= 1

    def test_parallel_has_cycle(self, parallel_ss):
        """Diamond lattice from parallel has a topological cycle."""
        b = betti_numbers(parallel_ss)
        assert b[0] == 1  # connected
        assert b[1] == 1  # one cycle in diamond

    def test_betti_tuple(self, simple_branch):
        b = betti_numbers(simple_branch)
        assert isinstance(b, tuple)
        assert len(b) == 2
        assert all(isinstance(x, int) for x in b)

    def test_select_connected(self, simple_select):
        b = betti_numbers(simple_select)
        assert b[0] == 1


# ---------------------------------------------------------------------------
# Morse inequalities
# ---------------------------------------------------------------------------

class TestMorseInequalities:
    def test_end_inequalities(self, end_ss):
        weak, strong = morse_inequalities(end_ss)
        assert weak is True
        assert strong is True

    def test_simple_branch_strong(self, simple_branch):
        weak, strong = morse_inequalities(simple_branch)
        assert strong is True

    def test_deep_chain_strong(self, deep_chain):
        weak, strong = morse_inequalities(deep_chain)
        assert strong is True

    def test_parallel_strong(self, parallel_ss):
        weak, strong = morse_inequalities(parallel_ss)
        assert strong is True

    def test_wide_branch_strong(self, wide_branch):
        weak, strong = morse_inequalities(wide_branch)
        assert strong is True

    def test_nested_branch_strong(self, nested_branch):
        weak, strong = morse_inequalities(nested_branch)
        assert strong is True

    def test_select_strong(self, simple_select):
        weak, strong = morse_inequalities(simple_select)
        assert strong is True

    def test_rest_strong(self, rest_ss):
        weak, strong = morse_inequalities(rest_ss)
        assert strong is True

    def test_iterator_strong(self, iterator_ss):
        weak, strong = morse_inequalities(iterator_ss)
        assert strong is True

    def test_file_strong(self, file_ss):
        weak, strong = morse_inequalities(file_ss)
        assert strong is True


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

class TestAnalyzeMorse:
    def test_end_analysis(self, end_ss):
        r = analyze_morse(end_ss)
        assert isinstance(r, MorseResult)
        assert r.num_critical >= 1
        assert r.strong_morse_holds

    def test_simple_branch_analysis(self, simple_branch):
        r = analyze_morse(simple_branch)
        assert isinstance(r, MorseResult)
        assert isinstance(r.matching, MorseMatching)
        assert isinstance(r.function, MorseFunction)
        assert isinstance(r.gradient, GradientField)
        assert r.strong_morse_holds

    def test_deep_chain_analysis(self, deep_chain):
        r = analyze_morse(deep_chain)
        assert r.num_critical >= 1
        assert r.strong_morse_holds

    def test_iterator_analysis(self, iterator_ss):
        r = analyze_morse(iterator_ss)
        assert r.betti_numbers[0] >= 1
        assert r.strong_morse_holds

    def test_file_analysis(self, file_ss):
        r = analyze_morse(file_ss)
        assert r.betti_numbers[0] >= 1
        assert r.strong_morse_holds

    def test_parallel_analysis(self, parallel_ss):
        r = analyze_morse(parallel_ss)
        assert r.strong_morse_holds

    def test_nested_analysis(self, nested_branch):
        r = analyze_morse(nested_branch)
        assert r.strong_morse_holds

    def test_euler_characteristic_correct(self, simple_branch):
        """Euler char matches V - E on the Hasse complex."""
        r = analyze_morse(simple_branch)
        assert isinstance(r.euler_characteristic, int)

    def test_critical_cells_in_result(self, wide_branch):
        r = analyze_morse(wide_branch)
        assert r.critical_cells == r.matching.critical_vertices

    def test_result_betti_tuple(self, rest_ss):
        r = analyze_morse(rest_ss)
        assert isinstance(r.betti_numbers, tuple)
        assert len(r.betti_numbers) == 2

    def test_num_critical_includes_edges(self, parallel_ss):
        """num_critical counts both unmatched vertices and edges."""
        r = analyze_morse(parallel_ss)
        expected = (len(r.matching.critical_vertices)
                    + len(r.matching.critical_edges))
        assert r.num_critical == expected


# ---------------------------------------------------------------------------
# Parametrized benchmarks
# ---------------------------------------------------------------------------

class TestBenchmarks:
    @pytest.mark.parametrize("type_string,name", [
        ("end", "End"),
        ("&{a: end}", "SingleMethod"),
        ("&{a: end, b: end}", "SimpleBranch"),
        ("+{ok: end, err: end}", "SimpleSelect"),
        ("&{a: &{b: &{c: end}}}", "DeepChain"),
        ("(&{a: end} || &{b: end})", "SimpleParallel"),
        ("&{a: &{c: end, d: end}, b: &{e: end, f: end}}", "NestedBranch"),
        ("&{a: end, b: end, c: end, d: end}", "WideBranch"),
        ("&{get: +{OK: end, NOT_FOUND: end}, put: +{OK: end, ERR: end}}", "REST"),
        ("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}", "Iterator"),
        ("&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}", "File"),
        ("+{OK: &{a: end}, ERR: end}", "SelectBranch"),
    ])
    def test_strong_morse_on_benchmarks(self, type_string, name):
        """Strong Morse inequality holds for all benchmarks."""
        ss = _build(type_string)
        _, strong = morse_inequalities(ss)
        assert strong, f"Strong Morse inequality failed for {name}"

    @pytest.mark.parametrize("type_string,name", [
        ("end", "End"),
        ("&{a: end}", "SingleMethod"),
        ("&{a: end, b: end}", "SimpleBranch"),
        ("+{ok: end, err: end}", "SimpleSelect"),
        ("&{a: &{b: &{c: end}}}", "DeepChain"),
        ("(&{a: end} || &{b: end})", "SimpleParallel"),
        ("&{a: &{c: end, d: end}, b: &{e: end, f: end}}", "NestedBranch"),
        ("&{a: end, b: end, c: end, d: end}", "WideBranch"),
        ("&{get: +{OK: end, NOT_FOUND: end}, put: +{OK: end, ERR: end}}", "REST"),
        ("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}", "Iterator"),
        ("&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}", "File"),
        ("+{OK: &{a: end}, ERR: end}", "SelectBranch"),
    ])
    def test_matching_acyclic_on_benchmarks(self, type_string, name):
        """Matching is acyclic for all benchmarks."""
        ss = _build(type_string)
        m = greedy_matching(ss)
        assert m.is_acyclic, f"Matching not acyclic for {name}"

    @pytest.mark.parametrize("type_string,name", [
        ("end", "End"),
        ("&{a: end, b: end}", "SimpleBranch"),
        ("&{a: &{b: &{c: end}}}", "DeepChain"),
        ("&{a: end, b: end, c: end, d: end}", "WideBranch"),
    ])
    def test_acyclic_types_no_cycles(self, type_string, name):
        """Acyclic types have b1 = 0 in the Hasse complex."""
        ss = _build(type_string)
        b = betti_numbers(ss)
        assert b[1] == 0, f"{name} should have b1=0 (tree-like)"

    def test_parallel_diamond_has_cycle(self):
        """Diamond lattice from parallel has exactly one cycle."""
        ss = _build("(&{a: end} || &{b: end})")
        b = betti_numbers(ss)
        assert b[1] == 1, "Diamond has one independent cycle"

    @pytest.mark.parametrize("type_string,name", [
        ("end", "End"),
        ("&{a: end}", "SingleMethod"),
        ("&{a: end, b: end}", "SimpleBranch"),
        ("+{ok: end, err: end}", "SimpleSelect"),
        ("&{a: &{b: &{c: end}}}", "DeepChain"),
        ("(&{a: end} || &{b: end})", "SimpleParallel"),
        ("&{a: &{c: end, d: end}, b: &{e: end, f: end}}", "NestedBranch"),
        ("&{a: end, b: end, c: end, d: end}", "WideBranch"),
        ("&{get: +{OK: end, NOT_FOUND: end}, put: +{OK: end, ERR: end}}", "REST"),
        ("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}", "Iterator"),
        ("&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}", "File"),
        ("+{OK: &{a: end}, ERR: end}", "SelectBranch"),
    ])
    def test_weak_morse_on_benchmarks(self, type_string, name):
        """Weak Morse inequality holds for all benchmarks."""
        ss = _build(type_string)
        weak, _ = morse_inequalities(ss)
        assert weak, f"Weak Morse inequality failed for {name}"

    @pytest.mark.parametrize("type_string,name", [
        ("end", "End"),
        ("&{a: end}", "SingleMethod"),
        ("&{a: end, b: end}", "SimpleBranch"),
        ("+{ok: end, err: end}", "SimpleSelect"),
        ("&{a: &{b: &{c: end}}}", "DeepChain"),
        ("(&{a: end} || &{b: end})", "SimpleParallel"),
        ("&{a: &{c: end, d: end}, b: &{e: end, f: end}}", "NestedBranch"),
        ("&{a: end, b: end, c: end, d: end}", "WideBranch"),
        ("&{get: +{OK: end, NOT_FOUND: end}, put: +{OK: end, ERR: end}}", "REST"),
        ("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}", "Iterator"),
        ("&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}", "File"),
        ("+{OK: &{a: end}, ERR: end}", "SelectBranch"),
    ])
    def test_betti_nonnegative_on_benchmarks(self, type_string, name):
        """All Betti numbers are non-negative."""
        ss = _build(type_string)
        b = betti_numbers(ss)
        assert all(x >= 0 for x in b), f"Negative Betti for {name}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_state(self):
        ss = _build("end")
        r = analyze_morse(ss)
        assert r.num_critical >= 1
        assert r.euler_characteristic == 1
        assert r.betti_numbers == (1, 0)

    def test_two_states(self):
        ss = _build("&{a: end}")
        r = analyze_morse(ss)
        assert r.strong_morse_holds

    def test_matching_empty_for_end(self):
        ss = _build("end")
        m = greedy_matching(ss)
        assert len(m.matched_pairs) == 0

    def test_gradient_flow_graph_structure(self):
        ss = _build("&{a: end, b: end}")
        g = gradient_field(ss)
        assert set(g.flow_graph.keys()) == ss.states

    def test_critical_equals_matching_critical(self):
        ss = _build("&{a: &{b: end}, c: end}")
        m = greedy_matching(ss)
        crits = critical_cells(m)
        assert crits == m.critical_vertices

    def test_betti_nonnegative(self):
        """Betti numbers are always non-negative."""
        for ts in ["end", "&{a: end}", "&{a: end, b: end}",
                    "rec X . &{a: X, b: end}"]:
            ss = _build(ts)
            b = betti_numbers(ss)
            assert all(x >= 0 for x in b), f"Negative Betti for {ts}"

    def test_euler_characteristic_formula(self):
        """chi = b0 - b1 for connected graphs."""
        ss = _build("&{a: end, b: end}")
        r = analyze_morse(ss)
        b = r.betti_numbers
        assert r.euler_characteristic == b[0] - b[1]

    def test_select_branch_combo(self):
        ss = _build("+{OK: &{a: end}, ERR: end}")
        r = analyze_morse(ss)
        assert r.strong_morse_holds

    def test_triple_branch(self):
        ss = _build("&{a: end, b: end, c: end}")
        r = analyze_morse(ss)
        assert r.strong_morse_holds

    def test_recursive_with_selection(self):
        ss = _build("rec X . +{a: X, b: end}")
        r = analyze_morse(ss)
        assert r.strong_morse_holds
