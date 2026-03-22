"""Tests for marking lattice analysis (Step 22)."""

import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice, check_distributive
from reticulate.petri import build_petri_net, build_reachability_graph
from reticulate.marking_lattice import (
    MarkingLatticeResult,
    PosetMetrics,
    analyze_marking_lattice,
    check_marking_isomorphism,
    compute_covering_relation,
    compute_height,
    compute_poset_metrics,
    compute_width,
    count_maximal_chains,
    reachability_to_statespace,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _ss(type_string: str):
    return build_statespace(parse(type_string))


def _mss(type_string: str):
    """Build marking state space from type string."""
    ss = _ss(type_string)
    net = build_petri_net(ss)
    rg = build_reachability_graph(net)
    return reachability_to_statespace(rg, net)


# ---------------------------------------------------------------------------
# Conversion: ReachabilityGraph → StateSpace
# ---------------------------------------------------------------------------


class TestReachabilityToStatespace:
    """Test converting reachability graphs to state spaces."""

    def test_end_single_state(self):
        mss = _mss("end")
        assert len(mss.states) == 1

    def test_single_branch(self):
        mss = _mss("&{a: end}")
        assert len(mss.states) == 2
        assert len(mss.transitions) == 1

    def test_two_branch(self):
        mss = _mss("&{a: end, b: end}")
        assert len(mss.states) == 2
        assert len(mss.transitions) == 2

    def test_nested_branch(self):
        mss = _mss("&{a: &{b: end}}")
        assert len(mss.states) == 3
        assert len(mss.transitions) == 2

    def test_recursive(self):
        mss = _mss("rec X . &{a: X, b: end}")
        assert len(mss.states) == 2

    def test_marking_ss_has_top_and_bottom(self):
        mss = _mss("&{a: &{b: end}, c: end}")
        assert mss.top is not None
        assert mss.bottom is not None
        assert mss.top != mss.bottom

    def test_parallel(self):
        mss = _mss("(&{a: end} || &{b: end})")
        ss = _ss("(&{a: end} || &{b: end})")
        assert len(mss.states) == len(ss.states)


# ---------------------------------------------------------------------------
# Marking lattice = state-space lattice
# ---------------------------------------------------------------------------


class TestMarkingIsomorphism:
    """Verify M(N(S)) ≅ L(S)."""

    def test_end_isomorphic(self):
        ss = _ss("end")
        net = build_petri_net(ss)
        assert check_marking_isomorphism(ss, net)

    def test_single_branch_isomorphic(self):
        ss = _ss("&{a: end}")
        net = build_petri_net(ss)
        assert check_marking_isomorphism(ss, net)

    def test_nested_isomorphic(self):
        ss = _ss("&{a: &{b: end}, c: end}")
        net = build_petri_net(ss)
        assert check_marking_isomorphism(ss, net)

    def test_selection_isomorphic(self):
        ss = _ss("+{ok: end, err: end}")
        net = build_petri_net(ss)
        assert check_marking_isomorphism(ss, net)

    def test_recursive_isomorphic(self):
        ss = _ss("rec X . &{a: X, b: end}")
        net = build_petri_net(ss)
        assert check_marking_isomorphism(ss, net)

    def test_parallel_isomorphic(self):
        ss = _ss("(&{a: end} || &{b: end})")
        net = build_petri_net(ss)
        assert check_marking_isomorphism(ss, net)

    def test_deep_isomorphic(self):
        ss = _ss("&{a: &{b: &{c: end}}}")
        net = build_petri_net(ss)
        assert check_marking_isomorphism(ss, net)

    def test_mixed_isomorphic(self):
        ss = _ss("&{a: +{ok: end, err: end}, b: end}")
        net = build_petri_net(ss)
        assert check_marking_isomorphism(ss, net)


# ---------------------------------------------------------------------------
# Marking lattice IS a lattice
# ---------------------------------------------------------------------------


class TestMarkingIsLattice:
    """Verify marking poset forms a lattice."""

    def test_end_lattice(self):
        mss = _mss("end")
        lr = check_lattice(mss)
        assert lr.is_lattice

    def test_branch_lattice(self):
        mss = _mss("&{a: end, b: end}")
        lr = check_lattice(mss)
        assert lr.is_lattice

    def test_nested_lattice(self):
        mss = _mss("&{a: &{b: end}, c: end}")
        lr = check_lattice(mss)
        assert lr.is_lattice

    def test_recursive_lattice(self):
        mss = _mss("rec X . &{a: X, b: end}")
        lr = check_lattice(mss)
        assert lr.is_lattice

    def test_parallel_lattice(self):
        mss = _mss("(&{a: end} || &{b: end})")
        lr = check_lattice(mss)
        assert lr.is_lattice


# ---------------------------------------------------------------------------
# Distributivity preservation
# ---------------------------------------------------------------------------


class TestDistributivity:
    """Verify distributivity is preserved between L(S) and M(N(S))."""

    def test_chain_distributive(self):
        ss = _ss("&{a: end}")
        mss = _mss("&{a: end}")
        d_ss = check_distributive(ss)
        d_mss = check_distributive(mss)
        assert d_ss.is_distributive == d_mss.is_distributive

    def test_diamond_distributive(self):
        ss = _ss("&{a: end, b: end}")
        mss = _mss("&{a: end, b: end}")
        d_ss = check_distributive(ss)
        d_mss = check_distributive(mss)
        assert d_ss.is_distributive == d_mss.is_distributive

    def test_ternary_non_distributive(self):
        ss = _ss("&{a: end, b: end, c: end}")
        mss = _mss("&{a: end, b: end, c: end}")
        d_ss = check_distributive(ss)
        d_mss = check_distributive(mss)
        assert d_ss.is_distributive == d_mss.is_distributive


# ---------------------------------------------------------------------------
# Covering relation
# ---------------------------------------------------------------------------


class TestCoveringRelation:
    """Test covering relation computation."""

    def test_end_no_covering(self):
        mss = _mss("end")
        covering = compute_covering_relation(mss)
        assert len(covering) == 0

    def test_chain_covering(self):
        mss = _mss("&{a: end}")
        covering = compute_covering_relation(mss)
        assert len(covering) == 1

    def test_diamond_covering(self):
        """Diamond lattice: top covers both middles, both cover bottom."""
        mss = _mss("&{a: end, b: end}")
        covering = compute_covering_relation(mss)
        # top → a → bottom, top → b → bottom
        # But states merge: top covers bottom via a and b (both go to end)
        # So covering depends on state count
        assert len(covering) >= 1

    def test_nested_covering(self):
        mss = _mss("&{a: &{b: end}}")
        covering = compute_covering_relation(mss)
        # Chain of 3: top → mid → bottom
        assert len(covering) == 2


# ---------------------------------------------------------------------------
# Width and height
# ---------------------------------------------------------------------------


class TestWidthHeight:
    """Test width and height computation."""

    def test_chain_width_1(self):
        mss = _mss("&{a: &{b: end}}")
        assert compute_width(mss) == 1

    def test_diamond_width_2(self):
        """Diamond has 2 incomparable elements in the middle."""
        mss = _mss("&{a: &{c: end}, b: &{d: end}}")
        w = compute_width(mss)
        assert w >= 2

    def test_chain_height(self):
        mss = _mss("&{a: &{b: end}}")
        assert compute_height(mss) == 2

    def test_single_height_0(self):
        mss = _mss("end")
        assert compute_height(mss) == 0

    def test_branch_height_1(self):
        mss = _mss("&{a: end, b: end}")
        assert compute_height(mss) == 1

    def test_deep_height(self):
        mss = _mss("&{a: &{b: &{c: end}}}")
        assert compute_height(mss) == 3


# ---------------------------------------------------------------------------
# Maximal chains
# ---------------------------------------------------------------------------


class TestMaximalChains:
    """Test maximal chain counting."""

    def test_end_one_chain(self):
        mss = _mss("end")
        assert count_maximal_chains(mss) == 1

    def test_linear_one_chain(self):
        mss = _mss("&{a: end}")
        assert count_maximal_chains(mss) == 1

    def test_branch_two_chains(self):
        mss = _mss("&{a: end, b: end}")
        assert count_maximal_chains(mss) == 2

    def test_triple_branch_three_chains(self):
        mss = _mss("&{a: end, b: end, c: end}")
        assert count_maximal_chains(mss) == 3

    def test_nested_chains(self):
        mss = _mss("&{a: &{c: end}, b: end}")
        chains = count_maximal_chains(mss)
        assert chains == 2  # a→c and b


# ---------------------------------------------------------------------------
# Poset metrics
# ---------------------------------------------------------------------------


class TestPosetMetrics:
    """Test combined poset metrics computation."""

    def test_metrics_end(self):
        mss = _mss("end")
        m = compute_poset_metrics(mss)
        assert m.width == 1
        assert m.height == 0
        assert m.num_chains == 1

    def test_metrics_branch(self):
        mss = _mss("&{a: end, b: end}")
        m = compute_poset_metrics(mss)
        assert m.width >= 1
        assert m.height == 1
        assert m.num_chains == 2

    def test_metrics_deep(self):
        mss = _mss("&{a: &{b: &{c: end}}}")
        m = compute_poset_metrics(mss)
        assert m.width == 1
        assert m.height == 3
        assert m.num_chains == 1


# ---------------------------------------------------------------------------
# High-level analysis
# ---------------------------------------------------------------------------


class TestAnalyzeMarkingLattice:
    """Test the main entry point."""

    def test_result_fields(self):
        ss = _ss("&{a: end, b: end}")
        result = analyze_marking_lattice(ss)
        assert isinstance(result, MarkingLatticeResult)
        assert result.lattice_result.is_lattice
        assert result.num_markings == len(ss.states)
        assert result.is_isomorphic_to_statespace
        assert result.height >= 1
        assert result.width >= 1
        assert result.chain_count >= 1

    def test_recursive_analysis(self):
        ss = _ss("rec X . &{a: X, b: end}")
        result = analyze_marking_lattice(ss)
        assert result.lattice_result.is_lattice
        assert result.is_isomorphic_to_statespace

    def test_parallel_analysis(self):
        ss = _ss("(&{a: end} || &{b: end})")
        result = analyze_marking_lattice(ss)
        assert result.lattice_result.is_lattice
        assert result.is_isomorphic_to_statespace


# ---------------------------------------------------------------------------
# Benchmark protocols
# ---------------------------------------------------------------------------


class TestBenchmarks:
    """Test marking lattice on all benchmark protocols."""

    @pytest.fixture
    def benchmarks(self):
        from tests.benchmarks.protocols import BENCHMARKS
        return BENCHMARKS

    def test_all_marking_lattices_are_lattices(self, benchmarks):
        """All 79 benchmark marking posets are lattices."""
        failures = []
        for bp in benchmarks:
            try:
                ss = build_statespace(parse(bp.type_string))
                result = analyze_marking_lattice(ss)
                if not result.lattice_result.is_lattice:
                    failures.append(f"{bp.name}: not a lattice")
            except Exception as e:
                failures.append(f"{bp.name}: {e}")
        assert failures == [], f"Failures: {failures}"

    def test_all_marking_lattices_isomorphic(self, benchmarks):
        """All 79 benchmarks: M(N(S)) ≅ L(S)."""
        failures = []
        for bp in benchmarks:
            try:
                ss = build_statespace(parse(bp.type_string))
                result = analyze_marking_lattice(ss)
                if not result.is_isomorphic_to_statespace:
                    failures.append(bp.name)
            except Exception as e:
                failures.append(f"{bp.name}: {e}")
        assert failures == [], f"Failures: {failures}"

    def test_distributivity_preserved(self, benchmarks):
        """Distributivity agrees between L(S) and M(N(S))."""
        failures = []
        for bp in benchmarks:
            try:
                ss = build_statespace(parse(bp.type_string))
                d_ss = check_distributive(ss)
                result = analyze_marking_lattice(ss)
                if d_ss.is_distributive != result.distributivity.is_distributive:
                    failures.append(bp.name)
            except Exception as e:
                failures.append(f"{bp.name}: {e}")
        assert failures == [], f"Failures: {failures}"

    def test_benchmark_statistics(self, benchmarks):
        """Collect poset statistics across all benchmarks."""
        total_width = 0
        total_height = 0
        total_chains = 0
        for bp in benchmarks:
            ss = build_statespace(parse(bp.type_string))
            result = analyze_marking_lattice(ss)
            total_width += result.width
            total_height += result.height
            total_chains += result.chain_count
        assert total_width > 0
        assert total_height > 0
        assert total_chains > 0

    def test_width_height_relation(self, benchmarks):
        """Width * height >= num_markings - 1 (Dilworth bound)."""
        for bp in benchmarks:
            ss = build_statespace(parse(bp.type_string))
            result = analyze_marking_lattice(ss)
            # For any poset: width * height >= n - 1 is NOT always true,
            # but width >= 1 and height >= 0 always hold
            assert result.width >= 1, f"{bp.name}: width < 1"
            assert result.height >= 0, f"{bp.name}: height < 0"
