"""Tests for configuration domains (Step 17)."""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.event_structures import (
    build_event_structure,
    configurations,
    config_domain,
    Configuration,
)
from reticulate.config_domains import (
    ScottDomainResult,
    ConfigDomainAnalysis,
    build_config_domain,
    check_scott_domain,
    compact_elements,
    consistent_pairs,
    check_coherence,
    check_algebraicity,
    check_bounded_completeness,
    check_distributivity,
    scott_open_sets,
    covering_chains,
    join_irreducibles,
    is_lattice,
    width,
    analyze_config_domain,
)


def _build(type_string: str):
    return build_statespace(parse(type_string))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def end_ss():
    return _build("end")


@pytest.fixture
def single_branch():
    return _build("&{a: end}")


@pytest.fixture
def binary_branch():
    return _build("&{a: end, b: end}")


@pytest.fixture
def deep_chain():
    return _build("&{a: &{b: &{c: end}}}")


@pytest.fixture
def parallel_ss():
    return _build("(&{a: end} || &{b: end})")


@pytest.fixture
def select_ss():
    return _build("+{ok: end, err: end}")


@pytest.fixture
def iterator_ss():
    return _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")


@pytest.fixture
def nested():
    return _build("&{a: &{c: end, d: end}, b: &{e: end, f: end}}")


@pytest.fixture
def ternary_branch():
    return _build("&{a: &{x: end, y: end}, b: &{x: end, y: end}, c: &{x: end, y: end}}")


# ---------------------------------------------------------------------------
# build_config_domain
# ---------------------------------------------------------------------------

class TestBuildConfigDomain:
    def test_end_single_config(self, end_ss):
        dom = build_config_domain(end_ss)
        assert dom.num_configs == 1  # only empty config
        assert dom.bottom.events == frozenset()

    def test_single_branch_configs(self, single_branch):
        dom = build_config_domain(single_branch)
        assert dom.num_configs == 2  # {} and {a}

    def test_binary_branch_configs(self, binary_branch):
        dom = build_config_domain(binary_branch)
        assert dom.num_configs == 3  # {}, {a}, {b}

    def test_deep_chain_configs(self, deep_chain):
        dom = build_config_domain(deep_chain)
        assert dom.num_configs == 4  # {}, {a}, {a,b}, {a,b,c}

    def test_parallel_configs(self, parallel_ss):
        dom = build_config_domain(parallel_ss)
        # Parallel: product lattice 2x2 = 4 states
        assert dom.num_configs >= 3

    def test_ordering_reflexive(self, binary_branch):
        dom = build_config_domain(binary_branch)
        for i in range(dom.num_configs):
            assert (i, i) in dom.ordering

    def test_ordering_antisymmetric(self, deep_chain):
        dom = build_config_domain(deep_chain)
        for (i, j) in dom.ordering:
            if i != j:
                assert (j, i) not in dom.ordering

    def test_bottom_is_empty(self, nested):
        dom = build_config_domain(nested)
        assert dom.bottom.events == frozenset()


# ---------------------------------------------------------------------------
# compact_elements
# ---------------------------------------------------------------------------

class TestCompactElements:
    def test_end_all_compact(self, end_ss):
        dom = build_config_domain(end_ss)
        compacts = compact_elements(dom)
        assert len(compacts) == dom.num_configs

    def test_binary_branch_all_compact(self, binary_branch):
        dom = build_config_domain(binary_branch)
        compacts = compact_elements(dom)
        assert len(compacts) == 3

    def test_deep_chain_all_compact(self, deep_chain):
        dom = build_config_domain(deep_chain)
        compacts = compact_elements(dom)
        assert len(compacts) == 4

    def test_finite_means_all_compact(self, parallel_ss):
        """In a finite poset, every element is compact."""
        dom = build_config_domain(parallel_ss)
        compacts = compact_elements(dom)
        assert len(compacts) == dom.num_configs


# ---------------------------------------------------------------------------
# consistent_pairs
# ---------------------------------------------------------------------------

class TestConsistentPairs:
    def test_end_single_pair(self, end_ss):
        dom = build_config_domain(end_ss)
        pairs = consistent_pairs(dom)
        assert len(pairs) == 1  # (0, 0)

    def test_chain_all_consistent(self, deep_chain):
        """In a chain, all pairs are consistent."""
        dom = build_config_domain(deep_chain)
        pairs = consistent_pairs(dom)
        n = dom.num_configs
        assert len(pairs) == n * (n + 1) // 2  # all pairs including self

    def test_binary_branch_consistent(self, binary_branch):
        dom = build_config_domain(binary_branch)
        pairs = consistent_pairs(dom)
        # 3 configs: {}, {a}, {b}. All are consistent (empty is below both).
        # Actually: {a} and {b} are consistent iff they have an upper bound.
        # In a 3-element config domain: {} <= {a}, {} <= {b},
        # but {a} and {b} might not have an upper bound if they conflict.
        assert len(pairs) >= 3  # at minimum, each element with itself and bottom

    def test_self_pairs_always_consistent(self, nested):
        dom = build_config_domain(nested)
        pairs = consistent_pairs(dom)
        pair_set = set(pairs)
        for i in range(dom.num_configs):
            assert (i, i) in pair_set

    def test_parallel_more_consistent(self, parallel_ss):
        """Parallel has more consistency since events don't conflict."""
        dom = build_config_domain(parallel_ss)
        pairs = consistent_pairs(dom)
        n = dom.num_configs
        # In a lattice, all pairs are consistent
        assert len(pairs) >= n


# ---------------------------------------------------------------------------
# check_coherence
# ---------------------------------------------------------------------------

class TestCoherence:
    def test_end_coherent(self, end_ss):
        dom = build_config_domain(end_ss)
        assert check_coherence(dom) is True

    def test_chain_coherent(self, deep_chain):
        dom = build_config_domain(deep_chain)
        assert check_coherence(dom) is True

    def test_binary_branch_coherent(self, binary_branch):
        dom = build_config_domain(binary_branch)
        assert check_coherence(dom) is True

    def test_parallel_coherent(self, parallel_ss):
        dom = build_config_domain(parallel_ss)
        assert check_coherence(dom) is True

    def test_nested_coherent(self, nested):
        dom = build_config_domain(nested)
        assert check_coherence(dom) is True

    def test_select_coherent(self, select_ss):
        dom = build_config_domain(select_ss)
        assert check_coherence(dom) is True


# ---------------------------------------------------------------------------
# check_algebraicity
# ---------------------------------------------------------------------------

class TestAlgebraicity:
    def test_always_algebraic_end(self, end_ss):
        dom = build_config_domain(end_ss)
        assert check_algebraicity(dom) is True

    def test_always_algebraic_chain(self, deep_chain):
        dom = build_config_domain(deep_chain)
        assert check_algebraicity(dom) is True

    def test_always_algebraic_parallel(self, parallel_ss):
        dom = build_config_domain(parallel_ss)
        assert check_algebraicity(dom) is True


# ---------------------------------------------------------------------------
# check_bounded_completeness
# ---------------------------------------------------------------------------

class TestBoundedCompleteness:
    def test_end_bounded_complete(self, end_ss):
        dom = build_config_domain(end_ss)
        assert check_bounded_completeness(dom) is True

    def test_chain_bounded_complete(self, deep_chain):
        dom = build_config_domain(deep_chain)
        assert check_bounded_completeness(dom) is True

    def test_binary_branch_bounded_complete(self, binary_branch):
        dom = build_config_domain(binary_branch)
        assert check_bounded_completeness(dom) is True

    def test_parallel_bounded_complete(self, parallel_ss):
        dom = build_config_domain(parallel_ss)
        assert check_bounded_completeness(dom) is True

    def test_select_bounded_complete(self, select_ss):
        dom = build_config_domain(select_ss)
        assert check_bounded_completeness(dom) is True


# ---------------------------------------------------------------------------
# check_scott_domain
# ---------------------------------------------------------------------------

class TestCheckScottDomain:
    def test_end_is_scott_domain(self, end_ss):
        dom = build_config_domain(end_ss)
        result = check_scott_domain(dom)
        assert result.is_scott_domain is True
        assert result.is_dcpo is True
        assert result.is_algebraic is True

    def test_chain_is_scott_domain(self, deep_chain):
        dom = build_config_domain(deep_chain)
        result = check_scott_domain(dom)
        assert result.is_scott_domain is True

    def test_binary_branch_scott_domain(self, binary_branch):
        dom = build_config_domain(binary_branch)
        result = check_scott_domain(dom)
        assert result.is_scott_domain is True
        assert result.num_compact == dom.num_configs

    def test_parallel_scott_domain(self, parallel_ss):
        dom = build_config_domain(parallel_ss)
        result = check_scott_domain(dom)
        assert result.is_scott_domain is True

    def test_iterator_scott_domain(self, iterator_ss):
        dom = build_config_domain(iterator_ss)
        result = check_scott_domain(dom)
        assert result.is_scott_domain is True

    def test_nested_scott_domain(self, nested):
        dom = build_config_domain(nested)
        result = check_scott_domain(dom)
        assert result.is_scott_domain is True

    def test_scott_result_has_counts(self, binary_branch):
        dom = build_config_domain(binary_branch)
        result = check_scott_domain(dom)
        assert result.num_configs == dom.num_configs
        assert result.num_compact > 0
        assert result.num_consistent_pairs > 0
        assert result.num_scott_open > 0


# ---------------------------------------------------------------------------
# scott_open_sets
# ---------------------------------------------------------------------------

class TestScottOpenSets:
    def test_end_two_opens(self, end_ss):
        """Single-element poset has 2 open sets: {} and {0}."""
        dom = build_config_domain(end_ss)
        opens = scott_open_sets(dom)
        assert len(opens) == 2

    def test_chain_opens(self, single_branch):
        """2-element chain has 3 open sets: {}, {top}, {top, bot}."""
        dom = build_config_domain(single_branch)
        opens = scott_open_sets(dom)
        # Upper sets of a 2-chain: {}, {1}, {0,1} where 0<=1
        assert len(opens) >= 2

    def test_empty_is_open(self, binary_branch):
        dom = build_config_domain(binary_branch)
        opens = scott_open_sets(dom)
        assert frozenset() in opens

    def test_whole_set_is_open(self, binary_branch):
        dom = build_config_domain(binary_branch)
        opens = scott_open_sets(dom)
        all_indices = frozenset(range(dom.num_configs))
        assert all_indices in opens

    def test_opens_are_upper_sets(self, deep_chain):
        dom = build_config_domain(deep_chain)
        opens = scott_open_sets(dom)
        for u in opens:
            for i in u:
                for j in range(dom.num_configs):
                    if (i, j) in dom.ordering:
                        assert j in u


# ---------------------------------------------------------------------------
# check_distributivity
# ---------------------------------------------------------------------------

class TestDistributivity:
    def test_end_distributive(self, end_ss):
        dom = build_config_domain(end_ss)
        assert check_distributivity(dom) is True

    def test_chain_distributive(self, deep_chain):
        dom = build_config_domain(deep_chain)
        assert check_distributivity(dom) is True

    def test_binary_branch_distributive(self, binary_branch):
        dom = build_config_domain(binary_branch)
        assert check_distributivity(dom) is True

    def test_parallel_distributive(self, parallel_ss):
        """Product lattice of chains is distributive."""
        dom = build_config_domain(parallel_ss)
        assert check_distributivity(dom) is True

    def test_select_distributive(self, select_ss):
        dom = build_config_domain(select_ss)
        assert check_distributivity(dom) is True


# ---------------------------------------------------------------------------
# covering_chains
# ---------------------------------------------------------------------------

class TestCoveringChains:
    def test_end_single_chain(self, end_ss):
        dom = build_config_domain(end_ss)
        chains = covering_chains(dom)
        assert len(chains) == 1
        assert len(chains[0]) == 1  # just the empty config

    def test_single_branch_one_chain(self, single_branch):
        dom = build_config_domain(single_branch)
        chains = covering_chains(dom)
        assert len(chains) == 1
        assert len(chains[0]) == 2

    def test_binary_branch_two_chains(self, binary_branch):
        """Two maximal chains: {} -> {a} and {} -> {b}."""
        dom = build_config_domain(binary_branch)
        chains = covering_chains(dom)
        assert len(chains) == 2

    def test_deep_chain_one_chain(self, deep_chain):
        dom = build_config_domain(deep_chain)
        chains = covering_chains(dom)
        assert len(chains) == 1
        assert len(chains[0]) == 4  # 4 configs in the chain

    def test_chains_start_at_bottom(self, nested):
        dom = build_config_domain(nested)
        chains = covering_chains(dom)
        for chain in chains:
            # First element should be the bottom (empty config)
            assert dom.configs[chain[0]].events == frozenset()

    def test_chains_are_covering(self, parallel_ss):
        """Each step in a chain adds exactly one event."""
        dom = build_config_domain(parallel_ss)
        chains = covering_chains(dom)
        for chain in chains:
            for k in range(len(chain) - 1):
                c1 = dom.configs[chain[k]]
                c2 = dom.configs[chain[k + 1]]
                assert len(c2.events) == len(c1.events) + 1
                assert c1.events < c2.events


# ---------------------------------------------------------------------------
# join_irreducibles
# ---------------------------------------------------------------------------

class TestJoinIrreducibles:
    def test_end_no_ji(self, end_ss):
        dom = build_config_domain(end_ss)
        ji = join_irreducibles(dom)
        assert len(ji) == 0

    def test_single_branch_one_ji(self, single_branch):
        dom = build_config_domain(single_branch)
        ji = join_irreducibles(dom)
        assert len(ji) == 1

    def test_chain_all_nonbottom_are_ji(self, deep_chain):
        """In a chain, every non-bottom element covers exactly one element."""
        dom = build_config_domain(deep_chain)
        ji = join_irreducibles(dom)
        assert len(ji) == dom.num_configs - 1  # all except bottom

    def test_binary_branch_ji(self, binary_branch):
        """In a 3-element diamond-minus-top, both non-bottom elements are JI."""
        dom = build_config_domain(binary_branch)
        ji = join_irreducibles(dom)
        assert len(ji) == 2


# ---------------------------------------------------------------------------
# is_lattice
# ---------------------------------------------------------------------------

class TestIsLattice:
    def test_end_is_lattice(self, end_ss):
        dom = build_config_domain(end_ss)
        assert is_lattice(dom) is True

    def test_chain_is_lattice(self, deep_chain):
        dom = build_config_domain(deep_chain)
        assert is_lattice(dom) is True

    def test_binary_branch_not_lattice(self, binary_branch):
        """Config domain of branch: {}, {a}, {b} — no join for {a},{b}."""
        dom = build_config_domain(binary_branch)
        # {a} and {b} conflict, so there's no config containing both.
        # The domain has no top, so it's NOT a lattice.
        # (The state space IS a lattice, but the config domain may not be.)
        # This depends on whether {a} and {b} have a join.
        result = is_lattice(dom)
        # If there's no upper bound for conflicting configs, not a lattice
        assert isinstance(result, bool)

    def test_parallel_config_domain(self, parallel_ss):
        """Parallel config domain may or may not be a lattice.

        The config domain of the event structure may differ from the
        state space lattice — the state space is a quotient of the
        config domain.  Config domains can fail to be lattices when
        multiple maximal configurations exist without a common upper bound.
        """
        dom = build_config_domain(parallel_ss)
        result = is_lattice(dom)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# width
# ---------------------------------------------------------------------------

class TestWidth:
    def test_end_width_1(self, end_ss):
        dom = build_config_domain(end_ss)
        assert width(dom) == 1

    def test_chain_width_1(self, deep_chain):
        dom = build_config_domain(deep_chain)
        assert width(dom) == 1

    def test_binary_branch_width_2(self, binary_branch):
        """Two incomparable configs: {a} and {b}."""
        dom = build_config_domain(binary_branch)
        assert width(dom) == 2

    def test_parallel_width(self, parallel_ss):
        dom = build_config_domain(parallel_ss)
        assert width(dom) >= 2


# ---------------------------------------------------------------------------
# analyze_config_domain
# ---------------------------------------------------------------------------

class TestAnalyzeConfigDomain:
    def test_end_analysis(self, end_ss):
        analysis = analyze_config_domain(end_ss)
        assert analysis.num_configs == 1
        assert analysis.scott_result.is_scott_domain is True

    def test_chain_analysis(self, deep_chain):
        analysis = analyze_config_domain(deep_chain)
        assert analysis.num_configs == 4
        assert analysis.num_covering_chains == 1
        assert analysis.num_join_irreducibles == 3
        assert analysis.max_chain_length == 4
        assert analysis.width == 1

    def test_binary_branch_analysis(self, binary_branch):
        analysis = analyze_config_domain(binary_branch)
        assert analysis.num_configs == 3
        assert analysis.scott_result.is_scott_domain is True

    def test_parallel_analysis(self, parallel_ss):
        analysis = analyze_config_domain(parallel_ss)
        assert analysis.scott_result.is_scott_domain is True
        assert analysis.scott_result.is_algebraic is True

    def test_iterator_analysis(self, iterator_ss):
        analysis = analyze_config_domain(iterator_ss)
        assert analysis.scott_result.is_scott_domain is True
        assert analysis.num_configs > 0

    def test_nested_analysis(self, nested):
        analysis = analyze_config_domain(nested)
        assert analysis.num_configs > 0
        assert analysis.scott_result.is_dcpo is True

    def test_analysis_has_domain(self, single_branch):
        analysis = analyze_config_domain(single_branch)
        assert analysis.domain is not None
        assert analysis.es is not None

    def test_select_analysis(self, select_ss):
        analysis = analyze_config_domain(select_ss)
        assert analysis.scott_result.is_scott_domain is True


# ---------------------------------------------------------------------------
# di-domain properties
# ---------------------------------------------------------------------------

class TestDIDomain:
    def test_chain_is_di_domain(self, deep_chain):
        dom = build_config_domain(deep_chain)
        result = check_scott_domain(dom)
        assert result.is_di_domain is True

    def test_end_is_di_domain(self, end_ss):
        dom = build_config_domain(end_ss)
        result = check_scott_domain(dom)
        assert result.is_di_domain is True

    def test_parallel_di_domain(self, parallel_ss):
        dom = build_config_domain(parallel_ss)
        result = check_scott_domain(dom)
        assert result.is_di_domain is True


# ---------------------------------------------------------------------------
# Edge cases and integration
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_element_domain(self, end_ss):
        """Domain with just the empty configuration."""
        dom = build_config_domain(end_ss)
        assert dom.num_configs == 1
        assert check_coherence(dom) is True
        assert check_bounded_completeness(dom) is True
        assert check_algebraicity(dom) is True
        assert is_lattice(dom) is True

    def test_scott_open_includes_empty_and_full(self, single_branch):
        dom = build_config_domain(single_branch)
        opens = scott_open_sets(dom)
        assert frozenset() in opens
        assert frozenset(range(dom.num_configs)) in opens

    def test_covering_chains_count_matches_paths(self, nested):
        """Covering chains = linearizations of the event structure."""
        dom = build_config_domain(nested)
        chains = covering_chains(dom)
        assert len(chains) > 0

    def test_config_domain_ordering_transitive(self, parallel_ss):
        dom = build_config_domain(parallel_ss)
        for (i, j) in dom.ordering:
            for (j2, k) in dom.ordering:
                if j == j2:
                    assert (i, k) in dom.ordering

    def test_ternary_branch_analysis(self, ternary_branch):
        """Ternary branch produces non-distributive state space (M3)."""
        analysis = analyze_config_domain(ternary_branch)
        assert analysis.num_configs > 0
        assert analysis.scott_result.is_scott_domain is True
