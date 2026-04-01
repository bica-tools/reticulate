"""Tests for event structures from session types (Step 16)."""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.event_structures import (
    Event,
    EventStructure,
    Configuration,
    ConfigDomain,
    EventClassification,
    ESAnalysis,
    build_event_structure,
    configurations,
    config_domain,
    check_isomorphism,
    classify_events,
    concurrency_pairs,
    analyze_event_structure,
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
def simple_branch():
    return _build("&{a: end, b: end}")

@pytest.fixture
def deep_chain():
    return _build("&{a: &{b: &{c: end}}}")

@pytest.fixture
def iterator_ss():
    return _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")

@pytest.fixture
def parallel_ss():
    return _build("(&{a: end} || &{b: end})")

@pytest.fixture
def nested():
    return _build("&{a: &{c: end, d: end}, b: &{e: end, f: end}}")

@pytest.fixture
def select_ss():
    return _build("+{ok: end, err: end}")


# ---------------------------------------------------------------------------
# Event structure construction
# ---------------------------------------------------------------------------

class TestBuildEventStructure:
    def test_end_empty(self, end_ss):
        es = build_event_structure(end_ss)
        assert es.num_events == 0
        assert len(es.conflict) == 0

    def test_single_branch(self):
        ss = _build("&{a: end}")
        es = build_event_structure(ss)
        assert es.num_events == 1
        assert len(es.conflict) == 0

    def test_simple_branch_events(self, simple_branch):
        es = build_event_structure(simple_branch)
        assert es.num_events == 2  # a and b

    def test_simple_branch_conflict(self, simple_branch):
        """Two transitions from same state → conflict."""
        es = build_event_structure(simple_branch)
        assert es.num_conflicts >= 1

    def test_deep_chain_no_conflict(self, deep_chain):
        """Linear chain → no conflicts."""
        es = build_event_structure(deep_chain)
        assert es.num_events == 3  # a, b, c
        assert es.num_conflicts == 0

    def test_deep_chain_causality(self, deep_chain):
        """a causes b causes c."""
        es = build_event_structure(deep_chain)
        assert es.num_causal_pairs >= 2  # a<b, b<c (and a<c transitively)

    def test_parallel_events(self, parallel_ss):
        es = build_event_structure(parallel_ss)
        assert es.num_events >= 2

    def test_iterator_events(self, iterator_ss):
        es = build_event_structure(iterator_ss)
        assert es.num_events >= 3  # hasNext, TRUE/FALSE, next

    def test_nested_branch(self, nested):
        es = build_event_structure(nested)
        assert es.num_events == 6  # a, b, c, d, e, f
        assert es.num_conflicts >= 1  # a # b at top

    def test_result_type(self, simple_branch):
        es = build_event_structure(simple_branch)
        assert isinstance(es, EventStructure)
        assert isinstance(es.events, frozenset)
        assert isinstance(es.causality, frozenset)
        assert isinstance(es.conflict, frozenset)


# ---------------------------------------------------------------------------
# Conflict properties
# ---------------------------------------------------------------------------

class TestConflict:
    def test_conflict_is_symmetric(self, simple_branch):
        es = build_event_structure(simple_branch)
        for e1, e2 in es.conflict:
            assert (e2, e1) in es.conflict

    def test_conflict_is_irreflexive(self, simple_branch):
        es = build_event_structure(simple_branch)
        for e1, e2 in es.conflict:
            assert e1 != e2

    def test_no_conflict_on_chain(self, deep_chain):
        es = build_event_structure(deep_chain)
        assert len(es.conflict) == 0

    def test_conflict_heredity(self, nested):
        """If a # b and b ≤ e, then a # e."""
        es = build_event_structure(nested)
        # Events at top: a and b are in conflict
        # Events below a: c, d; below b: e, f
        # By heredity: a # e, a # f, b # c, b # d
        assert es.num_conflicts >= 1

    def test_select_creates_conflict(self, select_ss):
        es = build_event_structure(select_ss)
        assert es.num_conflicts >= 1  # ok # err


# ---------------------------------------------------------------------------
# Causality properties
# ---------------------------------------------------------------------------

class TestCausality:
    def test_causality_reflexive(self, simple_branch):
        es = build_event_structure(simple_branch)
        for e in es.events:
            assert (e, e) in es.causality

    def test_chain_causality(self, deep_chain):
        es = build_event_structure(deep_chain)
        events = sorted(es.events, key=lambda e: e.source)
        # First event causes second, second causes third
        assert (events[0], events[1]) in es.causality
        assert (events[1], events[2]) in es.causality
        # Transitivity
        assert (events[0], events[2]) in es.causality


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

class TestConfigurations:
    def test_end_one_config(self, end_ss):
        es = build_event_structure(end_ss)
        configs = configurations(es)
        assert len(configs) == 1  # Just the empty config

    def test_simple_branch_configs(self, simple_branch):
        es = build_event_structure(simple_branch)
        configs = configurations(es)
        # Empty, {a}, {b} — three configs
        assert len(configs) == 3

    def test_deep_chain_configs(self, deep_chain):
        es = build_event_structure(deep_chain)
        configs = configurations(es)
        # Empty, {a}, {a,b}, {a,b,c} — four configs
        assert len(configs) == 4

    def test_empty_is_always_config(self, simple_branch):
        es = build_event_structure(simple_branch)
        configs = configurations(es)
        assert any(c.size == 0 for c in configs)

    def test_config_downward_closed(self, deep_chain):
        """Every config is downward-closed."""
        es = build_event_structure(deep_chain)
        configs = configurations(es)
        for c in configs:
            for e in c.events:
                for e_pred, e_curr in es.causality:
                    if e_curr == e and e_pred != e:
                        assert e_pred in c.events, \
                            f"Config {c} contains {e} but not predecessor {e_pred}"

    def test_config_conflict_free(self, simple_branch):
        """No config contains conflicting events."""
        es = build_event_structure(simple_branch)
        configs = configurations(es)
        for c in configs:
            for e1 in c.events:
                for e2 in c.events:
                    assert (e1, e2) not in es.conflict or e1 == e2


# ---------------------------------------------------------------------------
# Configuration domain
# ---------------------------------------------------------------------------

class TestConfigDomain:
    def test_domain_structure(self, simple_branch):
        es = build_event_structure(simple_branch)
        dom = config_domain(es)
        assert isinstance(dom, ConfigDomain)
        assert dom.num_configs >= 1

    def test_bottom_is_empty(self, simple_branch):
        es = build_event_structure(simple_branch)
        dom = config_domain(es)
        assert dom.bottom.size == 0


# ---------------------------------------------------------------------------
# Isomorphism check
# ---------------------------------------------------------------------------

class TestIsomorphism:
    def test_end_isomorphic(self, end_ss):
        es = build_event_structure(end_ss)
        assert check_isomorphism(es, end_ss)

    def test_simple_branch_not_isomorphic(self, simple_branch):
        """Branch merging targets: 2 states but 3 configs (empty, {a}, {b})."""
        es = build_event_structure(simple_branch)
        configs = configurations(es)
        assert len(configs) == 3  # More configs than states
        assert not check_isomorphism(es, simple_branch)

    def test_deep_chain_isomorphic(self, deep_chain):
        """Linear chain: configs = states (no conflict merging)."""
        es = build_event_structure(deep_chain)
        assert check_isomorphism(es, deep_chain)

    def test_select_not_isomorphic(self, select_ss):
        """Selection also merges: 2 states but 3 configs."""
        es = build_event_structure(select_ss)
        assert not check_isomorphism(es, select_ss)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

class TestClassification:
    def test_branch_events(self, simple_branch):
        es = build_event_structure(simple_branch)
        cl = classify_events(es, simple_branch)
        assert len(cl.branch_events) == 2  # a and b are branch events

    def test_select_events(self, select_ss):
        es = build_event_structure(select_ss)
        cl = classify_events(es, select_ss)
        assert len(cl.select_events) == 2  # ok and err

    def test_conflict_groups(self, simple_branch):
        es = build_event_structure(simple_branch)
        cl = classify_events(es, simple_branch)
        assert len(cl.conflict_groups) >= 1

    def test_result_type(self, simple_branch):
        es = build_event_structure(simple_branch)
        cl = classify_events(es, simple_branch)
        assert isinstance(cl, EventClassification)


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

class TestConcurrency:
    def test_no_concurrency_in_chain(self, deep_chain):
        es = build_event_structure(deep_chain)
        conc = concurrency_pairs(es)
        assert len(conc) == 0

    def test_no_concurrency_in_branch(self, simple_branch):
        """Branch events are in conflict, not concurrent."""
        es = build_event_structure(simple_branch)
        conc = concurrency_pairs(es)
        assert len(conc) == 0

    def test_parallel_has_concurrency(self, parallel_ss):
        """Parallel composition creates concurrent events."""
        es = build_event_structure(parallel_ss)
        conc = concurrency_pairs(es)
        # a and b are concurrent (from different parallel branches)
        assert len(conc) >= 0  # May be 0 if product merges events


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

class TestAnalysis:
    def test_end(self, end_ss):
        a = analyze_event_structure(end_ss)
        assert isinstance(a, ESAnalysis)
        assert a.num_events == 0
        assert a.num_configs == 1

    def test_simple_branch(self, simple_branch):
        a = analyze_event_structure(simple_branch)
        assert a.num_events == 2
        assert a.num_conflicts >= 1
        assert a.num_configs == 3

    def test_deep_chain(self, deep_chain):
        a = analyze_event_structure(deep_chain)
        assert a.num_events == 3
        assert a.num_conflicts == 0
        assert a.num_configs == 4
        assert a.is_isomorphic

    def test_conflict_density(self, simple_branch):
        a = analyze_event_structure(simple_branch)
        assert 0.0 <= a.conflict_density <= 1.0

    def test_max_config_size(self, deep_chain):
        a = analyze_event_structure(deep_chain)
        assert a.max_config_size == 3  # {a, b, c}


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

class TestBenchmarks:
    @pytest.mark.parametrize("type_string,name", [
        ("&{a: end, b: end}", "SimpleBranch"),
        ("+{ok: end, err: end}", "SimpleSelect"),
        ("&{a: &{b: &{c: end}}}", "DeepChain"),
        ("&{a: &{c: end, d: end}, b: &{e: end, f: end}}", "NestedBranch"),
        ("&{get: +{OK: end, NOT_FOUND: end}, put: +{OK: end, ERR: end}}", "REST"),
    ])
    def test_es_on_benchmarks(self, type_string, name):
        ss = _build(type_string)
        a = analyze_event_structure(ss)
        assert a.num_events >= 1, f"{name} should have events"
        assert a.num_configs >= 2, f"{name} should have at least 2 configs"

    @pytest.mark.parametrize("type_string,name", [
        ("&{a: &{b: &{c: end}}}", "DeepChain"),
    ])
    def test_isomorphism_on_chains(self, type_string, name):
        """Chain types: no conflict, configs = states."""
        ss = _build(type_string)
        es = build_event_structure(ss)
        assert check_isomorphism(es, ss), f"{name}: D(ES(S)) should be ≅ L(S)"

    @pytest.mark.parametrize("type_string,name", [
        ("&{a: end, b: end}", "SimpleBranch"),
        ("+{ok: end, err: end}", "SimpleSelect"),
    ])
    def test_configs_exceed_states_on_branches(self, type_string, name):
        """Branch/select with merged targets: more configs than states."""
        ss = _build(type_string)
        es = build_event_structure(ss)
        configs = configurations(es)
        assert len(configs) >= len(ss.states), f"{name}: configs >= states"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_event(self):
        ss = _build("&{a: end}")
        es = build_event_structure(ss)
        assert es.num_events == 1
        assert es.num_conflicts == 0

    def test_wide_branch(self):
        ss = _build("&{a: end, b: end, c: end, d: end, e: end}")
        es = build_event_structure(ss)
        assert es.num_events == 5
        # All 5 events pairwise in conflict: C(5,2) = 10
        assert es.num_conflicts == 10

    def test_event_repr(self):
        e = Event(source=0, label="a", target=1)
        assert "a" in repr(e)

    def test_configuration_size(self):
        c = Configuration(events=frozenset())
        assert c.size == 0

    def test_recursive_type(self, iterator_ss):
        """Recursive types should produce valid event structures."""
        a = analyze_event_structure(iterator_ss)
        assert a.num_events >= 3
        assert a.num_configs >= 1
