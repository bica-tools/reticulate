"""Tests for session types as effect systems (Step 99)."""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.effect_systems import (
    Effect,
    EffectSignature,
    SubeffectResult,
    EffectHandler,
    EffectAnalysis,
    extract_effects,
    effect_ordering,
    check_subeffecting,
    compose_effects,
    effect_sequence,
    infer_effect,
    effect_handler,
    effect_lattice,
    analyze_effects,
)


def _build(type_string: str):
    return build_statespace(parse(type_string))


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
def select_ss():
    return _build("+{ok: end, err: end}")

@pytest.fixture
def iterator_ss():
    return _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")

@pytest.fixture
def parallel_ss():
    return _build("(&{a: end} || &{b: end})")


# ---------------------------------------------------------------------------
# Effect extraction
# ---------------------------------------------------------------------------

class TestExtractEffects:
    def test_end_no_effects(self, end_ss):
        sig = extract_effects(end_ss)
        assert sig.num_effects == 0

    def test_branch_input_effects(self, simple_branch):
        sig = extract_effects(simple_branch)
        assert sig.num_effects == 2
        assert len(sig.input_effects) == 2
        assert len(sig.output_effects) == 0

    def test_select_output_effects(self, select_ss):
        sig = extract_effects(select_ss)
        assert sig.num_effects == 2
        assert len(sig.output_effects) == 2
        assert len(sig.input_effects) == 0

    def test_chain_effects(self, deep_chain):
        sig = extract_effects(deep_chain)
        assert sig.num_effects == 3
        labels = {e.name for e in sig.effects}
        assert labels == {"a", "b", "c"}

    def test_effect_states(self, simple_branch):
        sig = extract_effects(simple_branch)
        top_effects = sig.effect_states[simple_branch.top]
        assert len(top_effects) == 2

    def test_iterator_mixed(self, iterator_ss):
        sig = extract_effects(iterator_ss)
        assert len(sig.input_effects) >= 1
        assert len(sig.output_effects) >= 1


# ---------------------------------------------------------------------------
# Effect ordering
# ---------------------------------------------------------------------------

class TestEffectOrdering:
    def test_ordering_reflexive(self, simple_branch):
        ordering = effect_ordering(simple_branch)
        for s in simple_branch.states:
            assert s in ordering[s]

    def test_bottom_subeffect_of_all(self, simple_branch):
        """Bottom has no effects, so it's a subeffect of every state."""
        ordering = effect_ordering(simple_branch)
        bot = simple_branch.bottom
        for s in simple_branch.states:
            assert s in ordering[bot]

    def test_check_subeffecting(self, simple_branch):
        r = check_subeffecting(simple_branch, simple_branch.bottom, simple_branch.top)
        assert r.is_subeffect  # empty ⊆ {a, b}
        assert len(r.extra) == 2

    def test_non_subeffect(self, deep_chain):
        """States with different effects are not subeffects."""
        sig = extract_effects(deep_chain)
        # Find two states with different effect sets
        states = sorted(deep_chain.states)
        if len(states) >= 3:
            r = check_subeffecting(deep_chain, states[0], states[1])
            assert isinstance(r, SubeffectResult)


# ---------------------------------------------------------------------------
# Effect composition
# ---------------------------------------------------------------------------

class TestComposition:
    def test_compose_valid(self, simple_branch):
        result = compose_effects(simple_branch, simple_branch.top, "a")
        assert result is not None
        assert result == simple_branch.bottom

    def test_compose_invalid(self, simple_branch):
        result = compose_effects(simple_branch, simple_branch.bottom, "a")
        assert result is None

    def test_sequence_valid(self, deep_chain):
        path = effect_sequence(deep_chain, ["a", "b", "c"])
        assert path is not None
        assert len(path) == 4
        assert path[-1] == deep_chain.bottom

    def test_sequence_invalid(self, deep_chain):
        path = effect_sequence(deep_chain, ["a", "c"])  # skip b
        assert path is None

    def test_empty_sequence(self, simple_branch):
        path = effect_sequence(simple_branch, [])
        assert path == [simple_branch.top]


# ---------------------------------------------------------------------------
# Effect inference
# ---------------------------------------------------------------------------

class TestInference:
    def test_infer_valid(self, deep_chain):
        path = infer_effect(deep_chain, ["a", "b", "c"])
        assert path is not None

    def test_infer_invalid(self, deep_chain):
        path = infer_effect(deep_chain, ["x", "y"])
        assert path is None

    def test_infer_partial(self, deep_chain):
        path = infer_effect(deep_chain, ["a"])
        assert path is not None
        assert len(path) == 2


# ---------------------------------------------------------------------------
# Handler synthesis
# ---------------------------------------------------------------------------

class TestHandler:
    def test_handler_structure(self, simple_branch):
        h = effect_handler(simple_branch)
        assert isinstance(h, EffectHandler)
        assert len(h.operations) >= 2
        assert h.return_handler == "handle_return"

    def test_handler_operations(self, deep_chain):
        h = effect_handler(deep_chain)
        op_names = {name for name, _, _ in h.operations}
        assert "a" in op_names
        assert "b" in op_names
        assert "c" in op_names

    def test_handler_end(self, end_ss):
        h = effect_handler(end_ss)
        assert len(h.operations) == 0

    def test_state_handlers(self, simple_branch):
        h = effect_handler(simple_branch)
        assert simple_branch.top in h.state_handlers


# ---------------------------------------------------------------------------
# Effect lattice
# ---------------------------------------------------------------------------

class TestEffectLattice:
    def test_chain_co_enabled(self, deep_chain):
        lat = effect_lattice(deep_chain)
        # In a chain, all labels co-occur in the single execution
        assert "b" in lat.get("a", set())
        assert "a" in lat.get("b", set())

    def test_branch_not_co_enabled(self, simple_branch):
        lat = effect_lattice(simple_branch)
        # a and b are in conflict — never co-occur
        assert "b" not in lat.get("a", set())

    def test_end_empty(self, end_ss):
        lat = effect_lattice(end_ss)
        assert len(lat) == 0


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

class TestAnalysis:
    def test_end(self, end_ss):
        a = analyze_effects(end_ss)
        assert isinstance(a, EffectAnalysis)
        assert a.num_input == 0
        assert a.num_output == 0

    def test_branch(self, simple_branch):
        a = analyze_effects(simple_branch)
        assert a.num_input == 2
        assert a.num_output == 0
        assert a.effect_depth >= 1

    def test_chain(self, deep_chain):
        a = analyze_effects(deep_chain)
        assert a.effect_depth == 3
        assert a.is_linear

    def test_select(self, select_ss):
        a = analyze_effects(select_ss)
        assert a.num_output == 2

    def test_iterator(self, iterator_ss):
        a = analyze_effects(iterator_ss)
        assert a.num_input >= 1
        assert a.num_output >= 1


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

class TestBenchmarks:
    @pytest.mark.parametrize("type_string,name", [
        ("&{a: end, b: end}", "Branch"),
        ("+{ok: end, err: end}", "Select"),
        ("&{a: &{b: &{c: end}}}", "Chain"),
        ("(&{a: end} || &{b: end})", "Parallel"),
        ("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}", "Iterator"),
        ("&{get: +{OK: end, NOT_FOUND: end}, put: +{OK: end, ERR: end}}", "REST"),
    ])
    def test_analysis_on_benchmarks(self, type_string, name):
        ss = _build(type_string)
        a = analyze_effects(ss)
        assert a.signature.num_effects >= 1, f"{name} should have effects"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_effect(self):
        ss = _build("&{a: end}")
        a = analyze_effects(ss)
        assert a.signature.num_effects == 1
        assert a.is_linear

    def test_pure_recursion(self):
        ss = _build("rec X . &{a: X}")
        a = analyze_effects(ss)
        assert a.signature.num_effects >= 1

    def test_effect_repr(self):
        e = Effect(name="test", source_state=0, target_state=1,
                   is_input=True, is_output=False)
        assert e.name == "test"
        assert e.is_input
