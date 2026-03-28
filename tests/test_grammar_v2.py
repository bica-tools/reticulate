"""Tests for grammar_v2: real-world grammar extensions (Step 500).

Covers AST nodes, state-space construction, lattice preservation, the
survey function, and edge cases.
"""

from __future__ import annotations

import pytest

from reticulate.parser import Branch, End, Select, Rec, Var, parse
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice
from reticulate.extensions.grammar_v2 import (
    Guard,
    Interrupt,
    Probabilistic,
    Priority,
    Timeout,
    TryCatch,
    _add_escape_transitions,
    build_extension_statespace,
    build_guard_statespace,
    build_interrupt_statespace,
    build_priority_statespace,
    build_probabilistic_statespace,
    build_timeout_statespace,
    build_trycatch_statespace,
    check_extension_lattice,
    survey_lattice_preservation,
)


# =========================================================================
# AST node tests (6 tests)
# =========================================================================

class TestASTNodes:
    """Each new AST node is frozen, has correct fields, and repr works."""

    def test_timeout_frozen_fields(self) -> None:
        t = Timeout(body=End(), fallback=End(), limit=5)
        assert t.body == End()
        assert t.fallback == End()
        assert t.limit == 5
        with pytest.raises(AttributeError):
            t.limit = 10  # type: ignore[misc]

    def test_trycatch_frozen_fields(self) -> None:
        tc = TryCatch(body=End(), handler=End())
        assert tc.body == End()
        assert tc.handler == End()
        with pytest.raises(AttributeError):
            tc.body = End()  # type: ignore[misc]

    def test_priority_frozen_fields(self) -> None:
        p = Priority(choices=(("high", End()), ("low", End())))
        assert len(p.choices) == 2
        assert p.choices[0][0] == "high"
        with pytest.raises(AttributeError):
            p.choices = ()  # type: ignore[misc]

    def test_probabilistic_frozen_fields(self) -> None:
        pr = Probabilistic(choices=((0.9, End()), (0.1, End())))
        assert pr.choices[0][0] == pytest.approx(0.9)
        with pytest.raises(AttributeError):
            pr.choices = ()  # type: ignore[misc]

    def test_guard_frozen_fields(self) -> None:
        g = Guard(condition="ready", then_branch=End(), else_branch=End())
        assert g.condition == "ready"
        with pytest.raises(AttributeError):
            g.condition = "x"  # type: ignore[misc]

    def test_interrupt_frozen_fields(self) -> None:
        i = Interrupt(signal="SIGINT", body=End(), handler=End())
        assert i.signal == "SIGINT"
        with pytest.raises(AttributeError):
            i.signal = "x"  # type: ignore[misc]


# =========================================================================
# State-space construction tests (12 tests)
# =========================================================================

class TestStateSpaceConstruction:
    """State-space construction for each extension."""

    def test_timeout_adds_escape_transitions(self) -> None:
        body = parse("&{a: end, b: end}")
        node = Timeout(body=body, fallback=End())
        ss = build_timeout_statespace(node)
        labels = [lbl for _, lbl, _ in ss.transitions]
        assert "timeout" in labels

    def test_timeout_state_counts(self) -> None:
        body = parse("&{a: end, b: end}")
        fallback = parse("&{recover: end}")
        node = Timeout(body=body, fallback=fallback)
        ss = build_timeout_statespace(node)
        body_ss = build_statespace(body)
        fallback_ss = build_statespace(fallback)
        # Handler bottom merged into body bottom, so -1 from handler
        expected = len(body_ss.states) + len(fallback_ss.states) - 1
        assert len(ss.states) == expected

    def test_trycatch_uses_exception_label(self) -> None:
        body = parse("&{a: end, b: end}")
        node = TryCatch(body=body, handler=End())
        ss = build_trycatch_statespace(node)
        labels = [lbl for _, lbl, _ in ss.transitions]
        assert "exception" in labels

    def test_trycatch_structurally_same_as_timeout(self) -> None:
        body = parse("&{a: end, b: end}")
        handler = parse("&{recover: end}")
        timeout_ss = build_timeout_statespace(Timeout(body=body, fallback=handler))
        trycatch_ss = build_trycatch_statespace(TryCatch(body=body, handler=handler))
        assert len(timeout_ss.states) == len(trycatch_ss.states)
        assert len(timeout_ss.transitions) == len(trycatch_ss.transitions)

    def test_priority_same_state_count_as_branch(self) -> None:
        choices = (("high", End()), ("low", End()))
        prio_ss = build_priority_statespace(Priority(choices=choices))
        branch_ss = build_statespace(Branch(choices))
        assert len(prio_ss.states) == len(branch_ss.states)
        assert len(prio_ss.transitions) == len(branch_ss.transitions)

    def test_priority_three_choices(self) -> None:
        body = parse("&{a: end, b: end}")
        choices = (("high", End()), ("medium", body), ("low", End()))
        ss = build_priority_statespace(Priority(choices=choices))
        assert ss.top in ss.states
        assert ss.bottom in ss.states

    def test_probabilistic_same_state_count_as_select(self) -> None:
        s1, s2 = End(), End()
        prob_ss = build_probabilistic_statespace(
            Probabilistic(choices=((0.7, s1), (0.3, s2)))
        )
        select_ss = build_statespace(Select((("a", s1), ("b", s2))))
        assert len(prob_ss.states) == len(select_ss.states)

    def test_probabilistic_labels_contain_probabilities(self) -> None:
        ss = build_probabilistic_statespace(
            Probabilistic(choices=((0.9, End()), (0.1, End())))
        )
        labels = {lbl for _, lbl, _ in ss.transitions}
        assert any("0.9" in lbl for lbl in labels)
        assert any("0.1" in lbl for lbl in labels)

    def test_guard_creates_two_branch_structure(self) -> None:
        node = Guard(condition="ready", then_branch=End(), else_branch=End())
        ss = build_guard_statespace(node)
        labels = {lbl for _, lbl, _ in ss.transitions}
        assert "ready_true" in labels
        assert "ready_false" in labels

    def test_guard_state_count_matches_branch(self) -> None:
        then_b = parse("&{a: end}")
        else_b = parse("&{b: end}")
        guard_ss = build_guard_statespace(
            Guard(condition="c", then_branch=then_b, else_branch=else_b)
        )
        branch_ss = build_statespace(Branch((("c_true", then_b), ("c_false", else_b))))
        assert len(guard_ss.states) == len(branch_ss.states)

    def test_interrupt_escape_from_all_non_bottom(self) -> None:
        body = parse("&{a: end, b: end}")
        handler = parse("&{recover: end}")
        node = Interrupt(signal="cancel", body=body, handler=handler)
        ss = build_interrupt_statespace(node)
        body_ss = build_statespace(body)
        cancel_transitions = [(s, t) for s, lbl, t in ss.transitions if lbl == "cancel"]
        # Every non-bottom body state should have a cancel transition
        non_bottom_body = {s for s in body_ss.states if s != body_ss.bottom}
        sources = {s for s, _ in cancel_transitions}
        assert non_bottom_body <= sources

    def test_interrupt_preserves_body_transitions(self) -> None:
        body = parse("&{a: end, b: end}")
        node = Interrupt(signal="sig", body=body, handler=End())
        ss = build_interrupt_statespace(node)
        body_ss = build_statespace(body)
        # All body transitions should still be present
        for src, lbl, tgt in body_ss.transitions:
            assert (src, lbl, tgt) in ss.transitions


# =========================================================================
# Lattice preservation tests (12 tests)
# =========================================================================

class TestLatticePreservation:
    """Lattice preservation for each extension."""

    def test_priority_preserves_lattice_simple(self) -> None:
        r = check_extension_lattice("priority", "&{a: end, b: end}")
        assert r["is_lattice"] is True

    def test_priority_preserves_lattice_nested(self) -> None:
        r = check_extension_lattice("priority", "&{a: +{x: end, y: end}, b: end}")
        assert r["is_lattice"] is True

    def test_probabilistic_preserves_lattice(self) -> None:
        r = check_extension_lattice("probabilistic", "+{a: end, b: end}")
        assert r["is_lattice"] is True

    def test_probabilistic_preserves_lattice_nested(self) -> None:
        r = check_extension_lattice(
            "probabilistic", "&{a: +{x: end, y: end}, b: end}"
        )
        assert r["is_lattice"] is True

    def test_guard_preserves_lattice(self) -> None:
        r = check_extension_lattice("guard", "&{a: end}", "&{b: end}")
        assert r["is_lattice"] is True

    def test_guard_preserves_lattice_complex(self) -> None:
        r = check_extension_lattice(
            "guard", "&{a: +{x: end, y: end}, b: end}", "&{c: end}"
        )
        assert r["is_lattice"] is True

    def test_timeout_simple_body_is_lattice(self) -> None:
        r = check_extension_lattice("timeout", "&{a: end, b: end}", "&{recover: end}")
        assert r["is_lattice"] is True

    def test_trycatch_simple_body_is_lattice(self) -> None:
        r = check_extension_lattice(
            "trycatch", "&{a: end, b: end}", "&{recover: end}"
        )
        assert r["is_lattice"] is True

    def test_interrupt_end_body_not_lattice(self) -> None:
        # An end body has top == bottom, so no escape transitions are added.
        # The handler states become unreachable -- correctly NOT a lattice,
        # because interrupting a terminated session is vacuous.
        r = check_extension_lattice("interrupt", "end", "&{recover: end}")
        assert r["is_lattice"] is False

    def test_interrupt_simple_branch_body(self) -> None:
        r = check_extension_lattice(
            "interrupt", "&{a: end, b: end}", "&{recover: end}"
        )
        # Record whether lattice holds -- this is the key research question
        assert isinstance(r["is_lattice"], bool)
        assert r["total_states"] > 0

    def test_interrupt_recursive_body(self) -> None:
        r = check_extension_lattice(
            "interrupt",
            "rec X . &{a: +{ok: X, done: end}}",
            "&{recover: end}",
        )
        assert isinstance(r["is_lattice"], bool)
        assert r["total_states"] > 0

    def test_timeout_with_complex_handler(self) -> None:
        r = check_extension_lattice(
            "timeout",
            "&{a: end, b: end}",
            "&{log: +{retry: end, abort: end}}",
        )
        assert isinstance(r["is_lattice"], bool)
        assert r["total_states"] > 0


# =========================================================================
# Survey tests (5 tests)
# =========================================================================

class TestSurvey:
    """Survey function returns expected structure."""

    def test_survey_returns_all_extensions(self) -> None:
        results = survey_lattice_preservation()
        expected = {"timeout", "trycatch", "interrupt", "priority", "probabilistic", "guard"}
        assert set(results.keys()) == expected

    def test_survey_each_extension_has_tests(self) -> None:
        results = survey_lattice_preservation()
        for ext, summary in results.items():
            assert summary["total_tests"] > 0, f"{ext} has no tests"

    def test_survey_priority_all_lattice(self) -> None:
        results = survey_lattice_preservation()
        prio = results["priority"]
        assert prio["non_lattice_count"] == 0
        assert prio["error_count"] == 0

    def test_survey_probabilistic_all_lattice(self) -> None:
        results = survey_lattice_preservation()
        prob = results["probabilistic"]
        assert prob["non_lattice_count"] == 0
        assert prob["error_count"] == 0

    def test_survey_guard_all_lattice(self) -> None:
        results = survey_lattice_preservation()
        guard = results["guard"]
        assert guard["non_lattice_count"] == 0
        assert guard["error_count"] == 0


# =========================================================================
# Edge cases (5 tests)
# =========================================================================

class TestEdgeCases:
    """Edge cases for grammar extensions."""

    def test_end_body_timeout(self) -> None:
        node = Timeout(body=End(), fallback=End())
        ss = build_timeout_statespace(node)
        result = check_lattice(ss)
        assert result.is_lattice

    def test_end_body_interrupt(self) -> None:
        # End body: top == bottom, handler is End too (single state merged),
        # so the result is a single-state lattice.
        node = Interrupt(signal="sig", body=End(), handler=End())
        ss = build_interrupt_statespace(node)
        # Both body and handler are End (single state), bottoms merge,
        # result is a single state -- trivially a lattice.
        result = check_lattice(ss)
        assert result.is_lattice

    def test_recursive_body_trycatch(self) -> None:
        body = parse("rec X . &{a: +{ok: X, done: end}}")
        handler = parse("&{recover: end}")
        node = TryCatch(body=body, handler=handler)
        ss = build_trycatch_statespace(node)
        assert len(ss.states) > 0
        assert ss.top in ss.states
        assert ss.bottom in ss.states

    def test_build_extension_dispatch(self) -> None:
        node = Timeout(body=End(), fallback=End())
        ss = build_extension_statespace(node)
        assert ss.top in ss.states

    def test_build_extension_unknown_type_raises(self) -> None:
        with pytest.raises(TypeError, match="No grammar-v2 builder"):
            build_extension_statespace("not a node")  # type: ignore[arg-type]
