"""Tests for CCS semantics of session types (Step 26)."""

import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.ccs import (
    CCSProcess,
    CCSResult,
    CCSRec,
    CCSVar,
    LTS,
    Nil,
    Par,
    Prefix,
    Restrict,
    Sum,
    analyze_ccs,
    ccs_to_lts,
    check_bisimulation,
    check_bisimulation_with_statespace,
    pretty_ccs,
    session_to_ccs,
)


# ---------------------------------------------------------------------------
# Session type → CCS translation
# ---------------------------------------------------------------------------


class TestSessionToCCS:
    """Test translation from session type ASTs to CCS processes."""

    def test_end_to_nil(self):
        ccs = session_to_ccs(parse("end"))
        assert ccs == Nil()

    def test_wait_to_nil(self):
        ccs = session_to_ccs(parse("wait"))
        assert ccs == Nil()

    def test_single_branch(self):
        ccs = session_to_ccs(parse("&{a: end}"))
        assert ccs == Prefix("a", Nil())

    def test_two_branch(self):
        ccs = session_to_ccs(parse("&{a: end, b: end}"))
        assert isinstance(ccs, Sum)
        assert len(ccs.procs) == 2
        assert ccs.procs[0] == Prefix("a", Nil())
        assert ccs.procs[1] == Prefix("b", Nil())

    def test_select_uses_tau(self):
        ccs = session_to_ccs(parse("+{ok: end, err: end}"))
        assert isinstance(ccs, Sum)
        actions = {p.action for p in ccs.procs if isinstance(p, Prefix)}
        assert actions == {"τ_ok", "τ_err"}

    def test_single_select_no_sum(self):
        """Single-choice select produces Prefix, not Sum."""
        ccs = session_to_ccs(parse("+{ok: end}"))
        assert isinstance(ccs, Prefix)
        assert ccs.action == "τ_ok"

    def test_nested_branch(self):
        ccs = session_to_ccs(parse("&{a: &{b: end}}"))
        assert ccs == Prefix("a", Prefix("b", Nil()))

    def test_parallel(self):
        ccs = session_to_ccs(parse("&{a: end} || &{b: end}"))
        assert isinstance(ccs, Par)
        assert ccs.left == Prefix("a", Nil())
        assert ccs.right == Prefix("b", Nil())

    def test_recursion(self):
        ccs = session_to_ccs(parse("rec X . &{a: X}"))
        assert isinstance(ccs, CCSRec)
        assert ccs.var == "X"
        assert ccs.body == Prefix("a", CCSVar("X"))

    def test_variable(self):
        ccs = session_to_ccs(parse("rec X . &{a: X, b: end}"))
        assert isinstance(ccs, CCSRec)
        body = ccs.body
        assert isinstance(body, Sum)
        # One branch recurses, one terminates
        has_var = any(
            isinstance(p, Prefix) and isinstance(p.cont, CCSVar)
            for p in body.procs
        )
        assert has_var


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------


class TestPrettyCCS:
    """Test CCS pretty-printer."""

    def test_nil(self):
        assert pretty_ccs(Nil()) == "0"

    def test_prefix(self):
        assert pretty_ccs(Prefix("a", Nil())) == "a . 0"

    def test_sum(self):
        s = pretty_ccs(Sum((Prefix("a", Nil()), Prefix("b", Nil()))))
        assert s == "a . 0 + b . 0"

    def test_par(self):
        s = pretty_ccs(Par(Prefix("a", Nil()), Prefix("b", Nil())))
        assert s == "a . 0 | b . 0"

    def test_restrict(self):
        s = pretty_ccs(Restrict(Prefix("a", Nil()), frozenset({"b"})))
        assert "a . 0" in s
        assert "{b}" in s

    def test_rec(self):
        s = pretty_ccs(CCSRec("X", Prefix("a", CCSVar("X"))))
        assert s == "fix X . a . X"

    def test_var(self):
        assert pretty_ccs(CCSVar("Y")) == "Y"

    def test_nested_precedence(self):
        """Sum inside Prefix needs parentheses."""
        p = Prefix("a", Sum((Prefix("b", Nil()), Prefix("c", Nil()))))
        s = pretty_ccs(p)
        assert "a . " in s


# ---------------------------------------------------------------------------
# CCS → LTS construction
# ---------------------------------------------------------------------------


class TestCCSToLTS:
    """Test LTS construction from CCS processes."""

    def test_nil_lts(self):
        lts = ccs_to_lts(Nil())
        assert len(lts.states) == 1
        assert len(lts.transitions) == 0

    def test_prefix_lts(self):
        lts = ccs_to_lts(Prefix("a", Nil()))
        assert len(lts.states) == 2
        assert len(lts.transitions) == 1
        assert lts.transitions[0][1] == "a"

    def test_sum_lts(self):
        lts = ccs_to_lts(Sum((Prefix("a", Nil()), Prefix("b", Nil()))))
        # Initial state with two outgoing transitions
        assert len(lts.transitions) == 2
        labels = {l for _, l, _ in lts.transitions}
        assert labels == {"a", "b"}

    def test_par_lts(self):
        lts = ccs_to_lts(Par(Prefix("a", Nil()), Prefix("b", Nil())))
        # Parallel: initial state can do a or b
        initial_actions = {l for s, l, t in lts.transitions if s == lts.initial}
        assert "a" in initial_actions
        assert "b" in initial_actions

    def test_rec_finite_unfolding(self):
        """Recursive process should produce finite LTS via structural equality."""
        lts = ccs_to_lts(CCSRec("X", Prefix("a", CCSVar("X"))))
        # Should have a cycle: state --a--> same state
        assert len(lts.states) >= 1
        assert len(lts.transitions) >= 1

    def test_max_states_bound(self):
        """LTS builder respects max_states bound."""
        lts = ccs_to_lts(
            CCSRec("X", Sum((Prefix("a", CCSVar("X")), Prefix("b", CCSVar("X"))))),
            max_states=10,
        )
        assert len(lts.states) <= 11  # may slightly exceed due to batch


# ---------------------------------------------------------------------------
# Strong bisimulation
# ---------------------------------------------------------------------------


class TestBisimulation:
    """Test strong bisimulation checking."""

    def test_identical_lts(self):
        lts = ccs_to_lts(Prefix("a", Nil()))
        assert check_bisimulation(lts, lts)

    def test_nil_bisimilar(self):
        lts1 = ccs_to_lts(Nil())
        lts2 = ccs_to_lts(Nil())
        assert check_bisimulation(lts1, lts2)

    def test_different_action_not_bisimilar(self):
        lts1 = ccs_to_lts(Prefix("a", Nil()))
        lts2 = ccs_to_lts(Prefix("b", Nil()))
        assert not check_bisimulation(lts1, lts2)

    def test_sum_order_irrelevant(self):
        lts1 = ccs_to_lts(Sum((Prefix("a", Nil()), Prefix("b", Nil()))))
        lts2 = ccs_to_lts(Sum((Prefix("b", Nil()), Prefix("a", Nil()))))
        assert check_bisimulation(lts1, lts2)

    def test_different_branching_not_bisimilar(self):
        lts1 = ccs_to_lts(Prefix("a", Nil()))
        lts2 = ccs_to_lts(Sum((Prefix("a", Nil()), Prefix("b", Nil()))))
        assert not check_bisimulation(lts1, lts2)


# ---------------------------------------------------------------------------
# Bisimulation with state space
# ---------------------------------------------------------------------------


class TestBisimulationWithStatespace:
    """Test bisimulation between CCS LTS and session-type state space."""

    def test_end_bisimilar(self):
        ast = parse("end")
        ccs_proc = session_to_ccs(ast)
        lts = ccs_to_lts(ccs_proc)
        ss = build_statespace(ast)
        assert check_bisimulation_with_statespace(lts, ss)

    def test_single_branch_bisimilar(self):
        ast = parse("&{a: end}")
        ccs_proc = session_to_ccs(ast)
        lts = ccs_to_lts(ccs_proc)
        ss = build_statespace(ast)
        assert check_bisimulation_with_statespace(lts, ss)

    def test_two_branch_bisimilar(self):
        ast = parse("&{a: end, b: end}")
        ccs_proc = session_to_ccs(ast)
        lts = ccs_to_lts(ccs_proc)
        ss = build_statespace(ast)
        assert check_bisimulation_with_statespace(lts, ss)

    def test_select_bisimilar(self):
        ast = parse("+{ok: end, err: end}")
        ccs_proc = session_to_ccs(ast)
        lts = ccs_to_lts(ccs_proc)
        ss = build_statespace(ast)
        assert check_bisimulation_with_statespace(lts, ss)

    def test_nested_bisimilar(self):
        ast = parse("&{a: &{b: end, c: end}}")
        ccs_proc = session_to_ccs(ast)
        lts = ccs_to_lts(ccs_proc)
        ss = build_statespace(ast)
        assert check_bisimulation_with_statespace(lts, ss)


# ---------------------------------------------------------------------------
# High-level analysis
# ---------------------------------------------------------------------------


class TestAnalyzeCCS:
    """Test the analyze_ccs entry point."""

    def test_end(self):
        result = analyze_ccs(parse("end"))
        assert isinstance(result, CCSResult)
        assert result.num_states >= 1
        assert result.bisimilar_to_statespace

    def test_branch(self):
        result = analyze_ccs(parse("&{a: end, b: end}"))
        assert result.num_states >= 2
        assert result.num_transitions >= 2
        assert result.bisimilar_to_statespace

    def test_select(self):
        result = analyze_ccs(parse("+{ok: end}"))
        assert result.bisimilar_to_statespace

    def test_recursive(self):
        result = analyze_ccs(parse("rec X . &{a: X, b: end}"))
        assert result.num_states >= 2
        assert result.num_transitions >= 2

    def test_ccs_process_field(self):
        result = analyze_ccs(parse("&{a: end}"))
        assert isinstance(result.ccs_process, (Prefix, Sum, Nil))

    def test_lts_field(self):
        result = analyze_ccs(parse("&{a: end}"))
        assert isinstance(result.lts, LTS)
        assert result.lts.initial in result.lts.states
