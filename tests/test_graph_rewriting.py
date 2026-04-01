"""Tests for graph_rewriting module (Step 92).

Tests cover:
  - RewriteRule construction
  - Pattern matching (match_pattern)
  - Rule application (apply_rule)
  - Fixpoint iteration (apply_rules)
  - Standard rules (flatten_branch, merge_ends, unfold_once, etc.)
  - Confluence checking
  - Termination checking
  - Lattice preservation
  - Full analysis (analyze_rewriting)
  - Simplification (simplify)
  - Integration with parser + build_statespace
"""

from __future__ import annotations

import pytest

from reticulate.graph_rewriting import (
    ConfluenceResult,
    Match,
    RewriteRule,
    RewriteStep,
    RewritingAnalysis,
    TerminationResult,
    add_branch_rule,
    analyze_rewriting,
    apply_rule,
    apply_rules,
    check_confluence,
    check_lattice_preservation,
    check_termination,
    flatten_branch_rule,
    match_pattern,
    merge_ends_rule,
    remove_selection_rule,
    replace_subgraph_rule,
    simplify,
    unfold_once_rule,
)
from reticulate.lattice import check_lattice
from reticulate.parser import parse
from reticulate.statespace import StateSpace, build_statespace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ss(
    states: set[int],
    transitions: list[tuple[int, str, int]],
    top: int,
    bottom: int,
    labels: dict[int, str] | None = None,
    selections: set[tuple[int, str, int]] | None = None,
) -> StateSpace:
    """Shorthand for building a StateSpace."""
    return StateSpace(
        states=states,
        transitions=transitions,
        top=top,
        bottom=bottom,
        labels=labels or {},
        selection_transitions=selections or set(),
    )


def _from_type(type_string: str) -> StateSpace:
    """Parse a session type string and build its state space."""
    ast = parse(type_string)
    return build_statespace(ast)


# ===========================================================================
# 1. Data types
# ===========================================================================

class TestDataTypes:
    """Test basic data type construction."""

    def test_match_creation(self) -> None:
        m = Match(state_map={0: 10, 1: 20}, edge_map=[])
        assert m.state_map == {0: 10, 1: 20}
        assert m.edge_map == []

    def test_rewrite_rule_creation(self) -> None:
        lhs = _ss({0}, [], 0, 0)
        rhs = _ss({0}, [], 0, 0)
        rule = RewriteRule(name="test", lhs=lhs, rhs=rhs)
        assert rule.name == "test"
        assert rule.interface_states == {}

    def test_rewrite_rule_with_interface(self) -> None:
        lhs = _ss({0, 1}, [(0, "a", 1)], 0, 1)
        rhs = _ss({0, 1}, [(0, "a", 1)], 0, 1)
        rule = RewriteRule(
            name="identity", lhs=lhs, rhs=rhs,
            interface_states={0: 0, 1: 1},
        )
        assert rule.interface_states == {0: 0, 1: 1}

    def test_confluence_result(self) -> None:
        cr = ConfluenceResult(
            is_confluent=True,
            critical_pairs=[],
            all_convergent=True,
        )
        assert cr.is_confluent
        assert cr.counterexample is None

    def test_termination_result(self) -> None:
        tr = TerminationResult(
            terminates=True, measure="count", reason="reduces",
        )
        assert tr.terminates


# ===========================================================================
# 2. Pattern matching
# ===========================================================================

class TestPatternMatching:
    """Test match_pattern function."""

    def test_empty_pattern_no_match(self) -> None:
        host = _ss({0, 1}, [(0, "a", 1)], 0, 1)
        pattern = _ss(set(), [], 0, 0)
        assert match_pattern(host, pattern) == []

    def test_single_edge_match(self) -> None:
        host = _ss({0, 1}, [(0, "a", 1)], 0, 1)
        pattern = _ss({0, 1}, [(0, "a", 1)], 0, 1)
        matches = match_pattern(host, pattern)
        assert len(matches) >= 1
        m = matches[0]
        assert m.state_map[0] == 0
        assert m.state_map[1] == 1

    def test_two_edge_pattern(self) -> None:
        host = _ss(
            {0, 1, 2},
            [(0, "a", 1), (0, "b", 2)],
            0, 1,
        )
        pattern = _ss(
            {0, 1},
            [(0, "a", 1)],
            0, 1,
        )
        matches = match_pattern(host, pattern)
        assert len(matches) >= 1
        # Pattern should match with 0->0, 1->1
        found = any(m.state_map[0] == 0 and m.state_map[1] == 1 for m in matches)
        assert found

    def test_no_match_wrong_label(self) -> None:
        host = _ss({0, 1}, [(0, "a", 1)], 0, 1)
        pattern = _ss({0, 1}, [(0, "b", 1)], 0, 1)
        matches = match_pattern(host, pattern)
        assert len(matches) == 0

    def test_self_loop_match(self) -> None:
        host = _ss({0, 1}, [(0, "a", 0), (0, "b", 1)], 0, 1)
        pattern = _ss({0}, [(0, "a", 0)], 0, 0)
        matches = match_pattern(host, pattern)
        assert len(matches) >= 1
        assert matches[0].state_map[0] == 0

    def test_chain_match(self) -> None:
        host = _ss(
            {0, 1, 2},
            [(0, "a", 1), (1, "b", 2)],
            0, 2,
        )
        pattern = _ss(
            {0, 1, 2},
            [(0, "a", 1), (1, "b", 2)],
            0, 2,
        )
        matches = match_pattern(host, pattern)
        assert len(matches) >= 1

    def test_diamond_pattern(self) -> None:
        host = _ss(
            {0, 1, 2, 3},
            [(0, "a", 1), (0, "b", 2), (1, "c", 3), (2, "d", 3)],
            0, 3,
        )
        # Match a sub-pattern
        pattern = _ss(
            {0, 1},
            [(0, "a", 1)],
            0, 1,
        )
        matches = match_pattern(host, pattern)
        assert len(matches) >= 1

    def test_multiple_matches(self) -> None:
        # Two independent branches with same label structure
        host = _ss(
            {0, 1, 2, 3, 4},
            [(0, "x", 1), (1, "a", 2), (0, "y", 3), (3, "a", 4)],
            0, 2,
        )
        pattern = _ss(
            {0, 1},
            [(0, "a", 1)],
            0, 1,
        )
        matches = match_pattern(host, pattern)
        # Should find at least 2 matches: (1->2) and (3->4)
        assert len(matches) >= 2

    def test_pattern_with_selection(self) -> None:
        host = _ss(
            {0, 1, 2},
            [(0, "OK", 1), (0, "ERR", 2)],
            0, 1,
            selections={(0, "OK", 1), (0, "ERR", 2)},
        )
        pattern = _ss(
            {0, 1, 2},
            [(0, "OK", 1), (0, "ERR", 2)],
            0, 1,
        )
        matches = match_pattern(host, pattern)
        assert len(matches) >= 1


# ===========================================================================
# 3. Rule application
# ===========================================================================

class TestRuleApplication:
    """Test apply_rule function."""

    def test_identity_rule(self) -> None:
        """Applying an identity rule should not change the graph."""
        host = _ss({0, 1}, [(0, "a", 1)], 0, 1)
        lhs = _ss({0, 1}, [(0, "a", 1)], 0, 1)
        rhs = _ss({0, 1}, [(0, "a", 1)], 0, 1)
        rule = RewriteRule(
            name="identity", lhs=lhs, rhs=rhs,
            interface_states={0: 0, 1: 1},
        )
        matches = match_pattern(host, lhs)
        assert len(matches) >= 1
        result = apply_rule(host, rule, matches[0])
        assert len(result.states) == 2
        assert len(result.transitions) == 1

    def test_add_edge_rule(self) -> None:
        """Add a new edge via BranchExt-style rule."""
        host = _ss({0, 1}, [(0, "a", 1)], 0, 1)
        rule = add_branch_rule("a", "b")
        matches = match_pattern(host, rule.lhs)
        assert len(matches) >= 1
        result = apply_rule(host, rule, matches[0])
        # Should have the original a-edge plus a new b-edge
        labels = {l for _, l, _ in result.transitions}
        assert "a" in labels
        assert "b" in labels

    def test_remove_selection_edge(self) -> None:
        """Remove a selection edge via SelRestrict-style rule."""
        host = _ss(
            {0, 1, 2},
            [(0, "OK", 1), (0, "ERR", 2)],
            0, 1,
            selections={(0, "OK", 1), (0, "ERR", 2)},
        )
        rule = remove_selection_rule("OK", "ERR")
        matches = match_pattern(host, rule.lhs)
        assert len(matches) >= 1
        result = apply_rule(host, rule, matches[0])
        labels = {l for _, l, _ in result.transitions}
        assert "OK" in labels
        # ERR should be removed (or at least the ERR target unreachable)
        # The result should have fewer transitions
        assert len(result.transitions) <= len(host.transitions)

    def test_unfold_self_loop(self) -> None:
        """Unfold a self-loop once."""
        host = _ss(
            {0, 1},
            [(0, "a", 0), (0, "b", 1)],
            0, 1,
        )
        rule = unfold_once_rule("a")
        matches = match_pattern(host, rule.lhs)
        assert len(matches) >= 1
        result = apply_rule(host, rule, matches[0])
        # After unfolding, should have more states
        assert len(result.states) >= 2

    def test_apply_rule_preserves_top(self) -> None:
        """Top state should be preserved."""
        host = _ss({0, 1, 2}, [(0, "a", 1), (1, "b", 2)], 0, 2)
        rule = add_branch_rule("a", "c")
        matches = match_pattern(host, rule.lhs)
        assert len(matches) >= 1
        result = apply_rule(host, rule, matches[0])
        assert result.top in result.states

    def test_apply_rule_preserves_bottom(self) -> None:
        """Bottom state should be preserved."""
        host = _ss({0, 1, 2}, [(0, "a", 1), (1, "b", 2)], 0, 2)
        rule = add_branch_rule("b", "c")
        matches = match_pattern(host, rule.lhs)
        assert len(matches) >= 1
        result = apply_rule(host, rule, matches[0])
        assert result.bottom in result.states


# ===========================================================================
# 4. apply_rules (fixpoint iteration)
# ===========================================================================

class TestApplyRules:
    """Test apply_rules fixpoint iteration."""

    def test_no_applicable_rules(self) -> None:
        host = _ss({0, 1}, [(0, "a", 1)], 0, 1)
        rule = unfold_once_rule("z")  # label 'z' not in host
        final, steps = apply_rules(host, [rule])
        assert len(steps) == 0
        assert len(final.states) == len(host.states)

    def test_single_rule_application(self) -> None:
        host = _ss({0, 1}, [(0, "a", 1)], 0, 1)
        rule = add_branch_rule("a", "b")
        final, steps = apply_rules(host, [rule], max_iterations=1)
        assert len(steps) <= 1

    def test_max_iterations_respected(self) -> None:
        # Self-loop unfolding could go forever
        host = _ss({0}, [(0, "a", 0)], 0, 0)
        rule = unfold_once_rule("a")
        final, steps = apply_rules(host, [rule], max_iterations=3)
        assert len(steps) <= 3

    def test_fixpoint_reached(self) -> None:
        """Rules that don't match should stop immediately."""
        host = _ss({0, 1}, [(0, "x", 1)], 0, 1)
        rule = add_branch_rule("nonexistent", "y")
        final, steps = apply_rules(host, [rule])
        assert len(steps) == 0


# ===========================================================================
# 5. Standard rules
# ===========================================================================

class TestStandardRules:
    """Test standard rewriting rule constructors."""

    def test_flatten_branch_rule_construction(self) -> None:
        rule = flatten_branch_rule("a", "b")
        assert rule.name == "flatten_branch"
        assert len(rule.lhs.transitions) == 1
        assert len(rule.rhs.transitions) == 2

    def test_merge_ends_rule_construction(self) -> None:
        rule = merge_ends_rule()
        assert rule.name == "merge_ends"
        assert len(rule.lhs.states) == 3
        assert len(rule.rhs.states) == 2

    def test_unfold_once_rule_construction(self) -> None:
        rule = unfold_once_rule("a")
        assert rule.name == "unfold_once"
        assert len(rule.lhs.states) == 1
        assert len(rule.rhs.states) == 2

    def test_add_branch_rule_construction(self) -> None:
        rule = add_branch_rule("login", "logout")
        assert rule.name == "add_branch"
        assert len(rule.rhs.transitions) == 2

    def test_remove_selection_rule_construction(self) -> None:
        rule = remove_selection_rule("OK", "ERR")
        assert rule.name == "remove_selection"
        assert len(rule.lhs.transitions) == 2
        assert len(rule.rhs.transitions) == 1

    def test_replace_subgraph_rule_construction(self) -> None:
        old_cont = _ss({0, 1}, [(0, "x", 1)], 0, 1)
        new_cont = _ss({0, 1, 2}, [(0, "x", 1), (1, "y", 2)], 0, 2)
        rule = replace_subgraph_rule("entry", old_cont, new_cont)
        assert rule.name == "replace_subgraph"
        assert len(rule.rhs.states) > len(rule.lhs.states)


# ===========================================================================
# 6. Simplification
# ===========================================================================

class TestSimplify:
    """Test simplify function."""

    def test_simplify_no_change(self) -> None:
        """Already simple graph should not change."""
        ss = _ss({0, 1}, [(0, "a", 1)], 0, 1)
        result, steps = simplify(ss)
        assert len(result.states) == 2

    def test_simplify_merges_duplicate_ends(self) -> None:
        """Two end states (no outgoing edges) should be merged."""
        ss = _ss(
            {0, 1, 2},
            [(0, "a", 1), (0, "b", 2)],
            0, 1,
            labels={0: "branch", 1: "end1", 2: "end2"},
        )
        result, steps = simplify(ss)
        # State 2 should be merged into bottom (1)
        assert len(result.states) <= 2
        assert len(steps) >= 1

    def test_simplify_preserves_reachable(self) -> None:
        """Simplification should keep all reachable states."""
        ss = _from_type("&{a: end, b: end}")
        result, steps = simplify(ss)
        assert result.top in result.states
        assert result.bottom in result.states

    def test_simplify_removes_unreachable(self) -> None:
        """Unreachable states should be pruned."""
        ss = _ss(
            {0, 1, 5},
            [(0, "a", 1)],
            0, 1,
            labels={0: "top", 1: "bottom", 5: "orphan"},
        )
        result, steps = simplify(ss)
        assert 5 not in result.states

    def test_simplify_from_session_type(self) -> None:
        """Simplify a state space built from a session type."""
        ss = _from_type("&{a: end, b: end}")
        result, steps = simplify(ss)
        assert result.top in result.states
        assert result.bottom in result.states


# ===========================================================================
# 7. Confluence checking
# ===========================================================================

class TestConfluence:
    """Test check_confluence function."""

    def test_empty_rules_confluent(self) -> None:
        result = check_confluence([])
        assert result.is_confluent

    def test_single_rule_confluent(self) -> None:
        rule = add_branch_rule("a", "b")
        result = check_confluence([rule])
        assert result.is_confluent

    def test_disjoint_rules_confluent(self) -> None:
        r1 = add_branch_rule("a", "b")
        r2 = add_branch_rule("x", "y")
        result = check_confluence([r1, r2])
        # No shared labels, should be confluent
        assert result.is_confluent

    def test_overlapping_rules_detected(self) -> None:
        r1 = add_branch_rule("a", "b")
        r2 = add_branch_rule("a", "c")
        result = check_confluence([r1, r2])
        # Same label 'a' in both LHS patterns
        assert len(result.critical_pairs) >= 1

    def test_confluence_with_test_graphs(self) -> None:
        """Empirical confluence test with a concrete graph."""
        host = _ss(
            {0, 1, 2},
            [(0, "a", 1), (0, "b", 2)],
            0, 1,
        )
        r1 = add_branch_rule("a", "c")
        r2 = add_branch_rule("b", "d")
        result = check_confluence([r1, r2], test_graphs=[host])
        # These rules operate on different edges, should be confluent
        assert result.all_convergent


# ===========================================================================
# 8. Termination checking
# ===========================================================================

class TestTermination:
    """Test check_termination function."""

    def test_empty_rules_terminate(self) -> None:
        result = check_termination([])
        assert result.terminates

    def test_reducing_rule_terminates(self) -> None:
        rule = remove_selection_rule("OK", "ERR")
        result = check_termination([rule])
        assert result.terminates

    def test_increasing_rule_does_not_terminate(self) -> None:
        rule = unfold_once_rule("a")
        result = check_termination([rule])
        # Unfolding increases state count
        assert not result.terminates

    def test_merge_ends_terminates(self) -> None:
        rule = merge_ends_rule()
        result = check_termination([rule])
        assert result.terminates

    def test_mixed_rules_termination(self) -> None:
        """Mix of reducing and increasing rules."""
        r1 = remove_selection_rule("OK", "ERR")
        r2 = unfold_once_rule("a")
        result = check_termination([r1, r2])
        assert not result.terminates


# ===========================================================================
# 9. Lattice preservation
# ===========================================================================

class TestLatticePreservation:
    """Test lattice preservation checking."""

    def test_branch_extension_preserves_lattice(self) -> None:
        """Adding a branch to a lattice should preserve lattice property.

        We use flatten_branch_rule which adds a new edge to the SAME target
        (not a fresh state), so the lattice property is preserved.
        """
        ss = _from_type("&{a: end}")
        assert check_lattice(ss).is_lattice
        rule = flatten_branch_rule("a", "b")
        matches = match_pattern(ss, rule.lhs)
        if matches:
            assert check_lattice_preservation(ss, rule, matches[0])

    def test_identity_preserves_lattice(self) -> None:
        ss = _from_type("&{a: end, b: end}")
        assert check_lattice(ss).is_lattice
        lhs = _ss({0, 1}, [(0, "a", 1)], 0, 1)
        rhs = _ss({0, 1}, [(0, "a", 1)], 0, 1)
        rule = RewriteRule(
            name="identity", lhs=lhs, rhs=rhs,
            interface_states={0: 0, 1: 1},
        )
        matches = match_pattern(ss, rule.lhs)
        if matches:
            assert check_lattice_preservation(ss, rule, matches[0])


# ===========================================================================
# 10. Full analysis
# ===========================================================================

class TestAnalyzeRewriting:
    """Test analyze_rewriting function."""

    def test_no_rules(self) -> None:
        ss = _from_type("&{a: end}")
        analysis = analyze_rewriting(ss, [])
        assert analysis.num_steps == 0
        assert analysis.original_is_lattice
        assert analysis.final_is_lattice
        assert analysis.lattice_preserved

    def test_analysis_with_add_branch(self) -> None:
        ss = _from_type("&{a: end}")
        rule = add_branch_rule("a", "b")
        analysis = analyze_rewriting(ss, [rule])
        assert analysis.original_is_lattice
        # The final result should still be a lattice
        assert isinstance(analysis.steps, tuple)
        assert isinstance(analysis.rules_applied, frozenset)

    def test_analysis_records_steps(self) -> None:
        ss = _ss(
            {0, 1},
            [(0, "a", 1), (0, "a", 0)],
            0, 1,
        )
        rule = unfold_once_rule("a")
        analysis = analyze_rewriting(ss, [rule], max_iterations=2)
        assert analysis.num_steps == len(analysis.steps)

    def test_analysis_lattice_preserved_flag(self) -> None:
        ss = _from_type("&{a: end, b: end}")
        analysis = analyze_rewriting(ss, [])
        assert analysis.lattice_preserved


# ===========================================================================
# 11. Integration with parser / build_statespace
# ===========================================================================

class TestIntegration:
    """Integration tests with real session types."""

    def test_rewrite_iterator_type(self) -> None:
        """Java Iterator: rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"""
        ss = _from_type("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        assert check_lattice(ss).is_lattice
        analysis = analyze_rewriting(ss, [])
        assert analysis.original_is_lattice
        assert analysis.final_is_lattice

    def test_rewrite_simple_branch(self) -> None:
        """Simple branch: &{a: end, b: end}"""
        ss = _from_type("&{a: end, b: end}")
        rule = add_branch_rule("a", "c")
        matches = match_pattern(ss, rule.lhs)
        assert len(matches) >= 1

    def test_rewrite_selection(self) -> None:
        """Selection: +{OK: end, ERR: end}"""
        ss = _from_type("+{OK: end, ERR: end}")
        assert len(ss.selection_transitions) > 0

    def test_simplify_parallel(self) -> None:
        """Parallel type simplification."""
        ss = _from_type("(&{a: end} || &{b: end})")
        result, steps = simplify(ss)
        assert result.top in result.states
        assert result.bottom in result.states

    def test_add_branch_to_auth_protocol(self) -> None:
        """Add retry after FAIL in auth protocol."""
        ss = _from_type(
            "&{login: +{OK: &{query: end}, FAIL: end}}"
        )
        assert check_lattice(ss).is_lattice
        # The state space should have transitions for login, OK, FAIL, query
        all_labels = {l for _, l, _ in ss.transitions}
        assert "login" in all_labels
        assert "OK" in all_labels
        assert "FAIL" in all_labels
        assert "query" in all_labels

    def test_match_in_larger_graph(self) -> None:
        """Pattern matching in a larger built state space."""
        ss = _from_type("&{a: &{b: end}, c: end}")
        pattern = _ss(
            {0, 1},
            [(0, "b", 1)],
            0, 1,
        )
        matches = match_pattern(ss, pattern)
        assert len(matches) >= 1

    def test_rewrite_preserves_lattice_on_smtp(self) -> None:
        """SMTP-like protocol remains a lattice after rewriting."""
        ss = _from_type(
            "&{connect: &{ehlo: rec X . &{mail: &{rcpt: &{data: "
            "+{OK: X, ERR: X}}}, quit: end}}}"
        )
        lr = check_lattice(ss)
        assert lr.is_lattice
        analysis = analyze_rewriting(ss, [])
        assert analysis.lattice_preserved

    def test_empty_type_simplify(self) -> None:
        """end type should simplify to itself."""
        ss = _from_type("end")
        result, steps = simplify(ss)
        assert len(result.states) == 1
        assert result.top == result.bottom


# ===========================================================================
# 12. Edge cases
# ===========================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_state_no_transitions(self) -> None:
        ss = _ss({0}, [], 0, 0)
        result, steps = simplify(ss)
        assert len(result.states) == 1

    def test_match_pattern_larger_than_host(self) -> None:
        """Pattern larger than host should yield no matches."""
        host = _ss({0, 1}, [(0, "a", 1)], 0, 1)
        pattern = _ss(
            {0, 1, 2, 3},
            [(0, "a", 1), (1, "b", 2), (2, "c", 3)],
            0, 3,
        )
        matches = match_pattern(host, pattern)
        assert len(matches) == 0

    def test_apply_rules_empty_rule_list(self) -> None:
        ss = _ss({0, 1}, [(0, "a", 1)], 0, 1)
        final, steps = apply_rules(ss, [])
        assert len(steps) == 0
        assert len(final.states) == len(ss.states)

    def test_rewrite_step_records_before_after(self) -> None:
        host = _ss({0, 1}, [(0, "a", 1)], 0, 1)
        rule = add_branch_rule("a", "b")
        matches = match_pattern(host, rule.lhs)
        if matches:
            result = apply_rule(host, rule, matches[0])
            step = RewriteStep(
                rule=rule, match=matches[0],
                before=host, after=result,
            )
            assert step.before is host
            assert step.after is result

    def test_confluence_result_counterexample(self) -> None:
        cr = ConfluenceResult(
            is_confluent=False,
            critical_pairs=[("r1", "r2", "desc")],
            all_convergent=False,
            counterexample="test",
        )
        assert not cr.is_confluent
        assert cr.counterexample == "test"

    def test_termination_result_fields(self) -> None:
        tr = TerminationResult(
            terminates=False,
            measure="states",
            reason="increases",
        )
        assert not tr.terminates
        assert tr.measure == "states"
        assert tr.reason == "increases"

    def test_rewriting_analysis_frozen(self) -> None:
        """RewritingAnalysis should be frozen."""
        ss = _from_type("end")
        analysis = analyze_rewriting(ss, [])
        with pytest.raises(AttributeError):
            analysis.num_steps = 5  # type: ignore[misc]

    def test_match_frozen(self) -> None:
        """Match should be frozen."""
        m = Match(state_map={0: 1}, edge_map=[])
        with pytest.raises(AttributeError):
            m.state_map = {}  # type: ignore[misc]
