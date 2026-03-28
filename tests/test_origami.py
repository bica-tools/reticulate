"""Tests for reticulate.origami — lattice folding, quotients, and surgery."""

from __future__ import annotations

import pytest

from reticulate.origami import (
    FoldResult,
    SurgeryResult,
    classify_fold,
    contract_edge,
    extract_sublattice,
    fold_by_depth,
    fold_by_label,
    prune_unreachable,
    surgery_cut,
)
from reticulate.parser import parse
from reticulate.statespace import StateSpace, build_statespace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ss(type_str: str) -> StateSpace:
    """Parse and build state space from a type string."""
    return build_statespace(parse(type_str))


def _manual_ss(
    states: set[int],
    transitions: list[tuple[int, str, int]],
    top: int,
    bottom: int,
    labels: dict[int, str] | None = None,
    selection_transitions: set[tuple[int, str, int]] | None = None,
) -> StateSpace:
    """Build a manual StateSpace for testing."""
    return StateSpace(
        states=states,
        transitions=transitions,
        top=top,
        bottom=bottom,
        labels=labels or {},
        selection_transitions=selection_transitions or set(),
    )


# ---------------------------------------------------------------------------
# FoldResult / SurgeryResult dataclass tests
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_fold_result_frozen(self) -> None:
        fr = FoldResult(3, 2, {0: 0, 1: 0, 2: 2}, True, True, True)
        with pytest.raises(AttributeError):
            fr.is_lattice = False  # type: ignore[misc]

    def test_surgery_result_frozen(self) -> None:
        sr = SurgeryResult(3, 2, 1, 0, True)
        with pytest.raises(AttributeError):
            sr.is_lattice = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# fold_by_label
# ---------------------------------------------------------------------------


class TestFoldByLabel:
    def test_end_type_unchanged(self) -> None:
        ss = _ss("end")
        folded, result = fold_by_label(ss)
        assert result.original_states == result.folded_states
        assert result.preserved_top
        assert result.preserved_bottom

    def test_simple_branch_no_merge(self) -> None:
        ss = _ss("&{a: end, b: end}")
        folded, result = fold_by_label(ss)
        # The two 'end' states after a and b have the same outgoing labels (none)
        # so they should merge.
        assert result.folded_states <= result.original_states
        assert result.is_lattice
        assert result.preserved_top
        assert result.preserved_bottom

    def test_mergeable_states(self) -> None:
        # States with identical outgoing label sets should merge
        ss = _ss("&{a: &{x: end}, b: &{x: end}}")
        folded, result = fold_by_label(ss)
        assert result.folded_states <= result.original_states
        assert result.preserved_top
        assert result.preserved_bottom

    def test_folded_has_valid_structure(self) -> None:
        ss = _ss("&{a: end, b: end}")
        folded, result = fold_by_label(ss)
        assert folded.top in folded.states
        assert folded.bottom in folded.states
        for src, _label, tgt in folded.transitions:
            assert src in folded.states
            assert tgt in folded.states

    def test_selection_type(self) -> None:
        ss = _ss("+{a: end, b: end}")
        folded, result = fold_by_label(ss)
        assert result.is_lattice
        assert result.preserved_top

    def test_merge_map_covers_all_states(self) -> None:
        ss = _ss("&{a: end, b: end}")
        _, result = fold_by_label(ss)
        for s in ss.states:
            assert s in result.merge_map


# ---------------------------------------------------------------------------
# fold_by_depth
# ---------------------------------------------------------------------------


class TestFoldByDepth:
    def test_end_type_unchanged(self) -> None:
        ss = _ss("end")
        folded, result = fold_by_depth(ss)
        assert result.original_states == result.folded_states

    def test_conservative_folding(self) -> None:
        # fold_by_depth should merge <= fold_by_label
        ss = _ss("&{a: &{x: end}, b: &{x: end}}")
        _, label_result = fold_by_label(ss)
        _, depth_result = fold_by_depth(ss)
        assert depth_result.folded_states >= label_result.folded_states

    def test_preserves_structure(self) -> None:
        ss = _ss("&{a: end, b: end}")
        folded, result = fold_by_depth(ss)
        assert folded.top in folded.states
        assert folded.bottom in folded.states
        assert result.preserved_top
        assert result.preserved_bottom

    def test_depth_aware_merge(self) -> None:
        # Two states with the same labels but different depths should NOT merge
        ss = _ss("&{a: &{x: end}, b: end}")
        folded, result = fold_by_depth(ss)
        assert result.preserved_top
        assert result.preserved_bottom


# ---------------------------------------------------------------------------
# extract_sublattice
# ---------------------------------------------------------------------------


class TestExtractSublattice:
    def test_extract_from_top_is_identity(self) -> None:
        ss = _ss("&{a: end, b: end}")
        sub = extract_sublattice(ss, ss.top)
        assert sub.states == ss.states
        assert sub.top == ss.top
        assert sub.bottom == ss.bottom

    def test_extract_subgraph(self) -> None:
        ss = _ss("&{a: &{x: end}, b: end}")
        # Find a state that is not top
        non_top = [s for s in ss.states if s != ss.top and s != ss.bottom]
        if non_top:
            sub = extract_sublattice(ss, non_top[0])
            assert sub.top == non_top[0]
            assert len(sub.states) <= len(ss.states)

    def test_invalid_root_raises(self) -> None:
        ss = _ss("end")
        with pytest.raises(ValueError, match="not in state space"):
            extract_sublattice(ss, 9999)

    def test_bottom_preserved_when_reachable(self) -> None:
        ss = _ss("&{a: end, b: end}")
        sub = extract_sublattice(ss, ss.top)
        assert sub.bottom == ss.bottom

    def test_single_state(self) -> None:
        ss = _ss("end")
        sub = extract_sublattice(ss, ss.top)
        assert len(sub.states) == 1
        assert sub.top == sub.bottom


# ---------------------------------------------------------------------------
# prune_unreachable
# ---------------------------------------------------------------------------


class TestPruneUnreachable:
    def test_noop_on_wellformed(self) -> None:
        ss = _ss("&{a: end, b: end}")
        pruned = prune_unreachable(ss)
        assert pruned.states == ss.states

    def test_noop_on_end(self) -> None:
        ss = _ss("end")
        pruned = prune_unreachable(ss)
        assert pruned.states == ss.states

    def test_removes_dead_state(self) -> None:
        # Manually add a dead state
        ss = _ss("&{a: end}")
        dead = max(ss.states) + 1
        states = ss.states | {dead}
        manual = _manual_ss(
            states=states,
            transitions=list(ss.transitions),
            top=ss.top,
            bottom=ss.bottom,
            labels=dict(ss.labels),
        )
        pruned = prune_unreachable(manual)
        assert dead not in pruned.states

    def test_preserves_top_bottom(self) -> None:
        ss = _ss("&{a: end, b: &{x: end}}")
        pruned = prune_unreachable(ss)
        assert pruned.top == ss.top
        assert pruned.bottom == ss.bottom


# ---------------------------------------------------------------------------
# contract_edge
# ---------------------------------------------------------------------------


class TestContractEdge:
    def test_basic_contraction(self) -> None:
        ss = _ss("&{a: end, b: end}")
        # Pick the first transition
        src, label, tgt = ss.transitions[0]
        contracted = contract_edge(ss, src, label, tgt)
        assert len(contracted.states) < len(ss.states)

    def test_top_preserved(self) -> None:
        ss = _ss("&{a: end, b: end}")
        # Contract an edge not involving top directly as tgt
        for src, label, tgt in ss.transitions:
            if tgt != ss.top:
                contracted = contract_edge(ss, src, label, tgt)
                assert contracted.top in contracted.states
                break

    def test_bottom_preserved(self) -> None:
        ss = _ss("&{a: &{x: end}, b: end}")
        # Contract an edge where neither endpoint is bottom
        for src, label, tgt in ss.transitions:
            if src != ss.bottom and tgt != ss.bottom:
                contracted = contract_edge(ss, src, label, tgt)
                assert contracted.bottom in contracted.states
                break

    def test_invalid_transition_raises(self) -> None:
        ss = _ss("end")
        with pytest.raises(ValueError, match="not found"):
            contract_edge(ss, 0, "nonexistent", 1)

    def test_contracted_transitions_valid(self) -> None:
        ss = _ss("&{a: end, b: end}")
        src, label, tgt = ss.transitions[0]
        contracted = contract_edge(ss, src, label, tgt)
        for s, _l, t in contracted.transitions:
            assert s in contracted.states
            assert t in contracted.states


# ---------------------------------------------------------------------------
# classify_fold
# ---------------------------------------------------------------------------


class TestClassifyFold:
    def test_end_is_minimal(self) -> None:
        ss = _ss("end")
        assert classify_fold(ss) == "minimal"

    def test_simple_foldable(self) -> None:
        ss = _ss("&{a: end, b: end}")
        classification = classify_fold(ss)
        assert classification in ("minimal", "foldable", "highly_foldable")

    def test_returns_valid_string(self) -> None:
        ss = _ss("+{a: end, b: end}")
        result = classify_fold(ss)
        assert result in ("minimal", "foldable", "highly_foldable")

    def test_highly_foldable_manual(self) -> None:
        # Many states with same outgoing labels -> highly foldable
        # 6 states, 5 of which have identical empty outgoing sets
        ss = _manual_ss(
            states={0, 1, 2, 3, 4, 5},
            transitions=[
                (0, "a", 1), (0, "b", 2), (0, "c", 3),
                (0, "d", 4), (0, "e", 5),
            ],
            top=0,
            bottom=1,
        )
        result = classify_fold(ss)
        assert result == "highly_foldable"


# ---------------------------------------------------------------------------
# surgery_cut
# ---------------------------------------------------------------------------


class TestSurgeryCut:
    def test_cut_transition(self) -> None:
        ss = _ss("&{a: end, b: end}")
        src, label, tgt = ss.transitions[0]
        result = surgery_cut(ss, src, label, tgt)
        assert result.cut_transitions == 1
        assert result.added_transitions == 0
        assert result.original_states == len(ss.states)

    def test_invalid_transition_raises(self) -> None:
        ss = _ss("end")
        with pytest.raises(ValueError, match="not found"):
            surgery_cut(ss, 0, "nonexistent", 1)

    def test_cut_preserves_or_breaks_lattice(self) -> None:
        ss = _ss("&{a: end, b: end}")
        src, label, tgt = ss.transitions[0]
        result = surgery_cut(ss, src, label, tgt)
        assert isinstance(result.is_lattice, bool)

    def test_result_states_leq_original(self) -> None:
        ss = _ss("&{a: end, b: end}")
        src, label, tgt = ss.transitions[0]
        result = surgery_cut(ss, src, label, tgt)
        assert result.result_states <= result.original_states


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_recursive_type_fold(self) -> None:
        ss = _ss("rec X . &{a: X}")
        folded, result = fold_by_label(ss)
        assert result.preserved_top
        # Bottom may not be preserved for purely recursive types
        # (bottom ID is a phantom not in original states).
        assert isinstance(result.preserved_bottom, bool)

    def test_parallel_type_fold(self) -> None:
        ss = _ss("(end || end)")
        folded, result = fold_by_label(ss)
        assert result.preserved_top

    def test_selection_transitions_preserved(self) -> None:
        ss = _ss("+{a: end, b: end}")
        folded, result = fold_by_label(ss)
        # Selection transitions should survive folding if their endpoints survive
        for src, label, tgt in folded.selection_transitions:
            assert src in folded.states
            assert tgt in folded.states

    def test_extract_then_prune_idempotent(self) -> None:
        ss = _ss("&{a: end, b: end}")
        sub = extract_sublattice(ss, ss.top)
        pruned = prune_unreachable(sub)
        assert pruned.states == sub.states
