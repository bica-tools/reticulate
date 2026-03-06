"""Tests for state-space construction (statespace.py) and product (product.py)."""

import pytest

from reticulate.parser import parse
from reticulate.statespace import StateSpace, build_statespace


# ===================================================================
# Helper utilities
# ===================================================================

def _transitions_from(ss: StateSpace, state: int) -> dict[str, int]:
    """Return {label: target} for transitions from *state*."""
    return {l: t for s, l, t in ss.transitions if s == state}


def _all_labels(ss: StateSpace) -> set[str]:
    """Return all transition labels in the state space."""
    return {l for _, l, _ in ss.transitions}


# ===================================================================
# Basic constructs
# ===================================================================

class TestEnd:
    def test_single_state(self) -> None:
        ss = build_statespace(parse("end"))
        assert len(ss.states) == 1
        assert ss.top == ss.bottom
        assert len(ss.transitions) == 0

    def test_label_is_end(self) -> None:
        ss = build_statespace(parse("end"))
        assert ss.labels[ss.top] == "end"


class TestWait:
    def test_wait_maps_to_end_id(self) -> None:
        ss = build_statespace(parse("&{a: wait}"))
        assert len(ss.states) == 2
        assert ss.top != ss.bottom
        tr = _transitions_from(ss, ss.top)
        assert tr["a"] == ss.bottom


class TestSimpleChain:
    def test_a_end(self) -> None:
        """``&{a: end}`` -> 2 states, 1 transition."""
        ss = build_statespace(parse("&{a: end}"))
        assert len(ss.states) == 2
        assert len(ss.transitions) == 1
        tr = _transitions_from(ss, ss.top)
        assert "a" in tr
        assert tr["a"] == ss.bottom

    def test_a_b_end(self) -> None:
        """``&{a: &{b: end}}`` -> 3 states (top ->[a] mid ->[b] end), 2 transitions."""
        ss = build_statespace(parse("&{a: &{b: end}}"))
        assert len(ss.states) == 3
        assert len(ss.transitions) == 2

        # Walk the chain
        tr1 = _transitions_from(ss, ss.top)
        assert set(tr1.keys()) == {"a"}
        mid = tr1["a"]

        tr2 = _transitions_from(ss, mid)
        assert set(tr2.keys()) == {"b"}
        assert tr2["b"] == ss.bottom

    def test_three_step_chain(self) -> None:
        """``&{a: &{b: &{c: end}}}`` -> 4 states, 3 transitions."""
        ss = build_statespace(parse("&{a: &{b: &{c: end}}}"))
        assert len(ss.states) == 4
        assert len(ss.transitions) == 3


class TestBranch:
    def test_two_branches(self) -> None:
        """``&{m: end, n: end}`` -> 2 states (top + end), 2 transitions."""
        ss = build_statespace(parse("&{m: end, n: end}"))
        assert len(ss.states) == 2
        tr = _transitions_from(ss, ss.top)
        assert set(tr.keys()) == {"m", "n"}
        assert tr["m"] == ss.bottom
        assert tr["n"] == ss.bottom

    def test_branches_to_different_states(self) -> None:
        """``&{m: &{a: end}, n: end}`` -> 3 states."""
        ss = build_statespace(parse("&{m: &{a: end}, n: end}"))
        assert len(ss.states) == 3
        tr = _transitions_from(ss, ss.top)
        assert "m" in tr and "n" in tr
        assert tr["m"] != ss.bottom  # m leads to intermediate state
        assert tr["n"] == ss.bottom  # n goes directly to end


class TestSelect:
    def test_two_selections(self) -> None:
        """``+{OK: end, ERR: end}`` -> 2 states, 2 transitions."""
        ss = build_statespace(parse("+{OK: end, ERR: end}"))
        assert len(ss.states) == 2
        tr = _transitions_from(ss, ss.top)
        assert set(tr.keys()) == {"OK", "ERR"}

    def test_select_with_continuations(self) -> None:
        """``+{OK: &{a: end}, ERR: end}`` -> 3 states."""
        ss = build_statespace(parse("+{OK: &{a: end}, ERR: end}"))
        assert len(ss.states) == 3


# ===================================================================
# Recursion
# ===================================================================

class TestRecursion:
    def test_simple_loop(self) -> None:
        """``rec X . &{next: X, done: end}`` -> 2 states, 2 transitions.

        State space: top --[next]--> top (loop), top --[done]--> end.
        """
        ss = build_statespace(parse("rec X . &{next: X, done: end}"))
        assert len(ss.states) == 2
        assert len(ss.transitions) == 2

        tr = _transitions_from(ss, ss.top)
        assert set(tr.keys()) == {"next", "done"}
        assert tr["next"] == ss.top    # loop back
        assert tr["done"] == ss.bottom  # exit

    def test_inner_state_in_loop(self) -> None:
        """``rec X . &{a: &{b: X}, done: end}`` -> 3 states.

        States: top(=rec) --[a]--> after_a --[b]--> top, top --[done]--> end.
        """
        ss = build_statespace(parse("rec X . &{a: &{b: X}, done: end}"))
        assert len(ss.states) == 3
        assert len(ss.transitions) == 3

        tr_top = _transitions_from(ss, ss.top)
        assert "a" in tr_top and "done" in tr_top
        after_a = tr_top["a"]
        assert after_a != ss.top

        tr_after = _transitions_from(ss, after_a)
        assert set(tr_after.keys()) == {"b"}
        assert tr_after["b"] == ss.top  # loops back

    def test_nested_rec(self) -> None:
        """``rec X . &{a: rec Y . &{b: Y, done: X}, done: end}``

        Outer loop (X) with inner loop (Y):
        - top --[a]--> inner --[b]--> inner, inner --[done]--> top
        - top --[done]--> end
        """
        ss = build_statespace(parse(
            "rec X . &{a: rec Y . &{b: Y, done: X}, done: end}"
        ))
        assert len(ss.states) == 3  # top, inner, end
        assert len(ss.transitions) == 4

        tr_top = _transitions_from(ss, ss.top)
        inner = tr_top["a"]
        assert tr_top["done"] == ss.bottom

        tr_inner = _transitions_from(ss, inner)
        assert tr_inner["b"] == inner       # inner loop
        assert tr_inner["done"] == ss.top   # back to outer

    def test_multiple_loop_back(self) -> None:
        """``rec X . &{read: X, peek: X, done: end}`` -> 2 states.

        Both read and peek loop back to top.
        """
        ss = build_statespace(parse("rec X . &{read: X, peek: X, done: end}"))
        assert len(ss.states) == 2
        tr = _transitions_from(ss, ss.top)
        assert tr["read"] == ss.top
        assert tr["peek"] == ss.top
        assert tr["done"] == ss.bottom


# ===================================================================
# Continuation (complex left-hand side)
# ===================================================================

class TestContinuation:
    def test_branch_then_branch(self) -> None:
        """``&{m: end} . &{close: end}`` -> 3 states.

        branch --[m]--> close_state --[close]--> end.
        """
        ss = build_statespace(parse("&{m: end} . &{close: end}"))
        assert len(ss.states) == 3
        tr_top = _transitions_from(ss, ss.top)
        assert "m" in tr_top
        mid = tr_top["m"]
        tr_mid = _transitions_from(ss, mid)
        assert "close" in tr_mid
        assert tr_mid["close"] == ss.bottom


# ===================================================================
# Parallel -- product construction (spec S4.4)
# ===================================================================

class TestParallelFinite:
    """Tests matching spec S4.4.1: ``(&{a: &{b: end}} || &{c: &{d: end}})`` -> 3x3 = 9 states."""

    def setup_method(self) -> None:
        self.ss = build_statespace(parse("(&{a: &{b: end}} || &{c: &{d: end}})"))

    def test_state_count(self) -> None:
        assert len(self.ss.states) == 9  # 3 x 3

    def test_all_transition_labels(self) -> None:
        assert _all_labels(self.ss) == {"a", "b", "c", "d"}

    def test_top_has_two_transitions(self) -> None:
        """From the fork point, can do 'a' (left) or 'c' (right)."""
        tr = _transitions_from(self.ss, self.ss.top)
        assert set(tr.keys()) == {"a", "c"}

    def test_bottom_has_no_transitions(self) -> None:
        """The join point (end, end) has no outgoing transitions."""
        tr = _transitions_from(self.ss, self.ss.bottom)
        assert len(tr) == 0

    def test_bottom_reachable(self) -> None:
        """Bottom (join point) is reachable from top (fork point)."""
        assert self.ss.bottom in self.ss.reachable_from(self.ss.top)

    def test_transition_count(self) -> None:
        """Each of the 9 states has transitions from both components.

        Left chain (3 states, 2 transitions), right chain (3 states, 2 transitions).
        Product transitions: for each (s1, s2), left_adj[s1] + right_adj[s2].
        Total = sum over grid of (left_out + right_out):
          left has: 1,1,0 outgoing; right has: 1,1,0 outgoing.
          Each left_out is applied to all 3 right states and vice versa.
          Left total: (1+1+0)*3 = 6; Right total: 3*(1+1+0) = 6. Total = 12.
        """
        assert len(self.ss.transitions) == 12


class TestParallelSimple:
    """Simple parallel: ``(&{a: end} || &{b: end})`` -> 2x2 = 4 states."""

    def setup_method(self) -> None:
        self.ss = build_statespace(parse("(&{a: end} || &{b: end})"))

    def test_state_count(self) -> None:
        assert len(self.ss.states) == 4  # 2 x 2

    def test_top_transitions(self) -> None:
        tr = _transitions_from(self.ss, self.ss.top)
        assert set(tr.keys()) == {"a", "b"}

    def test_interleaving(self) -> None:
        """Both orderings a->b and b->a reach bottom."""
        tr_top = _transitions_from(self.ss, self.ss.top)
        # Path 1: a then b
        after_a = tr_top["a"]
        tr_after_a = _transitions_from(self.ss, after_a)
        assert "b" in tr_after_a
        assert tr_after_a["b"] == self.ss.bottom

        # Path 2: b then a
        after_b = tr_top["b"]
        tr_after_b = _transitions_from(self.ss, after_b)
        assert "a" in tr_after_b
        assert tr_after_b["a"] == self.ss.bottom


class TestParallelRecursive:
    """Tests matching spec S4.4.2: recursive parallel -> 2x2 = 4 states."""

    def setup_method(self) -> None:
        src = "(rec X . &{a: X, done: end} || rec Y . &{c: Y, stop: end})"
        self.ss = build_statespace(parse(src))

    def test_state_count(self) -> None:
        assert len(self.ss.states) == 4  # 2 x 2

    def test_top_has_loop_and_exit(self) -> None:
        """From top (both looping), can advance either loop or exit either."""
        tr = _transitions_from(self.ss, self.ss.top)
        assert set(tr.keys()) == {"a", "c", "done", "stop"}

    def test_loop_back(self) -> None:
        """Loop transitions return to the same state."""
        tr = _transitions_from(self.ss, self.ss.top)
        assert tr["a"] == self.ss.top  # left loops
        assert tr["c"] == self.ss.top  # right loops

    def test_bottom_reachable(self) -> None:
        assert self.ss.bottom in self.ss.reachable_from(self.ss.top)


class TestParallelWithContinuation:
    """``(&{a: wait} || &{b: wait}) . &{close: end}`` -- parallel then continuation."""

    def setup_method(self) -> None:
        self.ss = build_statespace(parse(
            "(&{a: wait} || &{b: wait}) . &{close: end}"
        ))

    def test_state_count(self) -> None:
        """4 product states + 1 close state + 1 final end = 6 states.

        Actually: product bottom (end,end) chains to close state,
        so it replaces the product bottom. 4 product + close + end = 6.
        But product bottom IS the close state after chaining.
        So: 3 product non-bottom + close_state + end = 5 states.
        Wait, let me think again:
        - Left: &{a: wait} -> 2 states. Right: &{b: wait} -> 2 states. Product: 4 states.
        - Product bottom = (end, end). Via Continuation, this chains to close's entry.
        - So the "end" of the parallel becomes the entry of "&{close: end}".
        - &{close: end} -> 2 states (close_entry + global_end).
        - Product has 4 states, but its bottom is replaced by close_entry.
        - Total: 3 product non-bottom states + close_entry + global_end = 5.
        """
        assert len(self.ss.states) == 5

    def test_reachable(self) -> None:
        assert self.ss.bottom in self.ss.reachable_from(self.ss.top)


# ===================================================================
# Spec examples
# ===================================================================

class TestSpecSharedFile:
    """Full SharedFile from spec S3.2 (simplified without recursion):

    ``&{init: &{open: +{OK: (&{read: wait} || &{write: wait}) . &{close: end}, ERROR: end}}}``
    """

    def setup_method(self) -> None:
        src = "&{init: &{open: +{OK: (&{read: wait} || &{write: wait}) . &{close: end}, ERROR: end}}}"
        self.ss = build_statespace(parse(src))

    def test_reachable(self) -> None:
        """Bottom is reachable from top."""
        assert self.ss.bottom in self.ss.reachable_from(self.ss.top)

    def test_init_transition(self) -> None:
        """Top state has exactly one transition: init."""
        tr = _transitions_from(self.ss, self.ss.top)
        assert set(tr.keys()) == {"init"}

    def test_all_methods_present(self) -> None:
        """All expected method names appear as transition labels."""
        labels = _all_labels(self.ss)
        assert {"init", "open", "OK", "ERROR", "read", "write", "close"} <= labels


class TestSpecConcurrentFileRecursive:
    """Concurrent file with recursive branches from spec S3.2:

    ``&{open: (rec X . &{read: X, doneReading: wait} || rec Y . &{write: Y, doneWriting: wait}) . &{close: end}}``
    """

    def setup_method(self) -> None:
        src = (
            "&{open: (rec X . &{read: X, doneReading: wait} "
            "|| rec Y . &{write: Y, doneWriting: wait}) . &{close: end}}"
        )
        self.ss = build_statespace(parse(src))

    def test_state_count(self) -> None:
        """open -> product(2x2=4 states, bottom chains to close) -> close -> end.

        open_state + 3 product non-bottom + close_state + end = 6.
        """
        assert len(self.ss.states) == 6

    def test_all_methods(self) -> None:
        labels = _all_labels(self.ss)
        assert {"open", "read", "write", "doneReading", "doneWriting", "close"} <= labels

    def test_reachable(self) -> None:
        assert self.ss.bottom in self.ss.reachable_from(self.ss.top)


# ===================================================================
# StateSpace methods
# ===================================================================

class TestStateSpaceMethods:
    def test_enabled(self) -> None:
        ss = build_statespace(parse("&{m: end, n: end}"))
        enabled = ss.enabled(ss.top)
        labels = {l for l, _ in enabled}
        assert labels == {"m", "n"}

    def test_successors(self) -> None:
        ss = build_statespace(parse("&{a: &{b: end}}"))
        succ = ss.successors(ss.top)
        assert len(succ) == 1
        mid = succ.pop()
        assert ss.successors(mid) == {ss.bottom}

    def test_reachable_from_bottom(self) -> None:
        ss = build_statespace(parse("&{a: end}"))
        assert ss.reachable_from(ss.bottom) == {ss.bottom}


# ===================================================================
# TransitionKind -- selection vs method
# ===================================================================


class TestTransitionKind:
    def test_select_produces_selection_transitions(self) -> None:
        ss = build_statespace(parse("+{OK: end, ERR: end}"))
        for s, l, t in ss.transitions:
            assert ss.is_selection(s, l, t), f"({s}, {l}, {t}) should be selection"

    def test_branch_produces_method_transitions(self) -> None:
        ss = build_statespace(parse("&{m: end, n: end}"))
        for s, l, t in ss.transitions:
            assert not ss.is_selection(s, l, t), f"({s}, {l}, {t}) should be method"

    def test_mixed_protocol_method_then_selection(self) -> None:
        ss = build_statespace(parse("&{m: +{OK: end, ERR: end}}"))
        top_tr = [(l, t) for s, l, t in ss.transitions if s == ss.top]
        assert len(top_tr) == 1
        label, select_state = top_tr[0]
        assert label == "m"
        assert not ss.is_selection(ss.top, "m", select_state)

        select_tr = [(l, t) for s, l, t in ss.transitions if s == select_state]
        for l, t in select_tr:
            assert ss.is_selection(select_state, l, t)

    def test_enabled_methods_filters_selections(self) -> None:
        ss = build_statespace(parse("&{m: +{OK: end, ERR: end}}"))
        methods = ss.enabled_methods(ss.top)
        assert len(methods) == 1
        assert methods[0][0] == "m"
        assert ss.enabled_selections(ss.top) == []

    def test_enabled_selections_filters_methods(self) -> None:
        ss = build_statespace(parse("&{m: +{OK: end, ERR: end}}"))
        m_target = ss.enabled(ss.top)[0][1]
        selections = ss.enabled_selections(m_target)
        assert len(selections) == 2
        labels = {l for l, _ in selections}
        assert labels == {"OK", "ERR"}
        assert ss.enabled_methods(m_target) == []

    def test_product_preserves_selection_kind(self) -> None:
        ss = build_statespace(parse("(&{m: end} || +{OK: end, ERR: end})"))
        has_method_m = any(
            l == "m" and not ss.is_selection(s, l, t)
            for s, l, t in ss.transitions
        )
        has_sel_ok = any(
            l == "OK" and ss.is_selection(s, l, t)
            for s, l, t in ss.transitions
        )
        has_sel_err = any(
            l == "ERR" and ss.is_selection(s, l, t)
            for s, l, t in ss.transitions
        )
        assert has_method_m, "m should be METHOD"
        assert has_sel_ok, "OK should be SELECTION"
        assert has_sel_err, "ERR should be SELECTION"

    def test_recursion_preserves_selection_kind(self) -> None:
        ss = build_statespace(parse("rec X . +{OK: &{m: X}, ERR: end}"))
        has_sel_ok = any(
            l == "OK" and ss.is_selection(s, l, t)
            for s, l, t in ss.transitions
        )
        has_method_m = any(
            l == "m" and not ss.is_selection(s, l, t)
            for s, l, t in ss.transitions
        )
        assert has_sel_ok, "OK should be SELECTION after recursion"
        assert has_method_m, "m should be METHOD after recursion"

    def test_enabled_returns_all(self) -> None:
        ss = build_statespace(parse("&{m: +{OK: end, ERR: end}}"))
        m_target = ss.enabled(ss.top)[0][1]
        all_labels = {l for l, _ in ss.enabled(m_target)}
        assert all_labels == {"OK", "ERR"}

    def test_default_empty_selection_transitions(self) -> None:
        ss = build_statespace(parse("&{a: end}"))
        assert ss.selection_transitions == set()

    def test_is_selection_false_for_nonexistent(self) -> None:
        ss = build_statespace(parse("&{a: end}"))
        assert not ss.is_selection(999, "x", 999)


# ===================================================================
# Error cases
# ===================================================================


class TestErrors:
    def test_unbound_variable(self) -> None:
        with pytest.raises(ValueError, match="unbound"):
            build_statespace(parse("X"))
