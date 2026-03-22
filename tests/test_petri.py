"""Tests for Petri net construction from session types (Step 21)."""

import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.petri import (
    FiringResult,
    Marking,
    PetriNet,
    PetriNetResult,
    Place,
    ReachabilityGraph,
    Transition,
    build_petri_net,
    build_reachability_graph,
    concurrent_transitions,
    conflict_places,
    enabled_transitions,
    fire,
    is_enabled,
    petri_dot,
    place_invariants,
    session_type_to_petri_net,
    verify_isomorphism,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _ss(type_string: str):
    """Parse and build state space from a type string."""
    return build_statespace(parse(type_string))


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------


class TestBuildPetriNet:
    """Test Petri net construction from state spaces."""

    def test_end_single_place(self):
        ss = _ss("end")
        net = build_petri_net(ss)
        assert len(net.places) == 1
        assert len(net.transitions) == 0

    def test_single_branch(self):
        ss = _ss("&{a: end}")
        net = build_petri_net(ss)
        assert len(net.places) == 2
        assert len(net.transitions) == 1
        # Transition labeled "a"
        t = list(net.transitions.values())[0]
        assert t.label == "a"
        assert not t.is_selection

    def test_single_selection(self):
        ss = _ss("+{a: end}")
        net = build_petri_net(ss)
        assert len(net.transitions) == 1
        t = list(net.transitions.values())[0]
        assert t.label == "a"
        assert t.is_selection

    def test_two_branch(self):
        ss = _ss("&{a: end, b: end}")
        net = build_petri_net(ss)
        assert len(net.places) == 2  # top and bottom share end
        assert len(net.transitions) == 2
        labels = {t.label for t in net.transitions.values()}
        assert labels == {"a", "b"}

    def test_nested_branch(self):
        ss = _ss("&{a: &{b: end}}")
        net = build_petri_net(ss)
        assert len(net.places) == 3
        assert len(net.transitions) == 2

    def test_initial_marking(self):
        ss = _ss("&{a: end}")
        net = build_petri_net(ss)
        # One token on the top state
        assert sum(net.initial_marking.values()) == 1
        assert net.initial_marking.get(ss.top, 0) == 1

    def test_state_marking_correspondence(self):
        ss = _ss("&{a: &{b: end}}")
        net = build_petri_net(ss)
        # Every state maps to a marking with exactly one token
        for state_id, marking in net.state_to_marking.items():
            assert sum(marking.values()) == 1
            assert marking == {state_id: 1}

    def test_places_match_states(self):
        ss = _ss("&{a: end, b: &{c: end}}")
        net = build_petri_net(ss)
        assert len(net.places) == len(ss.states)
        for state_id in ss.states:
            assert state_id in net.places
            assert net.places[state_id].state == state_id

    def test_transitions_match_edges(self):
        ss = _ss("&{a: end, b: &{c: end}}")
        net = build_petri_net(ss)
        assert len(net.transitions) == len(ss.transitions)

    def test_pre_post_arcs(self):
        ss = _ss("&{a: end}")
        net = build_petri_net(ss)
        tid = 0
        # Pre: source state
        assert len(net.pre[tid]) == 1
        pre_place = list(net.pre[tid])[0][0]
        assert pre_place == ss.top
        # Post: target state
        assert len(net.post[tid]) == 1
        post_place = list(net.post[tid])[0][0]
        assert post_place == ss.bottom


# ---------------------------------------------------------------------------
# Firing semantics
# ---------------------------------------------------------------------------


class TestFiring:
    """Test Petri net firing rule."""

    def test_fire_enabled(self):
        ss = _ss("&{a: end}")
        net = build_petri_net(ss)
        result = fire(net, net.initial_marking, 0)
        assert result.enabled
        assert result.new_marking is not None
        # Token moved from top to bottom
        assert result.new_marking.get(ss.top, 0) == 0
        assert result.new_marking.get(ss.bottom, 0) == 1

    def test_fire_not_enabled(self):
        ss = _ss("&{a: &{b: end}}")
        net = build_petri_net(ss)
        # Transition for "b" should not be enabled at initial marking
        b_tid = None
        for tid, t in net.transitions.items():
            if t.label == "b":
                b_tid = tid
        assert b_tid is not None
        result = fire(net, net.initial_marking, b_tid)
        assert not result.enabled
        assert result.new_marking is None

    def test_fire_sequence(self):
        ss = _ss("&{a: &{b: end}}")
        net = build_petri_net(ss)
        # Fire "a" then "b"
        a_tid = None
        b_tid = None
        for tid, t in net.transitions.items():
            if t.label == "a":
                a_tid = tid
            elif t.label == "b":
                b_tid = tid
        assert a_tid is not None and b_tid is not None
        r1 = fire(net, net.initial_marking, a_tid)
        assert r1.enabled
        r2 = fire(net, r1.new_marking, b_tid)
        assert r2.enabled
        # End state reached
        assert r2.new_marking.get(ss.bottom, 0) == 1

    def test_is_enabled(self):
        ss = _ss("&{a: end, b: end}")
        net = build_petri_net(ss)
        # Both transitions enabled at initial marking (conflict)
        enabled = enabled_transitions(net, net.initial_marking)
        assert len(enabled) == 2

    def test_conflict_firing(self):
        """Firing one branch disables the other."""
        ss = _ss("&{a: end, b: end}")
        net = build_petri_net(ss)
        # Fire first transition
        r = fire(net, net.initial_marking, 0)
        assert r.enabled
        # Check that no transitions are enabled at bottom
        enabled = enabled_transitions(net, r.new_marking)
        # At bottom (end), nothing should fire
        assert len(enabled) == 0

    def test_one_safe(self):
        """All reachable markings are 1-safe (at most 1 token per place)."""
        ss = _ss("&{a: &{b: end}, c: end}")
        net = build_petri_net(ss)
        rg = build_reachability_graph(net)
        for fm in rg.markings:
            m = dict(fm)
            for count in m.values():
                assert count <= 1

    def test_token_conservation(self):
        """Total tokens are conserved (always exactly 1)."""
        ss = _ss("&{a: &{b: end}, c: end}")
        net = build_petri_net(ss)
        rg = build_reachability_graph(net)
        for fm in rg.markings:
            m = dict(fm)
            assert sum(m.values()) == 1


# ---------------------------------------------------------------------------
# Reachability graph
# ---------------------------------------------------------------------------


class TestReachabilityGraph:
    """Test reachability graph construction."""

    def test_end_single_marking(self):
        ss = _ss("end")
        net = build_petri_net(ss)
        rg = build_reachability_graph(net)
        assert rg.num_markings == 1
        assert len(rg.edges) == 0

    def test_single_branch_two_markings(self):
        ss = _ss("&{a: end}")
        net = build_petri_net(ss)
        rg = build_reachability_graph(net)
        assert rg.num_markings == 2
        assert len(rg.edges) == 1

    def test_reachability_matches_states(self):
        ss = _ss("&{a: &{b: end}, c: end}")
        net = build_petri_net(ss)
        rg = build_reachability_graph(net)
        assert rg.num_markings == len(ss.states)

    def test_edges_match_transitions(self):
        ss = _ss("&{a: &{b: end}, c: end}")
        net = build_petri_net(ss)
        rg = build_reachability_graph(net)
        assert len(rg.edges) == len(ss.transitions)


# ---------------------------------------------------------------------------
# Isomorphism verification
# ---------------------------------------------------------------------------


class TestIsomorphism:
    """Test reachability graph ≅ state space."""

    def test_end_isomorphic(self):
        ss = _ss("end")
        net = build_petri_net(ss)
        assert verify_isomorphism(ss, net)

    def test_single_branch_isomorphic(self):
        ss = _ss("&{a: end}")
        net = build_petri_net(ss)
        assert verify_isomorphism(ss, net)

    def test_two_branch_isomorphic(self):
        ss = _ss("&{a: end, b: end}")
        net = build_petri_net(ss)
        assert verify_isomorphism(ss, net)

    def test_nested_branch_isomorphic(self):
        ss = _ss("&{a: &{b: end}}")
        net = build_petri_net(ss)
        assert verify_isomorphism(ss, net)

    def test_selection_isomorphic(self):
        ss = _ss("+{a: end, b: end}")
        net = build_petri_net(ss)
        assert verify_isomorphism(ss, net)

    def test_mixed_isomorphic(self):
        ss = _ss("&{a: +{ok: end, err: end}, b: end}")
        net = build_petri_net(ss)
        assert verify_isomorphism(ss, net)

    def test_recursive_isomorphic(self):
        ss = _ss("rec X . &{a: X, b: end}")
        net = build_petri_net(ss)
        assert verify_isomorphism(ss, net)

    def test_deep_nesting_isomorphic(self):
        ss = _ss("&{a: &{b: &{c: end}}}")
        net = build_petri_net(ss)
        assert verify_isomorphism(ss, net)


# ---------------------------------------------------------------------------
# Structural properties
# ---------------------------------------------------------------------------


class TestOccurrenceNet:
    """Test occurrence net (acyclicity) detection."""

    def test_linear_is_occurrence(self):
        ss = _ss("&{a: &{b: end}}")
        net = build_petri_net(ss)
        assert net.is_occurrence_net

    def test_branch_is_occurrence(self):
        ss = _ss("&{a: end, b: end}")
        net = build_petri_net(ss)
        assert net.is_occurrence_net

    def test_recursive_not_occurrence(self):
        ss = _ss("rec X . &{a: X, b: end}")
        net = build_petri_net(ss)
        assert not net.is_occurrence_net

    def test_end_is_occurrence(self):
        ss = _ss("end")
        net = build_petri_net(ss)
        assert net.is_occurrence_net


class TestFreeChoice:
    """Test free-choice property."""

    def test_linear_free_choice(self):
        ss = _ss("&{a: &{b: end}}")
        net = build_petri_net(ss)
        assert net.is_free_choice

    def test_branch_free_choice(self):
        """Binary branch: both transitions share same pre-set → free-choice."""
        ss = _ss("&{a: end, b: end}")
        net = build_petri_net(ss)
        assert net.is_free_choice

    def test_state_machine_always_free_choice(self):
        """State-machine encoding is always free-choice."""
        for ts in ["end", "&{a: end}", "&{a: end, b: end}",
                    "&{a: &{b: end}}", "+{a: end, b: end}",
                    "rec X . &{a: X, b: end}"]:
            ss = _ss(ts)
            net = build_petri_net(ss)
            assert net.is_free_choice, f"Not free-choice for {ts}"


# ---------------------------------------------------------------------------
# Conflict and concurrency
# ---------------------------------------------------------------------------


class TestConflict:
    """Test conflict place detection."""

    def test_no_conflict_linear(self):
        ss = _ss("&{a: end}")
        net = build_petri_net(ss)
        conflicts = conflict_places(net)
        assert len(conflicts) == 0

    def test_branch_creates_conflict(self):
        ss = _ss("&{a: end, b: end}")
        net = build_petri_net(ss)
        conflicts = conflict_places(net)
        assert len(conflicts) == 1
        place_id, tids = conflicts[0]
        assert len(tids) == 2

    def test_triple_branch_conflict(self):
        ss = _ss("&{a: end, b: end, c: end}")
        net = build_petri_net(ss)
        conflicts = conflict_places(net)
        assert len(conflicts) == 1
        _, tids = conflicts[0]
        assert len(tids) == 3

    def test_selection_creates_conflict(self):
        ss = _ss("+{a: end, b: end}")
        net = build_petri_net(ss)
        conflicts = conflict_places(net)
        assert len(conflicts) == 1

    def test_nested_conflicts(self):
        ss = _ss("&{a: &{c: end, d: end}, b: end}")
        net = build_petri_net(ss)
        conflicts = conflict_places(net)
        assert len(conflicts) == 2  # top-level and inner branch


class TestConcurrency:
    """Test concurrent transition detection."""

    def test_no_concurrency_linear(self):
        ss = _ss("&{a: &{b: end}}")
        net = build_petri_net(ss)
        conc = concurrent_transitions(net)
        assert len(conc) == 0

    def test_no_concurrency_branch(self):
        """Branch transitions share an input place → not concurrent."""
        ss = _ss("&{a: end, b: end}")
        net = build_petri_net(ss)
        conc = concurrent_transitions(net)
        assert len(conc) == 0

    def test_sequential_no_concurrency(self):
        ss = _ss("&{a: &{b: end}}")
        net = build_petri_net(ss)
        conc = concurrent_transitions(net)
        assert len(conc) == 0


# ---------------------------------------------------------------------------
# Place invariants
# ---------------------------------------------------------------------------


class TestPlaceInvariants:
    """Test S-invariant computation."""

    def test_trivial_invariant_exists(self):
        ss = _ss("&{a: end, b: end}")
        net = build_petri_net(ss)
        invs = place_invariants(net)
        assert len(invs) >= 1
        # Trivial invariant: all-ones
        assert all(v == 1 for v in invs[0].values())

    def test_invariant_end(self):
        ss = _ss("end")
        net = build_petri_net(ss)
        invs = place_invariants(net)
        assert len(invs) == 1

    def test_invariant_recursive(self):
        ss = _ss("rec X . &{a: X, b: end}")
        net = build_petri_net(ss)
        invs = place_invariants(net)
        assert len(invs) >= 1


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


class TestSessionTypeToPetriNet:
    """Test the main entry point."""

    def test_result_fields(self):
        ss = _ss("&{a: end}")
        result = session_type_to_petri_net(ss)
        assert isinstance(result, PetriNetResult)
        assert result.num_places == 2
        assert result.num_transitions == 1
        assert result.reachability_isomorphic
        assert result.num_reachable_markings == 2

    def test_recursive_result(self):
        ss = _ss("rec X . &{a: X, b: end}")
        result = session_type_to_petri_net(ss)
        assert not result.is_occurrence_net
        assert result.reachability_isomorphic

    def test_selection_result(self):
        ss = _ss("+{ok: end, err: end}")
        result = session_type_to_petri_net(ss)
        assert result.is_occurrence_net
        assert result.is_free_choice
        assert result.reachability_isomorphic


# ---------------------------------------------------------------------------
# DOT visualization
# ---------------------------------------------------------------------------


class TestPetriDot:
    """Test DOT generation for Petri nets."""

    def test_dot_output(self):
        ss = _ss("&{a: end}")
        net = build_petri_net(ss)
        dot = petri_dot(net)
        assert "digraph PetriNet" in dot
        assert "shape=circle" in dot
        assert "shape=box" in dot

    def test_dot_has_places_and_transitions(self):
        ss = _ss("&{a: end, b: end}")
        net = build_petri_net(ss)
        dot = petri_dot(net)
        assert "p" in dot  # places
        assert "t" in dot  # transitions

    def test_dot_initial_token(self):
        ss = _ss("&{a: end}")
        net = build_petri_net(ss)
        dot = petri_dot(net)
        # Initial place should show a token marker
        assert "•" in dot  # bullet character for token

    def test_dot_selection_styled(self):
        ss = _ss("+{a: end}")
        net = build_petri_net(ss)
        dot = petri_dot(net)
        assert "filled" in dot  # selection transitions are filled


# ---------------------------------------------------------------------------
# Parallel types
# ---------------------------------------------------------------------------


class TestParallelTypes:
    """Test Petri net construction for parallel session types."""

    def test_parallel_basic(self):
        ss = _ss("(&{a: end} || &{b: end})")
        net = build_petri_net(ss)
        result = session_type_to_petri_net(ss)
        assert result.reachability_isomorphic

    def test_parallel_places_match_states(self):
        ss = _ss("(&{a: end} || &{b: end})")
        net = build_petri_net(ss)
        assert len(net.places) == len(ss.states)

    def test_parallel_transitions_match(self):
        ss = _ss("(&{a: end} || &{b: end})")
        net = build_petri_net(ss)
        assert len(net.transitions) == len(ss.transitions)


# ---------------------------------------------------------------------------
# Benchmark protocols
# ---------------------------------------------------------------------------


class TestBenchmarks:
    """Test Petri net construction on benchmark protocols."""

    @pytest.fixture
    def benchmarks(self):
        from tests.benchmarks.protocols import BENCHMARKS
        return BENCHMARKS

    def test_all_benchmarks_isomorphic(self, benchmarks):
        """All 79 benchmark protocols produce isomorphic Petri nets."""
        failures = []
        for bp in benchmarks:
            try:
                ss = build_statespace(parse(bp.type_string))
                result = session_type_to_petri_net(ss)
                if not result.reachability_isomorphic:
                    failures.append(bp.name)
            except Exception as e:
                failures.append(f"{bp.name}: {e}")
        assert failures == [], f"Failures: {failures}"

    def test_all_benchmarks_one_safe(self, benchmarks):
        """All benchmark Petri nets are 1-safe."""
        for bp in benchmarks:
            ss = build_statespace(parse(bp.type_string))
            net = build_petri_net(ss)
            rg = build_reachability_graph(net)
            for fm in rg.markings:
                m = dict(fm)
                for count in m.values():
                    assert count <= 1, (
                        f"{bp.name} not 1-safe: marking {m}"
                    )

    def test_all_benchmarks_token_conservation(self, benchmarks):
        """All benchmark Petri nets conserve tokens (exactly 1)."""
        for bp in benchmarks:
            ss = build_statespace(parse(bp.type_string))
            net = build_petri_net(ss)
            rg = build_reachability_graph(net)
            for fm in rg.markings:
                m = dict(fm)
                total = sum(m.values())
                assert total == 1, (
                    f"{bp.name} token count = {total}: {m}"
                )

    def test_all_benchmarks_free_choice(self, benchmarks):
        """State-machine encoded nets are always free-choice."""
        for bp in benchmarks:
            ss = build_statespace(parse(bp.type_string))
            net = build_petri_net(ss)
            assert net.is_free_choice, f"{bp.name} not free-choice"

    def test_all_benchmarks_have_invariant(self, benchmarks):
        """All benchmark Petri nets have the trivial S-invariant."""
        for bp in benchmarks:
            ss = build_statespace(parse(bp.type_string))
            net = build_petri_net(ss)
            invs = place_invariants(net)
            assert len(invs) >= 1, f"{bp.name} has no invariant"

    def test_recursive_benchmarks_with_cycles_not_occurrence(self, benchmarks):
        """Recursive benchmarks that produce cycles are non-occurrence nets."""
        for bp in benchmarks:
            if "rec" not in bp.type_string.lower():
                continue
            ss = build_statespace(parse(bp.type_string))
            # Only check benchmarks that actually have cycles (SCC > 1 state)
            if bp.expected_sccs < len(ss.states):
                net = build_petri_net(ss)
                assert not net.is_occurrence_net, (
                    f"{bp.name} has cycles but marked as occurrence net"
                )

    def test_non_recursive_benchmarks_are_occurrence(self, benchmarks):
        """Non-recursive benchmarks produce occurrence nets."""
        for bp in benchmarks:
            if "rec" not in bp.type_string.lower():
                ss = build_statespace(parse(bp.type_string))
                net = build_petri_net(ss)
                assert net.is_occurrence_net, (
                    f"{bp.name} has no rec but not occurrence net"
                )

    def test_benchmark_conflict_at_branches(self, benchmarks):
        """Benchmarks with branching produce conflict places."""
        for bp in benchmarks:
            # Count branch points (states with >1 outgoing transition)
            ss = build_statespace(parse(bp.type_string))
            branch_states = set()
            for s in ss.states:
                out = [(l, t) for src, l, t in ss.transitions if src == s]
                if len(out) > 1:
                    branch_states.add(s)
            net = build_petri_net(ss)
            conflicts = conflict_places(net)
            assert len(conflicts) == len(branch_states), (
                f"{bp.name}: {len(conflicts)} conflicts vs "
                f"{len(branch_states)} branch states"
            )

    def test_benchmark_statistics(self, benchmarks):
        """Collect statistics across all benchmarks."""
        total_places = 0
        total_transitions = 0
        occurrence_count = 0
        for bp in benchmarks:
            ss = build_statespace(parse(bp.type_string))
            result = session_type_to_petri_net(ss)
            total_places += result.num_places
            total_transitions += result.num_transitions
            if result.is_occurrence_net:
                occurrence_count += 1
        assert total_places > 0
        assert total_transitions > 0
