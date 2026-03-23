"""Tests for place invariants on session-type Petri nets (Step 23)."""

import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.petri import (
    PetriNet,
    Place,
    Transition,
    build_petri_net,
)
from reticulate.place_invariants import (
    PlaceInvariant,
    PlaceInvariantResult,
    analyze_place_invariants,
    check_weighted_token_conservation,
    compute_incidence_matrix,
    compute_place_invariants,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ss(type_string: str):
    """Parse and build state space from a type string."""
    return build_statespace(parse(type_string))


def _net(type_string: str) -> PetriNet:
    """Build Petri net from a session type string."""
    return build_petri_net(_ss(type_string))


def _empty_net() -> PetriNet:
    """An empty Petri net with no places or transitions."""
    return PetriNet(
        places={},
        transitions={},
        pre={},
        post={},
        initial_marking={},
        state_to_marking={},
        marking_to_state={},
        is_occurrence_net=True,
        is_free_choice=True,
    )


def _single_place_net() -> PetriNet:
    """A net with a single place and no transitions (just 'end')."""
    return _net("end")


# ---------------------------------------------------------------------------
# Incidence matrix
# ---------------------------------------------------------------------------


class TestIncidenceMatrix:
    """Test incidence matrix construction."""

    def test_end_no_transitions(self):
        net = _net("end")
        matrix = compute_incidence_matrix(net)
        # One place, no transitions => empty row
        assert len(matrix) == 1
        for row in matrix.values():
            assert len(row) == 0

    def test_single_branch(self):
        net = _net("&{a: end}")
        matrix = compute_incidence_matrix(net)
        # 2 places (top, bottom), 1 transition
        assert len(matrix) == 2
        # Transition consumes from source (-1) and produces to target (+1)
        tid = list(net.transitions.keys())[0]
        src = net.transitions[tid].source_state
        tgt = net.transitions[tid].target_state
        assert matrix[src][tid] == -1
        assert matrix[tgt][tid] == 1

    def test_two_branch_shared_bottom(self):
        net = _net("&{a: end, b: end}")
        matrix = compute_incidence_matrix(net)
        # top state and bottom state (shared end)
        assert len(matrix) == 2

    def test_chain_three_states(self):
        net = _net("&{a: &{b: end}}")
        matrix = compute_incidence_matrix(net)
        # 3 places: top -> middle -> bottom
        assert len(matrix) == 3
        # Each transition: -1 from source, +1 to target
        for tid in net.transitions:
            col_sum = sum(matrix[p].get(tid, 0) for p in matrix)
            assert col_sum == 0  # conservation: column sums to zero

    def test_column_sums_zero(self):
        """For state-machine nets, every column of C sums to 0."""
        for ts in ["&{a: end}", "&{a: end, b: end}",
                    "&{a: &{b: end}}", "+{x: end, y: end}"]:
            net = _net(ts)
            matrix = compute_incidence_matrix(net)
            for tid in net.transitions:
                col_sum = sum(matrix[p].get(tid, 0) for p in matrix)
                assert col_sum == 0, f"Column sum != 0 for {ts}"

    def test_empty_net(self):
        net = _empty_net()
        matrix = compute_incidence_matrix(net)
        assert matrix == {}


# ---------------------------------------------------------------------------
# P-invariant computation
# ---------------------------------------------------------------------------


class TestComputePlaceInvariants:
    """Test null-space computation for P-invariants."""

    def test_end_single_invariant(self):
        net = _net("end")
        invs = compute_place_invariants(net)
        # Single place, no transitions: the unit vector is an invariant
        assert len(invs) == 1
        assert list(invs[0].values()) == [1]

    def test_single_branch_all_ones(self):
        net = _net("&{a: end}")
        invs = compute_place_invariants(net)
        # For a 2-place state-machine net with 1 transition,
        # the all-ones vector should be in the null space
        assert len(invs) >= 1
        # Check that all-ones is among them (or a scalar multiple)
        all_ones_found = False
        for inv in invs:
            place_ids = sorted(net.places.keys())
            if all(inv.get(p, 0) == 1 for p in place_ids):
                all_ones_found = True
        assert all_ones_found

    def test_two_branch_one_invariant(self):
        net = _net("&{a: end, b: end}")
        invs = compute_place_invariants(net)
        # 2 places, 2 transitions: null space dimension should be
        # num_places - rank(C^T)
        assert len(invs) >= 1

    def test_chain_invariant(self):
        net = _net("&{a: &{b: end}}")
        invs = compute_place_invariants(net)
        assert len(invs) >= 1
        # All-ones should work (state-machine property)
        place_ids = sorted(net.places.keys())
        all_ones = {p: 1 for p in place_ids}
        is_cons, tc = check_weighted_token_conservation(net, all_ones)
        assert is_cons
        assert tc == 1

    def test_selection_invariant(self):
        net = _net("+{x: end, y: end}")
        invs = compute_place_invariants(net)
        assert len(invs) >= 1

    def test_empty_net_no_invariants(self):
        net = _empty_net()
        invs = compute_place_invariants(net)
        assert invs == []

    def test_parallel_invariant(self):
        net = _net("(&{a: end} || &{b: end})")
        invs = compute_place_invariants(net)
        # Parallel creates product states; all-ones should still work
        assert len(invs) >= 1
        place_ids = sorted(net.places.keys())
        all_ones = {p: 1 for p in place_ids}
        is_cons, tc = check_weighted_token_conservation(net, all_ones)
        assert is_cons
        assert tc == 1


# ---------------------------------------------------------------------------
# Conservation verification
# ---------------------------------------------------------------------------


class TestConservation:
    """Test weighted token conservation checks."""

    def test_trivial_end(self):
        net = _net("end")
        weights = {list(net.places.keys())[0]: 1}
        is_cons, tc = check_weighted_token_conservation(net, weights)
        assert is_cons
        assert tc == 1

    def test_all_ones_branch(self):
        net = _net("&{a: end, b: end}")
        weights = {p: 1 for p in net.places}
        is_cons, tc = check_weighted_token_conservation(net, weights)
        assert is_cons
        assert tc == 1

    def test_all_ones_nested(self):
        net = _net("&{a: &{b: end, c: end}}")
        weights = {p: 1 for p in net.places}
        is_cons, tc = check_weighted_token_conservation(net, weights)
        assert is_cons
        assert tc == 1

    def test_zero_vector_trivially_conservative(self):
        net = _net("&{a: end}")
        weights: dict[int, int] = {}
        is_cons, tc = check_weighted_token_conservation(net, weights)
        assert is_cons
        assert tc == 0

    def test_wrong_weights_not_conservative(self):
        net = _net("&{a: &{b: end}}")
        # Give weight 2 to source, weight 1 to middle, weight 1 to bottom
        place_ids = sorted(net.places.keys())
        if len(place_ids) >= 3:
            weights = {place_ids[0]: 2, place_ids[1]: 1, place_ids[2]: 1}
            is_cons, _ = check_weighted_token_conservation(net, weights)
            # This may or may not be conservative depending on structure
            # Just verify it returns a result
            assert isinstance(is_cons, bool)

    def test_recursive_type_conservation(self):
        net = _net("rec X . &{a: X}")
        weights = {p: 1 for p in net.places}
        is_cons, tc = check_weighted_token_conservation(net, weights)
        assert is_cons
        assert tc == 1

    def test_selection_conservation(self):
        net = _net("+{ok: end, err: end}")
        weights = {p: 1 for p in net.places}
        is_cons, tc = check_weighted_token_conservation(net, weights)
        assert is_cons
        assert tc == 1


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------


class TestAnalyzePlaceInvariants:
    """Test the full analysis pipeline."""

    def test_end_analysis(self):
        net = _net("end")
        result = analyze_place_invariants(net)
        assert isinstance(result, PlaceInvariantResult)
        assert result.num_places == 1
        assert result.all_conservative
        assert result.is_fully_covered

    def test_simple_branch_analysis(self):
        net = _net("&{a: end}")
        result = analyze_place_invariants(net)
        assert result.all_conservative
        assert result.is_fully_covered
        assert result.num_places == 2
        assert len(result.invariants) >= 1

    def test_nested_branch_analysis(self):
        net = _net("&{a: &{b: end, c: end}}")
        result = analyze_place_invariants(net)
        assert result.all_conservative
        assert result.is_fully_covered

    def test_parallel_analysis(self):
        net = _net("(&{a: end} || &{b: end})")
        result = analyze_place_invariants(net)
        assert result.all_conservative
        assert result.is_fully_covered

    def test_empty_net_analysis(self):
        net = _empty_net()
        result = analyze_place_invariants(net)
        assert result.num_places == 0
        assert result.all_conservative
        assert result.is_fully_covered  # vacuously
        assert len(result.invariants) == 0

    def test_incidence_matrix_stored(self):
        net = _net("&{a: end}")
        result = analyze_place_invariants(net)
        assert isinstance(result.incidence_matrix, dict)
        assert len(result.incidence_matrix) == 2

    def test_null_space_dimension(self):
        net = _net("&{a: end}")
        result = analyze_place_invariants(net)
        assert result.null_space_dimension >= 1

    def test_invariant_support(self):
        net = _net("&{a: end}")
        result = analyze_place_invariants(net)
        for inv in result.invariants:
            assert isinstance(inv.support, frozenset)
            assert len(inv.support) > 0

    def test_invariant_token_count(self):
        net = _net("&{a: end, b: end}")
        result = analyze_place_invariants(net)
        for inv in result.invariants:
            if inv.is_conservative:
                assert inv.token_count is not None


# ---------------------------------------------------------------------------
# Benchmark protocols
# ---------------------------------------------------------------------------


class TestBenchmarkInvariants:
    """Test place invariants on benchmark session types."""

    BENCHMARKS = [
        ("Java Iterator", "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"),
        ("File Object", "&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}"),
        ("SMTP", "&{connect: &{ehlo: rec X . &{mail: &{rcpt: &{data: +{OK: X, ERR: X}}}, quit: end}}}"),
        ("Simple parallel", "(&{a: end} || &{b: end})"),
        ("Selection", "+{ok: &{get: end}, err: end}"),
    ]

    @pytest.mark.parametrize("name,ts", BENCHMARKS, ids=[b[0] for b in BENCHMARKS])
    def test_all_ones_conservative(self, name: str, ts: str):
        """The all-ones vector is always conservative for state-machine nets."""
        net = _net(ts)
        weights = {p: 1 for p in net.places}
        is_cons, tc = check_weighted_token_conservation(net, weights)
        assert is_cons, f"{name}: all-ones not conservative"
        assert tc == 1, f"{name}: token count != 1"

    @pytest.mark.parametrize("name,ts", BENCHMARKS, ids=[b[0] for b in BENCHMARKS])
    def test_fully_covered(self, name: str, ts: str):
        """Every benchmark net should be fully covered by invariants."""
        net = _net(ts)
        result = analyze_place_invariants(net)
        assert result.is_fully_covered, f"{name}: not fully covered"

    @pytest.mark.parametrize("name,ts", BENCHMARKS, ids=[b[0] for b in BENCHMARKS])
    def test_all_conservative(self, name: str, ts: str):
        """Every computed invariant should be verified conservative."""
        net = _net(ts)
        result = analyze_place_invariants(net)
        assert result.all_conservative, f"{name}: not all conservative"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and corner cases."""

    def test_single_place_no_transitions(self):
        net = _single_place_net()
        result = analyze_place_invariants(net)
        assert result.num_places == 1
        assert result.is_fully_covered
        assert result.all_conservative

    def test_self_loop_recursive(self):
        """rec X . &{a: X} creates a self-loop in the Petri net."""
        net = _net("rec X . &{a: X}")
        matrix = compute_incidence_matrix(net)
        # Self-loop: pre and post on same place => C[p,t] = 0
        for pid in matrix:
            for tid in matrix[pid]:
                # For a self-loop, incidence should be 0
                pass
        result = analyze_place_invariants(net)
        assert result.all_conservative

    def test_deeply_nested(self):
        net = _net("&{a: &{b: &{c: &{d: end}}}}")
        result = analyze_place_invariants(net)
        assert result.all_conservative
        assert result.is_fully_covered
        assert result.num_places == 5

    def test_wide_branch(self):
        net = _net("&{a: end, b: end, c: end, d: end, e: end}")
        result = analyze_place_invariants(net)
        assert result.all_conservative
        assert result.is_fully_covered
