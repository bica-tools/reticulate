"""Tests for S-invariants and T-invariants on session-type Petri nets (Step 23b)."""

import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.petri import (
    PetriNet,
    Place,
    Transition,
    build_petri_net,
)
from reticulate.st_invariants import (
    SInvariant,
    TInvariant,
    StructuralInvariantResult,
    analyze_structural_invariants,
    compute_s_invariants,
    compute_t_invariants,
    verify_s_invariant,
    verify_t_invariant,
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


# ---------------------------------------------------------------------------
# S-invariant computation
# ---------------------------------------------------------------------------


class TestComputeSInvariants:
    """Test S-invariant (place invariant) computation."""

    def test_end_single_invariant(self):
        net = _net("end")
        invs = compute_s_invariants(net)
        # Single place, no transitions: unit vector is an invariant
        assert len(invs) == 1
        assert invs[0].is_semi_positive

    def test_single_branch_all_ones(self):
        net = _net("&{a: end}")
        invs = compute_s_invariants(net)
        assert len(invs) >= 1
        # All-ones vector should appear
        place_ids = sorted(net.places.keys())
        has_all_ones = any(
            all(inv.weights.get(p, 0) == 1 for p in place_ids)
            for inv in invs
        )
        assert has_all_ones

    def test_two_branch_invariant(self):
        net = _net("&{a: end, b: end}")
        invs = compute_s_invariants(net)
        assert len(invs) >= 1
        # All should be strict (y^T * C = 0)
        assert all(inv.is_strict for inv in invs)

    def test_chain_invariant(self):
        net = _net("&{a: &{b: end}}")
        invs = compute_s_invariants(net)
        assert len(invs) >= 1
        # Token count should be 1 for state-machine nets
        for inv in invs:
            if inv.token_count is not None:
                assert inv.token_count == 1

    def test_selection_invariant(self):
        net = _net("+{x: end, y: end}")
        invs = compute_s_invariants(net)
        assert len(invs) >= 1

    def test_parallel_invariant(self):
        net = _net("(&{a: end} || &{b: end})")
        invs = compute_s_invariants(net)
        assert len(invs) >= 1
        # All-ones should be among them
        place_ids = sorted(net.places.keys())
        all_ones = {p: 1 for p in place_ids}
        found = any(
            all(inv.weights.get(p, 0) == 1 for p in place_ids)
            for inv in invs
        )
        assert found

    def test_empty_net_no_invariants(self):
        net = _empty_net()
        invs = compute_s_invariants(net)
        assert invs == []

    def test_support_is_frozenset(self):
        net = _net("&{a: end}")
        invs = compute_s_invariants(net)
        for inv in invs:
            assert isinstance(inv.support, frozenset)

    def test_invariant_covers_all_places(self):
        """For state-machine nets, the all-ones S-invariant covers all places."""
        net = _net("&{a: &{b: end}}")
        invs = compute_s_invariants(net)
        all_places = set(net.places.keys())
        covered = set()
        for inv in invs:
            covered.update(inv.support)
        assert covered == all_places


# ---------------------------------------------------------------------------
# S-invariant verification
# ---------------------------------------------------------------------------


class TestVerifySInvariant:
    """Test verification of S-invariants against reachable markings."""

    def test_verify_trivial(self):
        net = _net("&{a: end}")
        invs = compute_s_invariants(net)
        for inv in invs:
            assert verify_s_invariant(net, inv)

    def test_verify_chain(self):
        net = _net("&{a: &{b: &{c: end}}}")
        invs = compute_s_invariants(net)
        for inv in invs:
            assert verify_s_invariant(net, inv)

    def test_verify_recursive(self):
        net = _net("rec X . &{a: X}")
        invs = compute_s_invariants(net)
        for inv in invs:
            assert verify_s_invariant(net, inv)


# ---------------------------------------------------------------------------
# T-invariant computation
# ---------------------------------------------------------------------------


class TestComputeTInvariants:
    """Test T-invariant (transition invariant) computation."""

    def test_end_no_t_invariants(self):
        """'end' has no transitions, hence no T-invariants."""
        net = _net("end")
        invs = compute_t_invariants(net)
        assert invs == []

    def test_acyclic_no_t_invariants(self):
        """Acyclic nets (non-recursive types) have no T-invariants."""
        net = _net("&{a: end, b: end}")
        invs = compute_t_invariants(net)
        # Acyclic state-machine net: C has full column rank, no null space
        assert len(invs) == 0

    def test_chain_no_t_invariants(self):
        """Linear chain: no cycles, no T-invariants."""
        net = _net("&{a: &{b: end}}")
        invs = compute_t_invariants(net)
        assert len(invs) == 0

    def test_recursive_has_t_invariant(self):
        """rec X . &{a: X} creates a self-loop => T-invariant exists."""
        net = _net("rec X . &{a: X}")
        invs = compute_t_invariants(net)
        assert len(invs) >= 1
        # The T-invariant should cover the self-loop transition
        for inv in invs:
            assert len(inv.support) >= 1
            assert len(inv.transition_labels) >= 1

    def test_recursive_t_invariant_labels(self):
        """T-invariant labels should include 'a' for rec X . &{a: X}."""
        net = _net("rec X . &{a: X}")
        invs = compute_t_invariants(net)
        assert len(invs) >= 1
        all_labels = set()
        for inv in invs:
            all_labels.update(inv.transition_labels)
        assert "a" in all_labels

    def test_recursive_branch_t_invariants(self):
        """rec X . &{a: X, b: end} — the 'a' loop gives a T-invariant."""
        net = _net("rec X . &{a: X, b: end}")
        invs = compute_t_invariants(net)
        # There should be a T-invariant for the 'a' self-loop
        assert len(invs) >= 1

    def test_empty_net_no_t_invariants(self):
        net = _empty_net()
        invs = compute_t_invariants(net)
        assert invs == []

    def test_selection_no_t_invariants(self):
        """Non-recursive selection: no cycles."""
        net = _net("+{ok: end, err: end}")
        invs = compute_t_invariants(net)
        assert len(invs) == 0

    def test_t_invariant_support_is_frozenset(self):
        net = _net("rec X . &{a: X}")
        invs = compute_t_invariants(net)
        for inv in invs:
            assert isinstance(inv.support, frozenset)


# ---------------------------------------------------------------------------
# T-invariant verification
# ---------------------------------------------------------------------------


class TestVerifyTInvariant:
    """Test algebraic verification of T-invariants."""

    def test_verify_recursive(self):
        net = _net("rec X . &{a: X}")
        invs = compute_t_invariants(net)
        for inv in invs:
            assert verify_t_invariant(net, inv)

    def test_verify_recursive_branch(self):
        net = _net("rec X . &{a: X, b: end}")
        invs = compute_t_invariants(net)
        for inv in invs:
            assert verify_t_invariant(net, inv)


# ---------------------------------------------------------------------------
# Full structural analysis
# ---------------------------------------------------------------------------


class TestAnalyzeStructuralInvariants:
    """Test the full analysis pipeline."""

    def test_end_analysis(self):
        net = _net("end")
        result = analyze_structural_invariants(net)
        assert isinstance(result, StructuralInvariantResult)
        assert result.num_s_invariants >= 1
        assert result.num_t_invariants == 0
        assert result.is_s_covered
        assert not result.has_recursion

    def test_simple_branch_analysis(self):
        net = _net("&{a: end}")
        result = analyze_structural_invariants(net)
        assert result.is_conservative
        assert result.is_s_covered
        assert not result.has_recursion

    def test_recursive_analysis(self):
        net = _net("rec X . &{a: X}")
        result = analyze_structural_invariants(net)
        assert result.has_recursion
        assert result.num_t_invariants >= 1
        assert result.is_s_covered  # Still conservative

    def test_recursive_branch_analysis(self):
        net = _net("rec X . &{a: X, b: end}")
        result = analyze_structural_invariants(net)
        assert result.has_recursion
        assert result.is_conservative

    def test_parallel_analysis(self):
        net = _net("(&{a: end} || &{b: end})")
        result = analyze_structural_invariants(net)
        assert result.is_conservative
        assert result.is_s_covered

    def test_nested_branch_analysis(self):
        net = _net("&{a: &{b: end, c: end}}")
        result = analyze_structural_invariants(net)
        assert result.is_conservative
        assert result.is_s_covered
        assert not result.has_recursion

    def test_empty_net_analysis(self):
        net = _empty_net()
        result = analyze_structural_invariants(net)
        assert result.num_s_invariants == 0
        assert result.num_t_invariants == 0
        assert result.is_s_covered  # vacuously
        assert not result.has_recursion

    def test_incidence_matrix_stored(self):
        net = _net("&{a: end}")
        result = analyze_structural_invariants(net)
        assert isinstance(result.incidence_matrix, dict)


# ---------------------------------------------------------------------------
# Benchmark protocols
# ---------------------------------------------------------------------------


class TestBenchmarkInvariants:
    """Test structural invariants on benchmark session types."""

    BENCHMARKS = [
        ("Java Iterator", "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"),
        ("File Object", "&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}"),
        ("SMTP", "&{connect: &{ehlo: rec X . &{mail: &{rcpt: &{data: +{OK: X, ERR: X}}}, quit: end}}}"),
        ("Simple parallel", "(&{a: end} || &{b: end})"),
        ("Selection", "+{ok: &{get: end}, err: end}"),
    ]

    @pytest.mark.parametrize("name,ts", BENCHMARKS, ids=[b[0] for b in BENCHMARKS])
    def test_s_covered(self, name: str, ts: str):
        """Every benchmark net should be S-covered (conservative)."""
        net = _net(ts)
        result = analyze_structural_invariants(net)
        assert result.is_s_covered, f"{name}: not S-covered"

    @pytest.mark.parametrize("name,ts", BENCHMARKS, ids=[b[0] for b in BENCHMARKS])
    def test_s_invariants_verified(self, name: str, ts: str):
        """All S-invariants should verify against reachable markings."""
        net = _net(ts)
        invs = compute_s_invariants(net)
        for inv in invs:
            assert verify_s_invariant(net, inv), f"{name}: S-invariant failed verification"

    @pytest.mark.parametrize("name,ts", BENCHMARKS, ids=[b[0] for b in BENCHMARKS])
    def test_t_invariants_verified(self, name: str, ts: str):
        """All T-invariants should algebraically verify."""
        net = _net(ts)
        invs = compute_t_invariants(net)
        for inv in invs:
            assert verify_t_invariant(net, inv), f"{name}: T-invariant failed verification"

    def test_recursive_benchmarks_have_t_invariants(self):
        """Recursive benchmarks should have at least one T-invariant."""
        recursive_benchmarks = [
            ("Java Iterator", "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"),
            ("File Object", "&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}"),
            ("SMTP", "&{connect: &{ehlo: rec X . &{mail: &{rcpt: &{data: +{OK: X, ERR: X}}}, quit: end}}}"),
        ]
        for name, ts in recursive_benchmarks:
            net = _net(ts)
            invs = compute_t_invariants(net)
            assert len(invs) >= 1, f"{name}: no T-invariants for recursive type"

    def test_non_recursive_no_t_invariants(self):
        """Non-recursive benchmarks should have no T-invariants."""
        non_recursive = [
            ("Simple parallel", "(&{a: end} || &{b: end})"),
            ("Selection", "+{ok: &{get: end}, err: end}"),
        ]
        for name, ts in non_recursive:
            net = _net(ts)
            invs = compute_t_invariants(net)
            assert len(invs) == 0, f"{name}: unexpected T-invariant"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and corner cases."""

    def test_deeply_nested(self):
        net = _net("&{a: &{b: &{c: &{d: end}}}}")
        result = analyze_structural_invariants(net)
        assert result.is_conservative
        assert result.is_s_covered
        assert not result.has_recursion

    def test_wide_branch(self):
        net = _net("&{a: end, b: end, c: end, d: end, e: end}")
        result = analyze_structural_invariants(net)
        assert result.is_conservative
        assert result.is_s_covered

    def test_self_loop_recursive(self):
        """rec X . &{a: X} creates self-loop — both S and T invariants."""
        net = _net("rec X . &{a: X}")
        result = analyze_structural_invariants(net)
        assert result.is_conservative
        assert result.has_recursion
        assert result.num_t_invariants >= 1
        assert result.num_s_invariants >= 1

    def test_nested_recursion(self):
        """rec X . &{a: &{b: X}, c: end} — cycle through multiple states."""
        net = _net("rec X . &{a: &{b: X}, c: end}")
        result = analyze_structural_invariants(net)
        assert result.is_conservative
        assert result.has_recursion

    def test_single_place_no_transitions(self):
        net = _net("end")
        result = analyze_structural_invariants(net)
        assert result.is_s_covered
        assert result.num_s_invariants >= 1
