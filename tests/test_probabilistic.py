"""Tests for probabilistic session types (Step 96).

Covers: probability assignment (uniform/custom), entropy, expected path length,
Markov chain construction, stationary distribution, expected visits, and full
probabilistic analysis.
"""

from __future__ import annotations

import math

import pytest

from reticulate.parser import parse
from reticulate.statespace import StateSpace, build_statespace
from reticulate.probabilistic import (
    ProbabilisticAssignment,
    ProbabilisticAnalysis,
    assign_uniform,
    assign_weights,
    state_entropy,
    total_entropy,
    path_entropy,
    markov_chain,
    expected_path_length,
    stationary_distribution,
    expected_visits,
    analyze_probabilistic,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ss(type_str: str) -> StateSpace:
    """Parse and build state space from type string."""
    return build_statespace(parse(type_str))


def _approx(val: float, expected: float, tol: float = 1e-6) -> bool:
    """Check approximate equality."""
    return abs(val - expected) < tol


# ---------------------------------------------------------------------------
# 1. Probability assignment: uniform
# ---------------------------------------------------------------------------

class TestAssignUniform:
    """Tests for assign_uniform()."""

    def test_end_type(self) -> None:
        ss = _ss("end")
        probs = assign_uniform(ss)
        # end has no outgoing transitions — no weights
        assert probs.weights == {}

    def test_single_branch(self) -> None:
        ss = _ss("&{a: end}")
        probs = assign_uniform(ss)
        # Top state has one outgoing transition -> p=1.0
        assert ss.top in probs.weights
        w = probs.weights[ss.top]
        assert _approx(w["a"], 1.0)

    def test_two_branches(self) -> None:
        ss = _ss("&{a: end, b: end}")
        probs = assign_uniform(ss)
        w = probs.weights[ss.top]
        assert _approx(w["a"], 0.5)
        assert _approx(w["b"], 0.5)

    def test_three_branches(self) -> None:
        ss = _ss("&{a: end, b: end, c: end}")
        probs = assign_uniform(ss)
        w = probs.weights[ss.top]
        for label in ["a", "b", "c"]:
            assert _approx(w[label], 1.0 / 3)

    def test_selection_uniform(self) -> None:
        ss = _ss("+{ok: end, err: end}")
        probs = assign_uniform(ss)
        w = probs.weights[ss.top]
        assert _approx(w["ok"], 0.5)
        assert _approx(w["err"], 0.5)

    def test_nested_branches(self) -> None:
        ss = _ss("&{a: &{x: end, y: end}, b: end}")
        probs = assign_uniform(ss)
        # Top has 2 outgoing transitions
        w_top = probs.weights[ss.top]
        assert _approx(w_top["a"], 0.5)
        assert _approx(w_top["b"], 0.5)
        # Inner branch state has 2 outgoing transitions
        inner_states = [s for s in ss.states if s != ss.top and s != ss.bottom]
        has_inner = False
        for s in inner_states:
            w = probs.weights.get(s, {})
            if "x" in w and "y" in w:
                has_inner = True
                assert _approx(w["x"], 0.5)
                assert _approx(w["y"], 0.5)
        assert has_inner

    def test_probability_method(self) -> None:
        ss = _ss("&{a: end, b: end}")
        probs = assign_uniform(ss)
        assert _approx(probs.probability(ss.top, "a"), 0.5)
        assert _approx(probs.probability(ss.top, "b"), 0.5)
        assert _approx(probs.probability(ss.top, "nonexistent"), 0.0)
        assert _approx(probs.probability(999, "a"), 0.0)


# ---------------------------------------------------------------------------
# 2. Probability assignment: custom weights
# ---------------------------------------------------------------------------

class TestAssignWeights:
    """Tests for assign_weights()."""

    def test_valid_custom_weights(self) -> None:
        ss = _ss("&{a: end, b: end}")
        weights = {ss.top: {"a": 0.7, "b": 0.3}}
        probs = assign_weights(ss, weights)
        assert _approx(probs.probability(ss.top, "a"), 0.7)
        assert _approx(probs.probability(ss.top, "b"), 0.3)

    def test_probabilities_not_summing_to_one(self) -> None:
        ss = _ss("&{a: end, b: end}")
        with pytest.raises(ValueError, match="sum to"):
            assign_weights(ss, {ss.top: {"a": 0.5, "b": 0.3}})

    def test_missing_label(self) -> None:
        ss = _ss("&{a: end, b: end}")
        with pytest.raises(ValueError, match="Missing weight"):
            assign_weights(ss, {ss.top: {"a": 1.0}})

    def test_extra_label(self) -> None:
        ss = _ss("&{a: end, b: end}")
        with pytest.raises(ValueError, match="does not match"):
            assign_weights(ss, {ss.top: {"a": 0.5, "b": 0.3, "c": 0.2}})

    def test_missing_state(self) -> None:
        ss = _ss("&{a: end, b: end}")
        with pytest.raises(ValueError, match="no weights assigned"):
            assign_weights(ss, {})

    def test_negative_probability(self) -> None:
        ss = _ss("&{a: end, b: end}")
        with pytest.raises(ValueError, match="Negative"):
            assign_weights(ss, {ss.top: {"a": 1.5, "b": -0.5}})

    def test_deterministic_weight(self) -> None:
        ss = _ss("&{a: end, b: end}")
        probs = assign_weights(ss, {ss.top: {"a": 1.0, "b": 0.0}})
        assert _approx(probs.probability(ss.top, "a"), 1.0)
        assert _approx(probs.probability(ss.top, "b"), 0.0)

    def test_end_type_empty_weights(self) -> None:
        ss = _ss("end")
        probs = assign_weights(ss, {})
        assert probs.weights == {}


# ---------------------------------------------------------------------------
# 3. Shannon entropy
# ---------------------------------------------------------------------------

class TestEntropy:
    """Tests for state_entropy(), total_entropy(), path_entropy()."""

    def test_entropy_uniform_two(self) -> None:
        ss = _ss("&{a: end, b: end}")
        probs = assign_uniform(ss)
        h = state_entropy(ss, probs, ss.top)
        assert _approx(h, 1.0)  # log2(2) = 1 bit

    def test_entropy_uniform_four(self) -> None:
        ss = _ss("&{a: end, b: end, c: end, d: end}")
        probs = assign_uniform(ss)
        h = state_entropy(ss, probs, ss.top)
        assert _approx(h, 2.0)  # log2(4) = 2 bits

    def test_entropy_deterministic(self) -> None:
        ss = _ss("&{a: end, b: end}")
        probs = assign_weights(ss, {ss.top: {"a": 1.0, "b": 0.0}})
        h = state_entropy(ss, probs, ss.top)
        assert _approx(h, 0.0)  # deterministic = 0 entropy

    def test_entropy_skewed(self) -> None:
        ss = _ss("&{ok: end, err: end}")
        probs = assign_weights(ss, {ss.top: {"ok": 0.9, "err": 0.1}})
        h = state_entropy(ss, probs, ss.top)
        expected = -(0.9 * math.log2(0.9) + 0.1 * math.log2(0.1))
        assert _approx(h, expected)

    def test_entropy_bottom_state(self) -> None:
        ss = _ss("&{a: end}")
        probs = assign_uniform(ss)
        h = state_entropy(ss, probs, ss.bottom)
        assert _approx(h, 0.0)

    def test_total_entropy_single_choice(self) -> None:
        ss = _ss("&{a: end, b: end}")
        probs = assign_uniform(ss)
        te = total_entropy(ss, probs)
        assert _approx(te, 1.0)

    def test_total_entropy_nested(self) -> None:
        ss = _ss("&{a: &{x: end, y: end}, b: end}")
        probs = assign_uniform(ss)
        te = total_entropy(ss, probs)
        # top: 1 bit, inner: 1 bit => total = 2 bits
        assert _approx(te, 2.0)

    def test_path_entropy_equals_total_for_connected(self) -> None:
        ss = _ss("&{a: end, b: end}")
        probs = assign_uniform(ss)
        pe = path_entropy(ss, probs)
        te = total_entropy(ss, probs)
        assert _approx(pe, te)


# ---------------------------------------------------------------------------
# 4. Markov chain construction
# ---------------------------------------------------------------------------

class TestMarkovChain:
    """Tests for markov_chain()."""

    def test_end_type_absorbing(self) -> None:
        ss = _ss("end")
        probs = assign_uniform(ss)
        mc = markov_chain(ss, probs)
        # Single absorbing state
        assert mc[ss.bottom][ss.bottom] == 1.0

    def test_simple_branch(self) -> None:
        ss = _ss("&{a: end, b: end}")
        probs = assign_uniform(ss)
        mc = markov_chain(ss, probs)
        # Top transitions to bottom with total prob 1.0
        assert _approx(mc[ss.top].get(ss.bottom, 0.0), 1.0)
        # Bottom is absorbing
        assert _approx(mc[ss.bottom].get(ss.bottom, 0.0), 1.0)

    def test_skewed_branch(self) -> None:
        ss = _ss("&{a: end, b: end}")
        probs = assign_weights(ss, {ss.top: {"a": 0.7, "b": 0.3}})
        mc = markov_chain(ss, probs)
        # Both a and b go to bottom, so P(top, bottom) = 1.0
        assert _approx(mc[ss.top].get(ss.bottom, 0.0), 1.0)

    def test_nested_branch_probabilities(self) -> None:
        ss = _ss("&{a: &{x: end, y: end}, b: end}")
        probs = assign_uniform(ss)
        mc = markov_chain(ss, probs)
        # Row sums should be 1.0 for all states
        for s in ss.states:
            row_sum = sum(mc[s].values())
            assert _approx(row_sum, 1.0), f"Row sum for state {s} is {row_sum}"

    def test_selection_markov_chain(self) -> None:
        ss = _ss("+{ok: end, err: end}")
        probs = assign_uniform(ss)
        mc = markov_chain(ss, probs)
        # Row sums = 1
        for s in ss.states:
            row_sum = sum(mc[s].values())
            assert _approx(row_sum, 1.0)

    def test_row_sums_three_states(self) -> None:
        ss = _ss("&{a: &{x: end, y: end}, b: &{u: end, v: end}}")
        probs = assign_uniform(ss)
        mc = markov_chain(ss, probs)
        for s in ss.states:
            row_sum = sum(mc[s].values())
            assert _approx(row_sum, 1.0), f"Row sum for state {s} is {row_sum}"


# ---------------------------------------------------------------------------
# 5. Expected path length
# ---------------------------------------------------------------------------

class TestExpectedPathLength:
    """Tests for expected_path_length()."""

    def test_end_type(self) -> None:
        ss = _ss("end")
        probs = assign_uniform(ss)
        epl = expected_path_length(ss, probs)
        assert epl is not None
        assert _approx(epl, 0.0)

    def test_single_step(self) -> None:
        ss = _ss("&{a: end}")
        probs = assign_uniform(ss)
        epl = expected_path_length(ss, probs)
        assert epl is not None
        assert _approx(epl, 1.0)

    def test_two_branches_to_end(self) -> None:
        # Both branches go directly to end: expected length = 1
        ss = _ss("&{a: end, b: end}")
        probs = assign_uniform(ss)
        epl = expected_path_length(ss, probs)
        assert epl is not None
        assert _approx(epl, 1.0)

    def test_two_step_chain(self) -> None:
        ss = _ss("&{a: &{b: end}}")
        probs = assign_uniform(ss)
        epl = expected_path_length(ss, probs)
        assert epl is not None
        assert _approx(epl, 2.0)

    def test_weighted_expected_length(self) -> None:
        # &{a: &{x: end}, b: end}
        # Path a-x: length 2, path b: length 1
        # With p(a)=0.3, p(b)=0.7: E = 0.3*2 + 0.7*1 = 1.3
        ss = _ss("&{a: &{x: end}, b: end}")
        # Find the inner state
        top = ss.top
        outgoing = ss.enabled(top)
        weights: dict[int, dict[str, float]] = {}
        weights[top] = {"a": 0.3, "b": 0.7}
        # Inner state with "x" has one transition
        for s in ss.states:
            out = ss.enabled(s)
            if s != top and s != ss.bottom and out:
                w = {}
                for label, _tgt in out:
                    w[label] = 1.0
                weights[s] = w
        probs = assign_weights(ss, weights)
        epl = expected_path_length(ss, probs)
        assert epl is not None
        assert _approx(epl, 1.3)

    def test_recursive_geometric(self) -> None:
        # rec X . &{done: end, retry: X}
        # Geometric distribution with p=0.5 => E[T] = 1/0.5 = 2
        ss = _ss("rec X . &{done: end, retry: X}")
        top = ss.top
        weights = {top: {"done": 0.5, "retry": 0.5}}
        probs = assign_weights(ss, weights)
        epl = expected_path_length(ss, probs)
        assert epl is not None
        assert _approx(epl, 2.0)

    def test_recursive_skewed(self) -> None:
        # rec X . &{done: end, retry: X}
        # p(done)=0.1, p(retry)=0.9 => E[T] = 1/0.1 = 10
        ss = _ss("rec X . &{done: end, retry: X}")
        top = ss.top
        weights = {top: {"done": 0.1, "retry": 0.9}}
        probs = assign_weights(ss, weights)
        epl = expected_path_length(ss, probs)
        assert epl is not None
        assert _approx(epl, 10.0)

    def test_selection_expected_length(self) -> None:
        ss = _ss("+{ok: end, err: end}")
        probs = assign_uniform(ss)
        epl = expected_path_length(ss, probs)
        assert epl is not None
        assert _approx(epl, 1.0)


# ---------------------------------------------------------------------------
# 6. Stationary distribution
# ---------------------------------------------------------------------------

class TestStationaryDistribution:
    """Tests for stationary_distribution()."""

    def test_end_type(self) -> None:
        ss = _ss("end")
        probs = assign_uniform(ss)
        sd = stationary_distribution(ss, probs)
        # Single state, trivially concentrates on it
        if sd is not None:
            assert _approx(sd[ss.bottom], 1.0)

    def test_simple_branch_absorbing(self) -> None:
        # Non-recursive type: bottom absorbs, stationary dist concentrates on bottom
        ss = _ss("&{a: end, b: end}")
        probs = assign_uniform(ss)
        sd = stationary_distribution(ss, probs)
        if sd is not None:
            assert _approx(sd.get(ss.bottom, 0.0), 1.0)

    def test_server_loop(self) -> None:
        # rec X . &{read: X, write: X}
        # All paths loop back. Steady state: pi = [1.0] (single state)
        ss = _ss("rec X . &{read: X, write: X}")
        probs = assign_uniform(ss)
        sd = stationary_distribution(ss, probs)
        # Should have a valid distribution
        if sd is not None:
            total = sum(sd.values())
            assert _approx(total, 1.0)


# ---------------------------------------------------------------------------
# 7. Expected visits
# ---------------------------------------------------------------------------

class TestExpectedVisits:
    """Tests for expected_visits()."""

    def test_single_step(self) -> None:
        ss = _ss("&{a: end}")
        probs = assign_uniform(ss)
        ev = expected_visits(ss, probs, ss.top)
        assert ev is not None
        assert _approx(ev, 1.0)

    def test_bottom_zero_visits(self) -> None:
        ss = _ss("&{a: end}")
        probs = assign_uniform(ss)
        ev = expected_visits(ss, probs, ss.bottom)
        assert ev is not None
        assert _approx(ev, 0.0)

    def test_recursive_expected_visits(self) -> None:
        # rec X . &{done: end, retry: X} with p=0.5
        # Expected visits to top = 1/(1-0.5) = 2
        ss = _ss("rec X . &{done: end, retry: X}")
        top = ss.top
        weights = {top: {"done": 0.5, "retry": 0.5}}
        probs = assign_weights(ss, weights)
        ev = expected_visits(ss, probs, top)
        assert ev is not None
        assert _approx(ev, 2.0)

    def test_nonexistent_state(self) -> None:
        ss = _ss("&{a: end}")
        probs = assign_uniform(ss)
        ev = expected_visits(ss, probs, 99999)
        assert ev is not None
        assert _approx(ev, 0.0)


# ---------------------------------------------------------------------------
# 8. Full analysis
# ---------------------------------------------------------------------------

class TestAnalyzeProbabilistic:
    """Tests for analyze_probabilistic()."""

    def test_end_type_analysis(self) -> None:
        ss = _ss("end")
        probs = assign_uniform(ss)
        result = analyze_probabilistic(ss, probs)
        assert isinstance(result, ProbabilisticAnalysis)
        assert result.expected_path_length is not None
        assert _approx(result.expected_path_length, 0.0)
        assert _approx(result.total_entropy, 0.0)

    def test_simple_analysis(self) -> None:
        ss = _ss("&{a: end, b: end}")
        probs = assign_uniform(ss)
        result = analyze_probabilistic(ss, probs)
        assert result.expected_path_length is not None
        assert _approx(result.expected_path_length, 1.0)
        assert _approx(result.total_entropy, 1.0)
        assert result.is_absorbing
        assert result.transition_matrix is not None

    def test_nested_analysis(self) -> None:
        ss = _ss("&{a: &{x: end, y: end}, b: end}")
        probs = assign_uniform(ss)
        result = analyze_probabilistic(ss, probs)
        assert result.expected_path_length is not None
        # E[T] = 0.5 * 2 + 0.5 * 1 = 1.5
        assert _approx(result.expected_path_length, 1.5)
        # Total entropy: top 1 bit + inner 1 bit = 2 bits
        assert _approx(result.total_entropy, 2.0)

    def test_analysis_has_all_fields(self) -> None:
        ss = _ss("&{a: end, b: end}")
        probs = assign_uniform(ss)
        result = analyze_probabilistic(ss, probs)
        assert result.assignment is probs
        assert isinstance(result.state_entropies, dict)
        assert isinstance(result.transition_matrix, dict)
        assert isinstance(result.is_absorbing, bool)

    def test_recursive_analysis(self) -> None:
        ss = _ss("rec X . &{done: end, retry: X}")
        top = ss.top
        weights = {top: {"done": 0.5, "retry": 0.5}}
        probs = assign_weights(ss, weights)
        result = analyze_probabilistic(ss, probs)
        assert result.expected_path_length is not None
        assert _approx(result.expected_path_length, 2.0)

    def test_selection_analysis(self) -> None:
        ss = _ss("+{ok: end, err: end}")
        probs = assign_uniform(ss)
        result = analyze_probabilistic(ss, probs)
        assert result.expected_path_length is not None
        assert _approx(result.expected_path_length, 1.0)
        assert _approx(result.total_entropy, 1.0)


# ---------------------------------------------------------------------------
# 9. Edge cases and properties
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and mathematical properties."""

    def test_entropy_non_negative(self) -> None:
        """Entropy is always non-negative."""
        for type_str in ["end", "&{a: end}", "&{a: end, b: end}",
                         "+{x: end, y: end}", "&{a: &{x: end}, b: end}"]:
            ss = _ss(type_str)
            probs = assign_uniform(ss)
            for s in ss.states:
                h = state_entropy(ss, probs, s)
                assert h >= 0.0, f"Negative entropy at state {s}"

    def test_entropy_bounded_by_log_n(self) -> None:
        """Entropy at each state is bounded by log2(n) where n = number of outgoing transitions."""
        ss = _ss("&{a: end, b: end, c: end}")
        probs = assign_uniform(ss)
        h = state_entropy(ss, probs, ss.top)
        n = len(ss.enabled(ss.top))
        assert h <= math.log2(n) + 1e-9

    def test_uniform_maximizes_entropy(self) -> None:
        """Uniform distribution maximizes entropy."""
        ss = _ss("&{a: end, b: end, c: end}")
        probs_uniform = assign_uniform(ss)
        h_uniform = state_entropy(ss, probs_uniform, ss.top)

        probs_skewed = assign_weights(ss, {ss.top: {"a": 0.8, "b": 0.1, "c": 0.1}})
        h_skewed = state_entropy(ss, probs_skewed, ss.top)

        assert h_uniform >= h_skewed - 1e-9

    def test_expected_path_length_non_negative(self) -> None:
        """Expected path length is non-negative."""
        for type_str in ["end", "&{a: end}", "&{a: end, b: end}",
                         "&{a: &{x: end}, b: end}"]:
            ss = _ss(type_str)
            probs = assign_uniform(ss)
            epl = expected_path_length(ss, probs)
            assert epl is not None
            assert epl >= 0.0

    def test_markov_chain_row_sums(self) -> None:
        """All rows in the transition matrix sum to 1."""
        for type_str in ["end", "&{a: end}", "&{a: end, b: end}",
                         "+{x: end, y: end}", "&{a: &{x: end}, b: end}"]:
            ss = _ss(type_str)
            probs = assign_uniform(ss)
            mc = markov_chain(ss, probs)
            for s in ss.states:
                row_sum = sum(mc[s].values())
                assert _approx(row_sum, 1.0), \
                    f"Row sum {row_sum} != 1.0 for type {type_str}, state {s}"

    def test_probabilistic_assignment_frozen(self) -> None:
        """ProbabilisticAssignment is frozen."""
        probs = ProbabilisticAssignment(weights={})
        with pytest.raises(AttributeError):
            probs.weights = {}  # type: ignore[misc]

    def test_analysis_result_frozen(self) -> None:
        """ProbabilisticAnalysis is frozen."""
        ss = _ss("end")
        probs = assign_uniform(ss)
        result = analyze_probabilistic(ss, probs)
        with pytest.raises(AttributeError):
            result.total_entropy = 0.0  # type: ignore[misc]

    def test_recursive_high_exit_prob(self) -> None:
        """Recursive type with high exit probability: short expected path."""
        ss = _ss("rec X . &{done: end, retry: X}")
        probs = assign_weights(ss, {ss.top: {"done": 0.9, "retry": 0.1}})
        epl = expected_path_length(ss, probs)
        assert epl is not None
        # E[T] = 1/0.9 ≈ 1.111
        assert _approx(epl, 1.0 / 0.9)

    def test_parallel_type_uniform(self) -> None:
        """Parallel type gets uniform probabilities assigned."""
        ss = _ss("(&{a: end} || &{b: end})")
        probs = assign_uniform(ss)
        # Should have weights for all non-bottom states with outgoing transitions
        for s in ss.states:
            outgoing = ss.enabled(s)
            if outgoing:
                assert s in probs.weights
                w = probs.weights[s]
                n = len(outgoing)
                for label, _tgt in outgoing:
                    assert _approx(w[label], 1.0 / n)


# ---------------------------------------------------------------------------
# 10. Benchmark protocols
# ---------------------------------------------------------------------------

class TestBenchmarks:
    """Tests on realistic protocol patterns."""

    def test_tcp_retransmission_pattern(self) -> None:
        """TCP-like: send, then either ack or retry."""
        ss = _ss("rec X . &{send: +{ack: end, retry: X}}")
        top = ss.top
        # Find the selection state (after send)
        outgoing_top = ss.enabled(top)
        assert len(outgoing_top) == 1
        send_label, send_tgt = outgoing_top[0]
        assert send_label == "send"

        weights: dict[int, dict[str, float]] = {}
        weights[top] = {"send": 1.0}
        weights[send_tgt] = {"ack": 0.8, "retry": 0.2}
        probs = assign_weights(ss, weights)

        epl = expected_path_length(ss, probs)
        assert epl is not None
        # Each attempt: 2 steps (send + ack/retry). E[attempts] = 1/0.8 = 1.25
        # E[T] = E[attempts] * 2 steps per attempt... but actually:
        # Let E be expected steps from top.
        # E = 1 + (0.8 * 1 + 0.2 * (1 + E))
        # E = 1 + 0.8 + 0.2 + 0.2*E = 2 + 0.2*E
        # 0.8*E = 2 => E = 2.5
        assert _approx(epl, 2.5)

    def test_oauth_like_protocol(self) -> None:
        """OAuth-like: request token, then success or error."""
        ss = _ss("&{request: +{success: end, error: end}}")
        probs = assign_uniform(ss)
        epl = expected_path_length(ss, probs)
        assert epl is not None
        assert _approx(epl, 2.0)

    def test_iterator_pattern(self) -> None:
        """Iterator: hasNext then TRUE(next, loop) or FALSE(end)."""
        ss = _ss("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        top = ss.top
        # Find all states and assign uniform
        probs = assign_uniform(ss)
        epl = expected_path_length(ss, probs)
        assert epl is not None
        assert epl > 0
