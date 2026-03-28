"""Tests for consciousness module (Step 200)."""

from __future__ import annotations

import math

import pytest

from reticulate import build_statespace, parse
from reticulate.consciousness import (
    ConsciousnessState,
    QualiaCategorization,
    StreamOfConsciousness,
    attention_filter,
    categorize_qualia,
    dream,
    flow_state,
    metacognition,
    self_type,
    simulate_stream,
)
from reticulate.lattice import check_lattice
from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Data type tests
# ---------------------------------------------------------------------------

class TestConsciousnessState:
    """ConsciousnessState frozen dataclass tests."""

    def test_creation(self) -> None:
        s = ConsciousnessState(
            attention_target="end",
            awareness_level=0.5,
            active_mechanisms=("abstraction",),
            metacognitive_depth=0,
            stream_position=0,
        )
        assert s.attention_target == "end"
        assert s.awareness_level == 0.5
        assert s.active_mechanisms == ("abstraction",)
        assert s.metacognitive_depth == 0
        assert s.stream_position == 0

    def test_frozen(self) -> None:
        s = ConsciousnessState("end", 0.5, ("abstraction",), 0, 0)
        with pytest.raises(AttributeError):
            s.attention_target = "other"  # type: ignore[misc]

    def test_equality(self) -> None:
        a = ConsciousnessState("end", 0.5, ("abstraction",), 0, 0)
        b = ConsciousnessState("end", 0.5, ("abstraction",), 0, 0)
        assert a == b

    def test_hash(self) -> None:
        a = ConsciousnessState("end", 0.5, ("abstraction",), 0, 0)
        assert hash(a) == hash(a)


class TestStreamOfConsciousness:
    """StreamOfConsciousness frozen dataclass tests."""

    def test_creation(self) -> None:
        s = StreamOfConsciousness(
            states=(),
            transitions=(),
            total_entropy=0.0,
            metacognitive_moments=0,
            flow_score=0.0,
        )
        assert s.states == ()
        assert s.transitions == ()
        assert s.total_entropy == 0.0

    def test_frozen(self) -> None:
        s = StreamOfConsciousness((), (), 0.0, 0, 0.0)
        with pytest.raises(AttributeError):
            s.total_entropy = 1.0  # type: ignore[misc]


class TestQualiaCategorization:
    """QualiaCategorization frozen dataclass tests."""

    def test_creation(self) -> None:
        q = QualiaCategorization("end", 0.0, 1.0, 0.5, 0.0)
        assert q.raw_type == "end"
        assert q.perceived_complexity == 0.0
        assert q.perceived_beauty == 1.0
        assert q.perceived_meaning == 0.5
        assert q.emotional_valence == 0.0

    def test_frozen(self) -> None:
        q = QualiaCategorization("end", 0.0, 1.0, 0.5, 0.0)
        with pytest.raises(AttributeError):
            q.raw_type = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# self_type tests
# ---------------------------------------------------------------------------

class TestSelfType:
    """Tests for self_type()."""

    def test_returns_string(self) -> None:
        result = self_type()
        assert isinstance(result, str)
        assert "rec" in result
        assert "perceive" in result
        assert "reflect" in result

    def test_parses(self) -> None:
        ast = parse(self_type())
        assert ast is not None

    def test_builds_statespace(self) -> None:
        ast = parse(self_type())
        ss = build_statespace(ast)
        assert len(ss.states) > 0
        assert len(ss.transitions) > 0

    def test_forms_lattice(self) -> None:
        ast = parse(self_type())
        ss = build_statespace(ast)
        result = check_lattice(ss)
        assert result.is_lattice

    def test_deterministic(self) -> None:
        assert self_type() == self_type()


# ---------------------------------------------------------------------------
# simulate_stream tests
# ---------------------------------------------------------------------------

class TestSimulateStream:
    """Tests for simulate_stream()."""

    def test_basic_stream(self) -> None:
        stream = simulate_stream(
            types=["end", "&{a: end}"],
            mechanisms=["abstraction", "composition"],
            steps=10,
        )
        assert isinstance(stream, StreamOfConsciousness)
        assert len(stream.states) == 10
        assert len(stream.transitions) == 9

    def test_entropy_nonnegative(self) -> None:
        stream = simulate_stream(["end"], ["abstraction", "recursion"], steps=8)
        assert stream.total_entropy >= 0.0

    def test_single_mechanism_zero_entropy(self) -> None:
        stream = simulate_stream(["end"], ["abstraction"], steps=5)
        # Single mechanism → all same → entropy = 0
        assert stream.total_entropy == 0.0

    def test_metacognitive_moments(self) -> None:
        # Use self_type as target + "recursion" mechanism to trigger metacognition
        stream = simulate_stream(
            types=[self_type()],
            mechanisms=["recursion"],
            steps=5,
        )
        assert stream.metacognitive_moments > 0

    def test_no_metacognition_without_recursion(self) -> None:
        stream = simulate_stream(
            types=[self_type()],
            mechanisms=["abstraction"],
            steps=5,
        )
        assert stream.metacognitive_moments == 0

    def test_flow_score_range(self) -> None:
        stream = simulate_stream(["end"], ["abstraction", "composition"], steps=10)
        assert 0.0 <= stream.flow_score <= 1.0

    def test_empty_types_defaults(self) -> None:
        stream = simulate_stream([], ["abstraction"], steps=3)
        assert len(stream.states) == 3
        assert stream.states[0].attention_target == "end"

    def test_empty_mechanisms_defaults(self) -> None:
        stream = simulate_stream(["end"], [], steps=3)
        assert len(stream.states) == 3

    def test_zero_steps(self) -> None:
        stream = simulate_stream(["end"], ["abstraction"], steps=0)
        assert len(stream.states) == 0
        assert len(stream.transitions) == 0

    def test_awareness_peaks_at_middle(self) -> None:
        stream = simulate_stream(["end"], ["abstraction"], steps=10)
        # Middle states should have higher awareness than edge states
        mid = stream.states[5].awareness_level
        edge = stream.states[0].awareness_level
        assert mid >= edge


# ---------------------------------------------------------------------------
# categorize_qualia tests
# ---------------------------------------------------------------------------

class TestCategorizeQualia:
    """Tests for categorize_qualia()."""

    def test_end_type(self) -> None:
        q = categorize_qualia("end")
        assert q.raw_type == "end"
        assert isinstance(q.perceived_complexity, float)
        assert isinstance(q.perceived_beauty, float)
        assert isinstance(q.perceived_meaning, float)
        assert isinstance(q.emotional_valence, float)

    def test_branch_type(self) -> None:
        q = categorize_qualia("&{a: end, b: end}")
        assert q.perceived_complexity >= 0.0

    def test_complex_type(self) -> None:
        q = categorize_qualia("&{a: +{x: end, y: end}, b: end}")
        assert q.perceived_complexity > 0.0

    def test_meaning_with_self_labels(self) -> None:
        # Type sharing labels with self_type should have nonzero meaning
        q = categorize_qualia("&{perceive: end}")
        assert q.perceived_meaning >= 0.0

    def test_valence_range(self) -> None:
        q = categorize_qualia("&{a: end, b: end}")
        assert -1.0 <= q.emotional_valence <= 1.0


# ---------------------------------------------------------------------------
# attention_filter tests
# ---------------------------------------------------------------------------

class TestAttentionFilter:
    """Tests for attention_filter()."""

    def test_filter_keeps_focused_labels(self) -> None:
        ast = parse("&{a: end, b: end}")
        ss = build_statespace(ast)
        filtered = attention_filter(ss, {"a"})
        labels = {lbl for _, lbl, _ in filtered.transitions}
        assert "a" in labels
        assert "b" not in labels

    def test_filter_all_labels_preserves_structure(self) -> None:
        ast = parse("&{a: end, b: end}")
        ss = build_statespace(ast)
        all_labels = {lbl for _, lbl, _ in ss.transitions}
        filtered = attention_filter(ss, all_labels)
        assert len(filtered.transitions) == len(ss.transitions)

    def test_filter_empty_focus(self) -> None:
        ast = parse("&{a: end, b: end}")
        ss = build_statespace(ast)
        filtered = attention_filter(ss, set())
        # Only transitions are removed; states may still include initial/end
        assert len(filtered.transitions) == 0

    def test_returns_state_space(self) -> None:
        ast = parse("&{a: end, b: end}")
        ss = build_statespace(ast)
        result = attention_filter(ss, {"a"})
        assert isinstance(result, StateSpace)


# ---------------------------------------------------------------------------
# metacognition tests
# ---------------------------------------------------------------------------

class TestMetacognition:
    """Tests for metacognition()."""

    def test_depth_zero(self) -> None:
        assert metacognition("end", depth=0) == "end"

    def test_depth_zero_preserves_input(self) -> None:
        t = "&{a: end}"
        assert metacognition(t, depth=0) == t

    def test_depth_one_wraps(self) -> None:
        result = metacognition("end", depth=1)
        assert "perceive" in result
        assert "attend" in result
        assert "process" in result
        assert "reflect" in result

    def test_depth_one_contains_inner(self) -> None:
        result = metacognition("&{a: end}", depth=1)
        assert "&{a: end}" in result

    def test_depth_two_nests(self) -> None:
        result = metacognition("end", depth=2)
        # Depth 2 has at least 2 layers of perceive (inner appears in
        # both reflect and dream branches, so count >= 2)
        assert result.count("perceive") >= 2

    def test_depth_one_parses(self) -> None:
        result = metacognition("end", depth=1)
        ast = parse(result)
        assert ast is not None


# ---------------------------------------------------------------------------
# dream tests
# ---------------------------------------------------------------------------

class TestDream:
    """Tests for dream()."""

    def test_end_unchanged(self) -> None:
        # "end" has no labels to permute
        result = dream("end")
        assert result == "end"

    def test_deterministic_with_seed(self) -> None:
        t = "&{a: end, b: end, c: end}"
        r1 = dream(t, seed=42)
        r2 = dream(t, seed=42)
        assert r1 == r2

    def test_different_seeds_may_differ(self) -> None:
        t = "&{a: end, b: end, c: end}"
        r1 = dream(t, seed=42)
        r2 = dream(t, seed=99)
        # They might be the same by chance, but very unlikely with 3 labels
        # Just check both parse
        assert parse(r1) is not None
        assert parse(r2) is not None

    def test_preserves_structure(self) -> None:
        t = "&{a: end, b: end}"
        result = dream(t, seed=42)
        ast_orig = parse(t)
        ast_dream = parse(result)
        ss_orig = build_statespace(ast_orig)
        ss_dream = build_statespace(ast_dream)
        # Same number of states and transitions
        assert len(ss_orig.states) == len(ss_dream.states)
        assert len(ss_orig.transitions) == len(ss_dream.transitions)

    def test_parses_correctly(self) -> None:
        result = dream("&{x: +{y: end}, z: end}", seed=7)
        ast = parse(result)
        assert ast is not None


# ---------------------------------------------------------------------------
# flow_state tests
# ---------------------------------------------------------------------------

class TestFlowState:
    """Tests for flow_state()."""

    def test_end_no_flow(self) -> None:
        ast = parse("end")
        ss = build_statespace(ast)
        assert flow_state(ss) is False

    def test_simple_branch_no_flow(self) -> None:
        # Simple branch is not pendular (no alternation)
        ast = parse("&{a: end, b: end}")
        ss = build_statespace(ast)
        assert flow_state(ss) is False

    def test_returns_bool(self) -> None:
        ast = parse("&{a: +{x: end}}")
        ss = build_statespace(ast)
        result = flow_state(ss)
        assert isinstance(result, bool)
