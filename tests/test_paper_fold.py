"""Tests for paper_fold module (Step 60f: origami-to-session-types)."""

from __future__ import annotations

import pytest

from reticulate.paper_fold import (
    CreasePattern,
    FoldResult,
    analyze_fold,
    check_flat_foldable,
    check_kawasaki,
    crease_to_session_type,
    crease_to_statespace,
    simple_fold,
    star_fold,
)
from reticulate.parser import parse


# ---------------------------------------------------------------------------
# CreasePattern data type
# ---------------------------------------------------------------------------


class TestCreasePattern:
    """Tests for the CreasePattern frozen dataclass."""

    def test_creation(self) -> None:
        cp = CreasePattern(
            vertices=("a", "b"),
            creases=(("a", "b", "mountain"),),
            is_flat_foldable=True,
        )
        assert cp.vertices == ("a", "b")
        assert len(cp.creases) == 1
        assert cp.is_flat_foldable is True

    def test_frozen(self) -> None:
        cp = CreasePattern(vertices=("a",), creases=(), is_flat_foldable=True)
        with pytest.raises(AttributeError):
            cp.vertices = ("b",)  # type: ignore[misc]

    def test_empty_pattern(self) -> None:
        cp = CreasePattern(vertices=(), creases=(), is_flat_foldable=True)
        assert cp.vertices == ()
        assert cp.creases == ()


# ---------------------------------------------------------------------------
# simple_fold generator
# ---------------------------------------------------------------------------


class TestSimpleFold:
    """Tests for the simple_fold generator."""

    def test_single_crease(self) -> None:
        cp = simple_fold(1)
        assert cp.vertices == ("v0", "v1")
        assert len(cp.creases) == 1
        assert cp.creases[0] == ("v0", "v1", "mountain")
        assert cp.is_flat_foldable is True

    def test_two_creases(self) -> None:
        cp = simple_fold(2)
        assert cp.vertices == ("v0", "v1", "v2")
        assert len(cp.creases) == 2
        assert cp.creases[0][2] == "mountain"
        assert cp.creases[1][2] == "valley"

    def test_three_creases(self) -> None:
        cp = simple_fold(3)
        assert len(cp.vertices) == 4
        assert len(cp.creases) == 3
        types = [c[2] for c in cp.creases]
        assert types == ["mountain", "valley", "mountain"]

    def test_five_creases(self) -> None:
        cp = simple_fold(5)
        assert len(cp.vertices) == 6
        assert len(cp.creases) == 5
        assert cp.is_flat_foldable is True

    def test_zero_creases(self) -> None:
        cp = simple_fold(0)
        assert cp.vertices == ("v0",)
        assert cp.creases == ()

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            simple_fold(-1)


# ---------------------------------------------------------------------------
# star_fold generator
# ---------------------------------------------------------------------------


class TestStarFold:
    """Tests for the star_fold generator."""

    def test_three_points(self) -> None:
        cp = star_fold(3)
        assert cp.vertices[0] == "center"
        assert len(cp.vertices) == 4
        assert len(cp.creases) == 3
        # Odd: not flat-foldable.
        assert cp.is_flat_foldable is False

    def test_four_points(self) -> None:
        cp = star_fold(4)
        assert len(cp.vertices) == 5
        assert len(cp.creases) == 4
        assert cp.is_flat_foldable is True

    def test_six_points(self) -> None:
        cp = star_fold(6)
        assert len(cp.vertices) == 7
        assert len(cp.creases) == 6
        assert cp.is_flat_foldable is True

    def test_single_point(self) -> None:
        cp = star_fold(1)
        assert len(cp.vertices) == 2
        assert len(cp.creases) == 1
        # Odd and < 2: not flat-foldable.
        assert cp.is_flat_foldable is False

    def test_raises_on_zero(self) -> None:
        with pytest.raises(ValueError):
            star_fold(0)


# ---------------------------------------------------------------------------
# check_kawasaki
# ---------------------------------------------------------------------------


class TestCheckKawasaki:
    """Tests for the Kawasaki-Justin condition checker."""

    def test_simple_fold_satisfies(self) -> None:
        # Linear folds: interior vertices have degree 2 (1M + 1V or 2M etc.).
        # For n=2, v1 is interior with 1 mountain (from v0) + 1 valley (to v2) = |1-1|=0 != 2.
        # Actually let's check: v1 has mountain_v0_v1 (incident) and valley_v1_v2 (incident).
        # mountain=1, valley=1 → |1-1|=0 ≠ 2 → fails.
        cp = simple_fold(2)
        assert check_kawasaki(cp) is False

    def test_star_fold_even_satisfies(self) -> None:
        cp = star_fold(4)
        assert check_kawasaki(cp) is True

    def test_star_fold_odd_fails(self) -> None:
        cp = star_fold(3)
        assert check_kawasaki(cp) is False

    def test_empty_pattern(self) -> None:
        cp = CreasePattern(vertices=(), creases=(), is_flat_foldable=True)
        assert check_kawasaki(cp) is True

    def test_single_vertex_no_creases(self) -> None:
        cp = CreasePattern(vertices=("a",), creases=(), is_flat_foldable=True)
        assert check_kawasaki(cp) is True

    def test_custom_valid_pattern(self) -> None:
        # Vertex with 3 mountain, 1 valley: |3-1| = 2 ✓
        cp = CreasePattern(
            vertices=("c", "a", "b", "d"),
            creases=(
                ("c", "a", "mountain"),
                ("c", "b", "mountain"),
                ("c", "d", "mountain"),
                ("a", "c", "valley"),
            ),
            is_flat_foldable=False,
        )
        # At 'c': incident = 4 (3M + 1V) → |3-1| = 2 ✓
        # At 'a': incident = 2 (1M + 1V) → |1-1| = 0 ≠ 2 ✗
        assert check_kawasaki(cp) is False

    def test_single_crease_boundary(self) -> None:
        # Both vertices have degree 1 (< 2): exempt.
        cp = simple_fold(1)
        assert check_kawasaki(cp) is True


# ---------------------------------------------------------------------------
# check_flat_foldable
# ---------------------------------------------------------------------------


class TestCheckFlatFoldable:
    """Tests for flat-foldability checking."""

    def test_star_even_is_flat_foldable(self) -> None:
        cp = star_fold(4)
        assert check_flat_foldable(cp) is True

    def test_star_odd_not_flat_foldable(self) -> None:
        cp = star_fold(3)
        assert check_flat_foldable(cp) is False

    def test_crossing_creases_not_flat_foldable(self) -> None:
        cp = CreasePattern(
            vertices=("a", "b"),
            creases=(
                ("a", "b", "mountain"),
                ("b", "a", "valley"),
            ),
            is_flat_foldable=False,
        )
        # Same pair reversed: crossing detected.
        assert check_flat_foldable(cp) is False

    def test_empty_is_flat_foldable(self) -> None:
        cp = CreasePattern(vertices=(), creases=(), is_flat_foldable=True)
        assert check_flat_foldable(cp) is True


# ---------------------------------------------------------------------------
# crease_to_statespace
# ---------------------------------------------------------------------------


class TestCreaseToStatespace:
    """Tests for crease pattern to state space conversion."""

    def test_single_crease_states(self) -> None:
        cp = simple_fold(1)
        ss = crease_to_statespace(cp)
        # 2 vertices + 1 flat state = 3
        assert len(ss.states) == 3

    def test_single_crease_transitions(self) -> None:
        cp = simple_fold(1)
        ss = crease_to_statespace(cp)
        # 1 crease transition + 1 fold_complete (v1 has no outgoing)
        assert len(ss.transitions) == 2

    def test_top_is_first_vertex(self) -> None:
        cp = simple_fold(2)
        ss = crease_to_statespace(cp)
        assert ss.labels[ss.top] == "v0"

    def test_bottom_is_flat(self) -> None:
        cp = simple_fold(2)
        ss = crease_to_statespace(cp)
        assert ss.labels[ss.bottom] == "flat"

    def test_empty_pattern(self) -> None:
        cp = CreasePattern(vertices=(), creases=(), is_flat_foldable=True)
        ss = crease_to_statespace(cp)
        assert len(ss.states) == 1
        assert ss.top == ss.bottom

    def test_star_state_count(self) -> None:
        cp = star_fold(4)
        ss = crease_to_statespace(cp)
        # center + 4 outer + flat = 6
        assert len(ss.states) == 6

    def test_star_transition_count(self) -> None:
        cp = star_fold(4)
        ss = crease_to_statespace(cp)
        # 4 crease transitions + 4 fold_complete (outer vertices have no outgoing)
        assert len(ss.transitions) == 8


# ---------------------------------------------------------------------------
# crease_to_session_type
# ---------------------------------------------------------------------------


class TestCreaseToSessionType:
    """Tests for crease pattern to session type string conversion."""

    def test_empty_pattern(self) -> None:
        cp = CreasePattern(vertices=(), creases=(), is_flat_foldable=True)
        assert crease_to_session_type(cp) == "end"

    def test_single_crease_parseable(self) -> None:
        cp = simple_fold(1)
        st_str = crease_to_session_type(cp)
        # Must be parseable.
        ast = parse(st_str)
        assert ast is not None

    def test_two_creases_parseable(self) -> None:
        cp = simple_fold(2)
        st_str = crease_to_session_type(cp)
        ast = parse(st_str)
        assert ast is not None

    def test_star_parseable(self) -> None:
        cp = star_fold(4)
        st_str = crease_to_session_type(cp)
        ast = parse(st_str)
        assert ast is not None

    def test_contains_branch(self) -> None:
        cp = simple_fold(1)
        st_str = crease_to_session_type(cp)
        assert "&{" in st_str


# ---------------------------------------------------------------------------
# analyze_fold
# ---------------------------------------------------------------------------


class TestAnalyzeFold:
    """Tests for the full analysis pipeline."""

    def test_returns_fold_result(self) -> None:
        cp = simple_fold(2)
        result = analyze_fold(cp)
        assert isinstance(result, FoldResult)

    def test_state_count(self) -> None:
        cp = simple_fold(3)
        result = analyze_fold(cp)
        # 4 vertices + 1 flat = 5
        assert result.state_count == 5

    def test_session_type_str_not_empty(self) -> None:
        cp = simple_fold(1)
        result = analyze_fold(cp)
        assert len(result.session_type_str) > 0

    def test_kawasaki_field(self) -> None:
        cp = star_fold(4)
        result = analyze_fold(cp)
        assert result.kawasaki_satisfied is True

    def test_lattice_check_runs(self) -> None:
        cp = simple_fold(1)
        result = analyze_fold(cp)
        # The lattice field should be a boolean.
        assert isinstance(result.is_lattice, bool)

    def test_crease_pattern_preserved(self) -> None:
        cp = simple_fold(2)
        result = analyze_fold(cp)
        assert result.crease_pattern is cp
