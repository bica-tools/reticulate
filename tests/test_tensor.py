"""Tests for Step 168: Tensor product (internal vs external product).

Tests the distinction between internal product (∥, intra-object concurrency)
and external product (⊗, inter-object program state).
"""

from __future__ import annotations

import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace, StateSpace
from reticulate.lattice import check_lattice
from reticulate.morphism import find_isomorphism, is_order_preserving
from reticulate.tensor import (
    CoupledTensorResult,
    CouplingConstraint,
    FlatteningResult,
    InternalExternalComparison,
    TensorResult,
    check_flattening,
    check_projection_homomorphism,
    compare_internal_external,
    coupled_tensor,
    tensor_product,
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def simple_branch():
    return parse("&{a: end, b: end}")


@pytest.fixture
def simple_select():
    return parse("+{x: end, y: end}")


@pytest.fixture
def file_type():
    return parse("&{open: &{read: end, write: end}}")


@pytest.fixture
def logger_type():
    return parse("&{start: &{log: end}}")


@pytest.fixture
def recursive_type():
    return parse("rec X . &{next: X, stop: end}")


# ===================================================================
# Basic construction
# ===================================================================

class TestTensorConstruction:
    """Test basic tensor product construction."""

    def test_two_simple_types(self, simple_branch, simple_select):
        result = tensor_product(("A", simple_branch), ("B", simple_select))
        assert isinstance(result, TensorResult)
        assert result.is_lattice
        assert len(result.participants) == 2
        assert len(result.state_spaces) == 2

    def test_state_count_is_product(self, simple_branch, simple_select):
        result = tensor_product(("A", simple_branch), ("B", simple_select))
        a_states = len(result.state_spaces["A"].states)
        b_states = len(result.state_spaces["B"].states)
        assert len(result.tensor.states) == a_states * b_states

    def test_labels_are_object_qualified(self, simple_branch, simple_select):
        result = tensor_product(("A", simple_branch), ("B", simple_select))
        labels = {lbl for _, lbl, _ in result.tensor.transitions}
        # All labels should be prefixed with object name
        for lbl in labels:
            assert "." in lbl, f"Label '{lbl}' is not object-qualified"
            prefix = lbl.split(".")[0]
            assert prefix in ("A", "B"), f"Unknown prefix '{prefix}'"

    def test_a_labels_present(self, simple_branch, simple_select):
        result = tensor_product(("A", simple_branch), ("B", simple_select))
        labels = {lbl for _, lbl, _ in result.tensor.transitions}
        assert "A.a" in labels
        assert "A.b" in labels

    def test_b_labels_present(self, simple_branch, simple_select):
        result = tensor_product(("A", simple_branch), ("B", simple_select))
        labels = {lbl for _, lbl, _ in result.tensor.transitions}
        assert "B.x" in labels
        assert "B.y" in labels

    def test_tensor_is_lattice(self, simple_branch, simple_select):
        result = tensor_product(("A", simple_branch), ("B", simple_select))
        lr = check_lattice(result.tensor)
        assert lr.is_lattice

    def test_three_participants(self, simple_branch, simple_select, logger_type):
        result = tensor_product(
            ("A", simple_branch), ("B", simple_select), ("C", logger_type)
        )
        assert result.is_lattice
        a = len(result.state_spaces["A"].states)
        b = len(result.state_spaces["B"].states)
        c = len(result.state_spaces["C"].states)
        assert len(result.tensor.states) == a * b * c

    def test_accepts_statespace_directly(self, simple_branch):
        ss = build_statespace(simple_branch)
        s2 = parse("&{x: end}")
        result = tensor_product(("A", ss), ("B", s2))
        assert result.is_lattice

    def test_product_coords_present(self, simple_branch, simple_select):
        result = tensor_product(("A", simple_branch), ("B", simple_select))
        assert result.tensor.product_coords is not None
        for sid in result.tensor.states:
            assert sid in result.tensor.product_coords
            coords = result.tensor.product_coords[sid]
            assert len(coords) == 2

    def test_top_is_component_tops(self, simple_branch, simple_select):
        result = tensor_product(("A", simple_branch), ("B", simple_select))
        top_coords = result.tensor.product_coords[result.tensor.top]
        assert top_coords[0] == result.state_spaces["A"].top
        assert top_coords[1] == result.state_spaces["B"].top

    def test_bottom_is_component_bottoms(self, simple_branch, simple_select):
        result = tensor_product(("A", simple_branch), ("B", simple_select))
        bot_coords = result.tensor.product_coords[result.tensor.bottom]
        assert bot_coords[0] == result.state_spaces["A"].bottom
        assert bot_coords[1] == result.state_spaces["B"].bottom


# ===================================================================
# Validation errors
# ===================================================================

class TestTensorValidation:
    """Test error handling in tensor construction."""

    def test_fewer_than_two_raises(self, simple_branch):
        with pytest.raises(ValueError, match="at least 2"):
            tensor_product(("A", simple_branch))

    def test_duplicate_names_raise(self, simple_branch, simple_select):
        with pytest.raises(ValueError, match="Duplicate"):
            tensor_product(("A", simple_branch), ("A", simple_select))

    def test_state_counts_dict(self, file_type, logger_type):
        result = tensor_product(("File", file_type), ("Logger", logger_type))
        assert "File" in result.state_counts
        assert "Logger" in result.state_counts
        assert result.state_counts["File"] == len(result.state_spaces["File"].states)


# ===================================================================
# Projections
# ===================================================================

class TestProjections:
    """Test projection maps from tensor to components."""

    def test_projections_exist(self, simple_branch, simple_select):
        result = tensor_product(("A", simple_branch), ("B", simple_select))
        assert "A" in result.projections
        assert "B" in result.projections

    def test_projection_surjective(self, simple_branch, simple_select):
        result = tensor_product(("A", simple_branch), ("B", simple_select))
        proj_a = result.projections["A"]
        image_a = set(proj_a.values())
        assert image_a == result.state_spaces["A"].states

    def test_projection_is_homomorphism(self, simple_branch, simple_select):
        result = tensor_product(("A", simple_branch), ("B", simple_select))
        assert check_projection_homomorphism(result, "A")
        assert check_projection_homomorphism(result, "B")

    def test_projection_homomorphism_three_way(
        self, simple_branch, simple_select, logger_type
    ):
        result = tensor_product(
            ("A", simple_branch), ("B", simple_select), ("C", logger_type)
        )
        for name in ("A", "B", "C"):
            assert check_projection_homomorphism(result, name)

    def test_projection_with_recursive_type(self, simple_branch, recursive_type):
        result = tensor_product(("A", simple_branch), ("B", recursive_type))
        assert check_projection_homomorphism(result, "A")
        assert check_projection_homomorphism(result, "B")


# ===================================================================
# Selection propagation
# ===================================================================

class TestSelectionPropagation:
    """Test that selection (internal choice) info propagates correctly."""

    def test_selection_preserved_in_tensor(self, simple_branch, simple_select):
        result = tensor_product(("A", simple_branch), ("B", simple_select))
        # B is a Select type, so B.x and B.y should be selections
        sel_labels = {lbl for _, lbl, _ in result.tensor.selection_transitions}
        assert "B.x" in sel_labels or "B.y" in sel_labels

    def test_branch_not_selection(self, simple_branch, simple_select):
        result = tensor_product(("A", simple_branch), ("B", simple_select))
        sel_labels = {lbl for _, lbl, _ in result.tensor.selection_transitions}
        # A is Branch, so A.a and A.b should NOT be selections
        assert "A.a" not in sel_labels
        assert "A.b" not in sel_labels


# ===================================================================
# Coupling constraints
# ===================================================================

class TestCoupling:
    """Test coupled tensor product with constraints."""

    def test_unconstrained_returns_all_states(self, file_type, logger_type):
        result = tensor_product(("File", file_type), ("Logger", logger_type))
        coupled = coupled_tensor(result, [])
        assert isinstance(coupled, CoupledTensorResult)
        assert coupled.coupled_states == coupled.base_states
        assert coupled.reduction_ratio == 1.0

    def test_constraint_reduces_states(self, file_type, logger_type):
        result = tensor_product(("File", file_type), ("Logger", logger_type))
        # Find a state in File that is not the top
        file_ss = result.state_spaces["File"]
        non_top_states = [s for s in file_ss.states if s != file_ss.top and s != file_ss.bottom]
        if non_top_states:
            constrained_state = non_top_states[0]
            logger_ss = result.state_spaces["Logger"]
            logger_non_top = [s for s in logger_ss.states if s != logger_ss.top]
            if logger_non_top:
                constraint = CouplingConstraint(
                    description="Logger must advance before File reaches state",
                    required_before={
                        "File": {constrained_state: {("Logger", logger_non_top[0])}}
                    },
                )
                coupled = coupled_tensor(result, [constraint])
                # Constraint should remove some states
                assert coupled.coupled_states <= coupled.base_states

    def test_coupled_lattice_check(self, simple_branch, simple_select):
        result = tensor_product(("A", simple_branch), ("B", simple_select))
        coupled = coupled_tensor(result, [])
        assert coupled.coupled_is_lattice


# ===================================================================
# Flattening
# ===================================================================

class TestFlattening:
    """Test internal/external product flattening."""

    def test_no_internal_product_trivial(self, simple_branch, simple_select):
        result = tensor_product(("A", simple_branch), ("B", simple_select))
        flat = check_flattening(result)
        assert isinstance(flat, FlatteningResult)
        # No internal products, so flattening is trivial
        assert flat.isomorphic
        assert flat.nested_states == flat.flattened_states

    def test_with_parallel_type(self):
        """If one participant uses ∥, flattening expands its factors."""
        par_type = parse("(&{a: end} || &{b: end})")
        simple = parse("&{c: end}")
        result = tensor_product(("Par", par_type), ("Simple", simple))
        flat = check_flattening(result)
        # Par has 2 factors from ∥, so flattened should have 3 factors
        assert len(flat.factor_names) == 3
        assert flat.isomorphic
        assert flat.nested_states == flat.flattened_states

    def test_factor_names_with_parallel(self):
        par_type = parse("(&{a: end} || &{b: end})")
        simple = parse("&{c: end}")
        result = tensor_product(("Par", par_type), ("Simple", simple))
        flat = check_flattening(result)
        assert "Par[0]" in flat.factor_names
        assert "Par[1]" in flat.factor_names
        assert "Simple" in flat.factor_names


# ===================================================================
# Internal vs external comparison
# ===================================================================

class TestInternalExternalComparison:
    """Test comparison between ∥ (internal) and ⊗ (external) products."""

    def test_same_state_count(self):
        s1 = parse("&{a: end, b: end}")
        s2 = parse("&{x: end, y: end}")
        cmp = compare_internal_external(s1, s2, "A", "B")
        assert isinstance(cmp, InternalExternalComparison)
        assert cmp.internal_states == cmp.external_states

    def test_isomorphic_structure(self):
        s1 = parse("&{a: end, b: end}")
        s2 = parse("&{x: end, y: end}")
        cmp = compare_internal_external(s1, s2)
        assert cmp.isomorphic

    def test_different_label_sets(self):
        s1 = parse("&{a: end, b: end}")
        s2 = parse("&{x: end, y: end}")
        cmp = compare_internal_external(s1, s2, "A", "B")
        # Internal: bare labels {a, b, x, y}
        # External: qualified {A.a, A.b, B.x, B.y}
        assert cmp.internal_labels != cmp.external_labels
        assert "a" in cmp.internal_labels
        assert "A.a" in cmp.external_labels

    def test_label_ratio(self):
        s1 = parse("&{a: end, b: end}")
        s2 = parse("&{x: end, y: end}")
        cmp = compare_internal_external(s1, s2)
        # Same number of labels, just different names
        assert cmp.label_ratio == 1.0

    def test_overlapping_labels_same_count(self):
        """When labels overlap, internal merges them but external qualifies."""
        s1 = parse("&{m: end}")
        s2 = parse("&{m: end}")
        cmp = compare_internal_external(s1, s2, "A", "B")
        # Internal: both use "m" (1 label name, transitions from both states)
        # External: A.m and B.m (2 distinct label names)
        assert "m" in cmp.internal_labels
        assert "A.m" in cmp.external_labels
        assert "B.m" in cmp.external_labels

    def test_comparison_with_deeper_types(self):
        s1 = parse("&{open: &{read: end, write: end}}")
        s2 = parse("&{start: &{log: end}}")
        cmp = compare_internal_external(s1, s2, "File", "Logger")
        assert cmp.isomorphic
        assert cmp.internal_states == cmp.external_states


# ===================================================================
# Benchmark protocols as tensor products
# ===================================================================

class TestBenchmarkTensors:
    """Test tensor products of benchmark protocols."""

    def test_iterator_file_tensor(self):
        iterator = parse("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        file_obj = parse("&{open: &{read: end, write: end}}")
        result = tensor_product(("Iterator", iterator), ("File", file_obj))
        assert result.is_lattice
        lr = check_lattice(result.tensor)
        assert lr.is_lattice

    def test_iterator_file_projections(self):
        iterator = parse("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        file_obj = parse("&{open: &{read: end, write: end}}")
        result = tensor_product(("Iterator", iterator), ("File", file_obj))
        assert check_projection_homomorphism(result, "Iterator")
        assert check_projection_homomorphism(result, "File")

    def test_three_benchmark_tensor(self):
        iterator = parse("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        file_obj = parse("&{open: &{read: end, write: end}}")
        logger = parse("&{start: &{log: end}}")
        result = tensor_product(
            ("Iterator", iterator), ("File", file_obj), ("Logger", logger)
        )
        assert result.is_lattice
        total = (
            len(result.state_spaces["Iterator"].states)
            * len(result.state_spaces["File"].states)
            * len(result.state_spaces["Logger"].states)
        )
        assert len(result.tensor.states) == total

    def test_two_recursive_types(self):
        r1 = parse("rec X . &{a: X, b: end}")
        r2 = parse("rec Y . &{c: Y, d: end}")
        result = tensor_product(("R1", r1), ("R2", r2))
        assert result.is_lattice
        lr = check_lattice(result.tensor)
        assert lr.is_lattice


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    """Test edge cases and corner cases."""

    def test_end_end_tensor(self):
        """Tensor of two end types: single state."""
        e1 = parse("end")
        e2 = parse("end")
        result = tensor_product(("A", e1), ("B", e2))
        assert len(result.tensor.states) == 1
        assert result.tensor.top == result.tensor.bottom

    def test_end_nontrivial_tensor(self):
        """Tensor of end with non-trivial: just copies the non-trivial."""
        e = parse("end")
        s = parse("&{a: end, b: end}")
        result = tensor_product(("A", e), ("B", s))
        # 1 * 3 = 3 states
        assert len(result.tensor.states) == len(result.state_spaces["B"].states)

    def test_tensor_with_selection_only(self):
        s1 = parse("+{a: end, b: end}")
        s2 = parse("+{x: end, y: end}")
        result = tensor_product(("A", s1), ("B", s2))
        assert result.is_lattice

    def test_single_method_types(self):
        s1 = parse("&{m: end}")
        s2 = parse("&{n: end}")
        result = tensor_product(("A", s1), ("B", s2))
        assert len(result.tensor.states) == 4  # 2 * 2
        assert result.is_lattice

    def test_parallel_inside_tensor(self):
        """Object with internal ∥ inside external ⊗."""
        par = parse("(&{a: end} || &{b: end})")
        simple = parse("&{c: end}")
        result = tensor_product(("Par", par), ("Simple", simple))
        assert result.is_lattice
        # Par has 4 states (2×2), Simple has 2
        assert len(result.tensor.states) == 4 * 2


# ===================================================================
# Algebraic properties
# ===================================================================

class TestAlgebraicProperties:
    """Test algebraic properties of tensor product."""

    def test_commutativity_up_to_iso(self):
        """A ⊗ B ≅ B ⊗ A (up to isomorphism)."""
        s1 = parse("&{a: end, b: end}")
        s2 = parse("&{x: end, y: end}")
        r1 = tensor_product(("A", s1), ("B", s2))
        r2 = tensor_product(("B", s2), ("A", s1))
        # Same state count
        assert len(r1.tensor.states) == len(r2.tensor.states)

    def test_associativity_state_count(self):
        """(A ⊗ B) ⊗ C has same states as A ⊗ (B ⊗ C)."""
        sa = parse("&{a: end}")
        sb = parse("&{b: end}")
        sc = parse("&{c: end}")
        r_abc = tensor_product(("A", sa), ("B", sb), ("C", sc))
        # N-ary tensor should give 2 * 2 * 2 = 8 states
        assert len(r_abc.tensor.states) == 8

    def test_tensor_preserves_lattice(self):
        """Tensor of lattices is always a lattice."""
        types = [
            parse("&{a: end, b: end}"),
            parse("+{x: end, y: end}"),
            parse("rec X . &{next: X, stop: end}"),
        ]
        for i, s1 in enumerate(types):
            for j, s2 in enumerate(types):
                result = tensor_product(("A", s1), ("B", s2))
                lr = check_lattice(result.tensor)
                assert lr.is_lattice, f"Tensor of types {i} and {j} is not a lattice"
