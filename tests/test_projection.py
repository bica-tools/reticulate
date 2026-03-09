"""Tests for MPST projection (Step 12)."""

import pytest

from reticulate.global_types import (
    GEnd,
    GMessage,
    GRec,
    GVar,
    parse_global,
    roles,
)
from reticulate.parser import (
    Branch,
    End,
    Rec,
    Select,
    Var,
    parse,
    pretty,
)
from reticulate.projection import (
    ProjectionError,
    ProjectionResult,
    check_projection,
    project,
    project_all,
    verify_projection_morphism,
)
from reticulate.statespace import build_statespace


# ---------------------------------------------------------------------------
# Basic projection rules
# ---------------------------------------------------------------------------


class TestProjectionBasic:
    """Test the core projection rules."""

    def test_end_projects_to_end(self):
        assert project(GEnd(), "A") == End()

    def test_var_projects_to_var(self):
        assert project(GVar("X"), "A") == Var("X")

    def test_sender_gets_select(self):
        g = GMessage("A", "B", (("m", GEnd()),))
        local = project(g, "A")
        assert isinstance(local, Select)
        assert local.choices == (("m", End()),)

    def test_receiver_gets_branch(self):
        g = GMessage("A", "B", (("m", GEnd()),))
        local = project(g, "B")
        assert isinstance(local, Branch)
        assert local.choices == (("m", End()),)

    def test_uninvolved_merges(self):
        g = GMessage("A", "B", (("m", GEnd()),))
        local = project(g, "C")
        assert local == End()

    def test_sender_multiple_choices(self):
        g = GMessage("A", "B", (("m1", GEnd()), ("m2", GEnd())))
        local = project(g, "A")
        assert isinstance(local, Select)
        assert len(local.choices) == 2

    def test_receiver_multiple_choices(self):
        g = GMessage("A", "B", (("m1", GEnd()), ("m2", GEnd())))
        local = project(g, "B")
        assert isinstance(local, Branch)
        assert len(local.choices) == 2


# ---------------------------------------------------------------------------
# Uninvolved role merging
# ---------------------------------------------------------------------------


class TestMerge:
    """Test merging for uninvolved roles."""

    def test_identical_projections_merge(self):
        # A -> B : {m1: C -> D : {n: end}, m2: C -> D : {n: end}}
        # For role C: both branches project to +{n: end}, so merge succeeds
        g = parse_global(
            "A -> B : {m1: C -> D : {n: end}, m2: C -> D : {n: end}}"
        )
        local = project(g, "C")
        assert isinstance(local, Select)

    def test_different_projections_fail(self):
        # A -> B : {m1: C -> D : {n: end}, m2: D -> C : {x: end}}
        # For role C: m1 → +{n: end}, m2 → &{x: end} — different!
        g = parse_global(
            "A -> B : {m1: C -> D : {n: end}, m2: D -> C : {x: end}}"
        )
        with pytest.raises(ProjectionError):
            project(g, "C")

    def test_uninvolved_all_end(self):
        g = parse_global("A -> B : {m1: end, m2: end}")
        local = project(g, "C")
        assert local == End()


# ---------------------------------------------------------------------------
# Recursive projection
# ---------------------------------------------------------------------------


class TestRecursiveProjection:
    """Test projection of recursive global types."""

    def test_sender_recursive(self):
        g = parse_global("rec X . A -> B : {m: X, done: end}")
        local = project(g, "A")
        assert isinstance(local, Rec)
        assert isinstance(local.body, Select)

    def test_receiver_recursive(self):
        g = parse_global("rec X . A -> B : {m: X, done: end}")
        local = project(g, "B")
        assert isinstance(local, Rec)
        assert isinstance(local.body, Branch)

    def test_uninvolved_recursive_ends(self):
        g = parse_global("rec X . A -> B : {m: X, done: end}")
        local = project(g, "C")
        assert local == End()

    def test_negotiation(self):
        g = parse_global(
            "rec X . Buyer -> Seller : {offer: "
            "Seller -> Buyer : {accept: end, counter: X}}"
        )
        buyer = project(g, "Buyer")
        seller = project(g, "Seller")
        # Buyer: rec X . +{offer: &{accept: end, counter: X}}
        assert isinstance(buyer, Rec)
        assert isinstance(buyer.body, Select)
        # Seller: rec X . &{offer: +{accept: end, counter: X}}
        assert isinstance(seller, Rec)
        assert isinstance(seller.body, Branch)


# ---------------------------------------------------------------------------
# Multi-role projection
# ---------------------------------------------------------------------------


class TestProjectAll:
    """Test projecting onto all roles."""

    def test_simple_two_roles(self):
        g = parse_global("A -> B : {m: end}")
        result = project_all(g)
        assert set(result.keys()) == {"A", "B"}
        assert isinstance(result["A"], Select)
        assert isinstance(result["B"], Branch)

    def test_three_roles(self):
        g = parse_global(
            "A -> B : {m: B -> C : {n: end}}"
        )
        result = project_all(g)
        assert set(result.keys()) == {"A", "B", "C"}

    def test_two_buyer_all(self):
        g = parse_global(
            "Buyer1 -> Seller : {lookup: "
            "Seller -> Buyer1 : {price: "
            "Buyer1 -> Buyer2 : {share: "
            "Buyer2 -> Seller : {accept: end, reject: end}}}}"
        )
        result = project_all(g)
        assert set(result.keys()) == {"Buyer1", "Buyer2", "Seller"}


# ---------------------------------------------------------------------------
# Check projection result type
# ---------------------------------------------------------------------------


class TestCheckProjection:
    """Test the check_projection function."""

    def test_well_defined(self):
        g = parse_global("A -> B : {m: end}")
        result = check_projection(g, "A")
        assert isinstance(result, ProjectionResult)
        assert result.is_well_defined
        assert result.role == "A"
        assert "+" in result.local_type_str  # Select

    def test_undefined(self):
        g = parse_global(
            "A -> B : {m1: C -> D : {n: end}, m2: D -> C : {x: end}}"
        )
        result = check_projection(g, "C")
        assert not result.is_well_defined


# ---------------------------------------------------------------------------
# Projection and state spaces
# ---------------------------------------------------------------------------


class TestProjectionStateSpace:
    """Test that projected local types produce valid state spaces."""

    def test_local_type_builds_statespace(self):
        g = parse_global("A -> B : {m: end}")
        local = project(g, "A")
        ss = build_statespace(local)
        assert len(ss.states) >= 2

    def test_local_types_are_lattices(self):
        from reticulate.lattice import check_lattice

        g = parse_global(
            "Buyer1 -> Seller : {lookup: "
            "Seller -> Buyer1 : {price: "
            "Buyer1 -> Buyer2 : {share: "
            "Buyer2 -> Seller : {accept: end, reject: end}}}}"
        )
        for role in sorted(roles(g)):
            local = project(g, role)
            ss = build_statespace(local)
            result = check_lattice(ss)
            assert result.is_lattice, f"Role {role}: not a lattice"


# ---------------------------------------------------------------------------
# Morphism verification (Step 12 claim)
# ---------------------------------------------------------------------------


class TestProjectionMorphism:
    """Test the projection morphism properties."""

    def test_simple_morphism(self):
        g = parse_global("A -> B : {m: end}")
        result = verify_projection_morphism(g, "A")
        assert result.global_is_lattice
        assert result.local_is_lattice

    def test_order_preserving(self):
        g = parse_global("A -> B : {m: B -> A : {reply: end}}")
        result = verify_projection_morphism(g, "A")
        assert result.is_order_preserving

    def test_surjective(self):
        g = parse_global("A -> B : {m: end}")
        result = verify_projection_morphism(g, "A")
        assert result.is_surjective

    def test_two_buyer_morphism(self):
        g = parse_global(
            "Buyer1 -> Seller : {lookup: "
            "Seller -> Buyer1 : {price: "
            "Buyer1 -> Buyer2 : {share: "
            "Buyer2 -> Seller : {accept: end, reject: end}}}}"
        )
        for role in sorted(roles(g)):
            result = verify_projection_morphism(g, role)
            assert result.global_is_lattice, f"{role}: global not lattice"
            assert result.local_is_lattice, f"{role}: local not lattice"


# ---------------------------------------------------------------------------
# Benchmark verification
# ---------------------------------------------------------------------------


class TestProjectionBenchmarks:
    """Test projection on all multiparty benchmarks."""

    @pytest.fixture
    def benchmarks(self):
        from tests.benchmarks.multiparty_protocols import MULTIPARTY_BENCHMARKS
        return MULTIPARTY_BENCHMARKS

    def test_all_projections_defined(self, benchmarks):
        for b in benchmarks:
            g = parse_global(b.global_type_string)
            for role in sorted(b.expected_roles):
                result = check_projection(g, role)
                assert result.is_well_defined, (
                    f"{b.name}/{role}: projection not well-defined"
                )

    def test_all_local_types_are_lattices(self, benchmarks):
        from reticulate.lattice import check_lattice

        for b in benchmarks:
            g = parse_global(b.global_type_string)
            for role in sorted(b.expected_roles):
                local = project(g, role)
                ss = build_statespace(local)
                result = check_lattice(ss)
                assert result.is_lattice, (
                    f"{b.name}/{role}: local type not a lattice"
                )

    def test_projection_statistics(self, benchmarks):
        """Print projection statistics across all benchmarks."""
        total_roles = 0
        total_well_defined = 0
        total_surjective = 0
        total_order_preserving = 0

        for b in benchmarks:
            g = parse_global(b.global_type_string)
            for role in sorted(b.expected_roles):
                total_roles += 1
                result = check_projection(g, role)
                if result.is_well_defined:
                    total_well_defined += 1
                    morph = verify_projection_morphism(g, role)
                    if morph.is_surjective:
                        total_surjective += 1
                    if morph.is_order_preserving:
                        total_order_preserving += 1

        assert total_roles > 0
        print(f"\n{'='*60}")
        print(f"Step 12: Projection Statistics ({len(benchmarks)} benchmarks)")
        print(f"{'='*60}")
        print(f"Total role projections:    {total_roles}")
        print(f"Well-defined:             {total_well_defined}/{total_roles}")
        print(f"Surjective:               {total_surjective}/{total_well_defined}")
        print(f"Order-preserving:         {total_order_preserving}/{total_well_defined}")
        print(f"{'='*60}")
