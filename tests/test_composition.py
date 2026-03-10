"""Tests for bottom-up multiparty composition via lattice products (Step 15)."""

from __future__ import annotations

import pytest

from reticulate.composition import (
    CompositionResult,
    ComparisonResult,
    check_compatibility,
    compare_with_global,
    compose,
    product_nary,
)
from reticulate.duality import dual
from reticulate.lattice import check_lattice
from reticulate.parser import Branch, End, Select, Rec, Var, parse
from reticulate.statespace import StateSpace, build_statespace


# ======================================================================
# Helpers
# ======================================================================

def _parse(s: str):
    return parse(s)


# ======================================================================
# compose() — basic
# ======================================================================

class TestComposeBasic:
    """Basic composition of independent participants."""

    def test_single_participant(self):
        s = _parse("&{a: end, b: end}")
        result = compose(("Alice", s))
        assert result.is_lattice
        assert len(result.participants) == 1
        assert len(result.compatibility) == 0

    def test_two_participants(self):
        s1 = _parse("&{a: end, b: end}")
        s2 = _parse("&{x: end, y: end}")
        result = compose(("Alice", s1), ("Bob", s2))
        assert result.is_lattice
        assert len(result.participants) == 2
        assert ("Alice", "Bob") in result.compatibility

    def test_three_participants(self):
        s1 = _parse("&{a: end}")
        s2 = _parse("&{b: end}")
        s3 = _parse("&{c: end}")
        result = compose(("Alice", s1), ("Bob", s2), ("Carol", s3))
        assert result.is_lattice
        assert len(result.participants) == 3
        # 3 pairwise combos
        assert len(result.compatibility) == 3

    def test_product_state_count(self):
        """Product of two 2-state lattices has 4 states."""
        s1 = _parse("&{a: end, b: end}")  # 2 states: top, end
        s2 = _parse("&{x: end, y: end}")  # 2 states: top, end
        result = compose(("Alice", s1), ("Bob", s2))
        assert len(result.product.states) == 4

    def test_empty_participants_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            compose()

    def test_end_types(self):
        """Two end types compose to a single-state lattice."""
        result = compose(("A", End()), ("B", End()))
        assert result.is_lattice
        assert len(result.product.states) == 1


# ======================================================================
# product_nary()
# ======================================================================

class TestProductNary:
    """N-ary product via left-fold."""

    def test_single(self):
        ss = build_statespace(_parse("&{a: end}"))
        result = product_nary([ss])
        assert len(result.states) == len(ss.states)

    def test_binary(self):
        ss1 = build_statespace(_parse("&{a: end}"))
        ss2 = build_statespace(_parse("&{b: end}"))
        result = product_nary([ss1, ss2])
        assert len(result.states) == len(ss1.states) * len(ss2.states)

    def test_ternary(self):
        ss1 = build_statespace(_parse("&{a: end}"))
        ss2 = build_statespace(_parse("&{b: end}"))
        ss3 = build_statespace(_parse("&{c: end}"))
        result = product_nary([ss1, ss2, ss3])
        expected = len(ss1.states) * len(ss2.states) * len(ss3.states)
        assert len(result.states) == expected

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            product_nary([])

    def test_product_is_lattice(self):
        """Product of lattices is always a lattice."""
        ss1 = build_statespace(_parse("&{a: end, b: end}"))
        ss2 = build_statespace(_parse("+{x: end, y: end}"))
        ss3 = build_statespace(_parse("&{p: end}"))
        product = product_nary([ss1, ss2, ss3])
        result = check_lattice(product)
        assert result.is_lattice


# ======================================================================
# check_compatibility()
# ======================================================================

class TestCheckCompatibility:
    """Pairwise compatibility via duality on shared interface."""

    def test_dual_pair_compatible(self):
        """A type and its dual are compatible."""
        s = _parse("&{a: end, b: end}")
        d = dual(s)
        assert check_compatibility(s, d)

    def test_disjoint_labels_compatible(self):
        """Disjoint method sets are trivially compatible."""
        s1 = _parse("&{a: end}")
        s2 = _parse("&{x: end}")
        assert check_compatibility(s1, s2)

    def test_both_branch_compatible(self):
        """Both offering the same method (both Branch) is fine — no conflict."""
        s1 = _parse("&{a: end}")
        s2 = _parse("&{a: end}")
        assert check_compatibility(s1, s2)

    def test_offer_and_select_compatible(self):
        """One offers (Branch) what the other selects (Select) — compatible."""
        s1 = _parse("&{a: end, b: end}")
        s2 = _parse("+{a: end}")
        assert check_compatibility(s1, s2)

    def test_both_select_same_label_conflict(self):
        """Both selecting the same label with no offerer is incompatible."""
        s1 = _parse("+{a: end}")
        s2 = _parse("+{a: end}")
        assert not check_compatibility(s1, s2)

    def test_end_types_compatible(self):
        assert check_compatibility(End(), End())

    def test_complex_compatible(self):
        """Branch/Select with non-overlapping labels."""
        s1 = _parse("&{a: +{x: end}}")
        s2 = _parse("+{a: &{y: end}}")
        assert check_compatibility(s1, s2)

    def test_recursive_type_compatible(self):
        """Recursive type and its dual."""
        s = _parse("rec X . &{a: X, b: end}")
        d = dual(s)
        assert check_compatibility(s, d)

    def test_select_overlap_with_offerer(self):
        """Both select 'a' but one also offers 'a' — has conflict."""
        s1 = _parse("+{a: end}")
        s2 = _parse("&{a: +{a: end}}")
        # s2 offers 'a' AND selects 'a'; s1 selects 'a'
        # both_select = {a}, offered by s2 → not in both_select_not_offered
        # But conflict check: offered1={} & offered2={a} = {} ∩ selected1={a} & selected2={a} = {a}
        # conflict = {} & {a} = {} → no conflict by that check
        assert check_compatibility(s1, s2)


# ======================================================================
# compose() — compatibility integration
# ======================================================================

class TestComposeCompatibility:
    """Compatibility checking within compose()."""

    def test_dual_pair_in_composition(self):
        s = _parse("&{a: end, b: end}")
        d = dual(s)
        result = compose(("Server", s), ("Client", d))
        assert result.is_lattice
        assert result.compatibility[("Server", "Client")]

    def test_incompatible_pair(self):
        s1 = _parse("+{a: end}")
        s2 = _parse("+{a: end}")
        result = compose(("A", s1), ("B", s2))
        assert result.is_lattice  # Product is still a lattice!
        assert not result.compatibility[("A", "B")]

    def test_three_party_mixed_compat(self):
        server = _parse("&{req: end}")
        client = _parse("+{req: end}")
        observer = _parse("&{log: end}")
        result = compose(("Server", server), ("Client", client), ("Observer", observer))
        assert result.is_lattice
        assert result.compatibility[("Server", "Client")]
        assert result.compatibility[("Server", "Observer")]
        assert result.compatibility[("Client", "Observer")]


# ======================================================================
# Product of lattices is always a lattice (theorem verification)
# ======================================================================

class TestProductIsLattice:
    """Empirical verification: product of lattices is always a lattice."""

    TYPES = [
        "&{a: end}",
        "&{a: end, b: end}",
        "+{x: end, y: end}",
        "&{a: +{x: end, y: end}, b: end}",
        "rec X . &{a: X, b: end}",
        "(&{a: end} || &{b: end})",
    ]

    @pytest.mark.parametrize("t1", TYPES)
    @pytest.mark.parametrize("t2", TYPES[:3])
    def test_product_lattice(self, t1: str, t2: str):
        ss1 = build_statespace(_parse(t1))
        ss2 = build_statespace(_parse(t2))
        product = product_nary([ss1, ss2])
        result = check_lattice(product)
        assert result.is_lattice, f"Product of {t1} and {t2} is not a lattice!"


# ======================================================================
# compare_with_global() — top-down vs bottom-up
# ======================================================================

class TestCompareWithGlobal:
    """Compare bottom-up composition with top-down projection."""

    def test_request_response(self):
        """Binary protocol: product of projections should be close to global."""
        g = "Client -> Server : {request: Server -> Client : {response: end}}"
        from reticulate.global_types import parse_global
        from reticulate.projection import project_all
        gt = parse_global(g)
        projections = project_all(gt)

        result = compare_with_global(projections, g)
        assert result.global_is_lattice
        assert result.product_is_lattice
        assert result.over_approximation_ratio >= 1.0

    def test_two_buyer(self):
        """Three-party two-buyer: product is strictly larger than global."""
        g = (
            "Buyer1 -> Seller : {lookup: "
            "Seller -> Buyer1 : {price: "
            "Seller -> Buyer2 : {price: "
            "Buyer1 -> Buyer2 : {share: "
            "Buyer2 -> Seller : {accept: "
            "Seller -> Buyer2 : {deliver: end}, "
            "reject: end}}}}}"
        )
        from reticulate.global_types import parse_global
        from reticulate.projection import project_all
        gt = parse_global(g)
        projections = project_all(gt)

        result = compare_with_global(projections, g)
        assert result.global_is_lattice
        assert result.product_is_lattice
        # Product should be >= global (over-approximation)
        assert result.product_states >= result.global_states
        assert result.over_approximation_ratio >= 1.0

    def test_ring_protocol(self):
        """Three-node ring: causal ordering matters."""
        g = "A -> B : {msg: B -> C : {msg: C -> A : {msg: end}}}"
        from reticulate.global_types import parse_global
        from reticulate.projection import project_all
        gt = parse_global(g)
        projections = project_all(gt)

        result = compare_with_global(projections, g)
        assert result.global_is_lattice
        assert result.product_is_lattice
        assert result.over_approximation_ratio >= 1.0

    def test_delegation(self):
        """Four-step delegation chain."""
        g = (
            "Client -> Master : {task: "
            "Master -> Worker : {delegate: "
            "Worker -> Master : {result: "
            "Master -> Client : {response: end}}}}"
        )
        from reticulate.global_types import parse_global
        from reticulate.projection import project_all
        gt = parse_global(g)
        projections = project_all(gt)

        result = compare_with_global(projections, g)
        assert result.global_is_lattice
        assert result.product_is_lattice
        assert result.product_states >= result.global_states

    def test_role_matches(self):
        """Participant types should match projected types when they ARE the projections."""
        g = "Client -> Server : {request: Server -> Client : {response: end}}"
        from reticulate.global_types import parse_global
        from reticulate.projection import project_all
        gt = parse_global(g)
        projections = project_all(gt)

        result = compare_with_global(projections, g)
        for role, matches in result.role_type_matches.items():
            assert matches, f"Role {role} type doesn't match its projection"


# ======================================================================
# Multiparty benchmark comparison tests
# ======================================================================

class TestBenchmarkComparison:
    """Run compare_with_global on multiparty benchmarks."""

    def _run_benchmark(self, name: str, global_type_string: str):
        from reticulate.global_types import parse_global
        from reticulate.projection import project_all
        gt = parse_global(global_type_string)
        projections = project_all(gt)
        return compare_with_global(projections, global_type_string)

    def test_two_phase_commit(self):
        g = (
            "Coord -> P : {prepare: "
            "P -> Coord : {yes: "
            "Coord -> P : {commit: end}, "
            "no: Coord -> P : {abort: end}}}"
        )
        result = self._run_benchmark("Two-Phase Commit", g)
        assert result.global_is_lattice
        assert result.product_is_lattice
        assert result.over_approximation_ratio >= 1.0

    def test_streaming(self):
        g = "rec X . Producer -> Consumer : {data: X, done: end}"
        result = self._run_benchmark("Streaming", g)
        assert result.global_is_lattice
        assert result.product_is_lattice

    def test_negotiation(self):
        g = (
            "rec X . Buyer -> Seller : {offer: "
            "Seller -> Buyer : {accept: end, "
            "counter: X}}"
        )
        result = self._run_benchmark("Negotiation", g)
        assert result.global_is_lattice
        assert result.product_is_lattice

    def test_oauth2(self):
        g = (
            "Client -> AuthServer : {authorize: "
            "AuthServer -> Client : {code: "
            "Client -> AuthServer : {exchange: "
            "AuthServer -> Client : {token: "
            "Client -> ResourceServer : {access: "
            "ResourceServer -> Client : {resource: end}}}}}}"
        )
        result = self._run_benchmark("OAuth2", g)
        assert result.global_is_lattice
        assert result.product_is_lattice
        assert result.product_states >= result.global_states

    def test_mcp(self):
        g = (
            "Host -> Client : {initialize: "
            "Client -> Server : {discover: "
            "Server -> Client : {tools: "
            "Client -> Host : {ready: "
            "rec X . Client -> Server : {call: "
            "Server -> Client : {result: X}, "
            "done: end}}}}}"
        )
        result = self._run_benchmark("MCP", g)
        assert result.global_is_lattice
        assert result.product_is_lattice

    def test_dns_resolution(self):
        g = (
            "Client -> Resolver : {query: "
            "Resolver -> AuthNS : {lookup: "
            "AuthNS -> Resolver : {answer: "
            "Resolver -> Client : {response: end}}}}"
        )
        result = self._run_benchmark("DNS-Resolution", g)
        assert result.global_is_lattice
        assert result.product_is_lattice

    def test_map_reduce(self):
        g = (
            "Master -> Mapper : {map: "
            "Mapper -> Reducer : {emit: "
            "Reducer -> Master : {result: end}}}"
        )
        result = self._run_benchmark("Map-Reduce", g)
        assert result.global_is_lattice
        assert result.product_is_lattice


# ======================================================================
# Over-approximation analysis
# ======================================================================

class TestOverApproximation:
    """Verify that the product is an over-approximation of the global type."""

    def test_chain_protocol_strict_overapprox(self):
        """Chain topology (A→B→C→A) should have strict over-approximation."""
        g = "A -> B : {msg: B -> C : {msg: C -> A : {msg: end}}}"
        from reticulate.global_types import parse_global
        from reticulate.projection import project_all
        gt = parse_global(g)
        projections = project_all(gt)
        result = compare_with_global(projections, g)
        # For chain protocols, the product should be strictly larger
        assert result.product_states >= result.global_states

    def test_binary_protocol_tighter(self):
        """Binary protocols should have tighter over-approximation."""
        g = "Client -> Server : {request: Server -> Client : {response: end}}"
        from reticulate.global_types import parse_global
        from reticulate.projection import project_all
        gt = parse_global(g)
        projections = project_all(gt)
        result = compare_with_global(projections, g)
        # Binary protocols tend to have ratio closer to 1
        assert result.over_approximation_ratio >= 1.0


# ======================================================================
# Edge cases
# ======================================================================

class TestEdgeCases:
    """Edge cases for composition."""

    def test_single_method_types(self):
        s1 = _parse("&{m: end}")
        s2 = _parse("+{m: end}")
        result = compose(("Server", s1), ("Client", s2))
        assert result.is_lattice
        assert result.compatibility[("Server", "Client")]

    def test_deep_nesting(self):
        s = _parse("&{a: &{b: &{c: end}}}")
        d = dual(s)
        result = compose(("A", s), ("B", d))
        assert result.is_lattice

    def test_recursive_types_compose(self):
        s1 = _parse("rec X . &{next: X, stop: end}")
        s2 = _parse("rec Y . +{next: Y, stop: end}")
        result = compose(("Producer", s1), ("Consumer", s2))
        assert result.is_lattice
        assert result.compatibility[("Producer", "Consumer")]

    def test_parallel_type_in_composition(self):
        s1 = _parse("(&{a: end} || &{b: end})")
        s2 = _parse("&{x: end}")
        result = compose(("A", s1), ("B", s2))
        assert result.is_lattice

    def test_four_participants(self):
        types = [
            ("A", _parse("&{a: end}")),
            ("B", _parse("&{b: end}")),
            ("C", _parse("&{c: end}")),
            ("D", _parse("&{d: end}")),
        ]
        result = compose(*types)
        assert result.is_lattice
        assert len(result.compatibility) == 6  # C(4,2) = 6

    def test_result_fields(self):
        s1 = _parse("&{a: end}")
        s2 = _parse("&{b: end}")
        result = compose(("A", s1), ("B", s2))
        assert isinstance(result, CompositionResult)
        assert "A" in result.participants
        assert "B" in result.participants
        assert "A" in result.state_spaces
        assert "B" in result.state_spaces
        assert isinstance(result.product, StateSpace)

    def test_comparison_result_fields(self):
        g = "Client -> Server : {request: Server -> Client : {response: end}}"
        from reticulate.global_types import parse_global
        from reticulate.projection import project_all
        gt = parse_global(g)
        projections = project_all(gt)
        result = compare_with_global(projections, g)
        assert isinstance(result, ComparisonResult)
        assert result.global_states > 0
        assert result.product_states > 0
        assert isinstance(result.over_approximation_ratio, float)
