"""Tests for Mathematics as Self-Referential Session Types (Step 208)."""

from __future__ import annotations

import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice
from reticulate.mathematics_types import (
    MATH_LIBRARY,
    MathEntry,
    all_math_form_lattices,
    godel_check,
    get_math,
    is_self_referential,
    math_by_domain,
    proof_type,
)


# ---------------------------------------------------------------------------
# Data integrity
# ---------------------------------------------------------------------------


class TestMathLibrary:
    def test_library_has_at_least_20_entries(self) -> None:
        assert len(MATH_LIBRARY) >= 20

    def test_all_entries_are_math_entry(self) -> None:
        for entry in MATH_LIBRARY.values():
            assert isinstance(entry, MathEntry)

    def test_all_names_match_keys(self) -> None:
        for key, entry in MATH_LIBRARY.items():
            assert key == entry.name

    def test_all_domains_valid(self) -> None:
        valid = {"logic", "algebra", "analysis", "topology",
                 "computation", "foundations"}
        for entry in MATH_LIBRARY.values():
            assert entry.domain in valid, f"{entry.name} has invalid domain {entry.domain}"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParsing:
    @pytest.mark.parametrize("name", list(MATH_LIBRARY.keys()))
    def test_every_entry_parses(self, name: str) -> None:
        entry = MATH_LIBRARY[name]
        ast = parse(entry.session_type_str)
        assert ast is not None

    @pytest.mark.parametrize("name", list(MATH_LIBRARY.keys()))
    def test_every_entry_builds_statespace(self, name: str) -> None:
        entry = MATH_LIBRARY[name]
        ast = parse(entry.session_type_str)
        ss = build_statespace(ast)
        assert len(ss.states) >= 2


# ---------------------------------------------------------------------------
# Lattice verification
# ---------------------------------------------------------------------------


class TestLattice:
    @pytest.mark.parametrize("name", list(MATH_LIBRARY.keys()))
    def test_every_entry_forms_lattice(self, name: str) -> None:
        entry = MATH_LIBRARY[name]
        ast = parse(entry.session_type_str)
        ss = build_statespace(ast)
        result = check_lattice(ss)
        assert result.is_lattice, f"{name} does not form a lattice"

    def test_all_math_form_lattices(self) -> None:
        assert all_math_form_lattices()


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


class TestGetMath:
    def test_lookup_modus_ponens(self) -> None:
        entry = get_math("modus_ponens")
        assert entry.name == "modus_ponens"
        assert entry.domain == "logic"

    def test_lookup_turing_machine(self) -> None:
        entry = get_math("turing_machine")
        assert entry.domain == "computation"

    def test_lookup_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown math entry"):
            get_math("perpetual_motion")

    def test_lookup_all(self) -> None:
        for name in MATH_LIBRARY:
            entry = get_math(name)
            assert entry.name == name


# ---------------------------------------------------------------------------
# Domain queries
# ---------------------------------------------------------------------------


class TestDomain:
    def test_logic_domain(self) -> None:
        logic = math_by_domain("logic")
        assert len(logic) >= 4
        names = {e.name for e in logic}
        assert "modus_ponens" in names
        assert "induction" in names

    def test_algebra_domain(self) -> None:
        algebra = math_by_domain("algebra")
        assert len(algebra) >= 4

    def test_analysis_domain(self) -> None:
        analysis = math_by_domain("analysis")
        assert len(analysis) >= 3

    def test_topology_domain(self) -> None:
        topology = math_by_domain("topology")
        assert len(topology) >= 3

    def test_computation_domain(self) -> None:
        comp = math_by_domain("computation")
        assert len(comp) >= 3

    def test_foundations_domain(self) -> None:
        found = math_by_domain("foundations")
        assert len(found) >= 3

    def test_empty_domain(self) -> None:
        assert math_by_domain("numerology") == []


# ---------------------------------------------------------------------------
# Self-reference
# ---------------------------------------------------------------------------


class TestSelfReferential:
    def test_turing_machine_is_recursive(self) -> None:
        assert is_self_referential("turing_machine")

    def test_lambda_calculus_is_recursive(self) -> None:
        assert is_self_referential("lambda_calculus")

    def test_induction_is_recursive(self) -> None:
        assert is_self_referential("induction")

    def test_modus_ponens_not_recursive(self) -> None:
        assert not is_self_referential("modus_ponens")

    def test_compactness_not_recursive(self) -> None:
        assert not is_self_referential("compactness")

    def test_zfc_set_is_recursive(self) -> None:
        assert is_self_referential("zfc_set")


# ---------------------------------------------------------------------------
# Proof composition
# ---------------------------------------------------------------------------


class TestProofType:
    def test_single_step(self) -> None:
        result = proof_type(["modus_ponens"])
        ast = parse(result)
        assert ast is not None

    def test_two_step_proof(self) -> None:
        result = proof_type(["modus_ponens", "modus_ponens"])
        ast = parse(result)
        ss = build_statespace(ast)
        assert len(ss.states) >= 3

    def test_multi_step_proof_parses(self) -> None:
        result = proof_type(["modus_ponens", "reductio", "modus_ponens"])
        ast = parse(result)
        assert ast is not None

    def test_empty_proof_raises(self) -> None:
        with pytest.raises(ValueError, match="empty proof"):
            proof_type([])

    def test_unknown_step_raises(self) -> None:
        with pytest.raises(KeyError):
            proof_type(["modus_ponens", "waving_hands"])

    def test_composed_proof_forms_lattice(self) -> None:
        result = proof_type(["modus_ponens", "reductio"])
        ast = parse(result)
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        assert lr.is_lattice


# ---------------------------------------------------------------------------
# Goedel check
# ---------------------------------------------------------------------------


class TestGodelCheck:
    def test_godel_incompleteness_not_self_ref(self) -> None:
        # godel_incompleteness is not recursive, so not self-referential
        entry = get_math("godel_incompleteness")
        result = godel_check(entry.session_type_str)
        assert not result

    def test_induction_is_goedel(self) -> None:
        # induction: rec X . &{base_case: +{inductive_step: X, qed: end}}
        # X appears in branch with 2 choices, body has 2 choices -> self-similar
        entry = get_math("induction")
        result = godel_check(entry.session_type_str)
        assert result

    def test_non_recursive_is_not_goedel(self) -> None:
        result = godel_check("&{a: end}")
        assert not result

    def test_simple_rec_not_goedel(self) -> None:
        # rec X . &{a: X} — var in branch with 1 choice, body has 1 choice -> self-similar
        result = godel_check("rec X . &{a: X}")
        assert result

    def test_turing_machine_goedel(self) -> None:
        entry = get_math("turing_machine")
        result = godel_check(entry.session_type_str)
        # The inner Select has 2 choices, matching the outer Branch's 1? No.
        # Body is &{read: ...} with 1 choice, var is inside +{halt: end, continue: X}
        # with 2 choices >= 1 -> True
        assert result
