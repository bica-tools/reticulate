"""Tests for monoidal.py — Monoidal structure of session type composition (Step 167)."""

from __future__ import annotations

import pytest

from reticulate.parser import parse
from reticulate.statespace import StateSpace, build_statespace
from reticulate.product import product_statespace
from reticulate.monoidal import (
    AssociativityResult,
    CoherenceResult,
    MonoidalResult,
    SymmetryResult,
    UnitResult,
    check_associativity,
    check_coherence,
    check_monoidal_structure,
    check_monoidal_unit,
    check_symmetry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def end_ss() -> StateSpace:
    return build_statespace(parse("end"))


@pytest.fixture
def chain2() -> StateSpace:
    """&{a: end} — 2-state chain."""
    return build_statespace(parse("&{a: end}"))


@pytest.fixture
def chain3() -> StateSpace:
    """&{a: &{b: end}} — 3-state chain."""
    return build_statespace(parse("&{a: &{b: end}}"))


@pytest.fixture
def branch2() -> StateSpace:
    """&{a: end, b: end} — 3-state diamond-ish."""
    return build_statespace(parse("&{a: end, b: end}"))


@pytest.fixture
def select2() -> StateSpace:
    """+{x: end, y: end} — selection with 2 labels."""
    return build_statespace(parse("+{x: end, y: end}"))


# ---------------------------------------------------------------------------
# Unit tests: End is the monoidal unit
# ---------------------------------------------------------------------------

class TestMonoidalUnit:
    """Verify End is the monoidal unit for parallel composition."""

    def test_unit_end(self, end_ss: StateSpace) -> None:
        """End || End ~ End."""
        result = check_monoidal_unit(end_ss)
        assert result.left_unit
        assert result.right_unit

    def test_unit_chain2(self, chain2: StateSpace) -> None:
        """End || &{a: end} ~ &{a: end} ~ &{a: end} || End."""
        result = check_monoidal_unit(chain2)
        assert result.left_unit, "left unit law failed"
        assert result.right_unit, "right unit law failed"
        assert result.left_iso is not None
        assert result.right_iso is not None

    def test_unit_chain3(self, chain3: StateSpace) -> None:
        """End || &{a: &{b: end}} ~ &{a: &{b: end}}."""
        result = check_monoidal_unit(chain3)
        assert result.left_unit
        assert result.right_unit

    def test_unit_branch(self, branch2: StateSpace) -> None:
        """End || &{a: end, b: end} ~ &{a: end, b: end}."""
        result = check_monoidal_unit(branch2)
        assert result.left_unit
        assert result.right_unit

    def test_unit_select(self, select2: StateSpace) -> None:
        """End || +{x: end, y: end} ~ +{x: end, y: end}."""
        result = check_monoidal_unit(select2)
        assert result.left_unit
        assert result.right_unit

    def test_unit_iso_is_bijection(self, chain2: StateSpace) -> None:
        """The unit isomorphism is a bijection."""
        result = check_monoidal_unit(chain2)
        assert result.left_iso is not None
        # Check bijection: all target values are distinct and cover ss states
        iso = result.left_iso
        assert len(set(iso.values())) == len(iso)


# ---------------------------------------------------------------------------
# Associativity tests
# ---------------------------------------------------------------------------

class TestAssociativity:
    """Verify (S1||S2)||S3 ~ S1||(S2||S3)."""

    def test_assoc_three_ends(self, end_ss: StateSpace) -> None:
        """(End||End)||End ~ End||(End||End)."""
        result = check_associativity(end_ss, end_ss, end_ss)
        assert result.is_associative
        assert result.left_states == result.right_states == 1

    def test_assoc_chains(self, chain2: StateSpace) -> None:
        """Three 2-state chains."""
        result = check_associativity(chain2, chain2, chain2)
        assert result.is_associative
        assert result.left_states == result.right_states == 8

    def test_assoc_mixed(self, chain2: StateSpace, chain3: StateSpace, branch2: StateSpace) -> None:
        """Mixed types: chain2 || chain3 || branch2."""
        result = check_associativity(chain2, chain3, branch2)
        assert result.is_associative
        assert result.left_states == result.right_states

    def test_assoc_state_counts_match(self, chain2: StateSpace, branch2: StateSpace) -> None:
        """State counts always match for associativity."""
        end = build_statespace(parse("end"))
        result = check_associativity(chain2, branch2, end)
        assert result.left_states == result.right_states

    def test_assoc_iso_exists(self, chain2: StateSpace, chain3: StateSpace) -> None:
        """Isomorphism mapping is not None when associative."""
        end = build_statespace(parse("end"))
        result = check_associativity(chain2, chain3, end)
        assert result.is_associative
        assert result.iso is not None


# ---------------------------------------------------------------------------
# Symmetry (braiding) tests
# ---------------------------------------------------------------------------

class TestSymmetry:
    """Verify S1||S2 ~ S2||S1."""

    def test_symmetry_same_type(self, chain2: StateSpace) -> None:
        """&{a: end} || &{a: end} ~ &{a: end} || &{a: end} (trivially)."""
        result = check_symmetry(chain2, chain2)
        assert result.is_symmetric
        assert result.is_involution

    def test_symmetry_different_types(self, chain2: StateSpace, chain3: StateSpace) -> None:
        """chain2 || chain3 ~ chain3 || chain2."""
        result = check_symmetry(chain2, chain3)
        assert result.is_symmetric

    def test_symmetry_with_end(self, chain2: StateSpace, end_ss: StateSpace) -> None:
        """&{a: end} || end ~ end || &{a: end}."""
        result = check_symmetry(chain2, end_ss)
        assert result.is_symmetric
        assert result.is_involution

    def test_symmetry_involution(self, chain2: StateSpace, branch2: StateSpace) -> None:
        """sigma_{B,A} . sigma_{A,B} = id."""
        result = check_symmetry(chain2, branch2)
        assert result.is_symmetric
        assert result.is_involution

    def test_symmetry_end_end(self, end_ss: StateSpace) -> None:
        """End || End ~ End || End (trivial)."""
        result = check_symmetry(end_ss, end_ss)
        assert result.is_symmetric
        assert result.is_involution


# ---------------------------------------------------------------------------
# Coherence tests
# ---------------------------------------------------------------------------

class TestCoherence:
    """Verify Mac Lane coherence conditions."""

    def test_coherence_chains(self, chain2: StateSpace) -> None:
        """Pentagon, triangle, hexagon all hold for chain2."""
        result = check_coherence(chain2, chain2, chain2)
        assert result.pentagon, f"Pentagon failed: {result.counterexample}"
        assert result.triangle, f"Triangle failed: {result.counterexample}"
        assert result.hexagon, f"Hexagon failed: {result.counterexample}"
        assert result.counterexample is None

    def test_coherence_with_end(self, end_ss: StateSpace) -> None:
        """Coherence with the trivial type."""
        result = check_coherence(end_ss, end_ss, end_ss)
        assert result.pentagon
        assert result.triangle
        assert result.hexagon

    def test_coherence_mixed(self, chain2: StateSpace, chain3: StateSpace, branch2: StateSpace) -> None:
        """Mixed types satisfy coherence."""
        result = check_coherence(chain2, chain3, branch2)
        assert result.pentagon
        assert result.triangle
        assert result.hexagon


# ---------------------------------------------------------------------------
# Full monoidal structure
# ---------------------------------------------------------------------------

class TestMonoidalStructure:
    """Full symmetric monoidal category verification."""

    def test_monoidal_chains(self, chain2: StateSpace) -> None:
        """chain2 x3 forms a symmetric monoidal category."""
        result = check_monoidal_structure(chain2, chain2, chain2)
        assert result.is_monoidal
        assert result.unit.left_unit
        assert result.unit.right_unit
        assert result.associativity.is_associative
        assert result.symmetry.is_symmetric
        assert result.symmetry.is_involution
        assert result.coherence.pentagon
        assert result.coherence.triangle
        assert result.coherence.hexagon

    def test_monoidal_end(self, end_ss: StateSpace) -> None:
        """Trivial types form a symmetric monoidal category."""
        result = check_monoidal_structure(end_ss, end_ss, end_ss)
        assert result.is_monoidal

    def test_monoidal_mixed(self, chain2: StateSpace, chain3: StateSpace, branch2: StateSpace) -> None:
        """Mixed types form a symmetric monoidal category."""
        result = check_monoidal_structure(chain2, chain3, branch2)
        assert result.is_monoidal

    def test_monoidal_select(self, select2: StateSpace, chain2: StateSpace) -> None:
        """Selection types participate in monoidal structure."""
        end = build_statespace(parse("end"))
        result = check_monoidal_structure(select2, chain2, end)
        assert result.is_monoidal

    def test_result_dataclass_fields(self, chain2: StateSpace) -> None:
        """MonoidalResult has all expected fields."""
        result = check_monoidal_structure(chain2, chain2, chain2)
        assert isinstance(result, MonoidalResult)
        assert isinstance(result.unit, UnitResult)
        assert isinstance(result.associativity, AssociativityResult)
        assert isinstance(result.symmetry, SymmetryResult)
        assert isinstance(result.coherence, CoherenceResult)


# ---------------------------------------------------------------------------
# Benchmark protocols
# ---------------------------------------------------------------------------

class TestBenchmarkMonoidal:
    """Monoidal structure on benchmark protocols."""

    @pytest.mark.parametrize("type_str", [
        "&{a: end}",
        "&{a: end, b: end}",
        "&{a: &{b: end}}",
        "+{ok: end, err: end}",
        "&{open: &{read: end, write: end}}",
    ])
    def test_unit_law_benchmarks(self, type_str: str) -> None:
        """Unit law holds for various benchmark types."""
        ss = build_statespace(parse(type_str))
        result = check_monoidal_unit(ss)
        assert result.left_unit, f"Left unit failed for {type_str}"
        assert result.right_unit, f"Right unit failed for {type_str}"

    @pytest.mark.parametrize("t1,t2", [
        ("&{a: end}", "&{b: end}"),
        ("&{a: end, b: end}", "&{c: end}"),
        ("+{ok: end, err: end}", "&{a: end}"),
    ])
    def test_symmetry_benchmarks(self, t1: str, t2: str) -> None:
        """Symmetry holds for various type pairs."""
        ss1 = build_statespace(parse(t1))
        ss2 = build_statespace(parse(t2))
        result = check_symmetry(ss1, ss2)
        assert result.is_symmetric, f"Symmetry failed for {t1} || {t2}"
        assert result.is_involution, f"Involution failed for {t1} || {t2}"
