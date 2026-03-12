"""Tests for session type duality (Step 8)."""

import pytest

from reticulate.parser import (
    Branch, Continuation, End, Parallel, Rec, Select, Var, Wait,
    parse, pretty,
)
from reticulate.duality import (
    DualityResult,
    check_duality,
    dual,
    is_structurally_equal,
)


# ---------------------------------------------------------------------------
# Basic dual operation
# ---------------------------------------------------------------------------


class TestDualBasic:
    """Dual of atomic types."""

    def test_dual_end(self):
        assert dual(End()) == End()

    def test_dual_wait(self):
        assert dual(Wait()) == Wait()

    def test_dual_var(self):
        assert dual(Var("X")) == Var("X")


# ---------------------------------------------------------------------------
# Branch ↔ Selection swap
# ---------------------------------------------------------------------------


class TestDualBranchSelection:
    """Duality swaps branch and selection."""

    def test_branch_becomes_selection(self):
        s = parse("&{a: end}")
        d = dual(s)
        assert isinstance(d, Select)
        assert d.choices == (("a", End()),)

    def test_selection_becomes_branch(self):
        s = parse("+{a: end}")
        d = dual(s)
        assert isinstance(d, Branch)
        assert d.choices == (("a", End()),)

    def test_branch_multiple_methods(self):
        s = parse("&{a: end, b: end}")
        d = dual(s)
        assert isinstance(d, Select)
        assert len(d.choices) == 2
        assert dict(d.choices) == {"a": End(), "b": End()}

    def test_selection_multiple_labels(self):
        s = parse("+{a: end, b: end}")
        d = dual(s)
        assert isinstance(d, Branch)
        assert len(d.choices) == 2

    def test_nested_branch_in_selection(self):
        """dual(&{a: +{x: end}}) = +{a: &{x: end}}."""
        s = parse("&{a: +{x: end}}")
        d = dual(s)
        assert isinstance(d, Select)
        inner = dict(d.choices)["a"]
        assert isinstance(inner, Branch)
        assert dict(inner.choices) == {"x": End()}

    def test_nested_selection_in_branch(self):
        """dual(+{a: &{x: end}}) = &{a: +{x: end}}."""
        s = parse("+{a: &{x: end}}")
        d = dual(s)
        assert isinstance(d, Branch)
        inner = dict(d.choices)["a"]
        assert isinstance(inner, Select)

    def test_deeply_nested(self):
        """dual(&{a: &{b: +{c: end}}}) = +{a: +{b: &{c: end}}}."""
        s = parse("&{a: &{b: +{c: end}}}")
        d = dual(s)
        # outer: Select
        assert isinstance(d, Select)
        mid = dict(d.choices)["a"]
        # mid: Select (was Branch)
        assert isinstance(mid, Select)
        inner = dict(mid.choices)["b"]
        # inner: Branch (was Select)
        assert isinstance(inner, Branch)


# ---------------------------------------------------------------------------
# Recursion
# ---------------------------------------------------------------------------


class TestDualRecursion:
    """Duality distributes over recursion."""

    def test_recursive_branch(self):
        """dual(rec X . &{a: X, b: end}) = rec X . +{a: X, b: end}."""
        s = parse("rec X . &{a: X, b: end}")
        d = dual(s)
        assert isinstance(d, Rec)
        assert d.var == "X"
        assert isinstance(d.body, Select)

    def test_recursive_selection(self):
        """dual(rec X . +{a: X, b: end}) = rec X . &{a: X, b: end}."""
        s = parse("rec X . +{a: X, b: end}")
        d = dual(s)
        assert isinstance(d, Rec)
        assert isinstance(d.body, Branch)

    def test_recursive_var_preserved(self):
        """Variables inside recursion are unchanged by dual."""
        s = parse("rec X . &{a: X}")
        d = dual(s)
        body_choices = dict(d.body.choices)
        assert body_choices["a"] == Var("X")

    def test_nested_recursion(self):
        """dual(rec X . &{a: rec Y . +{b: Y, c: X}})."""
        s = parse("rec X . &{a: rec Y . +{b: Y, c: X}}")
        d = dual(s)
        assert isinstance(d, Rec)
        assert isinstance(d.body, Select)  # was Branch
        inner_rec = dict(d.body.choices)["a"]
        assert isinstance(inner_rec, Rec)
        assert isinstance(inner_rec.body, Branch)  # was Select


# ---------------------------------------------------------------------------
# Parallel and Continuation
# ---------------------------------------------------------------------------


class TestDualParallelContinuation:
    """Duality distributes over parallel and continuation."""

    def test_parallel(self):
        """dual(&{a: end} || +{b: end}) = (+{a: end} || &{b: end})."""
        s = Parallel((parse("&{a: end}"), parse("+{b: end}")))
        d = dual(s)
        assert isinstance(d, Parallel)
        assert isinstance(d.branches[0], Select)  # was Branch
        assert isinstance(d.branches[1], Branch)  # was Select

    def test_continuation(self):
        """dual((S₁ || S₂) . S₃) = (dual(S₁) || dual(S₂)) . dual(S₃)."""
        s = parse("(&{a: wait} || +{b: wait}) . &{c: end}")
        d = dual(s)
        assert isinstance(d, Continuation)
        assert isinstance(d.left, Parallel)
        assert isinstance(d.right, Select)  # was Branch


# ---------------------------------------------------------------------------
# Involution: dual(dual(S)) = S
# ---------------------------------------------------------------------------


class TestDualInvolution:
    """dual is an involution: dual(dual(S)) == S."""

    @pytest.mark.parametrize("type_str", [
        "end",
        "&{a: end}",
        "+{a: end}",
        "&{a: end, b: end}",
        "+{a: end, b: end}",
        "&{a: +{x: end, y: end}, b: end}",
        "+{a: &{x: end}, b: &{y: end}}",
        "rec X . &{a: X, b: end}",
        "rec X . +{a: X, b: end}",
        "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}",
    ])
    def test_involution(self, type_str: str):
        s = parse(type_str)
        assert is_structurally_equal(s, dual(dual(s)))

    def test_involution_parallel(self):
        s = Parallel((parse("&{a: end}"), parse("+{b: end}")))
        assert is_structurally_equal(s, dual(dual(s)))


# ---------------------------------------------------------------------------
# Pretty-printing of duals
# ---------------------------------------------------------------------------


class TestDualPrettyPrint:
    """Dual types should pretty-print correctly."""

    def test_branch_to_selection_pretty(self):
        s = parse("&{a: end, b: end}")
        d = dual(s)
        p = pretty(d)
        assert p == "+{a: end, b: end}"

    def test_selection_to_branch_pretty(self):
        s = parse("+{a: end, b: end}")
        d = dual(s)
        p = pretty(d)
        assert p == "&{a: end, b: end}"

    def test_recursive_pretty(self):
        s = parse("rec X . &{a: X, b: end}")
        d = dual(s)
        p = pretty(d)
        assert p == "rec X . +{a: X, b: end}"


# ---------------------------------------------------------------------------
# State-space isomorphism: L(S) ≅ L(dual(S))
# ---------------------------------------------------------------------------


class TestDualStateSpace:
    """L(S) and L(dual(S)) should be isomorphic as posets."""

    @pytest.mark.parametrize("type_str", [
        "end",
        "&{a: end}",
        "+{a: end}",
        "&{a: end, b: end}",
        "+{a: end, b: end}",
        "&{a: &{b: end}}",
        "+{a: +{b: end}}",
        "&{a: +{x: end, y: end}}",
        "rec X . &{a: X, b: end}",
        "rec X . +{a: X, b: end}",
    ])
    def test_isomorphic(self, type_str: str):
        from reticulate.morphism import find_isomorphism
        from reticulate.statespace import build_statespace

        s = parse(type_str)
        d = dual(s)
        ss_s = build_statespace(s)
        ss_d = build_statespace(d)

        iso = find_isomorphism(ss_s, ss_d)
        assert iso is not None, (
            f"L({type_str}) not isomorphic to L(dual({type_str}))"
        )

    def test_same_number_of_states(self):
        from reticulate.statespace import build_statespace

        s = parse("&{a: end, b: &{c: end}}")
        d = dual(s)
        ss_s = build_statespace(s)
        ss_d = build_statespace(d)
        assert len(ss_s.states) == len(ss_d.states)
        assert len(ss_s.transitions) == len(ss_d.transitions)

    def test_same_lattice_structure(self):
        """Both L(S) and L(dual(S)) should be lattices."""
        from reticulate.lattice import check_lattice
        from reticulate.statespace import build_statespace

        s = parse("&{a: end, b: &{c: end, d: end}}")
        d = dual(s)
        lr_s = check_lattice(build_statespace(s))
        lr_d = check_lattice(build_statespace(d))
        assert lr_s.is_lattice
        assert lr_d.is_lattice


# ---------------------------------------------------------------------------
# Selection annotation flip
# ---------------------------------------------------------------------------


class TestSelectionFlip:
    """Selection transitions should flip under duality."""

    def test_branch_selections_flip(self):
        """Branch transitions become selection transitions in dual."""
        from reticulate.statespace import build_statespace

        s = parse("&{a: end}")
        d = dual(s)
        ss_s = build_statespace(s)
        ss_d = build_statespace(d)

        # In L(S), the transition is a branch (not selection)
        assert len(ss_s.selection_transitions) == 0
        # In L(dual(S)), the transition is a selection
        assert len(ss_d.selection_transitions) == 1

    def test_selection_transitions_flip(self):
        """Selection transitions become branch transitions in dual."""
        from reticulate.statespace import build_statespace

        s = parse("+{a: end}")
        d = dual(s)
        ss_s = build_statespace(s)
        ss_d = build_statespace(d)

        # In L(S), the transition is a selection
        assert len(ss_s.selection_transitions) == 1
        # In L(dual(S)), the transition is a branch (not selection)
        assert len(ss_d.selection_transitions) == 0

    def test_mixed_annotations_flip(self):
        """Mixed branch/selection: all annotations flip."""
        from reticulate.statespace import build_statespace

        s = parse("&{a: +{x: end, y: end}}")
        d = dual(s)
        ss_s = build_statespace(s)
        ss_d = build_statespace(d)

        # L(S): 1 branch transition (a), 2 selection transitions (x, y)
        assert len(ss_s.selection_transitions) == 2
        assert len(ss_s.transitions) == 3

        # L(dual(S)): 1 selection transition (a), 2 branch transitions (x, y)
        assert len(ss_d.selection_transitions) == 1
        assert len(ss_d.transitions) == 3


# ---------------------------------------------------------------------------
# check_duality result
# ---------------------------------------------------------------------------


class TestCheckDuality:
    """Test the check_duality verification function."""

    def test_simple_branch(self):
        result = check_duality(parse("&{a: end, b: end}"))
        assert isinstance(result, DualityResult)
        assert result.is_involution
        assert result.is_isomorphic
        assert result.selection_flipped

    def test_simple_selection(self):
        result = check_duality(parse("+{a: end}"))
        assert result.is_involution
        assert result.is_isomorphic
        assert result.selection_flipped

    def test_recursive(self):
        result = check_duality(parse("rec X . &{a: X, b: end}"))
        assert result.is_involution
        assert result.is_isomorphic

    def test_end(self):
        result = check_duality(End())
        assert result.is_involution
        assert result.is_isomorphic

    def test_nested_mixed(self):
        result = check_duality(parse("&{a: +{x: end, y: end}, b: end}"))
        assert result.is_involution
        assert result.is_isomorphic
        assert result.selection_flipped

    def test_result_strings(self):
        result = check_duality(parse("&{a: end}"))
        assert result.type_str == "&{a: end}"
        assert result.dual_str == "+{a: end}"


# ---------------------------------------------------------------------------
# Subtyping and duality
# ---------------------------------------------------------------------------


class TestDualSubtyping:
    """Duality should reverse the subtyping direction for width."""

    def test_branch_width_reverses(self):
        """&{a,b} ≤ &{a} but dual: +{a,b} NOT ≤ +{a}; rather +{a} ≤ +{a,b}."""
        from reticulate.subtyping import is_subtype

        wide = parse("&{a: end, b: end}")
        narrow = parse("&{a: end}")

        # Branch: wider is subtype
        assert is_subtype(wide, narrow)
        assert not is_subtype(narrow, wide)

        # Dual (selection): narrower is subtype
        d_wide = dual(wide)
        d_narrow = dual(narrow)
        assert not is_subtype(d_wide, d_narrow)
        assert is_subtype(d_narrow, d_wide)

    def test_selection_width_reverses(self):
        """+{a} ≤ +{a,b} but dual: &{a} NOT ≤ &{a,b}; rather &{a,b} ≤ &{a}."""
        from reticulate.subtyping import is_subtype

        few = parse("+{a: end}")
        many = parse("+{a: end, b: end}")

        assert is_subtype(few, many)
        assert not is_subtype(many, few)

        d_few = dual(few)
        d_many = dual(many)
        assert not is_subtype(d_few, d_many)
        assert is_subtype(d_many, d_few)

    def test_subtype_contravariantly_dual(self):
        """S₁ ≤ S₂ iff dual(S₂) ≤ dual(S₁) — subtyping contravariancy."""
        from reticulate.subtyping import is_subtype

        pairs = [
            ("&{a: end, b: end}", "&{a: end}"),
            ("+{a: end}", "+{a: end, b: end}"),
            ("&{a: end, b: end, c: end}", "&{a: end}"),
            ("&{a: &{b: end, c: end}}", "&{a: &{b: end}}"),
        ]
        for s1_str, s2_str in pairs:
            s1 = parse(s1_str)
            s2 = parse(s2_str)
            assert is_subtype(s1, s2), f"{s1_str} should be ≤ {s2_str}"
            assert is_subtype(dual(s2), dual(s1)), (
                f"dual({s2_str}) should be ≤ dual({s1_str})"
            )

    def test_non_subtype_preserved(self):
        """If S₁ NOT ≤ S₂, then dual(S₂) NOT ≤ dual(S₁)."""
        from reticulate.subtyping import is_subtype

        s1 = parse("&{a: end}")
        s2 = parse("&{a: end, b: end}")
        assert not is_subtype(s1, s2)
        assert not is_subtype(dual(s2), dual(s1))


# ---------------------------------------------------------------------------
# Benchmarks: duality on all 34 protocols
# ---------------------------------------------------------------------------


class TestDualBenchmarks:
    """Verify duality properties on all benchmark protocols."""

    @pytest.fixture
    def benchmark_types(self):
        from tests.benchmarks.protocols import BENCHMARKS
        return [(b.name, parse(b.type_string)) for b in BENCHMARKS]

    def test_all_involutions(self, benchmark_types):
        """dual(dual(S)) == S for all benchmarks."""
        for name, s in benchmark_types:
            dd = dual(dual(s))
            assert is_structurally_equal(s, dd), (
                f"Involution failed for {name}"
            )

    def test_all_isomorphic(self, benchmark_types):
        """L(S) ≅ L(dual(S)) for all benchmarks."""
        from reticulate.morphism import find_isomorphism
        from reticulate.statespace import build_statespace

        for name, s in benchmark_types:
            d = dual(s)
            ss_s = build_statespace(s)
            ss_d = build_statespace(d)
            iso = find_isomorphism(ss_s, ss_d)
            assert iso is not None, (
                f"L({name}) not isomorphic to L(dual({name}))"
            )

    def test_all_dual_lattices(self, benchmark_types):
        """L(dual(S)) is a lattice for all benchmarks."""
        from reticulate.lattice import check_lattice
        from reticulate.statespace import build_statespace

        for name, s in benchmark_types:
            d = dual(s)
            ss_d = build_statespace(d)
            lr = check_lattice(ss_d)
            assert lr.is_lattice, (
                f"L(dual({name})) is not a lattice"
            )
