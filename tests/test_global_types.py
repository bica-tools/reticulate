"""Tests for multiparty global types (Step 11)."""

import pytest

from reticulate.global_types import (
    GEnd,
    GMessage,
    GParallel,
    GRec,
    GVar,
    GlobalParseError,
    GlobalType,
    build_global_statespace,
    parse_global,
    pretty_global,
    roles,
    tokenize_global,
)
from reticulate.lattice import check_lattice


# ---------------------------------------------------------------------------
# AST construction
# ---------------------------------------------------------------------------


class TestASTNodes:
    """Test direct AST construction."""

    def test_gend(self):
        assert GEnd() == GEnd()

    def test_gvar(self):
        assert GVar("X") == GVar("X")
        assert GVar("X") != GVar("Y")

    def test_gmessage_single(self):
        g = GMessage("A", "B", (("m", GEnd()),))
        assert g.sender == "A"
        assert g.receiver == "B"
        assert len(g.choices) == 1

    def test_gmessage_multiple(self):
        g = GMessage("A", "B", (("m1", GEnd()), ("m2", GEnd())))
        assert len(g.choices) == 2

    def test_gparallel(self):
        g = GParallel(GEnd(), GEnd())
        assert g.left == GEnd()

    def test_grec(self):
        g = GRec("X", GVar("X"))
        assert g.var == "X"

    def test_frozen(self):
        g = GEnd()
        with pytest.raises(AttributeError):
            g.x = 1  # type: ignore

    def test_hashable(self):
        s = {GEnd(), GEnd(), GVar("X")}
        assert len(s) == 2


# ---------------------------------------------------------------------------
# Role extraction
# ---------------------------------------------------------------------------


class TestRoles:
    """Test role extraction from global types."""

    def test_end_no_roles(self):
        assert roles(GEnd()) == frozenset()

    def test_var_no_roles(self):
        assert roles(GVar("X")) == frozenset()

    def test_single_message(self):
        g = GMessage("A", "B", (("m", GEnd()),))
        assert roles(g) == frozenset({"A", "B"})

    def test_nested_messages(self):
        g = GMessage("A", "B", (("m",
            GMessage("B", "C", (("n", GEnd()),))),))
        assert roles(g) == frozenset({"A", "B", "C"})

    def test_parallel_roles(self):
        g = GParallel(
            GMessage("A", "B", (("m", GEnd()),)),
            GMessage("C", "D", (("n", GEnd()),)),
        )
        assert roles(g) == frozenset({"A", "B", "C", "D"})

    def test_recursive_roles(self):
        g = GRec("X", GMessage("A", "B", (("m", GVar("X")),)))
        assert roles(g) == frozenset({"A", "B"})

    def test_two_buyer_roles(self):
        g = parse_global(
            "Buyer1 -> Seller : {lookup: "
            "Seller -> Buyer1 : {price: "
            "Seller -> Buyer2 : {price: end}}}"
        )
        assert roles(g) == frozenset({"Buyer1", "Buyer2", "Seller"})


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------


class TestPrettyGlobal:
    """Test pretty-printing of global types."""

    def test_end(self):
        assert pretty_global(GEnd()) == "end"

    def test_var(self):
        assert pretty_global(GVar("X")) == "X"

    def test_single_message(self):
        g = GMessage("A", "B", (("m", GEnd()),))
        assert pretty_global(g) == "A -> B : {m: end}"

    def test_multiple_choices(self):
        g = GMessage("A", "B", (("m1", GEnd()), ("m2", GEnd())))
        result = pretty_global(g)
        assert "A -> B" in result
        assert "m1: end" in result
        assert "m2: end" in result

    def test_rec(self):
        g = GRec("X", GMessage("A", "B", (("m", GVar("X")),)))
        assert pretty_global(g) == "rec X . A -> B : {m: X}"

    def test_parallel(self):
        g = GParallel(
            GMessage("A", "B", (("m", GEnd()),)),
            GMessage("C", "D", (("n", GEnd()),)),
        )
        result = pretty_global(g)
        assert "||" in result


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class TestTokenizer:
    """Test tokenization of global type strings."""

    def test_simple(self):
        tokens = tokenize_global("A -> B : {m: end}")
        kinds = [t.kind.name for t in tokens]
        assert "ARROW" in kinds
        assert "COLON" in kinds

    def test_arrow_token(self):
        tokens = tokenize_global("A -> B : {m: end}")
        arrows = [t for t in tokens if t.value == "->"]
        assert len(arrows) == 1

    def test_parallel(self):
        tokens = tokenize_global("(end || end)")
        pars = [t for t in tokens if t.value == "||"]
        assert len(pars) == 1

    def test_rec_keyword(self):
        tokens = tokenize_global("rec X . end")
        recs = [t for t in tokens if t.kind.name == "REC"]
        assert len(recs) == 1

    def test_error_on_invalid_char(self):
        with pytest.raises(GlobalParseError):
            tokenize_global("A -> B : {m: end} @")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class TestParseGlobal:
    """Test parsing of global type strings."""

    def test_end(self):
        assert parse_global("end") == GEnd()

    def test_var(self):
        assert parse_global("X") == GVar("X")

    def test_simple_message(self):
        g = parse_global("A -> B : {m: end}")
        assert isinstance(g, GMessage)
        assert g.sender == "A"
        assert g.receiver == "B"
        assert g.choices == (("m", GEnd()),)

    def test_multiple_choices(self):
        g = parse_global("A -> B : {m1: end, m2: end}")
        assert isinstance(g, GMessage)
        assert len(g.choices) == 2

    def test_nested_messages(self):
        g = parse_global("A -> B : {m: B -> C : {n: end}}")
        assert isinstance(g, GMessage)
        inner = g.choices[0][1]
        assert isinstance(inner, GMessage)
        assert inner.sender == "B"
        assert inner.receiver == "C"

    def test_rec(self):
        g = parse_global("rec X . A -> B : {m: X}")
        assert isinstance(g, GRec)
        assert g.var == "X"
        assert isinstance(g.body, GMessage)

    def test_parallel(self):
        g = parse_global("(A -> B : {m: end} || C -> D : {n: end})")
        assert isinstance(g, GParallel)

    def test_roundtrip(self):
        src = "A -> B : {m: end}"
        g = parse_global(src)
        assert parse_global(pretty_global(g)) == g

    def test_complex_roundtrip(self):
        src = "rec X . A -> B : {m: B -> A : {reply: X}, done: end}"
        g = parse_global(src)
        g2 = parse_global(pretty_global(g))
        assert g == g2

    def test_parse_error_on_invalid(self):
        with pytest.raises(GlobalParseError):
            parse_global("A -> B : @")

    def test_empty_choices(self):
        g = parse_global("A -> B : {}")
        assert isinstance(g, GMessage)
        assert len(g.choices) == 0

    def test_two_buyer(self):
        g = parse_global(
            "Buyer1 -> Seller : {lookup: "
            "Seller -> Buyer1 : {price: "
            "Buyer1 -> Buyer2 : {share: "
            "Buyer2 -> Seller : {accept: end, reject: end}}}}"
        )
        assert isinstance(g, GMessage)
        assert g.sender == "Buyer1"
        assert roles(g) == frozenset({"Buyer1", "Buyer2", "Seller"})

    def test_underscore_in_role(self):
        g = parse_global("role_A -> role_B : {msg: end}")
        assert g.sender == "role_A"


# ---------------------------------------------------------------------------
# State-space construction
# ---------------------------------------------------------------------------


class TestGlobalStateSpace:
    """Test state-space construction from global types."""

    def test_end_statespace(self):
        ss = build_global_statespace(GEnd())
        assert len(ss.states) == 1
        assert ss.top == ss.bottom

    def test_single_message(self):
        g = parse_global("A -> B : {m: end}")
        ss = build_global_statespace(g)
        assert len(ss.states) == 2  # top + end
        assert len(ss.transitions) == 1
        # Label should be role-annotated
        assert ss.transitions[0][1] == "A->B:m"

    def test_two_choices(self):
        g = parse_global("A -> B : {m1: end, m2: end}")
        ss = build_global_statespace(g)
        assert len(ss.states) == 2  # top + end (both go to end)
        assert len(ss.transitions) == 2
        labels = {t[1] for t in ss.transitions}
        assert labels == {"A->B:m1", "A->B:m2"}

    def test_sequential_messages(self):
        g = parse_global("A -> B : {m: B -> A : {reply: end}}")
        ss = build_global_statespace(g)
        assert len(ss.states) == 3  # top + middle + end
        assert len(ss.transitions) == 2

    def test_recursive_global(self):
        g = parse_global("rec X . A -> B : {m: X, done: end}")
        ss = build_global_statespace(g)
        assert len(ss.transitions) == 2
        labels = {t[1] for t in ss.transitions}
        assert "A->B:m" in labels
        assert "A->B:done" in labels

    def test_role_annotated_labels(self):
        g = parse_global("A -> B : {m: B -> C : {n: end}}")
        ss = build_global_statespace(g)
        labels = {t[1] for t in ss.transitions}
        assert "A->B:m" in labels
        assert "B->C:n" in labels


# ---------------------------------------------------------------------------
# Lattice properties (the Step 11 result)
# ---------------------------------------------------------------------------


class TestGlobalLattice:
    """Verify that global state spaces form lattices."""

    def test_single_message_lattice(self):
        g = parse_global("A -> B : {m: end}")
        ss = build_global_statespace(g)
        result = check_lattice(ss)
        assert result.is_lattice

    def test_branching_lattice(self):
        g = parse_global("A -> B : {m1: end, m2: end}")
        ss = build_global_statespace(g)
        result = check_lattice(ss)
        assert result.is_lattice

    def test_sequential_lattice(self):
        g = parse_global("A -> B : {m: B -> C : {n: end}}")
        ss = build_global_statespace(g)
        result = check_lattice(ss)
        assert result.is_lattice

    def test_recursive_lattice(self):
        g = parse_global("rec X . A -> B : {m: X, done: end}")
        ss = build_global_statespace(g)
        result = check_lattice(ss)
        assert result.is_lattice

    def test_two_buyer_lattice(self):
        g = parse_global(
            "Buyer1 -> Seller : {lookup: "
            "Seller -> Buyer1 : {price: "
            "Buyer1 -> Buyer2 : {share: "
            "Buyer2 -> Seller : {accept: end, reject: end}}}}"
        )
        ss = build_global_statespace(g)
        result = check_lattice(ss)
        assert result.is_lattice

    def test_two_phase_commit_lattice(self):
        g = parse_global(
            "Coord -> P1 : {prepare: "
            "P1 -> Coord : {yes: "
            "Coord -> P2 : {prepare: "
            "P2 -> Coord : {yes: "
            "Coord -> P1 : {commit: "
            "Coord -> P2 : {commit: end}}, "
            "no: Coord -> P1 : {abort: "
            "Coord -> P2 : {abort: end}}}}, "
            "no: Coord -> P2 : {abort: "
            "Coord -> P1 : {abort: end}}}}"
        )
        ss = build_global_statespace(g)
        result = check_lattice(ss)
        assert result.is_lattice


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------


class TestMultipartyBenchmarks:
    """Test all multiparty benchmarks."""

    @pytest.fixture
    def benchmarks(self):
        from tests.benchmarks.multiparty_protocols import MULTIPARTY_BENCHMARKS
        return MULTIPARTY_BENCHMARKS

    def test_all_parse(self, benchmarks):
        for b in benchmarks:
            g = parse_global(b.global_type_string)
            assert g is not None, f"{b.name}: parse failed"

    def test_all_roles_match(self, benchmarks):
        for b in benchmarks:
            g = parse_global(b.global_type_string)
            assert roles(g) == b.expected_roles, (
                f"{b.name}: expected {b.expected_roles}, got {roles(g)}"
            )

    def test_all_lattice(self, benchmarks):
        for b in benchmarks:
            g = parse_global(b.global_type_string)
            ss = build_global_statespace(g)
            result = check_lattice(ss)
            assert result.is_lattice, f"{b.name}: not a lattice"

    def test_all_roundtrip(self, benchmarks):
        for b in benchmarks:
            g = parse_global(b.global_type_string)
            s = pretty_global(g)
            g2 = parse_global(s)
            assert g == g2, f"{b.name}: roundtrip failed"
