"""Tests for reticulate.sugar — desugaring and ensugaring."""

import pytest

from reticulate.parser import (
    End, Wait, Var, Branch, Select, Parallel, Rec, Continuation, parse,
)
from reticulate.sugar import desugar, ensugar


# ---------------------------------------------------------------------------
# desugar — identity for all node types
# ---------------------------------------------------------------------------

class TestDesugarIdentity:
    """desugar returns structurally identical nodes (core ASTs are already normal)."""

    def test_end(self) -> None:
        node = End()
        assert desugar(node) == node

    def test_wait(self) -> None:
        node = Wait()
        assert desugar(node) == node

    def test_var(self) -> None:
        node = Var("X")
        assert desugar(node) == node

    def test_branch_single(self) -> None:
        node = Branch((("m", End()),))
        assert desugar(node) == node

    def test_branch_multi(self) -> None:
        node = Branch((("a", End()), ("b", Var("X"))))
        assert desugar(node) == node

    def test_select(self) -> None:
        node = Select((("OK", End()), ("ERR", Var("X"))))
        assert desugar(node) == node

    def test_parallel(self) -> None:
        node = Parallel((End(), Wait()))
        assert desugar(node) == node

    def test_rec(self) -> None:
        node = Rec("X", Branch((("m", Var("X")),)))
        assert desugar(node) == node

    def test_continuation(self) -> None:
        node = Continuation(Parallel((End(), End())), End())
        assert desugar(node) == node


class TestDesugarRecursive:
    """desugar walks into nested structures."""

    def test_nested_branch_in_rec(self) -> None:
        node = Rec("X", Branch((("a", Branch((("b", End()),))),)))
        result = desugar(node)
        assert result == node

    def test_nested_parallel_in_continuation(self) -> None:
        node = Continuation(
            Parallel((Branch((("a", End()),)), Branch((("b", End()),)))),
            End(),
        )
        result = desugar(node)
        assert result == node

    def test_deeply_nested(self) -> None:
        node = Rec("X", Branch((
            ("m", Select((
                ("OK", Parallel((Branch((("a", Wait()),)), Branch((("b", Wait()),))))),
                ("ERR", End()),
            ))),
        )))
        result = desugar(node)
        assert result == node

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(TypeError, match="unknown AST node"):
            desugar("not a node")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ensugar — syntactic sugar for single-method Branch
# ---------------------------------------------------------------------------

class TestEnsugarSingleBranch:
    """Single-method Branch(&{m: S}) becomes m . S."""

    def test_single_branch_to_dot(self) -> None:
        node = Branch((("open", End()),))
        assert ensugar(node) == "open . end"

    def test_nested_single_branches(self) -> None:
        node = Branch((("a", Branch((("b", End()),))),))
        assert ensugar(node) == "a . b . end"

    def test_triple_nested_single_branches(self) -> None:
        node = Branch((("a", Branch((("b", Branch((("c", End()),))),))),))
        assert ensugar(node) == "a . b . c . end"


class TestEnsugarMultiBranch:
    """Multi-method Branch stays as &{...}."""

    def test_multi_branch_unchanged(self) -> None:
        node = Branch((("a", End()), ("b", End())))
        assert ensugar(node) == "&{a: end, b: end}"

    def test_multi_branch_with_nested_single(self) -> None:
        node = Branch((
            ("a", Branch((("x", End()),))),
            ("b", End()),
        ))
        assert ensugar(node) == "&{a: x . end, b: end}"


class TestEnsugarOtherNodes:
    """ensugar handles all other node types correctly."""

    def test_end(self) -> None:
        assert ensugar(End()) == "end"

    def test_wait(self) -> None:
        assert ensugar(Wait()) == "wait"

    def test_var(self) -> None:
        assert ensugar(Var("X")) == "X"

    def test_select(self) -> None:
        node = Select((("OK", End()), ("ERR", Var("X"))))
        assert ensugar(node) == "+{OK: end, ERR: X}"

    def test_parallel(self) -> None:
        node = Parallel((End(), End()))
        assert ensugar(node) == "end || end"

    def test_parallel_parenthesised_in_tight(self) -> None:
        # Parallel inside continuation needs parens
        node = Continuation(Parallel((End(), End())), End())
        assert ensugar(node) == "(end || end) . end"

    def test_rec(self) -> None:
        node = Rec("X", Var("X"))
        assert ensugar(node) == "rec X . X"

    def test_rec_with_sugared_body(self) -> None:
        node = Rec("X", Branch((("m", Var("X")),)))
        assert ensugar(node) == "rec X . m . X"

    def test_continuation(self) -> None:
        node = Continuation(
            Parallel((Branch((("a", Wait()),)), Branch((("b", Wait()),)))),
            End(),
        )
        assert ensugar(node) == "(a . wait || b . wait) . end"

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(TypeError, match="unknown AST node"):
            ensugar(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Round-trip: parse → ensugar
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """ensugar(parse(string)) produces expected sugared output."""

    def test_single_method_roundtrip(self) -> None:
        ast = parse("&{open: end}")
        assert ensugar(ast) == "open . end"

    def test_chain_roundtrip(self) -> None:
        ast = parse("&{open: &{close: end}}")
        assert ensugar(ast) == "open . close . end"

    def test_multi_branch_roundtrip(self) -> None:
        ast = parse("&{read: end, write: end}")
        assert ensugar(ast) == "&{read: end, write: end}"

    def test_rec_with_single_branch_roundtrip(self) -> None:
        ast = parse("rec X . &{next: X}")
        assert ensugar(ast) == "rec X . next . X"

    def test_select_roundtrip(self) -> None:
        ast = parse("+{OK: end, ERR: end}")
        assert ensugar(ast) == "+{OK: end, ERR: end}"

    def test_parallel_roundtrip(self) -> None:
        ast = parse("end || end")
        assert ensugar(ast) == "end || end"


# ---------------------------------------------------------------------------
# Benchmark protocols
# ---------------------------------------------------------------------------

class TestBenchmarkProtocols:
    """Test ensugar with a few benchmark protocols."""

    def test_java_iterator(self) -> None:
        ast = parse("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        result = ensugar(ast)
        assert result == "rec X . hasNext . +{TRUE: next . X, FALSE: end}"

    def test_file_object(self) -> None:
        ast = parse("&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}")
        result = ensugar(ast)
        assert result == "open . rec X . read . +{data: X, eof: close . end}"

    def test_simple_parallel(self) -> None:
        ast = parse("(&{a: wait} || &{b: wait}) . end")
        result = ensugar(ast)
        assert result == "(a . wait || b . wait) . end"

    def test_mixed_sugar(self) -> None:
        """Single branches sugared, multi branches kept."""
        ast = parse("&{init: &{open: +{OK: end, ERROR: end}}}")
        result = ensugar(ast)
        assert result == "init . open . +{OK: end, ERROR: end}"
