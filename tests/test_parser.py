"""Comprehensive tests for the session-type parser."""

import pytest

from reticulate.parser import (
    Branch,
    End,
    Parallel,
    ParseError,
    Rec,
    Select,
    Sequence,
    Token,
    TokenKind,
    Var,
    parse,
    pretty,
    tokenize,
)


# ===================================================================
# Tokenizer tests
# ===================================================================

class TestTokenizer:
    def test_empty(self) -> None:
        tokens = tokenize("")
        assert len(tokens) == 1
        assert tokens[0].kind is TokenKind.EOF

    def test_single_chars(self) -> None:
        tokens = tokenize("{ } ( ) & + : , .")
        kinds = [t.kind for t in tokens[:-1]]  # exclude EOF
        assert kinds == [
            TokenKind.LBRACE, TokenKind.RBRACE,
            TokenKind.LPAREN, TokenKind.RPAREN,
            TokenKind.AMPERSAND, TokenKind.PLUS,
            TokenKind.COLON, TokenKind.COMMA,
            TokenKind.DOT,
        ]

    def test_parallel_operator(self) -> None:
        tokens = tokenize("||")
        assert tokens[0].kind is TokenKind.PAR
        assert tokens[0].value == "||"

    def test_identifiers(self) -> None:
        tokens = tokenize("end rec X foo_bar baz123")
        idents = [t.value for t in tokens if t.kind is TokenKind.IDENT]
        assert idents == ["end", "rec", "X", "foo_bar", "baz123"]

    def test_positions(self) -> None:
        tokens = tokenize("a . b")
        assert tokens[0].pos == 0   # a
        assert tokens[1].pos == 2   # .
        assert tokens[2].pos == 4   # b

    def test_unicode_oplus(self) -> None:
        tokens = tokenize("\u2295{")
        assert tokens[0].kind is TokenKind.PLUS

    def test_unicode_parallel(self) -> None:
        tokens = tokenize("\u2225")
        assert tokens[0].kind is TokenKind.PAR

    def test_unicode_mu(self) -> None:
        tokens = tokenize("\u03bcX")
        assert tokens[0].kind is TokenKind.IDENT
        assert tokens[0].value == "rec"

    def test_unexpected_character(self) -> None:
        with pytest.raises(ParseError, match="unexpected character"):
            tokenize("@")


# ===================================================================
# AST node tests
# ===================================================================

class TestASTNodes:
    def test_end_hashable(self) -> None:
        assert hash(End()) == hash(End())
        assert End() == End()

    def test_var_hashable(self) -> None:
        assert hash(Var("X")) == hash(Var("X"))
        assert Var("X") != Var("Y")

    def test_branch_hashable(self) -> None:
        b = Branch((("m", End()),))
        assert hash(b) == hash(Branch((("m", End()),)))

    def test_nodes_in_set(self) -> None:
        s = {End(), End(), Var("X"), Var("X")}
        assert len(s) == 2

    def test_frozen(self) -> None:
        with pytest.raises(AttributeError):
            End().name = "oops"  # type: ignore[attr-defined]

    def test_equality(self) -> None:
        a = Branch((("m", End()), ("n", Var("X"))))
        b = Branch((("m", End()), ("n", Var("X"))))
        assert a == b

    def test_inequality(self) -> None:
        a = Branch((("m", End()),))
        b = Select((("m", End()),))
        assert a != b


# ===================================================================
# Parser — basic constructs
# ===================================================================

class TestParserBasic:
    def test_end(self) -> None:
        assert parse("end") == End()

    def test_variable(self) -> None:
        assert parse("X") == Var("X")

    def test_branch(self) -> None:
        result = parse("&{m: end, n: end}")
        assert result == Branch((("m", End()), ("n", End())))

    def test_select(self) -> None:
        result = parse("+{OK: end, ERROR: end}")
        assert result == Select((("OK", End()), ("ERROR", End())))

    def test_parallel(self) -> None:
        result = parse("(end || end)")
        assert result == Parallel(End(), End())

    def test_parallel_bare_infix(self) -> None:
        result = parse("a . end || b . end")
        expected = Parallel(
            Branch((("a", End()),)),
            Branch((("b", End()),)),
        )
        assert result == expected

    def test_rec(self) -> None:
        result = parse("rec X . end")
        assert result == Rec("X", End())

    def test_parenthesized_grouping(self) -> None:
        result = parse("(end)")
        assert result == End()


# ===================================================================
# Parser — sequencing and desugaring
# ===================================================================

class TestSequencing:
    def test_simple_method_call(self) -> None:
        """``m . end`` desugars to ``Branch(((m, End),))``."""
        result = parse("m . end")
        assert result == Branch((("m", End()),))

    def test_chained_methods(self) -> None:
        """``a . b . end`` desugars right-associatively."""
        result = parse("a . b . end")
        assert result == Branch((("a", Branch((("b", End()),))),))

    def test_method_then_branch(self) -> None:
        """``init . &{m: end, n: end}``."""
        result = parse("init . &{m: end, n: end}")
        expected = Branch((("init", Branch((("m", End()), ("n", End())))),))
        assert result == expected

    def test_complex_left_becomes_sequence(self) -> None:
        """``(a || b) . end`` — complex left becomes Sequence."""
        result = parse("(a || b) . end")
        expected = Sequence(Parallel(Var("a"), Var("b")), End())
        assert result == expected

    def test_branch_then_method(self) -> None:
        """``&{m: end} . close . end`` — Branch on left becomes Sequence."""
        result = parse("&{m: end} . close . end")
        expected = Sequence(Branch((("m", End()),)), Branch((("close", End()),)))
        assert result == expected

    def test_rec_body_with_sequencing(self) -> None:
        """``rec X . m . X`` — recursion body includes desugared seq."""
        result = parse("rec X . m . X")
        assert result == Rec("X", Branch((("m", Var("X")),)))


# ===================================================================
# Parser — complex / spec examples
# ===================================================================

class TestSpecExamples:
    def test_shared_file(self) -> None:
        """SharedFile from spec §2.2: init . &{open: +{OK: use . close . end, ERROR: end}}"""
        src = "init . &{open: +{OK: use . close . end, ERROR: end}}"
        result = parse(src)
        expected = Branch((
            ("init", Branch((
                ("open", Select((
                    ("OK", Branch((("use", Branch((("close", End()),))),))),
                    ("ERROR", End()),
                ))),
            ))),
        ))
        assert result == expected

    def test_concurrent_file_access(self) -> None:
        """(read . end || write . end)"""
        src = "(read . end || write . end)"
        result = parse(src)
        expected = Parallel(
            Branch((("read", End()),)),
            Branch((("write", End()),)),
        )
        assert result == expected

    def test_full_shared_file(self) -> None:
        """Full SharedFile: init . &{open: +{OK: (read.end || write.end) . close . end, ERROR: end}}"""
        src = "init . &{open: +{OK: (read . end || write . end) . close . end, ERROR: end}}"
        result = parse(src)
        expected = Branch((
            ("init", Branch((
                ("open", Select((
                    ("OK", Sequence(
                        Parallel(
                            Branch((("read", End()),)),
                            Branch((("write", End()),)),
                        ),
                        Branch((("close", End()),)),
                    )),
                    ("ERROR", End()),
                ))),
            ))),
        ))
        assert result == expected

    def test_recursive_protocol(self) -> None:
        """rec X . &{next: X, close: end}"""
        src = "rec X . &{next: X, close: end}"
        result = parse(src)
        expected = Rec("X", Branch((("next", Var("X")), ("close", End()))))
        assert result == expected

    def test_nested_recursion(self) -> None:
        """rec X . rec Y . &{a: X, b: Y}"""
        src = "rec X . rec Y . &{a: X, b: Y}"
        result = parse(src)
        expected = Rec("X", Rec("Y", Branch((("a", Var("X")), ("b", Var("Y"))))))
        assert result == expected

    def test_parallel_inside_branch(self) -> None:
        """&{go: (a . end || b . end), stop: end}"""
        src = "&{go: (a . end || b . end), stop: end}"
        result = parse(src)
        expected = Branch((
            ("go", Parallel(Branch((("a", End()),)), Branch((("b", End()),)))),
            ("stop", End()),
        ))
        assert result == expected


# ===================================================================
# Parser — error handling
# ===================================================================

class TestParserErrors:
    def test_empty_input(self) -> None:
        with pytest.raises(ParseError, match="unexpected token EOF"):
            parse("")

    def test_trailing_garbage(self) -> None:
        with pytest.raises(ParseError, match="unexpected token"):
            parse("end end")

    def test_missing_brace_branch(self) -> None:
        with pytest.raises(ParseError):
            parse("&{m: end")

    def test_missing_brace_select(self) -> None:
        with pytest.raises(ParseError):
            parse("+{m: end")

    def test_missing_paren(self) -> None:
        with pytest.raises(ParseError):
            parse("(end")

    def test_empty_branch(self) -> None:
        with pytest.raises(ParseError, match="at least one choice"):
            parse("&{}")

    def test_empty_select(self) -> None:
        with pytest.raises(ParseError, match="at least one choice"):
            parse("+{}")

    def test_missing_colon_in_branch(self) -> None:
        with pytest.raises(ParseError, match="colon"):
            parse("&{m end}")

    def test_rec_keyword_as_variable(self) -> None:
        with pytest.raises(ParseError, match="keyword"):
            parse("rec rec . end")

    def test_end_keyword_as_rec_variable(self) -> None:
        with pytest.raises(ParseError, match="keyword"):
            parse("rec end . end")

    def test_dot_without_right_side(self) -> None:
        with pytest.raises(ParseError):
            parse("m .")

    def test_unexpected_token(self) -> None:
        with pytest.raises(ParseError):
            parse("}")

    def test_error_has_position(self) -> None:
        try:
            parse("m . }")
        except ParseError as e:
            assert e.pos is not None
            assert e.pos == 4  # position of '}'


# ===================================================================
# Pretty-printer
# ===================================================================

class TestPrettyPrinter:
    def test_end(self) -> None:
        assert pretty(End()) == "end"

    def test_var(self) -> None:
        assert pretty(Var("X")) == "X"

    def test_single_branch_as_sugar(self) -> None:
        """Single-method Branch prints as sequencing sugar."""
        assert pretty(Branch((("m", End()),))) == "m . end"

    def test_multi_branch(self) -> None:
        node = Branch((("m", End()), ("n", Var("X"))))
        assert pretty(node) == "&{m: end, n: X}"

    def test_select(self) -> None:
        node = Select((("OK", End()), ("ERR", End())))
        assert pretty(node) == "+{OK: end, ERR: end}"

    def test_parallel(self) -> None:
        node = Parallel(End(), Var("X"))
        assert pretty(node) == "end || X"

    def test_rec(self) -> None:
        node = Rec("X", Var("X"))
        assert pretty(node) == "rec X . X"

    def test_sequence(self) -> None:
        node = Sequence(Parallel(End(), End()), Var("X"))
        assert pretty(node) == "(end || end) . X"

    def test_roundtrip_simple(self) -> None:
        """parse → pretty → parse is identity."""
        src = "rec X . &{next: X, close: end}"
        assert parse(pretty(parse(src))) == parse(src)

    def test_roundtrip_complex(self) -> None:
        src = "init . &{open: +{OK: (read . end || write . end) . close . end, ERROR: end}}"
        assert parse(pretty(parse(src))) == parse(src)

    def test_roundtrip_parallel(self) -> None:
        src = "(a . end || b . end)"
        assert parse(pretty(parse(src))) == parse(src)


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    def test_deeply_nested(self) -> None:
        """Deeply nested structure parses without stack overflow."""
        # a . b . c . d . e . end
        result = parse("a . b . c . d . e . end")
        # Should be 5 layers of Branch
        node = result
        for name in ("a", "b", "c", "d", "e"):
            assert isinstance(node, Branch)
            assert len(node.choices) == 1
            assert node.choices[0][0] == name
            node = node.choices[0][1]
        assert node == End()

    def test_whitespace_insensitive(self) -> None:
        compact = parse("&{m:end,n:end}")
        spaced = parse("&{  m  :  end  ,  n  :  end  }")
        assert compact == spaced

    def test_newlines_ok(self) -> None:
        src = """&{
            m: end,
            n: end
        }"""
        result = parse(src)
        assert result == Branch((("m", End()), ("n", End())))

    def test_parallel_with_complex_branches(self) -> None:
        src = "(&{a: end, b: end} || +{c: end, d: end})"
        result = parse(src)
        expected = Parallel(
            Branch((("a", End()), ("b", End()))),
            Select((("c", End()), ("d", End()))),
        )
        assert result == expected

    def test_rec_with_parallel_body(self) -> None:
        src = "rec X . (X || end)"
        result = parse(src)
        expected = Rec("X", Parallel(Var("X"), End()))
        assert result == expected

    def test_select_with_sequencing_in_body(self) -> None:
        src = "+{ok: m . n . end, err: end}"
        result = parse(src)
        expected = Select((
            ("ok", Branch((("m", Branch((("n", End()),))),))),
            ("err", End()),
        ))
        assert result == expected
