"""Comprehensive tests for the session-type parser.

Updated for the core grammar (no sequencing sugar).  Every construct has
exactly one meaning:

- ``&{m: end}`` is a Branch (the only way to express a method call).
- ``m`` alone is ``Var("m")``.
- ``m . end`` is ``Continuation(Var("m"), End())``.
- ``wait`` is ``Wait()`` (parallel branch completion).
- ``(S1 || S2) . S3`` is ``Continuation(Parallel((S1, S2)), S3)``.
"""

import pytest

from reticulate.parser import (
    Branch,
    Continuation,
    End,
    Parallel,
    ParseError,
    Rec,
    Select,
    Sequence,       # backward-compat alias for Continuation
    Token,
    TokenKind,
    Var,
    Wait,
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
        tokens = tokenize("end rec X foo_bar baz123 wait")
        idents = [t.value for t in tokens if t.kind is TokenKind.IDENT]
        assert idents == ["end", "rec", "X", "foo_bar", "baz123", "wait"]

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

    def test_wait_hashable(self) -> None:
        assert hash(Wait()) == hash(Wait())
        assert Wait() == Wait()

    def test_var_hashable(self) -> None:
        assert hash(Var("X")) == hash(Var("X"))
        assert Var("X") != Var("Y")

    def test_branch_hashable(self) -> None:
        b = Branch((("m", End()),))
        assert hash(b) == hash(Branch((("m", End()),)))

    def test_continuation_hashable(self) -> None:
        c = Continuation(End(), Var("X"))
        assert hash(c) == hash(Continuation(End(), Var("X")))

    def test_nodes_in_set(self) -> None:
        s = {End(), End(), Var("X"), Var("X"), Wait(), Wait()}
        assert len(s) == 3

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

    def test_end_not_wait(self) -> None:
        assert End() != Wait()

    def test_sequence_alias(self) -> None:
        """Sequence is an alias for Continuation."""
        assert Sequence is Continuation
        node = Sequence(End(), Var("X"))
        assert isinstance(node, Continuation)
        assert node == Continuation(End(), Var("X"))


# ===================================================================
# Parser -- basic constructs
# ===================================================================

class TestParserBasic:
    def test_end(self) -> None:
        assert parse("end") == End()

    def test_wait(self) -> None:
        assert parse("wait") == Wait()

    def test_variable(self) -> None:
        assert parse("X") == Var("X")

    def test_bare_identifier_is_var(self) -> None:
        """A bare identifier like ``m`` is Var("m"), NOT a Branch."""
        assert parse("m") == Var("m")

    def test_branch(self) -> None:
        result = parse("&{m: end, n: end}")
        assert result == Branch((("m", End()), ("n", End())))

    def test_single_branch(self) -> None:
        """Single-method branch: ``&{m: end}``."""
        result = parse("&{m: end}")
        assert result == Branch((("m", End()),))

    def test_select(self) -> None:
        result = parse("+{OK: end, ERROR: end}")
        assert result == Select((("OK", End()), ("ERROR", End())))

    def test_parallel(self) -> None:
        result = parse("(end || end)")
        assert result == Parallel((End(), End()))

    def test_parallel_bare_infix(self) -> None:
        """``X || Y`` at top level is Parallel((Var, Var))."""
        result = parse("X || Y")
        expected = Parallel((Var("X"), Var("Y")))
        assert result == expected

    def test_parallel_ternary(self) -> None:
        """``X || Y || Z`` is flat Parallel with 3 branches."""
        result = parse("X || Y || Z")
        expected = Parallel((Var("X"), Var("Y"), Var("Z")))
        assert result == expected

    def test_parallel_quaternary(self) -> None:
        """``a || b || c || d`` is flat Parallel with 4 branches."""
        result = parse("a || b || c || d")
        expected = Parallel((Var("a"), Var("b"), Var("c"), Var("d")))
        assert result == expected

    def test_parallel_ternary_parenthesized(self) -> None:
        """``(end || end || end)``."""
        result = parse("(end || end || end)")
        expected = Parallel((End(), End(), End()))
        assert result == expected

    def test_parallel_ternary_branches(self) -> None:
        """``(&{a: end} || &{b: end} || &{c: end})``."""
        result = parse("(&{a: end} || &{b: end} || &{c: end})")
        expected = Parallel((
            Branch((("a", End()),)),
            Branch((("b", End()),)),
            Branch((("c", End()),)),
        ))
        assert result == expected

    def test_rec(self) -> None:
        result = parse("rec X . end")
        assert result == Rec("X", End())

    def test_rec_body_is_atom(self) -> None:
        """rec X . body — body is parsed at atom level."""
        result = parse("rec X . &{m: end}")
        assert result == Rec("X", Branch((("m", End()),)))

    def test_parenthesized_grouping(self) -> None:
        result = parse("(end)")
        assert result == End()


# ===================================================================
# Parser -- continuation construct
# ===================================================================

class TestContinuation:
    def test_simple_continuation(self) -> None:
        """``(end || end) . X`` is Continuation(Parallel((...)), Var)."""
        result = parse("(end || end) . X")
        expected = Continuation(Parallel((End(), End())), Var("X"))
        assert result == expected

    def test_parallel_then_branch(self) -> None:
        """``(X || Y) . &{m: end}``."""
        result = parse("(X || Y) . &{m: end}")
        expected = Continuation(
            Parallel((Var("X"), Var("Y"))),
            Branch((("m", End()),)),
        )
        assert result == expected

    def test_parallel_then_end(self) -> None:
        """``(end || end) . end``."""
        result = parse("(end || end) . end")
        expected = Continuation(Parallel((End(), End())), End())
        assert result == expected

    def test_chained_continuation(self) -> None:
        """``(X || Y) . (end || end) . Z`` -- right-associative."""
        result = parse("(X || Y) . (end || end) . Z")
        expected = Continuation(
            Parallel((Var("X"), Var("Y"))),
            Continuation(Parallel((End(), End())), Var("Z")),
        )
        assert result == expected

    def test_var_dot_var_is_continuation(self) -> None:
        """``a . end`` is Continuation(Var("a"), End()), NOT a branch."""
        result = parse("a . end")
        expected = Continuation(Var("a"), End())
        assert result == expected

    def test_chained_var_dot_is_continuation(self) -> None:
        """``a . b . end`` is nested Continuation."""
        result = parse("a . b . end")
        expected = Continuation(Var("a"), Continuation(Var("b"), End()))
        assert result == expected

    def test_branch_then_continuation(self) -> None:
        """``&{m: end} . &{n: end}`` is Continuation(Branch, Branch)."""
        result = parse("&{m: end} . &{n: end}")
        expected = Continuation(
            Branch((("m", End()),)),
            Branch((("n", End()),)),
        )
        assert result == expected

    def test_continuation_dot_binds_tighter_than_par(self) -> None:
        """Dot has higher precedence than ``||``."""
        # a . end || b . end
        result = parse("a . end || b . end")
        expected = Parallel((
            Continuation(Var("a"), End()),
            Continuation(Var("b"), End()),
        ))
        assert result == expected


# ===================================================================
# Parser -- Wait keyword
# ===================================================================

class TestWait:
    def test_wait_standalone(self) -> None:
        assert parse("wait") == Wait()

    def test_wait_in_branch(self) -> None:
        result = parse("&{m: wait}")
        assert result == Branch((("m", Wait()),))

    def test_wait_in_parallel_branches(self) -> None:
        """``(&{read: wait} || &{write: wait})``."""
        result = parse("(&{read: wait} || &{write: wait})")
        expected = Parallel((
            Branch((("read", Wait()),)),
            Branch((("write", Wait()),)),
        ))
        assert result == expected

    def test_wait_parallel_then_continuation(self) -> None:
        """``(&{read: wait} || &{write: wait}) . &{close: end}``."""
        result = parse("(&{read: wait} || &{write: wait}) . &{close: end}")
        expected = Continuation(
            Parallel((
                Branch((("read", Wait()),)),
                Branch((("write", Wait()),)),
            )),
            Branch((("close", End()),)),
        )
        assert result == expected

    def test_wait_not_valid_as_rec_var(self) -> None:
        with pytest.raises(ParseError, match="keyword"):
            parse("rec wait . end")


# ===================================================================
# Parser -- complex / spec examples
# ===================================================================

class TestSpecExamples:
    def test_shared_file(self) -> None:
        """SharedFile from spec: &{init: &{open: +{OK: &{use: &{close: end}}, ERROR: end}}}"""
        src = "&{init: &{open: +{OK: &{use: &{close: end}}, ERROR: end}}}"
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
        """(&{read: end} || &{write: end})"""
        src = "(&{read: end} || &{write: end})"
        result = parse(src)
        expected = Parallel((
            Branch((("read", End()),)),
            Branch((("write", End()),)),
        ))
        assert result == expected

    def test_full_shared_file(self) -> None:
        """Full SharedFile with parallel and wait:
        &{init: &{open: +{OK: (&{read: wait} || &{write: wait}) . &{close: end}, ERROR: end}}}
        """
        src = "&{init: &{open: +{OK: (&{read: wait} || &{write: wait}) . &{close: end}, ERROR: end}}}"
        result = parse(src)
        expected = Branch((
            ("init", Branch((
                ("open", Select((
                    ("OK", Continuation(
                        Parallel((
                            Branch((("read", Wait()),)),
                            Branch((("write", Wait()),)),
                        )),
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
        """&{go: (&{a: end} || &{b: end}), stop: end}"""
        src = "&{go: (&{a: end} || &{b: end}), stop: end}"
        result = parse(src)
        expected = Branch((
            ("go", Parallel((Branch((("a", End()),)), Branch((("b", End()),))))),
            ("stop", End()),
        ))
        assert result == expected


# ===================================================================
# Parser -- error handling
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

    def test_wait_keyword_as_rec_variable(self) -> None:
        with pytest.raises(ParseError, match="keyword"):
            parse("rec wait . end")

    def test_dot_without_right_side(self) -> None:
        with pytest.raises(ParseError):
            parse("X .")

    def test_unexpected_token(self) -> None:
        with pytest.raises(ParseError):
            parse("}")

    def test_error_has_position(self) -> None:
        try:
            parse("X . }")
        except ParseError as e:
            assert e.pos is not None
            assert e.pos == 4  # position of '}'


# ===================================================================
# Pretty-printer
# ===================================================================

class TestPrettyPrinter:
    def test_end(self) -> None:
        assert pretty(End()) == "end"

    def test_wait(self) -> None:
        assert pretty(Wait()) == "wait"

    def test_var(self) -> None:
        assert pretty(Var("X")) == "X"

    def test_single_branch(self) -> None:
        """Single-method Branch prints as ``&{m: end}``, NOT sugar."""
        assert pretty(Branch((("m", End()),))) == "&{m: end}"

    def test_multi_branch(self) -> None:
        node = Branch((("m", End()), ("n", Var("X"))))
        assert pretty(node) == "&{m: end, n: X}"

    def test_select(self) -> None:
        node = Select((("OK", End()), ("ERR", End())))
        assert pretty(node) == "+{OK: end, ERR: end}"

    def test_parallel(self) -> None:
        node = Parallel((End(), Var("X")))
        assert pretty(node) == "end || X"

    def test_parallel_ternary(self) -> None:
        node = Parallel((Var("a"), Var("b"), Var("c")))
        assert pretty(node) == "a || b || c"

    def test_parallel_ternary_roundtrip(self) -> None:
        src = "(&{a: end} || &{b: end} || &{c: end})"
        assert parse(pretty(parse(src))) == parse(src)

    def test_rec(self) -> None:
        node = Rec("X", Var("X"))
        assert pretty(node) == "rec X . X"

    def test_continuation(self) -> None:
        node = Continuation(Parallel((End(), End())), Var("X"))
        assert pretty(node) == "(end || end) . X"

    def test_continuation_parallel_parens(self) -> None:
        """Parallel inside continuation gets parenthesized."""
        node = Continuation(
            Parallel((Var("a"), Var("b"))),
            Branch((("close", End()),)),
        )
        assert pretty(node) == "(a || b) . &{close: end}"

    def test_rec_body_parallel_parens(self) -> None:
        """Parallel as rec body gets parenthesized."""
        node = Rec("X", Parallel((Var("X"), End())))
        assert pretty(node) == "rec X . (X || end)"

    def test_roundtrip_simple(self) -> None:
        """parse -> pretty -> parse is identity."""
        src = "rec X . &{next: X, close: end}"
        assert parse(pretty(parse(src))) == parse(src)

    def test_roundtrip_complex(self) -> None:
        src = "&{init: &{open: +{OK: (&{read: wait} || &{write: wait}) . &{close: end}, ERROR: end}}}"
        assert parse(pretty(parse(src))) == parse(src)

    def test_roundtrip_parallel(self) -> None:
        src = "(&{read: end} || &{write: end})"
        assert parse(pretty(parse(src))) == parse(src)

    def test_roundtrip_wait(self) -> None:
        src = "(&{a: wait} || &{b: wait}) . end"
        assert parse(pretty(parse(src))) == parse(src)

    def test_roundtrip_var_continuation(self) -> None:
        src = "a . b . end"
        assert parse(pretty(parse(src))) == parse(src)

    def test_roundtrip_branch_single(self) -> None:
        """Single branch roundtrips via &{m: end} format."""
        src = "&{m: end}"
        assert parse(pretty(parse(src))) == parse(src)


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    def test_deeply_nested_branches(self) -> None:
        """Deeply nested explicit branches parse correctly."""
        # &{a: &{b: &{c: &{d: &{e: end}}}}}
        result = parse("&{a: &{b: &{c: &{d: &{e: end}}}}}")
        node = result
        for name in ("a", "b", "c", "d", "e"):
            assert isinstance(node, Branch)
            assert len(node.choices) == 1
            assert node.choices[0][0] == name
            node = node.choices[0][1]
        assert node == End()

    def test_deeply_nested_continuations(self) -> None:
        """``a . b . c . d . e . end`` is nested Continuation(Var, ...)."""
        result = parse("a . b . c . d . e . end")
        node = result
        for name in ("a", "b", "c", "d", "e"):
            assert isinstance(node, Continuation)
            assert node.left == Var(name)
            node = node.right
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
        expected = Parallel((
            Branch((("a", End()), ("b", End()))),
            Select((("c", End()), ("d", End()))),
        ))
        assert result == expected

    def test_rec_with_parallel_body(self) -> None:
        src = "rec X . (X || end)"
        result = parse(src)
        expected = Rec("X", Parallel((Var("X"), End())))
        assert result == expected

    def test_select_with_nested_branches(self) -> None:
        """``+{ok: &{m: &{n: end}}, err: end}``."""
        src = "+{ok: &{m: &{n: end}}, err: end}"
        result = parse(src)
        expected = Select((
            ("ok", Branch((("m", Branch((("n", End()),))),))),
            ("err", End()),
        ))
        assert result == expected

    def test_wait_in_both_parallel_branches(self) -> None:
        """Both branches of parallel use wait, then continue."""
        src = "(&{a: wait} || &{b: wait}) . &{c: end}"
        result = parse(src)
        expected = Continuation(
            Parallel((
                Branch((("a", Wait()),)),
                Branch((("b", Wait()),)),
            )),
            Branch((("c", End()),)),
        )
        assert result == expected

    def test_rec_body_is_atom_not_continuation(self) -> None:
        """``rec X . X . end`` parses as Continuation(Rec(X, X), End()),
        because rec body is atom-level (just X), then ``. end`` is continuation."""
        result = parse("rec X . X . end")
        expected = Continuation(Rec("X", Var("X")), End())
        assert result == expected

    def test_rec_body_branch_then_continuation(self) -> None:
        """``rec X . &{m: X} . end`` -- rec body is &{m: X}, then . end."""
        result = parse("rec X . &{m: X} . end")
        expected = Continuation(Rec("X", Branch((("m", Var("X")),))), End())
        assert result == expected

    def test_parallel_of_wait(self) -> None:
        """``(wait || wait)``."""
        result = parse("(wait || wait)")
        expected = Parallel((Wait(), Wait()))
        assert result == expected
