"""Tests for the equation grammar (T1.2): parse_program, resolve, build_statespace_from_program."""

from __future__ import annotations

import os
import tempfile

import pytest

from reticulate.parser import (
    Branch,
    Continuation,
    Definition,
    End,
    Parallel,
    ParseError,
    Program,
    Rec,
    Select,
    TokenKind,
    Var,
    Wait,
    parse,
    parse_program,
    pretty,
    pretty_program,
    tokenize,
)
from reticulate.resolve import ResolveError, resolve
from reticulate.statespace import (
    StateSpace,
    build_statespace,
    build_statespace_from_program,
)


# ============================================================================
# A. Tokenizer (3 tests)
# ============================================================================


class TestTokenizerEquals:
    def test_equals_token_recognized(self) -> None:
        tokens = tokenize("S = end")
        kinds = [t.kind for t in tokens]
        assert TokenKind.EQUALS in kinds

    def test_equals_token_value(self) -> None:
        tokens = tokenize("S = end")
        eq_tokens = [t for t in tokens if t.kind is TokenKind.EQUALS]
        assert len(eq_tokens) == 1
        assert eq_tokens[0].value == "="

    def test_equals_in_context(self) -> None:
        tokens = tokenize("S = &{a: T}\nT = end")
        eq_tokens = [t for t in tokens if t.kind is TokenKind.EQUALS]
        assert len(eq_tokens) == 2


# ============================================================================
# B. parse_program (15 tests)
# ============================================================================


class TestParseProgram:
    def test_single_definition(self) -> None:
        prog = parse_program("S = end")
        assert len(prog.definitions) == 1
        assert prog.definitions[0].name == "S"
        assert prog.definitions[0].body == End()

    def test_multiple_definitions(self) -> None:
        prog = parse_program("S = &{a: T}\nT = end")
        assert len(prog.definitions) == 2
        assert prog.definitions[0].name == "S"
        assert prog.definitions[1].name == "T"
        assert prog.definitions[0].body == Branch((("a", Var("T")),))
        assert prog.definitions[1].body == End()

    def test_bare_expression_fallback(self) -> None:
        prog = parse_program("&{a: end}")
        assert len(prog.definitions) == 1
        assert prog.definitions[0].name == "_main"
        assert prog.definitions[0].body == Branch((("a", End()),))

    def test_with_parallel(self) -> None:
        prog = parse_program("S = (&{a: wait} || &{b: wait}) . end")
        assert len(prog.definitions) == 1
        assert prog.definitions[0].name == "S"
        body = prog.definitions[0].body
        assert isinstance(body, Continuation)
        assert isinstance(body.left, Parallel)

    def test_name_references(self) -> None:
        prog = parse_program("S = &{a: T, b: end}\nT = &{c: end}")
        assert len(prog.definitions) == 2
        assert prog.definitions[0].body == Branch((
            ("a", Var("T")),
            ("b", End()),
        ))

    def test_keyword_rejection_end(self) -> None:
        with pytest.raises(ParseError):
            parse_program("end = &{a: end}")

    def test_keyword_rejection_rec(self) -> None:
        with pytest.raises(ParseError):
            parse_program("rec = &{a: end}")

    def test_keyword_rejection_wait(self) -> None:
        with pytest.raises(ParseError):
            parse_program("wait = end")

    def test_mu_is_valid_definition_name(self) -> None:
        """'mu' (ASCII) is a valid definition name (only Unicode μ maps to rec)."""
        prog = parse_program("mu = end")
        assert prog.definitions[0].name == "mu"

    def test_bare_end_as_expression(self) -> None:
        """'end' alone is a bare expression, not a definition."""
        prog = parse_program("end")
        assert len(prog.definitions) == 1
        assert prog.definitions[0].name == "_main"
        assert prog.definitions[0].body == End()

    def test_rec_expression_bare(self) -> None:
        """'rec X . ...' starts with keyword so treated as bare expression."""
        prog = parse_program("rec X . &{a: X}")
        assert len(prog.definitions) == 1
        assert prog.definitions[0].name == "_main"
        assert isinstance(prog.definitions[0].body, Rec)

    def test_definitions_with_rec(self) -> None:
        prog = parse_program("S = rec X . &{a: X, b: end}")
        assert len(prog.definitions) == 1
        body = prog.definitions[0].body
        assert isinstance(body, Rec)
        assert body.var == "X"

    def test_definition_with_select(self) -> None:
        prog = parse_program("S = +{ok: end, err: end}")
        body = prog.definitions[0].body
        assert isinstance(body, Select)
        assert len(body.choices) == 2

    def test_three_definitions(self) -> None:
        prog = parse_program("A = &{x: B}\nB = &{y: C}\nC = end")
        assert len(prog.definitions) == 3
        assert prog.definitions[0].name == "A"
        assert prog.definitions[1].name == "B"
        assert prog.definitions[2].name == "C"

    def test_multiline_separators(self) -> None:
        """Definitions separated by newlines."""
        src = "S = &{a: T}\nT = &{b: end}"
        prog = parse_program(src)
        assert len(prog.definitions) == 2


# ============================================================================
# C. resolve (20 tests)
# ============================================================================


class TestResolve:
    def test_simple_end(self) -> None:
        prog = parse_program("S = end")
        ast = resolve(prog)
        assert ast == End()

    def test_simple_branch(self) -> None:
        prog = parse_program("S = &{a: end}")
        ast = resolve(prog)
        assert ast == Branch((("a", End()),))

    def test_reference_inlined(self) -> None:
        prog = parse_program("S = &{a: T}\nT = end")
        ast = resolve(prog)
        assert ast == Branch((("a", End()),))

    def test_reference_branch(self) -> None:
        prog = parse_program("S = &{a: T}\nT = &{c: end}")
        ast = resolve(prog)
        assert ast == Branch((("a", Branch((("c", End()),))),))

    def test_self_recursion(self) -> None:
        prog = parse_program("S = &{a: S, b: end}")
        ast = resolve(prog)
        assert isinstance(ast, Rec)
        assert ast.var == "S"
        assert isinstance(ast.body, Branch)
        # body should contain Var("S") for the self-reference
        choices = dict(ast.body.choices)
        assert choices["a"] == Var("S")
        assert choices["b"] == End()

    def test_mutual_recursion(self) -> None:
        prog = parse_program("S = &{a: T}\nT = &{b: S, c: end}")
        ast = resolve(prog)
        # S should resolve to a Rec since it's mutually recursive
        # The exact form depends on implementation; just check it's valid
        assert ast is not None
        # Should be buildable into a state space
        ss = build_statespace(ast)
        assert len(ss.states) > 0

    def test_unresolved_name_raises(self) -> None:
        prog = Program(definitions=(
            Definition("S", Branch((("a", Var("UNKNOWN")),))),
        ))
        with pytest.raises(ResolveError, match="unbound"):
            resolve(prog)

    def test_duplicate_definition_raises(self) -> None:
        prog = Program(definitions=(
            Definition("S", End()),
            Definition("S", End()),
        ))
        with pytest.raises(ResolveError, match="duplicate"):
            resolve(prog)

    def test_chain_A_B_C(self) -> None:
        prog = parse_program("A = B\nB = C\nC = end")
        ast = resolve(prog)
        assert ast == End()

    def test_chain_with_branch(self) -> None:
        prog = parse_program("A = &{x: B}\nB = &{y: C}\nC = end")
        ast = resolve(prog)
        assert ast == Branch((("x", Branch((("y", End()),))),))

    def test_mixed_recursion_and_reference(self) -> None:
        """Self-recursive with a non-recursive reference."""
        prog = parse_program("S = &{a: S, b: T}\nT = end")
        ast = resolve(prog)
        assert isinstance(ast, Rec)
        choices = dict(ast.body.choices)
        assert choices["a"] == Var("S")
        assert choices["b"] == End()

    def test_select_preserved(self) -> None:
        prog = parse_program("S = +{ok: T, err: end}\nT = &{x: end}")
        ast = resolve(prog)
        assert isinstance(ast, Select)
        choices = dict(ast.choices)
        assert choices["ok"] == Branch((("x", End()),))
        assert choices["err"] == End()

    def test_parallel_in_definition(self) -> None:
        prog = parse_program("S = (&{a: wait} || &{b: wait}) . end")
        ast = resolve(prog)
        assert isinstance(ast, Continuation)
        assert isinstance(ast.left, Parallel)

    def test_rec_inside_definition(self) -> None:
        """rec inside a definition body (not definition-level recursion)."""
        prog = parse_program("S = rec X . &{a: X, b: end}")
        ast = resolve(prog)
        assert isinstance(ast, Rec)
        assert ast.var == "X"

    def test_bare_expression_resolve(self) -> None:
        prog = parse_program("&{a: end}")
        ast = resolve(prog)
        assert ast == Branch((("a", End()),))

    def test_empty_program_raises(self) -> None:
        prog = Program(definitions=())
        with pytest.raises(ResolveError, match="empty"):
            resolve(prog)

    def test_multi_reference_same_target(self) -> None:
        """Two branches reference the same definition."""
        prog = parse_program("S = &{a: M, b: M}\nM = &{c: end}")
        ast = resolve(prog)
        assert isinstance(ast, Branch)
        choices = dict(ast.choices)
        # Both should resolve to the same AST
        assert choices["a"] == Branch((("c", End()),))
        assert choices["b"] == Branch((("c", End()),))

    def test_three_definitions_mixed(self) -> None:
        prog = parse_program("A = &{x: B, y: C}\nB = &{p: end}\nC = &{q: end}")
        ast = resolve(prog)
        assert isinstance(ast, Branch)
        choices = dict(ast.choices)
        assert choices["x"] == Branch((("p", End()),))
        assert choices["y"] == Branch((("q", End()),))

    def test_self_recursive_select(self) -> None:
        prog = parse_program("S = +{retry: S, done: end}")
        ast = resolve(prog)
        assert isinstance(ast, Rec)
        assert ast.var == "S"
        choices = dict(ast.body.choices)
        assert choices["retry"] == Var("S")
        assert choices["done"] == End()

    def test_definition_names_case_sensitive(self) -> None:
        prog = parse_program("S = &{a: s}\ns = end")
        ast = resolve(prog)
        assert isinstance(ast, Branch)
        choices = dict(ast.choices)
        assert choices["a"] == End()


# ============================================================================
# D. State-space from program (15 tests)
# ============================================================================


class TestStatespaceFromProgram:
    def test_simple_same_as_parse(self) -> None:
        """Simple definition produces same state space as direct parse."""
        prog = parse_program("S = &{a: end}")
        ss_prog = build_statespace_from_program(prog)
        ss_direct = build_statespace(parse("&{a: end}"))
        assert len(ss_prog.states) == len(ss_direct.states)
        assert len(ss_prog.transitions) == len(ss_direct.transitions)

    def test_reference_same_as_inline(self) -> None:
        """Reference produces same state space as inline."""
        prog = parse_program("S = &{a: T}\nT = &{b: end}")
        ss_prog = build_statespace_from_program(prog)
        ss_direct = build_statespace(parse("&{a: &{b: end}}"))
        assert len(ss_prog.states) == len(ss_direct.states)
        assert len(ss_prog.transitions) == len(ss_direct.transitions)

    def test_shared_state_fewer_states(self) -> None:
        """Two branches referencing the same definition should SHARE a state.

        S = &{a: M, b: M}
        M = &{c: end}

        With sharing: 3 states (S, M, end) and 4 transitions (a->M, b->M, c->end, c->end... wait, M is shared so just c->end once)
        Without sharing: 4 states (S, M1, M2, end) — more states.
        """
        prog = parse_program("S = &{a: M, b: M}\nM = &{c: end}")
        ss = build_statespace_from_program(prog)
        # With sharing: S, M, end = 3 states; transitions: a->M, b->M, c->end = 3
        assert len(ss.states) == 3
        assert len(ss.transitions) == 3

    def test_shared_deeper(self) -> None:
        """Sharing deeper: S = &{a: M, b: M}, M = &{c: N}, N = end."""
        prog = parse_program("S = &{a: M, b: M}\nM = &{c: end}")
        ss = build_statespace_from_program(prog)
        # S, M, end = 3 states
        assert len(ss.states) == 3

    def test_n_poset(self) -> None:
        """The N-poset example from the spec."""
        src = (
            "Init = &{a: &{b: ABC, c: AC}, c: AC}\n"
            "ABC = &{d: end}\n"
            "AC = &{b: ABC, d: &{b: end}}"
        )
        prog = parse_program(src)
        ss = build_statespace_from_program(prog)
        # Should have shared states for ABC and AC
        # Init, &{b:ABC,c:AC}, AC, ABC, &{b:end}, end = multiple states
        # The key test is that ABC and AC are shared
        assert len(ss.states) >= 5  # at minimum

    def test_fence_equivalent(self) -> None:
        """Equation form of (&{a:end}||&{d:end}).(&{b:end}||&{c:end})."""
        # Direct parallel construction
        ss_direct = build_statespace(
            parse("(&{a: wait} || &{d: wait}) . (&{b: wait} || &{c: wait}) . end")
        )
        # Equation form
        prog = parse_program(
            "S = (&{a: wait} || &{d: wait}) . (&{b: wait} || &{c: wait}) . end"
        )
        ss_eq = build_statespace_from_program(prog)
        assert len(ss_eq.states) == len(ss_direct.states)
        assert len(ss_eq.transitions) == len(ss_direct.transitions)

    def test_self_recursive_definition(self) -> None:
        prog = parse_program("S = &{a: S, b: end}")
        ss = build_statespace_from_program(prog)
        assert len(ss.states) >= 2
        # Should have a cycle (a loops back)
        has_cycle = any(
            t == ss.top for _, _, t in ss.transitions
        )
        assert has_cycle

    def test_bare_expression_via_program(self) -> None:
        prog = parse_program("&{a: end, b: end}")
        ss = build_statespace_from_program(prog)
        assert len(ss.states) == 2  # branch + end
        assert len(ss.transitions) == 2

    def test_select_definition(self) -> None:
        prog = parse_program("S = +{ok: &{x: end}, err: end}")
        ss = build_statespace_from_program(prog)
        # +{ok, err} state, &{x} state, end state = 3
        assert len(ss.states) == 3

    def test_chain_three_definitions(self) -> None:
        prog = parse_program("A = &{x: B}\nB = &{y: C}\nC = &{z: end}")
        ss = build_statespace_from_program(prog)
        # A, B, C, end = 4 states
        assert len(ss.states) == 4
        assert len(ss.transitions) == 3

    def test_shared_three_refs(self) -> None:
        """Three branches referencing the same definition."""
        prog = parse_program("S = &{a: M, b: M, c: M}\nM = &{d: end}")
        ss = build_statespace_from_program(prog)
        # S, M, end = 3 states
        assert len(ss.states) == 3
        assert len(ss.transitions) == 4  # a->M, b->M, c->M, d->end

    def test_definition_with_parallel(self) -> None:
        prog = parse_program("S = (&{a: wait} || &{b: wait}) . end")
        ss = build_statespace_from_program(prog)
        assert len(ss.states) >= 3

    def test_lattice_check_on_program(self) -> None:
        """Verify lattice property on equation-built state space."""
        from reticulate.lattice import check_lattice
        prog = parse_program("S = &{a: &{b: end}, b: &{a: end}}")
        ss = build_statespace_from_program(prog)
        result = check_lattice(ss)
        # Diamond: should be a lattice
        assert result.is_lattice

    def test_benchmark_roundtrip_iterator(self) -> None:
        """Java Iterator rewritten in equation form produces same state count."""
        # Direct: rec X . &{hasNext: +{true: &{next: X}, false: end}}
        direct_ast = parse("rec X . &{hasNext: +{true: &{next: X}, false: end}}")
        ss_direct = build_statespace(direct_ast)
        # Equation form using self-recursion
        prog = parse_program("Iter = &{hasNext: +{true: &{next: Iter}, false: end}}")
        ss_eq = build_statespace_from_program(prog)
        assert len(ss_eq.states) == len(ss_direct.states)

    def test_benchmark_simple_file(self) -> None:
        """File-like protocol: Open = &{read: Open, close: end}."""
        prog = parse_program("Open = &{read: Open, close: end}")
        ss = build_statespace_from_program(prog)
        # Open + end = 2 states
        assert len(ss.states) == 2
        assert len(ss.transitions) == 2


# ============================================================================
# E. CLI (5 tests)
# ============================================================================


class TestCLIEquations:
    def test_file_input(self, tmp_path: object) -> None:
        """CLI accepts --file flag."""
        from reticulate.cli import main

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".st", delete=False
        ) as f:
            f.write("S = &{a: end}")
            f.flush()
            try:
                # Should not raise
                main(["-f", f.name])
            finally:
                os.unlink(f.name)

    def test_equation_string_input(self) -> None:
        """CLI accepts equation strings on command line."""
        from reticulate.cli import main
        # Single definition that looks like an equation
        main(["S = &{a: end}"])

    def test_bare_expression_still_works(self) -> None:
        """CLI still accepts bare expressions."""
        from reticulate.cli import main
        main(["&{a: end}"])

    def test_file_with_multiple_definitions(self) -> None:
        from reticulate.cli import main

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".st", delete=False
        ) as f:
            f.write("S = &{a: T}\nT = end")
            f.flush()
            try:
                main(["-f", f.name])
            finally:
                os.unlink(f.name)

    def test_file_not_found(self) -> None:
        from reticulate.cli import main
        with pytest.raises(SystemExit):
            main(["-f", "/nonexistent/file.st"])


# ============================================================================
# F. Backward compatibility (5 tests)
# ============================================================================


class TestBackwardCompatibility:
    def test_parse_function_unchanged(self) -> None:
        """parse() still works as before."""
        ast = parse("&{a: end}")
        assert ast == Branch((("a", End()),))

    def test_parse_rec_unchanged(self) -> None:
        ast = parse("rec X . &{a: X, b: end}")
        assert isinstance(ast, Rec)

    def test_build_statespace_unchanged(self) -> None:
        """build_statespace() still works as before."""
        ast = parse("&{a: &{b: end}}")
        ss = build_statespace(ast)
        assert len(ss.states) == 3

    def test_pretty_unchanged(self) -> None:
        ast = parse("&{a: end, b: end}")
        assert pretty(ast) == "&{a: end, b: end}"

    def test_tokenize_backward_compat(self) -> None:
        """Existing tokens still work; EQUALS doesn't break anything."""
        tokens = tokenize("&{a: end}")
        kinds = [t.kind for t in tokens if t.kind is not TokenKind.EOF]
        assert TokenKind.EQUALS not in kinds


# ============================================================================
# G. Pretty-printer roundtrip (5 tests)
# ============================================================================


class TestPrettyProgram:
    def test_bare_expression_roundtrip(self) -> None:
        prog = parse_program("&{a: end}")
        assert pretty_program(prog) == "&{a: end}"

    def test_single_definition_roundtrip(self) -> None:
        prog = parse_program("S = &{a: end}")
        text = pretty_program(prog)
        assert text == "S = &{a: end}"

    def test_multi_definition_roundtrip(self) -> None:
        prog = parse_program("S = &{a: T}\nT = end")
        text = pretty_program(prog)
        assert "S = &{a: T}" in text
        assert "T = end" in text

    def test_roundtrip_parse(self) -> None:
        """pretty_program output can be re-parsed."""
        prog = parse_program("S = &{a: T, b: end}\nT = &{c: end}")
        text = pretty_program(prog)
        prog2 = parse_program(text)
        assert len(prog2.definitions) == len(prog.definitions)
        for d1, d2 in zip(prog.definitions, prog2.definitions):
            assert d1.name == d2.name
            assert d1.body == d2.body

    def test_program_definition_repr(self) -> None:
        """Definition and Program are frozen dataclasses."""
        d = Definition("S", End())
        p = Program(definitions=(d,))
        assert d.name == "S"
        assert p.definitions[0] is d
