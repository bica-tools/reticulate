"""Session type parser: AST nodes, tokenizer, recursive-descent parser, pretty-printer.

Grammar
-------

::

    S  ::=  &{ m₁ : S₁ , … , mₙ : Sₙ }    -- branch (external choice)
         |  +{ l₁ : S₁ , … , lₙ : Sₙ }    -- selection (internal choice)
         |  S₁ || S₂                        -- parallel
         |  S₁ . S₂                          -- sequencing
         |  rec X . S                        -- recursion
         |  ( S )                            -- grouping
         |  X                                -- type variable
         |  end                              -- terminated

Precedence (tightest first): ``.`` > ``||``.

Desugaring
~~~~~~~~~~
When the left-hand side of ``.`` is a bare identifier, the parser desugars
it into a single-method ``Branch``::

    ident . S   →   Branch(((ident, S),))

This means ``a . b . end`` is equivalent to ``&{a: &{b: end}}``: plain
method sequencing *is* branch with one arm.

Why ``Sequence`` exists
~~~~~~~~~~~~~~~~~~~~~~~
Given the desugaring above, one might ask whether ``Sequence`` is redundant.
It is not.  The desugaring only applies when the left operand is a bare
identifier (a method name).  When a *compound* expression appears on the
left of ``.``, there is no method name to form a ``Branch`` from, so the
parser produces a ``Sequence`` node instead.

The canonical case is a parallel block followed by a continuation::

    (read . end || write . end) . close . end

Here the parallel fork-join must complete before ``close`` can be called.
There is no single method name for the left side — it is a full ``Parallel``
sub-expression — so ``Sequence(Parallel(…), Branch(("close", End())))`` is
the only faithful representation.  7 of the 34 benchmark protocols rely on
this pattern (File Channel, Reticulate Pipeline, GitHub CI Workflow,
MQTT Client, Saga Orchestrator, Leader Replication, Ion Channel Na+/K+).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Union


# ---------------------------------------------------------------------------
# AST nodes (frozen dataclasses, hashable)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class End:
    """Terminated session."""


@dataclass(frozen=True)
class Var:
    """Type variable reference (e.g. ``X`` inside ``rec X . …``)."""
    name: str


@dataclass(frozen=True)
class Branch:
    """External choice ``&{ m₁ : S₁ , … , mₙ : Sₙ }``."""
    choices: tuple[tuple[str, "SessionType"], ...]


@dataclass(frozen=True)
class Select:
    """Internal choice ``+{ l₁ : S₁ , … , lₙ : Sₙ }``."""
    choices: tuple[tuple[str, "SessionType"], ...]


@dataclass(frozen=True)
class Parallel:
    """Parallel composition ``S₁ || S₂``."""
    left: "SessionType"
    right: "SessionType"


@dataclass(frozen=True)
class Rec:
    """Recursive type ``rec X . S``."""
    var: str
    body: "SessionType"


@dataclass(frozen=True)
class Sequence:
    """Sequential composition ``S₁ . S₂`` for compound left-hand sides.

    When the left operand of ``.`` is a bare identifier, the parser desugars
    it to a single-method ``Branch`` (so ``a . S`` becomes ``&{a: S}``).
    ``Sequence`` is produced only when the left operand is a compound
    expression with no method name to branch on — typically a ``Parallel``
    block followed by a continuation, e.g. ``(S₁ || S₂) . close . end``.
    """
    left: "SessionType"
    right: "SessionType"


SessionType = Union[End, Var, Branch, Select, Parallel, Rec, Sequence]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class TokenKind(Enum):
    LBRACE = auto()    # {
    RBRACE = auto()    # }
    LPAREN = auto()    # (
    RPAREN = auto()    # )
    AMPERSAND = auto() # &
    PLUS = auto()      # +
    COLON = auto()     # :
    COMMA = auto()     # ,
    DOT = auto()       # .
    PAR = auto()       # ||
    IDENT = auto()     # identifier or keyword (end, rec)
    EOF = auto()


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    value: str
    pos: int


class ParseError(Exception):
    """Raised on invalid session-type syntax, with character position."""

    def __init__(self, message: str, pos: int | None = None) -> None:
        self.pos = pos
        if pos is not None:
            message = f"at position {pos}: {message}"
        super().__init__(message)


def tokenize(source: str) -> list[Token]:
    """Scan *source* into a list of ``Token``s (including a trailing ``EOF``)."""
    tokens: list[Token] = []
    i = 0
    n = len(source)

    while i < n:
        ch = source[i]

        # skip whitespace
        if ch in " \t\n\r":
            i += 1
            continue

        # two-character token: ||
        if ch == "|" and i + 1 < n and source[i + 1] == "|":
            tokens.append(Token(TokenKind.PAR, "||", i))
            i += 2
            continue

        # single-character tokens
        single = {
            "{": TokenKind.LBRACE,
            "}": TokenKind.RBRACE,
            "(": TokenKind.LPAREN,
            ")": TokenKind.RPAREN,
            "&": TokenKind.AMPERSAND,
            "+": TokenKind.PLUS,
            ":": TokenKind.COLON,
            ",": TokenKind.COMMA,
            ".": TokenKind.DOT,
        }
        if ch in single:
            tokens.append(Token(single[ch], ch, i))
            i += 1
            continue

        # Unicode alternatives (checked before general identifier scan
        # because μ, ⊕, ∥ all satisfy ch.isalpha() in Python)
        if ch == "\u2295":  # ⊕
            tokens.append(Token(TokenKind.PLUS, "\u2295", i))
            i += 1
            continue
        if ch == "\u2225":  # ∥
            tokens.append(Token(TokenKind.PAR, "\u2225", i))
            i += 1
            continue
        if ch == "\u03bc":  # μ
            tokens.append(Token(TokenKind.IDENT, "rec", i))
            i += 1
            continue

        # identifiers: [A-Za-z_][A-Za-z0-9_]*
        if ch.isalpha() or ch == "_":
            start = i
            while i < n and (source[i].isalnum() or source[i] == "_"):
                i += 1
            word = source[start:i]
            tokens.append(Token(TokenKind.IDENT, word, start))
            continue

        raise ParseError(f"unexpected character {ch!r}", i)

    tokens.append(Token(TokenKind.EOF, "", n))
    return tokens


# ---------------------------------------------------------------------------
# Recursive-descent parser
# ---------------------------------------------------------------------------

class _Parser:
    """Internal parser state."""

    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    # -- helpers -------------------------------------------------------------

    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, kind: TokenKind, context: str = "") -> Token:
        tok = self._peek()
        if tok.kind is not kind:
            ctx = f" ({context})" if context else ""
            raise ParseError(
                f"expected {kind.name}, got {tok.kind.name} ({tok.value!r}){ctx}",
                tok.pos,
            )
        return self._advance()

    # -- grammar rules -------------------------------------------------------

    def parse(self) -> SessionType:
        result = self._seq_expr()
        if self._peek().kind is not TokenKind.EOF:
            tok = self._peek()
            raise ParseError(
                f"unexpected token {tok.kind.name} ({tok.value!r}) after expression",
                tok.pos,
            )
        return result

    def _seq_expr(self) -> SessionType:
        """Parse a sequence expression (right-associative ``.`` and ``||`` operators).

        If the left operand is a bare identifier we desugar ``ident . S``
        to ``Branch(((ident, S),))``.  Otherwise we produce a ``Sequence``
        node.

        ``||`` binds looser than ``.`` — ``a.b || c.d`` means ``(a.b) || (c.d)``.
        """
        left = self._dot_expr()

        # Infix || (parallel) — no parentheses required
        if self._peek().kind is TokenKind.PAR:
            self._advance()  # consume '||'
            right = self._seq_expr()  # right-associative
            return Parallel(left, right)

        return left

    def _dot_expr(self) -> SessionType:
        """Parse dot-sequencing (right-associative ``.`` operator).

        Binds tighter than ``||``.
        """
        left = self._atom()

        if self._peek().kind is TokenKind.DOT:
            self._advance()  # consume '.'
            right = self._dot_expr()  # right-associative

            # Desugaring: bare identifier becomes single-method Branch
            if isinstance(left, Var):
                return Branch(((left.name, right),))
            return Sequence(left, right)

        return left

    def _atom(self) -> SessionType:
        """Parse a self-delimiting construct."""
        tok = self._peek()

        # &{ ... }
        if tok.kind is TokenKind.AMPERSAND:
            return self._branch()

        # +{ ... } or ⊕{ ... }
        if tok.kind is TokenKind.PLUS:
            return self._select()

        # ( ... )  — grouping
        if tok.kind is TokenKind.LPAREN:
            return self._paren()

        # rec X . S
        if tok.kind is TokenKind.IDENT and tok.value == "rec":
            return self._rec()

        # end
        if tok.kind is TokenKind.IDENT and tok.value == "end":
            self._advance()
            return End()

        # plain identifier (type variable or method name)
        if tok.kind is TokenKind.IDENT:
            self._advance()
            return Var(tok.value)

        raise ParseError(
            f"unexpected token {tok.kind.name} ({tok.value!r})", tok.pos
        )

    def _choice_list(self, context: str) -> tuple[tuple[str, SessionType], ...]:
        """Parse ``m₁ : S₁ , … , mₙ : Sₙ`` inside braces."""
        entries: list[tuple[str, SessionType]] = []
        while True:
            label_tok = self._expect(TokenKind.IDENT, f"{context} label")
            self._expect(TokenKind.COLON, f"{context} colon after {label_tok.value!r}")
            body = self._seq_expr()
            entries.append((label_tok.value, body))
            if self._peek().kind is TokenKind.COMMA:
                self._advance()
            else:
                break
        return tuple(entries)

    def _branch(self) -> Branch:
        self._advance()  # consume '&'
        self._expect(TokenKind.LBRACE, "branch")
        if self._peek().kind is TokenKind.RBRACE:
            raise ParseError("branch must have at least one choice", self._peek().pos)
        choices = self._choice_list("branch")
        self._expect(TokenKind.RBRACE, "branch closing")
        return Branch(choices)

    def _select(self) -> Select:
        self._advance()  # consume '+'
        self._expect(TokenKind.LBRACE, "select")
        if self._peek().kind is TokenKind.RBRACE:
            raise ParseError("select must have at least one choice", self._peek().pos)
        choices = self._choice_list("select")
        self._expect(TokenKind.RBRACE, "select closing")
        return Select(choices)

    def _paren(self) -> SessionType:
        """Parse a parenthesized expression ``( S )``."""
        self._advance()  # consume '('
        inner = self._seq_expr()
        self._expect(TokenKind.RPAREN, "closing ')'")
        return inner

    def _rec(self) -> Rec:
        self._advance()  # consume 'rec'
        var_tok = self._expect(TokenKind.IDENT, "recursion variable")
        if var_tok.value in ("rec", "end"):
            raise ParseError(
                f"{var_tok.value!r} is a keyword, not a valid variable name",
                var_tok.pos,
            )
        self._expect(TokenKind.DOT, "recursion dot")
        body = self._dot_expr()
        return Rec(var_tok.value, body)


def parse(source: str) -> SessionType:
    """Parse a session-type string into an AST.

    Raises ``ParseError`` on invalid syntax.

    Examples::

        >>> parse("end")
        End()
        >>> parse("m . end")
        Branch(choices=(('m', End()),))
    """
    tokens = tokenize(source)
    return _Parser(tokens).parse()


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------

def pretty(node: SessionType) -> str:
    """Render an AST back to a human-readable session-type string.

    Single-element ``Branch`` is printed as sequencing sugar (``m . S``).
    Parentheses are added around ``||`` only when needed for correct
    precedence (inside ``.`` or ``rec``).
    """
    return _pretty(node, in_tight=False)


def _pretty(node: SessionType, *, in_tight: bool) -> str:
    """Internal pretty-printer.

    *in_tight* is ``True`` when we are inside a context that binds tighter
    than ``||`` (i.e. inside ``.`` sequencing or ``rec`` body).  In that
    case a ``Parallel`` node needs parentheses to roundtrip correctly.
    """
    match node:
        case End():
            return "end"
        case Var(name=name):
            return name
        case Branch(choices=choices) if len(choices) == 1:
            label, body = choices[0]
            return f"{label} . {_pretty(body, in_tight=True)}"
        case Branch(choices=choices):
            inner = ", ".join(
                f"{label}: {_pretty(body, in_tight=False)}"
                for label, body in choices
            )
            return f"&{{{inner}}}"
        case Select(choices=choices):
            inner = ", ".join(
                f"{label}: {_pretty(body, in_tight=False)}"
                for label, body in choices
            )
            return f"+{{{inner}}}"
        case Parallel(left=left, right=right):
            s = f"{_pretty(left, in_tight=False)} || {_pretty(right, in_tight=False)}"
            return f"({s})" if in_tight else s
        case Rec(var=var, body=body):
            return f"rec {var} . {_pretty(body, in_tight=True)}"
        case Sequence(left=left, right=right):
            return f"{_pretty(left, in_tight=True)} . {_pretty(right, in_tight=True)}"
        case _:
            raise TypeError(f"unknown AST node: {type(node).__name__}")
