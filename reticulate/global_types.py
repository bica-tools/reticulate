"""Multiparty session types: global type AST, parser, and state-space (Step 11).

A global type describes a protocol between multiple roles (participants).
Each interaction is annotated with sender and receiver roles:

    G  ::=  sender -> receiver : { m₁ : G₁ , … , mₙ : Gₙ }
         |  G₁ || G₂
         |  rec X . G
         |  X
         |  end

Projection onto a role yields a local (binary) session type — see projection.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Union


# ---------------------------------------------------------------------------
# AST nodes (frozen dataclasses, hashable)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GEnd:
    """Global end — all roles have terminated."""


@dataclass(frozen=True)
class GMessage:
    """Role-to-role interaction: sender -> receiver : { m₁: G₁, …, mₙ: Gₙ }.

    The sender makes an internal choice among the labels; the receiver
    observes the choice as an external choice.
    """
    sender: str
    receiver: str
    choices: tuple[tuple[str, "GlobalType"], ...]


@dataclass(frozen=True)
class GParallel:
    """Parallel composition of global types: G₁ || G₂."""
    left: "GlobalType"
    right: "GlobalType"


@dataclass(frozen=True)
class GRec:
    """Recursive global type: rec X . G."""
    var: str
    body: "GlobalType"


@dataclass(frozen=True)
class GVar:
    """Global type variable reference."""
    name: str


GlobalType = Union[GEnd, GMessage, GParallel, GRec, GVar]


# ---------------------------------------------------------------------------
# Role extraction
# ---------------------------------------------------------------------------

def roles(g: GlobalType) -> frozenset[str]:
    """Extract the set of role names from a global type."""
    match g:
        case GEnd():
            return frozenset()
        case GVar():
            return frozenset()
        case GMessage(sender=s, receiver=r, choices=cs):
            child_roles = frozenset()
            for _, body in cs:
                child_roles = child_roles | roles(body)
            return frozenset({s, r}) | child_roles
        case GParallel(left=left, right=right):
            return roles(left) | roles(right)
        case GRec(body=body):
            return roles(body)
        case _:
            raise TypeError(f"unknown global type node: {type(g).__name__}")


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------

def pretty_global(g: GlobalType) -> str:
    """Pretty-print a global type."""
    match g:
        case GEnd():
            return "end"
        case GVar(name=name):
            return name
        case GMessage(sender=s, receiver=r, choices=cs):
            parts = [f"{l}: {pretty_global(body)}" for l, body in cs]
            return f"{s} -> {r} : {{{', '.join(parts)}}}"
        case GParallel(left=left, right=right):
            return f"({pretty_global(left)} || {pretty_global(right)})"
        case GRec(var=v, body=body):
            return f"rec {v} . {pretty_global(body)}"
        case _:
            raise TypeError(f"unknown global type node: {type(g).__name__}")


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class _TokenKind(Enum):
    ARROW = auto()       # ->
    COLON = auto()       # :
    COMMA = auto()       # ,
    LBRACE = auto()      # {
    RBRACE = auto()      # }
    LPAREN = auto()      # (
    RPAREN = auto()      # )
    PAR = auto()         # ||
    DOT = auto()         # .
    IDENT = auto()       # identifier
    END = auto()         # 'end' keyword
    REC = auto()         # 'rec' keyword
    EOF = auto()


@dataclass
class _Token:
    kind: _TokenKind
    value: str
    pos: int


_KEYWORDS = {"end": _TokenKind.END, "rec": _TokenKind.REC}


def tokenize_global(source: str) -> list[_Token]:
    """Tokenize a global type string."""
    tokens: list[_Token] = []
    i = 0
    while i < len(source):
        c = source[i]
        if c.isspace():
            i += 1
            continue
        if c == '-' and i + 1 < len(source) and source[i + 1] == '>':
            tokens.append(_Token(_TokenKind.ARROW, "->", i))
            i += 2
        elif c == '|' and i + 1 < len(source) and source[i + 1] == '|':
            tokens.append(_Token(_TokenKind.PAR, "||", i))
            i += 2
        elif c == ':':
            tokens.append(_Token(_TokenKind.COLON, ":", i))
            i += 1
        elif c == ',':
            tokens.append(_Token(_TokenKind.COMMA, ",", i))
            i += 1
        elif c == '{':
            tokens.append(_Token(_TokenKind.LBRACE, "{", i))
            i += 1
        elif c == '}':
            tokens.append(_Token(_TokenKind.RBRACE, "}", i))
            i += 1
        elif c == '(':
            tokens.append(_Token(_TokenKind.LPAREN, "(", i))
            i += 1
        elif c == ')':
            tokens.append(_Token(_TokenKind.RPAREN, ")", i))
            i += 1
        elif c == '.':
            tokens.append(_Token(_TokenKind.DOT, ".", i))
            i += 1
        elif c.isalpha() or c == '_':
            start = i
            while i < len(source) and (source[i].isalnum() or source[i] == '_'):
                i += 1
            word = source[start:i]
            kind = _KEYWORDS.get(word, _TokenKind.IDENT)
            tokens.append(_Token(kind, word, start))
        else:
            raise GlobalParseError(f"unexpected character {c!r} at position {i}")
    tokens.append(_Token(_TokenKind.EOF, "", len(source)))
    return tokens


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class GlobalParseError(Exception):
    """Raised when a global type string cannot be parsed."""


def parse_global(source: str) -> GlobalType:
    """Parse a global type string into an AST."""
    tokens = tokenize_global(source)
    parser = _GlobalParser(tokens, source)
    result = parser._parse_global()
    parser._expect(_TokenKind.EOF)
    return result


class _GlobalParser:
    """Recursive-descent parser for global types."""

    def __init__(self, tokens: list[_Token], source: str) -> None:
        self._tokens = tokens
        self._source = source
        self._pos = 0

    def _peek(self) -> _Token:
        return self._tokens[self._pos]

    def _advance(self) -> _Token:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, kind: _TokenKind) -> _Token:
        tok = self._advance()
        if tok.kind != kind:
            raise GlobalParseError(
                f"expected {kind.name}, got {tok.kind.name} ({tok.value!r}) "
                f"at position {tok.pos}"
            )
        return tok

    def _parse_global(self) -> GlobalType:
        """Parse a global type, handling parallel at the lowest precedence."""
        left = self._parse_atom()
        while self._peek().kind == _TokenKind.PAR:
            self._advance()
            right = self._parse_atom()
            left = GParallel(left, right)
        return left

    def _parse_atom(self) -> GlobalType:
        tok = self._peek()

        if tok.kind == _TokenKind.END:
            self._advance()
            return GEnd()

        if tok.kind == _TokenKind.REC:
            self._advance()
            var_tok = self._expect(_TokenKind.IDENT)
            self._expect(_TokenKind.DOT)
            body = self._parse_global()
            return GRec(var_tok.value, body)

        if tok.kind == _TokenKind.LPAREN:
            self._advance()
            inner = self._parse_global()
            self._expect(_TokenKind.RPAREN)
            return inner

        if tok.kind == _TokenKind.IDENT:
            # Could be a variable reference (single ident) or
            # an interaction: sender -> receiver : { ... }
            if (self._pos + 1 < len(self._tokens)
                    and self._tokens[self._pos + 1].kind == _TokenKind.ARROW):
                return self._parse_message()
            else:
                self._advance()
                return GVar(tok.value)

        raise GlobalParseError(
            f"unexpected token {tok.kind.name} ({tok.value!r}) "
            f"at position {tok.pos}"
        )

    def _parse_message(self) -> GMessage:
        """Parse: sender -> receiver : { m₁: G₁, …, mₙ: Gₙ }"""
        sender_tok = self._expect(_TokenKind.IDENT)
        self._expect(_TokenKind.ARROW)
        receiver_tok = self._expect(_TokenKind.IDENT)
        self._expect(_TokenKind.COLON)
        self._expect(_TokenKind.LBRACE)

        choices: list[tuple[str, GlobalType]] = []
        if self._peek().kind != _TokenKind.RBRACE:
            choices.append(self._parse_choice())
            while self._peek().kind == _TokenKind.COMMA:
                self._advance()
                choices.append(self._parse_choice())

        self._expect(_TokenKind.RBRACE)
        return GMessage(sender_tok.value, receiver_tok.value, tuple(choices))

    def _parse_choice(self) -> tuple[str, GlobalType]:
        """Parse: label : G"""
        label_tok = self._expect(_TokenKind.IDENT)
        self._expect(_TokenKind.COLON)
        body = self._parse_global()
        return (label_tok.value, body)


# ---------------------------------------------------------------------------
# Global state-space construction
# ---------------------------------------------------------------------------

def build_global_statespace(g: GlobalType) -> "StateSpace":
    """Build the state space of a global type.

    Transition labels are role-annotated: "sender->receiver:method".
    The resulting StateSpace can be checked for lattice properties using
    the existing check_lattice() function.
    """
    from reticulate.statespace import StateSpace
    from reticulate.product import product_statespace

    builder = _GlobalBuilder()
    return builder.build(g)


class _GlobalBuilder:
    """Accumulates states and transitions for global state-space construction."""

    def __init__(self) -> None:
        self._next_id: int = 0
        self._states: dict[int, str] = {}
        self._transitions: list[tuple[int, str, int]] = []
        self._selection_transitions: set[tuple[int, str, int]] = set()

    def _fresh(self, label: str) -> int:
        sid = self._next_id
        self._next_id += 1
        self._states[sid] = label
        return sid

    def build(self, g: GlobalType) -> "StateSpace":
        from reticulate.statespace import StateSpace

        end_id = self._fresh("end")
        top_id = self._build(g, {}, end_id)

        reachable = self._reachable(top_id)
        reachable_transitions = [
            (s, l, t) for s, l, t in self._transitions if s in reachable
        ]
        reachable_selections = {
            (s, l, t) for s, l, t in self._selection_transitions if s in reachable
        }
        return StateSpace(
            states=reachable,
            transitions=reachable_transitions,
            top=top_id,
            bottom=end_id,
            labels={s: self._states[s] for s in reachable if s in self._states},
            selection_transitions=reachable_selections,
        )

    def _build(
        self,
        g: GlobalType,
        env: dict[str, int],
        end_id: int,
    ) -> int:
        match g:
            case GEnd():
                return end_id

            case GVar(name=name):
                if name not in env:
                    raise ValueError(f"unbound global type variable: {name!r}")
                return env[name]

            case GMessage(sender=s, receiver=r, choices=cs):
                if len(cs) == 1:
                    label = f"{s}->{r}:{cs[0][0]}"
                else:
                    labels_str = ",".join(l for l, _ in cs)
                    label = f"{s}->{r}:{{{labels_str}}}"
                entry = self._fresh(label)
                for lbl, body in cs:
                    target = self._build(body, env, end_id)
                    tr_label = f"{s}->{r}:{lbl}"
                    self._transitions.append((entry, tr_label, target))
                return entry

            case GParallel(left=left, right=right):
                return self._build_parallel(left, right, end_id)

            case GRec(var=var, body=body):
                placeholder = self._fresh(f"rec_{var}")
                new_env = {**env, var: placeholder}
                body_entry = self._build(body, new_env, end_id)
                if body_entry != placeholder:
                    self._merge(placeholder, body_entry)
                    return body_entry
                return placeholder

            case _:
                raise TypeError(
                    f"unknown global type node: {type(g).__name__}"
                )

    def _build_parallel(
        self,
        left: GlobalType,
        right: GlobalType,
        end_id: int,
    ) -> int:
        from reticulate.product import product_statespace

        left_ss = build_global_statespace(left)
        right_ss = build_global_statespace(right)
        prod = product_statespace(left_ss, right_ss)

        remap: dict[int, int] = {}
        for sid in prod.states:
            if sid == prod.bottom:
                remap[sid] = end_id
            else:
                remap[sid] = self._fresh(prod.labels.get(sid, "?"))

        for src, lbl, tgt in prod.transitions:
            remapped = (remap[src], lbl, remap[tgt])
            self._transitions.append(remapped)
            if prod.is_selection(src, lbl, tgt):
                self._selection_transitions.add(remapped)

        return remap[prod.top]

    def _merge(self, old_id: int, new_id: int) -> None:
        self._transitions = [
            (
                new_id if s == old_id else s,
                l,
                new_id if t == old_id else t,
            )
            for s, l, t in self._transitions
        ]
        self._selection_transitions = {
            (
                new_id if s == old_id else s,
                l,
                new_id if t == old_id else t,
            )
            for s, l, t in self._selection_transitions
        }
        if old_id in self._states:
            del self._states[old_id]

    def _reachable(self, start: int) -> set[int]:
        visited: set[int] = set()
        stack = [start]
        while stack:
            s = stack.pop()
            if s in visited:
                continue
            visited.add(s)
            for src, _, tgt in self._transitions:
                if src == s:
                    stack.append(tgt)
        return visited
