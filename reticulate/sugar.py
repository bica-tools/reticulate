"""Desugaring and ensugaring of session type ASTs.

desugar(node) — normalizes a session type to the core grammar.
  Currently identity since the Python parser produces core ASTs.

ensugar(node) — returns a pretty-printed string with syntactic sugar applied.
  Single-method Branch(&{m: S}) is rendered as m . S.
"""

from reticulate.parser import (
    SessionType, End, Wait, Var, Branch, Select, Parallel, Rec, Continuation,
    _pretty,
)


def desugar(node: SessionType) -> SessionType:
    """Normalize an AST to the core grammar (recursive identity walk)."""
    match node:
        case End() | Wait() | Var():
            return node
        case Branch(choices=choices):
            return Branch(tuple((label, desugar(body)) for label, body in choices))
        case Select(choices=choices):
            return Select(tuple((label, desugar(body)) for label, body in choices))
        case Parallel(branches=branches):
            return Parallel(tuple(desugar(b) for b in branches))
        case Rec(var=var, body=body):
            return Rec(var, desugar(body))
        case Continuation(left=left, right=right):
            return Continuation(desugar(left), desugar(right))
        case _:
            raise TypeError(f"unknown AST node: {type(node).__name__}")


def ensugar(node: SessionType) -> str:
    """Pretty-print with syntactic sugar: single-method Branch becomes m . S."""
    return _ensugar(node, in_tight=False)


def _ensugar(node: SessionType, *, in_tight: bool) -> str:
    match node:
        case End():
            return "end"
        case Wait():
            return "wait"
        case Var(name=name):
            return name
        case Branch(choices=choices) if len(choices) == 1:
            label, body = choices[0]
            return f"{label} . {_ensugar(body, in_tight=in_tight)}"
        case Branch(choices=choices):
            inner = ", ".join(
                f"{label}: {_ensugar(body, in_tight=False)}"
                for label, body in choices
            )
            return f"&{{{inner}}}"
        case Select(choices=choices):
            inner = ", ".join(
                f"{label}: {_ensugar(body, in_tight=False)}"
                for label, body in choices
            )
            return f"+{{{inner}}}"
        case Parallel(branches=branches):
            s = " || ".join(_ensugar(b, in_tight=False) for b in branches)
            return f"({s})" if in_tight else s
        case Rec(var=var, body=body):
            return f"rec {var} . {_ensugar(body, in_tight=True)}"
        case Continuation(left=left, right=right):
            return f"{_ensugar(left, in_tight=True)} . {_ensugar(right, in_tight=True)}"
        case _:
            raise TypeError(f"unknown AST node: {type(node).__name__}")
