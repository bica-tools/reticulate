"""Name resolution for the equation grammar.

Transforms a Program (named definitions) into a single SessionType AST
by resolving name references. Self-recursive and mutually recursive
definitions are wrapped with Rec nodes.
"""

from __future__ import annotations

from reticulate.parser import (
    Branch,
    Continuation,
    Definition,
    End,
    Parallel,
    Program,
    Rec,
    Select,
    SessionType,
    Var,
    Wait,
)


class ResolveError(Exception):
    """Raised on unresolvable name references or duplicate definitions."""
    pass


def resolve(program: Program) -> SessionType:
    """Resolve a Program into a single SessionType AST.

    1. Check for duplicate definition names.
    2. For each Var(name) in bodies: if name is a definition, it's a reference.
    3. Non-recursive references: inline by substitution.
    4. Self-recursive references: wrap with Rec(name, body).
    5. Mutual recursion: expand by substitution + Rec wrapping.
    6. Return the resolved AST for the first definition.
    """
    if not program.definitions:
        raise ResolveError("empty program: no definitions to resolve")

    # Check for duplicate definition names
    seen: dict[str, int] = {}
    for i, defn in enumerate(program.definitions):
        if defn.name in seen:
            raise ResolveError(
                f"duplicate definition: {defn.name!r} "
                f"(first at index {seen[defn.name]}, again at {i})"
            )
        seen[defn.name] = i

    env: dict[str, SessionType] = {d.name: d.body for d in program.definitions}
    def_names = set(env.keys())

    entry_name = program.definitions[0].name

    # Check for unresolved names in all bodies
    for defn in program.definitions:
        _check_unresolved(defn.body, def_names, set())

    # Resolve the entry definition
    result = _resolve_name(entry_name, env, def_names, set())
    return result


def _check_unresolved(
    node: SessionType,
    def_names: set[str],
    rec_bound: set[str],
) -> None:
    """Check that all Var references are either rec-bound or definition names."""
    match node:
        case End() | Wait():
            pass
        case Var(name=name):
            if name not in def_names and name not in rec_bound:
                raise ResolveError(f"unbound type variable: {name!r}")
        case Branch(choices=choices) | Select(choices=choices):
            for _, body in choices:
                _check_unresolved(body, def_names, rec_bound)
        case Parallel(branches=branches):
            for b in branches:
                _check_unresolved(b, def_names, rec_bound)
        case Rec(var=var, body=body):
            _check_unresolved(body, def_names, rec_bound | {var})
        case Continuation(left=left, right=right):
            _check_unresolved(left, def_names, rec_bound)
            _check_unresolved(right, def_names, rec_bound)


def _resolve_name(
    name: str,
    env: dict[str, SessionType],
    def_names: set[str],
    expanding: set[str],
) -> SessionType:
    """Resolve a single definition name into a SessionType AST.

    *expanding* tracks the set of names currently being expanded to detect
    recursion.
    """
    if name not in env:
        raise ResolveError(f"undefined name: {name!r}")

    body = env[name]

    # Find which definition names appear free in this body (directly)
    free = _free_def_names(body, def_names, set())

    # Check if self-recursive (directly or via mutual recursion)
    is_self_recursive = name in free

    # Also check if any referenced definition eventually leads back to name
    # (mutual recursion detection)
    is_mutually_recursive = _reaches(name, env, def_names)

    if is_self_recursive or is_mutually_recursive:
        # Wrap with Rec: keep Var(name) as the recursion variable
        # Resolve all OTHER references in the body
        resolved_body = _resolve_body(body, name, env, def_names, expanding | {name})
        return Rec(name, resolved_body)
    else:
        # Not recursive: substitute all definition references
        resolved_body = _resolve_body(body, name, env, def_names, expanding)
        return resolved_body


def _reaches(name: str, env: dict[str, SessionType], def_names: set[str]) -> bool:
    """Check if definition *name* is part of a mutual recursion cycle."""
    # BFS/DFS from name through definition references
    visited: set[str] = set()
    stack = list(_free_def_names(env[name], def_names, set()) - {name})

    while stack:
        current = stack.pop()
        if current == name:
            return True
        if current in visited or current not in env:
            continue
        visited.add(current)
        refs = _free_def_names(env[current], def_names, set())
        for ref in refs:
            if ref not in visited:
                stack.append(ref)
    return False


def _resolve_body(
    node: SessionType,
    current_name: str,
    env: dict[str, SessionType],
    def_names: set[str],
    expanding: set[str],
) -> SessionType:
    """Recursively resolve definition references in *node*.

    *current_name* is the definition currently being resolved (for self-recursion).
    *expanding* is the set of names in the expansion stack (for mutual recursion).
    """
    match node:
        case End() | Wait():
            return node

        case Var(name=vname):
            if vname not in def_names:
                # Not a definition reference -- leave as-is (rec variable or unbound)
                return node
            if vname == current_name:
                # Self-reference -- keep as Var for the enclosing Rec
                return node
            if vname in expanding:
                # Mutual recursion: this name is being expanded up the call stack.
                # Keep as Var -- the enclosing Rec for the mutually recursive
                # definition will bind it.
                return Var(vname)
            # Non-recursive reference: inline it
            return _resolve_name(vname, env, def_names, expanding)

        case Branch(choices=choices):
            new_choices = tuple(
                (label, _resolve_body(body, current_name, env, def_names, expanding))
                for label, body in choices
            )
            return Branch(new_choices)

        case Select(choices=choices):
            new_choices = tuple(
                (label, _resolve_body(body, current_name, env, def_names, expanding))
                for label, body in choices
            )
            return Select(new_choices)

        case Parallel(branches=branches):
            new_branches = tuple(
                _resolve_body(b, current_name, env, def_names, expanding)
                for b in branches
            )
            return Parallel(new_branches)

        case Rec(var=var, body=body):
            # The rec-bound variable shadows any definition with the same name
            inner_defs = def_names - {var}
            resolved = _resolve_body(body, current_name, env, inner_defs, expanding)
            return Rec(var, resolved)

        case Continuation(left=left, right=right):
            new_left = _resolve_body(left, current_name, env, def_names, expanding)
            new_right = _resolve_body(right, current_name, env, def_names, expanding)
            return Continuation(new_left, new_right)

        case _:
            raise TypeError(f"unknown AST node: {type(node).__name__}")


def _free_def_names(
    node: SessionType,
    def_names: set[str],
    rec_bound: set[str],
) -> set[str]:
    """Collect definition names that appear free (not bound by rec) in *node*."""
    match node:
        case End() | Wait():
            return set()

        case Var(name=name):
            if name in def_names and name not in rec_bound:
                return {name}
            return set()

        case Branch(choices=choices) | Select(choices=choices):
            result: set[str] = set()
            for _, body in choices:
                result |= _free_def_names(body, def_names, rec_bound)
            return result

        case Parallel(branches=branches):
            result = set()
            for b in branches:
                result |= _free_def_names(b, def_names, rec_bound)
            return result

        case Rec(var=var, body=body):
            return _free_def_names(body, def_names, rec_bound | {var})

        case Continuation(left=left, right=right):
            return (
                _free_def_names(left, def_names, rec_bound)
                | _free_def_names(right, def_names, rec_bound)
            )

        case _:
            return set()
