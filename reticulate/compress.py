"""Session type compression: tree-to-equation transformation.

Detects shared (structurally equal) subtrees in a session type AST and
factors them into named equations, producing a Program.  This is the
inverse of ``resolve.resolve()``.

The compression is semantic-preserving:
    resolve(compress(S)) ≅ S    (alpha-equivalent)
    L(S) ≅ L(build_program_statespace(compress(S)))    (isomorphic state spaces)

Shared subtrees correspond to reconvergence points in the state space.
The number of shared equations is an upper bound on the reconvergence
degree ρ(S).

Usage:
    from reticulate.compress import compress, compression_ratio
    prog = compress(ast)
    print(pretty_program(prog))
    print(f"Compression ratio: {compression_ratio(ast):.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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

# Minimum subtree size worth factoring out.
# Leaf nodes (End, Wait, Var) are size 1 — never worth naming.
_MIN_SHARED_SIZE = 3


# ---------------------------------------------------------------------------
# AST size
# ---------------------------------------------------------------------------

def ast_size(node: SessionType) -> int:
    """Count the number of AST nodes in a session type."""
    match node:
        case End() | Wait() | Var():
            return 1
        case Branch(choices=choices) | Select(choices=choices):
            return 1 + sum(ast_size(body) for _, body in choices)
        case Parallel(branches=branches):
            return 1 + sum(ast_size(b) for b in branches)
        case Rec(body=body):
            return 1 + ast_size(body)
        case Continuation(left=left, right=right):
            return 1 + ast_size(left) + ast_size(right)
        case _:
            return 1


# ---------------------------------------------------------------------------
# Subtree collection
# ---------------------------------------------------------------------------

def _collect_subtrees(
    node: SessionType,
    counts: dict[SessionType, int],
) -> None:
    """Walk the AST and count occurrences of each structurally-equal subtree."""
    # Increment count for this node
    counts[node] = counts.get(node, 0) + 1

    # Recurse into children
    match node:
        case End() | Wait() | Var():
            pass
        case Branch(choices=choices) | Select(choices=choices):
            for _, body in choices:
                _collect_subtrees(body, counts)
        case Parallel(branches=branches):
            for b in branches:
                _collect_subtrees(b, counts)
        case Rec(body=body):
            _collect_subtrees(body, counts)
        case Continuation(left=left, right=right):
            _collect_subtrees(left, counts)
            _collect_subtrees(right, counts)


# ---------------------------------------------------------------------------
# Name generation
# ---------------------------------------------------------------------------

def _generate_name(index: int) -> str:
    """Generate a definition name from an index: S0, S1, ..., S25, S26, ..."""
    return f"S{index}"


# ---------------------------------------------------------------------------
# Replacement
# ---------------------------------------------------------------------------

def _replace_shared(
    node: SessionType,
    shared_map: dict[SessionType, str],
) -> SessionType:
    """Replace shared subtrees with Var references to their equation names.

    We check children bottom-up: if a subtree matches a shared entry,
    replace it with Var(name).  We do NOT recurse into a subtree that
    has already been replaced (its body will be in its own definition).
    """
    # If this exact node is shared, replace with a reference
    if node in shared_map:
        return Var(shared_map[node])

    # Otherwise recurse into children
    match node:
        case End() | Wait() | Var():
            return node

        case Branch(choices=choices):
            new_choices = tuple(
                (label, _replace_shared(body, shared_map))
                for label, body in choices
            )
            return Branch(new_choices)

        case Select(choices=choices):
            new_choices = tuple(
                (label, _replace_shared(body, shared_map))
                for label, body in choices
            )
            return Select(new_choices)

        case Parallel(branches=branches):
            new_branches = tuple(
                _replace_shared(b, shared_map)
                for b in branches
            )
            return Parallel(new_branches)

        case Rec(var=var, body=body):
            return Rec(var, _replace_shared(body, shared_map))

        case Continuation(left=left, right=right):
            return Continuation(
                _replace_shared(left, shared_map),
                _replace_shared(right, shared_map),
            )

        case _:
            return node


# ---------------------------------------------------------------------------
# Dominance filtering
# ---------------------------------------------------------------------------

def _filter_dominated(
    shared: dict[SessionType, str],
) -> dict[SessionType, str]:
    """Remove shared subtrees that are children of other shared subtrees.

    If S1 contains S2, and both are shared, we only need to name S1.
    S2 will be factored out when we process S1's body.

    We keep the LARGEST shared subtrees (by ast_size) and remove any
    that are proper subtrees of a larger shared entry.
    """
    # Sort by size descending — largest first
    entries = sorted(shared.items(), key=lambda kv: ast_size(kv[0]), reverse=True)

    kept: dict[SessionType, str] = {}
    kept_nodes: list[SessionType] = []

    for node, name in entries:
        # Check if this node is a proper subtree of any already-kept node
        is_dominated = False
        for parent in kept_nodes:
            if _contains(parent, node) and parent != node:
                is_dominated = True
                break

        if not is_dominated:
            kept[node] = name
            kept_nodes.append(node)

    return kept


def _contains(parent: SessionType, target: SessionType) -> bool:
    """Check if target appears as a proper subtree of parent."""
    if parent == target:
        return False  # Not a PROPER subtree

    match parent:
        case End() | Wait() | Var():
            return False
        case Branch(choices=choices) | Select(choices=choices):
            return any(
                body == target or _contains(body, target)
                for _, body in choices
            )
        case Parallel(branches=branches):
            return any(
                b == target or _contains(b, target)
                for b in branches
            )
        case Rec(body=body):
            return body == target or _contains(body, target)
        case Continuation(left=left, right=right):
            return (
                left == target or _contains(left, target)
                or right == target or _contains(right, target)
            )
        case _:
            return False


# ---------------------------------------------------------------------------
# Core: compress
# ---------------------------------------------------------------------------

def compress(
    ast: SessionType,
    *,
    min_size: int = _MIN_SHARED_SIZE,
    entry_name: str = "Main",
) -> Program:
    """Compress a tree-like session type into an equation system.

    Detects structurally equal subtrees that appear more than once
    and factors them into named definitions.

    Args:
        ast: The session type AST to compress.
        min_size: Minimum AST node count for a subtree to be worth naming.
            Default 3 (leaf nodes are never factored out).
        entry_name: Name for the top-level definition. Default "Main".

    Returns:
        A Program with named definitions. The first definition is the
        entry point. Subsequent definitions represent shared subtrees.
    """
    # Step 1: Count all subtree occurrences
    counts: dict[SessionType, int] = {}
    _collect_subtrees(ast, counts)

    # Step 2: Find subtrees that appear 2+ times and are large enough
    candidates: list[SessionType] = [
        node for node, count in counts.items()
        if count >= 2 and ast_size(node) >= min_size
    ]

    # Step 3: Sort by size descending (name largest first)
    candidates.sort(key=ast_size, reverse=True)

    # Step 4: Assign names
    preliminary: dict[SessionType, str] = {}
    for i, node in enumerate(candidates):
        preliminary[node] = _generate_name(i)

    # Step 5: Filter dominated subtrees
    shared_map = _filter_dominated(preliminary)

    # Step 6: If nothing to share, return a trivial program
    if not shared_map:
        return Program(definitions=(Definition(entry_name, ast),))

    # Step 7: Build definitions — shared subtrees get their own definitions
    # Process shared bodies: replace any nested shared subtrees with Var refs
    definitions: list[Definition] = []

    # Main definition: replace shared subtrees in the original AST
    main_body = _replace_shared(ast, shared_map)
    definitions.append(Definition(entry_name, main_body))

    # Shared definitions: each shared subtree becomes a definition
    # Sort by name for deterministic output
    for node, name in sorted(shared_map.items(), key=lambda kv: kv[1]):
        # In the shared subtree's body, replace any OTHER shared subtrees
        # (but not itself — that would be a recursive self-reference)
        other_shared = {k: v for k, v in shared_map.items() if k != node}
        body = _replace_shared(node, other_shared)
        definitions.append(Definition(name, body))

    return Program(definitions=tuple(definitions))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compression_ratio(ast: SessionType, *, min_size: int = _MIN_SHARED_SIZE) -> float:
    """Compute the compression ratio: tree_size / equation_size.

    A ratio > 1.0 means the equation form is smaller (more sharing).
    A ratio of 1.0 means no sharing was found.

    The equation size counts only AST nodes in definition bodies
    (not the definition names/overhead), since those are metadata,
    not protocol structure.
    """
    tree_sz = ast_size(ast)

    prog = compress(ast, min_size=min_size)
    if len(prog.definitions) == 1:
        return 1.0  # No sharing — no compression

    eq_sz = sum(ast_size(d.body) for d in prog.definitions)

    if eq_sz == 0:
        return 1.0
    return tree_sz / eq_sz


@dataclass(frozen=True)
class CompressionResult:
    """Result of session type compression analysis."""
    original_size: int
    compressed_size: int
    ratio: float
    num_definitions: int
    num_shared: int
    shared_names: tuple[str, ...]
    program: Program

    @property
    def has_sharing(self) -> bool:
        """True if compression found shared subtrees."""
        return self.num_shared > 0


def analyze_compression(
    ast: SessionType,
    *,
    min_size: int = _MIN_SHARED_SIZE,
    entry_name: str = "Main",
) -> CompressionResult:
    """Full compression analysis: compress + compute metrics."""
    prog = compress(ast, min_size=min_size, entry_name=entry_name)

    original_size = ast_size(ast)
    compressed_size = sum(ast_size(d.body) for d in prog.definitions)
    num_shared = len(prog.definitions) - 1  # exclude Main

    shared_names = tuple(d.name for d in prog.definitions[1:])

    if num_shared == 0 or compressed_size == 0:
        ratio = 1.0
    else:
        ratio = original_size / compressed_size

    return CompressionResult(
        original_size=original_size,
        compressed_size=compressed_size,
        ratio=ratio,
        num_definitions=len(prog.definitions),
        num_shared=num_shared,
        shared_names=shared_names,
        program=prog,
    )
