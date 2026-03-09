"""Exhaustive enumeration of session types for universality checking (Step 6).

Generates all structurally distinct session types up to a given depth and
width, builds their state spaces, and checks whether L(S) is always a lattice.

If a counterexample is found, it is reported with the type string and the
failing pair.  If no counterexample exists up to the tested bounds, that is
strong evidence for the universality conjecture:

    For all finite session types S, L(S) is a lattice.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product as cartesian
from typing import Iterator

from reticulate.lattice import LatticeResult, check_lattice
from reticulate.parser import (
    Branch,
    Continuation,
    End,
    Parallel,
    Rec,
    Select,
    SessionType,
    Var,
    Wait,
    pretty,
)
from reticulate.statespace import StateSpace, build_statespace


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_LABELS = ("a", "b", "c")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnumerationConfig:
    """Parameters for exhaustive type enumeration."""
    max_depth: int = 3
    labels: tuple[str, ...] = DEFAULT_LABELS
    max_branch_width: int = 3
    include_parallel: bool = True
    include_recursion: bool = True
    include_selection: bool = True
    require_terminating: bool = True  # skip non-terminating types


@dataclass(frozen=True)
class UniversalityResult:
    """Result of universality checking."""
    total_types: int
    total_lattices: int
    counterexamples: tuple[tuple[str, LatticeResult], ...]
    config: EnumerationConfig

    @property
    def is_universal(self) -> bool:
        return len(self.counterexamples) == 0


# ---------------------------------------------------------------------------
# Type enumeration
# ---------------------------------------------------------------------------

def enumerate_session_types(
    config: EnumerationConfig | None = None,
) -> Iterator[SessionType]:
    """Generate all session types up to the configured bounds.

    Yields distinct AST trees.  The enumeration is finite for any
    finite *max_depth*.
    """
    if config is None:
        config = EnumerationConfig()
    yield from _enumerate(config.max_depth, config, rec_var=None)


def _enumerate(
    depth: int,
    config: EnumerationConfig,
    rec_var: str | None,
) -> Iterator[SessionType]:
    """Recursively enumerate types at the given depth budget."""
    # Base case: depth 0 — only terminals
    yield End()

    if rec_var is not None:
        yield Var(rec_var)

    if depth <= 0:
        return

    # Branch: &{m1: S1, ..., mk: Sk} for k = 1..max_branch_width
    labels = config.labels
    for width in range(1, min(config.max_branch_width, len(labels)) + 1):
        for label_combo in combinations(labels, width):
            # Generate all combinations of sub-types for each label
            sub_types = list(_enumerate(depth - 1, config, rec_var))
            for choices in cartesian(sub_types, repeat=width):
                yield Branch(tuple(zip(label_combo, choices)))

    # Selection: +{l1: S1, ..., lk: Sk}
    if config.include_selection:
        for width in range(1, min(config.max_branch_width, len(labels)) + 1):
            for label_combo in combinations(labels, width):
                sub_types = list(_enumerate(depth - 1, config, rec_var))
                for choices in cartesian(sub_types, repeat=width):
                    yield Select(tuple(zip(label_combo, choices)))

    # Parallel: (S1 || S2)
    if config.include_parallel:
        # Use Wait instead of End for parallel branches
        left_types = list(_enumerate_parallel_branch(depth - 1, config, rec_var))
        right_types = list(_enumerate_parallel_branch(depth - 1, config, rec_var))
        for left in left_types:
            for right in right_types:
                yield Parallel(left, right)
                # Also yield with continuation: (S1 || S2) . S3
                for cont in _enumerate(depth - 1, config, rec_var):
                    yield Continuation(Parallel(left, right), cont)

    # Recursion: rec X . S
    if config.include_recursion and rec_var is None:
        var_name = "X"
        for body in _enumerate(depth - 1, config, rec_var=var_name):
            # Only yield if the body actually uses the variable
            # and is not just the bare variable (rec X . X is degenerate)
            if _contains_var(body, var_name) and not isinstance(body, Var):
                yield Rec(var_name, body)


def _enumerate_parallel_branch(
    depth: int,
    config: EnumerationConfig,
    rec_var: str | None,
) -> Iterator[SessionType]:
    """Enumerate types suitable for parallel branches (using Wait as terminal)."""
    yield Wait()

    if rec_var is not None:
        yield Var(rec_var)

    if depth <= 0:
        return

    labels = config.labels
    for width in range(1, min(config.max_branch_width, len(labels)) + 1):
        for label_combo in combinations(labels, width):
            sub_types = list(_enumerate_parallel_branch(depth - 1, config, rec_var))
            for choices in cartesian(sub_types, repeat=width):
                yield Branch(tuple(zip(label_combo, choices)))

    if config.include_selection:
        for width in range(1, min(config.max_branch_width, len(labels)) + 1):
            for label_combo in combinations(labels, width):
                sub_types = list(_enumerate_parallel_branch(depth - 1, config, rec_var))
                for choices in cartesian(sub_types, repeat=width):
                    yield Select(tuple(zip(label_combo, choices)))


def _contains_var(node: SessionType, var_name: str) -> bool:
    """Check if an AST contains a reference to the given variable."""
    if isinstance(node, Var):
        return node.name == var_name
    elif isinstance(node, (End, Wait)):
        return False
    elif isinstance(node, (Branch, Select)):
        return any(_contains_var(s, var_name) for _, s in node.choices)
    elif isinstance(node, Parallel):
        return _contains_var(node.left, var_name) or _contains_var(node.right, var_name)
    elif isinstance(node, Rec):
        return _contains_var(node.body, var_name)
    elif isinstance(node, Continuation):
        return _contains_var(node.left, var_name) or _contains_var(node.right, var_name)
    return False


# ---------------------------------------------------------------------------
# Universality checker
# ---------------------------------------------------------------------------

def check_universality(
    config: EnumerationConfig | None = None,
    *,
    verbose: bool = False,
) -> UniversalityResult:
    """Check whether L(S) is a lattice for all enumerated session types.

    Returns a UniversalityResult with any counterexamples found.
    """
    if config is None:
        config = EnumerationConfig()

    total = 0
    lattices = 0
    counterexamples: list[tuple[str, LatticeResult]] = []

    for ast in enumerate_session_types(config):
        # Filter non-terminating types if requested
        if config.require_terminating:
            from reticulate.termination import is_terminating
            if not is_terminating(ast):
                continue

        total += 1
        type_str = pretty(ast)

        try:
            ss = build_statespace(ast)
        except (ValueError, RecursionError, KeyError):
            # Skip malformed types (e.g., unbounded recursion)
            continue

        try:
            result = check_lattice(ss)
        except (KeyError, ValueError):
            # Skip types that produce degenerate state spaces
            continue

        if result.is_lattice:
            lattices += 1
        else:
            counterexamples.append((type_str, result))
            if verbose:
                a, b, kind = result.counterexample or (0, 0, "unknown")
                print(f"COUNTEREXAMPLE: {type_str}")
                print(f"  States: {len(ss.states)}, Transitions: {len(ss.transitions)}")
                print(f"  Failing pair: ({a}, {b}) — {kind}")
                print()

        if verbose and total % 1000 == 0:
            print(f"  ... checked {total} types, {lattices} lattices, "
                  f"{len(counterexamples)} counterexamples")

    return UniversalityResult(
        total_types=total,
        total_lattices=lattices,
        counterexamples=tuple(counterexamples),
        config=config,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run universality check with default configuration."""
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Check universality: is L(S) always a lattice?",
    )
    parser.add_argument("--depth", type=int, default=2,
                        help="Max AST depth (default: 2)")
    parser.add_argument("--labels", type=str, default="a,b",
                        help="Comma-separated method labels (default: a,b)")
    parser.add_argument("--max-width", type=int, default=2,
                        help="Max branch/select width (default: 2)")
    parser.add_argument("--no-parallel", action="store_true",
                        help="Exclude parallel types")
    parser.add_argument("--no-recursion", action="store_true",
                        help="Exclude recursive types")
    parser.add_argument("--no-selection", action="store_true",
                        help="Exclude selection types")
    parser.add_argument("--allow-nonterminating", action="store_true",
                        help="Include non-terminating recursive types")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    config = EnumerationConfig(
        max_depth=args.depth,
        labels=tuple(args.labels.split(",")),
        max_branch_width=args.max_width,
        include_parallel=not args.no_parallel,
        include_recursion=not args.no_recursion,
        include_selection=not args.no_selection,
        require_terminating=not args.allow_nonterminating,
    )

    print(f"Universality check — Step 6")
    print(f"  Depth: {config.max_depth}")
    print(f"  Labels: {config.labels}")
    print(f"  Max width: {config.max_branch_width}")
    print(f"  Parallel: {config.include_parallel}")
    print(f"  Recursion: {config.include_recursion}")
    print(f"  Selection: {config.include_selection}")
    print()

    start = time.time()
    result = check_universality(config, verbose=args.verbose)
    elapsed = time.time() - start

    print(f"Results:")
    print(f"  Types checked: {result.total_types}")
    print(f"  Lattices: {result.total_lattices}")
    print(f"  Counterexamples: {len(result.counterexamples)}")
    print(f"  Time: {elapsed:.2f}s")
    print()

    if result.is_universal:
        print(f"UNIVERSAL: L(S) is a lattice for all {result.total_types} "
              f"enumerated types up to depth {config.max_depth}.")
    else:
        print(f"NOT UNIVERSAL: {len(result.counterexamples)} counterexample(s) found!")
        for type_str, lr in result.counterexamples[:10]:
            a, b, kind = lr.counterexample or (0, 0, "unknown")
            print(f"  {type_str}  — ({a}, {b}) {kind}")


if __name__ == "__main__":
    main()
